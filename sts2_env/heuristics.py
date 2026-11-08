"""Simple combat action selectors used as baselines for learned policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sts2_env.core.combat import CombatState
from sts2_env.core.creature import Creature
from sts2_env.core.damage import DamageResult
from sts2_env.core.enums import CombatSide, PowerId, PowerType
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    is_potion_action,
)
from sts2_env.gym_env.combat_env import STS2CombatEnv

HeuristicName = Literal["block_first", "attack_first", "block_without_wasting"]
ActionSelector = Callable[[CombatState, np.ndarray], int]


@dataclass(frozen=True)
class CombatTrace:
    """Health-loss measurements collected after each completed player turn."""

    turns: list[int]
    lost_health: list[int]
    won: bool
    terminated: bool


@dataclass(frozen=True)
class CombatTurnSummary:
    """Actions and combat effects recorded during one player turn."""

    round_number: int
    hand: tuple[str, ...]
    played_cards: tuple[str, ...]
    blocked_damage: int
    damage_dealt: int
    damage_received: int
    wasted_block: int
    buffs: tuple[str, ...]
    debuffs: tuple[str, ...]
    buff_amount: int
    debuff_amount: int


@dataclass
class _CombatTurnAccumulator:
    round_number: int
    hand: list[str]
    played_cards: list[str]
    blocked_damage: int = 0
    damage_dealt: int = 0
    damage_received: int = 0
    wasted_block: int = 0
    buffs: dict[tuple[str, str], int] = field(default_factory=dict)
    debuffs: dict[tuple[str, str], int] = field(default_factory=dict)


def incoming_attack_damage(combat: CombatState) -> int:
    """Return the visible total damage of all living enemies' current moves."""
    total = 0
    for enemy in combat.enemies:
        if not enemy.is_alive:
            continue
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is None:
            continue
        total += sum(intent.total_damage for intent in ai.current_move.intents if intent.is_attack)
    return total


def _card_value(card: object, attribute: str) -> int:
    value = getattr(card, attribute, None)
    if value is not None:
        return int(value)
    effect_vars = getattr(card, "effect_vars", {})
    return int(effect_vars.get(attribute.removeprefix("base_"), 0))


def _display_name(value: object) -> str:
    name = getattr(value, "name", str(value))
    return str(name).replace("_", " ").title()


def _card_name(card: object) -> str:
    card_id = getattr(card, "card_id", card)
    raw_name = getattr(card_id, "name", str(card_id))
    for character in ("IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"):
        suffix = f"_{character}"
        if raw_name.endswith(suffix):
            raw_name = raw_name.removesuffix(suffix)
            break
    name = _display_name(raw_name)
    return f"{name}+" if getattr(card, "upgraded", False) else name


def _creature_name(creature: Creature) -> str:
    if creature.is_player:
        return "Player"
    return _display_name(creature.monster_id or f"Enemy {creature.combat_id}")


def _power_snapshot(
    combat: CombatState,
) -> dict[tuple[Creature, PowerId], tuple[int, PowerType]]:
    return {
        (creature, power_id): (power.amount, power.power_type)
        for creature in combat.all_creatures
        for power_id, power in creature.powers.items()
    }


def _record_power_gains(
    turn: _CombatTurnAccumulator,
    previous: dict[tuple[Creature, PowerId], tuple[int, PowerType]],
    current: dict[tuple[Creature, PowerId], tuple[int, PowerType]],
) -> None:
    for (creature, power_id), (amount, power_type) in current.items():
        previous_amount = previous.get((creature, power_id), (0, power_type))[0]
        gained = (
            abs(amount) - abs(previous_amount)
            if power_type == PowerType.DEBUFF and amount < 0
            else amount - previous_amount
        )
        if gained <= 0:
            continue
        changes = turn.buffs if power_type == PowerType.BUFF else turn.debuffs
        key = (_creature_name(creature), _display_name(power_id))
        changes[key] = changes.get(key, 0) + gained


def _damage_totals(
    results: list[tuple[Creature | None, DamageResult]],
    player: Creature,
) -> tuple[int, int, int]:
    blocked = dealt = received = 0
    for dealer, result in results:
        if result.target is player:
            blocked += result.blocked
            received += result.unblocked_damage
        elif (
            dealer is not None
            and dealer.side == CombatSide.PLAYER
            and result.target.side == CombatSide.ENEMY
        ):
            dealt += result.unblocked_damage
    return blocked, dealt, received


def _turn_summary(turn: _CombatTurnAccumulator) -> CombatTurnSummary:
    def format_powers(changes: dict[tuple[str, str], int]) -> tuple[str, ...]:
        if not changes:
            return ()
        return tuple(
            f"{target}: {power} +{amount}"
            for (target, power), amount in sorted(changes.items())
        )

    return CombatTurnSummary(
        round_number=turn.round_number,
        hand=tuple(turn.hand),
        played_cards=tuple(turn.played_cards),
        blocked_damage=turn.blocked_damage,
        damage_dealt=turn.damage_dealt,
        damage_received=turn.damage_received,
        wasted_block=turn.wasted_block,
        buffs=format_powers(turn.buffs),
        debuffs=format_powers(turn.debuffs),
        buff_amount=sum(turn.buffs.values()),
        debuff_amount=sum(turn.debuffs.values()),
    )


def _is_block_card(card: object) -> bool:
    return bool(getattr(card, "gains_block", False) or _card_value(card, "base_block") > 0)


def _action_card(combat: CombatState, action: int):
    hand_index, _ = action_to_card_and_target(action)
    if hand_index is None or hand_index >= len(combat.hand):
        return None
    return combat.hand[hand_index]


def _card_actions(combat: CombatState, action_mask: np.ndarray) -> list[tuple[int, object]]:
    actions: list[tuple[int, object]] = []
    for action in np.flatnonzero(action_mask):
        action = int(action)
        if action == 0 or is_potion_action(action):
            continue
        card = _action_card(combat, action)
        if card is not None:
            actions.append((action, card))
    return actions


def _best_action(actions: list[tuple[int, object]], attribute: str) -> int:
    def score(item: tuple[int, object]) -> tuple[float, int, int]:
        action, card = item
        value = _card_value(card, attribute)
        cost = max(int(getattr(card, "cost", 0)), 1)
        return (value / cost, value, -action)

    return max(actions, key=score)[0]


def choose_heuristic_action(
    combat: CombatState,
    action_mask: np.ndarray,
    heuristic: HeuristicName,
) -> int:
    """Choose one legal card action or end the turn.

    The selectors deliberately use only information exposed by the combat
    state: current hand, energy legality, block, and visible enemy intents.
    """
    if combat.pending_choice is not None:
        return 0

    actions = _card_actions(combat, action_mask)
    if not actions:
        return 0

    attacks = [(action, card) for action, card in actions if card.is_attack]
    blocks = [(action, card) for action, card in actions if _is_block_card(card)]
    incoming = incoming_attack_damage(combat)

    if heuristic == "block_first":
        preferred = blocks if combat.player.block < incoming else attacks
        if preferred:
            return _best_action(preferred, "base_block" if preferred is blocks else "base_damage")
    elif heuristic == "attack_first":
        preferred = attacks or blocks
        if preferred:
            return _best_action(preferred, "base_damage" if attacks else "base_block")
    elif heuristic == "block_without_wasting":
        usable_blocks = [
            (action, card)
            for action, card in blocks
            if combat.player.block + _card_value(card, "base_block") <= incoming
        ]
        if usable_blocks:
            return _best_action(usable_blocks, "base_block")
        if attacks:
            return _best_action(attacks, "base_damage")
        if blocks:
            return _best_action(blocks, "base_block")
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    return _best_action(actions, "base_damage")


def simulate_combat(
    encounter_setup,
    *,
    character_id: str = "Ironclad",
    seed: int = 0,
    heuristic: HeuristicName | None = None,
    action_selector: ActionSelector | None = None,
    max_steps: int = 1_000,
) -> CombatTrace:
    """Run one fixed encounter using a heuristic or custom action selector."""
    if (heuristic is None) == (action_selector is None):
        raise ValueError("Provide exactly one of heuristic or action_selector")

    env = STS2CombatEnv(
        encounter_pool=[encounter_setup],
        character_id=character_id,
        max_turns=max_steps,
    )
    _, info = env.reset(seed=seed)
    assert env.combat is not None
    initial_hp = env.combat.player.current_hp
    turns = [1]
    lost_health = [0]
    previous_round = env.combat.round_number

    for _ in range(max_steps):
        assert env.combat is not None
        action = (
            choose_heuristic_action(env.combat, info["action_mask"], heuristic)
            if heuristic is not None
            else action_selector(env.combat, info["action_mask"])
        )
        _, _, terminated, truncated, info = env.step(int(action))
        assert env.combat is not None
        if env.combat.round_number != previous_round or terminated or truncated:
            turns.append(env.combat.round_number)
            lost_health.append(initial_hp - env.combat.player.current_hp)
            previous_round = env.combat.round_number
        if terminated or truncated:
            break

    return CombatTrace(
        turns=turns,
        lost_health=lost_health,
        won=env.combat.player_won,
        terminated=env.combat.is_over,
    )


def evaluate_heuristics(
    encounter_setup,
    seeds: list[int],
    heuristics: dict[str, HeuristicName],
    *,
    character_id: str = "Ironclad",
    max_steps: int = 1_000,
) -> dict[str, list[CombatTrace]]:
    """Run every heuristic on the same sequence of combat seeds."""
    return {
        label: [
            simulate_combat(
                encounter_setup,
                character_id=character_id,
                seed=seed,
                heuristic=heuristic,
                max_steps=max_steps,
            )
            for seed in seeds
        ]
        for label, heuristic in heuristics.items()
    }


def show_combat(
    encounter_setup,
    *,
    heuristic: HeuristicName,
    character_id: str = "Ironclad",
    seed: int = 0,
    max_steps: int = 1_000,
) -> list[CombatTurnSummary]:
    """Run and print one combat, with a per-player-turn action and effect summary."""
    env = STS2CombatEnv(
        encounter_pool=[encounter_setup],
        character_id=character_id,
        max_turns=max_steps,
    )
    _, info = env.reset(seed=seed)
    assert env.combat is not None
    combat = env.combat
    turn = _CombatTurnAccumulator(
        round_number=combat.round_number,
        hand=[_card_name(card) for card in combat.hand],
        played_cards=[],
    )
    summaries: list[CombatTurnSummary] = []

    for _ in range(max_steps):
        assert env.combat is not None
        combat = env.combat
        action = choose_heuristic_action(combat, info["action_mask"], heuristic)
        hand_index, _ = action_to_card_and_target(action)
        if (
            hand_index is not None
            and hand_index < len(combat.hand)
            and not is_potion_action(action)
        ):
            turn.played_cards.append(_card_name(combat.hand[hand_index]))

        damage_start = len(combat._damage_results_combat)
        block_clear_start = len(combat._block_clear_events_combat)
        powers_before = _power_snapshot(combat)
        previous_round = combat.round_number
        _, _, terminated, truncated, info = env.step(int(action))
        assert env.combat is not None
        combat = env.combat

        step_results = combat._damage_results_combat[damage_start:]
        blocked, dealt, received = _damage_totals(step_results, combat.player)
        turn.blocked_damage += blocked
        turn.damage_dealt += dealt
        turn.damage_received += received
        turn.wasted_block += sum(
            amount
            for creature, amount in combat._block_clear_events_combat[block_clear_start:]
            if creature is combat.player
        )
        _record_power_gains(turn, powers_before, _power_snapshot(combat))

        if combat.round_number != previous_round or terminated or truncated:
            summaries.append(_turn_summary(turn))
            if terminated or truncated:
                break
            turn = _CombatTurnAccumulator(
                round_number=combat.round_number,
                hand=[_card_name(card) for card in combat.hand],
                played_cards=[],
            )

    else:
        summaries.append(_turn_summary(turn))

    print(f"Walka: {getattr(encounter_setup, '__name__', encounter_setup)}")
    print(f"Heurystyka: {heuristic} | seed: {seed}")
    for summary in summaries:
        print(f"\nTura {summary.round_number}")
        print(f"  Ręka: {', '.join(summary.hand) or '—'}")
        print(f"  Zagrane karty: {', '.join(summary.played_cards) or '—'}")
        print(f"  Zablokowane obrażenia: {summary.blocked_damage}")
        print(f"  Zadane obrażenia: {summary.damage_dealt}")
        print(f"  Otrzymane obrażenia: {summary.damage_received}")
        print(f"  Zmarnowany blok: {summary.wasted_block}")
        print(
            f"  Buffy (suma: {summary.buff_amount}): "
            f"{', '.join(summary.buffs) or '—'}"
        )
        print(
            f"  Debuffy (suma: {summary.debuff_amount}): "
            f"{', '.join(summary.debuffs) or '—'}"
        )

    outcome = "wygrana" if combat.player_won else "przegrana" if combat.is_over else "limit kroków"
    print(f"\nWynik: {outcome}")
    print(
        "Suma: "
        f"zablokowane {sum(row.blocked_damage for row in summaries)}, "
        f"zadane {sum(row.damage_dealt for row in summaries)}, "
        f"otrzymane {sum(row.damage_received for row in summaries)}, "
        f"zmarnowany blok {sum(row.wasted_block for row in summaries)}, "
        f"buffy {sum(row.buff_amount for row in summaries)}, "
        f"debuffy {sum(row.debuff_amount for row in summaries)}"
    )
    return summaries
