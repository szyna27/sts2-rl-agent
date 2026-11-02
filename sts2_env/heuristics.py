"""Simple combat action selectors used as baselines for learned policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sts2_env.core.combat import CombatState
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
