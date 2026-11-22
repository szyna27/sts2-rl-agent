"""STS2 Combat Gymnasium Environment."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.cards.base import reset_instance_counter
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE, IRONCLAD_STARTING_HP
from sts2_env.encounters.act1 import ALL_ACT1_ENCOUNTERS, EncounterSetup
from sts2_env.encounters.room_set import (
    ROOM_TYPE_BOSS,
    ROOM_TYPE_ELITE,
    EncounterSpec,
)
from sts2_env.core.rng import INT_MAX_EXCLUSIVE, Rng
from sts2_env.gym_env.deck_builder import build_training_deck
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    get_action_mask,
    is_potion_action,
)
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.reward import compute_reward

logger = logging.getLogger(__name__)


class STS2CombatEnv(gymnasium.Env):
    """Gymnasium environment for a single STS2 combat encounter.

    Observation: flat float32 vector encoding player state, hand, piles, enemies.
    Action: fixed discrete combat action space including cards, end turn, and potions.

    Episode end contract (Gymnasium termination convention): once ``step()``
    returns ``terminated=True`` or ``truncated=True`` the episode is over and
    the underlying combat is frozen -- no further internal state advances.
    Any ``step()`` call after the episode ended raises
    :class:`gymnasium.error.ClosedEnvironmentError`; ``reset()`` starts a new
    episode and clears the flag.  Truncation (``turn_count > max_turns``) is
    reported exclusively through ``truncated`` and keeps ``terminated``
    untouched, preserving the Gymnasium terminated/truncated separation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        encounter_pool: list[EncounterSetup] | None = None,
        character_id: str = "Ironclad",
        player_hp: int | None = None,
        player_max_hp: int | None = None,
        max_turns: int = 200,
        render_mode: str | None = None,
        training_act: int | None = None,
        training_acts: Sequence[int] | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.encounter_pool = encounter_pool or ALL_ACT1_ENCOUNTERS
        if training_act is not None and training_acts is not None:
            raise ValueError("Set training_act or training_acts, not both")
        if training_act is not None and training_act not in {1, 2, 3, 4}:
            raise ValueError(f"Unsupported training_act: {training_act}")
        if training_acts is not None:
            if not training_acts or any(act not in {1, 2, 3, 4} for act in training_acts):
                raise ValueError("training_acts must contain acts from 1 through 4")
            if len(set(training_acts)) != len(training_acts):
                raise ValueError("training_acts must not contain duplicates")
            if encounter_pool is not None:
                raise ValueError("training_acts cannot be combined with a custom encounter_pool")
        self.training_act = training_act
        self.training_acts = tuple(training_acts or ())
        self.current_training_act: int | None = training_act
        self.current_encounter_id: str | None = None
        self.current_encounter_type: str | None = None
        self.current_deck_cards: tuple[tuple[str, bool], ...] = ()
        self._last_training_act_index: int | None = None
        self._custom_encounter_pool = encounter_pool is not None
        self.character_id = character_id
        default_hp = {
            "Ironclad": IRONCLAD_STARTING_HP,
            "Silent": 70,
            "Defect": 75,
            "Necrobinder": 75,
            "Regent": 75,
        }
        if character_id not in default_hp:
            raise ValueError(f"Unknown character_id: {character_id}")
        self.player_hp = default_hp[character_id] if player_hp is None else player_hp
        self.player_max_hp = (
            self.player_hp if player_max_hp is None else player_max_hp
        )
        self.max_turns = max_turns
        self.render_mode = render_mode

        self.combat: CombatState | None = None
        #: Set when the episode ends (terminated or truncated). While set,
        #: step() raises ClosedEnvironmentError and the internal combat is frozen.
        self._episode_ended = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_ended = False
        reset_instance_counter()

        rng_seed = int(self.np_random.integers(0, INT_MAX_EXCLUSIVE))
        rng = Rng(rng_seed)

        selected_training_act = self.training_act
        if self.training_acts:
            if self._last_training_act_index is None:
                self._last_training_act_index = int(
                    self.np_random.integers(0, len(self.training_acts))
                )
            else:
                self._last_training_act_index = (
                    self._last_training_act_index + 1
                ) % len(self.training_acts)
            selected_training_act = self.training_acts[self._last_training_act_index]
        self.current_training_act = selected_training_act

        encounter_type = "custom"
        if selected_training_act is None:
            encounter_idx = int(self.np_random.integers(0, len(self.encounter_pool)))
            encounter_setup = self.encounter_pool[encounter_idx]
            deck_factories = {
                "Ironclad": "ironclad",
                "Silent": "silent",
                "Defect": "defect",
                "Necrobinder": "necrobinder",
                "Regent": "regent",
            }
            module_name = deck_factories[self.character_id]
            module = __import__(f"sts2_env.cards.{module_name}", fromlist=["starter"])
            deck = getattr(module, f"create_{module_name}_starter_deck")()
            encounter_specs = None
        else:
            encounter_specs = self._get_training_encounter_specs(selected_training_act)
            if self._custom_encounter_pool:
                encounter_setups = self.encounter_pool
            else:
                encounter_setups = [spec.setup for spec in encounter_specs]
            encounter_idx = int(self.np_random.integers(0, len(encounter_setups)))
            encounter_setup = encounter_setups[encounter_idx]
            encounter_spec = next(
                (spec for spec in encounter_specs if spec.setup is encounter_setup),
                None,
            )
            encounter_type = "normal"
            if encounter_spec is not None:
                if encounter_spec.room_type == ROOM_TYPE_BOSS:
                    encounter_type = "boss"
                elif encounter_spec.room_type == ROOM_TYPE_ELITE:
                    encounter_type = "elite"
                elif encounter_spec.is_weak:
                    encounter_type = "weak"
            deck = build_training_deck(
                self.character_id,
                selected_training_act,
                encounter_type,
                rng,
            )

        self.current_encounter_id = getattr(
            encounter_setup, "__name__", type(encounter_setup).__name__
        )
        self.current_encounter_type = encounter_type
        self.current_deck_cards = tuple(
            (card.card_id.name, card.upgraded) for card in deck
        )

        # Create combat
        self.combat = CombatState(
            player_hp=self.player_hp,
            player_max_hp=self.player_max_hp,
            deck=deck,
            rng_seed=rng_seed,
            character_id=self.character_id,
        )

        # Setup encounter
        encounter_setup(self.combat, rng)

        # Start combat
        self.combat.start_combat()

        obs = encode_observation(self.combat)
        info = {
            "action_mask": get_action_mask(self.combat),
            "training_act": self.current_training_act,
            "encounter_id": self.current_encounter_id,
            "encounter_type": self.current_encounter_type,
        }
        return obs, info

    def _get_training_encounter_specs(self, training_act: int) -> list[EncounterSpec]:
        if training_act == 1:
            from sts2_env.encounters.act1 import ALL_ENCOUNTER_SPECS
        elif training_act == 2:
            from sts2_env.encounters.act2 import ALL_ENCOUNTER_SPECS
        elif training_act == 3:
            from sts2_env.encounters.act3 import ALL_ENCOUNTER_SPECS
        else:
            from sts2_env.encounters.act4 import (
                BOSS_ENCOUNTERS,
                ELITE_ENCOUNTERS,
                NORMAL_ENCOUNTERS,
                WEAK_ENCOUNTERS,
            )
            from sts2_env.encounters.room_set import (
                ROOM_TYPE_BOSS as ACT4_ROOM_TYPE_BOSS,
                ROOM_TYPE_ELITE as ACT4_ROOM_TYPE_ELITE,
            )

            return (
                [EncounterSpec(setup, is_weak=True) for setup in WEAK_ENCOUNTERS]
                + [EncounterSpec(setup) for setup in NORMAL_ENCOUNTERS]
                + [
                    EncounterSpec(setup, room_type=ACT4_ROOM_TYPE_ELITE)
                    for setup in ELITE_ENCOUNTERS
                ]
                + [
                    EncounterSpec(setup, room_type=ACT4_ROOM_TYPE_BOSS)
                    for setup in BOSS_ENCOUNTERS
                ]
            )
        return ALL_ENCOUNTER_SPECS

    def step(self, action: int):
        if self._episode_ended:
            raise gymnasium.error.ClosedEnvironmentError(
                "STS2CombatEnv episode has ended (terminated or truncated); "
                "call reset() before stepping again."
            )
        assert self.combat is not None, "Must call reset() first"

        prev_hp = self.combat.player.current_hp
        if self.combat.pending_choice is not None:
            if action == ACTION_END_TURN:
                self.combat.resolve_pending_choice(None)
            else:
                self.combat.resolve_pending_choice(action - 1)
        else:
            if action == ACTION_END_TURN:
                self.combat.end_player_turn()
            elif is_potion_action(action):
                slot_idx, target_idx = action_to_potion_and_target(action)
                success = (
                    slot_idx is not None
                    and self.combat.use_potion(slot_idx, target_index=target_idx)
                )
                if not success:
                    logger.debug("Ignored invalid potion action %d", action)
            else:
                hand_idx, target_idx = action_to_card_and_target(action)
                success = hand_idx is not None and self.combat.play_card(hand_idx, target_idx)
                if not success:
                    logger.debug("Ignored invalid card action %d", action)

        obs = encode_observation(self.combat)
        reward = compute_reward(self.combat, prev_hp)
        terminated = self.combat.is_over
        # Gymnasium TimeLimit convention: truncation is reported through
        # ``truncated`` only, keeping it mutually exclusive with ``terminated``.
        truncated = not terminated and self.combat.turn_count > self.max_turns
        if terminated or truncated:
            # Freeze the episode: the combat above stops advancing here (this
            # final observation is returned as-is) and any later step() call
            # is rejected by the _episode_ended guard.
            self._episode_ended = True
        info = {"action_mask": get_action_mask(self.combat)}

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current mask of legal actions."""
        if self.combat is None or self._episode_ended:
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
            mask[0] = 1
            return mask
        return get_action_mask(self.combat)

    def render(self):
        if self.render_mode == "ansi" and self.combat is not None:
            return str(self.combat)
        return None
