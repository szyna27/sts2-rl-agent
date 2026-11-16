"""State-value model and state encoding for full-run metagame decisions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from sts2_env.cards.base import preserve_instance_counter
from sts2_env.core.enums import CardId, RoomType
from sts2_env.map.acts import ALL_ACTS
from sts2_env.relics.base import RelicId
from sts2_env.run.run_manager import RunManager

METAGAME_MODEL_VERSION = 1
METAGAME_KEYS_TO_ACT4 = 3
METAGAME_CANDIDATE_FEATURES = 64

_CARD_IDS = tuple(card_id.name for card_id in CardId)
_RELIC_IDS = tuple(relic_id.name for relic_id in RelicId)
_BOSS_IDS = tuple(boss_id for act in ALL_ACTS for boss_id in act.boss_ids)
_CHARACTER_IDS = ("Ironclad", "Silent", "Defect", "Necrobinder", "Regent")
_PHASES = (
    RunManager.PHASE_MAP_CHOICE,
    RunManager.PHASE_COMBAT,
    RunManager.PHASE_CARD_REWARD,
    RunManager.PHASE_BOSS_RELIC,
    RunManager.PHASE_SHOP,
    RunManager.PHASE_REST_SITE,
    RunManager.PHASE_EVENT,
    RunManager.PHASE_TREASURE,
    RunManager.PHASE_RUN_OVER,
)
_ROOM_TYPES = tuple(RoomType)
_NUMERIC_FEATURES = (
    "hp_ratio",
    "max_hp",
    "gold",
    "floor",
    "act",
    "act_floor",
    "keys_to_act4",
    "ascension",
)


class MetagameStateEncoder:
    """Encode the persistent run state as a fixed-size float32 vector."""

    def __init__(self) -> None:
        self.card_ids = _CARD_IDS
        self.relic_ids = _RELIC_IDS
        self.boss_ids = _BOSS_IDS
        self.character_ids = _CHARACTER_IDS
        self.phase_ids = _PHASES
        self.room_types = _ROOM_TYPES
        self.card_offsets = {
            _canonical_id(name): index for index, name in enumerate(self.card_ids)
        }
        self.relic_offsets = {
            _canonical_id(name): index for index, name in enumerate(self.relic_ids)
        }
        self.boss_offsets = {
            _canonical_boss_id(name): index
            for index, name in enumerate(self.boss_ids)
        }
        self.character_offsets = {
            _canonical_id(name): index
            for index, name in enumerate(self.character_ids)
        }
        self.phase_offsets = {name: index for index, name in enumerate(self.phase_ids)}
        self.room_offsets = {
            room_type: index for index, room_type in enumerate(self.room_types)
        }

    @property
    def observation_size(self) -> int:
        return (
            len(self.card_ids)
            + len(self.relic_ids)
            + len(_NUMERIC_FEATURES)
            + METAGAME_KEYS_TO_ACT4
            + len(self.boss_ids)
            + len(self.character_ids)
            + len(self.phase_ids)
            + len(self.room_types)
            + METAGAME_CANDIDATE_FEATURES
        )

    def encode_manager(
        self,
        manager: RunManager,
        *,
        candidate_action: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode a simulator run, including its act's possible bosses."""
        run_state = manager.run_state
        player = run_state.player
        cards = Counter(card.card_id.name for card in player.deck)
        boss_ids = (
            ALL_ACTS[run_state.current_act_index].boss_ids
            if run_state.current_act_index < len(ALL_ACTS)
            else []
        )
        return self.encode_context(
            {
                "character_id": player.character_id,
                "deck": dict(cards),
                "relics": player.relics,
                "hp": player.current_hp,
                "max_hp": player.max_hp,
                "gold": player.gold,
                "floor": run_state.total_floor,
                "act_index": run_state.current_act_index,
                "act_floor": run_state.act_floor,
                "keys_to_act4": 0,
                "ascension": run_state.ascension_level,
                "boss_ids": boss_ids,
                "phase": manager.phase,
                "room_type": manager._current_room_type,
            },
            candidate_action=candidate_action,
        )

    def encode_context(
        self,
        context: Mapping[str, Any],
        *,
        candidate_action: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode a bridge ``metagame_state`` context with the same schema."""
        observation = np.zeros(self.observation_size, dtype=np.float32)
        offset = 0

        deck = context.get("deck", [])
        if isinstance(deck, Mapping):
            card_counts = {str(card_id): int(count) for card_id, count in deck.items()}
        else:
            card_counts = Counter(_read_model_id(card) for card in deck)
        for card_id, count in card_counts.items():
            index = self.card_offsets.get(_canonical_id(card_id))
            if index is None:
                raise ValueError(f"Unknown card ID in metagame state: {card_id}")
            observation[offset + index] = min(max(count, 0), 10) / 10.0
        offset += len(self.card_ids)

        relic_values = context.get("relics", [])
        relics = (
            relic_values.keys()
            if isinstance(relic_values, Mapping)
            else (_read_model_id(relic) for relic in relic_values)
        )
        for relic_id in relics:
            index = self.relic_offsets.get(_canonical_id(str(relic_id)))
            if index is None:
                raise ValueError(f"Unknown relic ID in metagame state: {relic_id}")
            observation[offset + index] = 1.0
        offset += len(self.relic_ids)

        hp = _read_int(context, "hp")
        max_hp = max(_read_int(context, "max_hp"), 1)
        numeric_values = (
            hp / max_hp,
            max_hp / 150.0,
            _read_int(context, "gold") / 1000.0,
            _read_int(context, "floor") / 50.0,
            _read_int(context, "act_index") / max(len(ALL_ACTS) - 1, 1),
            _read_int(context, "act_floor") / 20.0,
            _read_int(context, "keys_to_act4") / METAGAME_KEYS_TO_ACT4,
            _read_int(context, "ascension") / 20.0,
        )
        observation[offset : offset + len(numeric_values)] = np.clip(
            numeric_values, 0.0, 1.0
        )
        offset += len(numeric_values)

        keys = min(max(_read_int(context, "keys_to_act4"), 0), METAGAME_KEYS_TO_ACT4)
        observation[offset : offset + keys] = 1.0
        offset += METAGAME_KEYS_TO_ACT4

        boss_ids = context.get("boss_ids", [])
        if not boss_ids and context.get("boss_id"):
            boss_ids = [context["boss_id"]]
        for boss_id in boss_ids:
            boss_index = self.boss_offsets.get(_canonical_boss_id(str(boss_id)))
            if boss_index is not None:
                observation[offset + boss_index] = 1.0
        offset += len(self.boss_ids)

        character_index = self.character_offsets.get(
            _canonical_id(str(context.get("character_id", "")))
        )
        if character_index is not None:
            observation[offset + character_index] = 1.0
        offset += len(self.character_ids)

        phase = str(context.get("phase", ""))
        phase_index = self.phase_offsets.get(phase)
        if phase_index is not None:
            observation[offset + phase_index] = 1.0
        offset += len(self.phase_ids)

        room_type = context.get("room_type")
        if room_type is not None:
            if not isinstance(room_type, RoomType):
                room_type = _room_type_from_name(str(room_type))
            room_index = self.room_offsets.get(room_type)
            if room_index is not None:
                observation[offset + room_index] = 1.0
        offset += len(self.room_types)

        if candidate_action is not None:
            action_json = json.dumps(
                candidate_action,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            digest = hashlib.sha256(action_json.encode("utf-8")).digest()
            for byte_index in range(0, 16, 2):
                feature_index = int.from_bytes(
                    digest[byte_index : byte_index + 2], "big"
                ) % METAGAME_CANDIDATE_FEATURES
                observation[offset + feature_index] = 1.0

        return observation

    def schema(self) -> dict[str, Any]:
        """Return the ordered feature names stored with a model checkpoint."""
        return {
            "version": METAGAME_MODEL_VERSION,
            "card_ids": list(self.card_ids),
            "relic_ids": list(self.relic_ids),
            "numeric_features": list(_NUMERIC_FEATURES),
            "keys_to_act4": METAGAME_KEYS_TO_ACT4,
            "boss_ids": list(self.boss_ids),
            "character_ids": list(self.character_ids),
            "phase_ids": list(self.phase_ids),
            "room_types": [room_type.name for room_type in self.room_types],
            "candidate_features": METAGAME_CANDIDATE_FEATURES,
        }


class StateValueNetwork(nn.Module):
    """Estimate long-term run return from a metagame state vector."""

    def __init__(
        self,
        observation_size: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or hidden_size <= 0:
            raise ValueError("Network dimensions must be positive")
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations).squeeze(-1)


class MetagameAgent:
    """Choose actions by evaluating each simulated destination state."""

    def __init__(
        self,
        model: StateValueNetwork,
        encoder: MetagameStateEncoder | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder or MetagameStateEncoder()
        if model.observation_size != self.encoder.observation_size:
            raise ValueError("Model and metagame encoder observation sizes differ")
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "MetagameAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        encoder = MetagameStateEncoder()
        if checkpoint.get("schema") != encoder.schema():
            raise ValueError(
                "Metagame checkpoint feature schema does not match this game version"
            )
        model = StateValueNetwork(
            observation_size=encoder.observation_size,
            hidden_size=int(checkpoint["hidden_size"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, encoder)

    def score_states(self, states: Sequence[np.ndarray]) -> np.ndarray:
        if not states:
            return np.empty(0, dtype=np.float32)
        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.encoder.observation_size:
            raise ValueError("Candidate states do not match the metagame schema")
        with torch.no_grad():
            return self.model(torch.from_numpy(values)).cpu().numpy()

    def choose_action(
        self,
        manager: RunManager,
        *,
        rng: np.random.Generator | None = None,
        exploration_rate: float = 0.0,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Return a legal action and its simulated post-choice state."""
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between 0 and 1")
        if manager.phase == RunManager.PHASE_COMBAT:
            raise ValueError("MetagameAgent only selects non-combat actions")
        actions = manager.get_available_actions()
        if not actions:
            raise RuntimeError(f"No available actions during {manager.phase}")
        candidate_states = simulate_candidate_states(manager, actions, self.encoder)
        if rng is not None and rng.random() < exploration_rate:
            selected = int(rng.integers(len(actions)))
        else:
            selected = int(np.argmax(self.score_states(candidate_states)))
        return actions[selected], candidate_states[selected]

    def choose_bridge_option(self, state: Mapping[str, Any]) -> int | None:
        """Choose an option from a real-game bridge message."""
        context = state.get("metagame_state")
        if not isinstance(context, Mapping):
            raise ValueError(
                "Bridge state is missing metagame_state; rebuild the bridge mod"
            )
        state_type = str(state.get("type", ""))
        context = dict(context)
        context.setdefault("phase", _phase_for_bridge_state(state_type))
        options, option_kind = _bridge_candidates(state, state_type)
        if not options:
            raise ValueError(f"Bridge state {state_type!r} has no model candidates")

        encoded_states = [
            self.encoder.encode_context(
                _project_bridge_choice(context, option, option_kind, state_type),
                candidate_action=_bridge_action(option, option_kind, state_type),
            )
            for option in options
        ]
        best_index = int(np.argmax(self.score_states(encoded_states)))
        selected = options[best_index]
        selected_index = _read_int(selected, "index") if "index" in selected else None
        return None if selected_index is not None and selected_index < 0 else selected_index


def simulate_candidate_states(
    manager: RunManager,
    actions: Sequence[dict[str, Any]] | None = None,
    encoder: MetagameStateEncoder | None = None,
) -> list[np.ndarray]:
    """Apply each legal choice to a private run copy and encode its result."""
    if manager.phase == RunManager.PHASE_COMBAT:
        raise ValueError("Candidate simulation only supports non-combat phases")
    candidate_actions = list(actions if actions is not None else manager.get_available_actions())
    if not candidate_actions:
        raise ValueError(f"No available actions during {manager.phase}")
    state_encoder = encoder or MetagameStateEncoder()
    encoded: list[np.ndarray] = []
    for action in candidate_actions:
        candidate = _copy_manager_for_candidate(manager)
        with preserve_instance_counter():
            result = candidate.take_action(copy.deepcopy(action))
        if result.get("success") is False:
            raise RuntimeError(
                f"RunManager rejected legal candidate action {action!r}: {result}"
            )
        encoded.append(
            state_encoder.encode_manager(candidate, candidate_action=action)
        )
    return encoded


def _copy_manager_for_candidate(manager: RunManager) -> RunManager:
    memo: dict[int, Any] = {}
    candidate = copy.deepcopy(manager, memo)
    candidate._resume_after_run_choice = _rebind_callback(
        candidate._resume_after_run_choice, memo
    )
    _rebind_pending_choice(candidate.run_state.pending_choice, memo)
    event_model = candidate._event_model
    if event_model is not None:
        _rebind_pending_choice(event_model.pending_choice, memo)
    return candidate


def _rebind_pending_choice(choice: Any, memo: dict[int, Any]) -> None:
    if choice is not None:
        choice.resolver = _rebind_callback(choice.resolver, memo)


def _rebind_callback(callback: Any, memo: dict[int, Any]) -> Any:
    if not isinstance(callback, FunctionType) or callback.__closure__ is None:
        return callback
    closure = tuple(
        _make_cell(
            _rebind_callback(cell.cell_contents, memo)
            if isinstance(cell.cell_contents, FunctionType)
            else copy.deepcopy(cell.cell_contents, memo)
        )
        for cell in callback.__closure__
    )
    rebound = FunctionType(
        callback.__code__,
        callback.__globals__,
        callback.__name__,
        callback.__defaults__,
        closure,
    )
    rebound.__kwdefaults__ = callback.__kwdefaults__
    rebound.__annotations__ = callback.__annotations__
    rebound.__dict__.update(callback.__dict__)
    return rebound


def _make_cell(value: Any) -> Any:
    return (lambda: value).__closure__[0]


def save_metagame_checkpoint(
    path: str | Path,
    model: StateValueNetwork,
    encoder: MetagameStateEncoder,
    *,
    training_metadata: Mapping[str, Any],
) -> Path:
    """Save model weights together with the feature schema and run settings."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": METAGAME_MODEL_VERSION,
            "schema": encoder.schema(),
            "hidden_size": model.hidden_size,
            "state_dict": model.state_dict(),
            "training_metadata": json.loads(json.dumps(dict(training_metadata))),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _read_model_id(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("id", ""))
    return str(getattr(value, "name", value))


def _read_int(context: Mapping[str, Any], name: str) -> int:
    value = context.get(name, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metagame field {name!r} must be an integer") from exc


def _canonical_id(value: str) -> str:
    return value.replace("_", "").replace("-", "").casefold()


def _canonical_boss_id(value: str) -> str:
    canonical = _canonical_id(value)
    if canonical.startswith("setup"):
        canonical = canonical[5:]
    if canonical.endswith("boss"):
        canonical = canonical[:-4]
    return canonical


def _room_type_from_name(value: str) -> RoomType | None:
    canonical = _canonical_id(value)
    for room_type in RoomType:
        if _canonical_id(room_type.name) == canonical:
            return room_type
    return None


def _bridge_candidates(
    state: Mapping[str, Any],
    state_type: str,
) -> tuple[list[dict[str, Any]], str]:
    if state_type == "map_select":
        return list(state.get("nodes", [])), "map"
    if state_type == "card_reward":
        cards = list(state.get("cards", []))
        if state.get("can_skip", False):
            cards.append({"index": -1, "id": "skip", "action": "skip"})
        return cards, "card"
    if state_type == "card_bundle":
        return list(state.get("bundles", [])), "bundle"
    if state_type == "boss_relic":
        return _enabled_bridge_options(state), "boss_relic"
    if state_type in {"shop", "rest_site", "event", "crystal_sphere", "treasure"}:
        return _enabled_bridge_options(state), state_type
    raise ValueError(f"Bridge state type {state_type!r} is not a metagame choice")


def _enabled_bridge_options(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        option
        for option in state.get("options", [])
        if bool(option.get("enabled", True))
    ]


def _bridge_action(
    option: Mapping[str, Any],
    option_kind: str,
    state_type: str,
) -> dict[str, Any]:
    index = _read_int(option, "index") if "index" in option else 0
    if option_kind == "map":
        return {"action": "move", "index": index}
    if option_kind == "card":
        return {
            "action": "skip" if option.get("action") == "skip" else "pick_card",
            "index": index,
        }
    if option_kind == "bundle":
        return {"action": "pick_card_bundle", "index": index}
    if option_kind == "boss_relic":
        return {"action": "pick_relic", "index": index}
    if option_kind == "rest_site":
        return {
            "action": "rest_option",
            "option_id": option.get("id", ""),
        }
    if option_kind == "event":
        return {
            "action": "event_choice",
            "option_id": option.get("option_id", ""),
        }
    if option_kind == "crystal_sphere":
        return {"action": "divine_cell", "index": index}
    if option_kind == "treasure":
        return {"action": "collect", "index": index}
    if option_kind == "shop":
        return {
            "action": option.get("action", ""),
            "index": index,
        }
    raise ValueError(f"Unsupported bridge candidate type {state_type!r}")


def _project_bridge_choice(
    current: Mapping[str, Any],
    option: Mapping[str, Any],
    option_kind: str,
    state_type: str,
) -> dict[str, Any]:
    projected = dict(current)
    deck = list(current.get("deck", []))
    relics = list(current.get("relics", []))
    option_id = str(option.get("id", ""))
    action = str(option.get("action", ""))

    if option_kind == "map":
        projected["floor"] = _read_int(current, "floor") + 1
        projected["act_floor"] = _read_int(option, "row") + 1
        projected["room_type"] = option.get("type")
        projected["phase"] = _phase_after_map(option.get("type"))
    elif option_kind == "card":
        if action != "skip" and option_id:
            deck.append(option_id)
        projected["deck"] = deck
        projected["phase"] = RunManager.PHASE_MAP_CHOICE
    elif option_kind == "bundle":
        for card in option.get("cards", []):
            card_id = _read_model_id(card)
            if card_id:
                deck.append(card_id)
        projected["deck"] = deck
        projected["phase"] = RunManager.PHASE_MAP_CHOICE
    elif option_kind == "boss_relic":
        if option_id:
            relics.append(option_id)
        projected["relics"] = relics
        projected["phase"] = RunManager.PHASE_MAP_CHOICE
    elif option_kind == "shop":
        projected["gold"] = max(
            0, _read_int(current, "gold") - _read_int(option, "price")
        )
        if action == "buy_card" and option_id:
            deck.append(option_id)
        elif action == "buy_relic" and option_id:
            relics.append(option_id)
        projected["deck"] = deck
        projected["relics"] = relics
    elif option_kind == "rest_site":
        if _canonical_id(option_id) == "heal":
            hp = _read_int(current, "hp")
            max_hp = _read_int(current, "max_hp")
            projected["hp"] = min(max_hp, hp + max_hp * 30 // 100)
        projected["phase"] = RunManager.PHASE_MAP_CHOICE
    elif option_kind == "event":
        projected["phase"] = RunManager.PHASE_MAP_CHOICE
    elif option_kind == "treasure":
        projected["phase"] = RunManager.PHASE_MAP_CHOICE

    if state_type == "treasure" and option_id:
        projected["relics"] = relics + [option_id]
    return projected


def _phase_after_map(room_type: Any) -> str:
    room = _room_type_from_name(str(room_type or ""))
    if room in {RoomType.MONSTER, RoomType.ELITE, RoomType.BOSS}:
        return RunManager.PHASE_COMBAT
    if room == RoomType.SHOP:
        return RunManager.PHASE_SHOP
    if room == RoomType.REST_SITE:
        return RunManager.PHASE_REST_SITE
    if room == RoomType.EVENT:
        return RunManager.PHASE_EVENT
    if room == RoomType.TREASURE:
        return RunManager.PHASE_TREASURE
    return RunManager.PHASE_MAP_CHOICE


def _phase_for_bridge_state(state_type: str) -> str:
    return {
        "map_select": RunManager.PHASE_MAP_CHOICE,
        "card_reward": RunManager.PHASE_CARD_REWARD,
        "card_bundle": RunManager.PHASE_CARD_REWARD,
        "boss_relic": RunManager.PHASE_BOSS_RELIC,
        "shop": RunManager.PHASE_SHOP,
        "rest_site": RunManager.PHASE_REST_SITE,
        "event": RunManager.PHASE_EVENT,
        "crystal_sphere": RunManager.PHASE_EVENT,
        "treasure": RunManager.PHASE_TREASURE,
    }.get(state_type, "")
