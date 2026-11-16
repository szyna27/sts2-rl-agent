"""Training loop for the full-run metagame state-value model."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sts2_env.run.run_manager import RunManager
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.enums import RoomType
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    get_action_mask,
    is_potion_action,
)
from sts2_env.gym_env.observation import encode_observation
from sts2_env.heuristics import HeuristicName, choose_heuristic_action
from sts2_env.metagame import (
    MetagameAgent,
    MetagameStateEncoder,
    StateValueNetwork,
    save_metagame_checkpoint,
)

CombatPolicy = Callable[[CombatState], int]
ProgressCallback = Callable[[int, float, bool, int, int], None]


@dataclass(frozen=True)
class MetagameTrainingResult:
    episode_returns: tuple[float, ...]
    episode_wins: tuple[bool, ...]
    decision_counts: tuple[int, ...]
    updates: int


def compute_metagame_reward(
    *,
    floors_advanced: int = 0,
    elite_won: bool = False,
    boss_won: bool = False,
    run_won: bool = False,
    died: bool = False,
) -> float:
    """Apply long-term progression and terminal rewards to one run decision."""
    if floors_advanced < 0:
        raise ValueError("floors_advanced must be non-negative")
    if run_won and died:
        raise ValueError("A run cannot be both won and lost")
    return (
        float(floors_advanced)
        + (10.0 if elite_won else 0.0)
        + (50.0 if boss_won else 0.0)
        + (100.0 if run_won else -50.0 if died else 0.0)
    )


def make_heuristic_combat_policy(heuristic: HeuristicName) -> CombatPolicy:
    """Create a fixed combat policy for warm-start metagame runs."""
    def select_action(combat: CombatState) -> int:
        return choose_heuristic_action(combat, get_action_mask(combat), heuristic)

    return select_action


def make_maskable_ppo_combat_policy(model: Any) -> CombatPolicy:
    """Wrap a trained MaskablePPO combat model as a frozen run policy."""
    def select_action(combat: CombatState) -> int:
        action, _ = model.predict(
            encode_observation(combat),
            deterministic=True,
            action_masks=get_action_mask(combat).astype(np.bool_),
        )
        return int(np.asarray(action).item())

    return select_action


def train_metagame(
    *,
    episodes: int,
    combat_policy: CombatPolicy,
    character_id: str = "Ironclad",
    ascension_level: int = 0,
    seed: int = 0,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    replay_capacity: int = 20_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_decisions: int = 10_000,
    target_update_interval: int = 100,
    max_combat_turns: int = 200,
    hidden_size: int = 256,
    progress_callback: ProgressCallback | None = None,
) -> tuple[StateValueNetwork, MetagameStateEncoder, MetagameTrainingResult]:
    """Train a state-value network on complete runs with frozen combat."""
    _validate_training_options(
        episodes=episodes,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_decisions=epsilon_decay_decisions,
        target_update_interval=target_update_interval,
        max_combat_turns=max_combat_turns,
    )
    rng = np.random.default_rng(seed)
    replay_rng = random.Random(seed)
    encoder = MetagameStateEncoder()
    model = StateValueNetwork(encoder.observation_size, hidden_size)
    target_model = StateValueNetwork(encoder.observation_size, hidden_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay: deque[tuple[np.ndarray, float, np.ndarray, bool]] = deque(
        maxlen=replay_capacity
    )

    episode_returns: list[float] = []
    episode_wins: list[bool] = []
    decision_counts: list[int] = []
    updates = 0
    total_decisions = 0

    for episode in range(episodes):
        manager = RunManager(
            seed=seed + episode,
            character_id=character_id,
            ascension_level=ascension_level,
        )
        episode_return = 0.0
        decisions = 0

        while not manager.is_over:
            if manager.phase == RunManager.PHASE_COMBAT:
                _resolve_combat(manager, combat_policy, max_combat_turns)
                continue

            actions = manager.get_available_actions()
            if not actions:
                raise RuntimeError(
                    f"RunManager has no actions during {manager.phase}"
                )
            agent = MetagameAgent(model, encoder)
            progress = min(total_decisions / epsilon_decay_decisions, 1.0)
            epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
            selected_action, selected_candidate = agent.choose_action(
                manager,
                rng=rng,
                exploration_rate=epsilon,
            )

            floor_before = manager.run_state.total_floor
            room_type = manager._current_room_type
            result = manager.take_action(selected_action)
            if result.get("success") is False:
                raise RuntimeError(
                    f"RunManager rejected chosen action {selected_action!r}: {result}"
                )
            combat_won = False
            if manager.phase == RunManager.PHASE_COMBAT:
                combat_won = _resolve_combat(
                    manager,
                    combat_policy,
                    max_combat_turns,
                )
                if room_type is None:
                    room_type = manager._current_room_type

            reward = compute_metagame_reward(
                floors_advanced=max(
                    0, manager.run_state.total_floor - floor_before
                ),
                elite_won=combat_won and room_type == RoomType.ELITE,
                boss_won=combat_won and room_type == RoomType.BOSS,
                run_won=manager.is_over and manager.player_won,
                died=manager.is_over and not manager.player_won,
            )

            next_state = encoder.encode_manager(manager)
            replay.append(
                (
                    selected_candidate,
                    reward,
                    next_state,
                    manager.is_over,
                )
            )
            episode_return += reward
            decisions += 1
            total_decisions += 1
            updates += _train_from_replay(
                model,
                target_model,
                optimizer,
                replay,
                replay_rng,
                batch_size=batch_size,
                gamma=gamma,
            )
            if updates and updates % target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

        episode_returns.append(episode_return)
        episode_wins.append(manager.player_won)
        decision_counts.append(decisions)
        if progress_callback is not None:
            progress_callback(
                episode + 1,
                episode_return,
                manager.player_won,
                decisions,
                updates,
            )

    return (
        model,
        encoder,
        MetagameTrainingResult(
            episode_returns=tuple(episode_returns),
            episode_wins=tuple(episode_wins),
            decision_counts=tuple(decision_counts),
            updates=updates,
        ),
    )


def save_trained_metagame(
    path: str,
    model: StateValueNetwork,
    encoder: MetagameStateEncoder,
    result: MetagameTrainingResult,
    *,
    character_id: str,
    ascension_level: int,
    combat_model: str,
    seed: int,
) -> str:
    """Persist weights and reproducibility information for a training run."""
    checkpoint_path = save_metagame_checkpoint(
        path,
        model,
        encoder,
        training_metadata={
            "character_id": character_id,
            "ascension_level": ascension_level,
            "combat_model": combat_model,
            "seed": seed,
            "episodes": len(result.episode_returns),
            "episode_returns": list(result.episode_returns),
            "episode_wins": list(result.episode_wins),
            "decision_counts": list(result.decision_counts),
            "updates": result.updates,
            "combat_policy_frozen": True,
            "reward": {
                "per_floor": 1,
                "elite_win": 10,
                "boss_win": 50,
                "run_win": 100,
                "death": -50,
            },
        },
    )
    return str(checkpoint_path)


def _resolve_combat(
    manager: RunManager,
    combat_policy: CombatPolicy,
    max_combat_turns: int,
) -> bool:
    combat = manager.get_combat_state()
    if combat is None:
        raise RuntimeError("RunManager entered combat without a CombatState")
    while not combat.is_over:
        if combat.turn_count > max_combat_turns:
            manager.run_state.player.current_hp = 0
            manager.run_state.lose_run()
            return False
        mask = get_action_mask(combat)
        action = int(combat_policy(combat))
        if action < 0 or action >= len(mask) or not mask[action]:
            raise ValueError(f"Frozen combat policy returned illegal action {action}")
        result = _apply_combat_action(manager, combat, action)
        if result.get("success") is False:
            raise RuntimeError(f"Combat action failed: {result}")
    return combat.player_won


def _apply_combat_action(
    manager: RunManager,
    combat: CombatState,
    action: int,
) -> dict[str, Any]:
    if combat.pending_choice is not None:
        if action == ACTION_END_TURN:
            return manager.take_action({"action": "confirm_choice"})
        return manager.take_action({"action": "choose", "index": action - 1})
    if action == ACTION_END_TURN:
        return manager.take_action({"action": "end_turn"})
    if is_potion_action(action):
        slot, target = action_to_potion_and_target(action)
        if slot is None:
            raise ValueError(f"Invalid potion action index {action}")
        return manager.take_action(
            {
                "action": "use_potion",
                "slot_index": slot,
                "target_index": target,
            }
        )
    hand_index, target = action_to_card_and_target(action)
    if hand_index is None:
        return manager.take_action({"action": "end_turn"})
    card_action: dict[str, Any] = {
        "action": "play_card",
        "hand_index": hand_index,
    }
    if target is not None:
        card_action["target_index"] = target
    return manager.take_action(card_action)


def _train_from_replay(
    model: StateValueNetwork,
    target_model: StateValueNetwork,
    optimizer: torch.optim.Optimizer,
    replay: deque[tuple[np.ndarray, float, np.ndarray, bool]],
    rng: random.Random,
    *,
    batch_size: int,
    gamma: float,
) -> int:
    if not replay:
        return 0
    batch = rng.sample(replay, min(batch_size, len(replay)))
    states, rewards, next_states, terminal = zip(*batch, strict=True)
    state_tensor = torch.from_numpy(np.stack(states))
    next_state_tensor = torch.from_numpy(np.stack(next_states))
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    terminal_tensor = torch.tensor(terminal, dtype=torch.float32)
    with torch.no_grad():
        target_values = reward_tensor + gamma * target_model(next_state_tensor) * (
            1.0 - terminal_tensor
        )
    predictions = model(state_tensor)
    loss = nn.functional.smooth_l1_loss(predictions, target_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return 1


def _validate_training_options(
    *,
    episodes: int,
    gamma: float,
    learning_rate: float,
    batch_size: int,
    replay_capacity: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_decisions: int,
    target_update_interval: int,
    max_combat_turns: int,
) -> None:
    if episodes <= 0 or batch_size <= 0 or replay_capacity <= 0:
        raise ValueError("episodes, batch_size, and replay_capacity must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if epsilon_decay_decisions <= 0 or target_update_interval <= 0:
        raise ValueError("decay and target update intervals must be positive")
    if max_combat_turns <= 0:
        raise ValueError("max_combat_turns must be positive")
