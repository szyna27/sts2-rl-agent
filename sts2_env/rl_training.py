"""Reinforcement-learning utilities for training combat policies."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np

from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CombatSide
from sts2_env.encounters.act1 import EncounterSetup
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import encode_observation
from sts2_env.heuristics import (
    CombatTrace,
    HeuristicName,
    simulate_combat,
)

CARD_PERFORMANCE_FLUSH_INTERVAL = 25

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class RewardConfig:
    """Configurable combat reward coefficients."""

    damage_dealt: float = 1.0
    damage_received: float = -3.0
    win: float = 50.0
    death: float = -100.0
    per_turn: float = -5.0
    wasted_block: float = -0.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Reward coefficients must be finite numbers")


@dataclass(frozen=True)
class RewardBreakdown:
    damage_dealt: float
    damage_received: float
    turn_penalty: float
    wasted_block: float
    terminal: float

    @property
    def total(self) -> float:
        return (
            self.damage_dealt
            + self.damage_received
            + self.turn_penalty
            + self.wasted_block
            + self.terminal
        )


def compute_training_reward(
    config: RewardConfig,
    *,
    damage_dealt: int = 0,
    damage_received: int = 0,
    turns: int = 0,
    wasted_block: int = 0,
    won: bool = False,
    died: bool = False,
) -> RewardBreakdown:
    """Calculate reward components from observed combat outcomes."""
    if min(damage_dealt, damage_received, turns, wasted_block) < 0:
        raise ValueError("Damage, turn, and block counts must be non-negative")
    terminal = config.win if won else config.death if died else 0.0
    return RewardBreakdown(
        damage_dealt=config.damage_dealt * damage_dealt,
        damage_received=config.damage_received * damage_received,
        turn_penalty=config.per_turn * turns,
        wasted_block=config.wasted_block * wasted_block,
        terminal=terminal,
    )


def _serialize_card_counts(
    card_counts: Mapping[tuple[str, bool], int],
) -> list[dict[str, Any]]:
    return [
        {
            "card_id": card_id,
            "upgraded": upgraded,
            "count": count,
        }
        for (card_id, upgraded), count in sorted(card_counts.items())
    ]


class CombatTrainingEnv(gymnasium.Wrapper):
    """Apply configurable shaping rewards without changing combat semantics."""

    def __init__(
        self,
        env: STS2CombatEnv,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_config = reward_config or RewardConfig()
        self._charged_turns = 0
        self._episode_damage_dealt = 0
        self._episode_damage_received = 0
        self._episode_wasted_block = 0
        self._episode_reward_components: dict[str, float] = {}
        self._episode_training_act: int | None = None
        self._episode_encounter_id = ""
        self._episode_encounter_type = ""
        self._episode_cards_in_deck: Counter[tuple[str, bool]] = Counter()
        self._episode_cards_played: Counter[tuple[str, bool]] = Counter()

    @property
    def combat(self) -> CombatState:
        env = self.unwrapped
        if not isinstance(env, STS2CombatEnv) or env.combat is None:
            raise RuntimeError("CombatTrainingEnv must wrap a reset STS2CombatEnv")
        return env.combat

    def action_masks(self) -> np.ndarray:
        env = self.unwrapped
        if not isinstance(env, STS2CombatEnv):
            raise RuntimeError("CombatTrainingEnv must wrap an STS2CombatEnv")
        return env.action_masks()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        training_env = self.unwrapped
        if not isinstance(training_env, STS2CombatEnv):
            raise RuntimeError("CombatTrainingEnv must wrap an STS2CombatEnv")
        self._charged_turns = 0
        self._episode_damage_dealt = 0
        self._episode_damage_received = 0
        self._episode_wasted_block = 0
        self._episode_training_act = training_env.current_training_act
        self._episode_encounter_id = training_env.current_encounter_id or ""
        self._episode_encounter_type = training_env.current_encounter_type or "custom"
        self._episode_cards_in_deck = Counter(training_env.current_deck_cards)
        self._episode_cards_played = Counter()
        self._episode_reward_components = {
            name: 0.0 for name in RewardBreakdown.__dataclass_fields__
        }
        return observation, info

    def step(self, action: int):
        combat_before = self.combat
        damage_start = len(combat_before._damage_results_combat)
        block_start = len(combat_before._block_clear_events_combat)
        played_cards_start = len(combat_before._played_cards_combat)
        turn_count_before = combat_before.turn_count

        observation, _, terminated, truncated, info = self.env.step(action)
        combat_after = self.combat
        for card in combat_after._played_cards_combat[played_cards_start:]:
            self._episode_cards_played[(card.card_id.name, card.upgraded)] += 1
        new_damage_events = combat_after._damage_results_combat[damage_start:]
        damage_dealt = sum(
            result.hp_lost
            for dealer, result in new_damage_events
            if dealer is not None
            and dealer.side == CombatSide.PLAYER
            and result.target.side == CombatSide.ENEMY
        )
        damage_received = sum(
            result.hp_lost
            for _, result in new_damage_events
            if result.target is combat_after.player
        )
        wasted_block = sum(
            amount
            for creature, amount in combat_after._block_clear_events_combat[block_start:]
            if creature is combat_after.player
        )
        turns = max(0, turn_count_before - self._charged_turns)
        self._charged_turns += turns
        self._episode_damage_dealt += damage_dealt
        self._episode_damage_received += damage_received
        self._episode_wasted_block += wasted_block
        won = terminated and combat_after.player_won
        died = terminated and combat_after.player.is_dead
        breakdown = compute_training_reward(
            self.reward_config,
            damage_dealt=damage_dealt,
            damage_received=damage_received,
            turns=turns,
            wasted_block=wasted_block,
            won=won,
            died=died,
        )
        for name, value in asdict(breakdown).items():
            self._episode_reward_components[name] += value

        info = dict(info)
        info["reward_components"] = asdict(breakdown)
        info["is_success"] = bool(won)
        if terminated or truncated:
            info.update(
                {
                    "damage_received": int(self._episode_damage_received),
                    "wasted_block": int(self._episode_wasted_block),
                    "turns": int(self._charged_turns),
                    "training_act": self._episode_training_act,
                    "encounter_id": self._episode_encounter_id,
                    "encounter_type": self._episode_encounter_type,
                    "cards_in_deck": _serialize_card_counts(
                        self._episode_cards_in_deck
                    ),
                    "cards_played": _serialize_card_counts(
                        self._episode_cards_played
                    ),
                }
            )
            info["damage_dealt"] = int(self._episode_damage_dealt)
            for name, value in self._episode_reward_components.items():
                info[f"reward_{name}"] = value
        return observation, float(breakdown.total), terminated, truncated, info


def make_training_env(
    encounters: Sequence[EncounterSetup] | None = None,
    *,
    training_act: int | None = None,
    training_acts: Sequence[int] | None = None,
    character_id: str = "Ironclad",
    reward_config: RewardConfig | None = None,
    max_turns: int = 200,
    log_dir: str | Path | None = None,
) -> gymnasium.Wrapper:
    """Build a training environment for an encounter pool or one/more acts."""
    if not encounters and training_act is None and training_acts is None:
        raise ValueError("Provide encounters, a training_act, or training_acts")
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "RL training requires the optional dependencies; install them with "
            "`uv sync --extra rl --extra notebook`."
        ) from exc

    env = STS2CombatEnv(
        encounter_pool=list(encounters) if encounters else None,
        training_act=training_act,
        training_acts=training_acts,
        character_id=character_id,
        max_turns=max_turns,
    )
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        monitor_path = str(log_path / "monitor.csv")
    else:
        monitor_path = None

    return Monitor(
        CombatTrainingEnv(env, reward_config),
        filename=monitor_path,
        info_keywords=(
            "is_success",
            "training_act",
            "encounter_id",
            "encounter_type",
            "turns",
            "damage_dealt",
            "damage_received",
            "wasted_block",
            "reward_damage_dealt",
            "reward_damage_received",
            "reward_turn_penalty",
            "reward_wasted_block",
            "reward_terminal",
        ),
    )


def create_training_run_dir(log_root: str | Path) -> Path:
    """Create a unique, persistent directory for one training session."""
    run_dir = Path(log_root) / f"session-{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _evaluate_training_policy(
    model: Any,
    encounters: Sequence[EncounterSetup] | None,
    seeds: Sequence[int],
    *,
    character_id: str,
    reward_config: RewardConfig,
    max_turns: int,
    training_act: int | None = None,
    training_acts: Sequence[int] | None = None,
) -> dict[str, float]:
    """Evaluate a policy on fixed pairs and summarize combat outcomes."""
    if (not encounters and training_act is None and training_acts is None) or not seeds:
        raise ValueError("Periodic evaluation needs encounters or an act selection, and seeds")

    metrics: dict[str, list[float]] = {
        "reward": [],
        "win": [],
        "turns": [],
        "damage_dealt": [],
        "damage_received": [],
        "wasted_block": [],
    }
    shared_env = None
    if training_acts is not None:
        shared_env = CombatTrainingEnv(
            STS2CombatEnv(
                training_acts=training_acts,
                character_id=character_id,
                max_turns=max_turns,
            ),
            reward_config,
        )
    try:
        for index, seed in enumerate(seeds):
            env = shared_env
            if env is None:
                encounter_pool = (
                    [encounters[index % len(encounters)]]
                    if encounters
                    else None
                )
                env = CombatTrainingEnv(
                    STS2CombatEnv(
                        encounter_pool=encounter_pool,
                        training_act=training_act,
                        character_id=character_id,
                        max_turns=max_turns,
                    ),
                    reward_config,
                )
            try:
                observation, _ = env.reset(seed=int(seed))
                total_reward = 0.0
                for _ in range(max_turns * 10):
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                        action_masks=env.action_masks(),
                    )
                    observation, reward, terminated, truncated, info = env.step(
                        int(np.asarray(action).item())
                    )
                    total_reward += reward
                    if terminated or truncated:
                        metrics["reward"].append(total_reward)
                        metrics["win"].append(float(info["is_success"]))
                        metrics["turns"].append(float(info["turns"]))
                        metrics["damage_dealt"].append(float(info["damage_dealt"]))
                        metrics["damage_received"].append(
                            float(info["damage_received"])
                        )
                        metrics["wasted_block"].append(float(info["wasted_block"]))
                        break
                else:
                    raise RuntimeError(
                        f"Evaluation episode exceeded {max_turns * 10} environment steps"
                    )
            finally:
                if shared_env is None:
                    env.close()
    finally:
        if shared_env is not None:
            shared_env.close()

    return {
        "mean_reward": float(np.mean(metrics["reward"])),
        "win_rate": float(np.mean(metrics["win"])),
        "mean_turns": float(np.mean(metrics["turns"])),
        "mean_damage_dealt": float(np.mean(metrics["damage_dealt"])),
        "mean_damage_received": float(np.mean(metrics["damage_received"])),
        "mean_wasted_block": float(np.mean(metrics["wasted_block"])),
    }


def evaluate_training_policy(
    model: Any,
    seeds: Sequence[int],
    *,
    encounters: Sequence[EncounterSetup] | None = None,
    training_act: int | None = None,
    training_acts: Sequence[int] | None = None,
    character_id: str = "Ironclad",
    reward_config: RewardConfig | None = None,
    max_turns: int = 200,
) -> dict[str, float]:
    """Evaluate a policy on fixed seeds for an encounter pool or act sequence."""
    return _evaluate_training_policy(
        model,
        encounters,
        seeds,
        character_id=character_id,
        reward_config=reward_config or RewardConfig(),
        max_turns=max_turns,
        training_act=training_act,
        training_acts=training_acts,
    )


class CardPerformanceTracker:
    """Aggregate HP loss by card, act, and exact or act/type encounter group."""

    fieldnames = (
        "scope",
        "act",
        "encounter_type",
        "encounter_id",
        "card_id",
        "upgraded",
        "deck_battles",
        "deck_copies",
        "played_battles",
        "play_count",
        "mean_hp_lost_in_deck",
        "mean_hp_lost_when_played",
    )

    def __init__(self) -> None:
        self._totals: dict[tuple[str, str, str, str, str, bool], dict[str, int]] = {}

    def record_episode(self, info: Mapping[str, Any]) -> None:
        required_fields = {
            "training_act",
            "encounter_id",
            "encounter_type",
            "damage_received",
            "cards_in_deck",
            "cards_played",
        }
        missing_fields = required_fields - info.keys()
        if missing_fields:
            raise ValueError(
                f"Episode info is missing card-performance fields: {sorted(missing_fields)}"
            )

        deck_counts = self._read_card_counts(info["cards_in_deck"])
        play_counts = self._read_card_counts(info["cards_played"])
        card_keys = deck_counts.keys() | play_counts.keys()
        act = "" if info["training_act"] is None else str(info["training_act"])
        encounter_id = str(info["encounter_id"])
        encounter_type = str(info["encounter_type"])
        hp_lost = int(info["damage_received"])
        if hp_lost < 0:
            raise ValueError("damage_received must be non-negative")

        for card_id, upgraded in card_keys:
            deck_copies = deck_counts.get((card_id, upgraded), 0)
            play_count = play_counts.get((card_id, upgraded), 0)
            for scope, group_encounter_id in (
                ("encounter", encounter_id),
                ("act_type", "ALL"),
            ):
                key = (
                    scope,
                    act,
                    encounter_type,
                    group_encounter_id,
                    card_id,
                    upgraded,
                )
                totals = self._totals.setdefault(
                    key,
                    {
                        "deck_battles": 0,
                        "deck_copies": 0,
                        "played_battles": 0,
                        "play_count": 0,
                        "hp_lost_in_deck": 0,
                        "hp_lost_when_played": 0,
                    },
                )
                if deck_copies:
                    totals["deck_battles"] += 1
                    totals["deck_copies"] += deck_copies
                    totals["hp_lost_in_deck"] += hp_lost
                if play_count:
                    totals["played_battles"] += 1
                    totals["play_count"] += play_count
                    totals["hp_lost_when_played"] += hp_lost

    @staticmethod
    def _read_card_counts(
        card_entries: Sequence[Mapping[str, Any]],
    ) -> Counter[tuple[str, bool]]:
        counts: Counter[tuple[str, bool]] = Counter()
        for entry in card_entries:
            card_id = str(entry["card_id"])
            upgraded = bool(entry["upgraded"])
            count = int(entry["count"])
            if count <= 0:
                raise ValueError("Card counts must be positive")
            counts[(card_id, upgraded)] += count
        return counts

    def write_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            for key, totals in sorted(self._totals.items()):
                (
                    scope,
                    act,
                    encounter_type,
                    encounter_id,
                    card_id,
                    upgraded,
                ) = key
                deck_battles = totals["deck_battles"]
                played_battles = totals["played_battles"]
                writer.writerow(
                    {
                        "scope": scope,
                        "act": act,
                        "encounter_type": encounter_type,
                        "encounter_id": encounter_id,
                        "card_id": card_id,
                        "upgraded": upgraded,
                        "deck_battles": deck_battles,
                        "deck_copies": totals["deck_copies"],
                        "played_battles": played_battles,
                        "play_count": totals["play_count"],
                        "mean_hp_lost_in_deck": (
                            totals["hp_lost_in_deck"] / deck_battles
                            if deck_battles
                            else ""
                        ),
                        "mean_hp_lost_when_played": (
                            totals["hp_lost_when_played"] / played_battles
                            if played_battles
                            else ""
                        ),
                    }
                )


def make_training_metrics_callback(
    log_dir: str | Path,
    *,
    evaluation_encounters: Sequence[EncounterSetup] | None = None,
    evaluation_seeds: Sequence[int],
    evaluation_interval: int,
    evaluation_training_act: int | None = None,
    evaluation_training_acts: Sequence[int] | None = None,
    character_id: str = "Ironclad",
    reward_config: RewardConfig | None = None,
    max_turns: int = 200,
    start_timesteps: int = 0,
) -> BaseCallback:
    """Persist episode metrics and periodically evaluate the policy."""
    if evaluation_interval <= 0 or start_timesteps < 0:
        raise ValueError("evaluation_interval must be positive and start_timesteps non-negative")
    if (
        not evaluation_encounters
        and evaluation_training_act is None
        and evaluation_training_acts is None
    ) or not evaluation_seeds:
        raise ValueError("Provide evaluation encounters or an act selection, and evaluation seeds")
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise ImportError(
            "RL training requires the optional dependencies; install them with "
            "`uv sync --extra rl --extra notebook`."
        ) from exc

    run_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_fields = (
        "timesteps",
        "reward",
        "length",
        "is_success",
        "training_act",
        "encounter_id",
        "encounter_type",
        "turns",
        "damage_dealt",
        "damage_received",
        "wasted_block",
        "reward_damage_dealt",
        "reward_damage_received",
        "reward_turn_penalty",
        "reward_wasted_block",
        "reward_terminal",
    )
    evaluation_fields = (
        "timesteps",
        "num_episodes",
        "mean_reward",
        "win_rate",
        "mean_turns",
        "mean_damage_dealt",
        "mean_damage_received",
        "mean_wasted_block",
    )

    class TrainingMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.episode_file: Any = None
            self.evaluation_file: Any = None
            self.episode_writer: Any = None
            self.evaluation_writer: Any = None
            self.card_performance = CardPerformanceTracker()
            self.completed_episodes = 0
            self.next_evaluation = (
                start_timesteps // evaluation_interval + 1
            ) * evaluation_interval

        def _on_training_start(self) -> None:
            self.episode_file = (run_dir / "episodes.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self.evaluation_file = (run_dir / "evaluations.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self.episode_writer = csv.DictWriter(
                self.episode_file, fieldnames=episode_fields
            )
            self.evaluation_writer = csv.DictWriter(
                self.evaluation_file, fieldnames=evaluation_fields
            )
            self.episode_writer.writeheader()
            self.evaluation_writer.writeheader()
            self.episode_file.flush()
            self.evaluation_file.flush()
            self.card_performance.write_csv(run_dir / "card_performance.csv")

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is None:
                    continue
                self.card_performance.record_episode(info)
                self.completed_episodes += 1
                if self.completed_episodes % CARD_PERFORMANCE_FLUSH_INTERVAL == 0:
                    self.card_performance.write_csv(
                        run_dir / "card_performance.csv"
                    )
                row = {
                    "timesteps": self.num_timesteps,
                    "reward": episode["r"],
                    "length": episode["l"],
                    **{
                        field: info[field]
                        for field in episode_fields[3:]
                    },
                }
                self.episode_writer.writerow(row)
                self.episode_file.flush()
                for field in episode_fields[1:]:
                    if field in row and field not in {
                        "training_act",
                        "encounter_id",
                        "encounter_type",
                    }:
                        self.logger.record_mean(f"combat/{field}", float(row[field]))
                self.logger.record_mean(
                    "combat/win_rate", float(info["is_success"])
                )

            if self.num_timesteps >= self.next_evaluation:
                metrics = _evaluate_training_policy(
                    self.model,
                    evaluation_encounters,
                    evaluation_seeds,
                    character_id=character_id,
                    reward_config=reward_config or RewardConfig(),
                    max_turns=max_turns,
                    training_act=evaluation_training_act,
                    training_acts=evaluation_training_acts,
                )
                row = {
                    "timesteps": self.num_timesteps,
                    "num_episodes": len(evaluation_seeds),
                    **metrics,
                }
                self.evaluation_writer.writerow(row)
                self.evaluation_file.flush()
                for name, value in metrics.items():
                    self.logger.record(f"evaluation/{name}", value)
                self.next_evaluation = self.num_timesteps + evaluation_interval
            return True

        def _on_training_end(self) -> None:
            self.card_performance.write_csv(run_dir / "card_performance.csv")
            if self.episode_file is not None:
                self.episode_file.close()
            if self.evaluation_file is not None:
                self.evaluation_file.close()

    return TrainingMetricsCallback()


def make_progress_callback(
    total_timesteps: int,
    *,
    log_interval: int = 10_000,
    start_timesteps: int = 0,
) -> BaseCallback:
    """Create a callback reporting session progress, metrics, and ETA."""
    if total_timesteps <= 0 or log_interval <= 0 or start_timesteps < 0:
        raise ValueError(
            "total_timesteps and log_interval must be positive, "
            "and start_timesteps must be non-negative"
        )

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise ImportError(
            "RL training requires the optional dependencies; install them with "
            "`uv sync --extra rl --extra notebook`."
        ) from exc

    class ProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.started_at = 0.0
            self.next_log = log_interval

        def _on_training_start(self) -> None:
            self.started_at = time.monotonic()

        def _on_step(self) -> bool:
            completed = max(0, self.num_timesteps - start_timesteps)
            if completed < self.next_log and completed < total_timesteps:
                return True

            elapsed = time.monotonic() - self.started_at
            fraction = min(completed / total_timesteps, 1.0)
            remaining = elapsed * (1.0 / fraction - 1.0) if fraction > 0 else 0.0
            episodes = list(self.model.ep_info_buffer)
            if episodes:
                mean_reward = float(np.mean([episode["r"] for episode in episodes]))
                win_rate = float(
                    np.mean([episode["is_success"] for episode in episodes])
                )
                mean_turns = float(np.mean([episode["turns"] for episode in episodes]))
                mean_damage_dealt = float(
                    np.mean([episode["damage_dealt"] for episode in episodes])
                )
                mean_damage_received = float(
                    np.mean([episode["damage_received"] for episode in episodes])
                )
                metrics = (
                    f"recent reward={mean_reward:.1f}, win rate={win_rate:.1%}, "
                    f"turns={mean_turns:.1f}, damage dealt/received="
                    f"{mean_damage_dealt:.1f}/{mean_damage_received:.1f}"
                )
            else:
                metrics = "awaiting first completed episode"

            print(
                f"Training {completed:,}/{total_timesteps:,} "
                f"({fraction:.1%}) | {metrics} | "
                f"elapsed {elapsed / 60:.1f} min | ETA {remaining / 60:.1f} min"
            )
            self.next_log = completed + log_interval
            return True

    return ProgressCallback()


def save_training_metadata(
    path: str | Path,
    *,
    character_id: str,
    encounters: Sequence[EncounterSetup] | None,
    reward_config: RewardConfig,
    seed: int,
    total_timesteps: int,
    training_log_dir: str | Path | None = None,
    run_log_dir: str | Path | None = None,
    training_acts: Sequence[int] | None = None,
    training_stages: Sequence[Sequence[int]] | None = None,
    completed_training_acts: Sequence[int] | None = None,
    current_training_act: int | None = None,
    current_training_stage: Sequence[int] | None = None,
    current_act_timesteps: int = 0,
) -> Path:
    """Save reproducibility metadata beside a trained policy."""
    metadata_path = Path(path)
    if metadata_path.suffix == ".zip":
        metadata_path = metadata_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "character_id": character_id,
        "encounters": [
            getattr(encounter, "__name__", type(encounter).__name__)
            for encounter in (encounters or ())
        ],
        "reward_config": asdict(reward_config),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_log_dir": (
            str(Path(training_log_dir).resolve())
            if training_log_dir is not None
            else None
        ),
        "run_log_dir": (
            str(Path(run_log_dir).resolve()) if run_log_dir is not None else None
        ),
    }
    if training_acts is not None:
        metadata["training_acts"] = list(training_acts)
    if training_stages is not None:
        metadata["training_stages"] = [list(stage) for stage in training_stages]
    if completed_training_acts is not None:
        metadata["completed_training_acts"] = list(completed_training_acts)
    if current_training_act is not None:
        metadata["current_training_act"] = current_training_act
    if current_training_stage is not None:
        metadata["current_training_stage"] = list(current_training_stage)
    metadata["current_act_timesteps"] = current_act_timesteps
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def load_training_metadata(path: str | Path) -> dict[str, Any]:
    """Load metadata saved beside a model archive or model stem."""
    metadata_path = Path(path)
    if metadata_path.suffix != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _policy_action_selector(model: Any) -> Callable[[CombatState, np.ndarray], int]:
    def select_action(combat: CombatState, action_mask: np.ndarray) -> int:
        action, _ = model.predict(
            encode_observation(combat),
            deterministic=True,
            action_masks=action_mask.astype(bool),
        )
        return int(np.asarray(action).item())

    return select_action


def evaluate_policy_against_heuristics(
    model: Any,
    encounters: Sequence[EncounterSetup],
    seeds: Sequence[int],
    *,
    character_id: str = "Ironclad",
    max_steps: int = 1_000,
    heuristics: Mapping[str, HeuristicName] | None = None,
) -> dict[str, list[CombatTrace]]:
    """Compare a trained policy and baselines on identical encounter/seed pairs."""
    if len(encounters) == 0:
        raise ValueError("Provide at least one encounter for evaluation")
    if len(seeds) == 0:
        raise ValueError("Provide at least one seed for evaluation")

    baselines = heuristics or {
        "Block first": "block_first",
        "Attack first": "attack_first",
        "Block without wasting": "block_without_wasting",
    }
    results: dict[str, list[CombatTrace]] = {
        "MaskablePPO": [],
        **{label: [] for label in baselines},
    }
    policy_selector = _policy_action_selector(model)

    for index, seed in enumerate(seeds):
        encounter = encounters[index % len(encounters)]
        results["MaskablePPO"].append(
            simulate_combat(
                encounter,
                character_id=character_id,
                seed=int(seed),
                action_selector=policy_selector,
                max_steps=max_steps,
            )
        )
        for label, heuristic in baselines.items():
            results[label].append(
                simulate_combat(
                    encounter,
                    character_id=character_id,
                    seed=int(seed),
                    heuristic=heuristic,
                    max_steps=max_steps,
                )
            )
    return results
