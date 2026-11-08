"""Read and plot persistent reinforcement-learning training metrics."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

import numpy as np

TrainingRow: TypeAlias = dict[str, float | str]
TrainingHistory: TypeAlias = dict[str, list[TrainingRow]]


def _read_csv_tree(log_root: Path, filename: str) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for path in sorted(log_root.rglob(filename)):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for raw_row in reader:
                row: TrainingRow = {"session": path.parent.name}
                for key, value in raw_row.items():
                    if key is not None and value is not None:
                        row[key] = float(value)
                rows.append(row)
    return rows


def read_training_history(log_root: str | Path) -> TrainingHistory:
    """Load episode and periodic evaluation CSVs from a log root."""
    root = Path(log_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Training log directory does not exist: {root}")

    episodes = _read_csv_tree(root, "episodes.csv")
    if not episodes:
        raise FileNotFoundError(f"No episode history found under {root}")

    required_episode_fields = {
        "timesteps",
        "reward",
        "is_success",
        "turns",
        "damage_dealt",
        "damage_received",
        "reward_damage_dealt",
        "reward_damage_received",
        "reward_turn_penalty",
        "reward_wasted_block",
        "reward_terminal",
    }
    if not required_episode_fields.issubset(episodes[0]):
        missing = sorted(required_episode_fields - episodes[0].keys())
        raise ValueError(f"Episode history is missing required columns: {missing}")

    evaluations = _read_csv_tree(root, "evaluations.csv")
    return {"episodes": episodes, "evaluations": evaluations}


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    width = min(window, len(values))
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_training_history(
    log_root: str | Path,
    *,
    window: int = 50,
    show: bool = True,
):
    """Plot episode metrics, reward components, and periodic evaluation results."""
    if window <= 0:
        raise ValueError("window must be positive")

    history = read_training_history(log_root)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    colors = plt.get_cmap("tab10")
    grouped_episodes: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in history["episodes"]:
        grouped_episodes[str(row["session"])].append(row)

    for color_index, (session, rows) in enumerate(grouped_episodes.items()):
        rows.sort(key=lambda row: float(row["timesteps"]))
        timesteps = np.array([float(row["timesteps"]) for row in rows])
        color = colors(color_index % 10)

        reward = np.array([float(row["reward"]) for row in rows])
        reward_mean = _rolling_mean(reward, window)
        axes[0, 0].plot(
            timesteps[len(timesteps) - len(reward_mean) :],
            reward_mean,
            color=color,
            label=session,
        )

        wins = np.array([float(row["is_success"]) for row in rows])
        win_mean = _rolling_mean(wins, window) * 100
        axes[0, 1].plot(
            timesteps[len(timesteps) - len(win_mean) :],
            win_mean,
            color=color,
            label=session,
        )

        for field, style, label in (
            ("damage_dealt", "-", "dealt"),
            ("damage_received", "--", "received"),
        ):
            values = np.array([float(row[field]) for row in rows])
            mean = _rolling_mean(values, window)
            axes[0, 2].plot(
                timesteps[len(timesteps) - len(mean) :],
                mean,
                linestyle=style,
                color=color,
                label=f"{session} {label}",
            )

        turns = np.array([float(row["turns"]) for row in rows])
        turns_mean = _rolling_mean(turns, window)
        axes[1, 0].plot(
            timesteps[len(timesteps) - len(turns_mean) :],
            turns_mean,
            color=color,
            label=session,
        )

        for field, label in (
            ("reward_damage_dealt", "damage dealt"),
            ("reward_damage_received", "damage received"),
            ("reward_turn_penalty", "turn penalty"),
            ("reward_wasted_block", "wasted block"),
            ("reward_terminal", "terminal"),
        ):
            values = np.array([float(row[field]) for row in rows])
            mean = _rolling_mean(values, window)
            axes[1, 1].plot(
                timesteps[len(timesteps) - len(mean) :],
                mean,
                color=colors(
                    (
                        color_index * 5
                        + (
                            "reward_damage_dealt",
                            "reward_damage_received",
                            "reward_turn_penalty",
                            "reward_wasted_block",
                            "reward_terminal",
                        ).index(field)
                    )
                    % 10
                ),
                label=f"{session}: {label}",
                alpha=0.8,
            )

    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_ylabel("%")
    axes[1, 1].set_title("Reward components (rolling mean)")

    grouped_evaluations: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in history["evaluations"]:
        grouped_evaluations[str(row["session"])].append(row)
    if grouped_evaluations:
        for color_index, (session, rows) in enumerate(grouped_evaluations.items()):
            rows.sort(key=lambda row: float(row["timesteps"]))
            timesteps = [float(row["timesteps"]) for row in rows]
            color = colors(color_index % 10)
            axes[0, 0].plot(
                timesteps,
                [float(row["mean_reward"]) for row in rows],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{session}: evaluation",
            )
            axes[0, 1].plot(
                timesteps,
                [float(row["win_rate"]) * 100 for row in rows],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{session}: evaluation",
            )
            axes[0, 2].plot(
                timesteps,
                [float(row["mean_damage_received"]) for row in rows],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{session}: evaluation received",
            )
            axes[1, 0].plot(
                timesteps,
                [float(row["mean_turns"]) for row in rows],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{session}: evaluation",
            )
            axes[1, 2].plot(
                timesteps,
                [float(row["win_rate"]) * 100 for row in rows],
                marker="o",
                color=color,
                label=f"{session}: win rate",
            )
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].set_ylabel("Win rate (%)")
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "No periodic evaluation data",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )

    titles = (
        "Episode reward (rolling mean)",
        "Win rate (rolling mean)",
        "Damage dealt / received (rolling mean)",
        "Combat turns (rolling mean)",
        "Reward components (rolling mean)",
        "Periodic evaluation",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Training timesteps")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize="small")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
