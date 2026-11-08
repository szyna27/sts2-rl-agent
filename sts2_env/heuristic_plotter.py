"""Plot summaries for combat heuristic evaluations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from sts2_env.heuristics import CombatTrace


class CombatHeuristicPlotter:
    """Draw cumulative lost-HP distributions from heuristic combat results."""

    @staticmethod
    def plot(results: Mapping[str, Sequence[CombatTrace]]) -> tuple[Figure, Axes]:
        """Plot violin-like distributions and boxplots using only combat results."""
        if not results or any(not traces for traces in results.values()):
            raise ValueError("Combat results must contain at least one trace per heuristic")

        data = [
            [trace.lost_health[-1] for trace in traces]
            for traces in results.values()
        ]
        labels = list(results)
        positions = range(1, len(labels) + 1)
        colors = ["#1976D2", "#D32F2F", "#7B1FA2"]

        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (y_pos, strategy_data) in enumerate(zip(positions, data)):
            counts = Counter(strategy_data)
            x_vals = np.arange(min(0, min(strategy_data)), max(strategy_data) + 1)
            y_counts = np.array([counts.get(x, 0) for x in x_vals])
            max_count = y_counts.max() if y_counts.size else 1
            scaled_counts = (y_counts / max_count) * 0.45

            ax.fill_between(
                x_vals,
                y_pos - scaled_counts,
                y_pos + scaled_counts,
                facecolor=colors[i % len(colors)],
                alpha=0.3,
            )

        ax.boxplot(
            data,
            orientation="horizontal",
            showmeans=True,
            patch_artist=True,
            widths=0.35,
            meanprops={
                "marker": "^",
                "markerfacecolor": "#2ca02c",
                "markeredgecolor": "#2ca02c",
                "markersize": 8,
            },
            medianprops={"color": "orange", "linewidth": 3},
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
        )

        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cumulative lost HP")
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        fig.tight_layout()
        plt.show()
        return fig, ax
