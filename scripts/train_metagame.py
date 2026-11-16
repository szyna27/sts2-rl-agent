"""Train the metagame state-value model on complete simulated runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.metagame_training import (
    make_heuristic_combat_policy,
    make_maskable_ppo_combat_policy,
    save_trained_metagame,
    train_metagame,
)
from sts2_env.heuristics import HeuristicName


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a full-run metagame model with a frozen combat policy."
    )
    combat = parser.add_mutually_exclusive_group(required=True)
    combat.add_argument(
        "--combat-model",
        help="Path to a trained MaskablePPO combat model archive.",
    )
    combat.add_argument(
        "--combat-heuristic",
        choices=("attack_first", "block_first", "block_without_wasting"),
        help="Use a fixed combat heuristic instead of a trained combat model.",
    )
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/metagame_state_value.pt"),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")

    if args.combat_model:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise SystemExit(
                "Install RL dependencies with `uv sync --extra rl`."
            ) from exc
        combat_model = MaskablePPO.load(args.combat_model)
        combat_policy = make_maskable_ppo_combat_policy(combat_model)
        combat_policy_name = str(Path(args.combat_model).resolve())
    else:
        heuristic = _parse_heuristic(args.combat_heuristic)
        combat_policy = make_heuristic_combat_policy(heuristic)
        combat_policy_name = f"heuristic:{args.combat_heuristic}"

    def report_progress(
        episode: int,
        episode_return: float,
        won: bool,
        decisions: int,
        updates: int,
    ) -> None:
        if episode % args.log_interval == 0 or episode == 1:
            logging.info(
                "episode=%d/%d return=%.1f won=%s decisions=%d updates=%d",
                episode,
                args.episodes,
                episode_return,
                won,
                decisions,
                updates,
            )

    model, encoder, result = train_metagame(
        episodes=args.episodes,
        combat_policy=combat_policy,
        character_id=args.character,
        ascension_level=args.ascension,
        seed=args.seed,
        progress_callback=report_progress,
    )
    path = save_trained_metagame(
        str(args.output),
        model,
        encoder,
        result,
        character_id=args.character,
        ascension_level=args.ascension,
        combat_model=combat_policy_name,
        seed=args.seed,
    )
    win_rate = sum(result.episode_wins) / len(result.episode_wins)
    logging.info(
        "Saved %s (episodes=%d win_rate=%.3f mean_return=%.2f)",
        path,
        len(result.episode_returns),
        win_rate,
        sum(result.episode_returns) / len(result.episode_returns),
    )


def _parse_heuristic(value: str | None) -> HeuristicName:
    if value == "attack_first":
        return "attack_first"
    if value == "block_first":
        return "block_first"
    if value == "block_without_wasting":
        return "block_without_wasting"
    raise ValueError(f"Unknown combat heuristic: {value!r}")


if __name__ == "__main__":
    main()
