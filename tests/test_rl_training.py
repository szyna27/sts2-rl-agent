"""Tests for configurable rewards and RL combat training helpers."""

import csv
import json

import numpy as np
import pytest

from sts2_env.encounters.act1 import (
    setup_fuzzy_wurm_crawler_weak,
    setup_shrinker_beetle_weak,
)
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.enums import CombatSide
from sts2_env.gym_env.action_space import action_to_card_and_target, is_potion_action
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.heuristics import CombatTrace, choose_heuristic_action
from sts2_env.rl_training import (
    CardPerformanceTracker,
    CombatTrainingEnv,
    RewardConfig,
    _evaluate_training_policy,
    compute_training_reward,
    evaluate_training_policy,
    evaluate_policy_against_heuristics,
    load_training_metadata,
    make_training_env,
    save_training_metadata,
)
from sts2_env.rl_training_plotter import plot_training_history, read_training_history


def test_reward_config_matches_requested_default_coefficients():
    assert RewardConfig() == RewardConfig(
        damage_dealt=1,
        damage_received=-3,
        win=50,
        death=-100,
        per_turn=-5,
        wasted_block=-0.5,
    )


def test_reward_components_apply_hp_terminal_turn_and_block_weights():
    breakdown = compute_training_reward(
        RewardConfig(),
        damage_dealt=8,
        damage_received=2,
        turns=3,
        wasted_block=4,
        won=True,
    )

    assert breakdown.damage_dealt == 8
    assert breakdown.damage_received == -6
    assert breakdown.turn_penalty == -15
    assert breakdown.wasted_block == -2
    assert breakdown.terminal == 50
    assert breakdown.total == 35


def test_reward_config_rejects_non_finite_coefficients():
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(damage_dealt=float("nan"))


def test_training_env_applies_turn_penalty_and_preserves_action_mask():
    env = CombatTrainingEnv(
        STS2CombatEnv(encounter_pool=[setup_shrinker_beetle_weak])
    )
    _, info = env.reset(seed=7)
    assert (env.action_masks() == info["action_mask"]).all()

    _, reward, terminated, truncated, info = env.step(0)

    assert info["reward_components"]["turn_penalty"] == -5
    assert reward == pytest.approx(sum(info["reward_components"].values()))
    assert terminated is False
    assert truncated is False


def test_training_reward_uses_actual_hp_damage_and_cleared_player_block():
    env = CombatTrainingEnv(
        STS2CombatEnv(encounter_pool=[setup_shrinker_beetle_weak])
    )
    env.reset(seed=7)
    combat = env.combat
    combat.player.block = 20
    damage_start = len(combat._damage_results_combat)
    action = choose_heuristic_action(
        combat,
        env.action_masks(),
        "attack_first",
    )

    _, _, _, _, info = env.step(action)
    damage_events = combat._damage_results_combat[damage_start:]
    actual_hp_damage = sum(
        result.hp_lost
        for dealer, result in damage_events
        if dealer is not None
        and dealer.side == CombatSide.PLAYER
        and result.target.side == CombatSide.ENEMY
    )
    assert actual_hp_damage > 0
    assert info["reward_components"]["damage_dealt"] == actual_hp_damage

    block_start = len(combat._block_clear_events_combat)
    _, _, _, _, info = env.step(0)
    player_block_cleared = sum(
        amount
        for creature, amount in combat._block_clear_events_combat[block_start:]
        if creature is combat.player
    )
    assert player_block_cleared > 0
    assert info["reward_components"]["wasted_block"] == -0.5 * player_block_cleared


def test_evaluation_runs_every_policy_on_the_same_encounter_seed_pairs(monkeypatch):
    import sts2_env.rl_training as rl_training

    calls = []

    def fake_simulate(
        encounter,
        *,
        seed,
        heuristic=None,
        action_selector=None,
        **ignored_options,
    ):
        assert ignored_options["character_id"] == "Ironclad"
        calls.append((encounter, seed, heuristic, action_selector is not None))
        return CombatTrace([1], [0], True, True)

    monkeypatch.setattr(rl_training, "simulate_combat", fake_simulate)
    results = evaluate_policy_against_heuristics(
        model=object(),
        encounters=[setup_shrinker_beetle_weak, setup_fuzzy_wurm_crawler_weak],
        seeds=[7, 11, 19],
    )

    assert set(results) == {
        "MaskablePPO",
        "Block first",
        "Attack first",
        "Block without wasting",
    }
    assert all(len(traces) == 3 for traces in results.values())
    for seed_index, seed in enumerate([7, 11, 19]):
        paired_calls = calls[seed_index * 4 : (seed_index + 1) * 4]
        assert {call[0] for call in paired_calls} == {
            [setup_shrinker_beetle_weak, setup_fuzzy_wurm_crawler_weak][seed_index % 2]
        }
        assert {call[1] for call in paired_calls} == {seed}
        assert sum(call[3] for call in paired_calls) == 1


def test_evaluation_accepts_a_mask_aware_policy_predictor():
    class EndTurnModel:
        def predict(self, observation, *, deterministic, action_masks):
            assert observation.ndim == 1
            assert deterministic is True
            assert action_masks.dtype == np.bool_
            assert action_masks.any()
            return np.array(0), None

    results = evaluate_policy_against_heuristics(
        model=EndTurnModel(),
        encounters=[setup_shrinker_beetle_weak],
        seeds=[7],
        max_steps=5,
    )

    assert all(len(traces) == 1 for traces in results.values())
    assert all(trace.terminated is False for trace in results["MaskablePPO"])


def test_save_training_metadata_records_reproducibility_config(tmp_path):
    metadata_path = save_training_metadata(
        tmp_path / "ironclad.zip",
        character_id="Ironclad",
        encounters=[setup_shrinker_beetle_weak, setup_fuzzy_wurm_crawler_weak],
        reward_config=RewardConfig(),
        seed=17,
        total_timesteps=50_000,
        training_log_dir=tmp_path / "logs",
        run_log_dir=tmp_path / "logs" / "session-1",
        training_acts=[1, 4, 2, 3],
        training_stages=[[1, 4], [2], [3]],
        completed_training_acts=[1, 4],
        current_training_act=2,
        current_training_stage=[2],
        current_act_timesteps=20_000,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_path.name == "ironclad.json"
    assert metadata["character_id"] == "Ironclad"
    assert metadata["encounters"] == [
        "setup_shrinker_beetle_weak",
        "setup_fuzzy_wurm_crawler_weak",
    ]
    assert metadata["reward_config"]["wasted_block"] == -0.5
    assert metadata["total_timesteps"] == 50_000
    assert metadata["training_acts"] == [1, 4, 2, 3]
    assert metadata["training_stages"] == [[1, 4], [2], [3]]
    assert metadata["completed_training_acts"] == [1, 4]
    assert metadata["current_training_act"] == 2
    assert metadata["current_training_stage"] == [2]
    assert metadata["current_act_timesteps"] == 20_000
    assert metadata["training_log_dir"] == str((tmp_path / "logs").resolve())
    assert metadata["run_log_dir"] == str(
        (tmp_path / "logs" / "session-1").resolve()
    )
    assert load_training_metadata(tmp_path / "ironclad.zip") == metadata


def test_periodic_evaluation_returns_fixed_pair_combat_metrics():
    class EndTurnModel:
        def predict(self, observation, *, deterministic, action_masks):
            assert deterministic is True
            return np.flatnonzero(action_masks)[-1], None

    metrics = _evaluate_training_policy(
        EndTurnModel(),
        [setup_shrinker_beetle_weak],
        [7, 11],
        character_id="Ironclad",
        reward_config=RewardConfig(),
        max_turns=1,
    )

    assert set(metrics) == {
        "mean_reward",
        "win_rate",
        "mean_turns",
        "mean_damage_dealt",
        "mean_damage_received",
        "mean_wasted_block",
    }
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["mean_turns"] <= 1
    assert metrics["mean_damage_dealt"] >= 0
    assert metrics["mean_damage_received"] >= 0


def test_training_env_exposes_episode_total_reward_components():
    env = CombatTrainingEnv(
        STS2CombatEnv(
            encounter_pool=[setup_shrinker_beetle_weak],
            max_turns=1,
        )
    )
    env.reset(seed=7)
    component_totals = {}

    for _ in range(10):
        action = int(np.flatnonzero(env.action_masks())[-1])
        _, _, terminated, truncated, info = env.step(action)
        for name, value in info["reward_components"].items():
            component_totals[name] = component_totals.get(name, 0) + value
        if terminated or truncated:
            break
    else:
        pytest.fail("Combat did not finish within one turn")

    assert {
        f"reward_{name}": value for name, value in component_totals.items()
    } == pytest.approx(
        {f"reward_{name}": info[f"reward_{name}"] for name in component_totals}
    )
    assert info["damage_dealt"] >= 0
    assert info["damage_received"] >= 0
    env.close()


def test_training_env_reports_encounter_deck_and_played_cards():
    env = CombatTrainingEnv(
        STS2CombatEnv(
            encounter_pool=[setup_shrinker_beetle_weak],
            training_act=1,
            max_turns=1,
        )
    )
    try:
        env.reset(seed=7)
        combat = env.combat
        card_action = None
        played_card = None
        for action in np.flatnonzero(env.action_masks()):
            action = int(action)
            if action == ACTION_END_TURN or is_potion_action(action):
                continue
            hand_index, _ = action_to_card_and_target(action)
            if hand_index is None:
                continue
            card = combat.hand[hand_index]
            if card.card_id.name in {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}:
                card_action = action
                played_card = (card.card_id.name, card.upgraded)
                break
        assert card_action is not None
        assert played_card is not None

        finished_info = None
        for action in (card_action, ACTION_END_TURN):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                finished_info = info
                break

        assert finished_info is not None
        assert finished_info["training_act"] == 1
        assert finished_info["encounter_id"] == "setup_shrinker_beetle_weak"
        assert finished_info["encounter_type"] == "weak"
        assert any(
            card["card_id"] == played_card[0]
            and card["upgraded"] is played_card[1]
            for card in finished_info["cards_in_deck"]
        )
        assert any(
            card["card_id"] == played_card[0]
            and card["upgraded"] is played_card[1]
            and card["count"] == 1
            for card in finished_info["cards_played"]
        )
    finally:
        env.close()


def test_card_performance_tracker_separates_encounters_and_act_type_totals(tmp_path):
    tracker = CardPerformanceTracker()
    common_deck = [
        {"card_id": "BASH", "upgraded": False, "count": 1},
        {"card_id": "STRIKE_IRONCLAD", "upgraded": False, "count": 2},
    ]
    tracker.record_episode(
        {
            "training_act": 1,
            "encounter_id": "weak_encounter_a",
            "encounter_type": "weak",
            "damage_received": 4,
            "is_success": True,
            "cards_in_deck": common_deck,
            "cards_played": [
                {"card_id": "BASH", "upgraded": False, "count": 2}
            ],
        }
    )
    tracker.record_episode(
        {
            "training_act": 1,
            "encounter_id": "weak_encounter_b",
            "encounter_type": "weak",
            "damage_received": 8,
            "is_success": False,
            "cards_in_deck": common_deck,
            "cards_played": [],
        }
    )
    tracker.record_episode(
        {
            "training_act": 2,
            "encounter_id": "normal_encounter",
            "encounter_type": "normal",
            "damage_received": 10,
            "is_success": False,
            "cards_in_deck": common_deck,
            "cards_played": [],
        }
    )

    output_path = tmp_path / "card_performance.csv"
    tracker.write_csv(output_path)
    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    act1_weak_bash = next(
        row
        for row in rows
        if row["scope"] == "act_type"
        and row["act"] == "1"
        and row["encounter_type"] == "weak"
        and row["card_id"] == "BASH"
    )
    assert act1_weak_bash["encounter_id"] == "ALL"
    assert act1_weak_bash["deck_battles"] == "2"
    assert act1_weak_bash["played_battles"] == "1"
    assert act1_weak_bash["play_count"] == "2"
    assert float(act1_weak_bash["mean_hp_lost_in_deck"]) == 6.0
    assert float(act1_weak_bash["mean_hp_lost_when_played"]) == 4.0

    encounter_a_bash = next(
        row
        for row in rows
        if row["scope"] == "encounter"
        and row["encounter_id"] == "weak_encounter_a"
        and row["card_id"] == "BASH"
    )
    assert encounter_a_bash["deck_battles"] == "1"
    assert float(encounter_a_bash["mean_hp_lost_in_deck"]) == 4.0


def test_training_env_can_sample_an_act_without_explicit_encounters():
    pytest.importorskip("stable_baselines3")
    env = make_training_env(training_act=2)
    try:
        env.reset(seed=7)
        combat = env.unwrapped.combat
        assert combat is not None
        assert len(combat.current_player_state.player_state.deck) > 10
    finally:
        env.close()


def test_training_env_alternates_acts_within_one_stage():
    pytest.importorskip("stable_baselines3")
    env = make_training_env(training_acts=[1, 4])
    try:
        env.reset(seed=7)
        first_act = env.unwrapped.current_training_act
        env.reset()
        second_act = env.unwrapped.current_training_act

        assert {first_act, second_act} == {1, 4}
    finally:
        env.close()


def test_training_env_requires_encounters_or_an_act():
    with pytest.raises(ValueError, match="Provide encounters"):
        make_training_env()


def test_evaluate_training_policy_supports_act_sampling():
    class EndTurnModel:
        def predict(self, observation, *, deterministic, action_masks):
            return np.array(0), None

    metrics = evaluate_training_policy(
        EndTurnModel(),
        seeds=[7],
        training_acts=[1, 4],
        max_turns=1,
    )

    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["mean_turns"] <= 1


def test_evaluation_reuses_environment_to_alternate_training_acts(monkeypatch):
    import sts2_env.rl_training as rl_training

    reset_acts = []
    original_training_env = rl_training.CombatTrainingEnv

    class TrackingTrainingEnv(original_training_env):
        def reset(self, *, seed=None, options=None):
            result = super().reset(seed=seed, options=options)
            reset_acts.append(self.unwrapped.current_training_act)
            return result

    class EndTurnModel:
        def predict(self, observation, *, deterministic, action_masks):
            return np.flatnonzero(action_masks)[-1], None

    monkeypatch.setattr(rl_training, "CombatTrainingEnv", TrackingTrainingEnv)
    evaluate_training_policy(
        EndTurnModel(),
        seeds=[7, 11],
        training_acts=[1, 4],
        max_turns=1,
    )

    assert len(reset_acts) == 2
    assert reset_acts[0] != reset_acts[1]


def test_training_history_reads_persisted_csv_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    import csv

    session_dir = tmp_path / "session-1"
    session_dir.mkdir()
    episode_fields = [
        "timesteps",
        "reward",
        "length",
        "is_success",
        "turns",
        "damage_dealt",
        "damage_received",
        "wasted_block",
        "reward_damage_dealt",
        "reward_damage_received",
        "reward_turn_penalty",
        "reward_wasted_block",
        "reward_terminal",
    ]
    with (session_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerow(
            {
                "timesteps": 100,
                "reward": 10,
                "length": 8,
                "is_success": 1,
                "turns": 2,
                "damage_dealt": 15,
                "damage_received": 1,
                "wasted_block": 0,
                "reward_damage_dealt": 15,
                "reward_damage_received": -3,
                "reward_turn_penalty": -10,
                "reward_wasted_block": 0,
                "reward_terminal": 50,
            }
        )
    with (session_dir / "evaluations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timesteps",
                "num_episodes",
                "mean_reward",
                "win_rate",
                "mean_turns",
                "mean_damage_dealt",
                "mean_damage_received",
                "mean_wasted_block",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timesteps": 100,
                "num_episodes": 2,
                "mean_reward": 5,
                "win_rate": 0.5,
                "mean_turns": 3,
                "mean_damage_dealt": 8,
                "mean_damage_received": 2,
                "mean_wasted_block": 1,
            }
        )

    history = read_training_history(tmp_path)
    assert len(history["episodes"]) == 1
    assert history["episodes"][0]["session"] == "session-1"
    assert history["episodes"][0]["reward"] == 10
    assert history["evaluations"][0]["win_rate"] == 0.5

    figure, axes = plot_training_history(tmp_path, window=10, show=False)
    assert figure is not None
    assert axes.shape == (2, 3)


def test_training_metrics_callback_persists_episode_and_evaluation_rows(
    tmp_path, monkeypatch
):
    import csv
    import sys
    import types

    import sts2_env.rl_training as rl_training

    class FakeLogger:
        def __init__(self):
            self.values = []

        def record_mean(self, key, value):
            self.values.append((key, value))

        def record(self, key, value):
            self.values.append((key, value))

    class FakeBaseCallback:
        def __init__(self, verbose=0):
            self.logger = FakeLogger()

    stable_baselines = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    callbacks = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks.BaseCallback = FakeBaseCallback
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.callbacks", callbacks
    )
    evaluation_calls = []

    def fake_evaluate(*args, **kwargs):
        evaluation_calls.append(kwargs)
        return {
            "mean_reward": 4.0,
            "win_rate": 0.5,
            "mean_turns": 2.0,
            "mean_damage_dealt": 8.0,
            "mean_damage_received": 3.0,
            "mean_wasted_block": 1.0,
        }

    monkeypatch.setattr(rl_training, "_evaluate_training_policy", fake_evaluate)

    callback = rl_training.make_training_metrics_callback(
        tmp_path,
        evaluation_seeds=[7, 11],
        evaluation_interval=2,
        evaluation_training_act=2,
    )
    callback.model = object()
    callback.locals = {
        "infos": [
            {
                "episode": {"r": 10.0, "l": 8},
                "is_success": True,
                "training_act": 2,
                "encounter_id": "test_encounter",
                "encounter_type": "normal",
                "turns": 2,
                "damage_dealt": 15,
                "damage_received": 1,
                "wasted_block": 0,
                "cards_in_deck": [
                    {"card_id": "BASH", "upgraded": False, "count": 1}
                ],
                "cards_played": [
                    {"card_id": "BASH", "upgraded": False, "count": 1}
                ],
                "reward_damage_dealt": 15.0,
                "reward_damage_received": -3.0,
                "reward_turn_penalty": -10.0,
                "reward_wasted_block": 0.0,
                "reward_terminal": 50.0,
            }
        ]
    }

    callback._on_training_start()
    callback.num_timesteps = 1
    assert callback._on_step() is True
    callback.num_timesteps = 2
    callback.locals = {"infos": []}
    assert callback._on_step() is True
    callback._on_training_end()

    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as file:
        episodes = list(csv.DictReader(file))
    with (tmp_path / "evaluations.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        evaluations = list(csv.DictReader(file))
    assert len(episodes) == 1
    assert episodes[0]["reward_damage_dealt"] == "15.0"
    assert episodes[0]["training_act"] == "2"
    with (tmp_path / "card_performance.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        card_metrics = list(csv.DictReader(file))
    assert any(
        row["scope"] == "act_type"
        and row["act"] == "2"
        and row["encounter_type"] == "normal"
        and row["card_id"] == "BASH"
        and row["mean_hp_lost_in_deck"] == "1.0"
        and row["mean_hp_lost_when_played"] == "1.0"
        for row in card_metrics
    )
    assert len(evaluations) == 1
    assert evaluations[0]["win_rate"] == "0.5"
    assert evaluation_calls[0]["training_act"] == 2
