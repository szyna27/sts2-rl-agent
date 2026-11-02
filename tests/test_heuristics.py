"""Tests for simple combat heuristic selectors."""

import numpy as np

from sts2_env.encounters.act1 import setup_shrinker_beetle_weak
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.heuristics import (
    choose_heuristic_action,
    incoming_attack_damage,
    evaluate_heuristics,
    simulate_combat,
)


def test_incoming_damage_reads_current_enemy_intent():
    env = STS2CombatEnv(encounter_pool=[setup_shrinker_beetle_weak])
    _, info = env.reset(seed=7)
    assert env.combat is not None
    assert incoming_attack_damage(env.combat) >= 0
    action = choose_heuristic_action(env.combat, info["action_mask"], "block_first")
    assert np.asarray(info["action_mask"])[action] == 1


def test_heuristics_produce_a_complete_trace():
    trace = simulate_combat(
        setup_shrinker_beetle_weak,
        seed=7,
        heuristic="attack_first",
        max_steps=100,
    )
    assert trace.turns
    assert len(trace.turns) == len(trace.lost_health)
    assert trace.lost_health[0] == 0


def test_batch_evaluation_reuses_the_same_seed_set():
    seeds = [7, 11, 19]
    results = evaluate_heuristics(
        setup_shrinker_beetle_weak,
        seeds,
        {"Block first": "block_first", "Attack first": "attack_first"},
    )
    assert set(results) == {"Block first", "Attack first"}
    assert all(len(traces) == len(seeds) for traces in results.values())
    assert all(
        traces[0].lost_health == simulate_combat(
            setup_shrinker_beetle_weak, seed=seeds[0], heuristic=heuristic
        ).lost_health
        for traces, heuristic in (
            (results["Block first"], "block_first"),
            (results["Attack first"], "attack_first"),
        )
    )


def test_other_characters_use_their_starter_decks():
    env = STS2CombatEnv(
        encounter_pool=[setup_shrinker_beetle_weak],
        character_id="Silent",
    )
    env.reset(seed=7)
    assert env.combat is not None
    assert env.combat.character_id == "Silent"
    assert len(env.combat.draw_pile) + len(env.combat.hand) == 12
