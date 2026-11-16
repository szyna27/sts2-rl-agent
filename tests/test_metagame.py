"""Tests for the full-run metagame state-value model."""

import numpy as np
import pytest
import torch

from sts2_env.metagame import (
    MetagameAgent,
    MetagameStateEncoder,
    StateValueNetwork,
    save_metagame_checkpoint,
    simulate_candidate_states,
)
from sts2_env.metagame_training import (
    compute_metagame_reward,
    train_metagame,
)
from sts2_env.run.run_manager import RunManager


def test_encoder_represents_full_deck_relics_and_run_context():
    manager = RunManager(seed=1)
    encoder = MetagameStateEncoder()

    observation = encoder.encode_manager(manager)

    assert observation.shape == (encoder.observation_size,)
    assert observation.dtype == np.float32
    assert observation[encoder.card_offsets["strikeironclad"]] == pytest.approx(0.5)
    relic_start = len(encoder.card_ids)
    boss_start = relic_start + len(encoder.relic_ids) + 8 + 3
    character_start = boss_start + len(encoder.boss_ids)
    assert observation[relic_start + encoder.relic_offsets["burningblood"]] == 1.0
    assert observation[boss_start + encoder.boss_offsets["vantom"]] == 1.0
    assert observation[character_start + encoder.character_offsets["ironclad"]] == 1.0
    assert np.isfinite(observation).all()


def test_encoder_rejects_ids_not_in_the_model_schema():
    encoder = MetagameStateEncoder()

    with pytest.raises(ValueError, match="Unknown card ID"):
        encoder.encode_context({"deck": ["UNKNOWN_CARD"]})


def test_metagame_reward_shapes_floor_elite_boss_and_terminal_outcomes():
    assert compute_metagame_reward(
        floors_advanced=1,
        elite_won=True,
        boss_won=True,
        run_won=True,
    ) == 161.0
    assert compute_metagame_reward(died=True) == -50.0

    with pytest.raises(ValueError, match="non-negative"):
        compute_metagame_reward(floors_advanced=-1)


def test_candidate_simulation_does_not_mutate_the_live_run():
    manager = RunManager(seed=11)
    encoder = MetagameStateEncoder()
    initial_observation = encoder.encode_manager(manager)
    initial_deck = [card.card_id for card in manager.run_state.player.deck]

    candidates = simulate_candidate_states(manager, encoder=encoder)

    assert candidates
    assert manager.phase == RunManager.PHASE_MAP_CHOICE
    assert [card.card_id for card in manager.run_state.player.deck] == initial_deck
    np.testing.assert_array_equal(
        encoder.encode_manager(manager),
        initial_observation,
    )
    assert any(not np.array_equal(candidate, initial_observation) for candidate in candidates)


def test_candidate_simulation_rebinds_pending_choice_callbacks():
    manager = RunManager(seed=13)
    player = manager.run_state.player
    encoder = MetagameStateEncoder()

    def make_resolver(target_player):
        def resolver(_selected):
            target_player.current_hp -= 1

        return resolver

    player.request_deck_choice(
        prompt="Choose a card",
        cards=player.deck[:2],
        resolver=make_resolver(player),
        require_manual_confirmation=True,
    )

    candidates = simulate_candidate_states(manager, encoder=encoder)

    assert candidates
    assert player.current_hp == player.max_hp
    assert manager.run_state.pending_choice is not None
    hp_ratio_index = len(encoder.card_ids) + len(encoder.relic_ids)
    assert candidates[0][hp_ratio_index] == pytest.approx(
        (player.max_hp - 1) / player.max_hp
    )


def test_agent_selects_a_legal_candidate_action():
    manager = RunManager(seed=17)
    encoder = MetagameStateEncoder()
    model = StateValueNetwork(encoder.observation_size, hidden_size=8)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    agent = MetagameAgent(model, encoder)

    action, candidate_state = agent.choose_action(manager)

    assert action in manager.get_available_actions()
    assert candidate_state.shape == (encoder.observation_size,)


def test_bridge_option_scoring_uses_shared_state_encoder():
    encoder = MetagameStateEncoder()
    model = StateValueNetwork(encoder.observation_size, hidden_size=8)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    agent = MetagameAgent(model, encoder)
    state = {
        "type": "card_reward",
        "can_skip": True,
        "cards": [
            {"index": 0, "id": "BASH", "type": "Attack"},
            {"index": 1, "id": "INFLAME", "type": "Power"},
        ],
        "metagame_state": {
            "character_id": "Ironclad",
            "deck": ["STRIKE_IRONCLAD"],
            "relics": ["BURNING_BLOOD"],
            "hp": 70,
            "max_hp": 80,
            "gold": 99,
            "floor": 3,
            "act_index": 0,
            "act_floor": 3,
            "keys_to_act4": 0,
            "ascension": 0,
            "boss_id": "VantomBoss",
        },
    }

    choice = agent.choose_bridge_option(state)

    assert choice == 0


def test_training_runs_complete_episodes_with_a_frozen_combat_policy():
    model, encoder, result = train_metagame(
        episodes=1,
        combat_policy=lambda _: 0,
        seed=3,
        batch_size=4,
        replay_capacity=32,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_decisions=1,
        target_update_interval=2,
        max_combat_turns=1,
        hidden_size=8,
    )

    assert model.observation_size == encoder.observation_size
    assert len(result.episode_returns) == 1
    assert len(result.episode_wins) == 1
    assert result.decision_counts[0] > 0
    assert result.updates > 0


def test_checkpoint_round_trip_preserves_schema_and_predictions(tmp_path):
    encoder = MetagameStateEncoder()
    model = StateValueNetwork(encoder.observation_size, hidden_size=8)
    observation = encoder.encode_manager(RunManager(seed=23))
    expected = model(torch.from_numpy(observation)).item()
    path = save_metagame_checkpoint(
        tmp_path / "metagame.pt",
        model,
        encoder,
        training_metadata={"combat_policy_frozen": True},
    )

    loaded = MetagameAgent.load(path)

    assert loaded.score_states([observation])[0] == pytest.approx(expected)
