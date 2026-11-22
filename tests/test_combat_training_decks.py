import pytest

from sts2_env.core.enums import CardRarity, CardType
from sts2_env.core.rng import Rng
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.deck_builder import (
    _choose_early_act_removal_count,
    build_training_deck,
)


@pytest.mark.parametrize("act", [1, 2, 3, 4])
@pytest.mark.parametrize("encounter_type", ["weak", "normal", "elite", "boss"])
def test_training_deck_keeps_damage_cards_and_only_adds_expected_curses(act, encounter_type):
    deck = build_training_deck(
        "Ironclad",
        act,
        encounter_type,
        Rng(seed=act * 100 + len(encounter_type)),
    )

    assert sum(card.card_type is CardType.ATTACK for card in deck) >= 3
    curses = [card for card in deck if card.is_curse]
    if act in {1, 4}:
        assert all(card.card_id.name == "GREED" for card in curses)
    else:
        assert all(card.rarity is CardRarity.CURSE for card in curses)


@pytest.mark.parametrize("act", [1, 2, 3, 4])
def test_training_act_uses_generated_deck(act):
    env = STS2CombatEnv(training_act=act)
    env.reset(seed=7)

    deck = env.combat.current_player_state.player_state.deck
    assert sum(card.card_type is CardType.ATTACK for card in deck) >= 3
    assert env.combat.enemies


def test_training_environment_rejects_unknown_acts():
    with pytest.raises(ValueError, match="Unsupported training_act"):
        STS2CombatEnv(training_act=5)


def test_training_acts_choose_a_random_start_then_alternate():
    first_acts = set()
    for seed in range(12):
        env = STS2CombatEnv(training_acts=[1, 4])
        env.reset(seed=seed)
        first_act = env.current_training_act
        first_acts.add(first_act)

        env.reset()
        assert env.current_training_act != first_act
        env.reset()
        assert env.current_training_act == first_act
        env.close()

    assert first_acts == {1, 4}


def test_training_acts_reject_custom_encounter_pools():
    with pytest.raises(ValueError, match="custom encounter_pool"):
        STS2CombatEnv(
            encounter_pool=[],
            training_acts=[1, 4],
        )


@pytest.mark.parametrize(
    ("encounter_type", "expected_one", "expected_two"),
    [
        ("weak", 0.15, 0.10),
        ("normal", 0.25, 0.15),
        ("elite", 0.25, 0.15),
        ("boss", 0.50, 0.25),
    ],
)
def test_early_act_removal_probabilities_are_disjoint(
    encounter_type, expected_one, expected_two
):
    rng = Rng(seed=991)
    sample_count = 10_000
    removals = [
        _choose_early_act_removal_count(encounter_type, rng)
        for _ in range(sample_count)
    ]

    assert removals.count(1) / sample_count == pytest.approx(expected_one, abs=0.01)
    assert removals.count(2) / sample_count == pytest.approx(expected_two, abs=0.01)

