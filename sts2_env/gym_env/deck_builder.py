"""Randomized combat-training decks based on act and encounter difficulty."""

from __future__ import annotations

from dataclasses import dataclass, replace

from sts2_env.cards.base import CardInstance
from sts2_env.cards.factory import create_card, create_character_cards
from sts2_env.core.card_pools import CURSE_CARD_POOL
from sts2_env.core.enums import CardRarity, CardTag, CardType
from sts2_env.core.rng import Rng


@dataclass(frozen=True)
class _DeckProfile:
    remove_min: int = 0
    remove_max: int = 0
    common_min: int = 0
    common_max: int = 0
    uncommon_min: int = 0
    uncommon_max: int = 0
    uncommon_chance: float = 1.0
    rare_min: int = 0
    rare_max: int = 0
    rare_chance: float = 1.0
    upgrade_chance: float = 0.0
    max_upgrades: int | None = None
    curse_chance: float = 0.0
    curse_min: int = 1
    curse_max: int = 1


_EARLY_ACT_PROFILES = {
    "weak": _DeckProfile(
        common_max=3, uncommon_min=1, uncommon_max=1, uncommon_chance=0.25,
        rare_min=1, rare_max=1, rare_chance=0.05, upgrade_chance=0.05,
        max_upgrades=1, curse_chance=0.10,
    ),
    "normal": _DeckProfile(
        common_min=2, common_max=5, uncommon_min=1, uncommon_max=2,
        rare_min=1, rare_max=1, rare_chance=0.15, upgrade_chance=0.10,
        max_upgrades=2, curse_chance=0.10,
    ),
    "elite": _DeckProfile(
        common_max=5, uncommon_max=2, uncommon_chance=1.0,
        rare_min=1, rare_max=1, rare_chance=0.10, upgrade_chance=0.10,
        max_upgrades=2, curse_chance=0.10,
    ),
    "boss": _DeckProfile(
        common_min=4, common_max=6, uncommon_min=1, uncommon_max=3,
        rare_min=1, rare_max=1, rare_chance=0.50, upgrade_chance=0.15,
        max_upgrades=3, curse_chance=0.10,
    ),
}

_LATER_ACT_PROFILES = {
    2: {
        "small": _DeckProfile(
            3, 5, 2, 4, 2, 3, rare_min=1, rare_max=2,
            upgrade_chance=0.35, curse_chance=0.15,
        ),
        "medium": _DeckProfile(
            1, 3, 5, 8, 3, 5, rare_min=1, rare_max=3,
            upgrade_chance=0.25, curse_chance=0.25,
        ),
        "large": _DeckProfile(
            0, 1, 8, 12, 5, 8, rare_min=2, rare_max=4,
            upgrade_chance=0.20, curse_chance=0.35, curse_min=1, curse_max=2,
        ),
    },
    3: {
        "small": _DeckProfile(
            5, 8, 2, 5, 2, 4, rare_min=2, rare_max=3,
            upgrade_chance=0.50, curse_chance=0.20,
        ),
        "medium": _DeckProfile(
            2, 4, 6, 10, 4, 7, rare_min=2, rare_max=4,
            upgrade_chance=0.40, curse_chance=0.30, curse_min=1, curse_max=2,
        ),
        "large": _DeckProfile(
            0, 2, 10, 16, 6, 10, rare_min=3, rare_max=6,
            upgrade_chance=0.30, curse_chance=0.45, curse_min=1, curse_max=2,
        ),
    },
}

_ARCHETYPE_WEIGHTS = (("small", 30), ("medium", 45), ("large", 25))
_MIN_DAMAGE_CARDS = 3


def _randint(rng: Rng, low: int, high: int) -> int:
    return rng.next_int(low, high)


def _roll(rng: Rng, probability: float) -> bool:
    return rng.next_float() < probability


def _choose_archetype(rng: Rng) -> str:
    draw = rng.next_int(1, 100)
    for archetype, weight in _ARCHETYPE_WEIGHTS:
        draw -= weight
        if draw <= 0:
            return archetype
    raise AssertionError("Archetype weights must sum to 100")


def _choose_early_act_removal_count(encounter_type: str, rng: Rng) -> int:
    one_card_chance = {
        "weak": 0.15,
        "normal": 0.25,
        "elite": 0.25,
        "boss": 0.50,
    }[encounter_type]
    two_card_chance = {
        "weak": 0.10,
        "normal": 0.15,
        "elite": 0.15,
        "boss": 0.25,
    }[encounter_type]
    roll = rng.next_float()
    if roll < one_card_chance:
        return 1
    if roll < one_card_chance + two_card_chance:
        return 2
    return 0


def _remove_starting_cards(deck: list[CardInstance], count: int, rng: Rng) -> list[CardInstance]:
    remaining = list(deck)
    for _ in range(count):
        damage_count = sum(card.card_type is CardType.ATTACK for card in remaining)
        candidates = [
            card for card in remaining
            if (
                CardTag.DEFEND in card.tags
                or (CardTag.STRIKE in card.tags and damage_count > _MIN_DAMAGE_CARDS)
            )
        ]
        if not candidates:
            break
        remaining.remove(rng.choice(candidates))
    return remaining


def _add_cards(
    deck: list[CardInstance],
    character_id: str,
    rarity: CardRarity,
    count: int,
    rng: Rng,
) -> None:
    if count:
        deck.extend(create_character_cards(
            character_id,
            rng,
            count,
            rarity=rarity,
            distinct=False,
            generation_context=None,
        ))


def _add_curses(deck: list[CardInstance], count: int, *, act: int, rng: Rng) -> None:
    curse_pool = list(CURSE_CARD_POOL)
    if act == 1 or act == 4:
        from sts2_env.core.enums import CardId

        curse_pool = [CardId.GREED]
    if count:
        from sts2_env.cards.factory import create_cards_from_ids

        deck.extend(create_cards_from_ids(curse_pool, rng, count, distinct=False))


def _apply_upgrades(deck: list[CardInstance], profile: _DeckProfile, rng: Rng) -> list[CardInstance]:
    upgraded_deck = list(deck)
    candidates = [
        index for index, card in enumerate(upgraded_deck)
        if card.rarity in {
            CardRarity.BASIC, CardRarity.COMMON, CardRarity.UNCOMMON, CardRarity.RARE,
        } and _roll(rng, profile.upgrade_chance)
    ]
    if profile.max_upgrades is not None and len(candidates) > profile.max_upgrades:
        candidates = rng.sample(candidates, profile.max_upgrades)
    for index in candidates:
        upgraded = create_card(upgraded_deck[index].card_id, upgraded=True)
        if upgraded.upgraded:
            upgraded_deck[index] = upgraded
    return upgraded_deck


def build_training_deck(
    character_id: str,
    act: int,
    encounter_type: str,
    rng: Rng,
) -> list[CardInstance]:
    """Create a randomized deck for an act and a weak/normal/elite/boss fight."""
    if act not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported training act: {act}")
    if encounter_type not in {"weak", "normal", "elite", "boss"}:
        raise ValueError(f"Unsupported encounter type: {encounter_type}")

    from sts2_env.characters.all import create_starting_deck

    starting_deck = create_starting_deck(character_id)
    if act in {1, 4}:
        profile = _EARLY_ACT_PROFILES[encounter_type]
        remove_count = _choose_early_act_removal_count(encounter_type, rng)
        deck = _remove_starting_cards(starting_deck, remove_count, rng)
    else:
        archetype = _choose_archetype(rng)
        if encounter_type == "weak":
            archetype = "small"
        profile = _LATER_ACT_PROFILES[act][archetype]
        if act == 2 and encounter_type == "boss":
            remove_count = _randint(rng, 2, 4) if _roll(rng, 0.50) else 0
        else:
            remove_count = _randint(rng, profile.remove_min, profile.remove_max)
        deck = _remove_starting_cards(starting_deck, remove_count, rng)

    _add_cards(deck, character_id, CardRarity.COMMON, _randint(rng, profile.common_min, profile.common_max), rng)
    uncommon_count = _randint(rng, profile.uncommon_min, profile.uncommon_max)
    if not _roll(rng, profile.uncommon_chance):
        uncommon_count = 0
    _add_cards(deck, character_id, CardRarity.UNCOMMON, uncommon_count, rng)

    rare_count = _randint(rng, profile.rare_min, profile.rare_max)
    if not _roll(rng, profile.rare_chance):
        rare_count = 0
    _add_cards(deck, character_id, CardRarity.RARE, rare_count, rng)

    if act == 2 and encounter_type == "boss":
        _add_cards(deck, character_id, CardRarity.COMMON, 1, rng)
        _add_cards(deck, character_id, CardRarity.UNCOMMON, 1, rng)
        if _roll(rng, 0.70):
            _add_cards(deck, character_id, CardRarity.RARE, 1, rng)
        profile = replace(
            profile,
            upgrade_chance=min(1.0, profile.upgrade_chance + 0.10),
        )
    elif act == 3 and encounter_type == "boss":
        _add_cards(deck, character_id, CardRarity.COMMON, _randint(rng, 1, 2), rng)
        _add_cards(deck, character_id, CardRarity.UNCOMMON, _randint(rng, 1, 2), rng)
        if _roll(rng, 0.80):
            _add_cards(deck, character_id, CardRarity.RARE, 1, rng)
        profile = replace(
            profile,
            upgrade_chance=min(1.0, profile.upgrade_chance + 0.10),
        )

    if act in {1, 4}:
        from sts2_env.core.enums import CardId

        if _roll(rng, profile.curse_chance):
            from sts2_env.cards.factory import create_cards_from_ids

            deck.extend(create_cards_from_ids([CardId.GREED], rng, 1, distinct=False))
    elif _roll(rng, profile.curse_chance):
        _add_curses(deck, _randint(rng, profile.curse_min, profile.curse_max), act=act, rng=rng)

    return _apply_upgrades(deck, profile, rng)
