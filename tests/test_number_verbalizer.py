"""Tests for Persian number / date / time verbalization."""

from __future__ import annotations

import pytest

from ava_voice.core.speech_plan import plan_speech
from ava_voice.linguistics.number_verbalizer import (
    cardinal,
    ordinal,
    verbalize_numbers,
)


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "صفر"),
        (7, "هفت"),
        (13, "سیزده"),
        (20, "بیست"),
        (99, "نود و نه"),
        (100, "صد"),
        (205, "دویست و پنج"),
        (1000, "هزار"),
        (1001, "هزار و یک"),
        (1405, "هزار و چهارصد و پنج"),
        (21_000, "بیست و یک هزار"),
        (250_000, "دویست و پنجاه هزار"),
        (1_000_000, "یک میلیون"),
        (2_500_000_000, "دو میلیارد و پانصد میلیون"),
        (-12, "منفی دوازده"),
    ],
)
def test_cardinal(n: int, words: str) -> None:
    assert cardinal(n) == words


def test_cardinal_rejects_overflow() -> None:
    with pytest.raises(ValueError):
        cardinal(10**15)


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (1, "یکم"),
        (2, "دوم"),
        (3, "سوم"),
        (23, "بیست و سوم"),
        (30, "سی‌ام"),
        (100, "صدم"),
    ],
)
def test_ordinal(n: int, words: str) -> None:
    assert ordinal(n) == words


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("قیمت ۲۵۰,۰۰۰ تومان", "قیمت دویست و پنجاه هزار تومان"),
        ("قیمت ۲۵۰٬۰۰۰ تومان", "قیمت دویست و پنجاه هزار تومان"),
        ("۳.۵ کیلو", "سه ممیز پنج کیلو"),
        ("۳٫۵ کیلو", "سه ممیز پنج کیلو"),
        ("۲۵٪ تخفیف", "بیست و پنج درصد تخفیف"),
        ("25% off", "بیست و پنج درصد off"),
        ("دمای -۵ درجه", "دمای منفی پنج درجه"),
        ("صفحه ۲-۳", "صفحه دو تا سه"),
        ("فصل ۳م", "فصل سوم"),
        ("ساعت ۹:۰۰", "ساعت نه"),
        ("ساعت ۱۴:۳۰", "ساعت چهارده و نیم"),
        ("ساعت ۸:۱۵", "ساعت هشت و پانزده دقیقه"),
        ("۱۴۰۵/۰۹/۰۵", "پنجم آذر هزار و چهارصد و پنج"),
        ("2026-11-26", "بیست و ششم نوامبر دو هزار و بیست و شش"),
        ("کد ۰۰۷", "کد صفر صفر هفت"),
        (
            "۰۹۱۲۳۴۵۶۷۸۹",
            "صفر نهصد و دوازده، سیصد و چهل و پنج، شصت و هفت، هشتاد و نه",
        ),
        ("بدون عدد", "بدون عدد"),
    ],
)
def test_verbalize_numbers(text: str, spoken: str) -> None:
    assert verbalize_numbers(text) == spoken


def test_invalid_date_is_left_for_the_number_rule() -> None:
    # Month 13 is not a date; it must not be read with month names.
    out = verbalize_numbers("۱۴۰۵/۱۳/۰۱")
    assert "آذر" not in out and "۱" not in out


def test_no_digit_reaches_the_phoneme_stream() -> None:
    plan = plan_speech("قیمت این محصول ۲۵۰ هزار تومان است.")
    symbols = [
        ph.symbol
        for s in plan.sentences
        for ph in s.acoustic.phoneme_durations
    ]
    assert symbols, "plan must contain phonemes"
    assert not any(ch.isdigit() for sym in symbols for ch in sym)
