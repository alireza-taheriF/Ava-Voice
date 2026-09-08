"""Owned Persian evaluation sentences (Phase 0 gold-text set).

These were written for Ava Voice. They are not transcribed from a podcast
or a video. Tags mark the phenomena the patent claims care about: ezafe,
formality, affect, written drawl, questions.
"""

from __future__ import annotations

from ava_voice.corpus.schema import Utterance
from ava_voice.corpus.sources import OWNED_SEED

# (id, text, tags)
_SEED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("seed-ezafe-01", "کتابِ علی روی میز است.", ("ezafe", "kasra", "neutral")),
    ("seed-ezafe-02", "خانهِ بزرگ را دیدم.", ("ezafe", "kasra", "ye")),
    ("seed-ezafe-03", "رنگِ آبی آسمان را دوست دارم.", ("ezafe", "kasra")),
    ("seed-ezafe-04", "اسمِ این گل زیبا چیست؟", ("ezafe", "kasra", "question")),
    ("seed-ezafe-05", "صدایِ آرام رودخانه را می‌شنوم.", ("ezafe", "kasra", "calm")),
    ("seed-no-ezafe-01", "خانه بزرگ است.", ("no-ezafe", "lexical-e")),
    ("seed-no-ezafe-02", "کتاب روی میز است.", ("no-ezafe",)),
    ("seed-no-ezafe-03", "سلام. خوبی؟", ("no-ezafe", "dialogue")),
    ("seed-formal-01", "احتراماً کتابِ جناب مدیر را ملاحظه فرمایید.", ("formality", "ezafe")),
    ("seed-formal-02", "خواهشمندم حضورِ محترم را آگاه فرمایید.", ("formality", "ezafe")),
    ("seed-formal-03", "با کمال احترام گزارشِ جلسه تقدیم می‌گردد.", ("formality", "ezafe")),
    ("seed-warm-01", "چه کتابِ قشنگی!!!", ("warm", "ezafe", "intensity")),
    ("seed-warm-02", "خیلی خوشحالم از دیدنِ رویِ تو.", ("joy", "ezafe")),
    ("seed-warm-03", "عااالی بود امروز!", ("joy", "drawl", "intensity")),
    ("seed-sad-01", "دلم گرفته از رفتنِ تو.", ("sadness", "ezafe")),
    ("seed-sad-02", "خیلی ناراحتم.", ("sadness",)),
    ("seed-anger-01", "از دستِ این کار عصبانی‌ام!", ("anger", "ezafe")),
    ("seed-fear-01", "می‌ترسم از صدایِ آن.", ("fear", "ezafe")),
    ("seed-drawl-01", "سلاااام", ("drawl",)),
    ("seed-drawl-02", "نههههه این‌طور نیست.", ("drawl", "negation")),
    ("seed-question-01", "کجا می‌روی؟", ("question",)),
    ("seed-question-02", "آیا کتابِ او را خوانده‌ای؟", ("question", "ezafe")),
    ("seed-mixed-01", "خیلی خوشحالم! اما کتابِ سخت را نفهمیدم.", ("mixed", "ezafe")),
    ("seed-mixed-02", "احتراماً عرض می‌کنم که خیلی ناراحتم.", ("mixed", "formality", "sadness")),
    ("seed-para-01", "سلام! امروز عالی بود. کتابِ علی را خواندم.", ("paragraph", "ezafe")),
)


def seed_utterances() -> tuple[Utterance, ...]:
    """Return the owned seed set as :class:`Utterance` records."""
    rows: list[Utterance] = []
    for uid, text, tags in _SEED:
        rows.append(
            Utterance(
                id=uid,
                text=text,
                source="ava-seed",
                license=OWNED_SEED,
                tags=tags,
            )
        )
    return tuple(rows)
