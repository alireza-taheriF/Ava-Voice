"""Persian number, date and time verbalization for speech.

The phonemizer cannot pronounce digits: before this module existed, ``۲۵۰``
reached the phoneme stream as the literal symbols ``۲``, ``۵``, ``۰``. This
module rewrites numeric expressions into spoken Persian words *before*
tokenization, so every downstream stage sees speakable text.

Covered forms (all deterministic, no model, no network):

* cardinals up to 999 تریلیون, negatives, thousands separators
  (``۲۵۰,۰۰۰`` / ``۲۵۰٬۰۰۰``)
* decimals (``۳.۵`` / ``۳٫۵``) → ``سه ممیز پنج``
* percentages (``۲۵٪`` / ``25%``) → ``بیست و پنج درصد``
* ordinals written with a suffix (``۳م`` / ``۳ام``) and :func:`ordinal`
* numeric ranges (``۲-۳``) → ``دو تا سه``
* clock times (``۱۴:۲۰``) → ``چهارده و بیست دقیقه``, ``:۳۰`` → ``و نیم``
* dates ``YYYY/MM/DD`` — Jalali month names when the year looks Jalali
  (1300–1499), Gregorian month names when it looks Gregorian (1900–2199)
* digit strings with a leading zero (phone numbers, codes); Iranian mobile
  numbers ``09xxxxxxxxx`` are read in the conventional 4-3-2-2 grouping

>>> verbalize_numbers("قیمت ۲۵۰,۰۰۰ تومان")
'قیمت دویست و پنجاه هزار تومان'
>>> verbalize_numbers("ساعت ۱۴:۲۰")
'ساعت چهارده و بیست دقیقه'
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "cardinal",
    "ordinal",
    "verbalize_numbers",
    "JALALI_MONTHS",
    "GREGORIAN_MONTHS",
]

_ONES: Final[tuple[str, ...]] = (
    "صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه",
)
_TEENS: Final[tuple[str, ...]] = (
    "ده", "یازده", "دوازده", "سیزده", "چهارده",
    "پانزده", "شانزده", "هفده", "هجده", "نوزده",
)
_TENS: Final[tuple[str, ...]] = (
    "", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود",
)
_HUNDREDS: Final[tuple[str, ...]] = (
    "", "صد", "دویست", "سیصد", "چهارصد",
    "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد",
)
# Short scale as used in Iran: میلیارد = 10^9, تریلیون = 10^12.
_SCALES: Final[tuple[str, ...]] = ("", "هزار", "میلیون", "میلیارد", "تریلیون")
_MAX: Final[int] = 10 ** 15 - 1

JALALI_MONTHS: Final[tuple[str, ...]] = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
)
GREGORIAN_MONTHS: Final[tuple[str, ...]] = (
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)

_AND: Final[str] = " و "

_FA_DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹"
_AR_DIGITS: Final[str] = "٠١٢٣٤٥٦٧٨٩"
_TO_ASCII: Final[dict[int, str]] = {
    **{ord(c): str(i) for i, c in enumerate(_FA_DIGITS)},
    **{ord(c): str(i) for i, c in enumerate(_AR_DIGITS)},
}

_D: Final[str] = r"[0-9۰-۹٠-٩]"


def _ascii(s: str) -> str:
    return s.translate(_TO_ASCII)


# --------------------------------------------------------------------------- #
# Core number words
# --------------------------------------------------------------------------- #
def _below_thousand(n: int) -> list[str]:
    parts: list[str] = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rest >= 20:
        t, o = divmod(rest, 10)
        parts.append(_TENS[t])
        if o:
            parts.append(_ONES[o])
    elif rest >= 10:
        parts.append(_TEENS[rest - 10])
    elif rest:
        parts.append(_ONES[rest])
    return parts


def cardinal(n: int) -> str:
    """Return the spoken Persian cardinal for integer ``n``.

    ``1000`` is read ``هزار`` (not ``یک هزار``), matching everyday speech;
    larger scales keep the count (``یک میلیون``).

    >>> cardinal(1405)
    'هزار و چهارصد و پنج'
    >>> cardinal(-12)
    'منفی دوازده'
    """
    if n < 0:
        return "منفی " + cardinal(-n)
    if n == 0:
        return _ONES[0]
    if n > _MAX:
        raise ValueError(f"number too large to verbalize: {n}")
    groups: list[int] = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    words: list[str] = []
    for scale in range(len(groups) - 1, -1, -1):
        g = groups[scale]
        if not g:
            continue
        if scale == 1 and g == 1:
            words.append(_SCALES[1])
            continue
        chunk = _AND.join(_below_thousand(g))
        words.append(f"{chunk} {_SCALES[scale]}".strip())
    return _AND.join(words)


def ordinal(n: int) -> str:
    """Return the Persian ordinal (``-م`` form) for positive ``n``.

    >>> ordinal(1), ordinal(3), ordinal(30), ordinal(23)
    ('یکم', 'سوم', 'سی\\u200cام', 'بیست و سوم')
    """
    if n <= 0:
        raise ValueError("ordinals are defined for positive integers")
    word = cardinal(n)
    if word.endswith("سه"):
        return word[:-2] + "سوم"
    if word.endswith("ی"):  # سی → سی‌ام
        return word + "‌ام"
    return word + "م"


def _digits_one_by_one(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _read_digit_string(s: str) -> str:
    """Read a leading-zero digit string (phone number, code)."""
    if len(s) == 11 and s.startswith("09"):
        groups = (s[:4], s[4:7], s[7:9], s[9:])
    elif len(s) == 11 and s.startswith("0"):  # landline: 0 + area(2) + pairs
        groups = (s[:3], s[3:5], s[5:7], s[7:9], s[9:])
    else:
        return _digits_one_by_one(s)
    spoken: list[str] = []
    for g in groups:
        zeros = len(g) - len(g.lstrip("0"))
        head = " ".join([_ONES[0]] * zeros)
        tail = cardinal(int(g)) if g.lstrip("0") else ""
        spoken.append(" ".join(p for p in (head, tail) if p))
    return "، ".join(spoken)


# --------------------------------------------------------------------------- #
# Pattern handlers (most specific first)
# --------------------------------------------------------------------------- #
_DATE_RE: Final = re.compile(rf"(?<!{_D})({_D}{{4}})[/\-]({_D}{{1,2}})[/\-]({_D}{{1,2}})(?!{_D})")
_TIME_RE: Final = re.compile(rf"(?<!{_D})({_D}{{1,2}}):({_D}{{2}})(?!{_D})")
_RANGE_RE: Final = re.compile(rf"(?<![{_D[1:-1]}\-])({_D}+)\s?[-–]\s?({_D}+)(?!{_D})")
_PERCENT_RE: Final = re.compile(rf"(-?{_D}+(?:[.٫]{_D}+)?)\s?[%٪]")
_ORDINAL_RE: Final = re.compile(rf"(?<!{_D})({_D}+)(ام|م)(?![؀-ۿ])")
_NUMBER_RE: Final = re.compile(
    rf"(?<![\w.٫\-])(?:(?<![^\s])(-))?({_D}{{1,3}}(?:[,٬]{_D}{{3}})+|{_D}+)(?:[.٫]({_D}+))?(?!{_D})"
)


def _number(sign: str | None, int_part: str, frac: str | None) -> str:
    raw = _ascii(int_part).replace(",", "").replace("٬", "")
    if frac is None and len(raw) > 1 and raw.startswith("0"):
        return _read_digit_string(raw)
    if len(raw) > 15:
        return _digits_one_by_one(raw)
    words = cardinal(int(raw))
    if frac is not None:
        f = _ascii(frac)
        words += " ممیز " + (cardinal(int(f)) if not f.startswith("0") else _digits_one_by_one(f))
    return ("منفی " + words) if sign else words


def _sub_date(m: re.Match[str]) -> str:
    y, mo, d = (int(_ascii(g)) for g in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    if 1300 <= y <= 1499:
        months = JALALI_MONTHS
    elif 1900 <= y <= 2199:
        months = GREGORIAN_MONTHS
    else:
        return m.group(0)
    return f"{ordinal(d)} {months[mo - 1]} {cardinal(y)}"


def _sub_time(m: re.Match[str]) -> str:
    h, mi = int(_ascii(m.group(1))), int(_ascii(m.group(2)))
    if not (0 <= h <= 24 and 0 <= mi <= 59):
        return m.group(0)
    if mi == 0:
        return cardinal(h)
    if mi == 30:
        return f"{cardinal(h)} و نیم"
    return f"{cardinal(h)} و {cardinal(mi)} دقیقه"


def _sub_range(m: re.Match[str]) -> str:
    return f"{m.group(1)} تا {m.group(2)}"


def _sub_percent(m: re.Match[str]) -> str:
    token = m.group(1)
    sign = "-" if token.startswith("-") else None
    token = token.lstrip("-")
    int_part, _, frac = token.replace("٫", ".").partition(".")
    return _number(sign, int_part, frac or None) + " درصد"


def _sub_ordinal(m: re.Match[str]) -> str:
    n = int(_ascii(m.group(1)))
    return ordinal(n) if n > 0 else m.group(0)


def _sub_number(m: re.Match[str]) -> str:
    return _number(m.group(1), m.group(2), m.group(3))


def verbalize_numbers(text: str) -> str:
    """Rewrite every numeric expression in ``text`` as spoken Persian.

    Text without digits is returned unchanged (fast path).
    """
    if not re.search(_D, text):
        return text
    text = _DATE_RE.sub(_sub_date, text)
    text = _TIME_RE.sub(_sub_time, text)
    text = _RANGE_RE.sub(_sub_range, text)
    text = _PERCENT_RE.sub(_sub_percent, text)
    text = _ORDINAL_RE.sub(_sub_ordinal, text)
    text = _NUMBER_RE.sub(_sub_number, text)
    return text
