"""Display-only formatting; never round, invent or replace evidence values."""

import math
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import nepali_datetime

DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
NPT = timezone(timedelta(hours=5, minutes=45))


def number(value: int | float | None, language: str = "ne") -> str:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    if value is None:
        return "unknown" if language == "en" else "अज्ञात"
    if type(value) not in {int, float} or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("Expected finite numeric evidence")
    raw = format(Decimal(str(value)), "f")
    if language == "en":
        return raw
    sign = "-" if raw.startswith("-") else ""
    whole, _, fractional = raw.lstrip("-").partition(".")
    tail = whole[-3:]
    head = whole[:-3]
    groups = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return (sign + ",".join([*groups, tail]) + ("." + fractional if fractional else "")).translate(
        DIGITS
    )


def bs_date(day: date, language: str = "ne") -> str:
    try:
        converted = nepali_datetime.date.from_datetime_date(day).isoformat()
    except (ValueError, OverflowError):
        return (
            "BS unavailable (outside calendar table)"
            if language == "en"
            else "वि.सं. उपलब्ध छैन (पात्रोको सीमाबाहिर)"
        )
    return converted if language == "en" else converted.translate(DIGITS)


def timestamp(value: str, language: str = "ne") -> str:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    local = moment.astimezone(NPT)
    utc = moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
    clock = local.strftime("%H:%M:%S")
    if language == "ne":
        clock = clock.translate(DIGITS)
    return f"{bs_date(local.date(), language)} BS {clock} NPT | {utc}"


def event_day(value: str, language: str = "ne") -> str:
    # A supplied date has no exact instant; do not invent a UTC time.
    day = date.fromisoformat(value)
    label = "date only" if language == "en" else "मिति मात्र"
    return f"{bs_date(day, language)} BS | {day.isoformat()} ({label})"


def mapped_name(properties: dict, language: str = "ne") -> tuple[str | None, bool]:
    if (
        language == "ne"
        and isinstance(properties.get("name:ne"), str)
        and properties["name:ne"].strip()
    ):
        return properties["name:ne"], False
    name = properties.get("name")
    return (name if isinstance(name, str) and name.strip() else None, language == "ne")
