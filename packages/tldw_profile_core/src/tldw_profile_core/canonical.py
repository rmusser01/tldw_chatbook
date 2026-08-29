import hashlib
import hmac
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from rfc8785 import dumps

I_JSON_MAX_INTEGER = 2**53 - 1
CANONICAL_DATETIME_FORMAT = "utc-milliseconds-v1"


def normalize_datetime(value: datetime) -> str:
    """Return the portable V1 UTC millisecond representation of an aware datetime."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset().total_seconds() % 60:
        raise ValueError("timestamp offset must use whole minutes")
    if value.microsecond % 1_000:
        raise ValueError("timestamp precision must not exceed milliseconds")
    try:
        value = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ValueError("timestamp must normalize within years 0001-9999") from error
    milliseconds = value.microsecond // 1_000
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}T"
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}."
        f"{milliseconds:03d}Z"
    )


def _portable_datetime(value: datetime) -> datetime:
    normalize_datetime(value)
    return value


PortableDateTime = Annotated[datetime, AfterValidator(_portable_datetime)]


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return normalize_datetime(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with RFC 8785 JCS."""

    return dumps(_json_value(value))


def canonical_bytes(value: BaseModel) -> bytes:
    payload = value.model_dump(mode="python", exclude_none=False, by_alias=True)
    return canonical_json_bytes(payload)


def integrity_tag(value: BaseModel, key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("integrity key must be exactly 32 bytes")
    return f"hmac-sha256-v1:{hmac.new(key, canonical_bytes(value), hashlib.sha256).hexdigest()}"
