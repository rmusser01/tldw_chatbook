import hashlib
import hmac
import json

from pydantic import BaseModel


def canonical_bytes(value: BaseModel) -> bytes:
    payload = value.model_dump(mode="json", exclude_none=False, by_alias=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def integrity_tag(value: BaseModel, key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("integrity key must be exactly 32 bytes")
    return f"hmac-sha256-v1:{hmac.new(key, canonical_bytes(value), hashlib.sha256).hexdigest()}"
