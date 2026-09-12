"""Disk-backed store for discovered model snapshots with fetch timestamps."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel

CACHE_VERSION = 1
MODEL_CATALOG_DISK_MAX_BYTES = 2 * 1024 * 1024
MODEL_CATALOG_DISK_MAX_ENTRIES = 128
MODEL_CATALOG_DISK_MAX_RAW_ENTRIES = 4096
MODEL_CATALOG_DISK_MAX_MODELS_PER_ENTRY = 100
_PROVIDER_KEY_MAX_CHARS = 128
_ENDPOINT_FINGERPRINT_MAX_CHARS = 512
_MODEL_ID_MAX_CHARS = 120
_TIMESTAMP_MAX_CHARS = 64
_ENTRY_KEYS = frozenset(
    {"provider_list_key", "endpoint_fingerprint", "fetched_at", "models"}
)
_PAYLOAD_KEYS = frozenset({"version", "entries"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp as timezone-aware UTC; None when invalid."""
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > _TIMESTAMP_MAX_CHARS
        or not value.isprintable()
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _bounded_text(value: object, *, maximum: int) -> str | None:
    if type(value) is not str:
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > maximum or not normalized.isprintable():
        return None
    return normalized


def _bounded_endpoint_fingerprint(value: object) -> str | None:
    fingerprint = _bounded_text(value, maximum=_ENDPOINT_FINGERPRINT_MAX_CHARS)
    if fingerprint is None or "://" not in fingerprint:
        return fingerprint
    try:
        parsed = urlsplit(fingerprint)
    except ValueError:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if parsed.query or parsed.fragment:
        return None
    return fingerprint


def _bounded_model_ids(model_ids: Iterable[object]) -> tuple[str, ...]:
    accepted: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(model_ids):
        if index >= MODEL_CATALOG_DISK_MAX_MODELS_PER_ENTRY:
            raise ValueError("model snapshot exceeds disk cache bounds")
        model_id = _bounded_text(value, maximum=_MODEL_ID_MAX_CHARS)
        if model_id is None:
            raise ValueError("model snapshot is invalid")
        if model_id in seen:
            continue
        seen.add(model_id)
        accepted.append(model_id)
    return tuple(accepted)


def _decode_entry(
    entry: object,
) -> tuple[str, str, datetime, tuple[str, ...]] | None:
    if type(entry) is not dict or set(entry) != _ENTRY_KEYS:
        return None
    provider_list_key = _bounded_text(
        entry.get("provider_list_key"), maximum=_PROVIDER_KEY_MAX_CHARS
    )
    endpoint_fingerprint = _bounded_endpoint_fingerprint(
        entry.get("endpoint_fingerprint")
    )
    fetched = _parse_timestamp(entry.get("fetched_at"))
    raw_ids = entry.get("models")
    if (
        provider_list_key is None
        or endpoint_fingerprint is None
        or fetched is None
        or type(raw_ids) is not list
        or len(raw_ids) > MODEL_CATALOG_DISK_MAX_MODELS_PER_ENTRY
    ):
        return None
    try:
        model_ids = _bounded_model_ids(raw_ids)
    except (TypeError, ValueError):
        return None
    return provider_list_key, endpoint_fingerprint, fetched, model_ids


def _encode_cache_state(
    model_ids_by_key: dict[tuple[str, str], tuple[str, ...]],
    fetched_at_by_key: dict[tuple[str, str], datetime],
) -> bytes:
    entries: dict[str, dict[str, object]] = {}
    for key, model_ids in model_ids_by_key.items():
        fetched = fetched_at_by_key.get(key)
        if fetched is None:
            continue
        entry_key = json.dumps(
            key,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        entries[entry_key] = {
            "provider_list_key": key[0],
            "endpoint_fingerprint": key[1],
            "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
            "models": list(model_ids),
        }
    payload = {"version": CACHE_VERSION, "entries": entries}
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_serializable_state(
    model_ids_by_key: dict[tuple[str, str], tuple[str, ...]],
    fetched_at_by_key: dict[tuple[str, str], datetime],
) -> None:
    if (
        len(_encode_cache_state(model_ids_by_key, fetched_at_by_key))
        > MODEL_CATALOG_DISK_MAX_BYTES
    ):
        raise ValueError("model catalog cache exceeds disk bounds")


class ModelCatalogDiskStore:
    """JSON store mirroring ModelDiscoveryCache entries plus fetched_at.

    Stores model IDs and timestamps only — never credentials or headers.
    Not thread-safe; assumes a single writer (one startup refresh worker).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._model_ids: dict[tuple[str, str], tuple[str, ...]] = {}
        self._fetched_at: dict[tuple[str, str], datetime] = {}

    def fetched_at(
        self, provider_list_key: str, endpoint_fingerprint: str
    ) -> datetime | None:
        """Return when the entry was fetched.

        Args:
            provider_list_key: Provider list key of the entry.
            endpoint_fingerprint: Endpoint fingerprint of the entry.

        Returns:
            datetime | None: Timezone-aware UTC fetch time, or None when the
            entry is absent.
        """
        return self._fetched_at.get((str(provider_list_key), str(endpoint_fingerprint)))

    def is_stale(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        *,
        stale_after_hours: float,
        now: datetime | None = None,
    ) -> bool:
        """Return True when the entry is missing or older than the threshold.

        A threshold of 0 (or less) means always-stale: refetch every launch.
        A future-dated fetched_at (clock skew) also counts as stale.

        Args:
            provider_list_key: Provider list key of the entry.
            endpoint_fingerprint: Endpoint fingerprint of the entry.
            stale_after_hours: Maximum entry age in hours before it is stale.
            now: Reference time; defaults to the current UTC time.

        Returns:
            bool: True when the entry should be refetched.
        """
        if stale_after_hours <= 0:
            return True
        fetched = self.fetched_at(provider_list_key, endpoint_fingerprint)
        if fetched is None:
            return True
        current = now or _utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        age_seconds = (current - fetched).total_seconds()
        return age_seconds < 0 or age_seconds >= stale_after_hours * 3600

    def record(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        model_ids: Iterable[str],
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        """Store a fetched model ID snapshot for a provider/endpoint pair.

        Args:
            provider_list_key: Provider list key of the entry.
            endpoint_fingerprint: Endpoint fingerprint of the entry.
            model_ids: Fetched model IDs to store.
            fetched_at: Fetch time (naive treated as UTC); defaults to now.
        """
        provider_key = _bounded_text(provider_list_key, maximum=_PROVIDER_KEY_MAX_CHARS)
        endpoint_key = _bounded_endpoint_fingerprint(endpoint_fingerprint)
        if provider_key is None or endpoint_key is None:
            raise ValueError("model catalog cache identity is invalid")
        key = (provider_key, endpoint_key)
        if (
            key not in self._model_ids
            and len(self._model_ids) >= MODEL_CATALOG_DISK_MAX_ENTRIES
        ):
            raise ValueError("model catalog cache entry limit exceeded")
        bounded_ids = _bounded_model_ids(model_ids)
        stamp = fetched_at or _utc_now()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        stamp = stamp.astimezone(UTC)
        candidate_model_ids = dict(self._model_ids)
        candidate_fetched_at = dict(self._fetched_at)
        candidate_model_ids[key] = bounded_ids
        candidate_fetched_at[key] = stamp
        _require_serializable_state(candidate_model_ids, candidate_fetched_at)
        self._model_ids = candidate_model_ids
        self._fetched_at = candidate_fetched_at

    def prune(self, keep_provider_list_keys: set[str]) -> None:
        """Drop entries for providers no longer configured.

        Args:
            keep_provider_list_keys: Provider list keys whose entries survive;
                everything else is removed from the in-memory store.
        """
        for key in tuple(self._fetched_at):
            if key[0] not in keep_provider_list_keys:
                self._fetched_at.pop(key, None)
                self._model_ids.pop(key, None)

    def load_into(self, cache: ModelDiscoveryCache) -> None:
        """Populate the in-memory cache from disk; missing/corrupt loads empty.

        Args:
            cache: The runtime discovery cache to fill with entries decoded
                from the JSON store file.
        """
        self._model_ids.clear()
        self._fetched_at.clear()
        try:
            with self.path.open("rb") as cache_file:
                raw_payload = cache_file.read(MODEL_CATALOG_DISK_MAX_BYTES + 1)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("Ignoring model catalog cache (reason=read_error)")
            return
        if len(raw_payload) > MODEL_CATALOG_DISK_MAX_BYTES:
            logger.warning("Ignoring model catalog cache (reason=file_too_large)")
            return
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            logger.warning("Ignoring model catalog cache (reason=invalid_json)")
            return
        if (
            type(payload) is not dict
            or set(payload) != _PAYLOAD_KEYS
            or type(payload.get("version")) is not int
            or payload.get("version") != CACHE_VERSION
            or type(payload.get("entries")) is not dict
        ):
            logger.warning("Ignoring model catalog cache (reason=invalid_shape)")
            return
        entries = payload["entries"]
        if len(entries) > MODEL_CATALOG_DISK_MAX_RAW_ENTRIES:
            logger.warning("Ignoring model catalog cache (reason=too_many_entries)")
            return
        loaded_model_ids: dict[tuple[str, str], tuple[str, ...]] = {}
        loaded_fetched_at: dict[tuple[str, str], datetime] = {}
        accepted = 0
        rejected = 0
        examined = 0
        for entry in entries.values():
            if accepted >= MODEL_CATALOG_DISK_MAX_ENTRIES:
                rejected += len(entries) - examined
                break
            examined += 1
            decoded = _decode_entry(entry)
            if decoded is None:
                rejected += 1
                continue
            provider_list_key, endpoint_fingerprint, fetched, model_ids = decoded
            key = (provider_list_key, endpoint_fingerprint)
            # A disk file may alias one logical snapshot under many outer keys.
            # Keep the first valid snapshot and do not spend quota on duplicates.
            if key in loaded_model_ids:
                rejected += 1
                continue
            candidate_model_ids = dict(loaded_model_ids)
            candidate_fetched_at = dict(loaded_fetched_at)
            candidate_model_ids[key] = model_ids
            candidate_fetched_at[key] = fetched
            try:
                _require_serializable_state(candidate_model_ids, candidate_fetched_at)
            except ValueError:
                rejected += 1
                continue
            discovered_at = fetched.isoformat().replace("+00:00", "Z")
            try:
                cache.replace(
                    provider_list_key,
                    endpoint_fingerprint,
                    (
                        DiscoveredModel(
                            provider=provider_list_key,
                            provider_list_key=provider_list_key,
                            model_id=model_id,
                            display_name=model_id,
                            source="runtime_discovered",
                            endpoint_fingerprint=endpoint_fingerprint,
                            discovered_at=discovered_at,
                        )
                        for model_id in model_ids
                    ),
                )
            except Exception:  # noqa: BLE001 - one bad entry must not abort loading.
                rejected += 1
                continue
            loaded_model_ids = candidate_model_ids
            loaded_fetched_at = candidate_fetched_at
            accepted += 1
        self._model_ids = loaded_model_ids
        self._fetched_at = loaded_fetched_at
        if rejected:
            logger.warning(
                "Rejected model catalog cache entries (count={}); valid entries "
                "continue loading and discovery may refresh missing models.",
                rejected,
            )

    def save(self) -> None:
        """Atomically write the store (pid-scoped temp file + rename).

        Raises:
            OSError: if the write or rename fails (a leftover .tmp file may remain).
            ValueError: if internal state exceeds the serialized byte bound.
        """
        encoded = _encode_cache_state(self._model_ids, self._fetched_at)
        if len(encoded) > MODEL_CATALOG_DISK_MAX_BYTES:
            raise ValueError("model catalog cache exceeds disk bounds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_bytes(encoded)
        os.replace(tmp_path, self.path)
