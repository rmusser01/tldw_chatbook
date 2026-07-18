"""Disk-backed store for discovered model snapshots with fetch timestamps."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel

CACHE_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp as timezone-aware UTC; None when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ModelCatalogDiskStore:
    """JSON store mirroring ModelDiscoveryCache entries plus fetched_at.

    Stores model IDs and timestamps only — never credentials or headers.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._model_ids: dict[tuple[str, str], tuple[str, ...]] = {}
        self._fetched_at: dict[tuple[str, str], datetime] = {}

    def fetched_at(self, provider_list_key: str, endpoint_fingerprint: str) -> datetime | None:
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
        """
        if stale_after_hours <= 0:
            return True
        fetched = self.fetched_at(provider_list_key, endpoint_fingerprint)
        if fetched is None:
            return True
        current = now or _utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return (current - fetched).total_seconds() >= stale_after_hours * 3600

    def record(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        model_ids,
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        key = (str(provider_list_key), str(endpoint_fingerprint))
        self._model_ids[key] = tuple(str(model_id) for model_id in model_ids)
        stamp = fetched_at or _utc_now()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        self._fetched_at[key] = stamp

    def prune(self, keep_provider_list_keys: set[str]) -> None:
        """Drop entries for providers no longer configured."""
        for key in tuple(self._fetched_at):
            if key[0] not in keep_provider_list_keys:
                self._fetched_at.pop(key, None)
                self._model_ids.pop(key, None)

    def load_into(self, cache: ModelDiscoveryCache) -> None:
        """Populate the in-memory cache from disk; missing/corrupt loads empty."""
        self._model_ids.clear()
        self._fetched_at.clear()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable model catalog cache {self.path}: {exc}")
            return
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            return
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            provider_list_key = str(entry.get("provider_list_key") or "").strip()
            endpoint_fingerprint = str(entry.get("endpoint_fingerprint") or "").strip()
            fetched = _parse_timestamp(entry.get("fetched_at"))
            raw_ids = entry.get("models")
            if not provider_list_key or not endpoint_fingerprint or fetched is None:
                continue
            if not isinstance(raw_ids, list):
                continue
            model_ids = tuple(
                model_id.strip()
                for model_id in raw_ids
                if isinstance(model_id, str) and model_id.strip()
            )
            discovered_at = fetched.isoformat().replace("+00:00", "Z")
            cache.replace(
                provider_list_key,
                endpoint_fingerprint,
                tuple(
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
            key = (provider_list_key, endpoint_fingerprint)
            self._model_ids[key] = model_ids
            self._fetched_at[key] = fetched

    def save(self) -> None:
        """Atomically write the store (temp file + rename)."""
        entries: dict[str, dict] = {}
        for key, model_ids in self._model_ids.items():
            fetched = self._fetched_at.get(key)
            if fetched is None:
                continue
            entries[f"{key[0]}|{key[1]}"] = {
                "provider_list_key": key[0],
                "endpoint_fingerprint": key[1],
                "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
                "models": list(model_ids),
            }
        payload = {"version": CACHE_VERSION, "entries": entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)
