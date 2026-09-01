# models_dev_catalog.py
"""TASK-26023: models.dev as a lower-priority model-metadata source.

An upstream catalog (models.dev, ~4000 models) filling gaps BENEATH the
hand-maintained capability patterns and the seed price table. Design
constraints from the ACs:

* It is a MERGE LAYER, never an override: a hand-maintained entry always
  wins (the pricing/capability lookups consult this only after their own
  resolution misses).
* Fetched with a conditional GET (ETag / If-None-Match) and disk-cached;
  a 304 keeps the cached body. Fetching is explicit or background -- the
  lookup path never touches the network.
* Honest about unknowns: a model absent here returns None, so the caller
  keeps its existing "no fabricated price" behavior.
* Fully offline: no cache + no network = empty catalog, today's behavior.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

#: The upstream aggregate catalog.
MODELS_DEV_URL = "https://models.dev/api.json"

#: HTTP getter contract: (url, headers) -> (status, response_headers, body).
HttpGet = Callable[[str, dict], tuple]


@dataclass(frozen=True, slots=True)
class ModelsDevEntry:
    """One upstream model's gap-fill metadata."""

    context_window: int | None
    supports_vision: bool
    input_price_per_mtok: float | None
    output_price_per_mtok: float | None
    source: str = "models.dev"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def parse_models_dev(blob: Any) -> dict[tuple[str, str], ModelsDevEntry]:
    """Parse the models.dev api.json shape into gap-fill entries.

    Defensive: any provider/model whose shape is not what we expect is
    skipped, never raised on -- a schema drift upstream degrades to fewer
    entries, not a crash.
    """
    catalog: dict[tuple[str, str], ModelsDevEntry] = {}
    if not isinstance(blob, dict):
        return catalog
    for provider, provider_data in blob.items():
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            limit = model_data.get("limit")
            context = (
                _as_int(limit.get("context")) if isinstance(limit, dict) else None
            )
            modalities = model_data.get("modalities")
            inputs = (
                modalities.get("input") if isinstance(modalities, dict) else None
            )
            vision = isinstance(inputs, list) and "image" in inputs
            cost = model_data.get("cost")
            in_price = _as_float(cost.get("input")) if isinstance(cost, dict) else None
            out_price = (
                _as_float(cost.get("output")) if isinstance(cost, dict) else None
            )
            catalog[(str(provider).lower(), str(model_id).lower())] = ModelsDevEntry(
                context_window=context,
                supports_vision=vision,
                input_price_per_mtok=in_price,
                output_price_per_mtok=out_price,
            )
    return catalog


@dataclass(frozen=True)
class ModelsDevCache:
    """The parsed catalog plus the ETag it was fetched at."""

    catalog: dict[tuple[str, str], ModelsDevEntry]
    etag: str | None

    def lookup(self, provider: str, model: str) -> ModelsDevEntry | None:
        return self.catalog.get(
            ((provider or "").lower(), (model or "").lower())
        )

    @classmethod
    def empty(cls) -> "ModelsDevCache":
        return cls(catalog={}, etag=None)

    @classmethod
    def load(cls, disk_path: Path) -> "ModelsDevCache":
        """Load from disk; a missing/corrupt file is an empty cache (AC#7)."""
        try:
            raw = json.loads(Path(disk_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls.empty()
        if not isinstance(raw, dict):
            return cls.empty()
        return cls(
            catalog=parse_models_dev(raw.get("body")),
            etag=raw.get("etag") if isinstance(raw.get("etag"), str) else None,
        )


def _write_cache_file(disk_path: Path, body: Any, etag: str | None) -> None:
    disk_path = Path(disk_path)
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"etag": etag, "body": body}, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=disk_path.parent, prefix=".models-dev-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, disk_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fetch_models_dev(
    *,
    disk_path: Path,
    http_get: HttpGet,
    url: str = MODELS_DEV_URL,
) -> None:
    """Conditionally refresh the disk cache. Never raises, never blocks a
    lookup (callers run this explicitly or on a background timer).

    A 304 keeps the cached body; a 200 replaces it; any error leaves the
    existing cache untouched.
    """
    existing = ModelsDevCache.load(disk_path)
    headers = {"Accept": "application/json"}
    if existing.etag:
        headers["If-None-Match"] = existing.etag
    try:
        status, response_headers, body = http_get(url, headers)
    except Exception as exc:  # noqa: BLE001 -- a background refresh never raises out
        logger.debug(f"models.dev refresh failed (kept cache): {exc!r}")
        return
    if status == 304:
        return
    if status != 200:
        logger.debug(f"models.dev refresh got status {status}; kept cache")
        return
    try:
        parsed = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except (ValueError, AttributeError):
        logger.debug("models.dev refresh returned unparseable body; kept cache")
        return
    etag = None
    if isinstance(response_headers, dict):
        etag = response_headers.get("ETag") or response_headers.get("etag")
    try:
        _write_cache_file(disk_path, parsed, etag if isinstance(etag, str) else None)
    except Exception as exc:  # noqa: BLE001 -- disk failure keeps the old cache
        logger.debug(f"models.dev cache write failed: {exc!r}")


# ---------------------------------------------------------------------------
# Process-wide gap-fill layer. Lazily loads the disk cache into memory ONCE
# and answers lookups from memory -- the lookup path never touches the
# network or the disk after the first read (AC#4). Gated by
# ``[model_catalog] use_models_dev`` (default OFF), so unconfigured is
# byte-identical to today and fully offline (AC#6/#7).

_MEMORY_CACHE: "ModelsDevCache | None" = None


def default_cache_path() -> Path:
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "models_dev_catalog.json"


def _enabled() -> bool:
    try:
        from tldw_chatbook.config import coerce_bool_setting, get_cli_setting

        return coerce_bool_setting(
            get_cli_setting("model_catalog", "use_models_dev", False),
            default=False,
        )
    except Exception:  # noqa: BLE001 -- unreadable config disables the layer
        return False


def _memory_cache() -> ModelsDevCache:
    global _MEMORY_CACHE
    if _MEMORY_CACHE is None:
        try:
            _MEMORY_CACHE = ModelsDevCache.load(default_cache_path())
        except Exception:  # noqa: BLE001
            _MEMORY_CACHE = ModelsDevCache.empty()
    return _MEMORY_CACHE


def reset_memory_cache() -> None:
    """Test/refresh hook: drop the in-memory cache so the next lookup reloads."""
    global _MEMORY_CACHE
    _MEMORY_CACHE = None


def models_dev_entry(provider: str, model: str) -> ModelsDevEntry | None:
    """The upstream gap-fill entry for one pair, or None.

    Returns None when the layer is disabled, so callers keep their own
    honest-unknown behavior (AC#6). Never fetches.
    """
    if not _enabled():
        return None
    return _memory_cache().lookup(provider, model)
