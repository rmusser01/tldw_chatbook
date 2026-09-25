"""Shared TOML/keyring/secret-precedence config machinery (ADR-176).

Both modality packages parse their ``[<modality>_generation]`` section with
the same mechanics: nested TOML (globals + ``[<modality>_generation.<backend>]``
subsections), secret precedence env -> config -> keyring (namespaced per
modality), and warn-on-unknown-key for the flat-spelling mistake. The
per-modality tables stay in their packages; the mechanics live here once.

Behavioral deltas resolved per ADR-176: the config-key back-compat fallback
(image) is shared -- a no-op for backends whose config key already is
``api_key``; non-dict subsection/raw guards (video) are shared, being the
defensive-strict form; unknown-key scan failure logs use the error-type-only
debug form (the privacy-stricter pattern).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from typing import Any

import keyring
from loguru import logger


@dataclass(frozen=True)
class SecretSpec:
    """One backend's secret resolution data.

    Args:
        flat_field: Flat config field the resolved secret is stored under.
        env_vars: Environment variables, in precedence order.
        keyring_id: Keyring entry id inside the modality namespace.
        config_key: Nested TOML key the secret is read from/written to.
    """

    flat_field: str
    env_vars: tuple[str, ...]
    keyring_id: str
    config_key: str


@dataclass(frozen=True)
class ModalityConfigTables:
    """The per-modality data the shared machinery is parameterized by.

    Args:
        section_name: TOML section label for warnings, e.g. ``image_generation``.
        keyring_namespace: Keyring service name for this modality's secrets.
        keyring_label: Short label for keyring debug logs (``videogen``).
        global_keys: Known keys directly under the section.
        extra_exempt: Non-backend keys exempt from the unknown-key warning
            (image's ``styles``).
        secrets: Backend id -> secret resolution data.
        non_secret: ``(backend, toml_key)`` -> flat field name.
    """

    section_name: str
    keyring_namespace: str
    global_keys: tuple[str, ...]
    keyring_label: str = "media"
    extra_exempt: frozenset[str] = frozenset()
    secrets: Mapping[str, SecretSpec] = field(default_factory=dict)
    non_secret: Mapping[tuple[str, str], str] = field(default_factory=dict)

    @property
    def flat_map(self) -> dict[str, tuple[str, str]]:
        """Reverse map used only for the unknown-key warning."""
        mapping = {flat: key for key, flat in self.non_secret.items()}
        mapping.update(
            {
                spec.flat_field: (backend, spec.config_key)
                for backend, spec in self.secrets.items()
            }
        )
        return mapping

    @property
    def backend_names(self) -> set[str]:
        """Known ``[<section>.<backend>]`` subsection names."""
        return set(self.secrets) | {backend for backend, _ in self.non_secret}


# --- shared value coercers (byte-identical in both packages) ---


def coerce_int(value: Any, default: int) -> int:
    """Coerce a value to int, falling back to ``default`` on failure.

    Args:
        value: Raw config value.
        default: Fallback.

    Returns:
        The parsed integer.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, default: float) -> float:
    """Coerce a value to float, falling back to ``default`` on failure.

    Args:
        value: Raw config value.
        default: Fallback.

    Returns:
        The parsed float.
    """
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a value to bool, falling back to ``default``.

    Args:
        value: Raw config value (bool or common string spellings).
        default: Fallback.

    Returns:
        The parsed boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def coerce_bool_flag_or_warn(
    value: Any, default: bool, *, section: str, key: str
) -> bool:
    """Coerce a feature-flag setting, warning when a *present* value is malformed.

    ``Utils.coerce_bool_flag`` substitutes ``default`` for anything outside its
    vocabulary, which is right for a UI preference but hides a typo'd TOML gate:
    ``allow_uploads = "yess"`` reads exactly like an unset key, so the user sets
    a gate and nothing happens. Config loading still must never raise (same
    contract as :func:`warn_unknown_top_level_keys`), so the bad value is logged
    and the default used.

    Absence is not malformed: only a non-``None`` value is inspected, and the
    vocabulary itself accepts ``0``/``False``/``"off"``, so a deliberate
    switch-off never warns.

    Args:
        value: Raw config value (``None`` when the key is absent).
        default: Fallback used when the value is absent or unrecognized.
        section: TOML section label for the warning, e.g. ``video_generation.minimax``.
        key: TOML key name for the warning.

    Returns:
        The coerced boolean, or ``default`` when the value is absent/malformed.
    """
    from tldw_chatbook.Utils.Utils import coerce_bool_flag  # ADR-097: lazy

    if value is None:
        return default
    # Probing both defaults detects a fallback without restating the true/false
    # vocabulary here -- coerce_bool_flag stays its single source of truth.
    if coerce_bool_flag(value, True) != coerce_bool_flag(value, False):
        # The rejected spelling is what makes the warning actionable, but it is
        # user-typed config text going to a persistent sink -- bound it, the way
        # the unknown-key scan logs only the short key token.
        logger.warning(
            f"[{section}] {key} = {repr(value)[:40]} is not a recognized boolean "
            f"(true/false/1/0/yes/no/on/off) -- ignored, using {default}"
        )
        return default
    return coerce_bool_flag(value, default)


def coerce_choice(value: Any, *, default: str, allowed: set[str]) -> str:
    """Normalize a string choice to lowercase and return ``default`` when invalid."""
    raw = str(value or "").strip().lower()
    if raw in allowed:
        return raw
    return default


def parse_list(value: Any) -> list[str]:
    """Parse a value into a list of stripped non-empty strings.

    Args:
        value: Raw config value (list, JSON array, or comma-separated).

    Returns:
        The parsed strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_config_value(section: Mapping[str, Any], key: str) -> str | None:
    """Read a stripped, non-empty string value from a section.

    Args:
        section: Mapping to read from.
        key: Key to read.

    Returns:
        The stripped value, or None when absent/empty.
    """
    raw = section.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


# --- shared section machinery ---


def warn_unknown_top_level_keys(raw: Any, tables: ModalityConfigTables) -> None:
    """Warn once per unknown key found directly under the section.

    A backend field written using its *flat* dataclass name straight under
    ``[<section>]`` matches nothing and is silently ignored -- this surfaces
    that mistake with the exact nested replacement to use. Never raises
    (config loading must never crash on a malformed/unexpected key).
    """
    try:
        for key in raw:
            if not isinstance(key, str):
                continue
            if key in tables.global_keys or key in tables.extra_exempt:
                continue
            if key in tables.backend_names:
                continue
            target = tables.flat_map.get(key)
            if target is not None:
                backend, toml_key = target
                logger.warning(
                    f"[{tables.section_name}] unknown key '{key}' is ignored -- flat backend keys are "
                    f"not read here; use [{tables.section_name}.{backend}] {toml_key} = ... instead"
                )
            else:
                logger.warning(f"[{tables.section_name}] unknown key '{key}' is ignored")
    except Exception as e:  # never let a malformed section crash config loading
        logger.debug(
            "{} unknown-key scan failed (error_type={})",
            tables.section_name,
            type(e).__name__,
        )


# TASK-32924: Settings' Image/Video Gen panels resolve secrets in compose(),
# on the UI loop, on every open/save/revert/Test -- one keyring round trip
# per keyless backend (SecretService over D-Bus on Linux). The app never
# writes these entries, so a short TTL is the only invalidation needed.
# ponytail: a key added with `keyring set` shows up within this window.
_KEYRING_READ_TTL_SECONDS = 10.0
_KEYRING_READS: dict[tuple[str, str], tuple[float, str | None]] = {}


def keyring_get(backend: str, tables: ModalityConfigTables) -> str | None:
    """Namespaced keyring lookup through a short-lived cache; never raises.

    Args:
        backend: Backend id, used as the keyring username.
        tables: The modality's config tables (supplies the keyring service).

    Returns:
        The stored secret, or None when absent or when the keyring backend
        failed. Either result is reused for ``_KEYRING_READ_TTL_SECONDS``
        after the lookup returns.
    """
    key = (tables.keyring_namespace, backend)
    now = time.monotonic()
    hit = _KEYRING_READS.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    try:
        value = keyring.get_password(tables.keyring_namespace, backend)
    except Exception as e:  # keyring backend may be unavailable
        logger.debug(
            "keyring lookup failed for {}/{} (error_type={})",
            tables.keyring_label,
            backend,
            type(e).__name__,
        )
        value = None
    # Expiry starts when the (possibly blocking) lookup returns.
    _KEYRING_READS[key] = (time.monotonic() + _KEYRING_READ_TTL_SECONDS, value)
    return value


def resolve_secret(
    backend: str,
    sub: Any,
    tables: ModalityConfigTables,
    *,
    keyring_lookup: Callable[[str], str | None],
) -> tuple[str, str | None, str]:
    """Resolve one backend's secret and where it came from.

    Args:
        backend: Backend id whose secret is resolved.
        sub: The backend's nested TOML subsection (or None).
        tables: The modality's config tables.
        keyring_lookup: The modality's keyring read (its test patch point).

    Returns:
        ``(flat_field_name, value, source)`` where ``source`` is one of
        ``"env:<VAR>"``, ``"config"``, ``"keyring"``, or ``"missing"``.
    """
    spec = tables.secrets[backend]
    for ev in spec.env_vars:                  # 1. env
        v = os.getenv(ev)
        if v:
            return spec.flat_field, v, f"env:{ev}"
    sub = sub or {}
    cfg_val = sub.get(spec.config_key)        # 2. config (per-backend key)
    if not cfg_val and spec.config_key != "api_key":
        # Back-compat fallback (see the image _SECRETS comment) -- e.g. a
        # swarmui section hand-written (or saved before this fix) with
        # `api_key` instead of its real `swarm_token` key. A no-op for
        # backends whose config key already is ``api_key``.
        cfg_val = sub.get("api_key")
    if cfg_val and cfg_val != "<API_KEY_HERE>":
        return spec.flat_field, cfg_val, "config"
    if sub.get("auth_reference") == "recovery:setup_required":
        return spec.flat_field, None, "missing"
    kr = keyring_lookup(spec.keyring_id)      # 3. keyring
    if kr:
        return spec.flat_field, kr, "keyring"
    return spec.flat_field, None, "missing"


def load_generation_section(
    tables: ModalityConfigTables,
    *,
    read_toml: Callable[[], Any],
    keyring_lookup: Callable[[str], str | None],
) -> tuple[dict, dict[str, str]]:
    """Assemble the FLAT mapping the config builder expects.

    Args:
        tables: The modality's config tables.
        read_toml: The modality's raw-section reader (its test patch point).
        keyring_lookup: The modality's keyring read (its test patch point).

    Returns:
        ``(flat, key_sources)``; ``key_sources`` maps every known backend id
        to where its secret was resolved from.
    """
    raw = read_toml()
    if not isinstance(raw, dict):
        raw = {}
    warn_unknown_top_level_keys(raw, tables)
    flat: dict = {}
    for k in tables.global_keys:
        if k in raw:
            flat[k] = raw[k]
    for (backend, toml_key), flat_field in tables.non_secret.items():
        sub = raw.get(backend) or {}
        if not isinstance(sub, dict):
            sub = {}
        if toml_key in sub:
            flat[flat_field] = sub[toml_key]
    key_sources: dict[str, str] = {backend: "missing" for backend in tables.backend_names}
    for backend in tables.secrets:
        sub = raw.get(backend) or {}
        if not isinstance(sub, dict):
            sub = {}
        field_name, value, source = resolve_secret(
            backend, sub, tables, keyring_lookup=keyring_lookup
        )
        key_sources[backend] = source
        if value:
            flat[field_name] = value
    return flat, key_sources
