"""Pure registry for user-defined Console custom endpoints (ADR-146).

Entries persist as ``[custom_endpoints.<slug>]`` tables in the CLI config,
outside ``api_settings``, so provider-key iteration surfaces never see
registry plumbing. This module owns load/validate/mutate helpers only:
writes go through ``save_settings_to_cli_config`` (fed the mapping returned
by :func:`build_entry_mutation`) and removals through
``delete_settings_from_cli_config("custom_endpoints.<slug>", [...])`` --
both invoked by callers off-thread, never here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Chat.console_session_settings import normalize_llamacpp_base_url
from tldw_chatbook.Utils.input_validation import validate_url

logger = logging.getLogger(__name__)

#: Prefix marking a provider id that addresses a registry entry.
CUSTOM_ENDPOINT_ID_PREFIX = "custom-ep:"
#: Endpoint families a registry entry can execute as.
ENDPOINT_FAMILIES = frozenset({"llama_cpp", "openai_compatible", "ollama"})
#: Slugs are lowercase letters, digits, and hyphens, 1-64 chars.
SLUG_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")

_MAX_DISPLAY_NAME_LENGTH = 80
_MAX_SLUG_COLLISION_SUFFIX = 99
_DISPLAY_NAME_REQUIRED_COPY = "Display name is required."
_DISPLAY_NAME_TOO_LONG_COPY = (
    f"Display name must be {_MAX_DISPLAY_NAME_LENGTH} characters or fewer."
)
_INVALID_BASE_URL_COPY = "Base URL must be a valid http(s) URL."


@dataclass(frozen=True)
class CustomEndpointEntry:
    """One named endpoint entry persisted under ``custom_endpoints.<slug>``.

    Attributes:
        slug: Immutable lowercase ``[a-z0-9-]`` identity key.
        display_name: User-facing label; renames change only this.
        family: One of :data:`ENDPOINT_FAMILIES`; selects the execution path.
        base_url: Validated, family-normalized endpoint URL.
        api_key_env: Optional environment-variable credential reference.
        api_key: Optional stored key; display/log paths never read this,
            and it is excluded from the dataclass repr so secrets never
            reach logs or debug output.
        models: Cached model list discovered for this endpoint.
        created_from: Template provider id this entry was created from
            (informational only).
    """

    slug: str
    display_name: str
    family: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = field(default=None, repr=False)
    models: tuple[str, ...] = ()
    created_from: str | None = None


def split_custom_endpoint_id(provider: str | None) -> str | None:
    """Return the slug when ``provider`` is ``custom-ep:<slug>``, else None.

    Args:
        provider: Candidate provider id (may be None or any string).

    Returns:
        The slug after the prefix, or None when ``provider`` is not a
        registry id (including the bare prefix with an empty slug).
    """
    if not isinstance(provider, str) or not provider.startswith(
        CUSTOM_ENDPOINT_ID_PREFIX
    ):
        return None
    slug = provider[len(CUSTOM_ENDPOINT_ID_PREFIX) :]
    return slug or None


def load_custom_endpoints(
    app_config: Mapping[str, object],
) -> dict[str, CustomEndpointEntry]:
    """Return valid entries keyed by slug, dropping invalid ones.

    Invalid entries disable the registry surface for that entry only (one
    ``logger.warning`` naming the slug each); built-in providers and other
    entries are unaffected.

    Args:
        app_config: The full CLI config mapping.

    Returns:
        Valid :class:`CustomEndpointEntry` values keyed by slug; empty when
        the ``custom_endpoints`` table is absent or malformed.
    """
    raw_section = app_config.get("custom_endpoints")
    if not isinstance(raw_section, Mapping):
        raw_section = {}
    entries: dict[str, CustomEndpointEntry] = {}
    for slug, raw_entry in raw_section.items():
        if not isinstance(raw_entry, Mapping):
            logger.warning(
                "custom endpoint '%s' ignored: malformed section", slug
            )
            continue
        display_name = _string_value(raw_entry.get("display_name"))
        family = _string_value(raw_entry.get("family"))
        base_url = _string_value(raw_entry.get("base_url"))
        reasons = validate_entry(display_name, family, base_url)
        if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
            reasons.append(
                "slug must be lowercase letters, digits, and hyphens "
                "(1-64 chars)"
            )
        if reasons:
            logger.warning(
                "custom endpoint '%s' ignored: %s", slug, "; ".join(reasons)
            )
            continue
        entries[slug] = CustomEndpointEntry(
            slug=slug,
            display_name=display_name.strip(),
            family=family,
            base_url=_normalize_base_url(family, base_url),
            api_key_env=_optional_string(raw_entry.get("api_key_env")),
            api_key=_optional_string(raw_entry.get("api_key")),
            models=_parse_models(raw_entry.get("models")),
            created_from=_optional_string(raw_entry.get("created_from")),
        )
    return entries


def entry_for(
    app_config: Mapping[str, object], provider: str | None
) -> CustomEndpointEntry | None:
    """Resolve a ``custom-ep:<slug>`` provider id to its entry.

    Args:
        app_config: The full CLI config mapping.
        provider: Candidate provider id.

    Returns:
        The entry for the slug, or None when ``provider`` is not a registry
        id or its slug has no valid entry.
    """
    slug = split_custom_endpoint_id(provider)
    if slug is None:
        return None
    return load_custom_endpoints(app_config).get(slug)


def derive_slug(display_name: str, existing_slugs: Collection[str]) -> str:
    """Derive a unique slug from a display name.

    Lowercases, maps each run of non-``[a-z0-9]`` characters to a single
    ``-``, trims leading/trailing ``-``, and appends ``-2`` .. ``-99`` on
    collision with ``existing_slugs``.

    Args:
        display_name: User-facing name to derive from.
        existing_slugs: Slugs already taken.

    Returns:
        The derived slug (unmodified base when every suffixed candidate
        ``-2`` .. ``-99`` is taken).
    """
    base = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")
    if base not in existing_slugs:
        return base
    for suffix in range(2, _MAX_SLUG_COLLISION_SUFFIX + 1):
        candidate = f"{base}-{suffix}"
        if candidate not in existing_slugs:
            return candidate
    return base


def build_entry_mutation(
    entry: CustomEndpointEntry,
) -> dict[str, dict[str, object]]:
    """Build the config mutation section for persisting ``entry``.

    Feed the result to ``save_settings_to_cli_config``; delete an entry with
    ``delete_settings_from_cli_config("custom_endpoints.<slug>", ...)``.

    Args:
        entry: The entry to persist.

    Returns:
        ``{'custom_endpoints.<slug>': {...}}`` with ``None`` fields omitted;
        ``models`` is always present as a list.
    """
    values: dict[str, Any] = {
        "display_name": entry.display_name,
        "family": entry.family,
        "base_url": entry.base_url,
        "models": list(entry.models),
    }
    if entry.api_key_env is not None:
        values["api_key_env"] = entry.api_key_env
    if entry.api_key is not None:
        values["api_key"] = entry.api_key
    if entry.created_from is not None:
        values["created_from"] = entry.created_from
    return {f"custom_endpoints.{entry.slug}": values}


def validate_entry(display_name: str, family: str, base_url: str) -> list[str]:
    """Return user-facing validation errors for a candidate entry.

    Checks a non-blank display name of at most 80 characters, a known
    family, and a base URL that passes ``validate_url`` after
    family-appropriate normalization. URL checking is skipped for unknown
    families (there is no normalization rule to apply).

    Args:
        display_name: Candidate display name.
        family: Candidate family.
        base_url: Candidate base URL.

    Returns:
        Error messages in display order; empty when the candidate is valid.
    """
    errors: list[str] = []
    name = display_name.strip() if isinstance(display_name, str) else ""
    if not name:
        errors.append(_DISPLAY_NAME_REQUIRED_COPY)
    elif len(name) > _MAX_DISPLAY_NAME_LENGTH:
        errors.append(_DISPLAY_NAME_TOO_LONG_COPY)
    if family not in ENDPOINT_FAMILIES:
        errors.append(f"Unknown endpoint family: {family}.")
        return errors
    if not validate_url(_normalize_base_url(family, base_url)):
        errors.append(_INVALID_BASE_URL_COPY)
    return errors


def family_execution_key(family: str) -> str:
    """Map a registry family onto its Console execution provider key.

    Args:
        family: One of :data:`ENDPOINT_FAMILIES`.

    Returns:
        ``'llama_cpp'`` for ``llama_cpp``, ``'ollama'`` for ``ollama``, and
        ``'custom'`` for ``openai_compatible`` (which rides the generic
        OpenAI-compatible path).
    """
    return "custom" if family == "openai_compatible" else family


def family_normalizes_like_llama(family: str) -> bool:
    """Return whether ``family`` uses llama.cpp base-URL normalization.

    Args:
        family: Candidate family.

    Returns:
        True only for ``'llama_cpp'``.
    """
    return family == "llama_cpp"


def _normalize_base_url(family: str, base_url: str) -> str:
    """Normalize ``base_url`` per family.

    Args:
        family: Known endpoint family.
        base_url: Raw candidate URL.

    Returns:
        ``normalize_llamacpp_base_url`` output for the llama family;
        otherwise the stripped URL with trailing ``/`` removed.
    """
    raw = str(base_url or "").strip()
    if family_normalizes_like_llama(family):
        return normalize_llamacpp_base_url(raw)
    return raw.rstrip("/")


def _string_value(value: object) -> str:
    """Coerce a config value to ``str`` (blank-safe)."""
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str | None:
    """Return a non-blank ``str`` or None for missing/blank values."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _parse_models(value: object) -> tuple[str, ...]:
    """Keep the non-blank string items of a list/tuple ``models`` value."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
