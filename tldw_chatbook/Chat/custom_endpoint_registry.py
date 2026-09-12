"""Pure registry for user-defined Console custom endpoints (ADR-146).

Entries persist as ``[custom_endpoints.<slug>]`` tables in the CLI config,
outside ``api_settings``, so provider-key iteration surfaces never see
registry plumbing. This module owns load/validate/mutate helpers only:
writes go through ``save_settings_to_cli_config`` (fed the mapping returned
by :func:`build_entry_mutation`) and removals through
``delete_settings_from_cli_config("custom_endpoints", [slug])`` --
both invoked by callers off-thread, never here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

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
#: Suffix search runs ``-2`` .. ``-9999`` (4-digit suffixes) before giving up.
_MAX_SLUG_COLLISION_SUFFIX = 9999
_DISPLAY_NAME_REQUIRED_COPY = "Display name is required."
_DISPLAY_NAME_TOO_LONG_COPY = (
    f"Display name must be {_MAX_DISPLAY_NAME_LENGTH} characters or fewer."
)
_INVALID_BASE_URL_COPY = "Base URL must be a valid http(s) URL."
_SLUG_COLLISION_COPY = "That name is already in use; choose another."


class CustomEndpointSlugError(ValueError):
    """A display name could not derive a slug that avoids ``existing_slugs``.

    Subclasses :class:`ValueError` so the F9 convert worker (and any other
    ``except ValueError`` boundary) surfaces ``str(exc)`` -- the collision
    copy -- in user-facing status surfaces as-is.

    Args:
        message: User-facing collision copy carried by ``str(exc)``;
            defaults to the standard name-in-use message and may be
            overridden by callers with context-specific copy.
    """

    def __init__(self, message: str = _SLUG_COLLISION_COPY) -> None:
        super().__init__(message)


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


class _EndpointEntryConfig(BaseModel):
    """Validated raw shape of one ``[custom_endpoints.<slug>]`` table.

    The boundary model for ``load_custom_endpoints`` (ADR-146): every raw
    mapping is validated through it before any downstream construction, so
    malformed optional credentials and model collections surface as
    structured field errors instead of silent normalization. Business rules
    that need family context (URL normalization, family membership) stay in
    ``validate_entry`` -- the model carries only shape constraints.
    """

    model_config = {"extra": "ignore"}

    display_name: str
    family: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    models: tuple[str, ...] = ()
    created_from: str | None = None

    @field_validator("api_key_env", "api_key", "created_from")
    @classmethod
    def _optional_blank_is_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("models", mode="before")
    @classmethod
    def _models_reject_non_sequences(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return value
        raise ValueError("models must be a list of model ids")

    @field_validator("models")
    @classmethod
    def _models_keep_non_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item for item in value if item and item.strip())


def split_custom_endpoint_id(provider: str | None) -> str | None:
    """Return the slug when ``provider`` is ``custom-ep:<slug>``, else None.

    Accepts the canonicalized spelling ``custom_ep:<slug>`` as well:
    ``provider_config_key`` (the readiness id canonicalizer) rewrites dashes
    to underscores, so an id round-tripped through it reaches the registry in
    that form (ADR-146; the defaults-mutation path is the first such caller).

    Args:
        provider: Candidate provider id (may be None or any string).

    Returns:
        The slug after the prefix, or None when ``provider`` is not a
        registry id (including the bare prefix with an empty slug).
    """
    if not isinstance(provider, str):
        return None
    for prefix in (
        CUSTOM_ENDPOINT_ID_PREFIX,
        CUSTOM_ENDPOINT_ID_PREFIX.replace("-", "_"),
    ):
        if provider.startswith(prefix):
            slug = provider[len(prefix) :]
            return slug or None
    return None


def _custom_endpoints_section(
    app_config: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the ``custom_endpoints`` table from either config shape.

    Raw CLI config carries the table at the top level; the app's normalized
    ``load_settings()`` shape preserves it only nested under
    ``COMPREHENSIVE_CONFIG_RAW`` (the same projection gap the gateway's
    ``[caching]`` reader documents, PR #1239). A top-level table wins when
    both are present.
    """
    top_level = app_config.get("custom_endpoints")
    if isinstance(top_level, Mapping):
        return top_level
    raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    if isinstance(raw, Mapping):
        nested = raw.get("custom_endpoints")
        if isinstance(nested, Mapping):
            return nested
    return {}


def load_custom_endpoints(
    app_config: Mapping[str, object],
) -> dict[str, CustomEndpointEntry]:
    """Return valid entries keyed by slug, dropping invalid ones.

    Invalid entries disable the registry surface for that entry only (one
    ``logger.warning`` naming the slug each); built-in providers and other
    entries are unaffected.

    Args:
        app_config: The full CLI config mapping — either the raw CLI shape
            (``custom_endpoints`` at the top level) or the normalized
            ``load_settings()`` shape (nested under
            ``COMPREHENSIVE_CONFIG_RAW``).

    Returns:
        Valid :class:`CustomEndpointEntry` values keyed by slug; empty when
        the ``custom_endpoints`` table is absent or malformed.
    """
    raw_section = _custom_endpoints_section(app_config)
    entries: dict[str, CustomEndpointEntry] = {}
    for slug, raw_entry in raw_section.items():
        if not isinstance(raw_entry, Mapping):
            logger.warning("custom endpoint '%s' ignored: malformed section", slug)
            continue
        try:
            config = _EndpointEntryConfig.model_validate(dict(raw_entry))
        except ValueError as exc:
            # Pydantic messages embed rejected INPUT VALUES, which for a
            # malformed credential field would write the secret itself into
            # the persistent log -- log only the field/type taxonomy.
            details = "; ".join(
                f"{'.'.join(str(part) for part in error.get('loc', ()))}:"
                f"{error.get('type', 'invalid')}"
                for error in (
                    exc.errors() if hasattr(exc, "errors") else []
                )
            )
            logger.warning(
                "custom endpoint '%s' ignored: malformed entry (%s)",
                slug,
                details or "invalid entry",
            )
            continue
        display_name = config.display_name
        family = config.family
        base_url = config.base_url
        reasons = validate_entry(display_name, family, base_url)
        if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
            reasons.append(
                "slug must be lowercase letters, digits, and hyphens (1-64 chars)"
            )
        if reasons:
            logger.warning("custom endpoint '%s' ignored: %s", slug, "; ".join(reasons))
            continue
        entries[slug] = CustomEndpointEntry(
            slug=slug,
            display_name=display_name.strip(),
            family=family,
            base_url=_normalize_base_url(family, base_url),
            api_key_env=config.api_key_env,
            api_key=config.api_key,
            models=config.models,
            created_from=config.created_from,
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
    ``-``, trims leading/trailing ``-``, and appends ``-2`` .. ``-9999`` on
    collision with ``existing_slugs``. Long bases are stem-truncated so
    each suffix always fits whole inside the 64-char :data:`SLUG_PATTERN`
    contract (clamping the assembled candidate instead would shear the
    suffix off a long base and reject creatable names).

    Args:
        display_name: User-facing name to derive from.
        existing_slugs: Slugs already taken.

    Returns:
        The derived slug, unique against ``existing_slugs``.

    Raises:
        CustomEndpointSlugError: Every candidate -- the bare base and all
            suffixed stem forms -- collides with ``existing_slugs``.
            Returning a colliding slug here would make the creation path
            overwrite an existing entry's config section.
    """
    base = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")
    # SLUG_PATTERN contract: 1..64 chars of [a-z0-9-]. A punctuation-only
    # name derives the empty string and a long name overflows 64 chars --
    # both would be persisted by the creation path and then silently
    # discarded by ``load_custom_endpoints`` on the next reload, so clamp
    # here with a stable fallback instead.
    if not base:
        base = "endpoint"
    base = base[:64].rstrip("-") or "endpoint"
    if SLUG_PATTERN.fullmatch(base) and base not in existing_slugs:
        return base
    for suffix in range(2, _MAX_SLUG_COLLISION_SUFFIX + 1):
        suffix_text = f"-{suffix}"
        # Truncate the stem before appending so the suffix survives whole:
        # the base starts with an alphanumeric, so the stem cannot empty out.
        candidate = f"{base[: 64 - len(suffix_text)].rstrip('-')}{suffix_text}"
        if SLUG_PATTERN.fullmatch(candidate) and candidate not in existing_slugs:
            return candidate
    raise CustomEndpointSlugError()


def build_entry_mutation(
    entry: CustomEndpointEntry,
) -> dict[str, dict[str, object]]:
    """Build the config mutation section for persisting ``entry``.

    Feed the result to ``save_settings_to_cli_config``; delete an entry with
    ``delete_settings_from_cli_config("custom_endpoints", [slug])``.

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


def custom_endpoint_provider_settings(
    app_config: Mapping[str, object], provider: str | None
) -> Mapping[str, object] | None:
    """Provider-settings view for a custom-ep id: the entry flattened to
    the provider-settings key aliases.

    Args:
        app_config: The full CLI config mapping.
        provider: Candidate provider id.

    Returns:
        ``{'api_base_url': entry.base_url, 'api_url': entry.base_url,
        'api_key': entry.api_key or '', 'api_key_env': entry.api_key_env,
        'model': entry.models[0] if any}``, or None when ``provider`` is
        not a resolvable custom-ep id.
    """
    entry = entry_for(app_config, provider)
    if entry is None:
        return None
    view: dict[str, object] = {
        "api_base_url": entry.base_url,
        "api_url": entry.base_url,
        "api_key": entry.api_key or "",
        "api_key_env": entry.api_key_env,
    }
    if entry.models:
        view["model"] = entry.models[0]
    return view


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
