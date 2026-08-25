"""Pure state contracts for the first-run setup wizard.

No Textual imports, no I/O — every function is a pure transform over the
in-memory app config, mirroring Chat/console_onboarding_state.py. The wizard
Screen owns rendering and persistence; this module owns every decision.

Note on ``any_provider_configured``: it checks api_key / api_key_env_var
only, never endpoint-URL keys (api_url, api_base_url, ...). See that
function's docstring for the UAT incident that made endpoint-checking
actively wrong -- the shipped config.toml template ships ~12 default
[api_settings.*] endpoint URLs, which made every fresh install look
"configured" and the wizard never auto-offered.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from unicodedata import category as unicode_category

if TYPE_CHECKING:
    from tldw_chatbook.Chat.provider_setup_persistence import ProviderSetupMutation

WIZARD_STATE_SECTION = "first_run"
SETUP_STARTED_KEY = "setup_started"
SETUP_COMPLETED_KEY = "setup_completed"
SETUP_DRAFT_VERSION = 1

DRAFT_VERSION_KEY = "draft_version"
DRAFT_TRACK_KEY = "draft_track"
DRAFT_ACTIVE_STEP_KEY = "active_step_id"
DRAFT_VALUES_KEY = "draft_values"
DRAFT_RESUME_ATTEMPTED_KEY = "resume_attempted"
SETUP_DRAFT_KEYS = (
    DRAFT_VERSION_KEY,
    DRAFT_TRACK_KEY,
    DRAFT_ACTIVE_STEP_KEY,
    DRAFT_VALUES_KEY,
    DRAFT_RESUME_ATTEMPTED_KEY,
)

_MAX_SETUP_DRAFT_FIELDS = 64
_MAX_SETUP_DRAFT_BYTES = 16 * 1024
_MAX_PROVIDER_CHARS = 128
_MAX_ENDPOINT_CHARS = 4096
_MAX_MODEL_CHARS = 120
_MAX_CREDENTIAL_CHARS = 8192
_MAX_IDENTITY_COUNTER = 2**63 - 1
_SECRET_FIELD_TOKENS = ("api_key", "credential", "password", "token", "secret")
_CREDENTIAL_SOURCES = frozenset({"none", "draft", "environment", "stored"})
_ENDPOINT_REQUIRED_PROVIDER_KEYS = frozenset(
    {"custom", "custom_2", "llama_cpp", "local_llamacpp"}
)
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_ENV_VAR_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

_PLACEHOLDER_MARKERS = ("<", ">")

FirstRunCredentialSource = Literal["none", "draft", "environment", "stored"]
FirstRunSummaryAction = Literal[
    "start_chatting", "review_provider", "explore_home", "review_settings"
]


def is_untouched_default_session(
    session: object,
    messages: object,
    draft: object,
    staged_attachments: object,
) -> bool:
    """Delegate first-run eligibility to Console's canonical pure predicate."""

    from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Chat.console_chat_store import (
        is_untouched_default_session as console_session_is_untouched,
    )

    if (
        not isinstance(session, ConsoleChatSession)
        or session.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
    ):
        return False
    try:
        return console_session_is_untouched(
            session,
            messages,  # type: ignore[arg-type]
            draft,  # type: ignore[arg-type]
            staged_attachments,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        return False


class _CredentialValueOwner:
    """Provide an in-memory value slot that dataclass serialization cannot see."""

    __slots__ = ("_value",)


@dataclass(frozen=True, slots=True, init=False)
class ProviderCredentialDraft(_CredentialValueOwner):
    """One bounded credential decision whose value is never a dataclass field."""

    source: FirstRunCredentialSource = field(init=False)
    revision: int = field(init=False)

    def __init__(
        self,
        source: FirstRunCredentialSource,
        value: str,
        revision: int = 0,
    ) -> None:
        if type(source) is not str or source not in _CREDENTIAL_SOURCES:
            raise ValueError("Credential source is invalid.")
        if type(revision) is not int or not 0 <= revision <= _MAX_IDENTITY_COUNTER:
            raise ValueError("Credential revision is invalid.")
        if type(value) is not str or len(value) > _MAX_CREDENTIAL_CHARS:
            raise ValueError("Credential value is invalid.")
        if any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in value
        ):
            raise ValueError("Credential value is invalid.")
        if source in {"none", "stored"} and value:
            raise ValueError("Credential value conflicts with its source.")
        if source == "environment" and _ENV_VAR_PATTERN.fullmatch(value) is None:
            raise ValueError("Credential environment variable is invalid.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "_value", value)

    def __getattribute__(self, name: str) -> object:
        if name == "_value":
            raise AttributeError("credential value is memory-only")
        return object.__getattribute__(self, name)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ProviderCredentialDraft is sealed.")

    def __copy__(self) -> object:
        raise TypeError("Provider credentials are memory-only.")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("Provider credentials are memory-only.")

    def __reduce__(self) -> object:
        raise TypeError("Provider credentials are memory-only.")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("Provider credentials are memory-only.")


def _credential_value_for_boundary(credential: ProviderCredentialDraft) -> str:
    """Reveal a credential only to the first-run probe/persistence boundaries."""

    if type(credential) is not ProviderCredentialDraft:
        raise ValueError("Credential draft is invalid.")
    return object.__getattribute__(credential, "_value")


@dataclass(frozen=True, slots=True)
class FirstRunProviderDraft:
    """The provider connection staged between Provider and Model."""

    provider: str
    endpoint: str
    credential: ProviderCredentialDraft = field(repr=False)
    discovery_endpoint: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.provider) is not str
            or not self.provider
            or len(self.provider) > _MAX_PROVIDER_CHARS
            or self.provider != self.provider.strip()
            or _contains_unsafe_text(self.provider)
        ):
            raise ValueError("Provider is invalid.")
        if (
            type(self.endpoint) is not str
            or len(self.endpoint) > _MAX_ENDPOINT_CHARS
            or _contains_unsafe_text(self.endpoint)
        ):
            raise ValueError("Endpoint is invalid.")
        if (
            type(self.discovery_endpoint) is not str
            or len(self.discovery_endpoint) > _MAX_ENDPOINT_CHARS
            or _contains_unsafe_text(self.discovery_endpoint)
        ):
            raise ValueError("Discovery endpoint is invalid.")
        if type(self.credential) is not ProviderCredentialDraft:
            raise ValueError("Credential draft is invalid.")


@dataclass(frozen=True, slots=True)
class FirstRunModelDiscoveryKey:
    """Secret-free identity for model discovery against one exact connection."""

    provider_key: str
    connection_identity: tuple[str, str]
    credential_source: FirstRunCredentialSource
    credential_revision: int

    def __post_init__(self) -> None:
        from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

        if (
            type(self.credential_revision) is not int
            or not 0 <= self.credential_revision <= _MAX_IDENTITY_COUNTER
        ):
            raise ValueError("Credential revision is invalid.")
        if (
            type(self.credential_source) is not str
            or self.credential_source not in _CREDENTIAL_SOURCES
        ):
            raise ValueError("Credential source is invalid.")
        try:
            ProviderDraftIdentity(
                provider_key=self.provider_key,
                connection_identity=self.connection_identity,
                credential_source=self.credential_source,
                credential_revision=self.credential_revision,
                draft_generation=0,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Model discovery identity is invalid.") from exc


def _contains_unsafe_text(value: str) -> bool:
    return any(
        unicode_category(character) in _UNSAFE_TEXT_CATEGORIES for character in value
    )


def build_first_run_model_discovery_key(
    provider_draft: FirstRunProviderDraft,
) -> FirstRunModelDiscoveryKey:
    """Build a canonical cache/evidence key without reading credential value."""

    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
        resolve_provider_endpoint,
    )

    if type(provider_draft) is not FirstRunProviderDraft:
        raise ValueError("Provider draft is invalid.")
    provider_key = _first_run_provider_owner_key(provider_draft.provider)
    endpoint = provider_draft.discovery_endpoint or provider_draft.endpoint
    if not endpoint:
        from tldw_chatbook.Chat.console_provider_endpoints import (
            builtin_provider_endpoint,
        )

        endpoint = builtin_provider_endpoint(provider_key) or ""
    resolution = resolve_provider_endpoint(provider_key, endpoint)
    identity = canonical_connection_identity(provider_key, endpoint)
    if resolution.errors or identity is None:
        raise ValueError("Provider endpoint is invalid.")
    return FirstRunModelDiscoveryKey(
        provider_key=resolution.provider_key,
        connection_identity=identity,
        credential_source=provider_draft.credential.source,
        credential_revision=provider_draft.credential.revision,
    )


def _first_run_provider_owner_key(provider: object) -> str:
    """Return one shared provider owner key without reading configuration."""

    from tldw_chatbook.Chat.provider_setup_persistence import canonical_provider_key

    return canonical_provider_key(provider)


def _validate_first_run_app_config(
    app_config: object, provider: str
) -> Mapping[str, object]:
    """Reject malformed matching provider tables before constructing a write."""

    if type(app_config) is not dict:
        raise TypeError("Application configuration is invalid.")
    api_settings = app_config.get("api_settings")
    if api_settings is None:
        return app_config
    if type(api_settings) is not dict:
        raise TypeError("Provider configuration is invalid.")
    target = _first_run_provider_owner_key(provider)
    for index, (configured_provider, settings) in enumerate(api_settings.items()):
        if index >= 256:
            raise ValueError("Provider configuration is too large.")
        try:
            configured_owner = _first_run_provider_owner_key(configured_provider)
        except ValueError:
            continue
        if configured_owner != target:
            continue
        if type(settings) is not dict:
            raise TypeError("Provider configuration is invalid.")
    return app_config


def _first_run_provider_settings(
    app_config: Mapping[str, object], provider: str
) -> Mapping[str, object]:
    """Return the same owned provider table selected by shared persistence."""

    api_settings = app_config.get("api_settings")
    if type(api_settings) is not dict:
        return {}
    target = _first_run_provider_owner_key(provider)
    exact = api_settings.get(target)
    if type(exact) is dict:
        return exact
    for index, (configured_provider, settings) in enumerate(api_settings.items()):
        if index >= 256:
            raise ValueError("Provider configuration is too large.")
        try:
            configured_owner = _first_run_provider_owner_key(configured_provider)
        except ValueError:
            continue
        if configured_owner == target and type(settings) is dict:
            return settings
    return {}


def resolve_first_run_provider_draft(
    provider_draft: FirstRunProviderDraft,
    app_config: object,
) -> FirstRunProviderDraft:
    """Resolve an untouched endpoint without turning blank into implicit clear."""

    if type(provider_draft) is not FirstRunProviderDraft:
        raise ValueError("Provider draft is invalid.")
    config = _validate_first_run_app_config(app_config, provider_draft.provider)
    endpoint, discovery_endpoint = _resolve_first_run_provider_endpoints(
        provider_draft.provider,
        provider_draft.endpoint,
        config,
    )
    return replace(
        provider_draft,
        endpoint=endpoint,
        discovery_endpoint=discovery_endpoint,
    )


def _resolve_first_run_provider_endpoints(
    provider: str,
    editable_endpoint: str,
    app_config: Mapping[str, object],
) -> tuple[str, str]:
    """Resolve persisted and runtime endpoints without credential state."""

    from tldw_chatbook.Chat.console_provider_endpoints import (
        effective_provider_discovery_endpoint,
        first_configured_endpoint,
    )
    from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint

    endpoint = editable_endpoint.strip()
    owner_key = _first_run_provider_owner_key(provider)
    provider_settings = _first_run_provider_settings(app_config, provider)
    if not endpoint:
        endpoint = first_configured_endpoint(provider_settings) or ""
    if endpoint:
        resolution = resolve_provider_endpoint(owner_key, endpoint)
        if resolution.errors or resolution.persisted_endpoint is None:
            raise ValueError("Provider endpoint is invalid.")
        endpoint = resolution.persisted_endpoint
    if owner_key in _ENDPOINT_REQUIRED_PROVIDER_KEYS:
        if not endpoint:
            raise ValueError("Provider endpoint is required.")
    discovery_endpoint = effective_provider_discovery_endpoint(
        owner_key,
        endpoint or None,
        provider_settings,
    )
    if discovery_endpoint:
        discovery_resolution = resolve_provider_endpoint(
            owner_key, discovery_endpoint
        )
        if discovery_resolution.errors or discovery_resolution.chat_url is None:
            raise ValueError("Provider discovery endpoint is invalid.")
        discovery_endpoint = discovery_resolution.chat_url
    return endpoint, discovery_endpoint or ""


def build_current_first_run_model_discovery_key(
    *,
    provider: object,
    editable_endpoint: object,
    credential_source: object,
    credential_revision: object,
    app_config: object,
) -> FirstRunModelDiscoveryKey:
    """Resolve a current secret-free discovery key for persistence CAS."""

    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )

    if (
        type(provider) is not str
        or type(editable_endpoint) is not str
        or len(editable_endpoint) > _MAX_ENDPOINT_CHARS
        or _contains_unsafe_text(editable_endpoint)
        or type(credential_source) is not str
        or credential_source not in _CREDENTIAL_SOURCES
        or type(credential_revision) is not int
        or not 0 <= credential_revision <= _MAX_IDENTITY_COUNTER
    ):
        raise ValueError("Provider discovery identity is invalid.")
    config = _validate_first_run_app_config(app_config, provider)
    _, discovery_endpoint = _resolve_first_run_provider_endpoints(
        provider,
        editable_endpoint,
        config,
    )
    provider_key = _first_run_provider_owner_key(provider)
    identity = canonical_connection_identity(provider_key, discovery_endpoint)
    if identity is None:
        raise ValueError("Provider discovery identity is invalid.")
    return FirstRunModelDiscoveryKey(
        provider_key=provider_key,
        connection_identity=identity,
        credential_source=credential_source,
        credential_revision=credential_revision,
    )


def validate_first_run_model_id(model_id: object) -> str:
    """Return one bounded display/config-safe model identifier."""

    if type(model_id) is not str:
        raise ValueError("Model is invalid.")
    model = model_id.strip()
    if (
        not model
        or len(model) > _MAX_MODEL_CHARS
        or not model.isprintable()
        or _contains_unsafe_text(model)
    ):
        raise ValueError("Model is invalid.")
    return model


def build_first_run_provider_commit(
    provider_draft: FirstRunProviderDraft,
    model_id: object,
    app_config: object,
) -> ProviderSetupMutation:
    """Delegate one validated first-run provider/default commit to its owner."""

    from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
    from tldw_chatbook.Chat.provider_setup_persistence import (
        ProviderSetupDraft,
        ProviderSetupMutation,
        build_provider_setup_mutation,
    )

    if type(provider_draft) is not FirstRunProviderDraft:
        raise ValueError("Provider draft is invalid.")
    config = _validate_first_run_app_config(app_config, provider_draft.provider)
    effective_draft = resolve_first_run_provider_draft(provider_draft, config)
    model = validate_first_run_model_id(model_id)
    if effective_draft.endpoint:
        resolution = resolve_provider_endpoint(
            _first_run_provider_owner_key(effective_draft.provider),
            effective_draft.endpoint,
        )
        if resolution.errors:
            raise ValueError("Provider endpoint is invalid.")

    credential = provider_draft.credential
    credential_value = _credential_value_for_boundary(credential)

    def shared_draft(source: str) -> ProviderSetupDraft:
        return ProviderSetupDraft(
            provider=effective_draft.provider,
            model=model,
            endpoint=effective_draft.endpoint,
            credential_source=source,
            credential_revision=credential.revision,
            draft_generation=0,
            credential_value=(
                credential_value
                if credential.source == "draft" and credential_value
                else None
            ),
            credential_env_var=(
                credential_value if credential.source == "environment" else None
            ),
        )

    if credential.source == "none":
        from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

        readiness = get_provider_readiness(effective_draft.provider, config)
        if str(readiness.api_key_source or "").startswith("config:"):
            source = "stored"
        elif str(readiness.api_key_source or "").startswith("env:"):
            source = "environment"
        else:
            source = "none"
        mutation = build_provider_setup_mutation(shared_draft(source), config)
    elif credential.source == "draft" and not credential_value:
        mutation = build_provider_setup_mutation(shared_draft("none"), config)
    else:
        mutation = build_provider_setup_mutation(
            shared_draft(credential.source), config
        )
    if type(mutation) is not ProviderSetupMutation:
        raise ValueError("Provider setup mutation is invalid.")
    return mutation


#: TASK-21143 (UAT S-1/M-2/N-7): how the last model-discovery probe for the
#: CURRENT provider identity ended. "" means no failure is known — either
#: the probe succeeded, never ran, or its identity was superseded (the
#: wizard's existing per-discovery-key invalidation guarantees staleness
#: never reaches these values).
PROVIDER_PROBE_NONE = ""
PROVIDER_PROBE_AUTH = "authentication"
PROVIDER_PROBE_CONNECTION = "connection"


def classify_discovery_failure(
    discovery_state: str, failure_category: str
) -> str:
    """Collapse a discovery UI outcome into the trust-chain's three states.

    Args:
        discovery_state: The model step's discovery state string
            ("available", "connection_failed", "listing_unavailable", ...).
        failure_category: The human category rendered with a failure
            ("authentication", "connection error", "request failed", ...).

    Returns:
        PROVIDER_PROBE_AUTH for credential rejections,
        PROVIDER_PROBE_CONNECTION for any other failed probe, and
        PROVIDER_PROBE_NONE when discovery did not fail (including
        "listing_unavailable", which is a provider capability, not an
        error in the user's setup).
    """
    if discovery_state != "connection_failed":
        return PROVIDER_PROBE_NONE
    if failure_category == "authentication":
        return PROVIDER_PROBE_AUTH
    return PROVIDER_PROBE_CONNECTION


def probe_failure_summary_detail(probe_failure: str) -> str:
    """The summary row's honest wording for a failed probe."""

    if probe_failure == PROVIDER_PROBE_AUTH:
        return "saved, but the key failed an authentication check"
    if probe_failure == PROVIDER_PROBE_CONNECTION:
        return "saved, but the server couldn't be reached when models were checked"
    return ""


def apply_probe_failure_to_summary_rows(
    rows: tuple["SummaryRow", ...], probe_failure: str
) -> tuple["SummaryRow", ...]:
    """Overlay a known probe failure onto the config-derived summary rows.

    ``build_summary_rows`` reads the config file, and a saved-but-broken
    key is indistinguishable from a working one there — exactly the UAT
    S-1 incident where the summary said "✓ Provider" minutes after the
    probe got a 401. The overlay downgrades the Provider row to
    ROW_ATTENTION with the failure spelled out; rows the config already
    marks unconfigured are left alone (their message is more specific).
    """
    if not probe_failure:
        return rows
    detail = probe_failure_summary_detail(probe_failure)
    return tuple(
        replace(row, state=ROW_ATTENTION, detail=detail)
        if row.label == "Provider" and row.state == ROW_CONFIGURED
        else row
        for row in rows
    )


def build_first_run_summary_actions(
    *,
    provider_configured: bool,
    model_configured: bool,
    provider_probe_failed: bool = False,
) -> tuple[FirstRunSummaryAction, FirstRunSummaryAction, FirstRunSummaryAction]:
    """Return the exact primary, secondary, and tertiary summary hierarchy.

    TASK-21143: ``provider_probe_failed`` covers the saved-but-broken case
    (UAT S-1) — "configured" means written to disk, never "working", so a
    key that failed its authentication probe still counted as configured
    and the ``review_provider`` primary this function already had was
    unreachable exactly when it mattered most.
    """

    if type(provider_configured) is not bool or type(model_configured) is not bool:
        raise ValueError("Summary readiness must use booleans.")
    if type(provider_probe_failed) is not bool:
        raise ValueError("Summary probe state must use booleans.")
    primary: FirstRunSummaryAction = (
        "start_chatting"
        if provider_configured and model_configured and not provider_probe_failed
        else "review_provider"
    )
    return primary, "explore_home", "review_settings"


def coerce_wizard_flag(raw: Any) -> bool:
    """Tolerantly parse a persisted wizard flag.

    Args:
        raw: Whatever the TOML loader produced for the key.

    Returns:
        True only for bool True, int 1, or the string "true" (case-insensitive).
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw == 1
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _is_real_secret(value: Any) -> bool:
    """A non-empty string that is not a generic template placeholder."""
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    return not (
        stripped.startswith(_PLACEHOLDER_MARKERS[0])
        and stripped.endswith(_PLACEHOLDER_MARKERS[1])
    )


def _is_real_provider_api_key(value: Any) -> bool:
    """Return the shared canonical provider-credential validity decision."""
    from tldw_chatbook.config import is_valid_provider_api_key

    return is_valid_provider_api_key(value)


def any_provider_configured(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Return True when any provider has a real, user-supplied credential.

    Walks the NESTED ``app_config["api_settings"]`` dict. Do not replace this
    with config.get_detected_api_providers(): that helper matches
    "api_settings.<p>" as a top-level key and always returns [].

    Deliberately does NOT check endpoint-URL keys (api_url, api_base_url,
    etc.). UAT incident: the shipped config.toml template (config.py's
    CONFIG_TOML_CONTENT) pre-populates roughly a dozen [api_settings.*]
    blocks with default local-server endpoint URLs -- llama.cpp
    http://localhost:8080, Ollama, vLLM, the HuggingFace router, and more --
    on EVERY fresh install, none of them entered by the user. Counting those
    template defaults as "configured" made a truly fresh install
    indistinguishable from an already-configured one, so the wizard never
    auto-offered in the real app (confirmed live via tmux UAT; every
    existing unit test used a synthetic config and missed it because none
    reproduced the template's baked-in endpoint defaults).

    Consequence accepted deliberately: a user who has hand-configured a
    local endpoint only (no API key at all) will still get exactly ONE
    auto-offer of the wizard. Skipping it persists ``setup_completed``, so
    it never re-offers -- the same one-time-then-never-again behavior the
    upgrader path already relies on.
    """
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if not isinstance(settings, Mapping):
            continue
        if _is_real_provider_api_key(settings.get("api_key")):
            return True
        env_var = settings.get("api_key_env_var")
        if (
            isinstance(env_var, str)
            and env_var.strip()
            and _is_real_provider_api_key(environ.get(env_var.strip()))
        ):
            return True
    return False


def _wizard_flag(app_config: Mapping[str, object], key: str) -> bool:
    section = app_config.get(WIZARD_STATE_SECTION)
    if not isinstance(section, Mapping):
        return False
    return coerce_wizard_flag(section.get(key))


def _api_settings_entry_for_provider(
    app_config: Mapping[str, object], provider_value: str
) -> Mapping[str, object]:
    """The ``api_settings.<key>`` block matching ``provider_value``, or ``{}``.

    ``api_settings`` is always keyed by the raw provider_key form (e.g.
    "openai", "llama_cpp" -- see the ``[api_settings.*]`` headings in
    config.py's ``CONFIG_TOML_CONTENT``), which is also the exact form
    ``ProviderStep._display_value_for`` persists into
    ``chat_defaults.provider``. Comparing case-insensitively is a small
    extra safety margin (costs nothing, and the shipped template's OWN
    default ``chat_defaults.provider = "OpenAI"`` happens to use the
    capitalized display form rather than the wizard's raw-key form --
    matching case-insensitively doesn't change the outcome there, since
    ``[api_settings.openai]`` carries no endpoint field at all, but it keeps
    this helper from silently depending on that particular template detail
    staying inconsistent).
    """
    if not provider_value:
        return {}
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return {}
    target = provider_value.strip().lower()
    for key, settings in api_settings.items():
        if str(key).strip().lower() == target and isinstance(settings, Mapping):
            return settings
    return {}


def provider_summary_configured(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Whether the Summary step's Provider row should read configured (✓).

    Deliberately more permissive than ``any_provider_configured`` (the
    auto-offer gate, used as-is by ``should_offer_wizard``): a real inline
    ``api_key`` or resolved env var counts here exactly as it does there,
    but this row ALSO counts a bare endpoint (``api_url``/``api_base_url``,
    no key at all) for the provider named by ``chat_defaults.provider`` --
    something ``any_provider_configured`` intentionally ignores (see its
    own docstring).

    Why the extra case is needed. The wizard's own one-click "Use this
    server" path (``ProviderStep._on_use_detected`` ->
    ``build_provider_commit``) commits an ``api_url`` with NO ``api_key`` at
    all -- exactly the shape ``any_provider_configured`` was written to
    ignore. Reusing that helper verbatim for this row made the wizard's own
    one-click commit render "no credentials or endpoint" immediately after
    the user finished that exact flow on this exact screen.

    Why not just "any api_settings endpoint is present": the shipped
    config.toml template pre-populates roughly a dozen ``[api_settings.*]``
    blocks with default local-server endpoint URLs (llama.cpp, Ollama,
    vLLM, ...) on every fresh install, none of them touched by the user.
    Blanket-scanning every provider's endpoint would resurrect exactly the
    bug ``any_provider_configured``'s docstring describes, and would ALSO
    leak cross-provider: a user who selects Anthropic and enters no key
    would still see ✓ purely because some OTHER, untouched provider's
    default endpoint (e.g. llama_cpp) happens to sit in the same config.
    Scoping the endpoint check to ONLY the provider named by
    ``chat_defaults.provider`` (which the wizard always writes, in the raw
    provider_key form, whenever ``ProviderStep.commit()`` runs -- see
    ``invalidate_model_for_provider_change``) avoids that leak: the
    template's own default for that key is "OpenAI", a cloud provider whose
    ``[api_settings.openai]`` block carries no endpoint field at all, so an
    untouched template can never accidentally satisfy this check through
    its own baked-in provider value.

    Why also require ``first_run.setup_started``/``setup_completed``: belt
    and suspenders, and the literal "did the wizard actually do this"
    signal -- the wizard sets ``setup_started`` in the live app_config the
    moment it mounts (``FirstRunSetupWizard.on_mount`` ->
    ``_persist_started_flag``), before any step commits anything, and the
    shipped template never ships a ``[first_run]`` section at all. In
    practice this flag is already true for every real Summary render (the
    step cannot be reached before the wizard has mounted), so the
    chat_defaults-keyed scoping above is what actually prevents the
    cross-provider leak; this flag's job is to keep a synthetic/pristine
    ``app_config`` (e.g. a hand-built dict in a unit test, or any future
    caller that evaluates this outside of a live wizard run) from ever
    reading ✓ off endpoint state alone.

    Known limitation, accepted deliberately (same shape
    ``any_provider_configured`` already accepts for its own narrower case):
    a user who reruns the wizard on a config that already has
    ``setup_completed=True`` from a prior full run, then backs out of
    Provider without touching it this time, will still see this row read
    off whatever endpoint survives from that earlier run for whatever
    provider is currently named in ``chat_defaults.provider``.
    """
    if any_provider_configured(app_config, environ):
        return True
    if not (
        _wizard_flag(app_config, SETUP_STARTED_KEY)
        or _wizard_flag(app_config, SETUP_COMPLETED_KEY)
    ):
        return False
    provider_value = str(_section(app_config, "chat_defaults").get("provider") or "")
    settings = _api_settings_entry_for_provider(app_config, provider_value)
    return _is_real_secret(settings.get("api_url")) or _is_real_secret(
        settings.get("api_base_url")
    )


def should_offer_wizard(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Auto-offer once: no wizard state keys AND nothing configured."""
    if _wizard_flag(app_config, SETUP_STARTED_KEY):
        return False
    if _wizard_flag(app_config, SETUP_COMPLETED_KEY):
        return False
    return not any_provider_configured(app_config, environ)


def should_show_resume_toast(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Started but never finished: point at Settings, never re-push."""
    return _wizard_flag(app_config, SETUP_STARTED_KEY) and not _wizard_flag(
        app_config, SETUP_COMPLETED_KEY
    )


TRACK_QUICK = "quick"
TRACK_FULL = "full"

STEP_WELCOME = "welcome"
STEP_PROVIDER = "provider"
STEP_MODEL = "model"
STEP_VOICE = "voice"
STEP_RAG = "rag"
STEP_SPEECH = "speech"
STEP_TOOLS = "tools"
STEP_NOTES = "notes"
STEP_APPEARANCE = "appearance"
STEP_PROTECT = "protect-keys"
STEP_SUMMARY = "summary"

STEP_TITLES: Mapping[str, str] = {
    STEP_WELCOME: "Welcome",
    STEP_PROVIDER: "Provider",
    STEP_MODEL: "Model",
    STEP_VOICE: "Voice",
    STEP_RAG: "RAG",
    STEP_SPEECH: "Speech",
    STEP_TOOLS: "Tools",
    STEP_NOTES: "Notes",
    STEP_APPEARANCE: "Style",
    STEP_PROTECT: "Protect",
    STEP_SUMMARY: "Summary",
}


@dataclass(frozen=True, slots=True)
class SetupProgressItem:
    """One row in the progress tracker for the resolved setup path."""

    step_id: str
    title: str
    state: Literal["active", "complete", "upcoming", "attention"]


# TASK-1301: Speech transcription joins the FULL track only, right after RAG
# (both are optional model-setup steps) -- QUICK_TRACK stays byte-identical
# on purpose, see AC#1.
_FULL_TRACK = (
    STEP_WELCOME,
    STEP_PROVIDER,
    STEP_MODEL,
    STEP_VOICE,
    STEP_RAG,
    STEP_SPEECH,
    STEP_TOOLS,
    STEP_NOTES,
    STEP_APPEARANCE,
    STEP_PROTECT,
    STEP_SUMMARY,
)
_QUICK_TRACK = (
    STEP_WELCOME,
    STEP_PROVIDER,
    STEP_MODEL,
    STEP_VOICE,
    STEP_PROTECT,
    STEP_SUMMARY,
)

_SETUP_DRAFT_FIELD_TYPES: Mapping[str, Mapping[str, type]] = {
    STEP_WELCOME: {"track": str},
    STEP_PROVIDER: {"provider_key": str, "provider_value": str},
    STEP_MODEL: {"model_id": str},
    STEP_VOICE: {
        "endpoint": str,
        "authentication_mode": str,
        "model_id": str,
        "voice_id": str,
        "response_format": str,
        "speed": float,
        "sample_text": str,
        "use_as_default": bool,
    },
    STEP_RAG: {"embedding_model": str},
    STEP_SPEECH: {},
    STEP_TOOLS: {},
    STEP_NOTES: {},
    STEP_APPEARANCE: {"theme": str, "splash_card": str},
    STEP_PROTECT: {"encryption_enabled": bool},
    STEP_SUMMARY: {},
}


@dataclass(frozen=True, slots=True)
class SetupDraft:
    """Bounded, non-secret checkpoint for resuming first-run setup."""

    version: int
    track: str
    active_step_id: str
    values: Mapping[str, Mapping[str, object]]
    resume_attempted: bool = False


def _normalized_field_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    with_word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.casefold()).strip("_")


def _contains_secret_field_name(name: object) -> bool:
    normalized = _normalized_field_name(name)
    return any(token in normalized for token in _SECRET_FIELD_TOKENS)


def _canonical_setup_draft_size(payload: Mapping[str, object]) -> int | None:
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return len(serialized.encode("utf-8"))


def _mapping_contains_secret_key(value: object, seen: set[int] | None = None) -> bool:
    """Inspect nested mapping keys before applying the scalar allowlist."""

    if not isinstance(value, Mapping):
        return False
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    for key, nested_value in value.items():
        if _contains_secret_field_name(key):
            return True
        if _mapping_contains_secret_key(nested_value, seen):
            return True
    return False


def _validated_setup_draft(
    *,
    version: object,
    track: object,
    active_step_id: object,
    values: object,
    resume_attempted: object,
) -> SetupDraft | None:
    if type(version) is not int or version != SETUP_DRAFT_VERSION:
        return None
    if track not in (TRACK_QUICK, TRACK_FULL):
        return None
    if not isinstance(active_step_id, str):
        return None
    allowed_steps = active_step_ids(str(track), key_entered=True)
    if active_step_id not in allowed_steps:
        return None
    if not isinstance(resume_attempted, bool) or not isinstance(values, Mapping):
        return None

    field_count = 0
    clean_values: dict[str, dict[str, object]] = {}
    for step_id, step_values in values.items():
        if not isinstance(step_id, str) or step_id not in _SETUP_DRAFT_FIELD_TYPES:
            return None
        if step_id not in allowed_steps or not isinstance(step_values, Mapping):
            return None
        allowed_fields = _SETUP_DRAFT_FIELD_TYPES[step_id]
        clean_step: dict[str, object] = {}
        for field_name, value in step_values.items():
            field_count += 1
            if field_count > _MAX_SETUP_DRAFT_FIELDS:
                return None
            if _contains_secret_field_name(field_name):
                return None
            if not isinstance(field_name, str) or field_name not in allowed_fields:
                return None
            if _mapping_contains_secret_key(value):
                return None
            expected_type = allowed_fields[field_name]
            if expected_type is bool:
                if not isinstance(value, bool):
                    return None
            elif not isinstance(value, expected_type) or isinstance(value, bool):
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            clean_step[field_name] = value
        clean_values[step_id] = clean_step

    payload = {
        DRAFT_VERSION_KEY: version,
        DRAFT_TRACK_KEY: track,
        DRAFT_ACTIVE_STEP_KEY: active_step_id,
        DRAFT_VALUES_KEY: clean_values,
        DRAFT_RESUME_ATTEMPTED_KEY: resume_attempted,
    }
    size = _canonical_setup_draft_size(payload)
    if size is None or size > _MAX_SETUP_DRAFT_BYTES:
        return None
    welcome_values = clean_values.get(STEP_WELCOME)
    if welcome_values is not None and welcome_values.get("track") != track:
        return None
    return SetupDraft(
        version=version,
        track=str(track),
        active_step_id=active_step_id,
        values=clean_values,
        resume_attempted=resume_attempted,
    )


def read_setup_draft(app_config: Mapping[str, object]) -> SetupDraft | None:
    """Parse a setup checkpoint defensively; malformed drafts fail closed."""

    try:
        first_run = app_config.get(WIZARD_STATE_SECTION)
        if not isinstance(first_run, Mapping):
            return None
        return _validated_setup_draft(
            version=first_run.get(DRAFT_VERSION_KEY),
            track=first_run.get(DRAFT_TRACK_KEY),
            active_step_id=first_run.get(DRAFT_ACTIVE_STEP_KEY),
            values=first_run.get(DRAFT_VALUES_KEY),
            resume_attempted=first_run.get(DRAFT_RESUME_ATTEMPTED_KEY, False),
        )
    except Exception:
        return None


def setup_draft_checkpoint(
    *,
    track: str,
    active_step_id: str,
    values: Mapping[str, Mapping[str, object]],
    resume_attempted: bool = False,
) -> SetupDraft:
    """Select allowlisted checkpoint fields from completed wizard-step data."""

    clean_values: dict[str, dict[str, object]] = {}
    allowed_steps = set(active_step_ids(track, key_entered=True))
    for step_id, allowed_fields in _SETUP_DRAFT_FIELD_TYPES.items():
        if step_id not in allowed_steps:
            continue
        raw_step = values.get(step_id)
        if not isinstance(raw_step, Mapping):
            continue
        clean_step: dict[str, object] = {}
        for field_name, expected_type in allowed_fields.items():
            value = raw_step.get(field_name)
            if expected_type is bool:
                valid = isinstance(value, bool)
            else:
                valid = isinstance(value, expected_type) and not isinstance(value, bool)
            if valid:
                clean_step[field_name] = value
        if clean_step:
            clean_values[step_id] = clean_step

    draft = _validated_setup_draft(
        version=SETUP_DRAFT_VERSION,
        track=track,
        active_step_id=active_step_id,
        values=clean_values,
        resume_attempted=resume_attempted,
    )
    if draft is None:
        raise ValueError("setup draft checkpoint is invalid")
    return draft


def build_setup_draft_mutation(
    draft: SetupDraft | None,
) -> tuple[dict[str, dict[str, object]], dict[str, tuple[str, ...]]]:
    """Build the isolated first-run set/delete mutation for a checkpoint."""

    if draft is None:
        return {}, {WIZARD_STATE_SECTION: SETUP_DRAFT_KEYS}
    if type(draft) is not SetupDraft:
        raise TypeError("setup draft mutation requires SetupDraft")
    validated = _validated_setup_draft(
        version=draft.version,
        track=draft.track,
        active_step_id=draft.active_step_id,
        values=draft.values,
        resume_attempted=draft.resume_attempted,
    )
    if validated is None:
        raise ValueError("setup draft is invalid")
    return {
        WIZARD_STATE_SECTION: {
            DRAFT_VERSION_KEY: validated.version,
            DRAFT_TRACK_KEY: validated.track,
            DRAFT_ACTIVE_STEP_KEY: validated.active_step_id,
            DRAFT_VALUES_KEY: {
                key: dict(value) for key, value in validated.values.items()
            },
            DRAFT_RESUME_ATTEMPTED_KEY: validated.resume_attempted,
        }
    }, {}


def setup_recovery_action(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> Literal["offer", "prompt", "home", "none"]:
    """Choose the single startup action for setup or recovery."""

    if should_offer_wizard(app_config, environ):
        return "offer"
    if _wizard_flag(app_config, SETUP_COMPLETED_KEY):
        return "none"
    if not _wizard_flag(app_config, SETUP_STARTED_KEY):
        return "none"
    draft = read_setup_draft(app_config)
    if draft is None:
        return "none"
    return "home" if draft.resume_attempted else "prompt"


WIZARD_OWNED_SECTIONS = frozenset(
    {
        "chat_defaults",
        "embedding_config",
        "tools",
        "notes",
        "general",
        "splash_screen",
        "transcription",
        "provider_setup.confirmed",
        WIZARD_STATE_SECTION,
    }
)
_API_SETTINGS_PREFIX = "api_settings."


def active_step_ids(track: str, *, key_entered: bool) -> tuple[str, ...]:
    """Resolve the ordered active step ids for a track.

    Args:
        track: TRACK_QUICK or TRACK_FULL (anything else falls back to full).
        key_entered: Whether any secret was entered this run; gates STEP_PROTECT.
    """
    base = _QUICK_TRACK if track == TRACK_QUICK else _FULL_TRACK
    if key_entered:
        return base
    return tuple(step for step in base if step != STEP_PROTECT)


def build_setup_progress(
    active_ids: tuple[str, ...],
    current_index: int,
    attention_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[SetupProgressItem, ...]:
    """Project a resolved setup path into display-ready progress rows.

    TASK-21143 (UAT N-7): a visited step whose probe demonstrably failed
    must not wear the ✓ users read as "this part is OK" — ``attention_ids``
    downgrades those completed steps to the "attention" state (rendered as
    an amber "!"). Only completed steps downgrade: the active step keeps
    its position marker, upcoming steps have nothing to report on yet.
    """

    unknown_ids = tuple(step_id for step_id in active_ids if step_id not in STEP_TITLES)
    if unknown_ids:
        raise ValueError(f"unknown setup step: {unknown_ids[0]}")
    if not active_ids:
        return ()
    active_index = min(max(current_index, 0), len(active_ids) - 1)

    def state_for(index: int, step_id: str) -> str:
        if index < active_index:
            return "attention" if step_id in attention_ids else "complete"
        return "active" if index == active_index else "upcoming"

    return tuple(
        SetupProgressItem(
            step_id=step_id,
            title=STEP_TITLES[step_id],
            state=state_for(index, step_id),
        )
        for index, step_id in enumerate(active_ids)
    )


def stored_plaintext_key_present(app_config: Mapping[str, object]) -> bool:
    """Whether a real, unencrypted provider secret already sits on disk.

    Bug-4 fix: ``active_step_ids``'s ``key_entered`` gate previously only
    reflected secrets typed THIS run, so a rerun over a config that already
    has a plaintext key (e.g. hand-edited config.toml, or a completed prior
    run) could never reach Protect Keys without retyping a credential. This
    is the config-derived signal the spec actually wants: any configured
    key, regardless of when it was entered, as long as it is not already
    protected by encryption.

    Args:
        app_config: Loaded app configuration.

    Returns:
        True when at least one ``api_settings.<provider>.api_key`` holds a
        real (non-placeholder) secret AND config encryption is not enabled.
    """
    if coerce_wizard_flag(_section(app_config, "encryption").get("enabled")):
        return False
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if isinstance(settings, Mapping) and _is_real_provider_api_key(
            settings.get("api_key")
        ):
            return True
    return False


def build_provider_commit(
    *, provider_key: str, api_key: str | None, api_url: str | None
) -> dict[str, dict[str, Any]]:
    """Mutation for the provider step. Empty when the key lives in the env."""
    values: dict[str, Any] = {}
    if api_key:
        values["api_key"] = api_key
    if api_url:
        values["api_url"] = api_url
    if not values:
        return {}
    return {f"{_API_SETTINGS_PREFIX}{provider_key}": values}


def build_model_commit(
    *, provider_value: str, model_id: str
) -> dict[str, dict[str, Any]]:
    """Mutation for the model step.

    Args:
        provider_value: The ``chat_defaults.provider`` value the model is
            paired with (whatever form ProviderStep last committed).
        model_id: The chosen default model id.

    Returns:
        The section/value mapping to persist under ``chat_defaults``.
    """
    return {"chat_defaults": {"provider": provider_value, "model": model_id}}


def curated_models_for_provider(
    catalog: Mapping[str, Any], provider_value: str
) -> list[str]:
    """Look up curated fallback models for a provider, in ANY key form.

    ProviderStep persists ``chat_defaults.provider`` as the RAW provider_key
    (e.g. "llama_cpp", "openai") -- matching chat_screen's detected-server
    path -- while the curated ``[providers]`` table in config.toml (surfaced
    via ``config.get_cli_providers_and_models()``) is keyed by human display
    names (e.g. "OpenAI"). A plain ``catalog.get(provider_value)`` would
    silently return [] whenever the two forms disagree even though a
    matching entry exists, so normalize both sides via
    ``provider_readiness.provider_config_key`` before comparing.
    """
    if not provider_value:
        return []
    direct = catalog.get(provider_value)
    if direct:
        return list(direct)
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    target_key = provider_config_key(provider_value)
    if not target_key:
        return []
    for name, models in catalog.items():
        if provider_config_key(str(name)) == target_key:
            return list(models)
    return []


def build_rag_commit(*, default_model_id: str) -> dict[str, dict[str, Any]]:
    """Mutation for the RAG step.

    Args:
        default_model_id: The chosen default embedding model id.

    Returns:
        The section/value mapping to persist under ``embedding_config``.
    """
    return {"embedding_config": {"default_model_id": default_model_id}}


def build_tools_commit(*, gate_values: Mapping[str, bool]) -> dict[str, dict[str, Any]]:
    """Mutation for the tools step.

    Args:
        gate_values: Mapping of tool gate key to the desired enabled state.
            Callers typically pass only the delta (see
            ``tools_commit_delta``), not every gate.

    Returns:
        The section/value mapping to persist under ``tools``.
    """
    return {"tools": {key: bool(value) for key, value in gate_values.items()}}


def tools_commit_delta(
    *, gate_values: Mapping[str, bool], current_gates: Mapping[str, bool]
) -> dict[str, bool]:
    """Keys whose desired value differs from the currently persisted gate.

    ``current_gates`` should hold the EFFECTIVE value for every key present
    in ``gate_values`` (False when the ``[tools]`` section has no entry for
    it -- the same default ``BuiltinToolProvider``'s own gate check uses).
    A fresh config where every switch starts and stays False therefore
    produces an empty delta (no write), while flipping a previously-True
    gate back to False is still reported -- unlike a naive "only ever
    persist True" filter, which silently drops OFF-transitions on re-run.
    """
    return {
        key: bool(value)
        for key, value in gate_values.items()
        if bool(value) != bool(current_gates.get(key, False))
    }



def build_appearance_commit(
    *,
    default_theme: str | None,
    splash_card: str | None = None,
    reset_splash_to_random: bool = False,
) -> dict[str, dict[str, Any]]:
    """Mutation for the appearance step.

    Bug-2 fix: both ``default_theme`` and the splash-card write are now
    each individually optional, so the caller (``AppearanceStep.commit()``)
    can build a delta-aware commit -- e.g. a rerun that only changes the
    splash card omits ``general.default_theme`` entirely instead of
    rewriting it back to a (possibly stale) fallback value.

    Args:
        default_theme: Theme name to persist under
            ``general.default_theme``, or falsy (``None``/``""``) to omit
            that key entirely -- e.g. the chosen theme matches what is
            already persisted, so nothing needs to change.
        splash_card: A specific card name to persist under
            ``splash_screen.card_selection``, or falsy to leave that key
            out (e.g. nothing was chosen, or "Surprise me" was picked --
            see ``reset_splash_to_random``).
        reset_splash_to_random: When True and ``splash_card`` is falsy,
            write ``splash_screen.card_selection = "random"``. This is the
            explicit "Surprise me" choice over a previously persisted
            specific card, which otherwise has no truthy value of its own
            to signal that a write is even needed.

    Returns:
        The section/value mapping to persist; empty where nothing changed.
    """
    commit: dict[str, dict[str, Any]] = {}
    if default_theme:
        commit["general"] = {"default_theme": default_theme}
    if splash_card:
        commit["splash_screen"] = {"card_selection": splash_card}
    elif reset_splash_to_random:
        commit["splash_screen"] = {"card_selection": "random"}
    return commit


def build_wizard_state_commit(
    *, started: bool | None = None, completed: bool | None = None
) -> dict[str, dict[str, Any]]:
    """Mutation for the wizard's own progress flags.

    Args:
        started: Set ``first_run.setup_started`` to this value, or omit
            (leave as None) to leave that key out of the commit.
        completed: Set ``first_run.setup_completed`` to this value, or omit
            (leave as None) to leave that key out of the commit.

    Returns:
        The section/value mapping to persist under ``first_run``; empty
        when neither flag was passed.
    """
    values: dict[str, Any] = {}
    if started is not None:
        values[SETUP_STARTED_KEY] = started
    if completed is not None:
        values[SETUP_COMPLETED_KEY] = completed
    return {WIZARD_STATE_SECTION: values} if values else {}


def invalidate_model_for_provider_change(
    commit: dict[str, dict[str, Any]],
    *,
    previous_provider_value: str | None,
    new_provider_value: str,
) -> dict[str, dict[str, Any]]:
    """Supersede a stale model when the committed provider changes.

    Without this, Back-and-switch leaves chat_defaults pairing the new
    provider with the old provider's model.

    Bug-3 fix: ``previous_provider_value`` is compared with ``!=`` against
    an empty-string default rather than gated behind a truthiness check.
    The old ``if previous_provider_value and ...`` guard treated a falsy
    "no previous provider" (``None`` or ``""``) as "nothing to compare",
    which silently skipped the chat_defaults write on a genuinely first-ever
    provider selection (an empty/absent persisted ``chat_defaults.provider``
    is exactly as "different" from a real new provider as any other prior
    provider would be). Callers resolve ``previous_provider_value`` as the
    in-session previous commit if one exists this run, else the PERSISTED
    ``chat_defaults.provider`` (see ``read_wizard_prefill``), so this
    function no longer needs to special-case "no information at all".

    Args:
        commit: The commit built so far (e.g. from ``build_provider_commit``).
        previous_provider_value: The provider ``chat_defaults.provider`` was
            associated with before this commit -- the in-session value if
            this step has already committed once this run, else the
            persisted value (``""`` on a fresh/absent config). ``None`` is
            treated the same as ``""``.
        new_provider_value: The provider value this commit is selecting.

    Returns:
        ``commit`` unchanged when the provider is the same as before;
        otherwise a copy with ``chat_defaults`` set to the new provider and
        an emptied model.
    """
    effective_previous = previous_provider_value or ""
    if effective_previous != new_provider_value:
        merged = dict(commit)
        merged["chat_defaults"] = {"provider": new_provider_value, "model": ""}
        return merged
    return commit


def commit_sections_allowed(section_values: Mapping[str, Mapping[Any, Any]]) -> bool:
    """The invariant oracle: wizard commits touch only wizard-owned sections."""
    for section in section_values:
        if section in WIZARD_OWNED_SECTIONS:
            continue
        if section.startswith(_API_SETTINGS_PREFIX) and len(section) > len(
            _API_SETTINGS_PREFIX
        ):
            continue
        return False
    return True


@dataclass(frozen=True)
class SecretPresence:
    """Whether a provider secret exists — never the secret itself."""

    configured: bool
    inline_configured: bool = False
    env_var: str | None = None
    env_var_set: bool = False
    env_var_declared: bool = False


@dataclass(frozen=True)
class WizardPrefill:
    """Current config values for re-run prefill (no secrets)."""

    provider_value: str = ""
    model_id: str = ""
    default_theme: str = ""
    tool_gates: tuple[tuple[str, bool], ...] = ()
    card_selection: str = ""
    """Persisted ``splash_screen.card_selection`` -- "" when never set.

    Bug-2 fix: AppearanceStep needs this to tell "the user just explicitly
    re-picked Surprise-me over a persisted specific card" (a real reset)
    apart from "nothing was ever configured" (a no-op).
    """


ROW_CONFIGURED = "configured"
ROW_DEFAULT = "default"
ROW_ATTENTION = "attention"

_TEMPLATE_DEFAULT_THEME = "textual-dark"

#: Shipped template defaults that get merged into app_config at load time.
#: Like `_TEMPLATE_DEFAULT_THEME`, their mere presence is not a user choice
#: (task-2724: a Ctrl+N-only walk rendered "✓ Default model — gpt-5.6-terra"
#: and "✓ RAG — embedding model: e5-small-v2" beneath "✗ Provider").
_TEMPLATE_DEFAULT_RAG_MODEL = "e5-small-v2"
_TEMPLATE_DEFAULT_MODEL = "gpt-5.6-terra"


@dataclass(frozen=True)
class SummaryRow:
    """One line of the final summary matrix.

    TASK-1504: three states instead of a boolean --
    ``configured`` (✓, the user set this up), ``default`` (–, untouched and
    fine), ``attention`` (✗, worth acting on). ``ok`` survives as a derived
    convenience so older callers/tests keep working.

    Args:
        label: Row heading shown in the matrix.
        state: One of ROW_CONFIGURED / ROW_DEFAULT / ROW_ATTENTION.
        detail: Optional human explanation rendered after the label.
    """

    label: str
    state: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True only for user-configured rows (legacy boolean view)."""
        return self.state == ROW_CONFIGURED

    @property
    def glyph(self) -> str:
        """Matrix glyph: ✓ configured, – default, ✗ attention."""
        return {ROW_CONFIGURED: "✓", ROW_DEFAULT: "–"}.get(self.state, "✗")


def _section(app_config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = app_config.get(name)
    return section if isinstance(section, Mapping) else {}


def read_provider_secret_presence(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    provider_key: str,
) -> SecretPresence:
    """Resolve whether a provider secret is configured, without ever reading it.

    Falls back to the conventional ``<PROVIDER>_API_KEY`` environment variable
    name (via ``provider_readiness.default_api_key_env_var``, the same
    resolution Chat's own readiness check uses) when ``api_key_env_var`` is
    not explicitly present in ``app_config``. Without this fallback, a wizard
    run before the packaged default config.toml's ``api_key_env_var`` entries
    have been persisted to disk (or any app_config that omits them) could
    never detect an already-exported key -- exactly the first-run scenario
    this step exists for.
    """
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    settings = _first_run_provider_settings(app_config, provider_key)
    credential_source = settings.get("credential_source")
    explicit_source = (
        credential_source.strip().lower()
        if type(credential_source) is str
        and credential_source.strip().lower()
        in {"none", "stored", "environment"}
        else None
    )
    env_var_raw = settings.get("api_key_env_var")
    env_var_declared = isinstance(env_var_raw, str) and bool(env_var_raw.strip())
    env_var = (
        env_var_raw.strip()
        if env_var_declared
        else default_api_key_env_var(provider_key)
    )
    env_var_set = bool(env_var and _is_real_provider_api_key(environ.get(env_var)))
    inline = _is_real_provider_api_key(settings.get("api_key"))
    if explicit_source == "none":
        inline = False
        env_var_set = False
    elif explicit_source == "stored":
        env_var_set = False
    elif explicit_source == "environment":
        inline = False
    return SecretPresence(
        configured=inline or env_var_set,
        inline_configured=inline,
        env_var=env_var,
        env_var_set=env_var_set,
        env_var_declared=env_var_declared,
    )


def read_wizard_prefill(app_config: Mapping[str, object]) -> WizardPrefill:
    chat_defaults = _section(app_config, "chat_defaults")
    general = _section(app_config, "general")
    tools = _section(app_config, "tools")
    splash_screen = _section(app_config, "splash_screen")
    return WizardPrefill(
        provider_value=str(chat_defaults.get("provider") or ""),
        model_id=str(chat_defaults.get("model") or ""),
        default_theme=str(general.get("default_theme") or ""),
        tool_gates=tuple(
            (str(key), coerce_wizard_flag(value)) for key, value in tools.items()
        ),
        card_selection=str(splash_screen.get("card_selection") or ""),
    )


def build_summary_rows(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    rag_deps_installed: bool,
    speech_installed: bool = False,
    speech_runtime_installed: bool = False,
) -> tuple[SummaryRow, ...]:
    """Build the summary matrix strictly from persisted config.

    TASK-1504 semantics: a row only earns ✓ when the USER configured the
    thing; untouched-but-fine defaults render as – so the matrix never
    claims credit for template values; ✗ marks what deserves action
    (no provider, provider-without-model, plaintext keys).

    Args:
        app_config: The persisted (re-loaded) config mapping.
        environ: Process environment, for env-var key detection.
        rag_deps_installed: Whether the embeddings extras are importable.
        speech_installed: TASK-1301 AC#6 -- whether the managed Parakeet v2
            artifact is currently installed and active. Resolved off-loop by
            the caller (SummaryStep), exactly like ``rag_deps_installed`` --
            this function stays pure/I/O-free. Defaults to False so every
            existing caller that doesn't yet resolve this stays correct
            (an unconfigured/uninstalled row either way).
        speech_runtime_installed: Review Important 4 residual -- whether the
            ``onnx-asr`` runtime extra is importable (same probe the step
            itself gates on: ``Utils.optional_deps.parakeet_onnx_deps_installed``).
            Without this, a config persisted while the extra WAS installed
            (a completed setup) could still read ✓ configured/ready in the
            same run after the extra is removed, even though the step
            itself now says "runtime not installed" -- the two must agree.
            Defaults to False (conservative: unknown runtime state reads as
            not-ready, matching ``speech_installed``'s own default bias).

    Returns:
        Ordered tuple of SummaryRow for the Summary step to render.
    """
    prefill = read_wizard_prefill(app_config)
    # F2 fix: this row uses provider_summary_configured, NOT
    # any_provider_configured (the auto-offer gate) -- see that function's
    # docstring for why the two deliberately disagree on a bare, wizard-
    # committed endpoint (the one-click "Use this server" path).
    provider_ok = provider_summary_configured(app_config, environ)
    tools_on = [key for key, value in prefill.tool_gates if value]
    encryption_on = coerce_wizard_flag(
        _section(app_config, "encryption").get("enabled")
    )
    rag_model = str(
        _section(app_config, "embedding_config").get("default_model_id") or ""
    )

    if not rag_deps_installed:
        rag_row = SummaryRow(
            "RAG", ROW_DEFAULT, "optional — embeddings deps not installed"
        )
    elif rag_model and rag_model != _TEMPLATE_DEFAULT_RAG_MODEL:
        rag_row = SummaryRow("RAG", ROW_CONFIGURED, f"embedding model: {rag_model}")
    elif rag_model:
        # The untouched template value — matches the header's "RAG off"
        # sentence instead of contradicting it (task-2724).
        rag_row = SummaryRow(
            "RAG", ROW_DEFAULT, f"off by default — embedding model {rag_model}"
        )
    else:
        rag_row = SummaryRow("RAG", ROW_DEFAULT, "no embedding model selected")

    if prefill.model_id and provider_ok:
        model_row = SummaryRow("Default model", ROW_CONFIGURED, prefill.model_id)
    elif provider_ok:
        # A provider without a model is half-finished — worth flagging.
        model_row = SummaryRow("Default model", ROW_ATTENTION, "not selected")
    elif prefill.model_id and prefill.model_id != _TEMPLATE_DEFAULT_MODEL:
        # Typed/selected by the user, but with no provider it cannot take
        # effect — name it honestly without claiming a finished setup.
        model_row = SummaryRow(
            "Default model",
            ROW_DEFAULT,
            f"{prefill.model_id} — takes effect once a provider is connected",
        )
    else:
        # With no provider, whatever sits in chat_defaults.model is the
        # merged template default, not a choice (task-2724).
        model_row = SummaryRow("Default model", ROW_DEFAULT, "not selected")

    theme_is_custom = bool(
        prefill.default_theme and prefill.default_theme != _TEMPLATE_DEFAULT_THEME
    )

    if encryption_on:
        encryption_row = SummaryRow("Key encryption", ROW_CONFIGURED, "")
    elif stored_plaintext_key_present(app_config):
        encryption_row = SummaryRow(
            "Key encryption", ROW_ATTENTION, "API keys are stored as plain text"
        )
    else:
        encryption_row = SummaryRow("Key encryption", ROW_DEFAULT, "off")

    # TASK-1301 AC#6: read PERSISTED transcription config, never transient
    # widget state. "Configured by the wizard" is keyed off provider_id
    # matching the Parakeet ONNX provider specifically -- the shipped
    # [transcription] template always defaults to faster-whisper (or a
    # platform MLX provider) with model "distil-large-v3", which are never
    # blank, so a naive "model_id is set" check would resurrect the exact
    # template-poisoning bug any_provider_configured's own docstring warns
    # about.
    from tldw_chatbook.UI.Wizards.first_run_speech_step_state import (
        read_speech_prefill,
        routing_policy,
    )

    speech_prefill = read_speech_prefill(app_config)
    speech_configured = (
        speech_prefill.provider_id == routing_policy().parakeet_provider_id
    )
    if not speech_configured:
        speech_row = SummaryRow(
            "Speech transcription", ROW_DEFAULT, "not set up (optional)"
        )
    elif not speech_runtime_installed:
        # Review Important 4 residual: readiness must agree with the same
        # runtime probe the step itself gates on -- "files on disk" is not
        # "can actually run". Checked BEFORE speech_installed so the more
        # fundamental problem is the one reported when both are true.
        speech_row = SummaryRow(
            "Speech transcription",
            ROW_ATTENTION,
            "configured but the onnx-asr runtime isn't installed — revisit Lab ▸ Models",
        )
    elif speech_installed:
        speech_row = SummaryRow(
            "Speech transcription",
            ROW_CONFIGURED,
            f"{speech_prefill.model_id} ({speech_prefill.language})",
        )
    else:
        speech_row = SummaryRow(
            "Speech transcription",
            ROW_ATTENTION,
            # Review Important 1: there is no Settings speech/model category
            # -- the real, reachable destination for installing/deleting
            # this managed model is the Lab nav destination's Models screen
            # (screen_registry TAB_LLM "Models", rail rows Curated/Installed).
            "configured but not installed — revisit Lab ▸ Models",
        )

    return (
        SummaryRow(
            "Provider",
            ROW_CONFIGURED if provider_ok else ROW_ATTENTION,
            "" if provider_ok else "no credentials or saved endpoint",
        ),
        model_row,
        rag_row,
        speech_row,
        SummaryRow(
            "Tools",
            ROW_CONFIGURED if tools_on else ROW_DEFAULT,
            f"{len(tools_on)} enabled" if tools_on else "all off (default)",
        ),
        SummaryRow(
            "Notes folder sync",
            ROW_DEFAULT,
            "set up later in Library",
        ),
        SummaryRow(
            "Theme",
            ROW_CONFIGURED if theme_is_custom else ROW_DEFAULT,
            prefill.default_theme or "default",
        ),
        encryption_row,
    )


def rerun_model_prefill(
    app_config: Mapping[str, object], *, provider_value: str
) -> str:
    """Persisted default model to prefill when re-entering the Model step.

    TASK-1374: the prefill fires from a genuinely reachable condition — the
    session's provider matches the persisted ``chat_defaults.provider`` — so
    a re-run that keeps the same provider surfaces the saved model instead
    of blanking it. Both sides are normalized with
    ``provider_readiness.provider_config_key`` because the template stores a
    display-cased value ("OpenAI") while wizard commits store the raw key.

    Args:
        app_config: The in-memory app config mapping.
        provider_value: The session's current provider (raw key or display
            form); empty means no provider context, so no prefill.

    Returns:
        The persisted model id when providers match, else "".
    """
    if not provider_value:
        return ""
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    prefill = read_wizard_prefill(app_config)
    if not (prefill.provider_value and prefill.model_id):
        return ""
    if provider_config_key(provider_value) != provider_config_key(
        prefill.provider_value
    ):
        return ""
    return prefill.model_id
