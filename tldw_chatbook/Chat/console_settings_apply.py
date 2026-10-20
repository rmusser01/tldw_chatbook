"""Immutable contracts for applying Console chat settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_provider_support import (
    GENERATION_FIELD_REQUEST_KEYS,
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.console_settings_durability import (
    ConsoleSettingsDurabilityLease,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_target_default_console_session_settings,
    normalize_console_model_value,
)
from tldw_chatbook.Chat.custom_endpoint_registry import provider_identity_key
from tldw_chatbook.Chat.provider_readiness import provider_config_key


# ADR-095 amendment 2026-09-26 (D3, TASK-33004.1): the quick default mask. The
# quick surface's editable fields must be this same set (Max tokens becomes
# editable in TASK-33004.5).
QUICK_MODEL_DEFAULT_FIELDS = frozenset({"temperature", "max_tokens", "streaming"})
# The one quick field that may be blank: no cap is a real value, and a blank
# deletes the exact profile override so lower-precedence defaults apply.
QUICK_BLANKABLE_DEFAULT_FIELDS = frozenset({"max_tokens"})
# One field list: the fields the request-key table maps (TASK-33001.2).
FULL_MODEL_DEFAULT_FIELDS = frozenset(GENERATION_FIELD_REQUEST_KEYS)


class ConsoleSettingsAction(str, Enum):
    """A committing action exposed by a Console settings surface."""

    APPLY_TO_CHAT = "apply_to_chat"
    SAVE_MODEL_DEFAULT = "save_model_default"
    MAKE_NEW_CHAT_DEFAULT = "make_new_chat_default"


class ConsoleSettingsSurface(str, Enum):
    """The settings surface that produced one immutable submission."""

    QUICK_POPOVER = "quick_popover"
    FULL_SETTINGS = "full_settings"


@dataclass(frozen=True, slots=True)
class ConsoleSettingsOrigin:
    """Exact live-session binding observed when a settings surface opened."""

    session_id: str
    persisted_conversation_id: str | None
    conversation_binding_revision: int


class ConsoleSettingsFieldProvenance(str, Enum):
    """How a displayed generation field acquired its effective value."""

    INHERITED = "inherited"
    EXPLICIT = "explicit"
    CARRIED = "carried"


@dataclass(frozen=True, slots=True)
class ConsoleSettingsFieldDraft:
    """One generation field's conversation value and profile intent."""

    name: str
    effective_value: object | None
    profile_override: object | None
    provenance: ConsoleSettingsFieldProvenance
    dirty: bool


@dataclass(frozen=True, slots=True)
class ConsoleEndpointDraft:
    """An endpoint edit bound to one normalized provider configuration key."""

    value: str
    bound_provider_config_key: str
    dirty: bool
    checked: bool


@dataclass(frozen=True, slots=True)
class ConsoleModelDraft:
    """Process-local draft retained for one provider and literal model ID."""

    provider: str
    model: str | None
    settings: ConsoleSessionSettings
    field_drafts: tuple[ConsoleSettingsFieldDraft, ...]
    endpoint_draft: ConsoleEndpointDraft | None


@dataclass(frozen=True, slots=True)
class ConsoleSettingsDraftState:
    """Complete settings transaction state shared by both settings surfaces."""

    settings: ConsoleSessionSettings
    context_policy_overrides: ConsoleContextPolicyOverrides
    field_drafts: tuple[ConsoleSettingsFieldDraft, ...]
    model_drafts: tuple[ConsoleModelDraft, ...]
    endpoint_draft: ConsoleEndpointDraft | None


@dataclass(frozen=True, slots=True)
class ConsoleSettingsSubmission:
    """Validated intent submitted by a Console settings surface."""

    submission_id: str
    action: ConsoleSettingsAction
    surface: ConsoleSettingsSurface
    origin: ConsoleSettingsOrigin
    draft: ConsoleSettingsDraftState
    user_display_name_override: str | None
    default_field_mask: frozenset[str]

    def __post_init__(self) -> None:
        """Detach the immutable submission from a caller-owned mask."""
        if not isinstance(self.surface, ConsoleSettingsSurface):
            raise TypeError("surface must be ConsoleSettingsSurface")
        object.__setattr__(
            self, "default_field_mask", frozenset(self.default_field_mask)
        )


@dataclass(frozen=True, slots=True)
class ConsoleSettingsLiveCommit:
    """Exact live values and revisions accepted for one session."""

    submission_id: str
    session_id: str
    persisted_conversation_id: str | None
    conversation_binding_revision: int
    generation_revision: int
    context_policy_revision: int
    settings: ConsoleSessionSettings
    context_policy_overrides: ConsoleContextPolicyOverrides
    accepted_submission: ConsoleSettingsSubmission | None = None
    durability_admission: ConsoleSettingsDurabilityLease | None = None


@dataclass(frozen=True, slots=True)
class ConsoleSettingsCommittedSubmission:
    """A submission paired with its successful live session commit."""

    submission: ConsoleSettingsSubmission
    live_commit: ConsoleSettingsLiveCommit

    def __post_init__(self) -> None:
        """Expose the final normalized submission accepted by the live store."""

        accepted = self.live_commit.accepted_submission
        if accepted is None:
            return
        if accepted.submission_id != self.live_commit.submission_id:
            raise ValueError("Accepted submission does not match the live commit.")
        object.__setattr__(self, "submission", accepted)


@dataclass(frozen=True, slots=True)
class ConsoleSettingsTransfer:
    """Non-committing handoff from quick settings to the full settings view."""

    origin: ConsoleSettingsOrigin
    draft: ConsoleSettingsDraftState


def validate_console_settings_origin(
    origin: ConsoleSettingsOrigin,
    *,
    live_session_id: str | None,
    live_persisted_conversation_id: str | None,
    live_conversation_binding_revision: int,
) -> bool:
    """Return whether a live session still represents ``origin``.

    The one allowed identity transition is ordinary first persistence: an origin
    captured without a conversation ID may acquire its first durable ID while its
    binding revision remains unchanged. Explicit rebinding advances that revision
    and therefore fails closed even when the captured ID was ``None``.
    """

    if live_session_id is None or live_session_id != origin.session_id:
        return False
    if live_conversation_binding_revision != origin.conversation_binding_revision:
        return False
    if origin.persisted_conversation_id is None:
        return True
    return live_persisted_conversation_id == origin.persisted_conversation_id


def initial_settings_draft(
    settings: ConsoleSessionSettings,
    context_policy: ConsoleContextPolicyOverrides,
    *,
    exposed_fields: frozenset[str],
) -> ConsoleSettingsDraftState:
    """Build one process-local transaction from an exact live snapshot.

    ``ChatScreen._console_settings_initial_draft`` is this function (moved out
    of the screen with TASK-35190's `remember_session_pair`).
    """

    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=context_policy,
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in sorted(exposed_fields)
        ),
        model_drafts=(),
        endpoint_draft=None,
    )


def remember_model_draft(
    state: ConsoleSettingsDraftState,
) -> ConsoleSettingsDraftState:
    """Return ``state`` with its current exact provider/model draft remembered.

    Args:
        state: The draft whose current provider/model pair should survive a
            later switch back; never mutated.

    Returns:
        A new ``ConsoleSettingsDraftState`` whose ``model_drafts`` carry one
        entry for the current provider/model pair, keyed by the canonical
        provider IDENTITY (registry entries keep the dashed
        ``custom-ep:<slug>`` spelling) so rebase targets and popover
        carried-source lookups address the same key. That entry moves to the
        end, so the drafts run from least to most recently remembered
        (``rebased_endpoint`` reads the latest on a provider, TASK-35190).
    """
    # CE-001: drafts are keyed by provider IDENTITY. Registry ids
    # (``custom-ep:<slug>``) must stay dashed so the key matches the rebase
    # target and the popover's carried-source lookups -- the config-key
    # normalizer's underscore rewriting would fork the spellings.
    provider = provider_identity_key(state.settings.provider)
    model = normalize_console_model_value(state.settings.model)
    remembered = ConsoleModelDraft(
        provider=provider,
        model=model,
        settings=replace(state.settings, provider=provider, model=model),
        field_drafts=tuple(state.field_drafts),
        endpoint_draft=state.endpoint_draft,
    )
    key = (provider, model)
    drafts = [d for d in state.model_drafts if (d.provider, d.model) != key]
    return replace(state, model_drafts=(*drafts, remembered))


def remember_session_pair(
    draft: ConsoleSettingsDraftState,
    live: ConsoleSessionSettings | None,
) -> ConsoleSettingsDraftState:
    """Return a reopened Chat settings draft remembering the chat's own pair.

    TASK-35190: the chat's own pair stays remembered across a Settings round
    trip, so a pick back lands where it would without the round trip: the
    chat's own pair keeps its Endpoint, and on llama.cpp (whose direct path
    sends on an Endpoint set only on the chat, AC#5) an unseen model keeps
    it too.

    Args:
        draft: The draft Chat settings reopens on (a suspended one, whose
            pair may be another provider's); never mutated.
        live: The chat's committed settings, or None when it has none.

    Returns:
        ``draft`` itself when ``live`` is None; otherwise a copy whose
        ``model_drafts`` are exactly the live pair's remembered draft, its
        fields the full surface's (`FULL_MODEL_DEFAULT_FIELDS`).
    """
    if live is None:
        return draft
    own = initial_settings_draft(
        live, draft.context_policy_overrides, exposed_fields=FULL_MODEL_DEFAULT_FIELDS
    )
    return replace(draft, model_drafts=remember_model_draft(own).model_drafts)


def rebased_endpoint(
    state: ConsoleSettingsDraftState,
    source: ConsoleSettingsDraftState | ConsoleModelDraft,
    *,
    provider_id: str,
    config_key: str,
    default_base_url: str | None,
    preserve_snapshot: bool,
    full_surface: bool,
) -> tuple[str | None, ConsoleEndpointDraft | None]:
    """Return the base URL and endpoint draft a provider/model rebase lands on.

    A kept pair (the current one, or a remembered target) keeps its own
    endpoint. An unseen pair takes the endpoint of the latest draft on its
    provider: the current one, else the most recently remembered one. So a
    llama.cpp pick keeps the chat's Endpoint, after a detour through another
    provider too, and a pick onto another provider never carries it
    (TASK-35190). Only llama.cpp's direct path sends on an Endpoint set only
    on the chat; any other provider blocks one it has not saved
    (``endpoint_not_saved``), so there, as for an unset URL (the configured
    one, not "no endpoint") and a registry entry (whose own URL is its
    endpoint), the target's default chain applies. Only the full surface
    carries an endpoint edit.

    Args:
        state: The draft being rebased.
        source: The kept pair's draft (``state`` or the remembered target),
            else ``state``.
        provider_id: The target's provider identity.
        config_key: The target's provider configuration key.
        default_base_url: The base URL of the target's default chain.
        preserve_snapshot: Whether the target is the current or a
            remembered pair.
        full_surface: Whether the caller exposes every field (Chat settings).

    Returns:
        The base URL, and the endpoint draft (None when there is none).
    """
    peer = (
        source
        if preserve_snapshot
        or provider_identity_key(state.settings.provider) == provider_id
        else next(
            (
                draft
                for draft in reversed(state.model_drafts)
                if draft.provider == provider_id
            ),
            None,
        )
    )
    endpoint = peer.endpoint_draft if peer is not None else None
    bound = (
        endpoint
        if endpoint is not None and endpoint.bound_provider_config_key == config_key
        else None
    )
    if full_surface and bound is not None and bound.dirty:
        return bound.value or None, bound
    base_url = default_base_url
    if (
        peer is not None
        and (endpoint is None or (bound is not None and not bound.dirty))
        and (
            preserve_snapshot
            or (
                peer.settings.base_url
                # Config-free: no handler keys are read for the direct path.
                and resolve_console_provider_identity(
                    provider_id, handler_keys=()
                ).uses_direct_llama_path
            )
        )
    ):
        base_url = peer.settings.base_url
    if base_url is None or (preserve_snapshot and endpoint is None):
        return base_url, None
    return base_url, ConsoleEndpointDraft(
        value=base_url, bound_provider_config_key=config_key, dirty=False, checked=False
    )


def pick_base_url(
    state: ConsoleSettingsDraftState, provider: str, *, full_surface: bool
) -> str | None:
    """Return the URL a pick of an unseen model on ``provider`` lands on.

    The rebaser's own rule (``rebased_endpoint``), so Switch model's rows,
    its readiness and its probe read the server the rebase lands on and
    cannot drift from it (TASK-35190).

    Args:
        state: The draft a pick would rebase.
        provider: The provider picked onto.
        full_surface: Whether the rebase carries an endpoint edit (Chat
            settings, so pick mode).

    Returns:
        The URL, or None for the provider's default chain (its configured
        endpoint, or a registry entry's own URL).
    """
    provider_id = provider_identity_key(provider)
    return rebased_endpoint(
        state,
        state,
        provider_id=provider_id,
        config_key=provider_config_key(provider_id),
        default_base_url=None,
        preserve_snapshot=False,
        full_surface=full_surface,
    )[0]


def quick_blank_field_default(
    app_config: Mapping[str, object],
    target_defaults: ConsoleSessionSettings,
    name: str,
) -> tuple[object | None, object | None]:
    """Resolve one blank quick field the way its quick default Save will.

    A blankable field (Max tokens) skips the exact model profile, because a
    quick Save of that blank deletes the override (ADR-095 D3); resolving it
    through the profile would give the live chat the cap the Save removes.
    Its profile override stays ``None`` ("inherit"), so that Save deletes the
    override instead of writing the fallback cap (Qodo #2947). Every other
    field takes the target default as both values.

    Args:
        app_config: The live application configuration snapshot.
        target_defaults: The target's established default chain.
        name: The quick field whose submitted value is blank.

    Returns:
        ``(effective_value, profile_override)``: the value the live chat
        carries, and the value the rebased draft saves as the override.
    """
    if name not in QUICK_BLANKABLE_DEFAULT_FIELDS:
        value = getattr(target_defaults, name)
        return value, value
    fallback = build_target_default_console_session_settings(
        app_config,
        provider_identity_key(target_defaults.provider),
        normalize_console_model_value(target_defaults.model),
        excluded_model_profile_fields=frozenset({name}),
    )
    return getattr(fallback, name), None
