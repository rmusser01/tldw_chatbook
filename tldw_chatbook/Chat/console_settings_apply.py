"""Immutable contracts for applying Console conversation settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_settings_durability import (
    ConsoleSettingsDurabilityLease,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    normalize_console_model_value,
)
from tldw_chatbook.Chat.provider_readiness import provider_config_key


QUICK_MODEL_DEFAULT_FIELDS = frozenset({"temperature", "streaming"})
FULL_MODEL_DEFAULT_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
        "streaming",
    }
)


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


def remember_model_draft(
    state: ConsoleSettingsDraftState,
) -> ConsoleSettingsDraftState:
    """Return ``state`` with its current exact provider/model draft remembered."""

    provider = provider_config_key(state.settings.provider)
    model = normalize_console_model_value(state.settings.model)
    remembered = ConsoleModelDraft(
        provider=provider,
        model=model,
        settings=replace(state.settings, provider=provider, model=model),
        field_drafts=tuple(state.field_drafts),
        endpoint_draft=state.endpoint_draft,
    )
    key = (provider, model)
    drafts = list(state.model_drafts)
    for index, existing in enumerate(drafts):
        if (existing.provider, existing.model) == key:
            drafts[index] = remembered
            break
    else:
        drafts.append(remembered)
    return replace(state, model_drafts=tuple(drafts))
