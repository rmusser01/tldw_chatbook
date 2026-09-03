"""Console session settings modal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields, replace
import re
from typing import Any, Awaitable, Callable, Mapping, Protocol
from uuid import uuid4

from textual import events
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

from tldw_chatbook.Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
)
from tldw_chatbook.Chat.console_context_compaction import EffectiveMemoryKind
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyDefaults,
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
    ContextPolicyError,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    normalize_chat_display_name,
)
from tldw_chatbook.Chat.console_provider_endpoints import (
    first_configured_endpoint,
    normalize_generic_endpoint_for_compare,
)
from tldw_chatbook.Chat.console_provider_support import (
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.local_server_discovery import (
    normalize_probe_base_url,
    LocalModelProbeResult,
    endpoint_display,
    probe_models_endpoint,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SESSION_SETTINGS_SOURCES,
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    DEFAULT_LLAMACPP_BASE_URL,
    URL_BASED_PROVIDER_KEYS,
    build_console_model_options,
    build_console_provider_options,
    build_console_settings_readiness,
    console_settings_warnings,
    normalize_console_model_value,
    normalize_llamacpp_base_url,
    reasoning_effort_hint_for_model,
    validate_console_session_settings,
)
from tldw_chatbook.Chat.thinking_blocks import (
    ThinkingHistoryPolicy,
    normalize_thinking_history_policy,
)
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    ConsoleEndpointDraft,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSurface,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
    remember_model_draft,
)
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultSavePhase,
    build_console_default_readiness_preview,
    format_console_endpoint_preview,
)
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label
from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
    ProviderSettingsNavigationTarget,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.model_capabilities import is_vision_capable
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker
from .console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
    format_context_tokens,
)
from .console_settings_summary import build_console_readiness_presentation
from rich.markup import escape as escape_markup


MODEL_INPUT_PLACEHOLDER = "Enter model id"
MODAL_BODY_MIN_HEIGHT = 0
MODAL_CONTROL_HEIGHT = 3
MODAL_LABEL_WIDTH = 23
MODEL_CUSTOM_BUTTON_WIDTH = 18
MODEL_DISCOVER_BUTTON_ID = "console-settings-model-discover"
MODEL_DISCOVER_STATUS_ID = "console-settings-model-discover-status"
MODEL_DISCOVER_BUTTON_LABEL = "Discover models"
MODEL_DISCOVER_BUTTON_WIDTH = 19
_NO_CONFIGURED_MODELS_VALUE = "__no_configured_models__"
MODEL_DISCOVER_MISSING_URL_COPY = "Enter a base URL to discover models."
MODEL_DISCOVER_INVALID_URL_COPY = (
    "Enter a valid http(s) endpoint URL to discover models."
)
ModelProber = Callable[[str, str], Awaitable[LocalModelProbeResult]]
CurrentMemoryResetter = Callable[[], tuple[str, int] | None]
CurrentMemoryUndo = Callable[[str, int], bool]
AllMemoryResetter = Callable[[], int]
ContextCompactor = Callable[[], Awaitable[tuple[bool, str]]]
STREAMING_TOGGLE_WIDTH = 12
PROVIDER_CHOICE_INPUT_MAX_LENGTH = 64
COMPACTION_CLOSE_WARNING = "Provider work may continue and may still be billed."


_CONSOLE_SETTINGS_DRAFT_VERSION = 1
_SNAPSHOT_PROVIDER_RE = re.compile(r"[a-z0-9][a-z0-9_]{0,127}\Z")
_SNAPSHOT_SAFE_TEXT_RE = re.compile(r"[^\x00-\x1f\x7f\x80-\x9f]*\Z")
_SNAPSHOT_PRIVATE_TEXT_RE = re.compile(
    r"[^\x00-\x08\x0b-\x0c\x0e-\x1f\x7f\x80-\x9f]*\Z"
)
_SNAPSHOT_RAW_TEXT_LIMIT = 4096
_SNAPSHOT_MODEL_TEXT_LIMIT = 512
_SNAPSHOT_URL_TEXT_LIMIT = 2048
_SNAPSHOT_PRIVATE_TEXT_LIMIT = 100_000
_SNAPSHOT_PROVIDER_DRAFT_LIMIT = 256
_SNAPSHOT_SCROLL_ANCHOR_LIMIT = 1_000_000
_SNAPSHOT_RAW_VALUE_IDS = frozenset(
    {
        "console-settings-provider",
        "console-settings-model-picker",
        "console-settings-base-url",
        "console-settings-user-display-name",
        "console-settings-temperature",
        "console-settings-top-p",
        "console-settings-min-p",
        "console-settings-top-k",
        "console-settings-max-tokens",
        "console-settings-seed",
        "console-settings-presence-penalty",
        "console-settings-frequency-penalty",
        "console-settings-streaming",
        "console-settings-reasoning-effort",
        "console-settings-reasoning-summary",
        "console-settings-verbosity",
        "console-settings-thinking-effort",
        "console-settings-thinking-budget-tokens",
        "console-context-budget-mode",
        "console-context-custom-budget",
        "console-context-compaction-mode",
        "console-context-compaction-representation",
        "console-context-trigger-percent",
        "console-context-target-percent",
        "console-context-summary-max",
        "console-context-failure-behavior",
        "console-context-carry-forward",
    }
)
_SNAPSHOT_FOCUS_CONTROL_IDS = frozenset(
    {
        "console-settings-provider",
        "console-settings-model-picker",
        "console-settings-model-custom",
        "console-settings-base-url",
        "console-settings-user-display-name",
        "console-settings-temperature",
        "console-settings-top-p",
        "console-settings-min-p",
        "console-settings-top-k",
        "console-settings-max-tokens",
        "console-settings-seed",
        "console-settings-presence-penalty",
        "console-settings-frequency-penalty",
        "console-settings-streaming",
        "console-settings-reasoning-effort",
        "console-settings-reasoning-summary",
        "console-settings-verbosity",
        "console-settings-thinking-effort",
        "console-settings-thinking-budget-tokens",
        "console-settings-view-model",
        "console-settings-view-context",
        "console-context-budget-mode",
        "console-context-custom-budget",
        "console-context-compaction-mode",
        "console-context-compaction-representation",
        "console-context-trigger-percent",
        "console-context-target-percent",
        "console-context-summary-max",
        "console-context-failure-behavior",
        "console-context-carry-forward",
        "console-context-compact-now",
        "console-context-reset-current",
        "console-context-undo-reset",
        "console-context-reset-overrides",
        "console-context-reset-all",
        "console-context-confirm-reset-all",
    }
)
_SNAPSHOT_DISCLOSURE_KEYS = frozenset({"advanced_generation", "connection_details"})
_SNAPSHOT_CONTEXT_OVERRIDE_KEYS = frozenset(
    item.name for item in fields(ConsoleContextPolicyOverrides)
)
_SNAPSHOT_SETTINGS_FIELDS = tuple(item.name for item in fields(ConsoleSessionSettings))
_SNAPSHOT_OPTIONAL_STRING_FIELDS = frozenset(
    {
        "model",
        "base_url",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "system_prompt",
        "pinned_prefill",
    }
)
_SNAPSHOT_OPTIONAL_INT_FIELDS = frozenset(
    {"top_k", "max_tokens", "seed", "thinking_budget_tokens"}
)
_SNAPSHOT_OPTIONAL_FLOAT_FIELDS = frozenset(
    {"min_p", "presence_penalty", "frequency_penalty"}
)
_SNAPSHOT_REQUIRED_FLOAT_FIELDS = frozenset({"temperature", "top_p"})
_SNAPSHOT_REQUIRED_STRING_FIELDS = frozenset(
    {"provider", "character_label", "source"}
)
_SNAPSHOT_SETTING_CHOICE_DOMAINS = {
    "reasoning_effort": frozenset(
        {"none", "minimal", "low", "medium", "high", "xhigh"}
    ),
    "reasoning_summary": frozenset({"auto", "concise", "detailed", "none"}),
    "verbosity": frozenset({"low", "medium", "high"}),
    "thinking_effort": frozenset({"off", "low", "medium", "high", "xhigh", "max"}),
}
_SNAPSHOT_FLOAT_BOUNDS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
}
_SNAPSHOT_INT_MINIMUMS = {
    "top_k": 0,
    "max_tokens": 1,
    "seed": 0,
    "thinking_budget_tokens": 1024,
}
_SNAPSHOT_CONTEXT_ENUM_KEYS = frozenset(
    {
        "budget_mode",
        "compaction_mode",
        "compaction_representation",
        "failure_behavior",
        "carry_forward_mode",
    }
)
_SNAPSHOT_CONTEXT_INT_KEYS = frozenset(
    {"custom_budget_tokens", "summary_max_tokens"}
)
_SNAPSHOT_CONTEXT_FLOAT_KEYS = frozenset({"trigger_ratio", "target_ratio"})
_SNAPSHOT_CONTEXT_ENUM_TYPES = {
    "budget_mode": ContextBudgetMode,
    "compaction_mode": ContextCompactionMode,
    "compaction_representation": ContextCompactionRepresentation,
    "failure_behavior": CompactionFailureBehavior,
    "carry_forward_mode": ContextCarryForwardMode,
}


def _snapshot_text(
    value: object,
    *,
    limit: int,
    allow_blank: bool = True,
    allow_private_formatting: bool = False,
) -> bool:
    """Validate bounded primitive text retained in the process-memory snapshot."""
    return (
        type(value) is str
        and (allow_blank or bool(value))
        and len(value) <= limit
        and (
            _SNAPSHOT_PRIVATE_TEXT_RE.fullmatch(value) is not None
            if allow_private_formatting
            else _SNAPSHOT_SAFE_TEXT_RE.fullmatch(value) is not None
        )
    )


def _snapshot_provider(value: object) -> str | None:
    """Return one already-normalized bounded provider identifier, or ``None``."""
    if not _snapshot_text(value, limit=128, allow_blank=False):
        return None
    assert type(value) is str
    normalized = provider_config_key(value)
    if normalized != value or _SNAPSHOT_PROVIDER_RE.fullmatch(value) is None:
        return None
    return value


def _snapshot_model(
    value: object, *, allow_none: bool = False, allow_blank: bool = False
) -> bool:
    return (allow_none and value is None) or _snapshot_text(
        value, limit=_SNAPSHOT_MODEL_TEXT_LIMIT, allow_blank=allow_blank
    )


def _valid_snapshot_context_overrides(source: Mapping[str, object]) -> bool:
    """Reject config-style coercion before context enums are reconstructed."""
    for key, value in source.items():
        if key not in _SNAPSHOT_CONTEXT_OVERRIDE_KEYS:
            return False
        if value is None:
            continue
        if key in _SNAPSHOT_CONTEXT_ENUM_KEYS:
            enum_type = _SNAPSHOT_CONTEXT_ENUM_TYPES[key]
            if (
                not _snapshot_text(value, limit=64, allow_blank=False)
                or value not in {item.value for item in enum_type}
            ):
                return False
        elif key in _SNAPSHOT_CONTEXT_INT_KEYS:
            if type(value) is not int:
                return False
        elif key in _SNAPSHOT_CONTEXT_FLOAT_KEYS:
            if type(value) is not float:
                return False
        else:
            return False
    return True


def _copy_snapshot_string_mapping(
    source: Mapping[str, object],
    *,
    allow_none: bool = False,
    value_limit: int = _SNAPSHOT_MODEL_TEXT_LIMIT,
    allow_blank: bool = False,
) -> dict[str, str | None] | None:
    """Copy a small allowlisted provider draft mapping without generic deepcopy."""
    if len(source) > _SNAPSHOT_PROVIDER_DRAFT_LIMIT:
        return None
    copied: dict[str, str | None] = {}
    for key, value in source.items():
        provider = _snapshot_provider(key)
        if provider is None:
            return None
        if not (
            (allow_none and value is None)
            or _snapshot_text(value, limit=value_limit, allow_blank=allow_blank)
        ):
            return None
        copied[provider] = value  # type: ignore[assignment]
    return copied


def _valid_snapshot_settings_mapping(source: Mapping[str, object]) -> bool:
    """Validate config-independent snapshot shape and semantic invariants."""
    for name, value in source.items():
        if name in _SNAPSHOT_OPTIONAL_STRING_FIELDS:
            limit = (
                _SNAPSHOT_PRIVATE_TEXT_LIMIT
                if name in {"system_prompt", "pinned_prefill"}
                else _SNAPSHOT_URL_TEXT_LIMIT
                if name == "base_url"
                else _SNAPSHOT_MODEL_TEXT_LIMIT
            )
            if value is not None and not _snapshot_text(
                value,
                limit=limit,
                allow_private_formatting=name in {"system_prompt", "pinned_prefill"},
            ):
                return False
        elif name in _SNAPSHOT_OPTIONAL_INT_FIELDS:
            if value is not None and type(value) is not int:
                return False
        elif name in _SNAPSHOT_OPTIONAL_FLOAT_FIELDS:
            if value is not None and type(value) is not float:
                return False
        elif name in _SNAPSHOT_REQUIRED_FLOAT_FIELDS:
            if type(value) is not float:
                return False
        elif name in _SNAPSHOT_REQUIRED_STRING_FIELDS:
            if name == "provider":
                if _snapshot_provider(value) is None:
                    return False
            elif not _snapshot_text(value, limit=_SNAPSHOT_RAW_TEXT_LIMIT):
                return False
        elif name == "streaming":
            if type(value) is not bool:
                return False
        else:
            return False
    try:
        settings = ConsoleSessionSettings(**dict(source))
    except TypeError:
        return False
    if settings.source not in CONSOLE_SESSION_SETTINGS_SOURCES:
        return False
    if settings.base_url is not None and settings.base_url.strip():
        if not validate_url(settings.base_url):
            return False
    for name, (minimum, maximum) in _SNAPSHOT_FLOAT_BOUNDS.items():
        value = getattr(settings, name)
        if value is not None and not minimum <= value <= maximum:
            return False
    for name, minimum in _SNAPSHOT_INT_MINIMUMS.items():
        value = getattr(settings, name)
        if value is not None and value < minimum:
            return False
    for name, domain in _SNAPSHOT_SETTING_CHOICE_DOMAINS.items():
        value = getattr(settings, name)
        if value is not None and value.strip() and value not in domain:
            return False
    return True


@dataclass(frozen=True, slots=True)
class ConsoleSettingsDraftSnapshot:
    """Versioned, process-memory-only raw Conversation settings draft."""

    settings: ConsoleSessionSettings
    context_policy_overrides: ConsoleContextPolicyOverrides
    raw_values: Mapping[str, str | bool]
    provider_model_drafts: Mapping[str, str | None]
    provider_base_url_drafts: Mapping[str, str]
    active_view: str
    scroll_anchor: int
    focus_control_id: str | None
    disclosure_state: Mapping[str, bool]

    def __repr__(self) -> str:
        """Return diagnostic metadata without exposing Console-owned content."""
        return (
            f"{type(self).__name__}(version={_CONSOLE_SETTINGS_DRAFT_VERSION}, "
            f"active_view={self.active_view!r}, scroll_anchor={self.scroll_anchor}, "
            f"focus_control_id={self.focus_control_id!r}, "
            f"raw_value_count={len(self.raw_values)}, "
            f"provider_draft_count={len(self.provider_model_drafts)})"
        )

    def __post_init__(self) -> None:
        if type(self.settings) is not ConsoleSessionSettings:
            raise ValueError("settings draft is invalid")
        if not _valid_snapshot_settings_mapping(
            {name: getattr(self.settings, name) for name in _SNAPSHOT_SETTINGS_FIELDS}
        ):
            raise ValueError("settings draft is invalid")
        if type(self.context_policy_overrides) is not ConsoleContextPolicyOverrides:
            raise ValueError("context policy overrides are invalid")
        if not _valid_snapshot_context_overrides(
            self.context_policy_overrides.to_dict()
        ):
            raise ValueError("context policy overrides are invalid")
        if type(self.active_view) is not str or self.active_view not in {"model", "context"}:
            raise ValueError("active view is invalid")
        if (
            type(self.scroll_anchor) is not int
            or not 0 <= self.scroll_anchor <= _SNAPSHOT_SCROLL_ANCHOR_LIMIT
        ):
            raise ValueError("scroll anchor is invalid")
        if (
            self.focus_control_id is not None
            and (
                type(self.focus_control_id) is not str
                or self.focus_control_id not in _SNAPSHOT_FOCUS_CONTROL_IDS
            )
        ):
            raise ValueError("focus control is invalid")
        if not isinstance(self.raw_values, Mapping) or not set(self.raw_values).issubset(
            _SNAPSHOT_RAW_VALUE_IDS
        ):
            raise ValueError("raw modal values are invalid")
        raw_values: dict[str, str | bool] = {}
        for key, value in self.raw_values.items():
            if type(key) is not str:
                raise ValueError("raw modal values are invalid")
            if key == "console-settings-streaming":
                if type(value) is not bool:
                    raise ValueError("raw modal values are invalid")
            elif key == "console-settings-provider":
                if _snapshot_provider(value) is None:
                    raise ValueError("raw modal values are invalid")
            elif key == "console-settings-model-picker":
                if not _snapshot_model(value, allow_blank=True):
                    raise ValueError("raw modal values are invalid")
            elif key == "console-settings-base-url":
                if not _snapshot_text(value, limit=_SNAPSHOT_URL_TEXT_LIMIT):
                    raise ValueError("raw modal values are invalid")
            elif not _snapshot_text(value, limit=_SNAPSHOT_RAW_TEXT_LIMIT):
                raise ValueError("raw modal values are invalid")
            raw_values[key] = value
        model_drafts = _copy_snapshot_string_mapping(
            self.provider_model_drafts, allow_none=True, allow_blank=True
        )
        base_url_drafts = _copy_snapshot_string_mapping(
            self.provider_base_url_drafts,
            value_limit=_SNAPSHOT_URL_TEXT_LIMIT,
            allow_blank=True,
        )
        if model_drafts is None or base_url_drafts is None:
            raise ValueError("provider drafts are invalid")
        if not isinstance(self.disclosure_state, Mapping) or set(
            self.disclosure_state
        ) != _SNAPSHOT_DISCLOSURE_KEYS:
            raise ValueError("disclosure state is invalid")
        disclosure_state: dict[str, bool] = {}
        for key, value in self.disclosure_state.items():
            if type(key) is not str or type(value) is not bool:
                raise ValueError("disclosure state is invalid")
            disclosure_state[key] = value
        object.__setattr__(self, "raw_values", raw_values)
        object.__setattr__(self, "provider_model_drafts", model_drafts)
        object.__setattr__(self, "provider_base_url_drafts", base_url_drafts)
        object.__setattr__(self, "disclosure_state", disclosure_state)

    def to_mapping(self) -> dict[str, object]:
        """Return a structurally detached primitive mapping for screen state."""
        settings = {
            name: getattr(self.settings, name) for name in _SNAPSHOT_SETTINGS_FIELDS
        }
        return {
            "version": _CONSOLE_SETTINGS_DRAFT_VERSION,
            "settings": settings,
            "context_policy_overrides": dict(self.context_policy_overrides.to_dict()),
            "raw_values": dict(self.raw_values),
            "provider_model_drafts": dict(self.provider_model_drafts),
            "provider_base_url_drafts": dict(self.provider_base_url_drafts),
            "active_view": self.active_view,
            "scroll_anchor": self.scroll_anchor,
            "focus_control_id": self.focus_control_id,
            "disclosure_state": dict(self.disclosure_state),
        }

    @classmethod
    def from_mapping(
        cls, source: Mapping[str, object]
    ) -> "ConsoleSettingsDraftSnapshot | None":
        """Parse a complete snapshot mapping, refusing unknown or malformed state."""
        required_keys = {
            "version",
            "settings",
            "context_policy_overrides",
            "raw_values",
            "provider_model_drafts",
            "provider_base_url_drafts",
            "active_view",
            "scroll_anchor",
            "focus_control_id",
            "disclosure_state",
        }
        if not isinstance(source, Mapping) or set(source) != required_keys:
            return None
        if (
            type(source.get("version")) is not int
            or source.get("version") != _CONSOLE_SETTINGS_DRAFT_VERSION
        ):
            return None
        raw_settings = source.get("settings")
        raw_overrides = source.get("context_policy_overrides")
        raw_values = source.get("raw_values")
        model_drafts = source.get("provider_model_drafts")
        base_url_drafts = source.get("provider_base_url_drafts")
        disclosure_state = source.get("disclosure_state")
        if (
            not isinstance(raw_settings, Mapping)
            or set(raw_settings) != set(_SNAPSHOT_SETTINGS_FIELDS)
            or not _valid_snapshot_settings_mapping(raw_settings)
            or not isinstance(raw_overrides, Mapping)
            or not set(raw_overrides).issubset(_SNAPSHOT_CONTEXT_OVERRIDE_KEYS)
            or not _valid_snapshot_context_overrides(raw_overrides)
            or not isinstance(raw_values, Mapping)
            or not isinstance(model_drafts, Mapping)
            or not isinstance(base_url_drafts, Mapping)
            or not isinstance(disclosure_state, Mapping)
        ):
            return None
        try:
            settings = ConsoleSessionSettings(**dict(raw_settings))
            overrides = ConsoleContextPolicyOverrides.from_mapping(
                dict(raw_overrides)
            )
            return cls(
                settings=settings,
                context_policy_overrides=overrides,
                raw_values=dict(raw_values),
                provider_model_drafts=dict(model_drafts),
                provider_base_url_drafts=dict(base_url_drafts),
                active_view=source["active_view"],  # type: ignore[arg-type]
                scroll_anchor=source["scroll_anchor"],  # type: ignore[arg-type]
                focus_control_id=source["focus_control_id"],  # type: ignore[arg-type]
                disclosure_state=dict(disclosure_state),
            )
        except (ContextPolicyError, TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class ConsoleSettingsCredentialRequest:
    """Modal result that asks canonical Settings to configure one credential."""

    snapshot: ConsoleSettingsDraftSnapshot = field(repr=False)
    provider: str
    model: str | None

    def __post_init__(self) -> None:
        if type(self.snapshot) is not ConsoleSettingsDraftSnapshot:
            raise ValueError("credential request snapshot is invalid")
        target = ProviderSettingsNavigationTarget(
            category="providers-models",
            provider=self.provider,
            model=self.model,
            field="api_key",
            return_revision=1,
        )
        object.__setattr__(self, "provider", target.provider)
        object.__setattr__(self, "model", target.model)


@dataclass(frozen=True, slots=True)
class ConsoleSettingsResult:
    """Provider settings plus the separately owned per-chat display name."""

    settings: ConsoleSessionSettings
    user_display_name_override: str | None
    context_policy_overrides: ConsoleContextPolicyOverrides | None = None
    thinking_history_policy: ThinkingHistoryPolicy | None = None


# (label, input id, accepted-values placeholder) - placeholders mirror the
# Settings screen's enumerated hints for these provider-specific fields.
PROVIDER_CHOICE_INPUTS = (
    (
        "Reasoning effort",
        "console-settings-reasoning-effort",
        "none, minimal, low, medium, high, xhigh",
    ),
    (
        "Reasoning summary",
        "console-settings-reasoning-summary",
        "auto, concise, detailed, none",
    ),
    ("Verbosity", "console-settings-verbosity", "low, medium, high"),
    (
        "Thinking effort",
        "console-settings-thinking-effort",
        "off, low, medium, high, xhigh, max",
    ),
)
# ADR-066: local thinking execution keys. These providers consume only the
# reasoning-effort level (and the token budget) via their local thinking
# payload fields; the other provider-specific choice inputs have no effect.
_LOCAL_THINKING_EXECUTION_KEYS = frozenset(
    {
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local-llm",
        "vllm",
        "local_vllm",
        "local_mlx_lm",
        "custom-openai-api",
        "custom-openai-api-2",
    }
)
# Choice inputs with no effect on local thinking providers. The reasoning
# effort input is excluded: local providers consume it via
# chat_template_kwargs / reasoning_effort.
_LOCAL_NO_EFFECT_CHOICE_INPUT_IDS = frozenset(
    {
        "console-settings-thinking-effort",
        "console-settings-reasoning-summary",
        "console-settings-verbosity",
    }
)
PROVIDER_CHOICE_NO_EFFECT_SUFFIX = " (no effect on this provider)"
STREAMING_ON_LABEL = "On"
STREAMING_OFF_LABEL = "Off"
CONSOLE_SETTINGS_MODEL_SCOPE_COPY = (
    "Apply affects this chat. Save as model default writes this model's shown "
    "generation profile. Make default for new chats also selects this provider "
    "and model for eligible new chats."
)
CONSOLE_SETTINGS_CONTEXT_SCOPE_COPY = (
    "Apply affects this chat. Global context defaults are in "
    "F9 Settings > Console behavior."
)
CONSOLE_SETTINGS_SCOPE_COPY = CONSOLE_SETTINGS_MODEL_SCOPE_COPY
#: Debounce for the custom-model-id `Input` -- mirrors the picker/filter
#: family's 0.2 s shape (`console_prompt_picker_modal.py`). Each settle
#: rebuilds a full `ConsoleSessionSettings` draft from every form field and
#: re-validates it, which must not happen on every keystroke (task-15476).
CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS = 0.2
# Draft fields persisted under [api_settings.<provider>] by Save as default.
STREAMING_INHERIT_LABEL = "Inherit"
_GENERATION_FIELD_BY_INPUT_ID = {
    "console-settings-temperature": "temperature",
    "console-settings-top-p": "top_p",
    "console-settings-min-p": "min_p",
    "console-settings-top-k": "top_k",
    "console-settings-max-tokens": "max_tokens",
    "console-settings-seed": "seed",
    "console-settings-presence-penalty": "presence_penalty",
    "console-settings-frequency-penalty": "frequency_penalty",
    "console-settings-reasoning-effort": "reasoning_effort",
    "console-settings-reasoning-summary": "reasoning_summary",
    "console-settings-verbosity": "verbosity",
    "console-settings-thinking-effort": "thinking_effort",
    "console-settings-thinking-budget-tokens": "thinking_budget_tokens",
}


class DraftRebaser(Protocol):
    """Controller-owned provider/model rebase seam."""

    def __call__(
        self,
        state: ConsoleSettingsDraftState,
        *,
        provider: str,
        model: str | None,
        app_config: Mapping[str, object],
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState: ...


class DefaultReadinessResolver(Protocol):
    """Configuration-owned readiness seam for future-chat defaults."""

    def __call__(self, provider: str, model: str | None) -> Any: ...


LiveCommitter = Callable[[ConsoleSettingsSubmission], ConsoleSettingsLiveCommit]
DefaultRecoveryHandler = Callable[
    [ConsoleDefaultRecoveryRequest],
    Awaitable[ConsoleDefaultDurabilityState],
]


def _settings_screen_region(widget: Any) -> Any:
    """Return a mounted settings widget region in screen coordinates.

    Args:
        widget: Textual widget or test double with a mounted region.

    Returns:
        The widget's absolute screen region when the installed Textual version
        exposes one; otherwise the mounted widget region used by this project.
    """
    return getattr(widget, "screen_region", None) or widget.region


def _is_local_thinking_provider(provider: str | None) -> bool:
    """Return whether a provider executes through a local thinking key.

    Args:
        provider: Raw provider name from Console controls or session settings.

    Returns:
        True when the resolved execution key is one of the local thinking
        keys, i.e. the provider-specific choice inputs other than the
        reasoning-effort level have no effect on sends.
    """
    identity = resolve_console_provider_identity(provider)
    return identity.execution_key in _LOCAL_THINKING_EXECUTION_KEYS


async def _default_model_prober(
    base_url: str, provider_key: str
) -> LocalModelProbeResult:
    """Probe a models endpoint with the shared discovery helper.

    Args:
        base_url: Endpoint root taken from the current base-URL draft.
        provider_key: Normalized provider config key (enables the Ollama
            ``/api/tags`` fallback).

    Returns:
        The probe result, including honest failure copy on error.
    """
    return await probe_models_endpoint(base_url, provider_key=provider_key)


class ConsoleSettingsInput(Input):
    """Input field with browser-safe focus handoff behavior."""

    BINDINGS = [
        (
            Binding("home", "home", binding.description, show=binding.show)
            if binding.key == "home,ctrl+a"
            else binding
        )
        for binding in Input.BINDINGS
    ] + [
        Binding("ctrl+a,super+a", "select_all", "Select all", show=False),
    ]

    def on_click(self, event: events.Click | None = None) -> None:
        """Avoid trapping later Select clicks after browser text editing.

        Args:
            event: Optional click event to forward when Textual Web redirects a
                select click through the focused input.
        """
        self.select_all()
        self.release_mouse()
        if event is None:
            return
        handler = getattr(
            self.screen, "_open_select_from_redirected_settings_click", None
        )
        if callable(handler):
            handler(event)

    def on_blur(self) -> None:
        """Avoid trapping later Select clicks after browser text editing."""
        self.release_mouse()


class ConsoleSettingsModal(
    SafeModalDismissMixin,
    ModalScreen[
        ConsoleSettingsCommittedSubmission
        | ConsoleSettingsTransfer
        | ConsoleSettingsCredentialRequest
        | None
    ],
):
    """Edit a draft of the current Console session settings."""

    DEFAULT_CSS = f"""
    ConsoleSettingsModal #console-settings-body {{
        height: 1fr;
        min-height: {MODAL_BODY_MIN_HEIGHT};
        overflow-y: auto;
        overflow-x: hidden;
    }}

    /* TASK-363: the validation summary was near-body-text salience ($ds-status
       -error 10% bg / primary text), so "Save did nothing" read as no feedback
       in a taller-than-viewport modal. Make it unmistakably an error: bold
       error-coloured text, a stronger fill, and a heavy error rule down its
       edge. (Scoped here so it overrides the shared bundle rule.) */
    ConsoleSettingsModal .console-settings-error {{
        background: $error 25%;
        color: $text-error;
        text-style: bold;
        border-left: thick $error;
    }}

    ConsoleSettingsModal .console-settings-modal-section {{
        height: auto;
    }}

    ConsoleSettingsModal .console-settings-modal-row {{
        height: auto;
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal .console-settings-modal-label {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal Input,
    ConsoleSettingsModal Select,
    ConsoleSettingsModal Button {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal #console-settings-view-tabs {{
        height: 3;
        min-height: 3;
    }}

    ConsoleSettingsModal #console-settings-fold-hint {{
        height: 1;
        min-height: 1;
        color: $text-muted;
    }}

    ConsoleSettingsModal #console-settings-actions Button {{
        border: none;
    }}

    ConsoleSettingsModal #console-settings-new-chat-default-block,
    ConsoleSettingsModal #console-settings-default-recovery,
    ConsoleSettingsModal #console-settings-default-recovery-summary {{
        height: auto;
    }}

    ConsoleSettingsModal #console-settings-new-chat-default-block,
    ConsoleSettingsModal #console-settings-default-recovery {{
        margin: 1 0 0 0;
    }}

    ConsoleSettingsModal #console-settings-default-recovery-actions {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleSettingsModal #console-settings-memory-review {{
        height: auto;
        max-height: 12;
        overflow-y: auto;
        background: $surface;
        padding: 0 1;
    }}

    ConsoleSettingsModal .console-context-action-row {{
        height: auto;
        min-height: 3;
    }}

    ConsoleSettingsModal #console-settings-close-guard {{
        display: none;
        position: absolute;
        offset: 0 0;
        width: 100%;
        height: 100%;
        align: center middle;
        padding: 2;
        background: $panel;
    }}

    ConsoleSettingsModal #console-settings-close-guard.visible {{
        display: block;
    }}

    ConsoleSettingsModal #console-settings-close-actions {{
        width: 100%;
        height: auto;
        align: center middle;
    }}
    """

    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        Binding(
            "tab",
            "settings_focus_next",
            show=False,
            priority=True,
        ),
        Binding(
            "shift+tab",
            "settings_focus_previous",
            show=False,
            priority=True,
        ),
    ]
    SAFE_MODAL_CONTENT = "#console-settings-modal"

    def __init__(
        self,
        *,
        settings: ConsoleSessionSettings,
        origin: ConsoleSettingsOrigin | None = None,
        initial_draft: ConsoleSettingsDraftState | None = None,
        transfer: ConsoleSettingsTransfer | None = None,
        user_display_name_override: str | None = None,
        global_user_display_name: str = "User",
        app_config: Mapping[str, object],
        providers_models: Mapping[str, list[str]],
        context_estimate: ConsoleSettingsContextEstimate,
        context_state: ConsoleContextControlState | None = None,
        can_save: bool,
        focus_model: bool = False,
        focus_context: bool = False,
        reset_current_memory: CurrentMemoryResetter | None = None,
        undo_current_memory_reset: CurrentMemoryUndo | None = None,
        reset_all_memories: AllMemoryResetter | None = None,
        compact_now: ContextCompactor | None = None,
        model_prober: ModelProber | None = None,
        draft_rebaser: DraftRebaser | None = None,
        live_committer: LiveCommitter | None = None,
        default_readiness_resolver: DefaultReadinessResolver | None = None,
        default_durability_state: ConsoleDefaultDurabilityState | None = None,
        default_recovery_handler: DefaultRecoveryHandler | None = None,
        suspended_draft: ConsoleSettingsDraftSnapshot | None = None,
    ) -> None:
        super().__init__()
        if transfer is not None:
            origin = transfer.origin
            initial_draft = transfer.draft
            settings = transfer.draft.settings
        if (
            suspended_draft is not None
            and type(suspended_draft) is not ConsoleSettingsDraftSnapshot
        ):
            raise TypeError("suspended draft must be a ConsoleSettingsDraftSnapshot")
        self._suspended_draft = suspended_draft
        if suspended_draft is not None:
            settings = suspended_draft.settings
            user_display_name_override = suspended_draft.raw_values.get(
                "console-settings-user-display-name", user_display_name_override
            )
        self._origin = origin or ConsoleSettingsOrigin(
            "legacy-console-settings", None, 0
        )
        self._draft = initial_draft or self._initial_full_draft(settings)
        if suspended_draft is not None:
            self._draft = replace(
                self._draft,
                settings=settings,
                context_policy_overrides=suspended_draft.context_policy_overrides,
            )
        self._settings = self._draft.settings
        try:
            self._user_display_name_override = normalize_chat_display_name(
                user_display_name_override, blank_means_none=True
            )
        except ChatDisplayNameError:
            self._user_display_name_override = None
        try:
            self._global_user_display_name = (
                normalize_chat_display_name(
                    global_user_display_name, blank_means_none=False
                )
                or "User"
            )
        except ChatDisplayNameError:
            self._global_user_display_name = "User"
        self._app_config = app_config
        self._providers_models = providers_models
        self._context_estimate = context_estimate
        self._context_state = self._context_state_with_draft_overrides(
            context_state,
            settings=self._settings,
            estimate=context_estimate,
            overrides=self._draft.context_policy_overrides,
        )
        if suspended_draft is not None:
            self._context_state = self._context_state_with_overrides(
                suspended_draft.context_policy_overrides
            )
        self._can_save = can_save
        self._focus_model = focus_model
        self._active_view = (
            suspended_draft.active_view
            if suspended_draft is not None
            else ("context" if focus_context else "model")
        )
        self._reset_current_memory = reset_current_memory
        self._undo_current_memory_reset = undo_current_memory_reset
        self._reset_all_memories = reset_all_memories
        self._compact_now = compact_now
        self._memory_reset_token: tuple[str, int] | None = None
        self._confirm_reset_all = False
        self._context_overrides_reset = False
        self._advanced_generation_disclosed = bool(
            suspended_draft.disclosure_state["advanced_generation"]
            if suspended_draft is not None
            else False
        )
        self._connection_details_disclosed = bool(
            suspended_draft.disclosure_state["connection_details"]
            if suspended_draft is not None
            else False
        )
        self._model_prober: ModelProber = model_prober or _default_model_prober
        self._draft_rebaser = draft_rebaser
        self._live_committer = live_committer
        self._default_readiness_resolver = default_readiness_resolver
        self._default_durability_state = (
            default_durability_state or ConsoleDefaultDurabilityState()
        )
        self._default_recovery_handler = default_recovery_handler
        self._default_recovery_pending: (
            tuple[
                int,
                ConsoleDefaultSavePhase,
            ]
            | None
        ) = None
        self._default_recovery_layout_phase: ConsoleDefaultSavePhase | None = None
        self._discovered_model_ids: dict[str, tuple[str, ...]] = {}
        streaming_field = self._draft_field("streaming")
        self._streaming_draft: bool | None = (
            bool(suspended_draft.raw_values.get("console-settings-streaming"))
            if suspended_draft is not None
            and type(suspended_draft.raw_values.get("console-settings-streaming"))
            is bool
            else streaming_field.profile_override
            if streaming_field is not None
            and (
                streaming_field.profile_override is None
                or type(streaming_field.profile_override) is bool
            )
            else bool(self._settings.streaming)
        )
        self._streaming_effective_fallback = bool(self._settings.streaming)
        # Compose-time Input/Select echoes must not convert transferred draft
        # state into a new user edit. The guard is released after the first
        # refresh, once every mounted control reflects the transferred state.
        self._updating_controls = True
        self._initial_feedback_sync = True
        self._edited_generation_fields: set[str] = set()
        self._submit_pending = False
        self._rebase_event_guard = False
        snapshot_provider = (
            suspended_draft.raw_values.get("console-settings-provider")
            if suspended_draft is not None
            else None
        )
        self._active_provider = (
            snapshot_provider
            if _snapshot_provider(snapshot_provider) is not None
            else self._settings.provider
        )
        self._provider_model_drafts: dict[str, str | None] = {}
        if suspended_draft is not None:
            self._provider_model_drafts.update(suspended_draft.provider_model_drafts)
        if self._active_provider not in self._provider_model_drafts:
            initial_model = normalize_console_model_value(settings.model)
            if initial_model is not None:
                self._provider_model_drafts[self._active_provider] = initial_model
        self._provider_base_url_drafts: dict[str, str] = {}
        if suspended_draft is not None:
            self._provider_base_url_drafts.update(
                suspended_draft.provider_base_url_drafts
            )
        initial_base_url = self._initial_base_url_for_provider(
            self._active_provider,
            self._settings.base_url,
        )
        if (
            initial_base_url
            and self._active_provider not in self._provider_base_url_drafts
        ):
            self._provider_base_url_drafts[self._active_provider] = initial_base_url
        self._endpoint_draft = self._draft.endpoint_draft or ConsoleEndpointDraft(
            value=initial_base_url or "",
            bound_provider_config_key=provider_config_key(self._active_provider),
            dirty=False,
            checked=False,
        )
        self._restoring_suspended_draft = False
        self._last_logical_focus_control_id = (
            suspended_draft.focus_control_id if suspended_draft is not None else None
        )
        self._readiness_debounce_timer: Timer | None = None
        self._settings_close_guard_mode: str | None = None
        self._settings_close_guard_focus: Widget | None = None
        self._compaction_wait_worker: Any | None = None
        self._compaction_provider_task: asyncio.Task[tuple[bool, str]] | None = None
        self._compaction_result_definitive = False

    @staticmethod
    def _context_state_with_draft_overrides(
        state: ConsoleContextControlState | None,
        *,
        settings: ConsoleSessionSettings,
        estimate: ConsoleSettingsContextEstimate,
        overrides: ConsoleContextPolicyOverrides,
    ) -> ConsoleContextControlState:
        """Apply transferred policy intent without dropping live context metadata."""

        if state is None:
            return build_console_context_control_state(
                settings=settings,
                estimate=estimate,
                overrides=overrides,
            )
        if state.overrides == overrides:
            return state
        inherited = state.inherited_policy
        inherited_overrides = ConsoleContextPolicyOverrides(
            budget_mode=inherited.budget_mode,
            custom_budget_tokens=inherited.custom_budget_tokens,
            compaction_mode=inherited.compaction_mode,
            compaction_representation=inherited.compaction_representation,
            trigger_ratio=inherited.trigger_ratio,
            target_ratio=inherited.target_ratio,
            summary_max_tokens=inherited.summary_max_tokens,
            failure_behavior=inherited.failure_behavior,
            carry_forward_mode=inherited.carry_forward_mode,
        )
        reserved = (
            state.response_max_tokens
            + int(state.safety_margin_tokens or 0)
            + int(state.request_overhead_tokens or 0)
        )
        effective_cap = (
            state.safe_input_ceiling_tokens + reserved
            if state.safe_input_ceiling_tokens is not None
            else None
        )
        provider_cap = (
            effective_cap
            if effective_cap is not None
            and effective_cap > 0
            and (
                state.model_window_tokens is None
                or effective_cap < state.model_window_tokens
            )
            else None
        )
        return build_console_context_control_state(
            settings=settings,
            estimate=estimate,
            overrides=overrides,
            global_overrides=inherited_overrides,
            active_memory=state.active_memory,
            conversation_tokens=state.conversation_tokens,
            request_overhead_tokens=state.request_overhead_tokens,
            provider_input_cap_tokens=provider_cap,
            safety_margin_tokens=state.safety_margin_tokens,
            busy=state.busy,
            status_message=state.status_message,
        )

    @staticmethod
    def _initial_full_draft(
        settings: ConsoleSessionSettings,
    ) -> ConsoleSettingsDraftState:
        """Build a full-field draft for transitional callers without a transfer."""

        fields = tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in sorted(FULL_MODEL_DEFAULT_FIELDS)
        )
        return ConsoleSettingsDraftState(
            settings=settings,
            context_policy_overrides=ConsoleContextPolicyOverrides(),
            field_drafts=fields,
            model_drafts=(),
            endpoint_draft=None,
        )

    def _draft_field(self, name: str) -> ConsoleSettingsFieldDraft | None:
        return next(
            (field for field in self._draft.field_drafts if field.name == name),
            None,
        )

    def _updated_field_draft(
        self,
        name: str,
        effective_value: object | None,
    ) -> ConsoleSettingsFieldDraft:
        """Merge one mounted value without erasing inherited/carried intent."""

        prior = self._draft_field(name)
        # Textual may deliver a model Select change before a previously queued
        # Input.Changed notification. Comparing the mounted effective value to
        # the immutable draft makes that edit observable during the snapshot,
        # while an untouched inherited value still compares equal and remains
        # sparse rather than becoming an explicit override.
        direct_edit = bool(
            name in self._edited_generation_fields
            or (prior is not None and effective_value != prior.effective_value)
        )
        profile_override = (
            self._streaming_draft
            if name == "streaming"
            else effective_value
            if direct_edit
            else prior.profile_override
            if prior is not None
            else None
        )
        changed = bool(
            direct_edit
            or (
                name == "streaming"
                and (prior is None or prior.profile_override != profile_override)
            )
        )
        dirty = bool((prior is not None and prior.dirty) or changed)
        if changed:
            provenance = (
                ConsoleSettingsFieldProvenance.INHERITED
                if profile_override is None
                else ConsoleSettingsFieldProvenance.EXPLICIT
            )
        elif prior is not None:
            provenance = prior.provenance
        else:
            provenance = ConsoleSettingsFieldProvenance.INHERITED
        return ConsoleSettingsFieldDraft(
            name=name,
            effective_value=effective_value,
            profile_override=profile_override,
            provenance=provenance,
            dirty=dirty,
        )

    def compose(self) -> ComposeResult:
        provider_options = self._provider_select_options()
        provider_values = {value for _, value in provider_options}
        provider_value = (
            self._active_provider
            if self._active_provider in provider_values
            else Select.NULL
        )
        selected_model = self._model_for_provider(self._active_provider)
        base_url = self._base_url_for_provider(self._active_provider)
        uses_base_url = self._provider_uses_base_url(self._active_provider)
        model_options = self._model_select_options(
            self._active_provider, selected_model
        )
        model_select_options = model_options or [
            ("No configured models", _NO_CONFIGURED_MODELS_VALUE)
        ]
        model_option_values = {value for _, value in model_options}
        model_select_value = (
            selected_model if selected_model in model_option_values else Select.NULL
        )
        has_model_options = bool(model_options)
        use_model_select = self._should_use_model_select(
            self._active_provider,
            selected_model,
            model_options,
        )
        readiness = build_console_settings_readiness(
            self._settings_for_active_provider(), app_config=self._app_config
        )

        with Vertical(id="console-settings-modal"):
            yield Static("Console Settings", classes="console-modal-header")
            with Horizontal(id="console-settings-view-tabs"):
                yield Button(
                    "Model and generation",
                    id="console-settings-view-model",
                    variant="primary" if self._active_view == "model" else "default",
                )
                yield Button(
                    "Context and memory",
                    id="console-settings-view-context",
                    variant="primary" if self._active_view == "context" else "default",
                )
            yield Static(
                self._readiness_copy(readiness),
                id="console-settings-readiness",
                classes="console-settings-modal-row",
                markup=False,
            )
            credential_action = Button(
                "Configure credential…",
                id="console-settings-configure-credential",
                tooltip="Open F9 Settings > Providers & Models to configure this API key",
            )
            credential_action.display = self._missing_credential_recovery_available(
                readiness
            )
            yield credential_action
            yield Static(
                self._scope_copy(),
                id="console-settings-scope",
                classes="console-settings-modal-row",
                markup=False,
            )
            error_summary = Static(
                "",
                id="console-settings-error",
                classes="console-settings-error console-settings-error-summary",
                markup=False,
            )
            error_summary.display = False
            yield error_summary

            body = ScrollableContainer(
                id="console-settings-body",
                classes="console-settings-body",
            )
            # The global `*:focus` outline peeks through the 1-row section
            # margins as stray "|" fragments when this scroll container takes
            # focus (it was the first focusable widget on open). Keyboard
            # scrolling still works through its bindings while a child input
            # is focused, so the container stays out of the focus chain.
            body.can_focus = False
            with body:
                with Vertical(
                    id="console-settings-provider-model-section",
                    classes=self._provider_model_section_classes(),
                ):
                    yield Static("Provider and model", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Provider")
                        yield Select(
                            provider_options,
                            value=provider_value,
                            allow_blank=True,
                            id="console-settings-provider",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Model")
                        yield ModelSearchPicker(
                            id="console-settings-model-picker",
                            provider_select_id="#console-settings-provider",
                            current_model=selected_model,
                            providers_models=self._providers_models,
                            show_custom_button=False,
                        )
                    legacy_model_row = Horizontal(
                        id="console-settings-model-legacy-adapter"
                    )
                    legacy_model_row.display = False
                    with legacy_model_row:
                        model_select = Select(
                            model_select_options,
                            value=model_select_value,
                            allow_blank=True,
                            id="console-settings-model-select",
                            disabled=not use_model_select,
                            classes="console-settings-control",
                        )
                        model_select.styles.width = "1fr"
                        model_select.styles.min_width = 0
                        model_select.display = use_model_select
                        yield model_select
                        model_input = ConsoleSettingsInput(
                            value=selected_model or "",
                            placeholder=MODEL_INPUT_PLACEHOLDER,
                            id="console-settings-model-input",
                            disabled=has_model_options,
                            classes="console-settings-control",
                        )
                        model_input.styles.width = "1fr"
                        model_input.styles.min_width = 0
                        model_input.display = not use_model_select
                        yield model_input
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("")
                        model_custom = Button(
                            "Custom model",
                            id="console-settings-model-custom",
                            disabled=False,
                        )
                        model_custom.styles.width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.styles.min_width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.styles.max_width = MODEL_CUSTOM_BUTTON_WIDTH
                        model_custom.display = True
                        yield model_custom
                        supports_discovery = self._provider_supports_model_discovery(
                            self._active_provider
                        )
                        model_discover = Button(
                            MODEL_DISCOVER_BUTTON_LABEL,
                            id=MODEL_DISCOVER_BUTTON_ID,
                            disabled=not supports_discovery,
                        )
                        model_discover.tooltip = (
                            "List models served at the Base URL (/v1/models)"
                        )
                        model_discover.styles.width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.styles.min_width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.styles.max_width = MODEL_DISCOVER_BUTTON_WIDTH
                        model_discover.display = supports_discovery
                        yield model_discover
                    # Directly under the button that produces it. Rendering this
                    # below the Base URL row put four rows and an unrelated field
                    # between action and feedback, which read as a dead button.
                    discover_status = Static(
                        "",
                        id=MODEL_DISCOVER_STATUS_ID,
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    discover_status.display = False
                    yield discover_status
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Base URL")
                        base_url_input = ConsoleSettingsInput(
                            value=base_url or "",
                            id="console-settings-base-url",
                            disabled=not uses_base_url,
                            classes="console-settings-control",
                        )
                        base_url_input.display = uses_base_url
                        yield base_url_input
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("New-chat default")
                        endpoint_checkbox = Checkbox(
                            self._endpoint_save_copy(),
                            value=self._endpoint_checkbox_value(),
                            id="console-settings-save-endpoint",
                            disabled=not self._endpoint_can_be_checked(),
                        )
                        endpoint_checkbox.display = uses_base_url
                        yield endpoint_checkbox
                    endpoint_status = Static(
                        self._endpoint_status_copy(),
                        id="console-settings-endpoint-status",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    endpoint_status.display = uses_base_url
                    yield endpoint_status

                with Vertical(
                    classes="console-settings-modal-section console-settings-model-view"
                ):
                    yield Static("Chat identity", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Your name in this chat")
                        yield ConsoleSettingsInput(
                            value=self._user_display_name_override or "",
                            placeholder=self._global_user_display_name,
                            id="console-settings-user-display-name",
                            classes="console-settings-control",
                        )
                    yield Static(
                        "Leave blank to use the global default.",
                        id="console-settings-user-display-name-help",
                        classes="console-settings-modal-row",
                        markup=False,
                    )

                with Vertical(
                    classes="console-settings-modal-section console-settings-model-view"
                ):
                    yield Static("Sampling", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Temperature")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.temperature),
                            id="console-settings-temperature",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Top P")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.top_p),
                            id="console-settings-top-p",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Min P")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.min_p),
                            id="console-settings-min-p",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Top K")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.top_k),
                            id="console-settings-top-k",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Response max tokens")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.max_tokens),
                            id="console-settings-max-tokens",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Seed")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.seed),
                            id="console-settings-seed",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Presence")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.presence_penalty),
                            id="console-settings-presence-penalty",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Frequency")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.frequency_penalty),
                            id="console-settings-frequency-penalty",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Streaming")
                        streaming_toggle = Button(
                            self._streaming_toggle_label(),
                            id="console-settings-streaming",
                        )
                        streaming_toggle.tooltip = (
                            "Toggle streaming on or off for this session"
                        )
                        streaming_toggle.styles.width = STREAMING_TOGGLE_WIDTH
                        streaming_toggle.styles.min_width = STREAMING_TOGGLE_WIDTH
                        streaming_toggle.styles.max_width = STREAMING_TOGGLE_WIDTH
                        yield streaming_toggle

                with Vertical(
                    classes="console-settings-modal-section console-settings-model-view"
                ):
                    yield Static("Provider-specific", classes="destination-section")
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Reasoning")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.reasoning_effort),
                            placeholder=self._choice_placeholder(
                                "console-settings-reasoning-effort"
                            ),
                            id="console-settings-reasoning-effort",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Summary")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.reasoning_summary),
                            placeholder=self._choice_placeholder(
                                "console-settings-reasoning-summary"
                            ),
                            id="console-settings-reasoning-summary",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Verbosity")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.verbosity),
                            placeholder=self._choice_placeholder(
                                "console-settings-verbosity"
                            ),
                            id="console-settings-verbosity",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Thinking")
                        yield ConsoleSettingsInput(
                            value=self._format_value(self._settings.thinking_effort),
                            placeholder=self._choice_placeholder(
                                "console-settings-thinking-effort"
                            ),
                            id="console-settings-thinking-effort",
                            classes="console-settings-control",
                        )
                    with Horizontal(classes="console-settings-modal-row"):
                        yield self._modal_label("Budget")
                        yield ConsoleSettingsInput(
                            value=self._format_value(
                                self._settings.thinking_budget_tokens
                            ),
                            id="console-settings-thinking-budget-tokens",
                            classes="console-settings-control",
                        )

                with Vertical(
                    classes="console-settings-modal-section console-settings-model-view"
                ):
                    yield Static("Request preview", classes="destination-section")
                    yield Static(
                        f"Current         {self._context_label()}",
                        id="console-settings-context-current",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        f"Sources         {self._sources_label()}",
                        id="console-settings-context-sources",
                        classes="console-settings-modal-row",
                        markup=False,
                    )
                    yield Static(
                        "Estimate only; no truncation changes in this version. "
                        "Open Context and memory to manage the conversation budget.",
                        id="console-settings-context-note",
                        classes="console-settings-modal-row",
                        markup=False,
                    )

                with Vertical(
                    classes="console-settings-modal-section console-settings-model-view"
                ):
                    yield Static("Identity", classes="destination-section")
                    yield Static(
                        f"Current         {self._identity_current_label()}",
                        id="console-settings-identity-current",
                        classes="console-settings-modal-row",
                        markup=False,
                    )

                context_view = Vertical(
                    id="console-settings-context-view",
                    classes="console-settings-context-view",
                )
                # The body scroll owner can only advertise hidden content when
                # this nested view reports its full intrinsic height. A 1fr
                # default collapsed the view to the five-row viewport and made
                # lower sections both clipped and invisible to max_scroll_y.
                context_view.styles.height = "auto"
                with context_view:
                    with Vertical(classes="console-settings-modal-section"):
                        yield Static("Model capacity", classes="destination-section")
                        yield Static(
                            f"{self._model_window_label():<20}"
                            f"{format_context_tokens(self._context_state.model_window_tokens)} tokens",
                            id="console-context-model-window",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            "Response max tokens "
                            f"{format_context_tokens(self._context_state.response_max_tokens)} tokens",
                            id="console-context-response-max",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            "Safety margin       "
                            f"{format_context_tokens(self._context_state.safety_margin_tokens)} tokens",
                            id="console-context-safety-margin",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            "Safe input ceiling  "
                            f"{format_context_tokens(self._context_state.safe_input_ceiling_tokens)} tokens",
                            id="console-context-safe-input",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            self._context_validation_label(),
                            id="console-context-capacity-status",
                            classes="console-settings-modal-row",
                            markup=False,
                        )

                    with Vertical(classes="console-settings-modal-section"):
                        yield Static(
                            "Conversation budget", classes="destination-section"
                        )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Budget mode")
                            yield Select(
                                [
                                    ("Automatic", ContextBudgetMode.AUTOMATIC.value),
                                    ("Custom", ContextBudgetMode.CUSTOM.value),
                                ],
                                value=self._context_state.resolved_policy.policy.budget_mode.value,
                                id="console-context-budget-mode",
                                classes="console-settings-control",
                            )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Conversation max tokens")
                            yield ConsoleSettingsInput(
                                value=self._format_value(
                                    self._context_state.resolved_policy.policy.custom_budget_tokens
                                ),
                                placeholder="Required in Custom mode",
                                id="console-context-custom-budget",
                                classes="console-settings-control",
                            )
                        yield Static(
                            "Effective           "
                            f"{format_context_tokens(self._context_state.conversation_budget_tokens)} tokens",
                            id="console-context-effective-budget",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            "Next request        "
                            f"{format_context_tokens(self._context_state.request_tokens)} tokens",
                            id="console-context-next-request",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            "Request overhead    "
                            f"{format_context_tokens(self._context_state.request_overhead_tokens)} tokens",
                            id="console-context-overhead",
                            classes="console-settings-modal-row",
                            markup=False,
                        )

                    with Vertical(classes="console-settings-modal-section"):
                        yield Static("Compaction", classes="destination-section")
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Behavior")
                            yield Select(
                                [
                                    ("Ask", ContextCompactionMode.ASK.value),
                                    (
                                        "Automatic",
                                        ContextCompactionMode.AUTOMATIC.value,
                                    ),
                                    ("Off", ContextCompactionMode.OFF.value),
                                ],
                                value=self._context_state.resolved_policy.policy.compaction_mode.value,
                                id="console-context-compaction-mode",
                                classes="console-settings-control",
                            )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Representation")
                            yield Select(
                                [
                                    (
                                        "Text summary",
                                        ContextCompactionRepresentation.TEXT_SUMMARY.value,
                                    ),
                                    (
                                        "Visual transcript",
                                        ContextCompactionRepresentation.VISUAL_TRANSCRIPT.value,
                                    ),
                                    (
                                        "Hybrid",
                                        ContextCompactionRepresentation.HYBRID.value,
                                    ),
                                ],
                                value=self._context_state.resolved_policy.policy.compaction_representation.value,
                                id="console-context-compaction-representation",
                                classes="console-settings-control",
                                allow_blank=False,
                            )
                        yield Static(
                            "",
                            id="console-context-representation-status",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Compact at (%)")
                            yield ConsoleSettingsInput(
                                value=self._format_percent(
                                    self._context_state.resolved_policy.policy.trigger_ratio
                                ),
                                id="console-context-trigger-percent",
                                classes="console-settings-control",
                            )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Reduce conversation to (%)")
                            yield ConsoleSettingsInput(
                                value=self._format_percent(
                                    self._context_state.resolved_policy.policy.target_ratio
                                ),
                                id="console-context-target-percent",
                                classes="console-settings-control",
                            )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Summary response max")
                            yield ConsoleSettingsInput(
                                value=str(
                                    self._context_state.resolved_policy.policy.summary_max_tokens
                                ),
                                id="console-context-summary-max",
                                classes="console-settings-control",
                            )
                        yield Static(
                            "Summary response max applies only to Text summary and Hybrid.",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("If compaction fails")
                            yield Select(
                                [
                                    (
                                        "Stop and ask",
                                        CompactionFailureBehavior.STOP_AND_ASK.value,
                                    ),
                                    (
                                        "Omit older context",
                                        CompactionFailureBehavior.OMIT_OLDER_CONTEXT.value,
                                    ),
                                ],
                                value=self._context_state.resolved_policy.policy.failure_behavior.value,
                                id="console-context-failure-behavior",
                                classes="console-settings-control",
                            )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("Keep after compaction")
                            yield Select(
                                [
                                    (
                                        "Memory with recent turns",
                                        ContextCarryForwardMode.MEMORY_WITH_RECENT_TURNS.value,
                                    ),
                                    (
                                        "Memory with latest exchange",
                                        ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE.value,
                                    ),
                                ],
                                value=self._context_state.resolved_policy.policy.carry_forward_mode.value,
                                id="console-context-carry-forward",
                                classes="console-settings-control",
                            )
                        yield Static(
                            self._context_policy_provenance_label(),
                            id="console-context-policy-provenance",
                            classes="console-settings-modal-row",
                            markup=False,
                        )

                    with Vertical(classes="console-settings-modal-section"):
                        yield Static(
                            "Thinking history replay", classes="destination-section"
                        )
                        with Horizontal(classes="console-settings-modal-row"):
                            yield self._modal_label("History policy")
                            yield Select(
                                [
                                    ("Auto", "auto"),
                                    ("Include", "include"),
                                    ("Exclude", "exclude"),
                                ],
                                value=self._context_state.thinking_history.saved_policy,
                                id="console-context-thinking-history-policy",
                                classes="console-settings-control",
                                allow_blank=False,
                                disabled=(
                                    self._context_state.thinking_history.effective_label
                                    == "Required"
                                ),
                            )
                        yield Static(
                            self._thinking_history_effective_copy(),
                            id="console-context-thinking-history-effective",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Button(
                            "Save as default for new conversations",
                            id="console-context-thinking-history-save-default",
                            disabled=not self._can_save,
                        )

                    with Vertical(classes="console-settings-modal-section"):
                        yield Static("Current memory", classes="destination-section")
                        yield Static(
                            self._memory_metadata_label(),
                            id="console-context-memory-metadata",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        yield Static(
                            self._memory_review_text(),
                            id="console-settings-memory-review",
                            markup=False,
                        )
                        yield Static(
                            "",
                            id="console-context-action-status",
                            classes="console-settings-modal-row",
                            markup=False,
                        )
                        with Horizontal(classes="console-context-action-row"):
                            yield Button(
                                "Compact now",
                                id="console-context-compact-now",
                                disabled=(
                                    self._context_state.busy
                                    or self._compact_now is None
                                ),
                            )
                            yield Button(
                                "Reset current branch memory",
                                id="console-context-reset-current",
                                disabled=(
                                    self._context_state.effective_memory.kind
                                    is EffectiveMemoryKind.RAW
                                    or self._reset_current_memory is None
                                ),
                            )
                            undo = Button(
                                "Undo reset",
                                id="console-context-undo-reset",
                                disabled=True,
                            )
                            undo.display = False
                            yield undo
                        with Horizontal(classes="console-context-action-row"):
                            yield Button(
                                "Reset overrides",
                                id="console-context-reset-overrides",
                            )
                            yield Button(
                                "Reset all conversation memory…",
                                id="console-context-reset-all",
                                disabled=self._reset_all_memories is None,
                            )
                            confirm = Button(
                                "Confirm reset all branches",
                                id="console-context-confirm-reset-all",
                                variant="error",
                            )
                            confirm.display = False
                            yield confirm

                new_chat_block = Static(
                    "",
                    id="console-settings-new-chat-default-block",
                    markup=False,
                )
                new_chat_block.display = False
                yield new_chat_block
                recovery = Vertical(id="console-settings-default-recovery")
                recovery.display = False
                with recovery:
                    yield Static(
                        "",
                        id="console-settings-default-recovery-summary",
                        markup=False,
                    )

            fold_hint = Static(
                "▼ more — scroll for the rest",
                id="console-settings-fold-hint",
                markup=False,
            )
            fold_hint.display = False
            yield fold_hint
            recovery_actions = Horizontal(
                id="console-settings-default-recovery-actions"
            )
            recovery_actions.display = False
            with recovery_actions:
                yield Button(
                    "Retry default save",
                    id="console-settings-default-retry",
                )
                yield Button(
                    "Discard retry",
                    id="console-settings-default-discard",
                )
                yield Button(
                    "Refresh running app",
                    id="console-settings-default-refresh",
                )
                yield Button(
                    "Dismiss",
                    id="console-settings-default-dismiss",
                )
            with Vertical(
                id="console-settings-actions",
                classes="console-settings-modal-row console-settings-modal-actions",
            ):
                with Horizontal(classes="console-settings-action-group"):
                    yield Button("Cancel", id="console-settings-cancel")
                    save_default = Button(
                        "Save as model default",
                        id="console-settings-save-default",
                        disabled=not self._can_save,
                    )
                    save_default.tooltip = (
                        "Apply to this chat and save the shown generation profile "
                        "for this exact provider and model."
                    )
                    save_default.styles.width = 24
                    save_default.styles.min_width = 24
                    yield save_default
                with Horizontal(classes="console-settings-action-group"):
                    make_default = Button(
                        "Make default for new chats",
                        id="console-settings-make-default",
                        disabled=not self._can_save,
                    )
                    make_default.styles.width = 28
                    make_default.styles.min_width = 28
                    yield make_default
                    apply = Button(
                        "Apply to this chat",
                        id="console-settings-save",
                        variant="primary",
                        disabled=not self._can_save,
                    )
                    apply.styles.width = 20
                    apply.styles.min_width = 20
                    yield apply
            guard = Vertical(
                Static("", id="console-settings-close-message", markup=False),
                Horizontal(
                    Button(
                        "Undo and close",
                        id="console-settings-close-undo",
                    ),
                    Button(
                        "Keep reset and close",
                        id="console-settings-close-keep",
                    ),
                    Button(
                        "Close anyway",
                        id="console-settings-close-anyway",
                    ),
                    Button("Return", id="console-settings-close-return"),
                    id="console-settings-close-actions",
                ),
                id="console-settings-close-guard",
            )
            guard.display = False
            yield guard

    def on_mount(self) -> None:
        self._sync_action_layout(self.size.width)
        self._show_settings_view(self._active_view)
        self._sync_endpoint_controls()
        self._sync_default_recovery_region()
        self._sync_default_readiness()
        self.call_after_refresh(self._reveal_default_feedback)
        self.call_after_refresh(self._finish_initial_control_sync)
        if self._suspended_draft is not None:
            self._restore_suspended_draft(self._suspended_draft)
        elif self._default_recovery_layout_phase is None:
            if self._focus_model:
                self._focus_model_control()
            elif self._active_view == "context":
                self._sync_visual_representation_availability()
                self.call_after_refresh(self._focus_context_control)
        self.call_after_refresh(self._sync_fold_hint)

    def _finish_initial_control_sync(self) -> None:
        """Allow mounted control events after transferred state is projected."""

        self._updating_controls = False
        self.call_after_refresh(self._finish_initial_feedback_sync)

    def _finish_initial_feedback_sync(self) -> None:
        """Allow feedback reveals after the initial mount/resize callback batch."""

        self._initial_feedback_sync = False

    def _restore_suspended_draft(
        self, snapshot: ConsoleSettingsDraftSnapshot
    ) -> None:
        """Rehydrate raw, possibly invalid modal values after composition."""
        self._restoring_suspended_draft = True
        try:
            for control_id, value in snapshot.raw_values.items():
                if control_id == "console-settings-streaming":
                    continue
                if control_id == "console-settings-model-picker":
                    self.query_one(f"#{control_id}", ModelSearchPicker).set_model_value(
                        value if type(value) is str else None
                    )
                    continue
                try:
                    control = self.query_one(f"#{control_id}")
                except NoMatches:
                    continue
                if isinstance(control, Input) and type(value) is str:
                    control.value = value
                elif isinstance(control, Select) and type(value) is str:
                    control.value = value
            self._active_provider = self._select_value_text(
                self.query_one("#console-settings-provider", Select).value
            )
            self._streaming_draft = bool(
                snapshot.raw_values.get(
                    "console-settings-streaming", self._streaming_draft
                )
            )
            self.query_one("#console-settings-streaming", Button).label = (
                self._streaming_toggle_label()
            )
        finally:
            self._restoring_suspended_draft = False
        self._synchronize_restored_provider_state()
        self.call_after_refresh(self._restore_suspended_scroll_and_focus, snapshot)

    def _restore_suspended_scroll_and_focus(
        self, snapshot: ConsoleSettingsDraftSnapshot
    ) -> None:
        """Restore only a current logical focus target and the saved body anchor."""
        body = self.query_one("#console-settings-body", ScrollableContainer)
        body.scroll_to(y=snapshot.scroll_anchor, animate=False)
        control_id = snapshot.focus_control_id
        if control_id is None:
            self._focus_connection_fallback()
            return
        if control_id == "console-settings-model-picker":
            try:
                picker = self.query_one(f"#{control_id}", ModelSearchPicker)
                picker_input = picker.query_one("#model-search-picker-input", Input)
                if self._is_effectively_focusable(picker_input):
                    picker.focus_input()
                    return
            except (NoMatches, QueryError):
                pass
        else:
            try:
                control = self.query_one(f"#{control_id}")
                if self._is_effectively_focusable(control):
                    control.focus()
                    return
            except (NoMatches, QueryError):
                pass
        self._focus_connection_fallback()

    def _focus_connection_fallback(self) -> None:
        """Focus a live Connection control when an old logical target is unavailable."""
        self._show_settings_view("model")
        for selector, widget_type in (
            ("#console-settings-provider", Select),
            ("#console-settings-model-picker", ModelSearchPicker),
        ):
            try:
                control = self.query_one(selector, widget_type)
            except (NoMatches, QueryError):
                continue
            if isinstance(control, ModelSearchPicker):
                picker_input = control.query_one("#model-search-picker-input", Input)
                if not self._is_effectively_focusable(picker_input):
                    continue
                control.focus_input()
            elif self._is_effectively_focusable(control):
                control.focus()
            else:
                continue
            return

    @staticmethod
    def _is_effectively_focusable(control: Widget) -> bool:
        """Check the real target plus every mounted ancestor for focusability."""
        if not control.focusable:
            return False
        current: Widget | None = control
        while current is not None:
            if not current.display or current.disabled:
                return False
            parent = current.parent
            current = parent if isinstance(parent, Widget) else None
        return True

    def _settings_for_active_provider(self) -> ConsoleSessionSettings:
        """Project the current provider's raw model/endpoint into readiness only."""
        return replace(
            self._settings,
            provider=self._active_provider,
            model=self._model_for_provider(self._active_provider),
            base_url=self._base_url_for_provider(self._active_provider),
        )

    def _missing_credential_recovery_available(
        self, readiness: ConsoleSettingsReadiness | None = None
    ) -> bool:
        """Whether canonical Settings owns the current highest-priority blocker."""
        if readiness is None:
            readiness = build_console_settings_readiness(
                self._settings_for_active_provider(), app_config=self._app_config
            )
        return readiness.recovery_action == "configure_credential"

    def _synchronize_restored_provider_state(self) -> None:
        """Refresh dependent controls exactly once after suppressed rehydration."""
        provider = self._active_provider
        self._sync_model_controls(provider, self._model_for_provider(provider))
        self._sync_base_url_control(provider, self._base_url_for_provider(provider))
        self._sync_model_discover_controls(provider)
        self._sync_provider_choice_placeholders()
        self._sync_readiness_display()
        self._sync_visual_representation_availability()

    def capture_suspended_draft(self) -> ConsoleSettingsDraftSnapshot:
        """Capture the raw modal state before a credential-setup handoff."""
        self._store_current_model_for_provider(self._active_provider)
        self._store_current_base_url_for_provider(self._active_provider)
        try:
            context_overrides = self._build_context_policy_overrides()
        except (ContextPolicyError, ValueError):
            # Raw values below remain exact while semantic state stays on the
            # most recent complete policy the modal could validate.
            context_overrides = self._context_state.overrides
        else:
            self._context_state = self._context_state_with_overrides(
                context_overrides
            )
        raw_values: dict[str, str | bool] = {}
        for control_id in _SNAPSHOT_RAW_VALUE_IDS:
            if control_id == "console-settings-streaming":
                raw_values[control_id] = self._streaming_draft
                continue
            try:
                control = self.query_one(f"#{control_id}")
            except NoMatches:
                continue
            if control_id == "console-settings-model-picker":
                raw_values[control_id] = self._select_value_text(
                    getattr(control, "value", None)
                )
            elif isinstance(control, Input):
                raw_values[control_id] = control.value
            elif isinstance(control, Select):
                raw_values[control_id] = self._select_value_text(control.value)
        try:
            scroll_anchor = min(
                _SNAPSHOT_SCROLL_ANCHOR_LIMIT,
                max(
                    0,
                    int(
                        self.query_one(
                            "#console-settings-body", ScrollableContainer
                        ).scroll_y
                    ),
                ),
            )
        except (NoMatches, TypeError, ValueError):
            scroll_anchor = 0
        return ConsoleSettingsDraftSnapshot(
            settings=self._settings,
            context_policy_overrides=context_overrides,
            raw_values=raw_values,
            provider_model_drafts=dict(self._provider_model_drafts),
            provider_base_url_drafts=dict(self._provider_base_url_drafts),
            active_view=self._active_view,
            scroll_anchor=scroll_anchor,
            focus_control_id=(
                self._logical_focus_control_id()
                or self._last_logical_focus_control_id
            ),
            disclosure_state={
                "advanced_generation": bool(
                    getattr(self, "_advanced_generation_disclosed", False)
                ),
                "connection_details": bool(
                    getattr(self, "_connection_details_disclosed", False)
                ),
            },
        )

    def _context_state_with_overrides(
        self,
        overrides: ConsoleContextPolicyOverrides,
    ) -> ConsoleContextControlState:
        """Rebuild semantic context state while retaining live display inputs."""
        current = self._context_state
        inherited = current.inherited_policy
        inherited_overrides = ConsoleContextPolicyOverrides(
            budget_mode=inherited.budget_mode,
            custom_budget_tokens=inherited.custom_budget_tokens,
            compaction_mode=inherited.compaction_mode,
            compaction_representation=inherited.compaction_representation,
            trigger_ratio=inherited.trigger_ratio,
            target_ratio=inherited.target_ratio,
            summary_max_tokens=inherited.summary_max_tokens,
            failure_behavior=inherited.failure_behavior,
            carry_forward_mode=inherited.carry_forward_mode,
        )
        context_ceiling = None
        if current.model_window_tokens is not None:
            context_ceiling = (
                current.model_window_tokens
                - current.response_max_tokens
                - int(current.safety_margin_tokens or 0)
            )
        provider_input_cap = (
            current.safe_input_ceiling_tokens
            if current.safe_input_ceiling_tokens is not None
            and (
                context_ceiling is None
                or current.safe_input_ceiling_tokens < context_ceiling
            )
            else None
        )
        return build_console_context_control_state(
            settings=self._settings,
            estimate=self._context_estimate,
            overrides=overrides,
            global_overrides=inherited_overrides,
            active_memory=current.active_memory,
            conversation_tokens=current.conversation_tokens,
            request_overhead_tokens=current.request_overhead_tokens,
            provider_input_cap_tokens=provider_input_cap,
            safety_margin_tokens=current.safety_margin_tokens,
            busy=current.busy,
            status_message=current.status_message,
        )

    def _logical_focus_control_id(self, focused: Widget | None = None) -> str | None:
        """Return the public focus target, never a hidden legacy alias."""
        current = focused or self.focused or getattr(self.app, "focused", None)
        while current is not None:
            control_id = getattr(current, "id", None)
            if control_id in _SNAPSHOT_FOCUS_CONTROL_IDS:
                return control_id
            if control_id == "model-search-picker-input":
                return "console-settings-model-picker"
            current = getattr(current, "parent", None)
        return None

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Remember the last draft control before an action button takes focus."""
        logical_focus = self._logical_focus_control_id(event.widget)
        if logical_focus is not None:
            self._last_logical_focus_control_id = logical_focus

    def credential_request(self) -> ConsoleSettingsCredentialRequest:
        """Build the modal result consumed by Console's credential recovery route."""
        return ConsoleSettingsCredentialRequest(
            snapshot=self.capture_suspended_draft(),
            provider=self._active_provider,
            model=self._current_model_value(),
        )

    def on_resize(self, event: events.Resize) -> None:
        """Recompute the body fold affordance after viewport changes."""
        self._sync_action_layout(event.size.width)
        self.call_after_refresh(self._sync_fold_hint)
        self.call_after_refresh(self._reveal_default_feedback)

    def _sync_action_layout(self, viewport_width: int) -> None:
        """Keep every footer action reachable at supported terminal widths."""

        compact = viewport_width < 100
        recovery_active = (
            self._default_durability_state.recovery_intent is not None
            and self._default_durability_state.failure_phase is not None
        )
        view_tabs = self.query_one("#console-settings-view-tabs", Horizontal)
        view_height = 1 if compact else MODAL_CONTROL_HEIGHT
        view_tabs.styles.height = view_height
        view_tabs.styles.min_height = view_height
        for button in view_tabs.query(Button):
            button.styles.height = view_height
            button.styles.min_height = view_height
        for selector in ("#console-settings-readiness", "#console-settings-scope"):
            summary = self.query_one(selector, Static)
            summary.styles.height = "auto"
            summary.styles.min_height = 1 if compact else MODAL_CONTROL_HEIGHT
        recovery_actions = self.query_one(
            "#console-settings-default-recovery-actions", Horizontal
        )
        recovery_actions.styles.height = 1 if compact else MODAL_CONTROL_HEIGHT
        recovery_actions.styles.min_height = 1 if compact else MODAL_CONTROL_HEIGHT
        for button in recovery_actions.query(Button):
            button.styles.height = 1 if compact else MODAL_CONTROL_HEIGHT
            button.styles.min_height = 1 if compact else MODAL_CONTROL_HEIGHT
        actions = self.query_one("#console-settings-actions", Vertical)
        actions.styles.layout = "vertical" if compact else "horizontal"
        action_height = 1 if recovery_active or not compact else 2
        actions.styles.height = action_height
        actions.styles.min_height = action_height
        action_groups = list(actions.query(".console-settings-action-group"))
        for group in action_groups:
            group.styles.width = "100%" if compact else "auto"
            group.styles.height = 1
            group.styles.min_height = 1
            group.styles.align_horizontal = "right"
        if len(action_groups) > 1:
            action_groups[1].display = not recovery_active
        for button in actions.query(Button):
            button.styles.height = 1
            button.styles.min_height = 1

    def _show_settings_view(self, view: str) -> None:
        """Switch between the two stable in-modal destinations."""
        self._active_view = "context" if view == "context" else "model"
        recovery_active = (
            self._default_durability_state.recovery_intent is not None
            and self._default_durability_state.failure_phase is not None
        )
        show_model = self._active_view == "model" and not recovery_active
        for section in self.query(".console-settings-model-view"):
            section.display = show_model
        self.query_one("#console-settings-context-view", Vertical).display = (
            self._active_view == "context" and not recovery_active
        )
        self.query_one("#console-settings-view-model", Button).variant = (
            "primary" if show_model else "default"
        )
        self.query_one("#console-settings-view-context", Button).variant = (
            "default" if show_model else "primary"
        )
        self.query_one("#console-settings-scope", Static).update(self._scope_copy())
        self.query_one("#console-settings-save-default", Button).display = show_model
        self.query_one("#console-settings-make-default", Button).display = show_model
        self.query_one("#console-settings-save", Button).display = not recovery_active
        readiness = self.query_one("#console-settings-new-chat-default-block", Static)
        readiness.display = show_model and bool(str(readiness.renderable))
        self.call_after_refresh(self._sync_fold_hint)
        self.call_after_refresh(self._reveal_default_feedback)

    def _scope_copy(self) -> str:
        """Return save-scope copy for the active modal destination."""
        if self._active_view == "context":
            return CONSOLE_SETTINGS_CONTEXT_SCOPE_COPY
        return CONSOLE_SETTINGS_MODEL_SCOPE_COPY

    def _sync_default_readiness(self) -> None:
        """Gate default actions on an exact model and future-chat readiness."""

        try:
            save_button = self.query_one("#console-settings-save-default", Button)
            make_button = self.query_one("#console-settings-make-default", Button)
            block = self.query_one("#console-settings-new-chat-default-block", Static)
        except (NoMatches, QueryError):
            return
        settings = self._build_draft()
        if self._endpoint_is_authorized(self._endpoint_draft, settings.provider):
            readiness = build_console_default_readiness_preview(
                self._app_config,
                settings,
                self._endpoint_draft,
            )
        elif self._default_readiness_resolver is not None:
            readiness = self._default_readiness_resolver(
                settings.provider, settings.model
            )
        else:
            readiness = build_console_settings_readiness(
                settings,
                app_config=self._app_config,
            )
        if not settings.model:
            copy = "Unavailable: choose a model first."
        elif not readiness.native_send_supported:
            copy = f"Unavailable: {readiness.detail}"
        else:
            copy = ""
        recovery_active = (
            self._default_durability_state.recovery_intent is not None
            and self._default_durability_state.failure_phase is not None
        )
        save_button.disabled = not settings.model or not self._can_save
        make_button.disabled = bool(copy) or not self._can_save
        block.update(copy)
        block.display = (
            bool(copy) and self._active_view == "model" and not recovery_active
        )
        if block.display:
            self.call_after_refresh(self._sync_fold_hint)
            self.call_after_refresh(self._reveal_default_feedback)

    def _default_recovery_summary(self) -> str:
        state = self._default_durability_state
        intent = state.recovery_intent
        phase = state.failure_phase
        if intent is None or phase is None:
            return ""
        action = (
            "Make default for new chats"
            if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
            else "Save as model default"
        )
        fields = ", ".join(sorted(intent.field_mask))
        endpoint = ""
        if intent.endpoint_patch is not None:
            preview = format_console_endpoint_preview(intent.endpoint_patch.value)
            if preview is not None:
                endpoint = f" · connection {preview}"
        status = (
            "Not written to disk"
            if phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
            else "Saved on disk; running app refresh failed"
        )
        return (
            f"{status}\n{action}: {intent.provider_config_key}/"
            f"{intent.literal_model_id} · fields: {fields}{endpoint}"
        )

    def _sync_default_recovery_region(self) -> None:
        """Render the current app-owned recovery snapshot without consuming it."""

        try:
            region = self.query_one("#console-settings-default-recovery", Vertical)
            actions = self.query_one(
                "#console-settings-default-recovery-actions", Horizontal
            )
            summary = self.query_one(
                "#console-settings-default-recovery-summary", Static
            )
        except (NoMatches, QueryError):
            return
        state = self._default_durability_state
        phase = state.failure_phase
        visible = state.recovery_intent is not None and phase is not None
        region.display = visible
        actions.display = visible
        summary.update(self._default_recovery_summary())
        self._sync_default_recovery_layout(visible)
        if not visible:
            return
        before_replace = phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
        pending = self._default_recovery_pending is not None
        retry = self.query_one("#console-settings-default-retry", Button)
        discard = self.query_one("#console-settings-default-discard", Button)
        refresh = self.query_one("#console-settings-default-refresh", Button)
        dismiss = self.query_one("#console-settings-default-dismiss", Button)
        retry.display = before_replace
        discard.display = before_replace
        refresh.display = not before_replace
        dismiss.display = not before_replace
        for button in (retry, discard, refresh, dismiss):
            button.disabled = pending
        self.call_after_refresh(self._reveal_default_feedback)

    def _sync_default_recovery_layout(self, visible: bool) -> None:
        """Give the exact recovery summary the bounded body at narrow heights."""

        previous_phase = self._default_recovery_layout_phase
        current_phase = (
            self._default_durability_state.failure_phase if visible else None
        )
        self._default_recovery_layout_phase = current_phase
        self._sync_action_layout(self.size.width)
        for selector in (
            "#console-settings-view-tabs",
            "#console-settings-readiness",
            "#console-settings-scope",
        ):
            self.query_one(selector).display = not visible
        if visible:
            self.query_one(
                "#console-settings-new-chat-default-block", Static
            ).display = False
            for section in self.query(".console-settings-model-view"):
                section.display = False
            self.query_one("#console-settings-context-view", Vertical).display = False
            if current_phase is not previous_phase:
                self.call_after_refresh(self._scroll_default_recovery_summary_home)
                self.call_after_refresh(self._focus_default_recovery_action)
            self.call_after_refresh(self._sync_fold_hint)
            return
        self._show_settings_view(self._active_view)
        if previous_phase is not None:
            self.call_after_refresh(self._restore_focus_after_default_recovery)
        self.call_after_refresh(self._sync_fold_hint)

    def _scroll_default_recovery_summary_home(self) -> None:
        """Start a newly visible recovery at the beginning of its exact summary."""

        if not self.is_mounted or self._default_recovery_layout_phase is None:
            return
        try:
            body = self.query_one("#console-settings-body", ScrollableContainer)
        except (NoMatches, QueryError):
            return
        body.scroll_to(y=0, animate=False, immediate=True, force=True)

    def _focus_default_recovery_action(self) -> None:
        """Focus the first action valid for the current recovery phase."""

        if not self.is_mounted:
            return
        selector = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: ("#console-settings-default-retry"),
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: (
                "#console-settings-default-refresh"
            ),
        }.get(self._default_recovery_layout_phase)
        if selector is None:
            return
        try:
            button = self.query_one(selector, Button)
        except (NoMatches, QueryError):
            return
        if button.display and not button.disabled:
            button.focus()

    def _restore_focus_after_default_recovery(self) -> None:
        """Return focus to the visible editor after recovery succeeds."""

        if not self.is_mounted or self._default_recovery_layout_phase is not None:
            return
        try:
            if self._active_view == "context":
                self._focus_context_control()
            else:
                self._focus_model_control()
        except (NoMatches, QueryError):
            return

    def _reveal_default_feedback(self) -> None:
        """Reveal the applicable dynamic status inside the bounded modal body."""

        if not self.is_mounted or self._initial_feedback_sync:
            return
        try:
            recovery = self.query_one("#console-settings-default-recovery", Vertical)
            block = self.query_one("#console-settings-new-chat-default-block", Static)
        except (NoMatches, QueryError):
            return
        if recovery.display:
            return
        if block.display:
            body = self.query_one("#console-settings-body", ScrollableContainer)
            focused = self.app.focused
            if focused is not None and body in focused.ancestors:
                return
            block.scroll_visible(
                animate=False,
                immediate=True,
                force=True,
            )

    async def _request_default_recovery(
        self,
        action: ConsoleDefaultRecoveryAction,
    ) -> None:
        intent = self._default_durability_state.recovery_intent
        phase = self._default_durability_state.failure_phase
        allowed_actions = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: {
                ConsoleDefaultRecoveryAction.RETRY_SAVE,
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            },
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: {
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            },
        }
        if (
            intent is None
            or self._default_recovery_handler is None
            or action not in allowed_actions.get(phase, set())
            or self._default_recovery_pending is not None
        ):
            return
        assert isinstance(phase, ConsoleDefaultSavePhase)
        request = ConsoleDefaultRecoveryRequest(action, intent.generation)
        self._default_recovery_pending = (intent.generation, phase)
        self._sync_default_recovery_region()
        try:
            state = await self._default_recovery_handler(request)
        except Exception:
            self._set_validation_error("Default recovery failed; try again.")
            return
        finally:
            self._default_recovery_pending = None
            self._sync_default_recovery_region()
            self.call_after_refresh(self._focus_default_recovery_action)
        if not isinstance(state, ConsoleDefaultDurabilityState):
            self._set_validation_error("Default recovery returned invalid state.")
            self._sync_default_recovery_region()
            return
        self._clear_validation_error_summary()
        self._default_durability_state = state
        self._sync_default_recovery_region()
        self._sync_default_readiness()

    def on_key(self, event: events.Key) -> None:
        """Scroll an overflowing recovery summary while its actions keep focus."""

        if self._default_recovery_layout_phase is None:
            return
        try:
            body = self.query_one("#console-settings-body", ScrollableContainer)
        except (NoMatches, QueryError):
            return
        if event.key == "pageup":
            body.scroll_page_up(animate=False, force=True)
        elif event.key == "pagedown":
            body.scroll_page_down(animate=False, force=True)
        elif event.key == "home":
            body.scroll_home(animate=False, immediate=True, force=True)
        elif event.key == "end":
            body.scroll_end(animate=False, immediate=True, force=True)
        else:
            return
        event.stop()
        event.prevent_default()

    @on(Button.Pressed, "#console-settings-default-retry")
    async def _retry_default_save(self, event: Button.Pressed) -> None:
        event.stop()
        await self._request_default_recovery(ConsoleDefaultRecoveryAction.RETRY_SAVE)

    @on(Button.Pressed, "#console-settings-default-discard")
    async def _discard_default_retry(self, event: Button.Pressed) -> None:
        event.stop()
        await self._request_default_recovery(ConsoleDefaultRecoveryAction.DISCARD_RETRY)

    @on(Button.Pressed, "#console-settings-default-refresh")
    async def _refresh_running_app(self, event: Button.Pressed) -> None:
        event.stop()
        await self._request_default_recovery(
            ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
        )

    @on(Button.Pressed, "#console-settings-default-dismiss")
    async def _dismiss_runtime_refresh(self, event: Button.Pressed) -> None:
        event.stop()
        await self._request_default_recovery(
            ConsoleDefaultRecoveryAction.DISMISS_REFRESH
        )

    def _focus_context_control(self) -> None:
        """Focus and reveal the first editable current-conversation control."""
        try:
            control = self.query_one("#console-context-budget-mode", Select)
        except NoMatches:
            return
        control.focus()
        control.scroll_visible(
            animate=False,
            immediate=True,
            force=True,
        )

    def _sync_fold_hint(self) -> None:
        """Show a persistent cue whenever the modal body has hidden content."""
        try:
            body = self.query_one("#console-settings-body", ScrollableContainer)
            hint = self.query_one("#console-settings-fold-hint", Static)
        except NoMatches:
            return
        recovery_active = (
            self._default_durability_state.recovery_intent is not None
            and self._default_durability_state.failure_phase is not None
        )
        overflow = body.virtual_size.height > body.container_size.height
        if recovery_active:
            hint.update("▼ more — scroll recovery summary")
            hint.display = overflow
            return
        hint.update("▼ more — scroll for the rest")
        hint.display = overflow

    @on(Button.Pressed, "#console-settings-view-model")
    def _show_model_view(self, event: Button.Pressed) -> None:
        event.stop()
        self._show_settings_view("model")

    @on(Button.Pressed, "#console-settings-view-context")
    def _show_context_view(self, event: Button.Pressed) -> None:
        event.stop()
        self._show_settings_view("context")
        self.query_one("#console-context-budget-mode", Select).focus()

    @on(Button.Pressed, "#console-settings-configure-credential")
    def _configure_credential(self, event: Button.Pressed) -> None:
        """Dismiss through the typed, secret-free credential recovery result."""
        event.stop()
        if self._missing_credential_recovery_available():
            self.dismiss(self.credential_request())

    # Textual 8 composes same-named sync/async MRO message handlers; this is not
    # an ordinary OO override of the mixin hook.
    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """Recover select clicks redirected through focused Textual Web inputs.

        Args:
            event: Click event that may have been redirected from a focused
                settings input.
        """
        self._open_select_from_redirected_settings_click(event)

    def _open_select_from_redirected_settings_click(self, event: events.Click) -> None:
        """Recover settings controls when an input-held click lands on them.

        Args:
            event: Click event to recover when Textual Web keeps routing clicks
                through a focused settings input.
        """
        captured_widget = self.app.mouse_captured
        click_origin = getattr(event, "widget", None)
        focused_widget = self.app.focused
        screen_routed_click = click_origin is self and isinstance(
            focused_widget, ConsoleSettingsInput
        )
        if (
            not isinstance(captured_widget, ConsoleSettingsInput)
            and not isinstance(click_origin, ConsoleSettingsInput)
            and not screen_routed_click
        ):
            return
        if isinstance(captured_widget, ConsoleSettingsInput):
            captured_widget.release_mouse()

        if event.button != 1 or event.screen_x is None or event.screen_y is None:
            return

        for select in self.query(Select):
            if select.disabled or not select.display:
                continue
            select_region = _settings_screen_region(select)
            if select_region.contains(event.screen_x, event.screen_y):
                select.focus()
                select.action_show_overlay()
                event.stop()
                return
        for button in self.query(Button):
            if button.disabled or not button.display:
                continue
            button_region = _settings_screen_region(button)
            if button_region.contains(event.screen_x, event.screen_y):
                button.focus()
                button.press()
                event.stop()
                return

    def _has_selected_model(self) -> bool:
        try:
            return bool(self._current_model_value())
        except (NoMatches, QueryError):
            return bool(self._model_for_provider(self._active_provider))

    def _is_model_setup_mode(self) -> bool:
        return self._focus_model and not self._has_selected_model()

    @staticmethod
    def _readiness_copy(readiness: ConsoleSettingsReadiness) -> str:
        presentation = build_console_readiness_presentation(readiness)
        return "\n".join(
            (
                presentation.primary_label,
                presentation.detail,
                presentation.credential_row,
                presentation.endpoint_row,
                presentation.model_row,
                presentation.generation_row,
            )
        )

    def _provider_model_section_classes(self) -> str:
        classes = "console-settings-modal-section console-settings-model-view"
        if self._is_model_setup_mode():
            classes += " console-settings-primary-section"
        return classes

    def _modal_label(self, text: str) -> Static:
        label = Static(text, classes="console-settings-modal-label")
        label.styles.width = MODAL_LABEL_WIDTH
        label.styles.min_width = MODAL_LABEL_WIDTH
        label.styles.max_width = MODAL_LABEL_WIDTH
        return label

    def action_dismiss(self) -> None:
        """Route the dismiss action through the applicable close guard."""
        self._request_settings_close()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self._request_settings_close()

    def _request_settings_close(self) -> None:
        """Dismiss cleanly or reveal the one applicable side-effect guard."""
        guard = self.query_one("#console-settings-close-guard", Vertical)
        if guard.display:
            self._focus_settings_close_guard()
            return
        if self._memory_reset_token is not None:
            self._show_settings_close_guard("reset")
            return
        if self._compaction_is_active():
            self._show_settings_close_guard("compaction")
            return
        self.dismiss_safe_once(None)

    def _show_settings_close_guard(self, mode: str) -> None:
        guard = self.query_one("#console-settings-close-guard", Vertical)
        if guard.display and self._settings_close_guard_mode == mode:
            self._focus_settings_close_guard()
            return
        if not guard.display:
            self._settings_close_guard_focus = self.focused
        self._settings_close_guard_mode = mode
        is_reset = mode == "reset"
        self.query_one("#console-settings-close-undo", Button).display = is_reset
        self.query_one("#console-settings-close-keep", Button).display = is_reset
        self.query_one("#console-settings-close-anyway", Button).display = not is_reset
        message = self.query_one("#console-settings-close-message", Static)
        message.update(
            (
                "Current branch memory was reset. Undo it before closing, "
                "keep the reset, or return to Settings."
                if is_reset
                else "Compaction is still running. Closing stops waiting here. "
                f"{COMPACTION_CLOSE_WARNING}"
            )
        )
        guard.add_class("visible")
        guard.display = True
        self.call_after_refresh(self._focus_settings_close_guard)

    def _focus_settings_close_guard(self) -> None:
        selector = (
            "#console-settings-close-undo"
            if self._settings_close_guard_mode == "reset"
            else "#console-settings-close-anyway"
        )
        self.query_one(selector, Button).focus()

    def action_settings_focus_next(self) -> None:
        """Move focus forward within the active close guard or Settings."""
        guard = self.query_one("#console-settings-close-guard", Vertical)
        selector = "#console-settings-close-guard Button" if guard.display else "*"
        self.focus_next(selector)

    def action_settings_focus_previous(self) -> None:
        """Move focus backward within the active close guard or Settings."""
        guard = self.query_one("#console-settings-close-guard", Vertical)
        selector = "#console-settings-close-guard Button" if guard.display else "*"
        self.focus_previous(selector)

    def _hide_settings_close_guard(self) -> str | None:
        guard = self.query_one("#console-settings-close-guard", Vertical)
        mode = self._settings_close_guard_mode
        guard.remove_class("visible")
        guard.display = False
        self._settings_close_guard_mode = None
        return mode

    def _restore_settings_close_focus(self, widget: Widget | None) -> None:
        if widget is not None and widget.is_mounted and widget.is_attached:
            widget.focus()

    def _compaction_is_active(self) -> bool:
        task = self._compaction_provider_task
        worker = self._compaction_wait_worker
        return bool(
            (task is not None and not task.done())
            or (worker is not None and not worker.is_finished)
        )

    @on(Button.Pressed, "#console-settings-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-settings-close-undo")
    async def _close_after_undo(self, event: Button.Pressed) -> None:
        event.stop()
        if await self._undo_memory_reset():
            self._finish_reset_close_choice()

    @on(Button.Pressed, "#console-settings-close-keep")
    def _keep_reset_and_close(self, event: Button.Pressed) -> None:
        event.stop()
        self._memory_reset_token = None
        undo = self.query_one("#console-context-undo-reset", Button)
        undo.display = False
        undo.disabled = True
        self._finish_reset_close_choice()

    def _finish_reset_close_choice(self) -> None:
        if self._compaction_is_active():
            self.query_one("#console-context-action-status", Static).update(
                "Compacting… one additional model call may be billed."
            )
            self._show_settings_close_guard("compaction")
            return
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#console-settings-close-anyway")
    def _close_during_compaction(self, event: Button.Pressed) -> None:
        event.stop()
        if self._compaction_result_definitive:
            self.dismiss_safe_once(None)
            return
        worker = self._compaction_wait_worker
        if not self.dismiss_safe_once(None):
            return
        self._compaction_wait_worker = None
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self.notify(
            COMPACTION_CLOSE_WARNING,
            severity="warning",
            markup=False,
        )

    @on(Button.Pressed, "#console-settings-close-return")
    def _return_from_close_guard(self, event: Button.Pressed) -> None:
        event.stop()
        mode = self._hide_settings_close_guard()
        if mode == "reset":
            focus = self.query_one("#console-context-undo-reset", Button)
        else:
            focus = self._settings_close_guard_focus
        self._settings_close_guard_focus = None
        self.call_after_refresh(self._restore_settings_close_focus, focus)

    @on(Button.Pressed, "#console-settings-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(ConsoleSettingsAction.APPLY_TO_CHAT)

    @on(Button.Pressed, "#console-settings-save-default")
    def _save_as_default(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(ConsoleSettingsAction.SAVE_MODEL_DEFAULT)

    @on(Button.Pressed, "#console-settings-make-default")
    def _make_default(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT)

    @on(Button.Pressed, "#console-context-reset-overrides")
    def _reset_policy_overrides(self, event: Button.Pressed) -> None:
        """Restore the inherited policy draft without writing until Save."""
        event.stop()
        self._context_overrides_reset = True
        policy = self._context_state.inherited_policy
        self.query_one(
            "#console-context-budget-mode", Select
        ).value = policy.budget_mode.value
        self.query_one(
            "#console-context-custom-budget", Input
        ).value = self._format_value(policy.custom_budget_tokens)
        self.query_one(
            "#console-context-compaction-mode", Select
        ).value = policy.compaction_mode.value
        self.query_one(
            "#console-context-compaction-representation", Select
        ).value = policy.compaction_representation.value
        self.query_one("#console-context-trigger-percent", Input).value = str(
            self._format_percent(policy.trigger_ratio)
        )
        self.query_one("#console-context-target-percent", Input).value = str(
            self._format_percent(policy.target_ratio)
        )
        self.query_one("#console-context-summary-max", Input).value = str(
            policy.summary_max_tokens
        )
        self.query_one(
            "#console-context-failure-behavior", Select
        ).value = policy.failure_behavior.value
        self.query_one(
            "#console-context-carry-forward", Select
        ).value = policy.carry_forward_mode.value
        self.query_one("#console-context-policy-provenance", Static).update(
            "Inheriting Console Behavior defaults after Save."
        )

    @on(Button.Pressed, "#console-context-reset-current")
    async def _reset_current_branch_memory(self, event: Button.Pressed) -> None:
        """Deactivate only the selected branch-valid memory revision."""
        event.stop()
        if self._reset_current_memory is None:
            return
        event.button.disabled = True
        token = await asyncio.to_thread(self._reset_current_memory)
        if not self.is_mounted:
            return
        status = self.query_one("#console-context-action-status", Static)
        if token is None:
            event.button.disabled = False
            status.update("Memory changed before it could be reset.")
            return
        self._memory_reset_token = token
        self.query_one("#console-settings-memory-review", Static).update(
            "No conversation memory is active on this branch."
        )
        self.query_one("#console-context-memory-metadata", Static).update(
            "Current branch memory reset; transcript unchanged."
        )
        self.query_one("#console-context-reset-current", Button).display = False
        undo = self.query_one("#console-context-undo-reset", Button)
        undo.display = True
        undo.disabled = False
        status.update("Current branch memory reset. Undo is available.")

    @on(Button.Pressed, "#console-context-compact-now")
    def _compact_context_now(self, event: Button.Pressed) -> None:
        """Start one explicit auxiliary compaction call without sending chat."""
        event.stop()
        if self._compact_now is None:
            return
        event.button.disabled = True
        self._compaction_result_definitive = False
        self.query_one("#console-context-action-status", Static).update(
            "Compacting… one additional model call may be billed."
        )
        self._compaction_wait_worker = self.run_worker(
            self._compact_context_now_worker(),
            exclusive=True,
            group="console-settings-compaction",
        )

    async def _compact_context_now_worker(self) -> None:
        """Run the supplied controller action and restore the modal controls."""
        if self._compact_now is None:
            return
        provider_task = asyncio.create_task(self._run_context_compaction())
        self._compaction_provider_task = provider_task
        provider_task.add_done_callback(self._context_compaction_finished)
        try:
            succeeded, message = await asyncio.shield(provider_task)
        except asyncio.CancelledError:
            return
        self._compaction_wait_worker = None
        if not self.is_mounted:
            return
        self.query_one("#console-context-action-status", Static).update(message)
        button = self.query_one("#console-context-compact-now", Button)
        button.disabled = False
        if succeeded:
            self.query_one("#console-context-memory-metadata", Static).update(
                "Generated memory updated; transcript unchanged. Reopen to review provenance."
            )
        if self._settings_close_guard_mode == "compaction":
            focus = self._settings_close_guard_focus
            self.query_one("#console-settings-close-message", Static).update(message)
            self._hide_settings_close_guard()
            self._settings_close_guard_focus = None
            self.call_after_refresh(self._restore_settings_close_focus, focus)

    async def _run_context_compaction(self) -> tuple[bool, str]:
        """Run provider work independently of the modal-owned wait worker."""
        if self._compact_now is None:
            return False, "Conversation compaction is unavailable."
        try:
            return await self._compact_now()
        except Exception:
            return False, "Conversation compaction failed before memory was updated."

    def _context_compaction_finished(
        self, task: asyncio.Task[tuple[bool, str]]
    ) -> None:
        if self._compaction_provider_task is task:
            self._compaction_result_definitive = True
            self._compaction_provider_task = None

    @on(Button.Pressed, "#console-context-undo-reset")
    async def _undo_current_branch_memory_reset(self, event: Button.Pressed) -> None:
        """Reactivate the exact reset revision when it has not changed again."""
        event.stop()
        await self._undo_memory_reset()

    async def _undo_memory_reset(self) -> bool:
        """Reactivate the optimistic reset token and refresh its controls."""
        token = self._memory_reset_token
        if token is None or self._undo_current_memory_reset is None:
            return False
        restored = await asyncio.to_thread(self._undo_current_memory_reset, *token)
        if not self.is_mounted:
            return False
        status = self.query_one("#console-context-action-status", Static)
        if not restored:
            recovery = "Undo expired because conversation memory changed."
            status.update(recovery)
            if self._settings_close_guard_mode == "reset":
                self.query_one("#console-settings-close-message", Static).update(
                    recovery
                )
            return False
        self._memory_reset_token = None
        self.query_one("#console-settings-memory-review", Static).update(
            self._memory_review_text()
        )
        self.query_one("#console-context-memory-metadata", Static).update(
            self._memory_metadata_label()
        )
        reset = self.query_one("#console-context-reset-current", Button)
        reset.display = True
        reset.disabled = False
        undo = self.query_one("#console-context-undo-reset", Button)
        undo.display = False
        undo.disabled = True
        status.update("Current branch memory restored; transcript was unchanged.")
        return True

    @on(Button.Pressed, "#console-context-reset-all")
    def _request_reset_all_memories(self, event: Button.Pressed) -> None:
        """Reveal a separate cross-branch confirmation action."""
        event.stop()
        self._confirm_reset_all = True
        event.button.display = False
        confirm = self.query_one("#console-context-confirm-reset-all", Button)
        confirm.display = True
        confirm.focus()
        self.query_one("#console-context-action-status", Static).update(
            "This resets all generated and legacy conversation memory across "
            "every branch. Transcript messages will not change."
        )

    @on(Button.Pressed, "#console-context-confirm-reset-all")
    async def _confirm_reset_all_context_memories(
        self, event: Button.Pressed
    ) -> None:
        """Apply the separately confirmed all-branch memory reset."""
        event.stop()
        if not self._confirm_reset_all or self._reset_all_memories is None:
            return
        event.button.disabled = True
        count = await asyncio.to_thread(self._reset_all_memories)
        if not self.is_mounted:
            return
        event.button.disabled = False
        self._confirm_reset_all = False
        self._memory_reset_token = None
        event.button.display = False
        self.query_one("#console-context-reset-all", Button).display = True
        undo = self.query_one("#console-context-undo-reset", Button)
        undo.display = False
        undo.disabled = True
        self.query_one("#console-settings-memory-review", Static).update(
            "No conversation memory is active."
        )
        self.query_one("#console-context-memory-metadata", Static).update(
            "All generated and legacy conversation memory was reset; transcript "
            f"unchanged. Generated records deactivated: {count}."
        )
        self.query_one("#console-context-action-status", Static).update(
            "Reset all generated and legacy conversation memory across every "
            f"branch. Generated records deactivated: {count}."
        )

    def _validated_result_or_show_errors(self) -> ConsoleSettingsResult | None:
        """Build the result, surfacing validation errors without dismissing."""
        draft = self._build_draft()
        identity_errors: list[str] = []
        try:
            user_display_name_override = normalize_chat_display_name(
                self.query_one("#console-settings-user-display-name", Input).value,
                blank_means_none=True,
            )
        except ChatDisplayNameError as exc:
            user_display_name_override = None
            identity_errors.append(str(exc))
        try:
            context_overrides = self._build_context_policy_overrides()
            context_errors: list[str] = []
        except (ContextPolicyError, ValueError) as exc:
            context_overrides = self._context_state.overrides
            context_errors = [str(exc)]
        errors = [
            *identity_errors,
            *context_errors,
            *self._required_sampling_errors(),
            *self._provider_choice_input_errors(),
            *validate_console_session_settings(draft, app_config=self._app_config),
        ]
        if errors:
            # TASK-363: surface the error prominently AND bring it on-screen — in
            # a taller-than-viewport modal the summary can sit well above the fold
            # (the review's "Save did nothing" confusion), so scroll it into view.
            error_banner = self.query_one("#console-settings-error", Static)
            error_banner.update("\n".join(errors))
            error_banner.display = True
            error_banner.scroll_visible()
            return None
        return ConsoleSettingsResult(
            settings=draft,
            user_display_name_override=user_display_name_override,
            context_policy_overrides=context_overrides,
            thinking_history_policy=normalize_thinking_history_policy(
                self._select_value_text(
                    self.query_one(
                        "#console-context-thinking-history-policy", Select
                    ).value
                )
            ),
        )

    def _thinking_history_effective_copy(self) -> str:
        state = self._context_state.thinking_history
        if state.effective_label == "Required":
            return f"Effective: Required — {state.required_reason}"
        return f"Effective: {state.effective_label}"

    @on(Button.Pressed, "#console-context-thinking-history-save-default")
    async def _save_thinking_history_default(self, event: Button.Pressed) -> None:
        """Persist only the new-conversation default without dismissing."""

        event.stop()
        policy = normalize_thinking_history_policy(
            self._select_value_text(
                self.query_one("#console-context-thinking-history-policy", Select).value
            )
        )
        try:
            saved = await asyncio.to_thread(
                save_settings_to_cli_config,
                {"console": {"thinking_history_policy_default": policy}},
            )
        except Exception:
            saved = False
        status = self.query_one("#console-context-action-status", Static)
        if saved:
            if isinstance(self._app_config, dict):
                console = self._app_config.setdefault("console", {})
                if isinstance(console, dict):
                    console["thinking_history_policy_default"] = policy
            status.update(f"{policy.title()} will be used for new conversations only.")
        else:
            status.update("Could not save the new-conversation thinking default.")

    def _validated_submission_draft(
        self,
    ) -> tuple[ConsoleSettingsDraftState, str | None] | None:
        """Return the complete full-surface draft after mounted validation."""

        provider = self._select_value_text(
            self.query_one("#console-settings-provider", Select).value
        )
        model = self._current_model_value()
        if self._draft_rebaser is not None and (
            provider,
            model,
        ) != (
            self._draft.settings.provider,
            self._draft.settings.model,
        ):
            picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
            if not self._rebase_to(
                provider,
                model,
                preserve_custom_model_input=picker.custom_mode,
            ):
                return None
        result = self._validated_result_or_show_errors()
        if result is None:
            return None
        field_drafts: list[ConsoleSettingsFieldDraft] = []
        for name in sorted(FULL_MODEL_DEFAULT_FIELDS):
            effective = getattr(result.settings, name)
            field_drafts.append(self._updated_field_draft(name, effective))
        endpoint = replace(
            self._endpoint_draft,
            value=result.settings.base_url or "",
            checked=self._endpoint_checkbox_value(),
        )
        state = ConsoleSettingsDraftState(
            settings=result.settings,
            context_policy_overrides=(
                result.context_policy_overrides or ConsoleContextPolicyOverrides()
            ),
            field_drafts=tuple(field_drafts),
            model_drafts=self._draft.model_drafts,
            endpoint_draft=endpoint,
        )
        self._draft = remember_model_draft(state)
        return self._draft, result.user_display_name_override

    def _submission_for_action(
        self,
        action: ConsoleSettingsAction,
    ) -> ConsoleSettingsSubmission | None:
        """Build one discriminated exact-origin submission for every action."""

        validated = self._validated_submission_draft()
        if validated is None:
            return None
        draft, display_name = validated
        endpoint = draft.endpoint_draft
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            if not self._endpoint_is_authorized(endpoint, draft.settings.provider):
                endpoint = None
            draft = replace(
                draft,
                model_drafts=tuple(
                    replace(remembered, endpoint_draft=None)
                    for remembered in draft.model_drafts
                ),
            )
        elif not self._endpoint_is_live_bound(endpoint, draft.settings.provider):
            endpoint = None
        draft = replace(draft, endpoint_draft=endpoint)
        return ConsoleSettingsSubmission(
            submission_id=uuid4().hex,
            action=action,
            surface=ConsoleSettingsSurface.FULL_SETTINGS,
            origin=self._origin,
            draft=draft,
            user_display_name_override=display_name,
            default_field_mask=(
                frozenset()
                if action is ConsoleSettingsAction.APPLY_TO_CHAT
                else FULL_MODEL_DEFAULT_FIELDS
            ),
        )

    def _submit(self, action: ConsoleSettingsAction) -> None:
        """Live-commit one action and dismiss only after exact-origin success."""

        if self._submit_pending or self._safe_dismiss_committed:
            return
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            button = self.query_one("#console-settings-make-default", Button)
            if button.disabled:
                self._set_validation_error(
                    str(
                        self.query_one(
                            "#console-settings-new-chat-default-block", Static
                        ).renderable
                    )
                    or "Default is unavailable."
                )
                return
        submission = self._submission_for_action(action)
        if submission is None:
            return
        if action is not ConsoleSettingsAction.APPLY_TO_CHAT:
            self._sync_default_readiness()
            if normalize_console_model_value(submission.draft.settings.model) is None:
                self._set_validation_error("Unavailable: choose a model first.")
                return
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            button = self.query_one("#console-settings-make-default", Button)
            if button.disabled:
                self._set_validation_error(
                    str(
                        self.query_one(
                            "#console-settings-new-chat-default-block", Static
                        ).renderable
                    )
                    or "Default is unavailable."
                )
                return
        captured = self.app.mouse_captured
        if captured is not None:
            captured.release_mouse()
        self._submit_pending = True
        try:
            live_commit = (
                self._live_committer(submission)
                if self._live_committer is not None
                else self._transitional_live_commit(submission)
            )
        except ValueError as error:
            message = str(error).strip().rstrip(".")
            if message == "Chat closed; nothing applied":
                self.notify("Chat closed; nothing applied", severity="warning")
                self.dismiss_safe_once(None)
                return
            self._set_validation_error(str(error) or "Settings could not be applied.")
            self._submit_pending = False
            return
        except Exception:
            self._set_validation_error(
                "Settings could not be applied; nothing changed."
            )
            self._submit_pending = False
            return
        if not isinstance(live_commit, ConsoleSettingsLiveCommit):
            self._set_validation_error(
                "Settings could not be applied; nothing changed."
            )
            self._submit_pending = False
            return
        delivered = False
        try:
            for warning in console_settings_warnings(live_commit.settings):
                self.notify(warning, severity="warning", timeout=8000)
            delivered = self.dismiss_safe_once(
                ConsoleSettingsCommittedSubmission(submission, live_commit)
            )
        finally:
            if not delivered and live_commit.durability_admission is not None:
                live_commit.durability_admission.release()

    @staticmethod
    def _transitional_live_commit(
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleSettingsLiveCommit:
        """Keep pre-Task-8 callers mountable without mutating configuration."""

        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=submission.origin.session_id,
            persisted_conversation_id=submission.origin.persisted_conversation_id,
            conversation_binding_revision=(
                submission.origin.conversation_binding_revision
            ),
            generation_revision=0,
            context_policy_revision=0,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
            accepted_submission=submission,
        )

    def _set_validation_error(self, copy: str) -> None:
        error = self.query_one("#console-settings-error", Static)
        error.update(copy)
        error.display = bool(copy)
        if copy:
            error.scroll_visible()

    @on(Button.Pressed, "#console-settings-streaming")
    def _toggle_streaming(self, event: Button.Pressed) -> None:
        """Cycle the profile draft through Inherit, On, and Off."""
        event.stop()
        self._streaming_draft = {
            None: True,
            True: False,
            False: None,
        }[self._streaming_draft]
        event.button.label = self._streaming_toggle_label()

    def _streaming_toggle_label(self) -> str:
        if self._streaming_draft is None:
            return STREAMING_INHERIT_LABEL
        return STREAMING_ON_LABEL if self._streaming_draft else STREAMING_OFF_LABEL

    def _effective_streaming_value(self) -> bool:
        return (
            self._streaming_effective_fallback
            if self._streaming_draft is None
            else self._streaming_draft
        )

    def _endpoint_checkbox_value(self) -> bool:
        try:
            checkbox = self.query_one("#console-settings-save-endpoint", Checkbox)
        except (NoMatches, QueryError):
            return bool(self._endpoint_draft.checked)
        return bool(checkbox.value and not checkbox.disabled)

    def _endpoint_can_be_checked(self) -> bool:
        provider = provider_config_key(self._active_provider)
        endpoint = self._endpoint_draft
        return bool(
            endpoint.dirty
            and endpoint.bound_provider_config_key == provider
            and format_console_endpoint_preview(endpoint.value) is not None
        )

    def _endpoint_save_copy(self) -> str:
        preview = format_console_endpoint_preview(self._endpoint_draft.value)
        return (
            f"Also save connection: {preview}"
            if preview is not None
            else "Also save connection"
        )

    def _endpoint_status_copy(self) -> str:
        endpoint = self._endpoint_draft
        if not endpoint.dirty:
            return "Edit the Base URL to enable connection saving."
        if endpoint.bound_provider_config_key != provider_config_key(
            self._active_provider
        ):
            return "Connection edit belongs to another provider."
        if format_console_endpoint_preview(endpoint.value) is None:
            return "Enter a valid endpoint before saving the connection."
        return "Connection is saved only by Make default for new chats."

    @staticmethod
    def _endpoint_is_live_bound(
        endpoint: ConsoleEndpointDraft | None,
        provider: str,
    ) -> bool:
        return bool(
            endpoint is not None
            and endpoint.dirty
            and endpoint.bound_provider_config_key == provider_config_key(provider)
            and format_console_endpoint_preview(endpoint.value) is not None
        )

    @classmethod
    def _endpoint_is_authorized(
        cls,
        endpoint: ConsoleEndpointDraft | None,
        provider: str,
    ) -> bool:
        return bool(
            cls._endpoint_is_live_bound(endpoint, provider)
            and endpoint is not None
            and endpoint.checked
        )

    def _sync_endpoint_controls(self) -> None:
        try:
            checkbox = self.query_one("#console-settings-save-endpoint", Checkbox)
            status = self.query_one("#console-settings-endpoint-status", Static)
        except (NoMatches, QueryError):
            return
        uses_endpoint = self._provider_uses_base_url(self._active_provider)
        checkbox.display = uses_endpoint
        status.display = uses_endpoint
        checkbox.disabled = not self._endpoint_can_be_checked()
        checkbox.label = self._endpoint_save_copy()
        checkbox.value = bool(self._endpoint_draft.checked and not checkbox.disabled)
        status.update(self._endpoint_status_copy())

    @on(Input.Changed, "#console-settings-base-url")
    def _base_url_changed(self, event: Input.Changed) -> None:
        if self._updating_controls:
            return
        value = event.value.strip()
        if value == self._endpoint_draft.value:
            return
        self._endpoint_draft = ConsoleEndpointDraft(
            value=value,
            bound_provider_config_key=provider_config_key(self._active_provider),
            dirty=True,
            checked=False,
        )
        self._sync_endpoint_controls()
        self._sync_default_readiness()

    @on(Checkbox.Changed, "#console-settings-save-endpoint")
    def _endpoint_checked(self, event: Checkbox.Changed) -> None:
        if self._updating_controls:
            return
        self._endpoint_draft = replace(
            self._endpoint_draft,
            checked=bool(event.value and self._endpoint_can_be_checked()),
        )
        self._sync_default_readiness()

    def _choice_placeholder(self, input_id: str) -> str:
        """Return the accepted-values placeholder for an enumerated choice input."""
        if input_id == "console-settings-reasoning-effort":
            hint = reasoning_effort_hint_for_model(self._settings.model)
            if hint is not None:
                return " / ".join(sorted(hint)) + " (consumed by this model)"
        for _label, choice_input_id, placeholder in PROVIDER_CHOICE_INPUTS:
            if choice_input_id == input_id:
                if (
                    input_id in _LOCAL_NO_EFFECT_CHOICE_INPUT_IDS
                    and _is_local_thinking_provider(self._active_provider)
                ):
                    return placeholder + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
                return placeholder
        return ""

    def _sync_provider_choice_placeholders(self) -> None:
        """Refresh choice-input placeholders after the provider changes."""
        for _label, input_id, _placeholder in PROVIDER_CHOICE_INPUTS:
            self.query_one(
                f"#{input_id}", Input
            ).placeholder = self._choice_placeholder(input_id)

    @on(Input.Changed)
    @on(Select.Changed)
    def _invalidate_validation_summary_on_edit(self, event) -> None:
        """Clear a stale validation summary once the user edits any field.

        TASK-363: the summary was only refreshed on the next Save, so it lingered
        after the offending field was fixed ("Save did nothing" then a stale
        error). Any edit means the shown errors may no longer apply, so clear
        them; the next Save re-validates. Runs alongside the field-specific
        handlers below (it does not stop the event).
        """
        if not self._restoring_suspended_draft:
            self._clear_validation_error_summary()
        if (
            isinstance(event, Input.Changed)
            and not self._updating_controls
            and not self._restoring_suspended_draft
            and event.input.id in _GENERATION_FIELD_BY_INPUT_ID
        ):
            self._edited_generation_fields.add(
                _GENERATION_FIELD_BY_INPUT_ID[event.input.id]
            )

    def _clear_validation_error_summary(self) -> None:
        try:
            error = self.query_one("#console-settings-error", Static)
            error.update("")
            error.display = False
        except (QueryError, NoMatches):
            pass

    def _snapshot_before_rebase(self) -> ConsoleSettingsDraftState:
        """Capture current mounted edits under the still-active source key."""

        mounted = self._build_draft()
        source_settings = replace(
            mounted,
            provider=self._active_provider,
            # A model-change event arrives after its control already displays
            # the target. The immutable draft still identifies the source
            # whose mounted edits must be remembered.
            model=self._draft.settings.model,
            base_url=self._current_base_url_value(self._active_provider),
        )
        fields: list[ConsoleSettingsFieldDraft] = []
        for name in sorted(FULL_MODEL_DEFAULT_FIELDS):
            effective = getattr(source_settings, name)
            fields.append(self._updated_field_draft(name, effective))
        return remember_model_draft(
            replace(
                self._draft,
                settings=source_settings,
                field_drafts=tuple(fields),
                endpoint_draft=self._endpoint_draft,
            )
        )

    def _rebase_to(
        self,
        provider: str,
        model: str | None,
        *,
        preserve_custom_model_input: bool = False,
    ) -> bool:
        """Delegate provider/model changes to the controller-owned rebaser."""

        if self._draft_rebaser is None:
            return False
        source = self._snapshot_before_rebase()
        try:
            rebased = self._draft_rebaser(
                source,
                provider=provider,
                model=model,
                app_config=self._app_config,
                exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
            )
        except Exception:
            self._set_validation_error("Provider/model settings could not be rebased.")
            return False
        if not isinstance(rebased, ConsoleSettingsDraftState):
            self._set_validation_error("Provider/model rebase returned invalid state.")
            return False
        requested_provider = provider_config_key(provider)
        requested_model = normalize_console_model_value(model)
        provider_changed = (
            provider_config_key(source.settings.provider) != requested_provider
        )
        provider_matches = (
            provider_config_key(rebased.settings.provider) == requested_provider
        )
        model_matches = (
            normalize_console_model_value(rebased.settings.model) == requested_model
        )
        if not provider_matches or not (
            model_matches or (provider_changed and requested_model is None)
        ):
            self._set_validation_error(
                "Rebased settings did not match the requested provider/model."
            )
            self._sync_default_readiness()
            return False
        self._apply_rebased_state(
            rebased,
            preserve_custom_model_input=preserve_custom_model_input,
        )
        return True

    def _apply_rebased_state(
        self,
        state: ConsoleSettingsDraftState,
        *,
        preserve_custom_model_input: bool = False,
    ) -> None:
        """Project a controller-owned rebase back into mounted controls."""

        self._updating_controls = True
        self._rebase_event_guard = True
        try:
            self._draft = state
            self._settings = state.settings
            self._active_provider = state.settings.provider
            self._endpoint_draft = state.endpoint_draft or ConsoleEndpointDraft(
                value=state.settings.base_url or "",
                bound_provider_config_key=provider_config_key(state.settings.provider),
                dirty=False,
                checked=False,
            )
            self.query_one(
                "#console-settings-provider", Select
            ).value = state.settings.provider
            if preserve_custom_model_input:
                self.query_one(
                    "#console-settings-model-picker", ModelSearchPicker
                ).set_custom_value(state.settings.model)
                self.query_one(
                    "#console-settings-model-custom", Button
                ).label = "Model list"
            else:
                self._sync_model_controls(
                    state.settings.provider,
                    state.settings.model,
                )
            self._sync_base_url_control(
                state.settings.provider,
                state.settings.base_url,
            )
            input_fields = {
                "temperature": "console-settings-temperature",
                "top_p": "console-settings-top-p",
                "min_p": "console-settings-min-p",
                "top_k": "console-settings-top-k",
                "max_tokens": "console-settings-max-tokens",
                "seed": "console-settings-seed",
                "presence_penalty": "console-settings-presence-penalty",
                "frequency_penalty": "console-settings-frequency-penalty",
                "reasoning_effort": "console-settings-reasoning-effort",
                "reasoning_summary": "console-settings-reasoning-summary",
                "verbosity": "console-settings-verbosity",
                "thinking_effort": "console-settings-thinking-effort",
                "thinking_budget_tokens": "console-settings-thinking-budget-tokens",
            }
            for name, input_id in input_fields.items():
                self.query_one(f"#{input_id}", Input).value = self._format_value(
                    getattr(state.settings, name)
                )
            streaming = self._draft_field("streaming")
            self._streaming_effective_fallback = bool(state.settings.streaming)
            self._streaming_draft = (
                streaming.profile_override
                if streaming is not None
                and (
                    streaming.profile_override is None
                    or type(streaming.profile_override) is bool
                )
                else bool(state.settings.streaming)
            )
            self.query_one(
                "#console-settings-streaming", Button
            ).label = self._streaming_toggle_label()
            self._edited_generation_fields.clear()
            self._sync_model_discover_controls(state.settings.provider)
            self._sync_provider_choice_placeholders()
            self._sync_endpoint_controls()
        finally:
            self._updating_controls = False
        self._sync_readiness_display()
        self._sync_default_readiness()
        self._sync_visual_representation_availability()
        self.call_after_refresh(self._clear_rebase_event_guard)

    def _clear_rebase_event_guard(self) -> None:
        self._rebase_event_guard = False

    @on(Select.Changed, "#console-settings-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        if self._restoring_suspended_draft:
            return
        provider = self._select_value_text(event.value)
        if (
            self._updating_controls
            or self._rebase_event_guard
            or provider == self._active_provider
        ):
            return
        if self._draft_rebaser is not None:
            self._rebase_to(provider, None)
            return
        self._store_current_model_for_provider(self._active_provider)
        self._store_current_base_url_for_provider(self._active_provider)
        model = self._model_for_provider(provider)
        base_url = self._base_url_for_provider(provider)
        self._active_provider = provider
        self._sync_model_controls(provider, model)
        self._sync_base_url_control(provider, base_url)
        self._sync_model_discover_controls(provider)
        self._sync_provider_choice_placeholders()
        self._sync_readiness_display()
        self._sync_visual_representation_availability()
        self._sync_endpoint_controls()
        self._sync_default_readiness()

    @on(Select.Changed, "#console-settings-model-select")
    def _model_select_changed(self, event: Select.Changed) -> None:
        if self._restoring_suspended_draft:
            return
        model_id = normalize_console_model_value(self._select_value_text(event.value))
        # ``set_options`` queues a transient blank event before the concrete
        # replacement value. By dispatch time the Select already exposes the
        # final value, so ignore that stale adapter echo instead of rebasing a
        # second time to a model-less target.
        current_select_model = normalize_console_model_value(
            self._select_value_text(
                self.query_one("#console-settings-model-select", Select).value
            )
        )
        if model_id is None and current_select_model is not None:
            return
        self.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        ).set_model_value(model_id)
        if (
            not self._updating_controls
            and not self._rebase_event_guard
            and self._draft_rebaser is not None
            and model_id != self._draft.settings.model
        ):
            self._rebase_to(self._active_provider, model_id)
            return
        self._sync_readiness_display()
        self._sync_visual_representation_availability()

    @on(Input.Changed, "#console-settings-model-input")
    def _model_input_changed(self, event: Input.Changed) -> None:
        """Mirror the typed id into the picker now; debounce the rest.

        `picker.set_custom_value` only touches the picker's own small
        display state (its Input mirror, custom-mode button, status line),
        so it stays immediate. `_sync_readiness_display` rebuilds a full
        `ConsoleSessionSettings` draft from every form field and
        re-validates it, and `_sync_visual_representation_availability`
        does a model-capability lookup -- neither should run on every
        keystroke (task-15476).
        """
        if self._restoring_suspended_draft:
            return
        picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        if picker.custom_mode:
            picker.set_custom_value(event.value)
        self._schedule_readiness_sync()

    def _schedule_readiness_sync(self) -> None:
        """Debounce validation and custom-model rebasing while text is edited."""

        if self._readiness_debounce_timer is not None:
            self._readiness_debounce_timer.stop()
        self._readiness_debounce_timer = self.set_timer(
            CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS,
            self._apply_readiness_sync_debounced,
        )

    def _apply_readiness_sync_debounced(self) -> None:
        self._readiness_debounce_timer = None
        model = self._current_model_value()
        if (
            self._draft_rebaser is not None
            and not self._updating_controls
            and (self._active_provider, model)
            != (self._draft.settings.provider, self._draft.settings.model)
        ):
            picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
            self._rebase_to(
                self._active_provider,
                model,
                preserve_custom_model_input=picker.custom_mode,
            )
            return
        self._sync_readiness_display()
        self._sync_default_readiness()
        self._sync_visual_representation_availability()

    @on(ModelSearchPicker.ModelSelected)
    def _model_picker_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        event.stop()
        if self._restoring_suspended_draft:
            return
        if (
            self._draft_rebaser is not None
            and not self._updating_controls
            and not self._rebase_event_guard
            and event.model_id != self._draft.settings.model
        ):
            self._rebase_to(self._active_provider, event.model_id)
        self._set_provider_model_draft(self._active_provider, event.model_id)
        self._sync_model_controls(self._active_provider, event.model_id)
        self._sync_readiness_display()
        self._sync_visual_representation_availability()

    @on(ModelSearchPicker.ModelValueChanged)
    def _model_picker_value_changed(
        self, event: ModelSearchPicker.ModelValueChanged
    ) -> None:
        event.stop()
        if self._restoring_suspended_draft:
            return
        if event.custom:
            self._schedule_readiness_sync()
            return
        if (
            event.model_id is None
            and not event.custom
            and self._draft.settings.model is not None
        ):
            return
        if (
            self._draft_rebaser is not None
            and not self._updating_controls
            and not self._rebase_event_guard
            and event.model_id != self._draft.settings.model
        ):
            self._rebase_to(self._active_provider, event.model_id)
        self._set_provider_model_draft(self._active_provider, event.model_id)
        self._sync_readiness_display()
        self._sync_visual_representation_availability()

    @on(Select.Changed, "#console-context-compaction-representation")
    def _compaction_representation_changed(self, _event: Select.Changed) -> None:
        self._sync_visual_representation_availability()

    @on(Button.Pressed, "#console-settings-model-custom")
    def _model_custom_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        picker.toggle_custom_mode()
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        model_custom = self.query_one("#console-settings-model-custom", Button)
        if picker.custom_mode:
            model_select.display = False
            model_select.disabled = True
            model_input.display = True
            model_input.disabled = False
            model_input.value = picker.value or ""
            model_custom.label = "Model list"
        else:
            self._sync_model_controls(self._active_provider, picker.value)
        self._sync_readiness_display()

    @staticmethod
    def _provider_supports_model_discovery(provider: str) -> bool:
        """Return whether the provider serves an OpenAI-compatible model list."""
        return provider_config_key(provider) in URL_BASED_PROVIDER_KEYS

    @on(Button.Pressed, f"#{MODEL_DISCOVER_BUTTON_ID}")
    def _model_discover_pressed(self, event: Button.Pressed) -> None:
        """Probe the current base-URL draft for its served model list."""
        event.stop()
        provider = self._select_value_text(
            self.query_one("#console-settings-provider", Select).value
        )
        if not self._provider_supports_model_discovery(provider):
            return
        base_url = self._current_base_url_value(provider) or ""
        if not base_url:
            self._set_model_discover_status(MODEL_DISCOVER_MISSING_URL_COPY)
            return
        normalized_probe_url = normalize_probe_base_url(base_url)
        # PR #608 review: user-entered endpoint must pass the shared
        # input_validation boundary before any network use.
        if normalized_probe_url is None or not validate_url(normalized_probe_url):
            self._set_model_discover_status(MODEL_DISCOVER_INVALID_URL_COPY)
            return
        event.button.disabled = True
        self._set_model_discover_status(f"Contacting {endpoint_display(base_url)}…")
        self.run_worker(
            self._run_model_discovery(provider, base_url),
            exclusive=True,
            group="console-settings-model-discovery",
        )

    async def _run_model_discovery(self, provider: str, base_url: str) -> None:
        """Run the model probe off the draft URL and apply the outcome.

        Args:
            provider: Provider draft value at press time.
            base_url: Base-URL draft at press time.
        """
        try:
            result = await self._model_prober(base_url, provider_config_key(provider))
        except Exception:
            result = LocalModelProbeResult(
                ok=False,
                base_url=base_url,
                detail=f"No models endpoint at {endpoint_display(base_url)}.",
            )
        self._apply_model_discovery_result(provider, result)

    def _apply_model_discovery_result(
        self,
        provider: str,
        result: LocalModelProbeResult,
    ) -> None:
        """Surface a probe outcome: model Select on success, honest copy otherwise.

        Args:
            provider: Provider the probe was started for; results for a
                provider the user has since switched away from are dropped.
            result: Probe outcome from the discovery module.
        """
        try:
            discover = self.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        except (NoMatches, QueryError):
            return
        discover.disabled = not self._provider_supports_model_discovery(
            self._active_provider
        )
        if provider != self._active_provider:
            self._set_model_discover_status("")
            return
        display = endpoint_display(result.base_url)
        if not result.ok:
            self._set_model_discover_status(
                result.detail or f"No models endpoint at {display}."
            )
            return
        if not result.model_ids:
            self._set_model_discover_status(f"No models reported at {display}.")
            return
        self._discovered_model_ids[provider] = tuple(result.model_ids)
        picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        picker.set_discovered_models(provider, list(result.model_ids))
        count = len(result.model_ids)
        # With exactly one model there is nothing to choose, so choose it. The
        # previous behaviour kept whatever was already selected -- which is how
        # a TTS model stayed active against a chat endpoint after a successful
        # discovery, with only a status line hinting anything had changed.
        selected_model = self._current_model_value()
        if count == 1:
            selected_model = result.model_ids[0]
        self._sync_model_controls(provider, selected_model)
        if count == 1:
            self._set_model_discover_status(
                f"Found 1 model at {display}; selected {result.model_ids[0]}."
            )
        else:
            self._set_model_discover_status(f"Found {count} models at {display}.")
        self._sync_readiness_display()

    def _set_model_discover_status(self, text: str) -> None:
        """Update the inline discovery status line, hiding it when blank."""
        try:
            status = self.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        except (NoMatches, QueryError):
            return
        status.update(text)
        status.display = bool(text.strip())

    def _sync_model_discover_controls(self, provider: str) -> None:
        """Show the discovery affordance only for URL-based providers."""
        supports_discovery = self._provider_supports_model_discovery(provider)
        try:
            discover = self.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        except (NoMatches, QueryError):
            return
        discover.display = supports_discovery
        discover.disabled = not supports_discovery
        self._set_model_discover_status("")

    def _sync_readiness_display(self) -> None:
        draft = self._build_draft()
        readiness = build_console_settings_readiness(draft, app_config=self._app_config)
        self.query_one("#console-settings-readiness", Static).update(
            self._readiness_copy(readiness)
        )
        try:
            self.query_one(
                "#console-settings-configure-credential", Button
            ).display = self._missing_credential_recovery_available(readiness)
        except (NoMatches, QueryError):
            pass
        self._sync_provider_model_section_emphasis()

    def _sync_provider_model_section_emphasis(self) -> None:
        section = self.query_one("#console-settings-provider-model-section", Vertical)
        if self._is_model_setup_mode():
            section.add_class("console-settings-primary-section")
        else:
            section.remove_class("console-settings-primary-section")

    def _build_draft(self) -> ConsoleSessionSettings:
        provider = self._select_value_text(
            self.query_one("#console-settings-provider", Select).value
        )
        return ConsoleSessionSettings(
            provider=provider,
            model=self._current_model_value(),
            base_url=self._current_base_url_value(provider),
            temperature=self._parse_float_input(
                "console-settings-temperature", self._settings.temperature
            ),
            top_p=self._parse_float_input(
                "console-settings-top-p", self._settings.top_p
            ),
            min_p=self._parse_optional_float_input("console-settings-min-p"),
            top_k=self._parse_optional_int_input("console-settings-top-k"),
            max_tokens=self._parse_optional_int_input("console-settings-max-tokens"),
            seed=self._parse_optional_int_input("console-settings-seed"),
            presence_penalty=self._parse_optional_float_input(
                "console-settings-presence-penalty"
            ),
            frequency_penalty=self._parse_optional_float_input(
                "console-settings-frequency-penalty"
            ),
            reasoning_effort=self._parse_optional_choice_input(
                "console-settings-reasoning-effort"
            ),
            reasoning_summary=self._parse_optional_choice_input(
                "console-settings-reasoning-summary"
            ),
            verbosity=self._parse_optional_choice_input("console-settings-verbosity"),
            thinking_effort=self._parse_optional_choice_input(
                "console-settings-thinking-effort"
            ),
            thinking_budget_tokens=self._parse_optional_int_input(
                "console-settings-thinking-budget-tokens"
            ),
            streaming=self._effective_streaming_value(),
            character_label=self._settings.character_label,
            system_prompt=self._settings.system_prompt,
            source=self._settings.source,
            pinned_prefill=self._settings.pinned_prefill,
        )

    def _sync_model_controls(self, provider: str, current_model: str | None) -> None:
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        model_custom = self.query_one("#console-settings-model-custom", Button)
        raw_model_is_blank = type(current_model) is str and current_model == ""
        current_model = normalize_console_model_value(current_model)
        model_options = self._model_select_options(provider, current_model)
        if model_options:
            model_select.set_options(model_options)
            option_values = {str(value) for _, value in model_options}
            selected = (
                current_model
                if current_model in option_values
                else ("" if raw_model_is_blank else str(model_options[0][1]))
            )
            model_select.value = selected or Select.NULL
            use_model_select = self._should_use_model_select(
                provider, selected, model_options
            )
            model_select.disabled = not use_model_select
            model_select.display = use_model_select
            model_input.disabled = True
            model_input.display = not use_model_select
            model_input.value = selected
            model_custom.label = "Custom model"
            model_custom.disabled = False
            model_custom.display = True
            picker = self.query_one(
                "#console-settings-model-picker", ModelSearchPicker
            )
            picker.refresh_provider(
                provider,
                current_model="" if raw_model_is_blank else selected,
            )
            return

        fallback = current_model or ""
        model_select.set_options(
            [("No configured models", _NO_CONFIGURED_MODELS_VALUE)]
        )
        model_select.value = Select.NULL
        model_select.disabled = True
        model_select.display = False
        model_input.value = fallback
        model_input.disabled = False
        model_input.display = True
        model_custom.label = "Custom model"
        model_custom.disabled = False
        model_custom.display = True
        picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        picker.refresh_provider(
            provider,
            current_model="" if raw_model_is_blank else fallback or None,
        )

    def _toggle_manual_model_input(self) -> None:
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        model_custom = self.query_one("#console-settings-model-custom", Button)

        if model_input.display:
            if model_input.disabled:
                model_input.disabled = False
                model_custom.label = "Model list"
                model_input.focus()
                self._sync_readiness_display()
                return
            provider = self._select_value_text(
                self.query_one("#console-settings-provider", Select).value
            )
            current_model = normalize_console_model_value(model_input.value)
            self._sync_model_controls(provider, current_model)
            self._sync_readiness_display()
            if model_select.display and not model_select.disabled:
                model_select.focus()
            else:
                model_custom.focus()
            return

        model_input.value = (
            normalize_console_model_value(self._select_value_text(model_select.value))
            or ""
        )
        model_select.display = False
        model_select.disabled = True
        model_input.display = True
        model_input.disabled = False
        model_custom.label = "Model list"
        model_input.focus()
        self._sync_readiness_display()

    def _focus_model_control(self) -> None:
        picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        picker.focus_input()
        picker.query_one("#model-search-picker-input", Input).scroll_visible(
            animate=False,
            immediate=True,
            force=True,
        )

    def _provider_select_options(self) -> list[tuple[str, str]]:
        """Return provider options labeled with shared catalog display names.

        Option values stay raw provider config keys (task-191); only the
        rendered labels change, and the ``(WIP)`` marker from the underlying
        option builder is preserved.
        """
        options: list[tuple[str, str]] = []
        for option in build_console_provider_options(self._providers_models):
            label = provider_display_name(option.value)
            if option.label.endswith(" (WIP)"):
                label = f"{label} (WIP)"
            options.append((label, option.value))
        if self._settings.provider and self._settings.provider not in {
            value for _, value in options
        }:
            options.append(
                (
                    provider_display_name(self._settings.provider),
                    self._settings.provider,
                )
            )
        return options

    def _model_select_options(
        self, provider: str, current_model: str | None
    ) -> list[tuple[str, str]]:
        options = [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, current_model
            )
        ]
        option_values = {value for _, value in options}
        for model_id in self._discovered_model_ids.get(provider, ()):
            normalized = normalize_console_model_value(model_id)
            if normalized and normalized not in option_values:
                option_values.add(normalized)
                # Server-supplied text: escape so Rich-markup-like ids cannot
                # style or spoof the option label (PR #608 review).
                options.append((escape_markup(normalized), normalized))
        return options

    def _configured_model_select_options(self, provider: str) -> list[tuple[str, str]]:
        return [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, None
            )
        ]

    def _should_use_model_select(
        self,
        provider: str,
        selected_model: str | None,
        model_options: list[tuple[str, str]],
    ) -> bool:
        """Return whether the model list should be an interactive Select.

        The approved single-model fix only applies to the steady-state local
        runtime case where the user already has that exact model selected. Other
        single-option states still need an interactive list because they are
        recovery/setup flows, custom/freeform providers, or provider-switch
        transitions where saving must capture the resolved model.
        """
        if not model_options:
            return False
        if len(model_options) > 1:
            return True

        provider_key = provider_config_key(provider)
        if provider_key == "custom":
            return True
        configured_values = {
            str(value) for _, value in self._configured_model_select_options(provider)
        }
        selected_model = normalize_console_model_value(selected_model)

        if self._focus_model:
            return True
        if provider != self._settings.provider:
            if provider_key in {"llama_cpp", "local_llamacpp"}:
                return True
            return bool(selected_model and selected_model not in configured_values)

        settings_model = normalize_console_model_value(self._settings.model)
        if not settings_model:
            return True
        if provider_key not in {"llama_cpp", "local_llamacpp", "openai"}:
            return True
        if selected_model and selected_model not in configured_values:
            return True
        return not selected_model or selected_model != settings_model

    def _set_provider_model_draft(self, provider: str, value: object) -> None:
        """Store a per-provider draft model, normalized once at this boundary."""
        self._provider_model_drafts[provider] = normalize_console_model_value(value)

    def _provider_model_draft(self, provider: str) -> str | None:
        """Return the already-normalized draft model stored for a provider."""
        return self._provider_model_drafts.get(provider)

    def _store_current_model_for_provider(self, provider: str) -> None:
        if provider:
            picker = self.query_one(
                "#console-settings-model-picker", ModelSearchPicker
            )
            picker_input = picker.query_one("#model-search-picker-input", Input)
            current_model = self._current_model_value()
            if current_model is None and picker_input.value == "":
                if not (
                    provider in self._provider_model_drafts
                    and self._provider_model_drafts[provider] is None
                ):
                    self._provider_model_drafts[provider] = ""
            else:
                self._set_provider_model_draft(provider, current_model)

    def _store_current_base_url_for_provider(self, provider: str) -> None:
        if provider and self._provider_uses_base_url(provider):
            self._provider_base_url_drafts[provider] = self.query_one(
                "#console-settings-base-url", Input
            ).value

    def _model_for_provider(self, provider: str) -> str | None:
        if provider in self._provider_model_drafts:
            stored_value = self._provider_model_drafts[provider]
            if stored_value is None or (
                type(stored_value) is str and stored_value == ""
            ):
                return ""
            stored_model = normalize_console_model_value(stored_value)
            if stored_model:
                return stored_model
        configured_model = self._default_model_for_provider(provider)
        if configured_model:
            return configured_model
        if provider == self._settings.provider:
            settings_model = normalize_console_model_value(self._settings.model)
            if settings_model:
                return settings_model
        configured_model_options = self._configured_model_select_options(provider)
        if configured_model_options:
            return configured_model_options[0][1]
        return None

    def _default_model_for_provider(self, provider: str) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        for key in ("model", "api_model", "default_model"):
            configured_model = normalize_console_model_value(provider_settings.get(key))
            if configured_model:
                return configured_model
        return None

    def _sync_base_url_control(self, provider: str, base_url: str | None) -> None:
        base_url_input = self.query_one("#console-settings-base-url", Input)
        uses_base_url = self._provider_uses_base_url(provider)
        base_url_input.value = base_url or ""
        base_url_input.disabled = not uses_base_url
        base_url_input.display = uses_base_url

    def _current_base_url_value(self, provider: str) -> str | None:
        if not self._provider_uses_base_url(provider):
            return None
        return self.query_one("#console-settings-base-url", Input).value.strip() or None

    def _base_url_for_provider(self, provider: str) -> str | None:
        if not self._provider_uses_base_url(provider):
            return None
        if provider in self._provider_base_url_drafts:
            return self._provider_base_url_drafts[provider] or None
        if provider == self._settings.provider and self._settings.base_url:
            return self._initial_base_url_for_provider(
                provider, self._settings.base_url
            )
        return self._default_base_url_for_provider(provider)

    def _provider_uses_base_url(self, provider: str) -> bool:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        return provider_key in URL_BASED_PROVIDER_KEYS or any(
            key in provider_settings
            for key in ("api_base_url", "api_url", "base_url", "api_base")
        )

    def _provider_settings(self, provider_key: str) -> Mapping[str, object]:
        api_settings = self._app_config.get("api_settings", {})
        if not isinstance(api_settings, Mapping):
            return {}
        for configured_provider, configured_settings in api_settings.items():
            if provider_config_key(
                str(configured_provider)
            ) == provider_key and isinstance(configured_settings, Mapping):
                return configured_settings
        return {}

    def _default_base_url_for_provider(self, provider: str) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        base_url = first_configured_endpoint(provider_settings)
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            return normalize_llamacpp_base_url(base_url or DEFAULT_LLAMACPP_BASE_URL)
        return base_url

    def _initial_base_url_for_provider(
        self, provider: str, session_base_url: str | None
    ) -> str | None:
        provider_key = provider_config_key(provider)
        provider_settings = self._provider_settings(provider_key)
        configured_base_url = self._default_base_url_for_provider(provider)
        session_base_url = self._normalized_base_url_for_provider(
            provider_key, session_base_url
        )
        if not session_base_url:
            return configured_base_url
        if configured_base_url and self._matches_lower_priority_configured_endpoint(
            provider_key,
            session_base_url,
            provider_settings,
        ):
            return configured_base_url
        return session_base_url

    def _matches_lower_priority_configured_endpoint(
        self,
        provider_key: str,
        session_base_url: str,
        provider_settings: Mapping[str, object],
    ) -> bool:
        configured_endpoint = self._normalized_base_url_for_provider(
            provider_key,
            first_configured_endpoint(provider_settings),
        )
        if not configured_endpoint:
            return False
        session_identity = self._endpoint_identity_for_provider(
            provider_key, session_base_url
        )
        configured_identity = self._endpoint_identity_for_provider(
            provider_key, configured_endpoint
        )
        if session_identity == configured_identity:
            return False
        for key in ("api_url", "base_url", "api_base", "api_endpoint", "endpoint"):
            lower_priority_endpoint = self._normalized_base_url_for_provider(
                provider_key,
                provider_settings.get(key),
            )
            if (
                lower_priority_endpoint
                and session_identity
                == self._endpoint_identity_for_provider(
                    provider_key, lower_priority_endpoint
                )
            ):
                return True
        return False

    @staticmethod
    def _normalized_base_url_for_provider(
        provider_key: str, base_url: object | None
    ) -> str | None:
        value = str(base_url or "").strip()
        if not value:
            return None
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            return normalize_llamacpp_base_url(value)
        return value

    @staticmethod
    def _endpoint_identity_for_provider(provider_key: str, base_url: str | None) -> str:
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            base_url = normalize_llamacpp_base_url(base_url)
        return normalize_generic_endpoint_for_compare(base_url)

    @staticmethod
    def _first_string(*values: object) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _current_model_value(self) -> str | None:
        try:
            picker = self.query_one("#console-settings-model-picker", ModelSearchPicker)
        except (NoMatches, QueryError):
            picker = None
        if picker is not None:
            return normalize_console_model_value(picker.value)
        model_select = self.query_one("#console-settings-model-select", Select)
        model_input = self.query_one("#console-settings-model-input", Input)
        if model_select.display and not model_select.disabled:
            return normalize_console_model_value(
                self._select_value_text(model_select.value)
            )
        return normalize_console_model_value(model_input.value)

    @staticmethod
    def _select_value_text(value: object) -> str:
        """Normalize Textual's blank-select sentinel without stringifying it."""
        return "" if value is Select.NULL or value is None else str(value)

    def _context_label(self) -> str:
        label = self._context_estimate.label.strip() or "unknown"
        return label if "token" in label.lower() else f"{label} tokens"

    def _build_context_policy_overrides(self) -> ConsoleContextPolicyOverrides:
        """Build sparse per-conversation overrides from the context draft."""
        if self._context_overrides_reset:
            return ConsoleContextPolicyOverrides()
        inherited = self._context_state.inherited_policy
        budget_mode = ContextBudgetMode(
            self._select_value_text(
                self.query_one("#console-context-budget-mode", Select).value
            )
        )
        custom_text = self.query_one(
            "#console-context-custom-budget", Input
        ).value.strip()
        custom_budget = int(custom_text) if custom_text else None
        if budget_mode is ContextBudgetMode.CUSTOM and custom_budget is None:
            raise ContextPolicyError(
                "Custom conversation budget requires a positive token value."
            )
        if custom_budget is not None and custom_budget <= 0:
            raise ContextPolicyError(
                "Custom conversation budget must be a positive token value."
            )
        compaction_mode = ContextCompactionMode(
            self._select_value_text(
                self.query_one("#console-context-compaction-mode", Select).value
            )
        )
        compaction_representation = ContextCompactionRepresentation(
            self._select_value_text(
                self.query_one(
                    "#console-context-compaction-representation", Select
                ).value
            )
        )
        trigger_ratio = self._context_percent_ratio(
            "console-context-trigger-percent", "Trigger"
        )
        target_ratio = self._context_percent_ratio(
            "console-context-target-percent", "Compact-toward target"
        )
        summary_max = self._positive_context_int(
            "console-context-summary-max", "Summary response max tokens"
        )
        failure_behavior = CompactionFailureBehavior(
            self._select_value_text(
                self.query_one("#console-context-failure-behavior", Select).value
            )
        )
        carry_forward = ContextCarryForwardMode(
            self._select_value_text(
                self.query_one("#console-context-carry-forward", Select).value
            )
        )
        # Validate the complete draft even when only one side of a pair will
        # be persisted as a sparse override.
        ConsoleContextPolicyDefaults(
            budget_mode=budget_mode,
            custom_budget_tokens=custom_budget,
            compaction_mode=compaction_mode,
            compaction_representation=compaction_representation,
            trigger_ratio=trigger_ratio,
            target_ratio=target_ratio,
            summary_max_tokens=summary_max,
            failure_behavior=failure_behavior,
            carry_forward_mode=carry_forward,
        )

        def changed(value: object, inherited_value: object) -> object | None:
            return None if value == inherited_value else value

        return ConsoleContextPolicyOverrides(
            budget_mode=changed(budget_mode, inherited.budget_mode),  # type: ignore[arg-type]
            custom_budget_tokens=changed(  # type: ignore[arg-type]
                custom_budget, inherited.custom_budget_tokens
            ),
            compaction_mode=changed(  # type: ignore[arg-type]
                compaction_mode, inherited.compaction_mode
            ),
            compaction_representation=changed(  # type: ignore[arg-type]
                compaction_representation,
                inherited.compaction_representation,
            ),
            trigger_ratio=changed(trigger_ratio, inherited.trigger_ratio),  # type: ignore[arg-type]
            target_ratio=changed(target_ratio, inherited.target_ratio),  # type: ignore[arg-type]
            summary_max_tokens=changed(  # type: ignore[arg-type]
                summary_max, inherited.summary_max_tokens
            ),
            failure_behavior=changed(  # type: ignore[arg-type]
                failure_behavior, inherited.failure_behavior
            ),
            carry_forward_mode=changed(  # type: ignore[arg-type]
                carry_forward, inherited.carry_forward_mode
            ),
        )

    def _sync_visual_representation_availability(self) -> None:
        """Disable vision-only choices while preserving the saved intent."""

        try:
            select = self.query_one(
                "#console-context-compaction-representation", Select
            )
            options = select.query_one(OptionList)
            status = self.query_one("#console-context-representation-status", Static)
        except (NoMatches, QueryError):
            return
        model = self._current_model_value() or ""
        try:
            available = bool(model) and is_vision_capable(self._active_provider, model)
        except Exception:
            available = False
        for index in (1, 2):
            if available:
                options.enable_option_at_index(index)
            else:
                options.disable_option_at_index(index)
        if available:
            status.update(
                "Visual pages stay on-device until this request is sent; recent turns "
                "stay text. Provider image token cost is model-specific and may be estimated."
            )
            select.tooltip = "Choose how older conversation turns are compacted."
            return
        selected = self._select_value_text(select.value)
        effective = (
            " Saved visual intent will use Text summary for this model."
            if selected
            in {
                ContextCompactionRepresentation.VISUAL_TRANSCRIPT.value,
                ContextCompactionRepresentation.HYBRID.value,
            }
            else ""
        )
        status.update(
            f"Visual transcript and Hybrid require a vision-capable model.{effective}"
        )
        select.tooltip = "Vision-only choices are unavailable for the current model."

    def _context_percent_ratio(self, input_id: str, label: str) -> float:
        text = self.query_one(f"#{input_id}", Input).value.strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise ContextPolicyError(f"{label} must be a percentage.") from exc
        if value <= 0 or value >= 100:
            raise ContextPolicyError(f"{label} must be between 0 and 100.")
        return value / 100.0

    def _positive_context_int(self, input_id: str, label: str) -> int:
        text = self.query_one(f"#{input_id}", Input).value.strip()
        try:
            value = int(text)
        except ValueError as exc:
            raise ContextPolicyError(f"{label} must be a positive integer.") from exc
        if value <= 0:
            raise ContextPolicyError(f"{label} must be a positive integer.")
        return value

    def _context_policy_provenance_label(self) -> str:
        override_names = tuple(self._context_state.overrides.to_dict())
        if not override_names:
            return "Inherited from Console Behavior defaults."
        labels = {
            "budget_mode": "budget mode",
            "custom_budget_tokens": "custom tokens",
            "compaction_mode": "compaction",
            "compaction_representation": "representation",
            "trigger_ratio": "trigger",
            "target_ratio": "target",
            "summary_max_tokens": "summary max",
            "failure_behavior": "failure behavior",
            "carry_forward_mode": "carry forward",
        }
        fields = ", ".join(labels[name] for name in override_names)
        return f"Conversation overrides: {fields}. Other values are inherited."

    def _context_validation_label(self) -> str:
        resolved = self._context_state.resolved_policy
        messages = (*resolved.validation_errors, *resolved.warnings)
        if messages:
            return " ".join(messages)
        if resolved.safety_verified and self._context_state.model_window_verified:
            return "Capacity is verified for the selected model."
        if self._context_state.model_window_tokens is not None:
            return (
                "Estimated fallback only; model capacity is unverified. "
                "Set the actual context window in F9 Settings > Providers & Models."
            )
        return (
            "Model limit unknown; automatic safety cannot be verified. "
            "Set the context window in F9 Settings > Providers & Models."
        )

    def _model_window_label(self) -> str:
        """Label fallback model windows as estimates instead of verified facts."""
        if (
            self._context_state.model_window_tokens is not None
            and not self._context_state.model_window_verified
        ):
            return "Model window (est.)"
        return "Model window"

    def _memory_metadata_label(self) -> str:
        effective = self._context_state.effective_memory
        if effective.kind is EffectiveMemoryKind.RAW:
            return "No conversation memory. Transcript remains authoritative."
        if effective.kind is EffectiveMemoryKind.LEGACY_PREFIX:
            assert effective.legacy is not None
            return (
                "Legacy manual prefix memory · "
                f"Boundary {effective.legacy.boundary_message_id} · "
                "provenance unavailable"
            )
        memory = effective.memory
        scope = effective.scope
        if memory is None or scope is None:
            return "No conversation memory. Transcript remains authoritative."
        if effective.kind is EffectiveMemoryKind.GENERATED_RANGE:
            kind = (
                "Manual range memory · "
                f"Start {scope.selection_anchor_message_id} · "
                f"End {memory.boundary_message_id}"
            )
        elif scope.origin_kind.value == "manual_rewind":
            kind = "Manual prefix memory"
        else:
            kind = "Automatic prefix memory"
        return (
            f"{kind} · {memory.created_at} · {memory.provider}/{memory.model} · "
            f"prompt r{memory.prompt_revision} · "
            f"{memory.before_tokens:,} → {memory.after_tokens:,} tokens"
        )

    def _memory_review_text(self) -> str:
        effective = self._context_state.effective_memory
        if effective.memory is not None:
            return effective.memory.summary_text
        if effective.legacy is not None:
            return effective.legacy.summary_text
        return "No conversation memory is active on this branch."

    def _sources_label(self) -> str:
        if self._context_estimate.staged_context_summary.strip():
            return self._context_estimate.staged_context_summary.strip()
        return "None"

    def _identity_current_label(self) -> str:
        character = sanitize_character_display_label(
            self._settings.character_label,
            max_characters=180,
        )
        return f"Character: {character}" if character else "Assistant: General"

    @staticmethod
    def _format_value(value: object) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _format_percent(ratio: float) -> str:
        """Round-trip a stored ratio as a human percentage draft."""
        return f"{ratio * 100:.12g}"

    def _parse_float_input(self, input_id: str, fallback: float) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return fallback
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    def _required_sampling_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.query_one("#console-settings-temperature", Input).value.strip():
            errors.append("Temperature is required.")
        if not self.query_one("#console-settings-top-p", Input).value.strip():
            errors.append("Top P is required.")
        return errors

    def _provider_choice_input_errors(self) -> list[str]:
        errors: list[str] = []
        for label, input_id, _placeholder in PROVIDER_CHOICE_INPUTS:
            raw_value = self.query_one(f"#{input_id}", Input).value.strip()
            if raw_value and not validate_text_input(
                raw_value,
                max_length=PROVIDER_CHOICE_INPUT_MAX_LENGTH,
                allow_html=False,
            ):
                errors.append(f"{label} contains unsupported text.")
        return errors

    def _parse_optional_float_input(self, input_id: str) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return None
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    def _parse_optional_int_input(self, input_id: str) -> object:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            return raw_value

    def _parse_optional_text_input(self, input_id: str) -> str | None:
        raw_value = self.query_one(f"#{input_id}", Input).value.strip()
        return raw_value or None

    def _parse_optional_choice_input(self, input_id: str) -> str | None:
        raw_value = self._parse_optional_text_input(input_id)
        return raw_value.lower() if raw_value else None
