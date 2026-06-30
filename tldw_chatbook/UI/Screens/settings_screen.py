"""Settings destination shell for global app preferences."""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import logging
import os
from pathlib import Path
import re
import tomllib

from rich.cells import cell_len
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.events import DescendantFocus, Key
from textual.message_pump import NoActiveAppError
from textual.reactive import reactive
from textual.strip import Strip
from textual.widgets import Button, Input, Rule, Select, SelectionList, Static, TextArea

from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Chat.console_provider_support import (
    ConsoleProviderCatalogEntry,
    supported_console_provider_catalog,
)
from ...Chat.console_session_settings import CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
from ...ACP_Interop.runtime_session import ACPRuntimeSessionState
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...Sync_Interop.sync_promotion_state import SyncPromotionState, build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Sync_Interop.manual_sync_control import ManualSyncPreview, ManualSyncRunResult
from ...Workspaces.display_state import LIBRARY_WORKSPACE_VISIBILITY_COPY
from ...Widgets.destination_workbench import DestinationModeStrip
from ...config import (
    BASE_DATA_DIR_CLI,
    DEFAULT_CONFIG_FROM_TOML,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    coerce_bool_setting,
    coerce_int_setting,
    save_setting_to_cli_config,
)
from ...Utils.input_validation import (
    provider_api_key_validation_error,
    sanitize_string,
    validate_number_range,
    validate_text_input,
    validate_url,
)
from ...Utils.console_background_effects import (
    CONSOLE_BACKGROUND_EFFECTS,
    CONSOLE_BACKGROUND_INTENSITIES,
    CONSOLE_BACKGROUND_SCOPES,
    DEFAULT_CONSOLE_BACKGROUND_FPS,
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
    normalize_console_background_effects,
)
from ...Utils.path_validation import validate_path_simple
from ..Navigation.base_app_screen import BaseAppScreen
from .provider_model_resolution import EffectiveProviderModel, resolve_effective_provider_model
from .settings_config_adapter import SettingsConfigAdapter, redact_secret_text
from .settings_config_models import (
    SettingsCategoryId,
    SettingsCategorySummary,
    SettingsDomainCategoryContract,
    SettingsDraft,
    SettingsOwnershipRecord,
)
from .settings_appearance_defaults import (
    SettingsAppearanceDefaults,
    build_appearance_save_sections,
    load_appearance_defaults,
    validate_appearance_defaults,
)
from .settings_library_rag_defaults import (
    SettingsLibraryRagDefaults,
    build_library_rag_save_sections,
    load_library_rag_defaults,
    normalise_library_rag_citation_style,
    normalise_library_rag_search_mode,
    validate_library_rag_defaults,
)
from .settings_privacy_security import (
    SettingsPrivacyPosture,
    build_privacy_posture_rows,
    build_settings_privacy_posture,
)
from .settings_storage_defaults import (
    STORAGE_FIELD_LABELS,
    SettingsStorageDefaults,
    build_storage_check_rows,
    build_storage_save_sections,
    load_storage_defaults,
    validate_storage_defaults,
)
from ..Navigation.main_navigation import NavigateToScreen


logger = logging.getLogger(__name__)

MAX_CATEGORY_SEARCH_QUERY_CHARS = 80
PROVIDER_ENDPOINT_KEYS = ("api_base_url", "api_base", "base_url", "api_url", "endpoint")
PROVIDER_MODEL_PROFILE_FIELD_KEYS = {
    "model_profile_temperature": "temperature",
    "model_profile_top_p": "top_p",
    "model_profile_min_p": "min_p",
    "model_profile_top_k": "top_k",
    "model_profile_max_tokens": "max_tokens",
    "model_profile_seed": "seed",
    "model_profile_presence_penalty": "presence_penalty",
    "model_profile_frequency_penalty": "frequency_penalty",
    "model_profile_reasoning_effort": "reasoning_effort",
    "model_profile_reasoning_summary": "reasoning_summary",
    "model_profile_verbosity": "verbosity",
    "model_profile_thinking_effort": "thinking_effort",
    "model_profile_thinking_budget_tokens": "thinking_budget_tokens",
    "model_profile_streaming": "streaming",
}
REASONING_EFFORT_OPTIONS = frozenset({"", "none", "minimal", "low", "medium", "high", "xhigh"})
REASONING_SUMMARY_OPTIONS = frozenset({"", "auto", "concise", "detailed", "none"})
VERBOSITY_OPTIONS = frozenset({"", "low", "medium", "high"})
THINKING_EFFORT_OPTIONS = frozenset({"", "off", "low", "medium", "high", "xhigh", "max"})
OPENAI_REASONING_PROVIDER_KEYS = frozenset({"openai"})
ANTHROPIC_THINKING_PROVIDER_KEYS = frozenset({"anthropic"})
OPENAI_REASONING_PROFILE_FIELD_KEYS = frozenset(
    {
        "model_profile_reasoning_effort",
        "model_profile_reasoning_summary",
        "model_profile_verbosity",
    }
)
ANTHROPIC_THINKING_PROFILE_FIELD_KEYS = frozenset(
    {
        "model_profile_thinking_effort",
        "model_profile_thinking_budget_tokens",
    }
)
MODEL_PROFILE_INPUT_PLACEHOLDERS = {
    "model_profile_temperature": "0.0 - 2.0",
    "model_profile_top_p": "0.0 - 1.0",
    "model_profile_min_p": "optional 0.0 - 1.0",
    "model_profile_top_k": "optional whole number",
    "model_profile_max_tokens": "optional whole number",
    "model_profile_seed": "optional whole number",
    "model_profile_presence_penalty": "-2.0 - 2.0",
    "model_profile_frequency_penalty": "-2.0 - 2.0",
    "model_profile_reasoning_effort": "none, minimal, low, medium, high, xhigh",
    "model_profile_reasoning_summary": "auto, concise, detailed, none",
    "model_profile_verbosity": "low, medium, high",
    "model_profile_thinking_effort": "off, low, medium, high, xhigh, max",
    "model_profile_thinking_budget_tokens": "optional >= 1024",
    "model_profile_streaming": "true or false",
}
PROVIDER_MANUAL_SELECT_VALUE = "__manual__"
PROVIDER_MANUAL_SELECT_LABEL = "Manual / custom provider"
MODEL_DISCOVERY_IDLE_COPY = "Discover models from configured endpoint"
MODEL_DISCOVERY_EMPTY_COPY = (
    "No discovered models yet. Use Discover models after endpoint is configured."
)
MODEL_DISCOVERY_CAPABILITY_WARNING = (
    "Capabilities unknown until saved or verified; text chat is assumed."
)
MODEL_DISCOVERY_AMBIGUOUS_PROVIDER_COPY = (
    "Multiple provider entries match this provider. Rename or remove duplicates before "
    "saving discovered models."
)
MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY = (
    "This endpoint is not OpenAI-compatible for v1 discovery. Configure a /v1 endpoint "
    "to discover models."
)
CONSOLE_BEHAVIOR_CONSOLE_KEYS = frozenset(
    {
        "collapse_large_pastes",
        "paste_collapse_threshold",
    }
)
CONSOLE_BACKGROUND_EFFECT_KEYS = frozenset(
    {
        "background_effects.enabled",
        "background_effects.effect",
        "background_effects.scope",
        "background_effects.intensity",
        "background_effects.fps",
    }
)
CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER = (
    "background_effects.enabled",
    "background_effects.effect",
    "background_effects.scope",
    "background_effects.intensity",
    "background_effects.fps",
)
CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY = (
    "Workbench scope is not available in this build; using Transcript scope."
)
TEXTUAL_WEB_URL_AUTOLINK_BREAK = "\u200b"
TEXTUAL_WEB_URL_SCHEME_RE = re.compile(r"\b(https?)://", re.IGNORECASE)
CONSOLE_BEHAVIOR_CHAT_DEFAULT_KEYS = frozenset(
    {
        "streaming",
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
    }
)
CONSOLE_BEHAVIOR_SAVE_ORDER = (
    "collapse_large_pastes",
    "paste_collapse_threshold",
    "streaming",
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
    *CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER,
)
ADVANCED_CONFIG_GUIDED_PATHS = (
    (SettingsCategoryId.PROVIDERS_MODELS, "Providers"),
    (SettingsCategoryId.CONSOLE_BEHAVIOR, "Console"),
    (SettingsCategoryId.STORAGE, "Storage"),
    (SettingsCategoryId.PRIVACY_SECURITY, "Privacy"),
    (SettingsCategoryId.DIAGNOSTICS, "Diagnostics"),
)
ADVANCED_CONFIG_GUIDED_PATH_BUTTONS = {
    f"settings-advanced-open-{category.value}": category
    for category, _label in ADVANCED_CONFIG_GUIDED_PATHS
}
API_URL_PROVIDER_KEYS = {
    "aphrodite",
    "custom",
    "custom_2",
    "koboldcpp",
    "llama_cpp",
    "local_llamacpp",
    "local_llamafile",
    "local_llm",
    "local_mlx_lm",
    "local_ollama",
    "local_vllm",
    "ollama",
    "oobabooga",
    "tabbyapi",
    "vllm",
}
SETTINGS_SOURCE_LABELS = {
    "settings_draft": "Unsaved Settings draft",
    "console_control": "Console runtime override",
    "app_reactive": "Current app selection",
    "chat_defaults": "Saved chat defaults",
    "default": "Default fallback",
}
PROVIDER_ENDPOINT_PLACEHOLDERS = {
    "anthropic": "https://api.anthropic.com",
    "custom": "https://your-openai-compatible-host/v1",
    "custom_2": "https://your-openai-compatible-host/v1",
    "google": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai/v1",
    "koboldcpp": "http://127.0.0.1:5001",
    "llama_cpp": "http://127.0.0.1:9099/v1",
    "local_llamacpp": "http://127.0.0.1:9099/v1",
    "local_ollama": "http://127.0.0.1:11434",
    "local_vllm": "http://127.0.0.1:8000/v1",
    "mistral": "https://api.mistral.ai/v1",
    "mistralai": "https://api.mistral.ai/v1",
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "oobabooga": "http://127.0.0.1:5000/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}
PROVIDER_CREDENTIAL_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
GUIDED_SETTINGS_MUTATION_CATEGORIES = frozenset(
    {
        SettingsCategoryId.PROVIDERS_MODELS,
        SettingsCategoryId.APPEARANCE,
        SettingsCategoryId.CONSOLE_BEHAVIOR,
        SettingsCategoryId.LIBRARY_RAG,
        SettingsCategoryId.STORAGE,
    }
)
SETTINGS_OVERVIEW_BOUNDARY_ROWS = (
    ("Settings role", "Settings owns persisted defaults and validation"),
    ("Console boundary", "Console owns live chat/run state"),
    ("MCP boundary", "MCP owns server and tool management"),
    ("ACP boundary", "ACP owns runtime/session setup"),
    (
        "Sync/workspace boundary",
        "Sync and workspace handoff defaults are read-only until source contracts exist",
    ),
)
SETTINGS_SERVER_SYNC_WORKSPACE_SOURCE_CONTRACTS = (
    (
        "Server profile",
        "runtime_policy.types.RuntimeSourceState via app_instance.runtime_policy.state; "
        "runtime_policy.server_context.RuntimeServerContextProvider owns active server resolution",
    ),
    (
        "Sync safety",
        "Sync_Interop.sync_scope_service.SyncScopeService.list_write_sync_promotion_states; "
        "Sync_Interop.sync_promotion_state.SyncPromotionState for display copy",
    ),
    (
        "Workspace context",
        "Workspaces.LocalWorkspaceRegistryService.get_active_workspace; "
        "Chat.console_chat_store.ConsoleChatStore.workspace_context for Console context; "
        "Workspaces.display_state.LIBRARY_WORKSPACE_VISIBILITY_COPY for Library visibility policy",
    ),
    (
        "Handoff policy",
        "Workspaces.models.WorkspaceTransferPolicy defines copy/reference/metadata-only/local-only policy; "
        "Chat.chat_handoff_models.ChatHandoffPayload carries the staged Console context",
    ),
    (
        "ACP handoff readiness",
        "ACP_Interop.runtime_session.ACPRuntimeSessionState via app_instance.get_acp_runtime_session_state",
    ),
)
SETTINGS_DOMAIN_CATEGORY_CONTRACTS = (
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.LIBRARY_RAG,
        title="Library & RAG",
        owner_destination="Library",
        source_of_truth=(
            "Library source services",
            "RAG_Search retrieval adapters",
            "Library Collections local service",
        ),
        rows=(
            ("Browse/search visibility", "global Library browse/search remains visible across workspaces"),
            ("Console eligibility", "staging source evidence is limited to the active workspace"),
            ("Retrieval defaults", "AppRAGSearchConfig.rag.search and .retriever own result limits and blend defaults"),
            (
                "Citation/snippet defaults",
                "AppRAGSearchConfig.rag.search owns citations, snippets, and context budget defaults",
            ),
        ),
        settings_can_mutate=True,
        follow_up=(
            "Settings may edit persisted retrieval/display defaults only; Library owns indexing, "
            "query execution, source browse, Collections, and Console staging actions."
        ),
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.ARTIFACTS,
        title="Artifacts",
        owner_destination="Artifacts",
        source_of_truth=("Chatbook artifact store", "Artifacts destination display state"),
        rows=(
            ("Chatbooks", "Artifacts owns Chatbook browse, details, and Console resume actions"),
            ("Settings role", "show defaults/status only; do not move artifact operations here"),
        ),
        follow_up="Follow-up: add artifact export/default controls only after Artifacts exposes a persisted preference contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.PERSONAS,
        title="Personas",
        owner_destination="Personas",
        source_of_truth=("Character/persona scope service", "Personas destination runtime handoff"),
        rows=(
            ("Runtime selection", "Personas owns character/profile selection and Console attach payloads"),
            ("Settings role", "future defaults may choose discovery/display preferences, not active persona runtime"),
        ),
        follow_up="Follow-up: add persona display/default controls after Personas exposes a persisted category source.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.SKILLS,
        title="Skills",
        owner_destination="Skills",
        source_of_truth=("Skills repository", "Skills destination validation and attach paths"),
        rows=(
            ("Skill format", "Skills owns SKILL.md import, validation, and attach behavior"),
            ("Settings role", "future defaults can cover trust/display preferences only"),
        ),
        follow_up="Follow-up: add Skills defaults after import/attach policy has a persisted source contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.SCHEDULES,
        title="Schedules",
        owner_destination="Schedules",
        source_of_truth=("Schedules destination state", "schedule run handoff context"),
        rows=(
            ("Run control", "Schedules owns run, pause, retry, and Console handoff actions"),
            ("Settings role", "future defaults may cover timezone/notification preferences only"),
        ),
        follow_up="Follow-up: add schedule defaults after Schedules exposes a dedicated settings adapter.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.WATCHLISTS,
        title="Watchlists",
        owner_destination="Watchlists",
        source_of_truth=("Watchlists local service", "watchlist run snapshot adapter"),
        rows=(
            ("Monitoring", "Watchlists owns feeds, runs, status, and recovery actions"),
            ("Settings role", "future defaults may cover polling and notification preferences only"),
        ),
        follow_up="Follow-up: add watchlist defaults after Watchlists exposes persisted polling/notification settings.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.WORKFLOWS,
        title="Workflows",
        owner_destination="Workflows",
        source_of_truth=("Workflows destination procedure state", "workflow Console handoff payloads"),
        rows=(
            ("Execution", "Workflows owns procedure inputs, dry runs, approvals, and outputs"),
            ("Settings role", "future defaults may cover execution safety preferences only"),
        ),
        follow_up="Follow-up: add workflow defaults after Workflows exposes a persisted execution-safety contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.MCP_DEFAULTS,
        title="MCP Defaults",
        owner_destination="MCP",
        source_of_truth=("Unified MCP panel", "MCP configured server target store"),
        rows=(
            ("Runtime owner", "MCP owns server/tool runtime, target management, and tool readiness"),
            ("Settings role", "show global defaults/status only; server operations stay in MCP"),
        ),
        follow_up="Follow-up: add MCP defaults only after server-first settings are exposed without flattening tools into Settings.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.ACP_DEFAULTS,
        title="ACP Defaults",
        owner_destination="ACP",
        source_of_truth=("ACP runtime session state", "ACP destination launch/session setup"),
        rows=(
            ("Runtime owner", "ACP owns runtime launch, session setup, and task/run packages"),
            ("Settings role", "show defaults/status only; ACP setup stays in ACP"),
        ),
        follow_up="Follow-up: add ACP defaults after ACP exposes a persisted runtime/session preference contract.",
    ),
)


def _build_domain_contract_by_category(
    contracts: tuple[SettingsDomainCategoryContract, ...],
) -> Mapping[SettingsCategoryId, SettingsDomainCategoryContract]:
    contracts_by_category: dict[SettingsCategoryId, SettingsDomainCategoryContract] = {}
    for contract in contracts:
        if contract.category in contracts_by_category:
            raise ValueError(
                f"Duplicate Settings domain category contract: {contract.category.value}"
            )
        contracts_by_category[contract.category] = contract
    return contracts_by_category


DOMAIN_CONTRACT_BY_CATEGORY = _build_domain_contract_by_category(
    SETTINGS_DOMAIN_CATEGORY_CONTRACTS
)
DOMAIN_SETTINGS_CATEGORY_IDS = frozenset(DOMAIN_CONTRACT_BY_CATEGORY)
_WORKSPACE_RECORD_UNSET = object()


def _textual_web_safe_url_display(value: str) -> str:
    """Break URL schemes in rendered input text without changing the stored value."""
    return TEXTUAL_WEB_URL_SCHEME_RE.sub(
        lambda match: f"{match.group(1)}{TEXTUAL_WEB_URL_AUTOLINK_BREAK}://",
        value,
    )


def _textual_web_safe_url_display_index(value: str, index: int) -> int:
    display_index = index
    for match in TEXTUAL_WEB_URL_SCHEME_RE.finditer(value):
        insertion_index = match.start(1) + len(match.group(1))
        if index >= insertion_index:
            display_index += len(TEXTUAL_WEB_URL_AUTOLINK_BREAK)
    return display_index


class SettingsURLInput(Input):
    """Render endpoint URLs without browser autolinking.

    SettingsURLInput preserves the raw ``value`` used for validation, saving,
    selection, and event handling. Only the rendered display text is adjusted by
    inserting a zero-width break after URL schemes so textual-web/browser
    terminals do not treat provider endpoint values as clickable links.

    Args:
        *args: Positional arguments forwarded to ``textual.widgets.Input``.
        **kwargs: Keyword arguments forwarded to ``textual.widgets.Input``.
    """

    @property
    def _value(self) -> Text:
        if self.password:
            return super()._value
        text = Text(
            _textual_web_safe_url_display(self.value),
            no_wrap=True,
            overflow="ignore",
            end="",
        )
        if self.highlighter is not None:
            text = self.highlighter(text)
        return text

    @property
    def content_width(self) -> int:
        if self.placeholder and not self.value:
            return cell_len(self.placeholder)
        return self._value.cell_len + 1

    def _display_index(self, index: int) -> int:
        if self.password:
            return index
        return _textual_web_safe_url_display_index(self.value, index)

    def render_line(self, y: int) -> Strip:
        if y != 0:
            return Strip.blank(self.size.width, self.rich_style)

        console = self.app.console
        console_options = self.app.console_options
        max_content_width = self.scrollable_content_region.width

        if not self.value:
            placeholder = Text(self.placeholder, justify="left", end="")
            placeholder.stylize(self.get_component_rich_style("input--placeholder"))
            if self.has_focus:
                cursor_style = self.get_component_rich_style("input--cursor")
                if self._cursor_visible:
                    if len(placeholder) == 0:
                        placeholder = Text(" ", end="")
                    placeholder.stylize(cursor_style, 0, 1)

            strip = Strip(
                console.render(
                    placeholder,
                    console_options.update_width(max_content_width + 1),
                )
            )
        else:
            result = self._value

            value = self.value
            value_length = len(value)
            suggestion = self._suggestion
            show_suggestion = len(suggestion) > value_length and self.has_focus
            if show_suggestion:
                result += Text(
                    suggestion[value_length:],
                    self.get_component_rich_style("input--suggestion"),
                    end="",
                )

            if self.has_focus:
                if not self.selection.is_empty:
                    start, end = self.selection
                    start, end = sorted((start, end))
                    selection_style = self.get_component_rich_style("input--selection")
                    result.stylize_before(
                        selection_style,
                        self._display_index(start),
                        self._display_index(end),
                    )

                if self._cursor_visible:
                    cursor_style = self.get_component_rich_style("input--cursor")
                    cursor = self._display_index(self.cursor_position)
                    if not show_suggestion and self.cursor_at_end:
                        result.pad_right(1)
                    result.stylize(cursor_style, cursor, cursor + 1)

            segments = list(
                console.render(result, console_options.update_width(self.content_width))
            )

            strip = Strip(segments)
            scroll_x, _ = self.scroll_offset
            strip = strip.crop(scroll_x, scroll_x + max_content_width + 1)
            strip = strip.extend_cell_length(max_content_width + 1)

        return strip.apply_style(self.rich_style)


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    BINDINGS = [
        ("s", "settings_save_category", "Save Settings category"),
        ("r", "settings_revert_category", "Revert Settings category"),
        ("t", "settings_test_category", "Test Settings category"),
    ]

    active_category = reactive(SettingsCategoryId.OVERVIEW.value, recompose=True)
    category_search_query = reactive("")
    server_sync_workspace_handoff_rows = reactive((), recompose=True)
    manual_sync_rows = reactive((), recompose=True)

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)
        self._settings_drafts: dict[SettingsCategoryId, SettingsDraft] = {}
        self._provider_test_result = "Provider test has not run."
        self._provider_save_result = "Provider settings have not been saved this session."
        self._model_discovery_status = MODEL_DISCOVERY_IDLE_COPY
        self._model_discovery_models: tuple[object, ...] = ()
        self._model_discovery_selected_model_ids: set[str] = set()
        self._syncing_provider_endpoint = False
        self._syncing_provider_api_key = False
        self._syncing_provider_credential_env_var = False
        self._syncing_provider_model_profile = False
        self._syncing_provider_model_value = False
        self._syncing_provider_manual = False
        self._syncing_provider_selection = False
        self._syncing_console_threshold = False
        self._syncing_console_defaults = False
        self._syncing_console_background_effects = False
        self._syncing_library_rag_defaults = False
        self._syncing_appearance_defaults = False
        self._syncing_storage_defaults = False
        self._active_settings_field_id: str | None = None
        self._navigation_provider: str | None = None
        self._navigation_model: str | None = None
        self._navigation_field: str | None = None
        self._diagnostics_validation_result = "Config validation: not run"
        self._diagnostics_reload_result = "Config reload: not run"
        self._storage_check_rows: tuple[str, ...] = (
            "Storage check: not run",
            "Run Check Storage or press t to verify local path access.",
        )
        self._privacy_check_rows: tuple[str, ...] = (
            "Privacy check: not run",
            "Run Check Privacy or press t to verify redacted secret status.",
        )
        self._console_behavior_result = "Console behavior settings have not been saved this session."
        self._console_behavior_saved_this_session = False
        self._library_rag_result = "Library/RAG defaults have not been saved this session."
        self._appearance_result = "Appearance defaults have not been saved this session."
        self._storage_result = "Storage defaults have not been saved this session."
        self._advanced_config_result = "Advanced config validation: not run"
        self._advanced_config_validated_text: str | None = None
        self._ownership_by_category_cache = self._build_ownership_by_category()
        self.server_sync_workspace_handoff_rows = (
            self._server_sync_workspace_handoff_loading_rows()
        )
        self.manual_sync_rows = self._manual_sync_loading_rows()

    def on_mount(self) -> None:
        super().on_mount()
        self._queue_server_sync_workspace_handoff_refresh()
        self._queue_manual_sync_refresh()

    def on_screen_resume(self) -> None:
        self._queue_server_sync_workspace_handoff_refresh()
        self._queue_manual_sync_refresh()

    def _queue_server_sync_workspace_handoff_refresh(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        self._refresh_server_sync_workspace_handoff_rows()

    def _queue_manual_sync_refresh(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        self._refresh_manual_sync_rows()

    @staticmethod
    def _server_sync_workspace_handoff_loading_rows() -> tuple[tuple[str, str], ...]:
        return (
            ("Active server profile", "Loading Settings source contracts"),
            ("Local/server authority", "Loading Settings source contracts"),
            ("Sync safety", "Loading Settings source contracts"),
            ("Sync recovery", "Loading Settings source contracts"),
            ("Workspace default", "Loading Settings source contracts"),
            ("Library visibility", LIBRARY_WORKSPACE_VISIBILITY_COPY),
            (
                "Handoff policy",
                "copy/reference/metadata-only by source policy; "
                "Console staging is limited to the active workspace",
            ),
            ("ACP handoff readiness", "Loading Settings source contracts"),
        )

    @staticmethod
    def _manual_sync_loading_rows() -> tuple[tuple[str, str], ...]:
        return (
            ("Manual sync status", "loading"),
            ("Manual sync preview", "Loading manual Sync v2 preview."),
            ("Pending outgoing", "Loading"),
        )

    def _category_summaries(self) -> tuple[SettingsCategorySummary, ...]:
        return (
            SettingsCategorySummary(
                SettingsCategoryId.OVERVIEW,
                "Overview",
                "Readiness, storage, privacy, Console behavior, diagnostics.",
                "Active",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.PROVIDERS_MODELS,
                "Providers & Models",
                "Default provider, model, and readiness shared with Console.",
                "Shared",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.APPEARANCE,
                "Appearance",
                "Theme, density, and visual defaults shared with the app shell.",
                "Guided",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.STORAGE,
                "Storage",
                "Config path, local databases, and file locations.",
                "Guided",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.PRIVACY_SECURITY,
                "Privacy & Security",
                "Secrets, encryption, redaction, and local privacy boundaries.",
                "Local",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.CONSOLE_BEHAVIOR,
                "Console Behavior",
                "Composer, large paste handling, and chat-flow defaults.",
                "Console",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.LIBRARY_RAG,
                "Library & RAG",
                "Source search, retrieval, citations, snippets, and Console evidence defaults.",
                "Guided",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.ARTIFACTS,
                "Artifacts",
                "Chatbooks, saved outputs, and artifact resume defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.PERSONAS,
                "Personas",
                "Character/persona discovery and Console attach defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.SKILLS,
                "Skills",
                "Skill import, validation, trust, and attach defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.SCHEDULES,
                "Schedules",
                "Schedule run, notification, and Console follow defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.WATCHLISTS,
                "Watchlists",
                "Feed monitoring, polling, notification, and run defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.WORKFLOWS,
                "Workflows",
                "Procedure, dry-run, approval, and execution safety defaults.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.MCP_DEFAULTS,
                "MCP Defaults",
                "Server/tool management defaults without owning MCP runtime operations.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.ACP_DEFAULTS,
                "ACP Defaults",
                "ACP runtime/session defaults without owning ACP launch operations.",
                "Read-only",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.DIAGNOSTICS,
                "Diagnostics",
                "Config validation, logs, and troubleshooting signals.",
                "Validate",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.ADVANCED_CONFIG,
                "Advanced Config",
                "Raw TOML view and expert configuration editing.",
                "Advanced",
            ),
        )

    def _category_groups(self) -> tuple[tuple[str, tuple[SettingsCategoryId, ...]], ...]:
        return (
            (
                "Core",
                (
                    SettingsCategoryId.OVERVIEW,
                    SettingsCategoryId.PROVIDERS_MODELS,
                ),
            ),
            (
                "Interface",
                (
                    SettingsCategoryId.APPEARANCE,
                    SettingsCategoryId.CONSOLE_BEHAVIOR,
                ),
            ),
            (
                "Data & Privacy",
                (
                    SettingsCategoryId.STORAGE,
                    SettingsCategoryId.PRIVACY_SECURITY,
                ),
            ),
            ("Troubleshooting", (SettingsCategoryId.DIAGNOSTICS,)),
            ("Expert", (SettingsCategoryId.ADVANCED_CONFIG,)),
            (
                "Domain Defaults",
                (
                    SettingsCategoryId.LIBRARY_RAG,
                    SettingsCategoryId.ARTIFACTS,
                    SettingsCategoryId.PERSONAS,
                    SettingsCategoryId.SKILLS,
                    SettingsCategoryId.SCHEDULES,
                    SettingsCategoryId.WATCHLISTS,
                    SettingsCategoryId.WORKFLOWS,
                    SettingsCategoryId.MCP_DEFAULTS,
                    SettingsCategoryId.ACP_DEFAULTS,
                ),
            ),
        )

    def _domain_category_contracts(self) -> tuple[SettingsDomainCategoryContract, ...]:
        return SETTINGS_DOMAIN_CATEGORY_CONTRACTS

    def _domain_contract_by_category(self) -> Mapping[
        SettingsCategoryId, SettingsDomainCategoryContract
    ]:
        return DOMAIN_CONTRACT_BY_CATEGORY

    def _domain_category_contract(self, category: SettingsCategoryId) -> SettingsDomainCategoryContract:
        try:
            return self._domain_contract_by_category()[category]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Settings domain category contract: {category.value}"
            ) from exc

    def _domain_category_ownership_records(self) -> tuple[SettingsOwnershipRecord, ...]:
        records = []
        for contract in self._domain_category_contracts():
            if contract.category is SettingsCategoryId.LIBRARY_RAG:
                records.append(
                    SettingsOwnershipRecord(
                        category=contract.category,
                        owns_config_sections=(
                            "AppRAGSearchConfig.rag.search.default_search_mode",
                            "AppRAGSearchConfig.rag.search.default_top_k",
                            "AppRAGSearchConfig.rag.search.score_threshold",
                            "AppRAGSearchConfig.rag.search.include_citations",
                            "AppRAGSearchConfig.rag.search.citation_style",
                            "AppRAGSearchConfig.rag.search.snippet_max_chars",
                            "AppRAGSearchConfig.rag.search.max_context_size",
                            "AppRAGSearchConfig.rag.retriever.fts_top_k",
                            "AppRAGSearchConfig.rag.retriever.vector_top_k",
                            "AppRAGSearchConfig.rag.retriever.hybrid_alpha",
                        ),
                        reads_runtime_state_from=contract.source_of_truth,
                        writes_allowed=True,
                        runtime_owner="Settings persisted defaults; Library runtime actions",
                        boundary_copy=(
                            "Settings owns persisted retrieval and display defaults; Library owns "
                            "indexing, query execution, source browse, Collections, and staging."
                        ),
                        recovery_copy=(
                            "Revert unsaved defaults or open Library to validate query behavior."
                        ),
                    )
                )
                continue
            records.append(
                SettingsOwnershipRecord(
                    category=contract.category,
                    owns_config_sections=(),
                    reads_runtime_state_from=contract.source_of_truth,
                    writes_allowed=contract.settings_can_mutate,
                    runtime_owner=contract.owner_destination,
                    boundary_copy=(
                        f"{contract.owner_destination} owns the live workflow; Settings shows "
                        "read-only defaults/status until a persisted source contract exists."
                    ),
                    recovery_copy=f"Open {contract.owner_destination} for workflow actions and setup.",
                    read_only_reason=contract.follow_up,
                )
            )
        return tuple(records)

    def _category_ownership_records(self) -> tuple[SettingsOwnershipRecord, ...]:
        return (
            SettingsOwnershipRecord(
                category=SettingsCategoryId.OVERVIEW,
                owns_config_sections=("global defaults", "validation status", "recovery guidance"),
                reads_runtime_state_from=(
                    "Console",
                    "MCP",
                    "ACP",
                    "sync readiness",
                    "workspace status",
                ),
                writes_allowed=False,
                runtime_owner="owning destinations",
                boundary_copy="; ".join(value for _, value in SETTINGS_OVERVIEW_BOUNDARY_ROWS),
                recovery_copy=(
                    "Open the owning category or destination before changing runtime behavior; "
                    "sync and workspace handoff defaults stay read-only until source contracts exist."
                ),
                read_only_reason="Overview summarizes status and ownership only.",
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.PROVIDERS_MODELS,
                owns_config_sections=(
                    "chat_defaults.provider",
                    "chat_defaults.model",
                    "api_settings.<provider>.endpoint",
                    "api_settings.<provider>.api_key",
                    "api_settings.<provider>.api_key_env_var",
                    "api_settings.<provider>.model_defaults.<model>",
                ),
                reads_runtime_state_from=("Console provider readiness",),
                writes_allowed=True,
                runtime_owner="Settings persisted defaults; Console runtime selection",
                boundary_copy=(
                    "Provider, default model, endpoint, local config API key, credential "
                    "source, and selected "
                    "provider+model profile defaults are shared with Console."
                ),
                recovery_copy=(
                    "Test provider readiness, then use Console Defaults for sampling and transport settings."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.APPEARANCE,
                owns_config_sections=(
                    "general.default_theme",
                    "general.palette_theme_limit",
                    "web_server.font_size",
                    "appearance.density",
                    "appearance.animations_enabled",
                    "appearance.smooth_scrolling",
                ),
                reads_runtime_state_from=("app theme", "Customize destination"),
                writes_allowed=True,
                runtime_owner="Settings persisted defaults; Customize full theme editor",
                boundary_copy=(
                    "Settings owns launch/web visual defaults; Customize owns full "
                    "theme editing and deeper visual preview."
                ),
                recovery_copy=(
                    "Preview applies runtime-safe values for this session only; Save persists "
                    "defaults, Revert restores loaded values."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.STORAGE,
                owns_config_sections=(
                    "database.USER_DB_BASE_DIR",
                    "database.chachanotes_db_path",
                    "database.prompts_db_path",
                    "database.media_db_path",
                    "database.research_db_path",
                    "database.writing_db_path",
                    "database.library_collections_db_path",
                    "database.workspaces_db_path",
                ),
                reads_runtime_state_from=("local filesystem", "configured database paths"),
                writes_allowed=True,
                runtime_owner="Settings persisted defaults; storage services active handles",
                boundary_copy=(
                    "Settings edits persisted database path defaults only; active database "
                    "handles keep their current paths until restart."
                ),
                recovery_copy=(
                    "Validate paths, save the config-only change, then restart Chatbook to "
                    "activate new storage defaults."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.PRIVACY_SECURITY,
                owns_config_sections=("encryption", "api_settings.<provider>.credential_source"),
                reads_runtime_state_from=("config redaction", "environment credential status"),
                writes_allowed=False,
                runtime_owner="Privacy and credential services",
                boundary_copy="Settings exposes privacy posture without printing raw secrets.",
                recovery_copy="Rotate exposed credentials outside Chatbook and rerun privacy checks.",
                read_only_reason="Encryption and credential migration need a dedicated recovery flow.",
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.CONSOLE_BEHAVIOR,
                owns_config_sections=(
                    "console.collapse_large_pastes",
                    "console.paste_collapse_threshold",
                    "console.background_effects.*",
                    "chat_defaults.streaming",
                    "chat_defaults.temperature",
                    "chat_defaults.top_p",
                    "chat_defaults.max_tokens",
                ),
                reads_runtime_state_from=("Console composer/session defaults",),
                writes_allowed=True,
                runtime_owner="Console",
                boundary_copy=(
                    "Settings owns global Console fallbacks; provider+model profiles and "
                    "active Console sessions can override them."
                ),
                recovery_copy="Save or revert category drafts before testing live Console behavior.",
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.DIAGNOSTICS,
                owns_config_sections=("validation output", "reload status"),
                reads_runtime_state_from=("config adapter", "diagnostic services"),
                writes_allowed=False,
                runtime_owner="Diagnostics",
                boundary_copy="Diagnostics validates and reloads without mutating raw TOML.",
                recovery_copy="Validate before reload; use Advanced Config only for expert repairs.",
                read_only_reason="Diagnostic checks are non-destructive by design.",
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.ADVANCED_CONFIG,
                owns_config_sections=("raw TOML",),
                reads_runtime_state_from=("config file",),
                writes_allowed=True,
                runtime_owner="Settings advanced editor",
                boundary_copy="Advanced Config bypasses guided category controls.",
                recovery_copy="Validate exact current TOML before save; restore from backup if needed.",
            ),
            *self._domain_category_ownership_records(),
        )

    def _build_ownership_by_category(self) -> dict[SettingsCategoryId, SettingsOwnershipRecord]:
        return {record.category: record for record in self._category_ownership_records()}

    def _ownership_by_category(self) -> dict[SettingsCategoryId, SettingsOwnershipRecord]:
        return self._ownership_by_category_cache

    @staticmethod
    def _missing_ownership_record(category: SettingsCategoryId) -> SettingsOwnershipRecord:
        logger.warning("Settings ownership record missing for category %s", category.value)
        return SettingsOwnershipRecord(
            category=category,
            reads_runtime_state_from=("unknown",),
            writes_allowed=False,
            runtime_owner="Settings ownership matrix",
            boundary_copy="Ownership record missing; update the Settings ownership matrix.",
            recovery_copy="Update the Settings ownership matrix before enabling writes.",
            read_only_reason="Ownership record missing; update matrix before exposing actions.",
        )

    def _ownership_record(self, category: SettingsCategoryId) -> SettingsOwnershipRecord:
        return self._ownership_by_category().get(category) or self._missing_ownership_record(category)

    def _overview_ownership_rows(self) -> tuple[tuple[str, str], ...]:
        ownership = self._ownership_record(SettingsCategoryId.OVERVIEW)
        return (*SETTINGS_OVERVIEW_BOUNDARY_ROWS, ("Recovery", ownership.recovery_copy))

    def _active_summary(self) -> SettingsCategorySummary:
        for summary in self._category_summaries():
            if summary.category.value == self.active_category:
                return summary
        return self._category_summaries()[0]

    def _app_config_section_target(self):
        app_config = getattr(self.app_instance, "app_config", None)
        if (
            callable(getattr(app_config, "setdefault", None))
            and hasattr(app_config, "__setitem__")
        ):
            return app_config
        self.app_instance.app_config = {}
        return self.app_instance.app_config

    def _console_settings(self) -> dict:
        app_config = self._app_config_section_target()
        console_settings = app_config.setdefault("console", {})
        if not isinstance(console_settings, dict):
            console_settings = {}
            app_config["console"] = console_settings
        return console_settings

    def _chat_defaults(self) -> dict:
        app_config = self._app_config_section_target()
        chat_defaults = app_config.setdefault("chat_defaults", {})
        if not isinstance(chat_defaults, dict):
            chat_defaults = {}
            app_config["chat_defaults"] = chat_defaults
        return chat_defaults

    def _loaded_collapse_large_pastes_enabled(self) -> bool:
        return coerce_bool_setting(
            self._console_settings().get("collapse_large_pastes", True),
            True,
        )

    def _loaded_paste_collapse_threshold(self) -> int:
        return coerce_int_setting(
            self._console_settings().get(
                "paste_collapse_threshold",
                DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            ),
            DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            minimum=MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            maximum=MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        )

    @staticmethod
    def _coerce_float_default(
        value: object,
        fallback: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        if isinstance(value, bool):
            return fallback
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        if minimum <= number <= maximum:
            return number
        return fallback

    def _loaded_console_default_streaming(self) -> bool:
        chat_defaults = self._chat_defaults()
        if "streaming" in chat_defaults:
            return coerce_bool_setting(chat_defaults.get("streaming"), True)
        if "enable_streaming" in chat_defaults:
            return coerce_bool_setting(chat_defaults.get("enable_streaming"), True)
        return True

    def _loaded_console_default_temperature(self) -> float:
        return self._coerce_float_default(
            self._chat_defaults().get("temperature", 0.7),
            0.7,
            minimum=0.0,
            maximum=2.0,
        )

    def _loaded_console_default_top_p(self) -> float:
        return self._coerce_float_default(
            self._chat_defaults().get("top_p", 0.95),
            0.95,
            minimum=0.0,
            maximum=1.0,
        )

    def _loaded_console_default_min_p(self) -> float | str:
        return self._loaded_optional_float_default("min_p", minimum=0.0, maximum=1.0)

    def _loaded_console_default_top_k(self) -> int | str:
        return self._loaded_optional_int_default("top_k", minimum=0)

    def _loaded_console_default_max_tokens(self) -> int | str:
        return self._loaded_optional_int_default("max_tokens", minimum=1)

    def _loaded_console_default_seed(self) -> int | str:
        return self._loaded_optional_int_default("seed", minimum=0)

    def _loaded_console_default_presence_penalty(self) -> float | str:
        return self._loaded_optional_float_default(
            "presence_penalty",
            minimum=-2.0,
            maximum=2.0,
        )

    def _loaded_console_default_frequency_penalty(self) -> float | str:
        return self._loaded_optional_float_default(
            "frequency_penalty",
            minimum=-2.0,
            maximum=2.0,
        )

    def _loaded_console_default_choice(self, key: str, allowed: frozenset[str]) -> str:
        value = str(self._chat_defaults().get(key, "") or "").strip().lower()
        return value if value in allowed else ""

    def _loaded_console_default_thinking_budget_tokens(self) -> int | str:
        return self._loaded_optional_int_default("thinking_budget_tokens", minimum=1024)

    def _loaded_optional_float_default(
        self,
        key: str,
        *,
        minimum: float,
        maximum: float,
    ) -> float | str:
        value = self._chat_defaults().get(key, "")
        if value is None or str(value).strip() == "":
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return number if minimum <= number <= maximum else ""

    def _loaded_optional_int_default(self, key: str, *, minimum: int) -> int | str:
        value = self._chat_defaults().get(key, "")
        if value is None or str(value).strip() == "":
            return ""
        invalid_sentinel = minimum - 1
        coerced = coerce_int_setting(value, invalid_sentinel, minimum=minimum)
        return coerced if minimum <= coerced else ""

    def _console_behavior_loaded_values(self) -> dict[str, object]:
        return {
            "collapse_large_pastes": self._loaded_collapse_large_pastes_enabled(),
            "paste_collapse_threshold": self._loaded_paste_collapse_threshold(),
            "streaming": self._loaded_console_default_streaming(),
            "temperature": self._loaded_console_default_temperature(),
            "top_p": self._loaded_console_default_top_p(),
            "min_p": self._loaded_console_default_min_p(),
            "top_k": self._loaded_console_default_top_k(),
            "max_tokens": self._loaded_console_default_max_tokens(),
            "seed": self._loaded_console_default_seed(),
            "presence_penalty": self._loaded_console_default_presence_penalty(),
            "frequency_penalty": self._loaded_console_default_frequency_penalty(),
            "reasoning_effort": self._loaded_console_default_choice(
                "reasoning_effort",
                REASONING_EFFORT_OPTIONS,
            ),
            "reasoning_summary": self._loaded_console_default_choice(
                "reasoning_summary",
                REASONING_SUMMARY_OPTIONS,
            ),
            "verbosity": self._loaded_console_default_choice(
                "verbosity",
                VERBOSITY_OPTIONS,
            ),
            "thinking_effort": self._loaded_console_default_choice(
                "thinking_effort",
                THINKING_EFFORT_OPTIONS,
            ),
            "thinking_budget_tokens": self._loaded_console_default_thinking_budget_tokens(),
        }

    def _loaded_console_background_effects(self) -> dict[str, object]:
        return normalize_console_background_effects(
            self._console_settings().get("background_effects")
        ).to_config()

    def _raw_console_background_scope(self) -> object:
        raw_background_effects = self._console_settings().get("background_effects")
        if isinstance(raw_background_effects, Mapping):
            return raw_background_effects.get("scope")
        return None

    def _loaded_console_background_scope_is_unavailable(self) -> bool:
        return str(self._raw_console_background_scope()) == "workbench"

    def _console_background_effect_value(self, key: str) -> object:
        draft_key = f"background_effects.{key}"
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and draft_key in draft.values:
            return draft.values[draft_key]
        value = self._loaded_console_background_effects().get(key, "")
        if key == "scope":
            return self._available_console_background_scope(value)
        return value

    def _stage_console_background_effect_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            f"background_effects.{key}",
            self._loaded_console_background_effects().get(key),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    @staticmethod
    def _available_console_background_scope(scope: object) -> str:
        return "transcript" if str(scope) == "workbench" else str(scope or "transcript")

    def _console_background_effect_enabled_label(self) -> str:
        return (
            "Enabled"
            if bool(self._console_background_effect_value("enabled"))
            else "Disabled"
        )

    def _console_behavior_result_text(self) -> str:
        has_unsaved_changes = self._category_has_unsaved_changes(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if (
            self._loaded_console_background_scope_is_unavailable()
            and not has_unsaved_changes
            and self._console_behavior_result
            in {
                "Console behavior settings have not been saved this session.",
                "Console behavior settings staged.",
                "Console behavior settings saved.",
            }
        ):
            if self._console_behavior_saved_this_session:
                return (
                    "Console behavior settings saved. "
                    f"{CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY}"
                )
            return CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
        if (
            not has_unsaved_changes
            and self._console_behavior_result == "Console behavior settings staged."
        ):
            if self._console_behavior_saved_this_session:
                return "Console behavior settings saved."
            return "Console behavior settings have not been saved this session."
        return self._console_behavior_result

    def _console_behavior_value(self, key: str) -> object:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and key in draft.values:
            return draft.values[key]
        return self._console_behavior_loaded_values().get(key, "")

    @staticmethod
    def _console_input_value(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value if value is not None else "")

    def _collapse_large_pastes_enabled(self) -> bool:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and "collapse_large_pastes" in draft.values:
            return coerce_bool_setting(
                draft.values.get("collapse_large_pastes"),
                True,
            )
        return coerce_bool_setting(
            self._console_settings().get("collapse_large_pastes", True),
            True,
        )

    def _collapse_large_pastes_label(self) -> str:
        state = "Enabled" if self._collapse_large_pastes_enabled() else "Disabled"
        return f"{state}: collapse large pastes"

    def _collapse_large_pastes_button_label(self) -> str:
        return "Enabled" if self._collapse_large_pastes_enabled() else "Disabled"

    def _paste_collapse_threshold_value(self) -> int | str:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and "paste_collapse_threshold" in draft.values:
            return draft.values["paste_collapse_threshold"]
        return self._loaded_paste_collapse_threshold()

    def _paste_collapse_threshold_label(self) -> str:
        value = self._paste_collapse_threshold_value()
        try:
            threshold = self._normalise_paste_collapse_threshold(value)
        except ValueError:
            return f"Invalid threshold: {value}"
        return f"{threshold} characters"

    def _update_console_paste_summary(self) -> None:
        try:
            summary = self.query_one("#settings-overview-console-paste-collapse", Static)
        except QueryError:
            return
        summary.update(
            "Console paste collapse: "
            f"{self._collapse_large_pastes_label()} | Threshold: {self._paste_collapse_threshold_label()}"
        )

    def _app_config_mapping(self) -> Mapping[str, object]:
        app_config = getattr(self.app_instance, "app_config", {})
        return app_config if isinstance(app_config, Mapping) else {}

    def _appearance_loaded_defaults(self) -> SettingsAppearanceDefaults:
        return load_appearance_defaults(self._app_config_mapping())

    def _appearance_loaded_values(self) -> dict[str, object]:
        return asdict(self._appearance_loaded_defaults())

    def _appearance_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.APPEARANCE)

    def _appearance_setting_values(self) -> dict[str, object]:
        loaded = self._appearance_loaded_values()
        draft = self._appearance_draft()
        return {
            key: draft.values[key] if draft is not None and key in draft.values else value
            for key, value in loaded.items()
        }

    def _appearance_current_defaults(self) -> SettingsAppearanceDefaults:
        return SettingsAppearanceDefaults(**self._appearance_setting_values())

    def _appearance_validation_result(self):
        return validate_appearance_defaults(self._appearance_current_defaults())

    def _appearance_save_enabled(self) -> bool:
        if not self._category_has_unsaved_changes(SettingsCategoryId.APPEARANCE):
            return False
        return self._appearance_validation_result().valid

    def _library_rag_loaded_defaults(self) -> SettingsLibraryRagDefaults:
        return load_library_rag_defaults(self._app_config_mapping())

    def _library_rag_loaded_values(self) -> dict[str, object]:
        return asdict(self._library_rag_loaded_defaults())

    def _library_rag_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.LIBRARY_RAG)

    def _library_rag_setting_values(self) -> dict[str, object]:
        loaded = self._library_rag_loaded_values()
        draft = self._library_rag_draft()
        return {
            key: draft.values[key] if draft is not None and key in draft.values else value
            for key, value in loaded.items()
        }

    def _library_rag_current_defaults(self) -> SettingsLibraryRagDefaults:
        return SettingsLibraryRagDefaults(**self._library_rag_setting_values())

    def _library_rag_validation_result(self):
        return validate_library_rag_defaults(self._library_rag_current_defaults())

    def _library_rag_save_enabled(self) -> bool:
        if not self._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG):
            return False
        return self._library_rag_validation_result().valid

    def _storage_loaded_defaults(self) -> SettingsStorageDefaults:
        return load_storage_defaults(self._app_config_mapping())

    def _storage_loaded_values(self) -> dict[str, object]:
        return asdict(self._storage_loaded_defaults())

    def _storage_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.STORAGE)

    def _storage_setting_values(self) -> dict[str, object]:
        loaded = self._storage_loaded_values()
        draft = self._storage_draft()
        return {
            key: draft.values[key] if draft is not None and key in draft.values else value
            for key, value in loaded.items()
        }

    def _storage_current_defaults(self) -> SettingsStorageDefaults:
        return SettingsStorageDefaults(**self._storage_setting_values())

    def _storage_validation_result(self):
        return validate_storage_defaults(self._storage_current_defaults())

    def _storage_save_enabled(self) -> bool:
        if not self._category_has_unsaved_changes(SettingsCategoryId.STORAGE):
            return False
        return self._storage_validation_result().valid

    def _category_has_unsaved_changes(self, category: SettingsCategoryId) -> bool:
        draft = self._settings_drafts.get(category)
        return bool(draft and draft.is_dirty)

    def _guided_action_message(self, category: SettingsCategoryId) -> str:
        if category is SettingsCategoryId.APPEARANCE:
            if self._category_has_unsaved_changes(category):
                validation = self._appearance_validation_result()
                if not validation.valid:
                    return f"Guided edits: {validation.message}"
                return "Guided edits: Preview, Save, or Revert Appearance defaults."
            return "Guided edits: change an Appearance default first."
        if category is SettingsCategoryId.LIBRARY_RAG:
            if self._category_has_unsaved_changes(category):
                validation = self._library_rag_validation_result()
                if not validation.valid:
                    return f"Guided edits: {validation.message}"
                return "Guided edits: Save or Revert Library/RAG defaults."
            return "Guided edits: change a Library/RAG default first."
        if category is SettingsCategoryId.STORAGE:
            if self._category_has_unsaved_changes(category):
                validation = self._storage_validation_result()
                if not validation.valid:
                    return f"Guided edits: {validation.message}"
                return "Guided edits: Save or Revert Storage defaults."
            return "Guided edits: change a Storage default first."
        if category in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            if self._category_has_unsaved_changes(category):
                return "Guided edits: Save or Revert changes."
            return "Guided edits: change a field first."
        messages = {
            SettingsCategoryId.OVERVIEW: "Guided edits: choose Providers or Console.",
            SettingsCategoryId.PRIVACY_SECURITY: "Guided edits: use Check Privacy.",
            SettingsCategoryId.DIAGNOSTICS: "Guided edits: use Validate/Reload.",
            SettingsCategoryId.ADVANCED_CONFIG: "Guided edits: use Raw TOML controls.",
        }
        if category in DOMAIN_SETTINGS_CATEGORY_IDS:
            contract = self._domain_category_contract(category)
            return f"Guided edits: read-only/WIP; open {contract.owner_destination}."
        return messages.get(category, "Guided edits: read-only.")

    def _guided_actions_enabled(self, category: SettingsCategoryId) -> bool:
        if category is SettingsCategoryId.APPEARANCE:
            return self._appearance_save_enabled()
        if category is SettingsCategoryId.LIBRARY_RAG:
            return self._library_rag_save_enabled()
        if category is SettingsCategoryId.STORAGE:
            return self._storage_save_enabled()
        return (
            category in GUIDED_SETTINGS_MUTATION_CATEGORIES
            and self._category_has_unsaved_changes(category)
        )

    def _update_guided_action_widgets(self) -> None:
        category = self._active_category_id()
        actions_enabled = self._guided_actions_enabled(category)
        self._set_static_text("#settings-guided-action-state", self._guided_action_message(category))
        for selector in ("#settings-save-category", "#settings-revert-category"):
            try:
                self.query_one(selector, Button).disabled = not actions_enabled
            except QueryError:
                pass

    def _category_status(self, summary: SettingsCategorySummary) -> str:
        if self._category_has_unsaved_changes(summary.category):
            return "Unsaved"
        return summary.status

    def _category_button_label(
        self,
        summary: SettingsCategorySummary,
        *,
        is_active: bool | None = None,
    ) -> str:
        active = summary.category.value == self.active_category if is_active is None else is_active
        dirty_marker = " *" if self._category_has_unsaved_changes(summary.category) else ""
        return f"{'> ' if active else '  '}{summary.title}{dirty_marker}"

    def _refresh_category_button_label(self, category: SettingsCategoryId) -> None:
        try:
            summary = self._category_summary_by_id(category)
            button = self.query_one(f"#settings-category-{category.value}", Button)
        except QueryError:
            return
        button.label = self._category_button_label(summary)

    def _active_category_id(self) -> SettingsCategoryId:
        return SettingsCategoryId(self.active_category)

    def _update_draft_status_widgets(self, category: SettingsCategoryId) -> None:
        has_unsaved_changes = self._category_has_unsaved_changes(category)
        status = "Unsaved changes" if has_unsaved_changes else "No unsaved changes"
        try:
            self.query_one("#settings-selected-category-draft-status", Static).update(status)
        except QueryError:
            pass
        self._update_category_state_banner(category)
        try:
            category_status = "Unsaved" if has_unsaved_changes else self._category_summary_by_id(category).status
            category_status_widget = self.query_one(
                f"#settings-category-{category.value}-status", Static
            )
            category_status_widget.update(f"Status: {category_status}")
            if has_unsaved_changes:
                category_status_widget.add_class("settings-dirty-category")
            else:
                category_status_widget.remove_class("settings-dirty-category")
        except QueryError:
            pass
        self._refresh_category_button_label(category)
        if category is self._active_category_id():
            self._update_guided_action_widgets()

    def _category_summary_by_id(self, category: SettingsCategoryId) -> SettingsCategorySummary:
        for summary in self._category_summaries():
            if summary.category is category:
                return summary
        return self._category_summaries()[0]

    def _sanitize_category_search_query(self, query_text: str | None) -> str:
        raw_query = "" if query_text is None else str(query_text)
        sanitized_query = sanitize_string(
            raw_query,
            max_length=MAX_CATEGORY_SEARCH_QUERY_CHARS,
        )
        if validate_text_input(
            sanitized_query,
            max_length=MAX_CATEGORY_SEARCH_QUERY_CHARS,
            allow_html=False,
        ):
            return sanitized_query
        return ""

    def _category_search_text(self, query_text: str | None = None) -> str:
        raw_query = self.category_search_query if query_text is None else query_text
        return raw_query.strip() if isinstance(raw_query, str) else ""

    def _category_search_rank(
        self,
        summary: SettingsCategorySummary,
        query_text: str | None = None,
    ) -> int | None:
        query = self._category_search_text(query_text).lower()
        if not query:
            return 0
        primary_haystack = " ".join((summary.category.value, summary.title)).lower()
        if query in primary_haystack:
            return 0
        secondary_haystack = " ".join(
            (
                summary.description,
                self._category_status(summary),
            )
        ).lower()
        if query in secondary_haystack:
            return 1
        return None

    def _category_matches_search(
        self,
        summary: SettingsCategorySummary,
        query_text: str | None = None,
    ) -> bool:
        return self._category_search_rank(summary, query_text) is not None

    def _filtered_category_summaries(
        self,
        query_text: str | None = None,
    ) -> tuple[SettingsCategorySummary, ...]:
        ranked_summaries: list[tuple[int, int, SettingsCategorySummary]] = []
        for index, summary in enumerate(self._category_summaries()):
            rank = self._category_search_rank(summary, query_text)
            if rank is not None:
                ranked_summaries.append((rank, index, summary))
        return tuple(summary for _, _, summary in sorted(ranked_summaries))

    def _filtered_category_values(self, query_text: str | None = None) -> list[str]:
        return [
            summary.category.value
            for summary in self._filtered_category_summaries(query_text)
        ]

    def _category_search_status_text(self, query_text: str | None = None) -> str:
        query = self._category_search_text(query_text)
        if not query:
            return "No filter | / focus category search"
        matches = self._filtered_category_summaries(query)
        match_label = "match" if len(matches) == 1 else "matches"
        if matches:
            return f"Filter: {query} | {len(matches)} {match_label} | Enter opens {matches[0].title}"
        return f"Filter: {query} | 0 matches | Esc clears"

    @staticmethod
    def _category_group_dom_id(group_title: str) -> str:
        return f"settings-category-group-{group_title.lower().replace(' ', '-').replace('&', 'and')}"

    def _apply_category_search_filter(self) -> None:
        summaries_by_id = {summary.category: summary for summary in self._category_summaries()}
        visible_count = 0
        query = self._category_search_text()
        for group_title, category_ids in self._category_groups():
            group_visible = False
            for category_id in category_ids:
                summary = summaries_by_id[category_id]
                rank = self._category_search_rank(summary)
                is_visible = rank is not None
                group_visible = group_visible or is_visible
                visible_count += int(is_visible)
                try:
                    button = self.query_one(f"#settings-category-{summary.category.value}", Button)
                    button.display = is_visible
                    button.remove_class("settings-primary-search-match")
                    button.remove_class("settings-secondary-search-match")
                    if query and rank == 0:
                        button.add_class("settings-primary-search-match")
                    elif query and rank == 1:
                        button.add_class("settings-secondary-search-match")
                except QueryError:
                    pass
            try:
                self.query_one(f"#{self._category_group_dom_id(group_title)}", Static).display = group_visible
            except QueryError:
                pass

        try:
            status = self.query_one("#settings-category-search-status", Static)
            status.update(self._category_search_status_text())
        except QueryError:
            pass
        try:
            search = self.query_one("#settings-category-search", Input)
            search.set_class(bool(query), "settings-category-search-active")
        except QueryError:
            pass
        try:
            empty_state = self.query_one("#settings-category-search-empty", Static)
        except QueryError:
            return
        empty_state.update(f"No Settings categories match: {query}")
        empty_state.display = bool(query and visible_count == 0)

    def _submit_category_search(self, query_text: str) -> None:
        query_text = self._sanitize_category_search_query(query_text)
        self.category_search_query = query_text
        self._apply_category_search_filter()
        category_values = self._filtered_category_values(query_text)
        if category_values:
            self._select_category(category_values[0], restore_focus=True)

    def _category_state_banner_text(self, category: SettingsCategoryId) -> str:
        if category is SettingsCategoryId.APPEARANCE and self._category_has_unsaved_changes(category):
            validation = self._appearance_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if category is SettingsCategoryId.LIBRARY_RAG and self._category_has_unsaved_changes(category):
            validation = self._library_rag_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if category is SettingsCategoryId.STORAGE and self._category_has_unsaved_changes(category):
            validation = self._storage_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if self._category_has_unsaved_changes(category):
            return "State: Unsaved changes | Save or Revert before leaving this category."
        if category is SettingsCategoryId.ADVANCED_CONFIG:
            return "State: Guarded | Save blocked until the current text validates; backup created before overwrite."
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            return "State: Shared with Console"
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            return "State: Console scoped | Changes affect global Console fallbacks after save."
        if category is SettingsCategoryId.LIBRARY_RAG:
            return "State: Library scoped | Defaults affect future Library/RAG retrieval and display."
        if category is SettingsCategoryId.DIAGNOSTICS:
            return "State: Safe to run | Validation and reload expose status without writing raw TOML."
        if category is SettingsCategoryId.APPEARANCE:
            return "State: Visual defaults | Settings owns launch and web display defaults."
        if category is SettingsCategoryId.STORAGE:
            return (
                "State: Storage defaults | Changes apply on next launch; "
                "active handles stay unchanged."
            )
        if category is SettingsCategoryId.PRIVACY_SECURITY:
            return "State: Local privacy | Secrets stay redacted in validation and diagnostics."
        if category in DOMAIN_SETTINGS_CATEGORY_IDS:
            contract = self._domain_category_contract(category)
            return (
                "State: Read-only contract | "
                f"{contract.owner_destination} owns workflow actions and setup."
            )
        return "State: Active | Review readiness across Settings categories."

    def _render_category_state_banner(self, category: SettingsCategoryId) -> Static:
        banner = Static(
            self._category_state_banner_text(category),
            id="settings-category-state-banner",
            classes="settings-state-banner",
        )
        if self._category_has_unsaved_changes(category):
            banner.add_class("settings-dirty-category")
        return banner

    def _update_category_state_banner(self, category: SettingsCategoryId) -> None:
        try:
            banner = self.query_one("#settings-category-state-banner", Static)
        except QueryError:
            return
        banner.update(self._category_state_banner_text(category))
        if self._category_has_unsaved_changes(category):
            banner.add_class("settings-dirty-category")
        else:
            banner.remove_class("settings-dirty-category")

    @staticmethod
    def _normalise_appearance_int(value: object) -> int | str:
        text_value = str(value).strip()
        return int(text_value) if text_value.isdigit() else text_value

    def _stage_appearance_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.APPEARANCE
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            key,
            self._appearance_loaded_values().get(key),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _appearance_invalid_field_key(self) -> str | None:
        validation = self._appearance_validation_result()
        if validation.valid:
            return None
        message = validation.message
        if message.startswith("Theme"):
            return "default_theme"
        if message.startswith("Palette theme limit"):
            return "palette_theme_limit"
        if message.startswith("Font size"):
            return "font_size"
        if message.startswith("Density"):
            return "density"
        if message.startswith("Animations"):
            return "animations_enabled"
        if message.startswith("Smooth scrolling"):
            return "smooth_scrolling"
        return None

    def _appearance_field_selector(self, key: str) -> str | None:
        return {
            "default_theme": "#settings-appearance-theme",
            "palette_theme_limit": "#settings-appearance-palette-theme-limit",
            "font_size": "#settings-appearance-font-size",
            "density": "#settings-appearance-density",
            "animations_enabled": "#settings-appearance-animations-enabled",
            "smooth_scrolling": "#settings-appearance-smooth-scrolling",
        }.get(key)

    def _update_appearance_validation_classes(self) -> None:
        invalid_key = self._appearance_invalid_field_key()
        for key in (
            "default_theme",
            "palette_theme_limit",
            "font_size",
            "density",
            "animations_enabled",
            "smooth_scrolling",
        ):
            selector = self._appearance_field_selector(key)
            if selector is None:
                continue
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.set_class(key == invalid_key, "settings-invalid-input")

    def _mark_appearance_settings_staged(self) -> None:
        category = SettingsCategoryId.APPEARANCE
        if self._category_has_unsaved_changes(category):
            validation = self._appearance_validation_result()
            self._appearance_result = (
                "Appearance defaults staged." if validation.valid else validation.message
            )
        else:
            self._appearance_result = "Appearance defaults match loaded values."
        self._set_static_text("#settings-appearance-save-result", self._appearance_result)
        self._update_appearance_validation_classes()
        self._update_draft_status_widgets(category)

    def _appearance_theme_options(self) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = [
            ("Textual Dark", "textual-dark"),
            ("Textual Light", "textual-light"),
        ]
        seen = {value for _label, value in options}
        try:
            from tldw_chatbook.css.Themes.themes import ALL_THEMES
        except (ImportError, ModuleNotFoundError):
            ALL_THEMES = ()
        for theme in ALL_THEMES:
            theme_name = str(getattr(theme, "name", "") or "").strip()
            if not theme_name or theme_name in seen:
                continue
            seen.add(theme_name)
            options.append((theme_name.replace("_", " ").replace("-", " ").title(), theme_name))
        current_theme = str(self._appearance_setting_values()["default_theme"])
        if current_theme and current_theme not in seen:
            options.append((f"Current: {current_theme}", current_theme))
        return options

    def _appearance_bool_label(self, key: str) -> str:
        return "Enabled" if bool(self._appearance_setting_values()[key]) else "Disabled"

    def _appearance_summary_text(self) -> str:
        values = self._appearance_setting_values()
        return (
            f"Theme {values['default_theme']} | Density {values['density']} | "
            f"Font {values['font_size']} | Palette limit {values['palette_theme_limit']}"
        )

    def _stage_console_large_paste_value(self, value: bool) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            "collapse_large_pastes",
            self._loaded_collapse_large_pastes_enabled(),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _stage_console_default_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            key,
            self._console_behavior_loaded_values().get(key),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _normalise_paste_collapse_threshold(self, value: object) -> int:
        text_value = str(value).strip()
        if not text_value or not text_value.isdigit():
            raise ValueError("Paste collapse threshold must be a whole number.")
        if not validate_number_range(
            text_value,
            min_val=MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            max_val=MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        ):
            raise ValueError(
                "Paste collapse threshold must be between "
                f"{MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD} and "
                f"{MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD}."
            )
        return int(text_value)

    def _stage_console_paste_threshold_value(self, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        try:
            staged_value: object = self._normalise_paste_collapse_threshold(value)
        except ValueError:
            staged_value = str(value)
        draft.set_value(
            "paste_collapse_threshold",
            self._loaded_paste_collapse_threshold(),
            staged_value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    @staticmethod
    def _normalise_library_rag_int(value: object) -> int | str:
        text_value = str(value).strip()
        return int(text_value) if text_value.isdigit() else text_value

    @staticmethod
    def _normalise_library_rag_float(value: object) -> float | str:
        text_value = str(value).strip()
        if not text_value:
            return text_value
        try:
            return float(text_value)
        except ValueError:
            return text_value

    def _stage_library_rag_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.LIBRARY_RAG
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            key,
            self._library_rag_loaded_values().get(key),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _mark_library_rag_settings_staged(self) -> None:
        if self._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG):
            validation = self._library_rag_validation_result()
            self._library_rag_result = (
                "Library/RAG defaults staged." if validation.valid else validation.message
            )
        else:
            self._library_rag_result = "Library/RAG defaults match last loaded values."
        self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
        self._update_library_rag_preview()
        self._update_library_rag_validation_classes()
        self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)

    def _sync_library_rag_widgets(self) -> None:
        values = self._library_rag_loaded_values()
        self._syncing_library_rag_defaults = True
        try:
            try:
                self.query_one("#settings-library-rag-search-mode", Select).value = (
                    normalise_library_rag_search_mode(values["default_search_mode"])
                )
                self.query_one("#settings-library-rag-citation-style", Select).value = (
                    normalise_library_rag_citation_style(values["citation_style"])
                )
                self.query_one("#settings-library-rag-include-citations", Button).label = (
                    "Enabled" if bool(values["include_citations"]) else "Disabled"
                )
                for selector, key in (
                    ("#settings-library-rag-default-top-k", "default_top_k"),
                    ("#settings-library-rag-fts-top-k", "fts_top_k"),
                    ("#settings-library-rag-vector-top-k", "vector_top_k"),
                    ("#settings-library-rag-hybrid-alpha", "hybrid_alpha"),
                    ("#settings-library-rag-score-threshold", "score_threshold"),
                    ("#settings-library-rag-snippet-max-chars", "snippet_max_chars"),
                    ("#settings-library-rag-max-context-size", "max_context_size"),
                ):
                    self.query_one(selector, Input).value = str(values[key])
            except QueryError:
                pass
        finally:
            self._syncing_library_rag_defaults = False
        self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
        self._update_library_rag_preview()
        self._update_library_rag_validation_classes()

    def _library_rag_invalid_field_key(self) -> str | None:
        validation = self._library_rag_validation_result()
        if validation.valid:
            return None
        message = validation.message
        if message.startswith("Default results"):
            return "default_top_k"
        if message.startswith("Keyword results"):
            return "fts_top_k"
        if message.startswith("Vector results"):
            return "vector_top_k"
        if message.startswith("Hybrid balance"):
            return "hybrid_alpha"
        if message.startswith("Score threshold"):
            return "score_threshold"
        if message.startswith("Snippet characters"):
            return "snippet_max_chars"
        if message.startswith("Context budget"):
            return "max_context_size"
        if message.startswith("Search mode"):
            return "default_search_mode"
        if message.startswith("Citation style"):
            return "citation_style"
        return None

    def _library_rag_field_selector(self, key: str) -> str | None:
        return {
            "default_search_mode": "#settings-library-rag-search-mode",
            "default_top_k": "#settings-library-rag-default-top-k",
            "fts_top_k": "#settings-library-rag-fts-top-k",
            "vector_top_k": "#settings-library-rag-vector-top-k",
            "hybrid_alpha": "#settings-library-rag-hybrid-alpha",
            "score_threshold": "#settings-library-rag-score-threshold",
            "citation_style": "#settings-library-rag-citation-style",
            "snippet_max_chars": "#settings-library-rag-snippet-max-chars",
            "max_context_size": "#settings-library-rag-max-context-size",
        }.get(key)

    def _update_library_rag_validation_classes(self) -> None:
        invalid_key = self._library_rag_invalid_field_key()
        for key in (
            "default_search_mode",
            "default_top_k",
            "fts_top_k",
            "vector_top_k",
            "hybrid_alpha",
            "score_threshold",
            "citation_style",
            "snippet_max_chars",
            "max_context_size",
        ):
            selector = self._library_rag_field_selector(key)
            if selector is None:
                continue
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.set_class(key == invalid_key, "settings-invalid-input")

    def _stage_storage_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.STORAGE
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            key,
            self._storage_loaded_values().get(key),
            str(value if value is not None else ""),
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _storage_field_selector(self, key: str) -> str | None:
        return {
            "user_db_base_dir": "#settings-storage-user-db-base-dir",
            "chachanotes_db_path": "#settings-storage-chachanotes-db-path",
            "prompts_db_path": "#settings-storage-prompts-db-path",
            "media_db_path": "#settings-storage-media-db-path",
            "research_db_path": "#settings-storage-research-db-path",
            "writing_db_path": "#settings-storage-writing-db-path",
            "library_collections_db_path": "#settings-storage-library-collections-db-path",
            "workspaces_db_path": "#settings-storage-workspaces-db-path",
        }.get(key)

    def _storage_invalid_field_key(self) -> str | None:
        validation = self._storage_validation_result()
        if validation.valid:
            return None
        message = validation.message
        for key, label in STORAGE_FIELD_LABELS.items():
            if message.startswith(str(label)):
                return key
        return None

    def _update_storage_validation_classes(self) -> None:
        invalid_key = self._storage_invalid_field_key()
        for key in STORAGE_FIELD_LABELS:
            selector = self._storage_field_selector(key)
            if selector is None:
                continue
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.set_class(key == invalid_key, "settings-invalid-input")

    def _mark_storage_settings_staged(self) -> None:
        category = SettingsCategoryId.STORAGE
        if self._category_has_unsaved_changes(category):
            validation = self._storage_validation_result()
            self._storage_result = (
                "Storage defaults staged. Restart Chatbook to use saved paths."
                if validation.valid
                else validation.message
            )
        else:
            self._storage_result = "Storage defaults match last loaded values."
        self._set_static_text("#settings-storage-save-result", self._storage_result)
        self._update_storage_validation_classes()
        self._update_draft_status_widgets(category)

    def _sync_storage_widgets(self) -> None:
        values = self._storage_setting_values()
        self._syncing_storage_defaults = True
        try:
            for key, value in values.items():
                selector = self._storage_field_selector(key)
                if selector is None:
                    continue
                try:
                    self.query_one(selector, Input).value = str(value)
                except QueryError:
                    pass
        finally:
            self._syncing_storage_defaults = False
        self._set_static_text("#settings-storage-save-result", self._storage_result)
        self._update_storage_validation_classes()
        self._update_draft_status_widgets(SettingsCategoryId.STORAGE)

    def _library_rag_preview_rows(self) -> tuple[str, str, str]:
        values = self._library_rag_setting_values()
        search_mode = normalise_library_rag_search_mode(values["default_search_mode"])
        mode_label = {
            "plain": "Plain keyword",
            "semantic": "Semantic",
            "hybrid": "Hybrid",
        }[search_mode]
        citation_label = "No citations"
        if bool(values["include_citations"]):
            citation_style = normalise_library_rag_citation_style(values["citation_style"])
            citation_label = {
                "inline": "Inline citations",
                "footnote": "Footnote citations",
                "none": "No citations",
            }[citation_style]
        return (
            f"{mode_label} search | {values['default_top_k']} results | {citation_label}",
            (
                f"Keyword {values['fts_top_k']} | Vector {values['vector_top_k']} | "
                f"Hybrid alpha {values['hybrid_alpha']} | Min score {values['score_threshold']}"
            ),
            (
                f"Snippet: {values['snippet_max_chars']} chars | "
                f"Context budget: {values['max_context_size']} chars"
            ),
        )

    def _update_library_rag_preview(self) -> None:
        for selector, text in zip(
            (
                "#settings-library-rag-preview-summary",
                "#settings-library-rag-preview-retrieval",
                "#settings-library-rag-preview-context",
            ),
            self._library_rag_preview_rows(),
            strict=True,
        ):
            self._set_static_text(selector, text)

    def _normalise_console_default_streaming(self, value: object) -> bool:
        normalized = self._normalise_optional_bool(value)
        if normalized == "":
            raise ValueError("Streaming must be true or false.")
        return bool(normalized)

    def _normalise_console_default_temperature(self, value: object) -> float:
        normalized = self._normalise_optional_float(
            value,
            min_value=0.0,
            max_value=2.0,
            label="Temperature",
        )
        if normalized == "":
            raise ValueError("Temperature must be between 0.0 and 2.0.")
        return float(normalized)

    def _normalise_console_default_top_p(self, value: object) -> float:
        normalized = self._normalise_optional_float(
            value,
            min_value=0.0,
            max_value=1.0,
            label="Top P",
        )
        if normalized == "":
            raise ValueError("Top P must be between 0.0 and 1.0.")
        return float(normalized)

    @staticmethod
    def _normalise_console_default_max_tokens(value: object) -> int | str:
        text_value = "" if value is None else str(value).strip()
        if not text_value:
            return ""
        if not text_value.isdecimal() or int(text_value) < 1:
            raise ValueError("Max tokens must be a whole number of at least 1.")
        return int(text_value)

    @staticmethod
    def _normalise_console_background_fps(value: object) -> int:
        text_value = "" if value is None else str(value).strip()
        if (
            not text_value.isdecimal()
            or not validate_number_range(
                text_value,
                min_val=MIN_CONSOLE_BACKGROUND_FPS,
                max_val=MAX_CONSOLE_BACKGROUND_FPS,
            )
        ):
            raise ValueError(
                "Frame rate must be a whole number between "
                f"{MIN_CONSOLE_BACKGROUND_FPS} and {MAX_CONSOLE_BACKGROUND_FPS}."
            )
        return int(text_value)

    def _active_sync_scope(
        self,
        active_workspace: object = _WORKSPACE_RECORD_UNSET,
    ) -> dict[str, str | None]:
        runtime_state = self._runtime_source_state()
        active_source = str(getattr(runtime_state, "active_source", "local") or "local").lower()
        server_profile_value = getattr(runtime_state, "active_server_id", None)
        server_profile_id = str(server_profile_value or "").strip() or None
        source_authority = "server" if active_source == "server" and server_profile_id else "local"
        authenticated_principal_id = None
        if source_authority == "server":
            server_context_provider = getattr(self.app_instance, "server_context_provider", None)
            get_active_context = getattr(server_context_provider, "get_active_context", None)
            if callable(get_active_context):
                try:
                    authenticated_principal_id = event_principal_id_from_active_context(
                        get_active_context()
                    )
                except Exception:
                    logger.warning(
                        "Failed to resolve Settings sync authenticated principal scope.",
                        exc_info=True,
                    )
                    authenticated_principal_id = None

        workspace = (
            self._active_workspace_record()
            if active_workspace is _WORKSPACE_RECORD_UNSET
            else active_workspace
        )
        workspace_scope = None
        if workspace is not None:
            workspace_scope = str(getattr(workspace, "workspace_id", "") or "").strip() or None
        return {
            "server_profile_id": server_profile_id if source_authority == "server" else None,
            "authenticated_principal_id": authenticated_principal_id,
            "workspace_scope": workspace_scope,
        }

    def _sync_safety_states(
        self,
        scope: Mapping[str, str | None] | None = None,
    ) -> tuple[SyncPromotionState, ...]:
        labels = {
            "library_collections": "Collections",
            "workspaces": "Workspaces",
        }
        active_scope = dict(scope or self._active_sync_scope())
        sync_scope_service = getattr(self.app_instance, "sync_scope_service", None)
        list_states = getattr(sync_scope_service, "list_write_sync_promotion_states", None)
        if callable(list_states):
            try:
                return tuple(
                    list_states(
                        domains=list(labels),
                        surface_labels=labels,
                        server_profile_id=active_scope["server_profile_id"],
                        authenticated_principal_id=active_scope["authenticated_principal_id"],
                        workspace_scope=active_scope["workspace_scope"],
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load Settings sync safety states; using local fallback. error_type=%s",
                    type(exc).__name__,
                )
        return tuple(
            build_sync_promotion_state(
                domain=domain,
                surface_label=label,
                readiness=build_sync_readiness_report(
                    domain=domain,
                    server_profile_id=active_scope["server_profile_id"],
                    workspace_id=active_scope["workspace_scope"],
                    registry=DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
                ),
            )
            for domain, label in labels.items()
        )

    @staticmethod
    def _enum_display_value(value: object, fallback: str = "") -> str:
        enum_value = getattr(value, "value", value)
        text = str(enum_value or "").strip()
        return text or fallback

    def _runtime_source_state(self) -> object | None:
        return getattr(getattr(self.app_instance, "runtime_policy", None), "state", None)

    def _active_server_profile_label(self) -> str:
        state = self._runtime_source_state()
        source = str(
            getattr(
                state,
                "active_source",
                getattr(self.app_instance, "current_runtime_backend", "local"),
            )
            or "local"
        ).strip().lower()
        active_server_id = str(
            getattr(state, "active_server_id", getattr(self.app_instance, "active_server_id", None))
            or ""
        ).strip()
        server_label = str(getattr(state, "last_known_server_label", "") or "").strip()
        if source == "server" and active_server_id:
            if server_label and server_label != active_server_id:
                return f"{server_label} ({active_server_id})"
            return active_server_id
        if active_server_id:
            label = server_label or active_server_id
            return f"{label} ({active_server_id}) configured; current source is {source}"
        return "local-only; no active server profile"

    def _local_server_authority_label(self) -> str:
        get_source = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(get_source):
            try:
                source = str(get_source() or "local").strip().lower()
            except Exception:
                source = "local"
        else:
            state = self._runtime_source_state()
            source = str(getattr(state, "active_source", "local") or "local").strip().lower()
        if source not in {"local", "server"}:
            source = "local"
        return f"{source}; Settings is read-only"

    def _sync_safety_label(self, states: tuple[SyncPromotionState, ...] | None = None) -> str:
        sync_states = states if states is not None else self._sync_safety_states()
        if not sync_states:
            return "Sync: unavailable; owning sync surfaces control dry-run and recovery"
        return "; ".join(
            f"{state.surface_label}: {state.sync_label}"
            for state in sync_states
        )

    def _sync_recovery_label(self, states: tuple[SyncPromotionState, ...] | None = None) -> str:
        sync_states = states if states is not None else self._sync_safety_states()
        blocking_statuses = {"rollback-required", "conflict", "attention-required", "review-gated"}
        selected = next(
            (state for state in sync_states if state.status in blocking_statuses),
            sync_states[0] if sync_states else None,
        )
        if selected is None:
            return "Open the owning sync surface for dry-run setup and recovery."
        return selected.primary_recovery

    def _active_workspace_record(self) -> object | None:
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        get_active_workspace = getattr(registry_service, "get_active_workspace", None)
        if not callable(get_active_workspace):
            return None
        try:
            return get_active_workspace()
        except Exception:
            logger.warning(
                "Failed to read Settings active workspace state; using local fallback.",
                exc_info=True,
            )
            return None

    def _workspace_default_label(
        self,
        active_workspace: object = _WORKSPACE_RECORD_UNSET,
    ) -> str:
        workspace = (
            self._active_workspace_record()
            if active_workspace is _WORKSPACE_RECORD_UNSET
            else active_workspace
        )
        if workspace is not None:
            workspace_id = str(getattr(workspace, "workspace_id", "") or "").strip()
            workspace_name = str(getattr(workspace, "name", "") or "").strip() or workspace_id
            authority = self._enum_display_value(
                getattr(workspace, "authority", None),
                "local-only",
            )
            sync_status = self._enum_display_value(
                getattr(workspace, "sync_status", None),
                "not-configured",
            )
            if workspace_id:
                return (
                    f"Workspace: {workspace_name} ({workspace_id}); "
                    f"Authority: {authority}; Sync: {sync_status}"
                )
        store = getattr(self.app_instance, "console_chat_store", None)
        context = getattr(store, "workspace_context", None)
        active_workspace_id = str(getattr(context, "active_workspace_id", "") or "").strip()
        if active_workspace_id and active_workspace_id != "global":
            return (
                f"Workspace: {active_workspace_id}; Console context active; "
                "Library browse/search remains global"
            )
        return "Workspace: Local Default; Console/Home/Library own workspace switching"

    def _acp_runtime_session_state(self) -> ACPRuntimeSessionState:
        get_state = getattr(self.app_instance, "get_acp_runtime_session_state", None)
        if callable(get_state):
            try:
                return ACPRuntimeSessionState.from_any(get_state())
            except Exception:
                logger.warning(
                    "Failed to read Settings ACP runtime/session state; using unavailable fallback.",
                    exc_info=True,
                )
        return ACPRuntimeSessionState.from_any(
            getattr(self.app_instance, "acp_runtime_session_state", None)
        )

    def _acp_handoff_readiness_label(self) -> str:
        state = self._acp_runtime_session_state()
        if state.has_console_session_payload:
            status = state.session_status or "ready"
            return f"ACP session ready: {state.session_display_name} ({status})"
        if state.runtime_configured:
            return f"ACP runtime configured: {state.runtime_display_name}; no session payload"
        return "ACP runtime not configured; configure runtime and sessions in ACP"

    def _server_sync_workspace_handoff_rows(self) -> tuple[tuple[str, str], ...]:
        active_workspace = self._active_workspace_record()
        sync_scope = self._active_sync_scope(active_workspace)
        sync_states = self._sync_safety_states(sync_scope)
        return (
            ("Active server profile", self._active_server_profile_label()),
            ("Local/server authority", self._local_server_authority_label()),
            ("Sync safety", self._sync_safety_label(sync_states)),
            ("Sync recovery", self._sync_recovery_label(sync_states)),
            ("Workspace default", self._workspace_default_label(active_workspace)),
            ("Library visibility", LIBRARY_WORKSPACE_VISIBILITY_COPY),
            (
                "Handoff policy",
                "copy/reference/metadata-only by source policy; "
                "Console staging is limited to the active workspace",
            ),
            ("ACP handoff readiness", self._acp_handoff_readiness_label()),
        )

    @staticmethod
    def _manual_sync_rows_from_preview(
        preview: ManualSyncPreview,
    ) -> tuple[tuple[str, str], ...]:
        pending_copy = "; ".join(
            f"{domain}: {count}"
            for domain, count in preview.pending_by_domain.items()
        ) or "none"
        return (
            ("Manual sync status", preview.status),
            ("Manual sync preview", preview.user_message),
            ("Pending outgoing", pending_copy),
        )

    def _manual_sync_rows(self) -> tuple[tuple[str, str], ...]:
        control = getattr(self.app_instance, "manual_sync_control_service", None)
        if control is None:
            return (
                ("Manual sync status", "blocked"),
                ("Manual sync preview", "Manual Sync control is not available."),
                ("Pending outgoing", "unknown"),
            )
        active_workspace = self._active_workspace_record()
        sync_scope = self._active_sync_scope(active_workspace)
        server_profile_id = sync_scope["server_profile_id"]
        if not server_profile_id:
            return (
                ("Manual sync status", "blocked"),
                ("Manual sync preview", "Manual Sync requires an active server profile."),
                ("Pending outgoing", "none"),
            )
        try:
            preview = control.preview(
                server_profile_id=server_profile_id,
                authenticated_principal_id=sync_scope["authenticated_principal_id"],
                workspace_scope=sync_scope["workspace_scope"],
            )
        except Exception as exc:
            logger.warning("Failed to build Settings manual sync preview.", exc_info=True)
            return (
                ("Manual sync status", "blocked"),
                ("Manual sync preview", f"Manual Sync preview unavailable: {type(exc).__name__}"),
                ("Pending outgoing", "unknown"),
            )
        return self._manual_sync_rows_from_preview(preview)

    def _apply_server_sync_workspace_handoff_rows(
        self,
        rows: tuple[tuple[str, str], ...],
    ) -> None:
        self.server_sync_workspace_handoff_rows = rows

    def _apply_manual_sync_rows(
        self,
        rows: tuple[tuple[str, str], ...],
    ) -> None:
        self.manual_sync_rows = rows

    def _apply_manual_sync_result(self, result: ManualSyncRunResult) -> None:
        rows = [
            ("Manual sync status", result.status),
            ("Manual sync result", result.user_message),
            ("Pending outgoing", self._pending_copy(result.preview.pending_by_domain)),
        ]
        if result.conflict_reviews:
            first_review = result.conflict_reviews[0]
            rows.append(
                (
                    "Conflict review",
                    (
                        f"{first_review.domain} | {first_review.item_label} | {first_review.cause} | "
                        f"local: {first_review.local_summary} | remote: {first_review.remote_summary}"
                    ),
                )
            )
            rows.append(
                (
                    "Recovery options",
                    "; ".join(
                        f"{action}: {state}"
                        for action, state in first_review.recovery_options.items()
                    ),
                )
            )
        self.manual_sync_rows = tuple(rows)

    @staticmethod
    def _pending_copy(pending_by_domain: Mapping[str, int]) -> str:
        return "; ".join(
            f"{domain}: {count}"
            for domain, count in pending_by_domain.items()
        ) or "none"

    @work(exclusive=True, thread=True)
    def _refresh_server_sync_workspace_handoff_rows(self) -> None:
        try:
            rows = self._server_sync_workspace_handoff_rows()
        except Exception:
            logger.warning(
                "Failed to refresh Settings server/sync/workspace/handoff rows.",
                exc_info=True,
            )
            rows = self._server_sync_workspace_handoff_loading_rows()
        self.app.call_from_thread(self._apply_server_sync_workspace_handoff_rows, rows)

    @work(exclusive=True, thread=True, group="settings-manual-sync-preview")
    def _refresh_manual_sync_rows(self) -> None:
        try:
            rows = self._manual_sync_rows()
        except Exception:
            logger.warning("Failed to refresh Settings manual sync rows.", exc_info=True)
            rows = self._manual_sync_loading_rows()
        self.app.call_from_thread(self._apply_manual_sync_rows, rows)

    @work(exclusive=True, group="settings-manual-sync-run")
    async def _manual_sync_run_worker(self) -> None:
        control = getattr(self.app_instance, "manual_sync_control_service", None)
        if control is None:
            self._apply_manual_sync_rows(
                (
                    ("Manual sync status", "blocked"),
                    ("Manual sync result", "Manual Sync control is not available."),
                    ("Pending outgoing", "unknown"),
                ),
            )
            return
        active_workspace = self._active_workspace_record()
        sync_scope = self._active_sync_scope(active_workspace)
        server_profile_id = sync_scope["server_profile_id"]
        if not server_profile_id:
            self._apply_manual_sync_rows(
                (
                    ("Manual sync status", "blocked"),
                    ("Manual sync result", "Manual Sync requires an active server profile."),
                    ("Pending outgoing", "none"),
                ),
            )
            return
        try:
            result = await control.run_once(
                server_profile_id=server_profile_id,
                authenticated_principal_id=sync_scope["authenticated_principal_id"],
                workspace_scope=sync_scope["workspace_scope"],
            )
        except Exception as exc:
            logger.warning("Settings manual sync run failed.", exc_info=True)
            self._apply_manual_sync_rows(
                (
                    ("Manual sync status", "failed"),
                    ("Manual sync result", f"Manual Sync failed: {type(exc).__name__}"),
                    ("Pending outgoing", "unknown"),
                ),
            )
            return
        self._apply_manual_sync_result(result)

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def _config_path(self) -> Path:
        override = os.environ.get("TLDW_CONFIG_PATH")
        candidate = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
        return validate_path_simple(candidate, require_exists=False).resolve()

    def _config_writable_status(self) -> str:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"invalid path - {redact_secret_text(str(exc))}"
        target = config_path if config_path.exists() else config_path.parent
        writable = os.access(target, os.W_OK) if target.exists() else os.access(target.parent, os.W_OK)
        return "writable" if writable else "not writable"

    def _config_path_overview_value(self) -> str:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"invalid path - {redact_secret_text(str(exc))}"
        source = "Override config" if os.environ.get("TLDW_CONFIG_PATH") else "User config"
        filename = config_path.name or "config.toml"
        return f"{source}: {filename} ({self._config_writable_status()})"

    def _raw_config_text(self) -> str:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"# Unable to use config path: {redact_secret_text(str(exc))}\n"
        if config_path.exists():
            try:
                return config_path.read_text(encoding="utf-8")
            except OSError as exc:
                return f"# Unable to read {config_path}: {type(exc).__name__}"
        return "# Config file does not exist yet.\n"

    @staticmethod
    def _deep_merge_config_values(base: dict, update: Mapping) -> dict:
        merged = copy.deepcopy(base)
        for key, value in update.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key] = SettingsScreen._deep_merge_config_values(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _read_cli_config_without_writes(self) -> dict:
        loaded_config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError):
            return loaded_config
        if not config_path.exists():
            return loaded_config
        try:
            with open(config_path, "rb") as config_file:
                user_config = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError):
            return loaded_config
        if not isinstance(user_config, Mapping):
            return loaded_config
        return self._deep_merge_config_values(loaded_config, user_config)

    def _read_cli_config_value_without_writes(
        self,
        section: str,
        key: str,
        default: object = None,
    ) -> object:
        section_data = self._read_cli_config_without_writes().get(section)
        if isinstance(section_data, Mapping):
            return section_data.get(key, default)
        return default

    def _configured_user_folder_name(self) -> str:
        default_user = DEFAULT_CONFIG_FROM_TOML.get("general", {}).get("users_name", "default_user")
        user_name = self._read_cli_config_value_without_writes("general", "users_name", default_user)
        safe_user_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_name))
        return safe_user_name if safe_user_name else "default_user"

    def _configured_user_data_dir_path(self) -> Path:
        configured_data_dir = self._read_cli_config_value_without_writes("paths", "data_dir", None)
        if configured_data_dir is None:
            configured_data_dir = self._read_cli_config_value_without_writes("Paths", "data_dir", None)
        base_data_dir = (
            Path(str(configured_data_dir)).expanduser()
            if configured_data_dir
            else BASE_DATA_DIR_CLI
        )
        return validate_path_simple(
            base_data_dir / self._configured_user_folder_name(),
            require_exists=False,
        ).resolve()

    def _configured_database_path(self, key: str, default_filename: str) -> Path:
        custom_path = self._read_cli_config_value_without_writes("database", key, None)
        default_path = DEFAULT_CONFIG_FROM_TOML.get("database", {}).get(key)
        if custom_path and custom_path != default_path:
            return validate_path_simple(
                Path(str(custom_path)).expanduser(),
                require_exists=False,
            ).resolve()
        return self._configured_user_data_dir_path() / default_filename

    def _storage_path_entries(self) -> tuple[tuple[str, str, object, bool], ...]:
        return (
            (
                "User data directory",
                "user_data_dir",
                self._configured_user_data_dir_path,
                True,
            ),
            (
                "Notifications DB",
                "notifications_db_path",
                lambda: self._configured_database_path(
                    "notifications_db_path",
                    "tldw_chatbook_notifications.db",
                ),
                False,
            ),
            (
                "Watchlists DB",
                "subscriptions_db_path",
                lambda: self._configured_database_path(
                    "subscriptions_db_path",
                    "tldw_chatbook_subscriptions.db",
                ),
                False,
            ),
            (
                "Workspaces DB",
                "workspaces_db_path",
                lambda: self._configured_database_path(
                    "workspaces_db_path",
                    "tldw_chatbook_workspaces.db",
                ),
                False,
            ),
        )

    def _storage_path_value(self, attr_name: str, fallback_factory: object) -> object:
        value = getattr(self.app_instance, attr_name, None)
        if value:
            return value
        if callable(fallback_factory):
            return fallback_factory()
        return fallback_factory

    def _known_storage_paths(self) -> tuple[str, ...]:
        try:
            paths = [f"Config path: {self._config_path()}"]
        except (OSError, RuntimeError, ValueError) as exc:
            paths = [f"Config path: invalid - {redact_secret_text(str(exc))}"]
        for label, attr_name, fallback_factory, _directory in self._storage_path_entries():
            try:
                value = self._storage_path_value(attr_name, fallback_factory)
            except Exception as exc:
                paths.append(f"{label}: invalid - {redact_secret_text(str(exc))}")
            else:
                paths.append(f"{label}: {value}")
        return tuple(paths)

    @staticmethod
    def _nearest_existing_ancestor(path: Path) -> Path | None:
        candidate = path
        while candidate != candidate.parent:
            if candidate.exists():
                return candidate if candidate.is_dir() else None
            candidate = candidate.parent
        return candidate if candidate.exists() and candidate.is_dir() else None

    def _storage_path_status(self, label: str, path_value: object, *, directory: bool) -> str:
        if path_value is None or str(path_value).strip() in {"", "None"}:
            return f"{label}: not configured"
        try:
            raw_path = Path(str(path_value)).expanduser()
            path = validate_path_simple(raw_path, require_exists=False).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"{label}: invalid - {redact_secret_text(str(exc))}"

        target = path if directory else path.parent
        if target.exists():
            if not target.is_dir():
                return f"{label}: invalid - expected directory"
            writable = os.access(target, os.W_OK | os.X_OK)
            return f"{label}: {'writable' if writable else 'not writable'}"

        existing_target = self._nearest_existing_ancestor(target)
        if existing_target is None or not existing_target.is_dir():
            return f"{label}: not writable"
        writable = os.access(existing_target, os.W_OK | os.X_OK)
        return f"{label}: missing - parent {'writable' if writable else 'not writable'}"

    def _storage_check_results(self) -> tuple[str, ...]:
        rows = ["Storage check: complete"]
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            rows.append(f"Config path parent: invalid - {redact_secret_text(str(exc))}")
        else:
            rows.append(
                self._storage_path_status(
                    "Config path parent",
                    config_path,
                    directory=False,
                )
            )
        for label, attr_name, fallback_factory, directory in self._storage_path_entries():
            status_label = label if directory else f"{label} parent"
            try:
                value = self._storage_path_value(attr_name, fallback_factory)
            except Exception as exc:
                rows.append(f"{status_label}: invalid - {redact_secret_text(str(exc))}")
            else:
                rows.append(self._storage_path_status(status_label, value, directory=directory))
        rows.append("Storage safety: no files were created, moved, or rewritten.")
        return tuple(rows)

    def _storage_check_text(self) -> str:
        return "\n".join(self._storage_check_rows)

    def _update_storage_check_widgets(self) -> None:
        self._set_static_text("#settings-storage-check-result", self._storage_check_text())

    def _apply_storage_check_result(self, rows: tuple[str, ...]) -> None:
        self._storage_check_rows = rows
        self._update_storage_check_widgets()
        self.app.notify("Storage check finished.", severity="information")

    @work(exclusive=True, thread=True)
    def _storage_check_worker(self, values: SettingsStorageDefaults | None = None) -> None:
        rows = (
            build_storage_check_rows(values)
            if values is not None
            else self._storage_check_results()
        )
        self.app.call_from_thread(self._apply_storage_check_result, rows)

    def _skill_trust_posture(self) -> dict[str, object]:
        skill_trust_service = getattr(
            self.app_instance,
            "local_skill_trust_service",
            None,
        )
        if skill_trust_service is None:
            return {
                "enabled": False,
                "trust_status": "unavailable",
                "keyring_convenience_enabled": False,
                "reduced_rollback_protection": False,
            }

        trust_status = "unavailable"
        overall_status = getattr(skill_trust_service, "overall_status", None)
        if callable(overall_status):
            try:
                trust_status = overall_status()
            except Exception:
                logger.warning("Unable to read local skill trust posture.")
                trust_status = "unavailable_error"

        return {
            "enabled": True,
            "trust_status": trust_status,
            "keyring_convenience_enabled": bool(
                getattr(
                    skill_trust_service,
                    "keyring_convenience_enabled",
                    False,
                )
            ),
            "reduced_rollback_protection": bool(
                getattr(
                    skill_trust_service,
                    "reduced_rollback_protection",
                    False,
                )
            ),
        }

    def _settings_privacy_posture(
        self,
        app_config: object | None = None,
    ) -> SettingsPrivacyPosture:
        if app_config is None:
            app_config = getattr(self.app_instance, "app_config", {}) or {}
        return build_settings_privacy_posture(
            app_config,
            skill_trust=self._skill_trust_posture(),
        )

    def _privacy_posture_rows(self, app_config: object | None = None) -> tuple[str, ...]:
        return build_privacy_posture_rows(self._settings_privacy_posture(app_config))

    def _privacy_check_results(self, app_config: object | None = None) -> tuple[str, ...]:
        posture = self._settings_privacy_posture(app_config)
        return (
            "Privacy check: complete",
            *build_privacy_posture_rows(posture),
            (
                "Provider key source: "
                f"environment {posture.provider_env_present} present / "
                f"{posture.provider_env_missing} missing; provider config secrets "
                f"{posture.provider_config_secrets} present"
            ),
        )

    def _privacy_check_text(self) -> str:
        return "\n".join(self._privacy_check_rows)

    def _update_privacy_check_widgets(self) -> None:
        self._set_static_text("#settings-privacy-check-result", self._privacy_check_text())

    def _apply_privacy_check_result(self, rows: tuple[str, ...]) -> None:
        self._privacy_check_rows = rows
        self._update_privacy_check_widgets()
        self.app.notify("Privacy check finished.", severity="information")

    @work(exclusive=True, thread=True)
    def _privacy_check_worker(self, app_config: object) -> None:
        rows = self._privacy_check_results(app_config)
        self.app.call_from_thread(self._apply_privacy_check_result, rows)

    def _appearance_theme_summary(self) -> str:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        if not isinstance(app_config, Mapping):
            return "Theme: default"
        for section_name in ("appearance", "ui", "theme"):
            section = app_config.get(section_name, {})
            if isinstance(section, Mapping):
                theme = section.get("theme") or section.get("name")
                if theme:
                    return f"Theme: {theme} from [{section_name}]"
        return "Theme: default"

    def _set_static_text(self, selector: str, text: str) -> None:
        try:
            self.query_one(selector, Static).update(text)
        except QueryError:
            pass

    def _run_diagnostics_validation(self) -> None:
        self._diagnostics_validation_result = self._validate_current_config()
        self._set_static_text(
            "#settings-diagnostics-validation-result",
            self._diagnostics_validation_result,
        )

    def _run_diagnostics_reload(self) -> None:
        self._diagnostics_reload_result = self._reload_current_config()
        self._set_static_text(
            "#settings-diagnostics-reload-result",
            self._diagnostics_reload_result,
        )

    def _diagnostics_validation_and_reload_results(self) -> tuple[str, str, dict | None]:
        adapter = SettingsConfigAdapter()
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            message = redact_secret_text(str(exc))
            source = "Config source: invalid"
            return (
                f"Config validation: invalid - {message}\n{source}",
                f"Config reload: failed - {message}\n{source}",
                None,
            )
        source = f"Config source: {redact_secret_text(str(config_path))}"
        try:
            validation = adapter.validate_config_file(config_path)
        except Exception as exc:
            message = redact_secret_text(str(exc))
            return (
                f"Config validation: invalid - {message}\n{source}",
                f"Config reload: failed - {message}\n{source}",
                None,
            )

        validation_result = (
            f"Config validation: valid - {redact_secret_text(validation.message)}\n{source}"
            if validation.valid
            else f"Config validation: invalid - {redact_secret_text(validation.message)}\n{source}"
        )
        if not validation.valid:
            return (
                validation_result,
                f"Config reload: failed - {redact_secret_text(validation.message)}\n{source}",
                None,
            )

        try:
            loaded = adapter.load(force_reload=True)
        except Exception as exc:
            return (
                validation_result,
                f"Config reload: failed - {redact_secret_text(str(exc))}\n{source}",
                None,
            )
        if isinstance(loaded, dict):
            return validation_result, f"Config reload: loaded\n{source}", loaded
        return (
            validation_result,
            f"Config reload: failed - loaded config was not a table\n{source}",
            None,
        )

    def _apply_diagnostics_validation_and_reload_result(
        self,
        validation_result: str,
        reload_result: str,
        loaded_config: dict | None,
    ) -> None:
        if loaded_config is not None:
            self.app_instance.app_config = loaded_config
        self._diagnostics_validation_result = validation_result
        self._diagnostics_reload_result = reload_result
        self._set_static_text(
            "#settings-diagnostics-validation-result",
            self._diagnostics_validation_result,
        )
        self._set_static_text(
            "#settings-diagnostics-reload-result",
            self._diagnostics_reload_result,
        )
        self.app.notify("Diagnostics validation and reload finished.", severity="information")

    @work(exclusive=True, thread=True)
    def _diagnostics_validation_and_reload_worker(self) -> None:
        validation_result, reload_result, loaded_config = (
            self._diagnostics_validation_and_reload_results()
        )
        self.app.call_from_thread(
            self._apply_diagnostics_validation_and_reload_result,
            validation_result,
            reload_result,
            loaded_config,
        )

    def _validate_current_config(self) -> str:
        adapter = SettingsConfigAdapter()
        try:
            result = adapter.validate_config_file(self._config_path())
        except Exception as exc:
            return f"Config validation: invalid - {redact_secret_text(str(exc))}"
        if result.valid:
            return f"Config validation: valid - {redact_secret_text(result.message)}"
        return f"Config validation: invalid - {redact_secret_text(result.message)}"

    def _reload_current_config(self) -> str:
        adapter = SettingsConfigAdapter()
        try:
            validation = adapter.validate_config_file(self._config_path())
        except Exception as exc:
            return f"Config reload: failed - {redact_secret_text(str(exc))}"
        if not validation.valid:
            return f"Config reload: failed - {redact_secret_text(validation.message)}"
        try:
            loaded = adapter.load(force_reload=True)
        except Exception as exc:
            return f"Config reload: failed - {redact_secret_text(str(exc))}"
        if isinstance(loaded, dict):
            self.app_instance.app_config = loaded
            return "Config reload: loaded"
        return "Config reload: failed - loaded config was not a table"

    def _advanced_editor_text(self) -> str:
        try:
            return self.query_one("#settings-advanced-config-editor", TextArea).text
        except QueryError:
            return ""

    def _validate_advanced_config_text(self, text: str) -> str:
        result = SettingsConfigAdapter().validate_raw_toml(text)
        status = "valid" if result.valid else "invalid"
        return f"Advanced config validation: {status} - {redact_secret_text(result.message)}"

    def _advanced_validation_status(self, text: str | None = None) -> str:
        current_text = self._advanced_editor_text() if text is None else text
        if self._advanced_config_validated_text is None:
            return "Last validated: not validated"
        if self._advanced_config_validated_text == current_text:
            return "Last validated: current text"
        return "Last validated: stale after edits"

    def _advanced_save_allowed(self, text: str | None = None) -> bool:
        current_text = self._advanced_editor_text() if text is None else text
        return self._advanced_config_validated_text == current_text

    def _update_advanced_validation_status(self) -> None:
        self._set_static_text(
            "#settings-advanced-config-validation-status",
            self._advanced_validation_status(),
        )
        try:
            self.query_one("#settings-advanced-save-config", Button).disabled = (
                not self._advanced_save_allowed()
            )
        except QueryError:
            pass

    def _save_advanced_config_text(self, text: str) -> str:
        validation = SettingsConfigAdapter().validate_raw_toml(text)
        if not validation.valid:
            return f"Advanced config save: blocked - {redact_secret_text(validation.message)}"
        if self._advanced_config_validated_text != text:
            return "Advanced config save: blocked - validate current TOML before save"

        try:
            config_path = self._config_path()
        except ValueError as exc:
            return f"Advanced config save: failed - {redact_secret_text(str(exc))}"
        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        backup_created = False
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
                backup_created = True
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(config_path)
            backup_message = (
                f"backup: {backup_path}"
                if backup_created
                else "backup: none (new file)"
            )
            return f"Advanced config save: saved; {backup_message}"
        except OSError as exc:
            return f"Advanced config save: failed - {redact_secret_text(str(exc))}"

    def _read_advanced_backup_preview(self) -> tuple[str, str | None]:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Advanced config recovery: failed - {redact_secret_text(str(exc))}", None
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        if not backup_path.exists():
            return (
                f"Advanced config recovery: unavailable - no backup found at {backup_path}",
                None,
            )
        try:
            backup_text = backup_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"Advanced config recovery: failed - {redact_secret_text(str(exc))}", None
        return "Advanced config recovery: loaded backup preview; validate before save", backup_text

    def _load_advanced_backup_preview(self) -> str:
        result, backup_text = self._read_advanced_backup_preview()
        if backup_text is None:
            return result
        try:
            self.query_one("#settings-advanced-config-editor", TextArea).text = backup_text
        except QueryError:
            return "Advanced config recovery: failed - editor unavailable"
        self._advanced_config_validated_text = None
        self._update_advanced_validation_status()
        return result

    @work(exclusive=True, thread=True)
    def _advanced_validate_config_worker(self, text: str) -> None:
        validation = SettingsConfigAdapter().validate_raw_toml(text)
        status = "valid" if validation.valid else "invalid"
        result = f"Advanced config validation: {status} - {redact_secret_text(validation.message)}"
        self.app.call_from_thread(
            self._apply_advanced_validation_result,
            text,
            validation.valid,
            result,
        )

    def _apply_advanced_validation_result(self, text: str, valid: bool, result: str) -> None:
        self._advanced_config_result = result
        self._advanced_config_validated_text = text if valid else None
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)
        self._update_advanced_validation_status()

    @work(exclusive=True, thread=True)
    def _advanced_save_config_worker(self, text: str) -> None:
        result = self._save_advanced_config_text(text)
        loaded_config: dict | None = None
        if result.startswith("Advanced config save: saved"):
            try:
                loaded_config = SettingsConfigAdapter().load(force_reload=True)
            except Exception as exc:
                result = f"{result}; reload failed - {redact_secret_text(str(exc))}"
        self.app.call_from_thread(
            self._apply_advanced_save_result,
            result,
            loaded_config,
        )

    def _apply_advanced_save_result(self, result: str, loaded_config: dict | None) -> None:
        if loaded_config is not None:
            self.app_instance.app_config = loaded_config
        self._advanced_config_result = result
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)
        self._update_advanced_validation_status()

    @work(exclusive=True, thread=True)
    def _advanced_load_backup_worker(self) -> None:
        result, backup_text = self._read_advanced_backup_preview()
        self.app.call_from_thread(
            self._apply_advanced_backup_preview_result,
            result,
            backup_text,
        )

    def _apply_advanced_backup_preview_result(
        self,
        result: str,
        backup_text: str | None,
    ) -> None:
        final_result = result
        if backup_text is not None:
            try:
                self.query_one("#settings-advanced-config-editor", TextArea).text = backup_text
            except QueryError:
                final_result = "Advanced config recovery: failed - editor unavailable"
            else:
                self._advanced_config_validated_text = None
                self._update_advanced_validation_status()
        self._advanced_config_result = final_result
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)

    def _provider_readiness_label(self) -> str:
        resolved = self._resolve_provider_model_for_settings()
        provider = str(resolved.provider or "not selected").strip()
        model = str(resolved.model or "not selected").strip()
        if provider and provider != "not selected":
            return f"Provider readiness: {self._provider_display_name(provider)} / {model}"
        return "Provider readiness: needs provider and model"

    def _provider_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)

    def _provider_draft_value(self, key: str):
        draft = self._provider_draft()
        return draft.values.get(key) if draft is not None and key in draft.values else None

    def _resolve_provider_model_for_settings(self):
        draft = self._provider_draft()
        settings_provider = (
            draft.values["provider"] if draft is not None and "provider" in draft.values else None
        )
        settings_model = (
            draft.values["model"] if draft is not None and "model" in draft.values else None
        )
        resolved = resolve_effective_provider_model(
            self.app_instance,
            settings_provider=settings_provider,
            settings_model=settings_model,
        )
        if (
            draft is not None
            and "model" in draft.values
            and not str(draft.values.get("model") or "").strip()
        ):
            return EffectiveProviderModel(
                provider=resolved.provider,
                model="",
                provider_source=resolved.provider_source,
                model_source="settings_draft",
            )
        return resolved

    def _provider_loaded_setting_values(self) -> dict[str, object]:
        resolved = resolve_effective_provider_model(self.app_instance)
        provider = str(resolved.provider or "").strip()
        model = str(resolved.model or "").strip()
        profile = self._provider_model_profile(provider, model)
        return {
            "provider": provider,
            "model": model,
            "endpoint": self._provider_endpoint_value(provider),
            "api_key": "",
            "credential_env_var": self._provider_credential_env_var(provider),
            "model_profile_temperature": profile.get("temperature", ""),
            "model_profile_top_p": profile.get("top_p", ""),
            "model_profile_min_p": profile.get("min_p", ""),
            "model_profile_top_k": profile.get("top_k", ""),
            "model_profile_max_tokens": profile.get("max_tokens", ""),
            "model_profile_seed": profile.get("seed", ""),
            "model_profile_presence_penalty": profile.get("presence_penalty", ""),
            "model_profile_frequency_penalty": profile.get("frequency_penalty", ""),
            "model_profile_reasoning_effort": profile.get("reasoning_effort", ""),
            "model_profile_reasoning_summary": profile.get("reasoning_summary", ""),
            "model_profile_verbosity": profile.get("verbosity", ""),
            "model_profile_thinking_effort": profile.get("thinking_effort", ""),
            "model_profile_thinking_budget_tokens": profile.get("thinking_budget_tokens", ""),
            "model_profile_streaming": profile.get("streaming", ""),
        }

    def _provider_setting_values(self) -> dict[str, object]:
        loaded = self._provider_loaded_setting_values()
        draft = self._provider_draft()
        return {
            key: draft.values[key] if draft is not None and key in draft.values else value
            for key, value in loaded.items()
        }

    def _provider_setting_values_mapping(self) -> Mapping[str, object]:
        values = self._provider_setting_values()
        return values if isinstance(values, Mapping) else {}

    def _provider_display_setting_values(self) -> dict[str, object]:
        """Return provider values for rendering without staging navigation context."""
        values = dict(self._provider_setting_values_mapping())
        if self._provider_draft() is not None or not self._navigation_provider:
            return values
        provider = self._navigation_provider
        model = self._navigation_model or str(values.get("model") or "").strip()
        profile = self._provider_model_profile(provider, model)
        display_values = dict(values)
        display_values.update(
            {
                "provider": provider,
                "model": model,
                "endpoint": self._provider_endpoint_value(provider),
                "api_key": "",
                "credential_env_var": self._provider_credential_env_var(provider),
                "model_profile_temperature": profile.get("temperature", ""),
                "model_profile_top_p": profile.get("top_p", ""),
                "model_profile_min_p": profile.get("min_p", ""),
                "model_profile_top_k": profile.get("top_k", ""),
                "model_profile_max_tokens": profile.get("max_tokens", ""),
                "model_profile_seed": profile.get("seed", ""),
                "model_profile_presence_penalty": profile.get("presence_penalty", ""),
                "model_profile_frequency_penalty": profile.get("frequency_penalty", ""),
                "model_profile_reasoning_effort": profile.get("reasoning_effort", ""),
                "model_profile_reasoning_summary": profile.get("reasoning_summary", ""),
                "model_profile_verbosity": profile.get("verbosity", ""),
                "model_profile_thinking_effort": profile.get("thinking_effort", ""),
                "model_profile_thinking_budget_tokens": profile.get("thinking_budget_tokens", ""),
                "model_profile_streaming": profile.get("streaming", ""),
            }
        )
        return display_values

    def _clear_navigation_provider_context(self) -> None:
        self._navigation_provider = None
        self._navigation_model = None
        self._navigation_field = None

    @staticmethod
    def _normalise_optional_float(
        value: object,
        *,
        min_value: float,
        max_value: float,
        label: str,
    ) -> float | str:
        text = "" if value is None else str(value).strip()
        if not text:
            return ""
        if not validate_number_range(text, min_val=min_value, max_val=max_value):
            raise ValueError(f"{label} must be between {min_value:.1f} and {max_value:.1f}.")
        return float(text)

    def _normalise_model_profile_temperature(self, value: object) -> float | str:
        return self._normalise_optional_float(
            value,
            min_value=0.0,
            max_value=2.0,
            label="Temperature",
        )

    def _normalise_model_profile_top_p(self, value: object) -> float | str:
        return self._normalise_optional_float(
            value,
            min_value=0.0,
            max_value=1.0,
            label="Top P",
        )

    def _normalise_model_profile_min_p(self, value: object) -> float | str:
        return self._normalise_optional_float(
            value,
            min_value=0.0,
            max_value=1.0,
            label="Min P",
        )

    def _normalise_optional_int(
        self,
        value: object,
        *,
        min_value: int,
        label: str,
    ) -> int | str:
        text = "" if value is None else str(value).strip()
        if not text:
            return ""
        if not text.isdecimal() or int(text) < min_value:
            raise ValueError(f"{label} must be a whole number of at least {min_value}.")
        return int(text)

    def _normalise_model_profile_top_k(self, value: object) -> int | str:
        return self._normalise_optional_int(value, min_value=0, label="Top K")

    def _normalise_model_profile_max_tokens(self, value: object) -> int | str:
        return self._normalise_optional_int(value, min_value=1, label="Max tokens")

    def _normalise_model_profile_seed(self, value: object) -> int | str:
        return self._normalise_optional_int(value, min_value=0, label="Seed")

    def _normalise_model_profile_presence_penalty(self, value: object) -> float | str:
        return self._normalise_optional_float(
            value,
            min_value=-2.0,
            max_value=2.0,
            label="Presence penalty",
        )

    def _normalise_model_profile_frequency_penalty(self, value: object) -> float | str:
        return self._normalise_optional_float(
            value,
            min_value=-2.0,
            max_value=2.0,
            label="Frequency penalty",
        )

    @staticmethod
    def _normalise_optional_choice(
        value: object,
        *,
        allowed: frozenset[str],
        label: str,
    ) -> str:
        text = "" if value is None else str(value).strip().lower()
        if text in allowed:
            return text
        allowed_values = ", ".join(sorted(item for item in allowed if item))
        raise ValueError(f"{label} must be one of: {allowed_values}.")

    def _normalise_model_profile_reasoning_effort(self, value: object) -> str:
        return self._normalise_optional_choice(
            value,
            allowed=REASONING_EFFORT_OPTIONS,
            label="Reasoning effort",
        )

    def _normalise_model_profile_reasoning_summary(self, value: object) -> str:
        return self._normalise_optional_choice(
            value,
            allowed=REASONING_SUMMARY_OPTIONS,
            label="Reasoning summary",
        )

    def _normalise_model_profile_verbosity(self, value: object) -> str:
        return self._normalise_optional_choice(
            value,
            allowed=VERBOSITY_OPTIONS,
            label="Verbosity",
        )

    def _normalise_model_profile_thinking_effort(self, value: object) -> str:
        return self._normalise_optional_choice(
            value,
            allowed=THINKING_EFFORT_OPTIONS,
            label="Thinking effort",
        )

    def _normalise_model_profile_thinking_budget_tokens(self, value: object) -> int | str:
        return self._normalise_optional_int(
            value,
            min_value=1024,
            label="Thinking budget tokens",
        )

    @staticmethod
    def _normalise_optional_bool(value: object) -> bool | str:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip()
        if not text:
            return ""
        normalized = text.lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError("Streaming must be true or false.")

    @staticmethod
    def _provider_supports_openai_reasoning(provider: object) -> bool:
        return provider_config_key(str(provider or "")) in OPENAI_REASONING_PROVIDER_KEYS

    @staticmethod
    def _provider_supports_anthropic_thinking(provider: object) -> bool:
        return provider_config_key(str(provider or "")) in ANTHROPIC_THINKING_PROVIDER_KEYS

    def _model_profile_field_supported(self, provider: object, draft_key: str) -> bool:
        if draft_key in OPENAI_REASONING_PROFILE_FIELD_KEYS:
            return self._provider_supports_openai_reasoning(provider)
        if draft_key in ANTHROPIC_THINKING_PROFILE_FIELD_KEYS:
            return self._provider_supports_anthropic_thinking(provider)
        return True

    def _unsupported_model_profile_placeholder(self, provider: object) -> str:
        provider_label = self._provider_display_name(str(provider or "").strip())
        if not provider_label:
            provider_label = "this provider"
        return f"Unavailable for {provider_label}"

    def _model_profile_input_placeholder(self, provider: object, draft_key: str) -> str:
        if not self._model_profile_field_supported(provider, draft_key):
            return self._unsupported_model_profile_placeholder(provider)
        return MODEL_PROFILE_INPUT_PLACEHOLDERS[draft_key]

    def _model_profile_input_value(
        self,
        provider: object,
        draft_key: str,
        value: object,
    ) -> str:
        if not self._model_profile_field_supported(provider, draft_key):
            return ""
        return self._profile_input_value(value)

    def _provider_generation_support_copy(self, provider: object) -> str:
        provider_label = self._provider_display_name(str(provider or "").strip())
        if not provider_label:
            provider_label = "this provider"
        reasoning_status = (
            f"Reasoning available for {provider_label}"
            if self._provider_supports_openai_reasoning(provider)
            else f"Reasoning unavailable for {provider_label}"
        )
        thinking_status = (
            f"Thinking available for {provider_label}"
            if self._provider_supports_anthropic_thinking(provider)
            else f"Thinking unavailable for {provider_label}"
        )
        return f"{reasoning_status}; {thinking_status}."

    def _provider_form_values_from_widgets(self) -> dict[str, object]:
        loaded_values = self._provider_loaded_setting_values()
        provider_value = self._provider_widget_value()
        provider_draft = self._provider_draft()
        provider_explicitly_staged = (
            provider_draft is not None and "provider" in provider_draft.values
        )
        loaded_provider = str(loaded_values["provider"])
        provider = (
            loaded_provider
            if (
                provider_value
                and not provider_explicitly_staged
                and provider_config_key(provider_value) == provider_config_key(loaded_provider)
            )
            else (
                provider_value
                if provider_value or provider_explicitly_staged
                else loaded_provider
            )
        )
        model = (
            self.query_one("#settings-model-value", Input).value.strip()
            or str(loaded_values["model"])
        )
        endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
        api_key = self.query_one("#settings-provider-api-key", Input).value.strip()
        credential_env_var = self.query_one(
            "#settings-provider-credential-env-var",
            Input,
        ).value.strip()
        model_profile_temperature = self._normalise_model_profile_temperature(
            self.query_one("#settings-model-profile-temperature", Input).value
        )
        model_profile_top_p = self._normalise_model_profile_top_p(
            self.query_one("#settings-model-profile-top-p", Input).value
        )
        model_profile_min_p = self._normalise_model_profile_min_p(
            self.query_one("#settings-model-profile-min-p", Input).value
        )
        model_profile_top_k = self._normalise_model_profile_top_k(
            self.query_one("#settings-model-profile-top-k", Input).value
        )
        model_profile_max_tokens = self._normalise_model_profile_max_tokens(
            self.query_one("#settings-model-profile-max-tokens", Input).value
        )
        model_profile_seed = self._normalise_model_profile_seed(
            self.query_one("#settings-model-profile-seed", Input).value
        )
        model_profile_presence_penalty = self._normalise_model_profile_presence_penalty(
            self.query_one("#settings-model-profile-presence-penalty", Input).value
        )
        model_profile_frequency_penalty = self._normalise_model_profile_frequency_penalty(
            self.query_one("#settings-model-profile-frequency-penalty", Input).value
        )
        model_profile_reasoning_effort = self._normalise_model_profile_reasoning_effort(
            self.query_one("#settings-model-profile-reasoning-effort", Input).value
        )
        model_profile_reasoning_summary = self._normalise_model_profile_reasoning_summary(
            self.query_one("#settings-model-profile-reasoning-summary", Input).value
        )
        model_profile_verbosity = self._normalise_model_profile_verbosity(
            self.query_one("#settings-model-profile-verbosity", Input).value
        )
        model_profile_thinking_effort = self._normalise_model_profile_thinking_effort(
            self.query_one("#settings-model-profile-thinking-effort", Input).value
        )
        model_profile_thinking_budget_tokens = self._normalise_model_profile_thinking_budget_tokens(
            self.query_one("#settings-model-profile-thinking-budget-tokens", Input).value
        )
        model_profile_streaming = self._normalise_optional_bool(
            self.query_one("#settings-model-profile-streaming", Input).value
        )
        if not self._provider_supports_openai_reasoning(provider):
            model_profile_reasoning_effort = ""
            model_profile_reasoning_summary = ""
            model_profile_verbosity = ""
        if not self._provider_supports_anthropic_thinking(provider):
            model_profile_thinking_effort = ""
            model_profile_thinking_budget_tokens = ""
        return {
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "api_key": api_key,
            "credential_env_var": credential_env_var,
            "model_profile_temperature": model_profile_temperature,
            "model_profile_top_p": model_profile_top_p,
            "model_profile_min_p": model_profile_min_p,
            "model_profile_top_k": model_profile_top_k,
            "model_profile_max_tokens": model_profile_max_tokens,
            "model_profile_seed": model_profile_seed,
            "model_profile_presence_penalty": model_profile_presence_penalty,
            "model_profile_frequency_penalty": model_profile_frequency_penalty,
            "model_profile_reasoning_effort": model_profile_reasoning_effort,
            "model_profile_reasoning_summary": model_profile_reasoning_summary,
            "model_profile_verbosity": model_profile_verbosity,
            "model_profile_thinking_effort": model_profile_thinking_effort,
            "model_profile_thinking_budget_tokens": model_profile_thinking_budget_tokens,
            "model_profile_streaming": model_profile_streaming,
        }

    def _stage_provider_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.PROVIDERS_MODELS
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        if key == "api_key":
            provider = str(self._provider_setting_values_mapping().get("provider") or "").strip()
            original = self._provider_api_key_value(provider)
        else:
            original = self._provider_loaded_setting_values().get(key)
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _provider_config_entry(self, provider: str) -> tuple[str | None, Mapping[str, object]]:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        api_settings = app_config.get("api_settings", {}) if isinstance(app_config, Mapping) else {}
        if not isinstance(api_settings, Mapping):
            return None, {}
        target_key = provider_config_key(provider)
        if not target_key:
            return None, {}
        for configured_provider, configured_settings in api_settings.items():
            if provider_config_key(str(configured_provider)) == target_key:
                if isinstance(configured_settings, Mapping):
                    return str(configured_provider), configured_settings
                return str(configured_provider), {}
        return None, {}

    def _provider_config(self, provider: str) -> Mapping[str, object]:
        _section_key, provider_config = self._provider_config_entry(provider)
        return provider_config

    def _provider_credential_env_var(self, provider: str) -> str:
        env_var = self._provider_config(provider).get("api_key_env_var", "")
        return str(env_var or "").strip()

    def _provider_api_key_value(self, provider: str) -> str:
        api_key = self._provider_config(provider).get("api_key", "")
        return str(api_key or "").strip() if isinstance(api_key, str) else ""

    def _provider_readiness_app_config(self) -> Mapping[str, object]:
        """Return app config for provider-readiness checks."""
        try:
            app_config = getattr(self.app, "app_config")
        except (AttributeError, NoActiveAppError):
            return getattr(self.app_instance, "app_config", {}) or {}
        return app_config or {}

    def _provider_saved_api_key_present(self, provider: str) -> bool:
        readiness = get_provider_readiness(
            provider,
            self._provider_readiness_app_config(),
        )
        return bool(
            readiness.api_key_source
            and readiness.api_key_source.startswith("config:")
        )

    def _provider_api_key_placeholder(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return "Select provider first"
        readiness = get_provider_readiness(
            provider,
            self._provider_readiness_app_config(),
        )
        if not readiness.requires_api_key:
            return "No credential required"
        if self._provider_saved_api_key_present(provider):
            return "Local config key saved; paste a replacement to change it"
        return "Paste API key to save locally in config"

    def _provider_credential_status(self, provider: str) -> str:
        readiness = get_provider_readiness(
            provider,
            self._provider_readiness_app_config(),
        )
        if self._provider_saved_api_key_present(provider):
            return "API key source: local config key saved"
        if readiness.api_key_source and readiness.api_key_source.startswith("env:"):
            return f"API key source: {readiness.api_key_source}"
        if not readiness.requires_api_key:
            return "API key source: not required for this provider"
        if readiness.env_var:
            return f"API key source: missing; set {readiness.env_var} or paste a local key"
        return "API key source: missing"

    def _provider_credential_placeholder(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return "Select provider first"
        readiness = get_provider_readiness(
            provider,
            self._provider_readiness_app_config(),
        )
        if not readiness.requires_api_key:
            return "No credential required"
        if readiness.env_var:
            return readiness.env_var
        return f"{provider_key.upper()}_API_KEY"

    def _provider_catalog_entries(self) -> tuple[ConsoleProviderCatalogEntry, ...]:
        return supported_console_provider_catalog(
            handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        )

    def _provider_catalog_keys(self) -> frozenset[str]:
        return frozenset(entry.readiness_key for entry in self._provider_catalog_entries())

    def _provider_display_name(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        for entry in self._provider_catalog_entries():
            if entry.readiness_key == provider_key:
                return entry.display_name
        return provider

    def _provider_select_options(self) -> list[tuple[str, str]]:
        options = [
            (f"{entry.display_name} ({entry.readiness_key})", entry.readiness_key)
            for entry in self._provider_catalog_entries()
        ]
        options.append((PROVIDER_MANUAL_SELECT_LABEL, PROVIDER_MANUAL_SELECT_VALUE))
        return options

    def _provider_select_value_for_provider(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if provider_key in self._provider_catalog_keys():
            return provider_key
        return PROVIDER_MANUAL_SELECT_VALUE

    def _provider_catalog_model_default(self, provider: str) -> str:
        providers_models = getattr(self.app_instance, "providers_models", None)
        if not isinstance(providers_models, Mapping):
            return ""
        provider_key = provider_config_key(provider)
        for configured_provider, configured_models in providers_models.items():
            if provider_config_key(str(configured_provider)) != provider_key:
                continue
            if isinstance(configured_models, (str, bytes)) or not isinstance(
                configured_models,
                Sequence,
            ):
                continue
            for configured_model in configured_models:
                model = str(configured_model or "").strip()
                if model and model != "None":
                    return model
        return ""

    def _provider_model_default(self, provider: str) -> str:
        configured_model = str(self._provider_config(provider).get("model") or "").strip()
        if configured_model and configured_model != "None":
            return configured_model
        return self._provider_catalog_model_default(provider)

    @staticmethod
    def _select_value_text(value: object) -> str:
        if value is None or value is Select.BLANK:
            return ""
        return str(value).strip()

    def _provider_widget_value(self) -> str:
        try:
            provider_select = self.query_one("#settings-provider-value", Select)
            selected_value = self._select_value_text(provider_select.value)
            if selected_value == PROVIDER_MANUAL_SELECT_VALUE:
                try:
                    return self.query_one("#settings-provider-manual-value", Input).value.strip()
                except QueryError:
                    return ""
            return selected_value
        except QueryError:
            try:
                return self.query_one("#settings-provider-value", Input).value.strip()
            except QueryError:
                return str(self._provider_setting_values_mapping().get("provider") or "").strip()

    def _sync_provider_manual_widget(self, provider: str) -> None:
        try:
            provider_select = self.query_one("#settings-provider-value", Select)
            manual_row = self.query_one("#settings-provider-manual-row", Horizontal)
            manual_input = self.query_one("#settings-provider-manual-value", Input)
        except QueryError:
            return
        select_value = self._provider_select_value_for_provider(provider)
        uses_manual_entry = select_value == PROVIDER_MANUAL_SELECT_VALUE
        self._syncing_provider_selection = True
        try:
            provider_select.value = select_value
        finally:
            self._syncing_provider_selection = False
        self._syncing_provider_manual = True
        try:
            manual_input.disabled = not uses_manual_entry
            manual_input.value = provider if uses_manual_entry else ""
            manual_row.set_class(not uses_manual_entry, "settings-provider-manual-hidden")
        finally:
            self._syncing_provider_manual = False

    def _provider_catalog_summary(self) -> str:
        catalog = ", ".join(entry.readiness_key for entry in self._provider_catalog_entries())
        return f"Provider catalog: {catalog}"

    def _provider_catalog_key_policy(self) -> str:
        entries = self._provider_catalog_entries()
        key_required = sum(1 for entry in entries if entry.requires_api_key)
        keyless = sum(1 for entry in entries if not entry.requires_api_key)
        return f"Credential policy: {key_required} require keys; {keyless} local/keyless providers"

    def _sync_provider_runtime_defaults(self, provider: str, model: str) -> None:
        """Keep Console-facing app defaults aligned after a Settings save."""
        if hasattr(self.app_instance, "chat_api_provider_value"):
            self.app_instance.chat_api_provider_value = provider
        if hasattr(self.app_instance, "chat_api_model_value"):
            self.app_instance.chat_api_model_value = model
        if hasattr(self.app_instance, "chat_model_value"):
            self.app_instance.chat_model_value = model

    def _provider_model_defaults(self, provider: str) -> Mapping[str, object]:
        model_defaults = self._provider_config(provider).get("model_defaults", {})
        return model_defaults if isinstance(model_defaults, Mapping) else {}

    def _provider_model_profile(self, provider: str, model: str) -> Mapping[str, object]:
        model_name = str(model or "").strip()
        if not model_name:
            return {}
        profile = self._provider_model_defaults(provider).get(model_name, {})
        return profile if isinstance(profile, Mapping) else {}

    def _updated_model_defaults_for_values(
        self,
        provider: str,
        model: str,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        model_name = str(model or "").strip()
        model_defaults = copy.deepcopy(dict(self._provider_model_defaults(provider)))
        current_profile = model_defaults.get(model_name, {})
        next_profile = copy.deepcopy(current_profile) if isinstance(current_profile, Mapping) else {}
        for draft_key, profile_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS.items():
            if not self._model_profile_field_supported(provider, draft_key):
                next_profile.pop(profile_key, None)
                continue
            value = values.get(draft_key, "")
            if value == "":
                next_profile.pop(profile_key, None)
            else:
                next_profile[profile_key] = value
        if next_profile:
            model_defaults[model_name] = next_profile
        else:
            model_defaults.pop(model_name, None)
        return model_defaults

    @staticmethod
    def _profile_input_value(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value if value is not None else "")

    def _clear_provider_auxiliary_draft_keys(self) -> None:
        draft = self._provider_draft()
        if draft is None:
            return
        for key in (
            "endpoint",
            "api_key",
            "credential_env_var",
            *PROVIDER_MODEL_PROFILE_FIELD_KEYS,
        ):
            draft.values.pop(key, None)
            draft.originals.pop(key, None)
        if not draft.is_dirty:
            self._settings_drafts.pop(SettingsCategoryId.PROVIDERS_MODELS, None)

    def _clear_provider_model_profile_draft_keys(self) -> None:
        draft = self._provider_draft()
        if draft is None:
            return
        for key in PROVIDER_MODEL_PROFILE_FIELD_KEYS:
            draft.values.pop(key, None)
            draft.originals.pop(key, None)
        if not draft.is_dirty:
            self._settings_drafts.pop(SettingsCategoryId.PROVIDERS_MODELS, None)

    def _sync_provider_credential_widget(self, provider: str) -> None:
        try:
            credential_input = self.query_one("#settings-provider-credential-env-var", Input)
        except QueryError:
            credential_input = None
        try:
            api_key_input = self.query_one("#settings-provider-api-key", Input)
        except QueryError:
            api_key_input = None
        draft = self._provider_draft()
        self._syncing_provider_credential_env_var = True
        self._syncing_provider_api_key = True
        try:
            if credential_input is not None:
                credential_input.value = self._provider_credential_env_var(provider)
                credential_input.placeholder = self._provider_credential_placeholder(provider)
            if api_key_input is not None:
                api_key_input.value = (
                    str(draft.values.get("api_key") or "")
                    if draft is not None and "api_key" in draft.values
                    else ""
                )
                api_key_input.placeholder = self._provider_api_key_placeholder(provider)
        finally:
            self._syncing_provider_credential_env_var = False
            self._syncing_provider_api_key = False

    def _sync_provider_model_profile_widgets(self, provider: str, model: str) -> None:
        profile = self._provider_model_profile(provider, model)
        input_values = {
            "model_profile_temperature": profile.get("temperature", ""),
            "model_profile_top_p": profile.get("top_p", ""),
            "model_profile_min_p": profile.get("min_p", ""),
            "model_profile_top_k": profile.get("top_k", ""),
            "model_profile_max_tokens": profile.get("max_tokens", ""),
            "model_profile_seed": profile.get("seed", ""),
            "model_profile_presence_penalty": profile.get("presence_penalty", ""),
            "model_profile_frequency_penalty": profile.get("frequency_penalty", ""),
            "model_profile_reasoning_effort": profile.get("reasoning_effort", ""),
            "model_profile_reasoning_summary": profile.get("reasoning_summary", ""),
            "model_profile_verbosity": profile.get("verbosity", ""),
            "model_profile_thinking_effort": profile.get("thinking_effort", ""),
            "model_profile_thinking_budget_tokens": profile.get("thinking_budget_tokens", ""),
            "model_profile_streaming": profile.get("streaming", ""),
        }
        self._syncing_provider_model_profile = True
        try:
            for draft_key, value in input_values.items():
                selector = f"#settings-{draft_key.replace('_', '-')}"
                try:
                    widget = self.query_one(selector, Input)
                except QueryError:
                    continue
                supported = self._model_profile_field_supported(provider, draft_key)
                widget.disabled = not supported
                widget.placeholder = self._model_profile_input_placeholder(provider, draft_key)
                widget.value = self._profile_input_value(value) if supported else ""
        finally:
            self._syncing_provider_model_profile = False

    def _provider_endpoint_setting_key(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        provider_config = self._provider_config(provider)
        for key in PROVIDER_ENDPOINT_KEYS:
            if key in provider_config:
                return key
        return "api_url" if provider_key in API_URL_PROVIDER_KEYS else "api_base_url"

    def _provider_endpoint_value(self, provider: str) -> str:
        provider_config = self._provider_config(provider)
        endpoint_key = self._provider_endpoint_setting_key(provider)
        value = provider_config.get(endpoint_key)
        if value is None:
            for key in PROVIDER_ENDPOINT_KEYS:
                value = provider_config.get(key)
                if value is not None:
                    break
        return str(value or "").strip()

    def _provider_endpoint_placeholder(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return "Select a provider before setting an endpoint"
        if provider_key in PROVIDER_ENDPOINT_PLACEHOLDERS:
            return PROVIDER_ENDPOINT_PLACEHOLDERS[provider_key]
        if provider_key in API_URL_PROVIDER_KEYS:
            return "https://host:port/v1"
        return "Optional provider endpoint override"

    def _provider_endpoint_summary(self, provider: str, endpoint: object | None = None) -> str:
        provider_key = provider_config_key(provider)
        endpoint_key = self._provider_endpoint_setting_key(provider)
        endpoint_value = str(
            endpoint if endpoint is not None else self._provider_endpoint_value(provider)
        ).strip()
        if not provider_key:
            return "Endpoint: provider required before saving"
        if endpoint_value:
            return f"Endpoint: api_settings.{provider_key}.{endpoint_key}={endpoint_value}"
        if provider_key in API_URL_PROVIDER_KEYS:
            return f"Endpoint: api_settings.{provider_key}.{endpoint_key} not configured"
        return f"Endpoint: api_settings.{provider_key}.{endpoint_key} or provider default"

    def _provider_endpoint_display_value(self, provider: str, endpoint: object | None = None) -> str:
        provider_key = provider_config_key(provider)
        endpoint_value = str(
            endpoint if endpoint is not None else self._provider_endpoint_value(provider)
        ).strip()
        if not provider_key:
            return "provider required before saving"
        if endpoint_value:
            return endpoint_value
        if provider_key in API_URL_PROVIDER_KEYS:
            return "not configured"
        return "provider default"

    def _provider_endpoint_row(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return "Endpoint key: provider required"
        endpoint_key = self._provider_endpoint_setting_key(provider)
        return f"Endpoint key: api_settings.{provider_key}.{endpoint_key}"

    @staticmethod
    def _settings_source_label(source: object) -> str:
        source_key = str(source or "").strip()
        if not source_key:
            return "Unknown"
        return SETTINGS_SOURCE_LABELS.get(source_key, source_key.replace("_", " "))

    @staticmethod
    def _validate_provider_endpoint(endpoint: object) -> str | None:
        endpoint_text = str(endpoint or "").strip()
        if not endpoint_text:
            return None
        if not validate_url(endpoint_text):
            return "Endpoint must start with http:// or https:// and include a valid host."
        return None

    @staticmethod
    def _validate_credential_env_var(credential_env_var: object) -> str | None:
        env_var = str(credential_env_var or "").strip()
        if not env_var:
            return None
        sanitized = sanitize_string(env_var, max_length=128)
        if (
            sanitized != env_var
            or not validate_text_input(env_var, max_length=128, allow_html=False)
            or PROVIDER_CREDENTIAL_ENV_VAR_PATTERN.fullmatch(env_var) is None
        ):
            return (
                "Credential env var must use environment variable syntax: "
                "letters, numbers, and underscores; start with a letter or underscore."
            )
        return None

    @staticmethod
    def _validate_provider_api_key(api_key: object) -> str | None:
        return provider_api_key_validation_error(api_key)

    def _provider_key_status(self, provider: str) -> str:
        readiness = get_provider_readiness(
            provider,
            self._provider_readiness_app_config(),
        )
        if readiness.api_key_source:
            return f"API key: {readiness.api_key_source}"
        if not readiness.requires_api_key:
            return "API key: not required for this provider"
        if readiness.env_var:
            return f"{readiness.env_var}=missing"
        return "API key: missing"

    def _model_discovery_available(self, provider: str) -> bool:
        return bool(provider_config_key(provider)) and getattr(
            self.app_instance,
            "llm_provider_catalog_scope_service",
            None,
        ) is not None

    def _provider_discovery_staged_settings(self, provider: str) -> dict[str, object]:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return {"api_settings": {}}
        provider_section_key, _provider_config = self._provider_config_entry(provider)
        provider_save_key = provider_section_key or provider_key
        endpoint = ""
        credential_env_var = ""
        try:
            endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
            credential_env_var = self.query_one(
                "#settings-provider-credential-env-var",
                Input,
            ).value.strip()
        except QueryError:
            values = self._provider_setting_values_mapping()
            endpoint = str(values.get("endpoint") or "").strip()
            credential_env_var = str(values.get("credential_env_var") or "").strip()
        provider_settings: dict[str, object] = {}
        if endpoint:
            provider_settings[self._provider_endpoint_setting_key(provider)] = endpoint
        if credential_env_var:
            provider_settings["api_key_env_var"] = credential_env_var
        return {"api_settings": {provider_save_key: provider_settings}}

    def _model_discovery_selection_options(self) -> list[tuple[str, str, bool]]:
        options: list[tuple[str, str, bool]] = []
        for model in self._model_discovery_models:
            model_id = str(getattr(model, "model_id", "") or "").strip()
            if not model_id:
                continue
            source = str(getattr(model, "source", "runtime_discovered"))
            capability = str(getattr(model, "capability_status", "unknown"))
            persisted = "saved" if bool(getattr(model, "persisted", False)) else "runtime"
            label = f"{model_id} | {persisted} | {source} | capability={capability}"
            options.append(
                (
                    label,
                    model_id,
                    model_id in self._model_discovery_selected_model_ids,
                )
            )
        return options

    def _reset_provider_model_discovery_state(
        self,
        status: str = MODEL_DISCOVERY_IDLE_COPY,
    ) -> None:
        self._model_discovery_status = status
        self._model_discovery_models = ()
        self._model_discovery_selected_model_ids = set()
        self._refresh_model_discovery_widgets()

    def _discovery_status_from_error(self, result: object) -> str:
        error = getattr(result, "error", None)
        kind = str(getattr(error, "kind", "") or "")
        if kind == "ambiguous_provider_key":
            return MODEL_DISCOVERY_AMBIGUOUS_PROVIDER_COPY
        if kind == "unsupported_endpoint":
            return MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY
        message = str(getattr(error, "message", "") or "").strip()
        recovery = str(getattr(error, "recovery_hint", "") or "").strip()
        if recovery:
            return redact_secret_text(f"{message} {recovery}".strip())
        if message:
            return redact_secret_text(message)
        return "Model discovery failed. Check provider endpoint settings and try again."

    def _refresh_model_discovery_widgets(self) -> None:
        self._set_static_text("#settings-model-discovery-status", self._model_discovery_status)
        try:
            self.query_one("#settings-model-discovery-empty", Static).display = (
                not self._model_discovery_models
            )
        except QueryError:
            pass
        try:
            discover_button = self.query_one("#settings-discover-provider-models", Button)
            discover_button.disabled = not self._model_discovery_available(
                self._provider_widget_value()
            )
        except QueryError:
            pass
        try:
            save_button = self.query_one("#settings-save-discovered-provider-models", Button)
            save_button.disabled = not self._model_discovery_models
        except QueryError:
            pass
        try:
            clear_button = self.query_one("#settings-clear-discovered-provider-models", Button)
            clear_button.disabled = not self._model_discovery_models
        except QueryError:
            pass
        try:
            discovered_list = self.query_one("#settings-discovered-models-list", SelectionList)
            discovered_list.clear_options()
            discovered_list.add_options(self._model_discovery_selection_options())
            discovered_list.disabled = not self._model_discovery_models
        except QueryError:
            pass

    def _append_saved_discovered_models(
        self,
        provider_list_key: str | None,
        model_ids: tuple[str, ...],
    ) -> None:
        if not provider_list_key or not model_ids:
            return
        providers_models = getattr(self.app_instance, "providers_models", None)
        if not isinstance(providers_models, dict):
            providers_models = {}
            self.app_instance.providers_models = providers_models
        current = providers_models.get(provider_list_key, [])
        if not isinstance(current, list):
            current = list(current) if isinstance(current, tuple) else []
        seen = {model for model in current if isinstance(model, str)}
        for model_id in model_ids:
            if model_id not in seen:
                current.append(model_id)
                seen.add(model_id)
        providers_models[provider_list_key] = current

    @work(exclusive=True, group="settings-model-discovery")
    async def _discover_provider_models_worker(self) -> None:
        await self._discover_provider_models()

    async def _discover_provider_models(self) -> None:
        provider = self._provider_widget_value()
        provider_key = provider_config_key(provider)
        scope_service = getattr(self.app_instance, "llm_provider_catalog_scope_service", None)
        if not provider_key or scope_service is None:
            self._model_discovery_status = (
                "Provider is required before discovering models."
            )
            self._model_discovery_models = ()
            self._model_discovery_selected_model_ids = set()
            self._refresh_model_discovery_widgets()
            return

        staged_settings = self._provider_discovery_staged_settings(provider)
        self._model_discovery_status = "Model discovery: running"
        self._model_discovery_models = ()
        self._model_discovery_selected_model_ids = set()
        self._refresh_model_discovery_widgets()
        try:
            result = await scope_service.discover_models(
                mode="local",
                provider=provider_key,
                staged_settings=staged_settings,
            )
        except Exception as exc:
            logger.exception("Provider model discovery failed")
            self._model_discovery_status = redact_secret_text(
                f"Model discovery failed: {exc}"
            )
            self._model_discovery_models = ()
            self._model_discovery_selected_model_ids = set()
            self._refresh_model_discovery_widgets()
            return

        if str(getattr(result, "status", "")) == "success":
            models = tuple(getattr(result, "models", ()) or ())
            provider_list_key = str(
                getattr(result, "provider_list_key", None) or provider_key
            )
            self._model_discovery_models = models
            self._model_discovery_selected_model_ids = set()
            self._model_discovery_status = (
                f"Discovered {len(models)} model(s) from {provider_list_key}."
            )
            self._refresh_model_discovery_widgets()
            self.app.notify("Provider model discovery finished.", severity="information")
            return

        self._model_discovery_models = ()
        self._model_discovery_selected_model_ids = set()
        self._model_discovery_status = self._discovery_status_from_error(result)
        self._refresh_model_discovery_widgets()
        self.app.notify("Provider model discovery failed.", severity="warning")

    @work(exclusive=True, group="settings-model-discovery")
    async def _save_selected_discovered_provider_models_worker(self) -> None:
        await self._save_selected_discovered_provider_models()

    async def _save_selected_discovered_provider_models(self) -> None:
        provider = self._provider_widget_value()
        provider_key = provider_config_key(provider)
        scope_service = getattr(self.app_instance, "llm_provider_catalog_scope_service", None)
        if not provider_key or scope_service is None:
            self._model_discovery_status = "Provider is required before saving discovered models."
            self._refresh_model_discovery_widgets()
            return
        try:
            discovered_list = self.query_one("#settings-discovered-models-list", SelectionList)
            selected_model_ids = [str(model_id) for model_id in discovered_list.selected]
        except QueryError:
            selected_model_ids = sorted(self._model_discovery_selected_model_ids)
        if not selected_model_ids:
            self._model_discovery_status = "Select discovered models to save."
            self._refresh_model_discovery_widgets()
            self.app.notify("Select discovered models before saving.", severity="warning")
            return

        self._model_discovery_selected_model_ids = set(selected_model_ids)
        self._model_discovery_status = "Saving selected discovered models..."
        self._refresh_model_discovery_widgets()
        try:
            result = await scope_service.persist_discovered_models_to_settings(
                mode="local",
                provider=provider_key,
                model_ids=selected_model_ids,
            )
        except Exception as exc:
            logger.exception("Provider model discovery persistence failed")
            self._model_discovery_status = redact_secret_text(
                f"Could not save discovered models: {exc}"
            )
            self._refresh_model_discovery_widgets()
            return

        message = str(getattr(result, "message", "") or "").strip()
        status = str(getattr(result, "status", "") or "")
        saved_model_ids = tuple(
            str(model_id)
            for model_id in tuple(getattr(result, "saved_model_ids", ()) or ())
            if str(model_id).strip()
        )
        if status == "saved":
            provider_list_key = getattr(result, "provider_list_key", None)
            self._append_saved_discovered_models(provider_list_key, saved_model_ids)
            self._model_discovery_status = (
                message
                or f"Saved {len(saved_model_ids)} discovered model(s)."
            )
            self._refresh_model_discovery_widgets()
            self.app.notify("Discovered models saved.", severity="information")
            return

        if status == "ambiguous_provider_key":
            self._model_discovery_status = MODEL_DISCOVERY_AMBIGUOUS_PROVIDER_COPY
        else:
            self._model_discovery_status = redact_secret_text(
                message or "Could not save discovered models."
            )
        self._refresh_model_discovery_widgets()
        self.app.notify("Discovered model save failed.", severity="warning")

    @work(exclusive=True, group="settings-model-discovery")
    async def _clear_discovered_provider_models_worker(self) -> None:
        await self._clear_discovered_provider_models()

    async def _clear_discovered_provider_models(self) -> None:
        provider = self._provider_widget_value()
        provider_key = provider_config_key(provider)
        scope_service = getattr(self.app_instance, "llm_provider_catalog_scope_service", None)
        if provider_key and scope_service is not None:
            try:
                await scope_service.clear_discovered_models(
                    mode="local",
                    provider=provider_key,
                )
            except Exception:
                logger.exception("Provider discovered model cache clear failed")
        self._reset_provider_model_discovery_state("Discovered model cache cleared.")

    def _run_provider_readiness_test(self) -> str:
        try:
            provider = self._provider_widget_value()
            model = self.query_one("#settings-model-value", Input).value.strip()
        except QueryError:
            values = self._provider_setting_values()
            provider = str(values.get("provider") or "").strip()
            model = str(values.get("model") or "").strip()

        readiness = get_provider_readiness(
            provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )

        findings: list[str] = ["Provider test"]
        findings.append(readiness.user_message)
        if not model:
            findings.append("model=missing")
        else:
            findings.append(f"model={model}")

        if readiness.api_key_source:
            findings.append(f"api_key_source={readiness.api_key_source}")
        if readiness.env_var:
            raw_value = os.environ.get(readiness.env_var)
            findings.append(f"{readiness.env_var}={raw_value if raw_value else 'missing'}")
        elif not readiness.requires_api_key:
            findings.append("api_key=not required")
        findings.append(self._provider_endpoint_summary(provider))

        status = "ready" if readiness.ready and model else "blocked"
        findings.append(f"status={status}")
        return redact_secret_text(" | ".join(findings))

    def _update_provider_test_result(self) -> None:
        try:
            self.query_one("#settings-provider-test-result", Static).update(self._provider_test_result)
        except QueryError:
            pass

    def _update_provider_dynamic_widgets(self) -> None:
        try:
            provider = self._provider_widget_value()
        except QueryError:
            provider = str(self._provider_setting_values_mapping().get("provider") or "")
        try:
            endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
        except QueryError:
            endpoint = self._provider_endpoint_value(provider)
        readiness_label = self._provider_readiness_label()
        resolved = self._resolve_provider_model_for_settings()
        self._set_static_text(
            "#settings-provider-source",
            f"Provider source: {self._settings_source_label(resolved.provider_source)}",
        )
        self._set_static_text(
            "#settings-model-source",
            f"Model source: {self._settings_source_label(resolved.model_source)}",
        )
        try:
            self.query_one("#settings-provider-readiness", Static).update(
                f"Readiness: {readiness_label.removeprefix('Provider readiness: ')}"
            )
            self.query_one("#settings-provider-inspector-readiness", Static).update(readiness_label)
            self.query_one("#settings-provider-endpoint-key", Static).update(
                self._provider_endpoint_row(provider)
            )
            self.query_one("#settings-provider-endpoint", Static).update(
                f"Endpoint: {self._provider_endpoint_display_value(provider, endpoint)}"
            )
            self.query_one("#settings-provider-key-status", Static).update(
                self._provider_key_status(provider)
            )
            self.query_one("#settings-provider-credential-status", Static).update(
                self._provider_credential_status(provider)
            )
            api_key_input = self.query_one("#settings-provider-api-key", Input)
            api_key_input.placeholder = self._provider_api_key_placeholder(provider)
            clear_button = self.query_one("#settings-provider-api-key-clear", Button)
            clear_button.disabled = (
                not self._provider_saved_api_key_present(provider)
                and not bool(api_key_input.value.strip())
            )
        except QueryError:
            pass
        self._set_static_text(
            "#settings-provider-generation-support",
            self._provider_generation_support_copy(provider),
        )
        self._refresh_provider_field_guidance()

    def _detail_row(self, label: str, value: object, *, identifier: str | None = None) -> Static:
        return Static(
            f"{label}: {value}",
            id=identifier,
            classes="settings-detail-row",
        )

    def _provider_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        provider = str(self._provider_setting_values_mapping().get("provider") or "").strip()
        endpoint_key = self._provider_endpoint_row(provider).removeprefix("Endpoint key: ")
        provider_config_prefix = (
            f"api_settings.{provider_config_key(provider)}"
            if provider_config_key(provider)
            else "api_settings.<provider>"
        )
        field_id = self._active_settings_field_id
        if field_id == "settings-provider-value":
            return (
                ("Focused setting", "Provider"),
                ("Purpose", "Selects the provider used for Console generation defaults."),
                ("Saved as", "chat_defaults.provider"),
                ("Validation", "choose a catalog provider or use Manual for custom aliases"),
            )
        if field_id == "settings-provider-manual-value":
            return (
                ("Focused setting", "Manual provider"),
                ("Purpose", "Stores a custom provider key when the catalog has no match."),
                ("Saved as", "chat_defaults.provider"),
                ("Validation", "letters, numbers, hyphens, underscores, and provider aliases only"),
            )
        if field_id == "settings-model-value":
            return (
                ("Focused setting", "Model"),
                ("Purpose", "Selects the model used when Console has no narrower override."),
                ("Saved as", "chat_defaults.model"),
                ("Validation", "model name is required before provider-backed generation can run"),
            )
        if field_id == "settings-provider-endpoint-value":
            return (
                ("Focused setting", "Endpoint"),
                ("Purpose", "Controls the provider endpoint used by Console generation."),
                ("Saved as", endpoint_key),
                ("Validation", "must start with http:// or https:// when set"),
            )
        if field_id == "settings-provider-api-key":
            return (
                ("Focused setting", "API key"),
                ("Purpose", "Stores a provider API key in local config for Console generation."),
                ("Saved as", f"{provider_config_prefix}.api_key"),
                ("Validation", "single-line secret value; visible UI stays masked"),
            )
        if field_id == "settings-provider-credential-env-var":
            return (
                ("Focused setting", "Credential env"),
                ("Purpose", "Stores the environment variable name containing the API key."),
                ("Saved as", f"{provider_config_prefix}.api_key_env_var"),
                ("Validation", "environment variable names must start with a letter or underscore"),
            )
        if field_id == "settings-model-profile-temperature":
            return (
                ("Focused setting", "Temperature"),
                ("Purpose", "Optional creativity default for this provider and model profile."),
                ("Saved as", f"{provider_config_prefix}.model_defaults.<model>.temperature"),
                ("Validation", "number from 0.0 to 2.0, or blank for inherited default"),
            )
        if field_id == "settings-model-profile-top-p":
            return (
                ("Focused setting", "Top P"),
                ("Purpose", "Optional token-probability cutoff for this provider and model profile."),
                ("Saved as", f"{provider_config_prefix}.model_defaults.<model>.top_p"),
                ("Validation", "number from 0.0 to 1.0, or blank for inherited default"),
            )
        model_profile_guidance = {
            "settings-model-profile-min-p": (
                "Min P",
                "Optional minimum-probability sampling cutoff for local/provider profiles.",
                "min_p",
                "number from 0.0 to 1.0, or blank for inherited default",
            ),
            "settings-model-profile-top-k": (
                "Top K",
                "Optional token candidate count for providers that support top-k sampling.",
                "top_k",
                "whole number of at least 0, or blank for inherited default",
            ),
            "settings-model-profile-max-tokens": (
                "Max tokens",
                "Optional response length ceiling for this provider and model profile.",
                "max_tokens",
                "whole number of at least 1, or blank for inherited default",
            ),
            "settings-model-profile-seed": (
                "Seed",
                "Optional deterministic generation seed for providers that support it.",
                "seed",
                "whole number of at least 0, or blank for inherited default",
            ),
            "settings-model-profile-presence-penalty": (
                "Presence penalty",
                "Optional penalty for introducing tokens already present in the conversation.",
                "presence_penalty",
                "number from -2.0 to 2.0, or blank for inherited default",
            ),
            "settings-model-profile-frequency-penalty": (
                "Frequency penalty",
                "Optional penalty for repeating frequent tokens in the response.",
                "frequency_penalty",
                "number from -2.0 to 2.0, or blank for inherited default",
            ),
            "settings-model-profile-reasoning-effort": (
                "Reasoning effort",
                "Optional OpenAI Responses reasoning level for reasoning-capable models.",
                "reasoning_effort",
                "none, minimal, low, medium, high, xhigh, or blank for inherited default",
            ),
            "settings-model-profile-reasoning-summary": (
                "Reasoning summary",
                "Optional OpenAI reasoning summary detail for supported models.",
                "reasoning_summary",
                "auto, concise, detailed, none, or blank for inherited default",
            ),
            "settings-model-profile-verbosity": (
                "Verbosity",
                "Optional OpenAI text verbosity hint for GPT-5-style Responses models.",
                "verbosity",
                "low, medium, high, or blank for inherited default",
            ),
            "settings-model-profile-thinking-effort": (
                "Thinking effort",
                "Optional Anthropic-style thinking level mapped to provider token budgets.",
                "thinking_effort",
                "off, low, medium, high, xhigh, max, or blank for inherited default",
            ),
            "settings-model-profile-thinking-budget-tokens": (
                "Think budget",
                "Optional explicit thinking token budget for providers that expose it.",
                "thinking_budget_tokens",
                "whole number of at least 1024, or blank for inherited default",
            ),
        }
        if field_id in model_profile_guidance:
            label, purpose, key, validation = model_profile_guidance[field_id]
            draft_key = field_id.removeprefix("settings-").replace("-", "_")
            if not self._model_profile_field_supported(provider, draft_key):
                return (
                    ("Focused setting", label),
                    ("Availability", self._unsupported_model_profile_placeholder(provider)),
                    ("Saved as", "not saved for the selected provider"),
                    ("Validation", "select a provider that supports this control before editing"),
                )
            return (
                ("Focused setting", label),
                ("Purpose", purpose),
                ("Saved as", f"{provider_config_prefix}.model_defaults.<model>.{key}"),
                ("Validation", validation),
            )
        if field_id == "settings-model-profile-streaming":
            return (
                ("Focused setting", "Streaming"),
                ("Purpose", "Optional streaming preference for this provider and model profile."),
                ("Saved as", f"{provider_config_prefix}.model_defaults.<model>.streaming"),
                ("Validation", "true, false, or blank for inherited default"),
            )
        return (
            ("Focused setting", "Provider setup"),
            ("Purpose", "Configure the default provider, model, endpoint, and credential source."),
            ("Saved as", "chat_defaults plus provider-specific api_settings"),
            ("Validation", "test provider readiness before saving Console defaults"),
        )

    def _refresh_provider_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.PROVIDERS_MODELS:
            return
        for index, (label, value) in enumerate(self._provider_field_guidance_rows()):
            self._set_static_text(
                f"#settings-provider-field-guide-{index}",
                f"{label}: {value}",
            )

    def _appearance_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        field_id = self._active_settings_field_id
        if field_id == "settings-appearance-theme":
            return (
                ("Focused setting", "Theme"),
                ("Purpose", "Sets the launch/default app theme."),
                ("Saved as", "general.default_theme"),
                ("Validation", "choose a known theme or keep the loaded custom theme"),
            )
        if field_id == "settings-appearance-palette-theme-limit":
            return (
                ("Focused setting", "Palette limit"),
                ("Purpose", "Limits how many themes appear in the command palette; 0 shows all."),
                ("Saved as", "general.palette_theme_limit"),
                ("Validation", "whole number from 0 to 100"),
            )
        if field_id == "settings-appearance-font-size":
            return (
                ("Focused setting", "Web font size"),
                ("Purpose", "Controls Textual-web terminal cell density."),
                ("Saved as", "web_server.font_size"),
                ("Validation", "whole number from 6 to 32"),
            )
        if field_id == "settings-appearance-density":
            return (
                ("Focused setting", "Density"),
                ("Purpose", "Sets the global compact/normal/comfortable UI density default."),
                ("Saved as", "appearance.density"),
                ("Validation", "compact, normal, or comfortable"),
            )
        if field_id == "settings-appearance-animations-enabled":
            return (
                ("Focused setting", "Animations"),
                ("Purpose", "Controls whether optional UI motion is enabled by default."),
                ("Saved as", "appearance.animations_enabled"),
                ("Validation", "enabled or disabled"),
            )
        if field_id == "settings-appearance-smooth-scrolling":
            return (
                ("Focused setting", "Smooth scrolling"),
                ("Purpose", "Controls smooth scroll behavior where supported."),
                ("Saved as", "appearance.smooth_scrolling"),
                ("Validation", "enabled or disabled"),
            )
        return (
            ("Focused setting", "Appearance defaults"),
            ("Purpose", "Configure global visual defaults without replacing Customize."),
            ("Saved as", "general, web_server, and appearance config sections"),
            ("Validation", "preview safely, then save or revert"),
        )

    def _refresh_appearance_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.APPEARANCE:
            return
        for index, (label, value) in enumerate(self._appearance_field_guidance_rows()):
            self._set_static_text(
                f"#settings-appearance-field-guide-{index}",
                f"{label}: {value}",
            )

    def _storage_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        field_id = self._active_settings_field_id
        field_by_id = {
            (self._storage_field_selector(key) or "").removeprefix("#"): key
            for key in STORAGE_FIELD_LABELS
        }
        key = field_by_id.get(field_id or "")
        if key is None:
            return (
                ("Focused setting", "Storage defaults"),
                ("Purpose", "Configure persisted database path defaults for the next launch."),
                ("Saved as", "database.*"),
                ("Validation", "path text only; no files are moved, created, or reconnected"),
            )
        label = STORAGE_FIELD_LABELS[key]
        saved_key = (
            "database.USER_DB_BASE_DIR"
            if key == "user_db_base_dir"
            else f"database.{key}"
        )
        purpose = (
            "Base directory fallback for local Chatbook data."
            if key == "user_db_base_dir"
            else f"Path to the local {label} file used after restart."
        )
        return (
            ("Focused setting", label),
            ("Purpose", purpose),
            ("Saved as", saved_key),
            ("Validation", "must be a safe path; database paths must end in .db, .sqlite, or .sqlite3"),
        )

    def _refresh_storage_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.STORAGE:
            return
        for index, (label, value) in enumerate(self._storage_field_guidance_rows()):
            self._set_static_text(
                f"#settings-storage-field-guide-{index}",
                f"{label}: {value}",
            )

    def _split_detail_row(self, text: str) -> Static:
        label, separator, value = text.partition(":")
        if not separator:
            return self._detail_row("Path", text)
        return self._detail_row(label.strip(), value.strip())

    def _inspector_guidance(self, category: SettingsCategoryId) -> tuple[tuple[str, str], ...]:
        guidance: dict[SettingsCategoryId, tuple[tuple[str, str], ...]] = {
            SettingsCategoryId.OVERVIEW: (
                ("Affected config", "all Settings categories summarized for readiness"),
                ("Recovery", "open the specific category before changing values"),
                ("Boundary", "runtime MCP, ACP, and tool control stay in their own destinations"),
            ),
            SettingsCategoryId.PROVIDERS_MODELS: (
                (
                    "Affected config",
                    "provider, model, endpoint, and credential source defaults",
                ),
                (
                    "Recovery",
                    "test provider readiness before saving provider-backed Console defaults",
                ),
                (
                    "Boundary",
                    "Sampling and transport defaults are routed to Console Defaults",
                ),
            ),
            SettingsCategoryId.APPEARANCE: (
                ("Affected config", "theme, density, font size, and motion defaults"),
                (
                    "Recovery",
                    "open Customize for full theme editing; use Settings for persisted defaults",
                ),
                ("Boundary", "visual preferences do not change runtime or data access"),
            ),
            SettingsCategoryId.STORAGE: (
                ("Affected config", "config file path, local database paths, media storage roots"),
                ("Recovery", "verify paths, reload config, then restart only if storage roots changed"),
                ("Boundary", "server handoff does not move local source content unless explicitly requested"),
            ),
            SettingsCategoryId.PRIVACY_SECURITY: (
                ("Affected config", "encryption posture, credential-source status, and redaction status"),
                (
                    "Credential source",
                    "Environment variables are preferred for provider credentials.",
                ),
                (
                    "Recovery",
                    "open Providers & Models for provider defaults or Advanced Config for expert repair",
                ),
                (
                    "Boundary",
                    "raw secret values are never displayed; encryption mutation needs a password-gated flow",
                ),
            ),
            SettingsCategoryId.CONSOLE_BEHAVIOR: (
                ("Affected config", "chat_defaults fallbacks plus Console composer paste behavior"),
                ("Recovery", "revert unsaved changes or disable paste collapse if composer flow is disrupted"),
                ("Boundary", "active sessions and provider+model profiles override these global fallbacks"),
            ),
            SettingsCategoryId.LIBRARY_RAG: (
                (
                    "Affected config",
                    "AppRAGSearchConfig.rag.search and AppRAGSearchConfig.rag.retriever defaults",
                ),
                (
                    "Recovery",
                    "revert unsaved defaults or open Library to validate retrieval behavior",
                ),
                (
                    "Boundary",
                    "Library owns indexing, query execution, source browse, Collections, and staging",
                ),
            ),
            SettingsCategoryId.DIAGNOSTICS: (
                ("Affected config", "read-only validation, reload status, and troubleshooting output"),
                ("Recovery", "validate first, reload only after confirming the config source is correct"),
                ("Boundary", "diagnostics redact secrets and should not mutate advanced config"),
            ),
            SettingsCategoryId.ADVANCED_CONFIG: (
                ("Affected config", "raw TOML for every loaded configuration section"),
                ("Recovery", "validate current text, save atomically, then restore from backup if needed"),
                ("Boundary", "save is blocked until the exact current text validates"),
            ),
        }
        if category in DOMAIN_SETTINGS_CATEGORY_IDS:
            contract = self._domain_category_contract(category)
            return (
                (
                    "Affected config",
                    "none yet - this category is an ownership/status contract",
                ),
                (
                    "Recovery",
                    f"open {contract.owner_destination} for workflow actions and setup",
                ),
                (
                    "Boundary",
                    f"{contract.owner_destination} remains the runtime owner; Settings cannot mutate it yet",
                ),
            )
        return guidance[category]

    def _render_category_buttons(self) -> ComposeResult:
        summaries_by_id = {summary.category: summary for summary in self._category_summaries()}
        visible_count = 0
        for group_title, category_ids in self._category_groups():
            visible_categories = tuple(
                category_id
                for category_id in category_ids
                if self._category_matches_search(summaries_by_id[category_id])
            )
            group_heading = Static(
                group_title,
                id=self._category_group_dom_id(group_title),
                classes="settings-category-group-title",
            )
            group_heading.display = bool(visible_categories)
            yield group_heading
            for category_id in category_ids:
                summary = summaries_by_id[category_id]
                is_visible = category_id in visible_categories
                visible_count += int(is_visible)
                is_active = summary.category.value == self.active_category
                button = Button(
                    self._category_button_label(summary, is_active=is_active),
                    id=f"settings-category-{summary.category.value}",
                    classes="settings-category-button",
                    tooltip=summary.description,
                )
                if is_active:
                    button.add_class("settings-active-section")
                if self._category_search_text() and is_visible:
                    rank = self._category_search_rank(summary)
                    if rank == 0:
                        button.add_class("settings-primary-search-match")
                    elif rank == 1:
                        button.add_class("settings-secondary-search-match")
                button.display = is_visible
                yield button
                if summary.status:
                    status = Static(
                        f"Status: {self._category_status(summary)}",
                        id=f"settings-category-{summary.category.value}-status",
                        classes="destination-section settings-status-row settings-category-status-hidden",
                    )
                    if self._category_has_unsaved_changes(summary.category):
                        status.add_class("settings-dirty-category")
                    yield status
        empty_state = Static(
            f"No Settings categories match: {self._category_search_text()}",
            id="settings-category-search-empty",
            classes="settings-search-empty",
            markup=False,
        )
        empty_state.display = bool(self._category_search_text() and visible_count == 0)
        yield empty_state

    def _render_overview_detail(self) -> ComposeResult:
        yield Static("Overview", classes="destination-section settings-column-title")
        with Vertical(id="settings-overview-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.OVERVIEW)
            yield Static("Manual Sync v2", classes="destination-section")
            yield Static(
                "Preview pending Notes/Chat changes before any server mutation.",
                classes="settings-help-copy",
            )
            for label, value in self.manual_sync_rows:
                yield self._detail_row(label, value)
            with Horizontal(classes="settings-action-row"):
                yield Button(
                    "Preview manual sync",
                    id="settings-manual-sync-preview",
                    tooltip="Show pending Notes/Chat changes without mutating the server.",
                )
                yield Button(
                    "Run manual sync",
                    id="settings-manual-sync-run",
                    tooltip="Apply the previewed Notes/Chat changes to the server.",
                )
            yield Static("Configuration ownership", classes="destination-section")
            for label, value in self._overview_ownership_rows():
                yield self._detail_row(label, value)
            yield Static("Server, sync, workspace, and handoff", classes="destination-section")
            for label, value in self.server_sync_workspace_handoff_rows:
                yield self._detail_row(label, value)
            with Horizontal(classes="settings-action-row"):
                yield Button(
                    "Switch Source / Server",
                    id="settings-switch-runtime-source",
                    tooltip="Choose local-only or bind a tldw server as the runtime source.",
                )
            yield Static("Provider readiness", classes="destination-section")
            yield self._detail_row(
                "Provider readiness",
                self._provider_readiness_label().removeprefix("Provider readiness: "),
                identifier="settings-overview-provider-readiness",
            )
            yield Static("Storage", classes="destination-section")
            yield self._detail_row(
                "Config path",
                self._config_path_overview_value(),
                identifier="settings-overview-storage",
            )
            yield Static("Privacy", classes="destination-section")
            yield self._detail_row(
                "Privacy",
                "local config by default; secret-looking diagnostics are redacted",
                identifier="settings-overview-privacy",
            )
            yield self._detail_row(
                "Console paste collapse",
                self._collapse_large_pastes_label(),
                identifier="settings-overview-console-paste-collapse",
            )
            yield self._detail_row(
                "Diagnostics",
                "validate config before saving raw TOML changes",
            )

    def _render_provider_detail(self) -> ComposeResult:
        resolved = self._resolve_provider_model_for_settings()
        values = self._provider_display_setting_values()
        provider = str(values["provider"])
        yield Static("Providers & Models", classes="destination-section settings-column-title")
        with Vertical(id="settings-providers-models-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.PROVIDERS_MODELS)
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Provider", classes="settings-input-label")
                yield Select(
                    self._provider_select_options(),
                    value=self._provider_select_value_for_provider(provider),
                    id="settings-provider-value",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            manual_provider_classes = "settings-input-row"
            if self._provider_select_value_for_provider(provider) != PROVIDER_MANUAL_SELECT_VALUE:
                manual_provider_classes += " settings-provider-manual-hidden"
            with Horizontal(id="settings-provider-manual-row", classes=manual_provider_classes):
                yield Static("Manual", classes="settings-input-label")
                yield Input(
                    value=str(values["provider"])
                    if self._provider_select_value_for_provider(provider) == PROVIDER_MANUAL_SELECT_VALUE
                    else "",
                    id="settings-provider-manual-value",
                    classes="settings-compact-input",
                    placeholder="Custom provider key",
                    disabled=(
                        self._provider_select_value_for_provider(provider) != PROVIDER_MANUAL_SELECT_VALUE
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Model", classes="settings-input-label")
                yield Input(
                    value=str(values["model"]),
                    id="settings-model-value",
                    classes="settings-compact-input",
                    placeholder="Model name",
                )
            yield Static(
                "Selected model defaults",
                id="settings-selected-model-defaults-title",
                classes="destination-section",
            )
            yield Static(
                "Global fallbacks live under Console Defaults; these values apply only "
                "to the provider+model above.",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Temperature", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_temperature"]),
                    id="settings-model-profile-temperature",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top P", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_top_p"]),
                    id="settings-model-profile-top-p",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Min P", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_min_p"]),
                    id="settings-model-profile-min-p",
                    classes="settings-compact-input",
                    placeholder="optional 0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top K", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_top_k"]),
                    id="settings-model-profile-top-k",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Max tokens", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_max_tokens"]),
                    id="settings-model-profile-max-tokens",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Seed", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_seed"]),
                    id="settings-model-profile-seed",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Presence", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_presence_penalty"]),
                    id="settings-model-profile-presence-penalty",
                    classes="settings-compact-input",
                    placeholder="-2.0 - 2.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Frequency", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_frequency_penalty"]),
                    id="settings-model-profile-frequency-penalty",
                    classes="settings-compact-input",
                    placeholder="-2.0 - 2.0",
                )
            yield Static(
                self._provider_generation_support_copy(provider),
                id="settings-provider-generation-support",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Reasoning", classes="settings-input-label")
                yield Input(
                    value=self._model_profile_input_value(
                        provider,
                        "model_profile_reasoning_effort",
                        values["model_profile_reasoning_effort"],
                    ),
                    id="settings-model-profile-reasoning-effort",
                    classes="settings-compact-input",
                    placeholder=self._model_profile_input_placeholder(
                        provider,
                        "model_profile_reasoning_effort",
                    ),
                    disabled=not self._model_profile_field_supported(
                        provider,
                        "model_profile_reasoning_effort",
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Summary", classes="settings-input-label")
                yield Input(
                    value=self._model_profile_input_value(
                        provider,
                        "model_profile_reasoning_summary",
                        values["model_profile_reasoning_summary"],
                    ),
                    id="settings-model-profile-reasoning-summary",
                    classes="settings-compact-input",
                    placeholder=self._model_profile_input_placeholder(
                        provider,
                        "model_profile_reasoning_summary",
                    ),
                    disabled=not self._model_profile_field_supported(
                        provider,
                        "model_profile_reasoning_summary",
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Verbosity", classes="settings-input-label")
                yield Input(
                    value=self._model_profile_input_value(
                        provider,
                        "model_profile_verbosity",
                        values["model_profile_verbosity"],
                    ),
                    id="settings-model-profile-verbosity",
                    classes="settings-compact-input",
                    placeholder=self._model_profile_input_placeholder(
                        provider,
                        "model_profile_verbosity",
                    ),
                    disabled=not self._model_profile_field_supported(
                        provider,
                        "model_profile_verbosity",
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Thinking", classes="settings-input-label")
                yield Input(
                    value=self._model_profile_input_value(
                        provider,
                        "model_profile_thinking_effort",
                        values["model_profile_thinking_effort"],
                    ),
                    id="settings-model-profile-thinking-effort",
                    classes="settings-compact-input",
                    placeholder=self._model_profile_input_placeholder(
                        provider,
                        "model_profile_thinking_effort",
                    ),
                    disabled=not self._model_profile_field_supported(
                        provider,
                        "model_profile_thinking_effort",
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Think budget", classes="settings-input-label")
                yield Input(
                    value=self._model_profile_input_value(
                        provider,
                        "model_profile_thinking_budget_tokens",
                        values["model_profile_thinking_budget_tokens"],
                    ),
                    id="settings-model-profile-thinking-budget-tokens",
                    classes="settings-compact-input",
                    placeholder=self._model_profile_input_placeholder(
                        provider,
                        "model_profile_thinking_budget_tokens",
                    ),
                    restrict=r"^[0-9]*$",
                    disabled=not self._model_profile_field_supported(
                        provider,
                        "model_profile_thinking_budget_tokens",
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Streaming", classes="settings-input-label")
                yield Input(
                    value=self._profile_input_value(values["model_profile_streaming"]),
                    id="settings-model-profile-streaming",
                    classes="settings-compact-input",
                    placeholder="true or false",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Endpoint", classes="settings-input-label")
                yield SettingsURLInput(
                    value=str(values["endpoint"]),
                    id="settings-provider-endpoint-value",
                    classes="settings-compact-input",
                    placeholder=self._provider_endpoint_placeholder(provider),
                )
            yield Static("Credentials", classes="destination-section")
            yield Static(
                self._provider_credential_status(provider),
                id="settings-provider-credential-status",
                classes="settings-status-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("API key", classes="settings-input-label")
                yield Input(
                    value=str(values.get("api_key") or ""),
                    id="settings-provider-api-key",
                    classes="settings-compact-input",
                    placeholder=self._provider_api_key_placeholder(provider),
                    password=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("", classes="settings-input-label")
                yield Button(
                    "Clear saved key",
                    id="settings-provider-api-key-clear",
                    disabled=(
                        not self._provider_saved_api_key_present(provider)
                        and not bool(str(values.get("api_key") or "").strip())
                    ),
                    tooltip="Clear the API key saved in local config for this provider.",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Env var", classes="settings-input-label")
                yield Input(
                    value=str(values["credential_env_var"]),
                    id="settings-provider-credential-env-var",
                    classes="settings-compact-input",
                    placeholder=self._provider_credential_placeholder(provider),
                )
            yield Static(
                "Env vars are safer for shells, shared machines, and CI. This field stores the variable name, not the secret.",
                id="settings-provider-credential-guidance",
                classes="settings-status-row",
            )
            yield Static(
                self._provider_catalog_summary(),
                id="settings-provider-catalog",
                classes="settings-status-row",
            )
            yield Static(
                self._provider_catalog_key_policy(),
                id="settings-provider-catalog-policy",
                classes="settings-status-row",
            )
            yield Static(
                "Choose a catalog provider, or use Manual / custom provider for aliases.",
                id="settings-provider-manual-entry-policy",
                classes="settings-status-row",
            )
            yield Static(
                "Sampling and transport defaults are routed to Console Defaults.",
                id="settings-provider-sampling-route",
                classes="settings-status-row",
            )
            yield self._detail_row(
                "Endpoint key",
                self._provider_endpoint_row(str(values["provider"])).removeprefix("Endpoint key: "),
                identifier="settings-provider-endpoint-key",
            )
            yield Static("Provider readiness", classes="destination-section")
            yield self._detail_row(
                "Readiness",
                self._provider_readiness_label().removeprefix("Provider readiness: "),
                identifier="settings-provider-readiness",
            )
            yield self._detail_row(
                "Provider source",
                self._settings_source_label(resolved.provider_source),
                identifier="settings-provider-source",
            )
            yield self._detail_row(
                "Model source",
                self._settings_source_label(resolved.model_source),
                identifier="settings-model-source",
            )
            yield self._detail_row(
                "Endpoint",
                self._provider_endpoint_display_value(str(values["provider"]), values["endpoint"]),
                identifier="settings-provider-endpoint",
            )
            yield Static(
                self._provider_key_status(str(values["provider"])),
                id="settings-provider-key-status",
            )
            yield Static("Model discovery", classes="destination-section")
            yield Static(
                self._model_discovery_status,
                id="settings-model-discovery-status",
                classes="settings-status-row",
            )
            empty_state = Static(
                MODEL_DISCOVERY_EMPTY_COPY,
                id="settings-model-discovery-empty",
                classes="settings-status-row",
            )
            empty_state.display = not self._model_discovery_models
            yield empty_state
            yield Static(
                MODEL_DISCOVERY_CAPABILITY_WARNING,
                id="settings-model-discovery-capability-warning",
                classes="settings-status-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Button(
                    "Discover models",
                    id="settings-discover-provider-models",
                    disabled=not self._model_discovery_available(str(values["provider"])),
                    tooltip=(
                        "Query the configured OpenAI-compatible provider endpoint "
                        "for available models."
                    ),
                )
                yield Button(
                    "Save selected",
                    id="settings-save-discovered-provider-models",
                    disabled=not self._model_discovery_models,
                    tooltip="Append selected discovered model IDs to the local provider list.",
                )
                yield Button(
                    "Clear",
                    id="settings-clear-discovered-provider-models",
                    disabled=not self._model_discovery_models,
                    tooltip="Clear runtime-discovered models for this provider.",
                )
            yield SelectionList(
                *self._model_discovery_selection_options(),
                id="settings-discovered-models-list",
                classes="settings-discovered-models-list",
                disabled=not self._model_discovery_models,
            )
            yield Button(
                "Test Provider",
                id="settings-test-provider",
                tooltip="Run a local, non-network readiness check for this provider configuration.",
            )
            yield Static(self._provider_test_result, id="settings-provider-test-result")
            yield Static(
                self._provider_save_result,
                id="settings-provider-save-result",
                classes="settings-status-row",
            )

    def _render_console_behavior_card(self, *, compact: bool = False) -> ComposeResult:
        with Vertical(id="settings-console-behavior-card", classes="settings-secondary-card"):
            title = "Console paste collapse" if compact else "Console Behavior"
            yield Static(title, classes="destination-section")
            yield Static("Composer paste handling", classes="destination-section")
            yield Static(
                "Collapse large pasted chunks only when they exceed the threshold.",
                id="settings-console-collapse-large-pastes-label",
            )
            yield Button(
                self._collapse_large_pastes_button_label(),
                id="settings-console-collapse-large-pastes-toggle",
                tooltip="Toggle compact display for large pasted Console chunks.",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Threshold", classes="settings-input-label")
                yield Input(
                    value=str(self._paste_collapse_threshold_value()),
                    id="settings-console-paste-collapse-threshold",
                    classes="settings-compact-input",
                    placeholder=str(DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD),
                    restrict=r"^[0-9]*$",
                )
            yield Static(
                "Normal typing stays literal. The canonical message payload is preserved.",
                id="settings-console-collapse-large-pastes-help",
            )
            yield Static("Global fallback defaults", classes="destination-section")
            yield Static(
                "Used when no provider+model profile or active Console session overrides them.",
                id="settings-console-defaults-help",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Streaming", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("streaming")),
                    id="settings-console-default-streaming",
                    classes="settings-compact-input",
                    placeholder="true or false",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Temperature", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("temperature")),
                    id="settings-console-default-temperature",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top P", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("top_p")),
                    id="settings-console-default-top-p",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Min P", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("min_p")),
                    id="settings-console-default-min-p",
                    classes="settings-compact-input",
                    placeholder="optional 0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top K", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("top_k")),
                    id="settings-console-default-top-k",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Max tokens", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("max_tokens")),
                    id="settings-console-default-max-tokens",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Seed", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("seed")),
                    id="settings-console-default-seed",
                    classes="settings-compact-input",
                    placeholder="optional deterministic seed",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Presence", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("presence_penalty")
                    ),
                    id="settings-console-default-presence-penalty",
                    classes="settings-compact-input",
                    placeholder="-2.0 - 2.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Frequency", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("frequency_penalty")
                    ),
                    id="settings-console-default-frequency-penalty",
                    classes="settings-compact-input",
                    placeholder="-2.0 - 2.0",
                )
            yield Static(
                "Reasoning and thinking controls are sent only to providers that support them.",
                id="settings-console-reasoning-help",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Reasoning", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("reasoning_effort")
                    ),
                    id="settings-console-default-reasoning-effort",
                    classes="settings-compact-input",
                    placeholder="none, minimal, low, medium, high, xhigh",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Summary", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("reasoning_summary")
                    ),
                    id="settings-console-default-reasoning-summary",
                    classes="settings-compact-input",
                    placeholder="auto, concise, detailed, none",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Verbosity", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(self._console_behavior_value("verbosity")),
                    id="settings-console-default-verbosity",
                    classes="settings-compact-input",
                    placeholder="low, medium, high",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Thinking", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("thinking_effort")
                    ),
                    id="settings-console-default-thinking-effort",
                    classes="settings-compact-input",
                    placeholder="off, low, medium, high, xhigh, max",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Think budget", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("thinking_budget_tokens")
                    ),
                    id="settings-console-default-thinking-budget-tokens",
                    classes="settings-compact-input",
                    placeholder="optional tokens, min 1024",
                    restrict=r"^[0-9]*$",
                )
            yield Static(
                "chat_defaults.streaming is canonical; enable_streaming is read as fallback only.",
                id="settings-console-streaming-compatibility",
                classes="settings-status-row",
            )
            yield Static("Background effects", classes="destination-section")
            yield Button(
                self._console_background_effect_enabled_label(),
                id="settings-console-background-effect-enabled",
                tooltip="Toggle optional ambient effects behind the Console transcript.",
            )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Background effect", classes="settings-input-label")
                yield Select(
                    [
                        (label, value)
                        for label, value in (
                            ("None", "none"),
                            ("Snow", "snow"),
                            ("Rain", "rain"),
                            ("Matrix", "matrix"),
                        )
                        if value in CONSOLE_BACKGROUND_EFFECTS
                    ],
                    value=str(self._console_background_effect_value("effect") or "none"),
                    id="settings-console-background-effect-type",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Scope", classes="settings-input-label")
                yield Select(
                    [
                        (label, value)
                        for label, value in (
                            ("Transcript (recommended)", "transcript"),
                            ("Workbench (advanced)", "workbench"),
                        )
                        if value in CONSOLE_BACKGROUND_SCOPES
                    ],
                    value=str(self._console_background_effect_value("scope") or "transcript"),
                    id="settings-console-background-effect-scope",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Intensity", classes="settings-input-label")
                yield Select(
                    [
                        (label, value)
                        for label, value in (
                            ("Low", "low"),
                            ("Medium", "medium"),
                            ("High", "high"),
                        )
                        if value in CONSOLE_BACKGROUND_INTENSITIES
                    ],
                    value=str(self._console_background_effect_value("intensity") or "low"),
                    id="settings-console-background-effect-intensity",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Frame rate", classes="settings-input-label")
                yield Input(
                    value=str(
                        self._console_background_effect_value("fps")
                        or DEFAULT_CONSOLE_BACKGROUND_FPS
                    ),
                    id="settings-console-background-effect-fps",
                    classes="settings-compact-input",
                    placeholder=f"{DEFAULT_CONSOLE_BACKGROUND_FPS}",
                    restrict=r"^[0-9]*$",
                    tooltip=(
                        f"Frame rate from {MIN_CONSOLE_BACKGROUND_FPS} to "
                        f"{MAX_CONSOLE_BACKGROUND_FPS} FPS."
                    ),
                )
            yield Static(
                self._console_behavior_result_text(),
                id="settings-console-behavior-result",
                classes="settings-status-row",
            )

    def _library_rag_include_citations_label(self) -> str:
        return "Enabled" if bool(self._library_rag_setting_values()["include_citations"]) else "Disabled"

    def _render_library_rag_detail(self) -> ComposeResult:
        values = self._library_rag_setting_values()
        search_mode = normalise_library_rag_search_mode(values["default_search_mode"])
        citation_style = normalise_library_rag_citation_style(values["citation_style"])

        yield Static("Library & RAG", classes="destination-section settings-column-title")
        with Vertical(id="settings-library-rag-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.LIBRARY_RAG)
            yield Static("Search defaults", classes="destination-section")
            yield Static(
                "Used by future Library-native Search/RAG and Console evidence handoff defaults.",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Search mode", classes="settings-input-label")
                yield Select(
                    [
                        ("Plain keyword", "plain"),
                        ("Semantic", "semantic"),
                        ("Hybrid", "hybrid"),
                    ],
                    value=search_mode,
                    id="settings-library-rag-search-mode",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Default results", classes="settings-input-label")
                yield Input(
                    value=str(values["default_top_k"]),
                    id="settings-library-rag-default-top-k",
                    classes="settings-compact-input",
                    placeholder="1 - 100",
                    restrict=r"^[0-9]*$",
                )
            yield Static("Retriever balance", classes="destination-section")
            with Horizontal(classes="settings-input-row"):
                yield Static("Keyword results", classes="settings-input-label")
                yield Input(
                    value=str(values["fts_top_k"]),
                    id="settings-library-rag-fts-top-k",
                    classes="settings-compact-input",
                    placeholder="1 - 100",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Vector results", classes="settings-input-label")
                yield Input(
                    value=str(values["vector_top_k"]),
                    id="settings-library-rag-vector-top-k",
                    classes="settings-compact-input",
                    placeholder="1 - 100",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Hybrid alpha", classes="settings-input-label")
                yield Input(
                    value=str(values["hybrid_alpha"]),
                    id="settings-library-rag-hybrid-alpha",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Min score", classes="settings-input-label")
                yield Input(
                    value=str(values["score_threshold"]),
                    id="settings-library-rag-score-threshold",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                )
            yield Static("Citation and snippets", classes="destination-section")
            yield Button(
                self._library_rag_include_citations_label(),
                id="settings-library-rag-include-citations",
                tooltip="Toggle citation metadata in future RAG answers where supported.",
            )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Citation style", classes="settings-input-label")
                yield Select(
                    [
                        ("Inline", "inline"),
                        ("Footnote", "footnote"),
                        ("None", "none"),
                    ],
                    value=citation_style,
                    id="settings-library-rag-citation-style",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Snippet chars", classes="settings-input-label")
                yield Input(
                    value=str(values["snippet_max_chars"]),
                    id="settings-library-rag-snippet-max-chars",
                    classes="settings-compact-input",
                    placeholder="50 - 10000",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Context budget", classes="settings-input-label")
                yield Input(
                    value=str(values["max_context_size"]),
                    id="settings-library-rag-max-context-size",
                    classes="settings-compact-input",
                    placeholder="1000 - 1000000",
                    restrict=r"^[0-9]*$",
                )
            yield Static("Preview defaults", classes="destination-section")
            preview_summary, preview_retrieval, preview_context = self._library_rag_preview_rows()
            yield Static(
                preview_summary,
                id="settings-library-rag-preview-summary",
                classes="settings-detail-row",
            )
            yield Static(
                preview_retrieval,
                id="settings-library-rag-preview-retrieval",
                classes="settings-detail-row",
            )
            yield Static(
                preview_context,
                id="settings-library-rag-preview-context",
                classes="settings-detail-row",
            )
            yield Static("Save targets", classes="destination-section")
            yield self._detail_row("Search", "AppRAGSearchConfig.rag.search")
            yield self._detail_row("Retriever", "AppRAGSearchConfig.rag.retriever")
            yield Static(
                self._library_rag_result,
                id="settings-library-rag-save-result",
                classes="settings-status-row",
            )

    def _render_domain_category_detail(self, category: SettingsCategoryId) -> ComposeResult:
        contract = self._domain_category_contract(category)
        yield Static(contract.title, classes="destination-section settings-column-title")
        with Vertical(id=f"settings-{category.value}-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(category)
            yield Static("Domain ownership contract", classes="destination-section")
            yield self._detail_row("Owner destination", contract.owner_destination)
            yield self._detail_row("Settings mode", "read-only defaults/status contract")
            yield self._detail_row(
                "Writes allowed",
                "No - destination ownership must be implemented before mutation",
            )
            yield Static("Source of truth", classes="destination-section")
            for index, source in enumerate(contract.source_of_truth, start=1):
                yield self._detail_row(f"Source {index}", source)
            yield Static("Status and default boundaries", classes="destination-section")
            for label, value in contract.rows:
                yield self._detail_row(label, value)
            yield self._detail_row("Follow-up", contract.follow_up)

    def _render_detail_pane(self) -> ComposeResult:
        category = SettingsCategoryId(self.active_category)
        if category is SettingsCategoryId.OVERVIEW:
            yield from self._render_overview_detail()
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            yield from self._render_provider_detail()
        elif category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            yield Static("Console Behavior", classes="destination-section settings-column-title")
            with Vertical(id="settings-console-behavior-detail", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.CONSOLE_BEHAVIOR)
                yield from self._render_console_behavior_card(compact=False)
                yield Static("Composer behavior", classes="destination-section")
                yield self._detail_row(
                    "Paste collapse",
                    "pasted chunks over the threshold display as compact placeholders",
                )
                yield self._detail_row("Threshold", self._paste_collapse_threshold_label())
                yield self._detail_row(
                    "Typing rule",
                    "normal typing remains literal and never auto-collapses",
                )
                yield self._detail_row("Current default", self._collapse_large_pastes_label())
                yield Static("Global fallback defaults", classes="destination-section")
                yield self._detail_row(
                    "Fallback source",
                    "[chat_defaults].streaming, temperature, top_p, max_tokens",
                )
                yield self._detail_row(
                    "Compatibility",
                    "streaming is canonical; enable_streaming is read only when streaming is absent",
                )
                yield self._detail_row(
                    "Override order",
                    "active Console session, then provider+model profile, then global fallback",
                )
                yield self._detail_row(
                    "Save targets",
                    "[console] paste settings and [chat_defaults] global fallbacks",
                )
                yield self._detail_row(
                    "Console impact",
                    "new/default sessions use these only when no narrower override applies",
                )
        elif category is SettingsCategoryId.LIBRARY_RAG:
            yield from self._render_library_rag_detail()
        elif category is SettingsCategoryId.APPEARANCE:
            values = self._appearance_setting_values()
            yield Static("Appearance", classes="destination-section settings-column-title")
            with Vertical(id="settings-appearance-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.APPEARANCE)
                yield Static("Global visual defaults", classes="destination-section")
                yield Static(
                    "Settings owns launch and web display defaults. "
                    "Customize owns full theme editing and deeper visual preview.",
                    classes="settings-detail-row",
                )
                with Horizontal(classes="settings-input-row settings-select-row"):
                    yield Static("Theme", classes="settings-input-label")
                    yield Select(
                        self._appearance_theme_options(),
                        value=str(values["default_theme"]),
                        id="settings-appearance-theme",
                        classes="settings-compact-select",
                        allow_blank=False,
                        compact=True,
                    )
                with Horizontal(classes="settings-input-row"):
                    yield Static("Palette limit", classes="settings-input-label")
                    yield Input(
                        value=str(values["palette_theme_limit"]),
                        id="settings-appearance-palette-theme-limit",
                        classes="settings-compact-input",
                        placeholder="0 - 100",
                        restrict=r"^[0-9]*$",
                    )
                with Horizontal(classes="settings-input-row"):
                    yield Static("Web font size", classes="settings-input-label")
                    yield Input(
                        value=str(values["font_size"]),
                        id="settings-appearance-font-size",
                        classes="settings-compact-input",
                        placeholder="6 - 32",
                        restrict=r"^[0-9]*$",
                    )
                with Horizontal(classes="settings-input-row settings-select-row"):
                    yield Static("Density", classes="settings-input-label")
                    yield Select(
                        [
                            ("Compact", "compact"),
                            ("Normal", "normal"),
                            ("Comfortable", "comfortable"),
                        ],
                        value=str(values["density"]),
                        id="settings-appearance-density",
                        classes="settings-compact-select",
                        allow_blank=False,
                        compact=True,
                    )
                yield Static("Motion and scrolling", classes="destination-section")
                with Horizontal(classes="settings-input-row"):
                    yield Static("Animations", classes="settings-input-label")
                    yield Button(
                        self._appearance_bool_label("animations_enabled"),
                        id="settings-appearance-animations-enabled",
                        tooltip="Toggle optional UI animation defaults.",
                    )
                with Horizontal(classes="settings-input-row"):
                    yield Static("Smooth scroll", classes="settings-input-label")
                    yield Button(
                        self._appearance_bool_label("smooth_scrolling"),
                        id="settings-appearance-smooth-scrolling",
                        tooltip="Toggle smooth scrolling defaults where supported.",
                    )
                yield Static("Preview and boundary", classes="destination-section")
                yield self._detail_row("Current summary", self._appearance_summary_text())
                yield self._detail_row("Runtime preview", "applies safe values for this session only")
                yield self._detail_row(
                    "Open Customize",
                    "full theme editor, custom colors, and deeper visual preview",
                )
                yield self._detail_row("Save targets", "general, web_server, and appearance")
                with Horizontal(id="settings-appearance-actions", classes="settings-action-row"):
                    yield Button(
                        "Preview",
                        id="settings-preview-appearance",
                        tooltip="Apply runtime-safe Appearance values for this session only.",
                    )
                yield Static(
                    self._appearance_result,
                    id="settings-appearance-save-result",
                    classes="settings-status-row",
                )
        elif category is SettingsCategoryId.STORAGE:
            values = self._storage_setting_values()
            try:
                config_path: object = self._config_path()
            except (OSError, RuntimeError, ValueError) as exc:
                config_path = f"invalid - {redact_secret_text(str(exc))}"
            yield Static("Storage", classes="destination-section settings-column-title")
            with Vertical(id="settings-storage-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.STORAGE)
                yield Static("Storage defaults", classes="destination-section")
                yield self._detail_row("Scope", "persisted local database path defaults")
                yield self._detail_row(
                    "Activation",
                    "Changes apply on next launch; active database handles keep current paths",
                )
                yield self._detail_row(
                    "Safety",
                    "Save writes config only; no files are moved or reconnected",
                )
                yield self._detail_row("Config path", config_path)
                yield Static("Draft path check", classes="destination-section")
                yield self._detail_row(
                    "Check mode",
                    "non-mutating; reports parent readiness for the current config runtime",
                )
                with Horizontal(id="settings-storage-actions", classes="settings-action-row"):
                    yield Button(
                        "Check Storage",
                        id="settings-check-storage",
                        tooltip="Verify local storage path access without moving or writing data.",
                    )
                yield Static(
                    self._storage_check_text(),
                    id="settings-storage-check-result",
                    classes="settings-status-row settings-storage-check-result",
                )
                yield Static(
                    self._storage_result,
                    id="settings-storage-save-result",
                    classes="settings-status-row",
                )
                yield Static("Database paths", classes="destination-section")
                for key, label in STORAGE_FIELD_LABELS.items():
                    selector = self._storage_field_selector(key)
                    if selector is None:
                        continue
                    with Horizontal(classes="settings-input-row"):
                        yield Static(label, classes="settings-input-label")
                        yield Input(
                            value=str(values[key]),
                            id=selector.removeprefix("#"),
                            classes="settings-compact-input",
                            placeholder="~/path/to/database.db",
                        )
                yield Static("Runtime local paths", classes="destination-section")
                for path_summary in self._known_storage_paths():
                    yield self._split_detail_row(path_summary)
                yield self._detail_row("Config directory status", self._config_writable_status())
                yield self._detail_row(
                    "Handoff boundary",
                    "database and media paths remain local unless a server handoff is explicit",
                )
        elif category is SettingsCategoryId.PRIVACY_SECURITY:
            posture = self._settings_privacy_posture()
            yield Static("Privacy & Security", classes="destination-section settings-column-title")
            with Vertical(id="settings-privacy-security-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.PRIVACY_SECURITY)
                yield Static("Privacy posture", classes="destination-section")
                yield self._detail_row(
                    "Config encryption",
                    "enabled" if posture.encryption_enabled else "disabled",
                )
                yield self._detail_row(
                    "Redaction",
                    "active; raw secret values hidden",
                )
                yield self._detail_row(
                    "Sensitive config fields",
                    f"{posture.sensitive_config_fields} present",
                )
                yield self._detail_row(
                    "Provider config secrets",
                    f"{posture.provider_config_secrets} present",
                )
                yield Static("Credential sources", classes="destination-section")
                yield self._detail_row(
                    "Provider env vars",
                    (
                        f"{posture.provider_env_present} present / "
                        f"{posture.provider_env_missing} missing / "
                        f"{posture.provider_env_configured} configured"
                    ),
                )
                yield self._detail_row("Preferred source", "environment variables")
                yield self._detail_row("Config secrets", "counted only; raw values are never displayed")
                yield self._detail_row(
                    "Recovery actions",
                    "Check Privacy | Open Providers & Models | Open Advanced Config",
                )
                yield self._detail_row(
                    "Skill trust",
                    posture.skill_trust_status if posture.skill_trust_enabled else "disabled",
                )
                yield self._detail_row(
                    "Skill trust keyring convenience",
                    "enabled"
                    if posture.skill_trust_keyring_convenience_enabled
                    else "disabled",
                )
                yield self._detail_row(
                    "Skill trust rollback protection",
                    "reduced"
                    if posture.skill_trust_reduced_rollback_protection
                    else "full",
                )
                with Horizontal(id="settings-privacy-actions", classes="settings-action-row"):
                    yield Button(
                        "Check Privacy",
                        id="settings-check-privacy",
                        tooltip="Verify secret and redaction status without exposing values.",
                    )
                    yield Button(
                        "Open Providers & Models",
                        id="settings-open-provider-credentials",
                        tooltip="Review provider, endpoint, and credential-source defaults.",
                    )
                    yield Button(
                        "Open Advanced Config",
                        id="settings-open-advanced-config",
                        tooltip="Open guarded raw TOML recovery for expert repair.",
                    )
                yield Static("Data boundary", classes="destination-section")
                yield self._detail_row("Local data", posture.data_boundary)
                yield self._detail_row("Server tokens", posture.server_boundary)
                yield self._detail_row(
                    "Credential mutation",
                    "unavailable/WIP - password-gated flow required",
                )
                yield Static(
                    self._privacy_check_text(),
                    id="settings-privacy-check-result",
                    classes="settings-status-row",
                )
        elif category is SettingsCategoryId.DIAGNOSTICS:
            yield Static("Diagnostics", classes="destination-section settings-column-title")
            with Vertical(id="settings-diagnostics-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.DIAGNOSTICS)
                yield Static("Validate config", classes="destination-section")
                yield self._detail_row("Config path", self._config_path())
                yield self._detail_row("Validation", "raw TOML validation before advanced edits")
                yield self._detail_row("Reload", "load current config into the running app")
                yield self._detail_row("Redaction", "actionable errors without secrets")
                yield self._detail_row("Write safety", "validation is read-only")
                yield self._detail_row(
                    "Diagnostics writes",
                    "unavailable/WIP - raw edits remain gated in Advanced Config",
                )
                with Horizontal(id="settings-diagnostics-actions", classes="settings-action-row"):
                    yield Button(
                        "Validate Config",
                        id="settings-validate-config",
                        tooltip="Validate the current Settings config file.",
                    )
                    yield Button(
                        "Reload Config",
                        id="settings-reload-config",
                        tooltip="Reload the current Settings config into the running app.",
                    )
                yield Static(
                    self._diagnostics_validation_result,
                    id="settings-diagnostics-validation-result",
                    classes="settings-status-row",
                )
                yield Static(
                    self._diagnostics_reload_result,
                    id="settings-diagnostics-reload-result",
                    classes="settings-status-row",
                )
        elif category in DOMAIN_SETTINGS_CATEGORY_IDS:
            yield from self._render_domain_category_detail(category)
        else:
            yield Static("Advanced Config", classes="destination-section settings-column-title")
            with Vertical(id="settings-advanced-config-card", classes="settings-focus-card"):
                raw_config_text = self._raw_config_text()
                yield self._render_category_state_banner(SettingsCategoryId.ADVANCED_CONFIG)
                yield Static("Raw TOML", classes="destination-section")
                yield self._detail_row("Risk level", "expert-only raw configuration editing")
                yield self._detail_row(
                    "Save policy",
                    "Save blocked until the current text validates",
                )
                yield self._detail_row("Write mode", "atomic save with .bak backup before overwrite")
                yield self._detail_row("Required shape", "table-shaped TOML top-level value")
                yield self._detail_row(
                    "Guided path",
                    "prefer category controls unless raw TOML is required",
                )
                yield Static("Guided category paths", classes="destination-section")
                with Horizontal(id="settings-advanced-guided-paths", classes="settings-action-row"):
                    for target_category, label in ADVANCED_CONFIG_GUIDED_PATHS:
                        yield Button(
                            label,
                            id=f"settings-advanced-open-{target_category.value}",
                            classes="settings-advanced-guided-path-button",
                            tooltip=f"Open {label} guided settings instead of editing raw TOML.",
                        )
                yield Static("Raw TOML bypasses guided validation and should be used only for expert edits.")
                yield Static(
                    self._advanced_validation_status(),
                    id="settings-advanced-config-validation-status",
                    classes="settings-status-row settings-advanced-safety-status",
                )
                with Horizontal(id="settings-advanced-config-actions", classes="settings-action-row"):
                    yield Button(
                        "Validate Raw TOML",
                        id="settings-advanced-validate-config",
                        tooltip="Validate raw TOML before writing it to disk.",
                    )
                    yield Button(
                        "Load Backup",
                        id="settings-advanced-load-backup",
                        tooltip="Load the .bak file into the editor without saving.",
                    )
                    save_button = Button(
                        "Save Raw TOML",
                        id="settings-advanced-save-config",
                        tooltip="Atomically save raw TOML after validation.",
                    )
                    save_button.disabled = not self._advanced_save_allowed(raw_config_text)
                    yield save_button
                yield Static(
                    self._advanced_config_result,
                    id="settings-advanced-config-result",
                    classes="settings-status-row",
                )
                yield TextArea(
                    raw_config_text,
                    id="settings-advanced-config-editor",
                )

    def _render_impact_pane(self) -> ComposeResult:
        summary = self._active_summary()
        ownership = self._ownership_record(summary.category)
        yield Static("Scope Inspector", classes="destination-section settings-column-title")
        yield Static(f"Selected category: {summary.title}", classes="destination-section")
        yield Static(
            "Unsaved changes" if self._category_has_unsaved_changes(summary.category) else "No unsaved changes",
            id="settings-selected-category-draft-status",
            classes="destination-section",
        )
        yield Static(
            self._guided_action_message(summary.category),
            id="settings-guided-action-state",
            classes="settings-status-row",
        )
        save_button = Button(
            "Save",
            id="settings-save-category",
            tooltip="Save changes for the selected Settings category.",
        )
        save_button.disabled = not self._guided_actions_enabled(summary.category)
        yield save_button
        revert_button = Button(
            "Revert",
            id="settings-revert-category",
            tooltip="Discard unsaved changes for the selected Settings category.",
        )
        revert_button.disabled = not self._guided_actions_enabled(summary.category)
        yield revert_button
        if summary.category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            yield Static("Control guide", classes="destination-section")
            yield self._detail_row(
                "Streaming",
                "Global fallback for streaming responses when no Console session "
                "or provider+model profile overrides it",
            )
            yield self._detail_row(
                "Temperature",
                "Creativity fallback, 0.0 is focused and 2.0 is exploratory",
            )
            yield self._detail_row(
                "Top P",
                "Probability cutoff fallback; lower values narrow token choices",
            )
            yield self._detail_row(
                "Max tokens",
                "Optional response cap for new/default Console sends",
            )
            yield self._detail_row(
                "Paste collapse",
                "Only pasted chunks over the threshold become compact placeholders; "
                "typed text stays literal",
            )
            yield self._detail_row(
                "Threshold",
                "Minimum pasted chunk size before collapse",
            )
            yield Static("Override rules", classes="destination-section")
            yield self._detail_row(
                "Priority",
                "active Console session, then provider+model profile, then these global fallbacks",
            )
            yield self._detail_row(
                "Save scope",
                "[chat_defaults] response fallbacks and [console] paste display settings",
            )
            return
        elif summary.category is SettingsCategoryId.PROVIDERS_MODELS:
            yield Static("Affects Console and provider-backed generation.", classes="destination-section")
            yield Static(
                self._provider_readiness_label(),
                id="settings-provider-inspector-readiness",
                classes="settings-detail-row",
            )
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(self._provider_field_guidance_rows()):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-provider-field-guide-{index}",
                )
        elif summary.category is SettingsCategoryId.LIBRARY_RAG:
            yield Static("Affects Library search defaults and future RAG answers.", classes="destination-section")
            yield Static("Control guide", classes="destination-section")
            yield self._detail_row(
                "Search mode",
                "plain uses keyword search, semantic uses embeddings, hybrid blends both",
            )
            yield self._detail_row(
                "Result limits",
                "default, keyword, and vector result counts bound retrieval fan-out",
            )
            yield self._detail_row(
                "Hybrid balance",
                "0.0 favors keyword results; 1.0 favors semantic results",
            )
            yield self._detail_row(
                "Citations",
                "sets whether future RAG answers include source markers when supported",
            )
            yield self._detail_row(
                "Snippet/context",
                "snippet length controls preview text; context budget limits staged evidence",
            )
            yield Static("Boundary", classes="destination-section")
            yield self._detail_row(
                "Library owns",
                "indexing, query execution, source browse, Collections, and Console staging",
            )
            yield self._detail_row("Runtime owner", ownership.runtime_owner)
            yield self._detail_row("Writes allowed", "Yes")
            yield self._detail_row(
                "Config keys",
                "10 editable defaults under AppRAGSearchConfig",
            )
            yield self._detail_row("Recovery", ownership.recovery_copy)
            return
        elif summary.category is SettingsCategoryId.APPEARANCE:
            yield Static("Affects launch theme, web density, and visual defaults.", classes="destination-section")
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(self._appearance_field_guidance_rows()):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-appearance-field-guide-{index}",
                )
            yield Static("Boundary", classes="destination-section")
            yield self._detail_row(
                "Settings owns",
                "global defaults and validation before saving",
            )
            yield self._detail_row(
                "Customize owns",
                "full theme editing, custom colors, and deeper preview",
            )
            yield Button(
                "Open Customize",
                id="settings-open-appearance",
                tooltip="Open the dedicated Customize theme editor.",
            )
        elif summary.category is SettingsCategoryId.STORAGE:
            yield Static("Affects local database path defaults after restart.", classes="destination-section")
            yield self._detail_row(
                "Affected config",
                "config file path, local database paths, media storage roots",
            )
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(self._storage_field_guidance_rows()):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-storage-field-guide-{index}",
                )
            yield Static("Boundary", classes="destination-section")
            yield self._detail_row(
                "Restart required",
                "saved paths are picked up on next launch; active handles stay unchanged",
            )
            yield self._detail_row(
                "No migration",
                "Settings does not move files, create directories, or reconnect databases",
            )
            yield self._detail_row("Runtime owner", ownership.runtime_owner)
            yield self._detail_row("Writes allowed", "Yes")
            yield self._detail_row("Recovery", ownership.recovery_copy)
            return
        else:
            yield Static("Impact and boundaries", classes="destination-section")
            yield Static(summary.description)
        yield self._detail_row("Runtime owner", ownership.runtime_owner)
        yield self._detail_row(
            "Writes allowed",
            "Yes" if ownership.writes_allowed else "No",
        )
        if ownership.owns_config_sections:
            yield self._detail_row(
                "Owns",
                ", ".join(ownership.owns_config_sections),
            )
        if ownership.read_only_reason:
            yield self._detail_row("Read-only/WIP", ownership.read_only_reason)
        for label, value in self._inspector_guidance(summary.category):
            yield self._detail_row(
                label,
                value,
                identifier="settings-boundary-note" if label == "Boundary" else None,
            )
        yield Static("Mutation replay: disabled", id="settings-sync-mutation-disabled")
        yield Static(
            "Writes remain blocked until explicit review, conflict, rollback, and audit gates are implemented.",
            id="settings-sync-write-gates",
        )
        if summary.category is SettingsCategoryId.OVERVIEW:
            yield Button(
                "Open Customize",
                id="settings-open-appearance",
                tooltip="Open the dedicated Customize theme editor.",
            )

    def compose_content(self) -> ComposeResult:
        active_summary = self._active_summary()
        with Vertical(id="settings-shell"):
            yield Static(
                "Settings | Global preferences, appearance, accounts, storage | Local",
                id="settings-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="settings-category-strip", classes="destination-mode-strip"):
                yield Static(
                    f"Mode: {active_summary.title} | Runtime controls stay in MCP and ACP",
                    id="settings-category-label",
                    classes="destination-section",
                )
            with Horizontal(id="settings-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="settings-category-pane", classes="destination-workbench-pane"):
                    yield Static("Settings Sections", classes="destination-section settings-column-title")
                    yield Input(
                        value=self.category_search_query,
                        placeholder="Filter settings (/)",
                        id="settings-category-search",
                        classes="settings-category-search",
                    )
                    yield Static(
                        "/ filter | Enter open | Esc clear",
                        id="settings-category-search-help",
                        classes="settings-category-search-help",
                    )
                    yield Static(
                        self._category_search_status_text(),
                        id="settings-category-search-status",
                        classes="settings-category-search-status",
                        markup=False,
                    )
                    yield from self._render_category_buttons()
                yield self._column_divider("settings-category-detail-divider")
                detail_pane_container = (
                    Vertical
                    if active_summary.category is SettingsCategoryId.ADVANCED_CONFIG
                    else VerticalScroll
                )
                with detail_pane_container(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Preference Detail", classes="destination-section settings-column-title")
                    yield from self._render_detail_pane()
                yield self._column_divider("settings-detail-impact-divider")
                with VerticalScroll(
                    id="settings-impact-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield from self._render_impact_pane()

    def _category_value_from_button(self, button: Button) -> str | None:
        if not button.id or not button.has_class("settings-category-button"):
            return None
        prefix = "settings-category-"
        if not button.id.startswith(prefix):
            return None
        value = button.id.removeprefix(prefix)
        if value not in {summary.category.value for summary in self._category_summaries()}:
            return None
        return value

    def _focused_widget(self) -> object | None:
        try:
            return self.app.focused
        except NoActiveAppError:
            return None

    def _focused_category_value(self) -> str | None:
        focused = self._focused_widget()
        if isinstance(focused, Button):
            return self._category_value_from_button(focused)
        return None

    def _category_search_has_focus(self) -> bool:
        focused = self._focused_widget()
        return isinstance(focused, Input) and focused.id == "settings-category-search"

    def _settings_text_entry_has_focus(self) -> bool:
        return isinstance(self._focused_widget(), (Input, TextArea))

    def _focus_category_search(self) -> None:
        try:
            self.query_one("#settings-category-search", Input).focus()
        except QueryError:
            logger.debug("Unable to focus Settings category search")

    def _focus_category(self, category_value: str) -> None:
        try:
            self.query_one(f"#settings-category-{category_value}", Button).focus()
        except QueryError:
            logger.debug("Unable to focus Settings category button: %s", category_value)

    def _move_category_focus(self, delta: int) -> None:
        category_values = self._filtered_category_values()
        if not category_values:
            return
        current_value = self._focused_category_value() or self.active_category
        try:
            current_index = category_values.index(current_value)
        except ValueError:
            current_index = 0
        next_index = max(0, min(len(category_values) - 1, current_index + delta))
        self._focus_category(category_values[next_index])

    def apply_navigation_context(self, context: Mapping[str, object]) -> None:
        """Apply destination-specific navigation context after cross-screen routing.

        Args:
            context: Route context keys. `category` selects the Settings category, and
                optional `provider` / `model` values preselect the Providers & Models
                view when there are no unsaved provider edits.

        Returns:
            None. Navigation context only targets visible UI state; it does not stage or
            persist settings changes.
        """
        category = context.get("category")
        if isinstance(category, SettingsCategoryId):
            category_value = category.value
        elif isinstance(category, str):
            category_value = category
        else:
            return
        valid_categories = {summary.category.value for summary in self._category_summaries()}
        if category_value not in valid_categories:
            logger.debug("Ignoring unknown Settings navigation category: %s", category_value)
            return
        if category_value != SettingsCategoryId.PROVIDERS_MODELS.value:
            self._clear_navigation_provider_context()
            self._select_category(category_value, restore_focus=True)
            return
        provider = str(context.get("provider") or "").strip()
        if not provider:
            self._clear_navigation_provider_context()
            self._select_category(category_value, restore_focus=True)
            return
        model = str(context.get("model") or "").strip()
        field = str(context.get("field") or "").strip()
        if self._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS):
            self._clear_navigation_provider_context()
            self._select_category(category_value, restore_focus=True)
            logger.debug(
                "Preserving dirty Providers & Models draft while routing to provider=%s model=%s",
                provider,
                model,
            )
            return
        self._navigation_provider = provider
        self._navigation_model = model
        self._navigation_field = field
        self._select_category(category_value, restore_focus=True)
        self.call_after_refresh(self._apply_navigation_provider_context, provider, model, field)

    def _apply_navigation_provider_context(
        self,
        provider: str,
        model: str = "",
        field: str = "",
    ) -> None:
        """Synchronize mounted provider widgets after route-targeted navigation.

        Args:
            provider: Provider key to highlight in the mounted Providers & Models UI.
            model: Optional model name to show with the highlighted provider.
            field: Optional field intent to focus after provider/model sync.

        Returns:
            None. This method updates mounted widgets only and does not create a
            SettingsDraft.
        """
        if self.active_category != SettingsCategoryId.PROVIDERS_MODELS.value:
            return
        provider_value = str(provider or "").strip()
        if not provider_value:
            return
        if self._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS):
            return
        provider_settings = self._provider_setting_values_mapping()
        model_value = str(model or provider_settings.get("model") or "").strip()
        self._sync_provider_manual_widget(provider_value)
        self._sync_provider_credential_widget(provider_value)
        try:
            self._syncing_provider_model_value = True
            try:
                self.query_one("#settings-model-value", Input).value = model_value
            finally:
                self._syncing_provider_model_value = False
        except QueryError:
            pass
        self._sync_provider_model_profile_widgets(provider_value, model_value)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)
        self._focus_navigation_provider_field(field or self._navigation_field or "")

    def _focus_navigation_provider_field(self, field: str) -> None:
        field_selectors = {
            "api_key": "#settings-provider-api-key",
            "endpoint": "#settings-provider-endpoint-value",
            "credential_env_var": "#settings-provider-credential-env-var",
        }
        selector = field_selectors.get(str(field or "").strip())
        if not selector:
            return
        try:
            self.query_one(selector).focus()
        except QueryError:
            return

    def _select_category(self, category_value: str, *, restore_focus: bool = False) -> None:
        if category_value != SettingsCategoryId.PROVIDERS_MODELS.value:
            self._active_settings_field_id = None
        self.active_category = category_value
        if category_value == SettingsCategoryId.OVERVIEW.value:
            self._queue_server_sync_workspace_handoff_refresh()
            self._queue_manual_sync_refresh()
        if restore_focus:
            self.call_after_refresh(self._focus_category, category_value)

    @on(DescendantFocus)
    def handle_descendant_focus(self, event: DescendantFocus) -> None:
        active_category = self._active_category_id()
        widget_id = str(getattr(event.widget, "id", "") or "")
        if active_category is SettingsCategoryId.APPEARANCE:
            appearance_field_ids = {
                "settings-appearance-theme",
                "settings-appearance-palette-theme-limit",
                "settings-appearance-font-size",
                "settings-appearance-density",
                "settings-appearance-animations-enabled",
                "settings-appearance-smooth-scrolling",
            }
            self._active_settings_field_id = (
                widget_id if widget_id in appearance_field_ids else None
            )
            self._refresh_appearance_field_guidance()
            return
        if active_category is SettingsCategoryId.STORAGE:
            storage_field_ids = {
                "settings-storage-user-db-base-dir",
                "settings-storage-chachanotes-db-path",
                "settings-storage-prompts-db-path",
                "settings-storage-media-db-path",
                "settings-storage-research-db-path",
                "settings-storage-writing-db-path",
                "settings-storage-library-collections-db-path",
                "settings-storage-workspaces-db-path",
            }
            self._active_settings_field_id = (
                widget_id if widget_id in storage_field_ids else None
            )
            self._refresh_storage_field_guidance()
            return
        if active_category is not SettingsCategoryId.PROVIDERS_MODELS:
            self._active_settings_field_id = None
            return
        provider_field_ids = {
            "settings-provider-value",
            "settings-provider-manual-value",
            "settings-model-value",
            "settings-provider-endpoint-value",
            "settings-provider-api-key",
            "settings-provider-api-key-clear",
            "settings-provider-credential-env-var",
            "settings-model-profile-temperature",
            "settings-model-profile-top-p",
            "settings-model-profile-min-p",
            "settings-model-profile-top-k",
            "settings-model-profile-max-tokens",
            "settings-model-profile-seed",
            "settings-model-profile-presence-penalty",
            "settings-model-profile-frequency-penalty",
            "settings-model-profile-reasoning-effort",
            "settings-model-profile-reasoning-summary",
            "settings-model-profile-verbosity",
            "settings-model-profile-thinking-effort",
            "settings-model-profile-thinking-budget-tokens",
            "settings-model-profile-streaming",
        }
        self._active_settings_field_id = (
            widget_id if widget_id in provider_field_ids else None
        )
        self._refresh_provider_field_guidance()

    @on(Button.Pressed, "#settings-open-appearance")
    def open_appearance_settings(self) -> None:
        self.post_message(NavigateToScreen("customize"))

    @on(Select.Changed, "#settings-appearance-theme")
    def handle_appearance_theme_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_appearance_defaults:
            return
        self._stage_appearance_value("default_theme", str(event.value or "textual-dark"))
        self._mark_appearance_settings_staged()

    @on(Input.Changed, "#settings-appearance-palette-theme-limit")
    def handle_appearance_palette_theme_limit_changed(self, event: Input.Changed) -> None:
        if self._syncing_appearance_defaults:
            return
        self._stage_appearance_value(
            "palette_theme_limit",
            self._normalise_appearance_int(event.value),
        )
        self._mark_appearance_settings_staged()

    @on(Input.Changed, "#settings-appearance-font-size")
    def handle_appearance_font_size_changed(self, event: Input.Changed) -> None:
        if self._syncing_appearance_defaults:
            return
        self._stage_appearance_value(
            "font_size",
            self._normalise_appearance_int(event.value),
        )
        self._mark_appearance_settings_staged()

    @on(Select.Changed, "#settings-appearance-density")
    def handle_appearance_density_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_appearance_defaults:
            return
        self._stage_appearance_value("density", str(event.value or "normal"))
        self._mark_appearance_settings_staged()

    @on(Button.Pressed, "#settings-appearance-animations-enabled")
    def handle_appearance_animations_enabled_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not bool(self._appearance_setting_values()["animations_enabled"])
        self._stage_appearance_value("animations_enabled", next_value)
        event.button.label = self._appearance_bool_label("animations_enabled")
        self._mark_appearance_settings_staged()

    @on(Button.Pressed, "#settings-appearance-smooth-scrolling")
    def handle_appearance_smooth_scrolling_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not bool(self._appearance_setting_values()["smooth_scrolling"])
        self._stage_appearance_value("smooth_scrolling", next_value)
        event.button.label = self._appearance_bool_label("smooth_scrolling")
        self._mark_appearance_settings_staged()

    @on(Button.Pressed, "#settings-preview-appearance")
    def handle_preview_appearance(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category()

    @on(Button.Pressed, "#settings-switch-runtime-source")
    def handle_switch_runtime_source(self, event: Button.Pressed) -> None:
        """Open the runtime-source switch modal (callback-based, never wait)."""
        event.stop()
        from tldw_chatbook.Widgets.Settings_Widgets.server_switch_modal import ServerSwitchModal

        state = self._runtime_source_state()
        raw_config = (getattr(self.app_instance, "app_config", {}) or {}).get(
            "COMPREHENSIVE_CONFIG_RAW", {}
        )
        api_config = raw_config.get("tldw_api", {}) if isinstance(raw_config, dict) else {}
        modal = ServerSwitchModal(
            current_source=str(getattr(state, "active_source", "local") or "local"),
            current_server_label=str(
                getattr(state, "last_known_server_label", "")
                or getattr(state, "active_server_id", "")
                or ""
            ),
            current_base_url=str(api_config.get("base_url") or ""),
            current_auth_token=str(api_config.get("auth_token") or ""),
        )
        self.app.push_screen(modal, self._handle_runtime_source_switch_result)

    def _handle_runtime_source_switch_result(self, result: dict | None) -> None:
        if not result:
            return
        self.run_worker(
            self._perform_runtime_source_switch(result),
            exclusive=True,
            group="settings-runtime-source-switch",
        )

    async def _perform_runtime_source_switch(self, result: dict) -> None:
        """Apply the modal decision: persist, rebind, switch, enroll Sync v2."""
        app = self.app_instance
        if result.get("action") == "local":
            await app.handle_runtime_backend_changed("local")
            self.app.notify("Runtime source set to local.", severity="information")
            self._refresh_manual_sync_rows()
            return

        base_url = str(result.get("base_url") or "").strip()
        if not base_url:
            return
        from tldw_chatbook.config import load_settings, save_setting_to_cli_config
        from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app

        from tldw_chatbook.Utils.input_validation import validate_url

        if not validate_url(base_url):
            self.app.notify(
                "Rejected server URL; nothing was changed.", severity="error"
            )
            return
        save_setting_to_cli_config("tldw_api", "base_url", base_url)
        # Persist the token unconditionally so clearing it in the modal
        # actually removes the stored credential.
        auth_token = str(result.get("auth_token") or "").strip()
        save_setting_to_cli_config("tldw_api", "auth_token", auth_token)
        app.app_config = load_settings(force_reload=True)
        load_runtime_policy_for_app(app)
        await app.handle_runtime_backend_changed("server")

        state = self._runtime_source_state()
        server_id = str(getattr(state, "active_server_id", "") or "").strip()
        if not server_id:
            self.app.notify(
                "Server could not be bound from the entered URL.", severity="error"
            )
            return

        sync_scope_service = getattr(app, "sync_scope_service", None)
        prepare = getattr(sync_scope_service, "prepare_sync_v2_profile_mode", None)
        if callable(prepare):
            import platform

            try:
                summary = await prepare(
                    profile_mode="local_first_sync",
                    server_profile_id=server_id,
                    display_name=platform.node() or "tldw_chatbook",
                )
            except Exception as exc:
                logger.warning(
                    "Sync v2 profile preparation failed (server_profile_id={}, mode=local_first_sync).",
                    server_id,
                    exc_info=True,
                )
                self.app.notify(
                    f"Server activated, but Sync v2 setup failed: {exc}",
                    severity="warning",
                )
            else:
                dataset = str(summary.get("dataset_id") or "unknown")
                self.app.notify(
                    f"Server activated; Sync v2 prepared (dataset {dataset}).",
                    severity="information",
                )
        else:
            self.app.notify(
                "Server activated. Sync v2 service is unavailable in this runtime.",
                severity="warning",
            )
        self._refresh_manual_sync_rows()

    @on(Button.Pressed, "#settings-manual-sync-preview")
    def handle_manual_sync_preview(self, event: Button.Pressed) -> None:
        event.stop()
        self.manual_sync_rows = (
            ("Manual sync status", "loading"),
            ("Manual sync preview", "Refreshing manual Sync v2 preview."),
            ("Pending outgoing", "Loading"),
        )
        self._refresh_manual_sync_rows()

    @on(Button.Pressed, "#settings-manual-sync-run")
    def handle_manual_sync_run(self, event: Button.Pressed) -> None:
        event.stop()
        self.manual_sync_rows = (
            ("Manual sync status", "running"),
            ("Manual sync result", "Manual Sync is running after explicit user request."),
            ("Pending outgoing", "Refreshing"),
        )
        self._manual_sync_run_worker()

    @on(Button.Pressed, ".settings-category-button")
    def handle_category_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        category_value = self._category_value_from_button(event.button)
        if category_value is not None:
            self._select_category(category_value, restore_focus=event.button.has_focus)

    @on(Input.Changed, "#settings-category-search")
    def handle_category_search_changed(self, event: Input.Changed) -> None:
        event.stop()
        query_text = self._sanitize_category_search_query(event.value)
        self.category_search_query = query_text
        if query_text != event.value:
            event.input.value = query_text
        self._apply_category_search_filter()

    @on(Input.Submitted, "#settings-category-search")
    def handle_category_search_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit_category_search(event.value)

    @on(Button.Pressed, "#settings-console-collapse-large-pastes-toggle")
    def handle_console_collapse_large_pastes_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not self._collapse_large_pastes_enabled()
        self._stage_console_large_paste_value(next_value)
        event.button.label = self._collapse_large_pastes_button_label()
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-console-paste-collapse-threshold")
    def handle_console_paste_threshold_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_threshold:
            return
        self._stage_console_paste_threshold_value(event.value)
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result_text())
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-console-default-streaming")
    def handle_console_default_streaming_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_defaults:
            return
        try:
            value = self._normalise_console_default_streaming(event.value)
        except ValueError:
            value = event.value
        self._stage_console_default_value("streaming", value)
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-default-temperature")
    def handle_console_default_temperature_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_defaults:
            return
        try:
            value = self._normalise_console_default_temperature(event.value)
        except ValueError:
            value = event.value
        self._stage_console_default_value("temperature", value)
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-default-top-p")
    def handle_console_default_top_p_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_defaults:
            return
        try:
            value = self._normalise_console_default_top_p(event.value)
        except ValueError:
            value = event.value
        self._stage_console_default_value("top_p", value)
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-default-min-p")
    def handle_console_default_min_p_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input("min_p", event.value, self._normalise_model_profile_min_p)

    @on(Input.Changed, "#settings-console-default-top-k")
    def handle_console_default_top_k_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input("top_k", event.value, self._normalise_model_profile_top_k)

    @on(Input.Changed, "#settings-console-default-max-tokens")
    def handle_console_default_max_tokens_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_defaults:
            return
        try:
            value = self._normalise_console_default_max_tokens(event.value)
        except ValueError:
            value = event.value
        self._stage_console_default_value("max_tokens", value)
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-default-seed")
    def handle_console_default_seed_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input("seed", event.value, self._normalise_model_profile_seed)

    @on(Input.Changed, "#settings-console-default-presence-penalty")
    def handle_console_default_presence_penalty_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "presence_penalty",
            event.value,
            self._normalise_model_profile_presence_penalty,
        )

    @on(Input.Changed, "#settings-console-default-frequency-penalty")
    def handle_console_default_frequency_penalty_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "frequency_penalty",
            event.value,
            self._normalise_model_profile_frequency_penalty,
        )

    @on(Input.Changed, "#settings-console-default-reasoning-effort")
    def handle_console_default_reasoning_effort_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "reasoning_effort",
            event.value,
            self._normalise_model_profile_reasoning_effort,
        )

    @on(Input.Changed, "#settings-console-default-reasoning-summary")
    def handle_console_default_reasoning_summary_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "reasoning_summary",
            event.value,
            self._normalise_model_profile_reasoning_summary,
        )

    @on(Input.Changed, "#settings-console-default-verbosity")
    def handle_console_default_verbosity_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "verbosity",
            event.value,
            self._normalise_model_profile_verbosity,
        )

    @on(Input.Changed, "#settings-console-default-thinking-effort")
    def handle_console_default_thinking_effort_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "thinking_effort",
            event.value,
            self._normalise_model_profile_thinking_effort,
        )

    @on(Input.Changed, "#settings-console-default-thinking-budget-tokens")
    def handle_console_default_thinking_budget_tokens_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "thinking_budget_tokens",
            event.value,
            self._normalise_model_profile_thinking_budget_tokens,
        )

    def _stage_console_default_input(self, key: str, raw_value: object, normalizer) -> None:
        if self._syncing_console_defaults:
            return
        try:
            value = normalizer(raw_value)
        except ValueError:
            value = raw_value
        self._stage_console_default_value(key, value)
        self._mark_console_behavior_settings_staged()

    def _mark_console_behavior_settings_staged(self) -> None:
        if self._category_has_unsaved_changes(SettingsCategoryId.CONSOLE_BEHAVIOR):
            self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result_text())
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Button.Pressed, "#settings-console-background-effect-enabled")
    def handle_console_background_effect_enabled_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not bool(self._console_background_effect_value("enabled"))
        self._stage_console_background_effect_value("enabled", next_value)
        event.button.label = self._console_background_effect_enabled_label()
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-type")
    def handle_console_background_effect_type_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        self._stage_console_background_effect_value("effect", str(event.value or "none"))
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-scope")
    def handle_console_background_effect_scope_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        next_scope = self._available_console_background_scope(event.value)
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.get(category)
        if (
            next_scope == "transcript"
            and self._loaded_console_background_scope_is_unavailable()
            and (
                draft is None
                or "background_effects.scope" not in draft.values
            )
        ):
            self._console_behavior_result = CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
            self._update_draft_status_widgets(category)
            return
        if (
            next_scope == "transcript"
            and draft is not None
            and draft.originals.get("background_effects.scope") == "workbench"
            and draft.values.get("background_effects.scope") == "transcript"
        ):
            self._console_behavior_result = CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
            self._update_draft_status_widgets(category)
            return
        if str(event.value) == "workbench":
            draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
            draft.set_value("background_effects.scope", "workbench", next_scope)
            self._syncing_console_background_effects = True
            try:
                event.select.value = next_scope
            finally:
                self._syncing_console_background_effects = False
            self._console_behavior_result = CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
            self._update_draft_status_widgets(category)
            return
        self._stage_console_background_effect_value("scope", next_scope)
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-intensity")
    def handle_console_background_effect_intensity_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        self._stage_console_background_effect_value("intensity", str(event.value or "low"))
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-background-effect-fps")
    def handle_console_background_effect_fps_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_background_effects:
            return
        value: object = int(event.value) if str(event.value).isdigit() else event.value
        self._stage_console_background_effect_value("fps", value)
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-library-rag-search-mode")
    def handle_library_rag_search_mode_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value("default_search_mode", str(event.value or "semantic"))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-default-top-k")
    def handle_library_rag_default_top_k_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "default_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-fts-top-k")
    def handle_library_rag_fts_top_k_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "fts_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-vector-top-k")
    def handle_library_rag_vector_top_k_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "vector_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-hybrid-alpha")
    def handle_library_rag_hybrid_alpha_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "hybrid_alpha",
            self._normalise_library_rag_float(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-score-threshold")
    def handle_library_rag_score_threshold_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "score_threshold",
            self._normalise_library_rag_float(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Button.Pressed, "#settings-library-rag-include-citations")
    def handle_library_rag_include_citations_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not bool(self._library_rag_setting_values()["include_citations"])
        self._stage_library_rag_value("include_citations", next_value)
        event.button.label = "Enabled" if next_value else "Disabled"
        self._mark_library_rag_settings_staged()

    @on(Select.Changed, "#settings-library-rag-citation-style")
    def handle_library_rag_citation_style_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value("citation_style", str(event.value or "inline"))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-snippet-max-chars")
    def handle_library_rag_snippet_max_chars_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "snippet_max_chars",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-max-context-size")
    def handle_library_rag_max_context_size_changed(self, event: Input.Changed) -> None:
        if self._syncing_library_rag_defaults:
            return
        self._stage_library_rag_value(
            "max_context_size",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-storage-user-db-base-dir")
    def handle_storage_user_db_base_dir_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("user_db_base_dir", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-chachanotes-db-path")
    def handle_storage_chachanotes_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("chachanotes_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-prompts-db-path")
    def handle_storage_prompts_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("prompts_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-media-db-path")
    def handle_storage_media_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("media_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-research-db-path")
    def handle_storage_research_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("research_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-writing-db-path")
    def handle_storage_writing_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("writing_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-library-collections-db-path")
    def handle_storage_library_collections_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("library_collections_db_path", event.value)
        self._mark_storage_settings_staged()

    @on(Input.Changed, "#settings-storage-workspaces-db-path")
    def handle_storage_workspaces_db_path_changed(self, event: Input.Changed) -> None:
        if self._syncing_storage_defaults:
            return
        self._stage_storage_value("workspaces_db_path", event.value)
        self._mark_storage_settings_staged()

    def _apply_provider_value_change(self, provider: str) -> None:
        self._clear_navigation_provider_context()
        loaded_provider = str(self._provider_loaded_setting_values().get("provider") or "")
        previous_provider = str(self._provider_setting_values_mapping().get("provider") or "")
        provider_changed = (
            bool(provider)
            and provider_config_key(provider) != provider_config_key(previous_provider)
        )
        staged_provider = (
            loaded_provider
            if (
                provider
                and provider_config_key(provider) == provider_config_key(loaded_provider)
            )
            else provider
        )
        self._stage_provider_value("provider", staged_provider or None)
        self._sync_provider_manual_widget(staged_provider)
        try:
            endpoint_input = self.query_one("#settings-provider-endpoint-value", Input)
        except QueryError:
            endpoint_input = None
        if endpoint_input is not None:
            self._syncing_provider_endpoint = True
            try:
                endpoint_input.value = self._provider_endpoint_value(staged_provider)
                endpoint_input.placeholder = self._provider_endpoint_placeholder(staged_provider)
            finally:
                self._syncing_provider_endpoint = False
        self._sync_provider_credential_widget(staged_provider)
        provider_default_model = (
            self._provider_model_default(staged_provider) if provider_changed else ""
        )
        if provider_changed:
            self._stage_provider_value("model", provider_default_model or None)
            try:
                self.query_one("#settings-model-value", Input).value = provider_default_model
            except QueryError:
                pass
        model = str(self._provider_setting_values_mapping().get("model") or "")
        self._sync_provider_model_profile_widgets(staged_provider, model)
        self._clear_provider_auxiliary_draft_keys()
        self._reset_provider_model_discovery_state()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Select.Changed, "#settings-provider-value")
    def handle_provider_value_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_provider_selection:
            return
        selected_value = self._select_value_text(event.value)
        provider = (
            self._provider_widget_value()
            if selected_value == PROVIDER_MANUAL_SELECT_VALUE
            else selected_value
        )
        if (
            self._navigation_provider
            and provider_config_key(provider) == provider_config_key(self._navigation_provider)
        ):
            return
        current_provider = str(self._provider_setting_values_mapping().get("provider") or "")
        if provider_config_key(provider) == provider_config_key(current_provider):
            return
        self._apply_provider_value_change(provider)

    @on(Input.Changed, "#settings-provider-manual-value")
    def handle_provider_manual_value_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_manual:
            return
        self._apply_provider_value_change(event.value.strip())

    @on(Input.Changed, "#settings-model-value")
    def handle_model_value_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_model_value:
            return
        model_value = event.value.strip()
        if self._navigation_model is not None and model_value == self._navigation_model:
            return
        current_model = str(self._provider_setting_values_mapping().get("model") or "").strip()
        if model_value == current_model:
            return
        self._clear_navigation_provider_context()
        self._stage_provider_value("model", model_value or None)
        provider = str(self._provider_setting_values_mapping().get("provider") or "")
        self._sync_provider_model_profile_widgets(provider, model_value)
        self._clear_provider_model_profile_draft_keys()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-provider-endpoint-value")
    def handle_provider_endpoint_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_endpoint:
            self._update_provider_dynamic_widgets()
            return
        self._stage_provider_value("endpoint", event.value.strip())
        self._reset_provider_model_discovery_state()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-provider-credential-env-var")
    def handle_provider_credential_env_var_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_credential_env_var:
            self._update_provider_dynamic_widgets()
            return
        self._stage_provider_value("credential_env_var", event.value.strip())
        self._reset_provider_model_discovery_state()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-provider-api-key")
    def handle_provider_api_key_changed(self, event: Input.Changed) -> None:
        """Stage a local provider API-key draft.

        Args:
            event: Textual input change event containing the masked key value.

        Returns:
            None.
        """
        if self._syncing_provider_api_key:
            self._update_provider_dynamic_widgets()
            return
        self._stage_provider_value("api_key", event.value.strip())
        self._reset_provider_model_discovery_state()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Button.Pressed, "#settings-provider-api-key-clear")
    def handle_provider_api_key_clear_pressed(self, event: Button.Pressed) -> None:
        """Clear the staged local provider API key.

        Args:
            event: Textual button press event from the provider API-key clear control.

        Returns:
            None.
        """
        event.stop()
        try:
            api_key_input = self.query_one("#settings-provider-api-key", Input)
        except QueryError:
            api_key_input = None
        self._stage_provider_value("api_key", "")
        self._syncing_provider_api_key = True
        try:
            if api_key_input is not None:
                api_key_input.value = ""
        finally:
            self._syncing_provider_api_key = False
        self._reset_provider_model_discovery_state()
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-profile-temperature")
    def handle_model_profile_temperature_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_model_profile:
            return
        try:
            value = self._normalise_model_profile_temperature(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value("model_profile_temperature", value)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-profile-top-p")
    def handle_model_profile_top_p_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_model_profile:
            return
        try:
            value = self._normalise_model_profile_top_p(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value("model_profile_top_p", value)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-profile-min-p")
    def handle_model_profile_min_p_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_min_p",
            event.value,
            self._normalise_model_profile_min_p,
        )

    @on(Input.Changed, "#settings-model-profile-top-k")
    def handle_model_profile_top_k_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_top_k",
            event.value,
            self._normalise_model_profile_top_k,
        )

    @on(Input.Changed, "#settings-model-profile-max-tokens")
    def handle_model_profile_max_tokens_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_max_tokens",
            event.value,
            self._normalise_model_profile_max_tokens,
        )

    @on(Input.Changed, "#settings-model-profile-seed")
    def handle_model_profile_seed_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_seed",
            event.value,
            self._normalise_model_profile_seed,
        )

    @on(Input.Changed, "#settings-model-profile-presence-penalty")
    def handle_model_profile_presence_penalty_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_presence_penalty",
            event.value,
            self._normalise_model_profile_presence_penalty,
        )

    @on(Input.Changed, "#settings-model-profile-frequency-penalty")
    def handle_model_profile_frequency_penalty_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_frequency_penalty",
            event.value,
            self._normalise_model_profile_frequency_penalty,
        )

    @on(Input.Changed, "#settings-model-profile-reasoning-effort")
    def handle_model_profile_reasoning_effort_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_reasoning_effort",
            event.value,
            self._normalise_model_profile_reasoning_effort,
        )

    @on(Input.Changed, "#settings-model-profile-reasoning-summary")
    def handle_model_profile_reasoning_summary_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_reasoning_summary",
            event.value,
            self._normalise_model_profile_reasoning_summary,
        )

    @on(Input.Changed, "#settings-model-profile-verbosity")
    def handle_model_profile_verbosity_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_verbosity",
            event.value,
            self._normalise_model_profile_verbosity,
        )

    @on(Input.Changed, "#settings-model-profile-thinking-effort")
    def handle_model_profile_thinking_effort_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_thinking_effort",
            event.value,
            self._normalise_model_profile_thinking_effort,
        )

    @on(Input.Changed, "#settings-model-profile-thinking-budget-tokens")
    def handle_model_profile_thinking_budget_tokens_changed(self, event: Input.Changed) -> None:
        self._stage_model_profile_input(
            "model_profile_thinking_budget_tokens",
            event.value,
            self._normalise_model_profile_thinking_budget_tokens,
        )

    def _stage_model_profile_input(self, key: str, raw_value: object, normalizer) -> None:
        if self._syncing_provider_model_profile:
            return
        try:
            value = normalizer(raw_value)
        except ValueError:
            value = raw_value
        self._stage_provider_value(key, value)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-profile-streaming")
    def handle_model_profile_streaming_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_model_profile:
            return
        try:
            value = self._normalise_optional_bool(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value(
            "model_profile_streaming",
            value,
        )
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Button.Pressed, "#settings-save-category")
    def handle_save_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_save_category(allow_text_entry_focus=True)

    @on(Button.Pressed, "#settings-revert-category")
    def handle_revert_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_revert_category(allow_text_entry_focus=True)

    @on(Button.Pressed, "#settings-test-provider")
    def handle_test_provider(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category(allow_text_entry_focus=True)

    @on(Button.Pressed, "#settings-discover-provider-models")
    def handle_discover_provider_models(self, event: Button.Pressed) -> None:
        event.stop()
        self._discover_provider_models_worker()

    @on(Button.Pressed, "#settings-save-discovered-provider-models")
    def handle_save_discovered_provider_models(self, event: Button.Pressed) -> None:
        event.stop()
        self._save_selected_discovered_provider_models_worker()

    @on(Button.Pressed, "#settings-clear-discovered-provider-models")
    def handle_clear_discovered_provider_models(self, event: Button.Pressed) -> None:
        event.stop()
        self._clear_discovered_provider_models_worker()

    @on(Button.Pressed, "#settings-check-storage")
    def handle_check_storage(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category()

    @on(Button.Pressed, "#settings-check-privacy")
    def handle_check_privacy(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category()

    @on(Button.Pressed, "#settings-open-provider-credentials")
    def handle_open_provider_credentials(self, event: Button.Pressed) -> None:
        event.stop()
        self._select_category(SettingsCategoryId.PROVIDERS_MODELS.value, restore_focus=True)

    @on(Button.Pressed, "#settings-open-advanced-config")
    def handle_open_advanced_config_from_privacy(self, event: Button.Pressed) -> None:
        event.stop()
        self._select_category(SettingsCategoryId.ADVANCED_CONFIG.value, restore_focus=True)

    @on(Button.Pressed, "#settings-validate-config")
    def handle_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._run_diagnostics_validation()

    @on(Button.Pressed, "#settings-reload-config")
    def handle_reload_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._run_diagnostics_reload()

    @on(Button.Pressed, "#settings-advanced-validate-config")
    def handle_advanced_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        current_text = self._advanced_editor_text()
        self._advanced_config_result = "Advanced config validation: running"
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)
        self._update_advanced_validation_status()
        self._advanced_validate_config_worker(current_text)

    @on(Button.Pressed, "#settings-advanced-save-config")
    def handle_advanced_save_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = "Advanced config save: saving"
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)
        try:
            self.query_one("#settings-advanced-save-config", Button).disabled = True
        except QueryError:
            pass
        self._advanced_save_config_worker(self._advanced_editor_text())

    @on(Button.Pressed, "#settings-advanced-load-backup")
    def handle_advanced_load_backup(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = "Advanced config recovery: loading backup preview"
        self._set_static_text("#settings-advanced-config-result", self._advanced_config_result)
        self._advanced_load_backup_worker()

    @on(Button.Pressed, ".settings-advanced-guided-path-button")
    def handle_advanced_guided_path(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        target_category = ADVANCED_CONFIG_GUIDED_PATH_BUTTONS.get(button_id)
        if target_category is not None:
            self._select_category(target_category.value, restore_focus=True)

    @on(TextArea.Changed, "#settings-advanced-config-editor")
    def handle_advanced_config_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        self._update_advanced_validation_status()

    def action_settings_save_category(self, *, allow_text_entry_focus: bool = False) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        category = self._active_category_id()
        if category not in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            self.app.notify(self._guided_action_message(category), severity="information")
            return
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            try:
                values = self._provider_form_values_from_widgets()
            except ValueError as exc:
                self._provider_save_result = str(exc) or "Model profile values are invalid."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            loaded_values = self._provider_loaded_setting_values()
            chat_defaults_keys = {"provider", "model"}
            provider = str(values.get("provider") or "").strip()
            model = str(values.get("model") or "").strip()
            endpoint = str(values.get("endpoint") or "").strip()
            api_key = str(values.get("api_key") or "").strip()
            credential_env_var = str(values.get("credential_env_var") or "").strip()
            draft = self._settings_drafts.get(category)
            if not provider_config_key(provider):
                self._provider_save_result = "Provider is required."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            endpoint_touched = draft is not None and "endpoint" in draft.dirty_keys
            loaded_endpoint = str(loaded_values.get("endpoint") or "").strip()
            if (
                loaded_values.get("provider") != provider
                and not endpoint_touched
                and endpoint == loaded_endpoint
            ):
                endpoint = self._provider_endpoint_value(provider)
                values["endpoint"] = endpoint
                try:
                    self.query_one("#settings-provider-endpoint-value", Input).value = endpoint
                except QueryError:
                    pass
            endpoint_validation_error = self._validate_provider_endpoint(endpoint)
            if endpoint_validation_error:
                self._provider_save_result = endpoint_validation_error
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(endpoint_validation_error, severity="error")
                return
            credential_validation_error = self._validate_credential_env_var(credential_env_var)
            if credential_validation_error:
                self._provider_save_result = credential_validation_error
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(credential_validation_error, severity="error")
                return
            api_key_validation_error = self._validate_provider_api_key(api_key)
            if api_key_validation_error:
                self._provider_save_result = api_key_validation_error
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(api_key_validation_error, severity="error")
                return
            dirty_values = {
                key: value
                for key, value in values.items()
                if key in chat_defaults_keys and loaded_values.get(key) != value
            }
            dirty_keys = draft.dirty_keys if draft is not None else set()
            selected_profile = self._provider_model_profile(provider, model)
            model_profile_dirty = any(
                key in dirty_keys
                and self._model_profile_field_supported(provider, key)
                and values.get(key, "") != selected_profile.get(profile_key, "")
                for key, profile_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS.items()
            )
            provider_key = provider_config_key(provider)
            provider_section_key, _provider_config = self._provider_config_entry(provider)
            current_provider_endpoint = self._provider_endpoint_value(provider)
            current_credential_env_var = self._provider_credential_env_var(provider)
            api_key_dirty = draft is not None and "api_key" in draft.dirty_keys
            endpoint_dirty = endpoint != current_provider_endpoint
            credential_dirty = credential_env_var != current_credential_env_var
            if endpoint_dirty and not provider_key:
                self._provider_save_result = "Provider is required before saving an endpoint."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if credential_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving a credential source."
                )
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if api_key_dirty and not provider_key:
                self._provider_save_result = "Provider is required before saving an API key."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if model_profile_dirty and not model:
                self._provider_save_result = "Model is required before saving a model default profile."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if model_profile_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving a model default profile."
                )
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if (
                not dirty_values
                and not endpoint_dirty
                and not credential_dirty
                and not api_key_dirty
                and not model_profile_dirty
            ):
                self._settings_drafts.pop(category, None)
                self._update_provider_dynamic_widgets()
                self._update_draft_status_widgets(category)
                self._provider_save_result = "Provider settings: no changes to save."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify("No Settings changes to save.", severity="information")
                return
            saved = True
            if dirty_values:
                saved = SettingsConfigAdapter().save_values("chat_defaults", dirty_values)
            endpoint_key = self._provider_endpoint_setting_key(provider)
            provider_settings_values = {}
            if endpoint_dirty:
                provider_settings_values[endpoint_key] = endpoint
            if api_key_dirty:
                provider_settings_values["api_key"] = api_key
            if credential_dirty:
                provider_settings_values["api_key_env_var"] = credential_env_var
            if provider_settings_values and provider_key:
                provider_save_key = provider_section_key or provider_key
                provider_settings_saved = SettingsConfigAdapter().save_values(
                    f"api_settings.{provider_save_key}",
                    provider_settings_values,
                )
                saved = saved and provider_settings_saved
            next_model_defaults = None
            if model_profile_dirty and provider_key and model:
                provider_save_key = provider_section_key or provider_key
                next_model_defaults = self._updated_model_defaults_for_values(
                    provider,
                    model,
                    values,
                )
                profile_saved = SettingsConfigAdapter().save_values(
                    f"api_settings.{provider_save_key}",
                    {"model_defaults": next_model_defaults},
                )
                saved = saved and profile_saved
            if saved:
                defaults = self._chat_defaults()
                defaults.update(dirty_values)
                if dirty_values:
                    self._sync_provider_runtime_defaults(
                        str(values.get("provider") or "").strip(),
                        str(values.get("model") or "").strip(),
                    )
                if (
                    endpoint_dirty
                    or credential_dirty
                    or api_key_dirty
                    or next_model_defaults is not None
                ) and provider_key:
                    app_config = getattr(self.app_instance, "app_config", None)
                    if not isinstance(app_config, dict):
                        self.app_instance.app_config = {}
                        app_config = self.app_instance.app_config
                    api_settings = app_config.setdefault("api_settings", {})
                    if not isinstance(api_settings, dict):
                        api_settings = {}
                        app_config["api_settings"] = api_settings
                    provider_save_key = provider_section_key or provider_key
                    provider_settings = api_settings.setdefault(provider_save_key, {})
                    if not isinstance(provider_settings, dict):
                        provider_settings = {}
                        api_settings[provider_save_key] = provider_settings
                    provider_settings.update(provider_settings_values)
                    if next_model_defaults is not None:
                        provider_settings["model_defaults"] = next_model_defaults
                self._settings_drafts.pop(category, None)
                self._provider_save_result = "Provider settings saved."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self._sync_provider_credential_widget(provider)
                self._update_provider_dynamic_widgets()
                self._update_draft_status_widgets(category)
                self.app.notify("Provider and model settings saved.", severity="information")
            else:
                self._provider_save_result = "Failed to save provider and model settings."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify("Failed to save provider and model settings.", severity="error")
            return

        if category is SettingsCategoryId.STORAGE:
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._storage_current_defaults()
            validation = validate_storage_defaults(values)
            if not validation.valid:
                self._storage_result = validation.message
                self._set_static_text("#settings-storage-save-result", self._storage_result)
                self._update_storage_validation_classes()
                self._update_draft_status_widgets(category)
                self.app.notify(validation.message, severity="error")
                return
            section_values = build_storage_save_sections(
                self._app_config_mapping(),
                values,
            )
            self._storage_result = "Storage defaults saving..."
            self._set_static_text("#settings-storage-save-result", self._storage_result)
            self._settings_save_storage_worker(section_values)
            return

        if category is SettingsCategoryId.LIBRARY_RAG:
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._library_rag_current_defaults()
            validation = validate_library_rag_defaults(values)
            if not validation.valid:
                self._library_rag_result = validation.message
                self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
                self._update_draft_status_widgets(category)
                self.app.notify(validation.message, severity="error")
                return
            section_values = build_library_rag_save_sections(
                self._app_config_mapping(),
                values,
            )
            self._library_rag_result = "Saving Library/RAG defaults..."
            self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
            self._settings_save_library_rag_worker(section_values)
            return

        if category is SettingsCategoryId.APPEARANCE:
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._appearance_current_defaults()
            validation = validate_appearance_defaults(values)
            if not validation.valid:
                self._appearance_result = validation.message
                self._set_static_text("#settings-appearance-save-result", self._appearance_result)
                self._update_appearance_validation_classes()
                self._update_draft_status_widgets(category)
                self.app.notify(validation.message, severity="error")
                return
            section_values = build_appearance_save_sections(
                self._app_config_mapping(),
                values,
            )
            self._appearance_result = "Appearance defaults saving..."
            self._set_static_text("#settings-appearance-save-result", self._appearance_result)
            self._settings_save_appearance_worker(section_values)
            return

        draft = self._settings_drafts.get(category)
        if not draft or not draft.is_dirty:
            self.app.notify("No Settings changes to save.", severity="information")
            return

        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            dirty_values = {
                key: draft.values[key]
                for key in CONSOLE_BEHAVIOR_SAVE_ORDER
                if key in draft.dirty_keys
            }
            try:
                if "paste_collapse_threshold" in dirty_values:
                    dirty_values["paste_collapse_threshold"] = (
                        self._normalise_paste_collapse_threshold(
                            dirty_values["paste_collapse_threshold"]
                        )
                    )
                if "streaming" in dirty_values:
                    dirty_values["streaming"] = self._normalise_console_default_streaming(
                        dirty_values["streaming"]
                    )
                if "temperature" in dirty_values:
                    dirty_values["temperature"] = self._normalise_console_default_temperature(
                        dirty_values["temperature"]
                    )
                if "top_p" in dirty_values:
                    dirty_values["top_p"] = self._normalise_console_default_top_p(
                        dirty_values["top_p"]
                    )
                if "min_p" in dirty_values:
                    dirty_values["min_p"] = self._normalise_model_profile_min_p(
                        dirty_values["min_p"]
                    )
                if "top_k" in dirty_values:
                    dirty_values["top_k"] = self._normalise_model_profile_top_k(
                        dirty_values["top_k"]
                    )
                if "max_tokens" in dirty_values:
                    dirty_values["max_tokens"] = self._normalise_console_default_max_tokens(
                        dirty_values["max_tokens"]
                    )
                if "seed" in dirty_values:
                    dirty_values["seed"] = self._normalise_model_profile_seed(
                        dirty_values["seed"]
                    )
                if "presence_penalty" in dirty_values:
                    dirty_values["presence_penalty"] = (
                        self._normalise_model_profile_presence_penalty(
                            dirty_values["presence_penalty"]
                        )
                    )
                if "frequency_penalty" in dirty_values:
                    dirty_values["frequency_penalty"] = (
                        self._normalise_model_profile_frequency_penalty(
                            dirty_values["frequency_penalty"]
                        )
                    )
                if "reasoning_effort" in dirty_values:
                    dirty_values["reasoning_effort"] = (
                        self._normalise_model_profile_reasoning_effort(
                            dirty_values["reasoning_effort"]
                        )
                    )
                if "reasoning_summary" in dirty_values:
                    dirty_values["reasoning_summary"] = (
                        self._normalise_model_profile_reasoning_summary(
                            dirty_values["reasoning_summary"]
                        )
                    )
                if "verbosity" in dirty_values:
                    dirty_values["verbosity"] = self._normalise_model_profile_verbosity(
                        dirty_values["verbosity"]
                    )
                if "thinking_effort" in dirty_values:
                    dirty_values["thinking_effort"] = (
                        self._normalise_model_profile_thinking_effort(
                            dirty_values["thinking_effort"]
                        )
                    )
                if "thinking_budget_tokens" in dirty_values:
                    dirty_values["thinking_budget_tokens"] = (
                        self._normalise_model_profile_thinking_budget_tokens(
                            dirty_values["thinking_budget_tokens"]
                        )
                    )
                if "background_effects.fps" in dirty_values:
                    dirty_values["background_effects.fps"] = (
                        self._normalise_console_background_fps(
                            dirty_values["background_effects.fps"]
                        )
                    )
            except ValueError as exc:
                self._console_behavior_result = str(exc)
                self._set_static_text(
                    "#settings-console-behavior-result",
                    self._console_behavior_result,
                )
                self.app.notify(self._console_behavior_result, severity="error")
                return
            console_values = {
                key: value
                for key, value in dirty_values.items()
                if key in CONSOLE_BEHAVIOR_CONSOLE_KEYS
            }
            workbench_scope_fallback = False
            background_effects_dirty = any(
                key.startswith("background_effects.") and key in CONSOLE_BACKGROUND_EFFECT_KEYS
                for key in dirty_values
            )
            raw_scope = self._raw_console_background_scope()
            if background_effects_dirty or str(raw_scope) == "workbench":
                merged_background_effects = self._loaded_console_background_effects()
                for key in CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER:
                    if key in dirty_values:
                        merged_background_effects[key.removeprefix("background_effects.")] = (
                            dirty_values[key]
                        )
                previous_scope = merged_background_effects.get("scope")
                available_scope = self._available_console_background_scope(previous_scope)
                workbench_scope_fallback = (
                    str(previous_scope) == "workbench" or str(raw_scope) == "workbench"
                )
                merged_background_effects["scope"] = available_scope
                console_values["background_effects"] = (
                    normalize_console_background_effects(
                        merged_background_effects
                    ).to_config()
                )
            chat_default_values = {
                key: value
                for key, value in dirty_values.items()
                if key in CONSOLE_BEHAVIOR_CHAT_DEFAULT_KEYS
            }
            self._console_behavior_result = "Console behavior settings saving..."
            self._set_static_text(
                "#settings-console-behavior-result",
                self._console_behavior_result,
            )
            self._settings_save_console_behavior_worker(
                dict(console_values),
                dict(chat_default_values),
                workbench_scope_fallback,
            )
            return

        self.app.notify("This Settings category has no save action yet.", severity="warning")

    def action_settings_revert_category(self, *, allow_text_entry_focus: bool = False) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        category = self._active_category_id()
        if not self._category_has_unsaved_changes(category):
            self.app.notify("No Settings changes to revert.", severity="information")
            return
        self._settings_drafts.pop(category, None)
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            self._console_behavior_result = "Console behavior settings reverted to last loaded values."
            self._sync_console_behavior_widgets()
        elif category is SettingsCategoryId.APPEARANCE:
            self._appearance_result = "Appearance defaults reverted to last loaded values."
            self._sync_appearance_widgets()
            self._update_draft_status_widgets(category)
        elif category is SettingsCategoryId.LIBRARY_RAG:
            self._library_rag_result = "Library/RAG defaults reverted to last loaded values."
            self._sync_library_rag_widgets()
            self._update_draft_status_widgets(category)
        elif category is SettingsCategoryId.STORAGE:
            self._storage_result = "Storage defaults reverted to last loaded values."
            self._sync_storage_widgets()
            self._update_draft_status_widgets(category)
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            values = self._provider_setting_values()
            try:
                provider = str(values["provider"])
                self._syncing_provider_selection = True
                try:
                    self.query_one("#settings-provider-value", Select).value = (
                        self._provider_select_value_for_provider(provider)
                    )
                finally:
                    self._syncing_provider_selection = False
                self._sync_provider_manual_widget(provider)
                self.query_one("#settings-model-value", Input).value = str(values["model"])
                endpoint_input = self.query_one("#settings-provider-endpoint-value", Input)
                endpoint_input.value = str(values["endpoint"])
                endpoint_input.placeholder = self._provider_endpoint_placeholder(provider)
                api_key_input = self.query_one("#settings-provider-api-key", Input)
                api_key_input.value = str(values.get("api_key") or "")
                api_key_input.placeholder = self._provider_api_key_placeholder(provider)
                credential_input = self.query_one(
                    "#settings-provider-credential-env-var",
                    Input,
                )
                credential_input.value = str(values["credential_env_var"])
                credential_input.placeholder = self._provider_credential_placeholder(provider)
                for draft_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS:
                    profile_value = values[draft_key]
                    self.query_one(
                        f"#settings-{draft_key.replace('_', '-')}",
                        Input,
                    ).value = self._profile_input_value(profile_value)
            except QueryError:
                pass
            self._provider_save_result = "Provider settings reverted to last loaded values."
            self._set_static_text("#settings-provider-save-result", self._provider_save_result)
            self._update_provider_dynamic_widgets()
            self._update_draft_status_widgets(category)
        else:
            self._update_draft_status_widgets(category)
        self.app.notify("Settings category changes reverted.", severity="information")

    def action_settings_test_category(self, *, allow_text_entry_focus: bool = False) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        if self._active_category_id() is SettingsCategoryId.PROVIDERS_MODELS:
            self._provider_test_result = self._run_provider_readiness_test()
            self._update_provider_test_result()
            self.app.notify("Provider test finished.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.DIAGNOSTICS:
            self._diagnostics_validation_result = "Config validation: running"
            self._diagnostics_reload_result = "Config reload: waiting for validation"
            self._set_static_text(
                "#settings-diagnostics-validation-result",
                self._diagnostics_validation_result,
            )
            self._set_static_text(
                "#settings-diagnostics-reload-result",
                self._diagnostics_reload_result,
            )
            self._diagnostics_validation_and_reload_worker()
            self.app.notify("Diagnostics validation and reload started.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.STORAGE:
            self._storage_check_rows = ("Storage check: running",)
            self._update_storage_check_widgets()
            self._storage_check_worker(self._storage_current_defaults())
            self.app.notify("Storage check started.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.PRIVACY_SECURITY:
            self._privacy_check_rows = ("Privacy check: running",)
            self._update_privacy_check_widgets()
            app_config = copy.deepcopy(getattr(self.app_instance, "app_config", {}))
            self._privacy_check_worker(app_config)
            self.app.notify("Privacy check started.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.APPEARANCE:
            validation = self._appearance_validation_result()
            if not validation.valid:
                self._appearance_result = validation.message
                self._set_static_text("#settings-appearance-save-result", self._appearance_result)
                self._update_appearance_validation_classes()
                self._update_draft_status_widgets(SettingsCategoryId.APPEARANCE)
                self.app.notify(validation.message, severity="error")
                return
            values = self._appearance_current_defaults()
            preview_applied = False
            try:
                setattr(self.app_instance, "theme", str(values.default_theme))
                preview_applied = True
            except Exception:
                preview_applied = False
            self._appearance_result = (
                "Appearance preview applied for this session only."
                if preview_applied
                else "Appearance preview unavailable in this runtime; Save persists defaults."
            )
            self._set_static_text("#settings-appearance-save-result", self._appearance_result)
            self.app.notify("Appearance preview complete.", severity="information")
            return
        self.app.notify("No test action is available for this Settings category yet.", severity="warning")

    @staticmethod
    def _save_console_behavior_values(
        console_values: Mapping[str, object],
        chat_default_values: Mapping[str, object],
    ) -> bool:
        section_values = {}
        if console_values:
            section_values["console"] = dict(console_values)
        if chat_default_values:
            section_values["chat_defaults"] = dict(chat_default_values)
        if not section_values:
            return True
        return SettingsConfigAdapter().save_sections(section_values)

    @staticmethod
    def _save_appearance_sections(section_values: Mapping[str, object]) -> bool:
        return SettingsConfigAdapter().save_sections(section_values)

    @staticmethod
    def _save_library_rag_sections(section_values: Mapping[str, object]) -> bool:
        return SettingsConfigAdapter().save_sections(section_values)

    @staticmethod
    def _save_storage_sections(section_values: Mapping[str, object]) -> bool:
        return SettingsConfigAdapter().save_sections(section_values)

    def _app_config_update_target(self):
        app_config = getattr(self.app_instance, "app_config", None)
        if callable(getattr(app_config, "update", None)):
            return app_config
        self.app_instance.app_config = {}
        return self.app_instance.app_config

    def _apply_appearance_save_result(
        self,
        saved: bool,
        section_values: Mapping[str, object],
    ) -> None:
        if saved:
            self._app_config_update_target().update(copy.deepcopy(dict(section_values)))
            self._settings_drafts.pop(SettingsCategoryId.APPEARANCE, None)
            self._appearance_result = "Appearance defaults saved."
            self._set_static_text("#settings-appearance-save-result", self._appearance_result)
            self._sync_appearance_widgets()
            self.app.notify("Appearance defaults saved.", severity="information")
            return
        self._appearance_result = "Failed to save Appearance defaults."
        self._set_static_text("#settings-appearance-save-result", self._appearance_result)
        self.app.notify(self._appearance_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_appearance_worker(self, section_values: Mapping[str, object]) -> None:
        saved = self._save_appearance_sections(section_values)
        self.app.call_from_thread(
            self._apply_appearance_save_result,
            saved,
            dict(section_values),
        )

    def _apply_library_rag_save_result(
        self,
        saved: bool,
        section_values: Mapping[str, object],
    ) -> None:
        if saved:
            self._app_config_update_target().update(copy.deepcopy(dict(section_values)))
            self._settings_drafts.pop(SettingsCategoryId.LIBRARY_RAG, None)
            self._library_rag_result = "Library/RAG defaults saved."
            self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            self.app.notify("Library/RAG defaults saved.", severity="information")
            return
        self._library_rag_result = "Failed to save Library/RAG defaults."
        self._set_static_text("#settings-library-rag-save-result", self._library_rag_result)
        self.app.notify(self._library_rag_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_library_rag_worker(self, section_values: Mapping[str, object]) -> None:
        saved = self._save_library_rag_sections(section_values)
        self.app.call_from_thread(
            self._apply_library_rag_save_result,
            saved,
            dict(section_values),
        )

    def _apply_storage_save_result(
        self,
        saved: bool,
        section_values: Mapping[str, object],
    ) -> None:
        if saved:
            self._app_config_update_target().update(copy.deepcopy(dict(section_values)))
            self._settings_drafts.pop(SettingsCategoryId.STORAGE, None)
            self._storage_result = "Storage defaults saved. Restart Chatbook to use saved paths."
            self._set_static_text("#settings-storage-save-result", self._storage_result)
            self._sync_storage_widgets()
            self.app.notify("Storage defaults saved.", severity="information")
            return
        self._storage_result = "Failed to save Storage defaults."
        self._set_static_text("#settings-storage-save-result", self._storage_result)
        self.app.notify(self._storage_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_storage_worker(self, section_values: Mapping[str, object]) -> None:
        saved = self._save_storage_sections(section_values)
        self.app.call_from_thread(
            self._apply_storage_save_result,
            saved,
            dict(section_values),
        )

    def _apply_console_behavior_save_result(
        self,
        saved: bool,
        console_values: Mapping[str, object],
        chat_default_values: Mapping[str, object],
        workbench_scope_fallback: bool = False,
    ) -> None:
        if saved:
            normalized_console_values = dict(console_values)
            if "background_effects" in normalized_console_values:
                self._console_settings()["background_effects"] = dict(
                    normalized_console_values["background_effects"]
                )
                normalized_console_values = {
                    key: value
                    for key, value in normalized_console_values.items()
                    if key != "background_effects"
                }
            self._console_settings().update(normalized_console_values)
            self._chat_defaults().update(chat_default_values)
            self._settings_drafts.pop(SettingsCategoryId.CONSOLE_BEHAVIOR, None)
            if workbench_scope_fallback:
                self._console_behavior_result = (
                    "Console behavior settings saved. "
                    f"{CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY}"
                )
            else:
                self._console_behavior_result = "Console behavior settings saved."
            self._console_behavior_saved_this_session = True
            self._sync_console_behavior_widgets()
            self.app.notify("Console behavior settings saved.", severity="information")
            return
        self._console_behavior_result = "Failed to save Console behavior settings."
        self._set_static_text(
            "#settings-console-behavior-result",
            self._console_behavior_result,
        )
        self.app.notify("Failed to save Console behavior settings.", severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_console_behavior_worker(
        self,
        console_values: Mapping[str, object],
        chat_default_values: Mapping[str, object],
        workbench_scope_fallback: bool = False,
    ) -> None:
        saved = self._save_console_behavior_values(console_values, chat_default_values)
        self.app.call_from_thread(
            self._apply_console_behavior_save_result,
            saved,
            dict(console_values),
            dict(chat_default_values),
            workbench_scope_fallback,
        )

    def _sync_console_behavior_widgets(self) -> None:
        try:
            self.query_one("#settings-console-collapse-large-pastes-toggle", Button).label = (
                self._collapse_large_pastes_button_label()
            )
        except QueryError:
            pass
        try:
            self._syncing_console_threshold = True
            try:
                self.query_one("#settings-console-paste-collapse-threshold", Input).value = str(
                    self._paste_collapse_threshold_value()
                )
            finally:
                self._syncing_console_threshold = False
        except QueryError:
            pass
        input_values = {
            "#settings-console-default-streaming": self._console_behavior_value("streaming"),
            "#settings-console-default-temperature": self._console_behavior_value("temperature"),
            "#settings-console-default-top-p": self._console_behavior_value("top_p"),
            "#settings-console-default-min-p": self._console_behavior_value("min_p"),
            "#settings-console-default-top-k": self._console_behavior_value("top_k"),
            "#settings-console-default-max-tokens": self._console_behavior_value("max_tokens"),
            "#settings-console-default-seed": self._console_behavior_value("seed"),
            "#settings-console-default-presence-penalty": self._console_behavior_value(
                "presence_penalty"
            ),
            "#settings-console-default-frequency-penalty": self._console_behavior_value(
                "frequency_penalty"
            ),
            "#settings-console-default-reasoning-effort": self._console_behavior_value(
                "reasoning_effort"
            ),
            "#settings-console-default-reasoning-summary": self._console_behavior_value(
                "reasoning_summary"
            ),
            "#settings-console-default-verbosity": self._console_behavior_value("verbosity"),
            "#settings-console-default-thinking-effort": self._console_behavior_value(
                "thinking_effort"
            ),
            "#settings-console-default-thinking-budget-tokens": self._console_behavior_value(
                "thinking_budget_tokens"
            ),
        }
        self._syncing_console_defaults = True
        try:
            for selector, value in input_values.items():
                try:
                    self.query_one(selector, Input).value = self._console_input_value(value)
                except QueryError:
                    pass
        finally:
            self._syncing_console_defaults = False
        self._syncing_console_background_effects = True
        try:
            try:
                self.query_one("#settings-console-background-effect-enabled", Button).label = (
                    self._console_background_effect_enabled_label()
                )
            except QueryError:
                pass
            select_values = {
                "#settings-console-background-effect-type": self._console_background_effect_value("effect"),
                "#settings-console-background-effect-scope": self._console_background_effect_value("scope"),
                "#settings-console-background-effect-intensity": self._console_background_effect_value("intensity"),
            }
            for selector, value in select_values.items():
                try:
                    self.query_one(selector, Select).value = str(value)
                except QueryError:
                    pass
            try:
                self.query_one("#settings-console-background-effect-fps", Input).value = str(
                    self._console_background_effect_value("fps")
                    or DEFAULT_CONSOLE_BACKGROUND_FPS
                )
            except QueryError:
                pass
        finally:
            self._syncing_console_background_effects = False
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result_text())
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    def _sync_appearance_widgets(self) -> None:
        values = self._appearance_setting_values()
        self._syncing_appearance_defaults = True
        try:
            try:
                self.query_one("#settings-appearance-theme", Select).value = str(
                    values["default_theme"]
                )
            except QueryError:
                pass
            try:
                self.query_one("#settings-appearance-palette-theme-limit", Input).value = str(
                    values["palette_theme_limit"]
                )
            except QueryError:
                pass
            try:
                self.query_one("#settings-appearance-font-size", Input).value = str(
                    values["font_size"]
                )
            except QueryError:
                pass
            try:
                self.query_one("#settings-appearance-density", Select).value = str(
                    values["density"]
                )
            except QueryError:
                pass
            try:
                self.query_one("#settings-appearance-animations-enabled", Button).label = (
                    self._appearance_bool_label("animations_enabled")
                )
            except QueryError:
                pass
            try:
                self.query_one("#settings-appearance-smooth-scrolling", Button).label = (
                    self._appearance_bool_label("smooth_scrolling")
                )
            except QueryError:
                pass
        finally:
            self._syncing_appearance_defaults = False
        self._set_static_text("#settings-appearance-save-result", self._appearance_result)
        self._update_appearance_validation_classes()
        self._update_draft_status_widgets(SettingsCategoryId.APPEARANCE)

    def on_key(self, event: Key) -> None:
        focused = self._focused_widget()
        if (
            event.key in {"/", "slash"}
            or getattr(event, "character", None) == "/"
        ) and not isinstance(focused, (Input, TextArea)):
            self._focus_category_search()
            event.stop()
            event.prevent_default()
            return
        if event.key == "escape" and self.category_search_query:
            if self._category_search_has_focus() or not isinstance(focused, (Input, TextArea)):
                self.category_search_query = ""
                try:
                    self.query_one("#settings-category-search", Input).value = ""
                except QueryError:
                    pass
                self._apply_category_search_filter()
                self._focus_category_search()
                event.stop()
                event.prevent_default()
                return
        if event.key == "tab":
            if focused is None or getattr(focused, "has_class", lambda *_: False)("nav-button"):
                self._focus_category(SettingsCategoryId.OVERVIEW.value)
                event.stop()
                event.prevent_default()
            return
        if event.key in {"down", "j"} and self._focused_category_value() is not None:
            self._move_category_focus(1)
            event.stop()
            event.prevent_default()
            return
        if event.key in {"up", "k"} and self._focused_category_value() is not None:
            self._move_category_focus(-1)
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            if isinstance(focused, Button) and focused.id in {
                "settings-console-collapse-large-pastes-toggle",
                "settings-test-provider",
                "settings-check-storage",
                "settings-validate-config",
                "settings-reload-config",
                "settings-advanced-validate-config",
                "settings-advanced-save-config",
            }:
                focused.press()
                event.stop()
                event.prevent_default()
                return
            category_value = self._focused_category_value()
            if category_value is not None:
                self._select_category(category_value, restore_focus=True)
                event.stop()
                event.prevent_default()
