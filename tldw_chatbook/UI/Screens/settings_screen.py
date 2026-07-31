"""Settings destination shell for global app preferences."""

import asyncio
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import logging
import os
from pathlib import Path
import re
import tomllib

from rich.cells import cell_len
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.events import DescendantFocus, Key
from textual.message_pump import NoActiveAppError
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.suggester import SuggestFromList
from textual.validation import ValidationResult, Validator
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Rule,
    Select,
    SelectionList,
    Static,
    TextArea,
)

from ...Chat.console_chat_models import CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
from ...Chat.console_provider_endpoints import URL_BASED_PROVIDER_KEYS
from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Chat.console_provider_support import (
    ConsoleProviderCatalogEntry,
    supported_console_provider_catalog,
)
from ...Chat.console_session_settings import CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
from ...ACP_Interop.runtime_session import ACPRuntimeSessionState
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...Sync_Interop.sync_promotion_state import (
    SyncPromotionState,
    build_sync_promotion_state,
)
from ...Sync_Interop.sync_readiness import (
    DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
    build_sync_readiness_report,
)
from ...Sync_Interop.manual_sync_control import ManualSyncPreview, ManualSyncRunResult
from ...Workspaces.display_state import LIBRARY_WORKSPACE_VISIBILITY_COPY
from ...Workspaces.models import RuntimeBindingStatus
from ...Workspaces.registry_service import (
    DEFAULT_WORKSPACE_ID,
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
    next_local_workspace_identity,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.destination_workbench import DestinationModeStrip
from ...Chat.provider_catalog import (
    PROVIDER_CUSTOM_GROUP_KEYS,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_GROUP_CLOUD,
    PROVIDER_GROUP_CUSTOM,
    PROVIDER_GROUP_LOCAL,
    PROVIDER_GROUP_ORDER,
)
from ...config import (
    DEFAULT_CONFIG_FROM_TOML,
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
    _default_base_data_dir,
    coerce_bool_setting,
    coerce_int_setting,
    get_cli_config_path,
    load_settings,
    save_settings_to_cli_config,
)
from ...LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
    load_model_catalog_settings,
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
from ...UI.Workbench import WorkbenchHelpPanel, WorkbenchHelpState
from .provider_model_resolution import (
    EffectiveProviderModel,
    resolve_effective_provider_model,
)
from .settings_config_adapter import SettingsConfigAdapter, redact_secret_text
from .settings_endpoint_probe import probe_settings_endpoint
from .settings_config_models import (
    SettingsCategoryId,
    SettingsCategorySummary,
    SettingsDomainCategoryContract,
    SettingsDraft,
    SettingsOwnershipRecord,
)
from ...Widgets.settings_splash_screen_viewer import SettingsSplashScreenViewer
from ...Widgets.settings_theme_editor import SettingsThemeEditor
from ...Widgets.settings_internal_prompts_panel import InternalPromptsPanel
from ...Widgets.settings_image_gen_panel import (
    ImageGenSettingsPanel,
    _key_source_line as _image_gen_key_source_line,
    _secret_placeholder as _image_gen_secret_placeholder,
)
from ...Internal_Prompts import authoring as internal_prompts_authoring
from .settings_image_gen_defaults import (
    BACKEND_IDS as IMAGE_GEN_BACKEND_IDS,
    FIELD_SCHEMA as IMAGE_GEN_FIELD_SCHEMA,
    ImageGenDraftValues,
    canonical_backend_order as image_gen_canonical_backend_order,
    diff_to_sections as image_gen_diff_to_sections,
    effective_placeholder as image_gen_effective_placeholder,
    effective_secret_value as image_gen_effective_secret_value,
    key_source_after_clear as image_gen_key_source_after_clear,
    probe_backend as image_gen_probe_backend,
    validate_draft as validate_image_gen_draft,
)
from ...Image_Generation.config import (
    get_image_generation_config,
    reset_image_generation_config_cache,
)
from .settings_appearance_defaults import (
    SettingsAppearanceDefaults,
    build_appearance_save_sections,
    load_appearance_defaults,
    validate_appearance_defaults,
)
from .settings_library_rag_defaults import (
    SettingsLibraryRagDefaults,
    normalise_library_rag_chunking_method,
    normalise_library_rag_citation_style,
    normalise_library_rag_distance_metric,
    normalise_library_rag_search_mode,
    validate_library_rag_defaults,
)
from .settings_rag_profile_adapter import (
    activate_profile,
    active_profile_info,
    clone_profile_as,
    delete_user_profile,
    fetch_index_status,
    get_profile_defaults,
    index_change_pending,
    is_first_run_state,
    list_profiles_grouped,
    load_rag_defaults_from_active_profile,
    rename_user_profile,
    save_rag_defaults_to_active_profile,
    soft_config_warnings,
)
from ...RAG_Search.ingestion_indexing import (
    backfill_semantic_index,
    get_shared_rag_service,
    semantic_indexing_available,
)
from .settings_privacy_security import (
    SettingsPrivacyPosture,
    build_privacy_posture_rows,
    build_settings_privacy_posture,
    env_var_summary,
    skill_trust_display,
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
REASONING_EFFORT_OPTIONS = frozenset(
    {"", "none", "minimal", "low", "medium", "high", "xhigh"}
)
REASONING_SUMMARY_OPTIONS = frozenset({"", "auto", "concise", "detailed", "none"})
VERBOSITY_OPTIONS = frozenset({"", "low", "medium", "high"})
THINKING_EFFORT_OPTIONS = frozenset(
    {"", "off", "low", "medium", "high", "xhigh", "max"}
)
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
# task-180/191: provider display names + grouping now come from the shared
# catalog module (imported at the top) so Settings and Console match.

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
# ADR-020: ids of the [model_catalog] auto-refresh toggles so unrelated
# Checkbox.Changed events never trigger a config write.
MODEL_CATALOG_CHECKBOX_IDS = frozenset(
    {"settings-model-catalog-auto-refresh"}
    | {
        f"settings-mc-auto-{provider.lower()}"
        for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
    }
    | {
        f"settings-mc-write-{provider.lower()}"
        for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
    }
)
CONSOLE_BEHAVIOR_CONSOLE_KEYS = frozenset(
    {
        "collapse_large_pastes",
        "paste_collapse_threshold",
        "max_parallel_runs",
        "tool_result_display_chars",
    }
)
# Parallel-agents spec S4 (task-5): user-adjustable global cap on
# simultaneous Console runs. Aliases console_chat_models' single source of
# truth so the settings UI and ConsoleChatController.max_parallel_runs
# (which reads [console] max_parallel_runs via get_cli_setting with the
# same default and floor) can never drift apart. No upper bound is
# deliberate (user-owned trade-off).
DEFAULT_CONSOLE_MAX_PARALLEL_RUNS = CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
MIN_CONSOLE_MAX_PARALLEL_RUNS = 1
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
# Task 4 (SP3): shared honest re-index warning, appended to BOTH triggers that
# can re-point the active profile at a fresh (empty) fingerprinted vector
# collection -- saving an index-determining field (settings_rag_profile_adapter
# .index_change_pending) and switching the active profile itself
# (handle_library_rag_profile_set_active / _rag_after_set_active). See
# Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.
RAG_INDEX_CHANGE_WARNING = (
    "This change re-points to a new (empty) index — run Backfill."
)
RAG_INDEX_ABSENT_STATUS_TEXT = "Index: absent — will be created on next backfill"
RAG_INDEX_STATUS_CHECKING_TEXT = "Index: checking…"
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
    "max_parallel_runs",
    "tool_result_display_chars",
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
    # task-1565: labels mirror the sidebar's category names exactly so the
    # guided chips and the rail never disagree about what a place is called.
    (SettingsCategoryId.PROVIDERS_MODELS, "Providers & Models"),
    (SettingsCategoryId.CONSOLE_BEHAVIOR, "Console Behavior"),
    (SettingsCategoryId.STORAGE, "Storage"),
    (SettingsCategoryId.PRIVACY_SECURITY, "Privacy & Security"),
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
    # Keys mirror the source values resolve_effective_provider_model can
    # return (Provider/provider_model_resolution.py) -- task-648 renamed
    # console_control to console_session and deleted app_reactive; TASK-1310's
    # review caught the stale keys here rendering a raw "console session"
    # fallback label in Settings > Providers.
    "settings_draft": "Unsaved Settings draft",
    "console_session": "Console runtime override",
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
    "llama_cpp": "http://127.0.0.1:9099",
    "local_llamacpp": "http://127.0.0.1:9099",
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
# THEME and SPLASH_SCREEN are intentionally excluded; they manage their own
# persistence models (theme files and immediate splash config writes).
GUIDED_SETTINGS_MUTATION_CATEGORIES = frozenset(
    {
        SettingsCategoryId.PROVIDERS_MODELS,
        SettingsCategoryId.APPEARANCE,
        SettingsCategoryId.CONSOLE_BEHAVIOR,
        SettingsCategoryId.LIBRARY_RAG,
        SettingsCategoryId.STORAGE,
    }
)
# task-181: keep these rows in user language; they render on the Overview
# card, so avoid internal architecture/ownership phrasing.
SETTINGS_OVERVIEW_BOUNDARY_ROWS = (
    ("Settings", "edits saved defaults; changes apply after you save"),
    ("Console", "live chat and run controls stay on the Console screen"),
    ("MCP", "tool servers are managed on the MCP screen"),
    ("ACP", "agent runtime and sessions are managed on the ACP screen"),
    ("Sync & workspaces", "status shown here is read-only"),
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
        title="RAG",
        owner_destination="Library",
        source_of_truth=(
            "Library source services",
            "RAG_Search retrieval adapters",
            "Library Collections local service",
        ),
        rows=(
            (
                "Browse/search visibility",
                "global Library browse/search remains visible across workspaces",
            ),
            (
                "Console eligibility",
                "staging source evidence is limited to the active workspace",
            ),
            (
                "Retrieval defaults",
                "the active RAG profile (rag_profiles/<id>.json) owns result limits and blend defaults",
            ),
            (
                "Citation/snippet defaults",
                "the active RAG profile (rag_profiles/<id>.json) owns citations, snippets, and context budget defaults",
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
        source_of_truth=(
            "Chatbook artifact store",
            "Artifacts destination display state",
        ),
        rows=(
            (
                "Chatbooks",
                "Artifacts owns Chatbook browse, details, and Console resume actions",
            ),
            (
                "Settings role",
                "show defaults/status only; do not move artifact operations here",
            ),
        ),
        follow_up="add artifact export/default controls only after Artifacts exposes a persisted preference contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.PERSONAS,
        title="Roleplay",
        owner_destination="Roleplay",
        source_of_truth=(
            "Your saved characters and user profiles",
            "Whatever's currently open in Roleplay",
        ),
        rows=(
            (
                "What Roleplay controls",
                "Picking a character or user profile, and sending it to Console",
            ),
            (
                "What Settings might add later",
                "Browsing or display preferences - never which user profile is active",
            ),
        ),
        follow_up="Add user profile display/browsing preferences once Roleplay can hand Settings a saved preference to edit.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.SKILLS,
        title="Skills",
        owner_destination="Skills",
        source_of_truth=(
            "Skills repository",
            "Skills destination validation and attach paths",
        ),
        rows=(
            (
                "Skill format",
                "Skills owns SKILL.md import, validation, and attach behavior",
            ),
            (
                "Settings role",
                "future defaults can cover trust/display preferences only",
            ),
        ),
        follow_up="add Skills defaults after import/attach policy has a persisted source contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.SCHEDULES,
        title="Schedules",
        owner_destination="Schedules",
        source_of_truth=("Schedules destination state", "schedule run handoff context"),
        rows=(
            (
                "Run control",
                "Schedules owns run, pause, retry, and Console handoff actions",
            ),
            (
                "Settings role",
                "future defaults may cover timezone/notification preferences only",
            ),
        ),
        follow_up="add schedule defaults after Schedules exposes a dedicated settings adapter.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.WATCHLISTS,
        title="Watchlists",
        owner_destination="Watchlists",
        source_of_truth=("Watchlists local service", "watchlist run snapshot adapter"),
        rows=(
            ("Monitoring", "Watchlists owns feeds, runs, status, and recovery actions"),
            (
                "Settings role",
                "future defaults may cover polling and notification preferences only",
            ),
        ),
        follow_up="add watchlist defaults after Watchlists exposes persisted polling/notification settings.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.WORKFLOWS,
        title="Workflows",
        owner_destination="Workflows",
        source_of_truth=(
            "Workflows destination procedure state",
            "workflow Console handoff payloads",
        ),
        rows=(
            (
                "Execution",
                "Workflows owns procedure inputs, dry runs, approvals, and outputs",
            ),
            (
                "Settings role",
                "future defaults may cover execution safety preferences only",
            ),
        ),
        follow_up="add workflow defaults after Workflows exposes a persisted execution-safety contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.MCP_DEFAULTS,
        title="MCP Defaults",
        owner_destination="MCP",
        source_of_truth=("Unified MCP panel", "MCP configured server target store"),
        rows=(
            (
                "Runtime owner",
                "MCP owns server/tool runtime, target management, and tool readiness",
            ),
            (
                "Settings role",
                "show global defaults/status only; server operations stay in MCP",
            ),
        ),
        follow_up="add MCP defaults only after server-first settings are exposed without flattening tools into Settings.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.ACP_DEFAULTS,
        title="ACP Defaults",
        owner_destination="ACP",
        source_of_truth=(
            "ACP runtime session state",
            "ACP destination launch/session setup",
        ),
        rows=(
            (
                "Runtime owner",
                "ACP owns runtime launch, session setup, and task/run packages",
            ),
            ("Settings role", "show defaults/status only; ACP setup stays in ACP"),
        ),
        follow_up="add ACP defaults after ACP exposes a persisted runtime/session preference contract.",
    ),
    SettingsDomainCategoryContract(
        category=SettingsCategoryId.IMAGE_GENERATION,
        title="Image Gen",
        owner_destination="Console",
        source_of_truth=(
            "config.toml [image_generation] (+ nested backend sections)",
            "Image_Generation engine (adapters/worker)",
        ),
        rows=(
            (
                "Generation actions",
                "Console owns /generate-image, cards, variants",
            ),
            (
                "Settings role",
                "Settings edits persisted backend/config defaults only",
            ),
            (
                "Style templates",
                "config/templates dir own definitions; management UI planned",
            ),
        ),
        settings_can_mutate=True,
        follow_up=(
            "Follow-up: add a dedicated style-template create/edit/delete UI (v2); "
            "v1 shows a read-only count only."
        ),
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

# Task 3 (541 v2 UX AC3): RAG widget id -> guidance-group key. Mirrors the
# ids `_library_rag_field_selector` and the LIBRARY_RAG compose branch mint
# (search around "settings-library-rag-" in this file). Used by
# `_rag_field_guidance_rows()` so the Scope Inspector follows the focused
# field; falls back to `_active_rag_scope_group` (the last-expanded
# Collapsible) when the focused widget isn't one of these.
_RAG_FIELD_GROUP_BY_ID: dict[str, str] = {
    "settings-library-rag-search-mode": "search",
    "settings-library-rag-default-top-k": "search",
    "settings-library-rag-fts-top-k": "search",
    "settings-library-rag-vector-top-k": "search",
    "settings-library-rag-hybrid-alpha": "search",
    "settings-library-rag-score-threshold": "search",
    "settings-library-rag-include-citations": "search",
    "settings-library-rag-citation-style": "search",
    "settings-library-rag-snippet-max-chars": "search",
    "settings-library-rag-max-context-size": "search",
    "settings-library-rag-embedding-model": "embedding",
    "settings-library-rag-embedding-device": "embedding",
    "settings-library-rag-embedding-batch-size": "embedding",
    "settings-library-rag-embedding-max-length": "embedding",
    "settings-library-rag-chunk-size": "chunking",
    "settings-library-rag-chunk-overlap": "chunking",
    "settings-library-rag-chunking-method": "chunking",
    "settings-library-rag-distance-metric": "vector_store",
    "settings-library-rag-enable-reranking": "reranking",
    "settings-library-rag-reranker-model": "reranking",
    "settings-library-rag-reranker-top-k": "reranking",
    "settings-library-rag-profile-select": "profile",
    "settings-library-rag-profile-set-active": "profile",
    "settings-library-rag-profile-clone": "profile",
    "settings-library-rag-profile-rename": "profile",
    "settings-library-rag-profile-delete": "profile",
    "settings-library-rag-index-backfill": "index",
}

# Task 4 (541 v2 UX AC1): the Library/RAG editor field keys whose disabled
# state is driven PURELY by a read-only lock (builtin/active read_only, or a
# profile-picker PREVIEW) -- reranker_model/reranker_top_k are deliberately
# excluded here (their disabled state ALSO depends on whether reranking is
# enabled; see `_apply_library_rag_rerank_field_state`). Shared by
# `_sync_library_rag_profile_widgets` and `_sync_library_rag_widgets`'s
# `field_disabled` override so the two never drift out of sync with each
# other.
_LIBRARY_RAG_READ_LOCK_FIELD_KEYS: tuple[str, ...] = (
    "default_search_mode",
    "default_top_k",
    "fts_top_k",
    "vector_top_k",
    "hybrid_alpha",
    "score_threshold",
    "citation_style",
    "snippet_max_chars",
    "max_context_size",
    "embedding_model",
    "embedding_device",
    "embedding_batch_size",
    "embedding_max_length",
    "chunk_size",
    "chunk_overlap",
    "chunking_method",
    "distance_metric",
)
# The two Checkbox fields (Task 1) alongside the read-lock field keys above --
# also read-only-lock driven only, never rerank-enabled driven.
_LIBRARY_RAG_READ_LOCK_CHECKBOX_SELECTORS: tuple[str, ...] = (
    "#settings-library-rag-include-citations",
    "#settings-library-rag-enable-reranking",
)

# Collapsible id -> the same group keys. `@on(Collapsible.Toggled)` uses this
# so expanding a group (e.g. "Chunking") already switches the inspector's
# context even before any field inside it is focused. Textual 8.2.7: focus
# on Tab lands on the inner CollapsibleTitle, not the Collapsible itself, and
# `.collapsible--header` is a dead CSS class -- neither is used here; this
# keys off `event.collapsible.id`/`.collapsed` instead (see
# handle_settings_library_rag_collapsible_toggled).
_RAG_GROUP_BY_COLLAPSIBLE_ID: dict[str, str] = {
    "settings-library-rag-search-group": "search",
    "settings-library-rag-embedding-group": "embedding",
    "settings-library-rag-chunking-group": "chunking",
    "settings-library-rag-vector-store-group": "vector_store",
    "settings-library-rag-reranking-group": "reranking",
}

# Fleet-UX expert review F6 (task-1234): every guided category's "Focused
# field guide" rows share the SAME symptom task-1140 already fixed once for
# the fleet line -- the content updates in place (`_refresh_*_field_
# guidance`, no recompose) but focus never moves the Scope Inspector's
# scroll position, so the guide can sit below the pane's fold with only a
# thin scrollbar sliver hinting at it (worst case: only "Purpose:" visible,
# Consequences/Saved-as/Validation all clipped). Maps each guided category
# to its guide block's FIRST row id -- every id below already exists (see
# `_render_impact_pane`'s per-category "Focused field guide" loops) and is
# rendered unconditionally (the fallback rows use the same ids when no
# specific field is focused), so scrolling to it is meaningful even when
# `_active_settings_field_id` resolves to the generic fallback.
_FIELD_GUIDE_FIRST_ROW_ID: dict[SettingsCategoryId, str] = {
    SettingsCategoryId.APPEARANCE: "settings-appearance-field-guide-0",
    SettingsCategoryId.STORAGE: "settings-storage-field-guide-0",
    SettingsCategoryId.LIBRARY_RAG: "settings-library-rag-field-guide-0",
    SettingsCategoryId.CONSOLE_BEHAVIOR: "settings-console-behavior-field-guide-0",
    SettingsCategoryId.PROVIDERS_MODELS: "settings-provider-field-guide-0",
}

# One concise, fixed-length (5-row) entry per group (Task 3, 541 AC3).
# Fixed-length matters: `_refresh_rag_field_guidance` updates existing rows
# in place by index (`_set_static_text`, no recompose) the same way
# `_refresh_provider_field_guidance` does for Providers -- a variable row
# count would leave stale rows behind when switching groups. The "index-
# determining" facts (embedding model, embedding max length, every chunking
# field, distance metric) mirror
# RAG_Search/simplified/collection_fingerprint.py's `_index_fields()` --
# exactly what `index_change_pending()` (settings_rag_profile_adapter.py)
# fingerprints. Strings kept within the rail width (SP3 fit lesson, UX
# review item 9) -- no mid-sentence clipping at the QA viewport.
#
# Fallback when no RAG field is focused and no group has ever been expanded
# this session -- UNCHANGED from the terse rows the UX review (item 9)
# shortened this to; regression-locked by
# test_settings_library_rag_inspector_uses_shortened_terse_guidance. Defined
# ahead of `_RAG_GROUP_GUIDANCE` because the "search" entry below reuses it
# verbatim: Textual's Collapsible posts its own `Expanded` message from
# inside `__init__` whenever constructed with `collapsed=False` (the
# reactive's default is `True`, so the explicit `False` is a real change,
# queued and delivered once the widget starts running) -- which is exactly
# how the "Search" group (collapsed=False, expanded-by-default) composes.
# That message reaches `handle_settings_library_rag_collapsible_toggled`
# before the first `_render_impact_pane` pass finishes, so
# `_active_rag_scope_group` is ALREADY "search" at first paint, not None.
# Reusing the fallback tuple for "search" makes that a no-op: first paint
# reads identically whether the resolved group is None or "search".
_RAG_GROUP_GUIDANCE_FALLBACK: tuple[tuple[str, str], ...] = (
    ("Search mode", "plain=keyword, semantic=embeddings, hybrid=blend"),
    ("Result limits", "bounds default/keyword/vector result counts"),
    ("Hybrid balance", "0.0=keyword, 1.0=semantic"),
    ("Citations", "adds source markers to answers when supported"),
    ("Snippet/context", "snippet length + context budget for retrieved text"),
)
_RAG_GROUP_GUIDANCE: dict[str, tuple[tuple[str, str], ...]] = {
    "search": _RAG_GROUP_GUIDANCE_FALLBACK,
    "embedding": (
        ("Focused group", "Embedding"),
        ("Fields", "model, device, batch size, max length"),
        ("Purpose", "what the vector index is built from"),
        ("Impact", "⚠ model + max length rebuild the index -- Backfill after"),
        ("Saved as", "the profile's embedding settings"),
    ),
    "chunking": (
        ("Focused group", "Chunking"),
        ("Fields", "chunk size, overlap, method"),
        ("Purpose", "how source text is split before embedding"),
        ("Impact", "⚠ every field here rebuilds the index -- Backfill after"),
        ("Saved as", "the profile's chunking settings"),
    ),
    "vector_store": (
        ("Focused group", "Vector store"),
        ("Fields", "distance metric"),
        ("Purpose", "how embeddings are compared during retrieval"),
        ("Impact", "⚠ rebuilds the index -- Backfill after saving"),
        ("Saved as", "the profile's vector store settings"),
    ),
    "reranking": (
        ("Focused group", "Reranking"),
        ("Fields", "enable, reranker model, rerank results"),
        ("Purpose", "optional post-retrieval reordering of results"),
        ("Impact", "no index rebuild -- toggling adds/removes config"),
        ("Saved as", "the profile's reranking settings"),
    ),
    "profile": (
        ("Focused group", "Profiles"),
        ("Fields", "select, set active, clone, rename, delete"),
        ("Purpose", "switch which profile these fields edit"),
        ("Impact", "Set active is immediate; built-ins are read-only"),
        ("Saved as", "the [rag.service].profile pointer"),
    ),
    "index": (
        ("Focused group", "Index"),
        ("Fields", "Backfill"),
        ("Purpose", "rebuild the active profile's vector index"),
        ("Impact", "⚠ run after saving any warning field; safe to re-run"),
        ("Saved as", "not a config field -- runs Library ingestion"),
    ),
}

# Impact-pane guidance rows keyed by non-domain Settings category. Domain
# categories (DOMAIN_SETTINGS_CATEGORY_IDS) derive their guidance from their
# ownership contract instead and are intentionally absent here. Every other
# SettingsCategoryId MUST have an entry: this table is read inside compose, so
# a missing key would otherwise take down the whole app (see PR #713 / #742).
_INSPECTOR_GUIDANCE: dict[SettingsCategoryId, tuple[tuple[str, str], ...]] = {
    SettingsCategoryId.OVERVIEW: (
        ("Affected config", "all Settings categories summarized for readiness"),
        ("Recovery", "open the specific category before changing values"),
        (
            "Boundary",
            "runtime MCP, ACP, and tool control stay in their own destinations",
        ),
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
            "open Theme for full theme editing; use Settings for persisted defaults",
        ),
        ("Boundary", "visual preferences do not change runtime or data access"),
    ),
    SettingsCategoryId.STORAGE: (
        (
            "Affected config",
            "config file path, local database paths, media storage roots",
        ),
        (
            "Recovery",
            "verify paths, reload config, then restart only if storage roots changed",
        ),
        (
            "Boundary",
            "server handoff does not move local source content unless explicitly requested",
        ),
    ),
    SettingsCategoryId.WORKSPACES: (
        (
            "Affected config",
            "workspace lifecycle records and their bound folders",
        ),
        (
            "Recovery",
            "switch/rename/archive in Console (Alt+W); create a workspace in Library",
        ),
        (
            "Boundary",
            "lifecycle and folder bindings apply immediately; there is no draft state here",
        ),
    ),
    SettingsCategoryId.PRIVACY_SECURITY: (
        (
            "Affected config",
            "encryption posture, credential-source status, and redaction status",
        ),
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
        (
            "Affected config",
            "chat_defaults fallbacks plus Console composer paste behavior",
        ),
        (
            "Recovery",
            "revert unsaved changes or disable paste collapse if composer flow is disrupted",
        ),
        (
            "Boundary",
            "active sessions and provider+model profiles override these global fallbacks",
        ),
    ),
    SettingsCategoryId.LIBRARY_RAG: (
        (
            "Affected config",
            "the active RAG profile (rag_profiles/<id>.json) and the [rag.service].profile pointer",
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
        (
            "Affected config",
            "read-only validation, reload status, and troubleshooting output",
        ),
        (
            "Recovery",
            "validate first, reload only after confirming the config source is correct",
        ),
        (
            "Boundary",
            "diagnostics redact secrets and should not mutate advanced config",
        ),
    ),
    SettingsCategoryId.ADVANCED_CONFIG: (
        ("Affected config", "raw TOML for every loaded configuration section"),
        (
            "Recovery",
            "validate current text, save atomically, then restore from backup if needed",
        ),
        ("Boundary", "save is blocked until the exact current text validates"),
    ),
    SettingsCategoryId.THEME: (
        (
            "Affected config",
            "custom theme files under ~/.config/tldw_cli/themes/",
        ),
        (
            "Recovery",
            "use the editor's Apply/Save/Reset buttons; delete a theme file to remove it",
        ),
        (
            "Boundary",
            "launch visual defaults stay in Appearance; theme edits never touch config.toml",
        ),
    ),
    SettingsCategoryId.SPLASH_SCREEN: (
        (
            "Affected config",
            "splash_screen section defaults and card selection",
        ),
        (
            "Recovery",
            "reset defaults from this category or edit splash_screen values in Advanced Config",
        ),
        ("Boundary", "changes are saved immediately; no shared Settings draft state"),
    ),
    SettingsCategoryId.INTERNAL_PROMPTS: (
        (
            "Affected config",
            "config.toml [internal_prompts] overrides for built-in system prompts",
        ),
        (
            "Recovery",
            "use each prompt's own Save/Reset buttons to restore the packaged default",
        ),
        (
            "Boundary",
            "edits apply to internal tooling prompts only; no shared Settings draft state",
        ),
    ),
    # NOTE: unreachable via _inspector_guidance() -- IMAGE_GENERATION has
    # its own explicit branch there (checked before this dict), same as
    # LIBRARY_RAG's identical pre-existing shadowing. Kept accurate anyway
    # as a safety net for the dict-lookup fallback path.
    SettingsCategoryId.IMAGE_GENERATION: (
        (
            "Affected config",
            "[image_generation] backend enable/default, per-backend fields, and "
            "generation defaults",
        ),
        (
            "Recovery",
            "Revert discards unsaved edits; Console's /generate-image keeps "
            "working off the last saved config.toml regardless",
        ),
        (
            "Boundary",
            "Edits backend, key, and generation defaults here; Save applies "
            "to config.toml",
        ),
    ),
}
# Generic guidance for a category with no explicit entry. Kept as a runtime
# safety net only; test_inspector_guidance_covers_every_settings_category fails
# CI before an uncovered category can reach a user.
_INSPECTOR_GUIDANCE_FALLBACK: tuple[tuple[str, str], ...] = (
    ("Affected config", "this category manages its own settings"),
    ("Recovery", "use the controls in the category detail pane"),
    ("Boundary", "no shared Settings draft state is affected"),
)
# Categories already warned about, so the fallback logs once per run per
# category instead of on every compose pass.
_WARNED_MISSING_GUIDANCE_CATEGORIES: set[SettingsCategoryId] = set()


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


class ProviderEndpointURLValidator(Validator):
    """TASK-367: inline endpoint URL validation, run on blur/submit.

    An empty endpoint is allowed (not every provider needs one); a non-empty
    value must be a well-formed http/https URL. Surfacing this on blur flags a
    malformed URL (e.g. a dropped scheme character) at the field itself instead
    of only when the user later saves or runs Discover.
    """

    def validate(self, value: str) -> ValidationResult:
        """Validate a provider endpoint URL for the field's blur check.

        Args:
            value: The endpoint text currently in the field.

        Returns:
            ``success()`` for an empty value (endpoint optional) or a
            well-formed http/https URL; otherwise ``failure()`` with a
            corrective message.
        """
        text = str(value or "").strip()
        if not text or validate_url(text):
            return self.success()
        return self.failure(
            "Enter a full http:// or https:// URL, e.g. http://127.0.0.1:9099/v1."
        )


#: task-1583: widest token the ~26-34 cell Scope Inspector column shows
#: without folding mid-word.
_FOLD_TOKEN_LIMIT = 26


def _fold_long_tokens(text: str, limit: int = _FOLD_TOKEN_LIMIT) -> str:
    """Break over-long dotted keys and slashed paths at their separators.

    Rich wraps at spaces and folds longer tokens mid-word, so the narrow
    Scope Inspector rendered "crede/ntial_source" and "config.tom/l"
    (critique rescore P2). Tokens beyond ``limit`` that contain ``.`` or
    ``/`` separators gain newline break points after separators instead,
    continuation-indented; tokens without separators pass through.

    Args:
        text: The detail-row value, possibly multi-token or multi-line.
        limit: Maximum kept token length before folding.

    Returns:
        The text with pathological tokens folded at separator boundaries.
    """

    def fold_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) <= limit or not re.search(r"[./]", token):
            return token
        segments = re.split(r"(?<=[./])", token)
        lines: list[str] = []
        current = ""
        for segment in segments:
            if current and len(current) + len(segment) > limit:
                lines.append(current)
                current = segment
            else:
                current += segment
        if current:
            lines.append(current)
        return "\n  ".join(lines)

    return re.sub(r"\S+", fold_token, text)


class SettingsCategorySearchInput(Input):
    """Category filter input where "/" re-arms the query instead of typing.

    "/" is the screen-wide focus-the-filter key; once the filter itself has
    focus the screen's on_key never sees printable keys, so a second "/"
    would insert a literal slash into the query (task-1584's live trap).
    Intercept it here: select-all so the next keystroke replaces the stale
    text. Every other Input keeps literal "/" typing (endpoint URLs).
    """

    async def _on_key(self, event: Key) -> None:
        # Same slash representations the screen-level handler accepts --
        # some platforms/layouts emit key="slash" without character="/"
        # (Qodo review; the Playwright driver hit the "slash" name too).
        if event.key in {"/", "slash"} or event.character == "/":
            self.select_all()
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


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


def _mask_url_userinfo(url: object) -> str:
    """Mask a password embedded in a URL's userinfo before display.

    ``redact_secret_text`` is assignment-name based and misses credentials in
    ``scheme://user:pass@host`` form, so mask them positionally here. Non-URL
    or password-less input is returned unchanged.

    Args:
        url: A candidate endpoint string.

    Returns:
        The URL with any userinfo password replaced by ``***``.
    """
    from urllib.parse import urlsplit, urlunsplit

    text = str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    # Reconstruct from the raw netloc substring rather than the lazy
    # ``.hostname``/``.port`` properties: ``.port`` raises ValueError on a
    # malformed/out-of-range port (crashing the Test on a typo'd endpoint), and
    # ``.hostname`` strips IPv6 brackets. Keep host:port verbatim; only the
    # userinfo password is masked. (Never fall back to returning ``text`` with
    # the password intact -- redact_secret_text can't catch ``user:pass@host``.)
    netloc = parts.netloc
    userinfo, at, hostport = netloc.rpartition("@")
    if not at or ":" not in userinfo:
        # No userinfo, or a username with no password -> nothing to mask.
        return text
    user = userinfo.partition(":")[0]
    masked = f"{user}:***@{hostport}" if user else f"***@{hostport}"
    return urlunsplit((parts.scheme, masked, parts.path, parts.query, parts.fragment))


def overlay_provider_draft_config(
    app_config,
    *,
    provider_save_key: str,
    endpoint_key: str,
    draft_endpoint: str | None,
    draft_env_var: str | None,
    draft_api_key: str | None,
) -> dict:
    """Return a deep copy of ``app_config`` with unsaved draft provider fields overlaid.

    Args:
        app_config: The loaded application configuration.
        provider_save_key: The ``api_settings`` section key to overlay onto.
        endpoint_key: The endpoint setting key for this provider (e.g. ``api_url``).
        draft_endpoint: Draft endpoint, or ``None`` to leave the saved endpoint.
        draft_env_var: Draft credential env-var name, or ``None`` to leave saved.
        draft_api_key: Draft API key (``""`` models an explicit clear), or ``None``.

    Returns:
        A new config dict; ``app_config`` is never mutated.
    """
    merged = copy.deepcopy(dict(app_config)) if isinstance(app_config, Mapping) else {}
    api_settings = merged.get("api_settings")
    if not isinstance(api_settings, dict):
        api_settings = {}
        merged["api_settings"] = api_settings
    section = api_settings.get(provider_save_key)
    if not isinstance(section, dict):
        section = {}
        api_settings[provider_save_key] = section
    if draft_endpoint is not None:
        section[endpoint_key] = draft_endpoint
    if draft_env_var is not None:
        section["api_key_env_var"] = draft_env_var
    if draft_api_key is not None:
        section["api_key"] = draft_api_key
    return merged


class RagProfileNameModal(ModalScreen[str | None]):
    """Minimal name-prompt modal for the RAG profile-manager Clone/Rename actions.

    No existing text-prompt modal precedent lives in this screen (task-2
    brief) -- this follows the same dismiss-with-a-value + push_screen(modal,
    callback) shape as ``ConsoleSystemPromptModal``. Dismisses with the
    trimmed name, or ``None`` on Cancel/Escape/a blank submission.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, *, title: str, initial: str = "", confirm_label: str = "Save") -> None:
        super().__init__()
        self._modal_title = title
        self._initial = initial
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-rag-profile-name-modal", classes="settings-rag-profile-modal"):
            yield Static(self._modal_title, classes="destination-section")
            yield Input(value=self._initial, id="settings-rag-profile-name-input")
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-rag-profile-name-cancel")
                yield Button(
                    self._confirm_label,
                    id="settings-rag-profile-name-confirm",
                    variant="primary",
                )

    def on_mount(self) -> None:
        try:
            self.query_one("#settings-rag-profile-name-input", Input).focus()
        except QueryError:
            pass

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#settings-rag-profile-name-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#settings-rag-profile-name-confirm")
    def _handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit_current_value()

    @on(Input.Submitted, "#settings-rag-profile-name-input")
    def _handle_submit(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit_current_value()

    def _submit_current_value(self) -> None:
        try:
            value = self.query_one("#settings-rag-profile-name-input", Input).value
        except QueryError:
            value = ""
        self.dismiss(value.strip() or None)


class RagProfileSwitchConfirmModal(ModalScreen[str]):
    """Unsaved-Library/RAG-draft prompt before switching the active profile.

    Dismisses with ``"save"``, ``"discard"``, or ``"cancel"`` (also the
    Escape/no-choice outcome) -- never raises, never silently drops the
    caller's draft.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(
            id="settings-rag-profile-switch-modal", classes="settings-rag-profile-modal"
        ):
            yield Static("Unsaved Library/RAG changes", classes="destination-section")
            yield Static(
                "Save your changes before switching the active profile, or discard them?",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-rag-profile-switch-cancel")
                yield Button("Discard", id="settings-rag-profile-switch-discard")
                yield Button(
                    "Save", id="settings-rag-profile-switch-save", variant="primary"
                )

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    @on(Button.Pressed, "#settings-rag-profile-switch-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("cancel")

    @on(Button.Pressed, "#settings-rag-profile-switch-discard")
    def _handle_discard(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("discard")

    @on(Button.Pressed, "#settings-rag-profile-switch-save")
    def _handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("save")


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, storage, and app behavior."""

    BINDINGS = [
        ("s", "settings_save_category", "Save Settings category"),
        ("r", "settings_revert_category", "Revert Settings category"),
        ("t", "settings_test_category", "Test Settings category"),
        # Task 6 (541 AC6): RAG profile-workflow accelerators. Real Textual
        # bindings (so they're inert, by design, while an Input/TextArea has
        # focus -- those widgets consume printable keys before they bubble
        # here), but each action ALSO self-guards to the LIBRARY_RAG
        # category (see action_settings_rag_*) since this BINDINGS list is
        # shared by every Settings category, unlike s/r/t which dispatch
        # per-category internally.
        ("a", "settings_rag_set_active", "Set active RAG profile"),
        ("c", "settings_rag_clone", "Clone RAG profile"),
        ("b", "settings_rag_backfill", "Backfill RAG index"),
    ]

    #: Footer hint set — mirrors the show=True bindings the retired Textual
    #: Footer used to render (task-264 review).
    SETTINGS_SHORTCUTS = (
        ("s", "save category"),
        ("r", "revert category"),
        ("t", "test category"),
    )

    #: task-1564: categories whose `t` binding performs a real test action --
    #: everywhere else action_settings_test_category answers with the "No
    #: test action is available" toast, so the footer must not advertise it.
    TESTABLE_SETTINGS_CATEGORIES = frozenset(
        {
            SettingsCategoryId.PROVIDERS_MODELS,
            SettingsCategoryId.DIAGNOSTICS,
            SettingsCategoryId.STORAGE,
            SettingsCategoryId.PRIVACY_SECURITY,
            SettingsCategoryId.APPEARANCE,
            SettingsCategoryId.LIBRARY_RAG,
        }
    )

    #: Task 6 (541 AC6): RAG-only accelerator hints, appended to
    #: SETTINGS_SHORTCUTS whenever LIBRARY_RAG is the active category -- the
    #: a/c/b bindings above are no-ops everywhere else, so they're only
    #: advertised in the footer while they actually do something.
    LIBRARY_RAG_SHORTCUTS = (
        ("a", "set active"),
        ("c", "clone"),
        ("b", "backfill"),
    )

    #: Task 6 review (Important): action names of the RAG profile-workflow
    #: accelerators within BINDINGS -- action_show_workbench_help uses this
    #: to keep the app-level F1 help panel honest, mirroring the footer's
    #: LIBRARY_RAG gating above (the a/c/b bindings are guarded no-ops (see
    #: action_settings_rag_*) in every other category, so advertising them
    #: there would be a lie).
    _RAG_ACCELERATOR_ACTION_NAMES = frozenset(
        {
            "settings_rag_set_active",
            "settings_rag_clone",
            "settings_rag_backfill",
        }
    )

    active_category = reactive(SettingsCategoryId.OVERVIEW.value, recompose=True)
    category_search_query = reactive("")
    server_sync_workspace_handoff_rows = reactive((), recompose=True)
    manual_sync_rows = reactive((), recompose=True)
    # Deliberately NOT recompose=True (Qodo review of PR #1125): a recompose
    # here remounts the theme editor on the FIRST real user edit, discarding
    # the in-progress input and leaving the flag stale-True against a clean
    # editor. The two displays that read it (rail dirty marker, inspector
    # row) refresh in place via _refresh_theme_modified_widgets, mirroring
    # the InternalPromptsPanel.Modified idiom.
    theme_editor_modified = reactive(False)

    #: TASK-366: sentinel copies for the provider Test result row.
    _PROVIDER_TEST_NOT_RUN_COPY = "Provider test has not run."
    _PROVIDER_TEST_STALE_COPY = (
        "Provider settings changed since the last test — re-run Test Provider."
    )

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)
        self._settings_drafts: dict[SettingsCategoryId, SettingsDraft] = {}
        self._provider_test_result = self._PROVIDER_TEST_NOT_RUN_COPY
        self._provider_save_result = (
            "Provider settings have not been saved this session."
        )
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
        self._syncing_console_max_parallel_runs = False
        self._syncing_console_tool_result_display_chars = False
        self._syncing_console_defaults = False
        self._syncing_console_background_effects = False
        self._syncing_library_rag_defaults = False
        #: A profile id set by the "Save then switch" branch of the unsaved-
        #: changes prompt (RagProfileSwitchConfirmModal): remembered here so
        #: _apply_library_rag_save_result can dispatch the deferred set-active
        #: once the save worker reports back, then always self-clears.
        self._rag_profile_pending_activate: str | None = None
        self._syncing_appearance_defaults = False
        self._syncing_storage_defaults = False
        self._active_settings_field_id: str | None = None
        #: Task 3 (541 v2 UX AC3): the last-expanded Library/RAG Collapsible
        #: group (a key into `_RAG_GROUP_GUIDANCE`, e.g. "chunking"), used by
        #: `_rag_field_guidance_rows()` as the fallback scope when no RAG
        #: field is currently focused. Set by
        #: handle_settings_library_rag_collapsible_toggled; reset to None on
        #: any category switch away from LIBRARY_RAG (see _select_category).
        self._active_rag_scope_group: str | None = None
        #: Task 4 (541 v2 UX AC1): non-None while the profile Select is
        #: showing a DIFFERENT profile than the active one (browsed via the
        #: picker, never "Set active"'d) -- the editor renders THAT
        #: profile's values read-only and every
        #: handle_library_rag_*_changed handler early-returns before any
        #: draft staging (see _library_rag_edits_suppressed). None means
        #: "showing the active profile" (ordinary, draft-aware editing).
        #: Reset on any category switch away from LIBRARY_RAG (see
        #: _select_category) and cleared by _rag_after_set_active /
        #: _rag_after_profile_action.
        self._rag_preview_profile_id: str | None = None
        #: 541-v2 final review item 1: FIFO queue of the Select value(s)
        #: `_sync_library_rag_profile_widgets`'s OWN `set_options`/`value =`
        #: writes are about to cause the profile Select to post a
        #: `Select.Changed` for. A prior boolean-flag approach here
        #: (`_syncing_library_rag_profile_select`, set True around the
        #: writes, reset in a `finally`) could NOT actually suppress
        #: anything: Textual's `Select._watch_value` posts `Changed`
        #: through the widget's own async message queue, so
        #: handle_library_rag_profile_select_changed only receives it AFTER
        #: this resync has already returned and reset the flag (verified
        #: empirically with a mounted-pilot probe -- both the transient
        #: `Select.NULL` reset `set_options()` causes and the final
        #: resolved value arrive as separate deferred messages). Recording
        #: the actual value(s) a resync will cause Select to post here --
        #: and having the handler consume-and-ignore exactly those, in
        #: arrival order, popping each once matched -- suppresses them for
        #: real without swallowing a genuine user browse that later happens
        #: to land on the same value. Cleared on any category switch away
        #: from LIBRARY_RAG too (see _select_category): leaving the
        #: category recomposes the detail pane, minting a brand-new Select
        #: instance no stale expectation could ever legitimately match.
        self._rag_select_suppress_queue: list = []
        #: Same idiom as `_rag_select_suppress_queue`, applied to the
        #: Image Gen default-backend Select: constructing a Select with a
        #: non-blank `value=` fires `Select.Changed` the moment it mounts
        #: (verified empirically -- unlike Checkbox/Input, a fresh Select's
        #: reactive default IS blank, so any non-blank initial value is a
        #: real change from Select's own point of view). Every category
        #: (re)compose/recompose of `ImageGenSettingsPanel` -- initial
        #: category open, and the panel.recompose() after a successful
        #: Save or after Revert -- mints a brand-new Select instance that
        #: refires exactly once, re-staging its own already-current value
        #: into the draft as a spurious "edit" if left unguarded. Queued
        #: right before each (re)compose with the value that compose() is
        #: about to construct the Select with; the handler consumes and
        #: ignores exactly one matching entry per message. Cleared on any
        #: category switch away from IMAGE_GENERATION (see _select_category).
        self._image_gen_select_suppress_queue: list = []
        #: Task 6: guards the single-in-flight backend Test probe. Set
        #: True for the duration of one `_image_gen_probe_worker` run
        #: (Test buttons disabled meanwhile); a re-entrant Test click is a
        #: no-op while True (belt-and-suspenders alongside the disabled
        #: buttons themselves).
        self._image_gen_probe_in_flight: bool = False
        #: Bumped every time the user navigates AWAY from IMAGE_GENERATION
        #: (see _select_category). A probe result callback captures the
        #: session value at dispatch time; if it no longer matches when
        #: the callback lands (the category was left -- and possibly
        #: re-entered, minting a brand-new panel with fresh "Configured"/
        #: "Not configured" badges -- since dispatch), the stale result is
        #: dropped rather than clobbering an unrelated, freshly (re)opened
        #: panel's badge or in-flight state.
        self._image_gen_probe_session: int = 0
        #: Qodo PR #901 fix 3: `_image_gen_raw_section()`'s merged
        #: `[image_generation]` baseline, cached for the duration of one
        #: category "session" -- reached from every keystroke's staging
        #: handler (`Input.Changed` et al.), and `SettingsConfigAdapter
        #: ().load()` deepcopies the ENTIRE merged app config, making a
        #: fresh call per keystroke needlessly expensive. Invalidated
        #: (`None`) at exactly the three moments the on-disk truth can
        #: change: entering IMAGE_GENERATION (`_select_category`, so a
        #: stale cache from a PRIOR visit -- e.g. after an Advanced
        #: Config hand-edit in between -- is never reused), and after a
        #: successful Save or Revert (`_apply_image_gen_save_result` /
        #: `_handle_image_gen_revert`). `None` also means "not yet
        #: populated this session" -- lazily filled on first access, so
        #: merely opening the category (never editing anything) never
        #: triggers a load at all.
        self._image_gen_raw_section_cache: Mapping[str, object] | None = None
        self._navigation_provider: str | None = None
        self._navigation_model: str | None = None
        self._navigation_field: str | None = None
        #: One-shot focus intent (task-290): `Widget.focus()` defers its
        #: set_focus via call_later, so a storm recompose can destroy the
        #: target between intent and processing. Recorded when navigation
        #: focuses a provider field; cleared only when the INTENDED widget
        #: itself lands focus (any-focus clearing was too eager -- the
        #: stale category-chip focus the intent supersedes lands first),
        #: consumed by the post-recompose restore, and reset on
        #: navigation-away via _clear_navigation_provider_context.
        self._pending_navigation_focus_selector: str | None = None
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
        self._console_behavior_result = (
            "Console behavior settings have not been saved this session."
        )
        self._console_behavior_saved_this_session = False
        self._library_rag_result = (
            "Library/RAG defaults have not been saved this session."
        )
        self._library_rag_profile_result = (
            "No RAG profile action taken this session."
        )
        # Task 4 (SP3): index status readout + Backfill. The Static renders
        # this placeholder text at compose time -- the real state is fetched
        # off-thread (touches on-disk Chroma) on category show, after
        # set-active, and after a successful save; never during compose.
        self._library_rag_index_status_text = RAG_INDEX_STATUS_CHECKING_TEXT
        #: Task 2 (541 v2 UX): the raw dict behind `_library_rag_index_status_text`,
        #: kept in lockstep by `_apply_library_rag_index_status` (the single
        #: funnel every fetch trigger already goes through -- category show,
        #: post-save, post-set-active, 't' test). The pre-commit re-index
        #: confirm gate (`_confirm_reindex_then_save`) reads this to avoid
        #: an extra off-thread status fetch on every Save click; None until
        #: the first fetch completes.
        self._library_rag_index_status_cache: Mapping[str, object] | None = None
        #: Task 2 review (Important): debounces the cache-miss branch of
        #: `_confirm_reindex_then_save` against a second Save click landing
        #: before the first off-thread status fetch completes. Without this,
        #: a second click while `_library_rag_index_status_cache` is still
        #: None dispatches a SECOND `_rag_reindex_confirm_status_worker` in
        #: the same `exclusive=True` `@work` group as the first -- which
        #: CANCELS the first call, silently dropping ITS `pending_activate`
        #: (held only as a function-local inside the now-cancelled call,
        #: never handed back to anything). Set right before that dispatch,
        #: cleared once the flow's decision is made (worker callback, both
        #: the direct-dispatch and modal outcomes) or the modal resolves;
        #: see `_rag_reindex_confirm_status_worker` and
        #: `_handle_reindex_confirmation_result`.
        self._rag_reindex_confirm_in_flight = False
        self._library_rag_backfill_in_flight = False
        #: Task 5 review (541 v2 UX AC5, Important): tracks whether the LAST
        #: `_refresh_rag_first_run_panel_state` evaluation found the
        #: first-run starter panel active -- lets that method detect the
        #: actual first-run -> not-first-run TRANSITION (clone completes /
        #: backfill completes) rather than reacting to every ordinary status
        #: refresh. Without this, the fix that re-expands the Search group
        #: on first-run exit would fire unconditionally on every trigger
        #: (category show / Save / set-active), forcibly reopening Search
        #: even for a user in normal (non-first-run) state who deliberately
        #: collapsed it. False at construction to match the optimistic
        #: "not first-run" compose (`_render_library_rag_detail`, cache
        #: cold): the entering-first-run transition then fires correctly
        #: the first time a genuinely first-run status lands.
        self._rag_first_run_active = False
        self._appearance_result = (
            "Appearance defaults have not been saved this session."
        )
        self._storage_result = "Storage defaults have not been saved this session."
        #: Task 9 (workspace lifecycle card): the workspace row currently
        #: selected in the list, or None when nothing is selected -- the
        #: card renders nothing in that case. Reset whenever the user
        #: navigates away from WORKSPACES (see _select_category) and
        #: whenever a workspace is archived (its row disappears from the
        #: default, not-showing-archived list).
        self._settings_selected_workspace_id: str | None = None
        #: Task 9: mirrors the "Show archived" checkbox; read directly by
        #: `_render_workspaces_detail` on every (re)compose rather than
        #: cached separately, so there is no stale-watcher state to wipe.
        self._settings_show_archived_workspaces: bool = False
        self._settings_workspaces_result = ""
        self._advanced_config_result = "Advanced config validation: not run"
        self._advanced_config_validated_text: str | None = None
        self._ownership_by_category_cache = self._build_ownership_by_category()
        # Lazily-memoized cache, NOT a recompose=True reactive (P3 whole-branch
        # review Fix 1 + Fix 2): InternalPromptsPanel.Modified fires on every
        # prompt Save/Reset, and authoring.customized_count() iterates all
        # CATALOG entries with a config read each (~2.5ms). A recompose=True
        # reactive here would (a) unmount/remount the whole detail pane on
        # every save, wiping the panel's search text and scroll -- defeating
        # its own targeted _refresh_row design -- and (b) get recomputed live
        # on every Settings sidebar category-search keystroke via
        # _category_summaries(). Initialized to None and computed on first
        # DISPLAY (never in __init__): the count reads config, and reading
        # config during construction can force a config-file load/creation
        # before the app is ready (breaking storage-readiness checks). Kept
        # fresh afterward by _on_internal_prompts_modified via the panel's own
        # computed count (no extra live call).
        self._internal_prompts_customized_count: int | None = None
        # set_reactive, NOT plain assignment: assigning a recompose=True
        # reactive here fires refresh(recompose=True) on the not-yet-mounted
        # screen; the flag survives into mount and forces a full recompose of
        # the JUST-composed screen -- a wasted startup recompose (task-290
        # timeline: REFRESH pre-mount -> COMPOSE -> phantom re-COMPOSE).
        self.set_reactive(
            SettingsScreen.server_sync_workspace_handoff_rows,
            self._server_sync_workspace_handoff_loading_rows(),
        )
        self.set_reactive(
            SettingsScreen.manual_sync_rows,
            self._manual_sync_loading_rows(),
        )

    def save_state(self) -> dict[str, object]:
        """Save process-local Settings navigation and draft state.

        Returns:
            A deep-copy-safe state mapping for a fresh Settings screen.
        """
        state = super().save_state()
        if not isinstance(state, dict):
            state = {}
        state["active_category"] = self.active_category
        state["category_search_query"] = self._sanitize_category_search_query(
            self.category_search_query
        )
        state["settings_drafts"] = copy.deepcopy(self._settings_drafts)
        return state

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore validated process-local Settings state on a fresh screen.

        Args:
            state: Previously saved Settings navigation and draft state.
        """
        super().restore_state(state)
        if not isinstance(state, dict):
            return

        try:
            category = SettingsCategoryId(state.get("active_category"))
        except (TypeError, ValueError):
            pass
        else:
            self.active_category = category.value

        query = state.get("category_search_query")
        if isinstance(query, str):
            self.category_search_query = self._sanitize_category_search_query(query)

        drafts = state.get("settings_drafts")
        if isinstance(drafts, Mapping):
            valid_drafts = {
                category: draft
                for category, draft in drafts.items()
                if isinstance(category, SettingsCategoryId)
                and isinstance(draft, SettingsDraft)
                and draft.category is category
                and isinstance(draft.originals, dict)
                and isinstance(draft.values, dict)
                and all(isinstance(key, str) for key in draft.originals)
                and all(isinstance(key, str) for key in draft.values)
            }
            try:
                self._settings_drafts = copy.deepcopy(valid_drafts)
            except Exception:
                logger.debug("Ignoring malformed Settings draft state", exc_info=True)

    def _register_footer_shortcuts(self) -> None:
        """Register Settings shortcuts via BaseAppScreen's persisting API.

        Persistence matters here: this screen's `recompose=True` reactives
        (`active_category`, the sync-row tuples) replace the footer widget on
        every category switch; the registration must survive that.

        Task 6 (541 AC6): the rendered set is category-aware -- LIBRARY_RAG
        additionally advertises the a/c/b profile-workflow accelerators.
        Recomputed from `self.active_category` on every call, so re-calling
        this after a category switch (see `_select_category`) keeps the
        footer in sync without waiting for a recompose.
        """
        shortcuts = self._footer_shortcut_entries()
        self.register_footer_shortcuts(source="settings", shortcuts=shortcuts)

    def _footer_shortcut_entries(self) -> tuple[tuple[str, str], ...]:
        """Category- and focus-aware footer hints (task-1564/1560).

        Drops the ``s``/``r`` hints for categories outside the guided draft
        model (read-only pages, autosave Splash, immediate-apply Workspaces,
        the editor-owned Theme -- everywhere action_settings_save_category
        answers with an informational toast), drops the ``t`` hint for
        categories whose test action is the "No test action is available"
        toast, appends the RAG accelerators only where they act, and
        prefixes keys with "Esc, " while a text-entry widget owns focus
        (printable keys feed the field until Esc).
        """
        shortcuts = self.SETTINGS_SHORTCUTS
        active = self._active_category_id()
        if active not in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            shortcuts = tuple(
                entry for entry in shortcuts if entry[0] not in {"s", "r"}
            )
        if active not in self.TESTABLE_SETTINGS_CATEGORIES:
            shortcuts = tuple(
                entry for entry in shortcuts if entry[0] != "t"
            )
        if self._text_entry_focused():
            # task-1560: s/r/t are real bindings and therefore inert while an
            # Input/TextArea consumes printable keys -- advertising the bare
            # key would be a silent no-op (the critique's Alex trap). Tell
            # the truth about the escape hatch instead.
            shortcuts = tuple(
                (f"Esc, {key}", description) for key, description in shortcuts
            )
        if self._active_category_id() is SettingsCategoryId.LIBRARY_RAG:
            shortcuts = shortcuts + self.LIBRARY_RAG_SHORTCUTS
        return shortcuts

    def _text_entry_focused(self) -> bool:
        """Whether a printable-key-consuming widget owns focus right now."""
        try:
            focused = getattr(self.app, "focused", None)
        except Exception:
            # No active app (bare-screen tests / teardown) -- nothing focused.
            return False
        return isinstance(focused, (Input, TextArea))

    def on_descendant_focus(self, event) -> None:
        self._register_footer_shortcuts()

    def on_descendant_blur(self, event) -> None:
        self._register_footer_shortcuts()

    @staticmethod
    def _binding_entry_key_action_description(
        entry: object,
    ) -> tuple[str, str, str] | None:
        """(key, action, description) for a BINDINGS entry, or ``None`` if
        ``entry`` isn't a recognized shape.

        task-567: this used to only handle the tuple/list shape
        (``(key, action, description=...)``); a ``Binding(...)`` instance --
        Textual's OTHER valid BINDINGS entry shape -- silently vanished from
        the flattened help output below.
        """
        if isinstance(entry, Binding):
            return str(entry.key), str(entry.action), str(entry.description)
        if isinstance(entry, (tuple, list)) and entry:
            return (
                str(entry[0]),
                str(entry[1]),
                str(entry[2]) if len(entry) > 2 else "",
            )
        return None

    async def action_show_workbench_help(self) -> None:
        """F1 help, scoped to bindings that actually do something right now.

        `TldwCli.action_show_workbench_help` (app.py) delegates to this hook
        when it's present instead of falling back to its own generic
        BINDINGS flattener (`_show_generic_screen_help`), so this mirrors
        that fallback's output shape (same title/route id/shortcuts) except
        it drops the RAG profile-workflow accelerators (a/c/b) unless
        LIBRARY_RAG is the active category -- those bindings are guarded
        no-ops everywhere else (see action_settings_rag_*), same gating the
        footer already applies via LIBRARY_RAG_SHORTCUTS (task 6, 541 AC6
        review, Important).
        """
        show_rag_accelerators = (
            self._active_category_id() is SettingsCategoryId.LIBRARY_RAG
        )
        parsed_entries = (
            parts
            for entry in self.BINDINGS
            if (parts := self._binding_entry_key_action_description(entry))
            is not None
        )
        shortcuts = tuple(
            (key, description)
            for key, action, description in parsed_entries
            if show_rag_accelerators
            or action not in self._RAG_ACCELERATOR_ACTION_NAMES
        )
        screen_name = type(self).__name__
        state = WorkbenchHelpState(
            route_id=str(getattr(self.app, "current_tab", "") or screen_name),
            title=f"{screen_name} Shortcuts",
            shortcuts=shortcuts,
        )
        self.app.push_screen(WorkbenchHelpPanel(state))

    def on_mount(self) -> None:
        super().on_mount()
        self._register_footer_shortcuts()
        self._queue_sync_rows_refresh()
        # Task 4 (SP3): covers restored state (`restore_state` can set
        # `active_category` to LIBRARY_RAG before this screen is even
        # mounted) -- `_select_category` alone only covers a later in-session
        # switch INTO the category.
        self._maybe_refresh_rag_index_status_on_show()

    def on_screen_resume(self) -> None:
        self._queue_sync_rows_refresh()
        self._maybe_refresh_rag_index_status_on_show()
        self._maybe_refresh_workspaces_pane_on_show()

    def _maybe_refresh_rag_index_status_on_show(self) -> None:
        if self._active_category_id() is SettingsCategoryId.LIBRARY_RAG:
            self._refresh_library_rag_index_status()

    def _maybe_refresh_workspaces_pane_on_show(self) -> None:
        """Task 11: pick up registry changes made while this screen was
        suspended (e.g. a workspace created/archived from Console) as soon
        as the user resumes back into the WORKSPACES category -- mirrors
        `_maybe_refresh_rag_index_status_on_show`'s per-category resume
        guard above.
        """
        if self._active_category_id() is SettingsCategoryId.WORKSPACES:
            self._refresh_settings_workspaces_pane()

    def _queue_sync_rows_refresh(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        self._refresh_sync_rows()

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
                SettingsCategoryId.THEME,
                "Theme",
                "Full theme editor, custom colors, presets, and live preview.",
                "Custom",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.SPLASH_SCREEN,
                "Splash Screen",
                "Startup splash card selection, defaults, and preview gallery.",
                "Custom",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.STORAGE,
                "Storage",
                "Config path, local databases, and file locations.",
                "Guided",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.WORKSPACES,
                "Workspaces",
                "Create, rename, archive, and bind folders for agent file tools.",
                "Immediate actions",
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
                "RAG",
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
                "Roleplay",
                "Character and user profile browsing, plus how they attach to Console chats.",
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
                SettingsCategoryId.IMAGE_GENERATION,
                "Image Gen",
                "Image generation backend defaults for SwarmUI, OpenRouter, and "
                "other backend models.",
                "Guided",
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
            SettingsCategorySummary(
                SettingsCategoryId.INTERNAL_PROMPTS,
                "Internal Prompts",
                "View and edit the system prompts tldw_chatbook uses internally "
                "(RAG, web search, agents, summarization, more).",
                self._internal_prompts_status(),
            ),
        )

    def _get_internal_prompts_customized_count(self) -> int:
        """Memoized customized-prompt count for display.

        Computes the live count (authoring.customized_count(), ~2.5ms over all
        CATALOG entries with a config read each) only on the FIRST call, then
        caches it. Deferred to first display (never __init__) so it never
        forces a config load/creation during construction; recomputed at most
        once per screen since _on_internal_prompts_modified refreshes the cache
        directly from the panel's own event. Safe on a per-keystroke path
        (_category_summaries()) because every call after the first is a plain
        attribute read (task-P3 review Fix 2).
        """
        if self._internal_prompts_customized_count is None:
            try:
                self._internal_prompts_customized_count = (
                    internal_prompts_authoring.customized_count()
                )
            except Exception:
                self._internal_prompts_customized_count = 0
        return self._internal_prompts_customized_count

    def _internal_prompts_status(self) -> str:
        n = self._get_internal_prompts_customized_count()
        return f"{n} customized" if n else "Defaults"

    def _category_groups(
        self,
    ) -> tuple[tuple[str, tuple[SettingsCategoryId, ...]], ...]:
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
                    SettingsCategoryId.THEME,
                    SettingsCategoryId.SPLASH_SCREEN,
                    SettingsCategoryId.CONSOLE_BEHAVIOR,
                ),
            ),
            (
                "Data & Privacy",
                (
                    SettingsCategoryId.STORAGE,
                    SettingsCategoryId.WORKSPACES,
                    SettingsCategoryId.PRIVACY_SECURITY,
                ),
            ),
            ("Troubleshooting", (SettingsCategoryId.DIAGNOSTICS,)),
            (
                "Expert",
                (
                    SettingsCategoryId.INTERNAL_PROMPTS,
                    SettingsCategoryId.ADVANCED_CONFIG,
                ),
            ),
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
                    SettingsCategoryId.IMAGE_GENERATION,
                ),
            ),
        )

    def _domain_category_contracts(self) -> tuple[SettingsDomainCategoryContract, ...]:
        return SETTINGS_DOMAIN_CATEGORY_CONTRACTS

    def _domain_contract_by_category(
        self,
    ) -> Mapping[SettingsCategoryId, SettingsDomainCategoryContract]:
        return DOMAIN_CONTRACT_BY_CATEGORY

    def _domain_category_contract(
        self, category: SettingsCategoryId
    ) -> SettingsDomainCategoryContract:
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
                            "the active RAG profile (rag_profiles/<id>.json)",
                            "the [rag.service].profile pointer",
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
            if contract.category is SettingsCategoryId.IMAGE_GENERATION:
                records.append(
                    SettingsOwnershipRecord(
                        category=contract.category,
                        owns_config_sections=(
                            "image_generation.default_backend",
                            "image_generation.enabled_backends",
                            "image_generation.<backend>.*",
                            "image_generation.default_batch",
                            "image_generation.max_variants_per_message",
                            "image_generation.context_llm_enabled",
                            "image_generation.context_llm_turns",
                            "image_generation.context_llm_timeout_seconds",
                        ),
                        reads_runtime_state_from=contract.source_of_truth,
                        writes_allowed=True,
                        runtime_owner="Settings persisted defaults; Console /generate-image",
                        boundary_copy=(
                            "Settings owns persisted backend and generation-default config; "
                            "Console owns /generate-image, cards, and variant actions."
                        ),
                        recovery_copy=(
                            "Revert unsaved edits, or edit image_generation values "
                            "directly in Advanced Config."
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
                owns_config_sections=(
                    "global defaults",
                    "validation status",
                    "recovery guidance",
                ),
                reads_runtime_state_from=(
                    "Console",
                    "MCP",
                    "ACP",
                    "sync readiness",
                    "workspace status",
                ),
                writes_allowed=False,
                runtime_owner="owning destinations",
                boundary_copy="; ".join(
                    value for _, value in SETTINGS_OVERVIEW_BOUNDARY_ROWS
                ),
                recovery_copy=(
                    "Sync status here is read-only - manage workspaces in "
                    "Settings > Workspaces; switch in Console (Alt+W); run "
                    "sync from the owning sync surfaces."
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
                reads_runtime_state_from=("app theme",),
                writes_allowed=True,
                runtime_owner="Settings persisted defaults",
                boundary_copy=(
                    "Settings owns launch visual defaults; open the Theme category for full "
                    "theme editing and deeper visual preview."
                ),
                recovery_copy=(
                    "Preview applies runtime-safe values for this session only; Save persists "
                    "defaults, Revert restores loaded values."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.THEME,
                owns_config_sections=("custom theme files",),
                reads_runtime_state_from=("app theme", "custom theme files"),
                writes_allowed=True,
                runtime_owner="Theme editor",
                boundary_copy=(
                    "Settings Theme editor owns custom color palettes and theme files; "
                    "use the editor's Apply/Save/Reset buttons."
                ),
                recovery_copy=(
                    "Themes are saved to ~/.config/tldw_cli/themes/; reset or delete files there "
                    "to recover."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.SPLASH_SCREEN,
                owns_config_sections=("splash_screen",),
                reads_runtime_state_from=("splash_screen config",),
                writes_allowed=True,
                runtime_owner="Splash Screen viewer",
                boundary_copy=(
                    "Settings Splash Screen viewer owns startup splash defaults and card "
                    "selection; changes are saved immediately."
                ),
                recovery_copy=(
                    "Edit splash_screen values in Advanced Config or reset defaults from the "
                    "Splash Screen category."
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
                reads_runtime_state_from=(
                    "local filesystem",
                    "configured database paths",
                ),
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
                category=SettingsCategoryId.WORKSPACES,
                owns_config_sections=(),
                reads_runtime_state_from=("workspace registry",),
                writes_allowed=True,
                runtime_owner="Workspace registry (immediate actions)",
                boundary_copy=(
                    "Lifecycle and folder bindings apply immediately; no draft state."
                ),
                recovery_copy=(
                    "Quick actions: switch/rename/archive in Console (Alt+W); create in Library."
                ),
            ),
            SettingsOwnershipRecord(
                category=SettingsCategoryId.PRIVACY_SECURITY,
                owns_config_sections=(
                    "encryption",
                    "api_settings.<provider>.credential_source",
                ),
                reads_runtime_state_from=(
                    "config redaction",
                    "environment credential status",
                ),
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
                    "console.max_parallel_runs",
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
            SettingsOwnershipRecord(
                category=SettingsCategoryId.INTERNAL_PROMPTS,
                owns_config_sections=("internal_prompts.<prompt id>",),
                reads_runtime_state_from=("packaged internal prompt registry",),
                writes_allowed=True,
                runtime_owner="Internal Prompts panel",
                boundary_copy=(
                    "Settings Internal Prompts panel owns internal-tooling prompt overrides; "
                    "use each prompt's own Save/Reset buttons."
                ),
                recovery_copy=(
                    "Reset a prompt from its editor to restore the packaged default text."
                ),
            ),
            *self._domain_category_ownership_records(),
        )

    def _build_ownership_by_category(
        self,
    ) -> dict[SettingsCategoryId, SettingsOwnershipRecord]:
        return {
            record.category: record for record in self._category_ownership_records()
        }

    def _ownership_by_category(
        self,
    ) -> dict[SettingsCategoryId, SettingsOwnershipRecord]:
        return self._ownership_by_category_cache

    @staticmethod
    def _missing_ownership_record(
        category: SettingsCategoryId,
    ) -> SettingsOwnershipRecord:
        logger.warning(
            "Settings ownership record missing for category %s", category.value
        )
        return SettingsOwnershipRecord(
            category=category,
            reads_runtime_state_from=("unknown",),
            writes_allowed=False,
            runtime_owner="Settings ownership matrix",
            boundary_copy="Ownership record missing; update the Settings ownership matrix.",
            recovery_copy="Update the Settings ownership matrix before enabling writes.",
            read_only_reason="Ownership record missing; update matrix before exposing actions.",
        )

    def _ownership_record(
        self, category: SettingsCategoryId
    ) -> SettingsOwnershipRecord:
        return self._ownership_by_category().get(
            category
        ) or self._missing_ownership_record(category)

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
        if callable(getattr(app_config, "setdefault", None)) and hasattr(
            app_config, "__setitem__"
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

    def _loaded_console_max_parallel_runs(self) -> int:
        # No maximum -- deliberately unbounded (user-owned trade-off), see
        # DEFAULT_CONSOLE_MAX_PARALLEL_RUNS's docstring.
        return coerce_int_setting(
            self._console_settings().get(
                "max_parallel_runs",
                DEFAULT_CONSOLE_MAX_PARALLEL_RUNS,
            ),
            DEFAULT_CONSOLE_MAX_PARALLEL_RUNS,
            minimum=MIN_CONSOLE_MAX_PARALLEL_RUNS,
        )

    def _loaded_tool_result_display_chars(self) -> int:
        # TASK-870: how much of an agent tool result the Console DISPLAYS --
        # distinct from [agents]/RunBudget.max_tool_result_chars, which
        # governs how much the MODEL saw and is not user-configurable here.
        return coerce_int_setting(
            self._console_settings().get(
                "tool_result_display_chars",
                DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
            ),
            DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
            minimum=MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
            maximum=MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
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
            "max_parallel_runs": self._loaded_console_max_parallel_runs(),
            "tool_result_display_chars": self._loaded_tool_result_display_chars(),
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
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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
        has_unsaved_changes = self._category_has_unsaved_changes(
            SettingsCategoryId.CONSOLE_BEHAVIOR
        )
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

    def _remote_images_enabled(self) -> bool:
        """Return the live [chat.images].render_remote_images value."""
        from ...Chat.console_image_view import resolve_render_remote_images

        return resolve_render_remote_images(
            getattr(self.app_instance, "app_config", {}) or {}
        )

    def _remote_images_button_label(self) -> str:
        return "Enabled" if self._remote_images_enabled() else "Disabled"

    def _toggle_remote_images(self) -> bool:
        """Flip render_remote_images: persist it AND poke the live config.

        ADR-020-style immediate write (no category draft): the toggle is a
        single security-relevant boolean. The App captures ``app_config``
        once at startup, so persisting alone would not take effect until
        restart -- the raw in-memory tree the transcript gate reads is
        updated in place too.

        Returns:
            The new (post-toggle) enabled value.
        """
        next_value = not self._remote_images_enabled()
        save_settings_to_cli_config(
            {"chat.images": {"render_remote_images": next_value}}
        )
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
            if isinstance(raw, dict):
                raw.setdefault("chat", {}).setdefault("images", {})[
                    "render_remote_images"
                ] = next_value
            chat_section = app_config.get("chat")
            if isinstance(chat_section, dict) and isinstance(
                chat_section.get("images"), dict
            ):
                chat_section["images"]["render_remote_images"] = next_value
        return next_value

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

    def _console_max_parallel_runs_value(self) -> int | str:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and "max_parallel_runs" in draft.values:
            return draft.values["max_parallel_runs"]
        return self._loaded_console_max_parallel_runs()

    def _tool_result_display_chars_value(self) -> int | str:
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and "tool_result_display_chars" in draft.values:
            return draft.values["tool_result_display_chars"]
        return self._loaded_tool_result_display_chars()

    def _tool_result_display_chars_label(self) -> str:
        value = self._tool_result_display_chars_value()
        try:
            chars = self._normalise_tool_result_display_chars(value)
        except ValueError:
            return f"Invalid value: {value}"
        return f"{chars} characters"

    def _update_console_paste_summary(self) -> None:
        try:
            summary = self.query_one(
                "#settings-overview-console-paste-collapse", Static
            )
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
            key: draft.values[key]
            if draft is not None and key in draft.values
            else value
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
        return load_rag_defaults_from_active_profile()

    def _library_rag_loaded_values(self) -> dict[str, object]:
        return asdict(self._library_rag_loaded_defaults())

    def _library_rag_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.LIBRARY_RAG)

    def _library_rag_setting_values(self) -> dict[str, object]:
        loaded = self._library_rag_loaded_values()
        draft = self._library_rag_draft()
        return {
            key: draft.values[key]
            if draft is not None and key in draft.values
            else value
            for key, value in loaded.items()
        }

    def _library_rag_current_defaults(self) -> SettingsLibraryRagDefaults:
        return SettingsLibraryRagDefaults(**self._library_rag_setting_values())

    def _library_rag_validation_result(self):
        return validate_library_rag_defaults(self._library_rag_current_defaults())

    def _library_rag_soft_warnings(self) -> list[str]:
        """Advisory-only warnings (e.g. reranker top-k vs default results)
        for the current draft/loaded values. NEVER gates Save -- see
        _library_rag_save_enabled, which only consults
        _library_rag_validation_result (hard errors)."""
        return soft_config_warnings(self._library_rag_current_defaults())

    def _library_rag_save_enabled(self) -> bool:
        # Task 4 (541 v2 UX AC1): Save/Revert must be unavailable while the
        # editor is merely PREVIEWING a browsed (non-active) profile -- the
        # active profile's own draft (if any) is untouched and unaffected,
        # but nothing on screen right now is even editable.
        if self._rag_preview_profile_id is not None:
            return False
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
            key: draft.values[key]
            if draft is not None and key in draft.values
            else value
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

    # ------------------------------------------------------------------
    # Image Gen (task 5): draft/dirty editing + Save/Revert.
    #
    # Unlike APPEARANCE/LIBRARY_RAG/STORAGE (flat scalar fields), Image Gen
    # stages edits into `self._settings_drafts[IMAGE_GENERATION]` using
    # namespaced string keys so the ONE generic `SettingsDraft` still drives
    # the shared dirty-marker mechanism (`_category_has_unsaved_changes`,
    # the rail `*`, `settings-dirty-category`) -- no parallel draft
    # mechanism. Keys: "default_backend", "enabled_backends",
    # "context_llm_enabled", "default_batch", "max_variants_per_message",
    # "context_llm_turns", "context_llm_timeout_seconds",
    # "field::<backend_id>::<toml_key>" (per edited backend field), and
    # "cleared::<backend_id>::<toml_key>" (per Clear action).
    #
    # The diff/save baseline is ALWAYS `SettingsConfigAdapter().load()`'s
    # MERGED `[image_generation]` view -- never `load_user_image_generation_
    # table()`, which is display-only (see that helper's and
    # `diff_to_sections`'s docstrings). Using the merged view as the
    # "original" for dirty-tracking is what makes the Enabled checkboxes,
    # the default-backend Select, and `context_llm_enabled` safe to stage
    # straight from LIVE widget state on every change and at save time:
    # unlike free-text Inputs, Checkboxes/Select have no "blank value means
    # deferred to placeholder" ambiguity, and they're initialized from that
    # SAME merged config -- so an untouched widget's value always equals
    # its own "original" (never a spurious diff), while a genuinely toggled
    # one always differs.
    def _image_gen_raw_section(self) -> Mapping[str, object]:
        # Qodo PR #901 fix 3: cached per category session -- see
        # `_image_gen_raw_section_cache`'s docstring at its `__init__`
        # declaration for the exact three invalidation points.
        if self._image_gen_raw_section_cache is None:
            raw = SettingsConfigAdapter().load().get("image_generation")
            self._image_gen_raw_section_cache = (
                raw if isinstance(raw, Mapping) else {}
            )
        return self._image_gen_raw_section_cache

    def _image_gen_overlay_values(self) -> dict[str, object]:
        draft = self._settings_drafts.get(SettingsCategoryId.IMAGE_GENERATION)
        return dict(draft.values) if draft is not None else {}

    def _image_gen_expected_default_backend_select_value(
        self, overlay: Mapping[str, object]
    ) -> object:
        """The exact value `ImageGenSettingsPanel.compose()` is about to
        construct `#settings-imagegen-default_backend` with, for `overlay`
        -- must mirror that compose() logic exactly (see
        `_queue_image_gen_select_suppression`'s docstring)."""
        cfg = get_image_generation_config(reload=True)
        effective_default_backend = overlay.get("default_backend", cfg.default_backend)
        return (
            effective_default_backend
            if effective_default_backend in IMAGE_GEN_BACKEND_IDS
            else Select.NULL
        )

    def _queue_image_gen_select_suppression(self, overlay: Mapping[str, object]) -> None:
        """Record the value the about-to-(re)compose default-backend
        `Select` will mount with, if that value is non-blank -- a fresh
        `Select` only posts `Changed` on mount when constructed with a
        non-`Select.NULL` value (verified empirically; unlike Checkbox,
        which never refires on construction regardless of value). Call
        this immediately before every `ImageGenSettingsPanel` (re)compose:
        the initial category-open `_render_detail_pane` branch, and the
        `panel.recompose()` calls in `_apply_image_gen_save_result` /
        `_handle_image_gen_revert`. See `_rag_select_suppress_queue` for
        the sibling idiom this mirrors (a boolean in-progress flag cannot
        suppress a deferred `Select.Changed` message)."""
        expected_value = self._image_gen_expected_default_backend_select_value(overlay)
        if expected_value is not Select.NULL:
            self._image_gen_select_suppress_queue.append(expected_value)

    def _image_gen_stage(self, key: str, original: object, value: object) -> None:
        category = SettingsCategoryId.IMAGE_GENERATION
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)
        self._update_draft_status_widgets(category)

    def _image_gen_unstage(self, key: str) -> None:
        """Remove one staged key without touching any other (e.g. a fresh
        edit cancelling a pending Clear on the SAME field)."""
        category = SettingsCategoryId.IMAGE_GENERATION
        draft = self._settings_drafts.get(category)
        if draft is None:
            return
        draft.values.pop(key, None)
        draft.originals.pop(key, None)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _stage_image_gen_field(self, backend_id: str, toml_key: str, raw_value: str) -> None:
        raw_backend = self._image_gen_raw_section().get(backend_id) or {}
        original = raw_backend.get(toml_key) if isinstance(raw_backend, Mapping) else None
        original_str = "" if original is None else str(original)
        self._image_gen_stage(f"field::{backend_id}::{toml_key}", original_str, raw_value)
        # Typing into a field cancels a pending Clear on that SAME field --
        # otherwise diff_to_sections' documented "cleared wins" rule would
        # silently delete a key the user just retyped.
        self._image_gen_unstage(f"cleared::{backend_id}::{toml_key}")
        self._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)

    @staticmethod
    def _image_gen_coerce_int(raw_value: str) -> int | None:
        try:
            return int(raw_value.strip())
        except ValueError:
            return None

    @staticmethod
    def _image_gen_coerce_float(raw_value: str) -> float | None:
        try:
            return float(raw_value.strip())
        except ValueError:
            return None

    def _refresh_image_gen_default_markers(self, effective_default_backend: str) -> None:
        for backend_id in IMAGE_GEN_BACKEND_IDS:
            try:
                marker = self.query_one(
                    f"#settings-imagegen-default-marker-{backend_id}", Static
                )
            except QueryError:
                continue
            marker.update("★ Default" if backend_id == effective_default_backend else "")

    def _image_gen_draft_values_for_save(
        self, panel: ImageGenSettingsPanel
    ) -> ImageGenDraftValues:
        draft = self._settings_drafts.get(SettingsCategoryId.IMAGE_GENERATION)
        values = draft.values if draft is not None else {}
        backend_fields: dict[str, dict[str, str]] = {}
        cleared_fields: dict[str, list[str]] = {}
        for key, value in values.items():
            if key.startswith("field::"):
                _prefix, backend_id, toml_key = key.split("::", 2)
                backend_fields.setdefault(backend_id, {})[toml_key] = value
            elif key.startswith("cleared::"):
                _prefix, backend_id, toml_key = key.split("::", 2)
                cleared_fields.setdefault(backend_id, []).append(toml_key)

        # enabled_backends/default_backend/context_llm_enabled are ALWAYS
        # read live off the mounted widgets (never from `values`) -- see
        # the class-comment above for why this is safe and required (an
        # untouched checkbox/select must still be reflected, since it's
        # never staged unless the user actually toggles it).
        enabled_backends = [
            backend_id
            for backend_id in IMAGE_GEN_BACKEND_IDS
            if panel.query_one(
                f"#settings-imagegen-enabled-{backend_id}", Checkbox
            ).value
        ]
        default_backend_widget_value = panel.query_one(
            "#settings-imagegen-default_backend", Select
        ).value
        default_backend = (
            default_backend_widget_value
            if isinstance(default_backend_widget_value, str)
            else None
        )
        context_llm_enabled = bool(
            panel.query_one(
                "#settings-imagegen-context_llm_enabled", Checkbox
            ).value
        )

        return ImageGenDraftValues(
            default_backend=default_backend,
            enabled_backends=enabled_backends,
            default_batch=values.get("default_batch"),
            max_variants_per_message=values.get("max_variants_per_message"),
            context_llm_enabled=context_llm_enabled,
            context_llm_turns=values.get("context_llm_turns"),
            context_llm_timeout_seconds=values.get("context_llm_timeout_seconds"),
            backend_fields=backend_fields,
            cleared_fields=cleared_fields,
        )

    @on(Select.Changed, "#settings-imagegen-default_backend")
    def handle_image_gen_default_backend_changed(self, event: Select.Changed) -> None:
        event.stop()
        # A brand-new Select instance refires Changed with its OWN initial
        # value the moment it mounts (every category open + every post-
        # Save/Revert recompose) -- consume-and-ignore that expected
        # arrival rather than re-staging the value it's already showing as
        # a spurious "edit". See _queue_image_gen_select_suppression.
        queue = self._image_gen_select_suppress_queue
        if queue and event.value == queue[0]:
            queue.pop(0)
            return
        value = event.value if isinstance(event.value, str) else None
        if value is None:
            self._image_gen_unstage("default_backend")
            self._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)
            # Qodo PR #901 fix 1: clearing the Select to blank must clear
            # every row's "★ Default" marker too -- left unrefreshed, the
            # OLD default's marker keeps showing, now inconsistent with
            # the Select itself showing nothing selected.
            self._refresh_image_gen_default_markers("")
            return
        original = self._image_gen_raw_section().get("default_backend")
        self._image_gen_stage("default_backend", original, value)
        self._refresh_image_gen_default_markers(value)

    @on(Checkbox.Changed)
    def handle_image_gen_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = str(getattr(event.checkbox, "id", "") or "")
        if not checkbox_id.startswith("settings-imagegen-"):
            return
        event.stop()
        from ...Widgets.settings_image_gen_panel import switch_word, toggle_label

        if checkbox_id == "settings-imagegen-context_llm_enabled":
            event.checkbox.label = toggle_label("Context LLM", bool(event.value))
            original = bool(self._image_gen_raw_section().get("context_llm_enabled"))
            self._image_gen_stage("context_llm_enabled", original, bool(event.value))
            return
        prefix = "settings-imagegen-enabled-"
        if not checkbox_id.startswith(prefix):
            return
        event.checkbox.label = switch_word(bool(event.value))
        try:
            panel = self.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        except QueryError:
            return
        enabled_backends = [
            backend_id
            for backend_id in IMAGE_GEN_BACKEND_IDS
            if panel.query_one(
                f"#settings-imagegen-enabled-{backend_id}", Checkbox
            ).value
        ]
        # Normalized to canonical order (Minor 1, final review) -- `enabled_
        # backends` is already canonical (built by iterating IMAGE_GEN_
        # BACKEND_IDS above); without normalizing `original` the SAME way,
        # a config file whose list happens to be in a different order would
        # spuriously stage a "dirty" edit even when nothing actually
        # changed, and the rail marker would never clear on its own.
        original = image_gen_canonical_backend_order(
            self._image_gen_raw_section().get("enabled_backends")
        )
        self._image_gen_stage("enabled_backends", original, enabled_backends)

    _IMAGE_GEN_INT_GLOBAL_KEYS = {
        "settings-imagegen-default_batch": "default_batch",
        "settings-imagegen-max_variants_per_message": "max_variants_per_message",
        "settings-imagegen-context_llm_turns": "context_llm_turns",
    }

    @on(Input.Changed)
    def handle_image_gen_input_changed(self, event: Input.Changed) -> None:
        input_id = str(getattr(event.input, "id", "") or "")
        if not input_id.startswith("settings-imagegen-"):
            return
        prefix = "settings-imagegen-field-"
        if input_id.startswith(prefix):
            event.stop()
            remainder = input_id[len(prefix):]
            for backend_id in IMAGE_GEN_BACKEND_IDS:
                backend_prefix = f"{backend_id}-"
                if remainder.startswith(backend_prefix):
                    toml_key = remainder[len(backend_prefix):]
                    self._stage_image_gen_field(backend_id, toml_key, event.value)
                    return
            return
        global_key = self._IMAGE_GEN_INT_GLOBAL_KEYS.get(input_id)
        if global_key is not None:
            event.stop()
            original = self._image_gen_raw_section().get(global_key)
            # An unparseable edit is staged as the raw string rather than
            # silently dropped -- it still marks dirty, and validate_draft
            # (settings_image_gen_defaults.py) is what actually catches it
            # at Save time with an inline error, matching the per-backend
            # "int"-kind fields' treatment exactly instead of the edit
            # just vanishing with no feedback at all.
            coerced = self._image_gen_coerce_int(event.value)
            staged_value = coerced if coerced is not None else event.value
            self._image_gen_stage(global_key, original, staged_value)
            return
        if input_id == "settings-imagegen-context_llm_timeout_seconds":
            event.stop()
            original = self._image_gen_raw_section().get("context_llm_timeout_seconds")
            coerced_float = self._image_gen_coerce_float(event.value)
            staged_value = coerced_float if coerced_float is not None else event.value
            self._image_gen_stage(
                "context_llm_timeout_seconds", original, staged_value
            )

    def _handle_image_gen_clear(self, backend_id: str, toml_key: str) -> None:
        self._image_gen_stage(f"cleared::{backend_id}::{toml_key}", False, True)
        if backend_id == "swarmui" and toml_key == "swarm_token":
            # The loader also resolves the legacy `api_key` spelling as a
            # back-compat fallback; Clear must delete BOTH or a stale
            # hand-edited api_key resurrects the credential with no in-UI
            # recovery (deleting an absent key is a no-op, so this is free).
            self._image_gen_stage("cleared::swarmui::api_key", False, True)
        self._image_gen_unstage(f"field::{backend_id}::{toml_key}")
        new_source = image_gen_key_source_after_clear(backend_id)
        try:
            secret_input = self.query_one(
                f"#settings-imagegen-field-{backend_id}-{toml_key}", Input
            )
            secret_input.value = ""
            secret_input.placeholder = _image_gen_secret_placeholder(new_source)
        except QueryError:
            pass
        try:
            source_static = self.query_one(
                f"#settings-imagegen-key-source-{backend_id}", Static
            )
            source_static.update(_image_gen_key_source_line(new_source))
            secret_optional = backend_id == "swarmui"
            source_static.set_class(
                secret_optional and new_source == "missing",
                "settings-imagegen-key-source-neutral",
            )
        except QueryError:
            pass
        self._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)

    # ------------------------------------------------------------------
    # Image Gen (task 6): backend "Test" probes.
    #
    # A probe never persists anything -- it's a short live/filesystem
    # reachability check (`probe_backend`, settings_image_gen_defaults.py)
    # run against the CURRENT (possibly-unsaved) form values for one
    # backend. `probe_backend` is BLOCKING (a plain HTTP GET or filesystem
    # stat), so it always runs off the UI thread via `@work(thread=True)`;
    # any exception it fails to catch itself degrades to a safe closed-set
    # badge here rather than ever propagating exception text into a badge
    # or a notify(). Only one probe may be in flight at a time
    # (`_image_gen_probe_in_flight` + all six Test buttons disabled
    # meanwhile) -- see `_image_gen_probe_session`'s docstring (near its
    # declaration) for how a stale callback from a since-left-and-
    # reentered category is safely dropped instead of clobbering an
    # unrelated, freshly (re)opened panel.

    def _image_gen_test_form_values(
        self, panel: ImageGenSettingsPanel, backend_id: str
    ) -> dict[str, str]:
        """Gather the CURRENT non-secret form values for `backend_id`'s Test
        probe -- an edited-but-unsaved Input wins; a blank (untouched)
        Input falls back to the resolved effective value it's currently
        showing as its own placeholder (see `effective_placeholder`),
        never a blank string `probe_backend` could mistake for
        "explicitly cleared"."""
        cfg = get_image_generation_config(reload=True)
        form_values: dict[str, str] = {}
        for spec in IMAGE_GEN_FIELD_SCHEMA[backend_id]:
            if spec.kind == "secret":
                continue
            try:
                current = panel.query_one(
                    f"#settings-imagegen-field-{backend_id}-{spec.toml_key}", Input
                ).value.strip()
            except QueryError:
                current = ""
            form_values[spec.toml_key] = current or image_gen_effective_placeholder(
                cfg, backend_id, spec.toml_key
            )
        return form_values

    def _image_gen_test_secret(
        self, panel: ImageGenSettingsPanel, backend_id: str
    ) -> str | None:
        """The secret to probe with: this session's pasted-but-unsaved
        value if present, else the effective resolved secret (env/config/
        keyring) -- see `probe_backend`'s `secret` parameter docstring."""
        secret_spec = next(
            (
                spec
                for spec in IMAGE_GEN_FIELD_SCHEMA[backend_id]
                if spec.kind == "secret"
            ),
            None,
        )
        if secret_spec is None:
            return None
        try:
            pasted = panel.query_one(
                f"#settings-imagegen-field-{backend_id}-{secret_spec.toml_key}", Input
            ).value.strip()
        except QueryError:
            pasted = ""
        if pasted:
            return pasted
        cfg = get_image_generation_config(reload=True)
        return image_gen_effective_secret_value(cfg, backend_id)

    def _image_gen_set_test_buttons_disabled(self, disabled: bool) -> None:
        for backend_id in IMAGE_GEN_BACKEND_IDS:
            try:
                self.query_one(
                    f"#settings-imagegen-test-{backend_id}", Button
                ).disabled = disabled
            except QueryError:
                continue

    def _handle_image_gen_test(self, backend_id: str) -> None:
        if self._image_gen_probe_in_flight:
            return
        try:
            panel = self.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        except QueryError:
            return
        form_values = self._image_gen_test_form_values(panel, backend_id)
        secret = self._image_gen_test_secret(panel, backend_id)
        self._image_gen_probe_in_flight = True
        self._image_gen_set_test_buttons_disabled(True)
        self._image_gen_probe_worker(
            backend_id, form_values, secret, self._image_gen_probe_session
        )

    @work(thread=True, exclusive=False, exit_on_error=False)
    def _image_gen_probe_worker(
        self,
        backend_id: str,
        form_values: dict[str, str],
        secret: str | None,
        session: int,
    ) -> None:
        try:
            badge = image_gen_probe_backend(backend_id, form_values, secret).badge
        except Exception as exc:  # noqa: BLE001 - any escape must degrade safely
            # Qodo PR #901 fix 2: this probe builds Authorization headers
            # from a pasted-or-effective secret -- the spec's keys-never-
            # enter-logs contract forbids logging the raw exception text
            # (it could echo a header, URL, or secret embedded in some
            # library's error message). Log only the exception TYPE name
            # and the backend id, never str(exc).
            logger.debug(
                f"Image Gen probe for {backend_id!r} raised "
                f"{type(exc).__name__}"
            )
            badge = "Unreachable: probe error"
        finally:
            self.app.call_from_thread(
                self._apply_image_gen_probe_result, backend_id, badge, session
            )

    def _apply_image_gen_probe_result(
        self, backend_id: str, badge: str, session: int
    ) -> None:
        if session != self._image_gen_probe_session:
            # Stale: the category was left (and possibly re-entered, minting
            # a brand-new panel) since this probe was dispatched. Dropping
            # it here -- rather than clearing `_image_gen_probe_in_flight`
            # or touching any widget -- is what keeps a leftover result from
            # a PREVIOUS visit from ever re-enabling buttons for (or
            # overwriting a badge on) an unrelated, currently active probe
            # or freshly (re)opened panel.
            return
        self._image_gen_probe_in_flight = False
        self._image_gen_set_test_buttons_disabled(False)
        try:
            self.query_one(f"#settings-imagegen-status-{backend_id}", Static).update(
                badge
            )
        except QueryError:
            pass

    def _handle_image_gen_save(self) -> None:
        try:
            panel = self.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        except QueryError:
            return
        draft_values = self._image_gen_draft_values_for_save(panel)
        errors, warnings = validate_image_gen_draft(draft_values)
        if errors:
            message = " ".join(errors)
            self._set_static_text("#settings-imagegen-save-result", message)
            self.app.notify(message, severity="error")
            return
        self._set_static_text(
            "#settings-imagegen-save-result", "Saving Image Gen defaults..."
        )
        self._settings_save_image_gen_worker(draft_values, warnings)

    @work(exclusive=True, thread=True)
    def _settings_save_image_gen_worker(
        self, draft_values: ImageGenDraftValues, warnings: list[str]
    ) -> None:
        raw_config = SettingsConfigAdapter().load()
        sections, deletions = image_gen_diff_to_sections(draft_values, raw_config)
        adapter = SettingsConfigAdapter()
        ok = True
        if sections:
            ok = adapter.save_sections(sections) and ok
        for section, keys in deletions.items():
            if keys:
                ok = adapter.delete_values(section, keys) and ok
        if ok:
            reset_image_generation_config_cache()
        self.app.call_from_thread(self._apply_image_gen_save_result, ok, warnings)

    async def _apply_image_gen_save_result(
        self, saved: bool, warnings: list[str]
    ) -> None:
        if not saved:
            message = "Failed to save Image Gen defaults."
            self._set_static_text("#settings-imagegen-save-result", message)
            self.app.notify(message, severity="error")
            return
        self._settings_drafts.pop(SettingsCategoryId.IMAGE_GENERATION, None)
        # Qodo PR #901 fix 3: the save just changed the on-disk truth --
        # invalidate the cached raw-section baseline (see
        # `_image_gen_raw_section_cache`'s docstring).
        self._image_gen_raw_section_cache = None
        message = "Image Gen defaults saved."
        if warnings:
            message = f"{message} {' '.join(warnings)}"
        try:
            panel = self.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        except QueryError:
            panel = None
        if panel is not None:
            panel.overlay = {}
            self._queue_image_gen_select_suppression({})
            await panel.recompose()
            self._set_static_text("#settings-imagegen-save-result", message)
            # A fresh panel mounts its Test buttons enabled by default; if a
            # probe is still in flight (Save clicked mid-probe), re-assert
            # the disabled state on the newly-mounted buttons rather than
            # letting them render as clickable while ignored.
            if self._image_gen_probe_in_flight:
                self._image_gen_set_test_buttons_disabled(True)
        self._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)
        self.app.notify(message, severity="warning" if warnings else "information")

    async def _handle_image_gen_revert(self) -> None:
        self._settings_drafts.pop(SettingsCategoryId.IMAGE_GENERATION, None)
        # Qodo PR #901 fix 3: revert is the third of the exactly-three
        # invalidation points (see `_image_gen_raw_section_cache`'s
        # docstring) -- the file itself doesn't change on revert, but
        # this keeps the cache's lifetime scoped strictly to "since the
        # last time the draft was known-consistent with disk" rather
        # than silently spanning across a discard-and-restart edit.
        self._image_gen_raw_section_cache = None
        try:
            panel = self.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        except QueryError:
            panel = None
        if panel is not None:
            panel.overlay = {}
            self._queue_image_gen_select_suppression({})
            await panel.recompose()
            self._set_static_text("#settings-imagegen-save-result", "")
            # See the matching comment in _apply_image_gen_save_result.
            if self._image_gen_probe_in_flight:
                self._image_gen_set_test_buttons_disabled(True)
        self._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)

    @on(Button.Pressed)
    async def handle_image_gen_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(getattr(event.button, "id", "") or "")
        if button_id == "settings-imagegen-save":
            event.stop()
            self._handle_image_gen_save()
            return
        if button_id == "settings-imagegen-revert":
            event.stop()
            await self._handle_image_gen_revert()
            return
        test_prefix = "settings-imagegen-test-"
        if button_id.startswith(test_prefix):
            backend_id = button_id[len(test_prefix):]
            if backend_id in IMAGE_GEN_BACKEND_IDS:
                event.stop()
                self._handle_image_gen_test(backend_id)
            return
        prefix = "settings-imagegen-clear-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        remainder = button_id[len(prefix):]
        for backend_id in IMAGE_GEN_BACKEND_IDS:
            backend_prefix = f"{backend_id}-"
            if remainder.startswith(backend_prefix):
                toml_key = remainder[len(backend_prefix):]
                self._handle_image_gen_clear(backend_id, toml_key)
                return

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
        if category == SettingsCategoryId.THEME:
            return "Use the editor's Apply/Save/Reset buttons to manage themes."
        if category == SettingsCategoryId.SPLASH_SCREEN:
            return "Splash defaults are saved automatically."
        if category == SettingsCategoryId.WORKSPACES:
            return (
                "Immediate actions: workspace changes apply as you make them; "
                "there is no draft to save or revert."
            )
        if category == SettingsCategoryId.INTERNAL_PROMPTS:
            return "Use each prompt's Save / Reset buttons in the editor to manage overrides."
        if category == SettingsCategoryId.IMAGE_GENERATION:
            if self._category_has_unsaved_changes(category):
                return "Guided edits: use the panel's own Save/Revert controls below."
            return "Guided edits: change a backend or generation default first."
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
            return f"Guided edits: read-only here; open {contract.owner_destination}."
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

    @staticmethod
    def _guided_action_label(base: str, *, dirty: bool) -> str:
        """Save/Revert label with a text-carried inert-state annotation.

        Disabled buttons differed from enabled ones only by dimming
        (task-1582); in the clean state the label itself says why the pair
        is inert. A dirty-but-invalid draft keeps the plain label -- the
        guided-action state row explains the validation block there.

        Args:
            base: The plain label, e.g. "Save (s)".
            dirty: Whether the active category has unsaved changes.

        Returns:
            The plain label when dirty, otherwise the annotated form.
        """
        return base if dirty else f"{base} — no changes"

    def _update_guided_action_widgets(self) -> None:
        category = self._active_category_id()
        actions_enabled = self._guided_actions_enabled(category)
        dirty = self._category_has_unsaved_changes(category)
        self._set_static_text(
            "#settings-guided-action-state", self._guided_action_message(category)
        )
        for selector, base in (
            ("#settings-save-category", "Save (s)"),
            ("#settings-revert-category", "Revert (r)"),
        ):
            try:
                button = self.query_one(selector, Button)
            except QueryError:
                continue
            button.disabled = not actions_enabled
            button.label = self._guided_action_label(base, dirty=dirty)

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
        active = (
            summary.category.value == self.active_category
            if is_active is None
            else is_active
        )
        dirty_marker = ""
        if self._category_has_unsaved_changes(summary.category):
            dirty_marker = " *"
        elif summary.category == SettingsCategoryId.THEME and self.theme_editor_modified:
            dirty_marker = " *"
        # task-1563: view-only stub categories are full nav peers whose whole
        # page says "edit elsewhere" -- badge them in the rail so a third of
        # the navigation stops masquerading as editable surface.
        view_marker = ""
        try:
            if not self._ownership_record(summary.category).writes_allowed:
                view_marker = " (view)"
        except Exception:
            view_marker = ""
        return f"{'> ' if active else '  '}{summary.title}{view_marker}{dirty_marker}"

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
            self.query_one("#settings-selected-category-draft-status", Static).update(
                status
            )
        except QueryError:
            pass
        self._update_category_state_banner(category)
        try:
            category_status = (
                "Unsaved"
                if has_unsaved_changes
                else self._category_summary_by_id(category).status
            )
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
        if category is SettingsCategoryId.IMAGE_GENERATION:
            # Image Gen's Save/Revert live INSIDE the panel (not the generic
            # top guided-action bar, excluded above like THEME/INTERNAL_
            # PROMPTS) -- toggle them here so every staging path updates
            # them through this one shared draft-status refresh, matching
            # the sibling idiom rather than a parallel mechanism.
            for button_id in ("settings-imagegen-save", "settings-imagegen-revert"):
                try:
                    self.query_one(f"#{button_id}", Button).disabled = (
                        not has_unsaved_changes
                    )
                except QueryError:
                    pass

    def _category_summary_by_id(
        self, category: SettingsCategoryId
    ) -> SettingsCategorySummary:
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
            # task-1584: a match starting at a word boundary outranks a
            # mid-word substring hit -- "rag" must surface Library/RAG
            # before Storage (sto-RAG-e), which previously tied on tier
            # and won on list index.
            if re.search(rf"(?<![a-z0-9]){re.escape(query)}", primary_haystack):
                return 0
            return 1
        secondary_haystack = " ".join(
            (
                summary.description,
                self._category_status(summary),
            )
        ).lower()
        if query in secondary_haystack:
            return 2
        # task-1564: last tier -- the category's owned config keys. The Scope
        # Inspector already publishes them; indexing them lets "/" find the
        # category that OWNS a setting instead of forcing a 22-item scan.
        try:
            owned = " ".join(
                self._ownership_record(summary.category).owns_config_sections
            ).lower()
        except Exception:
            owned = ""
        if owned and query in owned:
            return 3
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
        summaries_by_id = {
            summary.category: summary for summary in self._category_summaries()
        }
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
                    button = self.query_one(
                        f"#settings-category-{summary.category.value}", Button
                    )
                    button.display = is_visible
                    button.remove_class("settings-primary-search-match")
                    button.remove_class("settings-secondary-search-match")
                    # task-1584 rescaled tiers: 0/1 are both primary
                    # (word-boundary vs substring); 2 is description/status.
                    if query and rank in (0, 1):
                        button.add_class("settings-primary-search-match")
                    elif query and rank == 2:
                        button.add_class("settings-secondary-search-match")
                except QueryError:
                    pass
            try:
                self.query_one(
                    f"#{self._category_group_dom_id(group_title)}", Static
                ).display = group_visible
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
        if (
            category is SettingsCategoryId.APPEARANCE
            and self._category_has_unsaved_changes(category)
        ):
            validation = self._appearance_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if (
            category is SettingsCategoryId.LIBRARY_RAG
            and self._category_has_unsaved_changes(category)
        ):
            validation = self._library_rag_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if (
            category is SettingsCategoryId.STORAGE
            and self._category_has_unsaved_changes(category)
        ):
            validation = self._storage_validation_result()
            if not validation.valid:
                return f"State: Needs correction | {validation.message}"
        if self._category_has_unsaved_changes(category):
            return (
                "State: Unsaved changes | Save or Revert before leaving this category."
            )
        if category is SettingsCategoryId.ADVANCED_CONFIG:
            return "State: Guarded | Save blocked until the current text validates; backup created before overwrite."
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            return "State: Shared with Console"
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            return "State: Console scoped | Changes affect global Console fallbacks after save."
        if category is SettingsCategoryId.LIBRARY_RAG:
            return "State: Library scoped | Defaults affect future Library/RAG retrieval and display."
        if category is SettingsCategoryId.IMAGE_GENERATION:
            return "State: Image Gen scoped | Defaults affect future Console image generations."
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
                "State: View only | "
                f"Manage this in {contract.owner_destination}."
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
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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
                "Appearance defaults staged."
                if validation.valid
                else validation.message
            )
        else:
            self._appearance_result = "Appearance defaults match loaded values."
        self._set_static_text(
            "#settings-appearance-save-result", self._appearance_result
        )
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
            options.append(
                (theme_name.replace("_", " ").replace("-", " ").title(), theme_name)
            )
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
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
        draft.set_value(
            "collapse_large_pastes",
            self._loaded_collapse_large_pastes_enabled(),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _stage_console_default_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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

    def _normalise_console_max_parallel_runs(self, value: object) -> int:
        # Parallel-agents spec S4 (task-5): integer, >= 1, no upper bound --
        # mirrors ConsoleChatController.max_parallel_runs' own floor.
        text_value = str(value).strip()
        if not text_value.isdigit() or int(text_value) < MIN_CONSOLE_MAX_PARALLEL_RUNS:
            raise ValueError(
                "Max parallel agent runs must be an integer of at least "
                f"{MIN_CONSOLE_MAX_PARALLEL_RUNS}."
            )
        return int(text_value)

    def _stage_console_max_parallel_runs_value(self, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
        try:
            staged_value: object = self._normalise_console_max_parallel_runs(value)
        except ValueError:
            staged_value = str(value)
        draft.set_value(
            "max_parallel_runs",
            self._loaded_console_max_parallel_runs(),
            staged_value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _normalise_tool_result_display_chars(self, value: object) -> int:
        # TASK-870: same bounded-integer shape as
        # _normalise_paste_collapse_threshold, with this setting's own
        # documented min/max (see config.DEFAULT_CONSOLE_TOOL_RESULT_
        # DISPLAY_CHARS's docstring for why 2000 is the ceiling).
        text_value = str(value).strip()
        if not text_value or not text_value.isdigit():
            raise ValueError("Tool result display cap must be a whole number.")
        if not validate_number_range(
            text_value,
            min_val=MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
            max_val=MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
        ):
            raise ValueError(
                "Tool result display cap must be between "
                f"{MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS} and "
                f"{MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS}."
            )
        return int(text_value)

    def _stage_tool_result_display_chars_value(self, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
        try:
            staged_value: object = self._normalise_tool_result_display_chars(value)
        except ValueError:
            staged_value = str(value)
        draft.set_value(
            "tool_result_display_chars",
            self._loaded_tool_result_display_chars(),
            staged_value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _console_behavior_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        """Focused-field guidance for Console Behavior (task-5).

        Only the "Max parallel agent runs" and (TASK-870) "Tool result
        display cap" fields have dedicated guidance today; other Console
        Behavior fields keep the always-visible "Control guide" static
        block rendered in `_render_category_impact_pane` instead of
        per-field guidance.
        """
        if self._active_settings_field_id == "settings-console-max-parallel-runs":
            return (
                (
                    "Purpose",
                    "How many agent runs may be in flight at once, across all tabs.",
                ),
                (
                    "Consequences",
                    "Each concurrent run holds a provider generation, its own tool "
                    "activity, and memory for its transcript. Local providers "
                    "(llama.cpp) typically serialize or slow under concurrent "
                    "generations; high values can exhaust provider slots, rate "
                    "limits, or RAM. Raise it as far as you like - the app "
                    "enforces no ceiling.",
                ),
                ("Saved as", "console.max_parallel_runs"),
                (
                    "Applies",
                    "Applies to new sends on save; running agents are never "
                    "stopped by lowering it.",
                ),
            )
        if self._active_settings_field_id == "settings-console-tool-result-display-chars":
            return (
                (
                    "Purpose",
                    "How much of an agent tool result the Console SHOWS you -- in "
                    "the live run rail, the transcript's tool-call markers, and a "
                    "resumed/historical run's step summaries.",
                ),
                (
                    "Consequences",
                    "This is NOT the same limit as max_tool_result_chars ([agents] "
                    "config, default 16,000), which caps what the MODEL saw and "
                    "stays fixed regardless of this setting. Raising this display "
                    f"cap past {MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS} cannot show "
                    "more -- the engine itself only records up to that many "
                    "characters per step. Use \"View full log\" on a run (Agent "
                    "rail) to read everything the model actually saw, beyond any "
                    "cap here.",
                ),
                ("Saved as", "console.tool_result_display_chars"),
                (
                    "Applies",
                    "Applies to newly rendered steps immediately on save -- no "
                    "restart needed. Steps already on screen keep their rendered "
                    "text until the transcript next redraws them.",
                ),
            )
        return (
            ("Purpose", "Focus a Console Behavior field for setting-specific guidance."),
            ("Consequences", "No field-specific guidance is active right now."),
            ("Saved as", "varies by field"),
            ("Applies", "varies by field"),
        )

    def _refresh_console_behavior_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.CONSOLE_BEHAVIOR:
            return
        for index, (label, value) in enumerate(
            self._console_behavior_field_guidance_rows()
        ):
            self._set_static_text(
                f"#settings-console-behavior-field-guide-{index}",
                f"{label}: {value}",
            )

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

    def _library_rag_edits_suppressed(self) -> bool:
        """Whether every `handle_library_rag_*_changed` handler must
        early-return WITHOUT staging a draft.

        Two independent reasons: (1) `_syncing_library_rag_defaults` -- a
        programmatic widget resync is currently writing values, not the
        user; (2) Task 4 (541 v2 UX AC1) -- the editor is showing a
        profile-picker PREVIEW (`_rag_preview_profile_id` is not None),
        which is READ-ONLY by design: drafts belong to the active profile
        only, and preview must never create or mutate one.
        """
        return (
            self._syncing_library_rag_defaults
            or self._rag_preview_profile_id is not None
        )

    def _stage_library_rag_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.LIBRARY_RAG
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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
                "Library/RAG defaults staged."
                if validation.valid
                else validation.message
            )
        else:
            self._library_rag_result = "Library/RAG defaults match last loaded values."
        self._set_static_text(
            "#settings-library-rag-save-result", self._library_rag_result
        )
        self._update_library_rag_preview()
        self._update_library_rag_validation_classes()
        self._update_library_rag_soft_warning()
        self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)

    def _sync_library_rag_widgets(
        self,
        values: Mapping[str, object] | None = None,
        *,
        field_disabled: bool | None = None,
    ) -> None:
        """Refresh the editor fields (Search/Embedding/Chunking/Vector
        store/Reranking) imperatively (no recompose).

        Args:
            values: Explicit field values to render. Defaults to the ACTIVE
                profile's raw loaded values (pre-task-4 behaviour, still
                what every set-active/clone/rename/delete/save resync below
                relies on) -- pass `self._library_rag_setting_values()` for
                a DRAFT-AWARE render (Task 4: restoring the active
                profile's editor after a profile-picker preview, where a
                staged draft must survive the round-trip).
            field_disabled: When given, forces EVERY editor field's
                disabled state, not just the reranker Inputs (which always
                follow `_library_rag_rerank_field_state`). Task 4 uses
                `True` for a profile PREVIEW (always read-only regardless
                of the previewed profile's own read_only flag) and the
                ACTIVE profile's `read_only` flag when restoring the
                ordinary editor after a preview. `None` (default) leaves
                those fields' disabled state untouched -- every pre-task-4
                caller already relies on that (driven separately by
                `_sync_library_rag_profile_widgets`).
        """
        if values is None:
            values = self._library_rag_loaded_values()
        self._syncing_library_rag_defaults = True
        try:
            try:
                self.query_one(
                    "#settings-library-rag-search-mode", Select
                ).value = normalise_library_rag_search_mode(
                    values["default_search_mode"]
                )
                self.query_one(
                    "#settings-library-rag-citation-style", Select
                ).value = normalise_library_rag_citation_style(values["citation_style"])
                self.query_one(
                    "#settings-library-rag-include-citations", Checkbox
                ).value = bool(values["include_citations"])
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
                self.query_one(
                    "#settings-library-rag-chunking-method", Select
                ).value = normalise_library_rag_chunking_method(
                    values["chunking_method"]
                )
                self.query_one(
                    "#settings-library-rag-distance-metric", Select
                ).value = normalise_library_rag_distance_metric(
                    values["distance_metric"]
                )
                self.query_one(
                    "#settings-library-rag-enable-reranking", Checkbox
                ).value = bool(values["enable_reranking"])
                for selector, key in (
                    ("#settings-library-rag-embedding-model", "embedding_model"),
                    ("#settings-library-rag-embedding-device", "embedding_device"),
                    (
                        "#settings-library-rag-embedding-batch-size",
                        "embedding_batch_size",
                    ),
                    (
                        "#settings-library-rag-embedding-max-length",
                        "embedding_max_length",
                    ),
                    ("#settings-library-rag-chunk-size", "chunk_size"),
                    ("#settings-library-rag-chunk-overlap", "chunk_overlap"),
                    ("#settings-library-rag-reranker-model", "reranker_model"),
                    ("#settings-library-rag-reranker-top-k", "reranker_top_k"),
                ):
                    self.query_one(selector, Input).value = str(values[key])
                resolved_field_disabled = (
                    field_disabled
                    if field_disabled is not None
                    else bool(active_profile_info()["read_only"])
                )
                self._apply_library_rag_rerank_field_state(
                    rerank_enabled=bool(values["enable_reranking"]),
                    field_disabled=resolved_field_disabled,
                )
                if field_disabled is not None:
                    for key in _LIBRARY_RAG_READ_LOCK_FIELD_KEYS:
                        selector = self._library_rag_field_selector(key)
                        if selector is None:
                            continue
                        self.query_one(selector).disabled = field_disabled
                    for selector in _LIBRARY_RAG_READ_LOCK_CHECKBOX_SELECTORS:
                        self.query_one(selector, Checkbox).disabled = field_disabled
            except QueryError:
                pass
        finally:
            self._syncing_library_rag_defaults = False
        self._set_static_text(
            "#settings-library-rag-save-result", self._library_rag_result
        )
        self._update_library_rag_preview()
        self._update_library_rag_validation_classes()
        self._update_library_rag_soft_warning()

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
        if message.startswith("Embedding model"):
            return "embedding_model"
        if message.startswith("Embedding max length"):
            return "embedding_max_length"
        if message.startswith("Chunking method"):
            return "chunking_method"
        if message.startswith("Rerank results"):
            return "reranker_top_k"
        # M3 (SP3 final review): chunk_size, chunk_overlap, distance_metric,
        # and embedding_batch_size are validated by RAGConfig.validate()
        # (simplified/config.py), routed through the adapter's
        # hard_config_errors() (see settings_library_rag_defaults.py's
        # validate_library_rag_defaults) -- so `message` here is RAGConfig's
        # own literal, lowercase/snake_case prose ("chunk_overlap must be
        # less than chunk_size", "embedding batch_size must be positive",
        # "Unknown distance metric: ..."), NOT this function's Title Case
        # field-label convention. The four startswith() checks this replaced
        # (`"Chunk size"`, `"Chunk overlap"`, `"Distance metric"`, `"Embedding
        # batch size"`) never matched that wording, so a hard error on any of
        # these fields blocked Save without ever highlighting the field red.
        # Matched by case-insensitive substring against RAGConfig's actual
        # wording instead, since that message can't be reworded here without
        # drifting from the single source of truth. Order matters: "chunk_overlap
        # must be less than chunk_size" contains BOTH substrings, so
        # chunk_overlap must be checked first.
        lowered = message.lower()
        if "chunk_overlap" in lowered:
            return "chunk_overlap"
        if "chunk_size" in lowered:
            return "chunk_size"
        if "batch_size" in lowered:
            return "embedding_batch_size"
        if "distance metric" in lowered:
            return "distance_metric"
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
            "embedding_model": "#settings-library-rag-embedding-model",
            "embedding_device": "#settings-library-rag-embedding-device",
            "embedding_batch_size": "#settings-library-rag-embedding-batch-size",
            "embedding_max_length": "#settings-library-rag-embedding-max-length",
            "chunk_size": "#settings-library-rag-chunk-size",
            "chunk_overlap": "#settings-library-rag-chunk-overlap",
            "chunking_method": "#settings-library-rag-chunking-method",
            "distance_metric": "#settings-library-rag-distance-metric",
            "reranker_model": "#settings-library-rag-reranker-model",
            "reranker_top_k": "#settings-library-rag-reranker-top-k",
        }.get(key)

    def _update_library_rag_validation_classes(self) -> None:
        # Task 4 (541 v2 UX AC1): the fields currently on screen may be
        # showing a PREVIEWED (different) profile's values while this
        # method's own validation is always computed from the ACTIVE
        # profile's draft -- highlighting a previewed field red over an
        # unrelated active-profile draft error would be misleading, so
        # nothing is ever highlighted while merely browsing a preview.
        invalid_key = (
            None
            if self._rag_preview_profile_id is not None
            else self._library_rag_invalid_field_key()
        )
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
            "embedding_model",
            "embedding_device",
            "embedding_batch_size",
            "embedding_max_length",
            "chunk_size",
            "chunk_overlap",
            "chunking_method",
            "distance_metric",
            "reranker_model",
            "reranker_top_k",
        ):
            selector = self._library_rag_field_selector(key)
            if selector is None:
                continue
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.set_class(key == invalid_key, "settings-invalid-input")

    def _update_library_rag_soft_warning(self) -> None:
        """Refresh the Reranking advisory Static (never gates Save --
        see _library_rag_soft_warnings)."""
        warnings = self._library_rag_soft_warnings()
        try:
            warning_widget = self.query_one(
                "#settings-library-rag-reranker-warning", Static
            )
        except QueryError:
            return
        warning_widget.update(" / ".join(warnings))
        warning_widget.display = bool(warnings)

    def _stage_storage_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.STORAGE
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
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
            citation_style = normalise_library_rag_citation_style(
                values["citation_style"]
            )
            citation_label = {
                "inline": "Inline citations",
                "footnote": "Footnote citations",
                "none": "No citations",
            }[citation_style]
        return (
            f"{mode_label} search | {values['default_top_k']} results | {citation_label}",
            (
                f"Keyword {values['fts_top_k']} | Vector {values['vector_top_k']} | "
                f"Hybrid balance {values['hybrid_alpha']} | Min score {values['score_threshold']}"
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
        if not text_value.isdecimal() or not validate_number_range(
            text_value,
            min_val=MIN_CONSOLE_BACKGROUND_FPS,
            max_val=MAX_CONSOLE_BACKGROUND_FPS,
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
        active_source = str(
            getattr(runtime_state, "active_source", "local") or "local"
        ).lower()
        server_profile_value = getattr(runtime_state, "active_server_id", None)
        server_profile_id = str(server_profile_value or "").strip() or None
        source_authority = (
            "server" if active_source == "server" and server_profile_id else "local"
        )
        authenticated_principal_id = None
        if source_authority == "server":
            server_context_provider = getattr(
                self.app_instance, "server_context_provider", None
            )
            get_active_context = getattr(
                server_context_provider, "get_active_context", None
            )
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
            workspace_scope = (
                str(getattr(workspace, "workspace_id", "") or "").strip() or None
            )
        return {
            "server_profile_id": server_profile_id
            if source_authority == "server"
            else None,
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
        list_states = getattr(
            sync_scope_service, "list_write_sync_promotion_states", None
        )
        if callable(list_states):
            try:
                return tuple(
                    list_states(
                        domains=list(labels),
                        surface_labels=labels,
                        server_profile_id=active_scope["server_profile_id"],
                        authenticated_principal_id=active_scope[
                            "authenticated_principal_id"
                        ],
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
        runtime_policy = getattr(self.app_instance, "runtime_policy", None)
        return runtime_policy.state if runtime_policy is not None else None

    def _active_server_profile_label(self) -> str:
        state = self._runtime_source_state()
        source = (
            str(
                getattr(
                    state,
                    "active_source",
                    getattr(self.app_instance, "current_runtime_backend", "local"),
                )
                or "local"
            )
            .strip()
            .lower()
        )
        active_server_id = str(
            getattr(
                state,
                "active_server_id",
                getattr(self.app_instance, "active_server_id", None),
            )
            or ""
        ).strip()
        server_label = str(getattr(state, "last_known_server_label", "") or "").strip()
        if source == "server" and active_server_id:
            if server_label and server_label != active_server_id:
                return f"{server_label} ({active_server_id})"
            return active_server_id
        if active_server_id:
            label = server_label or active_server_id
            return (
                f"{label} ({active_server_id}) configured; current source is {source}"
            )
        return "local-only; no active server profile"

    def _local_server_authority_label(self) -> str:
        get_source = getattr(
            self.app_instance, "get_authoritative_runtime_source", None
        )
        if callable(get_source):
            try:
                source = str(get_source() or "local").strip().lower()
            except Exception:
                source = "local"
        else:
            state = self._runtime_source_state()
            source = (
                str(getattr(state, "active_source", "local") or "local").strip().lower()
            )
        if source not in {"local", "server"}:
            source = "local"
        return f"{source}; Settings is read-only"

    def _sync_safety_label(
        self, states: tuple[SyncPromotionState, ...] | None = None
    ) -> str:
        sync_states = states if states is not None else self._sync_safety_states()
        if not sync_states:
            return (
                "Sync: unavailable; owning sync surfaces control dry-run and recovery"
            )
        # TASK-719: each surface's sync_label already begins with "Sync:";
        # joining them verbatim produced "Collections: Sync: dry-run only".
        return "; ".join(
            f"{state.surface_label}: "
            f"{str(state.sync_label).removeprefix('Sync:').strip() or state.sync_label}"
            for state in sync_states
        )

    def _sync_recovery_label(
        self, states: tuple[SyncPromotionState, ...] | None = None
    ) -> str:
        sync_states = states if states is not None else self._sync_safety_states()
        blocking_statuses = {
            "rollback-required",
            "conflict",
            "attention-required",
            "review-gated",
        }
        selected = next(
            (state for state in sync_states if state.status in blocking_statuses),
            sync_states[0] if sync_states else None,
        )
        if selected is None:
            return "Open the owning sync surface for dry-run setup and recovery."
        return selected.primary_recovery

    def _active_workspace_record(self) -> object | None:
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
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
            workspace_name = (
                str(getattr(workspace, "name", "") or "").strip() or workspace_id
            )
            authority = self._enum_display_value(
                getattr(workspace, "authority", None),
                "local-only",
            )
            sync_status = self._enum_display_value(
                getattr(workspace, "sync_status", None),
                "not-configured",
            )
            if workspace_id:
                # TASK-719: the row label already says "Workspace default";
                # repeating "Workspace:"/"Authority:"/"Sync:" as value
                # prefixes read like debug output.
                return (
                    f"{workspace_name} ({workspace_id}); "
                    f"authority {authority}; sync {sync_status}"
                )
        store = getattr(self.app_instance, "console_chat_store", None)
        context = getattr(store, "workspace_context", None)
        active_workspace_id = str(
            getattr(context, "active_workspace_id", "") or ""
        ).strip()
        if active_workspace_id and active_workspace_id != "global":
            return (
                f"{active_workspace_id}; Console context active; "
                "Library browse/search remains global"
            )
        # TASK-719: name the concrete owning surfaces instead of a vague list.
        return (
            "Local Default; switch in Console (Alt+W), "
            "manage in Settings > Workspaces"
        )

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
        pending_copy = (
            "; ".join(
                f"{domain}: {count}"
                for domain, count in preview.pending_by_domain.items()
            )
            or "none"
        )
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
                (
                    "Manual sync preview",
                    "Manual Sync requires an active server profile.",
                ),
                ("Pending outgoing", "none"),
            )
        try:
            preview = control.preview(
                server_profile_id=server_profile_id,
                authenticated_principal_id=sync_scope["authenticated_principal_id"],
                workspace_scope=sync_scope["workspace_scope"],
            )
        except Exception as exc:
            logger.warning(
                "Failed to build Settings manual sync preview.", exc_info=True
            )
            return (
                ("Manual sync status", "blocked"),
                (
                    "Manual sync preview",
                    f"Manual Sync preview unavailable: {type(exc).__name__}",
                ),
                ("Pending outgoing", "unknown"),
            )
        return self._manual_sync_rows_from_preview(preview)

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
        return (
            "; ".join(
                f"{domain}: {count}" for domain, count in pending_by_domain.items()
            )
            or "none"
        )

    @work(exclusive=True, thread=True, group="settings-sync-rows-refresh")
    def _refresh_sync_rows(self) -> None:
        """Compute BOTH sync row sets off-thread, apply in ONE hop (task-290).

        The previous shape ran two independent thread workers whose
        completions each set a ``recompose=True`` reactive at its own
        nondeterministic moment -- two full-screen recomposes shortly after
        mount (the "mount storm"). Computing both row sets in one pass and
        applying them in a single ``call_from_thread`` coalesces the storm:
        Textual's ``refresh(recompose=True)`` is flag-based, so two reactive
        assignments in the same message-loop slot recompose ONCE.

        Own worker group (not "settings-manual-sync-preview"): a solo manual
        refresh racing this one is benign last-wins between two fresh
        snapshots, whereas sharing the group would let a solo refresh CANCEL
        an in-flight combined pass and silently drop the handoff update.
        """
        try:
            handoff_rows = self._server_sync_workspace_handoff_rows()
        except Exception:
            logger.warning(
                "Failed to refresh Settings server/sync/workspace/handoff rows.",
                exc_info=True,
            )
            handoff_rows = self._server_sync_workspace_handoff_loading_rows()
        try:
            manual_rows = self._manual_sync_rows()
        except Exception:
            logger.warning(
                "Failed to refresh Settings manual sync rows.", exc_info=True
            )
            manual_rows = self._manual_sync_loading_rows()
        self.app.call_from_thread(self._apply_sync_rows, handoff_rows, manual_rows)

    def _apply_sync_rows(
        self,
        handoff_rows: tuple[tuple[str, str], ...],
        manual_rows: tuple[tuple[str, str], ...],
    ) -> None:
        """Apply both row sets in one hop, preserving focus across the
        recompose they trigger.

        The recompose replaces every child, dropping whatever the user (or a
        route-targeted navigation like ``_apply_navigation_provider_context``)
        had focused -- restore it by id once the fresh children mount.
        """
        changed = (
            handoff_rows != self.server_sync_workspace_handoff_rows
            or manual_rows != self.manual_sync_rows
        )
        focused = self.app.focused
        focused_id = (
            focused.id
            if focused is not None and focused.screen is self and focused.id
            else None
        )
        self.server_sync_workspace_handoff_rows = handoff_rows
        self.manual_sync_rows = manual_rows
        if changed:
            # Scheduled on ANY change, not only when something was focused:
            # the pending-intent case is precisely a focus that has NOT
            # landed yet (Widget.focus defers via call_later).
            self.call_after_refresh(self._restore_focus_after_sync_rows, focused_id)

    def _restore_focus_after_sync_rows(self, widget_id: str | None) -> None:
        pending = self._pending_navigation_focus_selector
        self._pending_navigation_focus_selector = None
        if self.app.focused is not None:
            # A post-recompose focus already exists (user action or a
            # late-landing deliberate focus) -- honor it, drop the intent.
            return
        if pending:
            # A navigation focus intent never landed (the recompose destroyed
            # its target before the deferred set_focus processed) -- re-issue
            # it against the fresh children. call_later is FIFO, so this
            # focus is queued after any stale earlier intents and wins.
            try:
                self.query_one(pending).focus()
                return
            except QueryError:
                pass
        if widget_id:
            try:
                self.query_one(f"#{widget_id}").focus()
            except QueryError:
                pass

    @work(exclusive=True, thread=True, group="settings-manual-sync-preview")
    def _refresh_manual_sync_rows(self) -> None:
        try:
            rows = self._manual_sync_rows()
        except Exception:
            logger.warning(
                "Failed to refresh Settings manual sync rows.", exc_info=True
            )
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
                    (
                        "Manual sync result",
                        "Manual Sync requires an active server profile.",
                    ),
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
        """Return the active CLI config path.

        Thin wrapper kept only so call sites in this module read naturally
        (``self._config_path()``); it must delegate to the shared effective-
        path resolver rather than re-spell the override/validate logic,
        since a second hand-copy would drift the moment either
        implementation changed -- see task-851 review finding 5.
        """
        return validate_path_simple(
            get_cli_config_path(),
            require_exists=False,
        )

    def _config_writable_status(self) -> str:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"invalid path - {redact_secret_text(str(exc))}"
        target = config_path if config_path.exists() else config_path.parent
        writable = (
            os.access(target, os.W_OK)
            if target.exists()
            else os.access(target.parent, os.W_OK)
        )
        return "writable" if writable else "not writable"

    def _config_path_overview_value(self) -> str:
        try:
            config_path = self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"invalid path - {redact_secret_text(str(exc))}"
        source = (
            "Override config" if os.environ.get("TLDW_CONFIG_PATH") else "User config"
        )
        filename = config_path.name or "config.toml"
        return f"{source}: {filename} ({self._config_writable_status()})"

    def _raw_config_text(self) -> str:
        try:
            self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"# Unable to use config path: {redact_secret_text(str(exc))}\n"
        try:
            return SettingsConfigAdapter().read_serialized()
        except OSError as exc:
            return f"# Unable to read config: {type(exc).__name__}"

    @staticmethod
    def _deep_merge_config_values(base: dict, update: Mapping) -> dict:
        merged = copy.deepcopy(base)
        for key, value in update.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key] = SettingsScreen._deep_merge_config_values(
                    merged[key], value
                )
            else:
                merged[key] = value
        return merged

    def _read_cli_config_without_writes(self) -> dict:
        try:
            self._config_path()
        except (OSError, RuntimeError, ValueError):
            return copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
        try:
            return SettingsConfigAdapter().load()
        except (OSError, tomllib.TOMLDecodeError):
            return copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)

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
        default_user = DEFAULT_CONFIG_FROM_TOML.get("general", {}).get(
            "users_name", "default_user"
        )
        user_name = self._read_cli_config_value_without_writes(
            "general", "users_name", default_user
        )
        safe_user_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_name))
        return safe_user_name if safe_user_name else "default_user"

    def _configured_user_data_dir_path(self) -> Path:
        """Read-only mirror of get_user_data_dir()'s resolution logic (minus
        the mkdir side effect), so the Settings display never diverges from
        the path the app actually uses. Uses _default_base_data_dir() (the
        same call-time HOME resolution as get_user_data_dir()'s fallback)
        rather than the import-time-frozen BASE_DATA_DIR_CLI constant --
        those two can disagree, e.g. under test-isolated HOME (task-519
        review)."""
        configured_data_dir = self._read_cli_config_value_without_writes(
            "paths", "data_dir", None
        )
        if configured_data_dir is None:
            configured_data_dir = self._read_cli_config_value_without_writes(
                "Paths", "data_dir", None
            )
        base_data_dir = (
            Path(str(configured_data_dir)).expanduser()
            if configured_data_dir
            else _default_base_data_dir()
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
        for (
            label,
            attr_name,
            fallback_factory,
            _directory,
        ) in self._storage_path_entries():
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

    def _storage_path_status(
        self, label: str, path_value: object, *, directory: bool
    ) -> str:
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
        for (
            label,
            attr_name,
            fallback_factory,
            directory,
        ) in self._storage_path_entries():
            status_label = label if directory else f"{label} parent"
            try:
                value = self._storage_path_value(attr_name, fallback_factory)
            except Exception as exc:
                rows.append(f"{status_label}: invalid - {redact_secret_text(str(exc))}")
            else:
                rows.append(
                    self._storage_path_status(status_label, value, directory=directory)
                )
        rows.append("Storage safety: no files were created, moved, or rewritten.")
        return tuple(rows)

    def _storage_check_text(self) -> str:
        return "\n".join(self._storage_check_rows)

    def _update_storage_check_widgets(self) -> None:
        self._set_static_text(
            "#settings-storage-check-result", self._storage_check_text()
        )

    def _apply_storage_check_result(self, rows: tuple[str, ...]) -> None:
        self._storage_check_rows = rows
        self._update_storage_check_widgets()
        self.app.notify("Storage check finished.", severity="information")

    @work(exclusive=True, thread=True)
    def _storage_check_worker(
        self, values: SettingsStorageDefaults | None = None
    ) -> None:
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

    def _privacy_posture_rows(
        self, app_config: object | None = None
    ) -> tuple[str, ...]:
        return build_privacy_posture_rows(self._settings_privacy_posture(app_config))

    def _privacy_check_results(
        self, app_config: object | None = None
    ) -> tuple[str, ...]:
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
        self._set_static_text(
            "#settings-privacy-check-result", self._privacy_check_text()
        )

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

    def _diagnostics_validation_and_reload_results(
        self,
    ) -> tuple[str, str, dict | None]:
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
        self.app.notify(
            "Diagnostics validation and reload finished.", severity="information"
        )

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
            self.query_one(
                "#settings-advanced-save-config", Button
            ).disabled = not self._advanced_save_allowed()
        except QueryError:
            pass

    def _save_advanced_config_text(self, text: str) -> str:
        validation = SettingsConfigAdapter().validate_raw_toml(text)
        if not validation.valid:
            return f"Advanced config save: blocked - {redact_secret_text(validation.message)}"
        if self._advanced_config_validated_text != text:
            return "Advanced config save: blocked - validate current TOML before save"

        try:
            self._config_path()
        except ValueError as exc:
            return f"Advanced config save: failed - {redact_secret_text(str(exc))}"
        try:
            _loaded, backup_path = SettingsConfigAdapter().replace_serialized(text)
            backup_message = (
                "backup: created"
                if backup_path is not None
                else "backup: none (new file)"
            )
            return f"Advanced config save: saved; {backup_message}"
        except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
            return f"Advanced config save: failed - {redact_secret_text(str(exc))}"

    def _read_advanced_backup_preview(self) -> tuple[str, str | None]:
        try:
            self._config_path()
        except (OSError, RuntimeError, ValueError) as exc:
            return (
                f"Advanced config recovery: failed - {redact_secret_text(str(exc))}",
                None,
            )
        try:
            backup_text = SettingsConfigAdapter().read_backup_serialized()
        except FileNotFoundError:
            return (
                "Advanced config recovery: unavailable - no backup found",
                None,
            )
        except (OSError, UnicodeDecodeError) as exc:
            return (
                f"Advanced config recovery: failed - {redact_secret_text(str(exc))}",
                None,
            )
        return (
            "Advanced config recovery: loaded backup preview; validate before save",
            backup_text,
        )

    def _load_advanced_backup_preview(self) -> str:
        result, backup_text = self._read_advanced_backup_preview()
        if backup_text is None:
            return result
        try:
            self.query_one(
                "#settings-advanced-config-editor", TextArea
            ).text = backup_text
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

    def _apply_advanced_validation_result(
        self, text: str, valid: bool, result: str
    ) -> None:
        self._advanced_config_result = result
        self._advanced_config_validated_text = text if valid else None
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )
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

    def _apply_advanced_save_result(
        self, result: str, loaded_config: dict | None
    ) -> None:
        if loaded_config is not None:
            self.app_instance.app_config = loaded_config
        self._advanced_config_result = result
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )
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
                self.query_one(
                    "#settings-advanced-config-editor", TextArea
                ).text = backup_text
            except QueryError:
                final_result = "Advanced config recovery: failed - editor unavailable"
            else:
                self._advanced_config_validated_text = None
                self._update_advanced_validation_status()
        self._advanced_config_result = final_result
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )

    def _provider_readiness_label(self) -> str:
        resolved = self._resolve_provider_model_for_settings()
        provider = str(resolved.provider or "not selected").strip()
        model = str(resolved.model or "not selected").strip()
        if provider and provider != "not selected":
            return (
                f"Provider readiness: {self._provider_display_name(provider)} / {model}"
            )
        return "Provider readiness: needs provider and model"

    def _provider_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)

    def _provider_draft_value(self, key: str):
        draft = self._provider_draft()
        return (
            draft.values.get(key) if draft is not None and key in draft.values else None
        )

    def _resolve_provider_model_for_settings(self):
        draft = self._provider_draft()
        settings_provider = (
            draft.values["provider"]
            if draft is not None and "provider" in draft.values
            else None
        )
        settings_model = (
            draft.values["model"]
            if draft is not None and "model" in draft.values
            else None
        )
        resolved = resolve_effective_provider_model(
            self._chat_defaults(),
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
        resolved = resolve_effective_provider_model(self._chat_defaults())
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
            "model_profile_thinking_budget_tokens": profile.get(
                "thinking_budget_tokens", ""
            ),
            "model_profile_streaming": profile.get("streaming", ""),
        }

    def _provider_setting_values(self) -> dict[str, object]:
        loaded = self._provider_loaded_setting_values()
        draft = self._provider_draft()
        return {
            key: draft.values[key]
            if draft is not None and key in draft.values
            else value
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
                "model_profile_thinking_budget_tokens": profile.get(
                    "thinking_budget_tokens", ""
                ),
                "model_profile_streaming": profile.get("streaming", ""),
            }
        )
        return display_values

    def _clear_navigation_provider_context(self) -> None:
        self._navigation_provider = None
        self._navigation_model = None
        self._navigation_field = None
        self._pending_navigation_focus_selector = None

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
            raise ValueError(
                f"{label} must be between {min_value:.1f} and {max_value:.1f}."
            )
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

    def _normalise_model_profile_thinking_budget_tokens(
        self, value: object
    ) -> int | str:
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
        return (
            provider_config_key(str(provider or "")) in OPENAI_REASONING_PROVIDER_KEYS
        )

    @staticmethod
    def _provider_supports_anthropic_thinking(provider: object) -> bool:
        return (
            provider_config_key(str(provider or "")) in ANTHROPIC_THINKING_PROVIDER_KEYS
        )

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
        """Summarize gated generation controls in one line (task-189).

        Instead of rendering rows of "Unavailable for <provider>" placeholder
        fields, the Generation defaults disclosure shows this single summary
        and hides the dead rows entirely.

        Returns:
            Copy such as ``"Reasoning/Thinking controls: unavailable for
            llama.cpp."`` or ``""`` when every gated control is available.
        """
        provider_label = self._provider_display_name(str(provider or "").strip())
        if not provider_label:
            provider_label = "this provider"
        unavailable: list[str] = []
        if not self._provider_supports_openai_reasoning(provider):
            unavailable.append("Reasoning")
        if not self._provider_supports_anthropic_thinking(provider):
            unavailable.append("Thinking")
        if not unavailable:
            return ""
        return f"{'/'.join(unavailable)} controls: unavailable for {provider_label}."

    @staticmethod
    def _gated_profile_row_classes(supported: bool) -> str:
        """Return input-row classes, hiding gated rows the provider lacks."""
        if supported:
            return "settings-input-row"
        return "settings-input-row settings-gated-profile-hidden"

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
                and provider_config_key(provider_value)
                == provider_config_key(loaded_provider)
            )
            else (
                provider_value
                if provider_value or provider_explicitly_staged
                else loaded_provider
            )
        )
        model = self.query_one("#settings-model-value", Input).value.strip() or str(
            loaded_values["model"]
        )
        endpoint = self.query_one(
            "#settings-provider-endpoint-value", Input
        ).value.strip()
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
        model_profile_frequency_penalty = (
            self._normalise_model_profile_frequency_penalty(
                self.query_one("#settings-model-profile-frequency-penalty", Input).value
            )
        )
        model_profile_reasoning_effort = self._normalise_model_profile_reasoning_effort(
            self.query_one("#settings-model-profile-reasoning-effort", Input).value
        )
        model_profile_reasoning_summary = (
            self._normalise_model_profile_reasoning_summary(
                self.query_one("#settings-model-profile-reasoning-summary", Input).value
            )
        )
        model_profile_verbosity = self._normalise_model_profile_verbosity(
            self.query_one("#settings-model-profile-verbosity", Input).value
        )
        model_profile_thinking_effort = self._normalise_model_profile_thinking_effort(
            self.query_one("#settings-model-profile-thinking-effort", Input).value
        )
        model_profile_thinking_budget_tokens = (
            self._normalise_model_profile_thinking_budget_tokens(
                self.query_one(
                    "#settings-model-profile-thinking-budget-tokens", Input
                ).value
            )
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
        draft = self._settings_drafts.setdefault(
            category, SettingsDraft(category=category)
        )
        if key == "api_key":
            provider = str(
                self._provider_setting_values_mapping().get("provider") or ""
            ).strip()
            original = self._provider_api_key_value(provider)
        else:
            original = self._provider_loaded_setting_values().get(key)
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)
        # TASK-366: any edit to a provider field outdates the last Test result.
        self._mark_provider_test_result_stale()

    def _provider_config_entry(
        self, provider: str
    ) -> tuple[str | None, Mapping[str, object]]:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        api_settings = (
            app_config.get("api_settings", {})
            if isinstance(app_config, Mapping)
            else {}
        )
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
            readiness.api_key_source and readiness.api_key_source.startswith("config:")
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
            return (
                f"API key source: missing; set {readiness.env_var} or paste a local key"
            )
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
        return frozenset(
            entry.readiness_key for entry in self._provider_catalog_entries()
        )

    def _provider_display_name(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        display_name = PROVIDER_DISPLAY_NAMES.get(provider_key)
        if display_name:
            return display_name
        for entry in self._provider_catalog_entries():
            if entry.readiness_key == provider_key:
                return entry.display_name
        return provider

    @staticmethod
    def _provider_catalog_group(entry: ConsoleProviderCatalogEntry) -> str:
        if entry.readiness_key in PROVIDER_CUSTOM_GROUP_KEYS:
            return PROVIDER_GROUP_CUSTOM
        if entry.requires_api_key:
            return PROVIDER_GROUP_CLOUD
        return PROVIDER_GROUP_LOCAL

    def _grouped_provider_catalog_entries(
        self,
    ) -> tuple[ConsoleProviderCatalogEntry, ...]:
        group_rank = {group: rank for rank, group in enumerate(PROVIDER_GROUP_ORDER)}
        return tuple(
            sorted(
                self._provider_catalog_entries(),
                key=lambda entry: (
                    group_rank[self._provider_catalog_group(entry)],
                    self._provider_display_name(entry.readiness_key).lower(),
                ),
            )
        )

    def _provider_select_options(self) -> list[tuple[str, str]]:
        # task-180: options are grouped by ordering (Cloud, then Local, then
        # Custom & legacy aliases) and labelled with human display names only;
        # raw config keys never render as labels. Textual's Select cannot show
        # disabled separator rows, so grouping is conveyed by ordering plus
        # "(legacy alias)" labels. Quick-find comes from the Select overlay's
        # built-in type-to-search (enabled by default).
        options = [
            (self._provider_display_name(entry.readiness_key), entry.readiness_key)
            for entry in self._grouped_provider_catalog_entries()
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
        configured_model = str(
            self._provider_config(provider).get("model") or ""
        ).strip()
        if configured_model and configured_model != "None":
            return configured_model
        return self._provider_catalog_model_default(provider)

    @staticmethod
    def _select_value_text(value: object) -> str:
        # task-565: `Select.NULL` is the real blank sentinel on this Textual
        # version -- `Select.BLANK` doesn't exist, it silently resolves to
        # the unrelated `Widget.BLANK` (`False`), so it never matched here.
        if value is None or value is Select.NULL:
            return ""
        return str(value).strip()

    def _provider_widget_value(self) -> str:
        try:
            provider_select = self.query_one("#settings-provider-value", Select)
            selected_value = self._select_value_text(provider_select.value)
            if selected_value == PROVIDER_MANUAL_SELECT_VALUE:
                try:
                    return self.query_one(
                        "#settings-provider-manual-value", Input
                    ).value.strip()
                except QueryError:
                    return ""
            return selected_value
        except QueryError:
            try:
                return self.query_one("#settings-provider-value", Input).value.strip()
            except QueryError:
                return str(
                    self._provider_setting_values_mapping().get("provider") or ""
                ).strip()

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
            manual_row.set_class(
                not uses_manual_entry, "settings-provider-manual-hidden"
            )
        finally:
            self._syncing_provider_manual = False

    def _provider_catalog_summary(self) -> str:
        # task-180: show grouped display names, never a raw config-key dump.
        grouped: dict[str, list[str]] = {}
        for entry in self._grouped_provider_catalog_entries():
            grouped.setdefault(self._provider_catalog_group(entry), []).append(
                self._provider_display_name(entry.readiness_key)
            )
        parts = [
            f"{group}: {', '.join(grouped[group])}"
            for group in PROVIDER_GROUP_ORDER
            if grouped.get(group)
        ]
        return "Provider catalog | " + " | ".join(parts)

    def _provider_catalog_key_policy(self) -> str:
        entries = self._provider_catalog_entries()
        key_required = sum(1 for entry in entries if entry.requires_api_key)
        keyless = sum(1 for entry in entries if not entry.requires_api_key)
        return f"Credential policy: {key_required} require keys; {keyless} local/keyless providers"

    def _provider_model_defaults(self, provider: str) -> Mapping[str, object]:
        model_defaults = self._provider_config(provider).get("model_defaults", {})
        return model_defaults if isinstance(model_defaults, Mapping) else {}

    def _provider_model_profile(
        self, provider: str, model: str
    ) -> Mapping[str, object]:
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
        next_profile = (
            copy.deepcopy(current_profile)
            if isinstance(current_profile, Mapping)
            else {}
        )
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
            credential_input = self.query_one(
                "#settings-provider-credential-env-var", Input
            )
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
                credential_input.placeholder = self._provider_credential_placeholder(
                    provider
                )
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
            "model_profile_thinking_budget_tokens": profile.get(
                "thinking_budget_tokens", ""
            ),
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
                widget.placeholder = self._model_profile_input_placeholder(
                    provider, draft_key
                )
                widget.value = self._profile_input_value(value) if supported else ""
                # task-189: gated rows are hidden (not rendered as disabled
                # placeholder noise); the disclosure shows one summary line.
                try:
                    row = self.query_one(f"{selector}-row", Horizontal)
                except QueryError:
                    continue
                row.set_class(not supported, "settings-gated-profile-hidden")
        finally:
            self._syncing_provider_model_profile = False
        self._refresh_generation_support_summary(provider)

    def _refresh_generation_support_summary(self, provider: str) -> None:
        """Update the one-line gated-controls summary and its visibility."""
        support_copy = self._provider_generation_support_copy(provider)
        try:
            summary = self.query_one("#settings-provider-generation-support", Static)
        except QueryError:
            return
        summary.update(support_copy)
        summary.set_class(not support_copy, "settings-gated-profile-hidden")

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

    def _provider_endpoint_summary(
        self, provider: str, endpoint: object | None = None
    ) -> str:
        provider_key = provider_config_key(provider)
        endpoint_key = self._provider_endpoint_setting_key(provider)
        endpoint_value = str(
            endpoint
            if endpoint is not None
            else self._provider_endpoint_value(provider)
        ).strip()
        if not provider_key:
            return "Endpoint: provider required before saving"
        if endpoint_value:
            return (
                f"Endpoint: api_settings.{provider_key}.{endpoint_key}={endpoint_value}"
            )
        if provider_key in API_URL_PROVIDER_KEYS:
            return (
                f"Endpoint: api_settings.{provider_key}.{endpoint_key} not configured"
            )
        return (
            f"Endpoint: api_settings.{provider_key}.{endpoint_key} or provider default"
        )

    def _provider_endpoint_display_value(
        self, provider: str, endpoint: object | None = None
    ) -> str:
        provider_key = provider_config_key(provider)
        endpoint_value = str(
            endpoint
            if endpoint is not None
            else self._provider_endpoint_value(provider)
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
            return (
                "Endpoint must start with http:// or https:// and include a valid host."
            )
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
        return (
            bool(provider_config_key(provider))
            and getattr(
                self.app_instance,
                "llm_provider_catalog_scope_service",
                None,
            )
            is not None
        )

    def _provider_test_staged_config(self, provider: str) -> Mapping[str, object]:
        """Return app_config with the unsaved draft provider fields overlaid.

        Only dirty fields are overlaid, so a provider with no unsaved edits tests
        exactly the saved config (task-432).

        Args:
            provider: The provider whose Test is running (the draft widget value).

        Returns:
            A config mapping the Test's readiness check can evaluate.
        """
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        draft = self._provider_draft()
        dirty = draft.dirty_keys if draft is not None else set()
        if not ({"endpoint", "credential_env_var", "api_key"} & dirty):
            return app_config
        provider_save_key, _config = self._provider_config_entry(provider)
        provider_save_key = provider_save_key or provider_config_key(provider)
        if not provider_save_key:
            return app_config
        try:
            endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
            env_var = self.query_one("#settings-provider-credential-env-var", Input).value.strip()
            api_key = self.query_one("#settings-provider-api-key", Input).value.strip()
        except QueryError:
            values = self._provider_setting_values_mapping()
            endpoint = str(values.get("endpoint") or "").strip()
            env_var = str(values.get("credential_env_var") or "").strip()
            api_key = str(values.get("api_key") or "").strip()
        return overlay_provider_draft_config(
            app_config,
            provider_save_key=provider_save_key,
            endpoint_key=self._provider_endpoint_setting_key(provider),
            draft_endpoint=endpoint if "endpoint" in dirty else None,
            draft_env_var=env_var if "credential_env_var" in dirty else None,
            draft_api_key=api_key if "api_key" in dirty else None,
        )

    def _provider_discovery_staged_settings(self, provider: str) -> dict[str, object]:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return {"api_settings": {}}
        provider_section_key, _provider_config = self._provider_config_entry(provider)
        provider_save_key = provider_section_key or provider_key
        endpoint = ""
        credential_env_var = ""
        try:
            endpoint = self.query_one(
                "#settings-provider-endpoint-value", Input
            ).value.strip()
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
            # TASK-387: humanize the row so a first-run user can read it instead
            # of decoding internal enum names (runtime_discovered / capability=…).
            saved_label = (
                "saved" if bool(getattr(model, "persisted", False)) else "session"
            )
            source_label = {
                "runtime_discovered": "discovered",
                "persisted_discovered": "discovered (cached)",
                "saved": "saved",
            }.get(source, source.replace("_", " "))
            capability_label = f"capabilities {capability}"
            label = (
                f"{model_id} · {saved_label} · {source_label} · {capability_label}"
            )
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
        message = str(getattr(error, "message", "") or "").strip()
        recovery = str(getattr(error, "recovery_hint", "") or "").strip()
        # TASK-367: surface the discovery client's DISTINCT message for endpoint
        # problems — a malformed URL and a valid-but-unsupported path now read
        # differently instead of collapsing into the single generic /v1 copy.
        if kind in {"unsupported_endpoint", "malformed_endpoint"}:
            if message or recovery:
                return redact_secret_text(f"{message} {recovery}".strip())
            return MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY
        if recovery:
            return redact_secret_text(f"{message} {recovery}".strip())
        if message:
            return redact_secret_text(message)
        return "Model discovery failed. Check provider endpoint settings and try again."

    def _refresh_model_discovery_widgets(self) -> None:
        self._set_static_text(
            "#settings-model-discovery-status", self._model_discovery_status
        )
        try:
            self.query_one(
                "#settings-model-discovery-empty", Static
            ).display = not self._model_discovery_models
        except QueryError:
            pass
        try:
            discover_button = self.query_one(
                "#settings-discover-provider-models", Button
            )
            discover_button.disabled = not self._model_discovery_available(
                self._provider_widget_value()
            )
        except QueryError:
            pass
        try:
            save_button = self.query_one(
                "#settings-save-discovered-provider-models", Button
            )
            save_button.disabled = not self._model_discovery_models
        except QueryError:
            pass
        try:
            clear_button = self.query_one(
                "#settings-clear-discovered-provider-models", Button
            )
            clear_button.disabled = not self._model_discovery_models
        except QueryError:
            pass
        try:
            discovered_list = self.query_one(
                "#settings-discovered-models-list", SelectionList
            )
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
        scope_service = getattr(
            self.app_instance, "llm_provider_catalog_scope_service", None
        )
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
            # No traceback: the log file sink runs with diagnose=True, which would
            # dump frame locals (api_key, headers) into the log file.
            logger.warning(f"Provider model discovery failed: {type(exc).__name__}")
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
            self._refresh_model_field_suggester()  # TASK-369: enable typeahead
            self.app.notify(
                "Provider model discovery finished.", severity="information"
            )
            return

        self._model_discovery_models = ()
        self._model_discovery_selected_model_ids = set()
        self._model_discovery_status = self._discovery_status_from_error(result)
        self._refresh_model_discovery_widgets()
        self.app.notify("Provider model discovery failed.", severity="warning")

    @work(exclusive=True, group="settings-model-discovery")
    async def _save_selected_discovered_provider_models_worker(self) -> None:
        await self._save_selected_discovered_provider_models()

    def _model_field_suggester(self) -> SuggestFromList | None:
        """TASK-369: typeahead of discovered model ids for the Model field.

        Recognition over recall — while a discovery result is on screen, typing a
        prefix (e.g. ``gemma``) completes to the full gguf id instead of forcing
        the user to recall a 56-character filename. Returns ``None`` when there
        is nothing to suggest.
        """
        ids = sorted(
            {
                str(getattr(model, "model_id", "") or "").strip()
                for model in self._model_discovery_models
                if str(getattr(model, "model_id", "") or "").strip()
            }
        )
        return SuggestFromList(ids, case_sensitive=False) if ids else None

    def _refresh_model_field_suggester(self) -> None:
        """Point the Model field's suggester at the current discovered models."""
        try:
            self.query_one("#settings-model-value", Input).suggester = (
                self._model_field_suggester()
            )
        except (QueryError, AttributeError):
            pass

    @staticmethod
    def _model_to_activate_after_save(
        current_model: object, saved_model_ids: tuple[str, ...]
    ) -> str:
        """TASK-369: the Model value to set after saving discovered models.

        An empty field is filled with the first saved model (so readiness can
        pass without retyping a long gguf name); a field the user already set is
        left untouched.
        """
        current = str(current_model or "").strip()
        if current:
            return current
        for model_id in saved_model_ids:
            candidate = str(model_id or "").strip()
            if candidate:
                return candidate
        return ""

    def _activate_saved_model_if_field_empty(
        self, saved_model_ids: tuple[str, ...]
    ) -> None:
        """Populate the empty Model field with a just-saved model (TASK-369)."""
        try:
            model_input = self.query_one("#settings-model-value", Input)
        except QueryError:
            return
        next_value = self._model_to_activate_after_save(
            model_input.value, saved_model_ids
        )
        if next_value and next_value != model_input.value:
            # Setting .value fires Input.Changed, which stages the model draft.
            model_input.value = next_value

    async def _save_selected_discovered_provider_models(self) -> None:
        provider = self._provider_widget_value()
        provider_key = provider_config_key(provider)
        scope_service = getattr(
            self.app_instance, "llm_provider_catalog_scope_service", None
        )
        if not provider_key or scope_service is None:
            self._model_discovery_status = (
                "Provider is required before saving discovered models."
            )
            self._refresh_model_discovery_widgets()
            return
        try:
            discovered_list = self.query_one(
                "#settings-discovered-models-list", SelectionList
            )
            selected_model_ids = [
                str(model_id) for model_id in discovered_list.selected
            ]
        except QueryError:
            selected_model_ids = sorted(self._model_discovery_selected_model_ids)
        if not selected_model_ids:
            self._model_discovery_status = "Select discovered models to save."
            self._refresh_model_discovery_widgets()
            self.app.notify(
                "Select discovered models before saving.", severity="warning"
            )
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
            # TASK-369: recognition over recall — offer the saved model for
            # activation instead of leaving an empty Model field the user must
            # retype from memory of a name the cleared discovery list no longer
            # shows.
            self._activate_saved_model_if_field_empty(saved_model_ids)
            self._model_discovery_status = (
                message or f"Saved {len(saved_model_ids)} discovered model(s)."
            )
            self._refresh_model_discovery_widgets()
            self._refresh_model_field_suggester()
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
        scope_service = getattr(
            self.app_instance, "llm_provider_catalog_scope_service", None
        )
        if provider_key and scope_service is not None:
            try:
                await scope_service.clear_discovered_models(
                    mode="local",
                    provider=provider_key,
                )
            except Exception:
                logger.exception("Provider discovered model cache clear failed")
        self._reset_provider_model_discovery_state("Discovered model cache cleared.")

    def _persist_model_catalog_settings(self) -> None:
        """Persist the model catalog toggles to ``[model_catalog]`` (ADR-020).

        The toggles gate a background behavior, so changes save immediately
        instead of staging into the category draft. States that match the
        saved config are skipped (defense-in-depth no-op guard) so merely
        viewing the category never rewrites config.toml.
        """
        try:
            auto_refresh_enabled = self.query_one(
                "#settings-model-catalog-auto-refresh", Checkbox
            ).value
            stale_hours_raw = self.query_one(
                "#settings-model-catalog-stale-hours", Input
            ).value
            auto_values = {
                provider: self.query_one(
                    f"#settings-mc-auto-{provider.lower()}", Checkbox
                ).value
                for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
            }
            write_values = {
                provider: self.query_one(
                    f"#settings-mc-write-{provider.lower()}", Checkbox
                ).value
                for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
            }
        except QueryError:
            return
        stale_hours_text = stale_hours_raw.strip()
        if not stale_hours_text:
            # Empty intermediate input; keep the last persisted value.
            return
        try:
            stale_after_hours: float | int = float(stale_hours_text)
        except (TypeError, ValueError):
            # Invalid intermediate input; keep the last persisted value.
            return
        if stale_after_hours < 0:
            return
        if stale_after_hours.is_integer():
            stale_after_hours = int(stale_after_hours)
        section_values = {
            "model_catalog": {
                "auto_refresh_enabled": auto_refresh_enabled,
                "stale_after_hours": stale_after_hours,
                "auto_refresh_disabled": [
                    provider
                    for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
                    if not auto_values[provider]
                ],
                "write_to_config": [
                    provider
                    for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
                    if write_values[provider]
                ],
            }
        }
        if load_model_catalog_settings(section_values) == load_model_catalog_settings(
            load_settings()
        ):
            return
        save_settings_to_cli_config(section_values)

    def _provider_readiness_test_report(self) -> tuple[str, str, bool]:
        """Run the local provider readiness test against the DRAFT config.

        Returns:
            Tuple of (detail line for the results row, toast summary stating
            the pass/fail outcome with its reason, whether the test passed).
        """
        try:
            provider = self._provider_widget_value()
            model = self.query_one("#settings-model-value", Input).value.strip()
            draft_endpoint = self.query_one(
                "#settings-provider-endpoint-value", Input
            ).value.strip()
        except QueryError:
            values = self._provider_setting_values()
            provider = str(values.get("provider") or "").strip()
            model = str(values.get("model") or "").strip()
            draft_endpoint = str(values.get("endpoint") or "").strip()
        draft = self._provider_draft()
        dirty = draft.dirty_keys if draft is not None else set()  # dirty_keys is a @property
        readiness = get_provider_readiness(
            provider, self._provider_test_staged_config(provider)
        )
        return self._build_provider_readiness_findings(
            provider, model, readiness, draft_endpoint=draft_endpoint, dirty=dirty
        )

    def _build_provider_readiness_findings(
        self,
        provider: str,
        model: str,
        readiness,
        *,
        draft_endpoint: str,
        dirty: set[str],
    ) -> tuple[str, str, bool]:
        """Assemble the Test evidence line + toast from resolved inputs.

        Reads only ``app_config`` (via helpers) and ``os.environ`` -- never widgets
        -- so it is unit-testable on a bare screen instance.

        Args:
            provider: Provider under test (draft widget value).
            model: Model under test (draft widget value).
            readiness: ``ProviderReadiness`` from the draft-overlaid config.
            draft_endpoint: The endpoint the test used (draft widget, may be empty).
            dirty: The provider draft's dirty field keys.

        Returns:
            Tuple of (redacted detail line, redacted toast summary, passed).
        """
        provider_key = provider_config_key(provider)
        passed = bool(readiness.ready and model)
        display_name = self._provider_display_name(provider) if provider else "Provider"
        # TASK-366: lead with ONE verdict consistent with the status line below.
        # A config-ready provider with no default model is still blocked, so it
        # must not read "<provider> is ready" next to "status=blocked".
        if readiness.ready and not model:
            verdict_message = f"{display_name} is configured, but no default model is set."
        else:
            verdict_message = readiness.user_message
        findings: list[str] = ["Provider test", verdict_message]

        if not model:
            findings.append("model=missing")
        else:
            findings.append(f"model={model}{' (draft)' if 'model' in dirty else ''}")

        # This literal marker holds no secret material (just a provenance
        # label). redact_secret_text() pattern-matches on "...key...=value",
        # so if it were run through the same redaction pass as the other
        # findings it would truncate the word "draft" right after "=". It is
        # therefore excluded from redaction below (see `api_key_relabelled`).
        draft_api_key_label = "api_key_source=draft api_key (unsaved)"
        api_key_relabelled = False
        if readiness.api_key_source:
            if (
                "api_key" in dirty
                and readiness.api_key_source
                == f"config:api_settings.{provider_key}.api_key"
            ):
                findings.append(draft_api_key_label)
                api_key_relabelled = True
            else:
                findings.append(f"api_key_source={readiness.api_key_source}")
        if readiness.env_var:
            # Report presence only, never the raw value. ``redact_secret_text``
            # is name-pattern based (it redacts only ``*_API_KEY``/``TOKEN``/...),
            # so a custom-named credential env var (e.g. ``MY_LLAMA_CRED``) would
            # otherwise print its secret verbatim into this screenshot-able UI.
            # Emitting the established ``<redacted>`` marker for any set value
            # keeps the standard-name output identical while closing that gap
            # (folds in task-483).
            env_present = bool(os.environ.get(readiness.env_var))
            env_tag = " (draft env var)" if "credential_env_var" in dirty else ""
            findings.append(
                f"{readiness.env_var}={'<redacted>' if env_present else 'missing'}{env_tag}"
            )
        elif not readiness.requires_api_key:
            findings.append("api_key=not required")

        # Mask any password embedded in the endpoint's userinfo before display
        # (name-pattern redaction misses ``scheme://user:pass@host``).
        endpoint_summary = self._provider_endpoint_summary(
            provider, endpoint=_mask_url_userinfo(draft_endpoint)
        )
        if "endpoint" in dirty:
            endpoint_summary = f"{endpoint_summary} (draft)"
        findings.append(endpoint_summary)

        findings.append(f"status={'ready' if passed else 'blocked'}")

        # task-185: the toast must state the outcome, not just "finished".
        if passed:
            summary = f"Provider test passed: {display_name} is ready; model {model}."
        elif not readiness.ready:
            summary = f"Provider test failed: {readiness.user_message}"
            if not model:
                summary += " Also set a default model."
        else:
            summary = (
                f"Provider test failed: {display_name} is ready but no default model is set."
            )
        if api_key_relabelled:
            detail = " | ".join(
                finding if finding == draft_api_key_label else redact_secret_text(finding)
                for finding in findings
            )
        else:
            detail = redact_secret_text(" | ".join(findings))
        return (
            detail,
            redact_secret_text(summary),
            passed,
        )

    def _run_provider_readiness_test(self) -> str:
        detail, _summary, _passed = self._provider_readiness_test_report()
        return detail

    def _provider_live_probe_base_url(self) -> str:
        """Return the endpoint to live-probe after a passing readiness test.

        task-191: only URL-based/local providers with a concrete endpoint
        (unsaved widget value first, then saved config) are probed; cloud and
        key-based providers keep the local-only Test behavior.

        Returns:
            The endpoint base URL, or ``""`` when no live probe applies.
        """
        provider = self._provider_widget_value()
        if provider_config_key(provider) not in URL_BASED_PROVIDER_KEYS:
            return ""
        try:
            endpoint = self.query_one(
                "#settings-provider-endpoint-value", Input
            ).value.strip()
        except QueryError:
            endpoint = ""
        return endpoint or self._provider_endpoint_value(provider)

    @work(exclusive=True, group="settings-endpoint-probe")
    async def _provider_endpoint_probe_worker(
        self,
        base_url: str,
        detail: str,
        summary: str,
    ) -> None:
        outcome = await probe_settings_endpoint(base_url)
        self._apply_provider_endpoint_probe_outcome(detail, summary, outcome)

    def _apply_provider_endpoint_probe_outcome(
        self,
        detail: str,
        summary: str,
        outcome,
    ) -> None:
        """Fold a live endpoint probe outcome into the Test result and toast.

        Args:
            detail: Readiness detail line shown in the results row.
            summary: Passing readiness toast summary the probe extends.
            outcome: ``SettingsEndpointProbeOutcome`` from the probe helper.
        """
        self._provider_test_result = redact_secret_text(
            f"{detail} | endpoint {outcome.summary}"
        )
        self._update_provider_test_result()
        combined = f"{summary.rstrip('.')}; endpoint {outcome.summary}."
        self.app.notify(
            redact_secret_text(combined),
            severity="information" if outcome.reachable else "warning",
        )

    def _update_provider_test_result(self) -> None:
        try:
            self.query_one("#settings-provider-test-result", Static).update(
                self._provider_test_result
            )
        except (QueryError, AttributeError):
            # QueryError: widget not mounted yet. AttributeError: called on an
            # unmounted screen (no DOM) — the state is still updated for the next
            # render either way.
            pass

    def _mark_provider_test_result_stale(self) -> None:
        """Invalidate a prior Test Provider result when provider inputs change.

        TASK-366: a stale ``ready``/``blocked`` verdict that no longer reflects
        the draft in the form is misleading (the review saw a ``blocked`` line
        persist after a successful save until Test was re-run). Once any provider
        field is edited or saved, the last result is replaced with a re-run
        prompt. A no-op when nothing has run yet or it is already marked stale, so
        it never clobbers the not-run sentinel or thrashes on every keystroke.
        """
        if self._provider_test_result in (
            self._PROVIDER_TEST_NOT_RUN_COPY,
            self._PROVIDER_TEST_STALE_COPY,
        ):
            return
        self._provider_test_result = self._PROVIDER_TEST_STALE_COPY
        self._update_provider_test_result()

    def _update_provider_dynamic_widgets(self) -> None:
        try:
            provider = self._provider_widget_value()
        except QueryError:
            provider = str(
                self._provider_setting_values_mapping().get("provider") or ""
            )
        try:
            endpoint = self.query_one(
                "#settings-provider-endpoint-value", Input
            ).value.strip()
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
            self.query_one("#settings-provider-inspector-readiness", Static).update(
                readiness_label
            )
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
            clear_button.disabled = not self._provider_saved_api_key_present(
                provider
            ) and not bool(api_key_input.value.strip())
        except QueryError:
            pass
        self._refresh_generation_support_summary(provider)
        self._refresh_provider_field_guidance()

    def _detail_row(
        self, label: str, value: object, *, identifier: str | None = None
    ) -> Static:
        if isinstance(value, str):
            # task-1583: dotted keys/paths fold at separators, never mid-word.
            value = _fold_long_tokens(value)
        return Static(
            f"{label}: {value}",
            id=identifier,
            classes="settings-detail-row",
        )

    def _provider_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        provider = str(
            self._provider_setting_values_mapping().get("provider") or ""
        ).strip()
        endpoint_key = self._provider_endpoint_row(provider).removeprefix(
            "Endpoint key: "
        )
        provider_config_prefix = (
            f"api_settings.{provider_config_key(provider)}"
            if provider_config_key(provider)
            else "api_settings.<provider>"
        )
        field_id = self._active_settings_field_id
        if field_id == "settings-provider-value":
            return (
                ("Focused setting", "Provider"),
                (
                    "Purpose",
                    "Selects the provider used for Console generation defaults.",
                ),
                ("Saved as", "chat_defaults.provider"),
                (
                    "Validation",
                    "choose a catalog provider or use Manual for custom aliases",
                ),
            )
        if field_id == "settings-provider-manual-value":
            return (
                ("Focused setting", "Manual provider"),
                (
                    "Purpose",
                    "Stores a custom provider key when the catalog has no match.",
                ),
                ("Saved as", "chat_defaults.provider"),
                (
                    "Validation",
                    "letters, numbers, hyphens, underscores, and provider aliases only",
                ),
            )
        if field_id == "settings-model-value":
            return (
                ("Focused setting", "Model"),
                (
                    "Purpose",
                    "Selects the model used when Console has no narrower override.",
                ),
                ("Saved as", "chat_defaults.model"),
                (
                    "Validation",
                    "model name is required before provider-backed generation can run",
                ),
            )
        if field_id == "settings-provider-endpoint-value":
            return (
                ("Focused setting", "Endpoint"),
                (
                    "Purpose",
                    "Controls the provider endpoint used by Console generation.",
                ),
                ("Saved as", endpoint_key),
                ("Validation", "must start with http:// or https:// when set"),
            )
        if field_id == "settings-provider-api-key":
            return (
                ("Focused setting", "API key"),
                (
                    "Purpose",
                    "Stores a provider API key in local config for Console generation.",
                ),
                ("Saved as", f"{provider_config_prefix}.api_key"),
                ("Validation", "single-line secret value; visible UI stays masked"),
            )
        if field_id == "settings-provider-credential-env-var":
            return (
                ("Focused setting", "Credential env"),
                (
                    "Purpose",
                    "Stores the environment variable name containing the API key.",
                ),
                ("Saved as", f"{provider_config_prefix}.api_key_env_var"),
                (
                    "Validation",
                    "environment variable names must start with a letter or underscore",
                ),
            )
        if field_id == "settings-model-profile-temperature":
            return (
                ("Focused setting", "Temperature"),
                (
                    "Purpose",
                    "Optional creativity default for this provider and model profile.",
                ),
                (
                    "Saved as",
                    f"{provider_config_prefix}.model_defaults.<model>.temperature",
                ),
                (
                    "Validation",
                    "number from 0.0 to 2.0, or blank for inherited default",
                ),
            )
        if field_id == "settings-model-profile-top-p":
            return (
                ("Focused setting", "Top P"),
                (
                    "Purpose",
                    "Optional token-probability cutoff for this provider and model profile.",
                ),
                ("Saved as", f"{provider_config_prefix}.model_defaults.<model>.top_p"),
                (
                    "Validation",
                    "number from 0.0 to 1.0, or blank for inherited default",
                ),
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
                    (
                        "Availability",
                        self._unsupported_model_profile_placeholder(provider),
                    ),
                    ("Saved as", "not saved for the selected provider"),
                    (
                        "Validation",
                        "select a provider that supports this control before editing",
                    ),
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
                (
                    "Purpose",
                    "Optional streaming preference for this provider and model profile.",
                ),
                (
                    "Saved as",
                    f"{provider_config_prefix}.model_defaults.<model>.streaming",
                ),
                ("Validation", "true, false, or blank for inherited default"),
            )
        return (
            ("Focused setting", "Provider setup"),
            (
                "Purpose",
                "Configure the default provider, model, endpoint, and credential source.",
            ),
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
                (
                    "Purpose",
                    "Limits how many themes appear in the command palette; 0 shows all.",
                ),
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
                (
                    "Purpose",
                    "Sets the global compact/normal/comfortable UI density default.",
                ),
                ("Saved as", "appearance.density"),
                ("Validation", "compact, normal, or comfortable"),
            )
        if field_id == "settings-appearance-animations-enabled":
            return (
                ("Focused setting", "Animations"),
                (
                    "Purpose",
                    "Controls whether optional UI motion is enabled by default.",
                ),
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
            (
                "Purpose",
                "Configure global visual defaults without replacing the Theme editor.",
            ),
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
                (
                    "Purpose",
                    "Configure persisted database path defaults for the next launch.",
                ),
                ("Saved as", "database.*"),
                (
                    "Validation",
                    "path text only; no files are moved, created, or reconnected",
                ),
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
            (
                "Validation",
                "must be a safe path; database paths must end in .db, .sqlite, or .sqlite3",
            ),
        )

    def _refresh_storage_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.STORAGE:
            return
        for index, (label, value) in enumerate(self._storage_field_guidance_rows()):
            self._set_static_text(
                f"#settings-storage-field-guide-{index}",
                f"{label}: {value}",
            )

    def _rag_field_guidance_rows(self) -> tuple[tuple[str, str], ...]:
        """Task 3 (541 v2 UX AC3): context-sensitive Library/RAG guidance.

        Mirrors `_provider_field_guidance_rows` / `_storage_field_guidance_rows`:
        keyed first on the focused field (`_active_settings_field_id`, via
        `_RAG_FIELD_GROUP_BY_ID`), then on the last-expanded Collapsible
        group (`_active_rag_scope_group`), then the unchanged static
        fallback so a first-ever visit (nothing focused, nothing expanded
        beyond the default "Search" group) reads exactly as before this
        task.
        """
        field_id = self._active_settings_field_id or ""
        group = _RAG_FIELD_GROUP_BY_ID.get(field_id) or self._active_rag_scope_group
        return _RAG_GROUP_GUIDANCE.get(group or "", _RAG_GROUP_GUIDANCE_FALLBACK)

    def _refresh_rag_field_guidance(self) -> None:
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        for index, (label, value) in enumerate(self._rag_field_guidance_rows()):
            self._set_static_text(
                f"#settings-library-rag-field-guide-{index}",
                f"{label}: {value}",
            )

    def _split_detail_row(self, text: str) -> Static:
        label, separator, value = text.partition(":")
        if not separator:
            return self._detail_row("Path", text)
        return self._detail_row(label.strip(), value.strip())

    def _inspector_guidance(
        self, category: SettingsCategoryId
    ) -> tuple[tuple[str, str], ...]:
        if category is SettingsCategoryId.IMAGE_GENERATION:
            # Final review Important 2: IMAGE_GENERATION sits in the
            # "Domain Defaults" rail group (needed for that grouping + the
            # settings_can_mutate=True domain contract test) but, unlike a
            # pure view-only delegation card, Settings genuinely owns and
            # writes this config -- an explicit branch here (mirroring
            # _guided_action_message's/_category_state_banner_text's own
            # IMAGE_GENERATION branches, both of which already win over the
            # generic domain fallback below) keeps this page from showing
            # the generic "nothing on this page is editable" copy that's
            # true for every OTHER domain category but not this one.
            return (
                (
                    "Affected config",
                    "[image_generation] backend enable/default, per-backend "
                    "fields (base URL, model, timeout, key), and generation "
                    "defaults",
                ),
                (
                    "Recovery",
                    "Revert discards unsaved edits; Console's /generate-image "
                    "keeps working off the last saved config.toml regardless",
                ),
                (
                    "Boundary",
                    "Edits backend, key, and generation defaults here; Save "
                    "applies to config.toml",
                ),
            )
        if category in DOMAIN_SETTINGS_CATEGORY_IDS:
            contract = self._domain_category_contract(category)
            return (
                (
                    "Affected config",
                    "none yet - nothing on this page is editable",
                ),
                (
                    "Recovery",
                    f"go to {contract.owner_destination} to make changes",
                ),
                (
                    "Boundary",
                    f"{contract.owner_destination} owns this; Settings only shows it",
                ),
            )
        guidance = _INSPECTOR_GUIDANCE.get(category)
        if guidance is not None:
            return guidance
        # A missing entry must degrade gracefully: this runs inside compose,
        # where an uncaught exception takes down the whole app. Warn once per
        # category so the coverage gap is visible in logs without spamming
        # every rebuild.
        if category not in _WARNED_MISSING_GUIDANCE_CATEGORIES:
            _WARNED_MISSING_GUIDANCE_CATEGORIES.add(category)
            logger.warning(
                "No inspector guidance entry for Settings category %r; using "
                "generic fallback. Add an entry to _INSPECTOR_GUIDANCE.",
                category,
            )
        return _INSPECTOR_GUIDANCE_FALLBACK

    def _render_category_buttons(self) -> ComposeResult:
        summaries_by_id = {
            summary.category: summary for summary in self._category_summaries()
        }
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
                    # task-1584 rescaled tiers: 0/1 primary, 2 secondary.
                    if rank in (0, 1):
                        button.add_class("settings-primary-search-match")
                    elif rank == 2:
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
        # task-181: lead with user-relevant readiness/storage/privacy; Manual
        # sync and the ownership summary sit at the bottom of the card.
        yield Static("Overview", classes="destination-section settings-column-title")
        with Vertical(id="settings-overview-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.OVERVIEW)
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
            yield Static(
                "Server, sync, workspace, and handoff", classes="destination-section"
            )
            for label, value in self.server_sync_workspace_handoff_rows:
                yield self._detail_row(label, value)
            with Horizontal(classes="settings-action-row"):
                yield Button(
                    "Switch Source / Server",
                    id="settings-switch-runtime-source",
                    tooltip="Choose local-only or bind a tldw server as the runtime source.",
                )
            yield Static("Manual sync", classes="destination-section")
            yield Static(
                "Preview pending Notes/Chat changes before anything is sent to a server.",
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
            yield Static("Where changes happen", classes="destination-section")
            for label, value in self._overview_ownership_rows():
                yield self._detail_row(label, value)

    def _render_provider_detail(self) -> ComposeResult:
        resolved = self._resolve_provider_model_for_settings()
        values = self._provider_display_setting_values()
        provider = str(values["provider"])
        yield Static(
            "Providers & Models", classes="destination-section settings-column-title"
        )
        with Vertical(
            id="settings-providers-models-card", classes="settings-focus-card"
        ):
            yield self._render_category_state_banner(
                SettingsCategoryId.PROVIDERS_MODELS
            )
            # task-189: the Connect block (provider, model, endpoint,
            # credentials, readiness/test) leads; sampling and tuning live in
            # the collapsed "Generation defaults" disclosure below it.
            yield Static(
                "Connect",
                id="settings-provider-connect-title",
                classes="destination-section",
            )
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
            if (
                self._provider_select_value_for_provider(provider)
                != PROVIDER_MANUAL_SELECT_VALUE
            ):
                manual_provider_classes += " settings-provider-manual-hidden"
            with Horizontal(
                id="settings-provider-manual-row", classes=manual_provider_classes
            ):
                yield Static("Manual", classes="settings-input-label")
                yield Input(
                    value=str(values["provider"])
                    if self._provider_select_value_for_provider(provider)
                    == PROVIDER_MANUAL_SELECT_VALUE
                    else "",
                    id="settings-provider-manual-value",
                    classes="settings-compact-input",
                    placeholder="Custom provider key",
                    disabled=(
                        self._provider_select_value_for_provider(provider)
                        != PROVIDER_MANUAL_SELECT_VALUE
                    ),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Model", classes="settings-input-label")
                yield Input(
                    value=str(values["model"]),
                    id="settings-model-value",
                    classes="settings-compact-input",
                    placeholder="Model name",
                    suggester=self._model_field_suggester(),
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Endpoint", classes="settings-input-label")
                yield SettingsURLInput(
                    value=str(values["endpoint"]),
                    id="settings-provider-endpoint-value",
                    classes="settings-compact-input",
                    placeholder=self._provider_endpoint_placeholder(provider),
                    validators=[ProviderEndpointURLValidator()],
                    validate_on={"blur", "submitted"},
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
            # task-189: the Test affordance closes the first-run Connect job
            # (provider -> model -> endpoint -> credentials -> test) before
            # the informational readiness and discovery sections.
            yield Button(
                "Test Provider",
                id="settings-test-provider",
                tooltip=(
                    "Run a local readiness check for this provider configuration; "
                    "URL-based local providers also get a short live endpoint probe."
                ),
            )
            # TASK-386 (AC#2): the readiness / live-probe explanation must also
            # exist as visible static text -- a hover tooltip is invisible to
            # keyboard users and self-occludes the result line below it.
            yield Static(
                "Runs a local readiness check; URL-based local providers also get "
                "a short live endpoint probe.",
                id="settings-test-provider-guidance",
                classes="settings-status-row",
            )
            yield Static(self._provider_test_result, id="settings-provider-test-result")
            yield Static(
                self._provider_save_result,
                id="settings-provider-save-result",
                classes="settings-status-row",
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
                self._provider_endpoint_display_value(
                    str(values["provider"]), values["endpoint"]
                ),
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
                    disabled=not self._model_discovery_available(
                        str(values["provider"])
                    ),
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
            # ADR-020: [model_catalog] auto-refresh toggles. Values initialize
            # inline from the saved config (the Connect block pattern) and
            # persist immediately on change via the handlers below.
            model_catalog_settings = load_model_catalog_settings(load_settings())
            # TASK-387: keep the internal decision-record id (ADR-020) out of the
            # user-facing heading; it survives in the code comment above.
            yield Static("Automatic refresh", classes="destination-section")
            yield Checkbox(
                "Auto-refresh model lists on startup",
                value=model_catalog_settings.auto_refresh_enabled,
                id="settings-model-catalog-auto-refresh",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Refresh after (hours):", classes="settings-status-row")
                yield Input(
                    f"{model_catalog_settings.stale_after_hours:g}",
                    id="settings-model-catalog-stale-hours",
                    type="integer",
                    tooltip="0 = refetch every launch.",
                )
            for _provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                _provider_key = provider_config_key(_provider)
                _pid = _provider.lower()
                with Horizontal(classes="settings-input-row"):
                    yield Checkbox(
                        f"{_provider}: auto-refresh",
                        value=(
                            _provider_key
                            not in model_catalog_settings.auto_refresh_disabled
                        ),
                        id=f"settings-mc-auto-{_pid}",
                    )
                    yield Checkbox(
                        "save to config",
                        value=_provider_key in model_catalog_settings.write_to_config,
                        id=f"settings-mc-write-{_pid}",
                        tooltip=(
                            "Append newly discovered models to config.toml — "
                            "large catalogs like OpenRouter only add newly released "
                            "models after a first baseline."
                        ),
                    )
            # task-189: sampling and provider-specific tuning live below the
            # Connect block in a collapsed-by-default disclosure.
            with Collapsible(
                title="Generation defaults",
                collapsed=True,
                id="settings-generation-defaults",
            ):
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
                        value=self._profile_input_value(
                            values["model_profile_temperature"]
                        ),
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
                        value=self._profile_input_value(
                            values["model_profile_max_tokens"]
                        ),
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
                        value=self._profile_input_value(
                            values["model_profile_presence_penalty"]
                        ),
                        id="settings-model-profile-presence-penalty",
                        classes="settings-compact-input",
                        placeholder="-2.0 - 2.0",
                    )
                with Horizontal(classes="settings-input-row"):
                    yield Static("Frequency", classes="settings-input-label")
                    yield Input(
                        value=self._profile_input_value(
                            values["model_profile_frequency_penalty"]
                        ),
                        id="settings-model-profile-frequency-penalty",
                        classes="settings-compact-input",
                        placeholder="-2.0 - 2.0",
                    )
                # task-189: one summary line replaces per-row "Unavailable
                # for <provider>" placeholders; unsupported rows are hidden.
                support_copy = self._provider_generation_support_copy(provider)
                support_summary = Static(
                    support_copy,
                    id="settings-provider-generation-support",
                    classes="settings-detail-row",
                )
                support_summary.set_class(
                    not support_copy, "settings-gated-profile-hidden"
                )
                yield support_summary
                with Horizontal(
                    id="settings-model-profile-reasoning-effort-row",
                    classes=self._gated_profile_row_classes(
                        self._model_profile_field_supported(
                            provider,
                            "model_profile_reasoning_effort",
                        )
                    ),
                ):
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
                with Horizontal(
                    id="settings-model-profile-reasoning-summary-row",
                    classes=self._gated_profile_row_classes(
                        self._model_profile_field_supported(
                            provider,
                            "model_profile_reasoning_summary",
                        )
                    ),
                ):
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
                with Horizontal(
                    id="settings-model-profile-verbosity-row",
                    classes=self._gated_profile_row_classes(
                        self._model_profile_field_supported(
                            provider,
                            "model_profile_verbosity",
                        )
                    ),
                ):
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
                with Horizontal(
                    id="settings-model-profile-thinking-effort-row",
                    classes=self._gated_profile_row_classes(
                        self._model_profile_field_supported(
                            provider,
                            "model_profile_thinking_effort",
                        )
                    ),
                ):
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
                with Horizontal(
                    id="settings-model-profile-thinking-budget-tokens-row",
                    classes=self._gated_profile_row_classes(
                        self._model_profile_field_supported(
                            provider,
                            "model_profile_thinking_budget_tokens",
                        )
                    ),
                ):
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
                        value=self._profile_input_value(
                            values["model_profile_streaming"]
                        ),
                        id="settings-model-profile-streaming",
                        classes="settings-compact-input",
                        placeholder="true or false",
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
                "Choose a catalog provider (type in the open list to jump to one), "
                "or use Manual / custom provider for other keys.",
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
                self._provider_endpoint_row(str(values["provider"])).removeprefix(
                    "Endpoint key: "
                ),
                identifier="settings-provider-endpoint-key",
            )

    def _render_console_behavior_card(self, *, compact: bool = False) -> ComposeResult:
        with Vertical(
            id="settings-console-behavior-card", classes="settings-secondary-card"
        ):
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
            yield Static("Chat images", classes="destination-section")
            yield Static(
                "Render images linked in assistant replies (remote fetch).",
                id="settings-console-remote-images-label",
            )
            yield Button(
                self._remote_images_button_label(),
                id="settings-console-remote-images-toggle",
                tooltip=(
                    "Fetch and render http(s) image links found in replies. "
                    "Applies immediately; fetches go through the egress SSRF "
                    "policy with size caps."
                ),
            )
            yield Static(
                "Off by default: fetching a model-suggested link reveals your "
                "IP address to that host.",
                id="settings-console-remote-images-help",
            )
            yield Static("Parallel agent runs", classes="destination-section")
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Max parallel agent runs", classes="settings-input-label"
                )
                yield Input(
                    value=str(self._console_max_parallel_runs_value()),
                    id="settings-console-max-parallel-runs",
                    classes="settings-compact-input",
                    placeholder=str(DEFAULT_CONSOLE_MAX_PARALLEL_RUNS),
                    restrict=r"^[0-9]*$",
                )
            yield Static(
                "Agent tool-result display cap", classes="destination-section"
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Tool result display cap", classes="settings-input-label"
                )
                yield Input(
                    value=str(self._tool_result_display_chars_value()),
                    id="settings-console-tool-result-display-chars",
                    classes="settings-compact-input",
                    placeholder=str(DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS),
                    restrict=r"^[0-9]*$",
                )
            yield Static(
                "How much of an agent tool result the Console shows you here -- "
                "distinct from max_tool_result_chars, which caps what the model "
                "itself saw. Open a run's \"View full log\" (Agent rail) to read "
                "everything beyond this cap.",
                id="settings-console-tool-result-display-chars-help",
                classes="settings-detail-row",
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
                    value=self._console_input_value(
                        self._console_behavior_value("streaming")
                    ),
                    id="settings-console-default-streaming",
                    classes="settings-compact-input",
                    placeholder="true or false",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Temperature", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("temperature")
                    ),
                    id="settings-console-default-temperature",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top P", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("top_p")
                    ),
                    id="settings-console-default-top-p",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Min P", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("min_p")
                    ),
                    id="settings-console-default-min-p",
                    classes="settings-compact-input",
                    placeholder="optional 0.0 - 1.0",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Top K", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("top_k")
                    ),
                    id="settings-console-default-top-k",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Max tokens", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("max_tokens")
                    ),
                    id="settings-console-default-max-tokens",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Seed", classes="settings-input-label")
                yield Input(
                    value=self._console_input_value(
                        self._console_behavior_value("seed")
                    ),
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
                    value=self._console_input_value(
                        self._console_behavior_value("verbosity")
                    ),
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
                    value=str(
                        self._console_background_effect_value("effect") or "none"
                    ),
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
                    value=str(
                        self._console_background_effect_value("scope") or "transcript"
                    ),
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
                    value=str(
                        self._console_background_effect_value("intensity") or "low"
                    ),
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

    # Task 1 (RAG settings v2 UX, AC #4): the citations/reranking toggles
    # used to be Buttons whose label just said "Enabled"/"Disabled" -- an
    # honest control here is a real Checkbox (label describes WHAT it does,
    # `.value` IS the state). The rerank model/results Inputs are dimmed
    # (never hidden) whenever reranking itself is off, distinct from the
    # builtin read-only lock: `_library_rag_rerank_field_state` computes
    # BOTH the shared `disabled` bool and the label suffix, so compose (which
    # can't query not-yet-mounted widgets) and the post-mount sync/handler
    # paths below can never drift from each other.
    @staticmethod
    def _library_rag_rerank_field_state(
        *, rerank_enabled: bool, field_disabled: bool
    ) -> tuple[bool, str]:
        disabled = (not rerank_enabled) or field_disabled
        # Review fix (AC #4): the suffix names reranking-off as the reason
        # ONLY when that's the actual, user-actionable reason -- not when a
        # builtin read-only lock is ALSO in play (the Enable-reranking
        # checkbox is itself disabled then, so "enable reranking to edit"
        # would be an unactionable instruction pointing at the wrong cause).
        suffix = (
            " (enable reranking to edit)"
            if (not rerank_enabled and not field_disabled)
            else ""
        )
        return disabled, suffix

    def _apply_library_rag_rerank_field_state(
        self, *, rerank_enabled: bool, field_disabled: bool
    ) -> None:
        """Post-mount refresh of the rerank model/results Inputs' disabled
        state and label suffix -- called after the checkbox is toggled and
        from ``_sync_library_rag_widgets``/``_sync_library_rag_profile_widgets``
        so a profile switch or revert can never leave a stale suffix/disabled
        combination behind."""
        disabled, suffix = self._library_rag_rerank_field_state(
            rerank_enabled=rerank_enabled, field_disabled=field_disabled
        )
        for input_selector, label_selector, base_label in (
            (
                "#settings-library-rag-reranker-model",
                "#settings-library-rag-reranker-model-label",
                "Reranker model",
            ),
            (
                "#settings-library-rag-reranker-top-k",
                "#settings-library-rag-reranker-top-k-label",
                "Rerank results",
            ),
        ):
            try:
                self.query_one(input_selector, Input).disabled = disabled
            except QueryError:
                pass
            self._set_static_text(label_selector, f"{base_label}{suffix}")

    @staticmethod
    def _library_rag_profile_select_options(grouped: dict) -> list[tuple[str, str]]:
        return [(f"{p['name']} (built-in)", p["id"]) for p in grouped["builtin"]] + [
            (p["name"], p["id"]) for p in grouped["user"]
        ]

    def _library_rag_selected_profile_id(self) -> str | None:
        try:
            select = self.query_one("#settings-library-rag-profile-select", Select)
        except QueryError:
            return None
        value = select.value
        # task-565: `Select.NULL` is the real blank sentinel on this Textual
        # version -- `Select.BLANK` doesn't exist, it silently resolves to
        # the unrelated `Widget.BLANK` (`False`), so it never matched here,
        # letting the stringified sentinel escape as a bogus profile id.
        if value is None or value is Select.NULL:
            return None
        return str(value)

    def _library_rag_profile_name(self, profile_id: str) -> str:
        grouped = list_profiles_grouped()
        for entry in grouped["builtin"] + grouped["user"]:
            if entry["id"] == profile_id:
                return entry["name"]
        return profile_id

    def _library_rag_index_status_line(self, status: Mapping[str, object]) -> str:
        """Render ``fetch_index_status()``'s dict as the status-row text.

        Graceful when the store is non-persistent or the collection hasn't
        been created yet (``state == "absent"``, ``provenance`` always
        empty in that case -- see ``collection_indexes.index_status``): a
        dedicated, friendlier line rather than "absent · 0 vectors". UX
        review item 3 (first-run Backfill nudge): when the active profile's
        ``default_search_mode`` actually NEEDS the vector index (semantic or
        hybrid), that friendlier line names the concrete consequence
        (results are keyword-only until Backfill runs) instead of the
        generic notice -- a brand-new install otherwise looks like search
        already works fully when only keyword search does. A `plain`-mode
        profile never needs the vector index, so it keeps the plain notice.
        When provenance is present (built/empty), append the "built with
        <model> / chunk <size>·<overlap>" tail; omit it when provenance is
        missing so a legacy/unstamped collection still renders sensibly.
        """
        state = str(status.get("state") or "unknown")
        if state == "absent":
            search_mode = normalise_library_rag_search_mode(
                self._library_rag_loaded_defaults().default_search_mode
            )
            if search_mode in ("semantic", "hybrid"):
                mode_label = "Hybrid" if search_mode == "hybrid" else "Semantic"
                return (
                    "Semantic index not built — "
                    f"{mode_label} search is keyword-only until you Backfill."
                )
            return RAG_INDEX_ABSENT_STATUS_TEXT
        count = status.get("count", 0)
        provenance = status.get("provenance") or {}
        model = provenance.get("embedding_model")
        chunk_size = provenance.get("chunk_size")
        chunk_overlap = provenance.get("chunk_overlap")
        base = f"Index: {state} · {count} vectors"
        if model is None or chunk_size is None or chunk_overlap is None:
            return base
        return f"{base} · built with {model} / chunk {chunk_size}·{chunk_overlap}"

    def _apply_library_rag_index_status(self, status: Mapping[str, object]) -> None:
        """Imperatively update the index-status Static from a freshly fetched
        status dict -- called from off-thread worker callbacks, never during
        compose (see ``_library_rag_index_status_text`` init comment).

        Also refreshes ``_library_rag_index_status_cache`` (Task 2, 541 v2
        UX) -- every caller of this method already has a fresh status in
        hand, so this is the one place that needs to remember it for the
        Save-path re-index confirm gate to read later.
        """
        # task-566: a `settings-rag-index-status` worker (category show /
        # 't' test / Save-path reindex confirm) that was already running
        # when the user navigated away still completes and still calls
        # back -- `_select_category`'s `cancel_group` is best-effort, not a
        # guarantee. Skip entirely rather than write a status Static / cache
        # entry that belongs to a category no longer on screen.
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        self._library_rag_index_status_cache = status
        self._library_rag_index_status_text = self._library_rag_index_status_line(
            status
        )
        self._set_static_text(
            "#settings-library-rag-index-status", self._library_rag_index_status_text
        )
        # Task 5 (541 v2 UX AC5): every trigger that lands a fresh index
        # status (category show / 't' test / backfill completion /
        # set-active) is exactly the set of moments the first-run starter
        # panel's predicate could have flipped -- funnel through here rather
        # than duplicating a second fetch-and-toggle path.
        self._refresh_rag_first_run_panel_state()
        # task-629: these SAME moments are also exactly when the active
        # profile identity itself could have silently changed underneath
        # this screen -- Backfill's get_shared_rag_service() call, on its
        # first-ever invocation in the process, imports-and-activates the
        # "Imported settings" profile as a side effect
        # (ensure_imported_profile). Without this, the "Active: .../
        # Editing: ..." rows and the editor card's border_title kept
        # showing whatever was active at mount (e.g. "Hybrid Basic") until
        # some UNRELATED direct profile action (Clone/Set active) next
        # happened to resync them -- surfacing a profile name ("Imported
        # settings") the user had never seen introduced, apparently out of
        # nowhere.
        self._refresh_library_rag_active_profile_identity_text()

    def _library_rag_cached_index_state(self) -> str:
        """The index-status row's OWN cached state string, or "unknown" when
        nothing has been fetched yet -- shared by the first-run predicate so
        it never triggers a fetch of its own (Task 5, 541 v2 UX AC5)."""
        cache = self._library_rag_index_status_cache
        if cache is None:
            return "unknown"
        return str(cache.get("state") or "unknown")

    def _library_rag_first_run_active(
        self, *, info: Mapping[str, object] | None = None, grouped: Mapping[str, object] | None = None
    ) -> bool:
        """Whether the Library/RAG editor is currently in the first-run
        state (Task 5, 541 v2 UX AC5) -- ``info``/``grouped`` may be passed
        in by a caller that already fetched them this pass (compose) to
        avoid a redundant adapter call; omitted, both are fetched fresh."""
        return is_first_run_state(
            info if info is not None else active_profile_info(),
            grouped if grouped is not None else list_profiles_grouped(),
            self._library_rag_cached_index_state(),
        )

    @staticmethod
    def _library_rag_starter_panel_copy(info: Mapping[str, object]) -> str:
        return (
            f"Search already works on {escape_markup(str(info['name']))}. Clone it "
            "to tune retrieval, or run Backfill to enable semantic results."
        )

    def _refresh_rag_first_run_panel_state(self) -> None:
        """Re-evaluate the first-run starter panel's visibility against the
        CURRENT active profile / profile list / cached index status, and
        reflect it onto the mounted widgets imperatively (no recompose) --
        Task 5 (541 v2 UX AC5). Called after anything that could flip the
        predicate: an index-status fetch landing (see
        ``_apply_library_rag_index_status``) and any completed profile
        action (clone/rename/delete, see ``_rag_after_profile_action``) --
        a clone is exactly how a first-run install gets its first user
        profile, which is what ends the first-run state without ever
        touching the index. Swallows ``QueryError``: this runs from
        off-thread worker callbacks and a not-yet-mounted or already-
        navigated-away-from screen must never crash it.

        Task 5 review (Important): the Search group's ``collapsed`` state is
        only ever touched on the actual first-run <-> not-first-run
        TRANSITION (tracked via ``_rag_first_run_active``), never on every
        ordinary re-evaluation. Before this, exiting first-run (clone
        completes / backfill completes) hid the starter panel but left
        Search collapsed behind it -- the user who just did what the panel
        told them saw editable fields hidden until an unrelated recompose
        self-healed it. Gating on the transition (rather than forcing
        ``collapsed`` unconditionally every time this runs) keeps a
        DIFFERENT user's deliberate collapse of Search, in ordinary
        already-not-first-run state, from being forcibly reopened by an
        unrelated status refresh (category show / Save / set-active).
        """
        try:
            panel = self.query_one("#settings-library-rag-starter-panel")
        except QueryError:
            return
        info = active_profile_info()
        first_run = self._library_rag_first_run_active(info=info)
        was_first_run = self._rag_first_run_active
        self._rag_first_run_active = first_run
        if not first_run:
            panel.display = False
            if was_first_run:
                # First-run just ENDED -- restore the Search group to its
                # ordinary expanded default so the newly-cloned/backfilled
                # profile's fields are immediately visible, not hidden
                # behind a collapse this predicate itself imposed on entry.
                try:
                    self.query_one(
                        "#settings-library-rag-search-group", Collapsible
                    ).collapsed = False
                except QueryError:
                    pass
            return
        self._set_static_text(
            "#settings-library-rag-starter-copy",
            self._library_rag_starter_panel_copy(info),
        )
        panel.display = True
        if not was_first_run:
            # The starter panel, not the disabled wall, is the first
            # impression while first-run -- collapse the Search group too
            # (the only one that composes expanded by default; Embedding/
            # Chunking/Vector store/Reranking already compose collapsed).
            # Symmetric with the exit branch above: only on the ENTERING
            # transition, never re-forced on every already-first-run
            # re-evaluation.
            try:
                self.query_one(
                    "#settings-library-rag-search-group", Collapsible
                ).collapsed = True
            except QueryError:
                pass

    @work(exclusive=True, thread=True, group="settings-rag-index-status")
    def _rag_index_status_worker(self) -> None:
        status = fetch_index_status()
        self.app.call_from_thread(self._apply_library_rag_index_status, status)

    def _refresh_library_rag_index_status(self) -> None:
        """Dispatch the off-thread index-status fetch (category show / after
        save). Guarded so a not-yet-mounted or already-navigated-away screen
        never queues pointless work."""
        if not getattr(self, "is_mounted", False):
            return
        self._rag_index_status_worker()

    def _apply_rag_test_category_result(self, status: Mapping[str, object]) -> None:
        """UX review item 8: 't test category' completion for RAG -- refresh
        the same index-status Static the other triggers do, then notify a
        one-line honest summary (index state + current preview defaults)
        instead of silently doing nothing."""
        self._apply_library_rag_index_status(status)
        state = str(status.get("state") or "unknown")
        preview_summary, _preview_retrieval, _preview_context = (
            self._library_rag_preview_rows()
        )
        # task-566 review (Important): a stale 't' test-category worker can
        # land after the user has already navigated away from Library/RAG --
        # `_apply_library_rag_index_status` above already no-ops in that
        # case, but this toast was still unconditional, surfacing "RAG
        # check: ..." over whatever category the user is now on.
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        self.app.notify(
            f"RAG check: {state} index · {preview_summary}", severity="information"
        )

    @work(exclusive=True, thread=True, group="settings-rag-index-status")
    def _rag_test_category_worker(self) -> None:
        status = fetch_index_status()
        self.app.call_from_thread(self._apply_rag_test_category_result, status)

    def _clear_library_rag_backfill_in_flight(self) -> None:
        """Main-thread flip of the in-flight flag -- see
        ``_rag_backfill_worker``'s ``finally`` block."""
        self._library_rag_backfill_in_flight = False

    @work(exclusive=True, thread=True, group="settings-rag-backfill")
    def _rag_backfill_worker(self) -> None:
        """Bulk-index existing media/notes/conversations into the active
        profile's resolved vector collection.

        Task 4 review (Finding 1): this originally shipped as a genuinely
        ``async`` worker, awaiting ``backfill_semantic_index`` directly on
        the UI event loop. That function has long *synchronous* stretches
        between its awaits (sync sqlite pagination generators, per-entry
        needs_reindexing/delete_document N+1 lookups), so a real backfill
        starved the Textual heartbeat and froze the whole TUI. Runs on a
        worker thread instead now, exactly like
        ``SearchRAGWindow._run_index_backfill``: a transient ``asyncio.run``
        loop keeps ALL of that sync work off the UI event loop, and every UI
        touch (notify, the in-flight flag, the status-row refresh) is
        marshalled back with ``call_from_thread``. Same default
        ``item_types`` covering media/notes/conversations, DBs sourced the
        same way from ``self.app_instance``, same start/finish/failure
        notify contract as before. Never crashes the screen: any exception
        is caught and reported via notify rather than propagating out of
        the worker.
        """
        try:
            if not semantic_indexing_available():
                self.app.call_from_thread(
                    self.app.notify,
                    "Semantic indexing is unavailable (missing embeddings "
                    "extras, or disabled in config).",
                    severity="warning",
                )
                return
            media_db = getattr(self.app_instance, "media_db", None)
            chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
            if media_db is None and chachanotes_db is None:
                self.app.call_from_thread(
                    self.app.notify,
                    "No local databases are available to backfill.",
                    severity="error",
                )
                return
            # M5 (SP3 final review): pre-resolve the shared RAG service
            # OUTSIDE the transient asyncio.run loop below -- mirrors
            # SearchRAGWindow._run_index_backfill's PR #700-hardened pattern.
            # backfill_semantic_index's own default (`rag_service or
            # get_shared_rag_service()`) would otherwise construct it for the
            # FIRST time from inside that loop, and the loop closes the
            # instant this run finishes.
            rag_service = get_shared_rag_service()
            if rag_service is None:
                # task-641 review: get_shared_rag_service() can return None
                # not only on a genuine construction failure but also when a
                # concurrent reset/set-active discarded an in-flight build
                # (the two-lock construction in ingestion_indexing.py). Never
                # fall through to backfill_semantic_index's own default arg
                # here -- that would retry construction for the FIRST time
                # INSIDE the transient asyncio.run loop below, exactly the
                # PR #700 hazard the pre-resolution above exists to prevent.
                # Mirrors SearchRAGWindow._run_index_backfill's explicit None
                # guard.
                self.app.call_from_thread(
                    self.app.notify,
                    "RAG backfill could not start: the shared RAG service "
                    "is unavailable right now. Try again shortly.",
                    severity="error",
                )
                return
            summary = asyncio.run(
                backfill_semantic_index(
                    media_db=media_db,
                    chachanotes_db=chachanotes_db,
                    rag_service=rag_service,
                )
            )
        except Exception as e:
            logger.error(f"RAG index backfill crashed: {e}")
            self.app.call_from_thread(
                self.app.notify, f"Backfill failed: {e}", severity="error"
            )
            return
        finally:
            self.app.call_from_thread(self._clear_library_rag_backfill_in_flight)
        status = summary.get("status")
        errors = summary.get("errors") or []
        if status in ("unavailable", "error") or errors:
            last_error = str(errors[-1]) if errors else None
            detail = f" Last error: {last_error}" if last_error else ""
            self.app.call_from_thread(
                self.app.notify,
                f"Backfill finished with problems: {summary.get('indexed', 0)} "
                f"indexed, {summary.get('failed', 0)} failed.{detail}",
                severity="error" if status == "error" else "warning",
            )
        else:
            self.app.call_from_thread(
                self.app.notify,
                f"Backfill complete: {summary.get('indexed', 0)} indexed, "
                f"{summary.get('skipped', 0)} already up-to-date.",
                severity="information",
            )
        self.app.call_from_thread(self._refresh_library_rag_index_status)

    def _render_library_rag_profile_block(self) -> ComposeResult:
        info = active_profile_info()
        grouped = list_profiles_grouped()
        options = self._library_rag_profile_select_options(grouped)
        valid_ids = {value for _, value in options}
        active_id = grouped["active_id"]
        # PR #863 review: fall back to `Select.NULL`, NOT `Select.BLANK` --
        # the latter doesn't exist on this Textual version's `Select`
        # (resolves to `Widget.BLANK` == False) and composing
        # `Select(value=False)` raises `InvalidSelectValueError`, breaking
        # the whole Settings screen the one time this fallback is hit (a
        # stale/deleted active-profile pointer -> synthetic "(missing)"
        # active id absent from the options).
        select_value = active_id if active_id in valid_ids else Select.NULL
        active_label = f"{info['name']} (built-in)" if info["read_only"] else info["name"]

        # Task 4 (541 v2 UX AC1): the "Profiles" heading is now the
        # enclosing container's border title (see `_render_library_rag_detail`)
        # instead of an inline Static -- avoids a doubled label.
        yield Static(
            f"Active: {active_label}",
            id="settings-library-rag-active-profile",
            classes="settings-detail-row",
        )
        # UX review item 6 (provenance): the active profile's own
        # description, most useful for a first-run "Imported settings"
        # snapshot -- hidden entirely (not just blank) when there's none.
        description_row = Static(
            info["description"],
            id="settings-library-rag-active-profile-description",
            classes="settings-status-row",
        )
        description_row.display = bool(info["description"])
        yield description_row
        with Horizontal(classes="settings-input-row settings-select-row"):
            yield Static("Profile", classes="settings-input-label")
            yield Select(
                options,
                value=select_value,
                id="settings-library-rag-profile-select",
                classes="settings-compact-select",
                allow_blank=True,
                compact=True,
            )
        # UX review item 2 (decoupling caption): the Select above lets a
        # user BROWSE profiles without editing them -- only "Set active"
        # actually switches which profile the fields below edit. Without
        # this line, picking a different profile in the dropdown reads as
        # "now editing that one" even though nothing has happened yet.
        # Always names the ACTIVE profile (never the Select's current
        # highlight); refreshed by _sync_library_rag_profile_widgets.
        yield Static(
            f"Editing: {info['name']}. Pick a profile and press 'Set active' "
            "to edit a different one.",
            id="settings-library-rag-editing-caption",
            classes="settings-status-row",
        )
        with Horizontal(classes="settings-action-row"):
            yield Button("Set active", id="settings-library-rag-profile-set-active")
            yield Button("Clone…", id="settings-library-rag-profile-clone")
            yield Button("Rename…", id="settings-library-rag-profile-rename")
            # UX review item 7 (delete danger styling): destructive action,
            # visually separated from the other three (margin-left, see
            # .settings-library-rag-profile-delete-button) and given the
            # repo's standard destructive-button variant (see e.g.
            # settings_theme_editor.py's own "Delete" button).
            yield Button(
                "Delete",
                id="settings-library-rag-profile-delete",
                variant="error",
                classes="settings-library-rag-profile-delete-button",
            )
        readonly_banner = Static(
            "Built-in profile — read-only. Clone it, then press Set active to edit the clone.",
            id="settings-library-rag-profile-readonly-banner",
            classes="settings-status-row settings-library-rag-readonly-banner",
        )
        readonly_banner.display = bool(info["read_only"])
        yield readonly_banner
        yield Static(
            self._library_rag_profile_result,
            id="settings-library-rag-profile-result",
            classes="settings-status-row",
        )
        # Task 4 (SP3): index status readout. Compose ALWAYS renders the
        # "checking…" placeholder -- the real state is fetched off-thread
        # (touches on-disk Chroma) by the category-show/set-active/save
        # triggers, never here.
        #
        # Deliberately NOT a `Static` + `Button` sharing one
        # `.settings-action-row` Horizontal: `.settings-detail-row`/
        # `.settings-status-row` are `width: 100%`, which -- inside a
        # Horizontal -- claims the whole row and pushes/clips a sibling
        # Button past the visible edge (the same class of bug already hit
        # once in this program, RAG-Scope-button-clipped-off-rail). Own
        # full-width row for the Static, Button in its own
        # `.settings-action-row` right below it, exactly like the
        # Set active/Clone/Rename/Delete row above.
        yield Static(
            self._library_rag_index_status_text,
            id="settings-library-rag-index-status",
            classes="settings-detail-row",
        )
        with Horizontal(classes="settings-action-row"):
            yield Button("Backfill", id="settings-library-rag-index-backfill")

    def _queue_rag_select_suppression(self, select: Select, expected_value: object) -> None:
        """Record ``expected_value`` as a ``Select.Changed`` value the NEXT
        (about-to-happen) programmatic mutation of ``select`` is expected to
        post -- see ``_rag_select_suppress_queue``'s docstring for why this
        replaces a boolean in-progress flag. Only queues an expectation when
        the mutation will actually change ``select.value`` (mirrors
        ``Reactive._set``'s own "only fires when the value differs" gate,
        which is what decides whether Textual posts a Changed at all): an
        unconditional queue entry a mutation never triggers a Changed for
        would sit there forever and wrongly swallow a LATER, unrelated
        genuine user selection that happens to land on that same value.
        """
        if select.value != expected_value:
            self._rag_select_suppress_queue.append(expected_value)

    def _set_library_rag_editing_caption_visible(self, visible: bool) -> None:
        """541-v2 final review item 2: the decoupling caption ("Editing: X.
        Pick a profile and press 'Set active' to edit a different one.")
        always names the ACTIVE profile -- directly contradictory sitting
        right above the editor's "Previewing: Y" title while a
        profile-picker PREVIEW is active. Hidden entirely during preview
        rather than reworded: the preview banner just below the Select
        already carries the equivalent "Previewing '<name>' (read-only) —
        press Set active to edit it" message.
        """
        try:
            caption = self.query_one("#settings-library-rag-editing-caption", Static)
        except QueryError:
            return
        caption.display = visible

    def _refresh_library_rag_active_profile_identity_text(self) -> None:
        """Refresh ONLY the active-profile-NAME-bearing text (the "Active:
        .../Editing: ..." rows and the editor card's border_title), without
        touching the profile Select's value/options or the editor's field
        values -- task-629.

        Extracted out of ``_sync_library_rag_profile_widgets`` (which still
        calls this first, preserving identical behaviour for its own
        callers) so it can ALSO be called from completion paths that never
        touch the Select but can still leave this text stale: specifically
        ``_apply_library_rag_index_status`` (category-show / 't' test /
        Backfill completion / set-active's index-status hop). Backfill in
        particular calls ``get_shared_rag_service()``, whose first-ever call
        in the process silently imports-and-activates the "Imported
        settings" profile (``ensure_imported_profile``, see
        ``RAG_Search/simplified/active_config.py``) as a side effect -- an
        active-profile-pointer flip this screen previously had NO resync
        path for until the user's NEXT direct profile action (e.g. Clone)
        incidentally called ``_sync_library_rag_profile_widgets`` and
        exposed the new name for the first time, out of nowhere.

        Guarded by ``_rag_preview_profile_id`` for the caption/border_title
        half only (never fights an in-progress PREVIEW's own "Previewing:
        ..." title/banner) -- the "Active: ..." summary row and its
        description are not preview-dependent and always reflect the true
        active profile.
        """
        info = active_profile_info()
        active_label = f"{info['name']} (built-in)" if info["read_only"] else info["name"]
        self._set_static_text(
            "#settings-library-rag-active-profile", f"Active: {active_label}"
        )
        try:
            description_row = self.query_one(
                "#settings-library-rag-active-profile-description", Static
            )
            description_row.update(info["description"])
            description_row.display = bool(info["description"])
        except QueryError:
            pass
        if self._rag_preview_profile_id is not None:
            return
        # UX review item 2 (decoupling caption): always names the ACTIVE
        # profile, independent of select_override / whatever the Select is
        # currently showing.
        self._set_static_text(
            "#settings-library-rag-editing-caption",
            f"Editing: {info['name']}. Pick a profile and press 'Set active' "
            "to edit a different one.",
        )
        # 541-v2 final review item 2: this resync always runs OUTSIDE a
        # preview (see class docstring on `_rag_select_suppress_queue`), so
        # unconditionally un-hide the caption here too -- covers the
        # set-active/clone/rename/delete exit-preview paths, which clear
        # `_rag_preview_profile_id` and call this method directly rather
        # than routing back through `_sync_rag_editor_display` (which
        # handles the "browsed back to the active profile" exit path).
        self._set_library_rag_editing_caption_visible(True)
        self._update_library_rag_editor_title()

    def _sync_library_rag_profile_widgets(
        self, *, select_override: str | None = None
    ) -> None:
        """Refresh the Profiles block imperatively (no recompose) after any
        set-active/clone/rename/delete action, and after a category revert.

        Mirrors ``_sync_library_rag_widgets``: query-and-update each widget,
        swallowing ``QueryError`` so a not-yet-mounted region never crashes a
        call from an off-thread worker's main-thread callback.

        Args:
            select_override: UX review item 1 (clone flow) -- when given (a
                valid profile id), the profile Select is left pointed at
                THIS profile rather than snapped back to the active one, so
                a just-cloned profile stays highlighted/selected for the
                user's next click ("Set active"). Callers that don't pass it
                keep the pre-existing "always show the active profile"
                behaviour (set-active/rename/delete/revert).
        """
        self._refresh_library_rag_active_profile_identity_text()
        info = active_profile_info()
        grouped = list_profiles_grouped()

        options = self._library_rag_profile_select_options(grouped)
        valid_ids = {value for _, value in options}
        active_id = grouped["active_id"]
        target_id = (
            select_override
            if select_override is not None and select_override in valid_ids
            else active_id
        )
        try:
            select = self.query_one("#settings-library-rag-profile-select", Select)
            # 541-v2 final review item 1: `set_options` alone resets the
            # selection to `Select.NULL` (posting a transient Changed, see
            # `Select._init_selected_option`), then the explicit assignment
            # below posts a second one -- both are recorded as expected
            # BEFORE the write that causes them so
            # handle_library_rag_profile_select_changed can consume and
            # ignore each once it actually arrives (see
            # `_rag_select_suppress_queue`'s docstring for why a boolean
            # flag here cannot work). NOTE: the resolved-value fallback
            # deliberately uses `Select.NULL`, not `Select.BLANK` --
            # `Select.BLANK` doesn't exist on this Textual version's
            # `Select`; the attribute lookup silently resolves to the
            # unrelated `Widget.BLANK` (`False`), and assigning that to
            # `.value` would raise `InvalidSelectValueError` the one time
            # this fallback (active id no longer a valid option) is ever
            # actually hit.
            resolved_target = target_id if target_id in valid_ids else Select.NULL
            self._queue_rag_select_suppression(select, Select.NULL)
            select.set_options(options)
            self._queue_rag_select_suppression(select, resolved_target)
            select.value = resolved_target
        except QueryError:
            pass

        try:
            self.query_one(
                "#settings-library-rag-profile-readonly-banner", Static
            ).display = bool(info["read_only"])
        except QueryError:
            pass

        # reranker_model/reranker_top_k are handled below via
        # _apply_library_rag_rerank_field_state instead of the blanket
        # read-only-only treatment `_LIBRARY_RAG_READ_LOCK_FIELD_KEYS` covers:
        # their disabled state also depends on whether reranking itself is
        # enabled, and this method runs AFTER _sync_library_rag_widgets on
        # every set-active/clone/rename/delete resync, so a naive
        # `disabled = read_only` here would silently re-enable them whenever
        # the active (non-builtin) profile just switched to has reranking off.
        for key in _LIBRARY_RAG_READ_LOCK_FIELD_KEYS:
            selector = self._library_rag_field_selector(key)
            if selector is None:
                continue
            try:
                self.query_one(selector).disabled = bool(info["read_only"])
            except QueryError:
                pass
        self._apply_library_rag_rerank_field_state(
            rerank_enabled=bool(self._library_rag_loaded_values()["enable_reranking"]),
            field_disabled=bool(info["read_only"]),
        )
        for selector in _LIBRARY_RAG_READ_LOCK_CHECKBOX_SELECTORS:
            try:
                self.query_one(selector, Checkbox).disabled = bool(info["read_only"])
            except QueryError:
                pass

    def _render_library_rag_detail(self) -> ComposeResult:
        values = self._library_rag_setting_values()
        search_mode = normalise_library_rag_search_mode(values["default_search_mode"])
        citation_style = normalise_library_rag_citation_style(values["citation_style"])
        chunking_method = normalise_library_rag_chunking_method(
            values["chunking_method"]
        )
        distance_metric = normalise_library_rag_distance_metric(
            values["distance_metric"]
        )
        # A built-in active profile is read-only: the editor fields render
        # disabled from the very first paint (not just after a later
        # set-active/clone/rename/delete action re-syncs them) -- this is the
        # state a brand-new install starts in (active = the "hybrid_basic"
        # builtin).
        info = active_profile_info()
        field_disabled = info["read_only"]
        rerank_enabled = bool(values["enable_reranking"])
        rerank_field_disabled, rerank_suffix = self._library_rag_rerank_field_state(
            rerank_enabled=rerank_enabled, field_disabled=field_disabled
        )
        # Task 5 (541 v2 UX AC5): whether to show the first-run starter
        # panel INSTEAD of the "wall of disabled fields" as the first
        # impression -- computed from whatever index status is already
        # cached (never a fresh fetch of its own; see
        # `_library_rag_first_run_active`). On the very first-ever compose
        # (nothing fetched yet) this reads False -- compose optimistically
        # WITHOUT the panel, exactly like the index-status row itself
        # always composes its "checking…" placeholder; the real state
        # lands imperatively once the off-thread fetch resolves (see
        # `_apply_library_rag_index_status` -> `_refresh_rag_first_run_panel_state`,
        # which flips this same panel + collapses Search below, without a
        # recompose). A revisit within the same session, where the cache is
        # already warm, composes the correct first-run appearance from the
        # very first paint -- no flicker either way.
        first_run = self._library_rag_first_run_active(info=info)
        # Task 5 review (Important): keep `_rag_first_run_active` -- the
        # transition tracker `_refresh_rag_first_run_panel_state` uses to
        # decide whether to force Search's collapsed state -- in lockstep
        # with whatever THIS compose actually rendered. Without this, a
        # recompose that lands with a warm cache (composing the correct
        # first-run appearance directly, per the comment above, entirely
        # bypassing `_refresh_rag_first_run_panel_state`) would leave the
        # tracker stale at its pre-compose value, so the NEXT first-run-exit
        # trigger (clone/backfill completing) could wrongly see "no
        # transition" and skip re-expanding Search.
        self._rag_first_run_active = first_run

        yield Static(
            "RAG", classes="destination-section settings-column-title"
        )
        with Vertical(id="settings-library-rag-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.LIBRARY_RAG)
            # Task 4 (541 v2 UX AC1): manage-vs-edit split -- the picker
            # (browse/Set active/Clone/Rename/Delete) and the editor
            # (Search/Embedding/Chunking/Vector store/Reranking fields) each
            # get their OWN titled container, rather than one undifferentiated
            # card. The editor's title flips to "Previewing: <name>" while
            # browsing a non-active profile (see `_update_library_rag_editor_title`).
            profiles_card = Vertical(
                id="settings-library-rag-profiles-card",
                classes="settings-secondary-card",
            )
            profiles_card.border_title = "Profiles"
            with profiles_card:
                yield from self._render_library_rag_profile_block()
            # Task 5 (541 v2 UX AC5): ALWAYS composed (never conditionally
            # mounted) so the post-fetch toggle is a plain `.display` flip,
            # never a dynamic mount/unmount -- see
            # `_refresh_rag_first_run_panel_state`.
            starter_panel = Vertical(
                id="settings-library-rag-starter-panel",
                classes="settings-secondary-card settings-library-rag-starter-panel",
            )
            starter_panel.display = first_run
            with starter_panel:
                yield Static(
                    self._library_rag_starter_panel_copy(info),
                    id="settings-library-rag-starter-copy",
                    classes="settings-detail-row",
                )
                with Horizontal(classes="settings-action-row"):
                    yield Button(
                        "Clone to tune…", id="settings-library-rag-starter-clone"
                    )
                    yield Button(
                        "Backfill now", id="settings-library-rag-starter-backfill"
                    )
            editor_card = Vertical(
                id="settings-library-rag-editor-card",
                classes="settings-secondary-card",
            )
            editor_card.border_title = f"Editing: {escape_markup(info['name'])}"
            with editor_card:
                yield from self._render_library_rag_editor_fields(
                    values=values,
                    search_mode=search_mode,
                    citation_style=citation_style,
                    chunking_method=chunking_method,
                    distance_metric=distance_metric,
                    field_disabled=field_disabled,
                    rerank_enabled=rerank_enabled,
                    rerank_field_disabled=rerank_field_disabled,
                    rerank_suffix=rerank_suffix,
                    search_group_collapsed=first_run,
                )

    def _render_library_rag_editor_fields(
        self,
        *,
        values: dict[str, object],
        search_mode: str,
        citation_style: str,
        chunking_method: str,
        distance_metric: str,
        field_disabled: bool,
        rerank_enabled: bool,
        rerank_field_disabled: bool,
        rerank_suffix: str,
        search_group_collapsed: bool = False,
    ) -> ComposeResult:
        # Task 4 (541 v2 UX AC1): read-only preview banner -- hidden at
        # first paint (the Select always starts pinned to the active
        # profile, see `_render_library_rag_profile_block`); shown by
        # `_set_library_rag_preview_banner` while browsing another profile.
        preview_banner = Static(
            "",
            id="settings-library-rag-preview-banner",
            classes="settings-status-row settings-library-rag-preview-banner",
        )
        preview_banner.display = False
        yield preview_banner
        # UX review item 5 (⚠ legend): the ⚠ markers on individual field
        # labels below (Embedding model, Max length, Chunk size/overlap/
        # method, Distance metric) are otherwise unexplained the first
        # time a user sees one.
        yield Static(
            "⚠ = changing this field rebuilds the index — run Backfill "
            "after saving.",
            id="settings-library-rag-warning-legend",
            classes="settings-status-row",
        )
        with Collapsible(
            title="Search",
            # Task 5 (541 v2 UX AC5): the ONLY group that composes expanded
            # by default (Embedding/Chunking/Vector store/Reranking already
            # compose collapsed) -- collapsed too while first-run, so the
            # starter panel above, not this wall of disabled fields, is the
            # first impression.
            collapsed=search_group_collapsed,
            id="settings-library-rag-search-group",
        ):
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
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Default results", classes="settings-input-label")
                yield Input(
                    value=str(values["default_top_k"]),
                    id="settings-library-rag-default-top-k",
                    classes="settings-compact-input",
                    placeholder="1 - 100",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
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
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Vector results", classes="settings-input-label")
                yield Input(
                    value=str(values["vector_top_k"]),
                    id="settings-library-rag-vector-top-k",
                    classes="settings-compact-input",
                    placeholder="1 - 100",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Hybrid balance", classes="settings-input-label")
                yield Input(
                    value=str(values["hybrid_alpha"]),
                    id="settings-library-rag-hybrid-alpha",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Min score", classes="settings-input-label")
                yield Input(
                    value=str(values["score_threshold"]),
                    id="settings-library-rag-score-threshold",
                    classes="settings-compact-input",
                    placeholder="0.0 - 1.0",
                    disabled=field_disabled,
                )
            yield Static("Citation and snippets", classes="destination-section")
            yield Checkbox(
                "Include citations",
                value=bool(values["include_citations"]),
                id="settings-library-rag-include-citations",
                tooltip="Toggle citation metadata in future RAG answers where supported.",
                disabled=field_disabled,
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
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Snippet chars", classes="settings-input-label")
                yield Input(
                    value=str(values["snippet_max_chars"]),
                    id="settings-library-rag-snippet-max-chars",
                    classes="settings-compact-input",
                    placeholder="50 - 10000",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Context budget", classes="settings-input-label")
                yield Input(
                    value=str(values["max_context_size"]),
                    id="settings-library-rag-max-context-size",
                    classes="settings-compact-input",
                    placeholder="1000 - 1000000",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
        with Collapsible(
            title="Embedding",
            collapsed=True,
            id="settings-library-rag-embedding-group",
        ):
            yield Static(
                "Changing this changes what the index is built from -- the "
                "index must be rebuilt (run Backfill).",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Embedding model ⚠", classes="settings-input-label"
                )
                yield Input(
                    value=str(values["embedding_model"]),
                    id="settings-library-rag-embedding-model",
                    classes="settings-compact-input",
                    placeholder="e.g. mxbai-embed-large-v1",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Device", classes="settings-input-label")
                yield Input(
                    value=str(values["embedding_device"]),
                    id="settings-library-rag-embedding-device",
                    classes="settings-compact-input",
                    placeholder="auto, cpu, cuda, mps",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Batch size", classes="settings-input-label")
                yield Input(
                    value=str(values["embedding_batch_size"]),
                    id="settings-library-rag-embedding-batch-size",
                    classes="settings-compact-input",
                    placeholder="> 0",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Max length ⚠", classes="settings-input-label"
                )
                yield Input(
                    value=str(values["embedding_max_length"]),
                    id="settings-library-rag-embedding-max-length",
                    classes="settings-compact-input",
                    placeholder="> 0",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
        with Collapsible(
            title="Chunking",
            collapsed=True,
            id="settings-library-rag-chunking-group",
        ):
            yield Static(
                "Changing this changes what the index is built from -- the "
                "index must be rebuilt (run Backfill).",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Chunk size ⚠", classes="settings-input-label"
                )
                yield Input(
                    value=str(values["chunk_size"]),
                    id="settings-library-rag-chunk-size",
                    classes="settings-compact-input",
                    placeholder="> 0 words",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Chunk overlap ⚠", classes="settings-input-label"
                )
                yield Input(
                    value=str(values["chunk_overlap"]),
                    id="settings-library-rag-chunk-overlap",
                    classes="settings-compact-input",
                    placeholder="0 - chunk size",
                    restrict=r"^[0-9]*$",
                    disabled=field_disabled,
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static(
                    "Method ⚠", classes="settings-input-label"
                )
                yield Select(
                    [
                        ("Words", "words"),
                        ("Sentences", "sentences"),
                        ("Paragraphs", "paragraphs"),
                    ],
                    value=chunking_method,
                    id="settings-library-rag-chunking-method",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                    disabled=field_disabled,
                )
        with Collapsible(
            title="Vector store",
            collapsed=True,
            id="settings-library-rag-vector-store-group",
        ):
            yield Static(
                "Changing this changes what the index is built from -- the "
                "index must be rebuilt (run Backfill).",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static(
                    "Distance metric ⚠",
                    classes="settings-input-label",
                )
                yield Select(
                    [
                        ("Cosine", "cosine"),
                        ("Euclidean (L2)", "l2"),
                        ("Inner product", "ip"),
                    ],
                    value=distance_metric,
                    id="settings-library-rag-distance-metric",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                    disabled=field_disabled,
                )
        with Collapsible(
            title="Reranking",
            collapsed=True,
            id="settings-library-rag-reranking-group",
        ):
            yield Static(
                "Enabling reranking creates the profile's reranker config; "
                "disabling it removes that config entirely.",
                classes="settings-detail-row",
            )
            yield Checkbox(
                "Enable reranking",
                value=rerank_enabled,
                id="settings-library-rag-enable-reranking",
                tooltip="Toggle LLM-based reranking of retrieved results for this profile.",
                disabled=field_disabled,
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    f"Reranker model{rerank_suffix}",
                    id="settings-library-rag-reranker-model-label",
                    classes="settings-input-label",
                )
                yield Input(
                    value=str(values["reranker_model"]),
                    id="settings-library-rag-reranker-model",
                    classes="settings-compact-input",
                    placeholder="blank = reranker default",
                    disabled=rerank_field_disabled,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    f"Rerank results{rerank_suffix}",
                    id="settings-library-rag-reranker-top-k-label",
                    classes="settings-input-label",
                )
                yield Input(
                    value=str(values["reranker_top_k"]),
                    id="settings-library-rag-reranker-top-k",
                    classes="settings-compact-input",
                    placeholder=">= 1",
                    restrict=r"^[0-9]*$",
                    disabled=rerank_field_disabled,
                )
            soft_warnings = self._library_rag_soft_warnings()
            reranker_warning = Static(
                " / ".join(soft_warnings),
                id="settings-library-rag-reranker-warning",
                classes="settings-status-row settings-library-rag-soft-warning",
            )
            reranker_warning.display = bool(soft_warnings)
            yield reranker_warning
        yield Static("Preview defaults", classes="destination-section")
        preview_summary, preview_retrieval, preview_context = (
            self._library_rag_preview_rows()
        )
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
        yield self._detail_row(
            "Profile", "the active RAG profile (rag_profiles/<id>.json)"
        )
        yield self._detail_row("Pointer", "the [rag.service].profile pointer")
        yield Static(
            self._library_rag_result,
            id="settings-library-rag-save-result",
            classes="settings-status-row",
        )

    def _render_domain_category_detail(
        self, category: SettingsCategoryId
    ) -> ComposeResult:
        contract = self._domain_category_contract(category)
        yield Static(
            contract.title, classes="destination-section settings-column-title"
        )
        with Vertical(
            id=f"settings-{category.value}-card", classes="settings-focus-card"
        ):
            yield self._render_category_state_banner(category)
            yield Static("How this page works", classes="destination-section")
            yield self._detail_row("Owner destination", contract.owner_destination)
            yield self._detail_row(
                "Settings mode", "View only - shows current defaults and status"
            )
            yield self._detail_row(
                "Writes allowed",
                f"No - change this in {contract.owner_destination} instead",
            )
            yield Static("Source of truth", classes="destination-section")
            for index, source in enumerate(contract.source_of_truth, start=1):
                yield self._detail_row(f"Source {index}", source)
            yield Static("Status and default boundaries", classes="destination-section")
            for label, value in contract.rows:
                yield self._detail_row(label, value)
            yield self._detail_row("Follow-up", contract.follow_up)

    def _render_workspaces_detail(self) -> ComposeResult:
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        yield Static("Workspace management", classes="destination-section")
        if registry is None:
            yield Static(
                "Workspace service is not ready. Restart Chatbook and retry.",
                id="settings-workspaces-result",
                classes="settings-status-row",
            )
            return
        show_archived = bool(self._settings_show_archived_workspaces)
        active = registry.get_active_workspace()
        active_id = active.workspace_id if active is not None else None
        with Horizontal(classes="settings-input-row"):
            yield Input(
                placeholder="New workspace name",
                id="settings-workspace-create-name",
                classes="settings-compact-input",
            )
            yield Button("Create", id="settings-workspace-create", compact=True)
        yield Checkbox(
            "Show archived", show_archived, id="settings-workspaces-show-archived"
        )
        with Vertical(id="settings-workspaces-list"):
            for record in registry.list_workspaces(include_archived=show_archived):
                marker = " (active)" if record.workspace_id == active_id else ""
                archived_suffix = " [archived]" if record.archived else ""
                folders = (
                    len(registry.list_folder_bindings(record.workspace_id))
                    if record.workspace_id != DEFAULT_WORKSPACE_ID
                    else 0
                )
                yield Button(
                    f"{record.name}{marker}{archived_suffix} - {folders} folders",
                    id=f"settings-workspace-row-{record.workspace_id}",
                    classes="settings-workspace-row",
                    compact=True,
                )
        yield Static(
            self._settings_workspaces_result,
            id="settings-workspaces-result",
            classes="settings-status-row",
        )
        yield from self._render_workspace_card(registry, active_id)

    def _render_workspace_card(
        self,
        registry: LocalWorkspaceRegistryService,
        active_id: str | None,
    ) -> ComposeResult:
        """Render the selected workspace's lifecycle card, if any (Task 9).

        Renders nothing when no workspace is selected. The built-in Default
        workspace gets ONLY the protection notice -- it keeps its identity
        (no rename/archive) and stays tool-less (no folder bindings, see
        Task 10). An archived workspace (final review Finding 3) gets ONLY
        an explanatory note + Unarchive -- rename/set-active/archive/folder
        controls are withheld since they act on a workspace_id that is
        currently archived. Every other workspace gets rename + set-active
        (or a Static when it is already active -- never a disabled Button
        expected to explain itself) + archive + folder bindings.
        """
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            # task-1585: rendering nothing here left the center pane a
            # near-empty box -- say what selecting does instead.
            yield Static(
                "Select a workspace above to rename it, set it active, "
                "archive it, or bind folders.",
                id="settings-workspace-card-hint",
                classes="settings-detail-row",
            )
            return
        record = registry.get_workspace(workspace_id)
        if record is None:
            # The selection outlived its workspace (e.g. removed by another
            # session) -- render nothing rather than a card for a ghost id.
            return
        yield Static("Selected workspace", classes="destination-section")
        with Vertical(id="settings-workspace-card", classes="settings-focus-card"):
            if record.workspace_id == DEFAULT_WORKSPACE_ID:
                yield Static(
                    "The built-in Default workspace keeps its identity and "
                    "stays tool-less; create a workspace to bind folders.",
                    classes="settings-detail-row",
                )
                return
            if record.archived:
                # Finding 3 (final review): rename/set-active/archive/folder
                # controls all require an ACTIVE workspace_id underneath --
                # offering them here let a user hit a bare-id error acting
                # on a workspace that is currently invisible everywhere
                # else. Unarchive first restores it to normal editing.
                yield Static(
                    "Archived workspace. Unarchive it to rename, activate, "
                    "or edit folders.",
                    id="settings-workspace-archived-note",
                    classes="settings-status-row",
                )
                yield Button(
                    "Unarchive", id="settings-workspace-unarchive", compact=True
                )
                return
            with Horizontal(classes="settings-input-row"):
                yield Input(
                    value=record.name,
                    id="settings-workspace-rename-input",
                    classes="settings-compact-input",
                )
                yield Button(
                    "Rename", id="settings-workspace-rename-apply", compact=True
                )
            if record.workspace_id == active_id:
                yield Static(
                    "This workspace is active.", classes="settings-detail-row"
                )
            else:
                yield Button(
                    "Set active", id="settings-workspace-set-active", compact=True
                )
            yield Button(
                "Archive", id="settings-workspace-archive", compact=True
            )
            yield from self._render_workspace_folder_bindings(registry, record.workspace_id)

    def _render_workspace_folder_bindings(
        self,
        registry: LocalWorkspaceRegistryService,
        workspace_id: str,
    ) -> ComposeResult:
        """Render the folder-bindings editor for the selected workspace (task 10).

        One row per bound folder with its access level and freshness
        (recomputed from disk by `list_folder_bindings`), a per-row
        ro/rw toggle and remove button, then an add row. Toggle/remove
        buttons stash `binding_id` as a plain attribute at compose time
        (mirrors the conversation browser's `conversation_id` stash) so
        the handler never has to parse a uuid out of the button id.
        """
        yield Static(
            "Folders (agent file-tool access)", classes="destination-section"
        )
        for binding in registry.list_folder_bindings(workspace_id):
            access = binding.metadata.get("access", "ro")
            freshness = (
                "ready" if binding.status == RuntimeBindingStatus.READY else "missing"
            )
            yield Static(
                f"{binding.locator} [{access}] {freshness}",
                id=f"settings-workspace-folder-{binding.binding_id}",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                toggle_button = Button(
                    "Allow write" if access != "rw" else "Read-only",
                    id=f"settings-workspace-folder-toggle-{binding.binding_id}",
                    classes="settings-workspace-folder-toggle",
                    compact=True,
                )
                toggle_button.binding_id = binding.binding_id
                yield toggle_button
                remove_button = Button(
                    "Remove",
                    id=f"settings-workspace-folder-remove-{binding.binding_id}",
                    classes="settings-workspace-folder-remove",
                    compact=True,
                )
                remove_button.binding_id = binding.binding_id
                yield remove_button
        with Horizontal(classes="settings-input-row"):
            yield Input(
                placeholder="~/path/to/folder",
                id="settings-workspace-folder-path",
                classes="settings-compact-input",
            )
            yield Button(
                "Add folder", id="settings-workspace-folder-add", compact=True
            )

    def _set_settings_workspaces_result(self, text: str) -> None:
        self._settings_workspaces_result = text
        self._set_static_text("#settings-workspaces-result", text)

    def _refresh_settings_workspaces_pane(self) -> None:
        """Re-render the Workspaces category via the screen's existing
        category-recompose path (task 9).

        `active_category` is a `recompose=True` reactive that already
        drives every category switch (`_select_category`); forcing it here
        with `mutate_reactive` reuses that same, already-proven path
        instead of recomposing a bespoke nested container (Textual only
        regenerates a widget's children from ITS OWN `compose()` -- a
        generic `Vertical` yielded inline here has none, so recomposing it
        directly would just wipe it). `_render_workspaces_detail` reads
        the registry plus the plain `_settings_selected_workspace_id` /
        `_settings_show_archived_workspaces` attributes fresh on every
        call, so there is no separate watcher-populated cache that could
        go stale or get wiped by the recompose.
        """
        self.mutate_reactive(SettingsScreen.active_category)

    def _render_detail_pane(self) -> ComposeResult:
        category = SettingsCategoryId(self.active_category)
        if category is SettingsCategoryId.OVERVIEW:
            yield from self._render_overview_detail()
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            yield from self._render_provider_detail()
        elif category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            yield Static(
                "Console Behavior", classes="destination-section settings-column-title"
            )
            with Vertical(
                id="settings-console-behavior-detail", classes="settings-focus-card"
            ):
                yield self._render_category_state_banner(
                    SettingsCategoryId.CONSOLE_BEHAVIOR
                )
                yield from self._render_console_behavior_card(compact=False)
                yield Static("Composer behavior", classes="destination-section")
                yield self._detail_row(
                    "Paste collapse",
                    "pasted chunks over the threshold display as compact placeholders",
                )
                yield self._detail_row(
                    "Threshold", self._paste_collapse_threshold_label()
                )
                yield self._detail_row(
                    "Typing rule",
                    "normal typing remains literal and never auto-collapses",
                )
                yield self._detail_row(
                    "Current default", self._collapse_large_pastes_label()
                )
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
            yield Static(
                "Appearance", classes="destination-section settings-column-title"
            )
            with Vertical(id="settings-appearance-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.APPEARANCE)
                yield Static("Global visual defaults", classes="destination-section")
                yield Static(
                    "Settings owns launch visual defaults. "
                    "Open the Theme category for full theme editing and deeper visual preview.",
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
                yield self._detail_row(
                    "Current summary", self._appearance_summary_text()
                )
                yield self._detail_row(
                    "Runtime preview", "applies safe values for this session only"
                )
                yield self._detail_row(
                    "Open Theme",
                    "full theme editor, custom colors, and deeper visual preview",
                )
                yield self._detail_row(
                    "Save targets", "general, web_server, and appearance"
                )
                with Horizontal(
                    id="settings-appearance-actions", classes="settings-action-row"
                ):
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
        elif category is SettingsCategoryId.THEME:
            yield Static("Theme", classes="destination-section settings-column-title")
            yield SettingsThemeEditor(id="settings-theme-editor")
        elif category is SettingsCategoryId.SPLASH_SCREEN:
            yield Static("Splash Screen", classes="destination-section settings-column-title")
            yield SettingsSplashScreenViewer(id="settings-splash-screen-viewer")
        elif category is SettingsCategoryId.INTERNAL_PROMPTS:
            yield Static("Internal Prompts", classes="destination-section settings-column-title")
            yield InternalPromptsPanel(id="settings-internal-prompts-panel")
        elif category is SettingsCategoryId.IMAGE_GENERATION:
            yield Static("Image Gen", classes="destination-section settings-column-title")
            image_gen_overlay = self._image_gen_overlay_values()
            self._queue_image_gen_select_suppression(image_gen_overlay)
            yield ImageGenSettingsPanel(
                id="settings-imagegen-panel",
                overlay=image_gen_overlay,
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
                yield self._detail_row(
                    "Scope", "persisted local database path defaults"
                )
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
                with Horizontal(
                    id="settings-storage-actions", classes="settings-action-row"
                ):
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
                yield Static(
                    "Database paths (configured)", classes="destination-section"
                )
                # TASK-720: the inputs edit config.toml values; the files a
                # session actually uses are resolved at runtime (a profile
                # from [general].users_name relocates defaults under a
                # per-profile directory). Without this note the configured
                # default and the resolved path read as two conflicting
                # current locations.
                yield Static(
                    "These are the configured config.toml values. The files "
                    "actually in use this session are listed under Active "
                    "files below and can differ when a user profile is set.",
                    id="settings-storage-configured-note",
                    classes="settings-status-row",
                )
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
                yield Static(
                    "Active files (resolved this session)",
                    classes="destination-section",
                )
                for path_summary in self._known_storage_paths():
                    yield self._split_detail_row(path_summary)
                yield self._detail_row(
                    "Config directory status", self._config_writable_status()
                )
                yield self._detail_row(
                    "Handoff boundary",
                    "database and media paths remain local unless a server handoff is explicit",
                )
        elif category is SettingsCategoryId.WORKSPACES:
            yield from self._render_workspaces_detail()
        elif category is SettingsCategoryId.PRIVACY_SECURITY:
            posture = self._settings_privacy_posture()
            yield Static(
                "Privacy & Security",
                classes="destination-section settings-column-title",
            )
            with Vertical(
                id="settings-privacy-security-card", classes="settings-focus-card"
            ):
                yield self._render_category_state_banner(
                    SettingsCategoryId.PRIVACY_SECURITY
                )
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
                    env_var_summary(
                        present=posture.provider_env_present,
                        missing=posture.provider_env_missing,
                        configured=posture.provider_env_configured,
                    ),
                )
                yield self._detail_row("Preferred source", "environment variables")
                yield self._detail_row(
                    "Config secrets", "counted only; raw values are never displayed"
                )
                yield self._detail_row(
                    "Recovery actions",
                    "Check Privacy | Open Providers & Models | Open Advanced Config",
                )
                yield self._detail_row(
                    "Skill trust",
                    skill_trust_display(posture.skill_trust_status)
                    if posture.skill_trust_enabled
                    else "disabled",
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
                with Horizontal(
                    id="settings-privacy-actions", classes="settings-action-row"
                ):
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
                    "not available yet - password-gated flow required",
                )
                yield Static(
                    self._privacy_check_text(),
                    id="settings-privacy-check-result",
                    classes="settings-status-row",
                )
        elif category is SettingsCategoryId.DIAGNOSTICS:
            yield Static(
                "Diagnostics", classes="destination-section settings-column-title"
            )
            with Vertical(
                id="settings-diagnostics-card", classes="settings-focus-card"
            ):
                yield self._render_category_state_banner(SettingsCategoryId.DIAGNOSTICS)
                yield Static("Validate config", classes="destination-section")
                yield self._detail_row("Config path", self._config_path())
                yield self._detail_row(
                    "Validation", "raw TOML validation before advanced edits"
                )
                yield self._detail_row(
                    "Reload", "load current config into the running app"
                )
                yield self._detail_row("Redaction", "actionable errors without secrets")
                yield self._detail_row("Write safety", "validation is read-only")
                yield self._detail_row(
                    "Diagnostics writes",
                    "not available yet - raw edits remain gated in Advanced Config",
                )
                with Horizontal(
                    id="settings-diagnostics-actions", classes="settings-action-row"
                ):
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
                    yield Button(
                        "Run Setup Wizard",
                        id="settings-run-setup-wizard",
                        tooltip="Re-run the guided first-run setup with current values.",
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
            yield Static(
                "Advanced Config", classes="destination-section settings-column-title"
            )
            with Vertical(
                id="settings-advanced-config-card", classes="settings-focus-card"
            ):
                raw_config_text = self._raw_config_text()
                yield self._render_category_state_banner(
                    SettingsCategoryId.ADVANCED_CONFIG
                )
                yield Static("Raw TOML", classes="destination-section")
                yield self._detail_row(
                    "Risk level", "expert-only raw configuration editing"
                )
                yield self._detail_row(
                    "Save policy",
                    "Save blocked until the current text validates",
                )
                yield self._detail_row(
                    "Write mode", "atomic save with .bak backup before overwrite"
                )
                yield self._detail_row(
                    "Required shape", "table-shaped TOML top-level value"
                )
                yield self._detail_row(
                    "Guided path",
                    "prefer category controls unless raw TOML is required",
                )
                yield Static("Guided category paths", classes="destination-section")
                with Horizontal(
                    id="settings-advanced-guided-paths", classes="settings-action-row"
                ):
                    for target_category, label in ADVANCED_CONFIG_GUIDED_PATHS:
                        yield Button(
                            label,
                            id=f"settings-advanced-open-{target_category.value}",
                            classes="settings-advanced-guided-path-button",
                            tooltip=f"Open {label} guided settings instead of editing raw TOML.",
                        )
                yield Static(
                    "Raw TOML bypasses guided validation and should be used only for expert edits."
                )
                yield Static(
                    self._advanced_validation_status(),
                    id="settings-advanced-config-validation-status",
                    classes="settings-status-row settings-advanced-safety-status",
                )
                with Horizontal(
                    id="settings-advanced-config-actions", classes="settings-action-row"
                ):
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
                    save_button.disabled = not self._advanced_save_allowed(
                        raw_config_text
                    )
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

    def _mode_line_text(self, summary: SettingsCategorySummary) -> str:
        """Mode-line text for the category strip.

        The MCP/ACP runtime disclaimer orients once on Overview; repeating
        it verbatim on all 17 categories made it standing noise the eye
        learns to skip (rescore P3).

        Args:
            summary: The active category's summary.

        Returns:
            "Mode: <title>", with the runtime disclaimer only on Overview.
        """
        if summary.category is SettingsCategoryId.OVERVIEW:
            return (
                f"Mode: {summary.title} | Runtime controls stay in MCP and ACP"
            )
        return f"Mode: {summary.title}"

    def _render_impact_pane_header(self) -> ComposeResult:
        """Fixed (non-scrolling) inspector header (task-1560/task-1562).

        Identity, draft status, guided-action state, and the Save/Revert
        pair are pinned above the scrollable body so the commit affordance
        can never scroll out of sight -- the critique's live dirty-state
        capture showed the rail scrolled with no visible Save anywhere.
        """
        summary = self._active_summary()
        yield Static(
            "Scope Inspector", classes="destination-section settings-column-title"
        )
        yield Static(
            f"Selected category: {summary.title}", classes="destination-section"
        )
        yield Static(
            "Unsaved changes"
            if self._category_has_unsaved_changes(summary.category)
            else "No unsaved changes",
            id="settings-selected-category-draft-status",
            classes="destination-section",
        )
        yield Static(
            self._guided_action_message(summary.category),
            id="settings-guided-action-state",
            classes="settings-status-row",
        )
        # task-1585: render the pair ONLY where the draft model acts --
        # previously read-only categories showed it permanently disabled
        # (dim-on-dim noise) while five own-persistence categories omitted
        # it, with no stated rule. Mirrors the task-1580 footer gating.
        if summary.category in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            dirty = self._category_has_unsaved_changes(summary.category)
            save_button = Button(
                self._guided_action_label("Save (s)", dirty=dirty),
                id="settings-save-category",
                tooltip="Save changes for the selected Settings category.",
            )
            save_button.disabled = not self._guided_actions_enabled(summary.category)
            yield save_button
            revert_button = Button(
                self._guided_action_label("Revert (r)", dirty=dirty),
                id="settings-revert-category",
                tooltip="Discard unsaved changes for the selected Settings category.",
            )
            revert_button.disabled = not self._guided_actions_enabled(summary.category)
            yield revert_button
        elif summary.category is SettingsCategoryId.OVERVIEW:
            yield Button(
                "Theme",
                id="settings-open-appearance",
                tooltip="Open the dedicated Theme editor.",
            )
        # task-181 copy, task-1583 placement: this reassurance line used to
        # close the SCROLLABLE body, where 8 of 20 critique captures cut it
        # mid-sentence ("Nothing is sent to" reads ominous truncated).
        # Pinned here it is always fully visible.
        yield Static(
            "Saves apply to your local config file. Nothing is sent to a server "
            "unless you run Manual sync yourself.",
            id="settings-local-scope-note",
        )

    def _render_impact_pane_body(self) -> ComposeResult:
        """Scrollable inspector remainder: guides, ownership, boundaries."""
        summary = self._active_summary()
        ownership = self._ownership_record(summary.category)
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
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(
                self._console_behavior_field_guidance_rows()
            ):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-console-behavior-field-guide-{index}",
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
            yield Static(
                "Affects Console and provider-backed generation.",
                classes="destination-section",
            )
            yield Static(
                self._provider_readiness_label(),
                id="settings-provider-inspector-readiness",
                classes="settings-detail-row",
            )
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(
                self._provider_field_guidance_rows()
            ):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-provider-field-guide-{index}",
                )
        elif summary.category is SettingsCategoryId.LIBRARY_RAG:
            # UX review item 9 (Scope Inspector clipping): a blank spacer
            # ahead of the RAG-specific guidance, separating it from the
            # shared Save/Revert buttons yielded just above (this branch is
            # the only content that changes per category; the Button pair
            # itself is shared plumbing rendered for every category and is
            # left alone -- see the report for why).
            yield Static("")
            yield Static(
                "Affects Library search defaults and future RAG answers.",
                classes="destination-section",
            )
            yield Static("Control guide", classes="destination-section")
            # Task 3 (541 v2 UX AC3): context-sensitive -- follows the
            # focused field or the last-expanded Collapsible group instead
            # of always showing the same static blurb (see
            # _rag_field_guidance_rows, refreshed by handle_descendant_focus
            # and handle_settings_library_rag_collapsible_toggled). Guidance
            # values are intentionally terse (UX review item 9): the
            # original prose wrapped across enough lines in this narrow rail
            # that the "Citations" row clipped mid-sentence ("...source
            # markers when") at the pane's unscrolled fold.
            for index, (label, value) in enumerate(
                self._rag_field_guidance_rows()
            ):
                yield self._detail_row(
                    label,
                    value,
                    identifier=f"settings-library-rag-field-guide-{index}",
                )
            yield Static("Boundary", classes="destination-section")
            yield self._detail_row(
                "Library owns",
                "indexing, query, source browse, Collections, Console staging",
            )
            yield self._detail_row("Runtime owner", ownership.runtime_owner)
            yield self._detail_row("Writes allowed", "Yes")
            yield self._detail_row(
                "Config keys",
                "10 editable defaults in the active RAG profile",
            )
            yield self._detail_row("Recovery", ownership.recovery_copy)
            return
        elif summary.category is SettingsCategoryId.APPEARANCE:
            yield Static(
                "Affects launch theme, web density, and visual defaults.",
                classes="destination-section",
            )
            yield Static("Focused field guide", classes="destination-section")
            for index, (label, value) in enumerate(
                self._appearance_field_guidance_rows()
            ):
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
                "Theme owns",
                "full theme editing, custom colors, and deeper preview",
            )
            yield Button(
                "Open Theme",
                id="settings-open-appearance",
                tooltip="Open the dedicated Theme editor.",
            )
        elif summary.category is SettingsCategoryId.THEME:
            yield Static("Affects app colors and saved custom themes.", classes="destination-section")
            yield Static("Focused field guide", classes="destination-section")
            yield self._detail_row("Save target", "~/.config/tldw_cli/themes/")
            yield self._detail_row("Note", "Use the editor's own Apply/Save/Reset buttons.")
            modified = "Yes" if self.theme_editor_modified else "No"
            yield self._detail_row(
                "Unsaved theme changes",
                modified,
                identifier="settings-theme-unsaved-note",
            )
        elif summary.category is SettingsCategoryId.SPLASH_SCREEN:
            yield Static("Affects startup splash screen behavior.", classes="destination-section")
            yield Static("Focused field guide", classes="destination-section")
            yield self._detail_row("Config section", "splash_screen")
            yield self._detail_row("Note", "Splash defaults are saved automatically.")
        elif summary.category is SettingsCategoryId.INTERNAL_PROMPTS:
            yield Static("Edit the prompts used by internal tooling.", classes="destination-section")
            yield self._detail_row("Save target", "~/.config/tldw_cli/config.toml  [internal_prompts]")
            yield self._detail_row("Note", "Use each prompt's own Save / Reset buttons.")
            yield self._detail_row(
                "Customized prompts",
                str(self._get_internal_prompts_customized_count()),
                identifier="settings-internal-prompts-customized-count",
            )
        elif summary.category is SettingsCategoryId.STORAGE:
            yield Static(
                "Affects local database path defaults after restart.",
                classes="destination-section",
            )
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
                "\n".join(ownership.owns_config_sections),
            )
        if ownership.read_only_reason:
            yield self._detail_row("Read-only", ownership.read_only_reason)
        for label, value in self._inspector_guidance(summary.category):
            yield self._detail_row(
                label,
                value,
                identifier="settings-boundary-note" if label == "Boundary" else None,
            )

    def compose_content(self) -> ComposeResult:
        active_summary = self._active_summary()
        with Vertical(id="settings-shell"):
            yield Static(
                "Settings | Global preferences, appearance, storage | Local",
                id="settings-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(
                id="settings-category-strip", classes="destination-mode-strip"
            ):
                yield Static(
                    self._mode_line_text(active_summary),
                    id="settings-category-label",
                    classes="destination-section",
                )
            with Horizontal(
                id="settings-workbench", classes="ds-panel destination-workbench"
            ):
                with Vertical(
                    id="settings-category-pane", classes="destination-workbench-pane"
                ):
                    yield Static(
                        "Settings Sections",
                        classes="destination-section settings-column-title",
                    )
                    yield SettingsCategorySearchInput(
                        value=self.category_search_query,
                        placeholder="Filter categories (/)",
                        id="settings-category-search",
                        classes="settings-category-search",
                    )
                    yield Static(
                        self._category_search_status_text(),
                        id="settings-category-search-status",
                        classes="settings-category-search-status",
                        markup=False,
                    )
                    with VerticalScroll(
                        id="settings-category-list",
                        classes="settings-category-list",
                    ):
                        yield from self._render_category_buttons()
                yield self._column_divider("settings-category-detail-divider")
                detail_pane_container = (
                    Vertical
                    if active_summary.category is SettingsCategoryId.ADVANCED_CONFIG
                    else VerticalScroll
                )
                with detail_pane_container(
                    id="settings-detail-pane", classes="destination-workbench-pane"
                ):
                    yield Static(
                        "Preference Detail",
                        classes="destination-section settings-column-title",
                    )
                    yield from self._render_detail_pane()
                yield self._column_divider("settings-detail-impact-divider")
                impact_pane = Vertical(
                    id="settings-impact-pane",
                    classes="destination-workbench-pane ds-inspector",
                )
                # Explicit height: under the real CSS bundle the pane class
                # sizes a scroll container, not a plain Vertical -- without
                # this the 1fr body below collapses to zero (StyledSettings
                # harness caught it; the plain harness cannot).
                impact_pane.styles.height = "100%"
                with impact_pane:
                    yield from self._render_impact_pane_header()
                    impact_body = VerticalScroll(id="settings-impact-pane-body")
                    # Inline styles, not CSS: the app-tier bundle outranks
                    # screen CSS and a 100%-height default would collapse
                    # inside the auto-flow wrapper (same guard as the image
                    # viewer modal).
                    impact_body.styles.height = "1fr"
                    impact_body.styles.scrollbar_size_vertical = 1
                    with impact_body:
                        yield from self._render_impact_pane_body()

    def _category_value_from_button(self, button: Button) -> str | None:
        if not button.id or not button.has_class("settings-category-button"):
            return None
        prefix = "settings-category-"
        if not button.id.startswith(prefix):
            return None
        value = button.id.removeprefix(prefix)
        if value not in {
            summary.category.value for summary in self._category_summaries()
        }:
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
            search = self.query_one("#settings-category-search", Input)
        except QueryError:
            logger.debug("Unable to focus Settings category search")
            return
        search.focus()
        # task-1584: refocusing must not resume the stale query -- select it
        # so the next keystroke starts fresh (repeat searches concatenated
        # before, silently poisoning the next filter).
        search.select_all()

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
        valid_categories = {
            summary.category.value for summary in self._category_summaries()
        }
        if category_value not in valid_categories:
            logger.debug(
                "Ignoring unknown Settings navigation category: %s", category_value
            )
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
        self.call_after_refresh(
            self._apply_navigation_provider_context, provider, model, field
        )

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
        self._pending_navigation_focus_selector = selector
        try:
            self.query_one(selector).focus()
        except QueryError:
            return

    def _select_category(
        self, category_value: str, *, restore_focus: bool = False
    ) -> None:
        if category_value != SettingsCategoryId.PROVIDERS_MODELS.value:
            self._active_settings_field_id = None
        # Task 3 (541 v2 UX AC3): the remembered "last-expanded RAG group"
        # scope must not leak into a later LIBRARY_RAG visit -- e.g. leaving
        # with "Chunking" expanded and coming back to a freshly recomposed
        # (all-collapsed-but-Search) detail pane should start at the same
        # fallback guidance a first-ever visit shows.
        if category_value != SettingsCategoryId.LIBRARY_RAG.value:
            self._active_rag_scope_group = None
            # Task 4 (541 v2 UX AC1): a profile-picker PREVIEW is a purely
            # visual browse of the mounted editor widgets -- leaving the
            # category recomposes the detail pane from scratch (the Select
            # rebuilds pinned to the ACTIVE profile, see
            # `_render_library_rag_profile_block`), so the in-memory
            # "previewing X" flag must not survive to a later visit and
            # silently mismatch what's actually on screen.
            self._rag_preview_profile_id = None
            # 541-v2 final review item 1: same reasoning -- a still-pending
            # suppression expectation was queued against THIS (about to be
            # destroyed) Select instance's next Changed message(s); the
            # recomposed detail pane mints a brand-new Select that owes
            # nothing to it, so a leftover entry here would incorrectly
            # swallow that new instance's own first genuine Changed.
            self._rag_select_suppress_queue.clear()
            # task-566: the exclusive `settings-rag-index-status` worker
            # group (index-status fetch on category show / 't' test /
            # Save-path reindex confirm) must not be left running once the
            # user has navigated away -- its callback would otherwise land
            # later and pop a re-index confirm modal, or write a status
            # line, over a now-unrelated category. Cancellation is
            # best-effort (a thread already running still completes and
            # still calls back -- see the guards in
            # `_apply_library_rag_index_status` and
            # `_decide_reindex_confirmation`, which are what actually
            # matter), but it does stop a not-yet-started fetch from ever
            # landing at all. `is_mounted` guards `self.workers` (routes
            # through `self.app`, unavailable on a not-yet-mounted screen)
            # -- same guard shape as `_refresh_library_rag_index_status`.
            if getattr(self, "is_mounted", False):
                self.workers.cancel_group(self, "settings-rag-index-status")
        if category_value != SettingsCategoryId.IMAGE_GENERATION.value:
            # Same reasoning as the RAG queue clear immediately above: a
            # still-pending suppression expectation belongs to the (about
            # to be destroyed) default-backend Select instance; the
            # recomposed detail pane mints a brand-new one that owes it
            # nothing.
            self._image_gen_select_suppress_queue.clear()
            # Task 6: bump the probe session and drop the in-flight guard
            # -- see `_image_gen_probe_session`'s docstring. A probe
            # already running keeps running (best-effort only, matching
            # the RAG index-status precedent above -- an in-flight thread
            # worker can't be interrupted mid-blocking-call), but its
            # eventual callback will find this session stale and no-op
            # instead of touching a since-recomposed, unrelated panel.
            self._image_gen_probe_session += 1
            self._image_gen_probe_in_flight = False
        # Task 2 review (Important): a stale re-index-confirm in-flight
        # guard must never survive navigating away from (or back into) the
        # category -- e.g. the user backs out mid-fetch. Unconditional
        # reset; a no-op for every category but LIBRARY_RAG.
        self._rag_reindex_confirm_in_flight = False
        if category_value != SettingsCategoryId.WORKSPACES.value:
            # Task 9: leaving the category recomposes the detail pane from
            # scratch -- a selection that survived would point the freshly
            # (re)composed card at a workspace id a later visit's list may
            # not even show (e.g. after an archive elsewhere).
            self._settings_selected_workspace_id = None
        self.active_category = category_value
        # task-1565: keep the selection visible -- the rail does not follow
        # the active category on its own, so deep categories (Schedules,
        # Image Gen) could be selected while entirely off-viewport.
        def _reveal_active_button() -> None:
            try:
                self.query_one(
                    f"#settings-category-{category_value}", Button
                ).scroll_visible(animate=False)
            except Exception:
                pass

        if getattr(self, "is_mounted", False):
            self.call_after_refresh(_reveal_active_button)
        # Task 6 (541 AC6): keep the footer's a/c/b hint in sync with a
        # live in-session category switch (on_mount's call alone only
        # covers the initial/restored-state paint -- see
        # _register_footer_shortcuts' docstring).
        self._register_footer_shortcuts()
        if category_value == SettingsCategoryId.OVERVIEW.value:
            self._queue_sync_rows_refresh()
        if category_value == SettingsCategoryId.LIBRARY_RAG.value:
            self._refresh_library_rag_index_status()
        if category_value == SettingsCategoryId.IMAGE_GENERATION.value:
            # Qodo PR #901 fix 3: entering the category invalidates the
            # cached raw-section baseline (see `_image_gen_raw_section_
            # cache`'s docstring) -- a PRIOR visit's cache must never
            # survive into a new one (e.g. an Advanced Config hand-edit
            # to [image_generation] made while away).
            self._image_gen_raw_section_cache = None
        if restore_focus:
            self.call_after_refresh(self._focus_category, category_value)

    @on(DescendantFocus)
    def handle_descendant_focus(self, event: DescendantFocus) -> None:
        # Clear the pending navigation focus intent only when SATISFIED (the
        # intended widget landed focus). Clearing on any focus is too eager:
        # the stale category-chip focus the intent is meant to supersede
        # lands first and would erase it (task-290).
        pending = self._pending_navigation_focus_selector
        landed_id = str(getattr(event.widget, "id", "") or "")
        if pending and landed_id and f"#{landed_id}" == pending:
            self._pending_navigation_focus_selector = None
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
            self._scroll_impact_pane_to_field_guide(active_category)
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
            self._scroll_impact_pane_to_field_guide(active_category)
            return
        if active_category is SettingsCategoryId.LIBRARY_RAG:
            # Task 3 (541 v2 UX AC3): membership uses the shared
            # _RAG_FIELD_GROUP_BY_ID table (also read by
            # _rag_field_guidance_rows and the Collapsible.Toggled handler
            # below) instead of a locally-duplicated id set.
            self._active_settings_field_id = (
                widget_id if widget_id in _RAG_FIELD_GROUP_BY_ID else None
            )
            self._refresh_rag_field_guidance()
            self._scroll_impact_pane_to_field_guide(active_category)
            return
        if active_category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            # task-5 + TASK-870: only "Max parallel agent runs" and "Tool
            # result display cap" have dedicated focused-field guidance
            # today; other Console Behavior fields keep the always-visible
            # "Control guide" static block.
            console_behavior_field_ids = {
                "settings-console-max-parallel-runs",
                "settings-console-tool-result-display-chars",
            }
            self._active_settings_field_id = (
                widget_id if widget_id in console_behavior_field_ids else None
            )
            self._refresh_console_behavior_field_guidance()
            self._scroll_impact_pane_to_field_guide(active_category)
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
        self._scroll_impact_pane_to_field_guide(active_category)

    def _scroll_impact_pane_to_field_guide(self, category: SettingsCategoryId) -> None:
        """Scroll the Scope Inspector so the Focused field guide is visible.

        Fleet-UX expert review F6 (task-1234): focusing a guided field
        already refreshes the guide row TEXT in place (the ``_refresh_*_
        field_guidance`` methods above, no recompose) but never moved the
        pane's own scroll position, so the guide block could sit below
        ``#settings-impact-pane``'s fold with only a thin scrollbar sliver
        hinting at it -- reported live as "focusing Max parallel shows only
        Purpose:". Same disease task-1140 fixed for the fleet line, in a
        second location.

        Qodo PR #1074 finding 2: scrolling to only the FIRST row was
        insufficient on its own -- ``scroll_to_widget`` no-ops once its
        target is already fully inside the viewport, so a prior scroll
        position that happens to leave the first row sitting flush with
        the pane's own bottom edge (fully visible, technically) short-
        circuits the whole call, leaving every row after it (Consequences/
        Saved as/Applies, etc.) below the fold. Two passes fix this:
        first reveal the LAST row (pulls the whole guide into view when it
        fits the viewport), then re-target the FIRST row with ``top=True``
        to force-pin it to the pane's top edge regardless of whether
        anything actually needed to move. When the guide is short enough
        to fit the viewport this second pass is a no-op in effect (the
        first pass already made the whole block visible); when the guide
        is TALLER than the viewport, it deliberately re-prioritizes the
        first rows -- Purpose/Focused setting, the most load-bearing
        content -- over the tail, maximizing visible coverage starting
        from the top rather than the bottom.

        ``call_after_refresh`` + ``force=True`` mirrors the pattern proven
        in ``library_screen.LibraryScreen._preserve_library_rail_scroll``:
        an unforced ``scroll_to_widget`` clamps to 0 when a container's
        scroll bounds haven't been (re)computed yet -- relevant here
        because a CATEGORY switch recomposes ``#settings-impact-pane``
        from scratch (``_render_impact_pane``), and focus can land on the
        new category's first field before that layout has settled.

        Args:
            category: The Settings category whose "Focused field guide"
                block (if any -- see ``_FIELD_GUIDE_FIRST_ROW_ID``) should
                be scrolled into view.
        """
        row_id = _FIELD_GUIDE_FIRST_ROW_ID.get(category)
        if row_id is None:
            return
        row_prefix = row_id.rsplit("-", 1)[0]

        def _scroll() -> None:
            try:
                pane = self.query_one("#settings-impact-pane-body")
                first_row = self.query_one(f"#{row_id}")
            except Exception:
                return
            # Guide rows are a fixed-length, contiguous block of Static
            # widgets rendered unconditionally (see the per-category
            # ``*_field_guidance_rows`` methods) -- find the last one by
            # probing sequential ids rather than hardcoding a row count a
            # future guidance edit would silently drift out of sync with.
            last_row = first_row
            index = 1
            while True:
                try:
                    last_row = self.query_one(f"#{row_prefix}-{index}")
                except Exception:
                    break
                index += 1
            # Pass 1: reveal the guide's tail. If the whole guide fits the
            # viewport this already brings every row into view.
            pane.scroll_to_widget(last_row, animate=False, force=True)
            # Pass 2: force the first row to the pane's top edge -- see the
            # docstring above for why this must run unconditionally (not
            # only when pass 1 left it hidden) and why ``top=True`` beats a
            # plain minimal-scroll re-target.
            pane.scroll_to_widget(first_row, animate=False, force=True, top=True)

        self.call_after_refresh(_scroll)

    @on(Collapsible.Toggled)
    def handle_settings_library_rag_collapsible_toggled(
        self, event: Collapsible.Toggled
    ) -> None:
        """Task 3 (541 v2 UX AC3): expanding a Library/RAG group already
        switches the Scope Inspector's context, even before any field
        inside it is focused. Collapsing the currently-active group falls
        back to whatever `_active_settings_field_id` would otherwise
        resolve to (typically the static fallback, since focus can't land
        on a hidden collapsed field).

        Args:
            event: The Collapsible toggle whose widget id names the group
                this handler maps to a Scope Inspector context.
        """
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        collapsible_id = str(getattr(event.collapsible, "id", "") or "")
        group = _RAG_GROUP_BY_COLLAPSIBLE_ID.get(collapsible_id)
        if group is None:
            return
        if event.collapsible.collapsed:
            if self._active_rag_scope_group == group:
                self._active_rag_scope_group = None
        else:
            self._active_rag_scope_group = group
        self._refresh_rag_field_guidance()

    @on(Button.Pressed, "#settings-open-appearance")
    def open_appearance_settings(self) -> None:
        self.post_message(
            NavigateToScreen("settings", {"category": SettingsCategoryId.THEME})
        )

    @on(SettingsThemeEditor.ThemeModifiedStatus)
    def handle_theme_modified_status(
        self, event: SettingsThemeEditor.ThemeModifiedStatus
    ) -> None:
        if self.theme_editor_modified == event.is_modified:
            return
        self.theme_editor_modified = event.is_modified
        self._refresh_theme_modified_widgets()

    def _refresh_theme_modified_widgets(self) -> None:
        """In-place refresh of the Theme dirty displays (rail marker, inspector row).

        Targeted updates, never a recompose -- a recompose would remount the
        theme editor and wipe the very in-progress edit that raised this
        notification (see the theme_editor_modified reactive's comment).
        """
        self._refresh_category_button_label(SettingsCategoryId.THEME)
        try:
            row = self.query_one("#settings-theme-unsaved-note", Static)
        except QueryError:
            pass
        else:
            modified = "Yes" if self.theme_editor_modified else "No"
            row.update(f"Unsaved theme changes: {modified}")

    @on(InternalPromptsPanel.Modified)
    def _on_internal_prompts_modified(self, event: InternalPromptsPanel.Modified) -> None:
        # Deliberately NOT a recompose=True reactive assignment (P3
        # whole-branch review Fix 1): the panel already computed this count
        # for us, so we just cache it and push a TARGETED refresh into
        # whichever widgets currently show it. A recompose here would
        # unmount/remount the panel on every save/reset, wiping its search
        # text and scroll position.
        self._internal_prompts_customized_count = event.customized_count
        self._refresh_internal_prompts_customized_widgets()

    def _refresh_internal_prompts_customized_widgets(self) -> None:
        """In-place refresh of every Internal Prompts customized-count display.

        Safe to call whether or not the sidebar status row / impact-pane row
        are currently mounted (different active category, or impact pane not
        yet composed) -- each query is independently guarded.
        """
        self._update_draft_status_widgets(SettingsCategoryId.INTERNAL_PROMPTS)
        try:
            row = self.query_one("#settings-internal-prompts-customized-count", Static)
        except QueryError:
            pass
        else:
            row.update(f"Customized prompts: {self._internal_prompts_customized_count}")

    @on(Select.Changed, "#settings-appearance-theme")
    def handle_appearance_theme_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._syncing_appearance_defaults:
            return
        self._stage_appearance_value(
            "default_theme", str(event.value or "textual-dark")
        )
        self._mark_appearance_settings_staged()

    @on(Input.Changed, "#settings-appearance-palette-theme-limit")
    def handle_appearance_palette_theme_limit_changed(
        self, event: Input.Changed
    ) -> None:
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
    def handle_appearance_animations_enabled_changed(
        self, event: Button.Pressed
    ) -> None:
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
        from tldw_chatbook.Widgets.Settings_Widgets.server_switch_modal import (
            ServerSwitchModal,
        )

        state = self._runtime_source_state()
        raw_config = (getattr(self.app_instance, "app_config", {}) or {}).get(
            "COMPREHENSIVE_CONFIG_RAW", {}
        )
        api_config = (
            raw_config.get("tldw_api", {}) if isinstance(raw_config, dict) else {}
        )
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
            switched = await app.handle_runtime_backend_changed("local")
            if not switched:
                return
            self.app.notify("Runtime source set to local.", severity="information")
            self._refresh_manual_sync_rows()
            return

        base_url = str(result.get("base_url") or "").strip()
        if not base_url:
            return

        from tldw_chatbook.Utils.input_validation import validate_url

        if not validate_url(base_url):
            self.app.notify(
                "Rejected server URL; nothing was changed.", severity="error"
            )
            return
        auth_token = str(result.get("auth_token") or "").strip()
        saved = save_settings_to_cli_config(
            {
                "tldw_api": {
                    "base_url": base_url,
                    "auth_token": auth_token,
                }
            }
        )
        if not saved:
            self.app.notify(
                "Server settings could not be saved; "
                "the previous source remains active.",
                severity="error",
            )
            return

        try:
            refreshed_config = load_settings(force_reload=True)
        except Exception as exc:
            logger.warning(
                "Saved server settings could not be loaded "
                "(exception_category=%s).",
                type(exc).__name__,
            )
            self.app.notify(
                "Server settings were saved but could not be activated; "
                "the previous source remains active.",
                severity="error",
            )
            return

        switched = await app.handle_runtime_backend_changed(
            "server",
            app_config_override=refreshed_config,
        )
        if not switched:
            return

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
                    "Sync v2 profile preparation failed "
                    "(mode=local_first_sync, exception_category=%s).",
                    type(exc).__name__,
                )
                self.app.notify(
                    "Server activated, but Sync v2 setup could not be completed.",
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
            (
                "Manual sync result",
                "Manual Sync is running after explicit user request.",
            ),
            ("Pending outgoing", "Refreshing"),
        )
        self._manual_sync_run_worker()

    @on(Button.Pressed, ".settings-category-button")
    def handle_category_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        category_value = self._category_value_from_button(event.button)
        if category_value is not None:
            self._select_category(category_value, restore_focus=event.button.has_focus)

    @on(Button.Pressed, ".settings-workspace-row")
    def handle_workspace_row_pressed(self, event: Button.Pressed) -> None:
        """Select a workspace row, then refresh so its card renders (task 9)."""
        event.stop()
        button_id = str(getattr(event.button, "id", "") or "")
        prefix = "settings-workspace-row-"
        if not button_id.startswith(prefix):
            return
        workspace_id = button_id.removeprefix(prefix)
        if not workspace_id:
            return
        self._settings_selected_workspace_id = workspace_id
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Checkbox.Changed, "#settings-workspaces-show-archived")
    def handle_workspaces_show_archived_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._settings_show_archived_workspaces = event.value
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, "#settings-workspace-create")
    def handle_workspace_create(self, event: Button.Pressed) -> None:
        """Create a workspace from the typed name, or a generated one when
        left blank -- the id always comes from `next_local_workspace_identity`
        (task 9)."""
        event.stop()
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        try:
            name_input = self.query_one("#settings-workspace-create-name", Input)
        except QueryError:
            return
        typed_name = name_input.value.strip()
        workspace_id, generated_name = next_local_workspace_identity(registry)
        try:
            registry.create_workspace(
                workspace_id=workspace_id, name=typed_name or generated_name
            )
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, "#settings-workspace-rename-apply")
    def handle_workspace_rename_apply(self, event: Button.Pressed) -> None:
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        try:
            rename_input = self.query_one("#settings-workspace-rename-input", Input)
        except QueryError:
            return
        try:
            registry.rename_workspace(workspace_id, rename_input.value)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, "#settings-workspace-set-active")
    def handle_workspace_set_active(self, event: Button.Pressed) -> None:
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        try:
            registry.set_active_workspace(workspace_id)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, "#settings-workspace-archive")
    def handle_workspace_archive(self, event: Button.Pressed) -> None:
        """Confirm, then archive (task 9).

        Mirrors `ChatScreen._confirm_console_workspace_archive` (Console's
        own archive flow): the SAME verbatim copy, and the SAME shape --
        an async closure passed as `confirm_callback`, since
        `ConfirmationDialog.on_button_pressed` `await`s that callback and a
        plain sync function would raise there instead of archiving.
        """
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        record = registry.get_workspace(workspace_id)
        if record is None:
            return

        async def _archive() -> None:
            try:
                registry.archive_workspace(workspace_id)
            except WorkspaceRegistryServiceError as exc:
                self._set_settings_workspaces_result(str(exc))
                return
            # The row disappears from the default (not-showing-archived)
            # list -- a selection surviving would point the card at a
            # workspace no longer in view.
            self._settings_selected_workspace_id = None
            self._settings_workspaces_result = ""
            self._refresh_settings_workspaces_pane()

        self.app.push_screen(
            ConfirmationDialog(
                title="Archive workspace?",
                message=(
                    f"Archive {record.name}? Its conversations stay saved and "
                    "remain visible in Library; the workspace disappears from "
                    "the switcher and the Console browser."
                ),
                confirm_label="Archive",
                confirm_callback=_archive,
            )
        )

    @on(Button.Pressed, "#settings-workspace-unarchive")
    def handle_workspace_unarchive(self, event: Button.Pressed) -> None:
        """Restore a workspace to the default listing without activating it
        (spec: never auto-activate on unarchive, task 9)."""
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        try:
            registry.unarchive_workspace(workspace_id)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, "#settings-workspace-folder-add")
    def _settings_workspace_add_folder(self, event: Button.Pressed) -> None:
        """Bind a folder as a read-only file-tool access root (task 10)."""
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        raw = self.query_one("#settings-workspace-folder-path", Input).value
        try:
            registry.add_folder_binding(workspace_id, raw)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._set_settings_workspaces_result("Folder added (read-only).")
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, ".settings-workspace-folder-toggle")
    def _settings_workspace_toggle_folder_access(self, event: Button.Pressed) -> None:
        """Flip a folder binding between read-only and read-write (task 10).

        The binding id is read from `event.button.binding_id`, stashed at
        compose time -- never parsed out of the button's dom id, which
        would split a uuid on its own hyphens.
        """
        event.stop()
        workspace_id = self._settings_selected_workspace_id
        if not workspace_id:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        binding_id = str(getattr(event.button, "binding_id", "") or "")
        if not binding_id:
            return
        current = next(
            (
                binding
                for binding in registry.list_folder_bindings(workspace_id)
                if binding.binding_id == binding_id
            ),
            None,
        )
        if current is None:
            return
        allow_write = current.metadata.get("access") != "rw"
        try:
            registry.set_folder_binding_access(binding_id, allow_write=allow_write)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

    @on(Button.Pressed, ".settings-workspace-folder-remove")
    def _settings_workspace_remove_folder(self, event: Button.Pressed) -> None:
        """Unbind a folder from the selected workspace (task 10)."""
        event.stop()
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        binding_id = str(getattr(event.button, "binding_id", "") or "")
        if not binding_id:
            return
        try:
            registry.remove_runtime_binding(binding_id)
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._settings_workspaces_result = ""
        self._refresh_settings_workspaces_pane()

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
    def handle_console_collapse_large_pastes_changed(
        self, event: Button.Pressed
    ) -> None:
        event.stop()
        next_value = not self._collapse_large_pastes_enabled()
        self._stage_console_large_paste_value(next_value)
        event.button.label = self._collapse_large_pastes_button_label()
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Button.Pressed, "#settings-console-remote-images-toggle")
    def handle_console_remote_images_toggle(self, event: Button.Pressed) -> None:
        """Flip the remote-images toggle: immediate write, no category draft."""
        event.stop()
        enabled = self._toggle_remote_images()
        event.button.label = self._remote_images_button_label()
        self.app.notify(
            "Linked images in replies will now render."
            if enabled
            else "Linked images in replies will stay ignored.",
            severity="information",
        )

    @on(Input.Changed, "#settings-console-paste-collapse-threshold")
    def handle_console_paste_threshold_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_threshold:
            return
        self._stage_console_paste_threshold_value(event.value)
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text(
            "#settings-console-behavior-result", self._console_behavior_result_text()
        )
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-console-max-parallel-runs")
    def handle_console_max_parallel_runs_changed(self, event: Input.Changed) -> None:
        if self._syncing_console_max_parallel_runs:
            return
        self._stage_console_max_parallel_runs_value(event.value)
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text(
            "#settings-console-behavior-result", self._console_behavior_result_text()
        )
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-console-tool-result-display-chars")
    def handle_console_tool_result_display_chars_changed(
        self, event: Input.Changed
    ) -> None:
        if self._syncing_console_tool_result_display_chars:
            return
        self._stage_tool_result_display_chars_value(event.value)
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text(
            "#settings-console-behavior-result", self._console_behavior_result_text()
        )
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
        self._stage_console_default_input(
            "min_p", event.value, self._normalise_model_profile_min_p
        )

    @on(Input.Changed, "#settings-console-default-top-k")
    def handle_console_default_top_k_changed(self, event: Input.Changed) -> None:
        self._stage_console_default_input(
            "top_k", event.value, self._normalise_model_profile_top_k
        )

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
        self._stage_console_default_input(
            "seed", event.value, self._normalise_model_profile_seed
        )

    @on(Input.Changed, "#settings-console-default-presence-penalty")
    def handle_console_default_presence_penalty_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_console_default_input(
            "presence_penalty",
            event.value,
            self._normalise_model_profile_presence_penalty,
        )

    @on(Input.Changed, "#settings-console-default-frequency-penalty")
    def handle_console_default_frequency_penalty_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_console_default_input(
            "frequency_penalty",
            event.value,
            self._normalise_model_profile_frequency_penalty,
        )

    @on(Input.Changed, "#settings-console-default-reasoning-effort")
    def handle_console_default_reasoning_effort_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_console_default_input(
            "reasoning_effort",
            event.value,
            self._normalise_model_profile_reasoning_effort,
        )

    @on(Input.Changed, "#settings-console-default-reasoning-summary")
    def handle_console_default_reasoning_summary_changed(
        self, event: Input.Changed
    ) -> None:
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
    def handle_console_default_thinking_effort_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_console_default_input(
            "thinking_effort",
            event.value,
            self._normalise_model_profile_thinking_effort,
        )

    @on(Input.Changed, "#settings-console-default-thinking-budget-tokens")
    def handle_console_default_thinking_budget_tokens_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_console_default_input(
            "thinking_budget_tokens",
            event.value,
            self._normalise_model_profile_thinking_budget_tokens,
        )

    def _stage_console_default_input(
        self, key: str, raw_value: object, normalizer
    ) -> None:
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
        self._set_static_text(
            "#settings-console-behavior-result", self._console_behavior_result_text()
        )
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Button.Pressed, "#settings-console-background-effect-enabled")
    def handle_console_background_effect_enabled_changed(
        self, event: Button.Pressed
    ) -> None:
        event.stop()
        next_value = not bool(self._console_background_effect_value("enabled"))
        self._stage_console_background_effect_value("enabled", next_value)
        event.button.label = self._console_background_effect_enabled_label()
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-type")
    def handle_console_background_effect_type_changed(
        self, event: Select.Changed
    ) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        self._stage_console_background_effect_value(
            "effect", str(event.value or "none")
        )
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-scope")
    def handle_console_background_effect_scope_changed(
        self, event: Select.Changed
    ) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        next_scope = self._available_console_background_scope(event.value)
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft = self._settings_drafts.get(category)
        if (
            next_scope == "transcript"
            and self._loaded_console_background_scope_is_unavailable()
            and (draft is None or "background_effects.scope" not in draft.values)
        ):
            self._console_behavior_result = (
                CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            )
            self._set_static_text(
                "#settings-console-behavior-result", self._console_behavior_result
            )
            self._update_draft_status_widgets(category)
            return
        if (
            next_scope == "transcript"
            and draft is not None
            and draft.originals.get("background_effects.scope") == "workbench"
            and draft.values.get("background_effects.scope") == "transcript"
        ):
            self._console_behavior_result = (
                CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            )
            self._set_static_text(
                "#settings-console-behavior-result", self._console_behavior_result
            )
            self._update_draft_status_widgets(category)
            return
        if str(event.value) == "workbench":
            draft = self._settings_drafts.setdefault(
                category, SettingsDraft(category=category)
            )
            draft.set_value("background_effects.scope", "workbench", next_scope)
            self._syncing_console_background_effects = True
            try:
                event.select.value = next_scope
            finally:
                self._syncing_console_background_effects = False
            self._console_behavior_result = (
                CONSOLE_BACKGROUND_WORKBENCH_UNAVAILABLE_COPY
            )
            self._set_static_text(
                "#settings-console-behavior-result", self._console_behavior_result
            )
            self._update_draft_status_widgets(category)
            return
        self._stage_console_background_effect_value("scope", next_scope)
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-console-background-effect-intensity")
    def handle_console_background_effect_intensity_changed(
        self, event: Select.Changed
    ) -> None:
        event.stop()
        if self._syncing_console_background_effects:
            return
        self._stage_console_background_effect_value(
            "intensity", str(event.value or "low")
        )
        self._mark_console_behavior_settings_staged()

    @on(Input.Changed, "#settings-console-background-effect-fps")
    def handle_console_background_effect_fps_changed(
        self, event: Input.Changed
    ) -> None:
        if self._syncing_console_background_effects:
            return
        value: object = int(event.value) if str(event.value).isdigit() else event.value
        self._stage_console_background_effect_value("fps", value)
        self._mark_console_behavior_settings_staged()

    @on(Select.Changed, "#settings-library-rag-search-mode")
    def handle_library_rag_search_mode_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "default_search_mode", str(event.value or "semantic")
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-default-top-k")
    def handle_library_rag_default_top_k_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "default_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-fts-top-k")
    def handle_library_rag_fts_top_k_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "fts_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-vector-top-k")
    def handle_library_rag_vector_top_k_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "vector_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-hybrid-alpha")
    def handle_library_rag_hybrid_alpha_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "hybrid_alpha",
            self._normalise_library_rag_float(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-score-threshold")
    def handle_library_rag_score_threshold_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "score_threshold",
            self._normalise_library_rag_float(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Checkbox.Changed, "#settings-library-rag-include-citations")
    def handle_library_rag_include_citations_changed(
        self, event: Checkbox.Changed
    ) -> None:
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("include_citations", bool(event.value))
        self._mark_library_rag_settings_staged()

    @on(Select.Changed, "#settings-library-rag-citation-style")
    def handle_library_rag_citation_style_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("citation_style", str(event.value or "inline"))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-snippet-max-chars")
    def handle_library_rag_snippet_max_chars_changed(
        self, event: Input.Changed
    ) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "snippet_max_chars",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-max-context-size")
    def handle_library_rag_max_context_size_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "max_context_size",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-embedding-model")
    def handle_library_rag_embedding_model_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("embedding_model", str(event.value))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-embedding-device")
    def handle_library_rag_embedding_device_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("embedding_device", str(event.value))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-embedding-batch-size")
    def handle_library_rag_embedding_batch_size_changed(
        self, event: Input.Changed
    ) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "embedding_batch_size",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-embedding-max-length")
    def handle_library_rag_embedding_max_length_changed(
        self, event: Input.Changed
    ) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "embedding_max_length",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-chunk-size")
    def handle_library_rag_chunk_size_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "chunk_size",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-chunk-overlap")
    def handle_library_rag_chunk_overlap_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "chunk_overlap",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Select.Changed, "#settings-library-rag-chunking-method")
    def handle_library_rag_chunking_method_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("chunking_method", str(event.value or "words"))
        self._mark_library_rag_settings_staged()

    @on(Select.Changed, "#settings-library-rag-distance-metric")
    def handle_library_rag_distance_metric_changed(self, event: Select.Changed) -> None:
        """Stage a distance-metric change on the active profile's draft.

        Args:
            event: The Select change; ``event.value`` is the metric name
                (falls back to ``"cosine"`` when blank).
        """
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("distance_metric", str(event.value or "cosine"))
        self._mark_library_rag_settings_staged()

    @on(Checkbox.Changed, "#settings-library-rag-enable-reranking")
    def handle_library_rag_enable_reranking_changed(
        self, event: Checkbox.Changed
    ) -> None:
        """Stage the reranking toggle and live-update the rerank fields.

        Task 1 (541 v2 UX AC4): flipping the checkbox immediately dims or
        re-enables the reranker model / rerank results Inputs.

        Args:
            event: The Checkbox change; ``event.value`` is the new
                enable-reranking state to stage.
        """
        event.stop()
        if self._library_rag_edits_suppressed():
            return
        next_value = bool(event.value)
        self._stage_library_rag_value("enable_reranking", next_value)
        self._apply_library_rag_rerank_field_state(
            rerank_enabled=next_value,
            field_disabled=bool(active_profile_info()["read_only"]),
        )
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-reranker-model")
    def handle_library_rag_reranker_model_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value("reranker_model", str(event.value))
        self._mark_library_rag_settings_staged()

    @on(Input.Changed, "#settings-library-rag-reranker-top-k")
    def handle_library_rag_reranker_top_k_changed(self, event: Input.Changed) -> None:
        if self._library_rag_edits_suppressed():
            return
        self._stage_library_rag_value(
            "reranker_top_k",
            self._normalise_library_rag_int(event.value),
        )
        self._mark_library_rag_settings_staged()

    @on(Select.Changed, "#settings-library-rag-profile-select")
    def handle_library_rag_profile_select_changed(self, event: Select.Changed) -> None:
        """Task 4 (541 v2 UX AC1): browsing the profile picker PREVIEWS that
        profile's values read-only -- it never stages a draft (only "Set
        active" does that, via the existing dirty-prompt flow below).
        Selecting the ACTIVE profile's own id exits preview and restores
        the ordinary, draft-aware editor.

        541-v2 final review item 1: a message whose value matches the head
        of `_rag_select_suppress_queue` is `_sync_library_rag_profile_widgets`'s
        OWN imperative resync arriving (Textual delivers `Select.Changed`
        asynchronously, so this can land well after that resync call has
        returned) -- consumed and ignored rather than treated as a user
        browsing the dropdown. Only the ONE matching queued expectation is
        popped per message: a later genuine selection that happens to reuse
        the same value is never mistaken for a leftover resync.
        """
        event.stop()
        queue = self._rag_select_suppress_queue
        if queue and event.value == queue[0]:
            queue.pop(0)
            return
        selected = event.value
        active_id = active_profile_info()["id"]
        # PR #863 review: `Select.NULL` is the real blank sentinel on this
        # Textual version (`Select.BLANK` doesn't exist -- it silently
        # resolves to the unrelated `Widget.BLANK`). The picker is
        # `allow_blank=True`, so a user CAN pick the blank row: treat it as
        # "exit preview", never as a profile id to preview.
        if selected is None or selected is Select.NULL or str(selected) == active_id:
            self._rag_preview_profile_id = None
        else:
            self._rag_preview_profile_id = str(selected)
        self._sync_rag_editor_display()

    def _sync_rag_editor_display(self) -> None:
        """The ONE place that decides whether the Library/RAG editor
        renders the active profile (draft-aware, editable) or a PREVIEW of
        a different, browsed profile (read-only, draft-untouched) -- Task 4
        (541 v2 UX AC1). Called after every profile-Select change and
        after any completed action that could change which profile is
        active or must clear a stale preview
        (`_rag_after_set_active`, `_rag_after_profile_action`).
        """
        if self._rag_preview_profile_id is not None:
            self._render_rag_profile_preview(self._rag_preview_profile_id)
            return
        self._sync_library_rag_widgets(
            self._library_rag_setting_values(),
            field_disabled=bool(active_profile_info()["read_only"]),
        )
        self._update_library_rag_editor_title()
        self._set_library_rag_preview_banner(None)
        # 541-v2 final review item 2: restore the decoupling caption --
        # covers the "browsed back to the active profile" exit-preview path,
        # which reaches here without going through
        # `_sync_library_rag_profile_widgets` (that method's own exit-preview
        # paths -- set-active/clone/rename/delete -- restore it themselves).
        self._set_library_rag_editing_caption_visible(True)

    def _render_rag_profile_preview(self, profile_id: str) -> None:
        """Render `profile_id`'s OWN values into the editor, all disabled,
        with a banner naming it -- never touches `_settings_drafts` (the
        hard invariant: drafts belong to the active profile only)."""
        defaults = get_profile_defaults(profile_id)
        if defaults is None:
            # The previewed profile vanished mid-browse (e.g. deleted by
            # another action) -- fall back to exiting preview rather than
            # rendering a preview of nothing.
            self._rag_preview_profile_id = None
            self._sync_rag_editor_display()
            return
        self._sync_library_rag_widgets(asdict(defaults), field_disabled=True)
        name = self._library_rag_profile_name(profile_id)
        self._update_library_rag_editor_title(preview_name=name)
        self._set_library_rag_preview_banner(name)
        # 541-v2 final review item 2: the decoupling caption always names
        # the ACTIVE profile ("Editing: X...") -- directly contradictory
        # sitting right above this "Previewing: Y" title. The banner above
        # already carries the equivalent messaging, so hide it rather than
        # reword it.
        self._set_library_rag_editing_caption_visible(False)

    def _update_library_rag_editor_title(self, *, preview_name: str | None = None) -> None:
        """Task 4 (541 v2 UX AC1): the editor container's border title --
        "Editing: <active name>" ordinarily, "Previewing: <selected name>"
        while browsing a different profile. Names are escaped: profile
        names can contain markup-significant characters (repo lesson)."""
        if preview_name is not None:
            title = f"Previewing: {escape_markup(preview_name)}"
        else:
            title = f"Editing: {escape_markup(active_profile_info()['name'])}"
        try:
            self.query_one("#settings-library-rag-editor-card").border_title = title
        except QueryError:
            pass

    def _set_library_rag_preview_banner(self, name: str | None) -> None:
        try:
            banner = self.query_one("#settings-library-rag-preview-banner", Static)
        except QueryError:
            return
        if name is None:
            banner.display = False
            return
        banner.update(
            f"Previewing '{escape_markup(name)}' (read-only) — press Set active to edit it"
        )
        banner.display = True

    @on(Button.Pressed, "#settings-library-rag-profile-set-active")
    def handle_library_rag_profile_set_active(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_library_rag_profile_set_active()

    # Task 6 (541 v2 UX AC6): factored out so the 'a' keyboard accelerator
    # shares this SAME trigger -- identical dirty-draft switch-confirm modal
    # + preview-clear behavior as clicking the button (mirrors how Task 5
    # factored _trigger_library_rag_profile_clone/_index_backfill).
    def _trigger_library_rag_profile_set_active(self) -> None:
        profile_id = self._library_rag_selected_profile_id()
        if profile_id is None:
            self.app.notify("Choose a profile first.", severity="warning")
            return
        info = active_profile_info()
        if profile_id == info["id"]:
            self.app.notify(
                f"'{info['name']}' is already active.", severity="information"
            )
            return
        if self._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG):
            self.app.push_screen(
                RagProfileSwitchConfirmModal(),
                lambda result: self._handle_rag_profile_switch_confirm(
                    result, profile_id
                ),
            )
            return
        self._dispatch_rag_set_active(profile_id)

    def _handle_rag_profile_switch_confirm(
        self, result: str | None, profile_id: str
    ) -> None:
        # Task 4 review (Critical fix): reaching THIS modal at all required
        # `_library_rag_selected_profile_id()` (the Select's current value)
        # to differ from the active profile -- which is exactly what
        # entering a profile-picker PREVIEW means (handle_library_rag_
        # profile_select_changed fires on every such browse). "discard" and
        # "save" both act on the ACTIVE profile's own draft (discard pops
        # it, save persists it) -- a still-armed preview must never survive
        # into either: `action_settings_save_category`'s LIBRARY_RAG guard
        # would otherwise silently no-op the save AND leak
        # `_rag_profile_pending_activate` (its capture-and-clear sits BELOW
        # that guard), and a bare `_sync_library_rag_widgets()` alone would
        # leave the fields showing the previewed profile's stale,
        # forced-disabled values under a stale "Previewing: ..." banner/
        # title during the async gap before the set-active worker (discard)
        # or save worker (save) completes. "cancel"/None deliberately does
        # NOT clear it here -- the user chose to keep browsing/editing
        # exactly as they left it.
        if result == "discard":
            self._rag_preview_profile_id = None
            self._sync_rag_editor_display()
            self._settings_drafts.pop(SettingsCategoryId.LIBRARY_RAG, None)
            self._sync_library_rag_widgets()
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            self._dispatch_rag_set_active(profile_id)
        elif result == "save":
            self._rag_preview_profile_id = None
            self._sync_rag_editor_display()
            self._rag_profile_pending_activate = profile_id
            self.action_settings_save_category(allow_text_entry_focus=True)
        # "cancel"/None (Escape): leave the draft, active profile, and any
        # in-progress preview untouched.

    def _dispatch_rag_set_active(self, profile_id: str) -> None:
        self._library_rag_profile_result = "Setting active profile..."
        self._set_static_text(
            "#settings-library-rag-profile-result", self._library_rag_profile_result
        )
        self._rag_set_active_worker(profile_id)

    @work(exclusive=True, thread=True, group="settings-rag-set-active")
    def _rag_set_active_worker(self, profile_id: str) -> None:
        ok, reason = activate_profile(profile_id)
        # Task 4 (SP3): fetch the newly-active profile's index status in the
        # SAME off-thread hop that flips the pointer -- both touch the
        # profile/config store off the UI thread already, so this reuses that
        # trip instead of a second worker round-trip right after. `None` on
        # failure (nothing to report) preserves the pre-task-4 2-arg call
        # shape for `_rag_after_set_active`'s "no warning known" branch.
        new_status = fetch_index_status() if ok else None
        self.app.call_from_thread(
            self._rag_after_set_active, ok, reason, new_status
        )

    def _rag_after_set_active(
        self, ok: bool, reason: str, new_index_status: Mapping[str, object] | None = None
    ) -> None:
        # Task 4 (541 v2 UX AC1): either outcome ends any in-progress
        # profile-picker PREVIEW -- on success the active profile itself
        # changed (a stale preview reference is meaningless); on failure
        # the Select snaps back to the real active id below, so a lingering
        # "previewing <failed target>" flag would mismatch what's on
        # screen. Explicit rather than relying on the Select.Changed
        # cascade: `_sync_library_rag_profile_widgets` deliberately
        # suppresses that cascade (see `_rag_select_suppress_queue`), and
        # even without suppression a same-value reassignment (Set active on
        # the profile already being previewed) posts no Changed message at
        # all.
        self._rag_preview_profile_id = None
        if ok:
            self._settings_drafts.pop(SettingsCategoryId.LIBRARY_RAG, None)
            info = active_profile_info()
            message = f"Active profile: {info['name']}"
            # Honest re-index warning (SP3 spec §3, trigger (b)): only when we
            # actually KNOW the new profile's fingerprinted collection is
            # genuinely absent/empty -- never noisy-by-default when the
            # caller didn't supply a status (e.g. pre-task-4 callers/tests)
            # or when it's already built (switching to a profile whose
            # collection was indexed before). Task 4 review (Finding 2):
            # a status-read failure ("unknown", see fetch_index_status's
            # exception fallback) used to fall into this same "not built"
            # branch and claim the switch "re-points to a new (empty)
            # index" -- a false claim, since an unreadable status says
            # nothing about whether the index actually changed. That case
            # now gets its own honest, distinct notice instead.
            state = new_index_status.get("state") if new_index_status is not None else None
            index_empty = state in ("absent", "empty")
            status_unknown = state == "unknown"
            if index_empty:
                message = f"{message} {RAG_INDEX_CHANGE_WARNING}"
            elif status_unknown:
                message = f"{message} Index status unavailable — check the Index row."
            self._library_rag_profile_result = message
            self._set_static_text("#settings-library-rag-profile-result", message)
            self._sync_library_rag_widgets()
            self._sync_library_rag_profile_widgets()
            self._update_library_rag_editor_title()
            self._set_library_rag_preview_banner(None)
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            if new_index_status is not None:
                self._apply_library_rag_index_status(new_index_status)
            self.app.notify(
                message,
                severity="warning" if (index_empty or status_unknown) else "information",
            )
            return
        message = f"Couldn't switch active profile: {reason}"
        self._library_rag_profile_result = message
        self._set_static_text("#settings-library-rag-profile-result", message)
        # TASK-2 review (Finding 4): the profile Select was already showing
        # the user's (failed) target selection -- snap it back to the real
        # active profile rather than leaving a stale value on screen.
        self._sync_library_rag_profile_widgets()
        # Task 4: nothing actually changed about which profile is active,
        # but the editor was possibly showing a forced-disabled PREVIEW of
        # the failed target -- restore the real (still-active) values, not
        # just the disabled-state fix `_sync_library_rag_profile_widgets`
        # already applies above.
        self._sync_library_rag_widgets()
        self._update_library_rag_editor_title()
        self._set_library_rag_preview_banner(None)
        self.app.notify(message, severity="error")

    @on(Button.Pressed, "#settings-library-rag-index-backfill")
    def handle_library_rag_index_backfill(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_library_rag_index_backfill()

    # Task 5 (541 v2 UX AC5): the first-run starter panel's "Backfill now"
    # shares this SAME trigger -- never a bespoke reimplementation of the
    # in-flight guard/notify/dispatch.
    @on(Button.Pressed, "#settings-library-rag-starter-backfill")
    def handle_library_rag_starter_backfill(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_library_rag_index_backfill()

    def _trigger_library_rag_index_backfill(self) -> None:
        if self._library_rag_backfill_in_flight:
            self.app.notify("Backfill is already running.", severity="warning")
            return
        self._library_rag_backfill_in_flight = True
        self.app.notify(
            "Backfill started — this may take a while for large libraries.",
            severity="information",
        )
        self._rag_backfill_worker()

    @on(Button.Pressed, "#settings-library-rag-profile-clone")
    def handle_library_rag_profile_clone(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_library_rag_profile_clone()

    # Task 5 (541 v2 UX AC5): the first-run starter panel's "Clone to
    # tune…" shares this SAME trigger -- opens the identical name-modal,
    # seeded from whatever the profile Select currently shows (the active
    # builtin, at first-run compose time).
    @on(Button.Pressed, "#settings-library-rag-starter-clone")
    def handle_library_rag_starter_clone(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_library_rag_profile_clone()

    def _trigger_library_rag_profile_clone(self) -> None:
        source_id = (
            self._library_rag_selected_profile_id() or active_profile_info()["id"]
        )
        self.app.push_screen(
            RagProfileNameModal(title="Clone profile", confirm_label="Clone"),
            lambda name: self._handle_rag_profile_clone_result(name, source_id),
        )

    def _handle_rag_profile_clone_result(
        self, name: str | None, source_id: str
    ) -> None:
        if not name:
            return
        self._dispatch_rag_profile_action("clone", source_id, name)

    @on(Button.Pressed, "#settings-library-rag-profile-rename")
    def handle_library_rag_profile_rename(self, event: Button.Pressed) -> None:
        event.stop()
        profile_id = self._library_rag_selected_profile_id()
        if profile_id is None:
            self.app.notify("Choose a profile first.", severity="warning")
            return
        current_name = self._library_rag_profile_name(profile_id)
        self.app.push_screen(
            RagProfileNameModal(
                title="Rename profile", initial=current_name, confirm_label="Rename"
            ),
            lambda name: self._handle_rag_profile_rename_result(name, profile_id),
        )

    def _handle_rag_profile_rename_result(
        self, name: str | None, profile_id: str
    ) -> None:
        if not name:
            return
        self._dispatch_rag_profile_action("rename", profile_id, name)

    @on(Button.Pressed, "#settings-library-rag-profile-delete")
    def handle_library_rag_profile_delete(self, event: Button.Pressed) -> None:
        event.stop()
        profile_id = self._library_rag_selected_profile_id()
        if profile_id is None:
            self.app.notify("Choose a profile first.", severity="warning")
            return
        name = self._library_rag_profile_name(profile_id)
        modal = ConfirmationDialog(
            title="Delete profile",
            message=f'Delete the "{name}" RAG profile? This cannot be undone.',
            confirm_label="Delete",
            cancel_label="Cancel",
        )
        self.app.push_screen(
            modal,
            lambda confirmed: self._handle_rag_profile_delete_result(
                confirmed, profile_id
            ),
        )

    def _handle_rag_profile_delete_result(
        self, confirmed: bool | None, profile_id: str
    ) -> None:
        if not confirmed:
            return
        self._dispatch_rag_profile_action("delete", profile_id, "")

    def _dispatch_rag_profile_action(
        self, action: str, profile_id: str, arg: str
    ) -> None:
        self._library_rag_profile_result = f"{action.capitalize()} profile..."
        self._set_static_text(
            "#settings-library-rag-profile-result", self._library_rag_profile_result
        )
        self._rag_profile_action_worker(action, profile_id, arg)

    @work(exclusive=True, thread=True, group="settings-rag-profile-crud")
    def _rag_profile_action_worker(
        self, action: str, profile_id: str, arg: str
    ) -> None:
        if action == "clone":
            ok, result = clone_profile_as(profile_id, arg)
        elif action == "rename":
            ok, result = rename_user_profile(profile_id, arg)
        elif action == "delete":
            ok, result = delete_user_profile(profile_id)
        else:
            ok, result = False, "unknown-action"
        self.app.call_from_thread(self._rag_after_profile_action, action, ok, result)

    def _rag_after_profile_action(self, action: str, ok: bool, result: str) -> None:
        if ok:
            # Task 4 (541 v2 UX AC1): a successful clone/rename/delete can
            # change the active profile's own name (rename) or identity
            # (delete-the-active-profile's hybrid_basic fallback), or move
            # the Select onto a brand-new id (clone) -- any lingering
            # PREVIEW reference is stale either way. A FAILED action
            # changes nothing, so an unrelated in-progress preview is left
            # alone (not cleared) below.
            self._rag_preview_profile_id = None
            # UX review item 1 (P0, clone flow): clone_profile_as returns
            # (True, new_profile_id) on success -- `result` IS that id here.
            # Land the user ON the new clone (picker selection) with an
            # actionable next step, instead of the generic "Profile cloned."
            # that silently snapped the Select back to whatever was active
            # (the id was discarded entirely pre-fix).
            new_clone_id = result if action == "clone" else None
            if action == "clone":
                clone_name = self._library_rag_profile_name(new_clone_id)
                message = f"Cloned to '{clone_name}'. Select 'Set active' to edit it."
            else:
                messages = {
                    "rename": "Profile renamed.",
                    "delete": "Profile deleted.",
                }
                message = messages.get(action, "Done.")
            # I1 (SP3 final review): delete_user_profile's success `result` is
            # normally "" (nothing to add), but carries a human-readable note
            # when the delete just fell back the active-profile pointer to
            # the hybrid_basic builtin (it deleted the profile that WAS
            # active) -- surface that alongside the plain "Profile deleted."
            # so the user knows their active profile changed too. `result`
            # means something different for clone/rename on success (the new
            # profile id / nothing), so this is delete-only.
            if action == "delete" and result:
                message = f"{message} {result}"
            self._library_rag_profile_result = message
            self._set_static_text("#settings-library-rag-profile-result", message)
            self._sync_library_rag_widgets()
            if new_clone_id:
                self._sync_library_rag_profile_widgets(select_override=new_clone_id)
            else:
                self._sync_library_rag_profile_widgets()
            self._update_library_rag_editor_title()
            self._set_library_rag_preview_banner(None)
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            # Task 5 (541 v2 UX AC5): a clone is exactly how a first-run
            # install gets its first user profile -- this is the one
            # first-run-ending trigger that never touches the index status,
            # so it needs its own explicit re-evaluation (rename/delete
            # never flip the predicate in practice, but recomputing is
            # cheap and keeps every successful profile action consistent).
            self._refresh_rag_first_run_panel_state()
            self.app.notify(message, severity="information")
            return
        reason = result or "failed"
        message = f"Couldn't {action} profile: {reason}"
        self._library_rag_profile_result = message
        self._set_static_text("#settings-library-rag-profile-result", message)
        self.app.notify(message, severity="error")

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
    def handle_storage_library_collections_db_path_changed(
        self, event: Input.Changed
    ) -> None:
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
        loaded_provider = str(
            self._provider_loaded_setting_values().get("provider") or ""
        )
        previous_provider = str(
            self._provider_setting_values_mapping().get("provider") or ""
        )
        provider_changed = bool(provider) and provider_config_key(
            provider
        ) != provider_config_key(previous_provider)
        staged_provider = (
            loaded_provider
            if (
                provider
                and provider_config_key(provider)
                == provider_config_key(loaded_provider)
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
                endpoint_input.placeholder = self._provider_endpoint_placeholder(
                    staged_provider
                )
            finally:
                self._syncing_provider_endpoint = False
        self._sync_provider_credential_widget(staged_provider)
        provider_default_model = (
            self._provider_model_default(staged_provider) if provider_changed else ""
        )
        if provider_changed:
            self._stage_provider_value("model", provider_default_model or None)
            try:
                self.query_one(
                    "#settings-model-value", Input
                ).value = provider_default_model
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
        if self._navigation_provider and provider_config_key(
            provider
        ) == provider_config_key(self._navigation_provider):
            return
        current_provider = str(
            self._provider_setting_values_mapping().get("provider") or ""
        )
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
        current_model = str(
            self._provider_setting_values_mapping().get("model") or ""
        ).strip()
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
    def handle_model_profile_presence_penalty_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_model_profile_input(
            "model_profile_presence_penalty",
            event.value,
            self._normalise_model_profile_presence_penalty,
        )

    @on(Input.Changed, "#settings-model-profile-frequency-penalty")
    def handle_model_profile_frequency_penalty_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_model_profile_input(
            "model_profile_frequency_penalty",
            event.value,
            self._normalise_model_profile_frequency_penalty,
        )

    @on(Input.Changed, "#settings-model-profile-reasoning-effort")
    def handle_model_profile_reasoning_effort_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_model_profile_input(
            "model_profile_reasoning_effort",
            event.value,
            self._normalise_model_profile_reasoning_effort,
        )

    @on(Input.Changed, "#settings-model-profile-reasoning-summary")
    def handle_model_profile_reasoning_summary_changed(
        self, event: Input.Changed
    ) -> None:
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
    def handle_model_profile_thinking_effort_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_model_profile_input(
            "model_profile_thinking_effort",
            event.value,
            self._normalise_model_profile_thinking_effort,
        )

    @on(Input.Changed, "#settings-model-profile-thinking-budget-tokens")
    def handle_model_profile_thinking_budget_tokens_changed(
        self, event: Input.Changed
    ) -> None:
        self._stage_model_profile_input(
            "model_profile_thinking_budget_tokens",
            event.value,
            self._normalise_model_profile_thinking_budget_tokens,
        )

    def _stage_model_profile_input(
        self, key: str, raw_value: object, normalizer
    ) -> None:
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

    @on(Checkbox.Changed)
    def handle_model_catalog_toggle_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = str(getattr(event.checkbox, "id", "") or "")
        if checkbox_id not in MODEL_CATALOG_CHECKBOX_IDS:
            return
        event.stop()
        self._persist_model_catalog_settings()

    @on(Input.Changed, "#settings-model-catalog-stale-hours")
    def handle_model_catalog_stale_hours_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._persist_model_catalog_settings()

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
        self._select_category(
            SettingsCategoryId.PROVIDERS_MODELS.value, restore_focus=True
        )

    @on(Button.Pressed, "#settings-open-advanced-config")
    def handle_open_advanced_config_from_privacy(self, event: Button.Pressed) -> None:
        event.stop()
        self._select_category(
            SettingsCategoryId.ADVANCED_CONFIG.value, restore_focus=True
        )

    @on(Button.Pressed, "#settings-validate-config")
    def handle_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._run_diagnostics_validation()

    @on(Button.Pressed, "#settings-reload-config")
    def handle_reload_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._run_diagnostics_reload()

    @on(Button.Pressed, "#settings-run-setup-wizard")
    def handle_run_setup_wizard(self, event: Button.Pressed) -> None:
        event.stop()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        # Wire the app-level result callback so a truthy exit_route off the
        # Summary step ("Go to Chat") still navigates -- without it, the
        # exit_route is silently dropped and re-run's "Go to Chat" is dead.
        self.app.push_screen(
            FirstRunSetupWizard(self.app_instance, rerun=True),
            self.app_instance.handle_first_run_wizard_result,
        )

    @on(Button.Pressed, "#settings-advanced-validate-config")
    def handle_advanced_validate_config(self, event: Button.Pressed) -> None:
        event.stop()
        current_text = self._advanced_editor_text()
        self._advanced_config_result = "Advanced config validation: running"
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )
        self._update_advanced_validation_status()
        self._advanced_validate_config_worker(current_text)

    @on(Button.Pressed, "#settings-advanced-save-config")
    def handle_advanced_save_config(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = "Advanced config save: saving"
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )
        try:
            self.query_one("#settings-advanced-save-config", Button).disabled = True
        except QueryError:
            pass
        self._advanced_save_config_worker(self._advanced_editor_text())

    @on(Button.Pressed, "#settings-advanced-load-backup")
    def handle_advanced_load_backup(self, event: Button.Pressed) -> None:
        event.stop()
        self._advanced_config_result = (
            "Advanced config recovery: loading backup preview"
        )
        self._set_static_text(
            "#settings-advanced-config-result", self._advanced_config_result
        )
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

    def action_settings_save_category(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        category = self._active_category_id()
        if category not in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            self.app.notify(
                self._guided_action_message(category), severity="information"
            )
            return
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            try:
                values = self._provider_form_values_from_widgets()
            except ValueError as exc:
                self._provider_save_result = (
                    str(exc) or "Model profile values are invalid."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
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
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
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
                    self.query_one(
                        "#settings-provider-endpoint-value", Input
                    ).value = endpoint
                except QueryError:
                    pass
            endpoint_validation_error = self._validate_provider_endpoint(endpoint)
            if endpoint_validation_error:
                self._provider_save_result = endpoint_validation_error
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(endpoint_validation_error, severity="error")
                return
            credential_validation_error = self._validate_credential_env_var(
                credential_env_var
            )
            if credential_validation_error:
                self._provider_save_result = credential_validation_error
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(credential_validation_error, severity="error")
                return
            api_key_validation_error = self._validate_provider_api_key(api_key)
            if api_key_validation_error:
                self._provider_save_result = api_key_validation_error
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
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
            provider_section_key, _provider_config = self._provider_config_entry(
                provider
            )
            current_provider_endpoint = self._provider_endpoint_value(provider)
            current_credential_env_var = self._provider_credential_env_var(provider)
            api_key_dirty = draft is not None and "api_key" in draft.dirty_keys
            endpoint_dirty = endpoint != current_provider_endpoint
            credential_dirty = credential_env_var != current_credential_env_var
            if endpoint_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving an endpoint."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(self._provider_save_result, severity="error")
                return
            if credential_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving a credential source."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(self._provider_save_result, severity="error")
                return
            if api_key_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving an API key."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(self._provider_save_result, severity="error")
                return
            if model_profile_dirty and not model:
                self._provider_save_result = (
                    "Model is required before saving a model default profile."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(self._provider_save_result, severity="error")
                return
            if model_profile_dirty and not provider_key:
                self._provider_save_result = (
                    "Provider is required before saving a model default profile."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
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
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify("No Settings changes to save.", severity="information")
                return
            saved = True
            if dirty_values:
                saved = SettingsConfigAdapter().save_values(
                    "chat_defaults", dirty_values
                )
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
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                # TASK-366: the last Test verdict described the pre-save draft;
                # don't let a stale ready/blocked line persist after saving.
                self._mark_provider_test_result_stale()
                self._sync_provider_credential_widget(provider)
                self._update_provider_dynamic_widgets()
                self._update_draft_status_widgets(category)
                self.app.notify(
                    "Provider and model settings saved.", severity="information"
                )
            else:
                self._provider_save_result = (
                    "Failed to save provider and model settings."
                )
                self._set_static_text(
                    "#settings-provider-save-result", self._provider_save_result
                )
                self.app.notify(
                    "Failed to save provider and model settings.", severity="error"
                )
            return

        if category is SettingsCategoryId.STORAGE:
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._storage_current_defaults()
            validation = validate_storage_defaults(values)
            if not validation.valid:
                self._storage_result = validation.message
                self._set_static_text(
                    "#settings-storage-save-result", self._storage_result
                )
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
            # Task 4 (541 v2 UX AC1): Save is blocked (no-op + notify)
            # while the editor is showing a profile-picker PREVIEW -- the
            # Save/Revert buttons already disable via
            # `_library_rag_save_enabled`, but this action can be reached
            # directly (keybinding, a test calling it, a stale enabled
            # button from a race), so it must be safe standing alone too.
            # Nothing in this branch below may run: staging is never
            # possible while previewing (see `_library_rag_edits_suppressed`),
            # so `_library_rag_current_defaults()` here would only ever
            # reflect the ACTIVE profile's own (unrelated) state anyway.
            if self._rag_preview_profile_id is not None:
                self.app.notify(
                    "Return to the active profile to save.", severity="warning"
                )
                return
            # TASK-2 review (Finding 2): a "Save" choice from
            # RagProfileSwitchConfirmModal arms `_rag_profile_pending_activate`
            # then calls back in here -- but `_apply_library_rag_save_result`
            # (the only clearing site) only runs once the save worker
            # dispatches below. Capture-and-clear up front so EVERY early
            # return in this branch (no-unsaved-changes, validation failure)
            # drops the stale pending id instead of leaking it into a later,
            # unrelated successful save; re-arm it only right before the
            # worker dispatch that will actually consume it.
            pending_activate = self._rag_profile_pending_activate
            self._rag_profile_pending_activate = None
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._library_rag_current_defaults()
            validation = validate_library_rag_defaults(values)
            if not validation.valid:
                self._library_rag_result = validation.message
                self._set_static_text(
                    "#settings-library-rag-save-result", self._library_rag_result
                )
                self._update_draft_status_widgets(category)
                self.app.notify(validation.message, severity="error")
                return
            # Task 2 (541 v2 UX): gate behind a pre-commit re-index confirm
            # when this save would re-point the active profile at a fresh,
            # EMPTY collection while the CURRENT one is actually built (see
            # _confirm_reindex_then_save's docstring). `pending_activate`
            # travels through the whole gate/confirm chain as a plain
            # argument -- `_rag_profile_pending_activate` stays cleared
            # (from the capture-and-clear above) until the save actually
            # dispatches, so a Cancel never re-arms it.
            self._confirm_reindex_then_save(values, pending_activate)
            return

        if category is SettingsCategoryId.APPEARANCE:
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to save.", severity="information")
                return
            values = self._appearance_current_defaults()
            validation = validate_appearance_defaults(values)
            if not validation.valid:
                self._appearance_result = validation.message
                self._set_static_text(
                    "#settings-appearance-save-result", self._appearance_result
                )
                self._update_appearance_validation_classes()
                self._update_draft_status_widgets(category)
                self.app.notify(validation.message, severity="error")
                return
            section_values = build_appearance_save_sections(
                self._app_config_mapping(),
                values,
            )
            self._appearance_result = "Appearance defaults saving..."
            self._set_static_text(
                "#settings-appearance-save-result", self._appearance_result
            )
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
                if "max_parallel_runs" in dirty_values:
                    dirty_values["max_parallel_runs"] = (
                        self._normalise_console_max_parallel_runs(
                            dirty_values["max_parallel_runs"]
                        )
                    )
                if "tool_result_display_chars" in dirty_values:
                    dirty_values["tool_result_display_chars"] = (
                        self._normalise_tool_result_display_chars(
                            dirty_values["tool_result_display_chars"]
                        )
                    )
                if "streaming" in dirty_values:
                    dirty_values["streaming"] = (
                        self._normalise_console_default_streaming(
                            dirty_values["streaming"]
                        )
                    )
                if "temperature" in dirty_values:
                    dirty_values["temperature"] = (
                        self._normalise_console_default_temperature(
                            dirty_values["temperature"]
                        )
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
                    dirty_values["max_tokens"] = (
                        self._normalise_console_default_max_tokens(
                            dirty_values["max_tokens"]
                        )
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
                key.startswith("background_effects.")
                and key in CONSOLE_BACKGROUND_EFFECT_KEYS
                for key in dirty_values
            )
            raw_scope = self._raw_console_background_scope()
            if background_effects_dirty or str(raw_scope) == "workbench":
                merged_background_effects = self._loaded_console_background_effects()
                for key in CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER:
                    if key in dirty_values:
                        merged_background_effects[
                            key.removeprefix("background_effects.")
                        ] = dirty_values[key]
                previous_scope = merged_background_effects.get("scope")
                available_scope = self._available_console_background_scope(
                    previous_scope
                )
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

        self.app.notify(
            "This Settings category has no save action yet.", severity="warning"
        )

    def action_settings_revert_category(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        category = self._active_category_id()
        if category is SettingsCategoryId.IMAGE_GENERATION:
            # Unlike THEME/SPLASH_SCREEN/INTERNAL_PROMPTS below (which have
            # no unified revert this action could drive at all), Image Gen
            # DOES have one -- `_handle_image_gen_revert` (the same coroutine
            # the panel's own Revert button calls). Routing the footer `r`
            # shortcut through the generic draft-pop-only path further down
            # would have popped the draft/cleared the `*` marker without
            # ever recomposing the panel, leaving its Input widgets stuck
            # showing the just-discarded unsaved text until the category was
            # re-entered. `run_worker` schedules the coroutine since this
            # action itself is sync (mirrors the Button.Pressed handler,
            # which is async and awaits it directly instead).
            if not self._category_has_unsaved_changes(category):
                self.app.notify("No Settings changes to revert.", severity="information")
                return
            self.run_worker(self._handle_image_gen_revert(), exclusive=False)
            return
        if category in (
            SettingsCategoryId.THEME,
            SettingsCategoryId.SPLASH_SCREEN,
            SettingsCategoryId.INTERNAL_PROMPTS,
            SettingsCategoryId.WORKSPACES,
        ):
            self.app.notify(
                "Use the editor's own buttons for this category", severity="information"
            )
            return
        # Task 4 review (Important): Revert is blocked (no-op + notify)
        # while the Library/RAG editor is showing a profile-picker PREVIEW
        # -- MUST run before the generic draft pop below, which would
        # otherwise silently discard the ACTIVE profile's own (unrelated
        # to whatever is being previewed) staged draft. Mirrors the Save
        # guard in action_settings_save_category.
        if (
            category is SettingsCategoryId.LIBRARY_RAG
            and self._rag_preview_profile_id is not None
        ):
            self.app.notify(
                "Return to the active profile to revert.", severity="warning"
            )
            return
        if not self._category_has_unsaved_changes(category):
            self.app.notify("No Settings changes to revert.", severity="information")
            return
        self._settings_drafts.pop(category, None)
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            self._console_behavior_result = (
                "Console behavior settings reverted to last loaded values."
            )
            self._sync_console_behavior_widgets()
        elif category is SettingsCategoryId.APPEARANCE:
            self._appearance_result = (
                "Appearance defaults reverted to last loaded values."
            )
            self._sync_appearance_widgets()
            self._update_draft_status_widgets(category)
        elif category is SettingsCategoryId.LIBRARY_RAG:
            self._library_rag_result = (
                "Library/RAG defaults reverted to last loaded values."
            )
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
                    self.query_one(
                        "#settings-provider-value", Select
                    ).value = self._provider_select_value_for_provider(provider)
                finally:
                    self._syncing_provider_selection = False
                self._sync_provider_manual_widget(provider)
                self.query_one("#settings-model-value", Input).value = str(
                    values["model"]
                )
                endpoint_input = self.query_one(
                    "#settings-provider-endpoint-value", Input
                )
                endpoint_input.value = str(values["endpoint"])
                endpoint_input.placeholder = self._provider_endpoint_placeholder(
                    provider
                )
                api_key_input = self.query_one("#settings-provider-api-key", Input)
                api_key_input.value = str(values.get("api_key") or "")
                api_key_input.placeholder = self._provider_api_key_placeholder(provider)
                credential_input = self.query_one(
                    "#settings-provider-credential-env-var",
                    Input,
                )
                credential_input.value = str(values["credential_env_var"])
                credential_input.placeholder = self._provider_credential_placeholder(
                    provider
                )
                for draft_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS:
                    profile_value = values[draft_key]
                    self.query_one(
                        f"#settings-{draft_key.replace('_', '-')}",
                        Input,
                    ).value = self._profile_input_value(profile_value)
            except QueryError:
                pass
            self._provider_save_result = (
                "Provider settings reverted to last loaded values."
            )
            self._set_static_text(
                "#settings-provider-save-result", self._provider_save_result
            )
            self._update_provider_dynamic_widgets()
            self._update_draft_status_widgets(category)
        else:
            self._update_draft_status_widgets(category)
        self.app.notify("Settings category changes reverted.", severity="information")

    def action_settings_test_category(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        if self._active_category_id() is SettingsCategoryId.PROVIDERS_MODELS:
            detail, summary, passed = self._provider_readiness_test_report()
            probe_base_url = self._provider_live_probe_base_url() if passed else ""
            if probe_base_url:
                # task-191: readiness passed for a URL-based provider; run a
                # short live probe in a worker and fold it into the toast.
                self._provider_test_result = f"{detail} | endpoint probe: checking"
                self._update_provider_test_result()
                self._provider_endpoint_probe_worker(probe_base_url, detail, summary)
                return
            self._provider_test_result = detail
            self._update_provider_test_result()
            self.app.notify(
                summary,
                severity="information" if passed else "warning",
            )
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
            self.app.notify(
                "Diagnostics validation and reload started.", severity="information"
            )
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
                self._set_static_text(
                    "#settings-appearance-save-result", self._appearance_result
                )
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
            self._set_static_text(
                "#settings-appearance-save-result", self._appearance_result
            )
            self.app.notify("Appearance preview complete.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.LIBRARY_RAG:
            # UX review item 8: 't test category' previously fell all the
            # way through to the generic "No test action..." toast for RAG
            # even though there's a cheap, honest check available -- refetch
            # the active profile's index status (same off-thread worker
            # pattern as category-show/set-active/save) and report it
            # alongside the current preview defaults.
            self.app.notify("RAG check started.", severity="information")
            self._rag_test_category_worker()
            return
        self.app.notify(
            "No test action is available for this Settings category yet.",
            severity="warning",
        )

    # --- Task 6 (541 AC6): RAG profile-workflow keyboard accelerators ---
    #
    # Unlike s/r/t (dispatched per-category from within one shared action),
    # these three are RAG-only: the guard lives in each action rather than a
    # branch inside a shared method, since there is no cross-category
    # meaning for "set active"/"clone"/"backfill". Each delegates to the
    # EXACT SAME trigger its corresponding button uses, so a key press
    # behaves identically to a click in every state (dirty-draft
    # switch-confirm modal, preview clear, first-run starter panel, the
    # backfill in-flight guard) -- no bespoke reimplementation here.

    def action_settings_rag_set_active(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        """'a' -- Set active for whatever profile the picker currently
        shows. No-op outside LIBRARY_RAG or while an Input/TextArea has
        focus (same guard shape as s/r/t; matters for direct callers, since
        a real keypress while typing is already swallowed by the focused
        widget before it would reach this binding).

        Args:
            allow_text_entry_focus: Skip the focused-text-entry no-op guard
                (for direct callers such as buttons/tests; a real keypress
                never arrives with a text entry focused).
        """
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        self._trigger_library_rag_profile_set_active()

    def action_settings_rag_clone(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        """'c' -- Clone the profile the picker currently shows. Same guard
        as action_settings_rag_set_active.

        Args:
            allow_text_entry_focus: Skip the focused-text-entry no-op guard
                (for direct callers; see action_settings_rag_set_active).
        """
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        self._trigger_library_rag_profile_clone()

    def action_settings_rag_backfill(
        self, *, allow_text_entry_focus: bool = False
    ) -> None:
        """'b' -- Backfill the active profile's index. Same guard as
        action_settings_rag_set_active.

        Args:
            allow_text_entry_focus: Skip the focused-text-entry no-op guard
                (for direct callers; see action_settings_rag_set_active).
        """
        if not allow_text_entry_focus and self._settings_text_entry_has_focus():
            return
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        self._trigger_library_rag_index_backfill()

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
            self._set_static_text(
                "#settings-appearance-save-result", self._appearance_result
            )
            self._sync_appearance_widgets()
            self.app.notify("Appearance defaults saved.", severity="information")
            return
        self._appearance_result = "Failed to save Appearance defaults."
        self._set_static_text(
            "#settings-appearance-save-result", self._appearance_result
        )
        self.app.notify(self._appearance_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_appearance_worker(
        self, section_values: Mapping[str, object]
    ) -> None:
        saved = self._save_appearance_sections(section_values)
        self.app.call_from_thread(
            self._apply_appearance_save_result,
            saved,
            dict(section_values),
        )

    def _confirm_reindex_then_save(
        self,
        values: SettingsLibraryRagDefaults,
        pending_activate: str | None,
    ) -> None:
        """Task 2 (541 v2 UX): pre-commit gate for the LIBRARY_RAG save.

        A save that would re-point the active profile at a fresh, EMPTY
        vector collection while the CURRENT collection is actually BUILT
        (has vectors worth losing) must be confirmed by the user BEFORE the
        save worker dispatches -- silently saving straight into "search
        returns nothing" is a trap. When the current index has nothing to
        lose (absent/empty/unknown, or the save doesn't change the
        collection at all), this proceeds straight through: the existing
        post-save ``RAG_INDEX_CHANGE_WARNING`` notice already covers that
        case honestly.

        ``index_change_pending`` is pure/fast (in-memory fingerprint
        compare), so it's always computed inline. The index STATUS check,
        by contrast, touches on-disk Chroma -- prefer the Static's last
        cached fetch (``_library_rag_index_status_cache``, kept fresh by
        every trigger that already calls ``_apply_library_rag_index_status``:
        category show, post-save, post-set-active, 't' test) to avoid
        adding save-click latency; only when nothing has been cached yet
        does this dispatch its own off-thread fetch before deciding.
        """
        if not index_change_pending(values):
            self._dispatch_library_rag_save(values, False, pending_activate)
            return
        cached_status = self._library_rag_index_status_cache
        if cached_status is not None:
            self._decide_reindex_confirmation(values, pending_activate, cached_status)
            return
        if self._rag_reindex_confirm_in_flight:
            # Debounce (Task 2 review, Important): a status fetch for an
            # earlier Save click's cache-miss window is already running --
            # dispatching a SECOND worker in the same exclusive @work group
            # would CANCEL the first one, silently dropping ITS
            # pending_activate. No-op instead; the in-flight fetch will
            # still complete and dispatch save for the FIRST click once it
            # lands.
            return
        self._rag_reindex_confirm_in_flight = True
        self._rag_reindex_confirm_status_worker(values, pending_activate)

    def _decide_reindex_confirmation(
        self,
        values: SettingsLibraryRagDefaults,
        pending_activate: str | None,
        status: Mapping[str, object],
    ) -> None:
        """Given a (cached or freshly fetched) index status, either push the
        re-index confirm modal (state == "built") or proceed straight to
        dispatch (absent/empty/unknown -- nothing built to lose)."""
        if str(status.get("state") or "unknown") != "built":
            self._dispatch_library_rag_save(values, True, pending_activate)
            return
        # task-566: this decision can be reached by a `settings-rag-index-
        # status` worker callback that was already in flight when the user
        # navigated away from Library/RAG (`_select_category`'s
        # `cancel_group` is best-effort, not a guarantee for an
        # already-running thread). Never surface the destructive "Re-index
        # required" modal over an unrelated category -- there's no one left
        # to confirm it, so the save attempt is dropped here rather than
        # auto-confirmed or shown out of context.
        if self._active_category_id() is not SettingsCategoryId.LIBRARY_RAG:
            return
        count = status.get("count", 0)
        # 541-v2 final review item 3: thousands separator -- a large library
        # can easily have a 6-7 digit vector count, unreadable at a glance
        # as a bare run of digits.
        modal = ConfirmationDialog(
            title="Re-index required",
            message=(
                "This change re-points to a new EMPTY index — the current "
                f"index ({count:,} vectors) stops being used and search "
                "returns nothing until you run Backfill. Save anyway?"
            ),
            confirm_label="Save anyway",
            cancel_label="Cancel",
        )
        self.app.push_screen(
            modal,
            lambda confirmed: self._handle_reindex_confirmation_result(
                confirmed, values, pending_activate
            ),
        )

    def _handle_reindex_confirmation_result(
        self,
        confirmed: bool | None,
        values: SettingsLibraryRagDefaults,
        pending_activate: str | None,
    ) -> None:
        # Task 2 review (Important): defensive clear -- by the time this
        # modal resolves the in-flight guard is normally already cleared
        # (the worker callback below clears it right after the decision
        # that pushed this very modal), but clearing it again here too,
        # unconditionally, on BOTH the Confirm and Cancel branches, means
        # this handler can never be the reason a future Save stays
        # debounced.
        self._rag_reindex_confirm_in_flight = False
        if not confirmed:
            # Cancel: the draft stays staged (never popped on this path) and
            # `_rag_profile_pending_activate` stays cleared (never re-armed
            # -- see the capture-and-clear comment at the LIBRARY_RAG save
            # branch) -- no save dispatched, nothing lost, nothing leaked.
            return
        self._dispatch_library_rag_save(values, True, pending_activate)

    def _clear_rag_reindex_confirm_in_flight(self) -> None:
        """Main-thread flip of the in-flight guard -- see
        ``_rag_reindex_confirm_status_worker``'s ``finally`` block."""
        self._rag_reindex_confirm_in_flight = False

    @work(exclusive=True, thread=True, group="settings-rag-index-status")
    def _rag_reindex_confirm_status_worker(
        self,
        values: SettingsLibraryRagDefaults,
        pending_activate: str | None,
    ) -> None:
        try:
            status = fetch_index_status()
            self.app.call_from_thread(self._apply_library_rag_index_status, status)
            self.app.call_from_thread(
                self._decide_reindex_confirmation, values, pending_activate, status
            )
        finally:
            # Task 2 review (Important): ALWAYS clears the in-flight guard,
            # even if something above raises -- fetch_index_status() itself
            # never raises (see its own except-fallback), but this is a
            # belt-and-suspenders net: without it, a failure here would
            # leave the flag stuck True forever, silently no-op-ing every
            # future Save on this category ("Save bricks"). `call_from_thread`
            # is synchronous from this (background) thread's point of view,
            # so this runs only AFTER `_decide_reindex_confirmation` above
            # has already returned -- covers both the direct-dispatch and
            # the modal-pushed outcome in this one place.
            self.app.call_from_thread(self._clear_rag_reindex_confirm_in_flight)

    def _dispatch_library_rag_save(
        self,
        values: SettingsLibraryRagDefaults,
        index_will_change: bool,
        pending_activate: str | None,
    ) -> None:
        self._library_rag_result = "Saving Library/RAG defaults..."
        self._set_static_text(
            "#settings-library-rag-save-result", self._library_rag_result
        )
        self._rag_profile_pending_activate = pending_activate
        self._settings_save_library_rag_worker(values, index_will_change)

    def _apply_library_rag_save_result(
        self,
        saved: bool,
        reason: str,
        index_will_change: bool = False,
    ) -> None:
        # A "Save" choice from RagProfileSwitchConfirmModal defers the profile
        # switch until this save completes; consumed (and cleared) exactly
        # once here regardless of outcome, so a later unrelated save never
        # replays a stale switch.
        pending_activate = self._rag_profile_pending_activate
        self._rag_profile_pending_activate = None
        if saved:
            self._settings_drafts.pop(SettingsCategoryId.LIBRARY_RAG, None)
            message = "Library/RAG defaults saved."
            # Task 4 (SP3), save-path trigger (a): honest re-index warning
            # when the just-saved fields re-point the fingerprinted
            # collection -- computed pre-save by index_change_pending, see
            # the dispatch site above.
            if index_will_change:
                message = f"{message} {RAG_INDEX_CHANGE_WARNING}"
            self._library_rag_result = message
            self._set_static_text(
                "#settings-library-rag-save-result", self._library_rag_result
            )
            self._sync_library_rag_widgets()
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            self.app.notify(
                message, severity="warning" if index_will_change else "information"
            )
            if pending_activate:
                # The deferred set-active worker fetches its own fresh index
                # status for the NEW active profile -- refreshing here first
                # would just be immediately-stale, wasted off-thread work.
                self._dispatch_rag_set_active(pending_activate)
            else:
                self._refresh_library_rag_index_status()
            return
        if reason == "builtin":
            self._library_rag_result = (
                "Built-in profile is read-only — Clone to edit."
            )
            self._set_static_text(
                "#settings-library-rag-save-result", self._library_rag_result
            )
            self._update_draft_status_widgets(SettingsCategoryId.LIBRARY_RAG)
            self.app.notify(
                "Built-in profile is read-only — Clone to edit", severity="warning"
            )
            return
        self._library_rag_result = "Failed to save Library/RAG defaults."
        self._set_static_text(
            "#settings-library-rag-save-result", self._library_rag_result
        )
        self.app.notify(self._library_rag_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_library_rag_worker(
        self, values: SettingsLibraryRagDefaults, index_will_change: bool = False
    ) -> None:
        saved, reason = save_rag_defaults_to_active_profile(values)
        self.app.call_from_thread(
            self._apply_library_rag_save_result,
            saved,
            reason,
            index_will_change,
        )

    def _apply_storage_save_result(
        self,
        saved: bool,
        section_values: Mapping[str, object],
    ) -> None:
        if saved:
            self._app_config_update_target().update(copy.deepcopy(dict(section_values)))
            self._settings_drafts.pop(SettingsCategoryId.STORAGE, None)
            self._storage_result = (
                "Storage defaults saved. Restart Chatbook to use saved paths."
            )
            self._set_static_text("#settings-storage-save-result", self._storage_result)
            self._sync_storage_widgets()
            self.app.notify("Storage defaults saved.", severity="information")
            return
        self._storage_result = "Failed to save Storage defaults."
        self._set_static_text("#settings-storage-save-result", self._storage_result)
        self.app.notify(self._storage_result, severity="error")

    @work(exclusive=True, thread=True)
    def _settings_save_storage_worker(
        self, section_values: Mapping[str, object]
    ) -> None:
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
            self.query_one(
                "#settings-console-collapse-large-pastes-toggle", Button
            ).label = self._collapse_large_pastes_button_label()
        except QueryError:
            pass
        try:
            self._syncing_console_threshold = True
            try:
                self.query_one(
                    "#settings-console-paste-collapse-threshold", Input
                ).value = str(self._paste_collapse_threshold_value())
            finally:
                self._syncing_console_threshold = False
        except QueryError:
            pass
        try:
            self._syncing_console_max_parallel_runs = True
            try:
                self.query_one(
                    "#settings-console-max-parallel-runs", Input
                ).value = str(self._console_max_parallel_runs_value())
            finally:
                self._syncing_console_max_parallel_runs = False
        except QueryError:
            pass
        try:
            self._syncing_console_tool_result_display_chars = True
            try:
                self.query_one(
                    "#settings-console-tool-result-display-chars", Input
                ).value = str(self._tool_result_display_chars_value())
            finally:
                self._syncing_console_tool_result_display_chars = False
        except QueryError:
            pass
        input_values = {
            "#settings-console-default-streaming": self._console_behavior_value(
                "streaming"
            ),
            "#settings-console-default-temperature": self._console_behavior_value(
                "temperature"
            ),
            "#settings-console-default-top-p": self._console_behavior_value("top_p"),
            "#settings-console-default-min-p": self._console_behavior_value("min_p"),
            "#settings-console-default-top-k": self._console_behavior_value("top_k"),
            "#settings-console-default-max-tokens": self._console_behavior_value(
                "max_tokens"
            ),
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
            "#settings-console-default-verbosity": self._console_behavior_value(
                "verbosity"
            ),
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
                    self.query_one(selector, Input).value = self._console_input_value(
                        value
                    )
                except QueryError:
                    pass
        finally:
            self._syncing_console_defaults = False
        self._syncing_console_background_effects = True
        try:
            try:
                self.query_one(
                    "#settings-console-background-effect-enabled", Button
                ).label = self._console_background_effect_enabled_label()
            except QueryError:
                pass
            select_values = {
                "#settings-console-background-effect-type": self._console_background_effect_value(
                    "effect"
                ),
                "#settings-console-background-effect-scope": self._console_background_effect_value(
                    "scope"
                ),
                "#settings-console-background-effect-intensity": self._console_background_effect_value(
                    "intensity"
                ),
            }
            for selector, value in select_values.items():
                try:
                    self.query_one(selector, Select).value = str(value)
                except QueryError:
                    pass
            try:
                self.query_one(
                    "#settings-console-background-effect-fps", Input
                ).value = str(
                    self._console_background_effect_value("fps")
                    or DEFAULT_CONSOLE_BACKGROUND_FPS
                )
            except QueryError:
                pass
        finally:
            self._syncing_console_background_effects = False
        self._set_static_text(
            "#settings-console-behavior-result", self._console_behavior_result_text()
        )
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
                self.query_one(
                    "#settings-appearance-palette-theme-limit", Input
                ).value = str(values["palette_theme_limit"])
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
                self.query_one(
                    "#settings-appearance-animations-enabled", Button
                ).label = self._appearance_bool_label("animations_enabled")
            except QueryError:
                pass
            try:
                self.query_one(
                    "#settings-appearance-smooth-scrolling", Button
                ).label = self._appearance_bool_label("smooth_scrolling")
            except QueryError:
                pass
        finally:
            self._syncing_appearance_defaults = False
        self._set_static_text(
            "#settings-appearance-save-result", self._appearance_result
        )
        self._update_appearance_validation_classes()
        self._update_draft_status_widgets(SettingsCategoryId.APPEARANCE)

    def on_key(self, event: Key) -> None:
        focused = self._focused_widget()
        is_slash = (
            event.key in {"/", "slash"} or getattr(event, "character", None) == "/"
        )
        if is_slash and self._category_search_has_focus():
            # task-1584: "/" on the already-focused filter re-arms it
            # (select-all so typing replaces) instead of inserting a
            # literal slash into the query.
            self._focus_category_search()
            event.stop()
            event.prevent_default()
            return
        if is_slash and not isinstance(focused, (Input, TextArea)):
            self._focus_category_search()
            event.stop()
            event.prevent_default()
            return
        if (
            event.key == "escape"
            and isinstance(focused, (Input, TextArea))
            and not self._category_search_has_focus()
        ):
            # task-1560: Esc releases field focus so the footer's advertised
            # "Esc, s save category" chain is true for EVERY text-entry
            # field -- previously only the filter input handled Esc, so
            # keys after Esc kept feeding the field (the critique's silent
            # no-op trap, one level deeper).
            self.set_focus(None)
            self._register_footer_shortcuts()
            event.stop()
            event.prevent_default()
            return
        if event.key == "escape" and self.category_search_query:
            if self._category_search_has_focus() or not isinstance(
                focused, (Input, TextArea)
            ):
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
            if focused is None or getattr(focused, "has_class", lambda *_: False)(
                "nav-button"
            ):
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
                "settings-run-setup-wizard",
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
