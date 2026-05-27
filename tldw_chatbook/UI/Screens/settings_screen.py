"""Settings destination shell for global app preferences."""

from collections.abc import Mapping
import logging
import os
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Button, Input, Rule, Static, TextArea

from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Sync_Interop.sync_promotion_state import SyncPromotionState, build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Widgets.destination_workbench import DestinationModeStrip
from ...config import DEFAULT_CONFIG_PATH, coerce_bool_setting, save_setting_to_cli_config
from ...Utils.input_validation import sanitize_string, validate_text_input, validate_url
from ...Utils.path_validation import validate_path_simple
from ..Navigation.base_app_screen import BaseAppScreen
from .provider_model_resolution import resolve_effective_provider_model
from .settings_config_adapter import SettingsConfigAdapter, redact_secret_text
from .settings_config_models import SettingsCategoryId, SettingsCategorySummary, SettingsDraft
from ..Navigation.main_navigation import NavigateToScreen


logger = logging.getLogger(__name__)

MAX_CATEGORY_SEARCH_QUERY_CHARS = 80
PROVIDER_ENDPOINT_KEYS = ("api_base_url", "api_base", "base_url", "api_url", "endpoint")
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
GUIDED_SETTINGS_MUTATION_CATEGORIES = frozenset(
    {
        SettingsCategoryId.PROVIDERS_MODELS,
        SettingsCategoryId.CONSOLE_BEHAVIOR,
    }
)


class SettingsScreen(BaseAppScreen):
    """Global preferences, appearance, accounts, storage, and app behavior."""

    BINDINGS = [
        ("s", "settings_save_category", "Save Settings category"),
        ("r", "settings_revert_category", "Revert Settings category"),
        ("t", "settings_test_category", "Test Settings category"),
    ]

    active_category = reactive(SettingsCategoryId.OVERVIEW.value, recompose=True)
    category_search_query = reactive("")

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "settings", **kwargs)
        self._settings_drafts: dict[SettingsCategoryId, SettingsDraft] = {}
        self._provider_test_result = "Provider test has not run."
        self._provider_save_result = "Provider settings have not been saved this session."
        self._syncing_provider_endpoint = False
        self._diagnostics_validation_result = "Config validation: not run"
        self._diagnostics_reload_result = "Config reload: not run"
        self._advanced_config_result = "Advanced config validation: not run"
        self._advanced_config_validated_text: str | None = None

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
                "Theme, density, and visual customization surface.",
                "Customize",
            ),
            SettingsCategorySummary(
                SettingsCategoryId.STORAGE,
                "Storage",
                "Config path, local databases, and file locations.",
                "Local",
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
        )

    def _active_summary(self) -> SettingsCategorySummary:
        for summary in self._category_summaries():
            if summary.category.value == self.active_category:
                return summary
        return self._category_summaries()[0]

    def _console_settings(self) -> dict:
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            self.app_instance.app_config = {}
            app_config = self.app_instance.app_config
        console_settings = app_config.setdefault("console", {})
        if not isinstance(console_settings, dict):
            console_settings = {}
            app_config["console"] = console_settings
        return console_settings

    def _chat_defaults(self) -> dict:
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            self.app_instance.app_config = {}
            app_config = self.app_instance.app_config
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

    def _update_console_paste_summary(self) -> None:
        try:
            summary = self.query_one("#settings-overview-console-paste-collapse", Static)
        except QueryError:
            return
        summary.update(f"Console paste collapse: {self._collapse_large_pastes_label()}")

    def _category_has_unsaved_changes(self, category: SettingsCategoryId) -> bool:
        draft = self._settings_drafts.get(category)
        return bool(draft and draft.is_dirty)

    def _guided_action_message(self, category: SettingsCategoryId) -> str:
        if category in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            if self._category_has_unsaved_changes(category):
                return "Guided edits: Save or Revert changes."
            return "Guided edits: change a field first."
        messages = {
            SettingsCategoryId.OVERVIEW: "Guided edits: choose Providers or Console.",
            SettingsCategoryId.APPEARANCE: "Guided edits: use Open Appearance.",
            SettingsCategoryId.STORAGE: "Guided edits: Storage is read-only.",
            SettingsCategoryId.PRIVACY_SECURITY: "Guided edits: Privacy/Security is read-only.",
            SettingsCategoryId.DIAGNOSTICS: "Guided edits: use Validate/Reload.",
            SettingsCategoryId.ADVANCED_CONFIG: "Guided edits: use Raw TOML controls.",
        }
        return messages.get(category, "Guided edits: read-only.")

    def _guided_actions_enabled(self, category: SettingsCategoryId) -> bool:
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
        if self._category_has_unsaved_changes(category):
            return "State: Unsaved changes | Save or Revert before leaving this category."
        if category is SettingsCategoryId.ADVANCED_CONFIG:
            return "State: Guarded | Save blocked until the current text validates; backup created before overwrite."
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            return "State: Shared with Console"
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            return "State: Console scoped | Changes affect composer behavior after save."
        if category is SettingsCategoryId.DIAGNOSTICS:
            return "State: Safe to run | Validation and reload expose status without writing raw TOML."
        if category is SettingsCategoryId.APPEARANCE:
            return "State: Routed | Open Appearance for theme and density controls."
        if category is SettingsCategoryId.STORAGE:
            return "State: Local paths | Verify write access before changing storage locations."
        if category is SettingsCategoryId.PRIVACY_SECURITY:
            return "State: Local privacy | Secrets stay redacted in validation and diagnostics."
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

    def _sync_safety_states(self) -> tuple[SyncPromotionState, ...]:
        labels = {
            "library_collections": "Collections",
            "workspaces": "Workspaces",
        }
        sync_scope_service = getattr(self.app_instance, "sync_scope_service", None)
        list_states = getattr(sync_scope_service, "list_write_sync_promotion_states", None)
        if callable(list_states):
            try:
                return tuple(
                    list_states(
                        domains=list(labels),
                        surface_labels=labels,
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
                    server_profile_id=None,
                    workspace_id=None,
                    registry=DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
                ),
            )
            for domain, label in labels.items()
        )

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
        except ValueError as exc:
            return f"invalid path - {redact_secret_text(str(exc))}"
        target = config_path if config_path.exists() else config_path.parent
        writable = os.access(target, os.W_OK) if target.exists() else os.access(target.parent, os.W_OK)
        return "writable" if writable else "not writable"

    def _raw_config_text(self) -> str:
        try:
            config_path = self._config_path()
        except ValueError as exc:
            return f"# Unable to use config path: {redact_secret_text(str(exc))}\n"
        if config_path.exists():
            try:
                return config_path.read_text(encoding="utf-8")
            except OSError as exc:
                return f"# Unable to read {config_path}: {type(exc).__name__}"
        return "# Config file does not exist yet.\n"

    def _known_storage_paths(self) -> tuple[str, ...]:
        try:
            paths = [f"Config path: {self._config_path()}"]
        except ValueError as exc:
            paths = [f"Config path: invalid - {redact_secret_text(str(exc))}"]
        user_data_dir = getattr(self.app_instance, "user_data_dir", None)
        if user_data_dir:
            paths.append(f"User data directory: {user_data_dir}")
        for attr_name, label in (
            ("notifications_db_path", "Notifications DB"),
            ("subscriptions_db_path", "Watchlists DB"),
            ("workspaces_db_path", "Workspaces DB"),
        ):
            value = getattr(self.app_instance, attr_name, None)
            if value:
                paths.append(f"{label}: {value}")
        return tuple(paths)

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

    def _run_diagnostics_validation_and_reload(self) -> None:
        self._run_diagnostics_validation()
        self._run_diagnostics_reload()

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

    def _provider_readiness_label(self) -> str:
        resolved = self._resolve_provider_model_for_settings()
        provider = str(resolved.provider or "not selected").strip()
        model = str(resolved.model or "not selected").strip()
        if provider and provider != "not selected":
            return f"Provider readiness: {provider} / {model}"
        return "Provider readiness: needs provider and model"

    def _provider_draft(self) -> SettingsDraft | None:
        return self._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)

    def _provider_draft_value(self, key: str):
        draft = self._provider_draft()
        return draft.values.get(key) if draft is not None and key in draft.values else None

    def _resolve_provider_model_for_settings(self):
        return resolve_effective_provider_model(
            self.app_instance,
            settings_provider=self._provider_draft_value("provider"),
            settings_model=self._provider_draft_value("model"),
        )

    def _provider_loaded_setting_values(self) -> dict[str, object]:
        defaults = self._chat_defaults()
        resolved = resolve_effective_provider_model(self.app_instance)
        return {
            "provider": str(resolved.provider or "").strip(),
            "model": str(resolved.model or "").strip(),
            "endpoint": self._provider_endpoint_value(str(resolved.provider or "")),
            "streaming": coerce_bool_setting(
                defaults.get("streaming", True),
                True,
            ),
            "temperature": defaults.get("temperature", 0.7),
        }

    def _provider_setting_values(self) -> dict[str, object]:
        loaded = self._provider_loaded_setting_values()
        return {
            key: self._provider_draft_value(key)
            if self._provider_draft_value(key) is not None
            else value
            for key, value in loaded.items()
        }

    @staticmethod
    def _normalise_temperature(value: object) -> float:
        return float(str(value).strip())

    def _provider_form_values_from_widgets(self) -> dict[str, object]:
        loaded_values = self._provider_loaded_setting_values()
        provider_value = self.query_one("#settings-provider-value", Input).value.strip()
        provider_draft = self._provider_draft()
        provider_explicitly_staged = (
            provider_draft is not None and "provider" in provider_draft.values
        )
        provider = (
            provider_value
            if provider_value or provider_explicitly_staged
            else str(loaded_values["provider"])
        )
        model = (
            self.query_one("#settings-model-value", Input).value.strip()
            or str(loaded_values["model"])
        )
        streaming = coerce_bool_setting(
            self.query_one("#settings-streaming-default", Input).value,
            True,
        )
        temperature = self._normalise_temperature(
            self.query_one("#settings-temperature-default", Input).value
        )
        endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
        return {
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "streaming": streaming,
            "temperature": temperature,
        }

    def _stage_provider_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.PROVIDERS_MODELS
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        original = self._provider_loaded_setting_values().get(key)
        draft.set_value(key, original, value)
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)

    def _provider_config(self, provider: str) -> Mapping[str, object]:
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        api_settings = app_config.get("api_settings", {}) if isinstance(app_config, Mapping) else {}
        if not isinstance(api_settings, Mapping):
            return {}
        provider_config = api_settings.get(provider_config_key(provider), {})
        return provider_config if isinstance(provider_config, Mapping) else {}

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

    def _provider_endpoint_row(self, provider: str) -> str:
        provider_key = provider_config_key(provider)
        if not provider_key:
            return "Endpoint key: provider required"
        endpoint_key = self._provider_endpoint_setting_key(provider)
        return f"Endpoint key: api_settings.{provider_key}.{endpoint_key}"

    @staticmethod
    def _validate_provider_endpoint(endpoint: object) -> str | None:
        endpoint_text = str(endpoint or "").strip()
        if not endpoint_text:
            return None
        if not validate_url(endpoint_text):
            return "Endpoint must start with http:// or https:// and include a valid host."
        return None

    def _provider_key_status(self, provider: str) -> str:
        readiness = get_provider_readiness(
            provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )
        if readiness.api_key_source:
            return f"API key: {readiness.api_key_source}"
        if readiness.env_var:
            return f"{readiness.env_var}=missing"
        if not readiness.requires_api_key:
            return "API key: not required for this provider"
        return "API key: missing"

    def _run_provider_readiness_test(self) -> str:
        try:
            provider = self.query_one("#settings-provider-value", Input).value.strip()
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
            provider = self.query_one("#settings-provider-value", Input).value.strip()
        except QueryError:
            provider = str(self._provider_setting_values().get("provider") or "")
        try:
            endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
        except QueryError:
            endpoint = self._provider_endpoint_value(provider)
        try:
            self.query_one("#settings-provider-readiness", Static).update(
                self._provider_readiness_label()
            )
            self.query_one("#settings-provider-endpoint-key", Static).update(
                self._provider_endpoint_row(provider)
            )
            self.query_one("#settings-provider-endpoint", Static).update(
                self._provider_endpoint_summary(provider, endpoint)
            )
            self.query_one("#settings-provider-key-status", Static).update(
                self._provider_key_status(provider)
            )
        except QueryError:
            pass

    def _detail_row(self, label: str, value: object, *, identifier: str | None = None) -> Static:
        return Static(
            f"{label}: {value}",
            id=identifier,
            classes="settings-detail-row",
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
                ("Affected config", "chat_defaults provider, model, streaming, and temperature"),
                ("Recovery", "test provider readiness before saving and revert if Console generation is blocked"),
                ("Boundary", "provider routing is shared with Console but runtime tool approval stays in Console"),
            ),
            SettingsCategoryId.APPEARANCE: (
                ("Affected config", "theme, density, and visual customization values"),
                ("Recovery", "open Appearance, preview changes, then return to Settings if needed"),
                ("Boundary", "visual preferences do not change runtime or data access"),
            ),
            SettingsCategoryId.STORAGE: (
                ("Affected config", "config file path, local database paths, media storage roots"),
                ("Recovery", "verify paths, reload config, then restart only if storage roots changed"),
                ("Boundary", "server handoff does not move local source content unless explicitly requested"),
            ),
            SettingsCategoryId.PRIVACY_SECURITY: (
                ("Affected config", "secret redaction, local privacy boundaries, and future encryption controls"),
                ("Recovery", "validate diagnostics output and rotate exposed credentials outside Chatbook"),
                ("Boundary", "raw secret values are not displayed in Settings validation results"),
            ),
            SettingsCategoryId.CONSOLE_BEHAVIOR: (
                ("Affected config", "composer behavior, paste collapse, and chat-flow defaults"),
                ("Recovery", "revert unsaved changes or disable paste collapse if composer flow is disrupted"),
                ("Boundary", "normal typing remains literal; only large paste chunks are transformed"),
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
                    f"{'> ' if is_active else '  '}{summary.title}",
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
            yield Static("Provider readiness", classes="destination-section")
            yield self._detail_row(
                "Provider readiness",
                self._provider_readiness_label().removeprefix("Provider readiness: "),
                identifier="settings-overview-provider-readiness",
            )
            yield Static("Storage", classes="destination-section")
            yield self._detail_row(
                "Config path",
                f"{self._config_path()} ({self._config_writable_status()})",
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
        values = self._provider_setting_values()
        yield Static("Providers & Models", classes="destination-section settings-column-title")
        with Vertical(id="settings-providers-models-card", classes="settings-focus-card"):
            yield self._render_category_state_banner(SettingsCategoryId.PROVIDERS_MODELS)
            with Horizontal(classes="settings-input-row"):
                yield Static("Provider", classes="settings-input-label")
                yield Input(
                    value=str(values["provider"]),
                    id="settings-provider-value",
                    classes="settings-compact-input",
                    placeholder="Provider, e.g. OpenAI or llama_cpp",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Model", classes="settings-input-label")
                yield Input(
                    value=str(values["model"]),
                    id="settings-model-value",
                    classes="settings-compact-input",
                    placeholder="Model name",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Endpoint", classes="settings-input-label")
                yield Input(
                    value=str(values["endpoint"]),
                    id="settings-provider-endpoint-value",
                    classes="settings-compact-input",
                    placeholder="http://127.0.0.1:9099/v1",
                )
            yield self._detail_row(
                "Endpoint key",
                self._provider_endpoint_row(str(values["provider"])).removeprefix("Endpoint key: "),
                identifier="settings-provider-endpoint-key",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Streaming", classes="settings-input-label")
                yield Input(
                    value=str(values["streaming"]).lower(),
                    id="settings-streaming-default",
                    classes="settings-compact-input",
                    placeholder="true or false",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Temperature", classes="settings-input-label")
                yield Input(
                    value=str(values["temperature"]),
                    id="settings-temperature-default",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            yield Static("Provider readiness", classes="destination-section")
            yield self._detail_row(
                "Readiness",
                self._provider_readiness_label().removeprefix("Provider readiness: "),
                identifier="settings-provider-readiness",
            )
            yield self._detail_row("Source", resolved.provider_source, identifier="settings-provider-source")
            yield self._detail_row("Model source", resolved.model_source, identifier="settings-model-source")
            yield Static(
                self._provider_endpoint_summary(str(values["provider"])),
                id="settings-provider-endpoint",
            )
            yield Static(
                self._provider_key_status(str(values["provider"])),
                id="settings-provider-key-status",
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
            yield Static(
                "Collapse large pasted chunks over 50 characters.",
                id="settings-console-collapse-large-pastes-label",
            )
            yield Button(
                self._collapse_large_pastes_label(),
                id="settings-console-collapse-large-pastes-toggle",
                tooltip="Toggle compact display for large pasted Console chunks.",
            )
            yield Static(
                "Keeps large paste chunks compact in Console. Disable to keep pasted text literal.",
                id="settings-console-collapse-large-pastes-help",
            )

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
                yield Static("Composer behavior", classes="destination-section")
                yield self._detail_row(
                    "Paste collapse",
                    "chunks over 50 characters display as compact placeholders",
                )
                yield self._detail_row(
                    "Typing rule",
                    "normal typing remains literal and never auto-collapses",
                )
                yield self._detail_row("Current default", self._collapse_large_pastes_label())
                yield self._detail_row("Save target", "[console].collapse_large_pastes")
                yield self._detail_row("Console impact", "composer display only; message payload is preserved")
                yield from self._render_console_behavior_card(compact=False)
        elif category is SettingsCategoryId.APPEARANCE:
            yield Static("Appearance", classes="destination-section settings-column-title")
            with Vertical(id="settings-appearance-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.APPEARANCE)
                yield Static("Visual configuration", classes="destination-section")
                yield self._split_detail_row(self._appearance_theme_summary())
                yield self._detail_row("Surface", "dedicated Appearance customization screen")
                yield self._detail_row("Scope", "theme, density, visual polish")
                yield self._detail_row("Settings role", "routing and status, not the full editor")
                yield self._detail_row("Next action", "open Appearance from the inspector")
        elif category is SettingsCategoryId.STORAGE:
            yield Static("Storage", classes="destination-section settings-column-title")
            with Vertical(id="settings-storage-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.STORAGE)
                yield Static("Local paths", classes="destination-section")
                for path_summary in self._known_storage_paths():
                    yield self._split_detail_row(path_summary)
                yield self._detail_row("Config directory status", self._config_writable_status())
                yield self._detail_row(
                    "Handoff boundary",
                    "database and media paths remain local unless a server handoff is explicit",
                )
                yield self._detail_row("Safety check", "verify write access before changing storage roots")
        elif category is SettingsCategoryId.PRIVACY_SECURITY:
            yield Static("Privacy & Security", classes="destination-section settings-column-title")
            with Vertical(id="settings-privacy-security-card", classes="settings-focus-card"):
                yield self._render_category_state_banner(SettingsCategoryId.PRIVACY_SECURITY)
                yield Static("Privacy posture", classes="destination-section")
                yield self._detail_row(
                    "Secrets",
                    "read from environment/config and hidden from diagnostics",
                )
                yield self._detail_row(
                    "Validation redaction",
                    "API key, token, password, and secret assignments",
                )
                yield self._detail_row("Encryption", "not configured from this Settings slice")
                yield self._detail_row("Secret redaction", "enabled for diagnostics and validation errors")
                yield self._detail_row("Audit posture", "expose status, not raw credentials")
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
                yield Static("Raw TOML bypasses guided validation and should be used only for expert edits.")
                yield Static(
                    self._advanced_validation_status(),
                    id="settings-advanced-config-validation-status",
                    classes="settings-status-row settings-advanced-safety-status",
                )
                yield TextArea(
                    raw_config_text,
                    id="settings-advanced-config-editor",
                )
                with Horizontal(id="settings-advanced-config-actions", classes="settings-action-row"):
                    yield Button(
                        "Validate Raw TOML",
                        id="settings-advanced-validate-config",
                        tooltip="Validate raw TOML before writing it to disk.",
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

    def _render_impact_pane(self) -> ComposeResult:
        summary = self._active_summary()
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
            yield Static("Affects Console", classes="destination-section")
            yield Static("Composer paste display, chat-flow defaults, and user input feedback.")
        elif summary.category is SettingsCategoryId.PROVIDERS_MODELS:
            yield Static("Affects Console and provider-backed generation.", classes="destination-section")
            yield Static(self._provider_readiness_label())
        elif summary.category is SettingsCategoryId.APPEARANCE:
            yield Static("Affects visual presentation only.", classes="destination-section")
            yield Button(
                "Open Appearance",
                id="settings-open-appearance",
                tooltip="Open appearance customization settings.",
            )
        else:
            yield Static("Impact and boundaries", classes="destination-section")
            yield Static(summary.description)
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
                "Open Appearance",
                id="settings-open-appearance",
                tooltip="Open appearance customization settings.",
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
                with Vertical(id="settings-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Preference Detail", classes="destination-section settings-column-title")
                    yield from self._render_detail_pane()
                yield self._column_divider("settings-detail-impact-divider")
                with Vertical(id="settings-impact-pane", classes="destination-workbench-pane ds-inspector"):
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

    def _focused_category_value(self) -> str | None:
        focused = self.app.focused
        if isinstance(focused, Button):
            return self._category_value_from_button(focused)
        return None

    def _category_search_has_focus(self) -> bool:
        focused = self.app.focused
        return isinstance(focused, Input) and focused.id == "settings-category-search"

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

    def _select_category(self, category_value: str, *, restore_focus: bool = False) -> None:
        self.active_category = category_value
        if restore_focus:
            self.call_after_refresh(self._focus_category, category_value)

    @on(Button.Pressed, "#settings-open-appearance")
    def open_appearance_settings(self) -> None:
        self.post_message(NavigateToScreen("customize"))

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
        event.button.label = self._collapse_large_pastes_label()
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-provider-value")
    def handle_provider_value_changed(self, event: Input.Changed) -> None:
        provider = event.value.strip()
        self._stage_provider_value("provider", provider or None)
        try:
            endpoint_input = self.query_one("#settings-provider-endpoint-value", Input)
        except QueryError:
            endpoint_input = None
        if endpoint_input is not None:
            self._syncing_provider_endpoint = True
            try:
                endpoint_input.value = self._provider_endpoint_value(provider)
            finally:
                self._syncing_provider_endpoint = False
        draft = self._provider_draft()
        if draft is not None:
            draft.values.pop("endpoint", None)
            draft.originals.pop("endpoint", None)
            if not draft.is_dirty:
                self._settings_drafts.pop(SettingsCategoryId.PROVIDERS_MODELS, None)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-model-value")
    def handle_model_value_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("model", event.value.strip() or None)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-provider-endpoint-value")
    def handle_provider_endpoint_changed(self, event: Input.Changed) -> None:
        if self._syncing_provider_endpoint:
            self._update_provider_dynamic_widgets()
            return
        self._stage_provider_value("endpoint", event.value.strip())
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-streaming-default")
    def handle_streaming_default_changed(self, event: Input.Changed) -> None:
        self._stage_provider_value("streaming", coerce_bool_setting(event.value, True))
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Input.Changed, "#settings-temperature-default")
    def handle_temperature_default_changed(self, event: Input.Changed) -> None:
        try:
            value = self._normalise_temperature(event.value)
        except ValueError:
            value = event.value
        self._stage_provider_value("temperature", value)
        self._update_provider_dynamic_widgets()
        self._update_draft_status_widgets(SettingsCategoryId.PROVIDERS_MODELS)

    @on(Button.Pressed, "#settings-save-category")
    def handle_save_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_save_category()

    @on(Button.Pressed, "#settings-revert-category")
    def handle_revert_category(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_revert_category()

    @on(Button.Pressed, "#settings-test-provider")
    def handle_test_provider(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_settings_test_category()

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

    @on(TextArea.Changed, "#settings-advanced-config-editor")
    def handle_advanced_config_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        self._update_advanced_validation_status()

    def action_settings_save_category(self) -> None:
        category = self._active_category_id()
        if category not in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            self.app.notify(self._guided_action_message(category), severity="information")
            return
        if category is SettingsCategoryId.PROVIDERS_MODELS:
            try:
                values = self._provider_form_values_from_widgets()
            except ValueError:
                self.app.notify("Temperature must be a number.", severity="error")
                return
            loaded_values = self._provider_loaded_setting_values()
            chat_defaults_keys = {"provider", "model", "streaming", "temperature"}
            provider = str(values.get("provider") or "").strip()
            endpoint = str(values.get("endpoint") or "").strip()
            draft = self._settings_drafts.get(category)
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
            dirty_values = {
                key: value
                for key, value in values.items()
                if key in chat_defaults_keys and loaded_values.get(key) != value
            }
            provider_key = provider_config_key(provider)
            current_provider_endpoint = self._provider_endpoint_value(provider)
            endpoint_dirty = endpoint != current_provider_endpoint
            if endpoint_dirty and not provider_key:
                self._provider_save_result = "Provider is required before saving an endpoint."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify(self._provider_save_result, severity="error")
                return
            if not dirty_values and not endpoint_dirty:
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
            if endpoint_dirty and provider_key:
                endpoint_saved = SettingsConfigAdapter().save_values(
                    f"api_settings.{provider_key}",
                    {endpoint_key: endpoint},
                )
                saved = saved and endpoint_saved
            if saved:
                defaults = self._chat_defaults()
                defaults.update(dirty_values)
                if endpoint_dirty and provider_key:
                    app_config = getattr(self.app_instance, "app_config", None)
                    if not isinstance(app_config, dict):
                        self.app_instance.app_config = {}
                        app_config = self.app_instance.app_config
                    api_settings = app_config.setdefault("api_settings", {})
                    if not isinstance(api_settings, dict):
                        api_settings = {}
                        app_config["api_settings"] = api_settings
                    provider_settings = api_settings.setdefault(provider_key, {})
                    if not isinstance(provider_settings, dict):
                        provider_settings = {}
                        api_settings[provider_key] = provider_settings
                    provider_settings[endpoint_key] = endpoint
                self._settings_drafts.pop(category, None)
                self._provider_save_result = "Provider settings saved."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self._update_provider_dynamic_widgets()
                self._update_draft_status_widgets(category)
                self.app.notify("Provider and model settings saved.", severity="information")
            else:
                self._provider_save_result = "Failed to save provider and model settings."
                self._set_static_text("#settings-provider-save-result", self._provider_save_result)
                self.app.notify("Failed to save provider and model settings.", severity="error")
            return

        draft = self._settings_drafts.get(category)
        if not draft or not draft.is_dirty:
            self.app.notify("No Settings changes to save.", severity="information")
            return

        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            dirty_values = {key: draft.values[key] for key in draft.dirty_keys}
            saved = True
            for key, value in dirty_values.items():
                if not save_setting_to_cli_config("console", key, value):
                    saved = False
            if saved:
                self._console_settings().update(dirty_values)
                self._settings_drafts.pop(category, None)
                self._sync_console_behavior_widgets()
                self.app.notify("Console behavior settings saved.", severity="information")
            else:
                self.app.notify("Failed to save Console behavior settings.", severity="error")
            return

        self.app.notify("This Settings category has no save action yet.", severity="warning")

    def action_settings_revert_category(self) -> None:
        category = self._active_category_id()
        if not self._category_has_unsaved_changes(category):
            self.app.notify("No Settings changes to revert.", severity="information")
            return
        self._settings_drafts.pop(category, None)
        if category is SettingsCategoryId.CONSOLE_BEHAVIOR:
            self._sync_console_behavior_widgets()
        elif category is SettingsCategoryId.PROVIDERS_MODELS:
            values = self._provider_setting_values()
            try:
                self.query_one("#settings-provider-value", Input).value = str(values["provider"])
                self.query_one("#settings-model-value", Input).value = str(values["model"])
                self.query_one("#settings-provider-endpoint-value", Input).value = str(values["endpoint"])
                self.query_one("#settings-streaming-default", Input).value = str(values["streaming"]).lower()
                self.query_one("#settings-temperature-default", Input).value = str(values["temperature"])
            except QueryError:
                pass
            self._provider_save_result = "Provider settings reverted to last loaded values."
            self._set_static_text("#settings-provider-save-result", self._provider_save_result)
            self._update_provider_dynamic_widgets()
            self._update_draft_status_widgets(category)
        else:
            self._update_draft_status_widgets(category)
        self.app.notify("Settings category changes reverted.", severity="information")

    def action_settings_test_category(self) -> None:
        if self._active_category_id() is SettingsCategoryId.PROVIDERS_MODELS:
            self._provider_test_result = self._run_provider_readiness_test()
            self._update_provider_test_result()
            self.app.notify("Provider test finished.", severity="information")
            return
        if self._active_category_id() is SettingsCategoryId.DIAGNOSTICS:
            self._run_diagnostics_validation_and_reload()
            self.app.notify("Diagnostics validation and reload finished.", severity="information")
            return
        self.app.notify("No test action is available for this Settings category yet.", severity="warning")

    def _sync_console_behavior_widgets(self) -> None:
        try:
            self.query_one("#settings-console-collapse-large-pastes-toggle", Button).label = (
                self._collapse_large_pastes_label()
            )
        except QueryError:
            pass
        self._update_console_paste_summary()
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    def on_key(self, event: Key) -> None:
        focused = self.app.focused
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
