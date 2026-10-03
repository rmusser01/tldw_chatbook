"""The Settings ▸ Providers & Models card (TASK-33007.1).

The card's composition, moved verbatim out of ``SettingsScreen`` (DESIGN.md's
One Home Rule). Only composition lives here: the card's handlers, workers and
state stay on the screen, which still receives every event the card's
widgets bubble (moving them is task-1378's remaining scope).

``settings_screen`` imports this module inside ``_render_provider_detail``,
so the Settings route's pre-import payload does not grow (ADR-097); by then
``settings_screen`` is fully loaded, which is what makes the import of its
constants below safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    OptionList,
    Select,
    SelectionList,
    Static,
)

from ...Chat.console_provider_endpoints import (
    first_configured_endpoint,
    safe_endpoint_display,
)
from ...Chat.console_provider_support import MODEL_FIELD_LABELS
from ...Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,
    load_custom_endpoints,
)
from ...Chat.provider_catalog import provider_display_name
from ...Chat.provider_readiness import provider_config_key
from ...config import provider_settings_for_key
from ...LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)
from ..Screens.settings_context_memory import model_context_window_state
from ..Screens.settings_provider_view_model import custom_endpoint_rows
from ..Screens.settings_screen import (
    ANTHROPIC_API_KEY_GUIDANCE_COPY,
    ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY,
    INSTANT_APPLY_BEHAVIOR_COPY,
    MODEL_DISCOVERY_CAPABILITY_WARNING,
    MODEL_DISCOVERY_EMPTY_COPY,
    MODEL_PROFILE_STREAMING_SELECT_OPTIONS,
    PROVIDER_MANUAL_SELECT_VALUE,
    PROVIDER_MODEL_PROFILE_FIELD_KEYS,
    PROVIDER_TEST_GUIDANCE,
    QWENCLOUD_API_MODE_HELP_COPY,
    QWENCLOUD_API_MODE_INVALID_COPY,
    QWENCLOUD_API_MODE_OPTIONS,
    QWENCLOUD_PROVIDER_TABLE_INVALID_COPY,
    ProviderEndpointURLValidator,
    SettingsRegion,
    SettingsURLInput,
    _anthropic_auth_source_options,
    _anthropic_auth_sources,
    _fold_long_tokens,
    _ProviderTestResult,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ..Screens.settings_screen import SettingsScreen


def compose_providers_models_card(screen: SettingsScreen) -> ComposeResult:
    """Compose the Providers & Models section title and card.

    Args:
        screen: The Settings screen that owns the card's state and handlers.

    Yields:
        The section title, then the card container with every row.
    """
    from tldw_chatbook.LLM_Management import (
        snapshot_settings as snapshot_preferences,
    )

    if screen._snapshot_preferences_loaded is None:
        try:
            screen._snapshot_preferences_loaded = (
                snapshot_preferences.load_snapshot_preferences()
            )
        except (ValueError, OSError):
            screen._snapshot_preferences_loaded = None
        screen._snapshot_preferences_unavailable = (
            screen._snapshot_preferences_loaded is None
        )
        screen._snapshot_preferences_raw = (
            (
                screen._snapshot_preferences_loaded.enabled,
                str(screen._snapshot_preferences_loaded.keep_count),
            )
            if screen._snapshot_preferences_loaded
            else (False, "")
        )
        screen.call_after_refresh(screen._update_guided_action_widgets)
    resolved = screen._resolve_provider_model_for_settings()
    values = screen._provider_display_setting_values()
    provider = str(values["provider"])
    context_window_state = model_context_window_state(
        screen._app_config_mapping(), provider, str(values["model"])
    )
    # Qodo #2876: a registry default is edited in Custom endpoints.
    registry_locked = screen._provider_is_registry_id(provider)
    yield Static(
        "Providers & Models", classes="destination-section settings-column-title"
    )
    provider_card = Vertical(
        id="settings-providers-models-card", classes="settings-focus-card"
    )
    provider_card.disabled = screen._vllm_default_recovery() is not None
    with provider_card:
        with Collapsible(
            title="Prompt-cache snapshots",
            collapsed=True,
            id="settings-snapshot-controls",
        ):
            yield Static(
                "Save processed context to reuse later. Restoring does not change your conversations.",
                classes="settings-help-copy",
            )
            yield Static(
                "Enable/disable applies on next launch.",
                id="settings-snapshot-launch-scope",
                classes="settings-help-copy",
            )
            yield Checkbox(
                "Enable snapshots",
                value=screen._snapshot_preferences_raw[0],
                disabled=screen._snapshot_preferences_unavailable,
                id="settings-snapshot-enabled",
            )
            yield Static(
                "Keep count (1–1000, across all models)",
                classes="settings-input-label",
            )
            yield Input(
                screen._snapshot_preferences_raw[1],
                disabled=screen._snapshot_preferences_unavailable,
                id="settings-snapshot-keep",
                type="integer",
            )
            yield Static(
                screen._SNAPSHOT_PREFERENCES_UNAVAILABLE_COPY
                if screen._snapshot_preferences_unavailable
                else "Draft — use category Save / Revert. Enable/disable applies on next launch.",
                id="settings-snapshot-result",
                classes="settings-help-copy",
            )
        # task-189: the Connect block (provider, model, endpoint,
        # credentials, readiness/test) leads; sampling and tuning live in
        # the collapsed "Generation defaults" disclosure below it.
        yield Static(
            "Connect",
            id="settings-provider-connect-title",
            classes="destination-section",
        )
        with Vertical(id="settings-provider-picker-block"):
            yield Static("Provider", classes="settings-input-label")
            yield Input(
                id="settings-provider-search",
                placeholder="Search providers by name or ID",
            )
            picker = OptionList(
                *screen._provider_picker_options(screen._provider_picker_groups()),
                id="settings-provider-picker",
                compact=True,
            )
            # task-16480: compose-time highlight so the configured
            # provider is selected on the very first paint; the
            # post-refresh highlight arrives too early (pre-mount) to
            # serve as the only source.
            screen._apply_provider_picker_highlight(picker)
            yield picker
            yield Static(
                "Choose a provider or enter a provider ID.",
                id="settings-provider-search-status",
                classes="settings-help-copy",
                markup=False,
            )
        with Horizontal(
            classes="settings-input-row settings-provider-manual-hidden"
        ):
            yield Select(
                screen._provider_select_options(),
                value=screen._provider_select_value_for_provider(provider),
                id="settings-provider-value",
                classes="settings-compact-select",
                allow_blank=False,
                compact=True,
            )
        manual_provider_classes = "settings-input-row"
        if (
            screen._provider_select_value_for_provider(provider)
            != PROVIDER_MANUAL_SELECT_VALUE
        ):
            manual_provider_classes += " settings-provider-manual-hidden"
        with Horizontal(
            id="settings-provider-manual-row", classes=manual_provider_classes
        ):
            yield Static("Manual", classes="settings-input-label")
            yield Input(
                value=str(values["provider"])
                if screen._provider_select_value_for_provider(provider)
                == PROVIDER_MANUAL_SELECT_VALUE
                else "",
                id="settings-provider-manual-value",
                classes="settings-compact-input",
                placeholder="Custom provider key",
                disabled=(
                    screen._provider_select_value_for_provider(provider)
                    != PROVIDER_MANUAL_SELECT_VALUE
                ),
            )
        edit_endpoint = Button(
            "Edit this endpoint in Custom endpoints",
            id="settings-provider-edit-custom-endpoint",
            tooltip=(
                "This provider is a named custom endpoint: its URL, "
                "credential, and models are edited there."
            ),
        )
        edit_endpoint.display = registry_locked and (
            screen._provider_registry_entry(provider) is not None
        )
        yield edit_endpoint
        with Horizontal(classes="settings-input-row"):
            yield Static("Model", classes="settings-input-label")
            yield Input(
                value=str(values["model"]),
                id="settings-model-value",
                classes="settings-compact-input",
                placeholder="Model name",
                suggester=screen._model_field_suggester(),
                disabled=registry_locked,
            )
        with Horizontal(classes="settings-input-row"):
            yield Static("Endpoint", classes="settings-input-label")
            yield SettingsURLInput(
                value=str(values["endpoint"]),
                id="settings-provider-endpoint-value",
                classes="settings-compact-input",
                placeholder=screen._provider_endpoint_placeholder(provider),
                validators=[ProviderEndpointURLValidator()],
                validate_on={"blur", "submitted"},
                disabled=registry_locked,
            )
        api_mode_value, api_mode_valid = screen._provider_api_mode_display_value(
            provider
        )
        provider_table_malformed = screen._qwencloud_provider_table_is_malformed(
            provider
        )
        api_mode_row_classes = "settings-input-row settings-select-row"
        if provider_config_key(provider) != "qwencloud":
            api_mode_row_classes += " settings-gated-profile-hidden"
        with Horizontal(
            id="settings-provider-api-mode-row", classes=api_mode_row_classes
        ):
            yield Static("API mode", classes="settings-input-label")
            yield Select(
                QWENCLOUD_API_MODE_OPTIONS,
                value=api_mode_value,
                id="settings-provider-api-mode",
                prompt="Choose Responses or Chat Completions",
                classes=(
                    "settings-compact-select"
                    if api_mode_valid and not provider_table_malformed
                    else "settings-compact-select settings-invalid-input"
                ),
                allow_blank=True,
                compact=True,
                disabled=provider_config_key(provider) != "qwencloud",
            )
        yield Static(
            (
                QWENCLOUD_PROVIDER_TABLE_INVALID_COPY
                if provider_config_key(provider) == "qwencloud"
                and provider_table_malformed
                else (
                    QWENCLOUD_API_MODE_INVALID_COPY
                    if provider_config_key(provider) == "qwencloud"
                    and not api_mode_valid
                    else QWENCLOUD_API_MODE_HELP_COPY
                )
            ),
            id="settings-provider-api-mode-guidance",
            classes=(
                "settings-status-row"
                if provider_config_key(provider) == "qwencloud"
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        yield Static("Credentials", classes="destination-section")
        yield Static(
            screen._provider_credential_status(provider),
            id="settings-provider-credential-status",
            classes="settings-status-row",
        )
        # TASK-34201: Anthropic only -- an API key or the Claude subscription.
        is_anthropic = provider_config_key(provider) == "anthropic"
        subscription_selected = (
            is_anthropic
            and screen._provider_auth_source_value(provider)
            == _anthropic_auth_sources()[1]
        )
        with Horizontal(
            id="settings-provider-auth-source-row",
            classes=(
                "settings-input-row settings-select-row"
                if is_anthropic
                else "settings-input-row settings-select-row settings-gated-profile-hidden"
            ),
        ):
            yield Static("Sign in with", classes="settings-input-label")
            yield Select(
                _anthropic_auth_source_options(),
                value=screen._provider_auth_source_value(provider),
                id="settings-provider-auth-source",
                classes="settings-compact-select",
                allow_blank=False,
                compact=True,
                disabled=not is_anthropic,
            )
        yield Static(
            ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY
            if subscription_selected
            else ANTHROPIC_API_KEY_GUIDANCE_COPY,
            id="settings-provider-auth-source-guidance",
            classes=(
                "settings-status-row"
                if is_anthropic
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        with Horizontal(classes="settings-input-row"):
            yield Static("API key", classes="settings-input-label")
            yield Input(
                value=str(values.get("api_key") or ""),
                id="settings-provider-api-key",
                classes="settings-compact-input",
                placeholder=screen._provider_api_key_placeholder(provider),
                password=True,
                disabled=registry_locked or subscription_selected,
            )
        with Horizontal(classes="settings-input-row"):
            yield Static("", classes="settings-input-label")
            yield Button(
                "Clear saved key",
                id="settings-provider-api-key-clear",
                disabled=subscription_selected or (
                    not screen._provider_saved_api_key_present(provider)
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
                placeholder=screen._provider_credential_placeholder(provider),
                disabled=registry_locked or subscription_selected,
            )
        yield Static(
            "Env vars are safer for shells, shared machines, and CI. This field stores the variable name, not the secret.",
            id="settings-provider-credential-guidance",
            classes="settings-status-row",
        )
        hosted_guidance = screen._hosted_provider_guidance(
            provider, values.get("model")
        )
        yield Static(
            hosted_guidance,
            id="settings-hosted-provider-guidance",
            classes=(
                "settings-status-row"
                if hosted_guidance
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        reconnect = Button(
            "Review restored OpenAI connection",
            id="settings-openai-reconnect-review",
        )
        reconnect.display = provider_config_key(provider) == "openai"
        reconnect.disabled = screen._openai_reconnect_busy
        yield reconnect
        # task-189: the Test affordance closes the first-run Connect job
        # (provider -> model -> endpoint -> credentials -> test) before
        # the informational readiness and discovery sections.
        yield Button(
            "Test Provider",
            id="settings-test-provider",
            tooltip=PROVIDER_TEST_GUIDANCE,
        )
        # TASK-386 (AC#2): the readiness / live-probe explanation must also
        # exist as visible static text -- a hover tooltip is invisible to
        # keyboard users and self-occludes the result line below it.
        yield Static(
            PROVIDER_TEST_GUIDANCE,
            id="settings-test-provider-guidance",
            classes="settings-status-row",
        )
        yield _ProviderTestResult(
            screen._adopt_shared_provider_test_evidence(),
            id="settings-provider-test-result",
            markup=False,
        )
        yield Static(
            screen._provider_save_result,
            id="settings-provider-save-result",
            classes="settings-status-row",
        )
        existing_changes = Static(
            screen._provider_existing_changes_copy(),
            id="settings-provider-existing-changes-summary",
            classes="settings-status-row",
            markup=False,
        )
        existing_changes.display = screen._provider_same_target_has_draft()
        yield existing_changes
        conflict_target = screen._provider_navigation_conflict_target
        conflict = Vertical(id="settings-provider-navigation-conflict")
        conflict.display = conflict_target is not None
        with conflict:
            yield Static(
                screen._provider_navigation_conflict_copy(),
                id="settings-provider-navigation-conflict-summary",
                classes="settings-status-row",
                markup=False,
            )
            yield Button(
                "Review existing changes",
                id="settings-provider-conflict-review",
            )
            yield Button(
                screen._provider_conflict_discard_label(),
                id="settings-provider-conflict-discard",
            )
            yield Button(
                "Return to Chat settings",
                id="settings-provider-conflict-return",
                disabled=screen._provider_return_actions_disabled(),
            )
        continuation = Vertical(id="settings-provider-return-continuation")
        continuation.display = screen._provider_return_outcome is not None
        with continuation:
            yield Static(
                screen._provider_return_continuation_copy(),
                id="settings-provider-return-continuation-status",
                classes="settings-status-row",
                markup=False,
            )
            yield Button(
                "Return to Chat settings",
                id="settings-provider-return",
                variant="primary",
                disabled=screen._provider_return_actions_disabled(),
            )
            yield Button(
                "Stay in Settings",
                id="settings-provider-stay",
            )
        return_without_saving = Button(
            "Return without saving",
            id="settings-provider-return-without-save",
            disabled=screen._provider_return_actions_disabled(),
        )
        return_without_saving.display = screen._provider_can_return_without_saving()
        yield return_without_saving
        yield Static("Provider readiness", classes="destination-section")
        yield screen._detail_row(
            "Readiness",
            screen._provider_readiness_label().removeprefix("Provider readiness: "),
            identifier="settings-provider-readiness",
        )
        yield screen._detail_row(
            "Provider source",
            screen._settings_source_label(resolved.provider_source),
            identifier="settings-provider-source",
        )
        yield screen._detail_row(
            "Model source",
            screen._settings_source_label(resolved.model_source),
            identifier="settings-model-source",
        )
        yield screen._detail_row(
            "Endpoint",
            screen._provider_endpoint_display_value(
                str(values["provider"]), values["endpoint"]
            ),
            identifier="settings-provider-endpoint",
        )
        yield Static(
            screen._provider_key_status(str(values["provider"])),
            id="settings-provider-key-status",
        )
        yield Static("Context capacity", classes="destination-section")
        yield Static(
            screen._provider_model_context_window_status(
                provider,
                str(values["model"]),
                values.get("model_context_window"),
            ),
            id="settings-model-context-window-status",
            classes="settings-status-row",
        )
        with Horizontal(classes="settings-input-row"):
            yield Static("Context window", classes="settings-input-label")
            yield Input(
                value=screen._profile_input_value(
                    values.get("model_context_window", "")
                ),
                id="settings-model-context-window",
                classes="settings-compact-input",
                placeholder="tokens (required when unknown)",
                restrict=r"^[0-9]*$",
                disabled=registry_locked,
            )
        with Horizontal(classes="settings-input-row"):
            yield Static("", classes="settings-input-label")
            yield Button(
                "Reset to detected",
                id="settings-model-context-window-reset",
                disabled=(
                    not context_window_state.has_configured_override
                    or registry_locked
                ),
                tooltip=(
                    "Remove only the configured context-window override and "
                    "return to the detected capability value."
                ),
            )
        yield Static(
            "This is the model's total token capacity, not a conversation "
            "length preference. Repairs update the existing model-capability "
            "registry used by request safety checks.",
            id="settings-model-context-window-help",
            classes="settings-detail-row",
        )
        yield Static("Model discovery", classes="destination-section")
        yield Static(
            screen._model_discovery_status,
            id="settings-model-discovery-status",
            classes="settings-status-row",
        )
        empty_state = Static(
            MODEL_DISCOVERY_EMPTY_COPY,
            id="settings-model-discovery-empty",
            classes="settings-status-row",
        )
        empty_state.display = not screen._model_discovery_models
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
                disabled=not screen._model_discovery_available(
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
                disabled=not screen._model_discovery_models,
                tooltip="Append selected discovered model IDs to the local provider list.",
            )
            yield Button(
                "Clear",
                id="settings-clear-discovered-provider-models",
                disabled=not screen._model_discovery_models,
                tooltip="Clear runtime-discovered models for this provider.",
            )
        yield SelectionList(
            *screen._model_discovery_selection_options(),
            id="settings-discovered-models-list",
            classes="settings-discovered-models-list",
            disabled=not screen._model_discovery_models,
        )
        # ADR-020: [model_catalog] auto-refresh toggles. Values initialize
        # inline from the saved config (the Connect block pattern) and
        # persist immediately on change via the handlers below.
        # task-1341: instant-apply is the labeled exception to the staged
        # default; the bordered group and hint line separate these
        # operational flags visually from the staged Connect fields.
        model_catalog_settings = screen._model_catalog_card_settings()
        # TASK-387: keep the internal decision-record id (ADR-020) out of the
        # user-facing heading; it survives in the code comment above.
        with Vertical(
            id="settings-model-catalog-group",
            classes="settings-instant-apply-group",
        ):
            yield Static("Automatic refresh", classes="destination-section")
            yield Static(
                INSTANT_APPLY_BEHAVIOR_COPY,
                id="settings-model-catalog-instant-hint",
                classes="settings-instant-apply-hint",
            )
            yield Checkbox(
                "Refresh on startup",
                value=model_catalog_settings.auto_refresh_enabled,
                id="settings-model-catalog-auto-refresh",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Refresh after (hours)", classes="settings-input-label"
                )
                yield Input(
                    (
                        str(
                            screen._model_catalog_form_values["model_catalog"][
                                "stale_after_hours"
                            ]
                        )
                        if screen._model_catalog_form_values is not None
                        else f"{model_catalog_settings.stale_after_hours:g}"
                    ),
                    id="settings-model-catalog-stale-hours",
                    type="number",
                    tooltip="0 = refetch every launch.",
                )
            yield Static(
                screen._model_catalog_save_status,
                id="settings-model-catalog-save-status",
                classes="settings-status-row",
                markup=False,
            )
            retry = Button("Retry", id="settings-model-catalog-retry")
            retry.display = screen._model_catalog_save_failed
            yield retry
            for _provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                _provider_key = provider_config_key(_provider)
                _pid = _provider.lower()
                with Horizontal(classes="settings-input-row"):
                    yield Checkbox(
                        f"{provider_display_name(_provider)}: refresh",
                        value=(
                            _provider_key
                            not in model_catalog_settings.auto_refresh_disabled
                        ),
                        id=f"settings-mc-auto-{_pid}",
                    )
                    yield Checkbox(
                        "save to config",
                        value=_provider_key
                        in model_catalog_settings.write_to_config,
                        id=f"settings-mc-write-{_pid}",
                        tooltip=(
                            "Append newly discovered models to config.toml — "
                            "large catalogs like OpenRouter only add newly released "
                            "models after a first baseline."
                        ),
                    )
        # ADR-146 task-7: named-endpoint management (rename / edit /
        # delete-with-reference-guard / slot conversion). Instant-apply
        # like the catalog block above: threaded config writes, one
        # shared status line, no partial-apply states.
        yield from compose_custom_endpoints_section(screen)
        # task-189: sampling and provider-specific tuning live below the
        # Connect block in a collapsed-by-default disclosure.
        model = str(values["model"])
        # TASK-33001.2: a row the provider+model request does not carry is
        # hidden and disabled (never a focus stop), as the gated rows were.
        row_supported = {
            draft_key: screen._model_profile_field_supported(
                provider, draft_key, model
            )
            for draft_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS
        }
        with Collapsible(
            title="Generation defaults",
            collapsed=screen._generation_defaults_collapsed,
            id="settings-generation-defaults",
            disabled=registry_locked,
        ):
            yield Static(
                "Selected model defaults",
                id="settings-selected-model-defaults-title",
                classes="destination-section",
            )
            yield Static(
                "Global fallbacks live under Console Behavior; these values apply only "
                "to the provider+model above.",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["temperature"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_temperature"]
                    ),
                    id="settings-model-profile-temperature",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            with Horizontal(
                id="settings-model-profile-top-p-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_top_p"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["top_p"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_top_p"]),
                    id="settings-model-profile-top-p",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_top_p"],
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(
                id="settings-model-profile-min-p-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_min_p"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["min_p"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_min_p"]),
                    id="settings-model-profile-min-p",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_min_p"],
                    placeholder="optional 0.0 - 1.0",
                )
            with Horizontal(
                id="settings-model-profile-top-k-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_top_k"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["top_k"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_top_k"]),
                    id="settings-model-profile-top-k",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_top_k"],
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["max_tokens"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_max_tokens"]
                    ),
                    id="settings-model-profile-max-tokens",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(
                id="settings-model-profile-seed-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_seed"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["seed"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_seed"]),
                    id="settings-model-profile-seed",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_seed"],
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(
                id="settings-model-profile-presence-penalty-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_presence_penalty"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["presence_penalty"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_presence_penalty"]
                    ),
                    id="settings-model-profile-presence-penalty",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_presence_penalty"],
                    placeholder="-2.0 - 2.0",
                )
            with Horizontal(
                id="settings-model-profile-frequency-penalty-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_frequency_penalty"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["frequency_penalty"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_frequency_penalty"]
                    ),
                    id="settings-model-profile-frequency-penalty",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_frequency_penalty"],
                    placeholder="-2.0 - 2.0",
                )
            # task-189: one summary line replaces per-row "Unavailable
            # for <provider>" placeholders; unsupported rows are hidden.
            support_copy = screen._provider_generation_support_copy(provider, model)
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
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_reasoning_effort"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["reasoning_effort"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_reasoning_effort",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-reasoning-summary-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_reasoning_summary"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["reasoning_summary"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_reasoning_summary",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-verbosity-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_verbosity"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["verbosity"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_verbosity",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-thinking-effort-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_thinking_effort"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["thinking_effort"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_thinking_effort",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-thinking-budget-tokens-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_thinking_budget_tokens"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["thinking_budget_tokens"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._model_profile_input_value(
                        provider,
                        "model_profile_thinking_budget_tokens",
                        model,
                        values["model_profile_thinking_budget_tokens"],
                    ),
                    id="settings-model-profile-thinking-budget-tokens",
                    classes="settings-compact-input",
                    placeholder=screen._model_profile_input_placeholder(
                        provider,
                        "model_profile_thinking_budget_tokens",
                        model,
                    ),
                    restrict=r"^[0-9]*$",
                    disabled=not row_supported[
                        "model_profile_thinking_budget_tokens"
                    ],
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static(
                    MODEL_FIELD_LABELS["streaming"], classes="settings-input-label"
                )
                yield Select(
                    list(MODEL_PROFILE_STREAMING_SELECT_OPTIONS),
                    value=screen._streaming_select_value(
                        values["model_profile_streaming"]
                    ),
                    id="settings-model-profile-streaming",
                    classes="settings-compact-select",
                    allow_blank=True,
                    prompt="Inherit default",
                    compact=True,
                )
        yield Static(
            screen._provider_catalog_summary(),
            id="settings-provider-catalog",
            classes="settings-status-row",
        )
        yield Static(
            screen._provider_catalog_key_policy(),
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
            "Sampling and transport defaults are routed to Console Behavior.",
            id="settings-provider-sampling-route",
            classes="settings-status-row",
        )
        yield screen._detail_row(
            "Endpoint key",
            screen._provider_endpoint_row(str(values["provider"])).removeprefix(
                "Endpoint key: "
            ),
            identifier="settings-provider-endpoint-key",
        )


def compose_custom_endpoints_section(screen: SettingsScreen) -> ComposeResult:
    """Compose the Custom endpoints section (ADR-146 management).

    Args:
        screen: The Settings screen that owns the section's state.

    Yields:
        The section heading, then the region rebuilt on every refresh.
    """
    yield Static("Custom endpoints", classes="destination-section")
    # SettingsRegion (task-15475): a plain Vertical yielded inline has no
    # compose() of its own, so a region-scoped refresh(recompose=True)
    # would wipe it instead of rebuilding it.
    yield SettingsRegion(
        partial(compose_custom_endpoints_children, screen),
        id="settings-custom-endpoints",
        classes="settings-instant-apply-group",
    )



def compose_custom_endpoints_children(screen: SettingsScreen) -> ComposeResult:
    """Yield the Custom endpoints region children (rebuilt per refresh).

    Args:
        screen: The Settings screen that owns the section's state.

    Yields:
        One row and action row per named endpoint, any open rename or edit
        form, the built-in slot rows, and the shared status line.
    """
    config = screen._custom_endpoints_view_config()
    entries = load_custom_endpoints(config)
    rows = custom_endpoint_rows(config)
    if not rows:
        yield Static(
            "No named endpoints yet. Create one from a template in "
            "Console's Chat settings (New endpoint…), or "
            "convert a built-in custom slot below.",
            id="settings-custom-endpoints-empty",
            classes="settings-status-row",
            markup=False,
        )
    for row in rows:
        slug = row.key[len(CUSTOM_ENDPOINT_ID_PREFIX) :]
        entry = entries.get(slug)
        if entry is None:
            continue
        yield Static(
            f"{row.label}: {_fold_long_tokens(row.value)}",
            id=f"settings-cep-row-{slug}",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="settings-action-row"):
            yield Button(
                "Rename",
                id=f"settings-cep-rename-{slug}",
                tooltip="Rename this endpoint. Its id never changes.",
            )
            yield Button(
                "Edit",
                id=f"settings-cep-edit-{slug}",
                tooltip=(
                    "Edit the endpoint URL, credential variable, or "
                    "model list. Existing conversations re-resolve "
                    "on their next send."
                ),
            )
            yield Button(
                "Duplicate",
                id=f"settings-cep-duplicate-{slug}",
                tooltip=(
                    "Create another endpoint from this one via the "
                    "template flow: same family, new URL (a full copy "
                    "stays one row below the preselected starter)."
                ),
            )
            yield Button(
                "Detach references",
                id=f"settings-cep-detach-{slug}",
                classes=(
                    ""
                    if screen._custom_endpoint_detach_slug == slug
                    else "settings-gated-profile-hidden"
                ),
                tooltip=(
                    "Keep each referencing conversation's current "
                    "endpoint as conversation-only, then delete the "
                    "entry."
                ),
            )
            yield Button(
                "Delete",
                id=f"settings-cep-delete-{slug}",
                tooltip=(
                    "Delete this endpoint. Blocked while any "
                    "conversation still references it."
                ),
            )
        if screen._custom_endpoint_rename_slug == slug:
            with Horizontal(classes="settings-input-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    value=entry.display_name,
                    id="settings-cep-rename-value",
                    classes="settings-compact-input",
                    placeholder="Display name",
                )
            with Horizontal(classes="settings-action-row"):
                yield Button("Save name", id="settings-cep-rename-save")
                yield Button("Cancel", id="settings-cep-rename-cancel")
        if screen._custom_endpoint_edit_slug == slug:
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["endpoint"], classes="settings-input-label"
                )
                yield Input(
                    value=entry.base_url,
                    id="settings-cep-edit-url",
                    classes="settings-compact-input",
                    placeholder="http://127.0.0.1:8080",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Credential env var (name)", classes="settings-input-label"
                )
                yield Input(
                    value=entry.api_key_env or "",
                    id="settings-cep-edit-key-env",
                    classes="settings-compact-input",
                    placeholder="API key environment variable (optional)",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Models", classes="settings-input-label")
                yield Input(
                    value=", ".join(entry.models),
                    id="settings-cep-edit-models",
                    classes="settings-compact-input",
                    placeholder="comma-separated model ids",
                )
            yield Static(
                "",
                id="settings-cep-edit-error",
                classes="settings-status-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Save endpoint", id="settings-cep-edit-save")
                yield Button("Cancel", id="settings-cep-edit-cancel")
    for slot_id in ("custom", "custom_2"):
        app_config = screen._app_config_mapping()
        api_settings = app_config.get("api_settings")
        slot_endpoint = first_configured_endpoint(
            provider_settings_for_key(
                api_settings if isinstance(api_settings, Mapping) else {},
                slot_id,
            )
        )
        if not slot_endpoint:
            continue
        yield Static(
            f"{provider_display_name(slot_id)}: "
            f"{safe_endpoint_display(slot_endpoint)} (built-in slot)",
            id=f"settings-cep-slot-{slot_id}",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="settings-action-row"):
            yield Button(
                "Convert to named endpoint",
                id=f"settings-cep-convert-{slot_id}",
                tooltip=(
                    "Create a named endpoint from this slot's "
                    "configured URL and models. The slot itself is "
                    "left untouched."
                ),
            )
    yield Static(
        screen._custom_endpoints_status,
        id="settings-custom-endpoints-status",
        classes="settings-status-row",
        markup=False,
    )
