import asyncio
import gc
import hashlib
import logging
import threading
import weakref
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from loguru import logger as loguru_logger
from textual import events

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.geometry import Region
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.UI.Console_Modules.wiring import build_console_settings_controllers
import tldw_chatbook.UI.Console_Modules.settings_navigation as settings_navigation_module
import tldw_chatbook.UI.Console_Modules.session as session_module
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
import tldw_chatbook.UI.Screens.settings_endpoint_probe as settings_endpoint_probe_module
import tldw_chatbook.Widgets.Console.console_settings_modal as settings_modal_module
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text as _screen_visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_settings_readiness,
    build_console_settings_summary_state,
    build_default_console_session_settings,
    validate_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Chat.local_server_discovery import LocalModelProbeResult
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
)
from tldw_chatbook.config import (
    API_MODELS_BY_PROVIDER,
    DEFAULT_CONFIG_FROM_TOML,
    RuntimeConfigSnapshot,
    save_setting_to_cli_config,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
)
from tldw_chatbook.UI.Screens import provider_model_resolution
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbeOutcome,
    SettingsEndpointProbePurpose,
)
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Console import (
    console_settings_summary as settings_summary_module,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS,
    MODAL_BODY_MIN_HEIGHT,
    MODAL_CONTROL_HEIGHT,
    MODEL_DISCOVER_BUTTON_ID,
    MODEL_DISCOVER_STATUS_ID,
    PROVIDER_CHOICE_NO_EFFECT_SUFFIX,
    ConsoleSettingsInput,
    ConsoleSettingsCredentialRequest,
    ConsoleModelDiscoveryIdentity,
    ConsoleSettingsDraftSnapshot,
    ConsoleSettingsModal,
    ConsoleSettingsResult,
    ConsoleUnverifiedModelDecision,
    _settings_screen_region,
)
from tldw_chatbook.Widgets.Console.console_provider_picker import (
    ConsoleProviderPicker,
)
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    APPLY_BUTTON_ID as SYSTEM_PROMPT_APPLY_BUTTON_ID,
    TEXT_AREA_ID as SYSTEM_PROMPT_TEXT_AREA_ID,
)
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker


def _assert_private_values_absent(
    surface: object,
    private_values: tuple[str, ...],
    *,
    surface_label: str,
) -> None:
    """Fail without echoing a private value or inspected surface."""
    rendered = surface if isinstance(surface, str) else repr(surface)
    if any(value in rendered for value in private_values):
        pytest.fail(
            f"private value leaked through {surface_label}",
            pytrace=False,
        )


def _assert_private_value_matches_opaquely(
    actual: object,
    expected: str,
    *,
    surface_label: str,
) -> None:
    """Compare private text by one-way digest and keep failures value-free."""
    if not isinstance(actual, str):
        pytest.fail(
            f"private value missing from {surface_label}",
            pytrace=False,
        )
    actual_digest = hashlib.sha256(actual.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    if actual_digest != expected_digest:
        pytest.fail(
            f"private value mismatch in {surface_label}",
            pytrace=False,
        )


def _assert_log_capture_is_live_and_private_free(
    *,
    saw_safe_marker: bool,
    saw_private_value: bool,
    surface_label: str,
) -> None:
    """Assert a logger capture without retaining or rendering its messages."""
    if not saw_safe_marker:
        pytest.fail(f"safe marker missing from {surface_label}", pytrace=False)
    if saw_private_value:
        pytest.fail(f"private value leaked through {surface_label}", pytrace=False)


def _assert_public_value_equal(
    actual: object,
    expected: object,
    *,
    surface_label: str,
) -> None:
    """Compare public structure without rendering an inspected value on failure."""
    if actual != expected:
        pytest.fail(f"unexpected public value in {surface_label}", pytrace=False)


def _assert_private_value_is_none(actual: object, *, surface_label: str) -> None:
    """Assert private optional state is empty without rendering it on failure."""
    if actual is not None:
        pytest.fail(f"unexpected private value in {surface_label}", pytrace=False)


def _assert_schema_key_absent(
    surface: object,
    forbidden_key: str,
    *,
    surface_label: str,
) -> None:
    """Inspect nested container keys without rendering any container values."""
    if isinstance(surface, dict):
        if forbidden_key in surface:
            pytest.fail(
                f"forbidden schema field present in {surface_label}",
                pytrace=False,
            )
        for value in surface.values():
            _assert_schema_key_absent(
                value,
                forbidden_key,
                surface_label=surface_label,
            )
    elif isinstance(surface, (list, tuple)):
        for value in surface:
            _assert_schema_key_absent(
                value,
                forbidden_key,
                surface_label=surface_label,
            )


def _bare_console_state_screen(store: ConsoleChatStore) -> ChatScreen:
    """Build the minimal real Console serializer fixture used by privacy tests."""
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    screen._console_runtime_ref = SimpleNamespace(
        chat_store=store,
        set_chat_store=lambda value: setattr(
            screen._console_runtime_ref, "chat_store", value
        ),
    )
    screen._ensure_console_chat_store = lambda: store
    screen._session = SimpleNamespace(_console_visible_draft_session_id=None)
    image_state = SimpleNamespace(
        prune=lambda _ids: None,
        serialize=lambda: {},
    )
    screen._stash_console_pending_attachments = lambda _store: None
    screen._console_visible_draft_session_id = None
    screen._console_composer_or_none = lambda: None
    screen._ensure_console_image_view = lambda: (image_state, SimpleNamespace())
    screen._task_resume_state = TaskResumeState()
    screen._console_library_rag_source_types = ("media", "notes", "conversations")
    screen._pending_console_launch_context = None
    screen._console_evidence_sent_notice = None
    screen._message = SimpleNamespace()
    return screen


def test_private_surface_assertions_keep_failure_messages_value_free() -> None:
    """A failing privacy oracle reports only its fixed surface label."""
    private_value = "TASK30010-helper-private-value"

    with pytest.raises(pytest.fail.Exception) as leak_failure:
        _assert_private_values_absent(
            private_value,
            (private_value,),
            surface_label="helper absence probe",
        )
    with pytest.raises(pytest.fail.Exception) as mismatch_failure:
        _assert_private_value_matches_opaquely(
            "TASK30010-helper-wrong-value",
            private_value,
            surface_label="helper digest probe",
        )
    with pytest.raises(pytest.fail.Exception) as log_failure:
        _assert_log_capture_is_live_and_private_free(
            saw_safe_marker=True,
            saw_private_value=True,
            surface_label="helper logger probe",
        )
    with pytest.raises(pytest.fail.Exception) as schema_failure:
        _assert_schema_key_absent(
            {"nested": {"api_key": private_value}},
            "api_key",
            surface_label="helper schema probe",
        )

    failure_artifacts = (
        str(leak_failure.value),
        str(mismatch_failure.value),
        str(log_failure.value),
        str(schema_failure.value),
    )
    for failure_artifact in failure_artifacts:
        _assert_private_values_absent(
            failure_artifact,
            (private_value,),
            surface_label="privacy-helper failure artifact",
        )
    _assert_public_value_equal(
        failure_artifacts[0],
        "private value leaked through helper absence probe",
        surface_label="absence-helper failure copy",
    )
    _assert_public_value_equal(
        failure_artifacts[1],
        "private value mismatch in helper digest probe",
        surface_label="digest-helper failure copy",
    )
    _assert_public_value_equal(
        failure_artifacts[2],
        "private value leaked through helper logger probe",
        surface_label="logger-helper failure copy",
    )
    _assert_public_value_equal(
        failure_artifacts[3],
        "forbidden schema field present in helper schema probe",
        surface_label="schema-helper failure copy",
    )


class SummaryHarness(ConsolidatedCSSApp):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self, state: ConsoleSettingsSummaryState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield ConsoleSettingsSummary(self.state)


def test_console_settings_screen_region_prefers_absolute_region() -> None:
    absolute_region = Region(10, 20, 30, 1)
    widget = SimpleNamespace(
        region=Region(1, 2, 30, 1),
        screen_region=absolute_region,
    )

    assert _settings_screen_region(widget) == absolute_region


def test_console_settings_screen_region_falls_back_to_mounted_region() -> None:
    mounted_region = Region(3, 4, 30, 1)
    widget = SimpleNamespace(region=mounted_region)

    assert _settings_screen_region(widget) == mounted_region


class ModalHarness(ConsolidatedCSSApp):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
                "openai": {"api_key": "test-key"},
            },
        }
        self.saved_settings: ConsoleSessionSettings | None = None
        self.saved_result: ConsoleSettingsResult | None = None

    def capture_saved_settings(
        self,
        result: ConsoleSettingsCommittedSubmission | ConsoleSettingsResult | None,
    ) -> None:
        if isinstance(result, ConsoleSettingsCommittedSubmission):
            submission = result.submission
            result = ConsoleSettingsResult(
                settings=result.live_commit.settings,
                user_display_name_override=submission.user_display_name_override,
                context_policy_overrides=result.live_commit.context_policy_overrides,
            )
        self.saved_result = result
        self.saved_settings = result.settings if result is not None else None


def _typed_ready_unverified_readiness() -> ConsoleSettingsReadiness:
    return ConsoleSettingsReadiness(
        label="LEGACY LABEL MUST NOT DRIVE UI",
        detail="PRIVATE legacy detail https://secret.invalid/token",
        native_send_supported=True,
        operability="ready_to_send",
        provider_display_name="OpenAI",
        configuration="configured",
        credential="present_unverified",
        credential_source="stored",
        endpoint="not_tested",
        model="unconfirmed",
        generation="not_tested",
    )


class StyledModalHarness(ModalHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class StyledConsoleHarness(ConsoleHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


def test_suspended_draft_round_trips_raw_modal_values_and_sensitive_session_fields() -> None:
    """Suspension keeps the modal's unvalidated values only in Console state."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
            system_prompt="private system text",
            pinned_prefill="private prefill text",
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(
            custom_budget_tokens=2048,
        ),
        raw_values={
            "console-settings-provider": "llama_cpp",
            "console-settings-model-picker": "model-a",
            "console-settings-base-url": "http://127.0.0.1:9099",
            "console-settings-temperature": "0.7.2",
            "console-settings-user-display-name": "Ada",
            "console-context-custom-budget": "2048",
        },
        provider_model_drafts={"llama_cpp": "model-a", "openai": "gpt-5"},
        provider_base_url_drafts={"llama_cpp": "http://127.0.0.1:9099"},
        active_view="context",
        scroll_anchor=17,
        focus_control_id="console-context-custom-budget",
        disclosure_state={"advanced_generation": True, "connection_details": False},
    )

    mapping = snapshot.to_mapping()
    restored = ConsoleSettingsDraftSnapshot.from_mapping(mapping)

    assert restored is not None
    assert restored.raw_values["console-settings-temperature"] == "0.7.2"
    assert restored.settings.system_prompt == "private system text"
    assert restored.settings.pinned_prefill == "private prefill text"
    assert restored.context_policy_overrides.custom_budget_tokens == 2048
    assert restored.provider_model_drafts == {
        "llama_cpp": "model-a",
        "openai": "gpt-5",
    }
    assert restored.provider_base_url_drafts == {
        "llama_cpp": "http://127.0.0.1:9099"
    }
    assert restored.active_view == "context"
    assert restored.scroll_anchor == 17
    assert restored.focus_control_id == "console-context-custom-budget"
    assert restored.disclosure_state == {
        "advanced_generation": True,
        "connection_details": False,
    }
    for private_value in (
        "private system text",
        "private prefill text",
        "127.0.0.1",
        "model-a",
    ):
        assert private_value not in repr(restored)

    mapping["raw_values"]["console-settings-temperature"] = "changed"  # type: ignore[index]
    mapping["provider_model_drafts"]["openai"] = "changed"  # type: ignore[index]
    assert restored.raw_values["console-settings-temperature"] == "0.7.2"
    assert restored.provider_model_drafts["openai"] == "gpt-5"


def test_suspended_draft_allows_incomplete_connection_and_multiline_private_text() -> None:
    """A suspended modal keeps editable blanks and private formatting verbatim."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="",
            base_url="",
            system_prompt="system line one\n\tsystem line two",
            pinned_prefill="prefill line one\n\tprefill line two",
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "llama_cpp",
            "console-settings-model-picker": "",
            "console-settings-base-url": "",
        },
        provider_model_drafts={"llama_cpp": ""},
        provider_base_url_drafts={"llama_cpp": ""},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    restored = ConsoleSettingsDraftSnapshot.from_mapping(snapshot.to_mapping())

    assert restored is not None
    assert restored.settings.model == ""
    assert restored.settings.base_url == ""
    assert restored.raw_values["console-settings-model-picker"] == ""
    assert restored.raw_values["console-settings-base-url"] == ""
    assert restored.provider_model_drafts == {"llama_cpp": ""}
    assert restored.provider_base_url_drafts == {"llama_cpp": ""}
    assert restored.settings.system_prompt == "system line one\n\tsystem line two"
    assert restored.settings.pinned_prefill == "prefill line one\n\tprefill line two"


def test_suspended_draft_allows_first_run_cloud_model_to_remain_unselected() -> None:
    """A private draft does not need to satisfy live send readiness."""
    mapping = _minimal_suspended_draft_mapping()
    mapping["settings"]["model"] = None  # type: ignore[index]
    mapping["settings"]["base_url"] = ""  # type: ignore[index]
    mapping["provider_model_drafts"] = {"openai": None}

    restored = ConsoleSettingsDraftSnapshot.from_mapping(mapping)

    assert restored is not None
    assert restored.settings.provider == "openai"
    assert restored.settings.model is None
    assert restored.settings.base_url == ""


@pytest.mark.parametrize(
    "invalid_url",
    ["not-a-url", "ftp://example.test", "https://bad host.test"],
)
def test_suspended_draft_rejects_malformed_nonblank_semantic_endpoint(
    invalid_url: str,
) -> None:
    """Semantic endpoint validation cannot depend on configured provider state."""
    mapping = _minimal_suspended_draft_mapping()
    mapping["settings"]["base_url"] = invalid_url  # type: ignore[index]

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


def _minimal_suspended_draft_mapping() -> dict[str, object]:
    """Return one detached valid snapshot mapping for fail-closed mutations."""
    return ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    ).to_mapping()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", -0.001),
        ("temperature", 2.001),
        ("top_p", float("nan")),
        ("top_p", -0.001),
        ("top_p", 1.001),
        ("min_p", float("inf")),
        ("min_p", -0.001),
        ("min_p", 1.001),
        ("top_k", -1),
        ("max_tokens", 0),
        ("seed", -1),
        ("presence_penalty", float("-inf")),
        ("presence_penalty", -2.001),
        ("presence_penalty", 2.001),
        ("frequency_penalty", float("inf")),
        ("frequency_penalty", -2.001),
        ("frequency_penalty", 2.001),
        ("thinking_budget_tokens", 1023),
        ("reasoning_effort", "arbitrary"),
        ("reasoning_summary", "arbitrary"),
        ("verbosity", "arbitrary"),
        ("thinking_effort", "arbitrary"),
        ("source", "arbitrary"),
    ],
)
def test_suspended_draft_rejects_out_of_domain_semantic_settings(
    field_name: str,
    invalid_value: object,
) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping["settings"][field_name] = invalid_value  # type: ignore[index]

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


@pytest.mark.parametrize(
    ("field_name", "boundary_value"),
    [
        ("temperature", 0.0),
        ("temperature", 2.0),
        ("top_p", 0.0),
        ("top_p", 1.0),
        ("min_p", 0.0),
        ("min_p", 1.0),
        ("top_k", 0),
        ("max_tokens", 1),
        ("seed", 0),
        ("presence_penalty", -2.0),
        ("presence_penalty", 2.0),
        ("frequency_penalty", -2.0),
        ("frequency_penalty", 2.0),
        ("thinking_budget_tokens", 1024),
        ("source", "derived"),
        ("source", "user"),
    ],
)
def test_suspended_draft_accepts_canonical_semantic_boundaries(
    field_name: str,
    boundary_value: object,
) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping["settings"][field_name] = boundary_value  # type: ignore[index]

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is not None


@pytest.mark.parametrize(
    ("field_name", "bool_value"),
    [
        ("temperature", True),
        ("min_p", False),
        ("top_k", True),
        ("max_tokens", False),
        ("seed", True),
        ("thinking_budget_tokens", False),
    ],
)
def test_suspended_draft_keeps_numeric_bool_rejection_exact(
    field_name: str,
    bool_value: bool,
) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping["settings"][field_name] = bool_value  # type: ignore[index]

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


@pytest.mark.parametrize("mapping_name", ["provider_model_drafts", "provider_base_url_drafts"])
def test_suspended_draft_caps_provider_draft_cardinality(mapping_name: str) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping[mapping_name] = {f"provider_{index}": "draft" for index in range(257)}

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


@pytest.mark.parametrize("mapping_name", ["provider_model_drafts", "provider_base_url_drafts"])
def test_suspended_draft_accepts_provider_draft_cardinality_boundary(
    mapping_name: str,
) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping[mapping_name] = {f"provider_{index}": "draft" for index in range(256)}

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is not None


@pytest.mark.parametrize("scroll_anchor", [-1, 1_000_001, True])
def test_suspended_draft_rejects_invalid_or_unbounded_scroll_anchor(
    scroll_anchor: object,
) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping["scroll_anchor"] = scroll_anchor

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


@pytest.mark.parametrize("scroll_anchor", [0, 1_000_000])
def test_suspended_draft_accepts_bounded_scroll_anchor(scroll_anchor: int) -> None:
    mapping = _minimal_suspended_draft_mapping()
    mapping["scroll_anchor"] = scroll_anchor

    assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is not None


def test_suspended_draft_rejects_equality_impersonating_view_and_focus() -> None:
    """Schema choices are exact strings, not arbitrary equality-compatible objects."""

    class Impersonator:
        def __init__(self, value: str) -> None:
            self.value = value

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return other == self.value

    common = dict(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        scroll_anchor=0,
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    with pytest.raises(ValueError):
        ConsoleSettingsDraftSnapshot(
            **common,
            active_view=Impersonator("model"),  # type: ignore[arg-type]
            focus_control_id="console-settings-model-picker",
        )
    with pytest.raises(ValueError):
        ConsoleSettingsDraftSnapshot(
            **common,
            active_view="model",
            focus_control_id=Impersonator("console-settings-model-picker"),  # type: ignore[arg-type]
        )


def test_credential_request_rejects_values_the_navigation_target_would_reject() -> None:
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    with pytest.raises(ValueError):
        ConsoleSettingsCredentialRequest(
            snapshot=snapshot,
            provider="openai\nunsafe",
            model="gpt-5",
        )
    with pytest.raises(ValueError):
        ConsoleSettingsCredentialRequest(
            snapshot=snapshot,
            provider="openai",
            model="gpt-5\nunsafe",
        )
    with pytest.raises(ValueError):
        ConsoleSettingsCredentialRequest(
            snapshot=snapshot,
            provider="openai",
            model=" gpt-5 ",
        )
    with pytest.raises(ValueError):
        ConsoleSettingsCredentialRequest(
            snapshot=snapshot,
            provider="openai",
            model="",
        )


def test_credential_request_stages_only_a_secret_free_return_and_navigation_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A credential deep link keeps Console-owned draft content off the route."""
    system_prompt = "TASK30010-private-system-prompt"
    pinned_prefill = "TASK30010-private-pinned-prefill"
    suspended_settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        system_prompt=system_prompt,
        pinned_prefill=pinned_prefill,
    )
    committed_settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        system_prompt="Committed session prompt",
        pinned_prefill=None,
    )
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=suspended_settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "llama_cpp",
            "console-settings-model-picker": "model-a",
            "console-settings-base-url": "http://127.0.0.1:9099",
            "console-settings-temperature": "0.7.2",
        },
        provider_model_drafts={"llama_cpp": "model-a"},
        provider_base_url_drafts={"llama_cpp": "http://127.0.0.1:9099"},
        active_view="model",
        scroll_anchor=3,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": True},
    )
    request = ConsoleSettingsCredentialRequest(
        snapshot=snapshot,
        provider="llama_cpp",
        model="model-a",
    )
    store = ConsoleChatStore()
    session = store.create_session(settings=committed_settings)
    messages = []
    screen = _bare_console_state_screen(store)
    screen.app_instance = SimpleNamespace(
        pending_handoffs=chat_screen_module.PendingHandoffStore(),
    )
    screen.post_message = messages.append

    from loguru import logger as loguru_logger

    caplog.set_level(logging.DEBUG)
    python_marker = "TASK30010-python-capture-live"
    loguru_marker = "TASK30010-loguru-capture-live"
    private_values = (system_prompt, pinned_prefill)
    loguru_messages: list[str] = []
    python_capture = (False, False)
    loguru_capture = (False, False)
    sink_id = loguru_logger.add(loguru_messages.append, level="DEBUG")

    try:
        screen._settings_navigation._stage_console_settings_credential_request(
            request,
            session_id=session.id,
        )
        serialized_console = screen._serialize_native_console_state()
        logging.getLogger(__name__).debug(python_marker)
        loguru_logger.debug(loguru_marker)
    finally:
        loguru_logger.remove(sink_id)
        python_log_text = caplog.text
        loguru_log_text = "".join(loguru_messages)
        python_capture = (
            python_marker in python_log_text,
            any(value in python_log_text for value in private_values),
        )
        loguru_capture = (
            loguru_marker in loguru_log_text,
            any(value in loguru_log_text for value in private_values),
        )
        caplog.clear()
        loguru_messages.clear()

    retained_snapshot = screen._suspended_conversation_settings
    if not isinstance(retained_snapshot, ConsoleSettingsDraftSnapshot):
        pytest.fail("Console suspended snapshot was not retained", pytrace=False)
    _assert_private_value_matches_opaquely(
        retained_snapshot.settings.system_prompt,
        system_prompt,
        surface_label="Console-owned suspended snapshot prompt",
    )
    _assert_private_value_matches_opaquely(
        retained_snapshot.settings.pinned_prefill,
        pinned_prefill,
        surface_label="Console-owned suspended snapshot prefill",
    )
    committed = store.session_settings(session.id)
    _assert_private_values_absent(
        committed,
        private_values,
        surface_label="committed Console session",
    )
    _assert_private_value_matches_opaquely(
        committed.system_prompt,
        "Committed session prompt",
        surface_label="committed Console session prompt",
    )
    _assert_private_value_is_none(
        committed.pinned_prefill,
        surface_label="committed Console session prefill",
    )
    assert store.session_settings_revision(session.id) == 0
    assert len(messages) == 1
    navigation = messages[0]
    settings_navigation_context = navigation.screen_context
    claim = screen.app_instance.pending_handoffs.claim(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN
    )
    assert claim is not None
    return_intent = claim.value
    assert return_intent.session_id == session.id
    assert return_intent.settings_revision == 0
    assert return_intent.active_view == "model"
    assert return_intent.focus_control_id == "console-settings-model-picker"
    _assert_private_values_absent(
        return_intent,
        private_values,
        surface_label="return handoff value",
    )
    _assert_private_values_absent(
        return_intent.to_context(),
        private_values,
        surface_label="return handoff context",
    )
    _assert_private_values_absent(
        settings_navigation_context,
        (*private_values, "http://127.0.0.1:9099"),
        surface_label="Settings navigation context",
    )
    _assert_public_value_equal(
        settings_navigation_context,
        {
            "category": "providers-models",
            "provider": "llama_cpp",
            "model": "model-a",
            "field": "api_key",
            "return_revision": claim.revision,
        },
        surface_label="Settings navigation context shape",
    )
    assert serialized_console is not None
    suspended = serialized_console["suspended_conversation_settings"]
    restored_snapshot = ConsoleSettingsDraftSnapshot.from_mapping(suspended)
    assert restored_snapshot is not None
    _assert_private_value_matches_opaquely(
        restored_snapshot.settings.system_prompt,
        system_prompt,
        surface_label="serialized suspended snapshot prompt",
    )
    _assert_private_value_matches_opaquely(
        restored_snapshot.settings.pinned_prefill,
        pinned_prefill,
        surface_label="serialized suspended snapshot prefill",
    )
    _assert_schema_key_absent(
        serialized_console,
        "api_key",
        surface_label="serialized Console snapshot",
    )
    transfer_coordinates = {
        key: value
        for key, value in serialized_console.items()
        if key != "suspended_conversation_settings"
    }
    _assert_private_values_absent(
        transfer_coordinates,
        private_values,
        surface_label="non-snapshot Console state",
    )
    _assert_log_capture_is_live_and_private_free(
        saw_safe_marker=python_capture[0],
        saw_private_value=python_capture[1],
        surface_label="stdlib logging capture",
    )
    _assert_log_capture_is_live_and_private_free(
        saw_safe_marker=loguru_capture[0],
        saw_private_value=loguru_capture[1],
        surface_label="loguru capture",
    )


def test_credential_route_staging_is_atomic_when_navigation_target_is_invalid(
    monkeypatch,
) -> None:
    """A target construction failure must not strand a return handoff."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    request = ConsoleSettingsCredentialRequest(snapshot, "openai", "gpt-5")
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    handoffs = chat_screen_module.PendingHandoffStore()
    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._ensure_console_chat_store = lambda: store
    screen.post_message = lambda _message: None
    monkeypatch.setattr(
        settings_navigation_module,
        "ProviderSettingsNavigationTarget",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("target rejected")),
    )

    with pytest.raises(ValueError, match="target rejected"):
        screen._settings_navigation._stage_console_settings_credential_request(
            request, session_id=session.id
        )

    assert screen._suspended_conversation_settings is None
    assert handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


def test_credential_route_navigation_rejection_clears_exact_staged_return_slot() -> (
    None
):
    """The source repair callback leaves no orphan when the Console guard vetoes."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    request = ConsoleSettingsCredentialRequest(snapshot, "openai", "gpt-5")
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    handoffs = chat_screen_module.PendingHandoffStore()
    messages: list[object] = []
    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._ensure_console_chat_store = lambda: store
    screen.post_message = messages.append

    screen._settings_navigation._stage_console_settings_credential_request(
        request, session_id=session.id
    )
    messages[0].report_completion(False)

    assert screen._suspended_conversation_settings == snapshot
    assert handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


def test_credential_route_rejected_delivery_repairs_the_exact_slot() -> None:
    """Textual refusing delivery uses the same source-safe failure settlement."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    request = ConsoleSettingsCredentialRequest(snapshot, "openai", "gpt-5")
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    handoffs = chat_screen_module.PendingHandoffStore()
    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._ensure_console_chat_store = lambda: store
    screen.post_message = lambda _message: False

    screen._settings_navigation._stage_console_settings_credential_request(
        request, session_id=session.id
    )

    assert screen._suspended_conversation_settings == snapshot
    assert handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


def test_credential_route_stale_identical_snapshot_cannot_reopen_newer_request(
    monkeypatch,
) -> None:
    """An old rejection cannot repair a structurally identical newer suspension."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    request = ConsoleSettingsCredentialRequest(snapshot, "openai", "gpt-5")
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    handoffs = chat_screen_module.PendingHandoffStore()
    messages: list[object] = []
    scheduled: list[object] = []
    fake_app = SimpleNamespace(screen_stack=(screen,))
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._ensure_console_chat_store = lambda: store
    screen.post_message = messages.append
    screen.run_worker = lambda worker, **_kwargs: scheduled.append(worker)

    screen._settings_navigation._stage_console_settings_credential_request(
        request, session_id=session.id
    )
    first_navigation = messages[-1]
    screen._settings_navigation._stage_console_settings_credential_request(
        request, session_id=session.id
    )
    second_navigation = messages[-1]

    first_navigation.report_completion(False)
    assert scheduled == []
    assert screen._suspended_conversation_settings_token == 2

    second_navigation.report_completion(False)
    assert len(scheduled) == 1
    scheduled[0].close()


@pytest.mark.asyncio
async def test_failed_source_reopen_retains_suspended_snapshot_and_token(
    monkeypatch,
) -> None:
    """A modal-push failure leaves the exact private draft available for retry."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    fake_app = SimpleNamespace(screen_stack=(screen,))
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen._ensure_console_chat_store = lambda: store
    screen._suspended_conversation_settings = snapshot
    screen._suspended_conversation_settings_token = 7

    async def failed_open(**_kwargs):
        return False

    screen._settings_navigation._open_console_settings = failed_open
    await screen._settings_navigation._reopen_suspended_console_settings(
        7,
        session_id=session.id,
        settings_revision=0,
    )

    assert screen._suspended_conversation_settings == snapshot
    assert screen._suspended_conversation_settings_token == 7


@pytest.mark.asyncio
async def test_cancelled_source_reopen_retains_suspended_snapshot_and_token(
    monkeypatch,
) -> None:
    """Cancelling the actual mount leaves the exact private retry state owned."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    fake_app = SimpleNamespace(screen_stack=(screen,))
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen._ensure_console_chat_store = lambda: store
    screen._suspended_conversation_settings = snapshot
    screen._suspended_conversation_settings_token = 11

    async def cancelled_open(**_kwargs):
        raise asyncio.CancelledError

    screen._settings_navigation._open_console_settings = cancelled_open
    with pytest.raises(asyncio.CancelledError):
        await screen._settings_navigation._reopen_suspended_console_settings(
            11,
            session_id=session.id,
            settings_revision=0,
        )

    assert screen._suspended_conversation_settings is snapshot
    assert screen._suspended_conversation_settings_token == 11


def _install_open_settings_dependencies(screen: ChatScreen) -> None:
    """Install the current ChatScreen settings seams on a constructor-free double."""

    async def effective_thinking_history_policy_for_session(_session_id):
        return "auto"

    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state_for=lambda _session_id: SimpleNamespace(is_send_allowed=True),
        effective_thinking_history_policy_for_session=(
            effective_thinking_history_policy_for_session
        ),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
        rebase_console_settings_draft=lambda draft, **_kwargs: draft,
    )
    screen._console_settings_context_estimate_for_session = (
        lambda *_args, **_kwargs: ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._console_context_control_state_for_session = (
        lambda *_args, **_kwargs: None
    )
    if not hasattr(screen, "app_instance"):
        screen.app_instance = SimpleNamespace()


@pytest.mark.asyncio
async def test_covered_cancelled_source_reopen_transfers_exact_draft_to_modal(
    monkeypatch,
) -> None:
    """A covered modal, not its source slot, owns the draft after cancellation."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    stack: list[object] = [screen]
    pushed: list[ConsoleSettingsModal] = []
    newer_overlay = object()
    mount_wait_started = asyncio.Event()

    class PendingMount:
        def __await__(self):
            async def wait_for_cancellation():
                mount_wait_started.set()
                await asyncio.Future()

            return wait_for_cancellation().__await__()

    def push_screen(modal, callback):
        stack.append(modal)
        pushed.append(modal)
        return PendingMount()

    fake_app = SimpleNamespace(screen_stack=stack, push_screen=push_screen)
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen.app_instance = SimpleNamespace()
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: snapshot.settings
    )
    screen._ensure_console_chat_store = lambda: store

    async def effective_thinking_history_policy_for_session(_session_id):
        return "auto"

    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state_for=lambda _session_id: SimpleNamespace(is_send_allowed=True),
        effective_thinking_history_policy_for_session=(
            effective_thinking_history_policy_for_session
        ),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
        rebase_console_settings_draft=lambda *args, **kwargs: args[0],
    )
    screen._console_settings_context_estimate_for_session = lambda *args, **kwargs: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._console_context_control_state_for_session = lambda *args, **kwargs: None
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(*_args, **_kwargs):
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = provider_models
    screen._suspended_conversation_settings = snapshot
    screen._suspended_conversation_settings_token = 13
    assert store.active_session_id == session.id
    assert store.session_settings_revision(session.id) == 0
    assert screen._owns_console_screen_stack()

    reopen_task = asyncio.create_task(
        screen._settings_navigation._reopen_suspended_console_settings(
            13,
            session_id=session.id,
            settings_revision=0,
        )
    )
    await asyncio.sleep(0)
    assert not reopen_task.done(), repr(reopen_task.exception())
    await asyncio.wait_for(mount_wait_started.wait(), timeout=1)
    stack.append(newer_overlay)
    reopen_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reopen_task

    assert len(pushed) == 1
    assert stack == [screen, pushed[0], newer_overlay]
    assert screen._suspended_conversation_settings is None
    assert screen._suspended_conversation_settings_token is None


@pytest.mark.asyncio
async def test_source_reopen_revalidates_exact_owner_after_model_resolution(
    monkeypatch,
) -> None:
    """A later top screen wins while the reopen awaits catalog resolution."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    store = ConsoleChatStore()
    session = store.create_session(settings=snapshot.settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    stack: list[object] = [screen]
    pushed: list[object] = []
    fake_app = SimpleNamespace(
        screen_stack=stack,
        push_screen=lambda modal, **_kwargs: pushed.append(modal),
    )
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen._suspended_conversation_settings = snapshot
    screen._suspended_conversation_settings_token = 7
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    resolution_started = asyncio.Event()
    release_resolution = asyncio.Event()

    async def delayed_provider_models(*_args, **_kwargs):
        resolution_started.set()
        await release_resolution.wait()
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = delayed_provider_models

    reopen_task = asyncio.create_task(
        screen._settings_navigation._reopen_suspended_console_settings(
            7,
            session_id=session.id,
            settings_revision=0,
        )
    )
    await resolution_started.wait()
    later_top_screen = object()
    stack.append(later_top_screen)
    release_resolution.set()
    await reopen_task

    assert pushed == []
    assert stack[-1] is later_top_screen
    assert screen._suspended_conversation_settings is snapshot
    assert screen._suspended_conversation_settings_token == 7


def test_resident_console_screen_is_not_active_stack_owner(monkeypatch) -> None:
    """A hidden resident source must not reopen over the actual top screen."""
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    top_screen = object()
    fake_app = SimpleNamespace(screen_stack=(screen, top_screen))
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))

    assert ChatScreen._owns_console_screen_stack(screen) is False


@pytest.mark.asyncio
async def test_open_console_settings_real_callback_stages_typed_credential_route(
    monkeypatch,
) -> None:
    """Production modal opening supplies the callback that stages its typed result."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    store = ConsoleChatStore()
    store.create_session(settings=settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    staged_modal: list[object] = []
    posted: list[object] = []
    mount_awaited = False

    class MountResult:
        def __await__(self):
            async def wait_for_mount():
                nonlocal mount_awaited
                mount_awaited = True

            return wait_for_mount().__await__()

    def push_screen(modal, callback):
        staged_modal.extend((modal, callback))
        return MountResult()

    fake_app = SimpleNamespace(push_screen=push_screen)
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen.app_instance = SimpleNamespace(
        pending_handoffs=chat_screen_module.PendingHandoffStore()
    )
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: settings
    )
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(*_args, **_kwargs):
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = provider_models
    screen.post_message = posted.append

    assert await screen._settings_navigation._open_console_settings() is True
    assert mount_awaited is True
    callback = staged_modal[1]
    callback(ConsoleSettingsCredentialRequest(snapshot, "openai", "gpt-5"))

    assert len(posted) == 1
    assert posted[0].screen_context["provider"] == "openai"


@pytest.mark.asyncio
async def test_open_console_settings_returns_false_when_mount_awaitable_fails(
    monkeypatch,
) -> None:
    """A synchronously pushed modal is not open until AwaitMount succeeds."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    store = ConsoleChatStore()
    store.create_session(settings=settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)

    class FailedMount:
        def __await__(self):
            async def fail_mount():
                raise RuntimeError("modal mount failed")

            return fail_mount().__await__()

    fake_app = SimpleNamespace(push_screen=lambda *_args, **_kwargs: FailedMount())
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: settings
    )
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(*_args, **_kwargs):
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = provider_models

    assert await screen._settings_navigation._open_console_settings() is False


@pytest.mark.asyncio
async def test_open_console_settings_unwinds_exact_modal_after_mutating_failed_mount(
    monkeypatch,
) -> None:
    """An AwaitMount failure cannot leave the modal sharing draft ownership."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    store = ConsoleChatStore()
    store.create_session(settings=settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    stack: list[object] = [screen]
    popped: list[object] = []
    callbacks: list[object] = []

    class FailedMount:
        def __await__(self):
            async def fail_mount():
                raise RuntimeError("modal mount failed after append")

            return fail_mount().__await__()

    class CompletedPop:
        def __await__(self):
            async def complete_pop():
                return None

            return complete_pop().__await__()

    def push_screen(modal, callback):
        stack.append(modal)
        callbacks.append(callback)
        return FailedMount()

    def pop_screen():
        popped.append(stack.pop())
        return CompletedPop()

    fake_app = SimpleNamespace(
        screen_stack=stack,
        push_screen=push_screen,
        pop_screen=pop_screen,
    )
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: settings
    )
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(*_args, **_kwargs):
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = provider_models

    assert await screen._settings_navigation._open_console_settings() is False
    assert len(popped) == 1
    assert isinstance(popped[0], ConsoleSettingsModal)
    assert stack == [screen]
    assert len(callbacks) == 1


@pytest.mark.asyncio
async def test_open_console_settings_propagates_mount_cancellation(monkeypatch) -> None:
    """AwaitMount cancellation remains visible to the owning Textual worker."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    store = ConsoleChatStore()
    store.create_session(settings=settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)

    class CancelledMount:
        def __await__(self):
            async def cancel_mount():
                raise asyncio.CancelledError

            return cancel_mount().__await__()

    fake_app = SimpleNamespace(push_screen=lambda *_args, **_kwargs: CancelledMount())
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: settings
    )
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {"api_settings": {"openai": {}}}
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(*_args, **_kwargs):
        return {"openai": ["gpt-5"]}

    screen._providers_models_for_console_settings = provider_models

    with pytest.raises(asyncio.CancelledError):
        await screen._settings_navigation._open_console_settings()


@pytest.mark.asyncio
async def test_suspended_open_uses_active_raw_provider_for_initial_discovery(
    monkeypatch,
) -> None:
    """Fresh composition is seeded from the draft provider, not its canonical origin."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "vllm",
            "console-settings-model-picker": "draft-model",
            "console-settings-base-url": "http://draft-vllm.invalid:8000",
        },
        provider_model_drafts={"openai": "gpt-5", "vllm": "draft-model"},
        provider_base_url_drafts={
            "vllm": "http://draft-vllm.invalid:8000"
        },
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-provider",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    store = ConsoleChatStore()
    store.create_session(settings=settings)
    screen = ChatScreen.__new__(ChatScreen)
    build_console_settings_controllers(screen)
    staged_modal: list[ConsoleSettingsModal] = []
    discovery_inputs: list[tuple[str, str | None]] = []

    class Mounted:
        def __await__(self):
            async def mounted():
                return None

            return mounted().__await__()

    def push_screen(modal, callback):
        staged_modal.append(modal)
        return Mounted()

    fake_app = SimpleNamespace(push_screen=push_screen)
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: fake_app))
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: settings
    )
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        run_state=SimpleNamespace(is_send_allowed=True),
        reset_active_context_memory=lambda _session_id: None,
        undo_context_memory_reset=lambda: None,
        reset_all_context_memories=lambda _session_id: None,
        compact_context_now=lambda _session_id: None,
    )
    screen._active_console_settings_context_estimate = lambda: (
        ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._active_console_context_control_state = lambda **_kwargs: None
    _install_open_settings_dependencies(screen)
    screen._provider_readiness_app_config = lambda: {
        "api_settings": {"openai": {}, "vllm": {}}
    }
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False

    async def provider_models(provider, *, current_model):
        discovery_inputs.append((provider, current_model))
        return {"vllm": ["draft-model"]}

    screen._providers_models_for_console_settings = provider_models

    assert (
        await screen._settings_navigation._open_console_settings(
            suspended_draft=snapshot
        )
        is True
    )
    assert discovery_inputs == [("vllm", "draft-model")]
    assert staged_modal[0]._active_provider == "vllm"
    assert staged_modal[0]._providers_models == {"vllm": ["draft-model"]}


@pytest.mark.asyncio
async def test_suspended_modal_initial_composition_uses_active_raw_provider() -> None:
    """Connection controls compose for the draft provider before mount events run."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "openai": {"api_key": "test-key"},
            "vllm": {"api_url": "http://canonical-vllm.invalid:8000"},
        }
    }
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5")
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "vllm",
            "console-settings-model-picker": "draft-model",
            "console-settings-base-url": "http://draft-vllm.invalid:8000",
        },
        provider_model_drafts={"openai": "gpt-5", "vllm": "draft-model"},
        provider_base_url_drafts={"vllm": "http://draft-vllm.invalid:8000"},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-provider",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    modal = _basic_modal(
        settings,
        app,
        providers_models={"openai": ["gpt-5"], "vllm": ["draft-model"]},
        suspended_draft=snapshot,
    )
    discovery_provider_calls: list[str] = []
    supports_discovery = modal._provider_supports_model_discovery

    def record_discovery_provider(provider: str) -> bool:
        discovery_provider_calls.append(provider)
        return supports_discovery(provider)

    modal._provider_supports_model_discovery = record_discovery_provider

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert discovery_provider_calls[0] == "vllm"
        assert modal.query_one("#console-settings-provider", Select).value == "vllm"
        assert modal.query_one(ModelSearchPicker).value == "draft-model"
        assert modal.query_one("#console-settings-base-url", Input).value == (
            "http://draft-vllm.invalid:8000"
        )
        assert modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).display is True


class FakeConsoleModelDiscoveryScope:
    def __init__(self, entries: tuple[MergedModelEntry, ...]) -> None:
        self.entries = entries
        self.merge_calls = []

    async def merge_saved_and_discovered_models(self, **kwargs):
        self.merge_calls.append(kwargs)
        return self.entries


class FailingConsoleModelDiscoveryScope:
    async def merge_saved_and_discovered_models(self, **kwargs):
        raise RuntimeError("merge failed")


class EmptyConsoleModelSnapshotScope:
    async def merge_saved_and_discovered_models(self, **_kwargs):
        return ()

    async def has_discovered_model_snapshot(self, **_kwargs):
        return True


def _visible_text(app: App[None]) -> str:
    return " ".join(str(widget.renderable) for widget in app.screen.query(Static))


def _summary_text(console) -> str:
    summary = console.query_one("#console-settings-summary")
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in summary.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


def test_groq_console_default_uses_current_catalog_model() -> None:
    groq_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"]["groq"]

    assert groq_settings["model"] == "llama-3.3-70b-versatile"
    assert groq_settings["model"] in API_MODELS_BY_PROVIDER["Groq"]
    assert groq_settings["model"] not in {"llama3-70b-8192", "llama3-8b-8192"}


def test_console_remote_defaults_use_smoke_verified_models() -> None:
    expected_defaults = {
        "anthropic": ("Anthropic", "claude-sonnet-5"),
        "cohere": ("Cohere", "command-a-03-2025"),
        "google": ("Google", "gemini-2.5-flash"),
        "huggingface": ("HuggingFace", "openai/gpt-oss-120b"),
    }

    for config_key, (catalog_key, expected_model) in expected_defaults.items():
        provider_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"][config_key]

        assert provider_settings["model"] == expected_model
        assert expected_model in API_MODELS_BY_PROVIDER[catalog_key]


async def _wait_for_console_settings_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-settings-modal")
            and host.screen_stack[-1].query("#console-settings-provider")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not open")


async def _apply_open_console_settings_modal(
    modal: ConsoleSettingsModal,
    pilot,
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    user_display_name_override: str | None = None,
) -> None:
    """Submit an open production modal through its live transaction path."""

    modal.query_one("#console-settings-provider", Select).value = provider
    for _ in range(20):
        await pilot.pause(0.05)
        if modal._active_provider == provider:
            break
    else:
        raise AssertionError(f"Provider did not settle to {provider!r}")
    model_select = modal.query_one("#console-settings-model-select", Select)
    if model_select.value != model:
        model_select.value = model
        await pilot.pause()
    if base_url is not None:
        modal.query_one("#console-settings-base-url", Input).value = base_url
    if user_display_name_override is not None:
        modal.query_one(
            "#console-settings-user-display-name", Input
        ).value = user_display_name_override
    await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.05)
    modal.query_one("#console-settings-save", Button).press()


async def _visible_console_settings_button(console: ChatScreen, pilot) -> Button:
    """Open the inspector rail and return the actionable settings summary button."""
    rail_state = replace(
        console._current_console_rail_state(),
        right_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    assert rail_state.right_open is True
    await _wait_for_selector(console, pilot, "#console-settings-open")
    for _ in range(40):
        button = console.query_one("#console-settings-open", Button)
        if button.display and button.region.width > 0 and button.region.height > 0:
            return button
        await pilot.pause(0.05)
    button = console.query_one("#console-settings-open", Button)
    raise AssertionError(
        "Console settings button is not visible/actionable: "
        f"display={button.display!r} region={button.region!r}"
    )


async def _wait_for_console_top_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            return
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not dismiss")


async def _wait_for_focused_id(host: App[None], pilot, widget_id: str) -> None:
    for _ in range(40):
        focused_id = getattr(host.focused, "id", None)
        if focused_id == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(host.focused, 'id', None)!r}"
    )


@pytest.mark.asyncio
async def test_missing_credential_action_is_mounted_only_for_missing_cloud_credentials() -> None:
    """Conversation settings exposes Settings-owned credential recovery only when blocked."""
    app = ModalHarness()
    app.app_config = {"api_settings": {"openai": {}}}
    results: list[object] = []
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5"),
        app,
        providers_models={"openai": ["gpt-5"]},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        action = app.screen.query_one("#console-settings-configure-credential", Button)
        assert action.display is True
        assert str(action.label) == "Configure credential…"
        await pilot.click("#console-settings-configure-credential")
        await pilot.pause()

    assert len(results) == 1
    assert isinstance(results[0], ConsoleSettingsCredentialRequest)
    assert results[0].provider == "openai"


@pytest.mark.asyncio
async def test_compact_missing_credential_pointer_click_after_picker_focus_dismisses_once() -> (
    None
):
    """A compact modal must not lose the recovery click to its scrollable body."""
    app = ModalHarness()
    app.app_config = {"api_settings": {"openai": {}}}
    results: list[object] = []
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5"),
        app,
        providers_models={"openai": ["gpt-5"]},
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        modal.query_one(ModelSearchPicker).focus_input()
        await pilot.click("#console-settings-configure-credential", button=3)
        await pilot.pause()
        assert results == []
        assert app.screen is modal

        await pilot.click("#console-settings-configure-credential")
        await pilot.pause()

    assert len(results) == 1
    assert isinstance(results[0], ConsoleSettingsCredentialRequest)
    assert results[0].provider == "openai"


@pytest.mark.asyncio
async def test_missing_credential_interior_pointer_click_activates_only_once(
    monkeypatch,
) -> None:
    """The normal Button path ignores right click and emits one left-click result."""
    app = ModalHarness()
    app.app_config = {"api_settings": {"openai": {}}}
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5"),
        app,
        providers_models={"openai": ["gpt-5"]},
    )
    dismissed: list[object] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        button = modal.query_one("#console-settings-configure-credential", Button)

        await pilot.click(button, offset=(2, 1), button=3)
        await pilot.pause()
        assert dismissed == []

        await pilot.click(button, offset=(2, 1))
        await pilot.pause()

    assert len(dismissed) == 1
    assert isinstance(dismissed[0], ConsoleSettingsCredentialRequest)


class RejectingConversationSettingsHarness(ConsoleHarness):
    """Host a real ChatScreen while rejecting its typed settings route."""

    def __init__(self, app_instance) -> None:
        super().__init__(app_instance)
        self.rejected_settings_routes: list[object] = []

    def on_navigate_to_screen(self, message) -> None:
        self.rejected_settings_routes.append(message)
        message.report_completion(False)


@pytest.mark.asyncio
async def test_mounted_configure_rejection_restores_picker_focus_through_production_route(
    monkeypatch,
) -> None:
    """A real Configure click reopens the draft at its prior logical input."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-5"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-5"}
    app.app_config["api_settings"]["openai"] = {}
    app.providers_models = {"openai": ["gpt-5"]}
    host = RejectingConversationSettingsHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="openai", model="gpt-5"),
        )

        assert await console._settings_navigation._open_console_settings() is True
        original = await _wait_for_console_settings_modal(host, pilot)
        picker = original.query_one(ModelSearchPicker)
        picker.focus_input()
        await _wait_for_focused_id(host, pilot, "model-search-picker-input")
        await pilot.click("#console-settings-configure-credential")

        fresh = None
        for _ in range(80):
            top = host.screen_stack[-1]
            if top is not original and top.query("#console-settings-modal"):
                fresh = top
                break
            await pilot.pause(0.05)
        assert fresh is not None
        await _wait_for_focused_id(host, pilot, "model-search-picker-input")

        assert len(host.rejected_settings_routes) == 1
        navigation = host.rejected_settings_routes[0]
        assert navigation.screen_context["provider"] == "openai"
        assert console._suspended_conversation_settings is None
        assert console._suspended_conversation_settings_token is None


@pytest.mark.asyncio
async def test_mounted_suspended_draft_rehydrates_raw_provider_drafts_and_focus() -> None:
    """Mounted capture and fresh modal rehydration retain raw, per-provider state."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://canonical-llama.invalid:8080"},
            "vllm": {"api_url": "http://canonical-vllm.invalid:8000"},
        }
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="canonical-model",
        base_url="http://canonical-llama.invalid:8080",
        system_prompt="private system prompt",
        pinned_prefill="private prefill",
    )
    providers_models = {
        "llama_cpp": ["canonical-model", "llama-draft"],
        "vllm": ["canonical-vllm", "vllm-draft"],
    }
    original = _basic_modal(settings, app, providers_models=providers_models)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(original)
        await pilot.pause()
        original.query_one("#console-settings-temperature", Input).value = "0.7.2"
        original.query_one("#console-settings-user-display-name", Input).value = "Ada"
        original.query_one("#console-context-custom-budget", Input).value = "raw-budget"
        original.query_one(ModelSearchPicker).set_model_value("llama-draft")
        original.query_one("#console-settings-base-url", Input).value = (
            "http://draft-llama.invalid:9090"
        )
        original.query_one("#console-settings-provider", Select).value = "vllm"
        await pilot.pause()
        original.query_one(ModelSearchPicker).set_model_value("vllm-draft")
        original.query_one("#console-settings-base-url", Input).value = (
            "http://draft-vllm.invalid:8001"
        )
        original.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        original.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed = False
        original._connection_details_disclosed = True
        original.query_one(ModelSearchPicker).focus_input()
        await pilot.pause()
        body = original.query_one("#console-settings-body", ScrollableContainer)
        body.scroll_to(y=7, animate=False)
        await pilot.pause()
        snapshot = original.capture_suspended_draft()
        expected_scroll_anchor = int(body.scroll_y)
        assert snapshot.settings.system_prompt == "private system prompt"
        assert snapshot.settings.pinned_prefill == "private prefill"
        assert snapshot.active_view == "model"
        assert snapshot.focus_control_id == "console-settings-model-picker"
        assert snapshot.scroll_anchor == expected_scroll_anchor
        assert snapshot.provider_model_drafts == {
            "llama_cpp": "llama-draft",
            "vllm": "vllm-draft",
        }
        assert snapshot.provider_base_url_drafts == {
            "llama_cpp": "http://draft-llama.invalid:9090",
            "vllm": "http://draft-vllm.invalid:8001",
        }
        detached = snapshot.to_mapping()
        detached["raw_values"]["console-settings-temperature"] = "changed"  # type: ignore[index]
        assert snapshot.raw_values["console-settings-temperature"] == "0.7.2"
        await app.pop_screen()

        restored = ConsoleSettingsDraftSnapshot.from_mapping(snapshot.to_mapping())
        assert restored is not None
        fresh = _basic_modal(
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="canonical-model",
                base_url="http://canonical-llama.invalid:8080",
            ),
            app,
            providers_models=providers_models,
            suspended_draft=restored,
        )
        await app.push_screen(fresh)
        await pilot.pause()
        assert fresh.query_one(ModelSearchPicker).value == "llama-draft"
        assert fresh.query_one("#console-settings-base-url", Input).value == (
            "http://draft-llama.invalid:9090"
        )
        assert fresh.query_one("#console-settings-temperature", Input).value == "0.7.2"
        assert fresh.query_one("#console-settings-user-display-name", Input).value == "Ada"
        assert fresh.query_one("#console-context-custom-budget", Input).value == "raw-budget"
        assert fresh._advanced_generation_disclosed is True
        assert fresh._connection_details_disclosed is True
        assert getattr(app.focused, "id", None) == "model-search-picker-input"
        assert int(
            fresh.query_one("#console-settings-body", ScrollableContainer).scroll_y
        ) == expected_scroll_anchor
        fresh.query_one("#console-settings-provider", Select).value = "vllm"
        await pilot.pause()
        assert fresh.query_one(ModelSearchPicker).value == "vllm-draft"
        assert fresh.query_one("#console-settings-base-url", Input).value == (
            "http://draft-vllm.invalid:8001"
        )
        fresh.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert fresh.query_one(ModelSearchPicker).value == "llama-draft"
        assert fresh.query_one("#console-settings-base-url", Input).value == (
            "http://draft-llama.invalid:9090"
        )


@pytest.mark.asyncio
async def test_mounted_suspended_draft_preserves_active_raw_base_url_whitespace() -> (
    None
):
    """Credential suspension cannot normalize an invalid active endpoint draft."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://canonical-llama.invalid:8080"},
            "vllm": {"api_url": "http://canonical-vllm.invalid:8000"},
        }
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="llama-model",
        base_url="http://canonical-llama.invalid:8080",
    )
    raw_active_endpoint = "  http://draft.invalid:9090  "
    other_provider_endpoint = "http://draft-vllm.invalid:8001"
    providers_models = {
        "llama_cpp": ["llama-model"],
        "vllm": ["vllm-model"],
    }

    async with app.run_test(size=(120, 40)) as pilot:
        original = _basic_modal(
            settings,
            app,
            providers_models=providers_models,
        )
        await app.push_screen(original)
        await pilot.pause()
        provider_select = original.query_one("#console-settings-provider", Select)
        provider_select.value = "vllm"
        await pilot.pause()
        original.query_one("#console-settings-base-url", Input).value = (
            other_provider_endpoint
        )
        provider_select.value = "llama_cpp"
        await pilot.pause()
        original.query_one("#console-settings-base-url", Input).value = (
            raw_active_endpoint
        )

        snapshot = original.capture_suspended_draft()
        assert snapshot.raw_values["console-settings-base-url"] == raw_active_endpoint
        assert snapshot.settings.base_url == "http://canonical-llama.invalid:8080"
        await app.pop_screen()

        restored = ConsoleSettingsDraftSnapshot.from_mapping(snapshot.to_mapping())
        assert restored is not None
        fresh = _basic_modal(
            settings,
            app,
            providers_models=providers_models,
            suspended_draft=restored,
        )
        await app.push_screen(fresh)
        await pilot.pause()

        fresh_base_url = fresh.query_one("#console-settings-base-url", Input)
        fresh_provider = fresh.query_one("#console-settings-provider", Select)
        assert fresh_base_url.value == raw_active_endpoint
        assert fresh.capture_suspended_draft().provider_base_url_drafts == {
            "llama_cpp": raw_active_endpoint,
            "vllm": other_provider_endpoint,
        }
        fresh_provider.value = "vllm"
        await pilot.pause()
        assert fresh_base_url.value == other_provider_endpoint
        fresh_provider.value = "llama_cpp"
        await pilot.pause()
        assert fresh_base_url.value == raw_active_endpoint


@pytest.mark.asyncio
async def test_mounted_suspended_draft_rehydrates_blank_connection_and_private_formatting() -> None:
    """Mounted recovery leaves incomplete editable fields and private whitespace intact."""
    app = ModalHarness()
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="",
            base_url="",
            system_prompt="system\n\tindented",
            pinned_prefill="prefill\n\tindented",
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "llama_cpp",
            "console-settings-model-picker": "",
            "console-settings-base-url": "",
        },
        provider_model_drafts={"llama_cpp": ""},
        provider_base_url_drafts={"llama_cpp": ""},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        modal = _basic_modal(
            ConsoleSessionSettings(provider="llama_cpp", model="canonical"),
            app,
            providers_models={"llama_cpp": ["canonical"]},
            suspended_draft=snapshot,
        )
        await app.push_screen(modal)
        await pilot.pause()
        assert modal.query_one(
            "#model-search-picker-input", Input
        ).value == ""
        assert modal.query_one("#console-settings-base-url", Input).value == ""
        captured = modal.capture_suspended_draft()
        assert captured.settings.system_prompt == "system\n\tindented"
        assert captured.settings.pinned_prefill == "prefill\n\tindented"


@pytest.mark.asyncio
async def test_mounted_suspended_none_model_stays_blank_after_provider_round_trip() -> (
    None
):
    """An explicitly unselected provider model cannot fall back to the catalog."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "openai": {"api_key": "test-key"},
            "anthropic": {"api_key": "test-key"},
        }
    }
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model=None),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "openai",
            "console-settings-model-picker": "",
        },
        provider_model_drafts={"openai": None},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        modal = _basic_modal(
            snapshot.settings,
            app,
            providers_models={
                "openai": ["catalog-model"],
                "anthropic": ["claude-model"],
            },
            suspended_draft=snapshot,
        )
        await app.push_screen(modal)
        await pilot.pause()

        picker_input = modal.query_one("#model-search-picker-input", Input)
        provider_select = modal.query_one("#console-settings-provider", Select)
        assert picker_input.value == ""
        assert modal._provider_model_drafts["openai"] is None

        provider_select.value = "anthropic"
        await pilot.pause()
        assert picker_input.value == "claude-model"
        assert modal._provider_model_drafts["openai"] is None
        provider_select.value = "openai"
        await pilot.pause()

        assert picker_input.value == ""
        captured = modal.capture_suspended_draft()
        assert captured.raw_values["console-settings-model-picker"] == ""
        assert captured.provider_model_drafts["openai"] is None


@pytest.mark.asyncio
async def test_suspended_focus_falls_back_to_connection_provider_when_target_hidden() -> None:
    app = ModalHarness()
    app.app_config = {"api_settings": {"openai": {"api_key": "test-key"}}}
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "openai",
            "console-settings-model-picker": "gpt-5",
        },
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-base-url",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                snapshot.settings,
                app,
                providers_models={"openai": ["gpt-5"]},
                suspended_draft=snapshot,
            )
        )
        await _wait_for_focused_id(
            app, pilot, "console-settings-provider-picker-input"
        )


@pytest.mark.asyncio
async def test_suspended_model_picker_focus_falls_back_when_ancestor_is_hidden() -> None:
    """The logical picker target uses its input's effective focusability."""
    app = ModalHarness()
    app.app_config = {"api_settings": {"openai": {"api_key": "test-key"}}}
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-model-picker": "gpt-5"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        modal = _basic_modal(
            snapshot.settings,
            app,
            providers_models={"openai": ["gpt-5"]},
        )
        await app.push_screen(modal)
        await pilot.pause()
        picker = modal.query_one("#console-settings-model-picker", ModelSearchPicker)
        assert picker.parent is not None
        picker.parent.display = False
        assert picker.query_one("#model-search-picker-input", Input).focusable
        assert modal.query_one(
            "#console-settings-provider-picker-input", Input
        ).focusable
        modal._restore_suspended_scroll_and_focus(snapshot)
        await _wait_for_focused_id(
            app, pilot, "console-settings-provider-picker-input"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "focus_control_id",
    [None, "console-context-undo-reset", "console-context-confirm-reset-all"],
)
async def test_suspended_context_missing_or_transient_focus_reveals_connection_fallback(
    focus_control_id: str | None,
) -> None:
    """Unavailable Context focus restores a usable visible Connection target."""
    app = ModalHarness()
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "openai",
            "console-settings-model-picker": "gpt-5",
        },
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="context",
        scroll_anchor=0,
        focus_control_id=focus_control_id,
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        modal = _basic_modal(
            snapshot.settings,
            app,
            providers_models={"openai": ["gpt-5"]},
            suspended_draft=snapshot,
        )
        await app.push_screen(modal)
        await _wait_for_focused_id(
            app, pilot, "console-settings-provider-picker-input"
        )

        assert modal._active_view == "model"
        assert all(section.display for section in modal.query(".console-settings-model-view"))
        assert modal.query_one("#console-settings-context-view").display is False


@pytest.mark.asyncio
async def test_mounted_suspended_draft_round_trips_context_view_and_focus() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 24)) as pilot:
        original = _basic_modal(settings, app)
        await app.push_screen(original)
        await pilot.click("#console-settings-view-context")
        await pilot.pause()
        context_budget = original.query_one("#console-context-custom-budget", Input)
        context_budget.value = "not-a-number-yet"
        context_budget.focus()
        await pilot.pause()
        snapshot = original.capture_suspended_draft()
        assert snapshot.active_view == "context"
        assert snapshot.focus_control_id == "console-context-custom-budget"
        assert snapshot.raw_values["console-context-custom-budget"] == "not-a-number-yet"
        await app.pop_screen()

        fresh = _basic_modal(
            settings,
            app,
            suspended_draft=ConsoleSettingsDraftSnapshot.from_mapping(
                snapshot.to_mapping()
            ),
        )
        await app.push_screen(fresh)
        await pilot.pause()
        assert fresh._active_view == "context"
        assert fresh.query_one("#console-settings-context-view").display is True
        assert fresh.query_one("#console-context-custom-budget", Input).value == (
            "not-a-number-yet"
        )
        await _wait_for_focused_id(app, pilot, "console-context-custom-budget")


@pytest.mark.asyncio
async def test_suspended_capture_updates_valid_context_semantics_and_preserves_them_when_raw_turns_invalid() -> None:
    """Raw invalid edits remain exact without replacing the last valid policy."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 24)) as pilot:
        modal = _basic_modal(settings, app)
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-context-budget-mode", Select).value = "custom"
        modal.query_one("#console-context-custom-budget", Input).value = "2048"
        valid = modal.capture_suspended_draft()

        assert valid.context_policy_overrides.budget_mode is ContextBudgetMode.CUSTOM
        assert valid.context_policy_overrides.custom_budget_tokens == 2048

        modal.query_one("#console-context-custom-budget", Input).value = "temporarily-invalid"
        invalid = modal.capture_suspended_draft()

        assert invalid.raw_values["console-context-custom-budget"] == (
            "temporarily-invalid"
        )
        assert invalid.context_policy_overrides == valid.context_policy_overrides


@pytest.mark.asyncio
async def test_suspended_rehydration_rebuilds_context_state_before_raw_overlay() -> None:
    """Semantic context survives independently from temporarily invalid raw text."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=3072,
    )
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=overrides,
        raw_values={
            "console-settings-provider": "llama_cpp",
            "console-settings-model-picker": "model-a",
            "console-context-budget-mode": "custom",
            "console-context-custom-budget": "temporarily-invalid",
        },
        provider_model_drafts={"llama_cpp": "model-a"},
        provider_base_url_drafts={},
        active_view="context",
        scroll_anchor=0,
        focus_control_id="console-context-custom-budget",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )

    async with app.run_test(size=(120, 24)) as pilot:
        modal = _basic_modal(settings, app, suspended_draft=snapshot)
        await app.push_screen(modal)
        await pilot.pause()

        assert modal._context_state.overrides == overrides
        assert modal._context_state.resolved_policy.policy.budget_mode is (
            ContextBudgetMode.CUSTOM
        )
        assert modal._context_state.resolved_policy.policy.custom_budget_tokens == 3072
        assert modal.query_one("#console-context-custom-budget", Input).value == (
            "temporarily-invalid"
        )


async def _press_new_console_tab(console, store, pilot) -> str:
    previous_session_id = store.active_session_id
    console.query_one("#console-new-chat-tab", Button).press()
    for _ in range(40):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError("New Console tab did not activate")


def _first_chat_config(provider: str = "openai", model: str = "model-a") -> dict:
    return {
        "chat_defaults": {"provider": provider, "model": model},
        "api_settings": {
            provider: {
                "api_key": "test-only-key",
                "model": model,
            }
        },
    }


def _pending_first_chat(app) -> ConsoleFirstChatIntent | None:
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    if claim is None:
        return None
    value = claim.value
    app.pending_handoffs.release(claim)
    return value if isinstance(value, ConsoleFirstChatIntent) else None


def _first_chat_owner(console: ChatScreen) -> ConsoleSessionController:
    return console._session


def _first_chat_session_snapshot(session: ConsoleChatSession) -> dict[str, object]:
    """Capture all session values without comparing holder object identity."""

    snapshot = {
        item.name: deepcopy(getattr(session, item.name))
        for item in fields(ConsoleChatSession)
        if item.name not in {"rag_scope_holder", "todo_store"}
    }
    snapshot["rag_scope_holder"] = deepcopy(session.rag_scope_holder.scope)
    snapshot["todo_store"] = deepcopy(session.todo_store.export_snapshot())
    return snapshot


@pytest.fixture(autouse=True)
def _first_chat_generation_guard_uses_session_snapshot(monkeypatch):
    """Keep synthetic Console snapshots internally consistent in this suite.

    The config module's real publication lock is covered in
    ``Tests/test_config_delete_settings.py``. Console tests intentionally use
    arbitrary generations, so their final guard must observe the same injected
    snapshot as the rest of the consumer transaction.
    """

    def guarded(expected_generation: int, action) -> bool:
        if (
            session_module.get_runtime_config_snapshot().generation
            != expected_generation
        ):
            return False
        return action() is True

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        guarded,
        raising=False,
    )


def _mounted_first_chat_projection(console: ChatScreen) -> dict[str, object]:
    """Capture first-chat-owned mounted state without retaining widgets."""

    store = console._ensure_console_chat_store()
    controller = console._ensure_console_chat_controller()
    control_bar = console.query_one("#console-control-bar")
    composer = console.query_one("#console-native-composer")
    tabs = tuple(
        (
            str(tab.id),
            str(getattr(tab, "label", "")),
            tuple(sorted(tab.classes)),
        )
        for tab in console.query(".console-session-tab")
    )
    return {
        "active_session_id": store.active_session_id,
        "controller": (
            controller.provider,
            controller.model,
            controller.configured_model,
            controller.system_prompt,
        ),
        "control_scalars": (
            console._console_control_provider,
            console._console_control_model,
        ),
        # Rail visibility is responsive presentation state and may settle after
        # a pending preference write. The rollback contract owns the summary's
        # projected content, not whether the responsive rail is currently open.
        "summary": " ".join(
            getattr(widget.renderable, "plain", str(widget.renderable))
            for widget in console.query_one("#console-settings-summary").query(Static)
            if hasattr(widget, "renderable") and str(widget.renderable)
        ),
        "control_state": deepcopy(control_bar.state),
        "provider_label": str(
            console.query_one("#console-provider-label", Static).renderable
        ),
        "model_label": str(
            console.query_one("#console-model-label", Static).renderable
        ),
        "tabs": tabs,
        "composer_draft": composer.draft_text(),
        "focus_id": getattr(console.app.focused, "id", None),
    }


async def _wait_for_first_chat_projection(
    console: ChatScreen,
    pilot,
    expected: dict[str, object],
) -> None:
    for _ in range(80):
        if _mounted_first_chat_projection(console) == expected:
            return
        await pilot.pause(0.05)
    assert _mounted_first_chat_projection(console) == expected


def test_console_store_can_reserve_an_exact_first_chat_session_id() -> None:
    settings = build_default_console_session_settings(_first_chat_config())
    store = ConsoleChatStore()

    session = store.create_session(
        session_id="first-chat-session",
        settings=settings,
        canonical_settings_baseline=settings,
    )

    assert session.id == "first-chat-session"
    with pytest.raises(ValueError, match="already exists"):
        store.create_session(session_id=session.id, settings=settings)


def test_first_chat_target_eligibility_query_is_read_only() -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    session = store.create_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    session_before = _first_chat_session_snapshot(session)
    active_before = store.active_session_id
    controls_before = (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    )

    assert (
        _first_chat_owner(console).eligible_console_first_chat_session_id()
        == session.id
    )
    assert store.active_session_id == active_before
    assert _first_chat_session_snapshot(session) == session_before
    assert (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    ) == controls_before

    store.set_session_draft(session.id, "user draft")
    changed_before = _first_chat_session_snapshot(session)
    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert _first_chat_session_snapshot(session) == changed_before


@pytest.mark.parametrize("user_provenance", ["custom-workspace", "renamed-back"])
def test_first_chat_eligibility_rejects_empty_user_owned_session_without_mutation(
    user_provenance: str,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    workspace_id = "workspace-user" if user_provenance == "custom-workspace" else None
    store = ConsoleChatStore(
        workspace_context=ConsoleWorkspaceContext(
            active_workspace_id=workspace_id or "global"
        )
    )
    console._console_chat_store = store
    user_session = store.create_session(
        title="Chat 1",
        workspace_id=workspace_id,
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    if user_provenance == "renamed-back":
        store.rename_session(user_session.id, "User planning")
        store.rename_session(user_session.id, "Chat 1")
    before = _first_chat_session_snapshot(user_session)

    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert store.active_session_id == user_session.id
    assert len(store.sessions()) == 1
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == before


def test_first_chat_eligibility_rejects_user_created_default_named_session() -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = build_default_console_session_settings(
        _first_chat_config("openai", "user-model")
    )
    user_session = store.create_session(title="Chat 1", settings=user_settings)
    before = _first_chat_session_snapshot(user_session)

    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert store.active_session_id == user_session.id
    assert len(store.sessions()) == 1
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == before


def test_session_owner_refuses_session_switch_and_config_generation_races(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(23, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    target = store.create_session(
        session_id="existing-first-chat-target",
        settings=build_default_console_session_settings(snapshot.values),
        canonical_settings_baseline=build_default_console_session_settings(
            snapshot.values
        ),
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 23)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    competing = store.create_session(
        settings=build_default_console_session_settings(snapshot.values),
        canonical_settings_baseline=build_default_console_session_settings(
            snapshot.values
        ),
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == competing.id
    assert _pending_first_chat(app) == intent

    store.switch_session(target.id)
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(24, snapshot.values),
        raising=False,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    assert _pending_first_chat(app) == intent


def test_first_chat_consumer_activates_once_and_acknowledges_exact_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(31, _first_chat_config("llama_cpp", "local-a"))
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    prior_settings = build_default_console_session_settings(
        _first_chat_config("openai", "prior-model")
    )
    prior = store.create_session(
        settings=prior_settings,
        canonical_settings_baseline=prior_settings,
    )
    store.set_session_draft(prior.id, "keep this draft")
    intent = ConsoleFirstChatIntent(
        "first-run-future-session", "llama_cpp", "local-a", snapshot.generation
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    presentation = MagicMock(side_effect=real_apply)
    restore_focus = MagicMock()
    owner._apply_first_chat_control_selection_fn = presentation
    owner._restore_first_chat_focus_fn = restore_focus

    assert owner._screen_mounted_accessor() is False
    assert owner.consume_pending_console_first_chat_intent() is True
    presentation.assert_called_once_with("llama_cpp", "local-a")
    restore_focus.assert_not_called()
    assert store.active_session_id == "first-run-future-session"
    assert store.session_settings("first-run-future-session").provider == "llama_cpp"
    assert store.session_settings("first-run-future-session").model == "local-a"
    assert store.session_draft(prior.id) == "keep this draft"
    assert store.session_settings(prior.id) == prior_settings
    assert console._console_control_provider == "llama_cpp"
    assert console._console_control_model == "local-a"
    assert _pending_first_chat(app) is None
    assert owner.consume_pending_console_first_chat_intent() is False


def test_first_chat_consumer_refuses_absent_nonreserved_target(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(37, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("deleted-target", "openai", "model-a", 37)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.sessions() == []
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_target_concurrent_id_claim_is_not_overwritten(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(39, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-target", "openai", "model-a", 39)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session
    competing = ConsoleSessionSettings(
        provider="openai",
        model="competing-model",
        source="user",
    )

    def create_with_concurrent_claim(**kwargs):
        if kwargs.get("session_id") == intent.session_id:
            original_create(session_id=intent.session_id, settings=competing)
        return original_create(**kwargs)

    monkeypatch.setattr(store, "create_session", create_with_concurrent_claim)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.session_settings(intent.session_id) == competing
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_target_never_adopts_preexisting_pristine_id(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(41, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-target", "openai", "model-a", 41)
    competing = build_default_console_session_settings(
        _first_chat_config("openai", "restored-model")
    )
    store.create_session(
        session_id=intent.session_id,
        settings=competing,
        canonical_settings_baseline=competing,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.session_settings(intent.session_id) == competing
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_create_preserves_concurrent_active_session_switch(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve prior draft")
    competing = store.create_session(
        title="Competing user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="competing-user-model",
            source="user",
        ),
    )
    store.set_session_draft(competing.id, "preserve competing draft")
    store.switch_session(prior.id)
    user_sessions_before = {
        item.id: _first_chat_session_snapshot(item) for item in store.sessions()
    }
    snapshot = RuntimeConfigSnapshot(42, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "reserved-active-session-race",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session

    def create_then_select_competing_session(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == intent.session_id:
            store.switch_session(competing.id)
        return created

    monkeypatch.setattr(
        store,
        "create_session",
        create_then_select_competing_session,
    )
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    presentation = MagicMock(side_effect=real_apply)
    acknowledgement = MagicMock(
        wraps=app.pending_handoffs.acknowledge_current,
    )
    owner._apply_first_chat_control_selection_fn = presentation
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        acknowledgement,
    )

    assert owner.consume_pending_console_first_chat_intent() is False
    assert all(item.id != intent.session_id for item in store.sessions())
    assert {
        item.id: _first_chat_session_snapshot(item) for item in store.sessions()
    } == user_sessions_before
    assert store.session_draft(prior.id) == "preserve prior draft"
    assert store.session_draft(competing.id) == "preserve competing draft"
    assert store.active_session_id == competing.id
    assert _pending_first_chat(app) == intent
    presentation.assert_called_once_with(None, None)
    acknowledgement.assert_not_called()


def test_first_chat_generation_change_during_reserved_create_rolls_back(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    console._console_control_provider = "prior-control-provider"
    console._console_control_model = "prior-control-model"
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = ConsoleSessionSettings(
        provider="openai",
        model="user-model",
        source="user",
    )
    user_session = store.create_session(
        title="User session",
        settings=user_settings,
    )
    store.set_session_draft(user_session.id, "preserve exactly")
    user_before = _first_chat_session_snapshot(user_session)
    current = [RuntimeConfigSnapshot(43, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-race", "openai", "model-a", 43)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session

    def create_then_advance_generation(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == intent.session_id:
            current[0] = RuntimeConfigSnapshot(44, current[0].values)
        return created

    monkeypatch.setattr(store, "create_session", create_then_advance_generation)
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    projections: list[tuple[object, object]] = []

    def apply_and_record(provider, model) -> None:
        projections.append((provider, model))
        real_apply(provider, model)

    owner._apply_first_chat_control_selection_fn = apply_and_record

    assert owner.consume_pending_console_first_chat_intent() is False
    assert projections == [("prior-control-provider", "prior-control-model")]
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == user_session.id
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == user_before
    assert _pending_first_chat(app) == intent
    assert len(notifications) == 1


def test_first_chat_generation_change_during_refresh_restores_exact_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    target = store.create_session(
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    target_before = _first_chat_session_snapshot(target)
    current = [RuntimeConfigSnapshot(47, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 47)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    original_refresh = store.refresh_pristine_session_settings

    def refresh_then_advance_generation(*args, **kwargs):
        refreshed = original_refresh(*args, **kwargs)
        current[0] = RuntimeConfigSnapshot(48, current[0].values)
        return refreshed

    monkeypatch.setattr(
        store,
        "refresh_pristine_session_settings",
        refresh_then_advance_generation,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before
    assert _pending_first_chat(app) == intent


def test_first_chat_generation_publish_at_ack_rolls_back_reserved_creation(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve before ack")
    sessions_before = [_first_chat_session_snapshot(item) for item in store.sessions()]
    current = [RuntimeConfigSnapshot(67, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent("ack-publish-new", "openai", "model-a", 67)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)

    def publish_before_guarded_ack(_generation, _acknowledge) -> bool:
        current[0] = RuntimeConfigSnapshot(68, current[0].values)
        return False

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        publish_before_guarded_ack,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == prior.id
    assert [
        _first_chat_session_snapshot(item) for item in store.sessions()
    ] == sessions_before
    assert _pending_first_chat(app) == intent


def test_first_chat_generation_publish_at_ack_restores_existing_refresh(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    target = store.create_session(
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    target_before = _first_chat_session_snapshot(target)
    current = [RuntimeConfigSnapshot(69, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 69)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

    def publish_before_guarded_ack(_generation, _acknowledge) -> bool:
        current[0] = RuntimeConfigSnapshot(70, current[0].values)
        return False

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        publish_before_guarded_ack,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before
    assert _pending_first_chat(app) == intent


def test_first_chat_replacement_and_session_switch_during_create_roll_back_old_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = ConsoleSessionSettings(
        provider="openai",
        model="user-model",
        source="user",
    )
    selected = store.create_session(title="Selected work", settings=user_settings)
    store.set_session_draft(selected.id, "selected draft")
    selected_before = _first_chat_session_snapshot(selected)
    snapshot = RuntimeConfigSnapshot(49, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    old_intent = ConsoleFirstChatIntent("old-reserved", "openai", "model-a", 49)
    replacement = ConsoleFirstChatIntent(
        "replacement-reserved",
        "openai",
        "model-a",
        49,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(old_intent)
    original_create = store.create_session

    def create_then_replace_and_reselect(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == old_intent.session_id:
            store.switch_session(selected.id)
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return created

    monkeypatch.setattr(store, "create_session", create_then_replace_and_reselect)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != old_intent.session_id for item in store.sessions())
    assert store.active_session_id == selected.id
    assert _first_chat_session_snapshot(selected) == selected_before
    assert _pending_first_chat(app) == replacement
    assert notifications == []
    assert _first_chat_owner(console)._first_chat_handoff_notified_revision is None


def test_first_chat_guarded_ack_replacement_rolls_back_original_claim(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Guarded-ack prior work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="guarded-ack-prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve guarded-ack draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(50, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "guarded-ack-original-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(
        original,
        session_id="guarded-ack-replacement-target",
    )
    app.pending_handoffs.stage_reserved_console_first_chat(original)

    def stage_replacement_before_acknowledgement(
        _expected_generation,
        acknowledge,
    ) -> bool:
        app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return acknowledge()

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        stage_replacement_before_acknowledgement,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != original.session_id for item in store.sessions())
    assert all(item.id != replacement.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert store.session_draft(prior.id) == "preserve guarded-ack draft"
    assert _pending_first_chat(app) == replacement


def test_first_chat_current_claim_fence_blocks_replacement_target_projection(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    console._console_control_provider = "prior-control-provider"
    console._console_control_model = "prior-control-model"
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Current-claim prior work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="current-claim-prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve current-claim draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(52, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "current-claim-original-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(
        original,
        session_id="current-claim-replacement-target",
    )
    app.pending_handoffs.stage_reserved_console_first_chat(original)
    original_create = store.create_session

    def create_then_stage_replacement(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == original.session_id:
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return created

    monkeypatch.setattr(
        store,
        "create_session",
        create_then_stage_replacement,
    )
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    projections: list[tuple[object, object]] = []

    def apply_and_record(provider, model) -> None:
        projections.append((provider, model))
        real_apply(provider, model)

    owner._apply_first_chat_control_selection_fn = apply_and_record

    assert owner.consume_pending_console_first_chat_intent() is False
    assert projections == [("prior-control-provider", "prior-control-model")]
    assert all(item.id != original.session_id for item in store.sessions())
    assert all(item.id != replacement.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert store.session_draft(prior.id) == "preserve current-claim draft"
    assert _pending_first_chat(app) == replacement
    assert notifications == []


def test_first_chat_failed_acknowledgement_rolls_back_and_requeues(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="User work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "keep")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(51, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("ack-race", "openai", "model-a", 51)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    events: list[tuple[object, ...]] = []
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    real_rollback = store.rollback_created_pristine_session
    real_release = app.pending_handoffs.release

    def apply_and_record(provider, model) -> None:
        events.append(("project", provider, model))
        real_apply(provider, model)

    def rollback_and_record(session_id, **kwargs) -> bool:
        events.append(("store-rollback", session_id))
        return real_rollback(session_id, **kwargs)

    def release_and_record(claim) -> bool:
        events.append(("release", claim.revision))
        return real_release(claim)

    owner._apply_first_chat_control_selection_fn = apply_and_record
    monkeypatch.setattr(
        store,
        "rollback_created_pristine_session",
        rollback_and_record,
    )
    monkeypatch.setattr(app.pending_handoffs, "release", release_and_record)
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: False,
    )

    assert owner.consume_pending_console_first_chat_intent() is False
    assert [event[0] for event in events] == [
        "project",
        "store-rollback",
        "project",
        "release",
    ]
    assert events[0][1:] == ("openai", "model-a")
    assert events[1][1] == intent.session_id
    assert events[2][1:] == (None, None)
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert _pending_first_chat(app) == intent


def test_first_chat_ack_exception_rolls_back_create_and_survives_release_error(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="User work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "keep exact")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(73, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "ack-exception-create-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    real_release = app.pending_handoffs.release
    secret = "PRIVATE_ACK_EXCEPTION_TEXT"
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        session_module.logger,
        "warning",
        lambda *args, **_kwargs: warnings.append(args),
    )

    def fail_acknowledge(_claim) -> bool:
        raise RuntimeError(secret)

    def fail_release(_claim) -> bool:
        raise RuntimeError("PRIVATE_RELEASE_EXCEPTION_TEXT")

    monkeypatch.setattr(app.pending_handoffs, "acknowledge_current", fail_acknowledge)
    monkeypatch.setattr(app.pending_handoffs, "release", fail_release)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    rendered_warnings = repr(warnings)
    assert secret not in rendered_warnings
    assert "PRIVATE_RELEASE_EXCEPTION_TEXT" not in rendered_warnings
    assert intent.session_id not in rendered_warnings

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    monkeypatch.setattr(app.pending_handoffs, "release", real_release)
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.active_session_id == intent.session_id
    assert _pending_first_chat(app) is None


def test_first_chat_exception_log_is_metadata_only() -> None:
    records: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(str(message)),
        level="WARNING",
    )
    try:
        ConsoleSessionController._log_first_chat_handoff_exception(
            "guarded-acknowledgement",
            RuntimeError("SECRET-FIRST-CHAT-EXCEPTION"),
        )
    finally:
        loguru_logger.remove(sink_id)

    rendered = "".join(records)
    assert "guarded-acknowledgement" in rendered
    assert "RuntimeError" in rendered
    assert "SECRET-FIRST-CHAT-EXCEPTION" not in rendered


def test_first_chat_ack_exception_restores_refresh_and_retries(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior_settings = build_default_console_session_settings(
        _first_chat_config("openai", "prior-model")
    )
    target = store.create_session(
        session_id="ack-exception-refresh-target",
        settings=prior_settings,
        canonical_settings_baseline=prior_settings,
    )
    target_before = _first_chat_session_snapshot(target)
    snapshot = RuntimeConfigSnapshot(75, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        target.id,
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_REFRESH")),
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.session_settings(target.id).model == "model-a"
    assert _pending_first_chat(app) is None


def test_first_chat_config_guard_exception_rolls_back_and_retries(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(77, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "config-guard-exception-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_guard = session_module.run_if_runtime_config_generation_current
    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("PRIVATE_GUARD")),
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        real_guard,
        raising=False,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert _pending_first_chat(app) is None


def test_first_chat_ack_exception_after_replacement_preserves_replacement(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior replacement work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve replacement draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(79, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "ack-exception-old-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(original, session_id="ack-exception-replacement-target")
    app.pending_handoffs.stage_reserved_console_first_chat(original)
    real_acknowledge = app.pending_handoffs.acknowledge_current

    def replace_then_raise(_claim) -> bool:
        app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        raise RuntimeError("PRIVATE_REPLACEMENT_ACK")

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        replace_then_raise,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != original.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert _pending_first_chat(app) == replacement

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.active_session_id == replacement.session_id
    assert _pending_first_chat(app) is None


def test_first_chat_failed_notification_tracking_is_bounded_to_latest_revision(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(999, _first_chat_config()),
        raising=False,
    )
    latest_revision = 0

    for index in range(128):
        latest_revision = app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_FIRST_CHAT,
            ConsoleFirstChatIntent(
                f"missing-{index}",
                "openai",
                "model-a",
                index + 1,
            ),
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )

    assert len(notifications) == 128
    assert (
        _first_chat_owner(console)._first_chat_handoff_notified_revision
        == latest_revision
    )


@pytest.mark.asyncio
async def test_mounted_first_chat_preserves_restored_and_concurrent_sessions(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        53,
        _first_chat_config("llama_cpp", "mounted-local"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        restored_settings = ConsoleSessionSettings(
            provider="openai",
            model="restored-model",
            source="user",
        )
        restored = ConsoleChatSession(
            id="restored-user-session",
            title="Restored work",
            workspace_id="workspace-restored",
            persisted_conversation_id="conversation-restored",
            settings=restored_settings,
            draft="restored draft",
            has_user_work=True,
        )
        concurrent = ConsoleChatSession(
            id="concurrent-user-session",
            title="Concurrent work",
            settings=replace(restored_settings, model="concurrent-model"),
            draft="concurrent draft",
            has_user_work=True,
        )
        store.restore_state(
            sessions=[restored, concurrent],
            active_session_id=concurrent.id,
        )
        before = [_first_chat_session_snapshot(item) for item in store.sessions()]
        intent = ConsoleFirstChatIntent(
            "mounted-first-chat",
            "llama_cpp",
            "mounted-local",
            snapshot.generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()[:2]
        ] == before
        assert store.session_settings(intent.session_id).provider == "llama_cpp"
        assert store.session_settings(intent.session_id).model == "mounted-local"
        assert _pending_first_chat(app) is None

        sessions_after_success = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_after_success


@pytest.mark.asyncio
async def test_mounted_first_chat_replacement_ack_exception_restores_prior_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    snapshot = RuntimeConfigSnapshot(
        59,
        _first_chat_config("llama_cpp", "replacement-race-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = ConsoleSessionSettings(
            provider="openai",
            model="prior-mounted-model",
            source="user",
            system_prompt="Preserve this system prompt.",
        )
        prior = ConsoleChatSession(
            id="prior-mounted-replacement",
            title="Prior mounted work",
            settings=prior_settings,
            draft="preserve mounted draft",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        old_intent = ConsoleFirstChatIntent(
            "mounted-old-target",
            "llama_cpp",
            "replacement-race-model",
            snapshot.generation,
        )
        replacement = replace(old_intent, session_id="mounted-replacement-target")
        app.pending_handoffs.stage_reserved_console_first_chat(old_intent)
        real_acknowledge_current = getattr(
            app.pending_handoffs,
            "acknowledge_current",
            None,
        )

        def replace_immediately_before_ack(_claim) -> bool:
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
            raise RuntimeError("PRIVATE_MOUNTED_REPLACEMENT")

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            replace_immediately_before_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != old_intent.session_id for item in store.sessions())
        assert store.active_session_id == prior.id
        assert _pending_first_chat(app) == replacement
        assert notifications == []

        assert real_acknowledge_current is not None
        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge_current,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == replacement.session_id
        assert _pending_first_chat(app) is None


@pytest.mark.asyncio
async def test_mounted_first_chat_generation_publish_at_ack_restores_reserved_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    current = [
        RuntimeConfigSnapshot(
            61,
            _first_chat_config("llama_cpp", "failed-ack-model"),
        )
    ]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = ConsoleSessionSettings(
            provider="openai",
            model="prior-failed-ack-model",
            source="user",
            system_prompt="Keep the mounted controller state.",
        )
        prior = ConsoleChatSession(
            id="prior-mounted-failed-ack",
            title="Prior failed-ack work",
            settings=prior_settings,
            draft="keep the composer exact",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            "mounted-failed-ack-target",
            "llama_cpp",
            "failed-ack-model",
            current[0].generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)

        def publish_at_guarded_ack(_generation, _acknowledge) -> bool:
            current[0] = RuntimeConfigSnapshot(62, current[0].values)
            return False

        monkeypatch.setattr(
            session_module,
            "run_if_runtime_config_generation_current",
            publish_at_guarded_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != intent.session_id for item in store.sessions())
        assert store.active_session_id == prior.id
        assert _pending_first_chat(app) == intent


@pytest.mark.asyncio
async def test_mounted_first_chat_generation_publish_at_ack_restores_refresh_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    current = [
        RuntimeConfigSnapshot(
            63,
            _first_chat_config("llama_cpp", "refresh-ack-model"),
        )
    ]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = build_default_console_session_settings(
            _first_chat_config("openai", "prior-refresh-model")
        )
        target = ConsoleChatSession(
            id="mounted-refresh-ack-target",
            title="Chat 1",
            settings=prior_settings,
            canonical_settings_baseline=prior_settings,
        )
        store.restore_state(sessions=[target], active_session_id=target.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            target.id,
            "llama_cpp",
            "refresh-ack-model",
            current[0].generation,
        )
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

        def publish_at_guarded_ack(_generation, _acknowledge) -> bool:
            current[0] = RuntimeConfigSnapshot(64, current[0].values)
            return False

        monkeypatch.setattr(
            session_module,
            "run_if_runtime_config_generation_current",
            publish_at_guarded_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert store.active_session_id == target.id
        assert _pending_first_chat(app) == intent


@pytest.mark.asyncio
async def test_mounted_first_chat_ack_exception_during_mount_is_retryable(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        81,
        _first_chat_config("llama_cpp", "mount-exception-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "mounted-on-mount-exception-target",
        "llama_cpp",
        "mount-exception-model",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_MOUNT")),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        await pilot.pause()
        assert all(item.id != intent.session_id for item in store.sessions())
        assert _pending_first_chat(app) == intent

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert _pending_first_chat(app) is None


@pytest.mark.asyncio
async def test_mounted_first_chat_ack_exception_during_resume_restores_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        83,
        _first_chat_config("llama_cpp", "resume-exception-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior = ConsoleChatSession(
            id="mounted-resume-prior",
            title="Resume prior",
            settings=ConsoleSessionSettings(
                provider="openai",
                model="resume-prior-model",
                source="user",
                system_prompt="Preserve resume UI.",
            ),
            draft="preserve resume composer",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        prior_focused_widget = console.app.focused
        assert prior_focused_widget is not None
        assert prior_focused_widget.is_mounted is True
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            "mounted-on-resume-exception-target",
            "llama_cpp",
            "resume-exception-model",
            snapshot.generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)
        real_acknowledge = app.pending_handoffs.acknowledge_current
        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_RESUME")),
        )
        owner = _first_chat_owner(console)
        real_restore_focus = owner._restore_first_chat_focus_fn
        restore_focus = MagicMock(side_effect=real_restore_focus)
        owner._restore_first_chat_focus_fn = restore_focus

        console.on_screen_resume()
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        for _ in range(80):
            if restore_focus.call_count:
                break
            await pilot.pause(0.05)
        restore_focus.assert_called_once_with(prior_focused_widget)
        assert console.app.focused is prior_focused_widget
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != intent.session_id for item in store.sessions())
        assert _pending_first_chat(app) == intent

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert _pending_first_chat(app) is None


async def _click_console_session_tab(console, store, pilot, session_id: str) -> None:
    await pilot.click(f"#console-session-tab-{session_id}")
    for _ in range(40):
        if store.active_session_id == session_id:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    console._ensure_console_chat_controller().switch_session(session_id)
    await console._sync_native_console_chat_ui()
    if store.active_session_id == session_id:
        await pilot.pause()
        return
    raise AssertionError(f"Console tab {session_id!r} did not activate")


def _select_values(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    values: set[str] = set()
    for option in options:
        value = getattr(option, "value", None)
        if value is None and isinstance(option, tuple) and len(option) >= 2:
            value = option[1]
        if value is not None and value is not Select.NULL:
            values.add(str(value))
    return values


def _select_ordered_values(select: Select) -> tuple[str, ...]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    values: list[str] = []
    for option in options:
        value = getattr(option, "value", None)
        if value is None and isinstance(option, tuple) and len(option) >= 2:
            value = option[1]
        if value is not None and value is not Select.NULL:
            values.append(str(value))
    return tuple(values)


def _merged_model(
    model_id: str,
    *,
    source: str = "saved",
    capability_status: str = "known",
    persisted: bool = True,
) -> MergedModelEntry:
    return MergedModelEntry(
        provider="openai",
        provider_list_key="openai",
        model_id=model_id,
        display_name=model_id,
        source=source,
        capability_status=capability_status,
        persisted=persisted,
    )


@pytest.mark.asyncio
async def test_console_settings_summary_renders_rows_and_button() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Conversation settings" in text
        assert "Provider: llama.cpp" in text
        assert "Model: model-a" in text
        assert "Context: 12 / 4k" in text
        assert "Sampling: T 0.70, P 0.95" in text
        assert "Assistant: General" in text
        header = app.query_one("#console-settings-header", Horizontal)
        title = app.query_one("#console-settings-title", Static)
        button = app.query_one("#console-settings-open", Button)
        assert title.parent is header
        assert button.parent is header
        assert title.region.y == button.region.y
        assert str(button.label) == "Configure"
        assert button.tooltip == "Configure Console settings"


@pytest.mark.asyncio
async def test_console_settings_summary_uses_direct_choose_model_action_when_setup_blocked() -> (
    None
):
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: Missing",
        context_row="Context: unavailable",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Missing model",
        action_label="Choose Model",
        action_tooltip="Choose a model for this Console session",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Provider: llama.cpp" in text
        assert "Model: Missing" in text
        button = app.query_one("#console-settings-open", Button)
        assert str(button.label) == "Choose Model"
        assert button.tooltip == "Choose a model for this Console session"


@pytest.mark.asyncio
async def test_console_settings_summary_treats_missing_provider_row_as_blank() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row=None,  # type: ignore[arg-type]
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        provider_row = app.query_one("#console-settings-provider-row", Static)
        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)

        updated_state = ConsoleSettingsSummaryState(
            provider_row=None,  # type: ignore[arg-type]
            model_row="Model: model-b",
            context_row="Context: 20 / 4k",
            sampling_row="Sampling: T 0.20, P 0.90",
            identity_row="Persona: Analyst",
            readiness_label="Ready",
        )
        app.query_one(ConsoleSettingsSummary).sync_state(updated_state)
        await pilot.pause()

        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)


def test_console_settings_summary_button_sizing_uses_named_constants() -> None:
    assert not hasattr(settings_summary_module, "CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT")
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING == 2
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MIN_WIDTH == 9
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MAX_WIDTH == 14
    assert settings_summary_module.CONSOLE_SETTINGS_ROW_HEIGHT == 1


@pytest.mark.asyncio
async def test_console_settings_header_is_external_and_unknown_context_is_named() -> (
    None
):
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: one row",
        model_row="",
        context_row="",
        sampling_row="",
        identity_row="",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        summary = app.query_one(ConsoleSettingsSummary)
        header = summary.query_one("#console-settings-header", Horizontal)
        body = summary.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        for _ in range(4):
            await pilot.pause()

        assert header.parent is summary
        assert body.parent is summary
        assert list(summary.children) == [header, body]
        assert body.desired_content_lines == 2
        assert body.viewport.content_region.height == 2
        assert body.hint.display is False


@pytest.mark.asyncio
async def test_console_settings_body_uses_exact_twenty_line_content_ceiling() -> None:
    state = ConsoleSettingsSummaryState(
        model_row="Model: visible",
        context_row="",
        sampling_row="",
        identity_row="",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 30)) as pilot:
        body = app.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        await body.viewport.remove_children()
        content = Static("\n".join(f"row {index}" for index in range(20)))
        await body.viewport.mount(content)
        body.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert body.viewport.content_region.height == 20
        assert body.hint.display is False

        content.update("\n".join(f"row {index}" for index in range(21)))
        content.refresh(layout=True)
        body.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert body.viewport.content_region.height == 20
        assert body.hint.display is True
        assert body.hint.region.height == 1


def test_console_settings_modal_sizing_uses_named_constants() -> None:
    assert MODAL_BODY_MIN_HEIGHT == 0
    assert MODAL_CONTROL_HEIGHT == 3
    assert f"min-height: {MODAL_BODY_MIN_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"min-height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS


def test_pending_launch_inspector_auto_open_docstring_is_google_style() -> None:
    docstring = ChatScreen._apply_pending_launch_inspector_auto_open.__doc__

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring


@pytest.mark.asyncio
async def test_summary_builder_mounted_rail_uses_typed_copy_and_provider_name() -> None:
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    readiness = build_console_settings_readiness(
        settings,
        app_config={
            "api_settings": {
                "openai": {"api_key_env_var": "ABSENT_SUMMARY_TEST_KEY"}
            }
        },
        environ={},
    )
    readiness = replace(readiness, label="READY legacy poison")
    state = build_console_settings_summary_state(
        settings,
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        readiness,
    )

    assert state.provider_row == "Provider: OpenAI"
    assert state.model_row == "Model: gpt-4.1"
    assert state.readiness_label == ""

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        painted = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )

        assert "Provider: OpenAI" in painted
        assert "Not ready — API key missing for OpenAI" in painted
        assert "READY legacy poison" not in painted
        assert "Provider: openai" not in painted


def test_summary_state_omits_legacy_readiness_from_visible_rows() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="WIP", detail="Provider not wired yet.", native_send_supported=False
        ),
    )

    assert state.provider_row == "Provider: Provider"
    assert state.model_row == "Model: model-a"
    assert state.readiness_label == ""


def test_default_console_session_settings_prefers_provider_model_profile() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": False,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": 0.2,
                        "top_p": 0.88,
                        "min_p": 0.04,
                        "top_k": 40,
                        "max_tokens": 1234,
                        "seed": 101,
                        "presence_penalty": 0.2,
                        "frequency_penalty": 0.3,
                        "reasoning_effort": "high",
                        "reasoning_summary": "auto",
                        "verbosity": "high",
                        "streaming": True,
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.provider == "openai"
    assert settings.model == "gpt-4.1"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.88
    assert settings.min_p == 0.04
    assert settings.top_k == 40
    assert settings.max_tokens == 1234
    assert settings.seed == 101
    assert settings.presence_penalty == 0.2
    assert settings.frequency_penalty == 0.3
    assert settings.reasoning_effort == "high"
    assert settings.reasoning_summary == "auto"
    assert settings.verbosity == "high"
    assert settings.streaming is True


def test_default_console_session_settings_prefers_chat_defaults_over_provider_scalars() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8
    assert settings.streaming is False


def test_default_console_settings_delegates_canonical_model_and_endpoint_resolution() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI-Compatible",
            "model": "chat-model",
            "temperature": 0.31,
        },
        "api_settings": {
            "openai": {
                "model": "provider-model",
                "api_base_url": "https://api.example.test/v1",
                "temperature": 0.79,
            }
        },
    }

    settings = build_default_console_session_settings(app_config)

    assert settings.provider == "openai"
    assert settings.model == "chat-model"
    assert settings.base_url == "https://api.example.test/v1"
    assert settings.temperature == 0.31
    assert app_config["chat_defaults"]["provider"] == "OpenAI-Compatible"


def test_console_session_settings_accepts_documented_effort_values() -> None:
    app_config = {
        "api_settings": {
            "openai": {"api_key": "test-key", "model": "gpt-5.1"},
            "anthropic": {"api_key": "test-key", "model": "claude-opus-4-8"},
        }
    }
    openai_settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-5.1",
        reasoning_effort="none",
    )
    anthropic_settings = ConsoleSessionSettings(
        provider="anthropic",
        model="claude-opus-4-8",
        thinking_effort="max",
    )

    assert (
        validate_console_session_settings(openai_settings, app_config=app_config) == []
    )
    assert (
        validate_console_session_settings(anthropic_settings, app_config=app_config)
        == []
    )


def test_default_console_session_settings_reads_enable_streaming_as_compatibility_fallback() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is False


def test_default_console_session_settings_prefers_canonical_streaming_over_enable_streaming() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "streaming": True,
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": False,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is True


def test_default_console_session_settings_uses_global_fallbacks_when_profile_is_absent() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.33,
            "top_p": 0.81,
            "max_tokens": 2048,
            "streaming": False,
        },
        "api_settings": {
            "openai": {},
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.33
    assert settings.top_p == 0.81
    assert settings.max_tokens == 2048
    assert settings.streaming is False


def test_default_console_session_settings_skips_blank_model_profile_values() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": "",
                        "top_p": " ",
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8


def test_summary_state_keeps_missing_model_row_compact() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        ConsoleSettingsContextEstimate(
            used_tokens=None, token_limit=None, label="unknown"
        ),
        ConsoleSettingsReadiness(
            label="READY legacy poison",
            detail="PRIVATE legacy detail poison",
            native_send_supported=False,
            operability="not_ready",
            blocker="model_missing",
            recovery_action="select_model",
            provider_display_name="llama.cpp",
            configuration="configured",
            credential="not_required",
            endpoint="not_tested",
            model="missing",
            generation="not_tested",
        ),
    )

    assert state.provider_row == "Provider: llama.cpp"
    assert state.model_row == "Model: Missing"
    assert state.readiness_label == ""
    assert state.action_label == "Choose Model"
    assert state.action_tooltip == "Choose a model for this Console session"


def test_summary_state_exposes_safe_credential_source() -> None:
    """Show safe env/config credential sources without exposing secret values."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:OPENAI_API_KEY.",
            native_send_supported=True,
            operability="ready_to_send",
            configuration="configured",
            credential="present_unverified",
            credential_source="environment",
            model="unconfirmed",
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:api_settings.anthropic.api_key.",
            native_send_supported=True,
            operability="ready_to_send",
            configuration="configured",
            credential="present_unverified",
            credential_source="stored",
            model="unconfirmed",
        ),
    )

    assert env_state.credential_row == "Credential: environment variable (not verified)"
    assert config_state.credential_row == "Credential: local config (not verified)"


def test_summary_state_handles_empty_credential_source_names() -> None:
    """Collapse empty env/config credential-source identifiers without padding."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:   .",
            native_send_supported=True,
            operability="ready_to_send",
            configuration="configured",
            credential="present_unverified",
            credential_source="environment",
            model="unconfirmed",
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:   .",
            native_send_supported=True,
            operability="ready_to_send",
            configuration="configured",
            credential="present_unverified",
            credential_source="stored",
            model="unconfirmed",
        ),
    )

    assert env_state.credential_row == "Credential: environment variable (not verified)"
    assert config_state.credential_row == "Credential: local config (not verified)"


def test_summary_state_ignores_warning_lines_after_credential_source() -> None:
    """Keep appended readiness warnings out of the credential summary row."""
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail=(
                "OpenAI is ready. API key found via env:OPENAI_API_KEY.\n"
                "Model warning: selected model may not support native tools."
            ),
            native_send_supported=True,
            operability="ready_to_send",
            configuration="configured",
            credential="present_unverified",
            credential_source="environment",
            model="unconfirmed",
        ),
    )

    assert state.credential_row == "Credential: environment variable (not verified)"


def test_summary_state_appends_optional_sampling_fields_only_when_set() -> None:
    without_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", temperature=0.7, top_p=0.95
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )
    with_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            temperature=0.7,
            top_p=0.95,
            min_p=0.05,
            top_k=40,
            max_tokens=512,
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert without_optional.sampling_row == "Sampling: T 0.70, P 0.95"
    assert (
        with_optional.sampling_row
        == "Sampling: T 0.70, P 0.95, min_p 0.05, top_k 40, max_tokens 512"
    )


def test_summary_state_normalizes_unknown_context_label() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=None, token_limit=None, label="Context: unknown"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert state.context_row == "Context: unavailable"


def test_summary_state_renders_character_or_generic_assistant_identity() -> None:
    character = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            character_label="Ada",
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )
    generic = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert character.identity_row == "Character: Ada"
    assert generic.identity_row == "Assistant: General"


def test_summary_state_projects_character_identity_to_one_line_without_mutating_settings() -> (
    None
):
    raw_name = "Nyx\n\tAdmin\x00[/bold]"
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        character_label=raw_name,
    )

    summary = build_console_settings_summary_state(
        settings,
        ConsoleSettingsContextEstimate(
            used_tokens=12,
            token_limit=4096,
            label="12 / 4k",
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Ready.",
            native_send_supported=True,
        ),
    )

    assert summary.identity_row == "Character: Nyx Admin?[/bold]"
    assert settings.character_label == raw_name


def test_legacy_identity_settings_helper_ignores_unknown_keys_without_mutation() -> (
    None
):
    source = {
        "provider": "llama_cpp",
        "model": "model-a",
        "persona_label": "Legacy A",
        "user_profile_label": "Legacy B",
    }
    source_before = dict(source)
    restored = ConsoleSessionController._restore_console_settings(source)

    assert restored is not None
    assert source == source_before
    assert not hasattr(restored, "user_profile_label")
    serialized = ConsoleSessionController._serialize_console_settings(restored)
    assert serialized is not None
    assert {"persona_label", "user_profile_label"}.isdisjoint(serialized)


def test_choose_model_action_label_normalization() -> None:
    assert ChatScreen._is_console_choose_model_action(" Choose Model ")
    assert ChatScreen._is_console_choose_model_action("choose model")
    assert ChatScreen._is_console_choose_model_action("CHOOSE MODEL")
    assert not ChatScreen._is_console_choose_model_action("Configure")


@pytest.mark.asyncio
async def test_console_model_resolution_includes_runtime_discovered_models() -> None:
    scope = FakeConsoleModelDiscoveryScope(
        (
            _merged_model("gpt-4.1", source="persisted_discovered"),
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    options = await provider_model_resolution.resolve_provider_model_options(
        {"openai": ["gpt-4.1"]},
        scope,
        provider="OpenAI",
    )

    assert [option.model_id for option in options] == ["gpt-4.1", "gpt-5"]
    assert (
        options[1].warning
        == "Capabilities unknown until saved or verified; text chat is assumed."
    )
    assert scope.merge_calls == [
        {
            "mode": "local",
            "provider": "openai",
        }
    ]


@pytest.mark.asyncio
async def test_console_model_resolution_failure_logs_provider_context(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.llm_provider_catalog_scope_service = FailingConsoleModelDiscoveryScope()
    console = ChatScreen(app)
    logged = []

    def fake_exception(message, *args, **kwargs):
        logged.append((message, args, kwargs))

    monkeypatch.setattr(chat_screen_module.logger, "exception", fake_exception)

    models = await console._providers_models_for_console_settings(
        "OpenAI",
        current_model="gpt-5",
    )

    assert models == {"openai": ["gpt-4.1"]}
    assert logged == [
        (
            "Unable to resolve Console runtime-discovered models for provider=%s model=%s",
            ("openai", "gpt-5"),
            {},
        )
    ]


@pytest.mark.asyncio
async def test_console_settings_model_resolution_preserves_configured_alternatives() -> (
    None
):
    app = _build_test_app()
    app.providers_models = {
        "local_llamacpp": ["uat-local-model", "uat-alt-local-model"],
    }
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model(
                "uat-local-model",
                source="runtime_discovered",
                capability_status="known",
                persisted=False,
            ),
        )
    )
    console = ChatScreen(app)

    models = await console._providers_models_for_console_settings(
        "local_llamacpp",
        current_model="uat-local-model",
    )

    assert models["local_llamacpp"] == ["uat-local-model", "uat-alt-local-model"]


@pytest.mark.asyncio
async def test_console_settings_model_resolution_keeps_empty_cloud_snapshot_authoritative() -> (
    None
):
    app = _build_test_app()
    app.providers_models = {"anthropic": ["retired-model"]}
    app.llm_provider_catalog_scope_service = EmptyConsoleModelSnapshotScope()
    console = ChatScreen(app)

    models = await console._providers_models_for_console_settings(
        "anthropic",
        current_model=None,
    )

    assert models["anthropic"] == []


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_discards_draft() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    used_tokens=10,
                    token_limit=4096,
                    label="10 / 4k",
                ),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_delayed_initial_focus_is_safe_after_unmount() -> None:
    """A queued initial-focus callback must tolerate a modal dismissed first."""
    app = ModalHarness()
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await app.pop_screen()
        await pilot.pause()

        modal._focus_highest_priority_connection()


@pytest.mark.asyncio
async def test_console_settings_modal_escape_dismisses_none() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        # First Escape belongs to the focused searchable picker and restores
        # its committed value; the next one dismisses the modal.
        await pilot.press("escape")
        assert app.screen.query_one(ConsoleProviderPicker).value == "llama_cpp"
        await pilot.press("escape")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_renders_typed_operability_and_verification_evidence(
    monkeypatch,
) -> None:
    readiness = _typed_ready_unverified_readiness()
    monkeypatch.setattr(
        settings_modal_module,
        "build_console_settings_readiness",
        lambda *_args, **_kwargs: readiness,
    )
    app = ModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        rendered = str(
            app.screen.query_one("#console-settings-readiness", Static).renderable
        )

    assert "Ready to send — credential not verified" in rendered
    assert "Credential · Present — not verified (local config)" in rendered
    assert "Endpoint · Not tested" in rendered
    assert "Model · Selected — not verified at this endpoint" in rendered
    assert "Generation · Not tested" in rendered
    assert "LEGACY LABEL" not in rendered
    assert "PRIVATE legacy detail" not in rendered


@pytest.mark.asyncio
async def test_console_settings_modal_save_returns_validated_settings() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-connection")
        assert "Choose a model to enable sending." not in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )
        app.screen.query_one("#console-settings-temperature", Input).value = "0.42"
        app.screen.query_one("#console-settings-top-p", Input).value = "0.88"
        app.screen.query_one(
            "#console-settings-user-display-name", Input
        ).value = "  Captain Rowan  "
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"
    assert app.saved_settings.temperature == 0.42
    assert app.saved_settings.top_p == 0.88
    assert app.saved_result is not None
    assert app.saved_result.user_display_name_override == "Captain Rowan"
    assert not hasattr(app.saved_result.settings, "user_display_name_override")


@pytest.mark.asyncio
async def test_console_settings_modal_renders_current_chat_identity() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                user_display_name_override="Captain Rowan",
                global_user_display_name="Default Name",
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        identity = app.screen.query_one("#console-settings-user-display-name", Input)
        identity_help = app.screen.query_one(
            "#console-settings-user-display-name-help", Static
        )
        assert identity.value == "Captain Rowan"
        assert identity.placeholder == "Default Name"
        assert "Conversation identity" in _visible_text(app)
        assert app.screen._is_effectively_focusable(identity) is False
        assert identity_help.region.height == 0

        await pilot.click("#console-settings-identity-advanced CollapsibleTitle")
        await pilot.pause()

        assert app.screen._is_effectively_focusable(identity) is True
        assert identity_help.region.height > 0
        assert "Leave blank to use the global default." in _visible_text(app)


@pytest.mark.asyncio
async def test_console_settings_modal_local_provider_marks_no_effect_choices() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()

        thinking = app.screen.query_one("#console-settings-thinking-effort", Select)
        summary = app.screen.query_one("#console-settings-reasoning-summary", Select)
        verbosity = app.screen.query_one("#console-settings-verbosity", Select)
        effort = app.screen.query_one("#console-settings-reasoning-effort", Select)
        # Local providers consume reasoning effort, while authoritative
        # no-effect choices are removed without rewriting retained values.
        assert thinking.parent is not None and thinking.parent.display is False
        assert summary.parent is not None and summary.parent.display is False
        assert verbosity.parent is not None and verbosity.parent.display is False
        assert effort.parent is not None and effort.parent.display is True


@pytest.mark.asyncio
async def test_console_settings_modal_remote_provider_keeps_thinking_hint_plain() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="anthropic", model="claude-3-5-sonnet-latest"
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()

        thinking = app.screen.query_one("#console-settings-thinking-effort", Select)
        assert thinking.parent is not None and thinking.parent.display is True


@pytest.mark.asyncio
async def test_console_settings_modal_provider_switch_refreshes_choice_hints() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={"openai": ["gpt-4.1"], "llama_cpp": ["model-a"]},
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        summary = app.screen.query_one("#console-settings-reasoning-summary", Select)
        assert summary.parent is not None and summary.parent.display is True

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        provider_select.value = "llama_cpp"
        await pilot.pause()

        assert summary.parent is not None and summary.parent.display is False


@pytest.mark.asyncio
async def test_console_settings_modal_hides_only_authoritatively_unsupported_controls() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        reasoning_effort="high",
        reasoning_summary="auto",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()

        reasoning = app.screen.query_one("#console-settings-reasoning-effort", Select)
        budget = app.screen.query_one("#console-settings-thinking-budget-tokens", Input)
        summary = app.screen.query_one("#console-settings-reasoning-summary", Select)

        assert reasoning.parent is not None and reasoning.parent.display is True
        assert budget.parent is not None and budget.parent.display is True
        assert summary.parent is not None and summary.parent.display is False
        assert app.screen._is_effectively_focusable(summary) is False

        await pilot.click("#console-settings-save")

    # Hiding a no-effect control does not rewrite a retained session draft.
    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_summary == "auto"


@pytest.mark.asyncio
async def test_console_settings_modal_keeps_unknown_support_visible_with_neutral_copy() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="future-custom-model",
        reasoning_effort="high",
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={"openai": ["future-custom-model"]},
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        reasoning = app.screen.query_one("#console-settings-reasoning-effort", Select)
        note = app.screen.query_one(
            "#console-settings-reasoning-effort-support", Static
        )
        assert reasoning.parent is not None and reasoning.parent.display is True
        assert str(note.renderable) == "Support not verified for this model."
        assert note.display is True


@pytest.mark.asyncio
async def test_console_settings_modal_tab_order_skips_hidden_support_rows() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="local_vllm", model="model-a")

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={"local_vllm": ["model-a"]},
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        reasoning = app.screen.query_one("#console-settings-reasoning-effort", Select)
        reasoning.focus()
        visited: list[str | None] = []
        for _ in range(8):
            await pilot.press("tab")
            visited.append(getattr(app.focused, "id", None))

        assert "console-settings-reasoning-summary" not in visited
        assert "console-settings-verbosity" not in visited
        assert "console-settings-thinking-effort" not in visited
        assert "console-settings-thinking-budget-tokens" not in visited


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_hidden_invalid_value_without_rewriting_it() -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["local_vllm"] = {
        "api_url": "http://127.0.0.1:8000"
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_vllm": ["model-b"],
                },
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        budget = app.screen.query_one(
            "#console-settings-thinking-budget-tokens", Input
        )
        budget.value = "unfinished-budget"
        app.screen.query_one("#console-settings-provider", Select).value = "local_vllm"
        await pilot.pause()

        assert budget.value == "unfinished-budget"
        assert budget.parent is not None and budget.parent.display is False
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "local_vllm"
    assert app.saved_settings.thinking_budget_tokens == 2048


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_hidden_parseable_current_value() -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["local_vllm"] = {
        "api_url": "http://127.0.0.1:8000"
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_vllm": ["model-b"],
                },
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        budget = app.screen.query_one(
            "#console-settings-thinking-budget-tokens", Input
        )
        budget.value = "4096"
        app.screen.query_one("#console-settings-provider", Select).value = "local_vllm"
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.thinking_budget_tokens == 4096


@pytest.mark.asyncio
async def test_console_settings_modal_blank_name_returns_separate_none_override() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                user_display_name_override="Captain Rowan",
                global_user_display_name="Default Name",
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-user-display-name", Input).value = "   "
        await pilot.click("#console-settings-save")

    assert app.saved_result is not None
    assert app.saved_result.user_display_name_override is None
    assert app.saved_result.settings == app.saved_settings
    assert not hasattr(app.saved_result.settings, "user_display_name_override")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid_name", "expected_error"),
    [
        ("界" * 25, "Display name must fit within 48 terminal cells."),
        ("Captain\x07Rowan", "Display name cannot contain control characters."),
        ("Captain\u202eRowan", "Display name cannot contain control characters."),
    ],
)
async def test_console_settings_modal_invalid_name_prevents_dismissal(
    invalid_name, expected_error
) -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-user-display-name", Input
        ).value = invalid_name
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal")
        assert expected_error in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_result is None


@pytest.mark.asyncio
async def test_console_settings_validation_error_clears_on_edit() -> None:
    """TASK-363: a validation error must clear as soon as the user edits any
    field, not linger stale until the next Save."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        temperature = app.screen.query_one("#console-settings-temperature", Input)
        temperature.value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        error = app.screen.query_one("#console-settings-error", Static)
        assert "Temperature is required" in str(error.renderable)

        # Editing the field invalidates the stale summary immediately.
        temperature.value = "0.5"
        await pilot.pause()
        assert str(error.renderable).strip() == ""


@pytest.mark.asyncio
async def test_console_settings_error_summary_is_visually_distinct() -> None:
    """TASK-363: the validation summary must read as an error (bold, error
    colour), not near-body-text salience."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        error = app.screen.query_one("#console-settings-error", Static)
        assert "bold" in str(error.styles.text_style)


@pytest.mark.asyncio
async def test_console_settings_modal_single_model_uses_readonly_value_not_dead_dropdown() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)

        assert model_select.display is False
        assert model_select.disabled is True
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert model_custom.display is True
        assert model_custom.disabled is False

        model_custom.press()
        await pilot.pause()
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_custom.label == "Model list"

        model_custom.press()
        await pilot.pause()
        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert getattr(app.focused, "id", None) == "model-search-picker-input"


@pytest.mark.asyncio
async def test_console_settings_modal_saves_replaced_temperature_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed = False
        await pilot.pause()
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("ctrl+a")
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        assert temperature.value == "0.7"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.temperature == 0.7


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_focused_sampling_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed = False
        await pilot.pause()
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        await pilot.press("2")

        assert temperature.value == "0.72"


@pytest.mark.parametrize(
    ("field_id", "attribute", "backspace_count", "typed_suffix", "expected"),
    [
        ("console-settings-temperature", "temperature", 0, "1", 0.71),
        ("console-settings-top-p", "top_p", 1, "6", 0.96),
        ("console-settings-min-p", "min_p", 1, "6", 0.06),
        ("console-settings-top-k", "top_k", 1, "1", 41),
        ("console-settings-max-tokens", "max_tokens", 1, "5", 65),
    ],
)
@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_sampling_inputs(
    field_id: str,
    attribute: str,
    backspace_count: int,
    typed_suffix: str,
    expected: float | int,
) -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.70,
        top_p=0.95,
        min_p=0.05,
        top_k=40,
        max_tokens=64,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.click("#console-settings-generation-advanced CollapsibleTitle")
        await pilot.pause()
        target_input = app.screen.query_one(f"#{field_id}", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(target_input)
        await pilot.pause()
        await pilot.click(target_input)
        target_input.focus()
        await pilot.pause()
        await pilot.press("end")
        for _ in range(backspace_count):
            await pilot.press("backspace")
        await pilot.press(typed_suffix)
        assert str(expected) in target_input.value

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert getattr(app.saved_settings, attribute) == expected


@pytest.mark.asyncio
async def test_console_settings_modal_body_is_scrollable_container_for_overflow_controls() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 32)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        body = app.screen.query_one("#console-settings-body")
        assert isinstance(body, ScrollableContainer)


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_provider_specific_generation_controls() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed = False
        await pilot.pause()

        for selector in (
            "#console-settings-seed",
            "#console-settings-presence-penalty",
            "#console-settings-frequency-penalty",
        ):
            input_widget = app.screen.query_one(selector, Input)
            body = app.screen.query_one("#console-settings-body")
            body.scroll_to_widget(input_widget)
            await pilot.pause()

            assert input_widget.display is True
            assert input_widget.disabled is False
            assert input_widget.value
            assert input_widget.content_region.height >= 1

        for selector in (
            "#console-settings-reasoning-effort",
            "#console-settings-reasoning-summary",
            "#console-settings-verbosity",
        ):
            choice = app.screen.query_one(selector, Select)
            body = app.screen.query_one("#console-settings-body")
            body.scroll_to_widget(choice)
            await pilot.pause()

            assert choice.display is True
            assert choice.disabled is False
            assert choice.value is not Select.NULL
            assert choice.content_region.height >= 1

        for selector in (
            "#console-settings-thinking-effort",
            "#console-settings-thinking-budget-tokens",
        ):
            control_type = (
                Input if selector == "#console-settings-thinking-budget-tokens" else Select
            )
            input_widget = app.screen.query_one(selector, control_type)
            assert input_widget.parent is not None
            assert input_widget.parent.display is False

        app.screen.query_one("#console-settings-seed", Input).value = "23"
        app.screen.query_one("#console-settings-presence-penalty", Input).value = "0.6"
        app.screen.query_one("#console-settings-frequency-penalty", Input).value = "0.7"
        app.screen.query_one(
            "#console-settings-reasoning-effort", Select
        ).value = "medium"
        app.screen.query_one(
            "#console-settings-reasoning-summary", Select
        ).value = "concise"
        app.screen.query_one("#console-settings-verbosity", Select).value = "high"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.seed == 23
    assert app.saved_settings.presence_penalty == 0.6
    assert app.saved_settings.frequency_penalty == 0.7
    assert app.saved_settings.reasoning_effort == "medium"
    assert app.saved_settings.reasoning_summary == "concise"
    assert app.saved_settings.verbosity == "high"
    assert app.saved_settings.thinking_effort == "low"
    assert app.saved_settings.thinking_budget_tokens == 2048


@pytest.mark.asyncio
async def test_console_settings_modal_normalizes_provider_specific_choices() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        reasoning_effort="medium",
        reasoning_summary="concise",
        verbosity="low",
        thinking_effort="medium",
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one(
            "#console-settings-reasoning-effort", Select
        ).value = "high"
        app.screen.query_one(
            "#console-settings-reasoning-summary", Select
        ).value = "auto"
        app.screen.query_one("#console-settings-verbosity", Select).value = "medium"
        app.screen.query_one(
            "#console-settings-thinking-effort", Select
        ).value = "low"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_effort == "high"
    assert app.saved_settings.reasoning_summary == "auto"
    assert app.saved_settings.verbosity == "medium"
    assert app.saved_settings.thinking_effort == "low"


@pytest.mark.asyncio
async def test_console_settings_modal_shows_inherited_provider_endpoint() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url=None
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.display is True
        assert base_url_input.disabled is False
        assert base_url_input.value == "http://127.0.0.1:9099"


@pytest.mark.asyncio
async def test_console_settings_modal_prefers_api_base_url_alias_over_default_api_url() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url=None
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_stale_lower_priority_endpoint_with_alias() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://localhost:8080",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_focus_mode_uses_ready_copy_when_model_selected() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-connection")
        assert "Ready to send" in str(readiness.renderable)
        assert "Credential · Not required" in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_clears_setup_copy_when_dropdown_model_is_available() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": ["freeform-model"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-connection")
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        readiness_copy = str(readiness.renderable)
        assert "Choose a model to enable sending." not in readiness_copy
        assert "not wired yet" not in readiness_copy
        assert "Ready to send" in str(readiness.renderable)
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_setup_copy_uses_typed_blocker_precedence() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model=None,
        base_url="ftp://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        readiness_copy = str(readiness.renderable)
        assert "Not ready — invalid base URL" in readiness_copy
        assert "Model · Missing" in readiness_copy


@pytest.mark.asyncio
async def test_console_settings_modal_invalid_temperature_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = "3.0"
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature must be between 0 and 2." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_temperature_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", temperature=0.7
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature is required." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_top_p_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", top_p=0.95)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-top-p", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Top P is required." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_when_cannot_save() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=False,
            )
        )
        await pilot.pause()

        assert app.screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_settings_modal_has_stable_body_error_and_footer_regions() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        body = app.screen.query_one("#console-settings-body")
        error = app.screen.query_one("#console-settings-error", Static)
        actions = app.screen.query_one("#console-settings-actions")
        temperature = app.screen.query_one("#console-settings-temperature", Input)

        assert "console-settings-body" in body.classes
        assert "console-settings-error-summary" in error.classes
        assert "console-settings-modal-actions" in actions.classes
        assert "console-settings-control" in temperature.classes
        assert error.region.y < body.region.y < actions.region.y


@pytest.mark.asyncio
async def test_console_settings_modal_inputs_keep_visible_content_row_when_unfocused() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed = False
        await pilot.pause()

        for selector in (
            "#console-settings-base-url",
            "#console-settings-temperature",
            "#console-settings-top-p",
            "#console-settings-max-tokens",
        ):
            input_widget = app.screen.query_one(selector, Input)

            assert input_widget.value
            assert input_widget.content_region.height >= 1


@pytest.mark.asyncio
async def test_console_settings_modal_renders_context_and_single_identity_row() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        character_label="Ada",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    10,
                    4096,
                    "10 / 4k",
                    staged_source_count=2,
                    staged_context_summary="2 staged sources",
                ),
                can_save=True,
            )
        )
        await pilot.pause()

        text = _visible_text(app)
        assert "Current" in str(
            app.screen.query_one("#console-settings-context-current", Static).renderable
        )
        assert "10 / 4k tokens" in text
        assert "2 staged sources" in str(
            app.screen.query_one("#console-settings-context-sources", Static).renderable
        )
        assert "Estimate only; no truncation changes in this version." in str(
            app.screen.query_one("#console-settings-context-note", Static).renderable
        )
        assert "Character: Ada" in str(
            app.screen.query_one(
                "#console-settings-identity-current", Static
            ).renderable
        )
        assert not app.screen.query("#console-settings-persona-readonly")
        assert not app.screen.query("#console-settings-character-readonly")
        assert "User Profile" not in text
        assert "As:" not in text
        assert not app.screen.query("#console-settings-persona-input")
        assert not app.screen.query("#console-settings-character-input")


@pytest.mark.asyncio
async def test_console_settings_modal_provider_select_lists_all_configured_providers() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "openai": ["gpt-4"],
                    "custom": [],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_values = _select_values(
            app.screen.query_one("#console-settings-provider", Select)
        )
        assert {"custom", "llama_cpp", "openai"}.issubset(provider_values)


@pytest.mark.asyncio
async def test_console_settings_modal_uses_model_dropdown_without_configured_models() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert "freeform-model" in _select_values(model_select)
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_uses_first_model_when_initial_model_missing() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "gpt-4.1"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_keyboard_selects_model_from_dropdown() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1", "gpt-5"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_select.focus()
        await pilot.press("enter")
        assert model_select.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        assert model_select.expanded is False
        assert model_select.value == "gpt-5"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-5"


@pytest.mark.asyncio
async def test_console_settings_modal_searchable_provider_picker_preserves_drafts() -> None:
    """Replacing the Select must retain provider-scoped model and endpoint drafts."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        picker = app.screen.query_one(
            "#console-settings-provider-picker", ConsoleProviderPicker
        )
        picker_input = picker.query_one(
            "#console-settings-provider-picker-input", Input
        )
        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        base_url = app.screen.query_one("#console-settings-base-url", Input)
        assert picker.value == "llama_cpp"
        assert provider_select.value == "llama_cpp"
        assert provider_select.display is False
        assert provider_select.disabled is True
        assert provider_select.focusable is False
        assert model_select.value == "model-a"

        base_url.value = "http://llama-draft.invalid:9090"
        picker.focus_input()
        await pilot.pause()
        picker_input.value = "legacy"
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert provider_select.value == "local_llamacpp"
        assert model_select.disabled is False
        assert model_select.value == "local-model"

        base_url.value = "http://legacy-draft.invalid:9091"
        picker.focus_input()
        await pilot.pause()
        picker_input.value = "llama_cpp"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert picker.value == "llama_cpp"
        assert provider_select.value == "llama_cpp"
        assert model_select.value == "model-a"
        assert base_url.value == "http://llama-draft.invalid:9090"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_searchable_provider_picker_focus_round_trips_as_public_picker_target() -> None:
    """Nested picker focus must not serialize the hidden compatibility Select."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5"),
        app,
        providers_models={"openai": ["gpt-5"], "llama_cpp": ["model-a"]},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        picker = modal.query_one(ConsoleProviderPicker)
        picker.focus_input()
        await pilot.pause()

        snapshot = modal.capture_suspended_draft()
        assert snapshot.focus_control_id == "console-settings-provider-picker"
        assert snapshot.raw_values["console-settings-provider"] == "openai"


@pytest.mark.asyncio
async def test_console_settings_modal_tabs_to_model_picker_after_provider_change() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        provider_picker = app.screen.query_one(ConsoleProviderPicker)
        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )

        provider_picker.focus_input()
        await pilot.pause()
        provider_select.value = "groq"
        await pilot.pause()

        assert (
            app.screen.query_one("#console-settings-model-legacy-adapter").display
            is False
        )
        assert model_select.value == "llama-3.3-70b-versatile"
        assert picker.value == "llama-3.3-70b-versatile"

        await pilot.press("tab")
        await _wait_for_focused_id(
            app, pilot, "console-settings-configure-credential"
        )
        await pilot.press("tab")
        await _wait_for_focused_id(app, pilot, "model-search-picker-input")
        await pilot.press("8")
        await pilot.pause()

        assert app.screen.query_one("#model-search-picker-results", OptionList).display


@pytest.mark.asyncio
async def test_console_settings_modal_reopens_provider_picker_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one("#console-settings-temperature", Input)
        provider_picker = app.screen.query_one(ConsoleProviderPicker)

        temperature.focus()
        temperature.value = "0.22"
        await pilot.pause()

        provider_picker.focus_input()
        await pilot.pause()

        assert app.screen.query_one(
            "#console-settings-provider-picker-results", OptionList
        ).display


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_picker_click_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_input = app.screen.query_one(
            "#console-settings-provider-picker-input", Input
        )

        await pilot.click("#console-settings-temperature")
        temperature.value = "0.72"
        await pilot.pause()
        await pilot.click("#console-settings-provider-picker-input")

        assert provider_input.has_focus
        assert app.screen.query_one(
            "#console-settings-provider-picker-results", OptionList
        ).display


@pytest.mark.asyncio
async def test_console_settings_modal_opens_screen_routed_provider_picker_click_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_input = app.screen.query_one(
            "#console-settings-provider-picker-input", Input
        )

        temperature.focus()
        temperature.value = "0.72"
        await pilot.pause()

        provider_region = _settings_screen_region(provider_input)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)
        await pilot.pause()

        assert provider_input.has_focus


@pytest.mark.asyncio
async def test_console_settings_input_releases_mouse_capture_after_click_to_replace() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        temperature.capture_mouse()

        assert app.mouse_captured is temperature

        temperature.on_click()

        assert app.mouse_captured is None
        assert temperature.selected_text == temperature.value


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_picker_from_redirected_input_click(
    monkeypatch,
) -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_input = app.screen.query_one(
            "#console-settings-provider-picker-input", Input
        )
        temperature.capture_mouse()
        temperature.value = "0.22"

        provider_screen_region = provider_input.region.translate((10, 0))
        monkeypatch.setattr(
            Input,
            "screen_region",
            property(
                lambda widget: (
                    provider_screen_region
                    if widget is provider_input
                    else widget.region
                )
            ),
            raising=False,
        )
        click = events.Click(
            temperature,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_screen_region.x + provider_screen_region.width - 1,
            screen_y=provider_screen_region.y,
        )

        temperature.on_click(click)
        await pilot.pause()

        assert app.mouse_captured is None
        assert provider_input.has_focus


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_plain_select_click_without_redirected_input() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            provider_select,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert app.mouse_captured is None
        assert provider_select.expanded is False


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_screen_routed_select_click_without_input_focus() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        cancel_button = app.screen.query_one("#console-settings-cancel", Button)
        cancel_button.focus()
        await pilot.pause()
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert getattr(app.focused, "id", None) == "console-settings-cancel"
        assert app.mouse_captured is None
        assert provider_select.expanded is False


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_missing_registry_model_for_current_provider() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="custom-openai-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "custom-openai-model"
        assert {"custom-openai-model", "gpt-4.1"}.issubset(_select_values(model_select))
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "custom-openai-model"


@pytest.mark.asyncio
async def test_console_settings_modal_allows_manual_model_when_registry_has_stale_options() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["anthropic"] = {"api_key": "test-key"}
    settings = ConsoleSessionSettings(
        provider="anthropic", model="claude-3-haiku-20240307"
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"anthropic": ["claude-3-haiku-20240307"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        custom_button = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_select.display is True
        assert model_input.display is False
        assert custom_button.display is True

        custom_button.press()
        await pilot.pause()

        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is False
        model_input.value = "claude-haiku-4-5-20251001"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "anthropic"
    assert app.saved_settings.model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_console_settings_modal_uses_shared_picker_and_saves_search_result() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1", "gpt-5"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        legacy_adapter = app.screen.query_one("#console-settings-model-legacy-adapter")
        assert picker.display is True
        assert legacy_adapter.display is False

        search_input = picker.query_one("#model-search-picker-input", Input)
        search_input.value = "gpt-5"
        await pilot.pause()
        results = picker.query_one("#model-search-picker-results", OptionList)
        option_id = "model-provenance-option-0"
        option = results.get_option(option_id)
        option_index = results.get_option_index(option_id)
        assert option.disabled is False
        results.post_message(
            OptionList.OptionSelected(results, option, option_index)
        )
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "gpt-5"


@pytest.mark.asyncio
async def test_console_settings_modal_refreshes_readiness_after_returning_to_model_list() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one("#console-settings-model-custom", Button).press()
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        for _ in range(500):
            if picker.custom_mode:
                break
            await pilot.pause(0.01)
        assert picker.custom_mode is True
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-connection")
        model_input.value = ""
        # Debounced (task-15476): let the production `Input.Changed`
        # handler settle instead of forcing `_sync_readiness_display()`
        # directly, which raced the (now-delayed) handler-driven update.
        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.1)

        assert model_input.value == ""
        assert picker.value is None
        assert "Not ready — choose a model" in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section") is True
        )

        app.screen._toggle_manual_model_input()
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.value == "model-a"
        assert "Ready to send" in str(readiness.renderable)
        assert "Credential · Not required" in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_configured_provider_model() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://127.0.0.1:9099",
        "model": "gemma-local-config-model",
    }
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "gemma-local-config-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "gemma-local-config-model"
        assert base_url_input.value == "http://127.0.0.1:9099"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "gemma-local-config-model"
    assert app.saved_settings.base_url == "http://127.0.0.1:9099"


@pytest.mark.parametrize(
    ("provider_settings", "expected_model"),
    (
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "api_model": "gemma-api-model",
            },
            "gemma-api-model",
        ),
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "model": "None",
                "api_model": "null",
                "default_model": "gemma-default-model",
            },
            "gemma-default-model",
        ),
    ),
)
@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_model_alias_fallbacks(
    provider_settings: dict[str, str],
    expected_model: str,
) -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = provider_settings
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.value == expected_model
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == expected_model


@pytest.mark.asyncio
async def test_console_settings_modal_can_select_runtime_discovered_model_with_warning() -> (
    None
):
    assert save_setting_to_cli_config(
        "api_settings.openai",
        "api_key",
        "test-key",
    )
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model("gpt-4.1"),
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 60)) as pilot:
        console = host.screen_stack[-1]
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        model_select = modal_screen.query_one("#console-settings-model-select", Select)
        assert {"gpt-4.1", "gpt-5"}.issubset(_select_values(model_select))

        model_select.value = "gpt-5"
        await pilot.pause()
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _visible_console_settings_button(console, pilot)
        for _ in range(40):
            summary_text = _summary_text(console)
            if "Model · Selected — not verified at this endpoint" in summary_text:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"Console summary did not show discovered-model provenance: {summary_text}"
            )

        _settings, readiness = console._active_console_settings_readiness()
        assert readiness.native_send_supported is True


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_to_no_models_allows_freeform_model_entry() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        picker = app.screen.query_one("#console-settings-model-picker")
        picker_input = picker.query_one("#model-search-picker-input", Input)
        picker_status = picker.query_one("#model-search-picker-status", Static)
        custom_button = app.screen.query_one("#console-settings-model-custom", Button)
        assert picker.value is None
        assert "No models reported" in str(picker_status.renderable)
        assert custom_button.display is True
        assert custom_button.disabled is False

        app.screen.query_one("#console-settings-model-custom", Button).press()
        await pilot.pause()
        picker_input.value = "freeform-model"
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_freeform_model_input() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "koboldcpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "koboldcpp"
        await pilot.pause()

        app.screen.query_one("#console-settings-model-custom", Button).press()
        await pilot.pause()
        model_input = app.screen.query_one("#model-search-picker-input", Input)
        assert model_input.placeholder == "Choose or search models"

        await pilot.click(model_input)
        for character in "local-model":
            await pilot.press(character)
        assert model_input.value == "local-model"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "koboldcpp"
    assert app.saved_settings.model == "local-model"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_target_provider_model() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is False
        assert model_select.disabled is True
        assert model_select.value == "gpt-4.1"
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "gpt-4.1"
        assert "model-a" not in _select_values(model_select)
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        assert picker.value == "gpt-4.1"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="koboldcpp",
        model=None,
        base_url="http://localhost:5001/api/v1/generate",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "koboldcpp": ["None"],
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_existing_none_model_sentinel_is_not_saved() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="None",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_does_not_carry_base_url_to_non_url_provider() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.disabled is True or base_url_input.display is False
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.base_url is None


@pytest.mark.asyncio
async def test_console_settings_modal_restores_freeform_model_after_provider_round_trip() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": [], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert (
            app.screen.query_one("#console-settings-model-select", Select).value
            == "model-a"
        )

        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_inspector_hosts_staged_context_above_source_readiness() -> None:
    """The pinned preamble precedes staged and readiness content.

    Project status and next-send authority form the pinned preamble above the
    Inspector rail body. The staged-context tray follows the task-9
    Environment/Tasks sections as the body's third child, ahead of the run
    inspector and its source-readiness content.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        rail = console.query_one("#console-right-rail")
        rail_body = console.query_one("#console-inspector-rail-body")
        project_status = console.query_one("#console-project-instruction-status")
        run_inspector = console.query_one("#console-run-inspector")
        readiness = console.query_one("#console-live-work-source-readiness")
        live_work = console.query_one("#console-live-work-section")
        project_status = console.query_one("#console-project-instruction-status")
        authority = console.query_one("#console-send-authority-summary")
        left_rail = console.query_one("#console-left-rail")

        # The pinned preamble stays outside the scroll body. Within the body,
        # staged context precedes the run inspector and live-work card.
        assert settings.parent.id == "console-run-inspector"
        assert staged_context.parent is rail_body
        assert readiness in live_work.query("*")
        assert project_status.parent is rail
        assert authority.parent is rail
        assert project_status not in tuple(rail_body.query("*"))
        assert authority not in tuple(rail_body.query("*"))
        children = list(rail_body.children)
        # task-9: Environment and Tasks sections now precede the
        # staged-context tray.
        assert children[0].id == "console-environment-section"
        assert children[1].id == "console-tasks-section"
        assert children[2] is staged_context
        assert children.index(staged_context) < children.index(run_inspector)
        assert children.index(run_inspector) < children.index(live_work)

        # The left rail no longer hosts a Context section (header, body, or
        # tray): only Session, Model, Agent, and Details remain.
        assert not list(left_rail.query("#console-staged-context-tray"))
        assert not list(console.query("#console-rail-section-header-context"))
        assert not list(console.query("#console-rail-section-body-context"))

        # With the Inspector opened, the pinned preamble measures above staged
        # context, which remains above both source-readiness presentations.
        console.query_one("#console-inspector-rail-open", Button).press()
        await pilot.pause()
        readiness_heading = console.query_one(
            "#console-inspector-source-readiness-heading"
        )
        for _ in range(40):
            if all(
                widget.region.height > 0
                for widget in (
                    project_status,
                    authority,
                    staged_context,
                    readiness_heading,
                    readiness,
                )
            ):
                break
            await pilot.pause(0.05)
        assert project_status.region.y < authority.region.y
        assert authority.region.y < staged_context.region.y
        assert staged_context.region.y < readiness_heading.region.y
        assert staged_context.region.y < readiness.region.y


@pytest.mark.asyncio
async def test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary() -> (
    None
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        left_rail = console.query_one("#console-left-rail")
        header = console.query_one(".console-rail-header")
        body = console.query_one("#console-left-rail-body")
        conversations_body = console.query_one(
            "#console-rail-section-body-conversations"
        )
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert header.region.height == 1
        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= left_rail.region.height - header.region.height
        assert settings.parent.id == "console-run-inspector"
        # TASK-14810 keeps the durable conversation browser in its dedicated
        # Conversations disclosure section while the whole section stack
        # remains inside the fixed-header rail scroller.
        assert workspace_context.parent is conversations_body
        conversations_section = body.query_one(
            "#console-bounded-section-conversations", ConsoleBoundedSection
        )
        assert conversations_section.parent is body
        assert conversations_body.parent is conversations_section.viewport
        viewport_width = conversations_section.viewport.region.width
        assert workspace_context.region.width <= viewport_width
        assert viewport_width - workspace_context.region.width <= 2


@pytest.mark.asyncio
async def test_console_settings_modal_save_updates_active_summary_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _apply_open_console_settings_modal(
            modal_screen,
            pilot,
            provider="openai",
            model="gpt-4.1",
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        summary_text = _summary_text(console)
        assert "Provider: OpenAI" in summary_text
        assert "Model: gpt-4.1" in summary_text
        assert store.session_settings(second_id).provider == "openai"
        assert store.session_settings(first.id).provider == "llama_cpp"

        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        summary_text = _summary_text(console)
        assert "Provider: llama.cpp" in summary_text
        assert "Model: model-a" in summary_text


@pytest.mark.asyncio
async def test_console_settings_modal_result_stays_bound_to_opening_session(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "user_display_name": "User",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id,
            ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", system_prompt="First prompt"
            ),
        )
        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id,
            ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", system_prompt="Second prompt"
            ),
        )
        await console._sync_native_console_chat_ui()

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()
        modal_screen.query_one(
            "#console-settings-model-select", Select
        ).value = "gpt-4.1"
        modal_screen.query_one(
            "#console-settings-user-display-name", Input
        ).value = "Captain Rowan"
        store.switch_session(first.id)
        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.05)
        modal_screen.query_one("#console-settings-save", Button).press()
        await _wait_for_console_top_screen(host, console, pilot)
        await pilot.pause()

        assert store.active_session_id == first.id
        assert store.session_settings(first.id) == ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", system_prompt="First prompt"
        )
        second = next(
            session for session in store.sessions() if session.id == second_id
        )
        second_settings = store.session_settings(second_id)
        assert second_settings.provider == "openai"
        assert second_settings.model == "gpt-4.1"
        assert second_settings.system_prompt == "Second prompt"
        assert second_settings.source == "user"
        assert second.user_display_name_override == "Captain Rowan"


@pytest.mark.asyncio
async def test_console_settings_save_preserves_omitted_prompt_prefill_and_source() -> (
    None
):
    """The real general-settings draft preserves prompt-owned session fields."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                pinned_prefill="Keep this pinned prefill",
            ),
        )
        store.seed_character_roleplay(
            session.id,
            system_template="Protect {{user}}.",
            greeting_template="",
            global_default="User",
        )
        system_prompt_writes: list[str | None] = []
        roleplay_writes: list[str | None] = []
        store.persistence = SimpleNamespace(
            update_conversation_system_prompt=lambda **kwargs: (
                system_prompt_writes.append(kwargs["system_prompt"]) or True
            ),
            update_conversation_roleplay_context=lambda **kwargs: (
                roleplay_writes.append(kwargs["character_system_template"]) or True
            ),
        )
        session.persisted_conversation_id = "conv-1"
        assert session.character_system_template == "Protect {{user}}."

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.query_one("#console-settings-temperature", Input).value = "0.5"
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await pilot.pause()

        settings = store.session_settings(session.id)
        assert settings.temperature == 0.5
        assert settings.system_prompt == "Protect User."
        assert settings.pinned_prefill == "Keep this pinned prefill"
        assert session.character_system_template == "Protect {{user}}."
        assert system_prompt_writes == []
        assert roleplay_writes == []


def test_console_settings_result_applies_name_override_without_losing_prompt_source(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default Name"}
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    )
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="",
        global_default="Default Name",
    )
    session.persisted_conversation_id = "conv-1"
    store.persistence = SimpleNamespace(
        update_conversation_system_prompt=lambda **_kwargs: True,
        update_conversation_roleplay_context=lambda **_kwargs: False,
    )
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: coroutine.close(),
    )

    console._apply_console_settings_result(
        ConsoleSettingsResult(
            settings=ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", temperature=0.5
            ),
            user_display_name_override="Captain Rowan",
        )
    )

    assert store.session_settings(session.id).temperature == 0.5
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert session.character_system_template == "Protect {{user}}."
    assert session.user_display_name_override == "Captain Rowan"
    assert (
        "Name changed for this session, but it may not survive reopening.",
        "warning",
    ) in notifications


@pytest.mark.asyncio
async def test_console_global_name_refresh_coalesces_and_respects_session_override(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default One"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    inherited = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    store.seed_character_roleplay(
        inherited.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    overridden = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    store.seed_character_roleplay(
        overridden.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    store.set_session_user_display_name_override(
        overridden.id,
        "Captain Rowan",
        global_default="Default One",
    )
    queued = []
    surface_syncs = []
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        lambda: surface_syncs.append(store.active_session_id),
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
    store.switch_session(inherited.id)
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert surface_syncs == [inherited.id]
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert queued == []

    assert store.presentation_context(inherited.id, "Default Two").user_name == (
        "Default Two"
    )
    assert store.session_settings(inherited.id).system_prompt == "Protect Default Two."

    store.switch_session(overridden.id)
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert surface_syncs[-1] == overridden.id
    assert queued == []
    assert store.presentation_context(overridden.id, "Default Two").user_name == (
        "Captain Rowan"
    )
    assert store.session_settings(overridden.id).system_prompt == (
        "Protect Captain Rowan."
    )

    store.set_session_user_display_name_override(
        overridden.id,
        None,
        global_default="Default Two",
    )
    assert store.presentation_context(overridden.id, "Default Two").user_name == (
        "Default Two"
    )
    assert store.session_settings(overridden.id).system_prompt == "Protect Default Two."
    assert surface_syncs == [
        inherited.id,
        overridden.id,
    ]


def test_console_identity_refresh_request_dispatches_without_transcript_tick(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default One"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    assert greeting is not None
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
    assert console.request_console_identity_refresh(1) is True
    assert console.request_console_identity_refresh(1) is False

    assert store.session_settings(session.id).system_prompt == "Protect Default Two."
    assert store.get_message(greeting.id).content == "Hello Default Two."


@pytest.mark.asyncio
async def test_real_inactive_console_tab_activation_dispatches_identity_refresh(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "user_display_name": "Default One",
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        first.assistant_kind = "character"
        first.character_name = "Alraune"
        greeting = store.seed_character_roleplay(
            first.id,
            system_template="Protect {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Default One",
        )
        assert greeting is not None
        await _press_new_console_tab(console, store, pilot)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console,
            pilot,
            f"#console-session-tab-{first.id}",
        )
        dispatches = []
        original_dispatch = console._dispatch_active_console_roleplay_refresh

        def audited_dispatch():
            result = original_dispatch()
            dispatches.append(
                (
                    store.active_session_id,
                    console._global_chat_display_name(),
                    result,
                )
            )
            return result

        monkeypatch.setattr(
            console,
            "_dispatch_active_console_roleplay_refresh",
            audited_dispatch,
        )

        app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
        await _click_console_session_tab(console, store, pilot, first.id)
        for _ in range(40):
            if store.session_settings(first.id).system_prompt == "Protect Default Two.":
                break
            await pilot.pause(0.05)

        assert (first.id, "Default Two", True) in dispatches
        assert store.session_settings(first.id).system_prompt == "Protect Default Two."
        assert store.get_message(greeting.id).content == "Hello Default Two."


@pytest.mark.asyncio
async def test_console_roleplay_refresh_serializes_blocked_b_then_c_without_stale_win(
    monkeypatch,
) -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.system_started = threading.Event()
            self.release_system = threading.Event()
            self.durable_system = "Speak with User."
            self.durable_greeting = "Hello User."
            self.writer_threads: list[int] = []

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
            self.writer_threads.append(threading.get_ident())
            if system_prompt == "Speak with Bravo.":
                self.system_started.set()
                assert self.release_system.wait(5)
            self.durable_system = system_prompt
            return True

        def update_message_content(self, **kwargs):
            self.writer_threads.append(threading.get_ident())
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "User"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = BlockingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Speak with User.",
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="User",
    )
    assert greeting is not None
    controller = console._ensure_console_chat_controller()
    queued = []
    owner_thread = threading.get_ident()
    prepare_threads: list[int] = []
    original_prepare = store.prepare_session_roleplay_projection_refresh

    def audited_prepare(*args, **kwargs):
        prepare_threads.append(threading.get_ident())
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        store, "prepare_session_roleplay_projection_refresh", audited_prepare
    )
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Bravo"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert store.session_settings(session.id).system_prompt == "Speak with Bravo."
    assert store.get_message(greeting.id).content == "Hello Bravo."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Speak with Bravo.")
    assert provider_system.endswith("Hello Bravo.")
    assert await asyncio.to_thread(persistence.system_started.wait, 5)
    assert len(queued) == 1
    waiter = asyncio.create_task(queued.pop(0)())
    waiter.cancel()
    await asyncio.sleep(0)
    assert waiter.done() is True

    names = [f"Commander {index}" for index in range(25)]
    for name in names:
        app.app_config["chat_defaults"]["user_display_name"] = name
        assert console._dispatch_active_console_roleplay_refresh() is True
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert store.session_settings(session.id).system_prompt == (
        "Speak with Commander 24."
    )
    assert store.get_message(greeting.id).content == "Hello Commander 24."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Speak with Commander 24.")
    assert provider_system.endswith("Hello Commander 24.")
    assert queued == []
    drain = console._console_roleplay_persistence_task
    assert drain is not None
    assert console._console_roleplay_active_plan is not None
    assert console._console_roleplay_pending_plan is not None
    assert (
        console._console_roleplay_pending_plan.generation == session.identity_revision
    )
    assert persistence.durable_system == "Speak with User."

    persistence.release_system.set()
    await drain

    assert console._console_roleplay_persistence_task is None
    assert console._console_roleplay_active_plan is None
    assert console._console_roleplay_pending_plan is None
    assert persistence.durable_system == "Speak with Commander 24."
    assert persistence.durable_greeting == "Hello Commander 24."
    assert prepare_threads == [owner_thread] * 26
    assert persistence.writer_threads
    assert all(thread_id != owner_thread for thread_id in persistence.writer_threads)


@pytest.mark.asyncio
async def test_console_roleplay_refresh_skips_plan_stale_before_writer() -> None:
    class RecordingPersistence:
        def __init__(self) -> None:
            self.system_writes: list[str | None] = []
            self.message_writes: list[str] = []

        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.system_writes.append(kwargs["system_prompt"])
            return True

        def update_message_content(self, **kwargs):
            self.message_writes.append(kwargs["content"])
            return True

    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = RecordingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", system_prompt="Speak with Alpha."
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )
    assert greeting is not None
    persistence.system_writes.clear()
    persistence.message_writes.clear()
    plan_b = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Bravo"
    )
    plan_c = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Cecelia"
    )
    assert plan_b is not None and plan_c is not None
    console._sync_console_identity_surfaces = lambda: None

    await console._refresh_console_roleplay_projections(plan_b)
    assert persistence.system_writes == []
    assert persistence.message_writes == []

    await console._refresh_console_roleplay_projections(plan_c)
    assert persistence.system_writes == ["Speak with Cecelia."]
    assert persistence.message_writes == ["Hello Cecelia."]


@pytest.mark.asyncio
async def test_roleplay_writer_cleanup_waits_for_owner_acceptance() -> None:
    class Persistence:
        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **_kwargs):
            return True

        def update_message_content(self, **_kwargs):
            return True

    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    store.persistence = Persistence()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Speak with Alpha.",
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )
    assert greeting is not None
    plan = store.prepare_session_roleplay_projection_refresh(
        session.id,
        global_default="Bravo",
        force_persistence=True,
    )
    assert plan is not None
    result = store.persist_roleplay_projection_plan(plan)
    future = asyncio.get_running_loop().create_future()
    future.set_result(result)

    chat_screen_module._release_console_roleplay_transition_after_writer(
        future,
        store=store,
        plan=plan,
    )

    assert session.id in store._fork_source_transitions
    assert store.accept_roleplay_projection_persistence_result(result) is True
    await asyncio.sleep(0)
    assert session.id not in store._fork_source_transitions


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("constructor", "start"))
async def test_roleplay_writer_startup_failure_releases_fork_transition(
    monkeypatch,
    failure_point: str,
) -> None:
    class Persistence:
        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **_kwargs):
            return True

        def update_message_content(self, **_kwargs):
            return True

    class StartFailureThread:
        def __init__(self, **_kwargs) -> None:
            if failure_point == "constructor":
                raise RuntimeError("thread constructor failed")

        def start(self) -> None:
            raise RuntimeError("thread start failed")

    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    store.persistence = Persistence()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Speak with Alpha.",
        ),
        assistant_kind="character",
        assistant_id="1",
        character_id=1,
        character_name="Alraune",
        ephemeral=True,
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )
    assert greeting is not None
    plan = store.prepare_session_roleplay_projection_refresh(
        session.id,
        global_default="Bravo",
        force_persistence=True,
    )
    assert plan is not None
    monkeypatch.setattr(chat_screen_module.threading, "Thread", StartFailureThread)

    with pytest.raises(RuntimeError, match=f"thread {failure_point} failed"):
        await console._refresh_console_roleplay_projections(plan)

    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}
    assert store.fork_eligibility(greeting.id).eligible is True


@pytest.mark.asyncio
async def test_cancelled_unmounted_drain_finishes_latest_plan(
    monkeypatch,
) -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.started.set()
            assert self.release.wait(5)
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    console = ChatScreen(app)
    queued = []
    monkeypatch.setattr(
        console, "run_worker", lambda coroutine, **_kwargs: queued.append(coroutine)
    )
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    store = console._ensure_console_chat_store()
    persistence = BlockingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert await asyncio.to_thread(persistence.started.wait, 5)
    drain = console._console_roleplay_persistence_task
    assert drain is not None
    drain.cancel()
    await asyncio.sleep(0)
    assert drain.done() is False
    persistence.release.set()
    await drain

    assert console._console_roleplay_persistence_task is None
    assert console._console_roleplay_pending_plan is None
    assert persistence.durable_system == "Speak with Cecelia."
    assert persistence.durable_greeting == "Hello Cecelia."


@pytest.mark.asyncio
async def test_mounted_console_cancel_latest_waiter_keeps_durable_c() -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            value = kwargs["system_prompt"]
            if value == "Speak with Bravo.":
                self.started.set()
                assert self.release.wait(5)
            self.durable_system = value
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    host = ConsoleHarness(app)
    persistence = BlockingPersistence()

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.persistence = persistence
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        session.persisted_conversation_id = "conv-1"
        greeting = store.seed_character_roleplay(
            session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )
        assert greeting is not None

        app.app_config["chat_defaults"]["user_display_name"] = "Bravo"
        assert console._dispatch_active_console_roleplay_refresh() is True
        assert await asyncio.to_thread(persistence.started.wait, 5)
        app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
        assert console._dispatch_active_console_roleplay_refresh() is True
        console.workers.cancel_group(console, "console-roleplay-refresh")
        await pilot.pause(0.05)
        persistence.release.set()
        for _ in range(100):
            if console._console_roleplay_persistence_task is None:
                break
            await pilot.pause(0.02)

        assert console._console_roleplay_persistence_task is None
        assert console._console_roleplay_active_plan is None
        assert console._console_roleplay_pending_plan is None
        assert persistence.durable_system == "Speak with Cecelia."
        assert persistence.durable_greeting == "Hello Cecelia."


@pytest.mark.asyncio
async def test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume(
    monkeypatch: pytest.MonkeyPatch,
):
    class HungFirstWritePersistence:
        """One shared-store double: the FIRST system-prompt write blocks.

        task-16815 fixture correction: the Console runtime/store became
        app-owned (task-15860), so two co-mounted ChatScreens share ONE
        store -- the original per-screen persistence pair aliased to a
        single store and the repair write bound to the hung double
        (stack-verified 2026-08-16). One double now serves both roles:
        the hung screen's refresh write blocks until released; every
        later write (the app-level repair force-persist) records. The
        contract under test is unchanged: unmount bounds a stuck writer,
        and the repair persists the latest identity even while the
        original write is still blocked.
        """

        def __init__(self) -> None:
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.first_system_write_seen = False

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-base"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            first_write = not self.first_system_write_seen
            self.first_system_write_seen = True
            if first_write:
                self.started.set()
                try:
                    assert self.release.wait(10)
                    # The released write still applies its side effect, as
                    # the original HungPersistence did via super() -- a
                    # completed write must persist what it carried (Qodo
                    # review, PR #1726).
                    self.durable_system = kwargs["system_prompt"]
                    return True
                finally:
                    self.finished.set()
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    app.app_config.setdefault("console", {})[
        "roleplay_refresh_teardown_timeout_seconds"
    ] = 0.05
    host = ConsoleHarness(app)
    hung_persistence = HungFirstWritePersistence()

    async with host.run_test(size=(160, 48)) as pilot:
        resumed = host.screen_stack[-1]
        await _wait_for_selector(resumed, pilot, "#console-settings-summary")
        resumed_store = resumed._ensure_console_chat_store()
        resumed_store.persistence = hung_persistence
        resumed_session = resumed_store.ensure_session()
        resumed_session.settings = ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        )
        resumed_session.assistant_kind = "character"
        resumed_session.character_name = "Alraune"
        resumed_session.persisted_conversation_id = "conv-base"
        resumed_store.seed_character_roleplay(
            resumed_session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )

        hung = ChatScreen(app)
        await host.push_screen(hung)
        await _wait_for_selector(hung, pilot, "#console-settings-summary")
        hung_store = hung._ensure_console_chat_store()
        hung_session = hung_store.ensure_session()
        hung_session.settings = ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        )
        hung_session.assistant_kind = "character"
        hung_session.character_name = "Alraune"
        hung_session.persisted_conversation_id = "conv-hung"
        hung_store.seed_character_roleplay(
            hung_session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )

        app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
        assert hung._dispatch_active_console_roleplay_refresh() is True
        assert await asyncio.to_thread(hung_persistence.started.wait, 5)
        writer_task = hung._console_roleplay_writer_task
        assert writer_task is not None
        assert writer_task.done() is False
        old_screen = weakref.ref(hung)
        event_loop = asyncio.get_running_loop()
        loop_errors: list[dict[str, object]] = []
        previous_exception_handler = event_loop.get_exception_handler()
        event_loop.set_exception_handler(
            lambda _loop, context: loop_errors.append(context)
        )
        try:
            started_at = asyncio.get_running_loop().time()
            await host.pop_screen()

            elapsed = asyncio.get_running_loop().time() - started_at
            assert elapsed < 0.5
            assert app._console_roleplay_repair_generation == 1
            assert app._console_roleplay_repair_global_name == "Cecelia"
            hung_persistence.release.set()
            for _ in range(100):
                if (
                    getattr(
                        app,
                        "_console_roleplay_repair_consumed_generation",
                        0,
                    )
                    == 1
                    and hung_persistence.durable_system == "Speak with Cecelia."
                    and hung_persistence.durable_greeting == "Hello Cecelia."
                ):
                    break
                await pilot.pause(0.01)
            assert app._console_roleplay_repair_consumed_generation == 1
            assert host.screen_stack[-1] is resumed
            assert hung_persistence.durable_system == "Speak with Cecelia."
            assert hung_persistence.durable_greeting == "Hello Cecelia."

            del hung, hung_store, hung_session
            for _ in range(50):
                gc.collect()
                if old_screen() is None:
                    break
                await pilot.pause(0.01)
            assert old_screen() is None
        finally:
            hung_persistence.release.set()
            assert await asyncio.to_thread(hung_persistence.finished.wait, 5)
            await pilot.pause(0.05)
            event_loop.set_exception_handler(previous_exception_handler)
        assert loop_errors == []


@pytest.mark.asyncio
async def test_roleplay_repair_marker_retries_partial_then_consumes(monkeypatch):
    class PartialPersistence:
        def __init__(self) -> None:
            self.fail_messages = True
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            if self.fail_messages:
                return False
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Cecelia"}
    app._console_roleplay_repair_generation = 1
    app._console_roleplay_repair_global_name = "Cecelia"
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    persistence = PartialPersistence()
    store = console._ensure_console_chat_store()
    store.persistence = persistence
    session = store.ensure_session()
    session.settings = ConsoleSessionSettings(
        provider="llama_cpp", system_prompt="Speak with Cecelia."
    )
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Cecelia",
    )
    assert greeting is not None
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    queued = []
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )

    assert console._consume_pending_console_roleplay_repair() is True
    await queued.pop(0)()
    assert getattr(app, "_console_roleplay_repair_consumed_generation", 0) == 0
    assert console._console_roleplay_repair_inflight_generation == 0
    assert len([note for note in notifications if note[1] == "warning"]) == 1

    persistence.fail_messages = False
    assert console._consume_pending_console_roleplay_repair() is True
    await queued.pop(0)()
    assert app._console_roleplay_repair_consumed_generation == 1
    assert persistence.durable_system == "Speak with Cecelia."
    assert persistence.durable_greeting == "Hello Cecelia."
    assert len([note for note in notifications if note[1] == "warning"]) == 1


@pytest.mark.asyncio
async def test_console_global_name_refresh_failure_notifies_once(monkeypatch) -> None:
    class RefusingPersistence:
        def __init__(self) -> None:
            self.system_writes: list[str | None] = []
            self.message_writes: list[str] = []

        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.system_writes.append(kwargs["system_prompt"])
            return False

        def update_message_content(self, **kwargs):
            self.message_writes.append(kwargs["content"])
            return False

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default Name"}
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = RefusingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Protect Default Name.",
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default Name",
    )
    assert greeting is not None
    persistence.system_writes.clear()
    persistence.message_writes.clear()
    controller = console._ensure_console_chat_controller()
    queued = []
    estimate_calls = []

    def estimate(messages, provider, model, **kwargs):
        estimate_calls.append((messages, provider, model, kwargs["system_prompt"]))
        return ConsoleSettingsContextEstimate(
            used_tokens=17,
            token_limit=4096,
            label="17 / 4096 tokens",
        )

    monkeypatch.setattr(chat_screen_module, "build_console_context_estimate", estimate)
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Captain Rowan"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Protect Captain Rowan.")
    assert provider_system.endswith("Hello Captain Rowan.")
    estimate_result = console._active_console_settings_context_estimate()
    assert estimate_result.used_tokens == 17
    assert estimate_calls[-1][0][-1]["content"] == "Hello Captain Rowan."
    assert estimate_calls[-1][3] == "Protect Captain Rowan."
    await queued.pop(0)()

    expected = (
        "Your chat name is active, but updated character templates may not survive "
        "reopening."
    )
    assert notifications.count((expected, "warning")) == 1
    assert persistence.system_writes == ["Protect Captain Rowan."]
    assert persistence.message_writes == ["Hello Captain Rowan."]
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Protect Captain Rowan.")
    assert provider_system.endswith("Hello Captain Rowan.")
    assert store.active_session_id == session.id


@pytest.mark.asyncio
async def test_system_prompt_editor_clears_character_template_source() -> None:
    """The dedicated prompt editor owns explicit prompt replacement."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        store.seed_character_roleplay(
            session.id,
            system_template="Protect {{user}}.",
            greeting_template="",
            global_default="User",
        )

        console.run_worker(
            console._open_console_system_prompt_editor(), exclusive=False
        )
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(
            f"#{SYSTEM_PROMPT_TEXT_AREA_ID}", TextArea
        ).text = "Manual prompt."
        modal.query_one(f"#{SYSTEM_PROMPT_APPLY_BUTTON_ID}", Button).press()
        await pilot.pause(0.2)

        assert store.session_settings(session.id).system_prompt == "Manual prompt."
        assert session.character_system_template is None


def test_system_prompt_command_clears_character_template_through_store(
    monkeypatch,
) -> None:
    """The `/system` apply path must retain the store's provenance revocation."""
    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
    )
    store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="",
        global_default="User",
    )
    monkeypatch.setattr(console, "_sync_console_chat_core_state", lambda: None)
    monkeypatch.setattr(console, "_sync_console_settings_summary", lambda: None)
    monkeypatch.setattr(console, "_sync_console_control_bar", lambda: None)

    console._session._apply_console_session_system_prompt("Manual slash prompt.")

    assert store.session_settings(session.id).system_prompt == "Manual slash prompt."
    assert session.character_system_template is None


@pytest.mark.asyncio
async def test_console_settings_are_isolated_between_native_tabs(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _apply_open_console_settings_modal(
            modal_screen,
            pilot,
            provider="openai",
            model="gpt-4.1",
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "llama_cpp"
        await _click_console_session_tab(console, store, pilot, second_id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        assert console._build_console_provider_selection().provider == "openai"


@pytest.mark.asyncio
async def test_console_native_tab_click_switches_without_programmatic_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first_id}")

        first_tab = console.query_one(f"#console-session-tab-{first_id}", Button)
        assert await pilot.click(first_tab, offset=(1, 0))
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_tab() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-1")

        first_conversation = console.query_one(
            "#console-workspace-conversation-1", Button
        )
        assert (
            getattr(first_conversation, "conversation_id", None) == f"native:{first_id}"
        )
        first_conversation.press()
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


@pytest.mark.asyncio
async def test_console_provider_selection_includes_generation_controls() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="openai",
                model="gpt-4.1",
                seed=17,
                presence_penalty=0.4,
                frequency_penalty=0.5,
                reasoning_effort="high",
                reasoning_summary="auto",
                verbosity="medium",
                thinking_effort="low",
                thinking_budget_tokens=2048,
            ),
        )
        await console._sync_native_console_chat_ui()

        selection = console._build_console_provider_selection()

    assert selection.seed == 17
    assert selection.presence_penalty == 0.4
    assert selection.frequency_penalty == 0.5
    assert selection.reasoning_effort == "high"
    assert selection.reasoning_summary == "auto"
    assert selection.verbosity == "medium"
    assert selection.thinking_effort == "low"
    assert selection.thinking_budget_tokens == 2048


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_keeps_original_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        await _visible_console_settings_button(console, pilot)
        original_summary = _summary_text(console)

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(None)
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert _summary_text(console) == original_summary
        assert store.session_settings(session.id).provider == "llama_cpp"


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_during_active_run() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"}
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        assert modal_screen.query_one("#console-settings-save", Button).disabled is True
        readiness_copy = str(
            modal_screen.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Not ready — current run is active" in readiness_copy
        assert "Ready to send" not in readiness_copy

        controller._set_run_state(ConsoleRunState(ConsoleRunStatus.IDLE, "Ready."))
        await pilot.pause()

        # The modal owns one opening-time snapshot: ChatScreen has no mounted-
        # modal update seam. Keep status and Save consistently blocked until
        # the user closes and reopens after the run transition.
        assert modal_screen.query_one("#console-settings-save", Button).disabled is True
        assert "Not ready — current run is active" in str(
            modal_screen.query_one("#console-settings-readiness", Static).renderable
        )


@pytest.mark.asyncio
async def test_console_settings_save_clears_stale_terminal_run_status() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "custom": {
            "api_url": "http://localhost:1234/v1/chat/completions",
            "model": "custom-model-beta",
        },
    }
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "custom": ["custom-model-alpha", "custom-model-beta"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        stale_copy = "Provider blocked: old llama.cpp failure."
        controller._set_run_state(ConsoleRunState.blocked(stale_copy))
        console._sync_console_mode_bar()
        assert stale_copy in str(
            console.query_one("#console-mode-bar", Static).renderable
        )

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _apply_open_console_settings_modal(
            modal_screen,
            pilot,
            provider="custom",
            model="custom-model-beta",
            base_url="http://localhost:1234/v1/chat/completions",
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "custom"
        assert controller.run_state.status is ConsoleRunStatus.IDLE
        assert stale_copy not in str(
            console.query_one("#console-mode-bar", Static).renderable
        )


@pytest.mark.asyncio
async def test_console_send_blocker_uses_saved_unsupported_session_provider() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="wip_provider", model="test-model"),
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        for _ in range(40):
            if "Provider blocked" in _screen_visible_text(console):
                break
            await pilot.pause(0.05)

        assert console._console_send_blocked_reason() == (
            "Console send blocked: Finish provider setup before sending."
        )
        assert "wip_provider" not in console._console_send_blocked_reason()


@pytest.mark.asyncio
async def test_console_missing_model_opens_console_settings_from_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _visible_console_settings_button(console, pilot)
        # The shared Workbench recovery banner stays hidden — the setup
        # card's action button carries this recovery instead (Phase 2 spec,
        # section 2).
        await _wait_for_selector(console, pilot, "#console-setup-modal-action")

        recovery_button = console.query_one("#console-setup-modal-action", Button)
        assert str(recovery_button.label) == "Choose model"
        assert recovery_button.display is True

        recovery_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _wait_for_focused_id(host, pilot, "model-search-picker-input")

        assert (
            modal_screen.query_one("#console-settings-provider", Select).value
            == "llama_cpp"
        )
        assert modal_screen.query_one(ModelSearchPicker).value == "model-a"
        readiness = modal_screen.query_one("#console-settings-readiness", Static)
        provider_model_section = modal_screen.query_one("#console-settings-connection")
        assert "Ready to send" in str(readiness.renderable)
        assert "Credential · Not required" in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )

        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        text = _screen_visible_text(console)
        assert "Model: model-a" in _summary_text(console)
        assert "Setup required: choose a model in Console Settings." not in text
        assert console._console_send_blocked_reason() == ""


@pytest.mark.asyncio
async def test_console_llamacpp_saved_missing_model_blocks_before_send() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        send_button = console.query_one("#console-send-message", Button)
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        # TASK-2154.6 (FR-04): Send is now genuinely disabled while setup
        # blocks, so the press above is a no-op by design; the persistent
        # reason strip (plus the kept tooltip) is the pre-click affordance.
        assert send_button.disabled is True
        reason = console.query_one("#console-send-disabled-reason")
        assert reason.styles.display == "block"
        assert reason.renderable.plain == (
            "Send blocked — choose a model to continue ›"
        )
        assert (
            send_button.tooltip == "Choose a model in Console Settings before sending."
        )
        assert (
            "Console send blocked: Select a model before sending."
            not in _screen_visible_text(console)
        )
        assert (
            "Setup required: choose a model in Console Settings."
            not in _screen_visible_text(console)
        )
        assert composer.draft_text() == "hello"


def test_console_default_settings_keep_configured_model_without_legacy_model() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    settings = screen._session._default_console_session_settings()

    assert settings.provider == "llama_cpp"
    assert settings.model == "configured-model"


def test_console_settings_summary_uses_effective_config_endpoint_for_llamacpp_defaults() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.endpoint_row == "Endpoint: http://127.0.0.1:9099"


def test_console_readiness_uses_saved_session_settings_over_stale_global_provider() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099/v1", "model": "local-model"},
        "openai": {},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model="local-model")
    )

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    run_recipe_row = next(
        row for row in inspector_state.rows if row.label == "Run recipe"
    )

    assert screen._console_provider_blocker_copy() == ""
    assert control_state.provider_label == "Provider: llama_cpp"
    assert control_state.model_label == "Model: local-model"
    assert "llama.cpp / local-model" in run_recipe_row.value
    assert "llama_cpp" not in run_recipe_row.value
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""


def test_console_control_state_reads_persona_label_without_storing_it_on_session(
    monkeypatch,
) -> None:
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    monkeypatch.setattr(
        screen._session,
        "_active_native_console_session",
        lambda: SimpleNamespace(
            assistant_kind="persona",
            assistant_name="Guide",
            assistant_id="persona-7",
            library_policy_holder=session.library_policy_holder,
        ),
    )

    state = screen._build_console_control_state(None)

    assert state.assistant_label == "Persona: Guide"
    assert session.assistant_kind == "generic"
    assert session.assistant_id == "console"
    assert session.assistant_authority_id is None
    assert "assistant_kind" in session.__dataclass_fields__
    assert "assistant_id" in session.__dataclass_fields__
    assert "assistant_name" not in session.__dataclass_fields__


def test_console_saved_openai_with_key_shows_ready_readiness() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )

    summary_state = screen._build_console_settings_summary_state()
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    blocker_copy = screen._console_provider_blocker_copy()

    assert summary_state.readiness_label == ""
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""
    assert blocker_copy == ""
    assert screen._console_send_blocked_reason() == ""


def test_console_missing_key_recovery_action_is_provider_specific() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )

    label, target, tooltip = screen._console_provider_recovery_action()

    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: API key missing for OpenAI"
    )
    assert label == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
    assert target == "settings"
    assert tooltip == "Configure OpenAI API and API key in Settings"
    assert screen._console_provider_recovery_field() == "api_key"
    assert (
        screen._console_setup_blocked_reason()
        == "Add API key in Settings > Providers & Models before sending."
    )


def test_console_unsaved_generic_endpoint_blocks_with_safe_in_modal_recovery() -> (
    None
):
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434", "model": "llama3"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:9999/v1",
        ),
    )

    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    label, target, tooltip = screen._console_provider_recovery_action()

    assert provider_row.value == "blocked"
    assert provider_row.recovery == (
        "Provider setup needed: save the endpoint in Conversation settings"
    )
    assert "127.0.0.1" not in provider_row.recovery
    assert (
        "save the endpoint in Conversation settings"
        in screen._console_provider_blocker_copy()
    )
    assert label == "Configure endpoint"
    assert target == "console"
    assert tooltip == "Save the Ollama endpoint in Conversation settings"
    assert screen._console_provider_recovery_field() == "endpoint"
    assert (
        screen._console_setup_blocked_reason()
        == "Save provider endpoint in Conversation settings before sending."
    )


def test_console_no_provider_recovery_action_and_card_step_are_provider_actions() -> (
    None
):
    """FR-05/FR-07: no provider at all -> provider action, no empty '' copy."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {}
    app.app_config["api_settings"] = {}
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="", model=None)
    )

    label, target, tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: choose a provider"
    )
    assert label == "Choose provider"
    assert target == "console"
    assert screen._console_provider_recovery_field() == ""
    assert readiness.label == "Unknown"
    assert "Select a provider" in readiness.detail
    assert "''" not in readiness.detail
    assert card_state.mode == "card"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Choose a supported provider"
    assert "''" not in step_one.label
    assert step_two.state == "pending"


def test_console_missing_key_no_model_recovery_action_is_provider_action() -> None:
    """FR-05: with provider blocked AND model missing, the provider blocker wins."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model=None)
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Missing key"
    assert readiness.native_send_supported is False
    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: API key missing for OpenAI"
    )
    assert label == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
    assert target == "settings"
    assert screen._console_provider_recovery_field() == "api_key"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Connect a provider (API key or local server)"
    assert step_two.state == "pending"


def test_console_provider_ready_missing_model_keeps_choose_model_action() -> None:
    """FR-05 regression: provider ready + model missing -> Choose model, step 1 done."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Missing model"
    assert readiness.native_send_supported is False
    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: choose a model"
    )
    assert label == "Choose model"
    assert target == "console"
    assert screen._console_provider_recovery_field() == ""
    assert (
        screen._console_send_blocked_reason()
        == "Console send blocked: Select a model before sending."
    )
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "done"
    assert step_one.label == "Provider ready"
    assert step_two.state == "active"
    assert step_two.label == "Pick a model"


def test_console_unsaved_endpoint_no_model_recovery_action_is_configure_endpoint() -> (
    None
):
    """FR-05: unsaved endpoint + no model -> Configure endpoint, step 1 active."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model=None,
            base_url="http://127.0.0.1:9999/v1",
        ),
    )

    label, target, tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()

    assert (
        "save the endpoint in Conversation settings"
        in screen._console_provider_blocker_copy()
    )
    assert label == "Configure endpoint"
    assert target == "console"
    assert tooltip == "Save the Ollama endpoint in Conversation settings"
    assert screen._console_provider_recovery_field() == "endpoint"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Save the provider's server address (endpoint)"
    assert step_two.state == "pending"


def test_console_invalid_endpoint_no_model_recovery_action_is_configure_endpoint() -> (
    None
):
    """FR-05: invalid endpoint + no model -> Configure endpoint, step 1 active."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model=None,
            base_url="not-a-url",
        ),
    )

    label, target, tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Invalid URL"
    assert "invalid base URL" in screen._console_provider_blocker_copy()
    assert label == "Configure endpoint"
    assert target == "console"
    assert tooltip == "Configure the provider endpoint before sending"
    assert screen._console_provider_recovery_field() == "endpoint"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Save the provider's server address (endpoint)"
    assert step_two.state == "pending"


def test_console_saved_llamacpp_missing_model_summary_is_not_ready_without_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label == ""
    assert summary_state.provider_row == "Provider: llama.cpp"
    assert summary_state.model_row == "Model: Missing"
    assert (
        screen._console_send_blocked_reason()
        == "Console send blocked: Select a model before sending."
    )


def test_console_saved_llamacpp_missing_model_summary_ready_with_configured_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label == ""
    assert "Select a model before sending" not in summary_state.model_row


@pytest.mark.asyncio
async def test_console_new_native_tab_receives_default_settings_snapshot() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        settings = store.session_settings(second_id)
        assert settings is not None
        assert settings.provider == "llama_cpp"
        assert settings.model == "configured-model"


@pytest.mark.asyncio
async def test_console_new_native_tab_uses_saved_global_default() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "local-model",
        },
    }
    app.providers_models = {
        "openai": ["gpt-4.1"],
        "local_llamacpp": ["local-model"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        active_settings = ConsoleSessionSettings(
            provider="local_llamacpp",
            model="local-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.2,
            top_p=0.8,
            streaming=False,
        )
        store.replace_session_settings(first_id, active_settings)
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        new_settings = store.session_settings(second_id)
        assert new_settings is not None
        assert new_settings.provider == "openai"
        assert new_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_model_switch_inherits_selected_model_default_profile() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        initial_settings = store.session_settings(session.id)
        assert initial_settings is not None
        assert initial_settings.model == "gpt-4.1"
        assert initial_settings.temperature == 0.2

        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.45
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False


@pytest.mark.asyncio
async def test_console_model_switch_preserves_explicit_session_overrides() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        console._sync_compact_shell_controls(temperature="0.33")
        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.33
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False


# --- task-177: readiness must follow Settings saves without an app restart ---


def _disk_loaded_snapshot(**overrides) -> dict:
    """Snapshot shaped like a real ``load_settings()`` boot config."""
    snapshot = {
        "general": {},
        "logging": {},
        "splash_screen": {},
        "api_settings": {"openai": {"api_key": ""}},
    }
    snapshot.update(overrides)
    return snapshot


def test_provider_readiness_config_refreshes_disk_loaded_snapshot(monkeypatch) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)
    fresh = _disk_loaded_snapshot(api_settings={"openai": {"api_key": "sk-fresh"}})
    monkeypatch.setattr(chat_screen_module, "load_settings", lambda: fresh)

    assert console._provider_readiness_app_config() is fresh


def test_provider_readiness_config_honors_injected_test_snapshot(monkeypatch) -> None:
    """Fakes without the disk-loaded marker sections stay authoritative."""
    app = _build_test_app()
    app.app_config = {"api_settings": {"openai": {"api_key": "injected"}}}
    console = ChatScreen(app)

    def _fail_load_settings():
        raise AssertionError(
            "load_settings must not be consulted for injected snapshots"
        )

    monkeypatch.setattr(chat_screen_module, "load_settings", _fail_load_settings)

    assert console._provider_readiness_app_config() is app.app_config


def test_provider_readiness_config_falls_back_when_load_settings_fails(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)

    def _boom():
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(chat_screen_module, "load_settings", _boom)

    assert console._provider_readiness_app_config() is app.app_config


def test_console_readiness_unblocks_after_provider_save_without_restart(
    monkeypatch, tmp_path
) -> None:
    """Save a provider key via the config API after boot; readiness must see it.

    Mirrors the live UAT failure: Settings saved the key, the config module
    cache reloaded, but Console kept reading the boot-time ``app_config``
    snapshot until restart.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    config_path = tmp_path / "console-readiness-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _build_test_app()
        # Boot-time snapshot: disk-loaded shape, but captured before the save.
        app.app_config = _disk_loaded_snapshot()
        console = ChatScreen(app)
        settings = ConsoleSessionSettings(provider="openai", model="gpt-4o")

        readiness_before = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_before.native_send_supported is False

        # The Settings screen save path: config API write + cache reload.
        assert config_module.save_setting_to_cli_config(
            "api_settings.openai", "api_key", "sk-saved-after-boot"
        )

        readiness_after = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_after.native_send_supported is True
        assert readiness_after.label == "Ready"
        # The stale snapshot alone would still be blocked - proving the fresh
        # read (not the snapshot) unblocked readiness.
        readiness_stale = build_console_settings_readiness(
            settings,
            app_config=app.app_config,
            environ={},
        )
        assert readiness_stale.native_send_supported is False
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


# --- task-178: settings modal persistence affordance, boolean streaming, focus artifact ---


def _basic_modal(
    settings: ConsoleSessionSettings, app: "ModalHarness", **kwargs
) -> ConsoleSettingsModal:
    return ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models=kwargs.pop("providers_models", {"llama_cpp": ["model-a"]}),
        context_estimate=kwargs.pop(
            "context_estimate", ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
        ),
        can_save=kwargs.pop("can_save", True),
        **kwargs,
    )


def test_console_settings_modal_exposes_ctrl_enter_primary_binding() -> None:
    """The documented Apply accelerator must remain visible and deterministic."""
    binding = next(
        binding
        for binding in ConsoleSettingsModal.BINDINGS
        if getattr(binding, "key", None) == "ctrl+enter"
    )

    assert binding.action == "activate_primary"
    assert binding.description == "Apply"
    assert binding.show is True


@pytest.mark.asyncio
async def test_console_settings_modal_focus_order_starts_provider_ends_cancel_and_skips_collapsed() -> None:
    """Tab order is logical and never enters undisclosed advanced fields."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert app.focused.id == "console-settings-provider-picker-input"
        await pilot.press("shift+tab")
        assert app.focused.id == "console-settings-cancel"

        focused_ids: list[str | None] = []
        for _ in range(40):
            await pilot.press("tab")
            focused_ids.append(app.focused.id)
            if app.focused.id == "console-settings-cancel":
                break

        assert "console-settings-save" in focused_ids
        assert "console-settings-temperature" not in focused_ids
        assert "console-settings-user-display-name" not in focused_ids


@pytest.mark.asyncio
async def test_console_settings_keyboard_tab_leaves_provider_results_in_logical_order() -> None:
    """Tab from an open compound list advances instead of reopening Provider."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.press("down")
        assert app.focused.id == "console-settings-provider-picker-results"

        await pilot.press("tab")

        assert app.focused.id == "console-settings-base-url"


@pytest.mark.asyncio
async def test_console_settings_focus_moves_to_reason_when_primary_becomes_disabled() -> None:
    """A state transition cannot leave focus attached to a disabled primary."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        primary = modal.query_one("#console-settings-save", Button)
        primary.focus()
        await pilot.pause()
        modal._can_save = False
        modal._sync_completion_actions()
        await pilot.pause()

        reason = modal.query_one(
            "#console-settings-primary-disabled-reason", Static
        )
        assert primary.disabled
        assert reason.display
        assert app.focused is reason


@pytest.mark.asyncio
async def test_console_settings_focus_moves_to_unavailable_copy_when_test_hides() -> None:
    """Provider changes cannot strand focus on a hidden generation-test button."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        test_button = modal.query_one("#console-settings-test-generation", Button)
        test_button.focus()
        await pilot.pause()
        modal._active_provider = "local_onnx"
        modal._sync_generation_test_controls()
        await pilot.pause()

        unavailable = modal.query_one(
            "#console-settings-generation-unavailable", Static
        )
        assert not test_button.display
        assert unavailable.display
        assert app.focused is unavailable


@pytest.mark.asyncio
async def test_generation_confirmation_cancel_restores_visible_action_focus() -> None:
    """Canceling consent cannot leave focus inside its hidden container."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        action = modal.query_one("#console-settings-test-generation", Button)
        action.press()
        await pilot.pause()
        cancel = modal.query_one("#console-settings-cancel-generation", Button)
        cancel.focus()
        await pilot.pause()
        assert app.focused is cancel

        await pilot.press("enter")
        await pilot.pause()

        assert app.focused is action
        assert modal._is_effectively_focusable(app.focused)
        assert str(action.label) == "Test generation"


@pytest.mark.asyncio
async def test_generation_confirmation_confirm_restores_running_action_focus() -> None:
    """Starting the probe moves focus to its visible Cancel-test action."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def waiting_tester(_request):
        started.set()
        await release.wait()
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        app,
        generation_tester=waiting_tester,
    )

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            action = modal.query_one("#console-settings-test-generation", Button)
            action.press()
            await pilot.pause()
            confirm = modal.query_one("#console-settings-confirm-generation", Button)
            confirm.focus()
            await pilot.pause()
            assert app.focused is confirm

            await pilot.press("enter")
            await asyncio.wait_for(started.wait(), timeout=1)
            await pilot.pause()

            assert app.focused is action
            assert modal._is_effectively_focusable(app.focused)
            assert str(action.label) == "Cancel test"
            action.press()
            await pilot.pause()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_model_picker_keyboard_escape_restores_then_dismisses_modal() -> None:
    """Model Escape is two-stage: cancel picker state, then request safe close."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        picker = modal.query_one("#console-settings-model-picker", ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is modal
        assert picker.value == "model-a"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not modal
        assert app.saved_result is None


@pytest.mark.asyncio
async def test_console_settings_ctrl_enter_activates_enabled_primary_or_focuses_reason() -> None:
    """The shortcut applies only a usable primary and explains blocked drafts."""
    ready_app = ModalHarness()
    ready_modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), ready_app
    )
    async with ready_app.run_test(size=(120, 40)) as pilot:
        await ready_app.push_screen(
            ready_modal, callback=ready_app.capture_saved_settings
        )
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()
    assert ready_app.saved_settings is not None

    blocked_app = ModalHarness()
    blocked_modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        blocked_app,
        providers_models={"llama_cpp": []},
    )
    async with blocked_app.run_test(size=(120, 40)) as pilot:
        await blocked_app.push_screen(
            blocked_modal, callback=blocked_app.capture_saved_settings
        )
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()

        reason = blocked_modal.query_one(
            "#console-settings-primary-disabled-reason", Static
        )
        assert blocked_app.screen is blocked_modal
        assert blocked_app.saved_settings is None
        assert reason.display
        assert blocked_app.focused is reason


@pytest.mark.parametrize(
    ("guard_mode", "expected_focus_id"),
    [
        ("reset", "console-settings-close-undo"),
        ("compaction", "console-settings-close-anyway"),
    ],
)
@pytest.mark.asyncio
async def test_console_settings_ctrl_enter_respects_visible_close_guard(
    guard_mode: str,
    expected_focus_id: str,
) -> None:
    """Apply cannot bypass reset or running-compaction close choices."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        if guard_mode == "reset":
            modal._memory_reset_token = ("memory-1", 2)
        else:
            modal._compaction_provider_task = SimpleNamespace(done=lambda: False)
        modal._show_settings_close_guard(guard_mode)
        await pilot.pause()

        await pilot.press("ctrl+enter")
        await pilot.pause()

        guard = modal.query_one("#console-settings-close-guard")
        expected_focus = modal.query_one(f"#{expected_focus_id}", Button)
        assert app.screen is modal
        assert app.saved_result is None
        assert guard.display
        assert app.focused is expected_focus
        assert modal._is_effectively_focusable(expected_focus)


@pytest.mark.asyncio
async def test_console_settings_accessible_inputs_have_names_and_bounded_descriptions() -> None:
    """Every visible editable field has a stable name and keyboard help copy."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        visible_fields = [
            control
            for control in modal.query("Input, Select")
            if modal._is_effectively_focusable(control)
        ]
        assert visible_fields
        for control in visible_fields:
            assert control.tooltip
            assert len(str(control.tooltip)) <= 120

        for control in modal._settings_focus_targets():
            visible_label = str(getattr(control, "label", "")).strip()
            description = str(control.tooltip or "").strip()
            assert visible_label or description
            assert len(description) <= 160

        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Ready to send" in readiness
        assert "Endpoint · Not tested" in readiness
        assert "Generation · Not tested" in readiness


@pytest.mark.asyncio
async def test_current_verification_result_announcement_fires_once_without_markup(monkeypatch) -> None:
    """Each current terminal connection/generation result has one bounded notice."""
    async def connection_tester(_identity):
        return ProviderProbeResult("reachable", ("model-a",))

    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        connection_tester=connection_tester,
        generation_tester=generation_tester,
    )
    notices: list[tuple[str, dict[str, object]]] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **kwargs: notices.append((str(message), kwargs)),
        )

        modal.query_one("#console-settings-model-discover", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_connection_probe_token is None:
                break
        assert notices == [("Connection test succeeded; 1 model listed.", {"markup": False})]

        notices.clear()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_generation_probe_token is None:
                break
        assert notices == [("Generation test succeeded.", {"markup": False})]


@pytest.mark.parametrize("probe_kind", ["connection", "generation"])
@pytest.mark.asyncio
async def test_operational_tester_exception_announces_sanitized_terminal_failure(
    monkeypatch,
    probe_kind: str,
) -> None:
    """Caught tester failures remain valid current outcomes and announce once."""

    async def throwing_tester(_request):
        raise RuntimeError("PRIVATE-TESTER-EXCEPTION")

    kwargs = (
        {"connection_tester": throwing_tester}
        if probe_kind == "connection"
        else {"generation_tester": throwing_tester}
    )
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        **kwargs,
    )
    notices: list[tuple[str, dict[str, object]]] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **notify_kwargs: notices.append(
                (str(message), notify_kwargs)
            ),
        )

        if probe_kind == "connection":
            modal.query_one("#console-settings-model-discover", Button).press()
            for _ in range(20):
                await pilot.pause(0.05)
                if modal._active_connection_probe_token is None:
                    break
            expected = "Connection test failed: connection error."
        else:
            modal.query_one("#console-settings-test-generation", Button).press()
            modal.query_one("#console-settings-confirm-generation", Button).press()
            for _ in range(20):
                await pilot.pause(0.05)
                if modal._active_generation_probe_token is None:
                    break
            expected = "Generation test failed: provider error."

        assert notices == [(expected, {"markup": False})]
        assert "PRIVATE-TESTER-EXCEPTION" not in repr(notices)


@pytest.mark.parametrize(
    ("model_ids", "expected"),
    [
        ((), "Connection test succeeded; no models reported."),
        (("model-a",), "Connection test succeeded; 1 model listed."),
        (
            ("model-a", "model-b", "model-c"),
            "Connection test succeeded; 3 models listed.",
        ),
    ],
)
def test_connection_success_announcement_has_bounded_zero_one_many_copy(
    monkeypatch,
    model_ids: tuple[str, ...],
    expected: str,
) -> None:
    """Connection announcements describe counts without endpoint or model IDs."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )
    notices: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        modal,
        "notify",
        lambda message, **kwargs: notices.append((str(message), kwargs)),
    )

    modal._announce_verification_result(ProviderProbeResult("reachable", model_ids))

    assert notices == [(expected, {"markup": False})]


@pytest.mark.asyncio
async def test_stale_or_cancelled_verification_has_no_announcement(monkeypatch) -> None:
    """Revoked probe capabilities cannot emit late success announcements."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def connection_tester(_identity):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return ProviderProbeResult("reachable", ("stale-model",))

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        connection_tester=connection_tester,
    )
    notices: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )
        modal.query_one("#console-settings-model-discover", Button).press()
        await asyncio.wait_for(started.wait(), timeout=1)
        modal.query_one("#console-settings-base-url", Input).value = (
            "http://127.0.0.1:9199"
        )
        release.set()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_connection_probe_token is None:
                break

        assert notices == []


@pytest.mark.asyncio
async def test_console_settings_modal_streaming_cycles_inherit_and_boolean_values() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", streaming=False
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        toggle = app.screen.query_one("#console-settings-streaming", Button)
        assert str(toggle.label) == "Off"

        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "Inherit"

        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "On"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.streaming is True


@pytest.mark.asyncio
async def test_console_settings_modal_enumerated_inputs_list_accepted_values() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        # llama.cpp is a local thinking provider: only the reasoning-effort
        # level is consumed, so the other choice inputs carry the no-effect
        # suffix.
        placeholders = {
            "console-settings-reasoning-effort": "none, minimal, low, medium, high, xhigh",
            "console-settings-reasoning-summary": (
                "auto, concise, detailed, none" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
            "console-settings-verbosity": (
                "low, medium, high" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
            "console-settings-thinking-effort": (
                "off, low, medium, high, xhigh, max" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
        }
        for input_id, expected in placeholders.items():
            control = app.screen.query_one(f"#{input_id}", Select)
            accepted_copy = expected.removesuffix(PROVIDER_CHOICE_NO_EFFECT_SUFFIX)
            assert _select_ordered_values(control) == tuple(accepted_copy.split(", "))
            assert control.tooltip == expected


@pytest.mark.asyncio
async def test_console_settings_modal_scope_line_names_session_and_default_scopes() -> (
    None
):
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        CONSOLE_SETTINGS_SCOPE_COPY,
    )

    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        scope = app.screen.query_one("#console-settings-scope", Static)
        assert str(scope.renderable) == CONSOLE_SETTINGS_SCOPE_COPY
        assert "conversation" in CONSOLE_SETTINGS_SCOPE_COPY.lower()
        assert "future provider conversations" in CONSOLE_SETTINGS_SCOPE_COPY.lower()
        assert (
            str(app.screen.query_one("#console-settings-save-default", Button).label)
            == "Save as provider defaults"
        )
        response_control = app.screen.query_one("#console-settings-max-tokens", Input)
        response_label = response_control.parent.query_one(
            ".console-settings-modal-label",
            Static,
        )
        assert str(response_label.renderable) == "Response max tokens"


@pytest.mark.asyncio
async def test_console_settings_modal_body_scroll_container_is_not_focusable() -> None:
    """The focused scroll body painted stray focus-outline fragments ("|")
    through the section margins with the real app CSS; keeping it out of the
    focus chain removes the artifact and lands first focus on a real control."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        body = app.screen.query_one("#console-settings-body", ScrollableContainer)
        assert body.can_focus is False
        assert app.focused is not body


# --- task-177 live regression: REAL journey (boot -> Settings save -> Console) ---


def _build_live_config_test_app():
    """Real TldwCli booted against the REAL (test-sandboxed) config file.

    Unlike ``_build_test_app`` this does NOT stub ``load_settings`` /
    ``get_cli_setting``: ``app.app_config`` is the genuine template config from
    the sandbox ``TLDW_CONFIG_PATH``, so the disk-loaded snapshot path (and the
    stale-snapshot bug it guards against) is exercised end to end.
    """
    import tempfile
    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    user_data_dir = Path(
        tempfile.mkdtemp(prefix="tldw-chatbook-live-config-test-")
    ).resolve(strict=True)

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app._publish_runtime_policy_projection(context.state)
        return context

    with ExitStack() as stack:
        stack.enter_context(
            patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerNotesWorkspaceService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerCharacterPersonaService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_notes_service",
                lambda self, _user: setattr(self, "notes_service", None),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_prompts_service",
                lambda self: setattr(self, "prompts_service_initialized", False),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_providers_models",
                lambda self: setattr(self, "providers_models", {}),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_media_db",
                lambda self: (
                    setattr(self, "media_db", None),
                    setattr(self, "_media_types_for_ui", ["All Media"]),
                ),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.load_runtime_policy_for_app",
                side_effect=fake_runtime_policy,
            )
        )
        for db_path_getter in (
            "get_notifications_db_path",
            "get_research_db_path",
            "get_writing_db_path",
        ):
            stack.enter_context(
                patch(f"tldw_chatbook.app.{db_path_getter}", return_value=":memory:")
            )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_subscriptions_db_path",
                return_value=user_data_dir / "subscriptions.sqlite",
            )
        )
        stack.enter_context(
            patch("tldw_chatbook.app.get_user_data_dir", return_value=user_data_dir)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_workspaces_db_path",
                return_value=user_data_dir / "workspaces.sqlite",
            )
        )
        return TldwCli()


async def _wait_for_screen(app, pilot, screen_type_name: str, *, attempts: int = 250):
    for _ in range(attempts):
        if type(app.screen).__name__ == screen_type_name:
            return app.screen
        await pilot.pause(0.02)
    raise AssertionError(
        f"Never reached {screen_type_name}; current screen: {type(app.screen).__name__}"
    )


@pytest.mark.asyncio
async def test_real_journey_settings_save_unblocks_console_without_restart(
    monkeypatch,
) -> None:
    """Live-UAT regression: boot -> blocked Console -> Settings save -> Console.

    Mirrors the exact live failure: the Settings adapter saves
    chat_defaults.provider/model + the llama.cpp endpoint (config caches reload),
    the user clicks the Console nav tab (fresh ChatScreen composes, prior screen
    state restores), and the setup card must NOT still be blocking.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.Widgets.Console import ConsoleSetupModal

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    # Prime the sandbox template config and keep the boot fast/deterministic.
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert config_module.save_setting_to_cli_config("splash_screen", "enabled", False)
    assert config_module.save_setting_to_cli_config(
        "first_run", "setup_completed", True
    )
    config_module.load_settings(force_reload=True)

    app = _build_live_config_test_app()
    # Sanity: the boot snapshot must look disk-loaded (markers present) so the
    # fresh-config branch is the one under test.
    assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config)

    async with app.run_test(size=(180, 50)) as pilot:
        # 1) First-run landing: Console blocked on the template OpenAI default.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        assert console._build_console_setup_card_state().mode == "card"

        # 2) Leave Console (screen state, including session settings, is saved).
        app.post_message(NavigateToScreen("home"))
        await _wait_for_screen(app, pilot, "HomeScreen")

        # 3) The real Settings save path (same three values as the live run).
        adapter = SettingsConfigAdapter()
        assert adapter.save_values(
            "chat_defaults",
            {"provider": "llama_cpp", "model": "Qwen3-Coder-Test.gguf"},
        )
        assert adapter.save_values(
            "api_settings.llama_cpp",
            {"api_url": "http://127.0.0.1:9099"},
        )

        # 4) Back to Console: a fresh ChatScreen composes and restores state.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")

        card_state = console._build_console_setup_card_state()
        assert card_state.mode != "card", (
            "Setup card still blocking after a provider save; "
            f"steps={[(step.state, step.label) for step in card_state.steps]}"
        )
        settings, readiness = console._active_console_settings_readiness()
        assert settings.provider == "llama_cpp"
        assert readiness.native_send_supported is True

        # The blocking modal must clear once guidance syncs.
        for _ in range(100):
            modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
            if not modal.is_blocking:
                break
            await pilot.pause(0.02)
        assert not console.query_one(
            "#console-setup-modal", ConsoleSetupModal
        ).is_blocking


def test_console_stale_default_refresh_respects_user_marked_settings() -> None:
    """Blocked derived defaults refresh; explicit user selections never do."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()

    user_choice = ConsoleSessionSettings(
        provider="openai", model="gpt-4o", source="user"
    )
    store.replace_session_settings(session.id, user_choice)
    assert console._session._ensure_active_console_session_settings() == user_choice

    # A user-work marker is intentionally durable, so exercise the untouched
    # stale-default case in a separate canonical session.
    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    session = store.create_session(
        settings=stale_derived,
        canonical_settings_baseline=stale_derived,
    )
    refreshed = console._session._ensure_active_console_session_settings()
    assert refreshed.provider == "llama_cpp"
    assert refreshed.source == "derived"


def test_console_stale_default_refresh_respects_applied_system_prompt() -> None:
    """A stale-default refresh must not overwrite an applied `/system` prompt.

    The user-work provenance introduced on ``dev`` treats `/system` as an
    explicit session choice. Automatic refresh therefore leaves the whole
    settings snapshot intact, including its provider and prompt.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()

    # Blocked derived defaults (openai, no key) -- as if snapshotted on a
    # fresh, never-configured session -- with a `/system` prompt applied
    # before any message was sent.
    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    store.replace_session_settings(
        session.id,
        stale_derived,
        mark_user_work=False,
        canonical_settings_baseline=stale_derived,
    )
    store.set_session_system_prompt(session.id, "Be concise.")

    refreshed = console._session._ensure_active_console_session_settings()

    assert refreshed.provider == "openai"
    assert refreshed.system_prompt == "Be concise."
    # The store itself must carry the preserved prompt forward too, not just
    # the returned snapshot.
    assert store.session_settings(session.id).system_prompt == "Be concise."


# --- task-188/191: provider display names + Discover models -----------------


def _select_labels(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    labels: set[str] = set()
    for option in options:
        prompt = getattr(option, "prompt", None)
        if prompt is None and isinstance(option, tuple) and option:
            prompt = option[0]
        if prompt is not None:
            labels.add(str(getattr(prompt, "plain", prompt)))
    return labels


@pytest.mark.asyncio
async def test_console_settings_modal_provider_labels_use_catalog_display_names() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        labels = _select_labels(provider_select)
        values = _select_values(provider_select)

    # Labels render shared-catalog display names; values stay raw config keys.
    assert "llama.cpp" in labels
    assert "OpenAI" in labels
    assert "Ollama" in labels
    assert "llama_cpp" not in labels
    assert {"llama_cpp", "openai", "ollama"}.issubset(values)


class _RecordingProber:
    def __init__(self, result: LocalModelProbeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, base_url: str, provider_key: str) -> LocalModelProbeResult:
        self.calls.append((base_url, provider_key))
        return self.result


async def _wait_for_discover_status(app, pilot, fragment: str) -> Static:
    status = app.screen.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
    for _ in range(60):
        if fragment in str(status.renderable):
            return status
        await pilot.pause(0.05)
    raise AssertionError(
        f"discover status never showed {fragment!r}; last: {str(status.renderable)!r}"
    )


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_success_swaps_input_for_select() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=True,
            base_url="http://127.0.0.1:9099",
            model_ids=("srv-a", "srv-b"),
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await _wait_for_discover_status(
            app, pilot, "2 models listed"
        )

        assert prober.calls == [("http://127.0.0.1:9099", "llama_cpp")]
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.disabled is False
        assert _select_values(model_select) == {"srv-a", "srv-b"}
        assert model_select.value == "srv-a"
        # Free-text fallback stays available after discovery.
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_custom.display is True
        assert model_custom.disabled is False
        assert app.screen._current_model_value() == "srv-a"
        assert app.screen._selected_model_requires_confirmation() is False
        readiness = build_console_settings_readiness(
            app.screen._build_draft(),
            app_config=app.app_config,
            active_run=False,
        )
        assert (readiness.operability, readiness.blocker) == ("ready_to_send", None)
        assert app.screen.query_one("#console-settings-save", Button).disabled is False

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "srv-a"


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_failure_shows_inline_copy() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=False,
            base_url="http://127.0.0.1:9099",
            detail="No models endpoint at http://127.0.0.1:9099.",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert discover.tooltip == settings_modal_module.MODEL_DISCOVER_SCOPE_COPY
        discover.press()
        await _wait_for_discover_status(
            app, pilot, "Connection failed: connection error."
        )
        status = app.screen.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        assert "No models endpoint" not in str(status.renderable)

        # Honest inline line, button usable again, manual entry still works.
        assert discover.disabled is False
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_input.display is True
        assert model_input.disabled is False


@pytest.mark.asyncio
async def test_console_settings_modal_discover_button_only_for_url_based_providers() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert discover.display is False
        assert discover.disabled is True
        unsupported = app.screen.query_one(
            "#console-settings-model-discover-scope", Static
        )
        assert unsupported.display is True
        assert str(unsupported.renderable) == (
            "No non-billable live connection check is available for this provider."
        )

        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert discover.display is True
        assert discover.disabled is False
        assert str(discover.label) == "Test connection & list models"
        assert len(
            [
                button
                for button in app.screen.query(Button)
                if button.display
                and str(button.label) == "Test connection & list models"
            ]
        ) == 1


class _BlockingConnectionTester:
    def __init__(self, result: ProviderProbeResult) -> None:
        self.result = result
        self.calls: list[ProviderDraftIdentity] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def __call__(self, identity: ProviderDraftIdentity) -> ProviderProbeResult:
        self.calls.append(identity)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.result


class _ImmediateConnectionTester:
    def __init__(self, result: ProviderProbeResult) -> None:
        self.result = result

    async def __call__(self, _identity: ProviderDraftIdentity) -> ProviderProbeResult:
        return self.result


class _MalformedConnectionTester:
    async def __call__(self, _identity: ProviderDraftIdentity):
        return {"detail": "PRIVATE-CONNECTION-PAYLOAD"}


class _CancellationResistantConnectionTester(_BlockingConnectionTester):
    async def __call__(self, identity: ProviderDraftIdentity) -> ProviderProbeResult:
        self.calls.append(identity)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            return self.result
        return self.result


@pytest.mark.parametrize(
    ("edit_kind", "control_id", "new_value"),
    [
        ("streaming", None, None),
        ("choice", "console-settings-reasoning-effort", "high"),
        ("choice", "console-settings-reasoning-summary", "concise"),
        ("choice", "console-settings-verbosity", "high"),
        ("choice", "console-settings-thinking-effort", "high"),
        ("numeric", "console-settings-temperature", "0.3"),
    ],
)
@pytest.mark.asyncio
async def test_generation_edit_preserves_active_connection_probe_and_settlement(
    edit_kind: str,
    control_id: str | None,
    new_value: str | None,
) -> None:
    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    tester = _CancellationResistantConnectionTester(
        ProviderProbeResult("reachable", ("model-a",))
    )
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        connection_tester=tester,
        generation_tester=generation_tester,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Generation · Succeeded" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break

        action = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        action.press()
        await asyncio.wait_for(tester.started.wait(), 1)
        token = modal._active_connection_probe_token
        assert token is not None

        if edit_kind == "streaming":
            modal.query_one("#console-settings-streaming", Button).press()
        elif edit_kind == "choice":
            assert control_id is not None and new_value is not None
            modal.query_one(f"#{control_id}", Select).value = new_value
        else:
            assert control_id is not None and new_value is not None
            modal.query_one(f"#{control_id}", Input).value = new_value
        await pilot.pause()

        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert tester.cancelled is False
        assert modal._active_connection_probe_token is token
        assert action.disabled
        assert "Endpoint · Testing" in readiness
        assert "Generation · Changed since test" in readiness

        tester.release.set()
        await _wait_for_discover_status(app, pilot, "1 model listed")
        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert modal._active_connection_probe_token is None
        assert action.disabled is False
        assert "Endpoint · Testing" not in readiness
        assert "Endpoint · Reachable" in readiness
        assert "Generation · Changed since test" in readiness


@pytest.mark.asyncio
async def test_malformed_connection_tester_result_restores_bounded_action(
    monkeypatch,
) -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        connection_tester=_MalformedConnectionTester(),
    )

    notices: list[str] = []
    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )
        action = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        action.press()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_connection_probe_token is None:
                break

        rendered = str(
            modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static).renderable
        )
        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert modal._active_connection_probe_token is None
        assert action.disabled is False
        assert "Testing" not in readiness
        assert "Connection failed: connection error." in rendered
        assert "PRIVATE-CONNECTION-PAYLOAD" not in rendered
        assert notices == []


@pytest.mark.asyncio
async def test_console_connection_tester_uses_chat_catalog_and_returns_typed_result(
    monkeypatch,
) -> None:
    """The Console seam must not accidentally invoke TTS or generation traffic."""
    calls: list[tuple[str, str, object]] = []

    async def probe(endpoint: str, *, provider: str, purpose: object):
        calls.append((endpoint, provider, purpose))
        return SettingsEndpointProbeOutcome(
            state="reachable",
            summary="reachable (1 model)",
            model_ids=("served-model",),
        )

    monkeypatch.setattr(
        settings_endpoint_probe_module,
        "probe_settings_endpoint",
        probe,
    )
    identity = ProviderDraftIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", "http://127.0.0.1:9099"),
        credential_source="none",
        credential_revision=3,
        draft_generation=7,
    )

    result = await ChatScreen._test_console_connection(identity)

    assert result == ProviderProbeResult("reachable", ("served-model",))
    assert calls == [
        (
            "http://127.0.0.1:9099",
            "llama_cpp",
            SettingsEndpointProbePurpose.CHAT_CATALOG,
        )
    ]


@pytest.mark.asyncio
async def test_connection_probe_publishes_only_bounded_current_identity_model_evidence() -> None:
    """A current typed result must drive provenance without claiming generation."""
    app = ModalHarness()
    tester = _BlockingConnectionTester(
        ProviderProbeResult("reachable", ("served-model",))
    )
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model=None,
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": []},
        connection_tester=tester,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await tester.started.wait()

        readiness = modal.query_one("#console-settings-readiness", Static)
        assert "Endpoint · Testing…" in str(readiness.renderable)
        testing_evidence = modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        )
        assert testing_evidence is not None
        assert testing_evidence.endpoint == "testing"

        assert len(tester.calls) == 1
        identity = tester.calls[0]
        assert type(identity) is ProviderDraftIdentity
        assert identity.provider_key == "llama_cpp"
        assert identity.connection_identity == (
            "llama_cpp",
            "http://127.0.0.1:9099",
        )

        tester.release.set()
        await _wait_for_discover_status(app, pilot, "1 model listed")

        assert modal._current_model_value() == "served-model"
        assert modal.query_one(ModelSearchPicker).provenance_for_model(
            "served-model", provider="llama_cpp"
        ) == settings_modal_module.ConsoleModelProvenance.SERVED_NOW
        evidence = modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        )
        assert evidence is not None
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == ("served-model",)
        assert evidence.generation == "not_tested"


@pytest.mark.asyncio
async def test_endpoint_edit_immediately_removes_reachable_confirmation_from_readiness() -> None:
    """A changed endpoint must not leave prior reachable/confirmed UI visible."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="served-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["served-model"]},
        connection_tester=_ImmediateConnectionTester(
            ProviderProbeResult("reachable", ("served-model",))
        ),
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await _wait_for_discover_status(app, pilot, "1 model listed")

        readiness = modal.query_one("#console-settings-readiness", Static)
        assert "Endpoint · Reachable" in str(readiness.renderable)
        assert "Model · Confirmed" in str(readiness.renderable)

        modal.query_one("#console-settings-base-url", Input).value = (
            "http://127.0.0.1:9100"
        )
        await pilot.pause()

        rendered = str(readiness.renderable)
        assert "Endpoint · Changed since test" in rendered
        assert "Model · Changed since test" in rendered
        assert "Endpoint · Reachable" not in rendered
        assert "Model · Confirmed" not in rendered
        assert modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        ) is None
        cancelled_readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Endpoint · Changed since test" in cancelled_readiness
        assert "Endpoint · Testing…" not in cancelled_readiness
        assert "Endpoint · Reachable" not in cancelled_readiness


@pytest.mark.parametrize(
    "change_path",
    ("select", "input", "picker_selected", "picker_value"),
)
@pytest.mark.asyncio
async def test_model_change_cancels_probe_restores_action_and_rejects_late_result(
    change_path: str,
) -> None:
    """Every model edit path must revoke the probe before a late result settles."""
    app = ModalHarness()
    tester = _CancellationResistantConnectionTester(
        ProviderProbeResult("reachable", ("stale-model",))
    )
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["model-a", "model-b"]},
        connection_tester=tester,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        action = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        action.press()
        await tester.started.wait()
        assert action.disabled is True

        if change_path == "select":
            model_select = modal.query_one("#console-settings-model-select", Select)
            modal._model_select_changed(Select.Changed(model_select, "model-b"))
        elif change_path == "input":
            model_input = modal.query_one("#console-settings-model-input", Input)
            modal._model_input_changed(Input.Changed(model_input, "model-b"))
        elif change_path == "picker_selected":
            modal._model_picker_selected(ModelSearchPicker.ModelSelected("model-b"))
        else:
            modal._model_picker_value_changed(
                ModelSearchPicker.ModelValueChanged("model-b", custom=True)
            )

        for _ in range(40):
            if tester.cancelled:
                break
            await pilot.pause(0.01)

        assert tester.cancelled is True
        assert action.display is True
        assert action.disabled is False
        assert "stale-model" not in modal._current_discovered_model_ids
        assert "model listed" not in str(
            modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static).renderable
        )
        assert modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        ) is None
        cancelled_readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Endpoint · Not tested" in cancelled_readiness
        assert "Endpoint · Testing…" not in cancelled_readiness
        assert "Endpoint · Reachable" not in cancelled_readiness


@pytest.mark.parametrize(
    ("result", "expected_copy"),
    (
        (
            ProviderProbeResult("model_listing_unavailable", (), "http_status"),
            "Connection reached; model listing unavailable. Generation not tested.",
        ),
        (
            ProviderProbeResult("unreachable", (), "timeout"),
            "Connection failed: request timed out.",
        ),
        (
            ProviderProbeResult("unreachable", (), "connection_refused"),
            "Connection failed: connection refused.",
        ),
        (
            ProviderProbeResult("unreachable", (), "unauthorized"),
            "Connection failed: unauthorized.",
        ),
        (
            ProviderProbeResult("unreachable", (), "forbidden"),
            "Connection failed: forbidden.",
        ),
        (
            ProviderProbeResult("unreachable", (), "http_status"),
            "Connection failed: endpoint returned an HTTP error.",
        ),
        (
            ProviderProbeResult("unreachable", (), "invalid_payload"),
            "Connection failed: invalid models response.",
        ),
        (
            ProviderProbeResult("unreachable", (), "connection_error"),
            "Connection failed: connection error.",
        ),
    ),
)
@pytest.mark.asyncio
async def test_connection_probe_renders_only_bounded_terminal_outcomes(
    result: ProviderProbeResult,
    expected_copy: str,
) -> None:
    """Transport categories must render fixed copy and re-enable the action."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="saved-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["saved-model"]},
        connection_tester=_ImmediateConnectionTester(result),
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        action = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        action.press()
        await _wait_for_discover_status(app, pilot, expected_copy)

        assert action.disabled is False
        assert "127.0.0.1" not in str(
            modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static).renderable
        )


@pytest.mark.asyncio
async def test_connection_probe_is_cancelled_and_cannot_publish_after_endpoint_edit() -> None:
    """Editing the endpoint must cancel, not merely ignore, the obsolete request."""
    app = ModalHarness()
    tester = _BlockingConnectionTester(
        ProviderProbeResult("reachable", ("stale-model",))
    )
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="saved-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["saved-model"]},
        connection_tester=tester,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await tester.started.wait()

        modal.query_one("#console-settings-base-url", Input).value = (
            "http://127.0.0.1:9100"
        )
        for _ in range(40):
            if tester.cancelled:
                break
            await pilot.pause(0.01)

        assert tester.cancelled is True
        assert modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        ) is None
        assert "stale-model" not in modal._current_discovered_model_ids
        cancelled_readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Endpoint · Not tested" in cancelled_readiness
        assert "Endpoint · Testing…" not in cancelled_readiness
        assert "Endpoint · Reachable" not in cancelled_readiness

        endpoint = modal.query_one("#console-settings-base-url", Input)
        endpoint.value = "not a valid endpoint"
        await pilot.pause()
        action = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert action.display is False
        assert str(
            modal.query_one(
                "#console-settings-model-discover-scope", Static
            ).renderable
        ) == settings_modal_module.CONNECTION_PROBE_UNAVAILABLE_COPY

        endpoint.value = "http://127.0.0.1:9200"
        await pilot.pause()
        assert action.display is True
        assert action.disabled is False


@pytest.mark.asyncio
async def test_connection_probe_is_cancelled_when_modal_closes() -> None:
    """Closing the modal must cancel its in-flight network request."""
    app = ModalHarness()
    tester = _BlockingConnectionTester(
        ProviderProbeResult("reachable", ("late-model",))
    )
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="saved-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["saved-model"]},
        connection_tester=tester,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await tester.started.wait()

        modal.query_one("#console-settings-cancel", Button).press()
        for _ in range(40):
            if tester.cancelled:
                break
            await pilot.pause(0.01)

        assert tester.cancelled is True


@pytest.mark.asyncio
async def test_console_settings_modal_discover_rejects_invalid_endpoint_url() -> None:
    """PR #608 review: user-entered endpoint must pass shared URL validation
    before any network probe; the prober must never be called."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model=None,
        base_url="http://[not-a-valid-url",
    )
    prober = _RecordingProber(
        LocalModelProbeResult(ok=True, base_url="", model_ids=("x",))
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert discover.display is False
        assert discover.disabled is True
        unavailable = app.screen.query_one(
            "#console-settings-model-discover-scope", Static
        )
        assert str(unavailable.renderable) == (
            settings_modal_module.CONNECTION_PROBE_UNAVAILABLE_COPY
        )

    assert prober.calls == []


# --- Roleplay UAT: model discovery looked like it did nothing ---
# Live repro (origin/dev @ f384a2807): pressing "Discover models" produced no
# visible change. The status line ("Found 1 model at http://127.0.0.1:9099.")
# was composed BELOW the unrelated Base URL field, four rows from the button
# that produced it, and the discovered model was not selected -- the
# known-broken model stayed in the box. It read as a dead button.


def test_discovery_status_renders_next_to_the_discover_button() -> None:
    """Feedback must sit with the control that produced it, not below another field."""
    source = (
        Path(chat_screen_module.__file__).resolve().parents[2]
        / "Widgets"
        / "Console"
        / "console_settings_modal.py"
    )
    text = source.read_text()

    base_url_pos = text.index('id="console-settings-base-url"')
    model_pos = text.index('id="console-settings-model-picker"')
    actions_pos = text.index('id="console-settings-connection-actions"')
    status_pos = text.index("id=MODEL_DISCOVER_STATUS_ID,")
    readiness_pos = text.index('id="console-settings-readiness-panel"')

    assert base_url_pos < model_pos < actions_pos < status_pos < readiness_pos


@pytest.mark.asyncio
async def test_discovery_selects_the_model_when_exactly_one_is_found() -> None:
    """One discovered model must be selected, not left for the user to notice.

    Leaving the previous (often wrong) model selected after a successful
    discovery is what let a TTS model stay active on a chat endpoint.
    """
    app = ModalHarness()
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="stale-model", base_url="http://127.0.0.1:9099"
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["stale-model"]},
        context_estimate=ConsoleSettingsContextEstimate(
            used_tokens=10, token_limit=16384, label="10 / 16k"
        ),
        can_save=True,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._apply_model_discovery_result(
            "llama_cpp",
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("only-real-model",),
            ),
        )
        await pilot.pause()

        assert modal._current_model_value() == "only-real-model"
        picker = modal.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        await pilot.press("o", "n", "l", "y")
        await pilot.pause()
        results = modal.query_one("#model-search-picker-results", OptionList)
        assert [str(option.prompt) for option in results.options] == [
            "Custom / unverified",
            "only-real-model",
        ]


@pytest.mark.asyncio
async def test_single_model_discovery_refreshes_generation_control_support(
    monkeypatch,
) -> None:
    app = ModalHarness()
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="stale-model", base_url="http://127.0.0.1:9099"
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["stale-model"]},
        context_estimate=ConsoleSettingsContextEstimate(
            used_tokens=10, token_limit=16384, label="10 / 16k"
        ),
        can_save=True,
    )

    def model_dependent_support(_provider, model, control):
        if control == "verbosity" and model == "only-real-model":
            return "unsupported"
        return "unknown"

    monkeypatch.setattr(
        settings_modal_module,
        "console_generation_control_support",
        model_dependent_support,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        verbosity = modal.query_one("#console-settings-verbosity", Select)
        assert verbosity.parent is not None and verbosity.parent.display is True

        picker = modal.query_one(ModelSearchPicker)
        with (
            modal.prevent(Select.Changed, ModelSearchPicker.ModelValueChanged),
            picker.prevent(ModelSearchPicker.ModelValueChanged),
        ):
            modal._apply_model_discovery_result(
                "llama_cpp",
                LocalModelProbeResult(
                    ok=True,
                    base_url="http://127.0.0.1:9099",
                    model_ids=("only-real-model",),
                ),
            )
        await pilot.pause()

        assert verbosity.parent is not None and verbosity.parent.display is False


# --- task-30012.3: connection-first composition and deliberate disclosure ---


def _task_30012_suspended_modal_draft(
    *,
    focus_control_id: str,
    advanced_generation: bool,
    raw_values: dict[str, str | bool] | None = None,
) -> ConsoleSettingsDraftSnapshot:
    settings = ConsoleSessionSettings(provider="openai", model="gpt-5.6-terra")
    return ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={
            "console-settings-provider": "openai",
            "console-settings-model-picker": "gpt-5.6-terra",
            **(raw_values or {}),
        },
        provider_model_drafts={"openai": "gpt-5.6-terra"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id=focus_control_id,
        disclosure_state={
            "advanced_generation": advanced_generation,
            "connection_details": False,
        },
    )


@pytest.mark.asyncio
async def test_console_settings_modal_connection_first_hierarchy_and_title() -> None:
    """Moving connection controls below tuning would break setup scanning order."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert (
            str(modal.query_one(".console-modal-header", Static).renderable)
            == "Conversation settings"
        )
        section_ids = [
            section.id
            for section in modal.query(".console-settings-modal-section")
            if section.id is not None
        ]
        assert section_ids[:4] == [
            "console-settings-connection",
            "console-settings-generation-advanced",
            "console-settings-identity-advanced",
            "console-settings-request-estimate",
        ]

        connection = modal.query_one("#console-settings-connection")
        connection_ids = [
            widget.id for widget in connection.query("*") if widget.id is not None
        ]
        assert connection_ids.index("console-settings-provider-picker") < (
            connection_ids.index("console-settings-base-url")
        )
        assert connection_ids.index("console-settings-base-url") < (
            connection_ids.index("console-settings-model-picker")
        )
        assert connection_ids.index("console-settings-model-picker") < (
            connection_ids.index("console-settings-connection-actions")
        )
        assert connection_ids.index("console-settings-connection-actions") < (
            connection_ids.index("console-settings-readiness-panel")
        )
        assert modal.query_one("#console-settings-configure-credential").parent is (
            connection
        )


@pytest.mark.asyncio
async def test_console_settings_modal_new_and_blocked_disclosures_start_closed() -> None:
    """First-run tuning must not compete with the incomplete connection path."""
    app = ModalHarness()
    app.app_config["api_settings"]["openai"] = {}
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5.6-terra"),
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed is True
        assert modal.query_one(
            "#console-settings-identity-advanced", Collapsible
        ).collapsed is True
        assert modal.query_one(
            "#console-settings-request-estimate", Collapsible
        ).collapsed is True
        assert app.focused is modal.query_one(
            "#console-settings-configure-credential", Button
        )
        assert modal._is_effectively_focusable(
            modal.query_one("#console-settings-temperature", Input)
        ) is False


@pytest.mark.asyncio
async def test_console_settings_modal_setup_emphasis_clears_on_connection_when_ready() -> (
    None
):
    """The setup cue belongs to Connection and must clear after model recovery."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        app,
        providers_models={"llama_cpp": []},
        focus_model=True,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        connection = modal.query_one("#console-settings-connection")
        compatibility_wrapper = modal.query_one(
            "#console-settings-provider-model-section"
        )
        assert connection.has_class("console-settings-primary-section") is True
        assert (
            compatibility_wrapper.has_class("console-settings-primary-section")
            is False
        )

        modal.query_one("#console-settings-model-custom", Button).press()
        await pilot.pause()
        manual_model = modal.query_one("#console-settings-model-input", Input)
        manual_model.value = "model-a"
        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.1)

        assert connection.has_class("console-settings-primary-section") is False


@pytest.mark.asyncio
async def test_console_settings_modal_tab_order_skips_collapsed_disclosure_children() -> (
    None
):
    """Both traversal directions reach headers, never hidden descendants."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        discover = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        discover.focus()

        forward_parents: list[str | None] = []
        for _ in range(4):
            await pilot.press("tab")
            assert app.focused is not None
            forward_parents.append(getattr(app.focused.parent, "id", None))
        assert forward_parents == [
            "console-settings-connection",
            "console-settings-generation-advanced",
            "console-settings-identity-advanced",
            "console-settings-request-estimate",
        ]
        for control_id in (
            "console-settings-view-model",
            "console-settings-view-context",
            "console-settings-save-default",
            "console-settings-make-default",
            "console-settings-save",
            "console-settings-cancel",
        ):
            await pilot.press("tab")
            assert app.focused is modal.query_one(f"#{control_id}")

        for control_id in (
            "console-settings-save",
            "console-settings-make-default",
            "console-settings-save-default",
            "console-settings-view-context",
            "console-settings-view-model",
        ):
            await pilot.press("shift+tab")
            assert app.focused is modal.query_one(f"#{control_id}")

        reverse_parents: list[str | None] = []
        for _ in range(3):
            await pilot.press("shift+tab")
            assert app.focused is not None
            reverse_parents.append(getattr(app.focused.parent, "id", None))
        assert reverse_parents == [
            "console-settings-request-estimate",
            "console-settings-identity-advanced",
            "console-settings-generation-advanced",
        ]

        advanced = modal.query_one(
            "#console-settings-generation-advanced", Collapsible
        )
        await pilot.press("enter")
        await pilot.pause()
        assert advanced.collapsed is False
        await pilot.press("tab")
        assert app.focused is modal.query_one("#console-settings-temperature", Input)


@pytest.mark.asyncio
async def test_console_settings_modal_targeted_advanced_control_opens_disclosure() -> None:
    """A deep-linked advanced target must never restore into hidden content."""
    app = ModalHarness()
    snapshot = _task_30012_suspended_modal_draft(
        focus_control_id="console-settings-reasoning-effort",
        advanced_generation=False,
    )
    modal = _basic_modal(
        snapshot.settings,
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
        suspended_draft=snapshot,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        advanced = modal.query_one(
            "#console-settings-generation-advanced", Collapsible
        )
        target = modal.query_one("#console-settings-reasoning-effort", Select)
        assert advanced.collapsed is False
        assert app.focused is target


@pytest.mark.asyncio
async def test_console_settings_modal_restores_non_targeted_disclosure_snapshot() -> None:
    """Returning users keep the disclosure state they explicitly chose."""
    app = ModalHarness()
    snapshot = _task_30012_suspended_modal_draft(
        focus_control_id="console-settings-provider-picker",
        advanced_generation=True,
    )
    modal = _basic_modal(
        snapshot.settings,
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
        suspended_draft=snapshot,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one(
            "#console-settings-generation-advanced", Collapsible
        ).collapsed is False


@pytest.mark.asyncio
async def test_console_settings_modal_generation_choices_are_constrained_selects() -> None:
    """Free-text provider choices allow values the generation API rejects."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
    )
    modal = _basic_modal(
        settings,
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        expected_domains = {
            "console-settings-reasoning-effort": {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
            },
            "console-settings-reasoning-summary": {
                "auto",
                "concise",
                "detailed",
                "none",
            },
            "console-settings-verbosity": {"low", "medium", "high"},
            "console-settings-thinking-effort": {
                "off",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            },
        }
        for control_id, expected in expected_domains.items():
            control = modal.query_one(f"#{control_id}", Select)
            assert _select_values(control) == expected

        modal.query_one(
            "#console-settings-reasoning-effort", Select
        ).value = Select.NULL
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_effort is None


@pytest.mark.parametrize("terminal_size", [(120, 60), (80, 24)])
@pytest.mark.asyncio
async def test_console_settings_modal_invalid_restored_choice_stays_inline_until_fixed(
    terminal_size: tuple[int, int],
) -> None:
    """An obsolete saved choice must be visible and cannot be silently erased."""
    app = ModalHarness()
    snapshot = _task_30012_suspended_modal_draft(
        focus_control_id="console-settings-provider-picker",
        advanced_generation=True,
        raw_values={"console-settings-reasoning-effort": "obsolete-effort"},
    )
    modal = _basic_modal(
        snapshot.settings,
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
        suspended_draft=snapshot,
    )

    async with app.run_test(size=terminal_size) as pilot:
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        choice = modal.query_one("#console-settings-reasoning-effort", Select)
        validation = modal.query_one(
            "#console-settings-reasoning-effort-validation", Static
        )
        assert choice.value is Select.NULL
        assert "Saved value is unavailable" in str(validation.renderable)
        assert modal.capture_suspended_draft().raw_values[
            "console-settings-reasoning-effort"
        ] == "obsolete-effort"

        await pilot.click("#console-settings-save-default")
        assert app.screen is modal
        assert app.saved_settings is None
        assert "Saved value is unavailable" in str(validation.renderable)

        await pilot.click("#console-settings-save")
        assert app.screen is modal
        assert app.saved_settings is None

        choice.value = "low"
        await pilot.pause()
        assert str(validation.renderable) == ""
        modal.query_one("#console-settings-save", Button).press()
        await pilot.pause()

    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_effort == "low"


def test_console_settings_modal_discovery_uses_exact_model_count_copy() -> None:
    """Discovery status must not use a plural noun for a single model."""
    endpoint = "http://127.0.0.1:9099"

    assert (
        settings_modal_module._model_availability_copy(0, endpoint)
        == "No models reported"
    )
    assert (
        settings_modal_module._model_availability_copy(
            1,
            endpoint,
            selected_model="only-model",
        )
        == "1 model listed"
    )
    assert (
        settings_modal_module._model_availability_copy(2, endpoint)
        == "2 models listed"
    )


def test_console_discovery_identity_and_unverified_decision_are_exact_values() -> None:
    """Confirmation carries the provider, canonical endpoint, generation, and model."""
    identity = ConsoleModelDiscoveryIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", "http://127.0.0.1:9099"),
        draft_generation=7,
    )

    assert ConsoleUnverifiedModelDecision(
        identity=identity,
        model_id="custom-model",
    ) != ConsoleUnverifiedModelDecision(
        identity=replace(identity, draft_generation=8),
        model_id="custom-model",
    )
    assert ConsoleUnverifiedModelDecision(
        identity=identity,
        model_id="custom-model",
    ) != ConsoleUnverifiedModelDecision(
        identity=identity,
        model_id="other-model",
    )
    assert "http://127.0.0.1:9099" not in repr(identity)


@pytest.mark.asyncio
async def test_console_discovery_identity_is_canonical_and_each_request_is_monotonic() -> None:
    """Equivalent endpoint spellings compare canonically; rapid probes remain distinct."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="draft-model",
            base_url="HTTP://LOCALHOST:80/v1/",
        ),
        app,
        providers_models={"llama_cpp": ["draft-model"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        first = modal._begin_model_discovery_identity(
            "llama_cpp", "HTTP://LOCALHOST:80/v1/"
        )
        second = modal._begin_model_discovery_identity(
            "llama_cpp", "http://localhost"
        )

        assert first.connection_identity == ("llama_cpp", "http://localhost")
        assert second.connection_identity == first.connection_identity
        assert second.draft_generation == first.draft_generation + 1

        endpoint = modal.query_one("#console-settings-base-url", Input)
        endpoint.value = "http://localhost/"
        await pilot.pause()
        assert modal._model_discovery_generation == second.draft_generation

        endpoint.value = "http://localhost:81"
        await pilot.pause()
        assert modal._model_discovery_generation == second.draft_generation + 1


@pytest.mark.asyncio
async def test_stale_discovery_results_never_clear_or_overwrite_newer_state() -> None:
    """Provider, endpoint, and rapid-request races all fail closed."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="draft-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={
            "llama_cpp": ["draft-model"],
            "vllm": ["vllm-model"],
        },
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        old = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        newer = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            newer,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("new-model", "new-model-2"),
            ),
        )
        modal._apply_model_discovery_result(
            old,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("old-model",),
            ),
        )
        status = modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        assert str(status.renderable) == "2 models listed"
        assert modal._current_model_discovery_identity == newer

        endpoint = modal.query_one("#console-settings-base-url", Input)
        endpoint.value = "http://127.0.0.1:9100"
        await pilot.pause()
        modal._apply_model_discovery_result(
            newer,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("endpoint-stale",),
            ),
        )
        assert modal._current_model_discovery_identity is None

        prior_copy = str(status.renderable)
        modal._switch_provider("vllm")
        await pilot.pause(0.1)
        modal._switch_provider("llama_cpp")
        await pilot.pause(0.1)
        endpoint.value = "http://127.0.0.1:9099"
        await pilot.pause()
        modal._apply_model_discovery_result(
            newer,
            LocalModelProbeResult(
                ok=False,
                base_url="http://127.0.0.1:9099",
                detail="private upstream detail",
            ),
        )
        await pilot.pause()
        assert str(status.renderable) == prior_copy


@pytest.mark.asyncio
async def test_zero_model_discovery_requires_confirmation_for_existing_selection() -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="custom-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["custom-model"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=(),
            ),
        )
        await pilot.pause()

        assert str(
            modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static).renderable
        ) == "No models reported"
        assert modal._selected_model_requires_confirmation() is True
        assert modal.query_one(
            "#console-settings-keep-unverified-model", Button
        ).display is True


@pytest.mark.asyncio
async def test_zero_result_provenance_events_settle_and_accept_later_catalog_refresh() -> None:
    """A marker-free empty listing must not recursively classify its own overlay."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="custom-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["custom-model"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        for _ in range(40):
            if "llama_cpp" in modal._base_provenance_options:
                break
            await pilot.pause(0.01)
        for _ in range(3):
            await pilot.pause()
        promote_calls = 0
        original_promote = modal._promote_current_discovery_options

        def counted_promote() -> None:
            nonlocal promote_calls
            promote_calls += 1
            # Bound the red-case feedback cycle so the regression fails quickly.
            if promote_calls <= 5:
                original_promote()

        modal._promote_current_discovery_options = counted_promote
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=(),
            ),
        )
        for _ in range(5):
            await pilot.pause()

        assert 1 <= promote_calls <= 2
        settled_calls = promote_calls
        for _ in range(5):
            await pilot.pause()
        assert promote_calls == settled_calls

        refreshed = provider_model_resolution.ResolvedProviderModelOption(
            label="catalog-new",
            model_id="catalog-new",
            source="saved",
            capability_status="known",
            persisted=True,
            provenance=provider_model_resolution.ConsoleModelProvenance.SAVED_FALLBACK,
        )
        modal.query_one(ModelSearchPicker).set_provenance_options(
            "llama_cpp", (refreshed,)
        )
        await pilot.pause()

        assert promote_calls == settled_calls + 1
        assert [
            option.model_id
            for option in modal._base_provenance_options["llama_cpp"]
        ] == ["catalog-new"]


@pytest.mark.asyncio
async def test_unverified_model_requires_exact_secondary_confirmation() -> None:
    """A successful list that omits the selected model cannot silently complete."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="custom-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["custom-model"]},
    )
    fallback_calls = 0
    original_focus_fallback = modal._focus_connection_fallback

    def counted_focus_fallback() -> None:
        nonlocal fallback_calls
        fallback_calls += 1
        # Keep a re-entrant focus regression bounded so it reports an
        # assertion instead of starving the Textual event loop indefinitely.
        if fallback_calls <= 8:
            original_focus_fallback()

    modal._focus_connection_fallback = counted_focus_fallback

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("served-a", "served-b"),
            ),
        )
        await pilot.pause()

        keep = modal.query_one("#console-settings-keep-unverified-model", Button)
        use = modal.query_one("#console-settings-save", Button)
        assert keep.display is True
        assert keep.disabled is False
        assert use.disabled is True

        keep.focus()
        await pilot.pause()
        assert keep.has_focus is True
        keep.press()
        await pilot.pause()
        assert modal._unverified_model_decision == ConsoleUnverifiedModelDecision(
            identity=identity,
            model_id="custom-model",
        )
        assert keep.display is False
        assert use.disabled is False
        assert use.has_focus is True
        assert fallback_calls <= 3

        picker = modal.query_one(ModelSearchPicker)
        picker.set_custom_value("changed-model")
        modal._model_picker_value_changed(
            ModelSearchPicker.ModelValueChanged("changed-model", custom=True)
        )
        await pilot.pause()
        assert modal._unverified_model_decision is None
        assert modal._current_model_discovery_identity == (
            modal._current_draft_discovery_identity()
        )
        assert keep.display is True
        assert use.disabled is True

        keep.focus()
        keep.press()
        await pilot.pause()
        assert modal._unverified_model_decision == ConsoleUnverifiedModelDecision(
            identity=modal._current_draft_discovery_identity(),
            model_id="changed-model",
        )
        assert keep.display is False
        assert use.disabled is False
        assert use.has_focus is True
        assert fallback_calls <= 3


@pytest.mark.parametrize(
    "modal_guard",
    ({"active_run": True}, {"can_save": False}),
)
@pytest.mark.asyncio
async def test_unverified_confirmation_respects_completion_guards(modal_guard) -> None:
    """A guarded modal cannot record an exception or redirect to disabled primary."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="custom-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["custom-model"]},
        **modal_guard,
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("served-a", "served-b"),
            ),
        )
        await pilot.pause()

        keep = modal.query_one("#console-settings-keep-unverified-model", Button)
        use = modal.query_one("#console-settings-save", Button)
        assert keep.display is True
        assert keep.disabled is True
        keep.press()
        await pilot.pause()
        assert modal._unverified_model_decision is None
        assert use.disabled is True


@pytest.mark.asyncio
async def test_exact_identity_discovery_promotes_models_to_served_now_without_generation_claim() -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="old-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["old-model", "served-model"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        before_generation = modal._model_discovery_generation
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("served-model",),
            ),
        )
        await pilot.pause()

        picker = modal.query_one(ModelSearchPicker)
        assert modal._current_model_value() == "served-model"
        assert modal._model_discovery_generation == before_generation + 1
        assert modal._current_model_discovery_identity == (
            modal._current_draft_discovery_identity()
        )
        assert picker.provenance_for_model(
            "served-model", provider="llama_cpp"
        ) == settings_modal_module.ConsoleModelProvenance.SERVED_NOW
        provenance = modal.query_one("#console-settings-model-provenance", Static)
        assert str(provenance.renderable) == "Served by this endpoint now"
        assert "generation" not in str(provenance.renderable).lower()


@pytest.mark.asyncio
async def test_discovery_scope_is_visible_before_activation_and_persists_after_result() -> None:
    """The list-only scope must not be hidden in hover-only tooltip text."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="served-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["served-model"], "openai": ["gpt-4.1"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        scope = modal.query_one("#console-settings-model-discover-scope", Static)
        assert scope.display is True
        assert str(scope.renderable) == settings_modal_module.MODEL_DISCOVER_SCOPE_COPY

        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("served-model",),
            ),
        )
        await pilot.pause()
        assert str(scope.renderable) == settings_modal_module.MODEL_DISCOVER_SCOPE_COPY
        assert str(
            modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static).renderable
        ) == "1 model listed"

        modal._switch_provider("openai")
        await pilot.pause(0.1)
        assert scope.display is True
        assert str(scope.renderable) == (
            settings_modal_module.CONNECTION_PROBE_UNAVAILABLE_COPY
        )


@pytest.mark.asyncio
async def test_selecting_served_now_row_rebinds_listing_to_new_model_generation() -> None:
    """Choosing another listed row retains exact listing evidence and provenance."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="served-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["served-a", "served-b"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        identity = modal._begin_model_discovery_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        )
        modal._apply_model_discovery_result(
            identity,
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("served-a", "served-b"),
            ),
        )
        await pilot.pause()
        before_generation = modal._model_discovery_generation
        status = modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        assert str(status.renderable) == "2 models listed"
        picker = modal.query_one(ModelSearchPicker)
        picker.set_model_value("served-b")
        modal._model_picker_selected(ModelSearchPicker.ModelSelected("served-b"))
        await pilot.pause()

        assert modal._model_discovery_generation > before_generation
        assert modal._current_model_discovery_identity == (
            modal._current_draft_discovery_identity()
        )
        assert modal._current_discovered_model_ids == ("served-a", "served-b")
        assert str(status.renderable) == "2 models listed"
        assert picker.provenance_for_model(
            "served-b", provider="llama_cpp"
        ) == settings_modal_module.ConsoleModelProvenance.SERVED_NOW
        assert modal._selected_model_requires_confirmation() is False


@pytest.mark.asyncio
async def test_selecting_another_listed_model_rebinds_connection_evidence() -> None:
    """A model-only choice from the same result must retain endpoint evidence."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="served-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["served-a", "served-b"]},
        connection_tester=_ImmediateConnectionTester(
            ProviderProbeResult("reachable", ("served-a", "served-b"))
        ),
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await _wait_for_discover_status(app, pilot, "2 models listed")

        picker = modal.query_one(ModelSearchPicker)
        picker.set_model_value("served-b")
        modal._model_picker_selected(ModelSearchPicker.ModelSelected("served-b"))
        await pilot.pause()

        evidence = modal._connection_evidence_store.evidence_for(
            modal._current_connection_probe_identity()
        )
        assert evidence is not None
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == ("served-a", "served-b")
        assert evidence.generation == "not_tested"


@pytest.mark.asyncio
async def test_console_settings_modal_shows_current_catalog_model_provenance() -> None:
    """A selected catalog model must expose its source beside the control."""
    app = ModalHarness()
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (_merged_model("gpt-5.6-terra", source="runtime_discovered"),)
    )
    modal = _basic_modal(
        ConsoleSessionSettings(provider="openai", model="gpt-5.6-terra"),
        app,
        providers_models={"openai": ["gpt-5.6-terra"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        provenance = modal.query_one("#console-settings-model-provenance", Static)
        for _ in range(40):
            if str(provenance.renderable) == "Current provider catalog":
                break
            await pilot.pause(0.01)

        assert str(provenance.renderable) == "Current provider catalog"
        picker = modal.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        picker.query_one("#model-search-picker-input", Input).value = "gpt"
        await pilot.pause()
        assert [
            str(option.prompt)
            for option in modal.query_one(
                "#model-search-picker-results", OptionList
            ).options
        ] == ["Current catalog", "gpt-5.6-terra"]


@pytest.mark.asyncio
async def test_console_settings_modal_unfenced_probe_stays_custom_unverified() -> None:
    """Before identity fencing, a manual probe must not claim Served now."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="saved-model",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["saved-model"]},
    )

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._apply_model_discovery_result(
            "llama_cpp",
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("probe-model",),
            ),
        )
        await pilot.pause()

        provenance = modal.query_one("#console-settings-model-provenance", Static)
        assert str(provenance.renderable) == (
            "Custom model ID; generation not verified."
        )
        picker = modal.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        prompts = [
            str(option.prompt)
            for option in modal.query_one(
                "#model-search-picker-results", OptionList
            ).options
        ]
        assert "Served now" not in prompts
        assert "Custom / unverified" in prompts


def _visible_enabled_primary_buttons(modal: ConsoleSettingsModal) -> list[Button]:
    """Return the actions competing for primary visual hierarchy."""
    return [
        button
        for button in modal.query(Button)
        if button.display and not button.disabled and button.variant == "primary"
    ]


@pytest.mark.asyncio
async def test_console_settings_modal_ready_state_has_one_primary_action() -> None:
    """Navigation state must not compete with the one completion action."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    modal = _basic_modal(settings, app)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        primary = _visible_enabled_primary_buttons(modal)
        assert [button.id for button in primary] == ["console-settings-save"]
        assert str(primary[0].label) == "Use for this conversation"
        assert str(
            modal.query_one(
                "#console-settings-primary-disabled-reason", Static
            ).renderable
        ) == ""


@pytest.mark.parametrize(
    ("case", "settings", "expected_reason"),
    [
        (
            "missing credential",
            ConsoleSessionSettings(provider="openai", model="gpt-5.6-terra"),
            "Add or verify the OpenAI API key to continue.",
        ),
        (
            "invalid endpoint",
            ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", base_url="ftp://127.0.0.1:9099"
            ),
            "Enter a valid llama.cpp Base URL to continue.",
        ),
        (
            "missing model",
            ConsoleSessionSettings(
                provider="llama_cpp",
                model=None,
                base_url="http://127.0.0.1:9099",
            ),
            "Choose a model to continue.",
        ),
        (
            "active run",
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                base_url="http://127.0.0.1:9099",
            ),
            "Available when the current run finishes.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_console_settings_modal_disabled_completion_has_persistent_reason(
    case: str,
    settings: ConsoleSessionSettings,
    expected_reason: str,
) -> None:
    """Each readiness blocker must explain why completion is unavailable."""
    app = ModalHarness()
    if case == "missing credential":
        app.app_config["api_settings"]["openai"] = {}
    modal = _basic_modal(
        settings,
        app,
        providers_models={settings.provider: [settings.model] if settings.model else []},
        can_save=case != "active run",
        active_run=case == "active run",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        session_action = modal.query_one("#console-settings-save", Button)
        reason = modal.query_one("#console-settings-primary-disabled-reason", Static)
        assert session_action.disabled is True
        assert str(reason.renderable) == expected_reason
        assert reason.display is True


@pytest.mark.asyncio
async def test_console_settings_modal_context_operation_disables_completion_with_reason() -> None:
    """A context mutation cannot race modal completion without explanation."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    estimate = ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    context_state = replace(
        build_console_context_control_state(settings=settings, estimate=estimate),
        busy=True,
    )
    modal = _basic_modal(
        settings,
        app,
        context_estimate=estimate,
        context_state=context_state,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#console-settings-save", Button).disabled is True
        assert str(
            modal.query_one(
                "#console-settings-primary-disabled-reason", Static
            ).renderable
        ) == "Available when the current context operation finishes."


@pytest.mark.asyncio
async def test_console_settings_modal_default_action_only_tracks_changed_persisted_fields() -> None:
    """An unchanged default must not offer a redundant global mutation."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    app.app_config = {
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            }
        },
        "console": {
            "provider_defaults": {
                "llama_cpp": {"temperature": 0.7, "top_p": 0.95}
            }
        },
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "model-a",
            "streaming": True,
        },
    }
    modal = _basic_modal(settings, app)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        default_action = modal.query_one("#console-settings-save-default", Button)
        scope = modal.query_one("#console-settings-default-scope", Static)
        assert default_action.display is False
        assert str(scope.renderable) == ""

        modal.query_one("#console-settings-temperature", Input).value = "0.8"
        await pilot.pause()

        assert default_action.display is True
        assert default_action.variant == "default"
        assert str(default_action.label) == "Save as generation defaults"
        assert str(scope.renderable) == "Used by future conversations for llama.cpp."


@pytest.mark.asyncio
async def test_console_settings_modal_provider_default_action_names_provider_scope() -> None:
    """Provider/model persistence must name both its action and impact scope."""
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"]["model"] = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "streaming": True,
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-b",
        base_url="http://127.0.0.1:9099",
    )
    modal = _basic_modal(
        settings,
        app,
        providers_models={"llama_cpp": ["model-a", "model-b"]},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        default_action = modal.query_one("#console-settings-save-default", Button)
        assert default_action.display is True
        assert str(default_action.label) == "Save as provider defaults"
        assert str(
            modal.query_one("#console-settings-default-scope", Static).renderable
        ) == "Used by future conversations for llama.cpp."
        assert modal.query_one("#console-settings-default-scope", Static).display


@pytest.mark.asyncio
async def test_console_settings_modal_unsaved_endpoint_requires_provider_setup_write() -> None:
    """An unsaved endpoint never exposes a conversation-only completion action."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "ollama": {
                "api_url": "http://127.0.0.1:11434",
                "model": "qwen-old",
            }
        },
        "chat_defaults": {
            "provider": "ollama",
            "model": "qwen-old",
            "streaming": True,
        },
    }
    settings = ConsoleSessionSettings(
        provider="ollama",
        model="qwen-new",
        base_url="http://127.0.0.1:22434",
    )
    modal = _basic_modal(
        settings,
        app,
        providers_models={"ollama": ["qwen-old", "qwen-new"]},
    )
    committed: list[ConsoleSettingsCommittedSubmission | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal, callback=committed.append)
        await pilot.pause()

        session_action = modal.query_one("#console-settings-save", Button)
        default_action = modal.query_one("#console-settings-save-default", Button)
        assert session_action.disabled is True
        assert session_action.display is False
        assert default_action.display is True
        assert default_action.disabled is False
        assert str(default_action.label) == "Save endpoint & use model"
        assert _visible_enabled_primary_buttons(modal) == [default_action]
        disabled_reason = modal.query_one(
            "#console-settings-primary-disabled-reason", Static
        )
        assert disabled_reason.display is False
        assert str(disabled_reason.renderable) == ""
        assert str(
            modal.query_one("#console-settings-default-scope", Static).renderable
        ) == "Updates this provider for future conversations."

        default_action.press()
        await pilot.pause()

        assert len(committed) == 1
        result = committed[0]
        assert isinstance(result, ConsoleSettingsCommittedSubmission)
        assert result.submission.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        endpoint = result.submission.draft.endpoint_draft
        assert endpoint is not None
        assert endpoint.value == "http://127.0.0.1:22434"
        assert endpoint.checked is True


@pytest.mark.asyncio
async def test_console_settings_modal_active_run_reason_precedes_endpoint_recovery() -> None:
    """An active run remains the immediate blocker when endpoint recovery also exists."""
    app = ModalHarness()
    app.app_config = {
        "api_settings": {
            "ollama": {
                "api_url": "http://127.0.0.1:11434",
                "model": "qwen-old",
            }
        }
    }
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="ollama",
            model="qwen-new",
            base_url="http://127.0.0.1:22434",
        ),
        app,
        providers_models={"ollama": ["qwen-new"]},
        can_save=False,
        active_run=True,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert str(
            modal.query_one(
                "#console-settings-primary-disabled-reason", Static
            ).renderable
        ) == "Available when the current run finishes."


@pytest.mark.asyncio
async def test_console_settings_modal_exposes_selected_view_in_tab_copy_and_tooltips() -> None:
    """Selected destination must remain legible without adding a layout row."""
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"), app
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        model_view = modal.query_one("#console-settings-view-model", Button)
        context_view = modal.query_one("#console-settings-view-context", Button)
        assert list(modal.query("#console-settings-selected-view")) == []
        assert str(model_view.label) == "Model and generation · Selected"
        assert str(context_view.label) == "Context and memory"
        assert model_view.tooltip == "Selected view: Model and generation"
        assert context_view.tooltip == "Show Context and memory"

        context_view.press()
        await pilot.pause()

        assert str(model_view.label) == "Model and generation"
        assert str(context_view.label) == "Context and memory · Selected"
        assert model_view.tooltip == "Show Model and generation"
        assert context_view.tooltip == "Selected view: Context and memory"


def test_summary_builder_reports_only_genuine_provider_endpoint_inheritance() -> None:
    """Real readiness distinguishes a usable provider default from invalid setup."""
    estimate = ConsoleSettingsContextEstimate(None, None, "")
    inherited = ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    inherited_readiness = build_console_settings_readiness(
        inherited,
        app_config={"api_settings": {"openai": {"api_key": "test-key"}}},
        environ={},
    )
    invalid = ConsoleSessionSettings(
        provider="ollama", model="qwen", base_url="ftp://invalid-endpoint"
    )
    invalid_readiness = build_console_settings_readiness(
        invalid,
        app_config={
            "api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}
        },
        environ={},
    )
    absent = ConsoleSessionSettings(provider="", model="model-a")
    absent_readiness = build_console_settings_readiness(
        absent,
        app_config={"api_settings": {}},
        environ={},
    )

    assert (
        build_console_settings_summary_state(
            inherited, estimate, inherited_readiness
        ).endpoint_row
        == "Endpoint: Provider default"
    )
    assert (
        build_console_settings_summary_state(invalid, estimate, invalid_readiness).endpoint_row
        == "Endpoint: Not configured"
    )
    assert (
        build_console_settings_summary_state(absent, estimate, absent_readiness).endpoint_row
        == "Endpoint: Not configured"
    )


@pytest.mark.asyncio
async def test_console_settings_empty_completion_status_rows_consume_no_layout() -> None:
    """Empty default scope and blocker copy must not reduce the body viewport."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    app.app_config = {
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            }
        },
        "console": {
            "provider_defaults": {
                "llama_cpp": {"temperature": 0.7, "top_p": 0.95}
            }
        },
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "model-a",
            "streaming": True,
        },
    }
    modal = _basic_modal(settings, app)

    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        default_scope = modal.query_one("#console-settings-default-scope", Static)
        disabled_reason = modal.query_one(
            "#console-settings-primary-disabled-reason", Static
        )
        scope = modal.query_one("#console-settings-scope", Static)
        assert default_scope.display is False
        assert disabled_reason.display is False
        assert default_scope.region.height == 0
        assert disabled_reason.region.height == 0
        assert scope.region.height == 1
        assert len(str(scope.renderable)) <= scope.content_region.width


@pytest.mark.parametrize("terminal_size", [(80, 24), (100, 30), (160, 40)])
@pytest.mark.asyncio
async def test_console_settings_task4_geometry_keeps_connection_and_footer_usable(
    terminal_size: tuple[int, int],
) -> None:
    """Task 4 labels stay complete and reachable at the supported size matrix."""
    app = StyledModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
            temperature=0.6,
        ),
        app,
    )

    async with app.run_test(size=terminal_size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        frame = modal.query_one("#console-settings-modal")
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        connection = modal.query_one("#console-settings-connection")
        actions = modal.query_one("#console-settings-actions", Vertical)
        cancel = modal.query_one("#console-settings-cancel", Button)
        defaults = modal.query_one("#console-settings-save-default", Button)
        use = modal.query_one("#console-settings-save", Button)

        assert body.container_size.height >= 1
        assert connection.region.overlaps(body.content_region)
        assert body.max_scroll_x == 0
        assert actions.virtual_size.width <= actions.container_size.width
        assert actions.region.bottom <= frame.content_region.bottom
        assert actions.region.right <= frame.content_region.right
        assert str(cancel.label) == "Cancel"
        assert str(defaults.label) == "Save as provider defaults"
        assert str(use.label) == "Use for this conversation"
        assert defaults.region.width >= len(str(defaults.label))
        assert use.region.width >= len(str(use.label))


@pytest.mark.asyncio
async def test_generation_test_requires_fresh_confirmation_for_every_paid_request() -> None:
    calls = []

    async def tester(request):
        calls.append(request)
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        test_button = modal.query_one("#console-settings-test-generation", Button)
        confirmation = modal.query_one("#console-settings-generation-confirmation")

        test_button.press()
        await pilot.pause()
        assert calls == []
        assert confirmation.display
        assert "may incur provider charges" in str(
            modal.query_one("#console-settings-generation-consent-copy", Static).renderable
        )

        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if calls:
                break
        assert len(calls) == 1
        assert confirmation.display is False
        assert calls[0].settings.model == "model-a"
        assert "Generation · Succeeded" in str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )

        test_button.press()
        await pilot.pause()
        assert len(calls) == 1
        assert confirmation.display


@pytest.mark.parametrize(
    "change_path", ("select", "input", "picker_selected", "picker_value")
)
@pytest.mark.asyncio
async def test_generation_test_model_edit_cancels_and_rejects_late_result(
    change_path: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_tester(_request):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={"llama_cpp": ["model-a", "model-b"]},
        generation_tester=cancellation_resistant_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        await asyncio.wait_for(entered.wait(), 1)

        if change_path == "select":
            model_select = modal.query_one("#console-settings-model-select", Select)
            modal._model_select_changed(Select.Changed(model_select, "model-b"))
        elif change_path == "input":
            modal.query_one("#console-settings-model-input", Input).value = "model-b"
        elif change_path == "picker_selected":
            modal._model_picker_selected(ModelSearchPicker.ModelSelected("model-b"))
        else:
            modal._model_picker_value_changed(
                ModelSearchPicker.ModelValueChanged("model-b", custom=True)
            )
        await pilot.pause()
        button = modal.query_one("#console-settings-test-generation", Button)
        assert str(button.label) == "Test generation"
        assert button.disabled is False
        assert "Testing generation" not in str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        cancellation_copy = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert "Stopped waiting" in cancellation_copy
        assert "may continue" in cancellation_copy
        assert "may still be billed" in cancellation_copy
        assert "Generation · Succeeded" not in str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )

        release.set()
        await pilot.pause(0.1)
        assert "Generation · Succeeded" not in str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )


@pytest.mark.parametrize(
    "change_path", ("provider", "endpoint", "generation", "connection_test")
)
@pytest.mark.asyncio
async def test_implicit_generation_cancellation_keeps_billing_warning_and_fences_late_result(
    change_path: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_tester(_request):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    async def connection_tester(_identity):
        return settings_modal_module.ProviderProbeResult("reachable", ("model-a",))

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={
            "llama_cpp": ["model-a"],
            "openai": ["gpt-4.1"],
        },
        generation_tester=cancellation_resistant_tester,
        connection_tester=connection_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        await asyncio.wait_for(entered.wait(), 1)

        if change_path == "provider":
            modal._switch_provider("openai")
        elif change_path == "endpoint":
            modal.query_one("#console-settings-base-url", Input).value = (
                "http://127.0.0.1:9199"
            )
        elif change_path == "generation":
            modal.query_one("#console-settings-streaming", Button).press()
        else:
            modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await pilot.pause()

        status = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert modal._active_generation_probe_token is None
        assert "Stopped waiting" in status
        assert "may continue" in status
        assert "may still be billed" in status
        assert "Generation · Succeeded" not in str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )

        release.set()
        await pilot.pause(0.1)
        status = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert "succeeded" not in status.lower()
        assert "may still be billed" in status


@pytest.mark.asyncio
async def test_generation_test_cancel_action_and_close_cancel_active_request() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def waiting_tester(_request):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    modal = _basic_modal(settings, app, generation_tester=waiting_tester)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        button = modal.query_one("#console-settings-test-generation", Button)
        button.press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        await asyncio.wait_for(entered.wait(), 1)
        assert str(button.label) == "Cancel test"
        running_copy = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert "Cancel stops waiting" in running_copy
        assert "may continue" in running_copy
        assert "may still be billed" in running_copy

        button.press()
        await asyncio.wait_for(cancelled.wait(), 1)
        await pilot.pause()
        assert str(button.label) == "Test generation"
        cancel_copy = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert "Stopped waiting" in cancel_copy
        assert "may continue" in cancel_copy
        assert "may still be billed" in cancel_copy

    entered = asyncio.Event()
    cancelled = asyncio.Event()
    second_app = ModalHarness()
    second = _basic_modal(settings, second_app, generation_tester=waiting_tester)
    async with second_app.run_test(size=(120, 40)) as pilot:
        await second_app.push_screen(second)
        await pilot.pause()
        second.query_one("#console-settings-test-generation", Button).press()
        second.query_one("#console-settings-confirm-generation", Button).press()
        await asyncio.wait_for(entered.wait(), 1)
        second.action_dismiss()
        await asyncio.wait_for(cancelled.wait(), 1)


@pytest.mark.asyncio
async def test_generation_test_unsupported_provider_has_fixed_copy_and_no_action() -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="local_onnx", model="model-a"),
        app,
        providers_models={"local_onnx": ["model-a"]},
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#console-settings-test-generation", Button).display is False
        unavailable = modal.query_one(
            "#console-settings-generation-unavailable", Static
        )
        assert unavailable.display
        assert str(unavailable.renderable) == (
            "Generation test unavailable for this provider."
        )


@pytest.mark.parametrize("change_path", ["legacy_select", "provider_picker"])
@pytest.mark.asyncio
async def test_provider_switch_syncs_generation_test_action_bidirectionally(
    change_path: str,
) -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={
            "llama_cpp": ["model-a"],
            "openai": ["gpt-4.1"],
        },
    )

    async def switch(provider: str) -> None:
        sync_trace.clear()
        if change_path == "legacy_select":
            provider_select = modal.query_one("#console-settings-provider", Select)
            with provider_select.prevent(Select.Changed):
                provider_select.value = provider
            modal._provider_changed(Select.Changed(provider_select, provider))
        else:
            modal._provider_picker_selected(
                ConsoleProviderPicker.ProviderSelected(provider)
            )
        await pilot.pause()
        assert sync_trace[-2:] == [
            ("generation_controls", provider),
            ("generation_test", provider),
        ]

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        # The paid-test action must not rely on incidental generation-choice
        # events to refresh after a provider transition.
        sync_trace: list[tuple[str, str]] = []
        sync_generation_test_controls = modal._sync_generation_test_controls

        def sync_test_controls() -> None:
            sync_trace.append(("generation_test", modal._active_provider))
            sync_generation_test_controls()

        modal._sync_generation_test_controls = sync_test_controls
        modal._sync_generation_control_support = lambda: sync_trace.append(
            ("generation_controls", modal._active_provider)
        )
        action = modal.query_one("#console-settings-test-generation", Button)
        unavailable = modal.query_one(
            "#console-settings-generation-unavailable", Static
        )

        assert action.display
        assert action.disabled is False
        assert unavailable.display is False

        await switch("openai")
        assert action.display is False
        assert action.disabled is True
        assert unavailable.display
        assert str(unavailable.renderable) == (
            "Generation test unavailable for this provider."
        )

        await switch("llama_cpp")
        assert action.display
        assert action.disabled is False
        assert unavailable.display is False


@pytest.mark.asyncio
async def test_switch_provider_explicitly_syncs_generation_test_controls() -> None:
    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        app,
        providers_models={
            "llama_cpp": ["model-a"],
            "openai": ["gpt-4.1"],
        },
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        calls: list[str] = []
        modal.query_one(
            "#console-settings-provider-picker", ConsoleProviderPicker
        ).set_provider = lambda _provider: None
        modal._cancel_connection_probe = lambda: None
        modal._invalidate_model_discovery_for_provider = lambda _provider: None
        modal._store_current_model_for_provider = lambda _provider: None
        modal._store_current_base_url_for_provider = lambda _provider: None
        modal._sync_model_controls = lambda _provider, _model: None
        modal._sync_base_url_control = lambda _provider, _base_url: None
        modal._advance_model_discovery_generation = lambda: False
        modal._sync_model_discover_controls = lambda _provider: None
        modal._sync_provider_choice_placeholders = lambda: None
        modal._sync_generation_control_support = lambda: None
        modal._sync_readiness_display = lambda: None
        modal._sync_visual_representation_availability = lambda: None
        modal._sync_generation_test_controls = lambda: calls.append(
            modal._active_provider
        )

        modal._switch_provider("openai")

        assert calls == ["openai"]


@pytest.mark.asyncio
async def test_connection_retest_preserves_succeeded_generation_evidence() -> None:
    connection_entered = asyncio.Event()
    connection_release = asyncio.Event()

    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    async def connection_tester(_identity):
        connection_entered.set()
        await connection_release.wait()
        return settings_modal_module.ProviderProbeResult("reachable", ("model-a",))

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=generation_tester,
        connection_tester=connection_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Generation · Succeeded" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break

        modal.query_one("#console-settings-model-discover", Button).press()
        await asyncio.wait_for(connection_entered.wait(), 1)
        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Endpoint · Testing" in readiness
        assert "Generation · Succeeded" in readiness
        connection_release.set()


@pytest.mark.parametrize(
    "edit_kind", ["provider", "endpoint", "model", "generation"]
)
@pytest.mark.asyncio
async def test_succeeded_generation_evidence_is_invalidated_by_relevant_edit(
    edit_kind: str,
) -> None:
    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    app.app_config["api_settings"]["ollama"] = {
        "api_url": "http://127.0.0.1:11434"
    }
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        providers_models={
            "llama_cpp": ["model-a", "model-b"],
            "ollama": ["model-a"],
        },
        generation_tester=generation_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Generation · Succeeded" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break

        if edit_kind == "provider":
            modal._switch_provider("ollama")
        elif edit_kind == "endpoint":
            modal.query_one("#console-settings-base-url", Input).value = (
                "http://127.0.0.1:9199"
            )
        elif edit_kind == "model":
            modal._model_picker_selected(ModelSearchPicker.ModelSelected("model-b"))
        else:
            modal.query_one("#console-settings-temperature", Input).value = "0.3"
        await pilot.pause()

        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Generation · Succeeded" not in readiness
        assert "Generation · Changed since test" in readiness
        status = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert "succeeded" not in status.lower()
        assert "Changed since test" in status


@pytest.mark.parametrize(
    ("control_id", "new_value"),
    [
        ("console-settings-reasoning-effort", "high"),
        ("console-settings-reasoning-summary", "concise"),
        ("console-settings-verbosity", "high"),
        ("console-settings-thinking-effort", "high"),
    ],
)
@pytest.mark.asyncio
async def test_generation_choice_edit_immediately_marks_succeeded_test_changed(
    control_id: str,
    new_value: str,
) -> None:
    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=generation_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Generation · Succeeded" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break

        modal.query_one(f"#{control_id}", Select).value = new_value
        await pilot.pause()

        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Generation · Succeeded" not in readiness
        assert "Generation · Changed since test" in readiness
        assert "Changed since test" in str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )


@pytest.mark.parametrize("edit_kind", ["streaming", "temperature", "enum"])
@pytest.mark.asyncio
async def test_generation_edit_marks_changed_without_invalidating_endpoint(
    edit_kind: str,
) -> None:
    async def generation_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult("succeeded")

    async def connection_tester(_identity):
        return settings_modal_module.ProviderProbeResult("reachable", ("model-a",))

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=generation_tester,
        connection_tester=connection_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-model-discover", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Endpoint · Reachable" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if "Generation · Succeeded" in str(
                modal.query_one("#console-settings-readiness", Static).renderable
            ):
                break

        if edit_kind == "streaming":
            modal.query_one("#console-settings-streaming", Button).press()
        elif edit_kind == "temperature":
            modal.query_one("#console-settings-temperature", Input).value = "0.3"
        else:
            modal.query_one(
                "#console-settings-reasoning-effort", Select
            ).value = "high"
        await pilot.pause()

        readiness = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        assert "Endpoint · Reachable" in readiness
        assert "Generation · Changed since test" in readiness


@pytest.mark.asyncio
async def test_malformed_generation_tester_result_fails_bounded_and_restores_action(
    monkeypatch,
) -> None:
    async def malformed_tester(_request):
        return {"text": "secret provider response"}

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=malformed_tester,
    )

    notices: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_generation_probe_token is None:
                break

        assert modal._active_generation_probe_token is None
        assert str(
            modal.query_one("#console-settings-test-generation", Button).label
        ) == "Test generation"
        status = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert status == "Provider generation test failed."
        assert "secret provider response" not in status
        assert notices == []


@pytest.mark.asyncio
async def test_generation_timeout_discloses_provider_work_may_continue_and_bill() -> None:
    async def timeout_tester(_request):
        return settings_modal_module.ProviderGenerationProbeResult(
            "failed", "timeout"
        )

    app = ModalHarness()
    modal = _basic_modal(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        ),
        app,
        generation_tester=timeout_tester,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal._active_generation_probe_token is None:
                break

        status = str(
            modal.query_one(
                "#console-settings-generation-test-status", Static
            ).renderable
        )
        assert status == (
            "Generation test timed out. Already-started provider work may continue "
            "and may still be billed."
        )
