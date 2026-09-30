"""Mechanical size repairs retain the existing public import and patch seams."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.bootstrap_profile]

_REEXPORTS = (
    (
        "UI.MCP_Modules.mcp_workbench",
        "UI.MCP_Modules.mcp_rail",
        "_target_id_from_server_key",
    ),
    (
        "UI.MCP_Modules.mcp_workbench",
        "UI.MCP_Modules.mcp_permissions_mode",
        "_cycled_ui_label",
    ),
    (
        "UI.Screens.watchlists_collections_screen",
        "UI.Watchlists_Modules.opml_dialogs",
        "watchlist_delete_consequence",
    ),
    (
        "UI.Screens.llm_screen",
        "UI.Screens.model_browser_state",
        "_insufficient_space_recovery",
    ),
    (
        "UI.Screens.personas_screen",
        "UI.Persona_Modules.personas_preview_coordinator",
        "_DrainedTaskResult",
    ),
    (
        "UI.Screens.personas_screen",
        "UI.Persona_Modules.personas_preview_coordinator",
        "_drain_async",
    ),
    (
        "UI.Screens.personas_screen",
        "UI.Persona_Modules.personas_preview_coordinator",
        "_drain_to_thread",
    ),
    (
        "Widgets.Console.console_transcript",
        "Widgets.Console.console_assistant_turn",
        "ConsoleMemoryBannerPresentation",
    ),
    (
        "Widgets.Console.console_transcript",
        "Widgets.Console.console_assistant_turn",
        "derive_console_memory_banner_presentation",
    ),
    (
        "UI.Screens.library_screen",
        "UI.Library_Modules.screen_helpers",
        "_assign_library_reader_preferences_attribute",
    ),
    (
        "UI.Screens.library_screen",
        "UI.Library_Modules.screen_helpers",
        "_library_note_editor_exit_veto_message",
    ),
    ("tldw_api.client", "tldw_api.utils", "_raise_api_error_from"),
    *(
        ("UI.Wizards.FirstRunSetupWizard", "UI.Wizards.first_run_setup_widgets", name)
        for name in (
            "SetupRadioButton",
            "SetupCheckbox",
            "SetupRadioSet",
            "SetupStep",
            "SetupStepFailure",
            "ProviderChoiceOption",
            "_radio_model_id",
            "manual_settings_context_for_required_step",
            "REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES",
        )
    ),
)


@pytest.mark.parametrize(("public", "owner", "name"), _REEXPORTS)
def test_helper_reexports_keep_the_canonical_object_and_checkout(public, owner, name):
    public_module = importlib.import_module("tldw_chatbook." + public)
    owner_module = importlib.import_module("tldw_chatbook." + owner)
    root = Path(__file__).resolve().parents[2]
    assert Path(public_module.__file__).resolve().is_relative_to(root)
    assert Path(owner_module.__file__).resolve().is_relative_to(root)
    assert getattr(public_module, name) is getattr(owner_module, name)


def test_api_request_primitive_static_methods_keep_the_canonical_helpers():
    from tldw_chatbook.tldw_api.client import TLDWAPIClient
    from tldw_chatbook.tldw_api.utils import _raise_if_redirected, _validate_timeout

    assert TLDWAPIClient._raise_if_redirected is _raise_if_redirected
    assert TLDWAPIClient._validate_timeout is _validate_timeout


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        ("UI.Library_Modules.screen_helpers", "_review_footer_entries"),
        ("Library.library_shell_state", "_copy_library_continue_receipt"),
        ("Library.library_rag_state", "_trailing_index"),
    ),
)
def test_library_static_methods_keep_the_canonical_helpers(owner, name):
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    owner_module = importlib.import_module("tldw_chatbook." + owner)
    assert getattr(LibraryScreen, name) is getattr(owner_module, name)
