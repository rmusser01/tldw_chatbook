"""Regression coverage for retired legacy app entrypoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETIRED_MODULES = (
    "tldw_chatbook.app_refactored",
    "tldw_chatbook.navigation",
    "tldw_chatbook.navigation.screen_registry",
    "tldw_chatbook.UI.Conv_Char_Window",
)

RETIRED_FILES = (
    "tldw_chatbook/app_refactored.py",
    "tldw_chatbook/navigation/__init__.py",
    "tldw_chatbook/navigation/navigation_manager.py",
    "tldw_chatbook/navigation/screen_registry.py",
    "tldw_chatbook/UI/Conv_Char_Window.py",
    "tldw_chatbook/UI/Conv_Char_Window.py.backup",
)

CCP_HANDLER_FILES = (
    "tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_dictionary_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_prompt_handler.py",
)


def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


def test_retired_legacy_entrypoint_modules_are_not_importable():
    """Verify retired legacy modules cannot be imported."""
    for module_name in RETIRED_MODULES:
        assert _find_spec(module_name) is None, module_name


def test_retired_legacy_entrypoint_files_are_removed():
    """Verify retired legacy source files are absent from the tree."""
    for relative_path in RETIRED_FILES:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_ccp_handlers_type_check_against_personas_screen():
    """Verify reused CCP handlers no longer type-check against CCPWindow."""
    for relative_path in CCP_HANDLER_FILES:
        source = (PROJECT_ROOT / relative_path).read_text()
        assert "Conv_Char_Window" not in source, relative_path
        assert "CCPWindow" not in source, relative_path
        assert "PersonasScreen" in source, relative_path


def test_active_ccp_route_still_resolves_to_personas_screen():
    """Verify the active compatibility route still targets PersonasScreen."""
    screen_name, canonical_tab, screen_class = resolve_screen_target("ccp")

    assert screen_name == "ccp"
    assert canonical_tab == "personas"
    assert screen_class is PersonasScreen
