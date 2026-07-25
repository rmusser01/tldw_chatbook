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
    # The Personas "prompts" mode chip is retired (Task 7): prompt handling
    # moved entirely into Library, so CCPPromptHandler is dead code.
    "tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler",
    # task-412: ChatWindow was never instantiated in production (the app uses
    # ChatWindowEnhanced) and its right-sidebar composers created the only
    # widgets that ever bore id="chat-right-sidebar".
    "tldw_chatbook.UI.Chat_Window",
    "tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar",
    "tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar_optimized",
    # task-562 Unit 5: whole-file retirements. The Chat tab has been the
    # native Console since 8ea71071f (2026-05-06) removed the only caller of
    # ChatWindowEnhanced; these modules backed its dead settings sidebar and
    # sidebar-resize keybindings and had zero live importers.
    "tldw_chatbook.Widgets.settings_sidebar",
    "tldw_chatbook.Widgets.settings_sidebar_optimized",
    "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar_resize",
    # task-577 T2: ChatWindowEnhanced itself and its entire descendant tree.
    # Never instantiated since 8ea71071f (2026-05-06) -- _ensure_chat_window
    # had zero callers, self.chat_window stayed None for the process
    # lifetime, and #chat-window/#chat-log/EnhancedSettingsSidebar never
    # existed in the live tree. Whole-file/package deletions gated per-module
    # (grep -rn against tldw_chatbook/ + Tests/, zero live importers).
    "tldw_chatbook.UI.Chat_Window_Enhanced",
    "tldw_chatbook.Widgets.enhanced_settings_sidebar",
    "tldw_chatbook.Widgets.minimal_settings_sidebar",
    "tldw_chatbook.UI.Chat_Modules",
    "tldw_chatbook.UI.Chat_Modules.chat_attachment_handler",
    "tldw_chatbook.UI.Chat_Modules.chat_input_handler",
    "tldw_chatbook.UI.Chat_Modules.chat_message_manager",
    "tldw_chatbook.UI.Chat_Modules.chat_messages",
    "tldw_chatbook.UI.Chat_Modules.chat_sidebar_handler",
    "tldw_chatbook.UI.Chat_Modules.chat_voice_handler",
    # task-577 T2: the chat-tabs subsystem -- composed only inside the
    # unmounted ChatWindowEnhanced tree (#console-chat-tabs has no composer,
    # so chat_screen._get_tab_container() always returned None).
    "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container",
    "tldw_chatbook.Widgets.Chat_Widgets.chat_session",
    "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_bar",
    "tldw_chatbook.Chat.tabs",
    "tldw_chatbook.Chat.tabs.tab_context",
    "tldw_chatbook.Chat.tabs.tab_state_manager",
    "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs",
    # task-577 T2: orphan packages with zero importers anywhere (confirmed
    # dead by the task-577 scout; tab_initializers registers nothing and is
    # NOT the Console's init path; sidebar_events.SIDEBAR_BUTTON_HANDLERS was
    # never imported).
    "tldw_chatbook.Event_Handlers.tab_initializers",
    "tldw_chatbook.Event_Handlers.tab_initializers.base_initializer",
    "tldw_chatbook.Event_Handlers.tab_initializers.chat_tab_initializer",
    "tldw_chatbook.Event_Handlers.tab_initializers.misc_tab_initializers",
    "tldw_chatbook.Event_Handlers.sidebar_events",
    # task-577 T4 (U6 residual): Utils/chat_diagnostics.py had zero importers
    # anywhere in tldw_chatbook/ or Tests/ -- a standalone diagnostic tool
    # that was never wired into any live caller.
    "tldw_chatbook.Utils.chat_diagnostics",
    # NOTE: tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar is
    # DEFERRED to task-577 PR2, not retired here -- chat_events.py has an
    # eager, module-level import of it (CHAT_BUTTON_HANDLERS is built from
    # its CHAT_SIDEBAR_BUTTON_HANDLERS at module scope); deleting the file
    # today would break chat_events.py's import and app boot. It retires
    # alongside chat_events.py itself in Phase 2 P1.
)

RETIRED_FILES = (
    "tldw_chatbook/app_refactored.py",
    "tldw_chatbook/navigation/__init__.py",
    "tldw_chatbook/navigation/navigation_manager.py",
    "tldw_chatbook/navigation/screen_registry.py",
    "tldw_chatbook/UI/Conv_Char_Window.py",
    "tldw_chatbook/UI/Conv_Char_Window.py.backup",
    "tldw_chatbook/UI/CCP_Modules/ccp_prompt_handler.py",
    # task-412
    "tldw_chatbook/UI/Chat_Window.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar_optimized.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py.backup",
    "tldw_chatbook/UI/Chat_Window_Enhanced.py.backup",
    "tldw_chatbook/Widgets/settings_sidebar.py.backup",
    # task-562 Unit 5
    "tldw_chatbook/Widgets/settings_sidebar.py",
    "tldw_chatbook/Widgets/settings_sidebar_optimized.py",
    "tldw_chatbook/Event_Handlers/Chat_Events/chat_events_sidebar_resize.py",
    # task-577 T2
    "tldw_chatbook/UI/Chat_Window_Enhanced.py",
    "tldw_chatbook/Widgets/enhanced_settings_sidebar.py",
    "tldw_chatbook/Widgets/minimal_settings_sidebar.py",
    "tldw_chatbook/UI/Chat_Modules/__init__.py",
    "tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py",
    "tldw_chatbook/UI/Chat_Modules/chat_input_handler.py",
    "tldw_chatbook/UI/Chat_Modules/chat_message_manager.py",
    "tldw_chatbook/UI/Chat_Modules/chat_messages.py",
    "tldw_chatbook/UI/Chat_Modules/chat_sidebar_handler.py",
    "tldw_chatbook/UI/Chat_Modules/chat_voice_handler.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_session.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_tab_bar.py",
    "tldw_chatbook/Chat/tabs/__init__.py",
    "tldw_chatbook/Chat/tabs/tab_context.py",
    "tldw_chatbook/Chat/tabs/tab_state_manager.py",
    "tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py",
    "tldw_chatbook/Event_Handlers/tab_initializers/__init__.py",
    "tldw_chatbook/Event_Handlers/tab_initializers/base_initializer.py",
    "tldw_chatbook/Event_Handlers/tab_initializers/chat_tab_initializer.py",
    "tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py",
    "tldw_chatbook/Event_Handlers/sidebar_events.py",
    # task-577 T4 (U6 residuals)
    "tldw_chatbook/Utils/chat_diagnostics.py",
    "tldw_chatbook/Docs/CHAT_TABS_GUIDE.md",
    "tldw_chatbook/css/features/_chat_tabs.tcss",
)

CCP_HANDLER_FILES = (
    "tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_dictionary_handler.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py",
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


def test_task_562_conversation_entry_chain_retired():
    """task-562: the dead Chat-tab conversation-entry chain must not return.

    The Chat tab has been the native Console since 8ea71071f (2026-05-06)
    removed the only caller of ChatWindowEnhanced. Tasks 1-3 of the task-562
    deletion campaign retired the whole conversation-load/save/clone/search/
    character-sidebar chain from ``chat_events`` and ``chat_events_tabs``
    with zero deferrals; Task 4 retired the whole-file settings-sidebar
    modules those handlers rendered into. This guard pins the complete
    deleted symbol set so none of it silently returns.
    """
    from tldw_chatbook.Event_Handlers.Chat_Events import chat_events

    # Unit 1 — save/clone/load-selected handlers + display fn
    # Unit 2 — new-conversation + save-details + convert-to-note handlers
    # Unit 3 — conversation-search stack
    # Unit 4 — character-load-into-sidebar family
    for name in (
        "display_conversation_in_chat_tab_ui",
        "handle_chat_save_current_chat_button_pressed",
        "handle_chat_clone_current_chat_button_pressed",
        "handle_chat_load_selected_button_pressed",
        "handle_chat_new_conversation_button_pressed",
        "handle_chat_convert_to_note_button_pressed",
        "handle_chat_save_details_button_pressed",
        "perform_chat_conversation_search",
        "handle_chat_conversation_search_bar_changed",
        "handle_chat_search_checkbox_changed",
        "is_general_history_conversation",
        "handle_chat_character_search_input_changed",
        "handle_chat_load_character_button_pressed",
        "handle_chat_character_attribute_changed",
        "handle_chat_clear_active_character_button_pressed",
        # Final-review fix — missed Unit-2-family handler
        "handle_chat_new_temp_chat_button_pressed",
    ):
        assert not hasattr(chat_events, name), f"{name} was retired in task-562"

    # Unit 1 — chat_events_tabs.py wrapper region: the whole module was
    # deleted outright by task-577 T2, which supersedes these per-symbol
    # pins (display_conversation_in_chat_tab_ui_with_tabs,
    # handle_chat_conversation_search_changed_with_tabs).

    for button_id in (
        "chat-save-current-chat-button",
        "chat-clone-current-chat-button",
        "chat-conversation-load-selected-button",
        "chat-new-conversation-button",
        "chat-save-conversation-details-button",
        "chat-convert-to-note-button",
        "chat-load-character-button",
        "chat-clear-active-character-button",
        # Final-review fix — missed Unit-2-family handler
        "chat-new-temp-chat-button",
    ):
        assert button_id not in chat_events.CHAT_BUTTON_HANDLERS


def test_task_577_pr1_window_family_retired():
    """task-577 PR1: the enhanced chat window family must not return.

    ``ChatWindowEnhanced`` has been unmounted since 8ea71071f (2026-05-06);
    T1 stripped every ``self.chat_window``/``_ensure_chat_window`` seam from
    ``chat_screen.py``, T2 deleted the window family + tabs subsystem +
    orphan packages outright, and T3 removed the flags that gated them.
    This guard pins the whole retired unit set: the modules stay
    unimportable, ``ChatScreen`` carries no window-family attributes, and
    the three retired config keys never reappear in the packaged default
    config text.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    # Re-affirm unimportability for the units this task's report calls out
    # by name (the full set is covered by RETIRED_MODULES/RETIRED_FILES
    # above; these are the ones the task-577 T4 brief pins explicitly).
    for module_name in (
        "tldw_chatbook.UI.Chat_Window_Enhanced",
        "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container",
        "tldw_chatbook.Widgets.Chat_Widgets.chat_session",
        "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_bar",
        "tldw_chatbook.Chat.tabs",
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs",
        "tldw_chatbook.Event_Handlers.tab_initializers",
        "tldw_chatbook.Event_Handlers.sidebar_events",
        "tldw_chatbook.Utils.chat_diagnostics",
    ):
        assert _find_spec(module_name) is None, module_name

    # chat_screen.py no longer defines the window-family seam: no
    # _ensure_chat_window method, and no chat_window attribute default
    # (class-level or instance-level -- self.chat_window stayed permanently
    # None before T1 removed the field entirely).
    assert not hasattr(ChatScreen, "_ensure_chat_window")
    assert not hasattr(ChatScreen, "chat_window")

    # Qodo fix (task-577 PR1): the infinite legacy-tabs restore retry loop
    # (`_perform_state_restoration`) is deleted outright -- `restore_state`
    # now just logs and moves on when it finds tabbed state.
    assert not hasattr(ChatScreen, "_perform_state_restoration")

    # The three retired config keys (use_enhanced_window, enable_tabs,
    # max_tabs -- T3) must not reappear in the packaged default config text.
    for key in ("use_enhanced_window", "enable_tabs", "max_tabs"):
        assert key not in config_module.CONFIG_TOML_CONTENT, key
