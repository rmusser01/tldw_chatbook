"""The approved private forwarding cleanup preserves real screen contracts."""

import ast
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
WIRING = ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py"
# Fixed review-approved inventory, not inferred from the current screen.
DELEGATES = {
    "_commit_console_settings_submission_live": (
        "settings_durability",
        "ConsoleSettingsDurabilityController",
        "_commit_console_settings_submission_live",
    ),
    "_open_console_settings": (
        "settings_navigation",
        "ConsoleSettingsNavigationController",
        "_open_console_settings",
    ),
    "_consume_pending_conversation_settings_return": (
        "settings_navigation",
        "ConsoleSettingsNavigationController",
        "_consume_pending_conversation_settings_return",
    ),
    "_dispatch_console_settings_submission": (
        "settings_durability",
        "ConsoleSettingsDurabilityController",
        "_dispatch_console_settings_submission",
    ),
    "_console_inspector_next_send_factories": (
        "context_cost",
        "ConsoleContextCostController",
        "_console_inspector_next_send_factories",
    ),
    "_console_next_send_token_estimate": (
        "context_cost",
        "ConsoleContextCostController",
        "_console_next_send_token_estimate",
    ),
    "_console_conversation_state": (
        "row_actions",
        "ConsoleRowActionsController",
        "_console_conversation_state",
    ),
    "_save_console_conversation_markdown": (
        "row_actions",
        "ConsoleRowActionsController",
        "_save_console_conversation_markdown",
    ),
    "_provider_readiness_app_config": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_provider_readiness_app_config",
    ),
    "_providers_models_for_console_settings": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_providers_models_for_console_settings",
    ),
    "_active_console_settings_context_estimate": (
        "context_cost",
        "ConsoleContextCostController",
        "_active_console_settings_context_estimate",
    ),
    "_console_settings_context_estimate_for_session": (
        "context_cost",
        "ConsoleContextCostController",
        "_console_settings_context_estimate_for_session",
    ),
    "_active_console_context_control_state": (
        "context_cost",
        "ConsoleContextCostController",
        "_active_console_context_control_state",
    ),
    "_console_context_control_state_for_session": (
        "context_cost",
        "ConsoleContextCostController",
        "_console_context_control_state_for_session",
    ),
    "_build_console_settings_summary_state": (
        "context_cost",
        "ConsoleContextCostController",
        "_build_console_settings_summary_state",
    ),
    "_build_console_provider_selection": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_build_console_provider_selection",
    ),
    "_build_console_provider_selection_for_settings": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_build_console_provider_selection_for_settings",
    ),
    "_active_console_provider_model_display": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_active_console_provider_model_display",
    ),
    "_active_console_settings_readiness": (
        "provider_selection",
        "ConsoleProviderSelectionController",
        "_active_console_settings_readiness",
    ),
    "_recent_console_image_messages": (
        "message",
        "ConsoleMessageController",
        "_recent_console_image_messages",
    ),
    "_ensure_console_prompt_history": (
        "prompts",
        "ConsolePromptsController",
        "_ensure_console_prompt_history",
    ),
    "_sync_console_dictation_availability": (
        "dictation",
        "ConsoleDictationController",
        "_sync_console_dictation_availability",
    ),
    "_open_console_prompts_modal": (
        "prompts",
        "ConsolePromptsController",
        "_open_console_prompts_modal",
    ),
    "_enter_console_hands_free_loop": (
        "hands_free",
        "ConsoleHandsFreeController",
        "_enter_console_hands_free_loop",
    ),
    "_request_console_dictation_stop": (
        "dictation",
        "ConsoleDictationController",
        "_request_console_dictation_stop",
    ),
    "_request_console_dictation_cancel": (
        "dictation",
        "ConsoleDictationController",
        "_request_console_dictation_cancel",
    ),
    "_request_console_dictation_start": (
        "dictation",
        "ConsoleDictationController",
        "_request_console_dictation_start",
    ),
    "_build_console_cost_state": (
        "context_cost",
        "ConsoleContextCostController",
        "_build_console_cost_state",
    ),
    "_build_console_inspector_cost_data": (
        "context_cost",
        "ConsoleContextCostController",
        "_build_console_inspector_cost_data",
    ),
    "_open_console_library_search": (
        "retrieval",
        "ConsoleRetrievalController",
        "open_library_search",
    ),
    "_current_console_conversation_id": (
        "session",
        "ConsoleSessionController",
        "_current_console_conversation_id",
    ),
    "_console_messages_from_conversation_tree": (
        "message",
        "ConsoleMessageController",
        "_console_messages_from_conversation_tree",
    ),
    "_rehydrate_console_message_image": (
        "message",
        "ConsoleMessageController",
        "_rehydrate_console_message_image",
    ),
    "_rehydrate_console_message_attachments": (
        "message",
        "ConsoleMessageController",
        "_rehydrate_console_message_attachments",
    ),
    "_rehydrate_console_message_generation_metadata": (
        "message",
        "ConsoleMessageController",
        "_rehydrate_console_message_generation_metadata",
    ),
    "_console_citation_message_body": (
        "message",
        "ConsoleMessageController",
        "_console_citation_message_body",
    ),
    "_append_native_console_system_message": (
        "message",
        "ConsoleMessageController",
        "_append_native_console_system_message",
    ),
    "_submit_console_native_draft": (
        "submission",
        "ConsoleSubmissionController",
        "_submit_console_native_draft",
    ),
    "_on_console_submission_accepted": (
        "submission",
        "ConsoleSubmissionController",
        "_on_console_submission_accepted",
    ),
    "_console_pending_image_attachment": (
        "submission",
        "ConsoleSubmissionController",
        "_console_pending_image_attachment",
    ),
    "_console_attachment_blocked_reason": (
        "submission",
        "ConsoleSubmissionController",
        "_console_attachment_blocked_reason",
    ),
    "_console_send_blocked_reason": (
        "submission",
        "ConsoleSubmissionController",
        "_console_send_blocked_reason",
    ),
    "_send_console_message_from_visible_action": (
        "submission",
        "ConsoleSubmissionController",
        "_send_console_message_from_visible_action",
    ),
    "_dispatch_console_draft_send": (
        "submission",
        "ConsoleSubmissionController",
        "_dispatch_console_draft_send",
    ),
    "_restore_console_send_stash": (
        "submission",
        "ConsoleSubmissionController",
        "_restore_console_send_stash",
    ),
    "_dispatch_console_command": (
        "commands",
        "ConsoleCommandsController",
        "_dispatch_console_command",
    ),
    "_console_command_insert_prompt": (
        "prompts",
        "ConsolePromptsController",
        "_console_command_insert_prompt",
    ),
    "_insert_prompt_text_into_composer": (
        "commands",
        "ConsoleCommandsController",
        "_insert_prompt_text_into_composer",
    ),
    "_consume_pending_console_prompt_insert": (
        "prompts",
        "ConsolePromptsController",
        "_consume_pending_console_prompt_insert",
    ),
    "_console_command_apply_system": (
        "prompts",
        "ConsolePromptsController",
        "_console_command_apply_system",
    ),
    "_console_command_generate_image": (
        "image",
        "ConsoleImageController",
        "_console_command_generate_image",
    ),
    "_console_command_generate_video": (
        "video",
        "ConsoleVideoController",
        "_console_command_generate_video",
    ),
    "_console_command_stream_video": (
        "video",
        "ConsoleVideoController",
        "_console_command_stream_video",
    ),
    "_console_command_rewind": (
        "commands",
        "ConsoleCommandsController",
        "_console_command_rewind",
    ),
    "_apply_console_rewind_choice": (
        "commands",
        "ConsoleCommandsController",
        "_apply_console_rewind_choice",
    ),
    "_clear_console_composer_draft": (
        "commands",
        "ConsoleCommandsController",
        "_clear_console_composer_draft",
    ),
    "_open_console_system_prompt_editor": (
        "prompts",
        "ConsolePromptsController",
        "_open_console_system_prompt_editor",
    ),
    "_console_command_skills": (
        "skill",
        "ConsoleSkillController",
        "_console_command_skills",
    ),
    "_console_save_as_destinations": (
        "message",
        "ConsoleMessageController",
        "_console_save_as_destinations",
    ),
    "_save_console_message_image": (
        "message",
        "ConsoleMessageController",
        "_save_console_message_image",
    ),
    "_save_console_message_as_note": (
        "message",
        "ConsoleMessageController",
        "_save_console_message_as_note",
    ),
    "_open_console_message_edit_modal": (
        "message",
        "ConsoleMessageController",
        "_open_console_message_edit_modal",
    ),
    "_select_console_message_variant": (
        "message",
        "ConsoleMessageController",
        "_select_console_message_variant",
    ),
    "_recover_stuck_console_send_stash": (
        "submission",
        "ConsoleSubmissionController",
        "_recover_stuck_console_send_stash",
    ),
}


@lru_cache(maxsize=None)
def _tree(path):
    return ast.parse(path.read_text())


def test_private_delegate_inventory_keeps_exact_approved_count_and_command_owners():
    assert len(DELEGATES) == 64
    expected_command_owners = {
        "_console_command_generate_image": "image",
        "_console_command_generate_video": "video",
        "_console_command_stream_video": "video",
        "_console_command_skills": "skill",
    }
    assert {
        name: DELEGATES[name][0] for name in expected_command_owners
    } == expected_command_owners


@pytest.mark.parametrize("name", DELEGATES)
def test_private_delegate_is_retired_and_its_owner_remains(name):
    module, owner_name, target_name = DELEGATES[name]
    screen = next(
        node
        for node in _tree(SCREEN).body
        if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
    )
    assert name not in {
        node.name
        for node in screen.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    owner_path = ROOT / f"tldw_chatbook/UI/Console_Modules/{module}.py"
    owner = next(
        node
        for node in _tree(owner_path).body
        if isinstance(node, ast.ClassDef) and node.name == owner_name
    )
    assert target_name in {
        node.name
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for path in (SCREEN, WIRING):
        references = [
            node.lineno
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"screen", "self"}
            and node.attr == name
        ]
        assert references == [], (path, name, references)


def test_documented_external_screen_seams_remain_real_methods():
    screen = next(
        node
        for node in _tree(SCREEN).body
        if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
    )
    methods = {
        node.name
        for node in screen.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_ensure_console_agent_bridge",
        "_console_active_session_is_ephemeral",
        "console_prompt_target_projection",
        "action_open_trajectory_view",
        "_console_review_notes_flow",
        "on_console_review_notes_requested",
    } <= methods
