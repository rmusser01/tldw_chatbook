"""Unchanged original optional-modal actions under current bootstrap ownership."""

import importlib

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_CASES = [
    (
        "Tests.Chat.test_console_style_picker",
        "test_style_picker_selection_inserts_style_token_into_draft",
    ),
    (
        "Tests.Chat.test_console_style_picker",
        "test_style_picker_insert_replaces_existing_leading_style_token",
    ),
    (
        "Tests.Chat.test_console_style_picker",
        "test_style_picker_escape_leaves_draft_untouched",
    ),
    (
        "Tests.UI.test_console_rewind_restore",
        "test_console_command_rewind_pushes_modal_with_newest_first_rows",
    ),
    (
        "Tests.UI.test_console_rewind_restore",
        "test_rewind_callback_refuses_sibling_descendant_path_change",
    ),
    (
        "Tests.UI.test_console_rewind_restore",
        "test_visible_send_rewind_cancel_clears_command_and_refocuses_empty_composer",
    ),
    (
        "Tests.UI.test_console_selection_end_to_end",
        "test_feedback_real_modal_submit_dispatches_composed_text",
    ),
    (
        "Tests.UI.test_console_selection_end_to_end",
        "test_feedback_real_modal_empty_submit_omits_comment_block",
    ),
    (
        "Tests.UI.test_console_selection_end_to_end",
        "test_feedback_real_modal_escape_dispatches_nothing",
    ),
    (
        "Tests.UI.test_console_native_chat_flow",
        "test_prompt_improvement_review_compares_and_can_keep_or_restore",
    ),
    (
        "Tests.UI.test_console_system_prompt",
        "test_action_open_console_system_prompt_editor_opens_modal",
    ),
    (
        "Tests.UI.test_console_system_prompt",
        "test_console_system_prompt_modal_apply_updates_settings_and_rail_preview",
    ),
    (
        "Tests.UI.test_console_system_prompt",
        "test_console_system_prompt_modal_cancel_leaves_settings_untouched",
    ),
    (
        "Tests.UI.test_console_system_prompt",
        "test_console_system_prompt_modal_save_to_library_creates_new_prompt",
    ),
]


@pytest.mark.parametrize("module_name,case", _CASES, ids=[case for _, case in _CASES])
async def test_original_optional_action_body(request, module_name, case):
    import inspect

    original = getattr(importlib.import_module(module_name), case)
    arguments = {
        name: request.getfixturevalue(name)
        for name in inspect.signature(original).parameters
    }
    await original(**arguments)
