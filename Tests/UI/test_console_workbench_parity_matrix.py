"""Console Workbench parity and closeout documentation gates."""

from __future__ import annotations

import ast
from pathlib import Path


DESIGN_DOC = Path("Docs/Design/chatbook-workbench-ui-system.md")
MASTER_SHELL_CONTRACT = Path("Docs/Design/master-shell-design-system-contract.md")


CONSOLE_PARITY_MATRIX: dict[str, tuple[str, ...]] = {
    "streaming_send_stop": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_stop_interrupts_stream_and_keeps_partial_message_visible",
        "Tests/UI/test_console_native_chat_flow.py::test_console_duplicate_send_during_stream_does_not_break_stop_control",
    ),
    "non_streaming_fallback": (
        "Tests/Chat/test_console_provider_gateway.py::test_llamacpp_stream_chat_falls_back_to_non_streaming_when_stream_rejected",
        "Tests/Chat/test_console_provider_gateway.py::test_stream_chat_non_streaming_resolution_yields_completion_once",
    ),
    "provider_model_selection": (
        "Tests/UI/test_console_session_settings.py",
        "Tests/Chat/test_console_provider_support.py",
        "Tests/Chat/test_console_provider_endpoints.py",
    ),
    "multi_session_tabs": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail",
        "Tests/UI/test_console_native_chat_flow.py::test_console_close_tab_with_messages_shows_confirmation",
        "Tests/UI/test_console_native_chat_flow.py::test_console_native_active_tab_title_opens_rename_modal",
        "Tests/Chat/test_console_chat_store.py::test_console_sessions_store_independent_settings_snapshots",
    ),
    "retry_regenerate_continue": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_failed_stream_renders_inline_retry_and_recovers",
        "Tests/UI/test_console_native_chat_flow.py::test_console_continue_action_streams_new_message_from_selected_turn",
        "Tests/UI/test_console_native_chat_flow.py::test_console_regenerate_action_streams_selected_variant",
    ),
    "edit_delete_copy_feedback": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_copy_action_uses_app_clipboard",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_edit_action_opens_modal_and_saves_content",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_delete_action_removes_message_from_transcript",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_feedback_action_records_rating",
    ),
    "attachments_images": (
        "Tests/UI/test_chat_image_attachment.py",
        "Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag",
    ),
    "tool_call_visibility": (
        "Tests/UI/test_console_internals_decomposition.py::test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions",
        "Tests/integration/test_chat_tool_flow.py",
    ),
    "workspace_and_live_work_handoffs": (
        "Tests/UI/test_console_workspace_context_rail.py",
        "Tests/UI/test_console_live_work_handoffs.py",
    ),
    "staged_context_sources": (
        "Tests/UI/test_console_workspace_context_rail.py",
        "Tests/UI/test_console_live_work_handoffs.py",
        "Tests/Chat/test_console_chat_controller.py::test_blocked_workspace_source_preserves_draft_and_skips_provider_call",
    ),
    "recovery_states": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_setup_required_state_groups_recovery_and_action_copy",
        "Tests/UI/test_console_native_chat_flow.py::test_console_setup_blocked_send_is_unreachable_behind_modal",
        "Tests/UI/test_console_native_chat_flow.py::test_console_failed_stream_renders_inline_retry_and_recovers",
    ),
    "persistence_behavior": (
        "Tests/Chat/test_console_chat_store.py",
        "Tests/Chat/test_console_chat_controller.py::test_retry_failed_message_streams_replacement_from_original_turn",
    ),
    "visible_workbench_actions": (
        "Tests/UI/test_console_workbench_contract.py::test_console_core_controls_are_visible_without_command_palette",
        "Tests/UI/test_console_workbench_contract.py::test_console_setup_card_recovery_action_button_is_visible_and_actionable",
    ),
    "command_palette_duplicates": (
        "Tests/UI/test_command_palette_basic.py",
        "Tests/UI/test_command_palette_shell_routes.py",
        "Tests/UI/test_command_palette_providers.py",
    ),
}


def _defined_test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}::{child.name}")
    return names


def test_console_parity_matrix_covers_required_behaviors() -> None:
    assert set(CONSOLE_PARITY_MATRIX) == {
        "streaming_send_stop",
        "non_streaming_fallback",
        "provider_model_selection",
        "multi_session_tabs",
        "retry_regenerate_continue",
        "edit_delete_copy_feedback",
        "attachments_images",
        "tool_call_visibility",
        "workspace_and_live_work_handoffs",
        "staged_context_sources",
        "recovery_states",
        "persistence_behavior",
        "visible_workbench_actions",
        "command_palette_duplicates",
    }


def test_console_parity_matrix_references_existing_tests() -> None:
    for test_refs in CONSOLE_PARITY_MATRIX.values():
        for test_ref in test_refs:
            file_part, _, test_name = test_ref.partition("::")
            path = Path(file_part)
            assert path.exists(), test_ref
            if test_name:
                assert test_name in _defined_test_names(path), test_ref


def test_workbench_design_doc_records_visible_action_and_responsiveness_rules() -> None:
    text = DESIGN_DOC.read_text()

    for expected in (
        "Stable composition",
        "Visible workflows",
        "Explicit state",
        "Responsiveness gates",
        "provider/model settings",
        "send/stop",
        "attach context",
        "Library RAG",
        "command palette",
        "heartbeat lag",
        "worker",
        "timer",
        "mount churn",
    ):
        assert expected in text


def test_master_shell_contract_links_workbench_ui_system() -> None:
    text = MASTER_SHELL_CONTRACT.read_text()

    assert "Docs/Design/chatbook-workbench-ui-system.md" in text
    assert "visible-action" in text
    assert "responsiveness gates" in text
