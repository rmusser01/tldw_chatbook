#!/bin/bash
PYTHONPATH=/private/tmp /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
 Tests/Chat/test_console_prompt_mixin_forks.py \
 Tests/DB/test_console_prompt_mixins.py::test_copy_same_row_retry_chain_publishes_exact_marker_once \
 Tests/DB/test_console_prompt_mixins.py::test_legacy_fork_ignores_unrelated_malformed_metadata \
 Tests/Chat/test_console_prompt_mixin_retry.py::test_current_choice_replay_refuses_source_rebind \
 Tests/Chat/test_console_prompt_mixin_retry.py::test_continue_refuses_source_rebind_during_hook \
 Tests/Chat/test_console_prompt_mixin_retry.py::test_unavailable_mixin_preview_has_bounded_recovery_copy \
 Tests/Chat/test_console_prompt_mixin_retry.py::test_retry_original_regenerate_current_and_clean_history \
 Tests/Chat/test_console_prompt_mixin_journey.py::test_local_queue_retry_current_regenerate_reopen \
 Tests/Chat/test_console_prompt_mixin_journey.py::test_selected_generation_never_keeps_another_receipts_count \
 Tests/UI/test_console_prompt_mixins.py::test_ordinary_browse_mixin_capability_refusal_and_local_open \
 Tests/UI/test_console_prompt_mixins.py::test_browse_mixin_open_awaited_transition_is_fenced \
 Tests/UI/test_console_prompt_mixins.py::test_held_save_preserves_newer_raw_editor_draft_and_can_save_again \
 Tests/Chatbooks/test_chatbook_prompt_round_trip.py::test_prompt_export_failure_is_atomic_bounded_and_private \
 Tests/Chat/test_console_chat_fork.py::test_fork_public_record_fields_are_exact \
 -q --tb=short --show-capture=no -p no:cacheprovider -p mixins_final_fd_audit \
 --basetemp=/private/tmp/mixins-final-covering-complete
