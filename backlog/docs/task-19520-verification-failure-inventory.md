# TASK-19520 verification inventory (evidence-only)

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-19520-trust-permissions`

HEAD: `ca4a03eb820a124651ce0958d67c3abc42b36970`

Preserved exact-node artifacts:

- `/tmp/task-19520-verification-lastfailed.json` — 301 exact node IDs.
- `/tmp/task-19520-verification-nodeids.json` — collection order used to separate the interrupted run from stale completed-Skills failures.

Partition rule:

- Current partial full-suite nodes: every `lastfailed` key whose index in `nodeids` is at or before index `28213`, the node `Tests/Skills/test_project_skills_import_modal.py::test_never_during_inflight_import_is_inert`. Count: **272**.
- Completed Skills-gate nodes: the 29 `lastfailed` keys after that cutoff, all in `Tests/Skills/test_skills_import.py` or `Tests/Skills/test_skills_library_flow.py`. Count: **29**.

The interrupted run ended with: **256 failed, 27,926 passed, 217 skipped, 1 xfailed, 16 errors, 205 warnings in 8,808.04s (2:26:48)**. It is partial evidence only.

## Completed Skills gate — Library shell/readiness fixture drift (29 failures)

Evidence class: completed `Tests/Skills/ -q` gate.

Count: **29**.

Confidence: **high** for the observed shared symptom; **medium** for one common root cause across all 29.

Representative captured traceback: `textual.css.query.NoMatches: No nodes match '#library-row-browse-skills' on LibraryScreen()` from `test_import_real_superpowers_skills_lands_trust_pending`. Captured background logs also repeatedly show `ValueError: Local quiz backend is unavailable` while Library composition/readiness is incomplete. The failures are confined to the import and Library-flow UI harnesses, not trust-permission assertions.

Already represented: `TASK-16236` (“Repair Skills full-suite contract fixtures”, Done) covers the same general fixture-contract class, so this is a recurrence/new drift rather than an untracked class.

Exact nodes:

1. `Tests/Skills/test_skills_import.py::test_import_real_superpowers_skills_lands_trust_pending`
2. `Tests/Skills/test_skills_import.py::test_import_skill_via_skill_md_file_path_derives_name_from_parent_directory`
3. `Tests/Skills/test_skills_import.py::test_loose_file_import_success_line_names_the_service_derived_skill`
4. `Tests/Skills/test_skills_import.py::test_import_skill_with_supporting_reference_file_threads_it_through`
5. `Tests/Skills/test_skills_import.py::test_import_skill_with_extra_frontmatter_fields_applies_recognized_and_drops_unknown`
6. `Tests/Skills/test_skills_import.py::test_reimporting_the_same_skill_name_is_skipped_not_duplicated`
7. `Tests/Skills/test_skills_import.py::test_import_row_reports_missing_skill_md_and_unknown_path_gracefully`
8. `Tests/Skills/test_skills_import.py::test_import_row_rejects_name_too_long_without_partial_state`
9. `Tests/Skills/test_skills_import.py::test_import_row_rejects_oversized_content_without_partial_state`
10. `Tests/Skills/test_skills_import.py::test_import_row_imports_nested_reference_subfolder`
11. `Tests/Skills/test_skills_import.py::test_import_row_url_routes_to_remote_install_and_primes_review`
12. `Tests/Skills/test_skills_import.py::test_import_row_url_remote_skill_error_becomes_status_line`
13. `Tests/Skills/test_skills_import.py::test_import_row_url_generic_failure_uses_classified_name_guess`
14. `Tests/Skills/test_skills_library_flow.py::test_saving_a_trusted_skill_warns_and_requeues_needs_review`
15. `Tests/Skills/test_skills_library_flow.py::test_trust_panel_review_then_approve_moves_skill_to_available`
16. `Tests/Skills/test_skills_library_flow.py::test_skill_editor_canvas_scrolls_trust_panel_into_view`
17. `Tests/Skills/test_skills_library_flow.py::test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_approve_flow`
18. `Tests/Skills/test_skills_library_flow.py::test_already_bootstrapped_store_never_shows_setup_state`
19. `Tests/Skills/test_skills_library_flow.py::test_uninitialized_trust_store_list_still_shows_needs_review_glyph`
20. `Tests/Skills/test_skills_library_flow.py::test_delete_skill_returns_to_list_and_decrements_count`
21. `Tests/Skills/test_skills_library_flow.py::test_skill_editor_opens_under_real_runtime_policy_enforcer`
22. `Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_row_opens_blank_editor`
23. `Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_save_creates_and_increments_count`
24. `Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_save_invalid_name_shows_classify_outcome`
25. `Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_save_arrives_needs_review_with_panel_primed`
26. `Tests/Skills/test_skills_library_flow.py::test_delete_cancel_preserves_edits_typed_during_confirm`
27. `Tests/Skills/test_skills_library_flow.py::test_derived_flag_cleared_when_snapshotting_populated_description`
28. `Tests/Skills/test_skills_library_flow.py::test_orphaned_manifest_is_one_click_resetup`
29. `Tests/Skills/test_skills_library_flow.py::test_list_mode_unlock_refreshes_snapshot_not_just_posture`

Result: **29 failed, 441 passed, 5 warnings in 106.68s**.

## Current partial-run groups (272 nodes total)

All exact nodes are in `/tmp/task-19520-verification-lastfailed.json`; use the cutoff above to exclude the 29 later Skills-gate nodes. Counts below account for all 272 current nodes exactly.

### 1. Sandbox-denied loopback listener setup (2 errors)

Count: **2 errors**. Confidence: **high**.

Representative traceback: `_DeepBacklogHTTPServer(("127.0.0.1", 0), ...)` reaches `socket.bind` and raises `PermissionError: [Errno 1] Operation not permitted`.

Exact nodes:

- `Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_survives_agent_bridge_style_loop_swap`
- `Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop`

Already represented: `TASK-15111` owns the test network/loopback marker contract; `TASK-16276` notes the same sandbox-only bind signature. Both are Done.

### 2. RAG embedding initialization escapes offline fixtures (13 errors)

Count: **13 errors**. Confidence: **high**.

Representative traceback: `_no_network_io` teardown raises because background initialization calls `socket.create_connection -> huggingface.co:443` twice while `sentence-transformers/all-MiniLM-L6-v2` requests `config.json`; the Hugging Face client catches the blocked `OSError` and schedules a retry, so the guard reports the swallowed attempt at teardown.

Exact nodes: the 12 current keys under `Tests/RAG/` and `Tests/RAG_Search/test_embeddings_performance.py::TestEmbeddingPerformance::test_real_model_performance` in the preserved artifact.

Already represented: `TASK-16232`, `TASK-16276`, and `TASK-16198` document/fix the same offline-fixture/teardown-egress class; all are Done, so this is a broad recurrence.

### 3. STT task-602 platform smoke setup/call error (1 error)

Count: **1 error**. Confidence: **low** on root cause because the tool return omitted this traceback.

Exact node: `Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations`.

Representative evidence: exact node and error status are preserved, but its traceback fell inside the truncated section. No focused rerun was performed after the stop instruction.

Already represented: TASK-602 design/evidence documents this smoke, but no matching open failure task was found by text search.

### 4. Console/Chat controller and harness failures (94 failures)

Count: **94 failures**. Confidence: **medium** that several fixture/API drift clusters exist; **low** that all 94 share one cause.

Exact-node selector: current artifact keys under `Tests/Chat/`, excluding the two `test_console_provider_gateway.py` errors above.

File clusters:

- 40 `test_console_generation_actions.py`
- 19 `test_console_h3_image_edit.py`
- 10 `test_kimi_zai_native_tools.py`
- 8 `test_qwencloud_native_tools.py`
- 4 `test_console_video_actions.py`
- 3 `test_console_fleet_wake_safety.py`
- 2 `test_console_skill_script_confirm.py`
- 1 each in attachment riders (2 nodes total), diff feedback, headless wake, local review hook, provider failure copy, stop reliability, and Kimi/Z.ai provider contract

Representative emitted summaries: generation-action routing/state assertions, H3 refusal/cancellation/persistence ordering, and hosted-native continuation/tool-history assertions. Their individual tracebacks were in the truncated portion, so a single root cause cannot be asserted from this run.

Already represented: `TASK-2769`/`TASK-2780` (Done) cover earlier `test_console_generation_actions.py` fixture drift; `TASK-18605` (In Progress) covers continuation/H3/skill-confirm stale guards. The current set is much broader than those tasks.

### 5. LLM transport/realtime contract failures (61 failures)

Count: **61 failures**. Confidence: **medium** by file-local clustering; **low** on a single shared cause.

Exact-node selector: current artifact keys under `Tests/LLM_Calls/`.

File clusters:

- 34 `test_openai_realtime_session.py`
- 15 `test_hosted_chat.py`
- 10 `test_qwencloud.py`
- 2 `test_summarization_diagnostic_privacy.py`

Representative emitted summaries: nearly the entire scripted realtime session contract, owned JSON/SSE retry and redaction behavior, QwenCloud retry/global-budget behavior, and diagnostic-manifest boundary checks. Tracebacks were truncated. The all-file breadth is consistent with fake-transport/fixture setup drift, but that is inference only.

Already represented: realtime testing has `TASK-14876` (To Do, xdist serialization), `TASK-19160` (Done), `TASK-2362` (Done), and `TASK-14811.6`; none exactly accounts for this 34-node local-run cluster from the preserved evidence alone.

### 6. Managed model-artifact acquisition failures (54 failures)

Count: **54 failures**. Confidence: **medium** that local HTTP/acquisition fixture setup is shared across many; **low** without tracebacks.

Exact-node selector: current artifact keys under `Tests/Model_Artifacts/`.

File clusters:

- 12 `test_stream_fetch.py`
- 10 `test_provision_fetch.py`
- 9 `test_source_map.py`
- 8 `test_preflight.py`
- 6 `test_provision_install.py`
- 6 `test_credentials_and_boundaries.py`
- 3 `test_provision_crash_recovery.py`

Representative emitted summaries cover local/cross-origin fetch, range resume, preflight, install/finalize, credential withholding, and crash recovery. No common traceback survived truncation. The run-end FD sentinel reported growth from 25 to 981 descriptors (+956), a cross-cutting contamination signal that may affect this HTTP-heavy family; it is not proof of causation.

Already represented: `TASK-1692`, `TASK-1695`, `TASK-1696`, and `TASK-1720` cover individual artifact-contract changes; no single existing task found covers this full cluster.

### 7. Database/schema/owner failures (13 failures)

Count: **13 failures**. Confidence: **high** for the two stale schema-head assertions; **medium/low** for the other 11.

Exact-node selector: current artifact keys under `Tests/DB/`, `Tests/ChaChaNotesDB/`, and `Tests/Media_DB/`.

Breakdown:

- 6 `test_core_sqlite_owner_privacy.py`
- 3 ChaChaNotesDB index/persona migration nodes
- 2 explicit stale-v40 nodes: `test_current_schema_version_is_40` and `test_fresh_db_lands_on_v40_with_the_annotations_table`
- 1 live-schema table census and 1 media DB migration/reopen node

Representative evidence: production `_CURRENT_SCHEMA_VERSION` is 42 while two tests explicitly assert v40. `TASK-19053` shipped v41 and `TASK-16320` shipped v42; `TASK-19044` is the prior Done task for this exact moving-literal class.

### 8. Packaging/install-distribution failures (7 failures)

Count: **7 failures**. Confidence: **medium** by file cluster; **low** on a common cause.

Exact-node selector: current artifact keys under `Tests/Packaging/`.

- 5 installed-distribution nodes
- 2 MCP unified wheel/sdist nodes

Representative emitted summaries include v35-to-current migration, immutable assets/loaders, Samira seeding, and isolated MCP extras. Tracebacks were truncated.

Already represented: `TASK-19044` (Done) covered prior installed migration/package-data drift; this run includes the renamed current-version nodes, so it is not the old literal itself.

### 9. MCP watchlists/provider failures (5 failures)

Count: **5 failures**. Confidence: **medium** that watchlists storage/service contract drift is shared.

Exact-node selector: current artifact keys under `Tests/MCP/`.

Representative emitted summaries: real watchlists provider structured outcomes, unexpected-failure scrubbing, off-loop database resolution, storage-lazy registration, and resolver replacement/close ordering. Tracebacks were truncated.

### 10. LLM Management MLX fixture failures (4 failures)

Count: **4 failures**. Confidence: **high** that all four share the `test_mlx_lm.py` fixture/config path; root cause traceback unavailable.

Exact-node selector: current artifact keys under `Tests/LLM_Management/`.

### 11. Startup performance policy failures (4 failures)

Count: **4 failures**. Confidence: **high** that the citation reconciliation/migration policy pair is shared; root cause traceback unavailable.

Exact-node selector: current artifact keys under `Tests/Performance/`.

### 12. Remaining bounded families (14 failures)

Count: **14 failures**. Confidence varies; exact nodes are preserved and are intentionally not collapsed into a fabricated common cause.

- 2 RAG_Search functional failures (`test_fts5_match_forms_shared.py`, `test_reranker_system_prompt.py`)
- 2 architecture ratchets (persistent diagnostic inventory, chat-screen size budget)
- 2 chunking failures (tokenizer override, CJK token parity)
- 2 localhost Parakeet artifact failures under `Tests/Local_Ingestion/`
- 1 each under Agents, Evals, Internal_Prompts, Persona_Visual, RuntimePolicy, and Skills

The six named one-node domains plus the four two-node families account for all 14. Tracebacks were truncated; each should be triaged from its preserved exact node rather than inferred from the count.

## Backlog mapping

The inventory is fully assigned as follows. Counts are node counts, including setup/teardown errors.

| Coverage | Nodes | Backlog task |
| --- | ---: | --- |
| Skills import and Library flow | 29 | `TASK-19642.1` |
| Project Skills import-modal session contamination | 1 | `TASK-19642.27` |
| Provider-gateway loopback setup | 2 | `TASK-19642.2` |
| RAG embedding egress/teardown | 13 | `TASK-19642.3` |
| STT task-602 smoke | 1 | `TASK-19642.4` |
| Console generation actions | 40 | `TASK-19642.5` |
| Console H3 image editing | 19 | `TASK-19642.6` |
| Kimi, Z.ai, and QwenCloud native-tool continuations | 19 | `TASK-19642.7` |
| Attachment save-toast escaping | 2 | `TASK-19642.8.1` |
| Video action routing and save behavior | 4 | `TASK-19642.8.2` |
| Fleet and headless wake authority | 4 | `TASK-19642.8.3` |
| Skill-confirm bridge wiring | 2 | `TASK-19642.8.4` |
| Diff-feedback payload identity | 1 | `TASK-19642.8.5` |
| Local-review root-swap confinement | 1 | `TASK-19642.8.6` |
| Provider failure recovery copy | 1 | `TASK-19642.8.7` |
| Stop content freeze | 1 | `TASK-19642.8.8` |
| OpenAI realtime sessions | 34 | `TASK-19642.9` |
| Hosted-chat and QwenCloud transports | 25 | `TASK-19642.10` |
| Model-artifact transfer, source maps, and credentials | 37 | `TASK-19642.11` |
| Model-artifact preflight, install, and recovery | 17 | `TASK-19642.12` |
| ChaChaNotes schema, migration, and census | 6 | `TASK-19642.13` |
| Media DB v2 migration reopen | 1 | `TASK-19642.28` |
| SQLite owner privacy | 6 | `TASK-19642.14` |
| Installed distribution and MCP packaging | 7 | `TASK-19642.15` |
| MCP watchlists provider | 5 | `TASK-19642.16` |
| MLX management fixtures | 4 | `TASK-19642.17` |
| Startup reconciliation policy | 4 | `TASK-19642.18` |
| FTS5 inflection search parity | 1 | `TASK-19642.19.1` |
| Local reranker system-prompt cardinality | 1 | `TASK-19642.19.2` |
| Persistent diagnostic inventory | 1 | `TASK-19642.20` |
| V2 chunker tokenizer override | 1 | `TASK-19642.21.1` |
| CJK token chunking golden parity | 1 | `TASK-19642.21.2` |
| Parakeet v2 local artifacts | 2 | `TASK-19642.22` |
| Tool-catalog snapshot concurrency | 1 | `TASK-19642.23` |
| Character-probe import boundary | 1 | `TASK-19642.24` |
| Websearch prompt parity | 1 | `TASK-19642.25` |
| Persona visual publication bounds | 1 | `TASK-19642.26` |
| Summarization diagnostic boundary | 2 | existing `TASK-18801` |
| Chat screen size ratchet | 1 | existing `TASK-3070` / `TASK-3070.11` |
| Server-client provider migration audit | 1 | existing `TASK-18610` AC 6 |
| **Total** | **301** | `TASK-19642` parent tracker |

## Cross-cutting contamination signal

The run-end sentinel reported **open file descriptors grew by 956** (`start=25`, `end=981`, limit 200). This is evidence of session contamination and may explain later HTTP/socket-heavy cascades, but the interrupted run did not bisect the owning test. Do not assign downstream failures to TASK-19520 or to the FD leak without a clean identical baseline/focused reproduction.

## TASK-19520 focused and mutation evidence

### Mutation RED evidence (all restored immediately)

1. JSON trust writer: changed only `_atomic_write_json` to pass `owner_only=False`; `test_trust_temp_is_owner_only_before_replace` failed at `assert all(observed_owner_only)`, observed `[False, False]`. **1 failed in 0.67s**.
2. Bytes trust writer: changed only `_atomic_write_bytes` to pass `owner_only=False`; `test_trust_bytes_writer_requests_owner_only_creation` failed at `[False] == [True]`. **1 failed in 0.63s**.
3. Directory guard: disabled POSIX directory chmod; `test_next_write_tightens_legacy_files_and_directories` failed with `493 == 448` (`0o755` vs `0o700`). **1 failed in 0.63s**.
4. Shared fchmod ordering: omitted `os.fchmod`; `test_owner_only_exclusively_opens_0o600_temp_before_real_replace` failed because events were only `[("writer", 384)]`, not `[("fchmod", 384), ("writer", 384)]`. **1 failed in 0.54s**.
5. Descriptor-close handling: omitted `os.close`; `test_owner_only_close_failure_after_setup_propagates_close_error` failed with `DID NOT RAISE OSError`. **1 failed in 0.55s**.

`git diff --exit-code` was clean after every restoration.

### Focused GREEN evidence

- `python -m pytest Tests/Skills/test_atomic_write_concurrency.py::TestOwnerOnlyTempCreation -q`: **8 passed, 1 warning in 0.67s**.

These tests pin exclusive `0o600` creation, fchmod-before-writer ordering, EEXIST preservation, writer/replace failure cleanup, descriptor closure, close-error precedence, and unchanged default behavior.

### Static evidence

- Ruff check on the four task files: **All checks passed**.
- Ruff format `--check` on the four task files: **4 files already formatted**.
- `git diff --check`: pass.
- Final `git status --short`: empty.
- Final `git diff --exit-code`: exit 0.

## Security self-review

- Diff scope is four files: shared atomic helper, trust store, and two Skills test files.
- Direct-call inventory confirms only `_atomic_write_json` and `_atomic_write_bytes` in the trust store opt into `owner_only=True`; ordinary shared text/bytes atomic writers keep default behavior.
- `O_EXCL` failure occurs before ownership is claimed, so an unexplained existing temp is preserved.
- Created descriptors close on setup success and failure; fchmod occurs before close and before the writer callback.
- Trust root is secured before snapshot-directory creation; snapshot directory is secured before publication.
- POSIX directory chmod and fchmod are guarded; non-POSIX creation keeps portable open semantics without requiring chmod.
- ACL/mount, same-UID, root/admin, and Windows ACL limits remain documentation only; no unsupported security boundary is claimed.
- No secrets, logs, user data, or ordinary skill writes are changed.

## Permanent exact-node appendix for the interrupted run

The 272 current partial-run nodes are JSON-quoted below so parameterized IDs containing newlines remain unambiguous. The 29 completed Skills-gate nodes are already enumerated above.

```text
"Tests/Agents/test_tool_catalog_concurrency.py::test_invoke_by_name_takes_exactly_one_catalog_snapshot"
"Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged"
"Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/chat_screen.py]"
"Tests/ChaChaNotesDB/test_index_census.py::TestChachanotesIndexCensusMatchesLiveSchema::test_no_unexpected_indexes[chain_migrated_from_v4]"
"Tests/ChaChaNotesDB/test_index_census.py::TestChachanotesIndexCensusMatchesLiveSchema::test_no_unexpected_indexes[fresh_bootstrap]"
"Tests/ChaChaNotesDB/test_persona_visual_migration.py::test_real_v40_upgrade_installs_separate_persona_visual_schema"
"Tests/Chat/test_console_attachment_riders.py::TestSaveImageToastEscaping::test_multi_save_toast_escapes_path"
"Tests/Chat/test_console_attachment_riders.py::TestSaveImageToastEscaping::test_single_save_toast_escapes_path"
"Tests/Chat/test_console_diff_feedback_delivery.py::test_no_pending_notes_leaves_payload_byte_identical"
"Tests/Chat/test_console_fleet_wake_safety.py::test_a_wake_defers_behind_a_pending_card_and_cannot_resolve_it"
"Tests/Chat/test_console_fleet_wake_safety.py::test_a_wake_dispatches_run_reply_under_the_same_authority_as_manual"
"Tests/Chat/test_console_fleet_wake_safety.py::test_a_woken_turns_gated_tool_still_raises_the_approval_card"
"Tests/Chat/test_console_generation_actions.py::test_browse_clamps_at_boundaries"
"Tests/Chat/test_console_generation_actions.py::test_browse_next_then_previous_mutates_screen_state_only"
"Tests/Chat/test_console_generation_actions.py::test_browse_noop_for_single_variant_message"
"Tests/Chat/test_console_generation_actions.py::test_dispatch_console_command_blocks_generate_image_when_ephemeral"
"Tests/Chat/test_console_generation_actions.py::test_failed_speech_clears_on_any_next_message_action"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_no_prompt_kill_switch_off_skips_llm_path"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_no_prompt_llm_call_raises_falls_back"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_no_prompt_llm_empty_response_falls_back"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_no_prompt_llm_timeout_falls_back"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_no_prompt_uses_llm_composed_context_end_to_end"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_prompt_present_never_resolves_llm_context"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_restores_draft_when_batch_raises"
"Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_threads_prepared_fields_into_batch"
"Tests/Chat/test_console_generation_actions.py::test_h3_edit_regenerate_refuses_before_capacity_or_inflight_checks"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_blocks_generation_regenerate_when_ephemeral"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_does_not_forge_user_speech_snapshot"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_posts_store_issued_speech_snapshot"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_routes_keep_button_for_generation_message"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_routes_regenerate_for_generation_message"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_routes_speak_for_generation_message"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_routes_speak_stop_to_tts_playback_event"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_routes_variant_next_for_generation_message"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_speak_marks_message_as_speaking"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_speak_stop_does_not_clear_other_message"
"Tests/Chat/test_console_generation_actions.py::test_handle_console_message_action_speak_stop_safe_when_nothing_speaking"
"Tests/Chat/test_console_generation_actions.py::test_keep_evicts_stale_render_cache_entries_so_rebuild_shows_kept_variant"
"Tests/Chat/test_console_generation_actions.py::test_keep_noop_when_browsed_index_is_zero"
"Tests/Chat/test_console_generation_actions.py::test_keep_reorders_store_and_resets_browse"
"Tests/Chat/test_console_generation_actions.py::test_llm_context_options_disabled_by_kill_switch"
"Tests/Chat/test_console_generation_actions.py::test_llm_context_options_provider_not_ready"
"Tests/Chat/test_console_generation_actions.py::test_llm_context_options_resolution_exception_degrades_gracefully"
"Tests/Chat/test_console_generation_actions.py::test_llm_context_options_resolves_ready_provider"
"Tests/Chat/test_console_generation_actions.py::test_real_handler_stop_order_settles_and_notifies_once"
"Tests/Chat/test_console_generation_actions.py::test_regenerate_failure_leaves_message_untouched_and_reports_error"
"Tests/Chat/test_console_generation_actions.py::test_regenerate_refused_at_cap"
"Tests/Chat/test_console_generation_actions.py::test_regenerate_refused_while_inflight"
"Tests/Chat/test_console_generation_actions.py::test_regenerate_success_appends_variant_and_browses_to_new_index"
"Tests/Chat/test_console_generation_actions.py::test_regenerate_success_inherits_style_from_position_zero_meta"
"Tests/Chat/test_console_generation_actions.py::test_rejected_owned_stop_retains_lifecycle_for_retry"
"Tests/Chat/test_console_generation_actions.py::test_rejected_stop_post_does_not_claim_stopped"
"Tests/Chat/test_console_h3_image_edit.py::test_app_owned_task_cancellation_drains_linearized_runner_before_reraise[cancel]"
"Tests/Chat/test_console_h3_image_edit.py::test_app_owned_task_cancellation_drains_linearized_runner_before_reraise[success]"
"Tests/Chat/test_console_h3_image_edit.py::test_failure_guidance_persistence_error_falls_back_without_masking_primary[after_append-expected_attempts1]"
"Tests/Chat/test_console_h3_image_edit.py::test_failure_guidance_persistence_error_falls_back_without_masking_primary[before_append-expected_attempts0]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_canonical_validation_performs_the_only_full_source_decode"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_command_uses_raw_instruction_one_memory_image_and_count_one"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_oversize_source_is_rejected_before_decode_or_dispatch"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui @anime change it-pendings1]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui change it-pendings2]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui change it-pendings3]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui change it-pendings4]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui change it-pendings5]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_refusals_happen_before_generic_preparation_or_generation[:comfyui-pendings0]"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_source_header_read_runs_off_loop_while_pump_remains_responsive"
"Tests/Chat/test_console_h3_image_edit.py::test_h3_warning_band_is_rejected_by_canonical_ceiling_before_full_decode"
"Tests/Chat/test_console_h3_image_edit.py::test_persistence_failure_retains_source_and_emits_sanitized_copy"
"Tests/Chat/test_console_h3_image_edit.py::test_postcommit_consume_exception_keeps_success_and_logs_only_type"
"Tests/Chat/test_console_h3_image_edit.py::test_stop_before_adapter_success_is_expected_and_retains_source"
"Tests/Chat/test_console_h3_image_edit.py::test_terminal_generation_never_syncs_stale_origin_screen"
"Tests/Chat/test_console_headless_wake_invariants.py::test_a_headless_wake_takes_the_same_agent_dispatch_and_budget"
"Tests/Chat/test_console_local_review_hook.py::test_selected_root_swap_fails_closed_before_local_invoke"
"Tests/Chat/test_console_provider_failure_copy.py::test_agent_failure_row_carries_body_and_image_recovery_hint"
"Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop"
"Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_survives_agent_bridge_style_loop_swap"
"Tests/Chat/test_console_skill_script_confirm.py::test_confirm_callback_absent_from_bridge_when_no_ui_sink_wired"
"Tests/Chat/test_console_skill_script_confirm.py::test_confirm_callback_present_when_ui_sink_wired"
"Tests/Chat/test_console_stop_reliability.py::test_stop_freezes_message_content_at_stop_point"
"Tests/Chat/test_console_video_actions.py::test_handle_console_message_action_routes_video_play_with_persisted_storage_id"
"Tests/Chat/test_console_video_actions.py::test_handle_console_message_action_routes_video_save_with_persisted_storage_id"
"Tests/Chat/test_console_video_actions.py::test_video_play_resolves_webm_from_metadata"
"Tests/Chat/test_console_video_actions.py::test_video_save_copy_preserves_webm_extension_and_collision_names"
"Tests/Chat/test_kimi_zai_native_tools.py::test_console_runs_two_native_calls_with_private_continuation[moonshot]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_console_runs_two_native_calls_with_private_continuation[zai]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_invalid_tool_history_fails_before_server_advancement[duplicate_ids-moonshot]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_invalid_tool_history_fails_before_server_advancement[duplicate_ids-zai]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_invalid_tool_history_fails_before_server_advancement[out_of_order-moonshot]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_invalid_tool_history_fails_before_server_advancement[out_of_order-zai]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_partial_call_cancellation_never_executes[moonshot]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_partial_call_cancellation_never_executes[zai]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_tool_error_continues_structurally[moonshot]"
"Tests/Chat/test_kimi_zai_native_tools.py::test_hosted_tool_error_continues_structurally[zai]"
"Tests/Chat/test_kimi_zai_provider_contract.py::test_streaming_adapter_carries_terminal_candidate_into_model_turn"
"Tests/Chat/test_qwencloud_native_tools.py::test_console_agent_bridge_runs_qwencloud_two_call_continuation[chat_completions]"
"Tests/Chat/test_qwencloud_native_tools.py::test_console_agent_bridge_runs_qwencloud_two_call_continuation[responses]"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_partial_call_cancellation_never_executes[chat_completions]"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_partial_call_cancellation_never_executes[responses]"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_responses_joined_runtime_history_pairs_out_of_order_results"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_responses_usage_enforces_agent_budget"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_tool_error_continues_structurally[chat_completions]"
"Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_tool_error_continues_structurally[responses]"
"Tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_tokenizer_override"
"Tests/Chunking/test_golden_parity.py::test_golden_parity[tokens-cjk]"
"Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py::test_current_schema_version_is_40"
"Tests/DB/test_chachanotes_transcript_annotations_migration.py::test_fresh_db_lands_on_v40_with_the_annotations_table"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_core_owner_rejects_unsafe_namespace_before_raw_sqlite[media-missing_parent]"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_core_owner_rejects_unsafe_namespace_before_raw_sqlite[media-symlink_parent]"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_core_owner_rejects_unsafe_namespace_before_raw_sqlite[media-symlink_target]"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_core_owner_rejects_unsafe_namespace_before_raw_sqlite[media-writable_parent]"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_domain_connection_exception_translation_is_preserved[media-DatabaseError]"
"Tests/DB/test_core_sqlite_owner_privacy.py::test_domain_connection_exception_translation_is_preserved_on_reconnect[media-DatabaseError]"
"Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema::test_no_missing_tables"
"Tests/Evals/character_probe/test_conversation_storage.py::test_character_probe_never_imports_the_word_bench_measurement_stack"
"Tests/Internal_Prompts/test_websearch_prompt_parity.py::test_result_relevance_eval_parity"
"Tests/LLM_Calls/test_hosted_chat.py::test_nonstreaming_2xx_malformed_body_is_not_retried[action0]"
"Tests/LLM_Calls/test_hosted_chat.py::test_nonstreaming_2xx_malformed_body_is_not_retried[action1]"
"Tests/LLM_Calls/test_hosted_chat.py::test_nonstreaming_2xx_malformed_body_is_not_retried[action2]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_honors_http_date_and_malformed_retry_after"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_http_failures_without_body_disclosure[400-ChatBadRequestError]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_http_failures_without_body_disclosure[401-ChatAuthenticationError]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_http_failures_without_body_disclosure[403-ChatAuthenticationError]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_http_failures_without_body_disclosure[429-ChatRateLimitError]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_http_failures_without_body_disclosure[500-ChatProviderError]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_retries_statuses_with_one_global_budget"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_sends_exact_route_headers_payload_and_timeout[chat/completions]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_sends_exact_route_headers_payload_and_timeout[responses]"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_sse_stream_does_not_retry_after_any_body_byte"
"Tests/LLM_Calls/test_hosted_chat.py::test_owned_sse_stream_transfers_ownership_and_closes_exactly_once"
"Tests/LLM_Calls/test_hosted_chat.py::test_sensitive_request_forces_transport_retries_to_zero"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_append_audio_after_close_from_foreign_thread_does_not_raise_or_queue"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_append_audio_base64_roundtrip"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_append_audio_from_foreign_thread_is_delivered"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_audio_delta_decodes_to_bytes_and_first_audio_fires_once"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_bad_frame_does_not_kill_recv_loop_and_response_done_still_fires"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_barge_in_after_response_done_still_truncates"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_callback_exception_is_isolated_and_routed_to_on_error"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_cancel_response_noops_when_no_response_active"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_cancel_response_sends_cancel_then_truncate_with_played_ms"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_close_does_not_hang_when_sender_task_is_stalled"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_configured_voice_is_sent_under_audio_output"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_connect_sends_correct_input_and_output_rates_without_swapping"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_connect_sends_session_update_and_fires_ready"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_default_config_sends_the_providers_own_mode"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_error_event_routes_to_on_error_not_crash"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_input_audio_buffer_committed_fires_on_turn_committed"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_input_transcript_completed_with_usage_fires_on_transcription_usage"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_input_transcript_completed_without_usage_does_not_fire_transcription_usage"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_output_item_added_ignores_non_assistant_role_items"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_output_item_added_only_resets_first_audio_once_per_response"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_response_done_cancelled_does_not_fire_reply_done"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_response_done_completed_fires_reply_done_and_usage"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_response_done_failed_routes_to_error_and_still_fires_reply_done"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_semantic_vad_never_carries_the_server_vad_knobs"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_semantic_vad_sends_the_bare_semantic_block"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_send_seed_creates_items_in_order_without_response"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_send_text_item_with_request_response_true_sends_response_create"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_sender_loop_death_marks_session_closed_and_fires_on_error_once"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_server_close_fires_on_closed_with_reason"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_server_vad_carries_both_knobs_when_both_are_set"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_server_vad_sends_only_the_knobs_that_are_set"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_speech_started_fires_during_active_response"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_transcripts_route_to_both_callbacks"
"Tests/LLM_Calls/test_openai_realtime_session.py::test_unset_voice_omits_the_voice_key"
"Tests/LLM_Calls/test_qwencloud.py::test_invalid_retry_after_uses_exponential_fallback_without_disclosure[429]"
"Tests/LLM_Calls/test_qwencloud.py::test_invalid_retry_after_uses_exponential_fallback_without_disclosure[503]"
"Tests/LLM_Calls/test_qwencloud.py::test_malformed_success_body_is_typed_redacted_and_not_retried[invalid-content]"
"Tests/LLM_Calls/test_qwencloud.py::test_malformed_success_body_is_typed_redacted_and_not_retried[invalid-json]"
"Tests/LLM_Calls/test_qwencloud.py::test_retry_policy_counts_status_connection_and_timeout_attempts"
"Tests/LLM_Calls/test_qwencloud.py::test_retry_policy_honors_retry_after_and_exponential_delay"
"Tests/LLM_Calls/test_qwencloud.py::test_retry_policy_uses_one_global_budget_across_mixed_failures"
"Tests/LLM_Calls/test_qwencloud.py::test_sensitive_request_forces_zero_retries"
"Tests/LLM_Calls/test_qwencloud.py::test_stalled_nonretryable_400_is_typed_redacted_and_not_retried"
"Tests/LLM_Calls/test_qwencloud.py::test_truncated_body_retries_once_and_closes_each_attempt"
"Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_changes_only_summarization_owner_diagnostics"
"Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_rejects_unreconciled_owned_digest"
"Tests/LLM_Management/test_mlx_lm.py::test_chat_with_mlx_lm_api_url_override"
"Tests/LLM_Management/test_mlx_lm.py::test_chat_with_mlx_lm_args_override_config"
"Tests/LLM_Management/test_mlx_lm.py::test_chat_with_mlx_lm_kwargs_passthrough"
"Tests/LLM_Management/test_mlx_lm.py::test_chat_with_mlx_lm_success_with_config"
"Tests/Local_Ingestion/test_parakeet_v2_artifact.py::test_run_parakeet_v2_preflight_and_provision_against_localhost_fixture"
"Tests/Local_Ingestion/test_parakeet_v2_artifact.py::test_vad_only_preflight_and_provision_never_include_a_parakeet_root"
"Tests/MCP/test_gateway_runtime_tools.py::test_real_watchlists_database_resolution_runs_off_event_loop"
"Tests/MCP/test_gateway_runtime_tools.py::test_real_watchlists_provider_preserves_structured_domain_outcomes"
"Tests/MCP/test_gateway_runtime_tools.py::test_real_watchlists_provider_scrubs_unexpected_failures"
"Tests/MCP/test_local_server_tools.py::test_watchlists_lazy_resolver_blocks_replacement_until_failed_close_succeeds"
"Tests/MCP/test_local_server_tools.py::test_watchlists_registration_is_storage_lazy_and_server_mode_never_resolves_path"
"Tests/Media_DB/test_media_db_v2.py::TestDatabaseCRUDAndSync::test_reading_progress_reopens_through_versioned_migration"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_credential_attached_to_same_origin_mapped_file"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_credential_withheld_from_cross_origin_mapped_file_but_both_download"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_cross_origin_redirect_strips_authorization_but_body_still_downloads"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_cross_origin_redirect_strips_client_level_default_authorization"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_gated_repo_with_resolver_provisions_and_never_leaks_token"
"Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_multi_file_source_map_urls_never_leak_into_state_manifests_or_errors"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_aggregates_and_grants"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_clamps_oversized_staged_credit_to_entry_total"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_counts_staged_credit"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_gated_repo_reports_instructions"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_insufficient_space_blocks_grant"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_stale_sidecar_credit_capped_by_actual_file_size"
"Tests/Model_Artifacts/test_preflight.py::test_preflight_upgrade_retains_prior_active_version"
"Tests/Model_Artifacts/test_preflight.py::test_probe_gating_head_does_not_follow_redirects"
"Tests/Model_Artifacts/test_provision_crash_recovery.py::test_kill_between_install_and_activate_fresh_provision_activates_with_zero_requests"
"Tests/Model_Artifacts/test_provision_crash_recovery.py::test_kill_mid_fetch_valid_sidecar_survives_and_fresh_provision_resumes"
"Tests/Model_Artifacts/test_provision_crash_recovery.py::test_reconcile_after_crash_removes_only_orphans_leaves_everything_else"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_enospc_raises_transfer_error_retryable_and_retains_staging"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_full_download_writes_sidecar_and_reports_progress"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_mid_body_disconnect_leaves_durable_sidecar_and_reprovision_resumes"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_multi_file_descriptor_raises_catalog_error_without_touching_anything"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_over_large_checkpoint_restarts_cleanly"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_restarts_from_zero_on_changed_etag"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_resumes_partial_file_with_range_request"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_skips_file_already_complete_in_sidecar"
"Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_zero_byte_file_creates_empty_destination_and_skips_network"
"Tests/Model_Artifacts/test_provision_fetch.py::test_provision_cancel_mid_fetch_releases_lease_and_preserves_prior_active"
"Tests/Model_Artifacts/test_provision_install.py::test_preverify_mismatch_persisting_past_max_refetches_raises_transfer_error"
"Tests/Model_Artifacts/test_provision_install.py::test_preverify_mismatch_refetches_once_and_recovers_on_good_content"
"Tests/Model_Artifacts/test_provision_install.py::test_provision_activates_already_installed_closure_with_zero_fetch_requests"
"Tests/Model_Artifacts/test_provision_install.py::test_provision_corrupt_payload_refetches_exactly_once_then_fails"
"Tests/Model_Artifacts/test_provision_install.py::test_provision_end_to_end_installs_and_activates_root_and_dependency"
"Tests/Model_Artifacts/test_provision_install.py::test_retryable_finalize_failure_leaves_staged_bytes_resumable_via_range"
"Tests/Model_Artifacts/test_source_map.py::test_gated_descriptor_url_does_not_block_public_mapped_files"
"Tests/Model_Artifacts/test_source_map.py::test_gated_mapped_file_detected_at_preflight_even_when_descriptor_url_is_public"
"Tests/Model_Artifacts/test_source_map.py::test_identical_source_map_at_provision_does_not_mismatch"
"Tests/Model_Artifacts/test_source_map.py::test_multi_file_artifact_provisions_end_to_end_with_source_map"
"Tests/Model_Artifacts/test_source_map.py::test_multi_file_artifact_with_dependency_provisions_via_source_map"
"Tests/Model_Artifacts/test_source_map.py::test_resolution_failure_at_provision_also_precedes_any_side_effect"
"Tests/Model_Artifacts/test_source_map.py::test_single_file_source_url_with_empty_source_map_still_works"
"Tests/Model_Artifacts/test_source_map.py::test_single_file_source_url_without_source_map_still_works"
"Tests/Model_Artifacts/test_source_map.py::test_source_url_changed_after_consent_raises_consent_mismatch"
"Tests/Model_Artifacts/test_stream_fetch.py::test_changed_last_modified_non_compliant_server_raises_restart"
"Tests/Model_Artifacts/test_stream_fetch.py::test_changed_last_modified_raises_restart_without_append"
"Tests/Model_Artifacts/test_stream_fetch.py::test_changed_validator_raises_restart"
"Tests/Model_Artifacts/test_stream_fetch.py::test_content_range_start_mismatch_raises_restart_without_append"
"Tests/Model_Artifacts/test_stream_fetch.py::test_full_fetch_writes_and_reports"
"Tests/Model_Artifacts/test_stream_fetch.py::test_last_modified_only_resume_succeeds"
"Tests/Model_Artifacts/test_stream_fetch.py::test_max_bytes_bounds_final_size"
"Tests/Model_Artifacts/test_stream_fetch.py::test_missing_content_range_on_resume_raises_restart_without_append"
"Tests/Model_Artifacts/test_stream_fetch.py::test_missing_etag_on_resume_raises_restart_without_append"
"Tests/Model_Artifacts/test_stream_fetch.py::test_no_range_support_raises_restart"
"Tests/Model_Artifacts/test_stream_fetch.py::test_resume_uses_range_and_appends"
"Tests/Model_Artifacts/test_stream_fetch.py::test_weak_etag_never_resumes"
"Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_current[sdist]"
"Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_current[source]"
"Tests/Packaging/test_installed_distribution.py::test_installed_distribution_validates_and_seeds_samira_without_package_writes[sdist]"
"Tests/Packaging/test_installed_distribution.py::test_installed_distribution_validates_and_seeds_samira_without_package_writes[source]"
"Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable"
"Tests/Packaging/test_mcp_unified_distribution.py::test_mcp_extra_installs_and_runs_from_each_isolated_artifact[sdist]"
"Tests/Packaging/test_mcp_unified_distribution.py::test_mcp_extra_installs_and_runs_from_each_isolated_artifact[wheel]"
"Tests/Performance/test_app_startup_performance.py::test_citation_artifact_reconciliation_is_deferred_and_policy_gated[False-0]"
"Tests/Performance/test_app_startup_performance.py::test_citation_artifact_reconciliation_is_deferred_and_policy_gated[True-1]"
"Tests/Performance/test_app_startup_performance.py::test_legacy_citation_migration_is_deferred_and_policy_gated[False-0]"
"Tests/Performance/test_app_startup_performance.py::test_legacy_citation_migration_is_deferred_and_policy_gated[True-1]"
"Tests/Persona_Visual/test_persona_visual_publication.py::test_publication_bounds_descriptors_at_contract_maximum"
"Tests/RAG/simplified/test_search_service.py::TestKeywordSearchRealRowMapping::test_media_types_filter_is_honored"
"Tests/RAG/simplified/test_simple_cache_basic.py::TestSimpleRAGCacheBasic::test_clear"
"Tests/RAG/simplified/test_simple_cache_concurrent.py::TestRaceConditions::test_stats_calculation_race"
"Tests/RAG/simplified/test_vector_store.py::TestChromaVectorStore::test_data_persists_across_reopen"
"Tests/RAG/simplified/test_vector_store_selection.py::TestDefaultWithoutEmbeddingsDeps::test_from_settings_defaults_to_memory"
"Tests/RAG/test_active_config_resolution.py::test_resolves_active_profiles_rag_config"
"Tests/RAG/test_active_config_resolution.py::test_top_k_only_resolution_does_not_import_torch"
"Tests/RAG/test_chunking_service.py::TestEveryChunkingMethodReturnsUsableText::test_method_yields_string_text[paragraphs-The first sentence is here. The second follows it. A third arrives.\\n\\nA second paragraph begins. It has two sentences.\\n\\nThird paragraph.The first sentence is here. The second follows it. A third arrives.\\n\\nA second paragraph begins. It has two sentences.\\n\\nThird paragraph.]"
"Tests/RAG/test_config_profiles.py::test_get_profile_manager_explicit_dir_bypasses_cache"
"Tests/RAG/test_config_profiles.py::test_legacy_blob_migration_isolates_per_entry_failures"
"Tests/RAG/test_ingestion_indexing.py::TestMediaPostIngestHook::test_callback_exception_does_not_break_ingestion"
"Tests/RAG/test_ingestion_indexing.py::TestMediaPostIngestHook::test_no_callback_for_duplicate_without_overwrite"
"Tests/RAG_Search/test_embeddings_performance.py::TestEmbeddingPerformance::test_real_model_performance"
"Tests/RAG_Search/test_fts5_match_forms_shared.py::test_plain_search_and_rag_answer_answer_the_same_inflection_miss"
"Tests/RAG_Search/test_reranker_system_prompt.py::test_a_local_provider_receives_exactly_one_system_instruction"
"Tests/RuntimePolicy/test_server_client_provider_migration_audit.py::test_legacy_server_client_builder_matches_are_listed_in_migration_audit"
"Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations"
"Tests/Skills/test_project_skills_import_modal.py::test_never_during_inflight_import_is_inert"
```
