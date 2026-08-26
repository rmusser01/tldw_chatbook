# Tests/UI full-suite sweep — task-15211 (2026-08-13)

**The first end-to-end run of the whole `Tests/UI` suite.** 503 modules in 16
checkpointed chunks against a worktree frozen at dev `e9f5f80e4`:
**10,811 passed, 117 failed rows, ~13 errors** (teardown-class, see below).
Raw per-chunk logs and the failure list were captured per-chunk so environment
kills lose one chunk, not the run — this is what let the sweep survive the
process kill and TCC lockout that had destroyed all three earlier monolithic
attempts.

## The task's core question: what attempts network egress?

**One distinct source in the entire suite**: `llm_screen._probe_local_server`
-> `127.0.0.1:11434`, firing at TEARDOWN of any test that mounts the Lab/LLM
screen (interval + deferred-mount one-shot flushing during shutdown).
Fixed during the sweep (PR #1596: widget-owned worker + the same autouse
harness stub the Console screen got in task-15111). The task-15111-era
"tests POST real inference" class did NOT reappear. Chunk 12 shows a
possible sibling: two provider-Test-button teardown errors in settings —
see the follow-up task.

## Fixed during the sweep (merged)

- **PR #1591** — four NEW unbounded background waits (the task-14912 hang
  class), caught by the guard on chunk 0.
- **PR #1596** — the Ollama probe above. Also cured the sweep's own chunk-5
  **failure-to-exit** (pytest printed its summary, then sat at zero CPU with
  two event loops parked in kevent and the main thread joining a non-daemon
  thread). Re-run of the same 32 modules on post-fix dev: clean exit, zero
  egress. This failure mode is the best explanation for every earlier
  full-suite attempt dying.
- **PR #1603** — 35 task files with lowercase `id: task-` frontmatter the
  maturity harness counts as no id at all (chunk 10's catch).

## Failure inventory (frozen-tree dev-red, attributed)

Chunk summaries:
- chunk 0 (rc=1): 1 failed, 344 passed, 4 warnings in 99.96s (0:01:39)
- chunk 1 (rc=1): 2 failed, 695 passed, 2 warnings in 473.66s (0:07:53)
- chunk 2 (rc=1): 5 failed, 880 passed, 1 xfailed, 8 warnings in 794.70s (0:13:14)
- chunk 3 (rc=1): 4 failed, 718 passed, 3 warnings in 581.16s (0:09:41)
- chunk 4 (rc=1): 13 failed, 310 passed, 2 warnings in 216.69s (0:03:36)
- chunk 5 (rc=143): 9 failed, 589 passed, 1 skipped, 3 warnings, 1 error in 894.06s (0:14:54)
- chunk 6 (rc=1): 1 failed, 575 passed, 1 skipped, 3 warnings, 6 errors in 459.39s (0:07:39)
- chunk 7 (rc=1): 32 failed, 954 passed, 2 warnings in 1860.63s (0:31:00)
- chunk 8 (rc=1): 16 failed, 1630 passed, 9 warnings, 2 errors in 1823.60s (0:30:23)
- chunk 9 (rc=1): 509 passed, 4 warnings, 1 error in 253.75s (0:04:13)
- chunk 10 (rc=1): 8 failed, 736 passed, 7 warnings, 2 errors in 707.39s (0:11:47)
- chunk 11 (rc=1): 8 failed, 766 passed, 3 warnings in 623.45s (0:10:23)
- chunk 12 (rc=1): 7 failed, 746 passed, 3 warnings, 2 errors in 597.79s (0:09:57)
- chunk 13 (rc=1): 2 failed, 482 passed, 3 warnings in 424.21s (0:07:04)
- chunk 14 (rc=1): 5 failed, 460 passed, 6 warnings, 4 errors in 549.97s (0:09:09)
- chunk 15 (rc=1): 4 failed, 417 passed, 2 warnings in 905.67s (0:15:05)

Failures by module:
-  15  test_library_file_notes_git.py
-  15  test_library_shell.py
-   7  test_console_staged_evidence_strip.py
-   6  test_destination_visual_parity_correction.py
-   4  test_console_shell_regions.py
-   4  test_library_file_notes_git_push.py
-   4  test_personas_generation_wiring.py
-   4  test_workbench_visual_snapshots.py
-   3  test_console_rail_width_budget.py
-   3  test_library_prompts_canvas.py
-   3  test_settings_rag_profile_region.py
-   2  test_console_dictionary_send_integration.py
-   2  test_console_world_info_send_integration.py
-   2  test_library_choice_strips.py
-   2  test_library_export_receipt.py
-   2  test_library_file_notes_workspace.py
-   2  test_library_ingest_structural.py
-   2  test_settings_workspaces_category.py
-   2  test_ui_responsiveness.py
-   1  test_background_signal_bounds.py
-   1  test_console_citation_sources.py
-   1  test_console_composer_collapse.py
-   1  test_console_fleet_discoverability.py
-   1  test_console_internals_decomposition.py
-   1  test_console_live_work_handoffs.py
-   1  test_console_rail_sections.py
-   1  test_console_shell_chip_actions.py
-   1  test_console_tab_scope.py
-   1  test_destination_headers.py
-   1  test_focus_accessibility.py
-   1  test_library_ingest_keyboard.py
-   1  test_library_multiselect_notes.py
-   1  test_library_skills_canvas.py
-   1  test_product_maturity_phase1_empty_setup_states.py
-   1  test_product_maturity_phase1_first_run.py
-   1  test_product_maturity_phase1_harness.py
-   1  test_product_maturity_phase1_keyboard_focus.py
-   1  test_product_maturity_phase6_first_time_release_replay.py
-   1  test_product_maturity_phase6_focus_visual_sweep.py
-   1  test_product_maturity_phase6_packaging_data_safety.py
-   1  test_product_maturity_phase6_power_user_replay.py
-   1  test_product_maturity_phase6_recovery_docs.py
-   1  test_schedules_ux_fixes.py
-   1  test_screen_navigation.py
-   1  test_settings_configuration_hub.py
-   1  test_settings_footer_hints.py
-   1  test_settings_model_catalog_toggles.py
-   1  test_speech_rail_navigation.py
-   1  test_speech_tts_settings_ownership_closeout.py
-   1  test_stts_profile_library.py
-   1  test_unified_shell_phase5_recovery_taxonomy.py
-   1  test_watchlists_check_now_failure.py

### Clusters

| cluster | size | attribution / next step |
|---|---|---|
| file-notes drift (color contract x10, push copy x3, library_shell notes focus/rows x15, workspace/export/choice-strip stragglers) | ~35 | the file-notes arcs (#1515/#1551 era); modules never re-run whole since — task filed |
| console contract drift (staged-evidence "Sources: N staged" x6 [7dbbc401b dropped the suffix], shell-region size2 geometry x4, rail width x2, chip-swap signature, dictionary/world-info sends x4, misc) | ~25 | console batch task filed |
| destination/workbench visual contracts (parity x~9, visual snapshots x4) | ~13 | one task; snapshot refresh needs a HUMAN eye on renders |
| settings (rag_profile_region x3, workspaces x2, catalog, footer-hints [known, 537451cb8..61f6ae575]) | ~7 | settings batch in the same task as the probe sibling |
| stale doubles (set_presentation_context, _library_export_quality_choices_visible, _library_notes_mutation_in_flight, memo-split residue) | >=5 | recurring class; folded into the batch tasks |
| StopIteration -> RuntimeError in coroutine (library_prompts_canvas x2) | 2 | POSSIBLE REAL BUG (PEP-479 conversion) — own task, high priority within the batch |
| personas_generation_wiring | 4 | singleton batch |
| product_maturity_phase1 timeouts ("condition not met within 10s") | 3 | suspect CONTENTION (4 concurrent suites); re-run before filing as real |
| speech / schedules / navigation / responsiveness / watchlists / skills / phase6 singletons | ~12 | singleton batch |

Frozen-tree residue that is NOT open work: settings_configuration_hub /
workbench_contract rows (fixed in #1554), stts wait-bounds rows (#1591),
every `ERROR at teardown` naming an Ollama/llamacpp/lab test (#1596).

### Notable negative results

- task-15741's blank-note `ConflictError` did NOT reproduce anywhere in the
  full sweep (library_shell's chunk-8 failures are all focus/rows drift).
- No test attempted egress to anything but 127.0.0.1:11434.

## Method notes (why this run finished when three before it did not)

Checkpointed chunks with per-chunk logs + resumability; a hung chunk was
diagnosed live (zero CPU accrual + native thread sample), killed, and its
already-printed summary still recorded; the worktree stayed FROZEN while
fixes went out on separate branches from a second worktree.
