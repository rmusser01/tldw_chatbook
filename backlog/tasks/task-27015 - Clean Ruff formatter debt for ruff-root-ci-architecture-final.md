---
id: TASK-27015
title: Clean Ruff formatter debt for ruff-root-ci-architecture-final
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
  - TASK-26933
  - TASK-26934
  - TASK-26935
  - TASK-26936
  - TASK-26937
  - TASK-26938
  - TASK-26939
  - TASK-26940
  - TASK-26941
  - TASK-26942
  - TASK-26943
  - TASK-26944
  - TASK-26945
  - TASK-26946
  - TASK-26947
  - TASK-26948
  - TASK-26949
  - TASK-26950
  - TASK-26951
  - TASK-26952
  - TASK-26953
  - TASK-26954
  - TASK-26955
  - TASK-26956
  - TASK-26957
  - TASK-26958
  - TASK-26959
  - TASK-26960
  - TASK-26961
  - TASK-26962
  - TASK-26963
  - TASK-26964
  - TASK-26965
  - TASK-26966
  - TASK-26967
  - TASK-26968
  - TASK-26969
  - TASK-26970
  - TASK-26971
  - TASK-26972
  - TASK-26973
  - TASK-26974
  - TASK-26975
  - TASK-26976
  - TASK-26977
  - TASK-26978
  - TASK-26979
  - TASK-26980
  - TASK-26981
  - TASK-26982
  - TASK-26983
  - TASK-26984
  - TASK-26985
  - TASK-26986
  - TASK-26987
  - TASK-26988
  - TASK-26989
  - TASK-26990
  - TASK-26991
  - TASK-26992
  - TASK-26993
  - TASK-26994
  - TASK-26995
  - TASK-26996
  - TASK-26997
  - TASK-26998
  - TASK-26999
  - TASK-27000
  - TASK-27001
  - TASK-27002
  - TASK-27003
  - TASK-27004
  - TASK-27005
  - TASK-27006
  - TASK-27007
  - TASK-27008
  - TASK-27009
  - TASK-27010
  - TASK-27011
  - TASK-27012
  - TASK-27013
  - TASK-27014
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-root-ci-architecture-final -->
<!-- TASK-26000-PATHS-SHA256: 2ac02c1f910e1b9a8d013de104b9cb236536073ee3f0f2e6dec95b6664b78679 -->
<!-- TASK-26000-FINAL: true -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-root-ci-architecture-final` Ruff formatter batch at the owner boundary recorded as: Root scripts, CI/architecture guards, packaging helpers, and the final repository gate; any post-cut unassigned failure blocks and requires a separate correction record.. The focused test surface recorded by TASK-26000 is `["Tests/App", "Tests/Architecture", "Tests/CI", "Tests/ProductionApp"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  ".github/scripts/generate_test_summary.py",
  "Docs/Development/test_logits_openai.py",
  "Docs/Development/textual_link_buttons_demo.py",
  "Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/drive_tui.py",
  "Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/probe_headless.py",
  "Docs/superpowers/qa/2026-08-15-local-thinking-controls-live-verification/live-verify-effort.py",
  "Docs/superpowers/qa/2026-08-15-local-thinking-controls-live-verification/live-verify.py",
  "Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/oracle_run.py",
  "Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py",
  "Docs/superpowers/qa/2026-08-16-rag-semantic-identity/route_probe.py",
  "Docs/superpowers/qa/2026-08-18-clarification-gate/census.py",
  "Docs/superpowers/qa/2026-08-18-granularity-census/granularity_census.py",
  "Docs/superpowers/qa/2026-08-18-hyde-census/hyde_census.py",
  "Docs/superpowers/qa/2026-08-18-hyde-census/hyde_probe.py",
  "Docs/superpowers/qa/2026-08-18-merge-tiering/tier_observability_census.py",
  "Docs/superpowers/qa/2026-08-18-prompts-seam/seam_effect.py",
  "Docs/superpowers/qa/2026-08-18-residual-zero-row/reachability_census.py",
  "Docs/superpowers/qa/anthropic-native-2026-07/anthropic_gate.py",
  "Docs/superpowers/qa/cohere-native-2026-07/cohere_gate.py",
  "Docs/superpowers/qa/console-prompt-improvement-2026-08/capture_qa.py",
  "Docs/superpowers/qa/console-ux-review-2026-08/uat_console.py",
  "Docs/superpowers/qa/console-watchlists-workflow-2026-08/redaction_check.py",
  "Docs/superpowers/qa/google-native-2026-07/google_gate.py",
  "Docs/superpowers/qa/library-media-reader-2026-08/capture_reader.py",
  "Docs/superpowers/qa/mcp-hub-phase5-2026-07/fake_llm_server.py",
  "Docs/superpowers/qa/native-tool-calls-2026-07/native_gate.py",
  "Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py",
  "Docs/superpowers/qa/rag-settings-sp3-2026-07/capture_rag_settings.py",
  "Docs/superpowers/qa/rag-settings-v2-2026-07/capture_rag_settings_v2.py",
  "Docs/superpowers/qa/skills-script-execution-2026-07-25/seed3.py",
  "Docs/superpowers/reviews/evidence/task-22033/task22033_live_matrix_runner.py",
  "Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py",
  "Helper_Scripts/Benchmarks/record_research_baseline.py",
  "Helper_Scripts/Benchmarks/token_estimate_benchmark.py",
  "Helper_Scripts/Examples/custom_splash_cards/custom_animation_effect.py",
  "Helper_Scripts/Examples/examples/audit_system_demo.py",
  "Helper_Scripts/Higgs-Install/verify_higgs_installation.py",
  "Helper_Scripts/Mass-Ingestion/mass_ingest.py",
  "Helper_Scripts/Prompts/Convert_Fabric_Prompts.py",
  "Helper_Scripts/Prompts/Ingest_Prompts.py",
  "Helper_Scripts/Prompts/Prompts_Dump.py",
  "Helper_Scripts/UI/visualize_layout_clean.py",
  "Helper_Scripts/UI/visualize_textual_layout.py",
  "Helper_Scripts/UI/visualize_textual_ui.py",
  "Helper_Scripts/UI/visualize_ui_simple.py",
  "Helper_Scripts/fixed_auto_review.py",
  "Packaging/check_manifest.py",
  "Packaging/common/version.py",
  "Packaging/macos/build_app.py",
  "Packaging/windows/build_exe.py",
  "Packaging/windows/build_windows.py",
  "Tests/App/test_startup_init_hygiene.py",
  "Tests/App/test_unhandled_exception_event.py",
  "Tests/App/test_worker_failure_event.py",
  "Tests/Architecture/test_backwards_select_option_guard.py",
  "Tests/Architecture/test_default_timeout_session_guard.py",
  "Tests/Architecture/test_framework_armed_clock_inventory.py",
  "Tests/Architecture/test_no_blocking_io_on_message_pump.py",
  "Tests/Architecture/test_on_mount_super_guard.py",
  "Tests/Architecture/test_persistent_diagnostic_inventory.py",
  "Tests/Architecture/test_progress_widget_clock_guard.py",
  "Tests/Architecture/test_python_floor_syntax.py",
  "Tests/Architecture/test_reactive_mutable_default_inventory.py",
  "Tests/Architecture/test_security_logger_write_surface.py",
  "Tests/Architecture/test_timer_path_static_update_inventory.py",
  "Tests/Architecture/test_vendor_pin_consistency.py",
  "Tests/CI/test_backlog_task_id_uniqueness.py",
  "Tests/CI/test_ci_queue_pressure_contract.py",
  "Tests/CI/test_task19637_platform_evidence.py",
  "Tests/Docs/test_console_library_controls_docs.py",
  "Tests/Helper_Scripts/test_record_research_baseline.py",
  "Tests/Packaging/test_exchange_export_trajectory_deferral.py",
  "Tests/Packaging/test_installed_distribution.py",
  "Tests/Packaging/test_persona_buddy_import_closure.py",
  "Tests/Packaging/test_rag_boot_import_closure.py",
  "Tests/Packaging/test_research_workspace_import_closure.py",
  "Tests/ProductionApp/test_chat_root_state_removal.py",
  "Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py",
  "Tests/ProductionApp/test_provider_selection_ownership.py",
  "Tests/ProductionApp/test_service_composition_lifecycle.py",
  "check_app.py",
  "check_textual.py",
  "clean_cache.py",
  "minimal_app.py",
  "run_all_tests_with_report.py",
  "run_rag_tests.py",
  "scripts/check_index_plan_pins.py",
  "scripts/check_schema_table_allowlist.py",
  "verify_ui.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any post-cut unassigned failure blocks this gate, is never absorbed into the pinned counts or current batches, and requires a separate correction record. <!-- TASK-26000-CONTRACT: repository-zero-gate --><!-- TASK-26000-CONTRACT: post-cut-unassigned-correction -->
<!-- AC:END -->
