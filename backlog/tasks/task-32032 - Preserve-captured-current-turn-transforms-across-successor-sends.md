---
id: TASK-32032
title: Preserve captured current-turn transforms across successor sends
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:11'
updated_date: '2026-09-08 15:50'
labels:
  - console
  - bug
  - tracing
dependencies: []
references:
  - backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
documentation:
  - backlog/docs/console-send-diagnostics.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dictionary and other admitted current-turn text transforms must keep their saved turn ownership and remain traceable when later sends return to the ordinary saved history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An admitted current-turn dictionary transform reaches the provider with the transformed text while its saved transcript and turn ownership remain unchanged.
- [x] #2 A following ordinary send completes through the real trace boundary after the transformed turn, with both historical calls reconstructed faithfully.
- [x] #3 Changed historical rows, unknown current owners, foreign revisions, stale reservations, and mismatched policies remain fail-closed.
- [x] #4 Targeted real-database regressions and an ADR-097 amendment document and verify the bounded transform and successor transition contract.
- [x] #5 A later send after a terminal failed or stopped transformed call restores only its pinned saved source, without inventing an assistant response or admitting a pending call.
- [x] #6 A transformed send and its warm or cold successor preserve eligible saved continuations and their exact owner; an unowned or changed continuation gains no replacement authority.
- [x] #7 Current dev project-instruction ownership and captured current-turn transforms remain compatible across initial, tool, fallback, retry and successor trace paths, with targeted integration evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Record exact saved-source ownership for current-turn provider artifacts and a bounded successor transition; permit that existing semantic-revision FK on call-boundary events with a schema migration.

1. Amend ADR-097 with the admitted current-owner text-transform contract, exact durable source pin, bounded completed-turn transition, recovery/policy guards, and rejected current-version inference.
2. Add a real controller, gateway, trace factory, and isolated SQLite regression for dictionary dispatch followed by ordinary and transformed sends; establish RED before production edits.
3. Add the typed source-bearing active-request artifact, preserve only the admitted current owner, and pin the resolved revision atomically at call reservation. Add v68-to-v69 event-shape migration and ownership guard without new tables or columns.
4. Extend the existing completed-turn witness only for the exact source-artifact-to-source plus linked assistant and next user transition, retaining immutable prior heads and all latest-call, range, source, value, policy, and recovery checks.
5. Verify cold restart, edited exact source, tool successor, unknown and changed history, foreign source, stale reservation, mismatched policy, and migration rollback/upgrade with targeted real-database tests. Run affected trace suites, baseline-relative lint, changed-range formatting, and diff checks. Update task notes and documentation without commits, merging, or live profile access.
6. Integrate restored eligible saved continuation groups through the actual controller. Prove transformed warm/cold successors and negative ownership/content controls; if a physical continuation tail exposes a mismatch, preserve the exact unchanged continuation domain and narrow the existing witness to its proven message range without generic artifact admission.
7. Integrate current dev project-instruction ownership (TASK-31976.1) with the pinned current-turn transform and continuation proof. Preserve both ADR-097 invariants, add combined-path regressions where needed, verify the affected Console/project suites against current dev, and record reviewed integration and publication evidence. User explicitly authorized a PR against dev and playback validation after the original isolated implementation.
8. PR #2512 Qodo follow-up: document accepted descriptors and optional source-revision returns for the two public provenance helpers; verify existing ownership regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Admitted current-user text transforms now retain their exact saved revision while the trace stores the transformed provider value as a policy-owned ACTIVE_REQUEST artifact. Historical or unowned changes do not receive this authority. Schema v69 pins that source on the existing call-boundary revision FK; it adds no table, column, source-body retention or ownership registry.

The existing completed-turn witness restores only the proven source/response/new-user range. It supports ordinary and agent sends, tool-loop completion, cold factories, owned pre-dispatch recovery, settled failures without assistant rows and a real streaming Stop with a verified partial assistant. Exact origin revision, owner, latest call, predecessor, bounded range, terminal state, response linkage and frozen policy semantics remain mandatory. Same-message newer or equal-text reverted revisions cannot replace the origin pin. Retained saved continuations preserve their exact raw source and unchanged suffix; filtered artifact equality grants no authority to unowned continuation descriptors.

Core changes: console_chat_controller.py, console_trace_provenance.py, console_trace_final_values.py, console_trace_runtime.py and console_trace_service.py; ChaChaNotes_DB.py and the v68-to-v69 migration. Regressions are in test_console_trace_current_turn_transforms.py, test_console_trace_transform_continuations.py, test_console_trace_runtime.py, test_console_trace_service.py and the v69 migration tests. ADR required: yes; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md records the source identity, successor, recovery and privacy contracts. Console send diagnostics and lessons-testing-evidence document the verified failure meaning and testing/migration traps.

Verification: tests first reproduced the real dictionary reservation failure, source substitution, terminal successor and continuation integration failures. Final combined trace/controller/settlement/reader/semantic-revision gate: 372 passed across 13 targeted modules, with four exact independently reproduced clean-dev baseline failures deselected. The baseline failures are the custom-openai-api credential-shape case, two oversized-settlement cases and cold recovery timestamp case. Seven introduced broader-gate failures were repaired and rerun GREEN. Required DB migration/adjacent-version checks: 22 passed; 40 other migration fixture failures were separately reproduced unchanged on clean dev. Independent final review found no remaining actionable defect. Baseline-relative Ruff across 34 changed Python files has 1,015 inherited/current diagnostics and zero introduced; changed ranges/new tests are formatted and git diff --check passes.

Implementation is in /private/tmp/tldw-trace-fix-32029, based on dev 3cccd9326c556a245fc87ab5013c192242e389cf. Verification used isolated real SQLite and the production controller/gateway/trace boundaries, replacing only inference; Python 3.12.11 and SQLite 3.49.1 differ from the reported 3.13.5/3.46.1. The single metadata log does not identify the reporter's exact transform or installed commit. A source pin retains identity only; unavailable retired source projections remain refused. No live profile, full suite, commit, merge or push was used.

Final artifact checks: all six derived preflight gates passed; all ten new test files passed Ruff formatting; baseline-relative Ruff remained 1,015 inherited/current findings across 34 changed Python files with zero introduced findings; git diff --check HEAD passed. The final independent review is clear and the 372-pass targeted trace gate covers the integrated changes.

PR integration, authorized 2026-09-08: preserved the verified 47-file fix as ded80ee8e on codex/kokoro-speech-trace-recovery and integrated dev 5aeac5ab221958ae612dd84ff47e047b23cd3f5d. Resolved the shared project-context witness and provider-input paths by composing the exact source pin with the bounded project/tool suffix; retained runtime credentials for owned retries and raw-before-filter saved continuation checks. The active descriptor index now follows dev project-context ownership while preserving current-user transforms. Both ADR-097 contracts and both lesson incidents are retained. Six added real controller/HTTP/SQLite cases cover dictionary transforms with ordinary, tool and fallback predecessors, warm/cold factories and changed/disabled project guidance.

Current-dev integration validation: 121 project/transform/continuation/redaction cases passed; six added combined-path cases passed; 552 gateway/preparation/thinking/history/grammar cases passed (two numeric-loopback listener cases rerun outside the sandbox and passed). The ten-module trace/settlement/reader gate passed 340 cases with three failures, all reproduced with the same assertion outcomes on an unchanged archive of exact dev 5aeac5ab2: test_oversized_response_is_replaced_by_one_bounded_labeled_artifact, test_queued_settlement_drops_oversized_response_and_usage_values, and test_cold_restart_recovers_open_calls_monotonically_and_idempotently. The previously excluded custom-OpenAI credential case now passes with the upstream correction. No full repository sweep was run. Logs: /private/tmp/console-dev-merge-initial.log, console-dev-combined-green.log, console-dev-merge-adjacent-gate.log, console-dev-merge-loopback.log, console-dev-merge-trace-gate.log and console-dev-current-baseline-failures.log.

All six current-dev derived gates passed, including 591 diagnostic owners and 3,445 task files with no duplicate IDs. Baseline-relative Ruff across 36 changed Python files has 1,014 inherited/current findings with zero introduced; changed Console ranges were formatted with AST equality checked, and the new combined test is fully formatted. Publication and real audio playback are handled by the coordinating task after final integration review.

Final merge review is clear. The independent reviewer compared the runtime, service and witness changes with both parents and passed 43 selected project/source/retained-value/owned-retry/redaction cases. Additional actual-controller warm/cold probes preserved all three together: a saved Moonshot continuation with a credential canary, real bound AGENTS context with credential/PII filtering, and transformed-user restoration on an ordinary successor under one frozen conversation privacy policy; exact raw provider checkpoints and the original trace remained unchanged, and Capture-Off controls passed. All ten added test modules and the new combined-project test passed Ruff formatting. Staged diff --check against origin/dev passes; four first-parent whitespace warnings are unchanged upstream Library/critique files outside the PR diff. Console integration is complete; the coordinating task is completing real audio validation before committing the merge and publishing the PR.

PR #2512 Qodo follow-up: added Google-style Args and Returns sections to current_turn_source_revision_id and saved_response_source_revision_id, describing accepted provenance descriptors, exact saved owner identifiers and unsupported-shape None results. No ownership or privacy logic changed. The adjacent admission/provenance gate passed 225 tests; the preceding rebased integration gate passed 173 relevant UI/TTS/DB/Console tests. Affected formatting, baseline-relative Ruff with zero new findings and all six derived checks pass. The existing ADR-097 contract is unchanged. Publication notes supersede the earlier no-commit/no-push investigation scope under the user's explicit PR and merge authorization.
<!-- SECTION:NOTES:END -->
