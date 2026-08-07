---
id: TASK-3035
title: 'Second refresh: reviewed production diagnostic inventory on dev'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 12:57'
updated_date: '2026-08-07 14:10'
labels:
  - testing
  - baseline
  - security
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
  - >-
    backlog/tasks/task-1822 -
    Refresh-stale-production-diagnostic-inventory-on-dev.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository architecture gate by reviewing the production diagnostic ownership and persistent-sink topology that drifted again since task-1822's first refresh, then recording the accepted current baseline without changing runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every production diagnostic owner changed since the reviewed baseline is inspected for metadata-only safety under ADR-029.
- [x] #2 Persistent sink topology is verified unchanged or any change is explicitly reviewed and documented.
- [x] #3 The checked production diagnostic inventory exactly matches current dev source.
- [x] #4 The focused diagnostic-inventory architecture tests pass.
- [x] #5 No production runtime behavior changes are introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: This refresh applies the existing metadata-only diagnostic boundary and records current ownership; it makes no new architecture, sink, storage, or security decision.

1. Reproduce the stale-inventory failure and generate a deterministic current inventory for comparison against the reviewed baseline (last set by commit a8ef42bff).
2. Diff owners and persistent-sink topology programmatically (added/removed/changed) between the baseline and current dev source; separate real content changes from pure line-shift digest churn.
3. Review every changed/added owner's actual diagnostic call sites against ADR-029 (no message text, no user/model content, no credentials); review the persistent-sink topology change explicitly.
4. Replace only the reviewed inventory artifact; do not bless any entry found to violate ADR-029.
5. Run the focused architecture tests, inventory checker, JSON validation, and diff hygiene; run the full Architecture suite and a repo-wide --collect-only sweep; ruff on touched files.
6. Record verification and review results in a full report file, check all acceptance criteria, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed refresh, not a blind regeneration. Diffed the checker's deterministic output against the last reviewed baseline (commit a8ef42bff) programmatically: 12 added owners, 0 removed, 90 changed (60 pure digest churn from unrelated line shifts, 30 with a real call-count delta). Read every added/changed owner's actual diagnostic call sites in context (43 entries total) plus the one persistent-sink topology change (a new loguru->stdlib bridge in app.py for the rebuilt Logs screen, ADR-031). All 43 are safe: ordinary, non-schema-validated Loguru/stdlib diagnostics that PersistentDiagnosticFilter unconditionally rejects (no `_tldw_metadata_only_record` marker), logging only ids, paths, provider/model names, and exception text -- never message/prompt/transcript content or credentials. Confirmed no new persist_event/log_persistent_metadata call sites were added anywhere in the 272-commit drift window, and the marker-confinement guard test still passes. The new app.py sink is UI-only fan-out (constructs a fresh LogRecord with no marker, so it cannot reach the disk-writing handler) -- reviewed and accepted, not a new persistent sink. One historical content-logging line (WebSearch_APIs.py demo snippet logging a search answer) was already removed within this same drift window before this review. Persistent sink topology: 6 files before and after, unchanged in substance.

Verification: focused test 3/3 passed (was 1 failed/2 passed). Full Tests/Architecture/: 28 passed, 2 failed -- both in test_profile_owned_path_inventory.py, a DIFFERENT unrelated gate broken by the same task-2951 TTSPlaygroundWidget deletion (stale exception rule referencing deleted STTS_Window.py functions); confirmed pre-existing and out of scope (that checker never reads the diagnostic-inventory JSON, and this branch touches no Python source) -- not fixed here per the diff-hygiene constraint; flagged as a follow-up. JSON validated (463 owners, 1144 TASK-492 calls, 6841 TASK-494 calls, 6 sink files). Repo-wide `pytest --collect-only -q`: 31873 tests collected, 0 collection errors. No Python source changed, so ruff has nothing to check. Diff hygiene: exactly two changes -- the inventory artifact and this task file.

Modified files: Docs/security/production-diagnostic-inventory.json and this task record. Full per-entry ADR-029 judgment table in .diag-refresh-report.md (worktree-local, git-ignored, not committed).

Post-rebase incremental re-review (third commit on this branch): the coordinator rebased onto a later dev tip (origin/dev 15407a641) that advanced through 5 merged PRs since this task's 6b38a13b8 baseline, not just PR #1415 as predicted -- reported explicitly rather than silently blessed, per the coordinator's own stop condition. Delta: 2 added owners (UI/Console_Modules/message.py, prompts.py -- PR #1408 console decomposition wave 3, same split pattern as before), 4 changed (chat_screen.py -16 confirmed 1:1 moved to the two new files; AgentRuns_DB.py, speech_catalog_mixin.py, WebSearch_APIs.py all confirmed pure digest churn with zero logger-line diffs). All 6 reviewed safe under ADR-029; closest call was prompts.py logging a user-typed search query string, judged safe since it's an ordinary non-persisted Loguru call PersistentDiagnosticFilter rejects regardless of content. Sink topology unchanged. Also verified (on a disposable, git-stash-free worktree off origin/dev, not this checkout) that the coordinator's separately-flagged test_screen_size_ratchet.py failure for chat_screen.py is dev's own pre-existing issue from PR #1408, byte-identical failure on clean dev, unrelated to this branch, not touched. Full accounting in .diag-refresh-report.md (Extension 2).
<!-- SECTION:NOTES:END -->
