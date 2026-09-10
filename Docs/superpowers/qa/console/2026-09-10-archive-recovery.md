# Console archive recovery verification — 2026-09-10

PR integration base: `origin/dev` at `068535986ed005e85045cf6e08397c372377cb28`.
Branch: `codex/console-archive-recovery`.
ADR: [147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md).

The implementation was ported from the original checkout using feature-only deltas. It preserves current dev's asynchronous Console hydration, canonical Library Reader, switcher modes, workspace persona annotations and tree-wide close-loss accounting. The archive migration is 70→71. No unrelated changes from the original checkout are included.

## Passing focused checks

- Archive persistence/service/action tests: 21 passed. Real SQLite covers archive/restore/Undo, restart, counts, paging, body search, retained identity/history, concurrent writes and cancellation-safe reservations.
- Conversation service and character seek compatibility: 100 passed, including archive scope combined with workspace-union and seek filters.
- Console archive publication/action checks: 15 passed; refresh ownership, intervening archive receipts, in-flight changes and navigation away during hydration are covered. Existing cache-neighbor checks also pass.
- Library state/canvas regressions: 83 passed. Five new recovery checks, nine canonical Reader regressions and nine source/link compatibility checks pass; mounted tests cover saved-body search, complete-transcript Find, archive/Undo, separate source reuse and original Resume at 100×30 and 160×44.
- Four Console workflow cases pass: compact/wide archive → Undo → exact resume → full search; and actual Restore & resume confirmation → simulated next send for both a global conversation and an archived workspace with a reused name. The simulated response and prior transcript persist under the original conversation ID, and another draft stays intact. Fixtures use the real Console persistence path, including current dev's durable Library policy metadata.
- Twenty existing switcher trust/reuse/mode regressions pass. Four additional activation-boundary and compact-viewport checks pass.
- Selected workspace registry, Console lifecycle, close and Settings tests pass. Settings tests wait for the exact rebuilt controls instead of fixed delays.

## Verification limits

These are targeted runs, not a full-suite result. Next-send checks use a deterministic provider gateway; they do not establish live provider behavior. Compact/wide SVG captures were inspected for review/action placement. Minimal Library harness captures deliberately lack unrelated source services and are not evidence of source-service availability.

A broader historical-migration run was stopped after encountering failures outside the archive tests, and its interruption also caused pytest teardown errors. That run is not counted as passing evidence. Migration failure triage is recorded below. All new archive migration checks pass against isolated databases; no real user database was migrated.

### Historical-migration triage

The interrupted run showed 44 failing cases. Three version/column expectation tests were corrected to distinguish their historical migration from the current schema, and all three pass in isolation.

Forty cases point to unchanged dev fixtures/behavior: 28 v41 partial objects lack `_db_diagnostic_ref` (the migration implementation is unchanged and both default-shape cases reproduce on dev); 11 thinking/quick-note/retention cases reproduce using unchanged dev DB code (existing semantic-authorization triggers or modern methods called against historical schemas); one bare-open content-hash case also reproduces on dev. These are outside the archive change.

One bare-open SIGKILL test did not reach its v48 child marker within 60 seconds. Its child-process baseline was not established, so it remains unverified rather than being described as a pre-existing failure. The broader interrupted run is not a clean test-suite result.

### Static and UI checks

New Python files pass Ruff and formatting. Changed existing files add no Ruff findings over the dev baseline; fatal-error checks and `git diff --check` pass. Consolidated CSS was regenerated successfully. A final cross-module review was performed before publication.

Final review correction: archive refusal now checks all session branches, including hidden unsaved/pending messages. The new regression failed before the fix; all 13 action tests pass afterward. The focused re-review checks this correction.
