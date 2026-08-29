---
id: TASK-19864
title: Diagnostics interpolate user file paths and workspace roots into log text
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-29 01:37'
labels:
  - privacy
  - diagnostics
  - logging
dependencies:
  - TASK-19321
  - TASK-19322
  - TASK-19555
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: surfaced during **TASK-19572**'s review round, whose reviewer corrected
an earlier over-statement of the same finding. Re-measured at `3605bd52d`.
Same class as the open **TASK-19321** / **TASK-19322**.

Production diagnostics interpolate the user's absolute file paths, workspace
roots and database locations directly into log message text. A path is not
inert: it names projects, clients, employers and directory structures the user
never chose to disclose.

**The scope of this finding has been corrected twice, so record it precisely
rather than re-deriving it from folklore:**

1. These calls do **not** reach a persistent on-disk sink.
   `PersistentDiagnosticFilter` (`Utils/persistent_diagnostics.py:257`) admits
   only records explicitly marked by `log_persistent_metadata` / `persist_event`;
   probing `filter(<record>)` for one of these calls returns `False`. An earlier
   report in this programme claimed they were writing to a persistent sink.
   They are not.
2. "Copy all" in the Logs window is **no longer** an exposure path. TASK-19555
   converted it to the metadata-only form (`UI/Logs_Window.py:533`) — timestamps,
   loggers, levels and exception types, no message bodies.

What remains, and is the actual reason to fix this:

- Terminal output, where the paths appear verbatim.
- The in-app Logs pane, which renders message bodies.
- **"Copy visible logs"** (`UI/Logs_Window.py:507`), which copies message text
  to the clipboard and whose own notification tells the user the truth:
  *"Recognised key formats and your account name were removed; file names and
  search terms were not."* `redact_user_paths` collapses the home prefix and
  the account name; a workspace root outside `$HOME` survives intact, and every
  leaf directory and file name survives in all cases.

Census at `3605bd52d` (larger than the 21-call sample the reviewer took — this
is a class, not a list):

| File | Interpolated value | Calls |
| --- | --- | --- |
| `Utils/file_handlers.py` | `{file_path}` | 10 |
| `DB/ChaChaNotes_DB.py` | `{self.db_path_str}` | 82 |
| `UI/Screens/change_review_screen.py` | `{root!r}` ×6, `{roots!r}`, `{remote!r} at {root!r}` | 8 |
| `Widgets/Console/console_conversation_inspector.py` | path arguments | ~5 |
| `Workspaces/git_workspace.py` | `{root}` | 1 |

Because the census keeps coming out different depending on who counts, the
outcome this task needs is a **rule plus a check**, not a list of edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In the five recorded owner files, production diagnostics do not place a
      user's absolute path, workspace root or database location into log
      message text where a less-identifying form (extension, path depth, a
      stable hash) carries the same diagnostic value
- [x] #2 In those owners, where a full path genuinely is the diagnostic — a "file
      not found" the user must act on — it goes through the redaction seam
      rather than being interpolated raw
- [x] #3 "Copy visible logs" no longer needs to warn that file names were not
      removed, or its warning is still accurate after the change
- [x] #4 A guard detects a newly-added diagnostic that interpolates a path-shaped
      value, so the census does not have to be retaken by hand a fourth time —
      mutation-checked by adding one such call and confirming it goes red
- [x] #5 The guard reports the whole set rather than aborting at the first hit
- [x] #6 The corrected scope is recorded in the implementation notes: these calls
      do NOT reach a persistent sink, and "Copy all" is metadata-only since
      TASK-19555 — so nobody re-rates this as a persistent-disclosure defect
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this enforces ADR-029's existing producer-side diagnostic privacy rule
without changing sink admission, storage, ownership, or service contracts.

Detailed plan: [TASK-19864 diagnostic path privacy implementation](../../Docs/superpowers/plans/2026-08-28-task-19864-diagnostic-path-privacy-implementation.md).

1. Add born-red AST-scanner tests for path expressions, safe transforms,
   alias propagation, multiplicity, and whole-set reporting.
2. Extend the existing diagnostic inventory scanner and advance its generated
   artifact to schema version 3 with explicit `legacy_unreviewed` candidates.
3. Add an owned-file gate and runtime sentinel tests for the five recorded
   owners, then prove them red against the current diagnostics.
4. Replace raw file/database locations with extensions, stable fingerprints,
   counts, or exception types; remove path-bearing traceback capture.
5. Repair Change Review, Git workspace detection, and Inspector diagnostics
   while preserving their user-facing recovery behavior.
6. Classify and repair the inherited `virtual_cli_provider.py` inventory drift
   that is already red on the branch's `dev` base.
7. Review and regenerate the production diagnostic inventory, restamp its
   dependent summarization fixture, and pin the still-accurate Copy visible
   warning.
8. Run focused behavior, architecture, formatting, and diff gates; self-review
   the complete patch before checking criteria and adding implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the corrected live-diagnostic privacy scope and regression guard.

- Exposure scope: the repaired calls do not enter the persistent sink. Remaining exposure was terminal output, the in-app Logs message pane, and deliberate Copy visible logs. Copy all remains metadata-only under TASK-19555. The existing Copy visible warning remains unchanged and accurate.
- Owner repairs: `Utils/file_handlers.py` now logs suffix, handler, stable path fingerprint, and exception type; `DB/ChaChaNotes_DB.py` uses a cached database fingerprint or fixed `memory` sentinel and removes raw exception/traceback detail without changing database operations or caller-facing exceptions; `UI/Screens/change_review_screen.py` fingerprints roots/remotes and includes operation plus exception type; `Widgets/Console/console_conversation_inspector.py` fingerprints log destinations while preserving actionable user-visible destination text; `Workspaces/git_workspace.py` fingerprints detection roots. TASK-19936 is folded into the Change Review repair, with its validation-failure early return and banner behavior preserved.
- Inherited drift: `Agents/virtual_cli_provider.py` was already red against the branch base because callback exceptions were logged raw; it now records fixed event metadata and exception type, with focused coverage.
- Guard and artifacts: the AST scanner covers f-strings, logger arguments, percent/dot formatting, aliases, lexical shadowing, traceback capture, safe transforms, duplicates, and complete-result reporting. Safe `type(value).__name__` recognition now requires the genuine unshadowed built-in; parameter, local-assignment, enclosing-scope, and module-scope shadows remain candidates. Inventory schema 3 records 541 owners, 1260 TASK-492 calls, 7396 TASK-494 calls, 8 sink files, and 717 candidates in 120 owner groups; every candidate is explicitly `legacy_unreviewed`, and the five governed owners have zero candidates. The no-write checker reports exact synchronization with the dependent summarization fixture.
- Fingerprint semantics: path/database references use deterministic SHA-256 `content_fingerprint` values so repeated values remain correlatable without emitting raw paths or leaf names; in-memory databases use a fixed sentinel. Scanner `call_digest` identity is line-independent, preserving identity across source movement while retaining duplicate findings.
- Behavior and review: user-facing copy, actionable destination notifications, Change Review early returns/banner text, file-processing outcomes, database statements/transactions/returns, and caller-facing exception behavior are preserved. TASK-15103 artifacts are untouched. Generated changes are limited to the schema-3 production inventory and synchronized summarization fixture. Formatter-only churn is confined to touched files; the two approved malformed-docstring normalizations are `ConsoleConversationInspector.action_refresh` and `test_status_badges`.
- Verification: the expanded 18-file Ruff check passed, and Ruff format check reported all 18 files already formatted. The no-write inventory checker reported 541 owners, 1260 TASK-492 calls, 7396 TASK-494 calls, and 8 sink files with no drift. The expanded focused suite passed 621 tests across the owner, architecture/inventory, summarization privacy, Logs share-path, UI/Inspector (including Change Review push and Console context modal), workspace, and virtual-CLI modules. This was not a full-suite run.
- Mutation evidence: (a) two raw `file_path` diagnostics in `_extract_local_ingest_text` produced two complete owner-gate candidates and restored green; (b) one `self.db_path_str` integrity diagnostic failed both the runtime database privacy assertion and owner architecture gate, then restored green; (c) the historical TASK-19936 `raw!r` diagnostic failed its UI/privacy test while the empty-banner early return remained intact, then restored green; (d) one path-bearing `logger.opt(exception=True)` call failed both runtime traceback privacy and the architecture traceback-capture gate, then restored green; (e) the four shadowed-`type` cases all failed born-red with zero candidates and passed after the scanner required the unshadowed built-in, while the genuine built-in control remained green. Every mutation was restored immediately and the worktree was clean before the next.
- Modified implementation/artifact files: `scripts/check_persistent_diagnostic_inventory.py`; `Docs/security/production-diagnostic-inventory.json`; `Tests/fixtures/summarization_diagnostic_review.json`; `tldw_chatbook/Utils/file_handlers.py`; `tldw_chatbook/DB/ChaChaNotes_DB.py`; `tldw_chatbook/UI/Screens/change_review_screen.py`; `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`; `tldw_chatbook/Workspaces/git_workspace.py`; `tldw_chatbook/Agents/virtual_cli_provider.py`; the modified test modules under `Tests/Architecture`, `Tests/Utils`, `Tests/DB`, `Tests/UI`, `Tests/Workspaces`, and `Tests/Agents`, plus `Tests/test_logs_share_path_privacy.py`; the approved design and detailed plan; and this task record. `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` was existing focused verification coverage and was not modified.
- ADR: no new ADR was required. This directly enforces [ADR-029](../decisions/029-local-private-data-boundary.md) without changing storage, sink admission, ownership, or service contracts.
- Lessons: no new generalized incident was found beyond the existing testing-evidence, backlog-hygiene, and live-verification lessons, so no lessons file was changed.
- Administrative closeout: after final technical/spec review, the user explicitly accepted TASK-19936's recorded process deviation. That acceptance authorizes both tasks' final Done status without retroactively erasing the administrative timing mistake.

Inherited warnings only: the environment reports a Requests dependency-version warning, existing SyntaxWarnings in unrelated modules, and sandbox-denied temporary-directory cleanup warnings. The local branch is ahead of and behind the current moving `origin/dev`; it was audited as requested without rebasing. None was a TASK-19864 failure.
<!-- SECTION:NOTES:END -->

## Notes

Medium, not high: the exposure is a live terminal, an in-app pane, and a
clipboard action the user takes deliberately and is warned about. It is filed
because it is the same untracked class as TASK-19321/19322 and because the
count has now been wrong in both directions — once too small (three calls, one
file), once too severe (persistent sink).

## Design

Approved design: [Diagnostic path privacy and regression guard](../../Docs/superpowers/specs/2026-08-28-diagnostic-path-privacy-and-guard-design.md).
