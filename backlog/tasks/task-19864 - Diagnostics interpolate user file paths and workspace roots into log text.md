---
id: TASK-19864
title: >-
  Diagnostics interpolate user file paths and workspace roots into log text
status: In Progress
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-28'
labels:
  - privacy
  - diagnostics
  - logging
priority: medium
dependencies:
  - TASK-19321
  - TASK-19322
  - TASK-19555
---

## Description

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

## Acceptance Criteria

- [ ] In the five recorded owner files, production diagnostics do not place a
      user's absolute path, workspace root or database location into log
      message text where a less-identifying form (extension, path depth, a
      stable hash) carries the same diagnostic value
- [ ] In those owners, where a full path genuinely is the diagnostic — a "file
      not found" the user must act on — it goes through the redaction seam
      rather than being interpolated raw
- [ ] "Copy visible logs" no longer needs to warn that file names were not
      removed, or its warning is still accurate after the change
- [ ] A guard detects a newly-added diagnostic that interpolates a path-shaped
      value, so the census does not have to be retaken by hand a fourth time —
      mutation-checked by adding one such call and confirming it goes red
- [ ] The guard reports the whole set rather than aborting at the first hit
- [ ] The corrected scope is recorded in the implementation notes: these calls
      do NOT reach a persistent sink, and "Copy all" is metadata-only since
      TASK-19555 — so nobody re-rates this as a persistent-disclosure defect

## Notes

Medium, not high: the exposure is a live terminal, an in-app pane, and a
clipboard action the user takes deliberately and is warned about. It is filed
because it is the same untracked class as TASK-19321/19322 and because the
count has now been wrong in both directions — once too small (three calls, one
file), once too severe (persistent sink).

## Design

Approved design: [Diagnostic path privacy and regression guard](../../Docs/superpowers/specs/2026-08-28-diagnostic-path-privacy-and-guard-design.md).

## Implementation Plan

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
