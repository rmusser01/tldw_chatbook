---
id: TASK-37300
title: Console trace copying and full-message note capture
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28'
labels:
  - console
  - clipboard
  - notes
dependencies: []
---

## Description

Let users share a complete assistant response or the current chat's active
branch for review in another chat, and capture a whole selected message in a
note without losing its tail. Distinguish staying in Console from creating a
note and opening it in Notes.

## Acceptance Criteria

- [x] #1 Copy all copies one assistant response's retained thinking, tool calls, full results, diffs, and answer as readable expanded Markdown, regardless of collapse state.
- [x] #2 Copy all messages includes the entire current chat's active branch, including user messages and all assistant traces, without depending on mounted history or including inactive branches/other chats.
- [x] #3 The two copy scopes use the same formatter, preserve textual payloads, and label known missing, truncated, proprietary, or incomplete content honestly.
- [x] #4 Clipboard actions preserve the draft and menu focus lifecycle; stale targets and failed formatting do not replace the clipboard, and old workers do not overwrite newer copy requests.
- [x] #5 The note buttons are labelled Create Note and Create+Go to Note; both save identical selected content, and only the latter opens the exact created note after a successful save.
- [x] #6 Select full message selects the complete current user/assistant body, and both note actions preserve selections beyond 4000 characters without substituting another arbitrary capture cap.
- [x] #7 Quoting, side-chat requests, review feedback, and bounded highlight previews retain their existing 4000-character behavior; abbreviated previews do not shorten copied or saved text.
- [x] #8 Note creation respects existing Notes identity, authorization and temporary-chat policy, runs blocking I/O off the UI loop, and avoids duplicate writes during activation/navigation recovery.
- [x] #9 Every menu action remains reachable by mouse/keyboard in short terminals, ANSI mode and resize cycles, with no new terminal-convention/global shortcut conflicts.
- [x] #10 Targeted automated checks, scoped static analysis, disposable-state live verification, self-review and relevant documentation support the behavior; limitations are recorded without claiming a full-suite pass.

## Implementation Plan

1. Apply the user-approved full-message selection/uncapped note extension in the updated design and implementation plan.
2. Add and test a pure per-response Markdown formatter and compose it for current-chat copying, preserving existing ownership and all retained thinking blocks.
3. Wire source-pinned copy requests, background rendering and guarded clipboard publication through the existing Console controllers.
4. Expose Copy all, Copy all messages and Select full message using existing menu state, focus and compact-layout behavior.
5. Route full selected text through one Notes creation path for both approved labels; use returned note-id navigation and safe recovery after save.
6. Run targeted functional/governance checks, scoped lint/format checks and disposable-state live journeys; update docs and record implementation evidence before DoD closeout.

Design: [Console Copy all response trace](../../Docs/superpowers/specs/2026-10-28-console-copy-all-response-trace-design.md)

Detailed plan: [Console trace copy and note capture](../../Docs/superpowers/plans/2026-10-28-console-trace-copy-and-note-capture.md)

ADR required: no

ADR path: backlog/decisions/068-console-text-selection-and-annotations.md;
backlog/decisions/195-console-live-tool-call-presentation.md;
backlog/decisions/210-console-region-ownership.md

Reason: additive Console actions within existing response ownership, display
capture, clipboard transport, Notes authority and navigation. No schema,
retention, dependency, provider/runtime or ownership boundary change. ADR-031
and ADR-150/161 continue to apply.

## Planning Context

The user approved the complete implementation plan on 2026-10-28, explicitly
including Select full message and removing the 4000-character cap from both
note actions. Execution is on codex/console-trace-copy in the original checkout.

Task ID allocation scanned 353 distinct local/remote ref heads and 110 readable
worktree buckets. Maximum filename ID observed: 37270; allocated with a 30-ID
lead. One worktree archive bucket was unreadable, so its uncommitted task IDs
were not verified. Recheck uniqueness before integration. The Backlog CLI was
not available in this shell; this planning record was created directly and
must not be reported as CLI-closeout complete.

## Implementation Notes

Implemented the approved Console copy scopes and full-selection note capture
on `codex/console-trace-copy` in the original checkout.

- Added one pure Markdown formatter using existing assistant ownership and
  activity ordering. It preserves displayable thinking, tool arguments, full
  retained results, before/after file contents, raw-command execution facts
  (including truncation), and the answer. Private provider captures, system
  messages and binary attachment contents stay outside the projection.
- Added **Copy all**, **Copy all messages** and **Select full message**. Menus
  capture source identity and active-path message IDs; stale branch requests
  are refused, and older formatting workers cannot overwrite newer requests.
  Selected-response More uses the same copy path, including incomplete runs.
- Added **Create Note** and **Create+Go to Note** using one uncapped, validated
  selection path, canonical Notes authority/identity, worker-owned SQLite
  lifetimes and the existing temporary-chat gate. Duplicate activation and
  navigation recovery reuse the saved note ID. Prompt quotes remain capped.
- The real Library arrival journey exposed a first-visit timeout while mounting.
  Its initial deep-linked note load now starts after first paint through the
  Notes controller, with captured-target/source/view guards and the original
  deadline. Both screen-size ratchets remain unchanged and pass.
- Updated the Console selection guide, chat basics and chat-pipeline architecture.

Core files: `Chat/console_response_trace.py`, `Chat/console_message_actions.py`,
`Widgets/Console/console_selection_menu.py`, `console_transcript.py`,
`UI/Console_Modules/{review_selection,message,wiring}.py`, the thin ChatScreen
boundary, `UI/Library_Modules/library_notes_controller.py` and Library's initial
load delegate. Generated selection-menu CSS was regenerated from Python sources.

ADR required: no. Existing ADR-068, ADR-195 and ADR-210 apply; ADR-031 and
ADR-150/161 govern keys and UI tokens. No schema, retention, provider, dependency
or navigation-protocol change.

### Verification and remaining closeout

Final focused feature run: **158 passed**. Final real Library arrival plus all
screen-size ratchets: **5 passed**. Four native terminal-driver journeys passed.

- Tests cover both note buttons at 3999, 4000, 4001, 10000 and 100000 characters,
  exact Unicode/whitespace, a blocked real SQLite write with an event-loop
  heartbeat, complete results, collapsed sections, offscreen history, native
  active-branch snapshots, stale menus/workers, cancellation, duplicate saves,
  temporary chats and failed navigation recovery.
- Real Console and Library screens verify the exact created note in an armed
  editor; the actual source session title is retained. The canonical existing
  Library loader control also passed with its documented bootstrap enrollment.
- Four disposable journeys passed through Textual's native terminal driver.
  OS acceptance of OSC 52, including terminal size limits, is not acknowledged
  by that transport and was not independently verified.
- One fresh read-only code review identified stale same-session branch copying
  and missing raw-command execution details. Both were reproduced and fixed;
  command argument retention was verified as well. No deferred minor findings.
- CSS reproduction, task-ID uniqueness and diff whitespace checks pass. Scoped
  Ruff comparison reports zero new findings and zero changed-range format drift;
  large legacy files retain their pre-existing diagnostics.
- Existing unrelated checks remain red: ten Canvas/refused-marker expectations,
  two legacy full-screen copy cases (both reproduced on pristine base), one
  obsolete frozen ChatScreen projection assertion, whole-tree scanners tripping
  on unrelated untracked AppleDouble files, and Windows path/regex mismatches in
  UI governance. No budgets or assertions were weakened to hide them. The full
  repository test suite was not requested or run.
- Full-app startup attempts also hit unrelated recovery/runtime stalls (including
  a Windows access violation); final arrival coverage uses the established
  real-screen harness with real persistence rather than claiming a clean full-app
  startup/live-server walk.
- The Backlog CLI is unavailable. Status stays **In Progress** until formal CLI
  closeout and repository DoD/integration checks are satisfied.

Execution evidence and rulings: [verification record](../../Docs/Development/evidence/2026-10-28-console-trace-copy-verification.md).

### Pre-PR review follow-up

At the user's request, rebased cleanly onto `origin/dev` at
`3efe8476cebf70eceafa605d1abed76a4aad4e0e` and completed a fresh read-only review.
Five confirmed P2 issues were fixed with regression coverage:

- Any later in-app clipboard write invalidates pending trace publication,
  including an identical-text copy.
- Full-message selection preserves source Markdown without row chrome in
  plain rendering; ordinary drag selection keeps its original text domain.
- Post-save navigation and failed-handoff receipts require foreground Console
  ownership as well as the captured chat ID. Queued receipts recheck both at
  execution, covering context changes after dispatch.
- Deferred Library loading requires its complete captured route and Notes rail.
- Response More actions remain visible/reachable through scrolling and resize
  cycles in a short terminal.

Additional core seam: the App owns the shared clipboard revision (moved to its
existing `TextualAppGuards` aggregate during CI closeout below). The guide's
remaining obsolete note-cap wording was corrected. The real-screen handoff
test qualifies the real saved snapshot through Notes before Library's UI load
deadline, isolating exact-note projection from cold-read latency; no production
deadline changed. See the verification record for the limits and final checks.
Existing ADRs remain sufficient. Formal CLI closeout stays pending; the user
authorized publishing this feature branch as a PR against `dev`.

Current-base verification: **175 focused checks passed**, followed by **25/25
Notes tests** after the final queued-receipt guard; **7 controller-boundary
checks passed**, with two documented baseline/environment failures excluded.
Scoped Ruff reports zero new findings/changed-range drift across 22 Python
files (188 inherited diagnostics). CSS reproduction, all active screen
ratchets, task-ID uniqueness across 5908 files and diff whitespace checks pass.
No full test sweep was requested or run.

### PR #3291 UAT follow-up

Checked CI and all review/comment pages at the user's request. No review threads
were present; Qodo is billing-blocked and CodeRabbit skipped the `dev` target.
An independent read-only source review found no new actionable issues. The
current-head Qodo requirement remains subject to an actual review or explicit
user exception before arming the enabled merge queue.

Native `WindowsDriver` journeys exercised the menus and a production Console
store/controllers/Notes SQLite path. The production journey copied and exactly
pasted a 150927-character trace, including both full 69038-character tool results,
and saved a 12449-character body while preserving the draft. Both note action
paths and exact Library editor arrival were checked with the existing qualified
Notes load; live-provider capture and OS clipboard acceptance remain unclaimed.

Visual UAT found a one-row More button's heavy focus outline replacing its label.
Removed that local override so the global canonical, non-obscuring Button focus
style applies, and keep long captions on one line with ellipsis so their
identifying prefix remains visible. Added a compositor-painted label assertion
to the short-terminal regression and regenerated CSS. This implements AC#9 and ADR-150's existing focus
contract; no new ADR, token, dependency or runtime boundary is needed.

Follow-up verification: **79/79 affected copy/selection-menu checks passed** in
90.95s after the painted-label regression first failed. The corrected More-menu
journey also passed under native `WindowsDriver` (5.77s); the independent real
Console/Notes copy, paste, save and exact-ID Library journey passed in 70.55s.
Both changed Python files pass Ruff and formatting; CSS reproduction and diff
whitespace checks pass. Existing full-suite/platform limitations remain recorded.

### Required CI size-budget fix

The completed original-head PR Fast Lane had one failure and 3256 passes:
`app.py` exceeded its 5712-line budget by the 14 new clipboard member lines.
Moved those unchanged members to the already-inherited `TextualAppGuards`
aggregate in `Utils/text_selection_crash_guard.py`, the existing home for
cross-cutting App guards without new boot imports or App bases. `app.py` is
again exactly 5712 lines; no budget was raised. Both member ASTs are unchanged,
and the independent follow-up review confirmed the MRO still delegates exactly
once to the existing clipboard transport. No new ADR is required for this move.

The two targeted App budget/slack checks pass. Real-App ordinary-copy race and
existing selection/Windows-driver guard cases also pass in the broader run.
That run returned 153 passed, 9 failed, 1 skipped: seven over-budget modules are
unchanged from `dev`; `console_transcript.py` was already 299 lines over its pin
on `dev` and contains 169 additional approved feature lines; the whole-tree
scanner still encounters unrelated invalid-UTF-8 AppleDouble files. Those
broader baseline debts remain recorded, not described as a green module sweep.
No size pins were changed in this feature.

The user explicitly approved substituting independent review/UAT for Qodo on
PR #3291 only. Auto-merge was disarmed and the branch synchronized before this
CI fix, as required by the enabled merge queue workflow; re-arm after publication.
