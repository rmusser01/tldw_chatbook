---
id: TASK-24704
title: Close the review findings raised against the Inspect rail burn-down
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30'
updated_date: '2026-08-30'
labels:
  - console
  - inspector
  - review
dependencies:
  - task-24700
  - task-24701
  - task-24702
  - task-24703
priority: high
---

## Description (the why)

The Inspect rail burn-down (TASK-24600–24612) and its second-round follow-ups
(TASK-24700–24703) shipped as PR #2220. Automated review of that PR raised
seventeen findings across four rounds, several of them real defects in the
burn-down's own fixes rather than in the code it changed.

This task is the close-out for those findings. It exists so the `TASK-24704`
markers left in the code have something to resolve to: without it, a reader
who follows one of those comments into the backlog finds nothing, and the
reasoning behind some non-obvious code (why a focus guard reads an event
instead of a flag, why one readiness row is skipped and the rest are not) is
recoverable only from a merged PR thread.

## Acceptance Criteria (the what)

- [x] Every finding raised against PR #2220 is either fixed or answered with
      evidence for why it is not a defect
- [x] Each fix has a test that was observed FAILING without it, not merely
      passing with it
- [x] Findings that turn out to be already-correct behaviour are confirmed by
      a test rather than by reading the code
- [x] No finding is closed by widening a guard so far that it hides the
      behaviour the guard was added to protect
- [x] Derived artifacts reproduce (`./scripts/preflight.sh` green)
- [x] The branch merges onto current dev with all required checks passing

## Implementation Plan (the how)

1. Take each review round in order; classify every finding as defect, rule
   violation, or already-correct before touching code
2. For each defect, write the failing test first, then fix
3. For each "confirm this is handled" finding, find the mechanism and pin it
   with a test if none exists
4. Re-request review after each round; do not assume a round is the last
5. Rebase onto dev between rounds, verifying any new red is dev's and not the
   branch's, against a pristine checkout at the current dev head

## Implementation Notes

Seventeen findings over four review rounds; all resolved. Three were real
defects, four were rule violations, and the rest confirmed already-correct
behaviour that had no test.

**The three real defects**, all in this burn-down's own fixes:

- *Boundary anchor went stale.* `n`/`p` parks focus on the outer scroller for
  an all-`Static` section and remembers the index so the next press continues.
  That memory was cleared only inside `_focus_boundary`, so Tab away and back
  left it set and the next `n`/`p` resumed from stale history. Fixed by
  clearing it from `on_descendant_focus` whenever focus lands anywhere other
  than the scroller. **Driving it from the focus EVENT is load-bearing**: an
  earlier attempt set a flag around `target.focus()`, and Textual delivers
  `DescendantFocus` asynchronously, so the flag was already reset by the time
  the handler read it — the guard silently did nothing and its test passed.
- *ACP snapshot could take down the sync tick.* `_sync_console_live_work_
  readiness_rows` read `acp_runtime_process_manager.snapshot()` unguarded. The
  card BUILDER makes the identical read and is deliberately still unguarded —
  it runs once, at compose. This call site is on the console sync tick, which
  re-raises on a live screen inside a worker whose `exit_on_error` is Textual's
  default `True`. One transient raise was the app exiting, five times a second
  during an active run. **The first fix was wrong in the other direction**: it
  returned early, which also froze the independently-probed MCP and RAG rows,
  so a persistent ACP failure would freeze them permanently. Only the ACP row
  is skipped now (via `ACP_READINESS_ROW_ID`), keeping its last known text
  rather than an unmeasured `not_configured` (TASK-24601's contract).
- *`Alt+I` reached the footer but not F1.* TASK-24604 added it to
  `CONSOLE_WORKBENCH_SHORTCUTS` and stopped; `action_show_workbench_help`
  renders `CONSOLE_WORKBENCH_SHORTCUT_GROUPS`, a separate constant. That is
  the wrong half to land — the footer degrades by keeping a PREFIX of its
  hints, so it drops entries at exactly the narrow widths where the rail's
  edge handle is hidden and `Alt+I` is the only route in. F1 never truncates.

**Two findings were answered rather than fixed.** The pinned FAILED authority
status is cleared on every path that matters — a new send's first act is
`_set_run_state(VALIDATING)`, cancellation lands on `STOPPED`, and session
switch is per-session — but only the FAILED→pinned direction had a test, so
the recovery direction got one.

**Trade-off.** `ConsoleInspectorState.from_values` gained a full 25-parameter
`Args:` block for two parameters this arc added. Documenting only the new two
would have left a partial block, which reads worse than none; the sibling
factory in the same module documents all of its parameters, so this matches.

Modified: `tldw_chatbook/UI/Screens/chat_screen.py`,
`UI/Console_Modules/right_rail.py`, `Chat/console_live_work.py`,
`Chat/console_display_state.py`, `Widgets/Console/console_run_inspector.py`,
`Tests/UI/test_console_{right_rail,run_inspector,inspector_navigation,
inspector_keyboard_route,inspector_focus_visibility}.py`,
`Docs/security/production-diagnostic-inventory.json`.

Merged as `c2f64f690b` (PR #2220).

## Lessons

Two worth keeping, both already borne out more than once in this arc:

- **A guard that depends on a flag set around `focus()` does not work in
  Textual.** `DescendantFocus` is delivered asynchronously; the flag is reset
  before the handler runs. Decide synchronously at the call site, or read the
  event. This cost two attempts.
- **After fixing a paired surface, ask which half was not touched.** Three
  separate findings in this arc were half-landed fixes — a widget pair, a
  footer/help pair, a builder/tick pair. The half that looks safe to skip is
  usually the one that degrades under the exact conditions the fix was for.
