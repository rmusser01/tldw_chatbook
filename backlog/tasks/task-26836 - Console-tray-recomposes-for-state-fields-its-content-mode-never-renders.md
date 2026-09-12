---
id: TASK-26836
title: Console tray recomposes for state fields its content mode never renders
status: Done
assignee:
  - '@claude'
created_date: '2026-09-01 14:51'
updated_date: '2026-09-01 15:04'
labels:
  - console
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26834 fix target 1, with direct probe evidence: a tree click pushed delta=conversation_browser into the workspaces tray (content=workspace, whose compose never reads that field) and it recomposed anyway -- opening one of the 2-3 nested App batches that hold all paints for 250-400ms. _can_skip_recompose condition 5 uses whole-state value equality, so ANY field delta forces recompose regardless of whether the mounted DOM depends on it. The fix records which state fields compose actually read (same pattern as the existing composed-row signature) and skips when only unread fields changed, adopting the new state. The guard's other five conditions and the TASK-251/15454 history are preserved: a changed READ field still recomposes exactly as today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A state push whose changed fields were not read by the tray's last compose does not recompose it, and the new state is adopted
- [x] #2 A changed field that compose DID read still recomposes exactly as before
- [x] #3 The existing recompose-guard suite stays green, with one deliberate revision: test_a_text_only_change_still_recomposes changes `heading` on a show_heading=False tray -- a field that instance provably never renders -- and is updated to use a genuinely rendered text field, with the unread-field skip pinned alongside
- [x] #4 The field-read recording is exercised for all three content modes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: browser-only delta on the workspace-content tray must not recompose\n2. state property with read-recording view during compose; store composed read set\n3. _can_skip_recompose: full-equality fast path, else read-fields-only diff; adopt state on skip\n4. Guard suite + tray suite + sweep; preflight; PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`compose()` now records which top-level state fields it reads, exactly as it
already records the composed row signature: `state` became a property whose
getter, while `_state_field_reads` is live (the same try/finally window that
brackets the row signature), returns a `_ComposeReadView` that logs attribute
names before delegating. Reads observed from interleaved coroutines during a
compose window can only ADD fields, so the bias is toward recomposing more,
never less. The frozen read set is stored alongside the composed signatures.

`_can_skip_recompose` condition 5 relaxes from whole-state equality to
read-fields equality: with no recorded read set it recomposes exactly as
before, and a changed READ field still recomposes. Conditions 1-4 and the
condition-6 DOM signature proof are untouched and still gate the skip. On a
skip, `sync_state` now adopts the new state so an unread delta is not
re-diffed forever.

One compose fix fell out: `browser = self.state.conversation_browser` was
read UNCONDITIONALLY before the content check, which recorded the browser as
read by every content mode -- the exact wasted recompose the probe observed.
The read now happens only for content in {"all", "conversations"}.

One deliberate test revision (AC #3): the guard suite's text-only test
changed `heading` on the show_heading=False conversations tray -- a field
that instance provably never renders -- and now uses the browser's
selected-summary line; the heading skip is pinned in the new suite instead.

Verified: 4 new tests (the probe-observed workspaces-tray skip, the mirror
conversations-tray skip, read-field recompose, fresh-instance healing) + the
TASK-15454 guard suite = 14 passed. Sweep of 7 tray-asserting files incl.
visual snapshots: 110 passed, 3 documented dev reds only. The
workspace-switch persistence red found by the sweep is deterministic on
pristine dev (0/5 both trees) and recorded on TASK-25715.

Files: `tldw_chatbook/Widgets/Console/console_workspace_context.py`,
`Tests/UI/test_console_tray_read_aware_recompose.py` (new),
`Tests/UI/test_console_workspace_tray_recompose_guard.py` (one test revised).
<!-- SECTION:NOTES:END -->
