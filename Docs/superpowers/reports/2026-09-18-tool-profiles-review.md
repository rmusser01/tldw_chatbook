# Tool Profiles component review

Review began at `2671067a5e` under the component migration workstream.
ADR-107 owns profile identity, publication and first-bind semantics.

## Refresh and initialization — TASK-32778

Two mounted regressions reproduced before repair:

- Hold a real row button press, refresh its row to a different profile or
  revision, then release delivery. Export, Edit, Bind and Remove all emitted
  the replacement row's authority. Actions now resolve by the originating
  control object; detached/replaced controls cannot acquire new authority.
- Hold an app-owned Textual initialization worker and refresh or remove
  Settings. Cancelling the Settings observer cancelled shared initialization.
  The observer now shields the shared wait and absorbs its eventual failure
  even if Settings has departed. Reopened Settings loads the completed service.

The regression fixture initially waited for every app worker before navigation,
including the deliberately held worker. It now settles mount work before
starting that controlled dependency. That interrupted fixture run does not
qualify cancellation behavior. The corrected pre-fix run failed all ten cases;
the first post-fix run passed all ten.

Final targeted verification passed 78 distinct cases: twelve new lifecycle
cases (including threaded composition and a second app observer), ten cold
workspace provisioning cases, 26 token/component governance cases, and all
30 existing Tool Profiles cases. Fifteen existing cases needed the repository's
private-profile process wrapper; two also needed their offscreen click targets
scrolled into view. Workflow assertions and publication authority are unchanged.
No new Ruff diagnostics were introduced; the existing Settings/panel/test
baselines remain 114/1/3. The new tests and changed small files pass formatting,
the changed Settings method passes isolated formatting, and backlog/diff guards
pass. Independent review found no introduced blocker.

This is mounted lifecycle evidence, not a new native visual qualification.
No visual values changed. The remaining management review below still needs
its own complete native journeys.

## Export recovery — TASK-32779

Existing regular files previously reached the no-overwrite publisher and
produced misleading unsupported-platform copy. Settings now retains the
reviewed snapshot and reopens filename selection with local copy for existing,
invalid or changed destinations. Publication authority remains unchanged.

Real-service mounted evidence first exposed an earlier crash: the review read
`payload.rules`, but the actual payload has `tools`. The old fake payload had
invented the same wrong attribute. The review now uses `tools`; its existing
policy-count assertions run against a real typed payload.

86 distinct targeted cases and four native dark/light × compact/wide journeys
qualify the repair. Native journeys verify the archive manifest, unchanged
incumbent and permission-store bytes, correction and cancellation. The three
terminal-outcome worker cases inject unsupported/failed/uncertain results;
real service/publication tests qualify the underlying boundaries. Capture
checks initial policy authority; publication uses the immutable reviewed
snapshot without recapturing current policy. [Evidence and visual gallery](../qa/2026-09-18-tool-profile-export/README.md).

## Review lifetime — TASK-32780

Import inspection and export capture could publish delayed review dialogs
over a newer category or Settings visit. Eight mounted cases reproduced the
problem; screen removal already suppressed late modals. A local intent now
tracks the initiating visit, exempts only its owned modal, and is rechecked
after preparation and modal boundaries before prompting or admitting a write.
Category changes, unrelated suspension and newer workflows invalidate it.

Independent review caught an initial overbroad error guard: it also hid errors
from already-admitted writes after navigation. Admission is now tracked
separately, preserving failure/uncertainty receipts alongside success while
obsolete preparation errors remain silent. The identical mounted publication
probe now retains its uncertainty receipt after a category roundtrip.

74 distinct targeted cases qualify navigation, same/cross-type replacement,
owned import revise/accept/cancel, destination capture, admitted outcomes and
existing workflow/loading behavior. [Evidence](../qa/2026-09-18-tool-profile-review-lifetime/README.md).
This is mounted production-CSS evidence, not new native visual qualification.

## Focus and native management journeys — TASK-32781

Unchanged listings now retain controls, while changed/reordered listings restore
the same profile/action after serialized recomposition. Newer focus wins; an
unavailable action falls back to another enabled action on the profile or Import.
Overlapping renders no longer cancel a teardown. Exact originating-control
identity still governs queued actions.

At 80×24, full-paint assertions exposed Remove clipping (10 columns allocated,
8 visible). The compact row now uses a token-backed two-column grid. Wide rows
retain their existing layout. 112 final-source targeted checks pass, with no
introduced Ruff diagnostics. Twelve prior action/loading lifecycle cases also
passed before that compact-only CSS adjustment. Independent reviews found no
introduced blocker.

Four real native import/removal journeys cover compact/wide dark/light, revise,
cancel, actual unbound import, removal cancellation, listing reorder and actual
removal with a permanent tombstone. All 24 captures were rendered and inspected;
focus and default/private-profile lifecycle assertions passed. [Evidence and
gallery](../qa/2026-09-18-tool-profile-focus/README.md).

## Removal outcomes — TASK-32782

Mounted checks reproduced definite-failure wording for an uncertain outcome,
stale row facts after errors, and offscreen feedback. Removal now gives truthful
recovery text, refreshes current facts after every terminal attempt, and keeps
plain-text feedback beside the profile or Import continuation. Guarded reveal
tracks focus and text reflow without taking over newer navigation. Mutation
authority and automatic-retry behavior are unchanged (no automatic retry).

167 targeted cases pass, including 18 new mounted outcome cases, prior lifecycle
regressions, governance and 25 real-service removal boundary cases. Independent
review found no introduced blocker, including an overlapping render/outcome
probe. A fixture comparison initially sampled an ongoing focus animation;
settling it repaired the test without changing production behavior.

Four real native size/theme journeys hold an actual profile lease, verify a
refusal with unchanged policy bytes, release it, reopen the category, and
explicitly confirm removal. All twelve captures show readable feedback and
keyboard continuation. Exact-source/private-profile lifecycle checks pass.
[Evidence and gallery](../qa/2026-09-18-tool-profile-removal/README.md).

## Bind handoff — TASK-32783

Bind no longer discards a same-workspace Persona/memory draft. It stages only the
selected profile, then reveals Persona or Apply with adjacent guidance after
category panes settle. Default receives a visible recovery continuation. Newer
focus/navigation wins; Bind never saves defaults automatically.

91 distinct targeted cases pass. Four final native cells and sixteen rendered,
inspected captures verify default recovery, staged Persona, roundtrip draft
retention, first-bind cancel and exact confirmed persistence. Independent review
caught queued focus masking a delayed valid callback; synchronous focus and a
precise regression fix it. Earlier fixture/source attempts are separated from
final evidence. [QA and gallery](../qa/2026-09-18-tool-profile-bind/README.md).

## Compact CSS indexing — TASK-32784

The PR structural guard reproduced 275 ancestor-scoped bare-type rules versus
the unchanged limit 274. A dedicated action class restores 274 at identical
specificity and declarations, without applying the compact rule to Import.
80 targeted checks pass; four mounted theme/size comparisons have identical
button styles, geometry, visibility and painted text. Independent review found
no issue. [Evidence](../qa/2026-09-18-tool-profile-css-fastpath/README.md).

## Remaining scope

The MCP Edit handoff selects the exact profile, including return visits, but
real routed probes show a geometry gap: focused permission rows are below the
canvas clip at 80×24, and Ctrl+End reaches a clipped row at 140×40. Standalone
profile selection tests pass but do not cover this routed layout. Repair and
native qualification remain open. Concurrent-workflow review also remains open.

The compact-selector CI failure is repaired locally by TASK-32784; remote
checks still need to run on the new head.
These bounded management reviews do not complete the broader component workstream.
