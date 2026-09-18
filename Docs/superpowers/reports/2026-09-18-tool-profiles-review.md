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

## MCP Edit handoff — TASK-32785

The real Settings route now reveals its permission table after content layout,
including restored MCP visits. The canvas permits keyboard reading of controls
and policy feedback; Space mutates only while the table owns focus. Exact
profile/revision authority and newer focus ownership remain intact.

129 distinct targeted cases pass. Four final native size/theme cells and 20
inspected captures verify real imports, first/last rows, horizontal State-column
access, saved selected-profile policy and fresh-revision return. Private lifecycle
checks pass. Older mode tests required the established process-isolation wrapper
and one stale CSS assertion required its already-existing token. Independent
review found no introduced blocker. [QA and gallery](../qa/2026-09-18-mcp-permission-handoff/README.md).

## Admitted write overlap — TASK-32786

Repeating Import, Export or Remove previously cancelled the observing UI worker
while its admitted thread could complete a write. Same-operation dispatch now
shows local progress until the write returns. Newer preparation, other reviews
and explicit worker cancellation preserve their existing boundaries. Import
uncertainty refreshes current facts and receives truthful recovery wording.

173 targeted cases pass, including fifteen overlap cases and 52 real-service
publication/removal checks. Independent review found no introduced blocker.
Four real native size/theme cells held the existing service mutation lock,
repeated Remove, then verified one revision increment, the tombstone and visible
continuation. All twelve captures were rendered and inspected; exact-source and
private lifecycle checks pass. [QA and gallery](../qa/2026-09-18-tool-profile-write-overlap/README.md).

## Write lifetime — TASK-32787

Leaving Settings could cancel its observer while the admitted write committed;
the replacement screen then retained stale profile facts and no result. The
application now owns admitted Import, Export and Remove tasks, limits duplicate
same-kind writes across visits, and exposes bounded pending/terminal receipts.
Settings observation cancels promptly and refreshes facts on completion without
taking newer focus. Normal shutdown closes admission and drains these tasks
before resource teardown, with the existing process-owned watchdog armed first.
ADR-167 records the lifecycle boundary; exact service review authority is unchanged.

229 distinct targeted cases pass. Independent review found watchdog ordering,
malformed nested-result and callable-cancellation gaps; their regression cases
now pass. Four final native recreation cells and a fifth pending-write shutdown
journey use real services. All seventeen captures were rendered and inspected;
exact-source and private lifecycle checks pass. Earlier fixture failures are
retained and explained. [QA and gallery](../qa/2026-09-18-tool-profile-write-lifetime/README.md).

## Remaining scope

In-visit overlap and screen-destruction/app-shutdown outcome ownership are
qualified above within their recorded bounds. The Edit repair does not close
the whole MCP destination: compact rail/header wrapping and simultaneous
tool/state readability remain part of its broader layout review. State remains
accessible with existing horizontal keyboard scroll, now qualified natively.

New PR heads require their own remote checks. These bounded management reviews
do not complete the broader component workstream.
