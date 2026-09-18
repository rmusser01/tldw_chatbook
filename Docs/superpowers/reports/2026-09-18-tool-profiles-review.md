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

## Confirmed remaining defect

- **Unchanged profile refresh loses focused actions.** Import/Edit/Remove
  focus falls to the detail-pane body after recomposition. Cancelling import
  or removal also triggers this through Settings resume. Restore by stable
  profile/action identity while respecting newer navigation and focus.

Read-only boundary probes confirmed that cancelling import review performs no
import, cancelling removal performs no removal, and real removal rejects a
stale revision without changing permission-store bytes. Those bounded probes
do not qualify the remaining native management journeys.

## Remaining scope

Remaining removal boundaries, focus through refresh, and complete native
import/removal compact/wide dark/light journeys remain to be qualified. This ledger does
not claim the Tool Profiles feature review or the broader component workstream
is complete.
