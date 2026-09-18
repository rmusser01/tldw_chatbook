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

## Export recovery — still open

The current save picker accepts an existing regular file. The publication
owner deliberately rejects replacement with `publication_unsupported`, but
Settings describes that result as missing platform primitives on POSIX.
Preserve the existing file and publication authority. Offer a new filename
with truthful recovery, retaining the already captured export review. Verify
an existing archive on supported POSIX as well as the separate Windows result.
This is source-confirmed and awaits a mounted regression and repair.

## Remaining scope

Import/export/removal cancellation and stale-context journeys, focus through
refresh, production CSS and native compact/wide dark/light presentation remain
to be qualified. This ledger does not claim the Tool Profiles feature review
or the broader component workstream is complete.
