# Full-implementation review corrections

Scope: the three confirmed findings after compositing implementation `c2333548d8`.
Changes are inline and sequential. Existing ADR-243 applies; no new architecture,
provider, dependency, storage format or speech loop is introduced.

## 1. Preserve manual authoring when reference preparation is unavailable

Commit `41e0e33c53` makes reference/region preparation optional. A valid saved source
still produces an ordinary draft and reopens existing frames when its generation
reference exceeds decode/PNG limits. Only region/generation controls are disabled;
direct calls refuse with an explanation. No memory limit is relaxed and source bytes,
manual validation and publication/removal CAS are unchanged.

A mounted 4096×4096 regression was red at Prepare before the fix. It now exercises
existing applied frames, replacement import, Apply, reopen and Remove with the exact
source preserved. Reviewed all reference callers, control reset and provenance paths.
The first affected authoring gate passed 62 tests; subsequent compatibility coverage
includes this regression.

## 2. Retire dialogs after persisted source/settings changes

Commit `90f9f51fba` extends the existing context poll only while a pair or region
dialog is open. One coalesced native operation verifies saved source and region
revision. A mismatch retires/clears the exact stale dialogs and draft, including
beneath a covering modal; active DB frames remain unchanged. A read error closes
review while retaining draft bytes. Own Save, revision, epoch and dialog identity
fence late work. Close drains the native validation before releasing its DB owner.

Five mounted tests were red for source/region edits through a second DB connection.
They now verify both dialog types, actual cleared content, deferred close beneath a
cover and unchanged active artwork. Held native work covers coalescing, replacement
isolation, close draining and failed-read retention. Another red test caught a queued
coroutine adopting already-discarded state; snapshots are now captured before
scheduling and checked before native work. The final region/generation/pair-viewer
gate passed 68 tests; the final copy-independent queued oracle also passed alone.

Reviewed every poll/close caller, source/settings race, unknown-recipe reselection
(the poll reads revision without deserializing unsupported recipes), own Save and
native cleanup. No unresolved implementation finding remains in this correction.

## 3. Preserve uncertainty evidence when live qualification is interrupted

The probe now retains safe artifacts and writes its report before propagating
KeyboardInterrupt/SystemExit after private temporary cleanup. Malformed interrupted
manifests are retained with an unknown (`null`) request count. It never retries and
never copies private transport logs/configuration into evidence.

Four new cases were red for missing reports before the fix. All ten probe controls
now pass, covering both interruption classes, complete/truncated manifests, identical
exception propagation, one attempt only, retained artifacts and removed temporary
credentials. Existing success/failure, no-overwrite and no-transport inventory controls
also pass. Reviewed the full private-config/subprocess/cleanup path; credentials stay
out of subprocess arguments and retained evidence. No live transport was invoked.

## Final affected verification

- Manual authoring, Console performance/radio and recovery ownership: **108 passed
  in 158.01s**.
- Region/generation/pair review, including all new lifetime controls: **68 passed
  in 198.94s**.
- Probe: **10 passed in 0.41s**; CSS/token guards: **13 passed in 62.26s**.
- All ten revised Python files pass Ruff/format; the diff passes whitespace checks.
- Fresh installed authoring/Console and full native F9 restore/edit journeys: **2 passed
  in 91.15s**, using newly built wheels. All 18 changed production files and v83 SQL
  match the installed copies; evidence is under `qa/portrait-compositing/2026-10-24/corrections/`.

An initial native child timed out. Isolated opening recovered all expected data and
completed the component edit; fixture review found that selection awaited only its
scheduling wrapper. The installed/restore fixtures now await the actual selected
character and visual-identity workers, then rendered confirmation focus. Final full
journeys pass with the original timeouts, real controls and unchanged admission.

**Final verdict:** all three findings are fixed and regression-tested. The second
review found and resolved the queued-cancellation race described above; no further
implementation defect was found. TASK-36553/36554 are Done after their corrective
review checkpoints; TASK-36555 stays In Progress only for its existing live-adapter AC5.

The combined UI/ProductionApp collection encountered `raw_source_selection_changed`
before tests ran. The implicated App/config/recovery/conftest files are unchanged;
the established isolated groups pass without bypassing admission or changing guards.

TASK-36555's real configured image-adapter AC5 remains open: the current default/only
enabled backend is SwarmUI, unsupported by this reference path. Fresh inventory records
zero requests and no fallback. No full suite, new live provider/hardware qualification,
push or merge.
