# PR3346: landing snapshot publication repair

Diagnostic published head: `f70eb635f5991d25b7cd5d32d789406d837be32b`.
Verified dev base: `2dcf038cb8be0af991f92c43b0c262f02471c51d`.
Current corrected bytes are identified by the enclosing source/evidence manifests.
All older captures retain their original head and scope qualification.

## Demonstrated defect and narrow repair

Hosted UI19 failed the unchanged healthy Library source-DOM wait before the
note-link journey started: rendered generation0, source generation1, dirty
reconciliation, loaded sources, no source/canvas error and current Library screen.
The same previously unexplained symptom was retained in f095 evidence. The exact
ordinary local test passed in30.81s; this alone did not establish a repair.

The new hosted traceback also records onboarding's retained landing sync querying
`#library-canvas-landing` while native recomposition has removed the children.
One bounded native system-event probe drove that actual empty-child mount boundary
with the original current `_finish_library_entry_canvas_sync` queued. Calling
the original onboarding sync once raised NoMatches through the original shared
failure path and synchronously cleared that same callback. The native remount
continued, but the unchanged original readiness wait failed at generation0/1,
dirtytrue. This proves callback loss for the explicit event schedule. The hosted
trace corroborates the missing-child failure; it does not prove its exact timing
or every triggering precondition.

`LibraryLandingCanvas.sync_state` still adopts the latest state and advances its
serial first. It now takes its existing recompose-return branch when native
recomposition is already requested or its lock is active, as well as when the
widget set changes. It therefore publishes through the original owner handoff
instead of querying absent or obsolete children. The pending-before-start case
matters: stored state can already name a new widget set while the DOM still has
the previous children. No new polling, fallback, suppression or callback slot.
Generic failure cleanup, focus supersession and route/generation fences are unchanged.

The production change is the accurate docstring, one ownership comment and the
initial condition only: existing widget module566→571 lines. No existing size,
performance, timeout, allowlist or assertion was relaxed. Textual is exactly8.2.8
in pyproject and the tested environment; its native recompose holds the public
RLock across removal/mount, and its `is_locked` property reports that ownership.

[Repair report](repair/implementation-report.md) ·
[Independent source/evidence review](review/candidate-review.md) ·
[Final root gates review](review/root-gates-addendum.md)

## Verification on frozen final bytes

Separate overlapping cohorts are not added together.

| Cohort | Result and qualification |
| --- | --- |
| Permanent pending/active native controls on original source |2 expected failures in2.79s at absent count/purpose children |
| Final ordinary native run |3 passed in33.63s: both new controls and the unchanged full note-link keyboard/autosave/backlink journey |
| Full original snapshot-owner replay on corrected source |1 passed in34.35s; same probe bytes as RED, exact current finish callback survives before/after sync, and original healthy-generation/link assertions pass |
| Existing leaf controls |3 passed in the initial five-case cohort with the new controls; that cohort preceded only the docstring wording change |
| Existing full-screen ownership controls |8 passed in22.87s with a collection-only private bootstrap-profile marker on those exact inherited cases; original bodies/authority unchanged |
| Final static/design/size checks |209 passed in31.33s |
| Final CI/workflow/census checks |254 passed in48.46s |
| Final preflights |Worker, markup537, diagnostic728 owners/16 sinks, and UI census1337 all pass unchanged apart from new census enrollment |
| Scoped Ruff/format |Widget2→2 inherited findings/new test0; metadata2→2 inherited EXE001/PIE810; zero additions and new/final blocks formatted |
| Five unchanged startup guards |5 passed,3 existing headroom warnings in42.99s |

The new leaf controls reuse the existing native landing harness and actual
mount/recompose lifecycle. They assert the latest literal purpose/count/action
and exactly-once completion in both windows. The actual snapshot probe separately
qualifies the screen-level generation chain; its test wrapper is explicit
fault-scheduled observation, not an uninstrumented hosted reproduction.

The inherited full-screen cases originally refused source-profile setup before
app construction; those failures are preserved. Their qualified rerun covers
retained controls, stale route, newer-generation publication, queued navigation,
detachment, same-generation/new-route retry and strict failure/retry. They are
not called ordinary-CI passes. The final ordinary three-case run uses normal
module-owned bootstrap and no private plugin. Original UI19 wait/deadline,
helpers, source-admission and generic callback files remain byte-identical.

Startup: UI-ready **1031/1031** modules; preimport **554/558** modules and
**426171/426199 LOC** (**28 lines headroom**); largest Library route
**132480/135111 LOC**; warmup **93/94**. Original CSS≤268 and worker/thread
allowlists pass. Original observer/guard bytes and all **3453** captured runtime,
bootstrap/config, Performance and modified-native-test paths match before/after.
These are the unchanged count/payload guards, not a new latency measurement.

The UI census preserves its exact1336-entry prefix and appends only
`Tests/UI/test_library_landing_recompose_publication.py`, covering both phases.
Floor1337. No overlap, replaced slot, lane, dependency or aggregate change.
Existing ADR103 and Library retained-canvas ownership apply; no new ADR.

## Evidence and remaining publication work

`artifact-map.json` retains **75 explicit source identities**, with producer
hash checks, collision checks, reversible whitespace-sensitive text wrappers and
deterministic gzip. Scratch profiles/caches/databases are excluded by the explicit
inventories. Credential-shaped text precheck found no new redaction requirement;
previous phase redactions remain untouched. Early profile/path setup failures
and preliminary observations remain qualified as such.

All local source/evidence and root-gate review findings are closed. Auto-merge was
held when UI19 failed; the corrected published head still needs fresh protected
CI before it is re-armed. Qodo remains bypassed under the owner's explicit
instruction. No new hosted success or completed merge is claimed here.
No full repository sweep, cloud-provider/cache-billing run or new PTY/font tour.
