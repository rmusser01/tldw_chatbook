# W2 performance/current-source fix report

Scope: shared working tree on HEAD022 before root's next commit/rebase. Read together with `perf-022-diagnosis.md` for pinned 3412/88af measurements. No budget constant, pinned census, source-evidence manifest, branch, index, commit or remote was changed by this agent. Root owns the subsequent commit/rebase and final evidence qualification.

## Outcome

All five exact original performance guards passed together in `/private/tmp/mixins-perf-qualified.log` (5 passed, 34.75s):

| Guard | Actual | Existing limit |
|---|---:|---:|
| Native warm UI-ready modules | 1031 | 1031 |
| Screen preimport added modules | 552 | 558 |
| Screen preimport added LOC | 425692 | 426199 |
| First-send warmup added modules | 92 | 94 |
| Ancestor-scoped bare-type CSS rules | 268 | 268 |
| Boot worker/thread inventory | Exact reviewed owners only | Unchanged authority/assertions |

After this run, a type-only import ordering cleanup removed one source line in LibraryScreen; current LOC is consequently one lower (425691), not a fresh measured number. Native bridge residency is now explicitly asserted in the original UI-ready census. After rebasing onto 88af, remove the newly upstream-added redundant explicit bridge warmup entry only against that native residency assertion; this branch did not yet contain that entry. Upstream first-send warmup changes and subsequent extractions require fresh measurements. There is zero module/CSS headroom.

## Changes and causal evidence

- Defer the complete Console Prompts modal, improvement view and legacy prompt picker to their actual open/apply methods. Preserve the existing Console public modal export with lazy `__getattr__`, all result/guard class identities, callbacks and source fences. No modal or view is mounted at UI-ready.
- Defer the Library delete confirmation import and type-only decision annotation. Defer the Library Note folder-name dialog to New/Rename, matching the existing first-use target-chooser imports. The latter was actually in the measured route closure and removed 527 LOC; no tests patch the former screen-global dialog import. An attempted Markdown export import deferral was correctly identified as outside the measured route closure and reverted in full.
- The SSH run-end registry still creates/tombstones its owner and drains its exact workers. Worker/start-failure/transport types are imported at acquire; public imported type identities remain available via lazy exports. Remove `remote_session_worker` from the ordinary local first-send warmup: a local run's retirement no longer imports SSH transport. Existing session failure/restart/concurrency/shutdown controls pass.
- Empty transcript projection returns no recovered-image specifications without launching a catalog read. It does not alter cached selection/rows, cancel a previous native reader, change nonempty loading, or relax any source/read/cancellation fence. New no-pending-reader case was seen red before the change; pending-reader retention and original real nonempty current/detached shutdown controls pass.
- Boot census now names two required existing owners: `(_reconcile_console_after_attach, console-attach-reconciliation)` and `(refresh, console-emergency-stop)`. The first completes arrival reconciliation before send admission. The second is scheduled at arrival, waits for UI-ready, and reads persisted emergency-stop state. No read was suppressed or delayed past its authority requirement. A timing-dependent recovered-image worker was removed by eliminating its unnecessary empty read, not by adding it to the inventory.
- CSS census now requires the exact eight mounted recovery-control IDs. This reaches the real CSS rule assertion (previous fixture stopped at seven); pinned dev88af itself then measures 273 rules versus 268. The feature adds four, producing 277. Nine exact class keys repay all nine, preserving IDs, original ancestor scopes, declarations and callbacks. Generated CSS was rebuilt by the original builder; no bundle was hand-edited.

### Nine selector changes

1. `ConsolePromptMixinsView Static` → `ConsolePromptMixinsView Static.console-mixins-copy`
2. `.console-mixins-actions Button` → `.console-mixins-actions Button.console-mixins-action`
3. `.console-mixin-row Checkbox` → `.console-mixin-row Checkbox.console-mixin-toggle`
4. `ConsolePromptsModal.-narrow .console-mixins-actions Button` → same ancestors plus `Button.console-mixins-action`
5. `#global-full-actions Button` → same ancestor plus `Button.global-full-action`
6. `#capture-policy-actions Button` → same ancestor plus `Button.capture-policy-action`
7. `#trace-privacy-actions Button` → same ancestor plus `Button.trace-privacy-action`
8. `#exchange-export-actions Button` → same ancestor plus `Button.exchange-export-action`
9. `#trace-export-actions Button` → same ancestor plus `Button.trace-export-action`

Classes are assigned at the known widget constructors, including each recompose. There is no dynamic tagging of unknown subtrees.

## Natural source-currentness failure

`test_preparing_changed_source_after_original_tab_await_replays_full[natural]` failed before making its intended source mutation. An unchanged original image recovery worker won the held surface lock, so the timer correctly declined overlapping FULL. Two observation-only exact reproductions captured the actual stack: `image._load_recovered_images` → original full sync, with the timer repeatedly returning False while `_console_sync_in_progress` remained true. The test now joins only the exact recovered-image workers issued by the first completed display via `worker.wait()`, within the original deadline (failure propagates), then qualifies a second original natural timer display before taking the lock. All original configuration custody/source mutation/post-await rejection/replay assertions remain.

Probe: `/private/tmp/mixins-natural-probe.{py,json,log}`. No callback was replaced in production; the test-local monitoring observer was augmented only to collect the stack. Natural and queued-full controls both pass after correction and again after the empty-image optimization.

## Targeted verification

- `/private/tmp/mixins-perf-regressions.log`: 53 passed, one Library test failed in setup with existing `raw_source_selection_changed` due its unmarked live-config fixture. The passing set includes the full SSH registry file, first-send inventory and cycle files, original current/detached real recovered-image shutdown, both new empty cases, both natural/queued source cases, Prompt open/system-stale controls and legacy picker control.
- `/private/tmp/mixins-library-perf-controls.log`: exact Library delete/cancel and Markdown-copy controls, with the existing `bootstrap_profile` contract applied by a private collection plugin: 2 passed. Production admission was not bypassed and no repository marker inventory was changed.
- `/private/tmp/mixins-perf-folder-controls.log`: 33 dialog tests passed, original New-folder route failed the same unmarked setup selection. `/private/tmp/mixins-folder-route-profile.log`: exact New-folder route with bootstrap profile, 1 passed. These are explicit setup failures/reruns, not an all-green aggregate.
- `/private/tmp/mixins-css-styles-{before,after}.log`: same 12 mounted controls each passed, covering capture/privacy/global-Full confirmation, exchange export/escape, trace export supported widths and actual Console Prompts action at 160×45, 100×30, 80×24.
- Observation-only Pilot snapshots: 444 matching state keys have identical outer/content geometry and display; 442 also have identical full computed style values after excluding node repr/classes. Two click-state samples differ only by the transient standard Button active tint; the probe did not synchronize that animation state, and no changed rule has a tint declaration. No whole visual-equivalence claim is made for those two transient samples. See `/private/tmp/mixins-css-style-comparison.json` and raw before/after JSON.
- `/private/tmp/mixins-perf-lint.json`: 18 owned/shared touched Python files, HEAD/current default-rule diagnostic multiset 275/275, zero added; all parse; `git diff --check` clean. Existing whole-file lint/format debt retained; only modified compose methods were range-formatted.
- The existing SSH AST test still searched app_lifecycle.on_unmount for a service phase that this feature had already extracted. It now asserts on_unmount awaits `_drain_app_unmount_services(self, ...)`, then checks the exact existing session-before-master order in one best-effort try in the actual root app_shutdown helper. Existing exception/order assertions remain.

## Owned artifacts and stop state

Explicit source/generated ownership: `/private/tmp/mixins-perf-owned-files.txt` (22 paths). Shared prompts/controller/screen files also contain other agents' work; preserve it. Generated Plugins sheets belong to root, not this agent.

Private artifacts: `/private/tmp/mixins-perf-probe.py`; `mixins-perf-{head022,dev3412,dev88af,fixed,final,complete,qualified}*` logs/JSON/profile directories; pinned archives `mixins-perf-baseline3412` and `mixins-perf-dev88af` (the latter's CSS test fixture alone later corrected for baseline rule measurement); `mixins-css-dev-fixed*`; `mixins-natural-probe*`; `mixins-empty-image-red*`; `mixins-perf-owned-targeted*`; `mixins-perf-regressions*`; `mixins-library-perf-controls*`; `mixins-folder-route-profile*`; `mixins-perf-folder-controls*`; `mixins-css-style-probe.py`, `mixins-css-styles-{before,after}*`, `mixins-css-style-comparison.json`; `mixins-classkey*`, `mixins-format*`, `mixins-prompts-imports*`, `mixins-library-imports*`, `mixins-library-screen-imports*`; `mixins-perf-css-build.log`; `mixins-perf-lint*`; `mixins-perf-owned-files.txt`.

All test subprocesses and builder processes owned by this agent have completed; no current process remains. No further product edits are planned before root's commit/rebase. These measurements precede that rebase and future size extractions.
