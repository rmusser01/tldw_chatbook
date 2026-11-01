# PR 2995 ownership CI triage

Read-only investigation. No tests, application launches, source changes, Git mutations, or nested agents were performed. The systematic-debugging skill supplied the evidence-first method. Production/test source references below are from immutable head `7a15e3e9b4998dba8d8726bbdb722ff60a3df83d`, not the concurrently edited checkout.

## CI identity and environment

- Log: `pr-initial-fastlane-job.log`. CI checked out merge `112298c452082b71c9a21572e7982f5bfd6bc9c9`, parents base `81c7c94f486491d6fde09584710dc3a3172225a0` and PR head `7a15e3e9b4998dba8d8726bbdb722ff60a3df83d` (log line 198). Root supplied `ci-initial-merge-tree-receipt.json`; merge and head trees both equal `9d2ff375673c6b46af8a85babf0d4960f68a271d`. Thus 7a is an exact source-tree authority for this CI failure.
- Ubuntu 24.04.5, x64 CPython 3.12.14. Editable base install, with no dev extras: `python -m pip install -e . pytest pytest-asyncio pytest-timeout packaging`.
- Relevant installed versions: Textual 8.2.8, pytest 9.1.1, pytest-asyncio 1.4.0, pytest-timeout 2.4.0, pluggy 1.6.0, Rich 15.0.0, platformdirs 4.12.3, anyio 4.15.1. Asyncio AUTO, function test loop scope, debug=False; timeout uses signal and includes fixtures.
- Logged step environment contains Python location/root variables, PKG_CONFIG_PATH and LD_LIBRARY_PATH. No explicit TLDW test profile, CI flag, or parallel runner option appears in that step. This is not proof of absence from the runner's inherited environment.

Exact failing step (log lines 637-647):

```sh
pytest \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/Chat/test_console_viewless_hooks.py \
  Tests/Agents/test_install_skill_runtime_tool.py \
  Tests/Chat/test_console_chat_create_integration.py \
  --timeout=300 \
  --tb=short
```

Collected 147: 4 failed, 142 passed, 1 inherited XFAIL, 5 warnings in 500.51s. All four failures were in the first file; later files had not executed their bodies yet, so their test-body pollution cannot explain the preceding failures. Their collection imports remain a possible difference.

Current local interpreter metadata (read only): Python 3.12.11, pytest 8.4.2, pytest-asyncio 1.2.0, Textual 8.2.8, Rich 14.3.3, anyio 4.12.1, platformdirs 4.9.2. This is the current environment, not a historical package receipt for the original local run.

## Failure observations and production seams

| Node | CI observation | What it establishes |
| --- | --- | --- |
| `test_second_console_visit_reuses_the_runtime` | line 2094, `controller_two.notify_run_outcome is None` | Shared runtime/controller/store/bridge identity and cancellation assertions before it passed; current view hooks were not live after the explicit `_ensure_console_chat_controller` call. A bare reconciliation wait omission cannot alone explain that explicit attach failing. |
| `test_a_superseded_screen_never_detaches_the_successors_runtime` | line 2252, `runtime.view is None`, expected successor ChatScreen | Actual claim loss, not merely delayed painting. The assertion's description blames outgoing detach, but the log does not identify which screen detached or claimed last. |
| `test_opening_console_during_a_headless_delivery_arms_the_poll` | line 2332, timer=None; `reconciled=False`, delivery active, same_controller=True | View reconciliation had not completed. DOM presence and pauses did not establish that the runtime could answer through that view. It does not yet prove the rearm implementation failed after reconciliation. |
| `test_native_acceptance_consumes_only_open_target_revision[unchanged]` | line 3272, 5s timeout waiting for held readiness entry | Failure happened before the barrier used to exercise receipt/composer consumption. No receipt or composer regression was reached. Other five variants passed. |

The native timeout also logs a `durable_commit`/`controller_submit` cancellation with duration 63,502ms (log line 1026), far longer than its nominal readiness wait. The log does not associate this diagnostics attempt with the rig's `start` task, so do not assume it is that submission. The rig's readiness barrier is upstream of its durable commit (`console_chat_controller.py:10779-10784` vs `:12773-12780`). An app-owned competing submission, blocked ledger operation, blocked event loop, or cleanup delay must be distinguished with task/attempt provenance.

Existing fences already handle stale-screen detach: `ConsoleRuntime.attach_view` rejects a superseded prior token (`console_runtime.py:4559-4567`), `detach_view` requires exact view and generation (`:4620-4623`), and `ChatScreen._console_runtime` is memoized (`chat_screen.py:9730-9753`). `on_unmount` retires the view and supplies its generation (`:16952-16959`). Therefore a claim stolen by some third screen can make the expected successor stale; it does not require the two-screen guard itself to be absent.

Reconciliation is deferred using `call_after_refresh` (`chat_screen.py:16664`), must synchronize UI and finish the exact runtime claim (`:16676-16687`), and only then starts polling (`:16728-16734`). `finish_view_reconciliation` re-arms an active wake (`console_runtime.py:4522-4536`). `_wait_for_selector` (`Tests/UI/test_destination_shells.py:1182-1193`) checks DOM presence, then one `pilot.pause`; it does not check reconciliation, current claim, or retired state.

## Leading hypothesis: competing startup screen in the mounted harness

This is a concrete hypothesis, not a reproduced cause. The first three tests use `_build_test_app`, enter `app.run_test`, then manually `push_screen(ChatScreen(app))`; they set `_initial_screen_pushed=True` only after awaiting that push. Meanwhile no-splash app mount schedules a retained deferred startup task (`app.py:3404`), and `_push_initial_screen` only checks `_initial_screen_pushed` at entry (`:4174-4177`), before construction and an awaited push (`:4248-4250`). A harness flag set during that await does not cancel a startup task that already passed the check.

Navigation constructs/restores its incoming screen before dismissing overlays (`app_navigation.py:767-910`, `:959-979`). `_dismiss_navigation_overlays` treats every stack entry above the two-entry content stack as a pushed overlay and dismisses it (`:410-425`). Thus an extra startup/harness Console can be resumed during dismissal, claim the runtime, and be unmounted during the subsequent switch. If its first runtime contact occurs there, its generation can supersede the already-constructed incoming screen; its detach can leave view=None, with the incoming screen's prior generation rejected on later ensure. This sequence fits all three lifecycle observations and should be instrumented before changing production.

It is also possible the lifecycle assertions merely happen before reconciliation, but that weaker explanation does not account for the second test's actual view=None or first test's explicit ensure without a further ownership transition. The fourth test likewise manually mounts an app Console and later replaces the runtime's store/controller/bridge with a separate rig, so competing startup work can affect it; the log does not prove a shared cause.

## Qualification of timing and isolation claims

The saved original `ownership-green.log` records 78 passes, 1 inherited strict XFAIL, 5 warnings in 521.98s. Its whole-file command is recorded in `task-1-report.md:206-216`; the runner supplied a fresh `TLDW_TEST_CONFIG_ROOT`, worktree PYTHONPATH and private basetemp. This differs from CI's four-file collection and package environment. `task-1-review.md` separately records source hashes matching immutable 07e and inherited FD warnings; no new passing claim is made here.

The three CI lifecycle calls were faster overall than their saved local passes: second visit 16.12s vs 20.00s; superseded screen 11.08s vs 21.00s; headless delivery 13.37s vs 22.59s. Native unchanged was slower, 78.14s vs 39.09s. These comparisons defeat a confident claim that all failures are simply slow CI; they do not rule out different scheduling within shorter totals. FD growth was 479 in CI and 489 in the saved local pass, so FD warnings alone are not a discriminating cause.

The whole ownership file explicitly keeps the bootstrap profile in `Tests/conftest.py:1245-1250`; native readiness setup persists splash disabled and refreshes configuration (`Tests/UI/test_console_native_chat_flow.py:239-257`). Do not attribute these failures to the familiar setup `RecoveryRequired` or enabled splash without evidence: the four tests reached their assertions/barrier.

Classification: unresolved. The leading explanation is test harness/startup ordering with real runtime claim consequences. CI does not yet establish a product defect in normal single-screen navigation, and local green does not dismiss one. Do not replace failing assertions with retries or widen timeouts as a repair before tracing claim ownership and admission.

## Sequential Task 5 reproduction recommendation

Use a pinned 7a tree for diagnosis, or record any new source hashes if Task 3/4 changes are included. Do not rerun the original 78 passing ownership nodes as a first step.

1. Run only the four failing nodes in their original order, fresh subprocess/profile, with the existing venv and a unique private profile/basetemp. Exact node selection:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_console_runtime_ownership.py::test_second_console_visit_reuses_the_runtime \
  Tests/UI/test_console_runtime_ownership.py::test_a_superseded_screen_never_detaches_the_successors_runtime \
  Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll \
  'Tests/UI/test_console_runtime_ownership.py::test_native_acceptance_consumes_only_open_target_revision[unchanged]' \
  --timeout=300 --tb=long -vv
```

2. Add disposable timestamped diagnostics at these boundaries, preserving production behavior: `_push_initial_screen` entry/construction/push completion; before/after each harness push/navigation; `attach_view`, `detach_view`, `finish_view_reconciliation`; reconciliation entry/return/retry. Record screen identity/class, full stack identities, current screen, generation/prior-generation, retired/closing flags, reconciled flags and startup-task done state. Record hook presence. No prompt text or credentials needed.
3. For native start, race the readiness Event with `start` completion for diagnostics: if start returns before entered, report its actual outcome; if still pending, capture preparation/ledger task states and identify controller/attempt for each diagnostics stage. The 63.5s durable commit log requires a provenance check, not an assumed readiness timeout fix.
4. If isolated four-node run is green, preserve the four-file CI collection imports while selecting only these nodes with `-k 'second_console_visit_reuses_the_runtime or a_superseded_screen_never_detaches_the_successors_runtime or opening_console_during_a_headless_delivery_arms_the_poll or (native_acceptance_consumes_only_open_target_revision and unchanged)'`. This separates collection influence without executing already-green neighbors.
5. Only if needed, repeat that same failing-node selection in a separate CI-matching environment (Python3.12.14/pytest9.1.1/asyncio1.4.0/Rich15) or under macOS `taskpolicy -b`, one variable at a time. Slow QoS is a diagnostic option from the repo lesson, not an established cause. If trace confirms startup overlap, deterministically hold startup's initial push while the harness mounts to pin it; if the same claim loss survives a single canonical Console with startup settled, investigate production navigation.

Any final repair must preserve assertions for runtime identity, exact successor claim, active-delivery polling and consumed receipt/composer ownership. Keep the inherited XFAIL and FD warning qualifications.
