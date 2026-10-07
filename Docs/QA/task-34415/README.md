# Console busy configuration entry — TASK-34415

Owner-approved bounded deferral, on `codex/console-busy-config-deferral`, originally
from dev `76d5d157aa6a628584ead573976c9adee72f5412`. Separate from PR3029's Library
test-only correction. The owner subsequently authorized latest-dev rebase,
scoped corrections and merge, accepting a fresh independent review instead of
credit-blocked Qodo for **PR3034 only**. This does not waive failed evidence or
unfinished qualification. [Approval recorded on the PR](https://github.com/rmusser01/tldw_chatbook/pull/3034#issuecomment-6029504207).

## Repair

Both incumbent refresh callers share the checked owner in
`UI/Console_Modules/config_sync.py`. It takes the unchanged native REBUILD then
FILE RLocks nonblocking and retains them across unchanged `operation(config)`.
Contention returns False and requests the existing coalesced 0.2s timer. Partial
acquisition is released; no selected-source read/render runs while busy. Replay
computes current state. Source checks, admission, native retirement, maintenance,
body/cleanup error precedence, whole-worker replay and teardown are unchanged.
No new config API, state/cache, lock, global policy or await.

The original branch shrank from 25231 lines/760 methods to 25204/759. Its published
25218/759 ceiling becomes 25204/759; nothing rises. Removing the redundant
private wrapper repairs the inherited method excess within this extraction.
The affected worker fixture supplies its incumbent session acknowledgement
callback without changing assertions. Formatting is mechanical; the ratchet's
unbounded `lru_cache` becomes stdlib `cache` with identical semantics.

ADR required: no — a routine scheduling repair under
[ADR126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md) and
[ADR120](../../../backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md).

## Initial branch evidence

Committed terminal receipts normalize trailing whitespace and final blank lines
only. Original captures remain in the `/tmp/task34415-*` directories; no message,
failure, warning, timing or assertion was removed.

- `red.log`: six real-thread held-lock scenarios fail with the old owner;
  both callers wait about 2.01–2.02s for the test watchdog, then render. GREEN
  asserts return while still held, not an arbitrary latency threshold. This run
  retains housekeeping warnings for old unrelated shared-temp debris; no cleanup
  or warning suppression was performed.
- `baseline.log`: current-config smoke passes; base line ratchet fails at
  25231 against 25218. Independent AST measurement finds 760 methods versus 759.
- `green.log`: six cases pass in 7.26s: both locks/callers, no read/publication,
  one retry, FILE-case partial REBUILD release proven on another thread, fresh
  checked config after save and source-drift refusal before replay publication.
- `first-combined.log`: 56 pass/five fail in 104.95s: four stale worker fixture
  callback failures and inherited method excess. Two inherited Splash
  SyntaxWarnings are retained, not fixed or waived.
- `base-worker-counterfactual.log`: the exact base checked owner reaches the
  same missing-callback error. Shared-owner counterfactual, not whole-repo BASE.
  The first driver attempt failed before fixture creation on a missing root;
  it remains in `/tmp` and is not regression evidence.
- `final-tests.log`: **61 pass in 113.84s, no pytest warnings**. Targeted config
  sync lifetime, native lock order, sync maintenance, mounted character activation,
  reduced motion and both Console ratchets. No full suite ran; earlier warnings
  remain recorded above.
- `initial-preflight.log` and `final-preflight.log`: all eleven derived-artifact
  guards pass, including the final source/receipt closeout rerun.
- `screen-lint-comparison.json`: full-screen Ruff has 192 diagnostics on base and
  work, none added/removed after accounting for shifted line references. Helper
  and both modified test paths are Ruff clean; four Python paths format clean.
  This is not a clean whole-screen lint claim.
- `design-detector.log`: Impeccable detector invoked once on two changed Python
  UI targets; exit 0/no output. It is not native terminal paint verification.

Independent read-only scoped review: no Critical/Important/Minor issues;
unchanged checked-scope/render AST, partial cleanup, fresh replay and final
test/size receipts confirmed. Review is not CI or Qodo acceptance.

## Latest-dev rebase

Rebased onto `cddc89d3e780b27392549b58ff9134b64e7d5907`. Both appended lessons
were retained in the sole conflict; the original repair replay is unchanged.
Upstream added 27 screen lines. The unchanged 25204 ceiling correctly went RED.
Caller census found no code/test/script users of `_get_shell_bar` or
`_collapse_console_hidden_control_bar`, and one user of `_get_compact_model_bar`.
Delete the unused helpers and inline that exact query/QueryError fallback;
remove only the unused `_summary_row_value` screen import, not its module-owned
implementation. Combined screen: **25204 lines/756 methods**; ceilings
25204/756. Both ratchets pass; neither ceiling rises.

All six existing compact-control/model-default/provider-mirror cases pass in
separate private pytest processes using the existing `bootstrap_profile` fixture
mode. Their assertions and production source checks are unchanged. The initial
ordinary five-case run failed before mounting on `raw_source_selection_changed`:
its fixtures reselected an already-bound config source. It is not a production
regression or passing UI evidence. The selected-profile driver adds only the
existing marker at collection, asserts exactly one original node and calls
`pytest.main`; it changes no test body, config getter or recovery admission.

The first combined rebase run recorded **60 passed/one overlay checkpoint
timeout/two inherited Splash SyntaxWarnings in 222.70s**. The unchanged six-case
activation-fault parametrization then passed **6/6 in 47.46s**, including overlay,
with its original deadlines and no warnings. Earlier failure/warnings remain
recorded; the passing rerun does not turn them into a warning-free qualification
claim. The final serial affected-suite rerun passed **61/61 in 235.40s, no pytest
warnings**, with unchanged assertions, deadlines and source guards. Fresh
independent review found no actionable code regressions; its historical-count
documentation finding was corrected before publication.

All eleven artifact guards pass on the combined source. Full-screen Ruff:
192 inherited diagnostics on latest dev, 189 on work; no additions after
normalizing shifted F811 line references. The three removals belong to deleted
code. Four Python paths format clean; helper and both changed tests Ruff clean.
Raw captures remain under `/private/tmp/pr3034-rebase-*`,
`/private/tmp/pr3034-rebased-*`, `/private/tmp/pr3034-compact-*` and
`/private/tmp/pr3034-overlay-recheck-SE88H7`; no warning or failure was suppressed.

Committed rebase receipts: `rebase-ratchet-red.log`, `rebase-first-affected.log`,
`rebase-compact-controls.log`, `rebase-fault-recheck.log`, `rebase-preflight.log`,
`rebase-screen-lint-comparison.json`, `rebase-final-affected.log`.
Only trailing whitespace was normalized. The compact-case driver below is run
once per original node, not on the whole module (configuration writes must not
leak between cases):

```python
import sys
import pytest


class SelectedBootstrapProfile:
    def pytest_collection_modifyitems(self, items):
        assert len(items) == 1
        items[0].add_marker(pytest.mark.bootstrap_profile)


raise SystemExit(pytest.main(sys.argv[1:], plugins=[SelectedBootstrapProfile()]))
```

## Exact-head CI mount-readiness correction

On head `abb9df541c4b8d375cc725994d33744b33e2055c`, Derived run
37562090298's PR lane failed only
`test_workbench_mounts_rail_canvas_inspector_and_loads_local_servers`:
1485 passed/one failed/one skipped. The rail had zero rows and test shutdown
pruned a replacement Select before its overlay composed. All four UI lanes and
Perf passed. The admission-sensitive step passed 335 tests/two expected failures
with six warnings, including aggregate FD growth; those warnings remain open.

The affected MCP source/test/workflow inputs were byte-identical to current dev.
The original test passed alone (one in 3.68s); its receipt retains unrelated
shared-temp housekeeping warnings. Holding the second source Select composition
for 0.2s reproduced the exact `SelectOverlay` exception once. The readiness fix
waits for the rail's asynchronous replacement and its three expected rows, under
the unchanged ten-second bound. All mode/server/canvas assertions remain intact;
no production code, profile selection, recovery guard, timeout or CI lane changed.

The same delayed-mount probe then passed once in 3.86s, with no pytest warnings.
The three unchanged real mount/compact-viewport cases passed in 7.35s, with no
pytest warnings. The changed file is formatted; its five inherited Ruff findings
are unchanged after accounting for shifted F811 line references. This is not a
clean whole-file lint claim. Production/scripts/package/workflow inputs remain
byte-identical to the reviewed head; the earlier 61 tests were not duplicated.
All eleven artifact guards pass in `ci-mcp-mount-preflight.log`; the complete
five-to-five Ruff comparison is in `ci-mcp-mount-lint-comparison.json`.

Receipts: `ci-pr-fast-lane-failed.log`, `ci-mcp-mount-alone.log`,
`ci-mcp-mount-red.log`, `ci-mcp-mount-green.log`, `ci-mcp-mount-targeted.log`.
Raw originals remain under `/private/tmp/pr3034-mcp-*` and
`/private/tmp/pr3034-pr-fast-lane-37562090298.log`. The fault probe runs the
original node with its ordinary fixtures, not an alternate profile or test body:

```python
import asyncio
import inspect
import sys
import pytest


class DelayedRailReplacement:
    @pytest.fixture(autouse=True)
    def delay_replacement_mount(self, monkeypatch):
        from textual.widgets import Select
        original = Select._on_compose
        source_compositions = 0

        async def delayed(select, event):
            nonlocal source_compositions
            if select.id == "mcp-rail-source":
                source_compositions += 1
                if source_compositions == 2:
                    print("PR3034 probe: second source Select compose held for 0.2s")
                    await asyncio.sleep(0.2)
            result = original(select, event)
            if inspect.isawaitable(result):
                await result
        monkeypatch.setattr(Select, "_on_compose", delayed)


raise SystemExit(pytest.main(sys.argv[1:], plugins=[DelayedRailReplacement()]))
```

Fresh read-only scoped review subsequently confirmed the rebased
`bc33d012bca665b3176dfc322821ef4131bd9ddc` correction with no
Critical/Important/Minor findings. Its production, test and artifact inputs
matched the reviewed pre-rebase tree; focused lock/size replay passed eight
tests in 14.28s without pytest warnings, with both Backlog guards passing.

## Exact-head CI pending-approval publication correction

On that head, Derived run 37565637296's UI4 lane failed the unchanged legacy
approval journey at Inspector count zero versus one: **240 passed/one failed/two
warnings in 568.83s**. PR and UI1/2/3 lanes and Perf passed; the failed lane is not
passing CI evidence. The live worker, round and approval card remained valid,
and the incumbent count builder returned one. A refresh may defer on a busy
native config lock; a single idle pause is not a publication receipt.
CI did not record lock contention attribution; the real-lock probe below
establishes that supported deferral path, not which lock the CI run encountered.

The original journey passed alone (parent one in 95.29s; private child one in
82.75s with an FD warning). A diagnostic separate thread then held the actual
native REBUILD RLock through the original idle pause. It reached the valid
unpublished frame, proved that the original sync returned False with the
installed retry scheduled, released and joined the holder, and reproduced the
exact zero-versus-one assertion. No count, sink, source/profile or recovery
state was substituted; the finite watchdog did not expire.

The test-only correction reuses the existing bounded settle helper to observe
the Inspector's actual count and original rendered approval text, then its
cleared count. All original worker, session, round, Files, decision and count
assertions remain; production, CI and qualification limits are unchanged. The
same real-lock probe passed (parent one in 76.70s; child one in 68.29s). The
ordinary affected module passed **13 tests in 71.62s**, including its original
six-journey private child (one in 61.32s). The changed test is Ruff and format
clean; production/scripts/package/workflows match the reviewed head.
All eleven existing artifact guards pass in `ci-pending-preflight.log`.

Resource warnings are retained, not fixed or waived: CI child FD growth 202
(14 to 216), ordinary baseline growth 205 (14 to 219), controlled RED and GREEN
growth 208 (14 to 222), final ordinary child growth 202 (14 to 216), against the
unchanged limit 200. A passing parent does not report its child's warning and
does not justify a warning-free whole-application qualification claim.

Receipts: `ci-pending-ui4-failed.log`, `ci-pending-baseline-child.log`,
`ci-pending-lock-red-child.log`, `ci-pending-lock-green-child.log`,
`ci-pending-final-parent.log`, `ci-pending-final-child.log` and the exact
diagnostic source `ci-pending-native-lock-probe.txt`. Only trailing whitespace
was normalized; full retained copies compare equal to normalized originals.
Raw originals remain in `/private/tmp/pr3034-pending-*` and
`/private/tmp/pr3034-ui4-failed-37565637296.log`. The probe runs the original
private-profile wrapper using `PYTEST_PLUGINS=pr3034_pending_lock_probe` and
`PYTHONPATH=/private/tmp/pr3034-pending-lock-probe-S7Aeia:<checkout>`; it changes
no test body, profile selection or config getter. Fresh independent read-only
review of the twelve staged files found no Critical/Important/Minor findings;
the original production and MCP reviews remain applicable by byte identity.
New exact-head CI remains required before protected merge.

## Exact-head CI completed-startup-timer correction

On `ee51e0b1e762463b38f7b5144161458c02d60702`, Derived run 37569263239's
UI1 lane failed before its approval geometry assertions: the shared readiness
helper asserted that the startup projection timer still existed. The lane
recorded 650 passed/one failed/two warnings in 520.06s. PR, UI2/3/4 and Perf
passed; this is not a passing aggregate CI claim. Installed Textual keeps timers
in a WeakSet, so a normally completed one-shot can disappear before lookup.

A call-through, weak-reference probe observed the real successful projections
and natural timer retirement, then ran the original private-profile geometry
node. It reproduced the exact None assertion (child one failed in 8.78s). The
minimal test-helper correction awaits a timer only if still present, retaining
the existing call_next drain and every readiness/layout/focus assertion and
timeout. The identical probe passed (child one in 9.58s). A separate gate on the
actual timer tick proved the helper still waits while that timer is pending,
then passed the same geometry assertions (child one in 9.66s). Neither probe
substitutes a projection result, profile, config getter or recovery admission.

The ordinary six-consumer selection passed geometry but the five other cases
refused changed config selection before mounting, before reaching this helper.
Those failures are retained, not repaired by bypassing admission. Each original
remaining control then passed in a separate process using the existing
bootstrap_profile fixture mode. The shared helper file is format clean; its
four inherited Ruff findings remain unchanged. No production/script/package/
workflow changes; resource warnings and qualification limits remain unwaived.

Receipts: `ci-projection-ui1-failed.log`, `ci-projection-red-child.log`,
`ci-projection-green-child.log`, `ci-projection-pending-child.log`, their parent
logs, `ci-projection-targeted.log` (the initial profile refusals),
`ci-projection-bootstrap-controls.log`, `ci-projection-lint-comparison.json`
and `ci-projection-preflight.log` (all eleven guards pass). Exact disposable
sources: `ci-projection-retired-probe.txt`, `ci-projection-pending-probe.txt`
and `ci-projection-bootstrap-driver.txt`. Originals remain in
`/private/tmp/pr3034-projection-timer-probe-DNpme6` and the failed hosted log in
`/private/tmp/pr3034-fd-origins.bGYAOJ/ui1-current-head.log`. Only trailing
whitespace was normalized; no failure, warning or assertion was omitted.
Fresh scoped review and latest-dev integration are recorded below. New
exact-head CI remains required before protected merge.

## Latest-dev dependency integration

Rebased all five authored patches onto dev
`6a08c6a18add254751023387d6e97203552efa2b` (PR3037). Every range-diff entry is
identical, and all authored production/test/script/package/workflow inputs are
byte-identical to tested timer-correction commit
`77c075f9a318eb7bdc4fd421ec38ea235734a8f7`. Upstream raw/storage admission and
saved SessionEnd dependencies did change; this is not a whole-tree identity
claim. The checked sync owner still retains native REBUILD then FILE locks and
fresh source/admission checks; this PR adds no cache or lifecycle policy.

On the combined source, the original affected selection passed **61 tests in
174.90s, no pytest warnings**. The original private-profile geometry node passed
again (child one in 9.98s), and the other five original helper consumers passed
in separate existing bootstrap-profile processes. All eleven artifact guards
passed. No full sweep, performance qualification or warning suppression ran.

The extra incoming raw/native-pause/SessionEnd and hook teardown selection is
**non-green: 29 passed, one teardown error in 487.88s**. Its mounted Interrupt
body passed, but the unchanged 300-second timeout fired during pytest-asyncio
executor shutdown; a worker remained in the real agent bridge's
`future.result`. The same node, body-pass/teardown-error outcome and blocked
bridge/executor stack are already reproduced on a complete exact-base export in
[the incoming SessionEnd report](../../superpowers/qa/2026-10-06-task33648-session-end/report.md).
The existing archive and its exact-base log/receipt hashes match that manifest.
This supplies a known-baseline disposition, not a unique-cause attribution,
passing aggregate certificate or lifetime fix. Full retirement and resource
warnings remain open; neither this test nor its timeout was changed or skipped.

Fresh independent scoped review of `6dc45406bd8c3470fb49a83d45d6a3c81169c870`
found no Critical/Important/Minor source findings, including the helper's six
consumers and relevant incoming contracts. The reviewer also accepted the
documented known-baseline disposition after comparing authentic retained
evidence. The non-green integration receipt is not credited as a pass. Source
and reviewed evidence are ready to publish, not cleared to merge without new CI.

Receipts: `rebase-latest-affected.log`, `rebase-latest-geometry.log`, its child
log, `rebase-latest-control-1.log` through `rebase-latest-control-5.log`,
`rebase-latest-incoming.log` and `rebase-latest-preflight.log`. They compare equal
to originals in `/private/tmp/pr3034-rebase-6a08-U06HKm` after trailing-whitespace
normalization only. The fourteen new timer-correction receipt files above also
compare equal to their normalized originals; no warning or failure was omitted.

## Prior bounded task closeout

Published head `57fa8b739d3b5def6c105b344ff9e230b454fba1` on current dev
`6a08c6a18add254751023387d6e97203552efa2b` completed
[Derived Artifacts 37573814397](https://github.com/rmusser01/tldw_chatbook/actions/runs/37573814397)
and [Perf Guard 37573814378](https://github.com/rmusser01/tldw_chatbook/actions/runs/37573814378)
successfully. The PR lane, all four UI lanes and required **Derived artifacts
reproduce from their sources** passed on that exact head. Fresh independent
scoped review has no Critical/Important/Minor findings; actual GitHub reviews
and threads were empty at closeout.

All six acceptance criteria and the documented scoped DoD are satisfied.
Only TASK34415 was marked Done through the current Backlog CLI. The closeout
changes documentation only; tested source, existing receipts, inherited static
debt and failures/warnings remain unchanged. This final closeout commit needs
its own exact-head CI before the authorized protected merge. No merge or
warning-free full qualification is claimed here; the non-green incoming
mounted Interrupt teardown receipt remains recorded above.

## Final-head callback-drain failure and bounded correction

The documentation-only head `97e64dd93b4aaa2c264ee8d73465376daef825e4`
did **not** receive merge clearance:
[Derived 37577168073](https://github.com/rmusser01/tldw_chatbook/actions/runs/37577168073)
failed UI1 at the original private-profile approval geometry readiness assertion,
`assert not screen._console_sync_requested`. UI1 reported 650 passed, one failed
and two warnings in 554.27s. PR, UI2/3/4 and Perf passed, but required derived
reproduction failed. Only TASK34415 was reopened, with AC7 added before the
test-only correction. The prior green closeout is historical evidence.

One drained callback is not completion of a full refresh deferred to the
installed coalesced timer. A real native REBUILD holder and a second real sync
request reproduced the exact original assertion with clean holder retirement.
A separate real pending-worker control exposed the interval after the replay
clears its flags but before its `console-sync` worker finishes. These controls
prove supported deferral, not which lock the hosted runner encountered.

The shared helper retains its unconditional callback drain and all original
readiness/geometry/focus assertions. Within the existing ten-second projection
phase it now observes all four real sync/replay flags and unfinished workers
owned by this screen in the existing `console-sync` group. No production,
profile, admission, workflow or qualification bound changes. Both native and
worker controls pass; a frozen replay still fails the original assertion within
the unchanged bound. All six original consumers pass, including the original
private-profile geometry child. Four inherited helper Ruff findings remain
unchanged; format and all eleven artifact guards pass.

Fresh review also caught the initial poll's independent 30-second
`Pilot.pause` screen drain. The final wait uses only an asyncio yield capped
by remaining phase time. A direct call-path guard is RED before this change
and GREEN with the actual native/pending-worker replay afterward; this is not
a claimed hosted 30-second timeout reproduction. All six original consumers,
the frozen-replay negative control, format, unchanged inherited lint comparison
and all eleven guards were rerun after that one-line adjustment.

The [audited receipt](ci-readiness-receipt.md) records positive, negative and
invalid preparation outcomes without crediting preparatory failures as RED.
The exact runnable [diagnostic source](ci-readiness-probe.txt) is retained.
The [phase guard](ci-readiness-budget-guard.txt) pins the non-draining poll.
Full unmodified raw parent/child logs and XML remain local, indexed by
[SHA-256 manifest](ci-readiness-sha256.txt). A safeguard rejected publication of
the diagnostic dumps because of possible environment/credential-like metadata;
this narrower receipt publishes outcomes and hashes, not those dumps.
It is not a full-log or normalized-copy claim. The hosted failed log remains
available in the linked run and locally; existing warning/failure receipts are
unchanged. Fresh scoped review and new exact-head CI are required before merge.

Fresh final staged review found no Critical/Important/Minor findings and
resolved both completion and polling-bound findings. The reviewer verified
all six consumers, exact probe sources and all 95 raw hashes; the first 65
entries are unchanged. This clears the scoped correction, not latest-dev
replay or new exact-head CI. All seven AC are checked; TASK34415 remains
In Progress for those publication gates and bounded closeout.

## Latest-dev File Notes replay

Latest-dev replay onto `5a607a14bffe9f9d52ec348985819e2c927a1fdd`
(merged PR3016 File Notes history/retention/delete safety) had no authored
path overlap. All eight patches have identical range-diff and every authored
Console/test/script/package/workflow input remains byte-identical to reviewed
`84d53ecfbb7087a69904f08113a9a4055748de43`; this is not whole-tree identity.
Library's File Notes factory and non-null-workspace shutdown boundary were
inspected. The combined tree passes nine focused lock/size/geometry cases in
17.72s, its original private geometry child in 5.94s, and all five separate
original bootstrap-profile helper controls. No pytest warnings; all eleven
artifact guards pass. The audited receipt and 111-entry manifest retain
the new integration evidence from `/private/tmp/pr3034-rebase-5a607-qtopba`.
Rebased source head `4382dafadc15a3a945fd67cc1c0b367cb69d9990` and the final
evidence-only notes still require reviewed publication and fresh exact-head CI.

Final replay-identity/evidence review confirmed that source head and the four
staged evidence-only paths: no Critical/Important/Minor findings. All 111 hashes
verify with the previous 95 unchanged; original production/readiness reviews
remain applicable through exact authored-input identity. The final evidence
commit preserves reviewed/tested source. Publication is ready; new exact-head
CI and bounded task closeout remain required, not merge or qualification clearance.

## Still open

Busy-lock responsiveness is not full unchanged 50ms activation qualification.
Native viewport/ordinary quit, actual Windows Terminal, three unfamiliar
participants, whole-app retirement, aggregate FD warnings and the separate
baseline closed-cursor finding remain follow-up. TASK31966/TASK31245 remain
In Progress; their missing criteria are neither checked nor waived. No semantic
implementation, global cache/GC/lifetime expansion or Terminal control workaround.
The original production, MCP and pending-approval correction reviews are
complete. New exact-head CI remains required before protected merge.
