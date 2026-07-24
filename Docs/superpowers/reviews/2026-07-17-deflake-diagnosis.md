# Deflake diagnosis: four reported flaky tests

Investigated on branch `claude/footer-hints-264` (origin/dev merged in), worktree
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`. All reproduction runs used
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`, run in the foreground, one process
at a time. No production or test files were edited — diagnosis only; fixes below are proposals for an implementer.

---

## 1. Tests/UI/test_console_native_chat_flow.py — "generic-provider-send... fixture HTTP 502"

**Test path as filed**: `Tests/Chat/test_console_native_chat_flow.py` — **this path does not exist** (verified via
`git log --all --follow` on that path: no history at all under `Tests/Chat/`). The file has always lived at
`Tests/UI/test_console_native_chat_flow.py`. The best name match for "the generic-provider-send test" is
`Tests/UI/test_console_native_chat_flow.py::test_console_native_generic_provider_send_renders_completed_message`
(line 905).

### Reproduction evidence

- Isolated, single-test invocation, both under `Tests/UI/pytest.ini` (picked up automatically when the test path
  is specified directly — `asyncio_mode=auto`) and forced under the root `pyproject.toml` config
  (`-c pyproject.toml`, `--import-mode=importlib`, `timeout=300` global): **1/1 passed** each way (~1.8s).
- 30 further solo repeats (2 batches of 15): **30/30 passed**, no variance in timing (~1.8–1.9s each).
- Full file (`Tests/UI/test_console_native_chat_flow.py`, 187 tests): **187/187 passed** in 152s.
- Total: **0 failures in 30 solo runs + 1 full-file run (187 tests)**. Does not reproduce.

### Root cause (from code reading, since it did not reproduce)

The test never touches a real HTTP fixture. It monkeypatches
`tldw_chatbook.Chat.Chat_Functions.chat_api_call` directly (line 915-918) with a synchronous fake that returns a
plain string. The Console gateway's generic (non-llama.cpp) send path
(`ConsoleProviderGateway._chat_api_call`, `tldw_chatbook/Chat/console_provider_gateway.py:1064-1069`) does a
late-binding import — `from tldw_chatbook.Chat.Chat_Functions import chat_api_call` **inside the method body**,
executed at call time from an `asyncio.to_thread` worker (`_stream_generic_chat`, lines 933-1013) — so whichever
function object is bound to `Chat_Functions.chat_api_call` at the moment of the call is used, regardless of which
thread performs the import. This is a safe, deliberate pattern (also used identically in
`tldw_chatbook/Agents/agent_service.py:82-83`) that structurally prevents the classic "monkeypatch raced by an
early-bound reference" bug. `ConsoleProviderGateway.resolve_for_send` (lines 624-749) — the resolution path this
test exercises for a generic (`openai`) provider — never performs a reachability probe or any `httpx` call; that
machinery (`_active_http_client`, `_is_reachable`) is exclusively used by the `llama_cpp`/`local_llamacpp` branch
(`resolve_llamacpp`, line 540), which this test does not take. There is therefore no code path in this test by
which a real HTTP request (and thus an HTTP 502) could occur today.

Grepped the whole feature area (`Tests/Chat/*.py`, `Tests/UI/test_console_native_chat_flow.py`) for any local
fixture HTTP server: the **only** real local server fixture in this feature area is `local_http_server` in
`Tests/Chat/test_console_provider_gateway.py:1282-1291` (a `http.server.ThreadingHTTPServer` on an ephemeral
port), used by exactly two tests — `test_owned_http_client_survives_agent_bridge_style_loop_swap` and
`test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop` (this task's flake #3, see
below). Both live in `console_provider_gateway.py`'s file family, not `test_console_native_chat_flow.py`.

**Conclusion**: this flake bucket, as filed, could not be reproduced and the exact test it names structurally
cannot produce an HTTP 502 today. The overwhelmingly likely explanation is that "fixture HTTP 502" is the same
underlying incident as flake #3 below (same feature area — Console provider-gateway readiness/reachability
probing — same `local_http_server` fixture, same probable failure shape: a real local socket under real
concurrent load timing out and surfacing as an HTTP-layer error), misfiled under the wrong test name/path when the
flake list was compiled. See flake #3's root cause and fix — it is the one concrete, reproduced mechanism in this
codebase that turns a local test HTTP server's behavior under load into an assertion failure that could plausibly
be summarized after the fact as "fixture HTTP 502."

### Classification

**Unable to confirm — no reproduction, no product-code path that can produce the symptom as described.** Not
classified as product bug, test bug, or environment issue; recommend re-filing.

### Proposed fix

No code change proposed against `test_console_native_generic_provider_send_renders_completed_message` itself —
there is nothing to fix; it is not exhibiting a bug. Concrete next steps:
1. **Do not touch this test.** If it is observed failing again, capture full `--tb=long` output (this
   investigation's tooling shows the default `-q` output truncates the interesting frames) and re-file with the
   actual traceback; that will show conclusively whether a 502 is really involved and from where.
2. Apply flake #3's fix (widen the local HTTP fixture's timeout/backlog headroom, see below) — if the two flakes
   are the same incident as hypothesized, this closes both at once.
3. If a *future* PR adds a real reachability probe to the generic (non-llama.cpp) provider path (mirroring
   `resolve_llamacpp`'s probing), that new probe should reuse the same widened-timeout `local_http_server` pattern
   from the start rather than the tight `PROBE_TIMEOUT_SECONDS = 5.0` default, given flake #3's evidence that 5s is
   already too tight under real thread contention against a stdlib `ThreadingHTTPServer`.

### Covering-test guidance

None needed for this specific test. If flake #3's fix is applied, re-run flake #3 per its own guidance below; if a
"generic provider reachability" fixture-server test is added in the future, give it the same timeout/backlog
headroom as flake #3's fix from day one.

---

## 2. test_skills_import — trust-pending intermittent failure

**Location**: `Tests/Skills/test_skills_import.py::test_import_real_superpowers_skills_lands_trust_pending`
(line 114). Confirmed via grep for `trust_pending`/`skills_import` across `Tests/`.

This exact test was already independently observed failing (out of scope) during a prior de-flake investigation
recorded in this same worktree at `.superpowers/sdd/flakes-report.md` (search "unrelated
`Tests/Skills/test_skills_import.py`" — reproduced live inside a large combined sweep of
`Tests/Skills Tests/Library Tests/UI/test_library_skills_canvas.py Tests/UI/test_library_shell.py
Tests/UI/test_screen_navigation.py Tests/UI/test_destination_shells.py`, ~923s, 1100+ tests in one process).

### Reproduction evidence (this investigation)

- Solo, single-test invocation: consistently **~18–19s per run** (this is itself notable — see root cause).
- 23 solo repeats (8 + 15, separate batches): **23/23 passed**, timings tightly clustered 17.4s–19s.
- `Tests/Skills` alone (128 tests): **128/128 passed**, 62s.
- `Tests/Skills` + `Tests/Library` together (676 tests): **676/676 passed**, 296s, 93% CPU utilization.
- Total: **0/23 solo + 0/128 + 0/676 combined = did not reproduce** within this investigation's time budget. The
  prior flakes-report.md investigation *did* observe a live failure of this same test, but only inside a ~15-minute,
  1100+-test single-process sweep — a scale of contention this investigation's foreground time budget could not
  replicate (the closest attempt, the 676-test Skills+Library combo, still passed).

### Root cause (from code reading + timing measurement)

`_run_skills_import_via_ui` (`Tests/Skills/test_skills_import.py:89-110`) polls for the Import row's status
`Static` text to *change* from its pre-press value, using a **fixed iteration count**, not a wall-clock deadline:

```python
for _ in range(attempts):        # attempts=150
    status_text = str(screen.query_one("#library-skills-import-status", Static).renderable)
    if status_text != previous:
        return status_text
    await pilot.pause(0.02)
return status_text
```

150 × 0.02s = a **hard 3.0-second ceiling**, after which it silently returns the stale (unchanged) value rather
than raising — the caller's `assert status == "1 imported · re-review it in the trust panel"`
(`test_skills_import.py:136`) is what actually surfaces the failure, comparing against whatever was on screen
before the press.

This is real, non-mocked work: `_run_library_skills_import`
(`tldw_chatbook/UI/Screens/library_screen.py:6561-6674`) reads the real `SKILL.md` file via
`asyncio.to_thread(skill_md_path.read_text, ...)` (line 6647-6649), then calls the real
`SkillsScopeService.import_skill(...)` through `_run_library_service_call(..., isolate_in_worker=True)`
(line 6661-6669). `_run_library_service_call` (`library_screen.py:1688-1715`) with `isolate_in_worker=True` runs the
service call via `asyncio.to_thread(invoke_service_in_worker)`, and `invoke_service_in_worker`
(lines 1697-1707) itself calls `asyncio.run(await_result())` — **a brand-new OS thread AND a brand-new asyncio
event loop are spun up per single skill import**, on top of real trust-store bootstrapping/hashing work in
`LocalSkillsService`/`SkillsScopeService`.

Measured cost on this (currently idle) machine: **~18.5s wall-clock for the whole test, 5 sequential real
imports ⇒ ~3.5s average per import** — i.e., the real per-import cost is *already close to or exceeding* the
test's own 3.0-second polling ceiling with **zero external load**. Any additional CPU contention (extra
threads/GC pauses/OS scheduling delays from other tests running in the same process, as in the prior
flakes-report.md combined-sweep reproduction) tips individual imports over the 3.0s ceiling.

Which import in the loop actually surfaces as a *failure* (vs. a masked false pass) matters: `previous` is
captured fresh before each press (line 99). For the **first** skill in `REAL_FIXTURE_SKILLS` ("executing-plans"),
`previous` is the Import row's pristine empty-string status, which differs from the eventual success string, so a
timeout on this first import produces a clean, visible failure:
`unexpected outcome importing 'executing-plans': ''`. For skills **2–5**, `previous` already equals the exact
success string from the *prior* skill's completed import (all five produce byte-identical success copy) — so if
one of *those* imports is slow enough to hit the same 3.0s ceiling, the loop returns the still-unchanged
`previous` value, which happens to already equal the expected string, producing a **false pass** that masks the
race rather than exposing it. This means the observable flake is concentrated on the first import in the loop,
though the underlying timing risk exists on all five.

### Classification

**Test bug** (timing assumption: fixed-iteration polling budget too tight for real, non-mocked, thread/loop/disk
work, not a wall-clock deadline). This is the exact same failure *shape* already diagnosed and fixed once in this
codebase for a sibling file — see `.superpowers/sdd/flakes-report.md`'s task-192 precedent
(`Tests/UI/test_library_shell.py`, converting `for _ in range(150): ... await pilot.pause(0.02) ... else: raise`
to a wall-clock `_wait_for_condition(pilot, predicate, timeout=..., message=...)` helper) — this is the same
precedent applied to a different file. Not a product bug: the import itself completes correctly every time
observed; the test just sometimes doesn't wait long enough to see it.

### Proposed fix

In `Tests/Skills/test_skills_import.py`, convert `_run_skills_import_via_ui` (lines 89-110) from a fixed
iteration count to a wall-clock deadline, with a generous ceiling (recommend **20–30 seconds**, given the
measured ~3.5s/import baseline with zero load):

```python
import time  # add to the file's stdlib imports

async def _run_skills_import_via_ui(
    screen, pilot, path: Path, *, timeout: float = 20.0
) -> str:
    previous = str(screen.query_one("#library-skills-import-status", Static).renderable)
    screen.query_one("#library-skills-import-path", Input).value = str(path)
    await pilot.pause()
    screen.query_one("#library-skills-import-run", Button).press()
    await pilot.pause()
    status_text = previous
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_text = str(screen.query_one("#library-skills-import-status", Static).renderable)
        if status_text != previous:
            return status_text
        await pilot.pause(0.02)
    return status_text
```

Only the polling mechanism changes (fixed count → wall-clock deadline); the fast path (status changes quickly)
is unaffected in speed. All 5 call sites in `test_import_real_superpowers_skills_lands_trust_pending`
(line 135) use the default, so no call-site changes are needed; other tests in the same file that call this
helper (e.g. around lines 156-448) inherit the same fix automatically.

**Secondary, lower-priority correctness gap** (do not conflate with the primary fix, flag as an optional
follow-up): because `previous` can coincidentally already equal the target success string for imports 2–5, a
slow-but-eventually-successful import in that position can still produce a *false pass* even after the wall-clock
fix, since "unchanged from previous" is used as the sole signal that nothing new happened. Closing this fully
would need a positive signal that this specific import round actually completed (e.g. asserting against a
per-round sequence/generation counter, or asserting the *set* of already-imported skill names grew) rather than
"the text is different from what it was before this round." Not required to fix the reported flake (which
manifests as a hard failure, not a silent pass), but worth a follow-up ticket.

### Covering-test guidance

After applying the fix, re-run the same combined sweep that reproduced it historically (from
`.superpowers/sdd/flakes-report.md`):
```
Tests/Skills Tests/Library Tests/UI/test_library_skills_canvas.py Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_navigation.py Tests/UI/test_destination_shells.py
```
to confirm the fixed test survives the same contention that previously tripped it (budget ~15+ minutes,
background run recommended for that specific verification pass, outside this investigation's foreground-only
constraint).

---

## 3. test_active_http_client_concurrent_swap — REPRODUCED

**Location**: `Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop`
(line 1468). Confirmed via grep for `test_active_http_client_concurrent_swap` across `Tests/`.

### Reproduction evidence

- Solo repeats, batch 1 (15 runs): **14 passed, 1 failed (run 11)**, ~1.8s/run.
- Solo repeats, batch 2 (25 runs): 25/25 passed.
- Solo repeats, batch 3 (40 runs): 40/40 passed.
- Solo repeats, batch 4 (50 runs): 50/50 passed.
- Solo repeats, batch 5 (150-run loop, logged to file, capturing full tracebacks on failure): **2 failures**
  observed, at run 25 and run 49 of that batch.
- **Total across all batches: ~3 failures in ~280 solo, single-process runs** (~1% failure rate), each run taking
  ~1.8–1.9s. **Reproduced.**

Captured failure (identical shape both times):

```
local_http_server = 'http://127.0.0.1:51370'
...
>       assert errors == []
E       AssertionError: assert [AssertionErr...ool>.result')] == []
E
E         Left contains one more item: AssertionError('assert False is True
 +  where False = result(timeout=10)
 +    where result = <Future at 0x1062d7bf0 state=finished returned bool>.result')
E         Use -v to get more diff

Tests/Chat/test_console_provider_gateway.py:1538: AssertionError
```

Critically, the future **completed** (did not hang/timeout at the outer `future.result(timeout=10)` level) and
raised **no** `RuntimeError` about being "bound to a different event loop" — the specific regression class this
test exists to catch (PR #629 Fix 1(a)). The only thing that failed is that `gateway._is_reachable(...)`
(the coroutine actually run) returned `False` instead of `True`.

### Root cause (confirmed by reading `_is_reachable` and the fixture)

`ConsoleProviderGateway._is_reachable` (`tldw_chatbook/Chat/console_provider_gateway.py:1132-1140`):

```python
async def _is_reachable(self, base_url: str) -> bool:
    try:
        await self._active_http_client().get(
            f"{base_url.rstrip('/')}/health",
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return False
    return True
```

`httpx.HTTPError` is the base for `httpx.TransportError`, which covers `ConnectTimeout`/`ReadTimeout` — i.e. any
timeout at `PROBE_TIMEOUT_SECONDS = 5.0` seconds (`console_provider_gateway.py:36`) is silently swallowed and
returned as a *legitimate-looking* `False`, not an exception the test's `except BaseException` branch would
distinguish from a real "unreachable" result.

The test (`test_console_provider_gateway.py:1484-1531`) is a genuine adversarial concurrency hammer: 6 OS threads,
each running its **own** persistent `asyncio` event loop, synchronized on a `threading.Barrier` so **all 6 race
`_active_http_client()`'s check-and-swap simultaneously, every one of 20 rounds**. `_active_http_client`
(`console_provider_gateway.py:467-521`) creates a **brand-new `httpx.AsyncClient`** (and thus a cold TCP
connection, no pooled/warm socket) whenever `self._client_loop is not loop` — which, given the barrier forces
6 different loops to contend for the single owned client on every round, is effectively **every single call** in
the worst case (whichever loop's thread wins the lock this round almost certainly differs from whoever won last
round). That means the test can force up to `6 × 20 = 120` **fresh TCP connection attempts** against the fixture
server in rapid, barrier-synchronized bursts.

The fixture (`local_http_server`, `test_console_provider_gateway.py:1282-1291`) is a plain
`http.server.ThreadingHTTPServer(("127.0.0.1", 0), _JSONOKHandler)` with **no `request_queue_size` override** —
confirmed via `socketserver.TCPServer.request_queue_size == 5` (stdlib default). With up to 6 threads opening new
connections in the same synchronized instant against a backlog of 5, plus 12+ live threads (6 hammer threads + 6
persistent loop-runner threads) contending for the GIL/CPU on the host machine, an individual probe's real,
non-mocked TCP-handshake-plus-GET occasionally takes long enough to exceed the hardcoded 5-second
`PROBE_TIMEOUT_SECONDS`, producing a genuine (not-mocked) `httpx.HTTPError` → `_is_reachable` returns `False` →
the test's own `assert future.result(timeout=10) is True` raises `AssertionError` → collected into `errors` →
final `assert errors == []` fails.

This is **not** a resurgence of the atomicity bug the test targets — the observed failure mode (a benign,
swallowed timeout, not a cross-loop `RuntimeError`) confirms the `_client_lock` guard
(`console_provider_gateway.py:441`, used inside `_active_http_client`) is working correctly. It is the test's own
adversarial design (maximal simultaneous fresh-connection churn against a tiny stdlib server with a small backlog)
occasionally outrunning a timeout budget that was sized for a single, uncontended probe, not a 120-connection
barrier-synchronized flood.

### Classification

**Test flakiness / environment sensitivity**, not a product bug. The production atomic-swap fix (PR #629 Fix
1(a)) remains intact; the regression test's own timing budget (a 5-second, hardcoded, non-overridable-from-the-test
`PROBE_TIMEOUT_SECONDS` combined with a 5-connection server backlog) is too tight for the deliberately extreme
concurrency it generates, under real OS/GIL scheduling variance.

### Proposed fix

Two independent, additive changes to `Tests/Chat/test_console_provider_gateway.py`, both test-only (zero
production risk):

1. **Widen the fixture's connection backlog.** Replace the plain `http.server.ThreadingHTTPServer` in
   `local_http_server` (lines 1282-1291) with a small subclass that raises `request_queue_size` (must be set
   before the socket starts listening, i.e. as a class attribute, not a post-construction instance mutation):
   ```python
   class _RobustThreadingHTTPServer(http.server.ThreadingHTTPServer):
       request_queue_size = 128

   @pytest.fixture
   def local_http_server():
       server = _RobustThreadingHTTPServer(("127.0.0.1", 0), _JSONOKHandler)
       ...  # unchanged
   ```

2. **Widen the probe timeout for this specific hammer test only**, via `monkeypatch` on the *actual* module
   attribute `_is_reachable` reads (not the test file's already-imported copy of the name — the test file does
   `from tldw_chatbook.Chat.console_provider_gateway import (..., PROBE_TIMEOUT_SECONDS, ...)`, which only
   creates a local alias; patching that alias would not affect `_is_reachable`, which reads its own module's
   global at call time). Add the module import and patch it directly:
   ```python
   from tldw_chatbook.Chat import console_provider_gateway as console_provider_gateway_module
   ...
   def test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop(
       local_http_server, monkeypatch,
   ):
       monkeypatch.setattr(console_provider_gateway_module, "PROBE_TIMEOUT_SECONDS", 20.0)
       ...
   ```
   and correspondingly raise the outer `future.result(timeout=10)` (line ~1520) to something comfortably above
   the new probe timeout, e.g. `timeout=30`, so the outer wait is never the binding constraint.

Together these remove the two concrete sources of unforced latency (small accept backlog + tight per-request
timeout) without weakening what the test actually verifies (the atomic check-and-swap under real concurrent
callers) — pass/fail should then depend purely on the swap's correctness, not on whether 120 barrier-synchronized
fresh TCP connections complete within 5 seconds on a possibly-busy machine. `thread_count`/`rounds` do not need to
be reduced; the fix targets the two artificial bottlenecks instead of weakening the adversarial coverage.

### Covering-test guidance

Re-run the modified test solo at least 200 times (now ~1.8s/run, so ~6 minutes total) to confirm the previously
~1% failure rate (3 failures in ~280 runs measured here) drops to 0. Also re-run
`test_owned_http_client_survives_agent_bridge_style_loop_swap` (the fixture's other consumer, lower risk since it
only issues 2 sequential requests) to confirm no regression from the backlog-size fixture change.

---

## 4. Tests/UI/test_console_composer_cursor_blink_keeps_row_count_stable_at_wrap_width — load-flaky

**Location**: `Tests/UI/test_console_internals_decomposition.py::test_console_composer_cursor_blink_keeps_row_count_stable_at_wrap_width`
(line 581). A sibling test with the identical hazard, `test_console_composer_cursor_blink_toggles` (line 557), was
also inspected since it shares the exact same root cause.

### Reproduction evidence

- Solo, single-test invocation: 1/1 passed, ~1.85s.
- 20 further solo repeats: **20/20 passed**.
- Full file (`Tests/UI/test_console_internals_decomposition.py`, 118 tests): **118/118 passed**, ~100s.
- Total: **0 failures in 21 solo runs + 1 full-file run (118 tests)**. Did not reproduce within this
  investigation's foreground, single-process time budget — consistent with the task's own framing ("fails when
  the machine is busy"): the race (below) requires enough *real* wall-clock elapsed time to cross a fixed
  0.53-second threshold, which a lightly-loaded machine reliably avoids and which repeating the single test in
  isolation (with no other concurrent load) does not reproduce either.

### Root cause (from code reading — confirms the task's own suspicion)

`ConsoleComposerBar` (`tldw_chatbook/Widgets/Console/console_composer_bar.py`) runs a **real background blink
timer**, not a simulated/virtual one:

- `CURSOR_BLINK_INTERVAL = 0.53` (line 75).
- `on_mount` (lines 595-604) creates the timer **paused**: `self._cursor_blink_timer = self.set_interval(0.53,
  self._toggle_cursor_blink, pause=True)`.
- `_sync_cursor_blink_state` (lines 584-593) is the only place that starts/stops it: `if
  self.has_focus_within: timer.resume() else: timer.pause()`.
- `on_focus` (lines 609-612) calls `_sync_cursor_blink_state()` — i.e. **the moment the composer gains focus, the
  real timer starts ticking every 0.53 real seconds**, each tick calling `_toggle_cursor_blink`
  (`self._cursor_visible = not self._cursor_visible`, lines 579-582) — **the exact same method the test calls
  manually** to drive its own deterministic before/after assertions.

Both cursor-blink tests do:
```python
composer.focus()
await pilot.pause(0.1)
...
composer._toggle_cursor_blink()      # manual call #1
assert ... CURSOR_GLYPH not in ...
composer._toggle_cursor_blink()      # manual call #2 (toggles test only)
assert ... CURSOR_GLYPH in ...
```
Nothing pauses or stops `composer._cursor_blink_timer` before or during this sequence. Under light load, the real
wall-clock time elapsed between `composer.focus()` (which starts the real timer) and the final manual-toggle
assertions is well under 0.53s, so the auto-timer has not yet fired and the test's own two manual toggles are the
only state changes — the assertions hold by luck of timing, not by design. `pilot.pause(delay)` in Textual does
not run on a mocked/virtual clock; the requested delay is a *minimum*, and the actual wall-clock time consumed
(including waiting for the message pump to go idle, layout, and OS scheduling) can be arbitrarily larger under
real CPU contention. If cumulative real elapsed time from `composer.focus()` onward crosses one or more full
0.53-second periods (plausible on "a busy machine," per the task's own framing, or during a large combined test
sweep with many concurrent/adjacent Textual apps churning), the real background timer fires at least once more
than the test accounts for, flipping `_cursor_visible` out from under the test via the identical
`_toggle_cursor_blink` method — desynchronizing the manual toggle count from the observed glyph-presence
assertions and producing an intermittent, load-correlated failure with no code defect involved (the blink
behavior itself is correct, intended UX).

### Classification

**Test bug** (a real, un-disabled background timer racing the test's own manual invocations of the same
state-mutating method, on real wall-clock time rather than a controlled/virtual clock). Not a product bug.

### Proposed fix

In both `test_console_composer_cursor_blink_toggles` (line 557-577) and
`test_console_composer_cursor_blink_keeps_row_count_stable_at_wrap_width` (line 581-618), pause the real blink
timer **immediately after focusing**, before any further `pilot.pause()` calls or manual toggles, so the automatic
timer can never fire during the rest of the test regardless of real elapsed wall-clock time:

```python
composer.focus()
composer._cursor_blink_timer.pause()   # deterministic: no more auto-ticks; only manual _toggle_cursor_blink() below can change _cursor_visible
await pilot.pause(0.1)
```

`Timer.pause()` in the installed Textual version (`textual.timer.Timer.pause`) is a plain synchronous method
(`self._active.clear()`; verified via `inspect.getsource`), not a coroutine, so no `await` is needed and this is a
drop-in one-line addition per test — it exactly mirrors the production code's own usage
(`console_composer_bar.py:591/593`, which also calls `timer.resume()`/`timer.pause()` unawaited). This removes
all real-clock dependency from both tests: the manual `_toggle_cursor_blink()` calls become the *only* source of
state changes, making the assertions deterministic on any machine regardless of load.

### Covering-test guidance

After the fix, to positively prove the fix actually closes the gap (rather than merely "still passes because it
never manifested"), an implementer can temporarily insert a real `await asyncio.sleep(0.6)` (one full blink
period plus margin) between `composer.focus()` and the manual toggle assertions in a scratch/local copy of the
**unfixed** test to confirm it now reliably fails (proving the race is real and reproducible on demand), then
confirm the same injected delay no longer causes a failure once `composer._cursor_blink_timer.pause()` is added
(proving the fix closes it). Do not commit the injected sleep — it is only a local verification aid. Re-run both
cursor-blink tests solo 20+ times and as part of the full file after the fix as a normal regression check.

---

## Summary table

| # | Test | Reproduced? | Classification | Fix location |
|---|------|-------------|-----------------|--------------|
| 1 | `test_console_native_generic_provider_send_renders_completed_message` (best match for filed path) | No (0/30 solo + 1 full-file run) | Unconfirmed / likely misfiled duplicate of #3 | None proposed against this test; re-file with full traceback if seen again |
| 2 | `test_import_real_superpowers_skills_lands_trust_pending` | No in this session (0/23 solo, 0/128, 0/676 combined); **yes historically** per `.superpowers/sdd/flakes-report.md` | Test bug: fixed-iteration poll (3.0s cap) too tight for real thread/loop/disk work (~3.5s/import measured) | `Tests/Skills/test_skills_import.py:89-110`, `_run_skills_import_via_ui` |
| 3 | `test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop` | **Yes** (~3/280 solo runs, ~1%) | Test/environment: tight 5s probe timeout + 5-connection server backlog vs. 120-connection barrier-synchronized hammer | `Tests/Chat/test_console_provider_gateway.py:1282-1291` (fixture) and `:1468-1538` (test) |
| 4 | `test_console_composer_cursor_blink_keeps_row_count_stable_at_wrap_width` | No (0/21 solo + 1 full-file run) — consistent with "load-flaky" framing | Test bug: real un-paused background blink timer races manual `_toggle_cursor_blink()` calls on real wall-clock time | `Tests/UI/test_console_internals_decomposition.py:557-577` and `:581-618` |
