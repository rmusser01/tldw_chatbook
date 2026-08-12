# TASK-3401.18 Generated-Video Playback Crash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make inline and modal generated-video playback degrade safely instead of blocking or terminating the Textual application.

**Architecture:** Keep the existing PyAV inline decoder and ffmpeg/ffplay modal pipeline. Move blocking activation and teardown into workers, marshal every UI mutation through `App.call_from_thread`, and attach decoder/process state to immutable generation identities so stale work cannot affect a replacement playback run. Use only existing modules and standard-library synchronization primitives; add no dependency or playback coordinator.

**Tech Stack:** Python 3.11, Textual 8 workers/Pilot, PyAV, ffmpeg/ffplay subprocesses, pytest, Loguru, Ruff.

---

## Scope and file ownership

Modify only these production files:

- `tldw_chatbook/Media_Playback/frame_source.py` — propagate decoder/native-close failures to the owning worker.
- `tldw_chatbook/Media_Playback/player_pipeline.py` — transactional subprocess startup/teardown and per-generation run state.
- `tldw_chatbook/Widgets/Console/console_video_preview.py` — worker-owned inline activation/decode/cleanup and generation-gated UI callbacks.
- `tldw_chatbook/UI/Screens/video_player_screen.py` — worker-owned modal activation/seek/teardown and pipeline/run-gated UI callbacks.
- `tldw_chatbook/UI/Console_Modules/agent.py` — closeout-only removal of the pre-existing controller bridge wrapper exposed by the required repo-wide guard.

Extend only these directly related tests:

- `Tests/Media_Playback/test_frame_source.py`
- `Tests/Media_Playback/test_player_pipeline.py`
- `Tests/Media_Playback/test_player_integration.py`
- `Tests/Widgets/test_console_video_preview.py`
- `Tests/Media_Playback/test_player_screen.py`
- `Tests/test_call_from_thread_guard.py` is verification-only; do not modify it. Verify the controller through its canonical owning-App bridge and the affected run-log rail regression instead of weakening the guard.

Update these task artifacts only at closeout:

- `backlog/tasks/task-3401.18 - Prevent-generated-video-preview-from-crashing-Console.md`
- `Docs/superpowers/plans/2026-08-09-task-3401-18-video-preview-crash.md` (check completed steps)

Do not change storage, generation adapters, workflow JSON, Console message identity, render-mode policy, CSS, keybindings, or dependencies. Do not run a full test collection or broad UI suite.

## Shared testing rules

- Use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B` so RED/mutation runs cannot reuse stale bytecode.
- Mounted GREEN-path regressions must use a real Textual `App.run_test()`, real worker execution, and real `App.call_from_thread`. Patch only the decoder/pipeline process seams.
- Use `threading.Event` barriers and observed state, never sleeps, to prove blocked open/read/seek/stop behavior and stale-generation interleavings.
- Capture Loguru with `logger.add(lambda message: records.append(str(message)))`; `capsys` is not evidence for this logger.
- Sensitive sentinels must be absent from logs and user-visible copy. Assert only `component`, `phase`, and exception class.
- After every load-bearing test is GREEN, temporarily remove the exact bridge/identity/cleanup guard and rerun that test to observe the intended failure, then restore with `apply_patch` (never `git checkout --`).

### Task 1: Make the real PyAV source report failures to its owner

**Files:**

- Modify: `tldw_chatbook/Media_Playback/frame_source.py:121-161`
- Test: `Tests/Media_Playback/test_frame_source.py`

- [x] **Step 1: Add RED tests for real iterator and close failures**

Add small fake container/stream objects around an `AvFrameSource` instance so the production `iter_frames()` and `close()` methods run without constructing a fake source API:

```python
def _opened_source(container, stream=object()):
    source = AvFrameSource("private-name.mp4")
    source._container = container
    source._stream = stream
    source._opened = True
    return source


def test_iter_frames_propagates_decoder_failure_without_logging_payload():
    secret = "PRIVATE-DECODE-SENTINEL"

    class Container:
        def decode(self, *, video):
            raise RuntimeError(secret)

    source = _opened_source(Container())
    with pytest.raises(RuntimeError, match=secret):
        list(source.iter_frames())


def test_close_clears_references_before_propagating_native_failure():
    source = None

    class Container:
        def close(self):
            assert source._container is None
            assert source._stream is None
            assert source._opened is False
            raise RuntimeError("PRIVATE-CLOSE-SENTINEL")

    source = _opened_source(Container())
    with pytest.raises(RuntimeError, match="PRIVATE-CLOSE-SENTINEL"):
        source.close()
    source.close()  # references were cleared, so retry is a no-op
```

Install a temporary Loguru sink around each test and assert the sentinel never appears. Keep the existing real-clip tests unchanged.

- [x] **Step 2: Run the focused RED command**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Media_Playback/test_frame_source.py -q
```

Expected: the new decode-propagation and close-propagation tests fail because both exceptions are currently swallowed.

- [x] **Step 3: Implement the minimal source-boundary correction**

Keep `GeneratorExit` normal and re-raise it, but remove the broad decode catch/log entirely. Detach state before native close:

```python
def close(self) -> None:
    container = self._container
    self._container = None
    self._stream = None
    self._opened = False
    if container is not None:
        container.close()
```

No new logger helper belongs in this file: the worker owner has the component/phase context and owns sanitized diagnostics.

- [x] **Step 4: Run GREEN and mutation proof**

Run the same focused command and expect all tests in `test_frame_source.py` to pass. Temporarily restore the old broad catches one at a time; each corresponding test must fail for the intended reason. Restore the implementation and rerun GREEN.

- [x] **Step 5: Commit Task 1**

```bash
git add tldw_chatbook/Media_Playback/frame_source.py Tests/Media_Playback/test_frame_source.py
git diff --cached --check
git commit -m "fix: surface video frame source failures"
```

### Task 2: Make PlayerPipeline runs isolated and teardown transactional

**Files:**

- Modify: `tldw_chatbook/Media_Playback/player_pipeline.py:139-370`
- Test: `Tests/Media_Playback/test_player_pipeline.py`
- Test: `Tests/Media_Playback/test_player_integration.py`

- [x] **Step 1: Add RED tests for startup/teardown resource ownership**

Extend `_FakeProc` and fake stdout with terminate/wait/kill/close counters. Use real `os.pipe()` descriptors for first-spawn and second-spawn failure tests, and assert closure with `os.fstat(fd)` raising `OSError`.

Cover these exact branches:

```python
def test_silent_start_never_allocates_audio_pipe(...): ...
def test_first_spawn_failure_closes_every_parent_fd(...): ...
def test_second_spawn_failure_stops_ffmpeg_and_closes_stdout_and_fds(...): ...
def test_stop_waits_after_forced_kill(...): ...
def test_stop_is_idempotent_and_stdout_closes_once(...): ...
```

The forced-timeout fake must record `terminate -> wait(timeout=2) -> kill -> wait()` in order.

- [x] **Step 2: Add RED tests for lazy-generator and stale-run isolation**

Pin the public shape the screen will use:

```python
run = pipeline.start(offset_seconds=1.0)
iterator = pipeline.iter_frames(run)
replacement = pipeline.seek(2.0)
```

Use separate identifiable stdout objects and barriers to prove:

- an old iterator first advanced after restart reads only `run.stdout`, never `replacement.stdout`;
- old frame/EOF completion changes only `run.frame_index`, `run.eof`, and `run.stats`;
- `replacement` remains non-EOF and its first PTS is `2.0` at frame index zero;
- `frame_due`, `frames_behind`, `note_rendered`, and `note_dropped` receive the originating run and never mutate the current run implicitly;
- natural EOF, successful stop, restart, and a captured-but-never-advanced iterator close the old stdout exactly once.

- [x] **Step 3: Run the focused pipeline RED command**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Media_Playback/test_player_pipeline.py Tests/Media_Playback/test_player_integration.py -q
```

Expected: new tests fail because startup is not exception-safe, silent playback always allocates a pipe, process kill is not reaped, and state is pipeline-global.

- [x] **Step 4: Add one private per-generation run object**

Keep it in `player_pipeline.py`; do not create a new abstraction module:

```python
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class PlayerRun:
    generation: int
    stdout: Any | None
    offset_seconds: float
    started_wall: float | None = None
    pause_started: float | None = None
    paused_total: float = 0.0
    frame_index: int = 0
    eof: bool = False
    stats: SyncStats = field(default_factory=SyncStats)
    _stdout_lock: Lock = field(default_factory=Lock, repr=False)
    _stdout_closed: bool = field(default=False, repr=False)

    def close_stdout_once(self) -> None:
        with self._stdout_lock:
            if self._stdout_closed:
                return
            self._stdout_closed = True
            stdout, self.stdout = self.stdout, None
        if stdout is not None:
            stdout.close()
```

`PlayerPipeline.start()` and `seek()` return `PlayerRun`. The pump must pass that exact object into these generation-aware APIs:

```python
def iter_frames(self, run: PlayerRun) -> Iterator[tuple[float, bytes]]: ...
def sync_clock(self, run: PlayerRun) -> float: ...
def frame_due(self, run: PlayerRun, pts: float) -> bool: ...
def frames_behind(self, run: PlayerRun, pts: float) -> bool: ...
def note_rendered(self, run: PlayerRun, pts: float) -> None: ...
def note_dropped(self, run: PlayerRun, pts: float) -> None: ...
```

`iter_frames(run)` captures `stdout = run.stdout` and uses only fields on `run`; its `finally` calls `run.close_stdout_once()`. It never reads the pipeline's current run after generator creation. Keep a read-only `current_run` property and a compatibility `stats` property returning current run stats for unchanged consumers.

- [x] **Step 5: Make startup and process teardown transactional**

Use local process/fd variables until both children start. Allocate `os.pipe()` only when `probe.has_audio`. In one `try/finally`, close every parent fd still owned by the parent. On failure, terminate/reap any local child and close the private run's stdout before re-raising; publish `_ffmpeg`, `_ffplay`, and `_run` only after success.

Use one small private helper for process reaping:

```python
def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        finally:
            proc.wait()
```

`stop()` detaches the process fields, marks the captured run EOF, stops/reaps both children, and calls `run.close_stdout_once()` in `finally`. It must not mutate a replacement run installed later.

- [x] **Step 6: Update existing pipeline/integration callers**

Change existing tests and `test_player_integration.py` to retain the returned run and call `iter_frames(run)`. Preserve the observable PTS, A/V, reconnect, pause/resume, seek, and tool-guidance behavior.

- [x] **Step 7: Run GREEN and the required race mutations**

Run the Task 2 focused command. Then separately mutate:

- `iter_frames(run)` back to `self._run.stdout`;
- one timing/stat helper back to current-run state;
- `close_stdout_once()` out of stop;
- the final `wait()` after kill.

Each load-bearing new test must fail specifically. Restore, rerun GREEN, and run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Media_Playback/test_player_integration.py -q
```

The integration file may skip only for its existing missing-tool condition.

- [x] **Step 8: Commit Task 2**

```bash
git add tldw_chatbook/Media_Playback/player_pipeline.py Tests/Media_Playback/test_player_pipeline.py Tests/Media_Playback/test_player_integration.py
git diff --cached --check
git commit -m "fix: isolate generated video playback runs"
```

### Task 3: Move inline preview activation and cleanup into its decode worker

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_video_preview.py:72-269`
- Test: `Tests/Widgets/test_console_video_preview.py`
- Verify: `Tests/test_call_from_thread_guard.py`

- [x] **Step 1: Replace vacuous worker stubs with mounted RED regressions**

Keep pure progress/card tests. Add a tiny `App` harness that composes a `ConsoleVideoPreview` plus a second control/state marker. Do not monkeypatch `run_worker` or the GREEN-path `App.call_from_thread`.

Use a fake source factory with `threading.Event` barriers and record thread ids. Add mounted tests for:

- click starts a real thread worker, renders one deterministic PIL frame, reaches EOF/paused, and leaves the app responsive with no worker error;
- blocked constructor/check/probe does not block `pilot.click()` or a second UI action;
- pause during a blocked decoder returns immediately; source closes on the decoder worker, not the UI thread;
- pause/resume installs a new generation; released old frame/EOF/finally cannot pause or overwrite the new run;
- immediate EOF leaves no timer, source, or `_active` owner;
- unmount ignores late frames/EOF and still lets the worker close its source;
- open, decode, render, timer-stop, and source-close failures are contained and log only component/phase/type;
- a rejected bridge is attempted once and has no direct worker-thread UI fallback.

Use `await app.workers.wait_for_complete()` only after releasing every barrier. Assert no worker has `WorkerState.ERROR`.

- [x] **Step 2: Run the focused inline RED command**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Widgets/test_console_video_preview.py Tests/test_call_from_thread_guard.py -q
```

Expected: mounted activation fails on `self.call_from_thread`, blocked probe freezes the old path, and stale/cleanup/privacy tests fail.

- [x] **Step 3: Add one private preview-run identity**

Keep it in the widget module:

```python
@dataclass(frozen=True)
class _PreviewRun:
    generation: int
    cancelled: Event
```

Store `self._run: _PreviewRun | None` and increment a generation for every Play. `play()` performs only UI state changes, one-active ownership, and worker dispatch via `partial`; it does not instantiate or probe `AvFrameSource`.

- [x] **Step 4: Implement worker-owned activation/decode/close**

The worker creates and probes its private source, then calls:

```python
accepted = self.app.call_from_thread(self._accept_source, run, source)
```

`_accept_source` checks current run/mount/state, attaches the source, and installs the off-screen timer before returning `True`. Only then may decode start. Every frame/EOF/failure callback carries both `run` and `source`; UI handlers reject mismatches before mutation.

Pause/unmount must:

1. invalidate/detach the current run/source;
2. set the run cancellation event;
3. clear `_active`;
4. stop/clear the timer inside an independent guarded cleanup step;
5. never call `source.close()` on the UI thread.

The worker closes its own source in `finally`. An unexpected close error is logged as `component=inline_preview phase=cleanup error_type=<type>` without exception text.

- [x] **Step 5: Contain UI/render and bridge failures**

Add one local sanitized logger helper and one bridge helper. The bridge helper attempts `app.call_from_thread` once and returns false on refusal; it never retries or calls the UI callback directly. `_show_frame` catches conversion/update errors, transitions the matching run to unavailable guidance, signals cancellation, and emits `phase=render` without raw exception text.

Keep activation/decode workers at Textual's default `exit_on_error=True`; the outer worker boundary must catch every expected failure so the mounted regression proves no exception escapes rather than relying on worker suppression.

- [x] **Step 6: Run GREEN and mutation proofs**

Run the Task 3 focused command. Then remove each of these in turn: `.app` from the bridge, the run/source identity check, timer-before-decode ordering, and worker-owned source close. The named mounted test must fail for each mutation. Restore and rerun GREEN.

- [x] **Step 7: Commit Task 3**

```bash
git add tldw_chatbook/Widgets/Console/console_video_preview.py Tests/Widgets/test_console_video_preview.py
git diff --cached --check
git commit -m "fix: contain inline video preview workers"
```

### Task 4: Make modal activation, seek, pump, and teardown generation-safe

**Files:**

- Modify: `tldw_chatbook/UI/Screens/video_player_screen.py:57-297`
- Test: `Tests/Media_Playback/test_player_screen.py`
- Verify: `Tests/test_call_from_thread_guard.py`

- [x] **Step 1: Add mounted modal RED tests**

Mount `VideoPlayerScreen` in a small real Textual app. Patch `playback_tools_available`, `probe_file`, and `PlayerPipeline` only. Use a deterministic fake pipeline exposing the Task 2 `PlayerRun` API and barriers.

Add tests for:

- blocked probe/start leaves the event loop responsive and closing the modal cleans up a subsequently started private pipeline;
- a real pump renders a frame and EOF status through `App.call_from_thread` with no worker error;
- seek is single-flight and non-blocking; successful completion attaches its returned run and starts a replacement pump;
- old queued frame/EOF/failure callbacks after seek or unmount are ignored;
- an old iterator fails after seek, yet the replacement pump still renders;
- blocked stop/reap after unmount does not block another app action and completes after release;
- partial start and mid-pump failures stop/reap, notify generically, and dismiss;
- render/dispatch/cleanup failures contain sensitive sentinels and do not escape.

- [x] **Step 2: Run the focused modal RED command**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Media_Playback/test_player_screen.py Tests/test_call_from_thread_guard.py -q
```

Expected: activation/seek/stop block the UI, worker callbacks use the nonexistent screen bridge, and stale/replacement/error tests fail.

- [x] **Step 3: Split modal work into activation, pump, and lifecycle workers**

Use only screen fields—no new coordinator:

```python
self._activation_token = 0
self._pipeline: PlayerPipeline | None = None
self._run: PlayerRun | None = None
self._seek_in_flight = False
```

`on_mount()` performs the cheap tool check and one-active pause, increments the token, and launches background probe/start. The worker owns a private pipeline until `_accept_activation(token, pipeline, run, probe)` accepts it on the UI thread; rejection or bridge refusal stops/reaps it in the same worker.

The pump receives `(token, pipeline, run)` explicitly. It iterates only `pipeline.iter_frames(run)`, calls only run-aware timing/stat methods, and queues frame/EOF/failure callbacks through `self.app.call_from_thread`. UI callbacks require the same token, pipeline identity, and run identity.

- [x] **Step 4: Make seek and unmount non-blocking**

Seek immediately rejects repeats while `_seek_in_flight`, invalidates/detaches the old UI callback identity, and starts one lifecycle worker. That worker calls `pipeline.seek(target)` and publishes the returned run only if its token is current. Acceptance starts a new pump before clearing the single-flight flag. A stale completion stops the pipeline.

Unmount invalidates/detaches state and asks the owning `App` to run a best-effort thread cleanup (`exit_on_error=False`) so screen removal cannot cancel process reaping. The cleanup body catches/logs its own failure. UI callbacks and dismissal/notification cleanup must be independently guarded.

- [x] **Step 5: Contain current-run pump/render failures**

Wrap the whole pump, including iterator, clock, stats, dispatch, and EOF, in one boundary. A current-run failure crosses the app bridge to a UI handler that invalidates playback, schedules stop/reap, shows generic recovery guidance, and dismisses. A stale failure only emits a sanitized record and returns. Render conversion/widget-update failures follow the same current-run handler with `phase=render`.

The bridge refusal path logs once and returns—no worker-thread fallback. Keep activation and pump workers at `exit_on_error=True`; correctness comes from containment, not suppression.

- [x] **Step 6: Run GREEN and mutation proofs**

Run the Task 4 focused command. Mutate away the app bridge, UI-side identity check, replacement-pump launch, single-flight guard, and app-owned unmount cleanup one at a time. Each dedicated test must fail. Restore and rerun GREEN.

- [x] **Step 7: Commit Task 4**

```bash
git add tldw_chatbook/UI/Screens/video_player_screen.py Tests/Media_Playback/test_player_screen.py
git diff --cached --check
git commit -m "fix: contain modal video player workers"
```

### Task 5: Focused verification, documentation, and task closeout

**Files:**

- Modify: `backlog/tasks/task-3401.18 - Prevent-generated-video-preview-from-crashing-Console.md`
- Modify: `Docs/superpowers/plans/2026-08-09-task-3401-18-video-preview-crash.md`

- [x] **Step 1: Run the complete touched-file test gate only**

Run exactly:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Media_Playback/test_frame_source.py \
  Tests/Media_Playback/test_player_pipeline.py \
  Tests/Media_Playback/test_player_integration.py \
  Tests/Widgets/test_console_video_preview.py \
  Tests/Media_Playback/test_player_screen.py \
  Tests/test_call_from_thread_guard.py \
  Tests/UI/test_console_agent_rail.py::test_view_full_log_loads_off_thread_then_opens_the_modal \
  -q
```

Do not add `Tests/`, `Tests/UI`, `Tests/Media_Playback`, RuntimePolicy, or the full repository suite.

- [x] **Step 2: Run static checks only on touched Python files**

Run Ruff on the five production files and five modified test files:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Media_Playback/frame_source.py \
  tldw_chatbook/Media_Playback/player_pipeline.py \
  tldw_chatbook/Widgets/Console/console_video_preview.py \
  tldw_chatbook/UI/Screens/video_player_screen.py \
  tldw_chatbook/UI/Console_Modules/agent.py \
  Tests/Media_Playback/test_frame_source.py \
  Tests/Media_Playback/test_player_pipeline.py \
  Tests/Media_Playback/test_player_integration.py \
  Tests/Widgets/test_console_video_preview.py \
  Tests/Media_Playback/test_player_screen.py
```

Run `ruff format --check` over the same ten files. Run `py_compile` for the five production files to a `TemporaryDirectory`/explicit `cfile` outputs so no repository `__pycache__` is created. Run `git diff --check` and inspect `git status --short`.

- [x] **Step 3: Perform the final privacy/resource self-review**

Confirm from the diff:

- no exception value, source path, filename, prompt, URL, traceback locals, or media bytes enter playback logs/copy;
- no direct `self.call_from_thread` remains in either touched UI file;
- every subprocess spawn/fd/stdout has a deterministic cleanup owner;
- every native PyAV close occurs on its decoder worker;
- every stale callback checks generation plus source/pipeline/run identity;
- no production/test file outside this plan changed.

- [x] **Step 4: Update task notes before Done**

Use the Backlog CLI to check ACs #1-#7 and set final notes, then inspect the resulting diff because `--notes` replaces the notes block. Record:

- approach and per-generation ownership;
- exact focused test count/command and any optional integration skips;
- exact Ruff/py_compile scopes;
- mutation evidence;
- ADR-044/no-new-ADR decision;
- live H3 generation was not repeated and no media was retained;
- any deviation from this plan.

The closeout-only deviation was the canonical Console agent bridge cleanup required to make the existing repo-wide guard honest after its safe-but-noncanonical controller wrapper was exposed. It changed no playback or agent behavior and was verified by the guard plus the single affected run-log rail regression.

Only then set TASK-3401.18 to Done via CLI.

- [x] **Step 5: Commit closeout artifacts**

```bash
git add "backlog/tasks/task-3401.18 - Prevent-generated-video-preview-from-crashing-Console.md" Docs/superpowers/plans/2026-08-09-task-3401-18-video-preview-crash.md
git diff --cached --check
git commit -m "docs: close generated-video playback crash task"
```

- [x] **Step 6: Final branch review and PR update**

Dispatch one final whole-change spec/correctness review over the TASK-3401.18 commit range. Fix and re-review every Critical/Important/Minor finding with the same touched-file-only gates. Then push the current branch and update draft PR #1460; do not mark it ready or merge without the user-requested final review checkpoint.
