# Windows-Safe POSIX Test Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the POSIX-specific media-signal and TTS file-lock tests collect truthfully on Windows so repository-wide pytest can advance beyond both current collection errors.

**Architecture:** Keep production code unchanged and gate tests at their real capability boundaries. The media module remains cross-platform and skips only real POSIX signal-delivery assertions, while the TTS module declares itself a POSIX `fcntl` contract suite and skips as a module when that standard-library capability is absent.

**Tech Stack:** Python 3.11+, pytest, Ruff, Backlog.md, Git/GitHub CLI

## Global Constraints

- Do not change production runtime behavior.
- Do not add dependencies, settings, schemas, migrations, or ADRs.
- Do not evaluate `SIGSTOP` or `SIGCONT` while pytest constructs parameters on an unsupported platform.
- Do not mutate Python's process-global `signal` module in tests.
- Treat the existing Windows collection exceptions as RED evidence; do not fabricate a failing product test for already-correct no-signal behavior.
- ADR required: no.
- ADR path: N/A.
- Reason: test-only portability correction preserving existing runtime boundaries.

---

### Task 1: Make media signal tests capability-safe

**Files:**
- Modify: `Tests/Media_Playback/test_player_pipeline.py:1-390`

**Interfaces:**
- Consumes: `PlayerPipeline.pause()`, `PlayerPipeline.resume()`, and their documented `hasattr(signal, ...)` degradation behavior.
- Produces: `_HAS_PROCESS_SUSPEND_SIGNALS: bool`, `requires_process_suspend_signals: pytest.MarkDecorator`, and collection-safe signal-name parameters.

- [ ] **Step 1: Preserve the Windows RED evidence**

Run:

```powershell
python -m pytest Tests/Media_Playback/test_player_pipeline.py --collect-only -q --basetemp=.pytest-tmp-task15742-media-red
```

Expected before the change: collection errors while evaluating
`pp.signal.SIGSTOP` in the parameter list.

- [ ] **Step 2: Add the capability marker without evaluating signal constants**

Add after imports:

```python
from types import SimpleNamespace

_HAS_PROCESS_SUSPEND_SIGNALS = all(
    hasattr(pp.signal, name) for name in ("SIGSTOP", "SIGCONT")
)
requires_process_suspend_signals = pytest.mark.skipif(
    not _HAS_PROCESS_SUSPEND_SIGNALS,
    reason="POSIX process suspend/resume signals are unavailable",
)
```

Apply `@requires_process_suspend_signals` to both tests that assert real signal
delivery. Change the concurrent parameterization to portable names:

```python
@pytest.mark.parametrize(
    ("action", "signal_name"),
    [("pause", "SIGSTOP"), ("resume", "SIGCONT")],
)
def test_pause_and_resume_signal_only_captured_run_during_restart(
    monkeypatch, action, signal_name
):
    expected_signal = getattr(pp.signal, signal_name)
```

- [ ] **Step 3: Add platform-neutral no-signal characterization**

Add this test beside the POSIX signal tests:

```python
def test_pause_resume_without_process_signals_updates_clock_without_kill(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(pp.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pp, "signal", SimpleNamespace())
    monkeypatch.setattr(
        pp.os,
        "kill",
        lambda *_args: pytest.fail("non-POSIX pause/resume must not signal"),
    )
    pipeline = pp.PlayerPipeline(
        "silent.mp4", _probe(has_audio=False), spawn=_SpawnRecorder()
    )
    run = pipeline.start()

    pipeline.pause()
    assert run.pause_started == 100.0
    now[0] = 103.0
    pipeline.resume()

    assert run.pause_started is None
    assert run.paused_total == pytest.approx(3.0)
    pipeline.stop()
```

This is characterization coverage. The collection exception from Step 1 is the
genuine RED for this test-only repair.

- [ ] **Step 4: Verify media collection and behavior GREEN**

Run:

```powershell
python -m pytest Tests/Media_Playback/test_player_pipeline.py --collect-only -q --basetemp=.pytest-tmp-task15742-media-collect
python -m pytest Tests/Media_Playback/test_player_pipeline.py -q --basetemp=.pytest-tmp-task15742-media-green
python -m ruff check Tests/Media_Playback/test_player_pipeline.py
python -m py_compile Tests/Media_Playback/test_player_pipeline.py
```

Expected: collection succeeds; all supported tests pass; only the two real
signal-delivery tests skip on Windows.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- Tests/Media_Playback/test_player_pipeline.py
git commit -m "test(media): gate POSIX playback signals"
```

---

### Task 2: Skip the POSIX TTS materialization suite when fcntl is absent

**Files:**
- Modify: `Tests/TTS/test_profile_reference_materialization.py:1-25`

**Interfaces:**
- Consumes: the production module's existing `_POSIX_SUPPORTED` and bounded `unsupported` outcome.
- Produces: an explicit module-level pytest skip on platforms without `fcntl`.

- [ ] **Step 1: Preserve the Windows RED evidence**

Run:

```powershell
python -m pytest Tests/TTS/test_profile_reference_materialization.py --collect-only -q --basetemp=.pytest-tmp-task15742-tts-red
```

Expected before the change: `ModuleNotFoundError: No module named 'fcntl'`.

- [ ] **Step 2: Replace the unconditional standard-library import with an explicit suite capability gate**

Remove `import fcntl`. Keep `import pytest` in the normal third-party import
group and, after all production imports, add:

```python
fcntl = pytest.importorskip(
    "fcntl",
    reason="POSIX profile-reference materialization contracts require fcntl",
)
```

Placing the capability call after imports keeps Ruff import ordering clean and
still allows the production module itself to prove import-safe on Windows.

- [ ] **Step 3: Verify TTS collection and module outcome GREEN**

Run:

```powershell
python -m pytest Tests/TTS/test_profile_reference_materialization.py --collect-only -q --basetemp=.pytest-tmp-task15742-tts-collect
python -m pytest Tests/TTS/test_profile_reference_materialization.py -q --basetemp=.pytest-tmp-task15742-tts-green
python -m ruff check Tests/TTS/test_profile_reference_materialization.py
python -m py_compile Tests/TTS/test_profile_reference_materialization.py
```

Expected on Windows: no collection error and one clear module skip citing
`fcntl`. Expected on POSIX: the existing suite collects and runs unchanged.

- [ ] **Step 4: Commit Task 2**

```powershell
git add -- Tests/TTS/test_profile_reference_materialization.py
git commit -m "test(tts): gate POSIX materialization contracts"
```

---

### Task 3: Verify globally, close TASK-15742, and create the PR

**Files:**
- Modify: `backlog/tasks/task-15742 - Make-POSIX-only-tests-collect-safely-on-Windows.md`

**Interfaces:**
- Consumes: the two collection-safe test modules from Tasks 1 and 2.
- Produces: verified Backlog closeout and a GitHub pull request targeting `dev`.

- [ ] **Step 1: Prove repository-wide collection advances past both former errors**

Run with the repository virtualenv and a worktree-local temp root:

```powershell
python -m pytest --collect-only -q --basetemp=.pytest-tmp-task15742-global-collect
```

Inspect the complete output and confirm neither `SIGSTOP` nor `fcntl` appears as
a collection error. Record any later unrelated collection failures separately.

- [ ] **Step 2: Start the repository-wide suite and run the affected focused matrix**

Run:

```powershell
python -m pytest -q --basetemp=.pytest-tmp-task15742-global
python -m pytest Tests/Media_Playback/test_player_pipeline.py Tests/TTS/test_profile_reference_materialization.py -q --basetemp=.pytest-tmp-task15742-focused
```

Stop the global command after five minutes without useful progress. Do not call
unrelated failures a TASK-15742 regression; report exact counts and provenance.

- [ ] **Step 3: Run final static and diff gates**

Run:

```powershell
python -m ruff check Tests/Media_Playback/test_player_pipeline.py Tests/TTS/test_profile_reference_materialization.py
python -m py_compile Tests/Media_Playback/test_player_pipeline.py Tests/TTS/test_profile_reference_materialization.py
git diff --check origin/dev...HEAD
git status --short
```

- [ ] **Step 4: Complete Backlog hygiene**

Check all six acceptance criteria only when supported by evidence. Add concise
Implementation Notes containing the root causes, capability gates, exact test
results, global collection outcome, and ADR disposition. Set TASK-15742 to Done
with the Backlog CLI and verify `backlog task 15742 --plain` preserved the notes.

- [ ] **Step 5: Commit closeout documentation**

```powershell
git add -- "backlog/tasks/task-15742 - Make-POSIX-only-tests-collect-safely-on-Windows.md"
git commit -m "docs: close Windows test collection task"
```

- [ ] **Step 6: Reconcile with current dev and reverify**

Fetch `origin/dev`, rebase the feature branch when needed, resolve only genuine
overlaps, then rerun the focused matrix, Ruff, compilation, and diff check on
the rebased HEAD.

- [ ] **Step 7: Push and create the pull request**

Push `codex/task-208-ingest-active-dedup`, create a ready pull request against
`dev` using GitHub CLI and the repository PR template, and report the PR URL.
