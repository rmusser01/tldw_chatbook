# TASK-3401.19 Video Store Startup Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run generated-video retention exactly once per `TldwCli` startup so Console navigation and screen recreation cannot delete current-run video bytes.

**Architecture:** `TldwCli` will create one ordinary `VideoStore` immediately after loading effective configuration, synchronously run the existing retention sweep, and retain that object for the app lifetime. `ChatScreen` will only return an explicit test override or borrow the app-owned store; it will never construct or clean a store. Existing `VideoStore` policy code remains unchanged, and focused production-app tests use real temporary files to prove session, TTL, navigation, restart, and failure behavior.

**Tech Stack:** Python 3.11, Textual 8, pytest, Loguru, existing `VideoStore` and `VideoGenerationMetadata` contracts.

---

## File map

- Modify `tldw_chatbook/app.py`: create and retain the app-owned store after config load; contain one startup sweep failure with bounded diagnostics.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: replace screen-owned construction/cleanup with explicit-override-or-app-owner lookup.
- Modify `Tests/UI/app_factory.py`: isolate every shared test-app construction from the developer's real generated-video directory before `TldwCli.__init__` runs.
- Modify `Tests/ProductionApp/test_chat_composition_retirement.py`: prove real app startup, navigation, restart, TTL, and failure containment using a temporary `VideoStore` root.
- Modify `Tests/Chat/test_console_video_message.py`: lock the narrow `ChatScreen` ownership boundary, explicit test override, and fail-loud missing-owner behavior.
- Modify `backlog/tasks/task-3401.19 - Run-session-video-retention-cleanup-only-at-app-startup.md`: record this plan, verification evidence, ADR-044 reuse, and completion notes.
- Keep `tldw_chatbook/Video_Generation/video_store.py` unchanged. TASK-3401.20 separately owns post-save size-cap enforcement.

## Test-scope constraint

Run only tests related to the touched files, as explicitly requested by the user. Do not run the full suite, broad `Tests/ProductionApp`, broad `Tests/Chat`, or RuntimePolicy collections. The final automated gate is exactly:

```bash
.venv/bin/python -B -m pytest \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/Chat/test_console_video_message.py -q
```

### Task 1: Isolate every touched test-app constructor

**Files:**
- Modify: `Tests/UI/app_factory.py`
- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`

- [ ] **Step 1: Make the shared factory patch the real VideoStore module before construction**

In `Tests/UI/app_factory.py`, add both patches to the existing `ExitStack` used around `TldwCli()`:

```python
patch(
    "tldw_chatbook.Video_Generation.video_store.get_user_data_dir",
    return_value=user_data_dir,
),
patch(
    "tldw_chatbook.Video_Generation.video_store.get_video_generation_config",
    return_value=SimpleNamespace(
        retention="session",
        retention_ttl_hours=24,
        max_store_mb=2048,
    ),
),
```

This is safety infrastructure, not product behavior: after startup retention moves into `TldwCli.__init__`, every one of the shared factory's callers must point at its already-tracked temporary user-data directory before constructing the app.

- [ ] **Step 2: Add module-wide ProductionApp isolation**

In `Tests/ProductionApp/test_chat_composition_retirement.py`, import `SimpleNamespace` and add this helper before the fixture:

```python
def _video_retention_config(**overrides):
    values = {
        "retention": "session",
        "retention_ttl_hours": 24,
        "max_store_mb": 2048,
    }
    values.update(overrides)
    return SimpleNamespace(**values)
```

Then add an autouse fixture so every existing and new `_production_app()` call is safe, not only tests that remember to call a helper:

```python
@pytest.fixture(autouse=True)
def isolated_video_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from tldw_chatbook.Video_Generation import video_store as video_store_module

    config = _video_retention_config()
    data_root = tmp_path / "profile"
    monkeypatch.setattr(video_store_module, "get_user_data_dir", lambda: data_root)
    monkeypatch.setattr(
        video_store_module,
        "get_video_generation_config",
        lambda: config,
    )
    return data_root / "generated_videos", config
```

Tests that need TTL may replace only `video_store_module.get_video_generation_config` inside their own body. No `TldwCli` in either touched test module may be constructed before temporary-root isolation is active.

- [ ] **Step 3: Commit the test safety boundary**

```bash
git add Tests/UI/app_factory.py Tests/ProductionApp/test_chat_composition_retirement.py
git diff --cached --check
git commit -m "test: isolate video retention app construction"
```

### Task 2: Prove and implement app-owned startup retention

**Files:**
- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`
- Modify: `tldw_chatbook/app.py`

- [ ] **Step 1: Add real-store startup test imports**

Add imports for `os`, Loguru's `logger`, `ConsoleChatMessage`, `ConsoleMessageRole`, `VideoGenerationMetadata`, and `VideoStore`. Reuse `_video_retention_config` and the autouse fixture committed in Task 1.

Use the root returned by the autouse `isolated_video_store` fixture only to seed pre-startup bytes. The production `TldwCli` must still construct its store through the normal no-argument path.

- [ ] **Step 2: Write failing startup/restart/TTL/failure tests**

Add these focused tests:

```python
def test_video_retention_runs_during_app_construction_once(
    monkeypatch, isolated_video_store
):
    root, config = isolated_video_store
    prior = VideoStore(root=root, config=config).save("prior", "clip", b"prior")
    calls = 0
    real_enforce = VideoStore.enforce_retention

    def count_enforce(self, **kwargs):
        nonlocal calls
        calls += 1
        return real_enforce(self, **kwargs)

    monkeypatch.setattr(VideoStore, "enforce_retention", count_enforce)
    app = _production_app(monkeypatch)

    assert calls == 1
    assert not prior.exists()
    assert app.generated_video_store.root == root
```

```python
def test_next_app_startup_applies_session_retention_again(
    monkeypatch, isolated_video_store
):
    root, _config = isolated_video_store
    first_app = _production_app(monkeypatch)
    current = first_app.generated_video_store.save("current", "clip", b"current")
    metadata = VideoGenerationMetadata(
        name="clip",
        prompt="current run",
        backend="comfyui",
    )
    message = ConsoleChatMessage(
        id="current",
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] clip",
        video_metadata=metadata,
    )
    assert current.exists()

    second_app = _production_app(monkeypatch)
    second_screen = ChatScreen(second_app)

    assert second_app.generated_video_store is not first_app.generated_video_store
    assert not current.exists()
    assert second_screen._build_video_card_specs([message])[message.id].status == "expired"
```

```python
def test_next_app_startup_keeps_fresh_ttl_video_and_removes_stale(
    monkeypatch, isolated_video_store
):
    from tldw_chatbook.Video_Generation import video_store as video_store_module

    config = _video_retention_config(retention="ttl", retention_ttl_hours=1)
    monkeypatch.setattr(
        video_store_module,
        "get_video_generation_config",
        lambda: config,
    )
    first_app = _production_app(monkeypatch)
    fresh = first_app.generated_video_store.save("fresh", "clip", b"fresh")
    stale = first_app.generated_video_store.save("stale", "clip", b"stale")
    old = stale.stat().st_mtime - 3700
    os.utime(stale, (old, old))

    second_app = _production_app(monkeypatch)

    assert fresh.exists()
    assert not stale.exists()
    assert second_app.generated_video_store.resolve("fresh", "clip") == fresh
```

For failure containment, add this complete named test. It makes the exception text contain distinct private path, message-id, and media-name sentinels, attaches a temporary Loguru sink with `format="{message}"`, and proves construction retains the store without leaking exception text:

```python
def test_video_retention_startup_failure_is_bounded(monkeypatch):
    private_values = (
        "/private/generated/video.mp4",
        "private-message-id",
        "private-media-name",
    )
    calls = 0

    def fail_retention(self, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(" | ".join(private_values))

    monkeypatch.setattr(VideoStore, "enforce_retention", fail_retention)
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="WARNING", format="{message}")
    try:
        app = _production_app(monkeypatch)
    finally:
        logger.remove(sink_id)

    assert calls == 1
    assert isinstance(app.generated_video_store, VideoStore)
    rendered = "\n".join(captured)
    assert "error_type=RuntimeError" in rendered
    assert all(value not in rendered for value in private_values)
```

Always remove the temporary Loguru sink in `finally`.

- [ ] **Step 3: Run the new startup tests to verify RED**

Run:

```bash
.venv/bin/python -B -m pytest \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  -q -k 'video_retention_runs_during_app_construction_once or next_app_startup or video_retention_startup_failure'
```

Expected: FAIL because `TldwCli` has no `generated_video_store` and does not own the startup sweep.

- [ ] **Step 4: Add the minimal app-owned startup helper**

In `TldwCli`, add:

```python
def _initialize_generated_video_store(self):
    """Create the app-owned video store and apply startup retention once."""
    from tldw_chatbook.Video_Generation.video_store import VideoStore

    store = VideoStore()
    try:
        store.enforce_retention()
    except Exception as exc:
        logger.warning(
            "Generated-video startup retention failed (error_type={}).",
            type(exc).__name__,
        )
    return store
```

Immediately after `self.app_config = load_settings()` in `TldwCli.__init__`, assign:

```python
self.generated_video_store = self._initialize_generated_video_store()
```

Do not add a global singleton, retry flag, async worker, progress UI, or new dependency.

- [ ] **Step 5: Run the startup tests to verify GREEN**

Run the same command from Step 3.

Expected: PASS; the failure case logs only the operation and exception type.

- [ ] **Step 6: Commit the app ownership boundary**

```bash
git add tldw_chatbook/app.py Tests/ProductionApp/test_chat_composition_retirement.py
git diff --cached --check
git commit -m "fix: own video retention at app startup"
```

### Task 3: Make Console screens borrow the app store

**Files:**
- Modify: `Tests/Chat/test_console_video_message.py`
- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Write the narrow ownership-boundary tests**

In `Tests/Chat/test_console_video_message.py`, add:

```python
def test_console_video_store_prefers_explicit_test_override(tmp_path):
    app_store = VideoStore(root=tmp_path / "app")
    override = VideoStore(root=tmp_path / "override")
    screen = ChatScreen(_build_test_app())
    screen.app_instance.generated_video_store = app_store
    screen._console_video_store = override

    assert screen._ensure_console_video_store() is override


def test_console_video_store_borrows_app_owner_without_cleanup(monkeypatch, tmp_path):
    store = VideoStore(root=tmp_path / "app")
    screen = ChatScreen(_build_test_app())
    screen.app_instance.generated_video_store = store
    monkeypatch.setattr(
        store,
        "enforce_retention",
        lambda: pytest.fail("screen must not run retention"),
    )

    assert screen._ensure_console_video_store() is store
    assert screen._ensure_console_video_store() is store


def test_console_video_store_fails_loudly_without_app_owner():
    app = _build_test_app()
    del app.generated_video_store
    screen = ChatScreen(app)

    with pytest.raises(RuntimeError, match="app-owned generated video store"):
        screen._ensure_console_video_store()
```

Keep `_console_video_store` as an explicit test-only override; do not initialize it in production `ChatScreen.__init__`.

Also extend `test_video_retention_startup_failure_is_bounded` after its app-construction and bounded-log assertions:

```python
first_screen = ChatScreen(app)
second_screen = ChatScreen(app)
assert first_screen._ensure_console_video_store() is app.generated_video_store
assert second_screen._ensure_console_video_store() is app.generated_video_store
assert calls == 1
```

These assertions are intentionally added in Task 3. Before the screen ownership change they fail because each fresh screen constructs another store and retries the patched failing sweep; after the change they pass without weakening Task 2's app-startup test.

- [ ] **Step 2: Extend the real navigation test with generated-video bytes**

In `test_registered_chat_route_uses_only_native_console_and_restores_snapshot`, bind a temporary session-retention root before constructing the app. After the first `ChatScreen` mounts:

```python
message_id = "current-run-video"
meta = VideoGenerationMetadata(
    name="current-run-clip",
    prompt="current run",
    backend="comfyui",
)
stored = app.generated_video_store.save(message_id, meta.name, b"current-run-bytes")
message = ConsoleChatMessage(
    id=message_id,
    role=ConsoleMessageRole.ASSISTANT,
    content="[video] current-run-clip",
    video_metadata=meta,
)
assert chat._ensure_console_video_store() is app.generated_video_store
assert chat._build_video_card_specs([message])[message_id].status == "ready"
```

After the existing real `NavigateToScreen("settings")` then `NavigateToScreen("chat")` flow, assert:

```python
assert restored_chat is not chat
assert restored_chat._ensure_console_video_store() is app.generated_video_store
spec = restored_chat._build_video_card_specs([message])[message_id]
assert spec.status == "ready"
assert spec.file_path == str(stored)
assert stored.read_bytes() == b"current-run-bytes"
```

Count `VideoStore.enforce_retention` around app construction and assert it remains exactly one after both screen instances.

- [ ] **Step 3: Run the ownership/navigation tests to verify RED**

Run:

```bash
.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  -q -k 'console_video_store or registered_chat_route_uses_only_native_console or video_retention_startup_failure'
```

Expected: FAIL because a fresh `ChatScreen` still constructs a screen-local store and reapplies session retention.

- [ ] **Step 4: Replace screen construction with app ownership lookup**

Replace `_ensure_console_video_store` with:

```python
def _ensure_console_video_store(self) -> VideoStore:
    """Return an explicit test override or the app-owned VideoStore."""
    override = getattr(self, "_console_video_store", None)
    if override is not None:
        return override
    store = getattr(self.app_instance, "generated_video_store", None)
    if store is None:
        raise RuntimeError("Console requires the app-owned generated video store")
    return store
```

Delete all `VideoStore()` construction, `enforce_retention()`, and sweep logging from this screen method. Do not modify generation, card-resolution, Play, Save, or Regenerate call sites; they already converge here.

- [ ] **Step 5: Run the ownership/navigation tests to verify GREEN**

Run the same command from Step 3.

Expected: PASS; the replacement screen resolves the original bytes through the identical app store and the sweep count remains one.

- [ ] **Step 6: Prove the navigation test is non-vacuous**

Temporarily insert `store.enforce_retention()` immediately before returning the app-owned store from `_ensure_console_video_store()`. Run only:

```bash
.venv/bin/python -B -m pytest \
  Tests/ProductionApp/test_chat_composition_retirement.py::test_registered_chat_route_uses_only_native_console_and_restores_snapshot -q
```

Expected: FAIL because the current-run file is removed during navigation/remount. Restore the production method exactly, rerun the same test, and expect PASS. Do not commit the mutation.

- [ ] **Step 7: Commit the screen ownership boundary**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py
git diff --cached --check
git commit -m "fix: preserve generated videos across Console navigation"
```

### Task 4: Verify, document, and close TASK-3401.19

**Files:**
- Modify: `backlog/tasks/task-3401.19 - Run-session-video-retention-cleanup-only-at-app-startup.md`
- Verify only: all files listed in the file map

- [ ] **Step 1: Run the complete authorized focused test gate**

```bash
.venv/bin/python -B -m pytest \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/Chat/test_console_video_message.py -q
```

Expected: all tests PASS. Record the exact count and warning count in the task notes.

- [ ] **Step 2: Run static checks only on touched Python files**

```bash
.venv/bin/python -m ruff check \
  Tests/UI/app_factory.py \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/Chat/test_console_video_message.py
.venv/bin/python -m ruff check --select E9,F63,F7,F82 \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/chat_screen.py
```

The first command is the authoritative full-rule gate for the three touched test/support files. The second is the authoritative fatal-rule gate for the two large production files, which carry unrelated full-rule baseline findings. Do not edit unrelated lint and do not claim a whole-file full-rule pass for those production files.

Compile the five touched Python files to a temporary directory so no repository artifacts remain:

```bash
.venv/bin/python -B -c 'import py_compile, tempfile; from pathlib import Path; files = [Path("tldw_chatbook/app.py"), Path("tldw_chatbook/UI/Screens/chat_screen.py"), Path("Tests/UI/app_factory.py"), Path("Tests/ProductionApp/test_chat_composition_retirement.py"), Path("Tests/Chat/test_console_video_message.py")]; out = Path(tempfile.mkdtemp()); [py_compile.compile(str(path), cfile=str(out / (path.name + ".pyc")), doraise=True) for path in files]'
git diff --check
```

Expected: focused lint/compile/diff checks PASS, or an identical pre-existing lint baseline is documented without unrelated cleanup.

- [ ] **Step 3: Self-review the lifecycle boundary**

Confirm from the final diff:

- `enforce_retention()` has exactly one production startup call owned by `TldwCli`;
- `ChatScreen` has no production fallback that creates or cleans a store;
- a new `TldwCli` creates a new store and reruns retention;
- TTL behavior still comes entirely from `VideoStore`;
- startup failure diagnostics contain no exception string, paths, message ids, or media names;
- `video_store.py` is unchanged and TASK-3401.20 remains the only size-cap follow-up;
- no broad or full test suite was run.

- [ ] **Step 4: Complete Backlog hygiene**

Using the Backlog CLI, check ACs 1-5, replace Implementation Notes with a concise PR-style summary including exact focused test/static commands and results, record the ADR check below, and set TASK-3401.19 to Done only after every gate passes.

```text
ADR required: no
ADR path: backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md
Reason: ADR-044 already assigns session/TTL retention to app startup; this fix only moves the existing call to the correct lifetime owner.
```

If implementation surfaces no reusable repository trap beyond the already documented lifecycle defect, record that no new lesson was required. Do not invent one.

- [ ] **Step 5: Commit task closeout**

```bash
git add 'backlog/tasks/task-3401.19 - Run-session-video-retention-cleanup-only-at-app-startup.md'
git diff --cached --check
git commit -m "docs: close video retention startup task"
```

- [ ] **Step 6: Verify branch/PR identity, then push and update the existing draft PR**

Run these read-only preflights:

```bash
git branch --show-current
gh pr view 1460 --json headRefName,baseRefName,isDraft,url
```

Expected: current branch and PR head are both `codex/task-3401-14-h3-generation-uat`, base is `dev`, and the PR is draft. Only then push that branch and update PR #1460's body with TASK-3401.19's outcome and exact focused evidence. Leave the PR draft unless the remaining workstream is complete or the user explicitly asks to mark it ready.
