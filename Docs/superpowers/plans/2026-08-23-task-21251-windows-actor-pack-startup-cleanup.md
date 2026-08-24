# Windows Actor Pack Startup Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow Windows to boot when Actor Pack staging privacy cannot be verified, without inspecting or deleting unverified staged data.

**Architecture:** Keep the authority decision inside `ActorPackImportService.sweep_staging()`. Treat a usable-but-unverified private-path result as a cleanup no-op returning `0`; preserve the existing cleanup error for unusable paths and the existing authenticated POSIX cleanup flow.

**Tech Stack:** Python 3.11+, pytest, Backlog.md CLI, existing `PrivatePathResult` contract

**Spec:** `Docs/superpowers/specs/2026-08-23-windows-actor-pack-startup-cleanup-design.md`

**Backlog task:** `backlog/tasks/task-21251 - Keep-Windows-startup-alive-when-Actor-Pack-cleanup-is-unsupported.md`

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** The fix preserves ADR-074's existing fail-closed authority boundary. Native Windows filesystem authority is separate work in TASK-21252.

---

## File map

- Modify `Tests/Actor_Packs/test_actor_pack_import.py`: add the Windows-equivalent private-path classification regression and prove that no staging enumeration occurs.
- Modify `tldw_chatbook/Actor_Packs/importer.py`: add the minimal early return for usable-but-unverified startup staging.
- Modify `backlog/tasks/task-21251 - Keep-Windows-startup-alive-when-Actor-Pack-cleanup-is-unsupported.md`: record completed acceptance criteria, verification evidence, ADR disposition, and implementation notes.

No new runtime module, dependency, configuration, schema, or ADR is needed.

### Task 1: Pin the unsupported-platform startup contract

**Files:**
- Modify: `Tests/Actor_Packs/test_actor_pack_import.py`
- Test: `Tests/Actor_Packs/test_actor_pack_import.py`

- [ ] **Step 1: Import the existing private-path result types**

Add this import after the Actor Pack repository import:

```python
from tldw_chatbook.Utils.private_paths import PrivatePathResult, PrivatePathStatus
```

- [ ] **Step 2: Write the failing regression test beside the existing startup sweep test**

```python
def test_startup_sweep_skips_usable_unverified_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "profile.db"), client_id="actor-pack-import-unverified"
    )
    repository = ActorPackRepository(db)
    staging_root = tmp_path / "staging"
    candidate = staging_root / f".import-{'0' * 32}"
    candidate.mkdir(parents=True)
    privacy = PrivatePathResult(
        staging_root,
        PrivatePathStatus.UNVERIFIED_PLATFORM,
        reason="native_acl_not_verified",
    )

    monkeypatch.setattr(
        importer_module,
        "secure_private_directory",
        lambda *_args, **_kwargs: privacy,
    )

    def unexpected_access(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unverified staging contents must not be examined")

    monkeypatch.setattr(importer_module.os, "scandir", unexpected_access)
    monkeypatch.setattr(
        importer_module, "_read_candidate_authority", unexpected_access
    )

    service = ActorPackImportService(
        repository,
        staging_root=staging_root,
        profile_root=tmp_path,
    )

    assert service.sweep_staging() == 0
    assert candidate.exists()
```

- [ ] **Step 3: Run the new test and verify the red state**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py::test_startup_sweep_skips_usable_unverified_staging -q
```

Expected: FAIL during `ActorPackImportService` construction with
`ActorPackImportError: actor_pack_import_cleanup_denied`.

- [ ] **Step 4: Commit the failing test**

```bash
git add Tests/Actor_Packs/test_actor_pack_import.py
git commit -m "test: reproduce Windows Actor Pack startup failure"
```

### Task 2: Make unsupported cleanup a non-destructive no-op

**Files:**
- Modify: `tldw_chatbook/Actor_Packs/importer.py:225-233`
- Test: `Tests/Actor_Packs/test_actor_pack_import.py`

- [ ] **Step 1: Add the minimal authority guard**

Replace the current unverified result check with:

```python
            if not privacy.verified_private:
                if privacy.usable:
                    return 0
                raise ValueError
```

Do not catch the error in `app.py`, run candidate helpers on the early-return
branch, or change the separate import-time private-staging checks.

- [ ] **Step 2: Run the new regression and verify the green state**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py::test_startup_sweep_skips_usable_unverified_staging -q
```

Expected: PASS.

- [ ] **Step 3: Run both startup cleanup contracts**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q -k startup_sweep
```

Expected: two tests PASS: the unsupported-platform no-op and the authenticated,
bounded supported-platform cleanup.

- [ ] **Step 4: Commit the minimal runtime fix**

```bash
git add tldw_chatbook/Actor_Packs/importer.py
git commit -m "fix: skip unverified Actor Pack startup cleanup"
```

### Task 3: Verify the Actor Pack boundary and close TASK-21251

**Files:**
- Modify: `backlog/tasks/task-21251 - Keep-Windows-startup-alive-when-Actor-Pack-cleanup-is-unsupported.md`
- Verify: `Tests/Actor_Packs/test_actor_pack_import.py`
- Verify: `Tests/Actor_Packs/`
- Verify: `tldw_chatbook/Actor_Packs/importer.py`

- [ ] **Step 1: Run the complete Actor Pack import test module**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run the Actor Pack test package**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs -q
```

Expected: all tests PASS. Record any pre-existing teardown warnings separately
from test failures.

- [ ] **Step 3: Run static and patch hygiene checks**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Actor_Packs/importer.py Tests/Actor_Packs/test_actor_pack_import.py
git diff --check origin/dev...HEAD
```

Expected: both commands exit `0` with no output.

- [ ] **Step 4: Review the final diff for scope and security boundary**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_chatbook/Actor_Packs/importer.py Tests/Actor_Packs/test_actor_pack_import.py
```

Expected: one focused test addition and one early-return guard; no `app.py`,
archive validation, activation, schema, dependency, or configuration changes.

- [ ] **Step 5: Update TASK-21251 through the Backlog CLI**

Check all four acceptance criteria, add concise implementation notes including
the commands and results above, state that no new lesson was needed, retain the
ADR-074 link and ADR check, and set the task to `Done` only if every required
check is green:

```bash
backlog task edit 21251 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --notes "Implemented the usable-but-unverified startup cleanup no-op in ActorPackImportService, with regression coverage proving service construction succeeds, sweep returns zero, and no candidate enumeration or deletion occurs. Existing authenticated cleanup behavior remains covered. ADR required: no; ADR-074 continues to govern the fail-closed boundary. Verification: record exact passing commands and counts here. Lessons: no new general lesson; the incident is specific to the documented platform capability split." -s Done --plain
```

- [ ] **Step 6: Commit task completion metadata**

```bash
git add "backlog/tasks/task-21251 - Keep-Windows-startup-alive-when-Actor-Pack-cleanup-is-unsupported.md"
git commit -m "docs: complete TASK-21251"
```

- [ ] **Step 7: Confirm a clean worktree and final task state**

Run:

```bash
git status --short --branch
backlog task 21251 --plain
backlog task 21252 --plain
```

Expected: the branch is clean, TASK-21251 is `Done` with checked acceptance
criteria and verification notes, and deferred native Windows support remains
`To Do` as TASK-21252.
