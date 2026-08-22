# TASK-19520 Skill Trust Material Permissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist local-skill trust manifests, encrypted snapshots, and file-backed generation markers with owner-only POSIX permissions from initial temp-file creation through atomic publication.

**Architecture:** Add an explicit `owner_only` opt-in to the shared atomic replace primitive, leaving its default behavior unchanged. The trust store opts into that path for JSON and bytes writes and secures both trust-owned directory levels before writing; focused tests prove pre-replace modes, legacy tightening, collision ownership, cleanup, and non-trust compatibility. A post-review hardening amendment makes owner-only writes recover from a stale deterministic temp by trying a bounded set of random same-directory siblings while preserving every path the current call did not create.

**Tech Stack:** Python 3.11+, `os.open`/`os.fchmod`, `pathlib`, pytest, existing `Skills_Interop.atomic_write` and `SkillTrustStore` APIs

**Design:** [TASK-19520 skill trust permissions design](../specs/2026-08-21-task-19520-skill-trust-permissions-design.md)

**Governing ADR:** [ADR-009: Local Skill Trust Boundary](../../../backlog/decisions/009-local-skill-trust-boundary.md)

---

## File Map

- Modify `tldw_chatbook/Skills_Interop/atomic_write.py`: own exclusive `0o600` temp pre-creation and ownership-aware cleanup behind `owner_only=True`.
- Modify `tldw_chatbook/Skills_Interop/skill_trust_store.py`: opt trust JSON/bytes writes into owner-only temp creation and normalize trust-owned directories to `0o700` on POSIX.
- Modify `Tests/Skills/test_atomic_write_concurrency.py`: characterize the shared helper's secure ordering, EEXIST ownership, descriptor cleanup, default behavior, and the existing trust-store spy signature.
- Create `Tests/Skills/test_skill_trust_permissions.py`: exercise manifest, snapshot, marker, pre-replace, snapshot-first, and legacy-mode behavior through production store APIs.
- Modify `backlog/tasks/task-19520 - Skill-trust-material-is-written-with-default-filesystem-permissions.md`: record the implementation plan before coding, then check acceptance criteria and add implementation evidence after verification.

The Backlog task's `## Implementation Plan` and ADR check were recorded with
this document before any test or production-code edit. TASK-19520 has five
digits, so all remaining task updates must edit the source Markdown directly;
the repository's Backlog CLI 1.44.0 can silently target a bogus task for
five-digit IDs.

Tasks 1–5 below are the completed historical implementation. Only Task 6 is
executable for the post-review amendment; its focused verification supersedes
the historical broad-suite commands at the user's explicit direction.

### Task 1: Characterize Owner-Only Atomic Temp Creation

**Files:**
- Modify: `Tests/Skills/test_atomic_write_concurrency.py:25-181`
- Modify: `Tests/Skills/test_atomic_write_concurrency.py:349-375`

- [ ] **Step 1: Add imports and a POSIX mode helper**

Add `errno`, `os`, and `stat` imports. Add this helper beside `_stray_entries`:

```python
def _mode(path: Path) -> int:
    """Return only the traditional POSIX permission bits for ``path``."""
    return stat.S_IMODE(path.stat().st_mode)
```

- [ ] **Step 2: Write the failing pre-replace mode test**

Add a POSIX-only test to `TestCleanupOnFailure` that delegates to the real `Path.replace` after observing the source:

```python
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits required")
def test_owner_only_temp_is_0600_before_replace(self, tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)
    observed_modes: list[int] = []
    opened_modes: list[int] = []
    original_replace = Path.replace
    original_open = aw.os.open

    def record_open(path, flags, mode):
        opened_modes.append(mode)
        return original_open(path, flags, mode)

    def inspect_then_replace(self, other):
        observed_modes.append(_mode(self))
        return original_replace(self, other)

    monkeypatch.setattr(aw.os, "open", record_open)
    monkeypatch.setattr(Path, "replace", inspect_then_replace)

    aw.replace_atomically(
        temp,
        target,
        lambda path: path.write_text("payload", encoding="utf-8"),
        owner_only=True,
    )

    assert opened_modes == [0o600]
    assert observed_modes == [0o600]
    assert _mode(target) == 0o600
```

- [ ] **Step 3: Write the failing ownership and cleanup tests**

Add four tests:

```python
def test_owner_only_collision_preserves_unowned_temp(self, tmp_path):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)
    temp.write_bytes(b"unowned sentinel")

    with pytest.raises(FileExistsError):
        aw.replace_atomically(
            temp,
            target,
            lambda path: path.write_text("replacement", encoding="utf-8"),
            owner_only=True,
        )

    assert temp.read_bytes() == b"unowned sentinel"
    assert not target.exists()


def test_owner_only_cleans_temp_after_writer_failure(self, tmp_path):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)

    def fail_after_precreate(path: Path) -> None:
        assert path.exists()
        raise RuntimeError("simulated secure writer failure")

    with pytest.raises(RuntimeError, match="secure writer failure"):
        aw.replace_atomically(temp, target, fail_after_precreate, owner_only=True)

    assert not temp.exists()
    assert not target.exists()


def test_owner_only_cleans_temp_after_replace_failure(self, tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)

    def fail_replace(self, other):
        del self, other
        raise OSError("simulated owner-only replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="owner-only replace failure"):
        aw.replace_atomically(
            temp,
            target,
            lambda path: path.write_text("payload", encoding="utf-8"),
            owner_only=True,
        )

    assert not temp.exists()
    assert not target.exists()


def test_default_replace_does_not_precreate_temp(self, tmp_path):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)

    def writer(path: Path) -> None:
        assert not path.exists()
        path.write_text("default behavior", encoding="utf-8")

    aw.replace_atomically(temp, target, writer)

    assert target.read_text(encoding="utf-8") == "default behavior"
```

- [ ] **Step 4: Write the failing descriptor-cleanup test**

On POSIX, wrap the real `os.open`, force `os.fchmod` to fail, then prove the captured descriptor is closed and the created path is cleaned:

```python
@pytest.mark.skipif(os.name != "posix", reason="os.fchmod required")
def test_owner_only_closes_fd_when_fchmod_fails(self, tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    temp = aw.unique_temp_path(target)
    opened_fds: list[int] = []
    original_open = aw.os.open

    def record_open(path, flags, mode):
        fd = original_open(path, flags, mode)
        opened_fds.append(fd)
        return fd

    def fail_fchmod(fd, mode):
        del fd, mode
        raise OSError("chmod failed")

    monkeypatch.setattr(aw.os, "open", record_open)
    monkeypatch.setattr(aw.os, "fchmod", fail_fchmod)

    with pytest.raises(OSError, match="chmod failed"):
        aw.replace_atomically(
            temp,
            target,
            lambda path: path.write_text("payload", encoding="utf-8"),
            owner_only=True,
        )

    assert len(opened_fds) == 1
    with pytest.raises(OSError) as exc_info:
        os.fstat(opened_fds[0])
    assert exc_info.value.errno == errno.EBADF
    assert not temp.exists()
```

- [ ] **Step 5: Update the existing trust-store spy for the new keyword**

Change `spy_replace_atomically` in `test_trust_store_temp_name_is_dot_prefixed_hidden_convention` to accept and forward `owner_only`:

```python
def spy_replace_atomically(
    temp_path, target_path, write_fn, *, owner_only=False
):
    observed_temp_names.append(temp_path.name)
    return original_replace_atomically(
        temp_path,
        target_path,
        write_fn,
        owner_only=owner_only,
    )
```

- [ ] **Step 6: Run the focused tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py \
  -q
```

Expected: the new tests that pass `owner_only=True` fail with `TypeError:
replace_atomically() got an unexpected keyword argument 'owner_only'`. The
updated existing trust-store spy also fails when it forwards
`owner_only=False` to the not-yet-extended helper; other pre-existing tests
remain green.

- [ ] **Step 7: Commit the RED tests**

```bash
git add Tests/Skills/test_atomic_write_concurrency.py
git commit -m "test(skills): specify owner-only atomic temp behavior"
```

### Task 2: Implement Owner-Only Shared Atomic Creation

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/atomic_write.py:34-99`
- Test: `Tests/Skills/test_atomic_write_concurrency.py`

- [ ] **Step 1: Add the owner-only constant and flag builder**

Add this constant after imports:

```python
_OWNER_ONLY_FILE_MODE = 0o600
```

Inside a private helper, build flags without assuming optional constants exist:

```python
def _owner_only_open_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_BINARY", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    return flags
```

- [ ] **Step 2: Extend `replace_atomically` with ownership-aware creation**

Change the signature and body to:

```python
def replace_atomically(
    temp_path: Path,
    target_path: Path,
    write_fn: Callable[[Path], None],
    *,
    owner_only: bool = False,
) -> None:
    """Write through ``temp_path`` and atomically replace ``target_path``.

    When ``owner_only`` is true, exclusively pre-create the temp file with
    owner-only permissions before ``write_fn`` can reopen it. Cleanup never
    unlinks an unexplained path when exclusive creation failed.
    """
    created_temp = False
    try:
        if owner_only:
            fd = os.open(temp_path, _owner_only_open_flags(), _OWNER_ONLY_FILE_MODE)
            created_temp = True
            try:
                if os.name == "posix":
                    os.fchmod(fd, _OWNER_ONLY_FILE_MODE)
            finally:
                os.close(fd)
        write_fn(temp_path)
        temp_path.replace(target_path)
    except BaseException:
        if not owner_only or created_temp:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
```

Retain and expand the existing Google-style `Args`/`Raises` documentation, including the `owner_only` behavior, `O_EXCL` collision ownership, and unchanged default semantics.

- [ ] **Step 3: Run the focused shared-helper tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 4: Run static checks for the changed helper**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  Tests/Skills/test_atomic_write_concurrency.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  Tests/Skills/test_atomic_write_concurrency.py
```

Expected: both commands exit zero.

- [ ] **Step 5: Commit the shared primitive**

```bash
git add tldw_chatbook/Skills_Interop/atomic_write.py Tests/Skills/test_atomic_write_concurrency.py
git commit -m "fix(skills): create owner-only atomic temp files"
```

### Task 3: Specify Trust-Store File and Directory Modes

**Files:**
- Create: `Tests/Skills/test_skill_trust_permissions.py`

- [ ] **Step 1: Add focused production-path fixtures**

Create the test file with POSIX-only mode assertions and helpers:

```python
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop import skill_trust_store as trust_store_module
from tldw_chatbook.Skills_Interop.skill_trust_crypto import derive_skill_trust_keys
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX mode-bit assertions"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _manifest(generation: int) -> dict:
    return {
        "version": 1,
        "generation": generation,
        "skills": {},
        "audit": [],
    }


def _store(tmp_path: Path):
    trust_dir = tmp_path / "trust"
    marker_path = trust_dir / "generation_marker.json"
    marker = FileSkillTrustGenerationMarkerStore(
        marker_path=marker_path,
        store_dir=trust_dir,
    )
    return (
        SkillTrustStore(store_dir=trust_dir, marker_store=marker),
        marker_path,
    )
```

- [ ] **Step 2: Write the snapshot-first and final-mode test**

```python
def test_snapshot_first_secures_store_snapshot_manifest_and_marker(tmp_path):
    store, marker_path = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"p" * 32)

    store.save_snapshot(
        "demo-1",
        {"files": {"SKILL.md": "# Demo"}},
        keys,
        generation=1,
    )

    snapshot_path = store.snapshots_dir / "demo-1.json"
    assert _mode(store.store_dir) == 0o700
    assert _mode(store.snapshots_dir) == 0o700
    assert _mode(snapshot_path) == 0o600

    store.save_manifest(_manifest(1), keys, salt=b"p" * 32)

    assert _mode(store.manifest_path) == 0o600
    assert _mode(marker_path) == 0o600
```

- [ ] **Step 3: Write the pre-replace production-path test**

Monkeypatch `Path.replace`, delegate to the real implementation, and assert every observed trust temp is hidden and `0o600`:

```python
def test_trust_temp_is_owner_only_before_replace(tmp_path, monkeypatch):
    store, _ = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"q" * 32)
    observed: list[tuple[str, int]] = []
    observed_owner_only: list[bool] = []
    original_replace = Path.replace
    original_replace_atomically = trust_store_module.replace_atomically

    def inspect_atomic_call(
        temp_path, target_path, write_fn, *, owner_only=False
    ):
        observed_owner_only.append(owner_only)
        return original_replace_atomically(
            temp_path,
            target_path,
            write_fn,
            owner_only=owner_only,
        )

    def inspect_then_replace(self, other):
        observed.append((self.name, _mode(self)))
        return original_replace(self, other)

    monkeypatch.setattr(
        trust_store_module, "replace_atomically", inspect_atomic_call
    )
    monkeypatch.setattr(Path, "replace", inspect_then_replace)
    store.save_manifest(_manifest(1), keys, salt=b"q" * 32)

    assert observed
    assert observed_owner_only
    assert all(observed_owner_only)
    assert all(name.startswith(".") for name, _ in observed)
    assert all(mode == 0o600 for _, mode in observed)
```

- [ ] **Step 4: Write the bytes-writer owner-only test**

Call the real private bytes writer because manifest rollback reaches this seam
directly. Spy on the module's imported `replace_atomically`, delegate to the
real helper, and require `owner_only=True`:

```python
def test_trust_bytes_writer_requests_owner_only_creation(tmp_path, monkeypatch):
    target = tmp_path / "trust" / "manifest-rollback.json"
    observed_owner_only: list[bool] = []
    original_replace_atomically = trust_store_module.replace_atomically

    def inspect_atomic_call(
        temp_path, target_path, write_fn, *, owner_only=False
    ):
        observed_owner_only.append(owner_only)
        return original_replace_atomically(
            temp_path,
            target_path,
            write_fn,
            owner_only=owner_only,
        )

    monkeypatch.setattr(
        trust_store_module, "replace_atomically", inspect_atomic_call
    )
    trust_store_module._atomic_write_bytes(
        target,
        b"previous manifest bytes",
        base_dir=target.parent,
    )

    assert observed_owner_only == [True]
    assert _mode(target) == 0o600
```

- [ ] **Step 5: Write the legacy-tightening test**

Create valid files first, widen them and their trust-owned directories, then rewrite through public APIs:

```python
def test_next_write_tightens_legacy_files_and_directories(tmp_path):
    store, marker_path = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"r" * 32)
    snapshot_path = store.snapshots_dir / "demo-1.json"

    store.save_snapshot("demo-1", {"files": {}}, keys, generation=1)
    store.save_manifest(_manifest(1), keys, salt=b"r" * 32)

    for directory in (store.store_dir, store.snapshots_dir):
        directory.chmod(0o755)
    for path in (snapshot_path, store.manifest_path, marker_path):
        path.chmod(0o666)

    store.save_snapshot("demo-1", {"files": {}}, keys, generation=2)
    store.save_manifest(_manifest(2), keys)

    assert _mode(store.store_dir) == 0o700
    assert _mode(store.snapshots_dir) == 0o700
    assert _mode(snapshot_path) == 0o600
    assert _mode(store.manifest_path) == 0o600
    assert _mode(marker_path) == 0o600
```

- [ ] **Step 6: Run the new tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skill_trust_permissions.py \
  -q
```

Expected on POSIX: failures show permissive final/temp modes, missing
`owner_only=True` on both JSON and bytes writers, or the missing trust root
during snapshot-first creation. On non-POSIX: tests skip cleanly.

- [ ] **Step 7: Commit the RED trust-store tests**

```bash
git add Tests/Skills/test_skill_trust_permissions.py
git commit -m "test(skills): specify trust material permissions"
```

### Task 4: Enforce Trust-Store File and Directory Permissions

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py:1-25`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py:504-528`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py:591-630`
- Test: `Tests/Skills/test_skill_trust_permissions.py`

- [ ] **Step 1: Add the POSIX directory constant**

Import `os` with the standard-library imports and define beside the trust filenames:

```python
_OWNER_ONLY_DIRECTORY_MODE = 0o700
```

- [ ] **Step 2: Secure the trust root before snapshot-first creation**

At the beginning of the persistence part of `SkillTrustStore.save_snapshot`, secure both owned levels in order:

```python
_ensure_trust_directory(self.store_dir)
snapshots_dir = _ensure_trust_directory(self.snapshots_dir)
```

This must happen before `_atomic_write_json`; do not recursively chmod ancestors above `self.store_dir`.

- [ ] **Step 3: Opt both trust writers into owner-only temp creation**

Change the two calls to:

```python
replace_atomically(
    temp_path,
    path,
    lambda t: t.write_text(text, encoding="utf-8"),
    owner_only=True,
)
```

and:

```python
replace_atomically(
    temp_path,
    path,
    lambda t: t.write_bytes(payload),
    owner_only=True,
)
```

Update the surrounding TASK-17963 comment to explain that TASK-19520 additionally pre-creates trust temps with owner-only permissions before the content callback.

- [ ] **Step 4: Normalize each trust-owned leaf directory**

Change `_ensure_trust_directory` to create the leaf with `0o700`, re-check the symlink invariant after creation, and normalize the leaf on POSIX:

```python
def _ensure_trust_directory(path: Path) -> Path:
    directory = validate_path_simple(path)
    if directory.is_symlink():
        raise ValueError("unsafe skill trust path")
    directory.mkdir(
        mode=_OWNER_ONLY_DIRECTORY_MODE,
        parents=True,
        exist_ok=True,
    )
    if directory.is_symlink():
        raise ValueError("unsafe skill trust path")
    if os.name == "posix":
        directory.chmod(_OWNER_ONLY_DIRECTORY_MODE)
    return directory
```

Do not swallow a POSIX `chmod` failure; the write must abort before creating sensitive material.

- [ ] **Step 5: Run the new trust permission tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skill_trust_permissions.py \
  -q
```

Expected: all tests pass on POSIX or skip on non-POSIX.

- [ ] **Step 6: Run the focused regression set**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_store.py \
  Tests/Skills/test_skill_trust_store_reset.py \
  Tests/Skills/test_skill_trust_permissions.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 7: Run formatting and lint checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
```

Expected: both commands exit zero.

- [ ] **Step 8: Commit the trust-store enforcement**

```bash
git add \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_skill_trust_permissions.py
git commit -m "fix(skills): restrict trust material permissions"
```

### Task 5: Mutation Check, Full Verification, and Backlog Close-out

**Files:**
- Modify: `backlog/tasks/task-19520 - Skill-trust-material-is-written-with-default-filesystem-permissions.md`
- Verify: `tldw_chatbook/Skills_Interop/atomic_write.py`
- Verify: `tldw_chatbook/Skills_Interop/skill_trust_store.py`
- Verify: `Tests/Skills/test_atomic_write_concurrency.py`
- Verify: `Tests/Skills/test_skill_trust_permissions.py`

- [ ] **Step 1: Mutation-check both trust-writer file-mode guards**

Temporarily disable `owner_only=True` in the JSON trust writer without
committing it and run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skill_trust_permissions.py::test_trust_temp_is_owner_only_before_replace \
  -q
```

Expected: FAIL because the production writer no longer reports the explicit
owner-only opt-in (and under a permissive umask the observed temp mode is also
not `0o600`). Restore the line immediately with `apply_patch`.

Then temporarily disable `owner_only=True` in the bytes trust writer and run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skill_trust_permissions.py::test_trust_bytes_writer_requests_owner_only_creation \
  -q
```

Expected: FAIL because the writer no longer requests owner-only creation.
Restore the line immediately with `apply_patch`.

- [ ] **Step 2: Mutation-check directory normalization**

Temporarily disable the POSIX `directory.chmod(0o700)` call without committing it and run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skill_trust_permissions.py::test_next_write_tightens_legacy_files_and_directories \
  -q
```

Expected: FAIL because widened trust directories remain `0o755`. Restore the line immediately with `apply_patch`.

- [ ] **Step 3: Run the complete Skills suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q
```

Expected: all Skills tests pass.

- [ ] **Step 4: Run the full test suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

Expected: the complete repository suite passes. If machine-load flakes occur,
compare the exact failure set from the identical command against a clean
`origin/dev` worktree per `backlog/docs/lessons-testing-evidence.md`; do not
substitute collection success for the full gate.

- [ ] **Step 5: Run reachability and static verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ --collect-only -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
git diff --check
```

Expected: collection succeeds with no errors; lint, formatting, and whitespace checks exit zero.

- [ ] **Step 6: Review the final diff for scope and security ordering**

Confirm:

- only trust-store callers pass `owner_only=True`;
- exclusive-create failure cannot unlink an unowned path;
- the descriptor closes if `fchmod` fails;
- `store_dir` is secured before snapshot-first creation;
- legacy files and owned directories tighten only on writes;
- no Windows ACL guarantee or same-UID isolation is claimed; and
- no unrelated working-tree changes are staged.

- [ ] **Step 7: Update TASK-19520 close-out sections**

In the task file:

- mark all five acceptance criteria `[x]` only after the evidence above is green;
- add concise implementation notes naming the shared `owner_only` path, trust-store opt-in, directory ordering, legacy migration, non-POSIX boundary, mutation results, and exact test totals;
- include:

```text
ADR required: no
ADR path: backlog/decisions/009-local-skill-trust-boundary.md
Reason: direct hardening of ADR-009's existing persistence boundary.
```

- [ ] **Step 8: Mark the five-digit task Done by editing its source file**

Do not call `backlog task edit/view 19520`: the repository's Backlog CLI 1.44.0
has a documented five-digit-ID bug that can create
`backlog/tasks/task-task- - .md`. Edit the task frontmatter to `status: Done`,
set `updated_date`, and verify the exact source file directly with `sed`/`rg`.
Run `git status --short backlog/` and confirm no bogus task file exists.

- [ ] **Step 9: Commit task close-out documentation**

```bash
git add 'backlog/tasks/task-19520 - Skill-trust-material-is-written-with-default-filesystem-permissions.md'
git commit -m "docs: close TASK-19520"
```

- [ ] **Step 10: Invoke completion verification and branch-finish workflows**

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`. After review findings are resolved and verification remains green, use `superpowers:finishing-a-development-branch` to present merge/PR/cleanup options.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/009-local-skill-trust-boundary.md`

Reason: the plan hardens the file-permission implementation of ADR-009's existing trust boundary without changing storage ownership, trust policy, cryptography, authentication, or platform ACL contracts.

### Task 6: Recover Owner-Only Writes from Stale Temp Collisions

**Post-review amendment:** Qodo review identified that a failed cleanup can leave
the deterministic PID/thread temp occupied, causing every later write from the
same thread to fail at `O_EXCL`. This task implements the approved design
revision without broadening the default atomic-write path.

**Files:**
- Modify: `Tests/Skills/test_atomic_write_concurrency.py:229-360`
- Modify: `tldw_chatbook/Skills_Interop/atomic_write.py:34-135`

- [ ] **Step 1: Replace the collision-fails test with a failing recovery test**

Pre-create the supplied deterministic temp with sentinel bytes, patch
`secrets.token_hex` to return a known token, and record the path passed to the
writer:

```python
def test_owner_only_collision_uses_fresh_sibling_and_preserves_existing_temp(
    self, tmp_path, monkeypatch
):
    target = tmp_path / "trust.json"
    temp = aw.unique_temp_path(target, hidden=True)
    alternate = temp.with_name(f"{temp.name}.0123456789abcdef")
    sentinel = b"do-not-overwrite-this-temp"
    temp.write_bytes(sentinel)
    written_paths: list[Path] = []

    monkeypatch.setattr(secrets, "token_hex", lambda size: "0123456789abcdef")

    def write(path: Path) -> None:
        written_paths.append(path)
        path.write_bytes(b"replacement")

    aw.replace_atomically(temp, target, write, owner_only=True)

    assert written_paths == [alternate]
    assert temp.read_bytes() == sentinel
    assert target.read_bytes() == b"replacement"
    assert not alternate.exists()
    if os.name == "posix":
        assert _mode(target) == 0o600
```

- [ ] **Step 2: Add a failing bounded-exhaustion test**

Patch `os.open` to raise a distinct `FileExistsError` on each call, record the
writer and cleanup seams, and assert exactly eight candidates are attempted:

```python
def test_owner_only_collision_retry_is_bounded_and_preserves_unowned_paths(
    self, tmp_path, monkeypatch
):
    target = tmp_path / "trust.json"
    temp = aw.unique_temp_path(target, hidden=True)
    collisions: list[FileExistsError] = []
    opened: list[Path] = []
    writer_calls: list[Path] = []
    cleanup_calls: list[Path] = []

    monkeypatch.setattr(secrets, "token_hex", lambda size: f"{len(opened):016x}")

    def collide(path, flags, mode):
        del flags, mode
        opened.append(Path(path))
        error = FileExistsError(errno.EEXIST, f"collision-{len(opened)}", path)
        collisions.append(error)
        raise error

    def record_unlink(self, *, missing_ok=False):
        del missing_ok
        cleanup_calls.append(self)

    monkeypatch.setattr(aw.os, "open", collide)
    monkeypatch.setattr(Path, "unlink", record_unlink)

    with pytest.raises(FileExistsError) as exc_info:
        aw.replace_atomically(
            temp, target, writer_calls.append, owner_only=True
        )

    assert exc_info.value is collisions[-1]
    assert len(opened) == 8
    assert len(set(opened)) == 8
    assert opened[0] == temp
    assert writer_calls == []
    assert cleanup_calls == []
    assert not target.exists()
```

- [ ] **Step 3: Add a failing alternate-ownership cleanup test**

Pre-create the supplied temp, force the writer to fail after writing the fresh
alternate, and prove only the owned alternate is removed:

```python
def test_owner_only_alternate_writer_failure_cleans_only_owned_sibling(
    self, tmp_path, monkeypatch
):
    target = tmp_path / "trust.json"
    temp = aw.unique_temp_path(target, hidden=True)
    alternate = temp.with_name(f"{temp.name}.fedcba9876543210")
    sentinel = b"unowned"
    temp.write_bytes(sentinel)
    monkeypatch.setattr(secrets, "token_hex", lambda size: "fedcba9876543210")

    def fail(path: Path) -> None:
        assert path == alternate
        path.write_bytes(b"partial")
        raise RuntimeError("alternate writer failed")

    with pytest.raises(RuntimeError, match="alternate writer failed"):
        aw.replace_atomically(temp, target, fail, owner_only=True)

    assert temp.read_bytes() == sentinel
    assert not alternate.exists()
    assert not target.exists()
```

- [ ] **Step 4: Run the three new tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py::TestOwnerOnlyTempCreation::test_owner_only_collision_uses_fresh_sibling_and_preserves_existing_temp \
  Tests/Skills/test_atomic_write_concurrency.py::TestOwnerOnlyTempCreation::test_owner_only_collision_retry_is_bounded_and_preserves_unowned_paths \
  Tests/Skills/test_atomic_write_concurrency.py::TestOwnerOnlyTempCreation::test_owner_only_alternate_writer_failure_cleans_only_owned_sibling \
  -q
```

Expected: all three fail because `replace_atomically` still surfaces the first
deterministic collision rather than selecting a fresh sibling. Add `secrets`
to the test module's standard-library imports before these tests so the RED
failures exercise production behavior rather than test setup.

- [ ] **Step 5: Implement bounded owner-only candidate selection**

Import `secrets`, add `_OWNER_ONLY_TEMP_CANDIDATES = 8`, and add a private
iterator that yields the supplied path followed by seven random siblings:

```python
def _owner_only_temp_paths(temp_path: Path):
    yield temp_path
    for _ in range(_OWNER_ONLY_TEMP_CANDIDATES - 1):
        yield temp_path.with_name(f"{temp_path.name}.{secrets.token_hex(8)}")
```

In the `owner_only` branch, retry only `FileExistsError`. Store the candidate
in `owned_temp_path` only after `os.open` succeeds. Invoke `write_fn` and
`replace` with `owned_temp_path`, and make the exception handler unlink only
that path. If all candidates collide, re-raise the final `FileExistsError`
without cleanup. Do not catch other setup errors and do not alter the default
branch.

- [ ] **Step 6: Run the owner-only class and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py::TestOwnerOnlyTempCreation -q
```

Expected: all owner-only tests pass.

- [ ] **Step 7: Run only the focused TASK-19520 verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_store.py \
  Tests/Skills/test_skill_trust_store_reset.py \
  Tests/Skills/test_skill_trust_permissions.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  tldw_chatbook/Skills_Interop/skill_trust_store.py \
  Tests/Skills/test_atomic_write_concurrency.py \
  Tests/Skills/test_skill_trust_permissions.py
git diff --check
```

Expected: focused tests, lint, formatting, and whitespace checks pass. Do not
run the broad Skills or repository suite; the user explicitly limited this
review fix to tests related to the modified functionality.

- [ ] **Step 8: Commit the production fix**

```bash
git add \
  tldw_chatbook/Skills_Interop/atomic_write.py \
  Tests/Skills/test_atomic_write_concurrency.py
git commit -m "fix(skills): recover from stale secure temps"
```

- [ ] **Step 9: Close the post-review task amendment**

Update the TASK-19520 source Markdown directly: add the bounded stale-temp
recovery to its implementation-plan and implementation-notes sections, record
the exact focused-test/static evidence, and return frontmatter status from
`In Progress` to `Done`. Do not use the Backlog CLI for this five-digit ID.

- [ ] **Step 10: Commit documentation and address the Qodo thread**

```bash
git add \
  Docs/superpowers/plans/2026-08-21-task-19520-skill-trust-permissions.md \
  'backlog/tasks/task-19520 - Skill-trust-material-is-written-with-default-filesystem-permissions.md'
git commit -m "docs: record TASK-19520 review amendment"
```

Push the branch, reply inline with the bounded-retry and focused-test evidence,
and resolve the review thread only after GitHub reflects the fix.
