# Private Path and Config Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a stdlib-only, descriptor-anchored private-file boundary and use it to create, harden, and read the effective Chatbook configuration without following links or trusting attacker-writable namespaces.

**Architecture:** A new dependency-leaf `private_paths` module retains lexical path spelling, walks POSIX paths through directory descriptors, returns structured posture outcomes, and exposes pinned reads plus owner-only creation. `config.py` uses that boundary only for effective-path selection and bootstrap read/create in TASK-488; TASK-491 will later route every remaining mutation, raw-editor, encryption, recovery, and export path through the same primitive.

**Tech Stack:** Python 3.11+, stdlib `os`/`stat`/`contextlib`/`dataclasses`/`enum`, TOML via the existing `tomllib`, pytest.

## Global Constraints

- Preserve the lexical absolute path until link and namespace checks finish; do not call `Path.resolve()` for selection.
- The private-path module is stdlib-only and must not import config, logging, Textual, database, or application modules.
- On POSIX, private files are current-effective-user-owned regular files with mode `0600`; application-owned directories use `0700`.
- A custom config parent is never chmodded or created by Chatbook; it must already be a trusted namespace.
- A shared sticky directory can protect an existing current-user-owned entry, but it is unsafe for creation of a missing selected entry.
- Unsafe POSIX config targets and POSIX runtimes lacking required descriptor guards fail closed. Only Windows returns `unverified_platform`, and it must not be described as owner-only or ACL-secure.
- A private success is returned only after descriptor-based identity, entry identity, type, owner, and final-mode postconditions pass.
- Do not open, move, delete, or inspect the contents of `openai-api-key.txt` or `moonshot-api-key.txt`.
- Follow red-green-refactor TDD for every production change.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/022-local-private-data-boundary.md`

Reason: ADR-022 already defines the private local-data boundary. This task directly implements that accepted decision and does not make a new architecture choice.

## File Map

- Create `tldw_chatbook/Utils/private_paths.py`: result model, lexical selection, trusted POSIX descriptor walk, owner-only directory/file lifecycle, pinned binary reads.
- Create `Tests/Utils/test_private_paths.py`: unit and behavioral coverage for status classification, modes, symlinks, sticky parents, and descriptor pinning.
- Create `Tests/test_config_private_bootstrap.py`: effective-path and config bootstrap integration coverage.
- Modify `tldw_chatbook/config.py`: preserve lexical selection, remove fallback creation, and bootstrap through `private_paths`.
- Create `Tests/Utils/test_repository_credential_ignore.py`: exact repository-root ignore regression.
- Modify `.gitignore`: add only `/openai-api-key.txt` and `/moonshot-api-key.txt`.
- Modify `backlog/tasks/task-488 - Establish-private-path-boundary-and-harden-config-bootstrap.md`: record this plan, then add implementation notes and completion evidence after verification.

---

### Task 1: Define structured private-path outcomes and lexical selection

**Files:**

- Create: `tldw_chatbook/Utils/private_paths.py`
- Create: `Tests/Utils/test_private_paths.py`

**Interfaces:**

- Produces: `PrivatePathStatus`, `PrivatePathResult`, `PrivatePathError`, `lexical_path()`.
- Consumes: only Python stdlib.

- [ ] **Step 1: Write failing result-model and lexical-path tests**

```python
from pathlib import Path

import pytest

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    lexical_path,
)


@pytest.mark.parametrize(
    ("status", "verified_private", "usable"),
    [
        (PrivatePathStatus.CREATED_PRIVATE, True, True),
        (PrivatePathStatus.HARDENED_PRIVATE, True, True),
        (PrivatePathStatus.ALREADY_PRIVATE, True, True),
        (PrivatePathStatus.UNVERIFIED_PLATFORM, False, True),
        (PrivatePathStatus.UNSAFE_PARENT, False, False),
        (PrivatePathStatus.WRONG_OWNER, False, False),
        (PrivatePathStatus.LINK_OR_NON_REGULAR, False, False),
        (PrivatePathStatus.OPERATION_FAILED, False, False),
    ],
)
def test_private_path_result_classifies_posture(status, verified_private, usable):
    result = PrivatePathResult(Path("/tmp/config.toml"), status)

    assert result.verified_private is verified_private
    assert result.usable is usable


def test_private_path_error_exposes_bounded_result_without_original_exception():
    result = PrivatePathResult(
        Path("/tmp/config.toml"),
        PrivatePathStatus.UNSAFE_PARENT,
        reason="shared_writable_parent",
    )

    error = PrivatePathError(result)

    assert error.result is result
    assert "shared_writable_parent" in str(error)


def test_lexical_path_normalizes_without_resolving_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    selected = lexical_path(Path("alias") / ".." / "alias" / "config.toml")

    assert selected == alias / "config.toml"
    assert selected != real / "config.toml"


def test_lexical_path_rejects_nul():
    with pytest.raises(ValueError, match="NUL"):
        lexical_path("bad\x00path")
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```bash
python3 -m pytest -q \
  Tests/Utils/test_private_paths.py::test_private_path_result_classifies_posture \
  Tests/Utils/test_private_paths.py::test_lexical_path_normalizes_without_resolving_symlink
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tldw_chatbook.Utils.private_paths'`.

- [ ] **Step 3: Implement the result model and lexical selector**

```python
"""Private local-file lifecycle primitives.

This module is deliberately dependency-leaf: callers choose failure policy and
diagnostics while this module performs lexical selection and filesystem checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

PathInput: TypeAlias = str | os.PathLike[str]


class PrivatePathStatus(StrEnum):
    CREATED_PRIVATE = "created_private"
    HARDENED_PRIVATE = "hardened_private"
    ALREADY_PRIVATE = "already_private"
    UNSAFE_PARENT = "unsafe_parent"
    WRONG_OWNER = "wrong_owner"
    LINK_OR_NON_REGULAR = "link_or_non_regular"
    OPERATION_FAILED = "operation_failed"
    UNVERIFIED_PLATFORM = "unverified_platform"


@dataclass(frozen=True)
class PrivatePathResult:
    lexical_path: Path
    status: PrivatePathStatus
    reason: str | None = None

    @property
    def verified_private(self) -> bool:
        return self.status in {
            PrivatePathStatus.CREATED_PRIVATE,
            PrivatePathStatus.HARDENED_PRIVATE,
            PrivatePathStatus.ALREADY_PRIVATE,
        }

    @property
    def usable(self) -> bool:
        return self.verified_private or (
            self.status is PrivatePathStatus.UNVERIFIED_PLATFORM
        )


class PrivatePathError(OSError):
    def __init__(self, result: PrivatePathResult) -> None:
        self.result = result
        reason = f": {result.reason}" if result.reason else ""
        super().__init__(f"{result.status.value}{reason}")


def lexical_path(path: PathInput) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    expanded = os.path.expanduser(raw)
    return Path(os.path.abspath(os.path.normpath(expanded)))
```

- [ ] **Step 4: Run result-model tests**

Run: `python3 -m pytest -q Tests/Utils/test_private_paths.py -k "result or lexical"`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the result model**

```bash
git add tldw_chatbook/Utils/private_paths.py Tests/Utils/test_private_paths.py
git commit -m "feat(security): define private path outcomes"
```

---

### Task 2: Implement trusted descriptor traversal and existing-file hardening

**Files:**

- Modify: `tldw_chatbook/Utils/private_paths.py`
- Modify: `Tests/Utils/test_private_paths.py`

**Interfaces:**

- Consumes: `lexical_path()`, `PrivatePathResult`, `PrivatePathError`.
- Produces: `open_private_binary(path) -> ContextManager[PrivateBinaryFile]` and `secure_private_directory(path, *, create, application_owned) -> PrivatePathResult`.

> **Implementation correction (2026-07-23):** P1 security review made
> `O_NONBLOCK` and `O_NOCTTY` mandatory POSIX guards, added a no-follow entry
> preclassification before the guarded final open, and rejects multiply-linked
> files before `fchmod` or read and again in the postcondition. These checks
> prevent FIFO/device blocking and hard-link alias hardening. Temporary parent
> descriptors close before caller code runs; the pinned file descriptor belongs
> to the returned stream and closes on context exit.

- [ ] **Step 1: Add failing POSIX traversal, ownership, and mode tests**

```python
import os
import stat


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_hardens_before_read(tmp_path):
    target = tmp_path / "config.toml"
    target.write_bytes(b"[chat]\nstreaming = true\n")
    target.chmod(0o644)

    with open_private_binary(target) as opened:
        assert opened.stream.read().startswith(b"[chat]")
        assert opened.result.status is PrivatePathStatus.HARDENED_PRIVATE
        assert stat.S_IMODE(os.fstat(opened.stream.fileno()).st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_final_symlink(tmp_path):
    outside = tmp_path / "outside.toml"
    outside.write_text("secret = true\n", encoding="utf-8")
    alias = tmp_path / "config.toml"
    alias.symlink_to(outside)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(alias):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert outside.stat().st_mode & 0o777 != 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_intermediate_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.toml").write_text("[chat]\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(alias / "config.toml"):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_non_regular_leaf(tmp_path):
    target = tmp_path / "config.toml"
    target.mkdir()

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
@pytest.mark.timeout(2, method="signal")
def test_open_private_binary_rejects_fifo_without_blocking(tmp_path):
    target = tmp_path / "config.toml"
    os.mkfifo(target, mode=0o644)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_multiply_linked_file_without_changing_alias(
    tmp_path,
):
    target = tmp_path / "config.toml"
    alias = tmp_path / "shared-alias.toml"
    target.write_bytes(b"shared private data")
    target.chmod(0o644)
    os.link(target, alias)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(alias.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_reports_hardening_failure(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_bytes(b"config")
    target.chmod(0o644)

    def fail_fchmod(file_fd, mode):
        raise OSError("simulated")

    monkeypatch.setattr(private_paths.os, "fchmod", fail_fchmod)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "OSError"


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_open_private_binary_fails_when_postcondition_is_not_verified(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_bytes(b"config")
    target.chmod(0o600)
    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_file_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_keeps_opened_identity_when_name_is_replaced(tmp_path):
    target = tmp_path / "config.toml"
    replacement = tmp_path / "replacement.toml"
    target.write_bytes(b"trusted")
    replacement.write_bytes(b"replacement")

    with open_private_binary(target) as opened:
        replacement.replace(target)
        assert opened.stream.read() == b"trusted"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_stays_on_pinned_parent_during_path_replacement(
    tmp_path,
    monkeypatch,
):
    selected_parent = tmp_path / "selected"
    child = selected_parent / "child"
    child.mkdir(parents=True)
    (child / "config.toml").write_bytes(b"trusted")
    displaced_parent = tmp_path / "selected-displaced"
    real_open_component = private_paths._open_directory_component
    raced = False

    def race_after_parent_is_pinned(parent_fd, component):
        nonlocal raced
        if component == "child" and not raced:
            raced = True
            selected_parent.rename(displaced_parent)
            replacement_child = selected_parent / "child"
            replacement_child.mkdir(parents=True)
            (replacement_child / "config.toml").write_bytes(b"replacement")
        return real_open_component(parent_fd, component)

    monkeypatch.setattr(
        private_paths,
        "_open_directory_component",
        race_after_parent_is_pinned,
    )

    with open_private_binary(selected_parent / "child" / "config.toml") as opened:
        assert opened.stream.read() == b"trusted"


def test_stat_classification_rejects_wrong_owner():
    fake = type(
        "FakeStat",
        (),
        {"st_mode": stat.S_IFREG | 0o600, "st_nlink": 1, "st_uid": 2222},
    )()

    assert (
        _classify_private_file_stat(fake, expected_uid=1111)
        is PrivatePathStatus.WRONG_OWNER
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_private_directory_never_reports_success_after_failed_postcondition(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config"
    monkeypatch.setattr(
        private_paths,
        "_private_directory_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        secure_private_directory(
            target,
            create=True,
            application_owned=True,
        )

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_directory_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_private_directory_closes_component_fd_when_entry_stat_disappears(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config"
    target.mkdir()
    opened_components = set()
    real_open_component = private_paths._open_directory_component
    real_stat = private_paths.os.stat

    def track_open(parent_fd, component):
        opened = real_open_component(parent_fd, component)
        opened_components.add(opened)
        return opened

    def fail_nofollow_entry_stat(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        if dir_fd is not None and follow_symlinks is False:
            raise FileNotFoundError("simulated entry replacement")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        private_paths,
        "_open_directory_component",
        track_open,
    )
    monkeypatch.setattr(private_paths.os, "stat", fail_nofollow_entry_stat)
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: True,
    )

    with pytest.raises(PrivatePathError) as caught:
        secure_private_directory(
            target,
            create=False,
            application_owned=True,
        )

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "FileNotFoundError"
    for opened_fd in opened_components:
        with pytest.raises(OSError):
            os.fstat(opened_fd)


def test_private_directory_rejects_filesystem_root():
    with pytest.raises(ValueError, match="filesystem root"):
        secure_private_directory(
            Path(os.path.abspath(os.sep)),
            create=True,
            application_owned=True,
        )


def test_open_private_binary_fails_closed_when_posix_guards_are_unavailable(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_bytes(b"private")
    target.chmod(0o644)
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX capability contract")
@pytest.mark.parametrize("missing_capability", ["_NONBLOCK", "_NOCTTY"])
def test_open_private_binary_fails_before_traversal_when_leaf_guard_is_unavailable(
    tmp_path,
    monkeypatch,
    missing_capability,
):
    target = tmp_path / "config.toml"
    target.write_bytes(b"private")
    target.chmod(0o644)
    monkeypatch.setattr(private_paths, missing_capability, 0, raising=False)
    monkeypatch.setattr(
        private_paths,
        "_open_verified_parent",
        lambda *args, **kwargs: pytest.fail(
            "target traversal occurred without required leaf guards"
        ),
    )

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
```

- [ ] **Step 2: Run the tests and confirm missing-interface failures**

Run: `python3 -m pytest -q Tests/Utils/test_private_paths.py -k "open_private or stat_classification"`

Expected: collection fails because `open_private_binary`, `PrivateBinaryFile`, and `_classify_private_file_stat` are not defined.

- [ ] **Step 3: Implement descriptor traversal and pinned reads**

Add these public shapes and equivalent internal helpers:

```python
import contextlib
import errno
import stat
from typing import BinaryIO, Iterator

_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_NOCTTY = getattr(os, "O_NOCTTY", 0)
_PRIVATE_FILE_OPEN_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _NOCTTY
_WINDOWS_PLATFORM = os.name == "nt"


def _posix_guards_available() -> bool:
    required_dir_fd = {os.open, os.stat, os.mkdir}
    return (
        os.name == "posix"
        and _NOFOLLOW != 0
        and _NONBLOCK != 0
        and _NOCTTY != 0
        and getattr(os, "O_DIRECTORY", 0) != 0
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "geteuid")
        and hasattr(os, "fstat")
        and hasattr(os, "fchmod")
        and hasattr(os, "fsync")
    )


@dataclass
class PrivateBinaryFile:
    stream: BinaryIO
    result: PrivatePathResult


def _classify_private_file_stat(
    file_stat: os.stat_result,
    *,
    expected_uid: int,
) -> PrivatePathStatus | None:
    if not stat.S_ISREG(file_stat.st_mode):
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_nlink != 1:
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_uid != expected_uid:
        return PrivatePathStatus.WRONG_OWNER
    return None


def _directory_allows_untrusted_change(directory_stat: os.stat_result) -> bool:
    return bool(stat.S_IMODE(directory_stat.st_mode) & 0o022)


def _trusted_directory_owner(directory_stat: os.stat_result, euid: int) -> bool:
    return directory_stat.st_uid in {0, euid}


def _open_directory_component(parent_fd: int, component: str) -> int:
    return os.open(
        component,
        _DIRECTORY_OPEN_FLAGS | _NOFOLLOW,
        dir_fd=parent_fd,
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _private_file_postcondition_holds(
    file_fd: int,
    parent_fd: int,
    leaf: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(file_fd)
    entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
    )


def _private_directory_postcondition_holds(
    directory_fd: int,
    parent_fd: int,
    component: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(directory_fd)
    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISDIR(opened.st_mode)
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_DIRECTORY_MODE
    )


def _open_verified_parent(
    selected: Path,
    *,
    missing_leaf_allowed: bool,
) -> tuple[int, str]:
    parts = selected.parts
    if len(parts) < 2 or parts[0] != os.sep:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="invalid_absolute_path",
            )
        )

    euid = os.geteuid()
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    try:
        current_stat = os.fstat(current_fd)
        for component in parts[1:-1]:
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            current_writable = bool(current_mode & 0o022)
            current_sticky = bool(current_mode & stat.S_ISVTX)
            if current_writable and not current_sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                    )
                )
            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="missing_parent",
                    )
                ) from None
            except OSError as exc:
                status = (
                    PrivatePathStatus.LINK_OR_NON_REGULAR
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else PrivatePathStatus.OPERATION_FAILED
                )
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        status,
                        reason=type(exc).__name__,
                    )
                ) from None

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_parent",
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                        )
                    )
                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                os.close(old_fd)
                current_stat = next_stat
            finally:
                if not transferred:
                    os.close(next_fd)

        if not _trusted_directory_owner(current_stat, euid):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason="untrusted_directory_owner",
                )
            )
        final_parent_mode = stat.S_IMODE(current_stat.st_mode)
        final_parent_writable = bool(final_parent_mode & 0o022)
        final_parent_sticky = bool(final_parent_mode & stat.S_ISVTX)
        if final_parent_writable and (
            not final_parent_sticky or missing_leaf_allowed
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason=(
                        "missing_leaf_in_shared_sticky_parent"
                        if final_parent_sticky and missing_leaf_allowed
                        else "shared_writable_parent"
                    ),
                )
            )
        return current_fd, parts[-1]
    except BaseException:
        os.close(current_fd)
        raise


@contextlib.contextmanager
def open_private_binary(path: PathInput) -> Iterator[PrivateBinaryFile]:
    selected = lexical_path(path)
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            with selected.open("rb") as stream:
                yield PrivateBinaryFile(
                    stream=stream,
                    result=PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNVERIFIED_PLATFORM,
                        reason="native_acl_not_verified",
                    ),
                )
            return
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=False,
    )
    file_fd = -1
    try:
        entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry_stat.st_mode):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.LINK_OR_NON_REGULAR,
                )
            )
        file_fd = os.open(leaf, _PRIVATE_FILE_OPEN_FLAGS, dir_fd=parent_fd)
        file_stat = os.fstat(file_fd)
        rejected = _classify_private_file_stat(
            file_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise PrivatePathError(PrivatePathResult(selected, rejected))
        prior_mode = stat.S_IMODE(file_stat.st_mode)
        if prior_mode != _PRIVATE_FILE_MODE:
            os.fchmod(file_fd, _PRIVATE_FILE_MODE)
            status = PrivatePathStatus.HARDENED_PRIVATE
        else:
            status = PrivatePathStatus.ALREADY_PRIVATE
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=file_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        stream = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = -1
    except (PrivatePathError, FileNotFoundError):
        raise
    except OSError as exc:
        status = (
            PrivatePathStatus.LINK_OR_NON_REGULAR
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}
            else PrivatePathStatus.OPERATION_FAILED
        )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                status,
                reason=type(exc).__name__,
            )
        ) from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)

    with stream:
        yield PrivateBinaryFile(
            stream=stream,
            result=PrivatePathResult(selected, status),
        )
```

The complete `_open_verified_parent` component loop above enforces:

1. Start from an `O_DIRECTORY` descriptor for `/`.
2. Open every existing component with `O_DIRECTORY | O_NOFOLLOW` relative to the prior descriptor.
3. Verify the opened component with `fstat`, not a pathname `stat`.
4. Accept only directory owners `0` or `os.geteuid()`.
5. Reject non-sticky group/world-writable directories.
6. For a sticky writable directory, inspect the already-opened next component and require its owner to be `0` or `os.geteuid()`.
7. Reject a missing final leaf under a sticky writable parent.
8. Convert `ELOOP`/`ENOTDIR` into `LINK_OR_NON_REGULAR`, namespace failures into `UNSAFE_PARENT`, and other failures into `OPERATION_FAILED`.
9. Close every superseded descriptor on every branch.

- [ ] **Step 4: Implement application-owned directory creation/hardening**

```python
def secure_private_directory(
    path: PathInput,
    *,
    create: bool,
    application_owned: bool,
) -> PrivatePathResult:
    selected = lexical_path(path)
    if not application_owned:
        raise ValueError("Only application-owned directories may be changed")
    if selected == Path(selected.anchor):
        raise ValueError("The filesystem root cannot be application-owned")
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if create:
                selected.mkdir(parents=True, exist_ok=True)
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    euid = os.geteuid()
    parts = selected.parts
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    created_final = False
    hardened_final = False
    try:
        current_stat = os.fstat(current_fd)
        for index, component in enumerate(parts[1:], start=1):
            is_final = index == len(parts) - 1
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            shared_writable = bool(current_mode & 0o022)
            sticky = bool(current_mode & stat.S_ISVTX)
            if shared_writable and not sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                    )
                )

            created_component = False
            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                if not create:
                    raise
                if shared_writable:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="missing_component_in_shared_sticky_parent",
                        )
                    ) from None
                os.mkdir(component, mode=_PRIVATE_DIRECTORY_MODE, dir_fd=current_fd)
                next_fd = _open_directory_component(current_fd, component)
                created_component = True
            except OSError as exc:
                status = (
                    PrivatePathStatus.LINK_OR_NON_REGULAR
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else PrivatePathStatus.OPERATION_FAILED
                )
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        status,
                        reason=type(exc).__name__,
                    )
                ) from None

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_component",
                        )
                    )
                if next_stat.st_uid != euid and (created_component or is_final):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.WRONG_OWNER,
                            reason="application_directory_wrong_owner",
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                        )
                    )
                if shared_writable and next_stat.st_uid not in {0, euid}:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="sticky_child_wrong_owner",
                        )
                    )

                if created_component or is_final:
                    before = stat.S_IMODE(next_stat.st_mode)
                    if before != _PRIVATE_DIRECTORY_MODE:
                        os.fchmod(next_fd, _PRIVATE_DIRECTORY_MODE)
                        if is_final:
                            hardened_final = True
                    if not _private_directory_postcondition_holds(
                        next_fd,
                        current_fd,
                        component,
                        expected_identity=next_stat,
                    ):
                        raise PrivatePathError(
                            PrivatePathResult(
                                selected,
                                PrivatePathStatus.OPERATION_FAILED,
                                reason="private_directory_postcondition_failed",
                            )
                        )
                if is_final and created_component:
                    created_final = True

                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                os.close(old_fd)
                current_stat = os.fstat(current_fd)
            finally:
                if not transferred:
                    os.close(next_fd)

        status = (
            PrivatePathStatus.CREATED_PRIVATE
            if created_final
            else (
                PrivatePathStatus.HARDENED_PRIVATE
                if hardened_final
                else PrivatePathStatus.ALREADY_PRIVATE
            )
        )
        return PrivatePathResult(selected, status)
    except PrivatePathError:
        raise
    except OSError as exc:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason=type(exc).__name__,
            )
        ) from None
    finally:
        os.close(current_fd)
```

Replace the implementation ellipsis with the complete descriptor loop before running tests.

- [ ] **Step 5: Run focused utility tests**

Run: `python3 -m pytest -q Tests/Utils/test_private_paths.py`

Expected: all tests pass on POSIX; Windows-only assertions use the simulated posture test added in Task 3.

- [ ] **Step 6: Commit descriptor traversal**

```bash
git add tldw_chatbook/Utils/private_paths.py Tests/Utils/test_private_paths.py
git commit -m "feat(security): pin private file traversal"
```

---

### Task 3: Add owner-only creation and platform-posture tests

**Files:**

- Modify: `tldw_chatbook/Utils/private_paths.py`
- Modify: `Tests/Utils/test_private_paths.py`

**Interfaces:**

- Consumes: verified parent traversal and structured results.
- Produces: `create_private_text(path, text, *, application_owned_directory=None, encoding="utf-8") -> PrivatePathResult`.

> **Implementation correction (2026-07-23):** Security review rejected
> pathname rollback after exclusive creation because a check-then-unlink sequence
> can delete an intervening replacement. The final implementation encodes before
> any filesystem mutation, never unlinks by the selected name, retains any
> post-create failure residue at owner-only mode, treats zero-byte writes as a
> bounded failure, and therefore does not require descriptor-relative
> `os.unlink` support.

- [ ] **Step 1: Add failing creation, sticky-parent, race, and Windows-posture tests**

```python
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_create_private_text_is_0600_under_0022_umask(tmp_path):
    target = tmp_path / "config.toml"
    previous = os.umask(0o022)
    try:
        result = create_private_text(target, "[chat]\n")
    finally:
        os.umask(previous)

    assert result.status is PrivatePathStatus.CREATED_PRIVATE
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_create_private_text_rejects_missing_leaf_in_shared_sticky_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(shared / "config.toml", "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT
    assert not (shared / "config.toml").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_create_private_text_does_not_replace_existing_target(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_private_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "existing"


def test_unverified_platform_does_not_claim_private(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    result = create_private_text(target, "[chat]\n")

    assert result.status is PrivatePathStatus.UNVERIFIED_PLATFORM
    assert result.usable is True
    assert result.verified_private is False


def test_unsupported_posix_guards_fail_closed_without_creating(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX capability contract")
def test_missing_unlink_dir_fd_capability_does_not_block_creation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    restricted = frozenset(
        capability
        for capability in private_paths.os.supports_dir_fd
        if capability is not private_paths.os.unlink
    )
    monkeypatch.setattr(private_paths.os, "supports_dir_fd", restricted)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    assert private_paths._posix_guards_available() is True
    result = create_private_text(target, "[chat]\n")

    assert result.status is PrivatePathStatus.CREATED_PRIVATE
    assert target.read_text(encoding="utf-8") == "[chat]\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_create_private_text_retains_private_entry_on_failed_postcondition(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_file_postcondition_failed"
    assert target.read_text(encoding="utf-8") == "[chat]\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mutation contract")
def test_posix_encoding_failure_has_no_filesystem_residue(tmp_path):
    owned_directory = tmp_path / "application-config"
    target = owned_directory / "config.toml"

    with pytest.raises(UnicodeEncodeError):
        create_private_text(
            target,
            "\ud800",
            application_owned_directory=owned_directory,
        )

    assert not owned_directory.exists()
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX rollback race contract")
def test_create_private_text_never_unlinks_name_after_postcondition_failure(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    unlink_calls = []

    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        private_paths.os,
        "unlink",
        lambda *args, **kwargs: unlink_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: True)

    with pytest.raises(PrivatePathError):
        create_private_text(target, "[chat]\n")

    assert unlink_calls == []
    assert target.read_text(encoding="utf-8") == "[chat]\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX write contract")
@pytest.mark.timeout(2, method="signal")
def test_create_private_text_zero_byte_write_fails_without_spinning(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(private_paths.os, "write", lambda *args, **kwargs: 0)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.reason == "zero_byte_write"
    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
```

- [ ] **Step 2: Run the creation tests and confirm the missing function**

Run: `python3 -m pytest -q Tests/Utils/test_private_paths.py -k "create_private or unverified_platform"`

Expected: collection fails because `create_private_text` is not defined.

- [ ] **Step 3: Implement exclusive owner-only creation**

```python
def create_private_text(
    path: PathInput,
    text: str,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
) -> PrivatePathResult:
    selected = lexical_path(path)
    payload = text.encode(encoding)
    if application_owned_directory is not None:
        owned_dir = lexical_path(application_owned_directory)
        if selected.parent != owned_dir:
            raise ValueError("Application-owned directory must be the target parent")
        secure_private_directory(
            owned_dir,
            create=True,
            application_owned=True,
        )

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            with selected.open("xb") as handle:
                handle.write(payload)
                handle.flush()
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    try:
        file_fd = _open_leaf_for_create(parent_fd, leaf)
        created_stat = os.fstat(file_fd)
        os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written == 0:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="zero_byte_write",
                    )
                )
            view = view[written:]
        os.fsync(file_fd)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=created_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        return PrivatePathResult(selected, PrivatePathStatus.CREATED_PRIVATE)
    except FileExistsError:
        raise
    except PrivatePathError:
        raise
    except OSError as exc:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason=type(exc).__name__,
            )
        ) from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)
```

Define the exclusive leaf seam immediately above `create_private_text`:

```python
def _open_leaf_for_create(parent_fd: int, leaf: str) -> int:
    return os.open(
        leaf,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
        _PRIVATE_FILE_MODE,
        dir_fd=parent_fd,
    )
```

- [ ] **Step 4: Add a deterministic final-component race test**

Monkeypatch the private leaf-open seam so the selected leaf is replaced by a symlink immediately before `os.open`. Assert `create_private_text` raises `FileExistsError` or `PrivatePathError`, never writes through the symlink, and leaves the outside sentinel byte-identical.

```python
@pytest.mark.skipif(os.name != "posix", reason="POSIX race contract")
def test_create_private_text_never_follows_raced_final_symlink(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    outside = tmp_path / "outside.toml"
    outside.write_text("sentinel", encoding="utf-8")
    real_open = private_paths._open_leaf_for_create

    def raced_open(parent_fd, leaf):
        target.symlink_to(outside)
        return real_open(parent_fd, leaf)

    monkeypatch.setattr(private_paths, "_open_leaf_for_create", raced_open)

    with pytest.raises((FileExistsError, PrivatePathError)):
        create_private_text(target, "private")

    assert outside.read_text(encoding="utf-8") == "sentinel"
```

The `_open_leaf_for_create(parent_fd, leaf)` seam from Step 3 lets the test trigger the race without weakening production behavior.

- [ ] **Step 5: Run the entire utility suite**

Run: `python3 -m pytest -q Tests/Utils/test_private_paths.py`

Expected: all tests pass.

- [ ] **Step 6: Commit private creation**

```bash
git add tldw_chatbook/Utils/private_paths.py Tests/Utils/test_private_paths.py
git commit -m "feat(security): create private files exclusively"
```

---

### Task 4: Route effective config bootstrap through the private boundary

**Files:**

- Modify: `tldw_chatbook/config.py`
- Create: `Tests/test_config_private_bootstrap.py`
- Modify: `Tests/test_config_console_defaults.py`

**Interfaces:**

- Consumes: `lexical_path`, `open_private_binary`, `create_private_text`, `PrivatePathError`, `PrivatePathStatus`.
- Produces: a lexical `_get_effective_config_path()` and fail-closed bootstrap behavior.

> **Implementation correction (2026-07-23):** Cache review requires every
> real load attempt to clear the prior cache immediately after the cache-hit
> fast path. Only a successful pinned parse plus decrypt, or a successful
> private creation, may populate the cache. Private creation failures propagate;
> malformed TOML and generic read fallbacks return internal defaults uncached so
> the next ordinary call retries the selected file.

- [ ] **Step 1: Add failing config-bootstrap tests**

```python
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook import config as config_module
import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import PrivatePathError


def _clear_config_cache():
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_first_config_creation_is_private(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()
    previous = os.umask(0o022)
    try:
        loaded = config_module.load_cli_config_and_ensure_existence(
            force_reload=True
        )
    finally:
        os.umask(previous)

    assert loaded["_first_run"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_default_application_config_directory_is_created_as_0700(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config" / "config.toml"
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", target)
    _clear_config_cache()

    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_config_is_hardened_before_read(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults]\nstreaming = false\n", encoding="utf-8")
    target.chmod(0o644)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert loaded["chat_defaults"]["streaming"] is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX link contract")
def test_config_loader_rejects_final_symlink_without_reading_outside(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside.toml"
    outside.write_text("[chat_defaults]\nstreaming = false\n", encoding="utf-8")
    outside.chmod(0o644)
    selected = tmp_path / "config.toml"
    selected.symlink_to(outside)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    _clear_config_cache()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_config_loader_rejects_missing_file_in_shared_sticky_parent(
    tmp_path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    selected = shared / "config.toml"
    fallback = tmp_path / ".tldw_cli_config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _clear_config_cache()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert not selected.exists()
    assert not fallback.exists()


def test_effective_path_preserves_symlink_spelling(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    selected = alias / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))

    assert config_module._get_effective_config_path() == selected


def test_config_loader_reports_unverified_platform_without_claiming_acl_safety(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults]\nstreaming = true\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    messages = []
    sink = config_module.logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    _clear_config_cache()
    try:
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
    finally:
        config_module.logger.remove(sink)

    text = "\n".join(messages).lower()
    assert "permission posture is unverified" in text
    assert "owner-only" not in text
    assert "acl-secure" not in text


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_failed_private_creation_clears_existing_config_cache(
    tmp_path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    selected = shared / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    config_module._CONFIG_CACHE = {"stale": True}
    config_module._CONFIG_CACHE_SOURCE = selected.absolute()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None


def test_malformed_config_defaults_are_not_cached_and_repaired_file_is_reloaded(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert loaded["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None

    target.write_text("[chat_defaults]\ntemperature = 0.17\n", encoding="utf-8")
    repaired = config_module.load_cli_config_and_ensure_existence()

    assert repaired["chat_defaults"]["temperature"] == 0.17
```

- [ ] **Step 2: Run the bootstrap tests and verify current failures**

Run: `python3 -m pytest -q Tests/test_config_private_bootstrap.py`

Expected failures:

- first creation mode is `0644` under a `0022` umask;
- existing `0644` config remains wide;
- symlink is resolved/followed;
- a missing shared-sticky target is created;
- effective path equals the resolved target instead of the lexical spelling.

- [ ] **Step 3: Replace eager resolution with lexical selection**

```python
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    create_private_text,
    lexical_path,
    open_private_binary,
)


def _get_effective_config_path() -> Path:
    """Return the lexical active CLI config path."""
    override = os.environ.get("TLDW_CONFIG_PATH")
    candidate = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
    return lexical_path(candidate)


def _application_owned_config_directory(config_path: Path) -> Path | None:
    default_path = lexical_path(DEFAULT_CONFIG_PATH)
    return default_path.parent if config_path == default_path else None


def _report_config_path_posture(result: PrivatePathResult) -> None:
    if result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
        logger.warning(
            "Config file permission posture is unverified on this platform."
        )
    elif result.status is PrivatePathStatus.HARDENED_PRIVATE:
        logger.info("Hardened the effective config file to the private posture.")
```

Import `PrivatePathResult` and `PrivatePathStatus` with the other private-path
symbols. Remove the `validate_path_simple` import from `config.py` only if no
other config function uses it.

- [ ] **Step 4: Replace first creation and existing reads**

Use the pinned stream for TOML:

```python
    config_path = _get_effective_config_path()
    if (
        _CONFIG_CACHE is not None
        and _CONFIG_CACHE_SOURCE == config_path
        and not force_reload
    ):
        return _CONFIG_CACHE

    _CONFIG_CACHE = None
    _CONFIG_CACHE_SOURCE = None
    loaded_config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    bootstrap_succeeded = False
    application_directory = _application_owned_config_directory(config_path)
    try:
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            user_config_from_file = tomllib.load(opened.stream)
        loaded_config = deep_merge_dicts(loaded_config, user_config_from_file)
        loaded_config = decrypt_config_section(loaded_config)
        bootstrap_succeeded = True
    except FileNotFoundError:
        created = create_private_text(
            config_path,
            CONFIG_TOML_CONTENT,
            application_owned_directory=application_directory,
        )
        _report_config_path_posture(created)
        loaded_config["_first_run"] = True
        bootstrap_succeeded = True
    except PrivatePathError as exc:
        if (
            application_directory is not None
            and exc.result.reason == "missing_parent"
        ):
            created = create_private_text(
                config_path,
                CONFIG_TOML_CONTENT,
                application_owned_directory=application_directory,
            )
            _report_config_path_posture(created)
            loaded_config["_first_run"] = True
            bootstrap_succeeded = True
        else:
            raise
    except tomllib.TOMLDecodeError:
        logger.opt(exception=True).error(
            "Invalid selected config; using internal defaults without caching."
        )
    except Exception:
        logger.opt(exception=True).error(
            "Config read failed; using internal defaults without caching."
        )

    if bootstrap_succeeded:
        _CONFIG_CACHE = loaded_config
        _CONFIG_CACHE_SOURCE = config_path
    return loaded_config
```

Delete the `Path.exists()` branch and the `~/.tldw_cli_config.toml` fallback.
Exceptions raised by either `create_private_text()` call propagate rather than
falling through to defaults. Preserve the existing detailed TOML/generic
diagnostics, but leave both cache fields empty on those fallbacks.

Do not route save/delete/encryption/raw-editor/export paths in this task; TASK-491 owns their single-persistence-owner conversion. The private write/read seam created here is the required dependency.

- [ ] **Step 5: Update the lexical-path assertion**

In `Tests/test_config_console_defaults.py`, change:

```python
assert config_module._get_effective_config_path() == isolated_config.resolve()
```

to:

```python
assert config_module._get_effective_config_path() == isolated_config.absolute()
```

- [ ] **Step 6: Run config-focused regression tests**

Run:

```bash
python3 -m pytest -q \
  Tests/test_config_private_bootstrap.py \
  Tests/test_config_console_defaults.py \
  Tests/test_config_delete_settings.py \
  Tests/test_config_app_config_encryption.py \
  Tests/Utils/test_config_import_hygiene.py \
  Tests/Utils/test_config_nested_settings.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit config bootstrap containment**

```bash
git add \
  tldw_chatbook/config.py \
  Tests/test_config_private_bootstrap.py \
  Tests/test_config_console_defaults.py
git commit -m "fix(security): contain config bootstrap paths"
```

---

### Task 5: Add the exact repository-root credential ignore guard

**Files:**

- Modify: `.gitignore`
- Create: `Tests/Utils/test_repository_credential_ignore.py`

**Interfaces:**

- Produces: exact root-only ignore behavior.
- Consumes: repository Git configuration only; never opens the named files.

- [ ] **Step 1: Add the failing `git check-ignore` test**

```python
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def test_exact_root_credential_filenames_are_ignored():
    assert _is_ignored("openai-api-key.txt")
    assert _is_ignored("moonshot-api-key.txt")


def test_same_filenames_in_subdirectories_are_not_covered_by_root_guard():
    assert not _is_ignored("nested/openai-api-key.txt")
    assert not _is_ignored("nested/moonshot-api-key.txt")
```

- [ ] **Step 2: Run the ignore test and confirm it fails**

Run: `python3 -m pytest -q Tests/Utils/test_repository_credential_ignore.py`

Expected: the root filename assertions fail.

- [ ] **Step 3: Add only the exact patterns**

Append:

```gitignore

# Local credential scratch files (repository root only)
/openai-api-key.txt
/moonshot-api-key.txt
```

- [ ] **Step 4: Run the ignore test**

Run: `python3 -m pytest -q Tests/Utils/test_repository_credential_ignore.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit the ignore guard**

```bash
git add .gitignore Tests/Utils/test_repository_credential_ignore.py
git commit -m "chore(security): ignore local credential scratch files"
```

---

### Task 6: Verify TASK-488 and close its Backlog record

**Files:**

- Modify: `backlog/tasks/task-488 - Establish-private-path-boundary-and-harden-config-bootstrap.md`

**Interfaces:**

- Consumes: every deliverable above.
- Produces: verified acceptance checklist and implementation notes.

- [ ] **Step 1: Run the complete focused test set**

Run:

```bash
python3 -m pytest -q \
  Tests/Utils/test_private_paths.py \
  Tests/Utils/test_repository_credential_ignore.py \
  Tests/test_config_private_bootstrap.py \
  Tests/test_config_console_defaults.py \
  Tests/test_config_delete_settings.py \
  Tests/test_config_app_config_encryption.py \
  Tests/Utils/test_config_import_hygiene.py \
  Tests/Utils/test_config_nested_settings.py
```

Expected: all tests pass with only explicit platform skips.

- [ ] **Step 2: Run static and source-boundary checks**

Run:

```bash
python3 -m compileall -q tldw_chatbook/Utils/private_paths.py tldw_chatbook/config.py
git diff --check
python3 -c "from pathlib import Path; source = Path('tldw_chatbook/Utils/private_paths.py').read_text(); forbidden = ('loguru', 'tldw_chatbook.config', 'textual', 'sqlite3'); assert not any(name in source for name in forbidden)"
```

Expected: all commands exit `0`.

- [ ] **Step 3: Run the broader config/UI regression slice**

Run:

```bash
python3 -m pytest -q \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_tools_settings_window.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
```

Expected: all tests pass. If a test asserts the removed fallback or resolved-path behavior, update only that obsolete assertion and rerun.

- [ ] **Step 4: Request a security-focused code review**

The reviewer must inspect:

- descriptor lifetime and close paths;
- final and intermediate link handling;
- sticky-parent missing-leaf behavior;
- owner/type checks before `fchmod` or read;
- no fallback creation;
- Windows `unverified_platform` wording;
- exact root-only `.gitignore` behavior.

Expected: no unresolved correctness or security findings.

- [ ] **Step 5: Update the Backlog task**

Use Backlog CLI to:

1. check all seven acceptance criteria;
2. add concise implementation notes listing the private-path module, config bootstrap integration, exact ignore guard, test evidence, and ADR-022;
3. set TASK-488 to `Done` only after all Definition of Done checks pass.

- [ ] **Step 6: Commit task completion metadata**

```bash
git add \
  "backlog/tasks/task-488 - Establish-private-path-boundary-and-harden-config-bootstrap.md"
git commit -m "docs(backlog): complete private path bootstrap task"
```

## Plan Self-Review

- Spec coverage: every TASK-488 acceptance criterion maps to Tasks 1–5; Task 6 verifies and records the evidence.
- Placeholder scan: no task contains a deferred implementation marker; every code-producing step specifies the complete behavior and concrete interface.
- Type consistency: all later steps consume `PrivatePathResult`, `PrivatePathError`, `PrivateBinaryFile`, `lexical_path`, `open_private_binary`, `secure_private_directory`, and `create_private_text` with the signatures defined above.
- Scope boundary: TASK-488 changes config selection/bootstrap only. TASK-491 remains responsible for the full persistence owner, raw editor, encryption operations, recovery, exports, backups, cache generations, and live runtime snapshots.
