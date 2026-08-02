# TASK-597 Direct Local GGUF Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely admit one explicitly selected local GGUF for the pinned transcribe.cpp runtime without copying, hashing, registering, or loading it.

**Architecture:** Keep the already reviewed bounded GGUF v3 reader and exact architecture/platform declarations in the active `gguf_admission.py` module. Preserve the prior store-facing descriptor prototype only in a private `_deferred_gguf_managed_import.py` reference module with no export, registration, import, or call path, then add one no-follow admission boundary that validates and inspects the same opened regular-file handle.

**Tech Stack:** Python 3.11+, standard-library `os`/`pathlib`/`platform`/`stat`/`struct`, existing `validate_path_simple`, dataclasses, pytest, Ruff, mypy.

---

## Preconditions and scope

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-597-local-gguf-import` on `codex/task-597-local-gguf-import`.
- Governing design: `Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md`.
- Reason: ADR-040 accepts direct local GGUF paths before managed acquisition and amends ADR-025 for this provider.
- Use `superpowers:test-driven-development` for new behavior and `superpowers:verification-before-completion` before completion claims.
- Do not add provider configuration, Textual UI, transcribe.cpp imports, inference, routing, downloads, copying, staging, activation, or artifact-store state. The preserved descriptor prototype remains dead reference code until TASK-1861 reviews it.

## File map

- Rename `tldw_chatbook/Model_Artifacts/gguf_import.py` to `tldw_chatbook/Model_Artifacts/gguf_admission.py` — bounded GGUF parsing, exact pinned compatibility, and direct-file admission only.
- Create `tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py` — private, unexported TASK-1861 reference containing the recovered store-facing descriptor prototype and no active registration/call path.
- Rename `Tests/Model_Artifacts/test_gguf_import.py` to `Tests/Model_Artifacts/test_gguf_admission.py` — retained parser/platform coverage plus new path-boundary, identity, privacy, and import-boundary tests.
- Keep `Tests/Model_Artifacts/gguf_test_helpers.py` unchanged unless a new deterministic binary fixture is genuinely required.
- Update `backlog/tasks/task-597 - Validate-explicit-local-transcribe.cpp-GGUF-files.md` through Backlog CLI before implementation and again at closeout.
- Do not modify `tldw_chatbook/Model_Artifacts/__init__.py`; TASK-604 can import the focused submodule directly.

Before Task 1, record this reviewed plan on TASK-597 with Backlog CLI and commit
the plan document plus task-file update. The implementation then starts from a
clean worktree. The TASK-597 branch base for complete-diff checks is
`f68e6b00a`.

### Task 1: Preserve the deferred prototype and enforce the active import boundary

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py`
- Modify: `Tests/Model_Artifacts/test_gguf_admission.py`
- Verify unchanged: `tldw_chatbook/Model_Artifacts/gguf_admission.py`
- Verify unchanged: `tldw_chatbook/Model_Artifacts/__init__.py`

- [ ] **Step 1: Write failing import-boundary and deferred-module tests**

Replace the weak denylist-only import test with helpers that normalize both
`ast.Import` aliases and `ast.ImportFrom` module/alias combinations. Enforce an
explicit allowlist for the active module:

```python
_ALLOWED_ADMISSION_STDLIB_ROOTS = frozenset(
    {
        "__future__",
        "os",
        "platform",
        "stat",
        "struct",
        "unicodedata",
        "dataclasses",
        "pathlib",
        "typing",
    }
)


def _normalized_import_targets(source: str) -> set[str]:
    tree = ast.parse(source)
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = f"{'.' * node.level}{node.module or ''}"
            if node.module:
                targets.add(module)
            for alias in node.names:
                targets.add(f"{module}.{alias.name}" if module else alias.name)
    return targets


def _is_allowed_admission_import(target: str) -> bool:
    if target.startswith("."):
        return target == "..Utils.path_validation" or target.startswith(
            "..Utils.path_validation."
        )
    return any(
        target == root or target.startswith(f"{root}.")
        for root in _ALLOWED_ADMISSION_STDLIB_ROOTS
    )


def test_admission_module_uses_only_approved_imports():
    source = Path(gguf.__file__).read_text(encoding="utf-8")
    assert all(
        _is_allowed_admission_import(target)
        for target in _normalized_import_targets(source)
    )
    assert not hasattr(gguf, "select_gguf_descriptor")
```

Add parameterized mutation cases proving the checker rejects:

```python
@pytest.mark.parametrize(
    "statement",
    [
        "from . import store",
        "from . import os",
        "from .dataclasses import dataclass",
        "from tldw_chatbook.Model_Artifacts import service",
        "import httpx",
        "import urllib.request",
        "import socket",
        "import ctypes",
        "import cffi",
        "from textual.app import App",
    ],
)
def test_admission_import_boundary_rejects_unapproved_import_shapes(statement):
    assert any(
        not _is_allowed_admission_import(target)
        for target in _normalized_import_targets(statement)
    )


@pytest.mark.parametrize(
    "statement",
    [
        "import os",
        "from dataclasses import dataclass",
        "from ..Utils.path_validation import validate_path_simple",
    ],
)
def test_admission_import_boundary_accepts_approved_import_shapes(statement):
    assert all(
        _is_allowed_admission_import(target)
        for target in _normalized_import_targets(statement)
    )
```

Add a source-only deferred-boundary test that expects:

- `_deferred_gguf_managed_import.py` exists and parses;
- its docstring contains both `DEFERRED` and `TASK-1861`;
- its AST assigns `__all__` to an empty tuple;
- its AST defines the recovered `runtime_constraint_admits_pinned_version`,
  `_eligible_curated_descriptor`, `_local_gguf_descriptor`, and
  `select_gguf_descriptor` functions but does not define or duplicate
  `inspect_gguf`;
- `Model_Artifacts/__init__.py` does not mention the deferred module; and
- no other production `.py` file mentions `_deferred_gguf_managed_import`.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py -k "import_boundary or deferred" -v
```

Expected: the alias-shape cases expose the old checker's blind spot and the
deferred-boundary test fails because the deferred module does not exist.

- [ ] **Step 3: Recover the store-facing prototype into the private deferred module**

Use this exact historical source as the recovery reference:

```bash
git show fd9956903^:tldw_chatbook/Model_Artifacts/gguf_import.py
```

Create `_deferred_gguf_managed_import.py` with:

- a `DEFERRED TASK-1861` reference-only docstring;
- `__all__: tuple[str, ...] = ()`;
- a local preserved `TRANSCRIBE_CPP_VERSION = "0.1.3"` and its bounded release
  tuple/constraint grammar rather than changing or depending on the active
  admission module for a constant it no longer exposes;
- the recovered `GGUFAmbiguousCuratedMatchError`, bounded runtime-constraint
  helpers, curated descriptor matching, local descriptor construction, and
  `select_gguf_descriptor` functions;
- required artifact-service imports local to this deferred module; and
- imports of `GGUFMetadata`, `normalize_platform_target`,
  `require_transcribe_cpp_architecture`, and any display helper required by the
  recovered functions from `gguf_admission` instead of duplicating the bounded
  parser.

Do not export this module from `Model_Artifacts/__init__.py`, add any call site,
restore descriptor tests as active behavior, or change `gguf_admission.py`.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/gguf_admission.py tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py Tests/Model_Artifacts/test_gguf_admission.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Model_Artifacts/gguf_admission.py tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py Tests/Model_Artifacts/test_gguf_admission.py
git diff --check 690eb29ad...HEAD
```

Expected: all commands exit zero; the focused module retains 142 or more passing
tests.

- [ ] **Step 5: Prove the deferred module is unreachable**

Run:

```bash
rg -n "_deferred_gguf_managed_import" tldw_chatbook --glob "!Model_Artifacts/_deferred_gguf_managed_import.py"
rg -n "_deferred_gguf_managed_import" tldw_chatbook/Model_Artifacts/__init__.py
rg -n "ArtifactDescriptor|select_gguf_descriptor|sha256|curated" tldw_chatbook/Model_Artifacts/gguf_admission.py
```

Expected: all three commands return no matches.

- [ ] **Step 6: Commit deferred preservation and boundary hardening**

```bash
git add tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py Tests/Model_Artifacts/test_gguf_admission.py
git commit -m "refactor: defer GGUF managed import prototype"
```

### Task 2: Add same-handle direct local file admission

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/gguf_admission.py`
- Modify: `Tests/Model_Artifacts/test_gguf_admission.py`

- [ ] **Step 1: Write failing success, privacy, and path-boundary tests**

Add tests equivalent to the following, using `make_gguf` from the existing helper:

```python
def _supported_runtime(monkeypatch):
    monkeypatch.setattr(gguf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gguf.platform, "machine", lambda: "arm64")


def test_validate_local_gguf_returns_bounded_path_private_admission(
    tmp_path, monkeypatch
):
    _supported_runtime(monkeypatch)
    model_path = tmp_path / "chosen.gguf"
    payload = make_gguf(architecture="whisper", name="Whisper Small")
    model_path.write_bytes(payload)

    result = gguf.validate_local_gguf(model_path)

    assert result.path == model_path.absolute()
    assert result.metadata.architecture == "whisper"
    assert result.source_identity.size_bytes == len(payload)
    assert result.platform_target == ("darwin", "arm64")
    assert str(model_path) not in repr(result)


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_validate_local_gguf_rejects_non_regular_sources_without_path_leak(
    tmp_path, monkeypatch, kind
):
    _supported_runtime(monkeypatch)
    secret_path = tmp_path / "private-model.gguf"
    if kind == "directory":
        secret_path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.gguf"
        target.write_bytes(make_gguf())
        try:
            secret_path.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")

    with pytest.raises(gguf.GGUFPathError) as raised:
        gguf.validate_local_gguf(secret_path)

    assert str(secret_path) not in str(raised.value)
    assert str(secret_path) not in repr(raised.value)


def test_validate_local_gguf_uses_project_validator_without_resolving_final_link(
    tmp_path, monkeypatch
):
    _supported_runtime(monkeypatch)
    model_path = tmp_path / "chosen.gguf"
    model_path.write_bytes(make_gguf())
    observed = {}

    def validate(value, require_exists=False, *, probe_existing=True):
        observed.update(
            value=value,
            require_exists=require_exists,
            probe_existing=probe_existing,
        )
        return Path(value)

    monkeypatch.setattr(gguf, "validate_path_simple", validate)
    gguf.validate_local_gguf(model_path)

    assert observed == {
        "value": model_path,
        "require_exists": False,
        "probe_existing": False,
    }
```

Also add a POSIX-only FIFO case guarded by `hasattr(os, "mkfifo")`; it must raise `GGUFPathError` without calling `os.open`, proving irregular inputs cannot block admission.

Add two small boundary cases: `.GGUF` is accepted case-insensitively while
`.gguf.tmp` is rejected, and a `ValueError` from `validate_path_simple` is
translated to `GGUFPathError` without retaining the selected path or the
validator's raw message.

- [ ] **Step 2: Write failing replacement and same-handle tests**

Add:

```python
def test_validate_local_gguf_rejects_replacement_between_lstat_and_open(
    tmp_path, monkeypatch
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    replacement = tmp_path / "replacement.gguf"
    selected.write_bytes(make_gguf(architecture="whisper"))
    replacement.write_bytes(make_gguf(architecture="parakeet"))
    real_open = gguf.os.open

    def replace_then_open(path, flags):
        replacement.replace(selected)
        return real_open(path, flags)

    monkeypatch.setattr(gguf.os, "open", replace_then_open)

    with pytest.raises(gguf.GGUFSourceChangedError):
        gguf.validate_local_gguf(selected)


def test_validate_local_gguf_inspects_the_open_descriptor(tmp_path, monkeypatch):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    selected.write_bytes(make_gguf())
    observed_fileno = []
    real_inspect = gguf.inspect_gguf

    def inspect(handle, *, file_size):
        observed_fileno.append(handle.fileno())
        assert gguf.os.fstat(handle.fileno()).st_size == file_size
        return real_inspect(handle, file_size=file_size)

    monkeypatch.setattr(gguf, "inspect_gguf", inspect)
    gguf.validate_local_gguf(selected)

    assert len(observed_fileno) == 1
    with pytest.raises(OSError):
        os.fstat(observed_fileno[0])
```

Add these two race/error cases:

```python
def test_validate_local_gguf_rechecks_name_when_nofollow_is_unavailable(
    tmp_path, monkeypatch
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    backing = tmp_path / "backing.gguf"
    selected.write_bytes(make_gguf())
    real_open = gguf.os.open
    real_flags = gguf._read_only_no_follow_flags()

    def flags_without_nofollow():
        return real_flags & ~getattr(os, "O_NOFOLLOW", 0)

    def replace_with_same_inode_symlink(path, flags):
        selected.replace(backing)
        try:
            selected.symlink_to(backing)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        return real_open(path, flags)

    monkeypatch.setattr(gguf, "_read_only_no_follow_flags", flags_without_nofollow)
    monkeypatch.setattr(gguf.os, "open", replace_with_same_inode_symlink)

    with pytest.raises(gguf.GGUFPathError):
        gguf.validate_local_gguf(selected)


def test_source_change_wins_when_inspection_also_fails(tmp_path, monkeypatch):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    selected.write_bytes(make_gguf())
    real_fstat = gguf.os.fstat
    calls = 0

    def changing_fstat(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        if calls < 2:
            return info
        values = list(info)
        values[6] = info.st_size + 1
        return os.stat_result(values)

    def malformed(handle, *, file_size):
        raise gguf.GGUFParseError("malformed test fixture")

    monkeypatch.setattr(gguf.os, "fstat", changing_fstat)
    monkeypatch.setattr(gguf, "inspect_gguf", malformed)

    with pytest.raises(gguf.GGUFSourceChangedError):
        gguf.validate_local_gguf(selected)
```

Also capture the opened descriptor in the existing replacement test and in a
separate parser-failure test. After each failure, assert `os.fstat(fd)` raises
`OSError`. Together with the success assertion above, these cover descriptor
closure on success, identity mismatch, and parser failure.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py -k "validate_local or fifo or replacement" -v
```

Expected: failures because the admission result, path errors, and `validate_local_gguf` are not implemented.

- [ ] **Step 3: Add the minimal typed admission values and errors**

Add these public values near `GGUFMetadata`:

```python
class GGUFPathError(GGUFError):
    """Raised when the selected local GGUF cannot be opened safely."""


class GGUFSourceChangedError(GGUFPathError):
    """Raised when the selected source changes during one admission."""


@dataclass(frozen=True)
class GGUFSourceIdentity:
    device: int
    inode: int
    mode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class LocalGGUFAdmission:
    path: Path = field(repr=False)
    metadata: GGUFMetadata
    source_identity: GGUFSourceIdentity
    platform_target: tuple[str, str]
```

Add imports for `os`, `platform`, `stat`, `field`, `Path`, and `validate_path_simple`. Keep the module standard-library-only apart from the existing project path validator.

- [ ] **Step 4: Implement the no-follow, same-handle boundary**

Implement these helpers and behavior below `inspect_gguf`:

```python
def _source_identity(info: os.stat_result) -> GGUFSourceIdentity:
    return GGUFSourceIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        size_bytes=info.st_size,
        modified_ns=info.st_mtime_ns,
        changed_ns=info.st_ctime_ns,
    )


def _read_only_no_follow_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
```

`validate_local_gguf(path: str | Path) -> LocalGGUFAdmission` must, in order:

1. call `validate_path_simple(path, probe_existing=False)` and convert the lexical path to an absolute `Path` without resolving the final component;
2. require a case-insensitive `.gguf` suffix;
3. `os.lstat` the name and reject symlinks and non-regular files before open;
4. open with `_read_only_no_follow_flags`, wrapping all `OSError` details in a path-free `GGUFPathError` raised `from None`;
5. compare the initial `lstat` identity with the first `os.fstat`, including type, device, inode, size, modification time, and change time;
6. immediately `lstat` the pathname again after open, reject a symlink or irregular name, and require that named identity to still equal the opened identity; this is mandatory even when `O_NOFOLLOW` is unavailable;
7. call `inspect_gguf` on that same binary handle with the opened size;
8. in a `finally` around inspection, compare a second `fstat` with the first so an observable mutation raises `GGUFSourceChangedError` even when parsing also raises;
9. close the handle on success, identity mismatch, parse failure, and every other exit;
10. call `normalize_platform_target(platform.system(), platform.machine())`; and
11. return `LocalGGUFAdmission` with the absolute lexical path excluded from `repr`.

Do not resolve the path, hash tensor bytes, preserve an open descriptor, or claim the snapshot is a lease.

- [ ] **Step 5: Run the focused admission tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py -k "validate_local or fifo or replacement or identity or path" -v
```

Expected: all focused tests pass.

- [ ] **Step 6: Run the entire TASK-597 test module**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit direct-file admission**

```bash
git add tldw_chatbook/Model_Artifacts/gguf_admission.py Tests/Model_Artifacts/test_gguf_admission.py
git commit -m "feat: admit explicit local GGUF files"
```

### Task 3: Verify scope, quality, and Backlog completion

**Files:**
- Modify: `backlog/tasks/task-597 - Validate-explicit-local-transcribe.cpp-GGUF-files.md` through Backlog CLI
- Verify: `tldw_chatbook/Model_Artifacts/gguf_admission.py`
- Verify: `tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py`
- Verify: `Tests/Model_Artifacts/test_gguf_admission.py`

- [ ] **Step 1: Run focused and neighboring regression tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_admission.py Tests/Utils/test_path_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run static checks on only the changed Python files**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/gguf_admission.py tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py Tests/Model_Artifacts/test_gguf_admission.py Tests/Model_Artifacts/gguf_test_helpers.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Model_Artifacts/gguf_admission.py tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py Tests/Model_Artifacts/test_gguf_admission.py Tests/Model_Artifacts/gguf_test_helpers.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/Model_Artifacts/gguf_admission.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Model_Artifacts/_deferred_gguf_managed_import.py
git diff --check f68e6b00a...HEAD
git status --short
```

Expected: every command exits zero and `git status --short` is empty.

- [ ] **Step 3: Run the repository-required full test suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

Expected: the full suite passes with no failures.

- [ ] **Step 4: Re-run scope and privacy scans**

```bash
rg -n "ArtifactDescriptor|ModelArtifactService|sha256|copy|install|activation|textual|ggml|httpx" tldw_chatbook/Model_Artifacts/gguf_admission.py
rg -n "gguf_import" tldw_chatbook Tests
rg -n "_deferred_gguf_managed_import" tldw_chatbook --glob "!Model_Artifacts/_deferred_gguf_managed_import.py"
```

Expected: all commands return no matches.

- [ ] **Step 5: Request independent code review**

Use `superpowers:requesting-code-review` against the complete TASK-597 diff. Address all Critical and Important findings, rerun the affected checks, and repeat review until approved.

- [ ] **Step 6: Close the Backlog task only after all evidence is fresh**

Use Backlog CLI to:

- check all seven acceptance criteria;
- add concise Implementation Notes naming the rename, retained parser/compatibility boundary, private deferred TASK-1861 prototype, same-handle admission, path privacy, tests, and ADR-040;
- set TASK-597 to `Done` only after the required tests, static checks, and review pass.

- [ ] **Step 7: Commit closeout metadata and verify the committed branch**

```bash
git add "backlog/tasks/task-597 - Validate-explicit-local-transcribe.cpp-GGUF-files.md"
git commit -m "docs: close TASK-597 local GGUF admission"
git diff --check f68e6b00a...HEAD
git status --short
```

## Completion boundary

TASK-597 is complete when a caller can validate one explicit local GGUF and receive bounded admission evidence suitable for TASK-604. The preserved TASK-1861 prototype remains private and unreachable. A user still cannot transcribe through transcribe.cpp until TASK-604 adds configuration, Library batch wiring, the pinned native provider, and persisted results.
