# Security Hardening Cluster Implementation Plan (TASK-330 + 331#1/#2 + 332)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four independent defensive-security fixes: git-clone transport/argument-injection hardening, a real file-tool sandbox root, pickle→json (+ a crash) in the tool-result cache, and surfacing the silently-unenforced eval-sandbox memory limit.

**Architecture:** Each fix is confined to one subsystem. New git-URL validators live in `Utils/input_validation.py`; the other fixes are in-place edits to `Media/local_media_reading_service.py`, `Tools/file_operation_tools.py`, `Tools/tool_executor.py`, and `Evals/specialized_runners.py`.

**Tech Stack:** Python 3.11, pytest, unittest.mock. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-24-security-hardening-cluster-design.md` (committed `b0cc04ab3`). TASK-331 **#3** (permission/confirmation gate) is DEFERRED to its own spec — do NOT implement it here.

## Global Constraints

1. **Git URL allowlist:** explicit scheme ∈ `{"https","ssh"}` ONLY. Reject `ext`, `file`, `fd`, `git`, `http`, scp-shorthand (`user@host:path`), absent scheme, whitespace/backslash/control chars, and a leading `-`. Same env on the clone: `GIT_ALLOW_PROTOCOL="https:ssh"`, `GIT_PROTOCOL_FROM_USER="0"`, plus a literal `"--"` argv separator before the URL.
2. **Sandbox root** default = `get_user_data_dir() / "tool_sandbox"` (the real API; `USER_DATA_DIR` does NOT exist), config-overridable via `[tools] file_sandbox_root`, `mkdir(parents=True, exist_ok=True)`.
3. **Cache serializer** = `json` (NOT pickle); no migration needed (no legacy pickle files exist — the path crashed before ever writing one). The existing try/except around load/save already degrades gracefully.
4. **RLIMIT surfacing** is PARENT-side (`platform.system()=="Darwin"` → not enforced); one-time WARNING + `results["sandbox_warnings"]`. Only `RLIMIT_AS` is the gap; do NOT touch `RLIMIT_NPROC`.
5. **DEFERRED (do not touch):** TASK-331 #3 permission gate; the `file://`-local-read vector (follow-up).
6. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-sec-hardening` (branch `feat/security-hardening-cluster`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only each task's listed files, never `-A`.
7. **Line numbers below are as-of origin/dev `5a402cbb7` — re-verify with `grep -n` before editing; the target text is authoritative.**

**Baseline:** `Tools/` and `Media/` and the git-clone path have essentially no existing tests for the touched code; `Tests/Evals/test_code_execution_security.py` exists. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/test_code_execution_security.py -q` before Task 5 and note any pre-existing failures — report, don't fix.

---

### Task 1: Git URL + ref validators (`Utils/input_validation.py`)

**Files:**
- Modify: `tldw_chatbook/Utils/input_validation.py` (add two functions near `validate_url`; `ValidationError` already exists at ~L455, `urlparse` imported at L9, `re` at L5)
- Test: `Tests/Utils/test_git_url_validation.py` (create)

**Interfaces:**
- Produces: `validate_git_repo_url(url: str) -> None` (raises `ValidationError` on reject) and `validate_git_ref(ref: str) -> None` (raises `ValidationError` on reject). Both no-op-return when valid.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Utils/test_git_url_validation.py`:

```python
import pytest

from tldw_chatbook.Utils.input_validation import (
    ValidationError,
    validate_git_repo_url,
    validate_git_ref,
)


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo.git",
    "https://gitlab.example.com/a/b",
    "ssh://git@github.com/owner/repo.git",
])
def test_valid_repo_urls_pass(url):
    validate_git_repo_url(url)  # no raise


@pytest.mark.parametrize("url", [
    "ext::sh -c 'touch /tmp/pwn'",          # RCE transport
    "ext::git-upload-pack",
    "file:///etc/passwd",
    "file::/etc/passwd",
    "fd::17",
    "git://example.com/repo.git",           # unauthenticated transport, not allowlisted
    "http://example.com/repo.git",          # http not allowlisted (https only)
    "git@github.com:owner/repo.git",        # scp-shorthand rejected (ambiguous) — use ssh://
    "-upload-pack=/bin/sh",                 # leading dash (arg injection)
    "--upload-pack=x",
    "  https://x/y ",                        # whitespace
    "https://exa\\mple.com/y",              # backslash
    "/local/path/repo",                      # no scheme
    "repo",                                  # no scheme
    "",                                       # empty
])
def test_malicious_or_disallowed_repo_urls_raise(url):
    with pytest.raises(ValidationError):
        validate_git_repo_url(url)


@pytest.mark.parametrize("ref", ["main", "v1.2.3", "feature/new-thing", "release_1"])
def test_valid_refs_pass(ref):
    validate_git_ref(ref)  # no raise


@pytest.mark.parametrize("ref", [
    "--upload-pack=/bin/sh",   # leading dash
    "-b",
    "a b",                      # whitespace
    "a\tb",
    "..",                       # traversal-ish / invalid ref
    "a..b",
    "a\nb",                     # control char
    "",
])
def test_malicious_or_invalid_refs_raise(ref):
    with pytest.raises(ValidationError):
        validate_git_ref(ref)
```

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_git_url_validation.py -q`
Expected: ImportError (`validate_git_repo_url`/`validate_git_ref` don't exist).

- [ ] **Step 3: Implement** — add to `tldw_chatbook/Utils/input_validation.py` (place after `validate_url`, before `ValidationError` is fine — `ValidationError` is referenced at call time, not import time; if the linter prefers, put them after the `ValidationError` class):

```python
_GIT_ALLOWED_SCHEMES = frozenset({"https", "ssh"})
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def validate_git_repo_url(url: str) -> None:
    """Validate a git clone repo URL against a strict transport allowlist.

    Only explicit ``https://`` and ``ssh://`` schemes are accepted. scp-style
    shorthand (``user@host:path``) is rejected because its no-scheme,
    colon-before-slash shape is ambiguous with git's ``ext::``/``fd::`` custom
    transports and with leading-dash argument-injection payloads; use
    ``ssh://git@host/path`` instead. Rejects custom transports (``ext``,
    ``file``, ``fd``, ``git``, ...), whitespace/backslash/control characters,
    a leading ``-`` (git-option injection), and anything without a scheme.

    Args:
        url: The candidate repo URL.

    Raises:
        ValidationError: If ``url`` is not an allowlisted, well-formed git URL.
    """
    if not isinstance(url, str) or not url:
        raise ValidationError("repo_url must be a non-empty string")
    if url != url.strip() or any(c.isspace() for c in url):
        raise ValidationError("repo_url must not contain whitespace")
    if "\\" in url or any(ord(c) < 0x20 for c in url):
        raise ValidationError("repo_url must not contain backslashes or control characters")
    if url.startswith("-"):
        raise ValidationError("repo_url must not start with '-' (git-option injection)")
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValidationError(f"repo_url is not a parseable URL: {exc}")
    if parsed.scheme.lower() not in _GIT_ALLOWED_SCHEMES:
        raise ValidationError(
            f"repo_url scheme {parsed.scheme!r} is not allowed; "
            "only https:// and ssh:// git URLs are permitted"
        )
    if not parsed.hostname:
        raise ValidationError("repo_url must include a host")


def validate_git_ref(ref: str) -> None:
    """Validate a git branch/ref name for use as a ``--branch`` value.

    Rejects a leading ``-`` (git-option injection), whitespace/control chars,
    ``..``, and any character outside ``[A-Za-z0-9._/-]``.

    Args:
        ref: The candidate branch/ref name.

    Raises:
        ValidationError: If ``ref`` is unsafe or not a well-formed ref name.
    """
    if not isinstance(ref, str) or not ref:
        raise ValidationError("ref must be a non-empty string")
    if ref.startswith("-"):
        raise ValidationError("ref must not start with '-' (git-option injection)")
    if ".." in ref:
        raise ValidationError("ref must not contain '..'")
    if not _GIT_REF_RE.match(ref):
        raise ValidationError(
            "ref may only contain letters, digits, '.', '_', '/', '-'"
        )
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_git_url_validation.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/input_validation.py Tests/Utils/test_git_url_validation.py
git commit -m "feat(security): git repo-url + ref validators (transport allowlist, no arg injection) [TASK-330]"
```

---

### Task 2: Wire `_clone_git_repository` (`Media/local_media_reading_service.py`)

**Files:**
- Modify: `tldw_chatbook/Media/local_media_reading_service.py` (`_clone_git_repository`, ~L4246)
- Test: `Tests/Media/test_git_clone_hardening.py` (create)

**Interfaces:**
- Consumes: `validate_git_repo_url`, `validate_git_ref`, `ValidationError` from `tldw_chatbook.Utils.input_validation`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Media/test_git_clone_hardening.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Utils.input_validation import ValidationError


def _clone(repo_url, ref=None):
    return LocalMediaReadingService._clone_git_repository(
        repo_url, Path("/tmp/checkout_target"), ref=ref
    )


@pytest.mark.parametrize("repo_url", [
    "ext::sh -c 'touch /tmp/pwn'",
    "file:///etc/passwd",
    "-upload-pack=/bin/sh",
    "git@github.com:owner/repo.git",
])
def test_malicious_repo_url_rejected_before_subprocess(repo_url):
    with patch("subprocess.run") as mock_run:
        with pytest.raises((ValidationError, ValueError, RuntimeError)):
            _clone(repo_url)
        mock_run.assert_not_called()


def test_malicious_ref_rejected_before_subprocess():
    with patch("subprocess.run") as mock_run:
        with pytest.raises((ValidationError, ValueError, RuntimeError)):
            _clone("https://github.com/owner/repo.git", ref="--upload-pack=/bin/sh")
        mock_run.assert_not_called()


def test_valid_clone_uses_separator_and_restricted_env():
    ok = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=ok) as mock_run:
        _clone("https://github.com/owner/repo.git", ref="main")
    assert mock_run.call_count == 1
    argv = mock_run.call_args[0][0]
    # "--" separator immediately precedes the repo URL
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1] == "https://github.com/owner/repo.git"
    assert "--branch" in argv and argv[argv.index("--branch") + 1] == "main"
    env = mock_run.call_args[1]["env"]
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh"
    assert env["GIT_PROTOCOL_FROM_USER"] == "0"
```

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Media/test_git_clone_hardening.py -q`
Expected: FAIL — malicious inputs currently reach `subprocess.run` (no validation), and the valid case has no `--`/env.

- [ ] **Step 3: Implement** — replace the body of `_clone_git_repository` (keep the `@staticmethod`/signature):

```python
    @staticmethod
    def _clone_git_repository(
        repo_url: str, checkout_path: Path, *, ref: Any = None
    ) -> None:
        import os
        import subprocess

        from ..Utils.input_validation import (
            validate_git_repo_url,
            validate_git_ref,
        )

        validate_git_repo_url(repo_url)
        command = ["git", "clone", "--depth", "1"]
        if ref:
            validate_git_ref(str(ref))
            command.extend(["--branch", str(ref)])
        # `--` separates options from positionals so a hostile URL/path can
        # never be parsed as a git option; the env restricts git to https/ssh
        # transports even across redirects/submodules (blocks ext::/file:: etc.).
        command.extend(["--", repo_url, str(checkout_path)])
        clone_env = {
            **os.environ,
            "GIT_ALLOW_PROTOCOL": "https:ssh",
            "GIT_PROTOCOL_FROM_USER": "0",
        }
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=clone_env
        )
        if completed.returncode != 0:
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "git clone failed"
            )
            raise RuntimeError(message)
```

(`Any` is already imported in this module — it's used in the existing signature.)

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Media/test_git_clone_hardening.py -q`
Expected: all pass.

- [ ] **Step 5: Regression-check the existing media suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Media/test_local_media_reading_service.py -q`
Expected: still green (the existing git-source test uses a local dir and takes the filesystem-copy branch, never reaching `_clone_git_repository`). If it regresses, your edit broke a contract — fix the edit.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Media/local_media_reading_service.py Tests/Media/test_git_clone_hardening.py
git commit -m "fix(security): harden git clone against ext:: transport + arg injection [TASK-330]"
```

---

### Task 3: Real file-tool sandbox root (`Tools/file_operation_tools.py`)

**Files:**
- Modify: `tldw_chatbook/Tools/file_operation_tools.py` (add helper; fix 3 `validate_path` calls at ~L63, ~L168, ~L334)
- Test: `Tests/Tools/test_file_tool_sandbox.py` (create; add `Tests/Tools/__init__.py` if the dir isn't a package — check `ls Tests/Tools/`)

**Interfaces:**
- Consumes: `validate_path` (already imported: `from ..Utils.path_validation import validate_path`); `get_user_data_dir` / `get_cli_setting` from `..config`.
- Produces: module-level `_tool_sandbox_root() -> Path`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Tools/test_file_tool_sandbox.py`:

```python
import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.Tools import file_operation_tools as fot


def test_sandbox_root_is_real_dir_not_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path / "tool_sandbox"))
    root = fot._tool_sandbox_root()
    assert root == (tmp_path / "tool_sandbox").resolve()
    assert root.is_dir()  # created
    assert root.name != "file" and root.name != "directory"


def test_read_file_rejects_traversal_outside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    # write a secret OUTSIDE the sandbox
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="../secret.txt"))
    assert result.get("success") is False or "error" in result  # rejected, not leaked
    assert "top secret" not in str(result)


def test_read_file_reads_inside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "hello.txt").write_text("inside content")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="hello.txt"))
    assert "inside content" in str(result)
```

(Verify `ReadFileTool().execute`'s success/error shape by reading the current method; adjust the assertions to its actual return keys — it returns a dict with either content or an `error`/`success` field. The load-bearing assertions are: traversal → not leaked; in-sandbox → content returned.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_file_tool_sandbox.py -q`
Expected: FAIL — `_tool_sandbox_root`/`_resolve_sandbox_config` don't exist; and with the literal `"file"` root the in-sandbox read fails.

- [ ] **Step 3: Implement** — add near the top of `tldw_chatbook/Tools/file_operation_tools.py` (after the imports):

```python
def _resolve_sandbox_config() -> str:
    """Return the configured sandbox root string (indirection for tests)."""
    from ..config import get_cli_setting, get_user_data_dir

    default_root = str(get_user_data_dir() / "tool_sandbox")
    return get_cli_setting("tools", "file_sandbox_root", default_root) or default_root


def _tool_sandbox_root() -> Path:
    """Resolve + create the file-tool sandbox root.

    The file tools confine all reads/writes/listings under this directory.
    Defaults to ``<user data dir>/tool_sandbox``; override with
    ``[tools] file_sandbox_root`` in config.toml.
    """
    root = Path(_resolve_sandbox_config()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
```

Then change the three call sites:
- `validate_path(file_path, "file")` (ReadFileTool, ~L63) → `validate_path(file_path, _tool_sandbox_root())`
- `validate_path(directory_path, "directory")` (ListDirectoryTool, ~L168) → `validate_path(directory_path, _tool_sandbox_root())`
- `validate_path(file_path, "file")` (WriteFileTool, ~L334) → `validate_path(file_path, _tool_sandbox_root())`

(Confirm `get_cli_setting`'s 3-arg form `get_cli_setting("tools", "file_sandbox_root", default)` matches its signature — the codebase uses both a 3-arg `(section, key, default)` and a dotted form; grep an existing 3-arg call to confirm. If only the section-dict form exists, use `get_cli_setting("tools", {}).get("file_sandbox_root", default_root)`.)

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_file_tool_sandbox.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/file_operation_tools.py Tests/Tools/test_file_tool_sandbox.py
git commit -m "fix(security): file tools use a real configured sandbox root (not literal strings) [TASK-331]"
```

---

### Task 4: pickle→json + fix the get_tool_executor crash (`Tools/tool_executor.py`)

**Files:**
- Modify: `tldw_chatbook/Tools/tool_executor.py` (`_load_from_disk` ~L205, `_save_to_disk` ~L227, `get_tool_executor` import ~L633)
- Test: `Tests/Tools/test_tool_cache_json.py` (create)

**Interfaces:**
- Consumes: `get_user_data_dir` from `..config`. `json` is already imported at the top of the module (no new import); `pickle` import (~L10) becomes unused — remove it.

- [ ] **Step 1: Write the failing test**

Create `Tests/Tools/test_tool_cache_json.py`:

```python
import asyncio
import json
import time
from pathlib import Path

import pytest

from tldw_chatbook.Tools.tool_executor import ToolResultCache


def test_cache_round_trips_through_json(tmp_path):
    persist = tmp_path / "tool_results.cache"

    async def run():
        c1 = ToolResultCache(persist_path=persist)
        await c1.set("mytool", {"a": 1}, {"result": "ok", "n": 3}, ttl=3600)
        await c1._save_to_disk()
        # the on-disk file is valid JSON (not pickle)
        raw = persist.read_text(encoding="utf-8")
        json.loads(raw)  # would raise if pickle
        c2 = ToolResultCache(persist_path=persist)
        await c2._load_from_disk()
        got = await c2.get("mytool", {"a": 1})
        return raw, got

    raw, got = asyncio.run(run())
    assert got == {"result": "ok", "n": 3}
    assert "\x80" not in raw  # not a pickle opcode stream


def test_corrupt_cache_file_degrades_gracefully(tmp_path):
    persist = tmp_path / "tool_results.cache"
    persist.write_text("not valid json {{{", encoding="utf-8")

    async def run():
        c = ToolResultCache(persist_path=persist)
        await c._load_from_disk()  # must not raise
        return await c.get("anything", {})

    assert asyncio.run(run()) is None


def test_get_tool_executor_with_cache_enabled_does_not_importerror(monkeypatch):
    # Previously `from ..config import USER_DATA_DIR` raised ImportError here.
    from tldw_chatbook.Tools import tool_executor as te

    monkeypatch.setattr(
        te, "get_tool_executor", te.get_tool_executor
    )  # ensure symbol import path is exercised
    import tldw_chatbook.config as cfg

    monkeypatch.setattr(
        cfg, "get_cli_setting",
        lambda section, key=None, default=None: (
            {"cache_enabled": True, "cache_persist": True} if section == "tools" and key is None
            else default
        ),
    )
    te.reset_tool_executor() if hasattr(te, "reset_tool_executor") else None
    ex = te.get_tool_executor()  # must not raise ImportError
    assert ex is not None
```

(The third test's exact mechanics depend on how `get_tool_executor` reads config and whether a reset helper exists — read the module and adapt: the ONLY load-bearing assertion is "calling `get_tool_executor()` with cache enabled does not raise `ImportError`." If a global singleton makes it hard to force a fresh build, assert instead that `import tldw_chatbook.Tools.tool_executor` + a direct call of the persist-path code path resolves `get_user_data_dir` without `ImportError`.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_tool_cache_json.py -q`
Expected: FAIL — the on-disk file is pickle (json.loads raises), and/or cache-enabled build raises `ImportError`.

- [ ] **Step 3a: Swap pickle→json in the cache.** In `_load_from_disk` change:
```python
                with open(self.persist_path, "rb") as f:
                    loaded_cache = pickle.load(f)
```
to:
```python
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    loaded_cache = json.load(f)
```
(The subsequent `for key, (result, expiry_time) in loaded_cache.items():` unpacks a 2-element JSON list identically to a tuple — no change needed.)

In `_save_to_disk` change:
```python
            with open(self.persist_path, "wb") as f:
                pickle.dump(cache_copy, f)
```
to:
```python
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(cache_copy, f)
```
(The existing `try/except Exception` around `_save_to_disk` already logs+skips if a value is non-JSON-serializable, degrading to in-memory-only — no extra code needed. Tuples serialize as JSON arrays.)

Remove the now-unused `import pickle` (~L10).

- [ ] **Step 3b: Fix the crash in `get_tool_executor`.** Change (~L633):
```python
            from ..config import USER_DATA_DIR

            cache_dir = Path(USER_DATA_DIR) / "tool_cache"
```
to:
```python
            from ..config import get_user_data_dir

            cache_dir = get_user_data_dir() / "tool_cache"
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_tool_cache_json.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/tool_executor.py Tests/Tools/test_tool_cache_json.py
git commit -m "fix(security): tool-result cache uses json not pickle; fix USER_DATA_DIR ImportError crash [TASK-331]"
```

---

### Task 5: Surface RLIMIT_AS non-enforcement (`Evals/specialized_runners.py`)

**Files:**
- Modify: `tldw_chatbook/Evals/specialized_runners.py` (add `import platform` + `_memory_limit_enforced()`; `_execute_code` docstring ~L258-272 + results dict ~L276-282 + a warning emission)
- Test: `Tests/Evals/test_rlimit_surfacing.py` (create)

**Interfaces:**
- Produces: module-level `_memory_limit_enforced() -> bool`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/test_rlimit_surfacing.py`:

```python
import pytest

from tldw_chatbook.Evals import specialized_runners as sr


def test_memory_limit_not_enforced_on_darwin(monkeypatch):
    monkeypatch.setattr(sr.platform, "system", lambda: "Darwin")
    assert sr._memory_limit_enforced() is False


def test_memory_limit_enforced_on_linux(monkeypatch):
    monkeypatch.setattr(sr.platform, "system", lambda: "Linux")
    assert sr._memory_limit_enforced() is True


def test_warns_and_records_when_unenforced(monkeypatch):
    # A helper that builds the results dict + appends the sandbox warning when
    # memory isn't enforced. On Darwin it must surface a warning entry.
    monkeypatch.setattr(sr, "_memory_limit_enforced", lambda: False)
    warnings = sr._sandbox_warnings()
    assert any("memory" in w.lower() for w in warnings)


def test_no_warning_when_enforced(monkeypatch):
    monkeypatch.setattr(sr, "_memory_limit_enforced", lambda: True)
    assert sr._sandbox_warnings() == []
```

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/test_rlimit_surfacing.py -q`
Expected: FAIL — `_memory_limit_enforced` / `_sandbox_warnings` don't exist (and `platform` not imported).

- [ ] **Step 3a: Add `import platform`** near the other stdlib imports (~L19-26) in `specialized_runners.py`, and add module-level helpers (after imports / near the top-level of the module):

```python
_MEMORY_LIMIT_WARNED = False


def _memory_limit_enforced() -> bool:
    """Whether the eval sandbox's RLIMIT_AS memory cap is enforced here.

    On macOS/BSD ``setrlimit(RLIMIT_AS, ...)`` raises ``ValueError`` (the
    address-space limit is aliased to RSS and cannot be lowered), so the
    256MB memory cap silently does not apply. Time is still bounded by the
    subprocess wall-clock timeout on every platform; only peak memory is
    unbounded where this returns False.
    """
    return platform.system() != "Darwin"


def _sandbox_warnings() -> list:
    """One-time-logged, per-result warnings about non-enforced sandbox limits."""
    global _MEMORY_LIMIT_WARNED
    warnings: list = []
    if not _memory_limit_enforced():
        msg = (
            "eval sandbox: RLIMIT_AS memory limit is NOT enforced on this "
            "platform (macOS/BSD); model-generated code is bounded by the "
            "execution timeout but not by peak memory"
        )
        warnings.append(msg)
        if not _MEMORY_LIMIT_WARNED:
            logger.warning(msg)
            _MEMORY_LIMIT_WARNED = True
    return warnings
```

- [ ] **Step 3b: Wire into `_execute_code`.** In the `results = {...}` init (~L276-282) add the key:
```python
            "execution_time": 0.0,
            "sandbox_warnings": _sandbox_warnings(),
```
And fix the docstring (~L258-272) — change the security-measures list so "memory" is qualified:
```python
        Security measures implemented:
        - CPU time + wall-clock timeout (cross-platform)
        - Memory cap via RLIMIT_AS -- best-effort; NOT enforced on macOS/BSD
          (setrlimit(RLIMIT_AS) raises ValueError there), surfaced via
          results["sandbox_warnings"]; see _memory_limit_enforced()
        - Process/file-descriptor/file-write limits (RLIMIT_NPROC/NOFILE/FSIZE)
        - Dangerous builtins disabled (eval, exec, compile, etc.)
        - Restricted environment with minimal PATH
        - Runs in temporary directory with restricted HOME
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/test_rlimit_surfacing.py Tests/Evals/test_code_execution_security.py -q`
Expected: the new tests pass; the existing security suite stays green (you didn't change the generated RLIMIT code, only added parent-side surfacing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/specialized_runners.py Tests/Evals/test_rlimit_surfacing.py
git commit -m "fix(security): surface non-enforced RLIMIT_AS memory limit on macOS; correct sandbox docstring [TASK-332]"
```

---

### Task 6: Backlog bookkeeping + follow-ups

**Files:**
- Modify: `backlog/tasks/task-330 - Harden-git-clone-against-transport-and-argument-injection.md`
- Modify: `backlog/tasks/task-331 - Tool-executor-security-hardening.md`
- Modify: `backlog/tasks/task-332 - Eval-runner-resource-limits-robust-on-macOS.md`
- Create: two follow-up task files (IDs via the collision-safe scan)

- [ ] **Step 1: Assign follow-up IDs.** Scan both namespaces against origin/dev + working tree:
```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-sec-hardening && git fetch -q origin
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
import os, re, subprocess
ids=set()
for name in subprocess.run(["git","ls-tree","-r","--name-only","origin/dev","backlog/"],capture_output=True,text=True).stdout.splitlines():
    m=re.search(r"task-(\d+)",name)
    if m: ids.add(int(m.group(1)))
for root in ("backlog/tasks","backlog/drafts"):
    if os.path.isdir(root):
        for name in os.listdir(root):
            m=re.search(r"task-(\d+)",name)
            if m: ids.add(int(m.group(1)))
print("next ids:", max(ids)+1, max(ids)+2)
EOF
```
Use the two printed ids for the follow-ups below (call them `<GATE_ID>` and `<FILEURL_ID>`).

- [ ] **Step 2: Close TASK-330** — read it; check all ACs (`- [ ]`→`- [x]`), `status: Done`, add `## Implementation Notes`: validators `validate_git_repo_url`/`validate_git_ref` in `Utils/input_validation.py` (https/ssh explicit-scheme allowlist; reject ext/file/fd/git/scp-shorthand/leading-dash/whitespace); `_clone_git_repository` now validates repo_url+ref, inserts `--`, and runs with `GIT_ALLOW_PROTOCOL=https:ssh` + `GIT_PROTOCOL_FROM_USER=0`; tests cover ext::/file::/leading-dash/malicious-ref rejection before subprocess. Note the follow-up (`file://`-resolves-to-local-dir read vector → task-`<FILEURL_ID>`).

- [ ] **Step 3: Close TASK-332** — check ACs, `status: Done`, `## Implementation Notes`: only `RLIMIT_AS` silently no-ops on macOS (RLIMIT_NPROC works); parent-side `_memory_limit_enforced()` (False on Darwin) → one-time WARNING + `results["sandbox_warnings"]`; docstring corrected; the pre-existing memory-exhaustion test was vacuous (static scan blocks its payload).

- [ ] **Step 4: Partially close TASK-331** — check ACs #1 and #2 (`- [x]`), LEAVE #3 unchecked, `status: Done` with `## Implementation Notes` stating: AC#1 (real sandbox root via `[tools] file_sandbox_root` default `get_user_data_dir()/tool_sandbox`) and AC#2 (pickle→json in ToolResultCache + fixed the `USER_DATA_DIR` ImportError that crashed `get_tool_executor` when caching was enabled) are done; **AC#3 (the confirmation/permission gate) is split to a dedicated follow-up, task-`<GATE_ID>`**, because it is a cross-system + UI integration (two call sites, risk-tagging the `Tool` ABC, reusing `MCPPermissionStore`/`resolve_effective_state` + `ChatApprovalCard`) far larger than this bundle.

- [ ] **Step 5: File the two follow-ups** (copy an existing task file's frontmatter format):
  - `backlog/tasks/task-<GATE_ID> - Wire-built-in-tool-executor-into-MCP-permission-gate.md` — labels `tools,security`; priority medium; deps `[task-331]`. Description: built-in fs/mutating tools (`write_file`/`create_note`/`update_note`, all default-off) auto-execute on model tool_calls with NO allow/ask/deny gate. Wire `ToolExecutor` (Site A: `Event_Handlers/…execute_tool_calls`, main-loop) and/or the agent-runtime `BuiltinToolProvider` (Site B: worker-thread) into the existing `MCP/permission_store.py` model (`resolve_effective_state`/`EffectiveToolState`, kill switch, `HIGH_RISK_TAGS`), add a risk-tag field to the `Tool` ABC and tag the mutating tools, and reuse `Widgets/Chat_Widgets/chat_approval_card.py` for the "ask" confirmation. AC: a mutating built-in tool requested by the model is gated allow/ask/deny before execution, with a test. NOTE the Task-3 interaction: the sandbox fix made these tools functional-within-a-sandbox, so this gate is the intended protection.
  - `backlog/tasks/task-<FILEURL_ID> - Restrict-file-scheme-local-read-in-git-ingestion-source.md` — labels `security,media`; priority low; deps `[task-330]`. Description: `_local_git_repository_path`/`_sync_git_repository_source_items` resolve a `file://`/no-scheme `repo_url` pointing at a real local dir and read it directly (skipping the clone) — a secondary local-file-read vector not covered by the clone-time transport allowlist. AC: local-path ingestion sources are restricted/validated (or explicitly opt-in).

- [ ] **Step 6: Commit**
```bash
git add "backlog/tasks/task-330 - Harden-git-clone-against-transport-and-argument-injection.md" "backlog/tasks/task-331 - Tool-executor-security-hardening.md" "backlog/tasks/task-332 - Eval-runner-resource-limits-robust-on-macOS.md" backlog/tasks/task-<GATE_ID>*.md backlog/tasks/task-<FILEURL_ID>*.md
git commit -m "docs(backlog): close TASK-330/332, partial TASK-331 (#1/#2); file gate + file:// follow-ups"
```

---

## Post-plan notes for the controller (not for task implementers)

- SDD models: Task 1 (isolated validators, complete code) → cheapest tier; Tasks 2-5 (in-place edits on real files) → mid tier; Task 6 (backlog) → cheapest tier. Final whole-branch review → most capable.
- The three fs tools become functional-within-a-sandbox after Task 3 while the gate is deferred — this is the intended, on-the-record tradeoff (default-off + narrow sandbox); the final review should confirm write_file's blast radius is confined to the sandbox root.
- Re-verify every `~Lnnn` against the current file before editing (concurrent dev drift).

