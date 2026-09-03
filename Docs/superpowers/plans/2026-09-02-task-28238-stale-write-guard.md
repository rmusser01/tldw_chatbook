# TASK-28238 Phase 1: Stale-Write Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** fs_write/fs_edit/fs_patch refuse when the target changed on disk since this run last read it, so concurrent fleet children can no longer silently clobber each other on the shared working tree.

**Architecture:** A thread-safe read-ledger keyed by `(run_id, canonical_path)` lives in a new small module and is owned by `LocalToolProvider`. The provider records a whole-file hash (or ABSENT) at fs_read dispatch by resolving the path itself; at write dispatch it refuses stale targets — fs_write by injecting the existing atomic CAS (`expected_sha256`/`expected_absent`), fs_edit/fs_patch by a provider pre-hash — and updates the ledger after successful writes.

**Tech Stack:** Python ≥3.11, stdlib only (hashlib, threading, dataclasses). No new dependencies.

**Spec:** `backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md` (phase 1 sections; read it first — it carries the review-hardened corrections this plan implements).

## Global Constraints

- Work on a NEW branch off `origin/dev` in a clean worktree (this repo's dev moves fast; other sessions' dirty files live in shared checkouts). Branch name: `feat/task-28238-stale-write-guard`.
- Test runner: the worktree venv only — `.venv/bin/python -m pytest` (a bare `pytest` or `uv run pytest` fails in this repo; if the venv lacks pytest: `VIRTUAL_ENV=.venv uv pip install -e . pytest pytest-asyncio`).
- Ledger canonical key MUST be `os.path.normcase(str(resolved.absolute()))` — byte-identical to `_write_lock_for`'s canonical key in `Tools/local_tool_impls.py:467`, so the ledger and the CAS agree on identity.
- `run_id` comes from `tldw_chatbook.Agents.run_context.current_run_id()` (a ContextVar; already visible inside tool execution — see `local_tool_provider.py:1527`). `""` (no run) is a valid key.
- Never record a ledger entry from fs_read's success/failure: a missing file RAISES the same `LocalToolError` type as a confinement refusal. Only the provider-side resolve+`is_file()` path records.
- Per-run entry cap: 512 paths per run_id, oldest-evicted (bounds the long-lived MCP server provider that runs forever with `run_id=""`).
- Do NOT add any `logger.warning`/`logger.error` to production paths in this change — a new boot-path diagnostic trips BOTH the boot-census ratchet and the derived-artifacts inventory check. Refusals are returned as results, not logged.
- fs_write CAS injection is SKIPPED when the model supplied `expected_sha256` or `expected_absent`, or when `_promotion_call_kind(name, args) is not None` (`local_tool_provider.py:2042` — promotion calls carry their own snapshot CAS; injecting would break `_application_matches`).
- Single-agent behavior unchanged: no ledger entry ⇒ every write proceeds exactly as today (AC#3).

---

### Task 1: ReadLedger module

**Files:**
- Create: `tldw_chatbook/Agents/fs_read_ledger.py`
- Test: `Tests/Agents/test_fs_read_ledger.py`

**Interfaces:**
- Consumes: stdlib only.
- Produces (later tasks rely on these exact names):
  - `@dataclass(frozen=True) ReadStamp: sha256: str | None; size: int` — `sha256 is None` means the stamp is ABSENT-kind; use `ReadStamp.absent()` classmethod.
  - `class ReadLedger:`
    - `record_present(run_id: str, canonical_path: str, sha256: str, size: int) -> None`
    - `record_absent(run_id: str, canonical_path: str) -> None`
    - `stamp_for(run_id: str, canonical_path: str) -> ReadStamp | None`
    - `update_written(run_id: str, canonical_path: str, sha256: str, size: int) -> None` (same as record_present; distinct name for call-site clarity)
    - constructor: `ReadLedger(max_paths_per_run: int = 512)`
  - `canonical_ledger_key(resolved: Path) -> str` — `os.path.normcase(str(resolved.absolute()))`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_fs_read_ledger.py
"""TASK-28238 phase 1: read-ledger for the stale-write guard."""

import threading
from pathlib import Path

from tldw_chatbook.Agents.fs_read_ledger import (
    ReadLedger,
    ReadStamp,
    canonical_ledger_key,
)


def test_record_and_lookup_present():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "ab" * 32, 10)
    stamp = ledger.stamp_for("run-a", "/x/f.txt")
    assert stamp is not None and stamp.sha256 == "ab" * 32 and stamp.size == 10


def test_runs_are_independent():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "aa" * 32, 1)
    ledger.record_present("run-b", "/x/f.txt", "bb" * 32, 2)
    assert ledger.stamp_for("run-a", "/x/f.txt").sha256 == "aa" * 32
    assert ledger.stamp_for("run-b", "/x/f.txt").sha256 == "bb" * 32
    assert ledger.stamp_for("run-c", "/x/f.txt") is None


def test_absent_stamp():
    ledger = ReadLedger()
    ledger.record_absent("run-a", "/x/missing.txt")
    stamp = ledger.stamp_for("run-a", "/x/missing.txt")
    assert stamp is not None and stamp.is_absent


def test_update_written_replaces():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "aa" * 32, 1)
    ledger.update_written("run-a", "/x/f.txt", "cc" * 32, 3)
    assert ledger.stamp_for("run-a", "/x/f.txt").sha256 == "cc" * 32


def test_per_run_cap_evicts_oldest():
    ledger = ReadLedger(max_paths_per_run=3)
    for i in range(5):
        ledger.record_present("run-a", f"/x/{i}.txt", "aa" * 32, i)
    assert ledger.stamp_for("run-a", "/x/0.txt") is None
    assert ledger.stamp_for("run-a", "/x/1.txt") is None
    assert ledger.stamp_for("run-a", "/x/4.txt") is not None
    # other runs unaffected by run-a's evictions
    ledger.record_present("run-b", "/x/0.txt", "bb" * 32, 0)
    assert ledger.stamp_for("run-b", "/x/0.txt") is not None


def test_thread_safety_no_lost_updates():
    ledger = ReadLedger(max_paths_per_run=10_000)

    def hammer(run_id):
        for i in range(500):
            ledger.record_present(run_id, f"/x/{i}.txt", "aa" * 32, i)

    threads = [threading.Thread(target=hammer, args=(f"run-{t}",)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in range(4):
        assert ledger.stamp_for(f"run-{t}", "/x/499.txt") is not None


def test_canonical_key_matches_cas_canonicalization(tmp_path):
    import os
    resolved = (tmp_path / "f.txt").resolve()
    assert canonical_ledger_key(resolved) == os.path.normcase(str(resolved.absolute()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_fs_read_ledger.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.fs_read_ledger'`

- [ ] **Step 3: Write the module**

```python
# tldw_chatbook/Agents/fs_read_ledger.py
"""TASK-28238 phase 1: per-run read-ledger for the fs stale-write guard.

Concurrent fleet children share ONE LocalToolProvider (that is why
RunToolPolicy keys its caps by (run_id, tool)); a per-provider ledger would
let one child's write mask a sibling's staleness. So entries key on
(run_id, canonical_path). The canonical path uses the SAME normalization as
the fs_write CAS lock (`os.path.normcase(str(p.absolute()))`,
Tools/local_tool_impls.py `_write_lock_for`) so the two mechanisms agree.

Bounded: at most ``max_paths_per_run`` entries per run_id, oldest evicted —
necessary because the MCP server provider (MCP/local_server_tools.py
build_server_local_provider) lives for the process with run_id always "".
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_PATHS_PER_RUN = 512


def canonical_ledger_key(resolved: Path) -> str:
    """The ledger's path identity; byte-identical to the fs_write CAS lock key.

    Args:
        resolved: An already-resolved path (resolve_workspace_path output).

    Returns:
        ``os.path.normcase(str(resolved.absolute()))``.
    """
    return os.path.normcase(str(resolved.absolute()))


@dataclass(frozen=True)
class ReadStamp:
    """What this run last saw at a path: a whole-file hash, or absence."""

    sha256: str | None
    size: int

    @classmethod
    def absent(cls) -> "ReadStamp":
        """A stamp recording that the path did not exist when read."""
        return cls(sha256=None, size=0)

    @property
    def is_absent(self) -> bool:
        """True when this stamp recorded a missing file."""
        return self.sha256 is None


class ReadLedger:
    """Thread-safe (run_id, canonical_path) -> ReadStamp map with a per-run cap."""

    def __init__(self, max_paths_per_run: int = DEFAULT_MAX_PATHS_PER_RUN) -> None:
        """Create a ledger.

        Args:
            max_paths_per_run: Entry cap per run_id; oldest evicted on overflow.
        """
        self._max = max(1, int(max_paths_per_run))
        self._lock = threading.Lock()
        # run_id -> OrderedDict[canonical_path, ReadStamp] (insertion-ordered
        # for oldest-first eviction; move_to_end on re-record).
        self._by_run: dict[str, OrderedDict[str, ReadStamp]] = {}

    def _put(self, run_id: str, canonical_path: str, stamp: ReadStamp) -> None:
        with self._lock:
            entries = self._by_run.setdefault(str(run_id), OrderedDict())
            if canonical_path in entries:
                entries.move_to_end(canonical_path)
            entries[canonical_path] = stamp
            while len(entries) > self._max:
                entries.popitem(last=False)

    def record_present(
        self, run_id: str, canonical_path: str, sha256: str, size: int
    ) -> None:
        """Record that ``run_id`` read a present file with this content hash."""
        self._put(run_id, canonical_path, ReadStamp(sha256=sha256, size=int(size)))

    def record_absent(self, run_id: str, canonical_path: str) -> None:
        """Record that ``run_id`` observed the path missing."""
        self._put(run_id, canonical_path, ReadStamp.absent())

    def update_written(
        self, run_id: str, canonical_path: str, sha256: str, size: int
    ) -> None:
        """Record the content ``run_id`` itself just wrote (same as a fresh read)."""
        self.record_present(run_id, canonical_path, sha256, size)

    def stamp_for(self, run_id: str, canonical_path: str) -> ReadStamp | None:
        """Return what ``run_id`` last saw at the path, or None if never read."""
        with self._lock:
            entries = self._by_run.get(str(run_id))
            if entries is None:
                return None
            return entries.get(canonical_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Agents/test_fs_read_ledger.py -q -p no:cacheprovider`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/fs_read_ledger.py Tests/Agents/test_fs_read_ledger.py
git commit -m "feat(agents): read-ledger for the fs stale-write guard (TASK-28238 P1 T1)"
```

---

### Task 2: Record-on-read wiring in LocalToolProvider

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (constructor ~line 570 `self._root = workspace_root`; dispatch closure `_invoke_allowed` ~line 1532)
- Test: `Tests/Agents/test_local_tool_provider.py` (append a new section at the end)

**Interfaces:**
- Consumes: Task 1's `ReadLedger`, `canonical_ledger_key`; existing `resolve_workspace_path(path, root, intent=...)` and `LocalToolError` (`Tools/local_tool_impls.py`); `current_run_id()` (`Agents/run_context.py`).
- Produces: `self._read_ledger: ReadLedger` on every provider (later tasks read/update it); a private helper with this exact signature:
  `def _record_fs_read_observation(self, args: dict, root: Path) -> None`
  and a shared hashing helper:
  `def _hash_file(path: Path) -> tuple[str, int] | None` (module-level; None when unreadable/missing).

**Placement note for the implementer:** inside `_invoke_allowed` (after the authority check, before `selected_spec.handler(clean_args)` is called) add, for `name == "fs_read"` only, a call to `self._record_fs_read_observation(clean_args, redaction_root or self._root)`. `redaction_root` is `authority.root` when an authority is pinned (see `_invoke_allowed`'s existing `redaction_root = authority.root if authority is not None else self._result_redaction_root` — use `authority.root if authority is not None else self._root` for RESOLUTION; do not resolve against `_result_redaction_root`). The record happens regardless of what the handler later returns; a resolve failure records nothing and never raises out.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Agents/test_local_tool_provider.py`)

```python
# --- TASK-28238 phase 1: stale-write guard -- record-on-read ---

def _guard_provider(tmp_path):
    """Real-executor provider rooted at tmp_path for guard tests."""
    return make_provider(root=tmp_path, use_default_executor=True, allow_write=True)


def test_fs_read_records_whole_file_hash(tmp_path):
    import hashlib
    from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key

    target = tmp_path / "a.txt"
    target.write_text("line1\nline2\nline3\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        result = provider.invoke("local:fs_read", {"path": "a.txt", "limit": 1})
    assert result.ok
    key = canonical_ledger_key(target.resolve())
    stamp = provider._read_ledger.stamp_for("run-a", key)
    assert stamp is not None
    # whole-file hash, not the windowed first line
    assert stamp.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()


def test_fs_read_of_missing_path_records_absent(tmp_path):
    from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key

    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        result = provider.invoke("local:fs_read", {"path": "nope.txt"})
    assert not result.ok  # fs_read itself still errors as today
    key = canonical_ledger_key((tmp_path / "nope.txt").resolve())
    stamp = provider._read_ledger.stamp_for("run-a", key)
    assert stamp is not None and stamp.is_absent


def test_fs_read_of_refused_path_records_nothing(tmp_path):
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        result = provider.invoke("local:fs_read", {"path": "../outside.txt"})
    assert not result.ok
    # nothing recorded under this run at all
    assert provider._read_ledger._by_run.get("run-a") in (None, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "records_whole_file or records_absent or records_nothing"`
Expected: FAIL — `AttributeError: 'LocalToolProvider' object has no attribute '_read_ledger'`

- [ ] **Step 3: Implement**

In `local_tool_provider.py`:

(a) Near the other lazy imports at the top of the module, do NOT add a top-level import (this file loads per run, not at boot, but stay consistent with its lazy-import style). Instead add a module-level helper below `_promotion_call_kind`:

```python
def _hash_file(path: "Path") -> "tuple[str, int] | None":
    """Whole-file (sha256, size) of ``path``; None when missing/unreadable."""
    import hashlib

    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest(), len(data)
```

(b) In `__init__`, next to `self._spill_lock = threading.Lock()` / `self._inline_bytes_by_run` (~line 719):

```python
        from tldw_chatbook.Agents.fs_read_ledger import ReadLedger

        # TASK-28238 phase 1: (run_id, canonical_path) -> what this run last
        # saw there. Keyed per run because fleet children SHARE this provider.
        self._read_ledger = ReadLedger()
```

(c) Add the method (near `_bounded_result`):

```python
    def _record_fs_read_observation(self, args: dict, root: "Path") -> None:
        """Stamp the ledger from a provider-side resolve of an fs_read target.

        Never keyed off fs_read's outcome: a missing file and a confinement
        refusal raise the same LocalToolError type, so the provider resolves
        the path itself -- refused -> record nothing; absent -> ABSENT;
        present -> whole-file hash. Never raises.
        """
        from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key
        from tldw_chatbook.Agents.run_context import current_run_id
        from tldw_chatbook.Tools.local_tool_impls import (
            LocalToolError,
            resolve_workspace_path,
        )

        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            return
        try:
            resolved = resolve_workspace_path(raw, Path(root).resolve(), intent="read")
        except (LocalToolError, OSError, ValueError):
            return  # refused path: not an observation
        key = canonical_ledger_key(resolved)
        run_id = current_run_id()
        if not resolved.is_file():
            self._read_ledger.record_absent(run_id, key)
            return
        hashed = _hash_file(resolved)
        if hashed is None:
            return
        digest, size = hashed
        self._read_ledger.record_present(run_id, key, digest, size)
```

(d) In `_invoke_allowed`, right after `dispatch_started = True` and before the `selected_spec` selection (~line 1549):

```python
                if name == "fs_read":
                    self._record_fs_read_observation(
                        clean_args,
                        authority.root if authority is not None else self._root,
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "records_whole_file or records_absent or records_nothing"`
Expected: 3 passed

- [ ] **Step 5: Run the whole provider suite for regressions**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py Tests/Agents/test_fs_read_ledger.py -q -p no:cacheprovider`
Expected: all pass (baseline count + 10)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat(agents): record fs_read observations in the stale-write ledger (TASK-28238 P1 T2)"
```

---

### Task 3: fs_write staleness via CAS injection

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_invoke_allowed`, same region as Task 2; new refusal constant next to `LOCAL_ROOT_CHANGED_REFUSAL` ~line 132)
- Test: `Tests/Agents/test_local_tool_provider.py`

**Interfaces:**
- Consumes: Task 2's `self._read_ledger`, `_hash_file`, `canonical_ledger_key`; `write_file`'s existing CAS params (`expected_sha256`, `expected_absent` — `Tools/local_tool_impls.py:441-448`); `_promotion_call_kind` (`local_tool_provider.py:2042`).
- Produces: module constant `LOCAL_STALE_WRITE_REFUSAL = "Stale write refused: {path} changed since you read it (was {old}, now {new}). Re-read the file and retry."` where `{old}`/`{new}` are `sha256[:8]/size` or the word `absent`; a helper
  `def _stale_write_refusal(self, shown_path: str, stamp, resolved: "Path") -> str` used by Task 4 too; reason-code string `"stale_write"` carried on the blocked ToolResult (mirror how `LOCAL_ROOT_CHANGED_REFUSAL` results are built with `ToolResult.blocked(...)`).

**Semantics to implement (exact):** for `name == "fs_write"`, before handler dispatch:
1. Resolve the target (same pattern as Task 2, `intent="write"`); on resolve failure do nothing (the handler will refuse identically as today).
2. `stamp = self._read_ledger.stamp_for(current_run_id(), key)`; if `None` → dispatch unchanged (blind write).
3. If the model supplied `expected_sha256` or `expected_absent` in `clean_args`, or `_promotion_call_kind(name, clean_args) is not None` → dispatch unchanged (explicit intent / promotion snapshot CAS wins).
4. Otherwise inject into a COPY of `clean_args`: `expected_sha256=stamp.sha256` when the stamp has a hash, `expected_absent=True` when the stamp is ABSENT.
5. Wrap the handler call: when it raises `LocalToolError` whose message starts with `"write precondition failed"` AND this call injected, return `ToolResult.blocked(self._stale_write_refusal(...))` with reason-code `"stale_write"` instead of the generic error — computing `{new}` from `_hash_file(resolved)` at refusal time (or `absent` if now missing). A precondition failure when the MODEL supplied the params keeps today's error text.

- [ ] **Step 1: Write the failing tests**

```python
# --- TASK-28238 phase 1: fs_write staleness (CAS injection) ---

def test_two_writer_race_refuses_second_writer(tmp_path):
    """AC#4: A reads, B writes, A's write refuses naming the conflict."""
    target = tmp_path / "shared.txt"
    target.write_text("original\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        assert provider.invoke("local:fs_read", {"path": "shared.txt"}).ok
    with use_run_id("run-b"):
        assert provider.invoke(
            "local:fs_write", {"path": "shared.txt", "content": "B's version\n"}
        ).ok
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_write", {"path": "shared.txt", "content": "A's version\n"}
        )
    assert not result.ok
    text = str(result.content)
    assert "Stale write refused" in text and "shared.txt" in text
    # B's content survived; A did not clobber
    assert target.read_text() == "B's version\n"


def test_own_read_write_write_chain_never_false_positives(tmp_path):
    target = tmp_path / "mine.txt"
    target.write_text("v1\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        assert provider.invoke("local:fs_read", {"path": "mine.txt"}).ok
        assert provider.invoke(
            "local:fs_write", {"path": "mine.txt", "content": "v2\n"}
        ).ok
        assert provider.invoke(
            "local:fs_write", {"path": "mine.txt", "content": "v3\n"}
        ).ok
    assert target.read_text() == "v3\n"


def test_blind_write_proceeds_unchanged(tmp_path):
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_write", {"path": "new.txt", "content": "hello\n"}
        )
    assert result.ok
    assert (tmp_path / "new.txt").read_text() == "hello\n"


def test_absent_then_created_by_peer_refuses(tmp_path):
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        provider.invoke("local:fs_read", {"path": "soon.txt"})  # records ABSENT
    with use_run_id("run-b"):
        assert provider.invoke(
            "local:fs_write", {"path": "soon.txt", "content": "B first\n"}
        ).ok
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_write", {"path": "soon.txt", "content": "A's create\n"}
        )
    assert not result.ok
    assert "Stale write refused" in str(result.content)
    assert (tmp_path / "soon.txt").read_text() == "B first\n"


def test_model_supplied_precondition_wins_over_ledger(tmp_path):
    import hashlib

    target = tmp_path / "explicit.txt"
    target.write_text("old\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        provider.invoke("local:fs_read", {"path": "explicit.txt"})
    # peer changes the file
    target.write_text("peer\n")
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_write",
            {"path": "explicit.txt", "content": "mine\n", "expected_sha256": current},
        )
    # model's explicit (correct, current) precondition wins -> write proceeds
    assert result.ok
    assert target.read_text() == "mine\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "two_writer_race or false_positives or blind_write_proceeds or absent_then_created or precondition_wins"`
Expected: `test_two_writer_race_refuses_second_writer` and `test_absent_then_created_by_peer_refuses` FAIL (writes succeed today); the other three PASS trivially (they pin no-regression behavior — that is fine, keep them).

- [ ] **Step 3: Implement**

In `local_tool_provider.py`:

(a) Next to `LOCAL_ROOT_CHANGED_REFUSAL` (~line 132):

```python
LOCAL_STALE_WRITE_REFUSAL = (
    "Stale write refused: {path} changed since you read it "
    "(was {old}, now {new}). Re-read the file and retry."
)
```

(b) Methods (near `_record_fs_read_observation`):

```python
    def _stale_write_refusal(self, shown_path: str, stamp, resolved: "Path") -> str:
        """Build the AC#2 refusal naming the conflict; never raises."""
        def _fmt(sha: "str | None", size: int) -> str:
            return "absent" if sha is None else f"{sha[:8]}/{size}"

        now = _hash_file(resolved)
        new_text = "absent" if now is None else _fmt(now[0], now[1])
        return LOCAL_STALE_WRITE_REFUSAL.format(
            path=shown_path, old=_fmt(stamp.sha256, stamp.size), new=new_text
        )

    def _fs_write_guard_injection(
        self, args: dict, root: "Path"
    ) -> "tuple[dict, object, Path] | None":
        """Return (args_with_cas, stamp, resolved) when the ledger arms fs_write.

        None means dispatch unchanged: no stamp, refused path, explicit
        model-supplied precondition, or a promotion call.
        """
        from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key
        from tldw_chatbook.Agents.run_context import current_run_id
        from tldw_chatbook.Tools.local_tool_impls import (
            LocalToolError,
            resolve_workspace_path,
        )

        if "expected_sha256" in args or "expected_absent" in args:
            return None
        if _promotion_call_kind("fs_write", args) is not None:
            return None
        raw = args.get("path")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            resolved = resolve_workspace_path(raw, Path(root).resolve(), intent="write")
        except (LocalToolError, OSError, ValueError):
            return None
        stamp = self._read_ledger.stamp_for(current_run_id(), canonical_ledger_key(resolved))
        if stamp is None:
            return None
        injected = dict(args)
        if stamp.is_absent:
            injected["expected_absent"] = True
        else:
            injected["expected_sha256"] = stamp.sha256
        return injected, stamp, resolved
```

(c) In `_invoke_allowed`, extend the Task-2 hook block:

```python
                stale_guard = None
                if name == "fs_read":
                    self._record_fs_read_observation(
                        clean_args,
                        authority.root if authority is not None else self._root,
                    )
                elif name == "fs_write":
                    stale_guard = self._fs_write_guard_injection(
                        clean_args,
                        authority.root if authority is not None else self._root,
                    )
                    if stale_guard is not None:
                        clean_args = stale_guard[0]
```

(d) Reword the refusal at the exception path. VERIFIED SEAM: the impl's `LocalToolError("write precondition failed: ...")` does NOT reach `_invoke_allowed` directly — the workspace worker sanitizes it and re-raises `WorkspaceToolExecutionError(code, message)` with the message text preserved (`Tools/workspace_tool_worker.py:91,129`; `Tools/workspace_tool_executor.py:49`), which lands in `_invoke_allowed`'s EXISTING `except WorkspaceToolExecutionError as exc:` clause (~line 1574). Extend THAT clause — add this at its top, before the existing `_workspace_execution_error_result` conversion:

```python
                except WorkspaceToolExecutionError as exc:
                    if (
                        stale_guard is not None
                        and "write precondition failed" in str(exc)
                    ):
                        provider_terminal = LocalProviderTerminal.RETURNED
                        return LocalToolInvocationResult(
                            result=ToolResult.blocked(
                                self._stale_write_refusal(
                                    str(clean_args.get("path", "")),
                                    stale_guard[1],
                                    stale_guard[2],
                                )
                            ),
                            final_gate=gate.verdict,
                            approval_consumed=gate.approval_consumed,
                            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
                            dispatch_started=True,
                            provider_terminal=provider_terminal,
                        )
                    # ... existing conversion continues unchanged ...
```

(Never let the stale branch swallow a precondition failure the MODEL parameterized — the `stale_guard is not None` check is that guarantee, because injection is skipped when the model supplied params. The substring match is on the impl's exact copy at `Tools/local_tool_impls.py:474,476`: "write precondition failed: target is present" / "write precondition failed: target digest changed".)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "two_writer_race or false_positives or blind_write_proceeds or absent_then_created or precondition_wins"`
Expected: 5 passed

NOTE for the implementer on `test_own_read_write_write_chain_never_false_positives`: it passes at this task only because Task 3's injection uses the stamp recorded at READ time, and the FIRST write invalidates disk vs stamp — so the SECOND write would refuse WITHOUT Task 5's update-after-write. If this test FAILS at this task with a stale-write refusal on v3: that is the expected intermediate state — move the test to Task 5's step 1 instead and note it in the commit. (Whether it fails here depends on ordering; the plan keeps it here so the implementer sees the dependency explicitly.)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat(agents): fs_write stale-write guard via atomic CAS injection (TASK-28238 P1 T3)"
```

---

### Task 4: fs_edit / fs_patch pre-hash staleness check

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (same `_invoke_allowed` region)
- Test: `Tests/Agents/test_local_tool_provider.py`

**Interfaces:**
- Consumes: Tasks 1-3's ledger, `_hash_file`, `_stale_write_refusal`, `canonical_ledger_key`; `parse_patch_targets` (`Tools/patch_tool_impls.py:184`) for fs_patch's multi-target list (mirror the existing loop in `_path_targets_without_authority`, `local_tool_provider.py:987-1000`).
- Produces: `def _stale_targets_for(self, name: str, args: dict, root: "Path") -> "list[tuple[str, object, Path]]"` — `(shown_path, stamp, resolved)` for every target whose ledger stamp mismatches current disk; empty list means proceed.

**Semantics:** for `name in {"fs_edit", "fs_patch"}` before handler dispatch: resolve each target (fs_edit: `args["path"]`; fs_patch: every `plan.new_path` from `parse_patch_targets(args["diff"])`, skipping plans whose parse fails — the handler will surface those). For each target with a ledger stamp: present-stamp whose disk hash differs, or ABSENT-stamp whose path now exists → stale. If ANY target is stale → return `ToolResult.blocked(...)` refusing the WHOLE call, message = `_stale_write_refusal` of the FIRST stale target (plus `" (+N more stale targets)"` when N>0). This is a pre-hash check (tiny TOCTOU accepted — spec §Check).

- [ ] **Step 1: Write the failing tests**

```python
# --- TASK-28238 phase 1: fs_edit / fs_patch staleness (pre-hash) ---

def test_edit_race_refuses_second_writer(tmp_path):
    target = tmp_path / "shared.py"
    target.write_text("x = 1\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        assert provider.invoke("local:fs_read", {"path": "shared.py"}).ok
    with use_run_id("run-b"):
        assert provider.invoke(
            "local:fs_write", {"path": "shared.py", "content": "x = 2\n"}
        ).ok
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_edit",
            {"path": "shared.py", "old_string": "x = 1", "new_string": "x = 99"},
        )
    assert not result.ok
    assert "Stale write refused" in str(result.content)
    assert target.read_text() == "x = 2\n"


def test_edit_without_prior_read_proceeds(tmp_path):
    target = tmp_path / "blind.py"
    target.write_text("y = 1\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_edit", {"path": "blind.py", "old_string": "y = 1", "new_string": "y = 2"}
        )
    assert result.ok
    assert target.read_text() == "y = 2\n"


def test_patch_with_one_stale_target_refuses_whole_patch(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha\n")
    b.write_text("beta\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        assert provider.invoke("local:fs_read", {"path": "a.txt"}).ok
        assert provider.invoke("local:fs_read", {"path": "b.txt"}).ok
    with use_run_id("run-b"):
        assert provider.invoke(
            "local:fs_write", {"path": "b.txt", "content": "beta CHANGED\n"}
        ).ok
    diff = (
        "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-alpha\n+alpha2\n"
        "--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-beta\n+beta2\n"
    )
    with use_run_id("run-a"):
        result = provider.invoke("local:fs_patch", {"diff": diff})
    assert not result.ok
    assert "Stale write refused" in str(result.content)
    # NEITHER file was touched -- whole patch refused
    assert a.read_text() == "alpha\n"
    assert b.read_text() == "beta CHANGED\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "edit_race or without_prior_read or one_stale_target"`
Expected: `test_edit_race_refuses_second_writer` and `test_patch_with_one_stale_target` FAIL (they succeed today); `test_edit_without_prior_read_proceeds` PASSES (no-regression pin).

- [ ] **Step 3: Implement**

Add the helper:

```python
    def _stale_targets_for(
        self, name: str, args: dict, root: "Path"
    ) -> "list[tuple[str, object, Path]]":
        """Targets of an fs_edit/fs_patch whose ledger stamp mismatches disk."""
        from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key
        from tldw_chatbook.Agents.run_context import current_run_id
        from tldw_chatbook.Tools.local_tool_impls import (
            LocalToolError,
            resolve_workspace_path,
        )

        run_id = current_run_id()
        base = Path(root).resolve()
        shown_paths: list[str] = []
        if name == "fs_edit":
            raw = args.get("path")
            if isinstance(raw, str) and raw:
                shown_paths.append(raw)
        elif name == "fs_patch":
            from tldw_chatbook.Tools.patch_tool_impls import (
                FilesystemPatchError,
                parse_patch_targets,
            )

            try:
                plans = parse_patch_targets(args.get("diff") or "")
            except FilesystemPatchError:
                return []  # the handler will refuse the malformed diff itself
            for plan in plans:
                if plan.new_path is not None:
                    shown_paths.append(plan.new_path)

        stale: list[tuple[str, object, Path]] = []
        for shown in shown_paths:
            try:
                resolved = resolve_workspace_path(shown, base, intent="write")
            except (LocalToolError, OSError, ValueError):
                continue  # handler will refuse identically
            stamp = self._read_ledger.stamp_for(run_id, canonical_ledger_key(resolved))
            if stamp is None:
                continue
            now = _hash_file(resolved)
            if stamp.is_absent:
                if now is not None:
                    stale.append((shown, stamp, resolved))
            elif now is None or now[0] != stamp.sha256:
                stale.append((shown, stamp, resolved))
        return stale
```

Extend the `_invoke_allowed` hook block from Task 3:

```python
                elif name in {"fs_edit", "fs_patch"}:
                    _stale = self._stale_targets_for(
                        name,
                        clean_args,
                        authority.root if authority is not None else self._root,
                    )
                    if _stale:
                        shown, stamp, resolved = _stale[0]
                        message = self._stale_write_refusal(shown, stamp, resolved)
                        if len(_stale) > 1:
                            message += f" (+{len(_stale) - 1} more stale targets)"
                        provider_terminal = LocalProviderTerminal.RETURNED
                        return LocalToolInvocationResult(
                            result=ToolResult.blocked(message),
                            final_gate=gate.verdict,
                            approval_consumed=gate.approval_consumed,
                            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
                            dispatch_started=True,
                            provider_terminal=provider_terminal,
                        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "edit_race or without_prior_read or one_stale_target"`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat(agents): fs_edit/fs_patch stale-target pre-check (TASK-28238 P1 T4)"
```

---

### Task 5: Ledger update after successful writes

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (same region)
- Test: `Tests/Agents/test_local_tool_provider.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `def _update_ledger_after_write(self, name: str, args: dict, root: "Path") -> None` — called ONLY on the handler-returned-ok path.

**Semantics:** after `selected_spec.handler(clean_args)` returns successfully for `fs_write`/`fs_edit`/`fs_patch` (not dry-run: skip when `args.get("dry_run") is True`): for every target (fs_write/fs_edit: `args["path"]`; fs_patch: `parse_patch_targets` list), re-resolve and `_hash_file` the file NOW (post-handler re-read — the spec's Update rule: for edit/patch the written bytes are only knowable from disk) and `update_written`. A now-missing target (patch deleted it) records ABSENT. Never raises.

- [ ] **Step 1: Write the failing test**

```python
# --- TASK-28238 phase 1: update-after-write ---

def test_read_edit_edit_chain_never_false_positives(tmp_path):
    target = tmp_path / "chain.py"
    target.write_text("n = 1\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        assert provider.invoke("local:fs_read", {"path": "chain.py"}).ok
        assert provider.invoke(
            "local:fs_edit", {"path": "chain.py", "old_string": "n = 1", "new_string": "n = 2"}
        ).ok
        second = provider.invoke(
            "local:fs_edit", {"path": "chain.py", "old_string": "n = 2", "new_string": "n = 3"}
        )
    assert second.ok, str(second.content)
    assert target.read_text() == "n = 3\n"


def test_write_updates_ledger_so_peer_race_still_detected_after(tmp_path):
    """After my own write, a PEER's change is still caught on my next write."""
    target = tmp_path / "then.txt"
    target.write_text("v1\n")
    provider = _guard_provider(tmp_path)
    with use_run_id("run-a"):
        provider.invoke("local:fs_read", {"path": "then.txt"})
        assert provider.invoke(
            "local:fs_write", {"path": "then.txt", "content": "v2\n"}
        ).ok
    with use_run_id("run-b"):
        assert provider.invoke(
            "local:fs_write", {"path": "then.txt", "content": "peer\n"}
        ).ok
    with use_run_id("run-a"):
        result = provider.invoke(
            "local:fs_write", {"path": "then.txt", "content": "v3\n"}
        )
    assert not result.ok
    assert "Stale write refused" in str(result.content)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -q -p no:cacheprovider -k "chain_never_false or still_detected_after"`
Expected: `test_read_edit_edit_chain_never_false_positives` FAILS (the second edit sees a stale stamp from the pre-edit read). If Task 3's own-rewrite test was deferred here, it fails the same way. `test_write_updates_ledger...` FAILS (run-a's second write proceeds blind or refuses wrongly depending on ordering — assert pins the correct end state).

- [ ] **Step 3: Implement**

```python
    def _update_ledger_after_write(self, name: str, args: dict, root: "Path") -> None:
        """Re-stamp every written target so an agent's own chain never trips.

        Post-handler re-read: for fs_edit/fs_patch the written bytes are only
        knowable from disk. Never raises.
        """
        from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key
        from tldw_chatbook.Agents.run_context import current_run_id
        from tldw_chatbook.Tools.local_tool_impls import (
            LocalToolError,
            resolve_workspace_path,
        )

        if args.get("dry_run") is True:
            return
        run_id = current_run_id()
        base = Path(root).resolve()
        shown_paths: list[str] = []
        if name in {"fs_write", "fs_edit"}:
            raw = args.get("path")
            if isinstance(raw, str) and raw:
                shown_paths.append(raw)
        elif name == "fs_patch":
            from tldw_chatbook.Tools.patch_tool_impls import (
                FilesystemPatchError,
                parse_patch_targets,
            )

            try:
                plans = parse_patch_targets(args.get("diff") or "")
            except FilesystemPatchError:
                return
            for plan in plans:
                if plan.new_path is not None:
                    shown_paths.append(plan.new_path)
        for shown in shown_paths:
            try:
                resolved = resolve_workspace_path(shown, base, intent="write")
            except (LocalToolError, OSError, ValueError):
                continue
            key = canonical_ledger_key(resolved)
            hashed = _hash_file(resolved)
            if hashed is None:
                self._read_ledger.record_absent(run_id, key)
            else:
                self._read_ledger.update_written(run_id, key, hashed[0], hashed[1])
```

Call it in `_invoke_allowed` on the success path, immediately after `result = ToolResult(ok=True, ...)` is constructed and before the `return`:

```python
                    if name in {"fs_write", "fs_edit", "fs_patch"}:
                        self._update_ledger_after_write(
                            name,
                            clean_args,
                            authority.root if authority is not None else self._root,
                        )
```

- [ ] **Step 4: Run the full guard test set + provider suite**

Run: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py Tests/Agents/test_fs_read_ledger.py -q -p no:cacheprovider`
Expected: all pass, including every Task 2-5 guard test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat(agents): re-stamp ledger after successful writes (TASK-28238 P1 T5)"
```

---

### Task 6: Guardrail sweeps (ratchets, lint, full suites)

**Files:**
- No new files; fixes only if a sweep is red.

- [ ] **Step 1: Boot-census + perf ratchets**

Run: `.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py Tests/Performance/test_app_import_weight.py -q -p no:cacheprovider`
Expected: PASS. (`fs_read_ledger` is imported lazily inside methods and by `local_tool_provider`, which is not boot-resident; if the census fails naming `fs_read_ledger`, convert any top-level import of it to the lazy in-method pattern used throughout `local_tool_provider.py`.)

- [ ] **Step 2: Diagnostic inventory unchanged**

Run: `.venv/bin/python scripts/check_persistent_diagnostic_inventory.py`
Expected: clean (this change adds NO logger diagnostics). If it reports drift naming your files, you added a logging call — remove it (refusals are results, not logs).

- [ ] **Step 3: Lint the changed files**

Run: `.venv/bin/ruff check tldw_chatbook/Agents/fs_read_ledger.py tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_fs_read_ledger.py --select F`
Expected: clean. (`ruff` may need `VIRTUAL_ENV=.venv uv pip install ruff`.)

- [ ] **Step 4: Adjacent suites**

Run: `.venv/bin/python -m pytest Tests/Agents/ -q -p no:cacheprovider -x --ignore=Tests/Agents/test_agent_service.py`
Then: `.venv/bin/python -m pytest Tests/Agents/test_agent_service.py -q -p no:cacheprovider`
Expected: no NEW failures vs a clean origin/dev baseline (compare failing-test NAMES against a stash/clean checkout if anything is red — this repo has known env-dependent baseline failures).

- [ ] **Step 5: Commit any fixes**

```bash
git add -A -- tldw_chatbook/Agents Tests/Agents
git commit -m "chore(agents): guardrail-sweep fixes for the stale-write guard (TASK-28238 P1 T6)"
```

(Skip the commit if nothing changed.)

---

### Task 7: Task hygiene + spec status

**Files:**
- Modify: `backlog/tasks/task-28238 - Worktree-isolation-and-stale-write-guard-for-parallel-sub-agents.md`
- Modify: `backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md` (status line only)

- [ ] **Step 1: Tick the phase-1 ACs**

In the task file flip to `- [x]`: AC#2 (refuse-on-change naming the conflict), AC#3 (single-agent unchanged by default), AC#4 (racing-writers test). AC#1 (worktree isolation) stays `- [ ]` — it is phase 2; the task stays In Progress.

- [ ] **Step 2: Append phase-1 implementation notes to the task**

Summarize: ledger module + provider wiring (record provider-side, CAS injection for fs_write, pre-hash for fs_edit/fs_patch multi-target, post-write re-stamp), the files touched, and the test names that pin each AC. State that phase 2 (worktree isolation) remains, and that its dispatch-wiring open question (shared single registry) is recorded in the spec.

- [ ] **Step 3: Update the spec status line**

Change the spec's `Status:` line to note phase 1 implemented on branch `feat/task-28238-stale-write-guard` (date), phase 2 not started.

- [ ] **Step 4: Set task status**

Run: `backlog task edit 28238 -s "In Progress" -a @claude` (In Progress, not Done — phase 2 remains.)

- [ ] **Step 5: Commit**

```bash
git add "backlog/tasks/task-28238 - Worktree-isolation-and-stale-write-guard-for-parallel-sub-agents.md" backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md
git commit -m "docs(agents): record TASK-28238 phase-1 completion; phase 2 remains"
```
