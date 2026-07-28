# Agent Run Log (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append every model turn, tool call, and tool result of an agent run losslessly to a segmented, searchable log file, and give the primary agent a `search_run_log` tool to query it.

**Architecture:** A pure codec (`run_log_format.py`) encodes/parses a line-anchored record format. A writer (`run_log.py`) owns path resolution, segmentation, and durability. The pure loop gains one injected callable, `LoopDeps.on_record`, called at the two points where full-fidelity values exist. `AgentService` constructs the writer once per run tree and wires it. `search_run_log` is registered as a runtime tool exactly like `install_skill`.

**Tech Stack:** Python 3.11+, stdlib only (`dataclasses`, `pathlib`, `threading`, `re`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md`

## Global Constraints

- **Phase 1 is additive.** `run_agent_loop`'s `messages` list is NOT modified. Existing runs must behave byte-identically when logging is disabled or unwired.
- **Every new `LoopDeps` field defaults to `None`/no-op**, matching `review_tool_calls`, `read_skill_file`, `install_skill`, `run_skill_script`.
- **`agent_runtime.py` stays pure**: no I/O, no DB, no Textual imports. It may only call injected callables.
- **A failing log write must never abort a run.** Wrap `on_record` in catch-and-continue, as `add()` already does for `on_step`.
- **Path resolution goes through `allowed_file_roots(write=True, sandbox_root=_tool_sandbox_root())`** → `is_within` → `is_sensitive_path`. Never construct or validate paths independently.
- **Record anchor is `#@#`** (never `###` — markdown H3 collides).
- **`bytes=N` is UTF-8 bytes of content, excluding the terminating newline.** Log files are read and written in **binary**.
- **Tests run from the venv:** `python -m pytest` with the project venv active. The `timeout` command is unavailable in this environment.
- Docstrings are Google style with Args/Returns/Raises. Type hints on public APIs.

---

### Task 1: Record codec

Pure encode/parse. No filesystem, no runtime imports.

**Files:**
- Create: `tldw_chatbook/Agents/run_log_format.py`
- Test: `Tests/Agents/test_run_log_format.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RECORD_ANCHOR: str = "#@#"`
  - `@dataclass(frozen=True) RunLogRecord(number: int, run_id: str, kind: str, type: str, ts: str, content: str, tool: str = "", status: str = "", call_id: str = "", truncated_from: int = 0)`
  - `encode_record(record: RunLogRecord) -> bytes`
  - `iter_records(data: bytes) -> Iterator[RunLogRecord]`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_format.py`:

```python
# Tests/Agents/test_run_log_format.py
"""Pure record codec: round-trip, adversarial content, partial tails."""

from tldw_chatbook.Agents.run_log_format import (
    RECORD_ANCHOR,
    RunLogRecord,
    encode_record,
    iter_records,
)


def rec(number=1, content="hello", **kw):
    base = dict(
        number=number,
        run_id="a3f9c1",
        kind="primary",
        type="tool_result",
        ts="2026-07-27T18:22:31.004Z",
        content=content,
    )
    base.update(kw)
    return RunLogRecord(**base)


def test_round_trip_preserves_content_exactly():
    original = rec(content="line one\nline two\n")
    (parsed,) = list(iter_records(encode_record(original)))
    assert parsed.content == original.content
    assert parsed.number == 1
    assert parsed.run_id == "a3f9c1"


def test_header_is_one_physical_line():
    blob = encode_record(rec(tool="grep_files", status="ok", call_id="call_7"))
    header = blob.split(b"\n", 1)[0].decode()
    assert header.startswith(RECORD_ANCHOR + " ")
    assert "tool=grep_files" in header
    assert "bytes=5" in header


def test_content_containing_the_anchor_does_not_corrupt_parsing():
    # The whole point of bytes=N: content is sliced by length, never scanned.
    evil = f"{RECORD_ANCHOR} 999999 run=x kind=primary type=model ts=z bytes=0\nnope"
    blob = encode_record(rec(number=1, content=evil)) + encode_record(rec(number=2))
    parsed = list(iter_records(blob))
    assert len(parsed) == 2
    assert parsed[0].content == evil
    assert parsed[1].number == 2


def test_multibyte_content_counts_bytes_not_characters():
    original = rec(content="héllo — ✅")
    blob = encode_record(original)
    assert f"bytes={len(original.content.encode('utf-8'))}".encode() in blob
    (parsed,) = list(iter_records(blob))
    assert parsed.content == original.content


def test_partial_trailing_record_is_ignored():
    blob = encode_record(rec(number=1)) + encode_record(rec(number=2, content="abcdef"))
    truncated = blob[:-3]  # content cut mid-write
    parsed = list(iter_records(truncated))
    assert [p.number for p in parsed] == [1]


def test_record_missing_only_its_terminator_is_ignored():
    blob = encode_record(rec(number=1))
    parsed = list(iter_records(blob[:-1]))
    assert parsed == []


def test_truncated_field_round_trips_and_is_absent_otherwise():
    assert b"truncated=" not in encode_record(rec())
    blob = encode_record(rec(content="cut", truncated_from=9000))
    assert b"truncated=9000" in blob
    (parsed,) = list(iter_records(blob))
    assert parsed.truncated_from == 9000


def test_whitespace_in_header_values_is_sanitised():
    # A header field containing a space or newline would break single-line parsing.
    (parsed,) = list(iter_records(encode_record(rec(tool="bad name\nx"))))
    assert " " not in parsed.tool and "\n" not in parsed.tool
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.run_log_format'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Agents/run_log_format.py`:

```python
# tldw_chatbook/Agents/run_log_format.py
"""Line-anchored, byte-exact record codec for the agent run log.

Pure module: no filesystem, no runtime imports. See the design spec
(Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md §4).

Format, one record:

    #@# 000412 run=a3f9c1 kind=primary type=tool_result tool=grep_files \
status=ok call=call_7 ts=2026-07-27T18:22:31.004Z bytes=1834
    <exactly 1834 UTF-8 bytes of content>

The header is always ONE physical line: a wrapped header would break
``^#@# `` matching and detach fields onto a continuation line. ``bytes=``
lets a parser slice content by length instead of scanning for the next
anchor, so content containing a literal ``#@#`` cannot corrupt parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

#: Never occurs naturally. ``###`` was rejected: it is a markdown H3, so
#: every heading in generated or fetched content would false-positive.
RECORD_ANCHOR = "#@#"

_ANCHOR_BYTES = RECORD_ANCHOR.encode("utf-8") + b" "
_PLACEHOLDER = "-"


def _sanitise(value: str) -> str:
    """Collapse whitespace so a value can never break the single-line header.

    Args:
        value: Raw field value (a tool name, run id, status, or call id).

    Returns:
        ``value`` with every whitespace run replaced by ``_``, or ``"-"``
        when empty.
    """
    cleaned = "_".join(str(value).split())
    return cleaned or _PLACEHOLDER


@dataclass(frozen=True)
class RunLogRecord:
    """One appended record: a model turn, tool call, tool result, or spawn."""

    number: int
    run_id: str
    kind: str
    type: str
    ts: str
    content: str
    tool: str = ""
    status: str = ""
    call_id: str = ""
    truncated_from: int = 0


def encode_record(record: RunLogRecord) -> bytes:
    """Serialise one record to its on-disk bytes.

    Args:
        record: The record to encode.

    Returns:
        The header line, a newline, the UTF-8 content, and a terminating
        newline. ``bytes=`` counts the content only, never the terminator.
    """
    body = record.content.encode("utf-8")
    header = (
        f"{RECORD_ANCHOR} {record.number:06d}"
        f" run={_sanitise(record.run_id)}"
        f" kind={_sanitise(record.kind)}"
        f" type={_sanitise(record.type)}"
        f" tool={_sanitise(record.tool)}"
        f" status={_sanitise(record.status)}"
        f" call={_sanitise(record.call_id)}"
        f" ts={_sanitise(record.ts)}"
        f" bytes={len(body)}"
    )
    if record.truncated_from:
        header += f" truncated={record.truncated_from}"
    return header.encode("utf-8") + b"\n" + body + b"\n"


def _parse_header(line: str) -> dict[str, str] | None:
    parts = line.split(" ")
    if len(parts) < 3 or parts[0] != RECORD_ANCHOR:
        return None
    fields: dict[str, str] = {"number": parts[1]}
    for token in parts[2:]:
        key, _, value = token.partition("=")
        if value:
            fields[key] = value
    return fields


def iter_records(data: bytes) -> Iterator[RunLogRecord]:
    """Parse every COMPLETE record in ``data``, in file order.

    A trailing record whose declared content (plus its terminating newline)
    is not fully present is skipped: the agent searches its own log while
    the writer is still appending to it, so a half-written tail is normal
    rather than corruption.

    Args:
        data: Raw bytes of one log segment.

    Yields:
        Each fully-present ``RunLogRecord``.
    """
    position = 0
    length = len(data)
    while position < length:
        if not data.startswith(_ANCHOR_BYTES, position):
            # Not at a record boundary; find the next anchor at a line start.
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        newline = data.find(b"\n", position)
        if newline == -1:
            return
        fields = _parse_header(data[position:newline].decode("utf-8", "replace"))
        if fields is None:
            return
        try:
            size = int(fields.get("bytes", "0"))
            number = int(fields["number"])
        except (KeyError, ValueError):
            return
        start = newline + 1
        end = start + size
        # end + 1 covers the terminating newline: only fully-terminated
        # records are yielded, so a record still being written is skipped.
        if size < 0 or end + 1 > length:
            return
        yield RunLogRecord(
            number=number,
            run_id=fields.get("run", _PLACEHOLDER),
            kind=fields.get("kind", _PLACEHOLDER),
            type=fields.get("type", _PLACEHOLDER),
            ts=fields.get("ts", _PLACEHOLDER),
            content=data[start:end].decode("utf-8", "replace"),
            tool=fields.get("tool", _PLACEHOLDER),
            status=fields.get("status", _PLACEHOLDER),
            call_id=fields.get("call", _PLACEHOLDER),
            truncated_from=int(fields.get("truncated", "0") or 0),
        )
        position = end + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_run_log_format.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_log_format.py Tests/Agents/test_run_log_format.py
git commit -m "feat(agents): byte-exact record codec for the agent run log"
```

---

### Task 2: RunLogWriter

Path resolution, two-phase binding, segmentation, durability, config.

**Files:**
- Create: `tldw_chatbook/Agents/run_log.py`
- Test: `Tests/Agents/test_run_log_writer.py`

**Interfaces:**
- Consumes: `RunLogRecord`, `encode_record` from Task 1.
- Produces:
  - `class RunLogWriter(*, dir_name=None, segment_bytes=None, max_record_bytes=None)` — `None` resolves from `[agents]` config
  - `RunLogWriter.bind(run_id: str) -> None` — idempotent; creates the directory
  - `RunLogWriter.append(*, run_id: str, kind: str, type: str, content: str, tool: str = "", status: str = "", call_id: str = "") -> int | None` — returns the assigned record number, or `None` when inactive/failed
  - `RunLogWriter.write_manifest(metadata: dict) -> None` — never raises
  - `RunLogWriter.close() -> None` — final fsync; idempotent
  - `RunLogWriter.is_active -> bool`, `RunLogWriter.log_dir -> Path | None`
  - `resolve_log_root() -> Path | None`
  - `_setting(key: str, default)` — `[agents]` config accessor and test seam
  - Config keys (spec §8): `run_log_enabled` (default `True`), `run_log_dir_name` (`agent-runs`), `run_log_segment_bytes` (`4_000_000`), `run_log_max_record_bytes` (`1_000_000`)

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_writer.py`:

```python
# Tests/Agents/test_run_log_writer.py
"""Writer: binding, segmentation, durability, degradation."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Pin the writer's resolved root to a temp dir."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def make(root_dir, **kw):
    writer = RunLogWriter(**kw)
    writer.bind("run-abc")
    return writer


def test_unbound_writer_writes_nothing(root):
    writer = RunLogWriter()
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None
    assert list(root.iterdir()) == []


def test_bind_creates_the_run_directory_and_gitignore(root):
    make(root)
    assert (root / "agent-runs" / "run-abc").is_dir()
    assert (root / "agent-runs" / ".gitignore").read_text() == "*\n"


def test_existing_gitignore_is_never_overwritten(root):
    (root / "agent-runs").mkdir()
    (root / "agent-runs" / ".gitignore").write_text("keep me\n")
    make(root)
    assert (root / "agent-runs" / ".gitignore").read_text() == "keep me\n"


def test_records_are_numbered_monotonically_from_one(root):
    writer = make(root)
    numbers = [
        writer.append(run_id="r", kind="primary", type="model", content=str(i))
        for i in range(3)
    ]
    assert numbers == [1, 2, 3]


def test_a_child_run_shares_the_parent_counter(root):
    writer = make(root)
    writer.append(run_id="parent", kind="primary", type="model", content="a")
    writer.append(run_id="child", kind="subagent", type="model", content="b")
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    parsed = list(iter_records(data))
    assert [(p.number, p.run_id) for p in parsed] == [(1, "parent"), (2, "child")]


def test_second_bind_is_ignored(root):
    writer = make(root)
    writer.bind("run-other")
    assert writer.log_dir.name == "run-abc"


def test_segment_rolls_and_no_record_spans_a_boundary(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    run_dir = root / "agent-runs" / "run-abc"
    segments = sorted(run_dir.glob("logs.*.txt"))
    assert len(segments) > 1
    # Every segment parses standalone: no record straddles a boundary.
    total = 0
    for segment in segments:
        parsed = list(iter_records(segment.read_bytes()))
        assert parsed, f"{segment.name} parsed to nothing"
        total += len(parsed)
    assert total == 6


def test_oversized_record_is_capped_and_marked(root):
    writer = make(root, max_record_bytes=50)
    writer.append(run_id="r", kind="primary", type="tool_result", content="y" * 500)
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    (parsed,) = list(iter_records(data))
    assert len(parsed.content.encode()) <= 50
    assert parsed.truncated_from == 500


def test_unresolvable_root_deactivates_the_writer(monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None


def test_write_failure_deactivates_rather_than_raising(root, monkeypatch):
    writer = make(root)
    assert writer.append(run_id="r", kind="primary", type="model", content="a") == 1

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_bytes", boom)
    assert writer.append(run_id="r", kind="primary", type="model", content="b") is None
    assert writer.is_active is False


def test_config_can_disable_logging_entirely(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: False if key == "run_log_enabled" else default
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert not (root / "agent-runs").exists()


def test_config_overrides_the_directory_name(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "my-logs" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert (root / "my-logs" / "run-abc").is_dir()


def test_write_manifest_emits_readable_json(root):
    writer = make(root)
    writer.write_manifest({"run_id": "run-abc", "status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert manifest["status"] == "done"
    assert manifest["segments"] == []  # nothing appended yet


def test_manifest_records_segments_after_appends(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    writer.write_manifest({"status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert len(manifest["segments"]) > 1
    assert manifest["record_count"] == 6


def test_manifest_failure_never_raises(root, monkeypatch):
    writer = make(root)
    monkeypatch.setattr(writer, "_write_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError))
    writer.write_manifest({"status": "done"})  # must not raise


def test_close_is_safe_to_call_twice_and_on_an_inactive_writer(root):
    writer = make(root)
    writer.close()
    writer.close()
    RunLogWriter().close()


def test_concurrent_appends_produce_unique_numbers_and_no_corruption(root):
    import threading

    writer = make(root)

    def worker(index):
        for _ in range(20):
            writer.append(
                run_id=f"r{index}", kind="primary", type="model", content=f"payload-{index}"
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = []
    for segment in sorted((root / "agent-runs" / "run-abc").glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    numbers = sorted(r.number for r in records)
    assert numbers == list(range(1, 81))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.run_log'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Agents/run_log.py`:

```python
# tldw_chatbook/Agents/run_log.py
"""Segmented, append-only run log writer.

Impure by design (filesystem + config). Path resolution reuses the file
tools' own chain -- ``allowed_file_roots`` -> ``is_within`` ->
``is_sensitive_path`` -- so this writer can never become a path-validation
bypass. See the design spec §3.3, §7, §8, §9.2.
"""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from .run_log_format import RunLogRecord, encode_record

#: Directory created inside the resolved root. Deliberately UNDOTTED: a
#: dotted directory is excluded by `_is_hidden_within`, which would hide
#: the log from the very tools meant to read it.
DEFAULT_DIR_NAME = "agent-runs"
DEFAULT_SEGMENT_BYTES = 4_000_000
DEFAULT_MAX_RECORD_BYTES = 1_000_000
MANIFEST_NAME = "MANIFEST"


def _setting(key: str, default):
    """Read one ``[agents]`` config key. Test seam: monkeypatched wholesale.

    Args:
        key: Key name within the ``[agents]`` section.
        default: Value returned when unset or unreadable.

    Returns:
        The configured value, or ``default``.
    """
    try:
        from tldw_chatbook.config import get_cli_setting

        value = get_cli_setting("agents", key, default)
    except Exception:
        return default
    return default if value is None else value


def resolve_log_root() -> Path | None:
    """Return the directory the log tree is created under, or ``None``.

    Prefers the run's first read-write workspace folder root so the log is
    a user-visible artifact; falls back to the tool sandbox root when no
    such folder is bound. Any failure resolves to ``None`` (logging off)
    rather than to a wider or unvalidated location.

    Returns:
        The chosen root directory, or ``None`` when none is usable.
    """
    try:
        from tldw_chatbook.Tools.file_operation_tools import _tool_sandbox_root
        from tldw_chatbook.Tools.workspace_file_roots import allowed_file_roots

        sandbox = _tool_sandbox_root()
        roots = allowed_file_roots(write=True, sandbox_root=sandbox)
    except Exception:
        logger.opt(exception=True).warning("run log: cannot resolve any root")
        return None
    if not roots:
        return None
    # allowed_file_roots returns (sandbox, *workspace_folders); prefer a
    # bound workspace folder, fall back to the sandbox.
    for candidate in roots[1:]:
        return candidate
    return roots[0]


class RunLogWriter:
    """Appends records for ONE run tree to a segmented log.

    Constructed unbound (the run id does not exist until ``_run_one`` calls
    ``create_run``), then bound once by the primary run. Child runs share
    the instance, and therefore the record counter, so parent and child
    record numbers can never collide.
    """

    def __init__(
        self,
        *,
        dir_name: str | None = None,
        segment_bytes: int | None = None,
        max_record_bytes: int | None = None,
    ) -> None:
        """Build an UNBOUND writer. Explicit args override ``[agents]`` config.

        Args:
            dir_name: Directory name; defaults to ``[agents] run_log_dir_name``.
            segment_bytes: Roll threshold; defaults to
                ``[agents] run_log_segment_bytes``.
            max_record_bytes: Per-record ceiling; defaults to
                ``[agents] run_log_max_record_bytes``.
        """
        self._dir_name = dir_name or str(_setting("run_log_dir_name", DEFAULT_DIR_NAME))
        self._segment_bytes = int(
            segment_bytes
            if segment_bytes is not None
            else _setting("run_log_segment_bytes", DEFAULT_SEGMENT_BYTES)
        )
        self._max_record_bytes = int(
            max_record_bytes
            if max_record_bytes is not None
            else _setting("run_log_max_record_bytes", DEFAULT_MAX_RECORD_BYTES)
        )
        self._lock = threading.Lock()
        self._counter = 0
        self._segment_index = 1
        self._segment_size = 0
        self._active = False
        self.log_dir: Path | None = None

    @property
    def is_active(self) -> bool:
        """Whether records are currently being written."""
        return self._active

    def bind(self, run_id: str) -> None:
        """Bind to ``run_id`` and create its directory. Idempotent.

        Args:
            run_id: The PRIMARY run's id. Later calls are ignored so a
                child run never rebinds its parent's writer.
        """
        if self.log_dir is not None:
            return
        if not _setting("run_log_enabled", True):
            self._active = False
            return
        root = resolve_log_root()
        if root is None:
            self._active = False
            return
        try:
            base = root / self._dir_name
            base.mkdir(parents=True, exist_ok=True)
            gitignore = base / ".gitignore"
            if not gitignore.exists():
                # Created only if absent: writing into a user's repository
                # is itself a mutation.
                gitignore.write_text("*\n", encoding="utf-8")
            run_dir = base / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.opt(exception=True).warning(
                "run log: cannot create log directory; logging disabled"
            )
            self._active = False
            return
        self.log_dir = run_dir
        self._active = True

    def _segment_path(self) -> Path:
        assert self.log_dir is not None
        return self.log_dir / f"logs.{self._segment_index:04d}.txt"

    def _write_bytes(self, path: Path, payload: bytes, *, sync: bool = False) -> None:
        """Append ``payload`` to ``path``.

        ``flush()`` on every record survives a process crash. ``fsync`` is
        reserved for segment rolls and run end (``sync=True``): calling it
        per record into a user's project directory is wasteful.

        Args:
            path: Target file, opened in append-binary mode.
            payload: Bytes to append.
            sync: Whether to force an ``fsync`` after flushing.
        """
        import os

        with open(path, "ab") as handle:
            handle.write(payload)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())

    def append(
        self,
        *,
        run_id: str,
        kind: str,
        type: str,
        content: str,
        tool: str = "",
        status: str = "",
        call_id: str = "",
    ) -> int | None:
        """Append one record and return its number.

        Args:
            run_id: Id of the run this record belongs to (parent or child).
            kind: ``primary`` or ``subagent``.
            type: ``model``, ``tool_call``, ``tool_result``, or ``spawn``.
            content: Full, untruncated text.
            tool: Tool name, when applicable.
            status: ``ok`` / ``error``, when applicable.
            call_id: Provider ``tool_call_id``, when applicable.

        Returns:
            The assigned record number, or ``None`` when the writer is
            inactive or the write failed. Never raises.
        """
        if not self._active or self.log_dir is None:
            return None
        with self._lock:
            truncated_from = 0
            body = content.encode("utf-8")
            if len(body) > self._max_record_bytes:
                truncated_from = len(body)
                # Cut on a character boundary, then re-encode.
                body = body[: self._max_record_bytes]
                content = body.decode("utf-8", "ignore")
            self._counter += 1
            record = RunLogRecord(
                number=self._counter,
                run_id=run_id,
                kind=kind,
                type=type,
                ts=_now_iso(),
                content=content,
                tool=tool,
                status=status,
                call_id=call_id,
                truncated_from=truncated_from,
            )
            payload = encode_record(record)
            # Roll BEFORE writing: a record must never span segments, or
            # bytes=-exact parsing (which assumes one file) breaks.
            if self._segment_size and self._segment_size + len(payload) > (
                self._segment_bytes
            ):
                try:
                    # fsync the segment being retired; it will not be
                    # appended to again.
                    self._write_bytes(self._segment_path(), b"", sync=True)
                except Exception:  # noqa: BLE001 — durability is best-effort
                    logger.opt(exception=True).warning("run log: segment fsync failed")
                self._segment_index += 1
                self._segment_size = 0
            try:
                self._write_bytes(self._segment_path(), payload)
            except Exception:
                logger.opt(exception=True).warning(
                    "run log: append failed; logging disabled for this run"
                )
                self._active = False
                return None
            self._segment_size += len(payload)
            return record.number

    def write_manifest(self, metadata: dict) -> None:
        """Write run-level convenience metadata. Never raises.

        The manifest is deliberately NOT load-bearing: segment discovery is
        glob + sort (``run_log_search.load_records``), so a crashed run that
        never reaches this call is still fully readable.

        Args:
            metadata: Run-level fields (model, budget, status, supersession).
        """
        if self.log_dir is None:
            return
        import json

        payload = dict(metadata)
        try:
            payload["segments"] = [p.name for p in sorted(self.log_dir.glob("logs.*.txt"))]
            payload["record_count"] = self._counter
            self._write_bytes(
                self.log_dir / MANIFEST_NAME,
                json.dumps(payload, indent=2, default=str).encode("utf-8"),
                sync=True,
            )
        except Exception:  # noqa: BLE001 — convenience metadata only
            logger.opt(exception=True).warning("run log: manifest write failed")

    def close(self) -> None:
        """Flush the final segment to disk. Idempotent and always safe."""
        if not self._active or self.log_dir is None:
            return
        try:
            self._write_bytes(self._segment_path(), b"", sync=True)
        except Exception:  # noqa: BLE001 — best-effort durability
            logger.opt(exception=True).warning("run log: final fsync failed")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_run_log_writer.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_log.py Tests/Agents/test_run_log_writer.py
git commit -m "feat(agents): segmented run-log writer with two-phase binding"
```

---

### Task 3: `on_record` hook in the pure loop

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (`LoopDeps`, ~line 256; model turn site ~line 424; content assembly ~line 642)
- Test: `Tests/Agents/test_run_log_on_record.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (the hook is a bare callable).
- Produces: `LoopDeps.on_record: Callable[[str, dict], None] | None = None`, called as `on_record(record_type, payload)` where `record_type` is `"model"`, `"tool_call"`, or `"tool_result"`, and `payload` carries `content`, `tool`, `status`, `call_id`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_on_record.py`:

```python
# Tests/Agents/test_run_log_on_record.py
"""on_record captures FULL fidelity at both loop call sites."""

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

from Tests.Agents.test_agent_runtime import make_deps


def collect():
    seen = []
    return seen, lambda kind, payload: seen.append((kind, payload))


def run(turns, *, invoke=None, budget=None):
    seen, hook = collect()
    deps = make_deps(turns, invoke=invoke)
    deps.on_record = hook
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=budget or RunBudget(max_steps=8, max_model_turns=8),
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    return seen, outcome


def test_model_record_carries_full_text_not_the_200_char_summary():
    long_text = "z" * 5000
    seen, _ = run([ModelTurn(text=long_text)])
    model_records = [p for kind, p in seen if kind == "model"]
    assert model_records and model_records[0]["content"] == long_text


def test_tool_result_record_carries_content_before_truncation():
    big = "q" * 40_000
    turns = [
        ModelTurn(
            text="",
            tool_calls=(ToolCall(name="calculator", args={}, call_id="c1"),),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(
        turns,
        invoke=lambda c: ToolResult(ok=True, content=big),
        budget=RunBudget(max_steps=8, max_model_turns=8, max_tool_result_chars=100),
    )
    results = [p for kind, p in seen if kind == "tool_result"]
    assert results and results[0]["content"] == big
    assert results[0]["call_id"] == "c1"


def test_tool_call_record_carries_full_args():
    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1+1"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(turns)
    calls = [p for kind, p in seen if kind == "tool_call"]
    assert calls and "1+1" in calls[0]["content"]


def test_runtime_tool_results_are_captured_too():
    # find_tools never reaches deps.invoke_tool -- a service-side wrapper
    # would have missed it entirely.
    turns = [
        ModelTurn(
            text="",
            tool_calls=(ToolCall(name="find_tools", args={"query": "x"}, call_id="c1"),),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(turns)
    tools = [p["tool"] for kind, p in seen if kind == "tool_result"]
    assert "find_tools" in tools


def test_failing_hook_never_aborts_the_run():
    def boom(kind, payload):
        raise RuntimeError("log is on fire")

    deps = make_deps([ModelTurn(text="fine")])
    deps.on_record = boom
    config = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert outcome.final_text == "fine"


def test_absent_hook_is_a_no_op():
    deps = make_deps([ModelTurn(text="fine")])
    config = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert outcome.final_text == "fine"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_on_record.py -v`
Expected: FAIL — `AttributeError` / `TypeError` on `deps.on_record`

- [ ] **Step 3: Add the field and both call sites**

In `tldw_chatbook/Agents/agent_runtime.py`, append to `LoopDeps` (after `run_skill_script`):

```python
    # on_record: full-fidelity capture for the run log (run_log.py). Called
    # with (record_type, payload) at the two points where the COMPLETE value
    # exists -- which the step log does not carry, since `add()` truncates
    # model turns to 200 chars and tool results to 2000. Captured in the
    # loop rather than in service wrappers because the loop assembles
    # `content` for EVERY dispatch branch at one point: a wrapper around
    # deps.invoke_tool would silently miss find_tools, load_tools,
    # spawn_subagent, skill_file, install_skill and run_skill_script.
    # `None` (the default) is a no-op: behavior is byte-identical to
    # pre-run-log runs.
    on_record: Callable[[str, dict], None] | None = None
```

Add a module-level helper beside `_catalog_lines`:

```python
def _emit_record(deps: "LoopDeps", record_type: str, **payload) -> int | None:
    """Best-effort run-log capture; a failing writer never aborts a run.

    Args:
        deps: The run's injected dependencies.
        record_type: ``model``, ``tool_call``, or ``tool_result``.
        **payload: ``content``, ``tool``, ``status``, ``call_id``.

    Returns:
        The assigned record number, or ``None`` when logging is off or the
        write failed. Task 7 threads this into the truncation trailer.
    """
    if deps.on_record is None:
        return None
    try:
        return deps.on_record(record_type, payload)
    except Exception:  # noqa: BLE001 — logging is never load-bearing
        logger.opt(exception=True).warning(
            f"on_record hook raised for a {record_type} record; continuing"
        )
        return None
```

At the model-turn site, immediately after `add(STEP_MODEL, summary=turn.text[:200])`:

```python
        _emit_record(
            deps,
            "model",
            content=turn.text,
            tool="",
            status="",
            call_id="",
        )
```

At the dispatch site, replace:

```python
                content = result.content if result.ok else f"ERROR: {result.error}"
```

with:

```python
                content = result.content if result.ok else f"ERROR: {result.error}"

            # Capture BEFORE _truncate_tool_result below: the log is the
            # lossless record, history is the capped view of it. This single
            # point covers every dispatch branch above -- builtin, MCP,
            # skill, runtime tools -- and the review-hook refusal path.
            _emit_record(
                deps,
                "tool_call",
                content=json.dumps(call.args, sort_keys=True, default=str),
                tool=call.name,
                status="",
                call_id=call.call_id,
            )
            _emit_record(
                deps,
                "tool_result",
                content=content,
                tool=call.name,
                status="ok" if verdict == "proceed" else "refused",
                call_id=call.call_id,
            )
```

Note the dedent: both `_emit_record` calls sit at the `for call in calls:` body level, so they run for the refusal path too (where `content` is the verdict string).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_run_log_on_record.py Tests/Agents/test_agent_runtime.py -v`
Expected: PASS — new tests pass and every existing runtime test still passes (the hook defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_run_log_on_record.py
git commit -m "feat(agents): on_record capture hook at the loop's full-fidelity points"
```

---

### Task 4: Wire the writer into AgentService

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`__init__`, `run_turn` ~line 766, `_run_one` ~line 423, `LoopDeps(...)` construction)
- Test: `Tests/Agents/test_run_log_service_wiring.py`

**Interfaces:**
- Consumes: `RunLogWriter` (Task 2), `LoopDeps.on_record` (Task 3).
- Produces: `AgentService(run_log_writer: RunLogWriter | None = None)`; `AgentService.run_log_writer` attribute readable by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_service_wiring.py`:

```python
# Tests/Agents/test_run_log_service_wiring.py
"""The service owns the writer: one counter per run tree, every caller logged."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return db, registry, tmp_path


def chat_call_returning(text):
    def call(**kwargs):
        return {"choices": [{"message": {"content": text}}]}

    return call


def read_all(root: Path):
    run_dirs = list((root / "agent-runs").iterdir())
    run_dirs = [d for d in run_dirs if d.is_dir()]
    assert len(run_dirs) == 1
    records = []
    for segment in sorted(run_dirs[0].glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    return records


def test_a_plain_run_writes_records_without_the_caller_wiring_anything(wired):
    db, registry, root = wired
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    records = read_all(root)
    assert [r.type for r in records] == ["model"]
    assert records[0].content == "hello"
    assert records[0].kind == "primary"


def test_record_numbers_are_unique_across_the_whole_run_tree(wired):
    db, registry, root = wired
    writer = RunLogWriter()
    service = AgentService(
        db, registry, chat_call=chat_call_returning("x"), run_log_writer=writer
    )
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    # Simulate a child appending through the same shared writer.
    writer.append(run_id="child", kind="subagent", type="model", content="child work")
    numbers = [r.number for r in read_all(root)]
    assert numbers == sorted(set(numbers))


def test_disabled_writer_leaves_the_run_untouched(wired, monkeypatch):
    db, registry, root = wired
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    _run_id, outcome = service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    assert outcome.final_text == "hello"
    assert not (root / "agent-runs").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_service_wiring.py -v`
Expected: FAIL — `TypeError: AgentService.__init__() got an unexpected keyword argument 'run_log_writer'`

- [ ] **Step 3: Wire it**

In `agent_service.py` `__init__`, add the parameter and store it:

```python
        run_log_writer: "RunLogWriter | None" = None,
```
```python
        # Constructed here (or by the caller) so EVERY caller gets a log --
        # on_step is passed in by the Console bridge, so anything riding
        # that hook would silently log nothing for other callers.
        from .run_log import RunLogWriter as _RunLogWriter

        self.run_log_writer = run_log_writer or _RunLogWriter()
```

In `_run_one`, immediately after `run_id = self.db.create_run(...)`:

```python
        # Two-phase: the writer was constructed before any run id existed.
        # Only the PRIMARY run binds; a child finds it already bound.
        if agent_kind == AGENT_KIND_PRIMARY:
            self.run_log_writer.bind(run_id)
```

Define the per-run adapter beside the other closures in `_run_one`:

```python
        def on_record(record_type: str, payload: dict) -> int | None:
            # MUST return the record number: Task 7 threads it into the
            # truncation trailer so a cut result points at its full copy.
            return self.run_log_writer.append(
                run_id=run_id,
                kind=agent_kind,
                type=record_type,
                content=str(payload.get("content", "")),
                tool=str(payload.get("tool", "")),
                status=str(payload.get("status", "")),
                call_id=str(payload.get("call_id", "")),
            )
```

Write the manifest once the run tree finishes, at the end of `run_turn` (it
needs run-level metadata the writer does not have, including supersession):

```python
        self.run_log_writer.write_manifest(
            {
                "run_id": run_id,
                "model": config.model,
                "api_endpoint": api_endpoint,
                "allowed_tools": list(config.allowed_tools),
                "budget": dataclasses.asdict(config.budget),
                "status": outcome.status,
                "superseded_run_id": supersede_run_id or "",
                "total_tokens": outcome.total_tokens,
            }
        )
        self.run_log_writer.close()
```

And pass it in the `LoopDeps(...)` construction:

```python
            on_record=on_record,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_run_log_service_wiring.py Tests/Agents/test_agent_service.py -v`
Expected: PASS — new tests plus the existing service suite.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_run_log_service_wiring.py
git commit -m "feat(agents): own the run-log writer in AgentService"
```

---

### Task 5: Log search (pure)

**Files:**
- Create: `tldw_chatbook/Agents/run_log_search.py`
- Test: `Tests/Agents/test_run_log_search.py`

**Interfaces:**
- Consumes: `RunLogRecord`, `iter_records` (Task 1).
- Produces:
  - `search_records(records: list[RunLogRecord], *, contains: str = "", pattern: str = "", tool: str = "", type: str = "", status: str = "", kind: str = "", from_record: int = 0, to_record: int = 0, context: int = 0, limit: int = 50) -> list[RunLogRecord]`
  - `load_records(log_dir: Path) -> list[RunLogRecord]`
  - `format_results(records: list[RunLogRecord], *, max_chars: int = 400) -> str`
  - `MAX_REGEX_SCAN_CHARS: int = 500`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_search.py`:

```python
# Tests/Agents/test_run_log_search.py
"""Search: literal by default, structured filters, bounded regex."""

from tldw_chatbook.Agents.run_log_format import RunLogRecord
from tldw_chatbook.Agents.run_log_search import (
    MAX_REGEX_SCAN_CHARS,
    format_results,
    search_records,
)


def rec(number, content, **kw):
    base = dict(
        number=number,
        run_id="r",
        kind="primary",
        type="tool_result",
        ts="t",
        content=content,
    )
    base.update(kw)
    return RunLogRecord(**base)


CORPUS = [
    rec(1, "opened the config file", tool="read_file", status="ok"),
    rec(2, "connection refused", tool="web_search", status="error"),
    rec(3, "thinking about it", type="model"),
    rec(4, "wrote the config file", tool="write_file", status="ok"),
]


def test_literal_contains_is_the_default_and_is_not_a_regex():
    assert [r.number for r in search_records(CORPUS, contains="config file")] == [1, 4]
    # A regex metacharacter is matched literally, never compiled.
    assert search_records(CORPUS, contains="config.file") == []


def test_literal_search_is_unbounded_by_line_length():
    long_record = [rec(9, "x" * 5000 + "NEEDLE")]
    assert len(search_records(long_record, contains="NEEDLE")) == 1


def test_structured_filters_compose():
    hits = search_records(CORPUS, status="error")
    assert [r.number for r in hits] == [2]
    assert [r.number for r in search_records(CORPUS, type="model")] == [3]
    assert [r.number for r in search_records(CORPUS, tool="write_file")] == [4]


def test_record_range_slices():
    assert [r.number for r in search_records(CORPUS, from_record=3)] == [3, 4]
    assert [r.number for r in search_records(CORPUS, to_record=2)] == [1, 2]


def test_context_returns_neighbours_in_order_without_duplicates():
    hits = search_records(CORPUS, contains="refused", context=1)
    assert [r.number for r in hits] == [1, 2, 3]


def test_regex_mode_is_opt_in_and_scan_bounded():
    assert [r.number for r in search_records(CORPUS, pattern=r"conn\w+")] == [2]
    # Beyond the scan window the pattern cannot match, by design.
    far = [rec(9, "y" * (MAX_REGEX_SCAN_CHARS + 50) + "NEEDLE")]
    assert search_records(far, pattern="NEEDLE") == []
    assert len(search_records(far, contains="NEEDLE")) == 1


def test_invalid_regex_returns_no_hits_rather_than_raising():
    assert search_records(CORPUS, pattern="(unclosed") == []


def test_limit_caps_results():
    assert len(search_records(CORPUS, limit=2)) == 2


def test_format_results_is_readable_and_truncates_long_content():
    text = format_results([rec(7, "z" * 900, tool="read_file")], max_chars=50)
    assert "record 000007" in text
    assert "read_file" in text
    assert len(text) < 300


def test_format_results_reports_no_matches():
    assert "no matching records" in format_results([]).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.run_log_search'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Agents/run_log_search.py`:

```python
# tldw_chatbook/Agents/run_log_search.py
"""Query the run log: literal by default, structured filters, bounded regex.

Pure module. Literal substring search is the DEFAULT and carries no
line-length cap: `str.__contains__` is linear and cannot backtrack. Regex
is opt-in and scan-bounded, because Python's `re` has no match timeout and
`agent_service._call_with_timeout` abandons rather than kills its worker
thread -- the same reasoning behind `grep_files`' own 500-char window.
"""

from __future__ import annotations

import re
from pathlib import Path

from .run_log_format import RunLogRecord, iter_records

#: Per-record scan window for opt-in regex mode; mirrors
#: `file_operation_tools._MAX_GREP_LINE_SEARCH_CHARS`.
MAX_REGEX_SCAN_CHARS = 500


def load_records(log_dir: Path) -> list[RunLogRecord]:
    """Load every complete record from every segment, in order.

    Segment discovery is glob + sort, never the MANIFEST: a crashed run
    writes no manifest, and those are exactly the runs worth inspecting.

    Args:
        log_dir: The run's log directory.

    Returns:
        All records in record-number order; empty when unreadable.
    """
    records: list[RunLogRecord] = []
    try:
        for segment in sorted(log_dir.glob("logs.*.txt")):
            records.extend(iter_records(segment.read_bytes()))
    except OSError:
        return records
    return sorted(records, key=lambda r: r.number)


def search_records(
    records: list[RunLogRecord],
    *,
    contains: str = "",
    pattern: str = "",
    tool: str = "",
    type: str = "",
    status: str = "",
    kind: str = "",
    from_record: int = 0,
    to_record: int = 0,
    context: int = 0,
    limit: int = 50,
) -> list[RunLogRecord]:
    """Filter ``records``; return hits plus optional neighbouring context.

    Args:
        records: All loaded records, in order.
        contains: Literal substring (case-insensitive). Never compiled.
        pattern: Opt-in regex, searched only over the first
            ``MAX_REGEX_SCAN_CHARS`` characters of each record.
        tool: Exact tool-name filter.
        type: Exact record-type filter.
        status: Exact status filter.
        kind: Exact agent-kind filter.
        from_record: Inclusive lower bound on record number.
        to_record: Inclusive upper bound on record number.
        context: Include this many records either side of each hit.
        limit: Maximum records returned.

    Returns:
        Matching records in record order, deduplicated, capped at ``limit``.
    """
    compiled = None
    if pattern:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
    needle = contains.lower()
    hit_indexes: list[int] = []
    for index, record in enumerate(records):
        if from_record and record.number < from_record:
            continue
        if to_record and record.number > to_record:
            continue
        if tool and record.tool != tool:
            continue
        if type and record.type != type:
            continue
        if status and record.status != status:
            continue
        if kind and record.kind != kind:
            continue
        if needle and needle not in record.content.lower():
            continue
        if compiled is not None and not compiled.search(
            record.content[:MAX_REGEX_SCAN_CHARS]
        ):
            continue
        hit_indexes.append(index)
    selected: set[int] = set()
    for index in hit_indexes:
        low = max(0, index - context)
        high = min(len(records) - 1, index + context)
        selected.update(range(low, high + 1))
    return [records[i] for i in sorted(selected)][:limit]


def format_results(records: list[RunLogRecord], *, max_chars: int = 400) -> str:
    """Render results for the model.

    Args:
        records: Records to render.
        max_chars: Per-record content ceiling in the rendering.

    Returns:
        One block per record, or a plain no-matches line.
    """
    if not records:
        return "No matching records."
    blocks = []
    for record in records:
        body = record.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"… (+{len(record.content) - max_chars} chars)"
        blocks.append(
            f"record {record.number:06d} [{record.type}"
            f"{'/' + record.tool if record.tool and record.tool != '-' else ''}"
            f"{'/' + record.status if record.status and record.status != '-' else ''}]"
            f"\n{body}"
        )
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_run_log_search.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_log_search.py Tests/Agents/test_run_log_search.py
git commit -m "feat(agents): literal-first run-log search with bounded regex mode"
```

---

### Task 6: `search_run_log` runtime tool

Mirrors `install_skill` exactly: schema in `tool_catalog`, `LoopDeps` field, dispatch branch, `AGENT_KIND_PRIMARY`-gated wiring.

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (name constant + `RUNTIME_TOOL_NAMES`)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (schema)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (`LoopDeps` field + dispatch branch)
- Modify: `tldw_chatbook/Agents/agent_service.py` (runtime_schemas gate + closure)
- Test: `Tests/Agents/test_search_run_log_runtime_tool.py`

**Interfaces:**
- Consumes: `load_records`, `search_records`, `format_results` (Task 5); `AgentService.run_log_writer` (Task 4).
- Produces: `SEARCH_RUN_LOG_TOOL_NAME = "search_run_log"`, `SEARCH_RUN_LOG_TOOL_SCHEMA`, `LoopDeps.search_run_log: Callable[[dict], ToolResult] | None`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_search_run_log_runtime_tool.py`:

```python
# Tests/Agents/test_search_run_log_runtime_tool.py
"""search_run_log: primary-only, no catalog slot, dispatched by the loop."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUNTIME_TOOL_NAMES,
    SEARCH_RUN_LOG_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.tool_catalog import SEARCH_RUN_LOG_TOOL_SCHEMA

from Tests.Agents.test_agent_runtime import make_deps


def test_name_is_registered_as_a_runtime_tool():
    assert SEARCH_RUN_LOG_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert SEARCH_RUN_LOG_TOOL_SCHEMA.name == SEARCH_RUN_LOG_TOOL_NAME
    props = SEARCH_RUN_LOG_TOOL_SCHEMA.parameters["properties"]
    assert "contains" in props and "pattern" in props and "from_record" in props


def test_loop_dispatches_to_the_injected_callable():
    seen = {}

    def handler(args):
        seen.update(args)
        return ToolResult(ok=True, content="record 000412 [model]")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=SEARCH_RUN_LOG_TOOL_NAME,
                    args={"contains": "refused"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="answered"),
    ]
    deps = make_deps(turns)
    deps.search_run_log = handler
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"contains": "refused"}
    assert outcome.final_text == "answered"


def test_unwired_name_falls_through_to_the_permission_gate():
    # deps.search_run_log is None -> the else branch -> deps.invoke_tool.
    invoked = []

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name=SEARCH_RUN_LOG_TOOL_NAME, args={}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns, invoke=invoke)
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert invoked == [SEARCH_RUN_LOG_TOOL_NAME]
```

Append to `Tests/Agents/test_run_log_service_wiring.py`:

```python
def test_tool_is_offered_to_the_primary_agent_only(wired, monkeypatch):
    from tldw_chatbook.Agents.agent_models import SEARCH_RUN_LOG_TOOL_NAME

    db, registry, root = wired
    offered = []

    def capture(**kwargs):
        names = [t["function"]["name"] for t in kwargs.get("tools", [])]
        offered.append(names)
        return {"choices": [{"message": {"content": "ok"}}]}

    service = AgentService(db, registry, chat_call=capture)
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            budget=RunBudget(max_subagents=0),
        ),
        api_endpoint="openai",
    )
    assert any(SEARCH_RUN_LOG_TOOL_NAME in names for names in offered)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_search_run_log_runtime_tool.py -v`
Expected: FAIL — `ImportError: cannot import name 'SEARCH_RUN_LOG_TOOL_NAME'`

- [ ] **Step 3: Implement across the four modules**

In `agent_models.py`, beside the other runtime-tool names:

```python
SEARCH_RUN_LOG_TOOL_NAME = "search_run_log"
```
and add `SEARCH_RUN_LOG_TOOL_NAME` to the `RUNTIME_TOOL_NAMES` frozenset.

In `tool_catalog.py`, beside `RUN_SKILL_SCRIPT_TOOL_SCHEMA`:

```python
SEARCH_RUN_LOG_TOOL_SCHEMA = ToolSchema(
    id="runtime:search_run_log",
    name=SEARCH_RUN_LOG_TOOL_NAME,
    description=(
        "Search this run's own complete log. Your context holds a truncated "
        "view; the log holds every model turn, tool call, and tool result in "
        "full. Use it to recover a truncated result or recall an earlier step "
        "instead of re-running work. Prefer 'contains' (literal substring, "
        "searches the whole record); 'pattern' is a regular expression and "
        "only examines each record's first 500 characters."
    ),
    parameters={
        "type": "object",
        "properties": {
            "contains": {
                "type": "string",
                "description": "Literal substring to find (case-insensitive).",
            },
            "pattern": {
                "type": "string",
                "description": "Regular expression; first 500 chars per record.",
            },
            "tool": {"type": "string", "description": "Filter by tool name."},
            "type": {
                "type": "string",
                "description": "Filter by record type: model, tool_call, tool_result.",
            },
            "status": {"type": "string", "description": "Filter: ok or error."},
            "from_record": {"type": "integer", "description": "Lowest record number."},
            "to_record": {"type": "integer", "description": "Highest record number."},
            "context": {
                "type": "integer",
                "description": "Records to include either side of each hit.",
            },
        },
        "required": [],
    },
)
```

In `agent_runtime.py`, add the `LoopDeps` field:

```python
    # search_run_log: the seventh runtime tool (run-log query). Wired ONLY
    # for the top-level agent (agent_kind == primary), like install_skill:
    # a depth-1 child has max_subagents clamped to 0, so its "subtree" is
    # itself and its short history is already in its context -- the tool
    # would buy it nothing while widening what it can see. `None` (the
    # default) means the run is not wired for it and a call by that name
    # falls through to the generic deps.invoke_tool path.
    search_run_log: Callable[[dict], ToolResult] | None = None
```

and the dispatch branch, immediately before the final `else:`:

```python
                elif (
                    call.name == SEARCH_RUN_LOG_TOOL_NAME
                    and deps.search_run_log is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.search_run_log(dict(call.args))
```

(import `SEARCH_RUN_LOG_TOOL_NAME` from `.agent_models` at the top.)

In `agent_service.py` `_run_one`, gate the schema beside the `install_skill` gate:

```python
        if (
            agent_kind == AGENT_KIND_PRIMARY
            and self.run_log_writer.is_active
        ):
            runtime_schemas.append(SEARCH_RUN_LOG_TOOL_SCHEMA)
```

and define the closure:

```python
        def search_run_log(args: dict) -> ToolResult:
            """Query THIS run's log. Reads only what this agent produced."""
            from .run_log_search import format_results, load_records, search_records

            log_dir = self.run_log_writer.log_dir
            if log_dir is None:
                return ToolResult(ok=False, error="No run log is available.")
            try:
                records = load_records(log_dir)
                hits = search_records(
                    records,
                    contains=str(args.get("contains", "")),
                    pattern=str(args.get("pattern", "")),
                    tool=str(args.get("tool", "")),
                    type=str(args.get("type", "")),
                    status=str(args.get("status", "")),
                    from_record=int(args.get("from_record") or 0),
                    to_record=int(args.get("to_record") or 0),
                    context=int(args.get("context") or 0),
                )
            except (TypeError, ValueError) as exc:
                return ToolResult(ok=False, error=f"Invalid search arguments: {exc}")
            return ToolResult(ok=True, content=format_results(hits))
```

Pass `search_run_log=search_run_log if agent_kind == AGENT_KIND_PRIMARY else None` in the `LoopDeps(...)` construction.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_search_run_log_runtime_tool.py Tests/Agents/test_run_log_service_wiring.py Tests/Agents/test_tool_catalog.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/ Tests/Agents/
git commit -m "feat(agents): search_run_log runtime tool, primary agent only"
```

---

### Task 7: Prompt integration

The connective tissue that makes an additive Phase 1 pay off: truncation trailers point at records, and the model is told the log exists — only when it really does.

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (`_truncate_tool_result`)
- Modify: `tldw_chatbook/Agents/agent_service.py` (system-prompt section)
- Test: `Tests/Agents/test_run_log_prompt_integration.py`

**Interfaces:**
- Consumes: `search_run_log` wiring (Task 6).
- Produces: `RUN_LOG_PROMPT_SECTION: str` in `agent_service.py`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_log_prompt_integration.py`:

```python
# Tests/Agents/test_run_log_prompt_integration.py
"""Truncation points at the log; the prompt mentions it only when real."""

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_trailer_points_at_the_record_when_one_exists():
    out = _truncate_tool_result("z" * 100, 10, "grep_files", record_number=412)
    assert "search_run_log" in out
    assert "412" in out


def test_trailer_keeps_the_old_wording_when_there_is_no_record():
    out = _truncate_tool_result("z" * 100, 10, "grep_files", record_number=None)
    assert "search_run_log" not in out
    assert "narrower query" in out


def test_untruncated_content_is_returned_unchanged():
    assert _truncate_tool_result("short", 100, "t", record_number=7) == "short"


def _service(tmp_path, capture):
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db, registry, chat_call=capture)


def _run(service):
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="BASE", budget=RunBudget()),
        api_endpoint="openai",
    )


def test_prompt_mentions_the_log_when_logging_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    prompts = []

    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    _run(_service(tmp_path, capture))
    assert any("search_run_log" in p for p in prompts)
    assert all(p.startswith("BASE") for p in prompts)


def test_prompt_is_silent_about_the_log_when_it_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    prompts = []

    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    _run(_service(tmp_path, capture))
    assert all("search_run_log" not in p for p in prompts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_run_log_prompt_integration.py -v`
Expected: FAIL — `TypeError: _truncate_tool_result() got an unexpected keyword argument 'record_number'`

- [ ] **Step 3: Implement**

In `agent_runtime.py`, change the signature and trailer:

```python
def _truncate_tool_result(
    content: str, max_chars: int, tool_name: str, record_number: int | None = None
) -> str:
```

and replace the return with:

```python
    if record_number is not None:
        recovery = (
            f" The full result is recorded at record {record_number:06d} — "
            f"search_run_log(from_record={record_number}, to_record={record_number})."
        )
    else:
        recovery = (
            " Re-issue the call with a narrower query, or use the tool's "
            "offset/limit arguments to read the rest."
        )
    return (
        content[:max_chars]
        + f"\n\n[truncated: {tool_name} returned {len(content)} characters; "
        f"showing the first {max_chars}.{recovery}]"
    )
```

At the call site, thread through the number `on_record` returned. Change `_emit_record` to return the writer's value, capture it for the `tool_result` record, and pass it:

```python
            record_number = _emit_record(deps, "tool_result", ...)
            ...
            content = _truncate_tool_result(
                content,
                budget.max_tool_result_chars,
                call.name,
                record_number=record_number,
            )
```

`_emit_record` returns `deps.on_record(...)`'s value (the record number) or `None`; `agent_service`'s `on_record` closure must `return self.run_log_writer.append(...)`.

In `agent_service.py`, add the section constant:

```python
RUN_LOG_PROMPT_SECTION = (
    "Run log: every model turn, tool call, and tool result of this run is "
    "recorded in full to a log file. Your context holds a truncated view of "
    "it. When a result was truncated, or you need something from earlier in "
    "this run, call search_run_log to read the complete record instead of "
    "re-running the work or guessing. Prefer the 'contains' argument (a "
    "literal substring, searched over the whole record) over 'pattern'. "
    "Search for specific content you know you need rather than browsing."
)
```

and in `_make_call_model`, append it to `config.system_prompt` when — and only when — this run wired the tool:

```python
            system_content = config.system_prompt
            if log_active:
                system_content = f"{system_content}\n\n{RUN_LOG_PROMPT_SECTION}"
```

where `_make_call_model` gains a `log_active: bool` parameter, passed from `_run_one` as `agent_kind == AGENT_KIND_PRIMARY and self.run_log_writer.is_active` — the same condition that gates `SEARCH_RUN_LOG_TOOL_SCHEMA`, so the prompt can never advertise a tool the run does not have.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/ -v`
Expected: PASS — the whole Agents suite, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/ Tests/Agents/
git commit -m "feat(agents): point truncation trailers at the run log and prompt for it"
```

---

### Task 8: Full-suite verification and live run

**Files:**
- Test: no new files; runs the existing suites.

- [ ] **Step 1: Run the full agent and chat suites**

Run: `python -m pytest Tests/Agents/ Tests/Chat/ -q`
Expected: PASS with **zero novel regressions** against an `origin/dev` baseline. Capture the baseline first if unsure:
`git stash && python -m pytest Tests/Agents/ Tests/Chat/ -q | tail -5 && git stash pop`

- [ ] **Step 2: Verify the additive guarantee**

Run: `python -m pytest Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service.py -q`
Expected: PASS. These predate this work and exercise the loop with `on_record=None`; they must not have needed edits. If any required a change, the additive guarantee was broken — stop and reassess.

- [ ] **Step 3: Live run against a real provider**

Per the repository's live-verification rule (tests alone have repeatedly missed defects here), and using `.claude/skills/verify` to drive the TUI:

1. Set `TLDW_CONFIG_PATH` to a scratch config so the live DB is untouched.
2. Enable a file tool so a run produces a large result: `[tools] read_file_enabled = true`.
3. Bind a scratch workspace folder with `rw` access.
4. Send a Console message that makes the agent read a large file, so a result exceeds `max_tool_result_chars`.
5. Confirm on disk: `<workspace>/agent-runs/<run_id>/logs.0001.txt` exists, contains `#@#`-anchored records, and holds the **full** result while the transcript shows the truncated one.
6. Confirm `<workspace>/agent-runs/.gitignore` contains `*`.
7. Ask the agent a question answerable only from the truncated remainder, and confirm it calls `search_run_log` and recovers it.

- [ ] **Step 4: Record the evidence**

Write the observed record count, file sizes, and the recovered-content exchange into the task's Implementation Notes. Live evidence, not test output, is what closes this.

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "test(agents): full-suite and live verification of the run log"
```
