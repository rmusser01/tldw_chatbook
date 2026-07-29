# Tests/Agents/test_run_log_workspace_isolation.py
"""TASK-1270: the run log in a BOUND WORKSPACE folder must be unreadable by
sub-agents through `glob_files`/`grep_files`, exactly like the
sandbox-fallback case covered by `test_run_log_sandbox_isolation.py`.

`RunLogWriter.bind()` used to dot the log directory name (`.agent-runs`,
excluded from `glob_files`/`grep_files` by `_is_hidden_within`) ONLY when
`resolve_log_root()` resolved via the SANDBOX FALLBACK. When a read-write
workspace folder was bound instead, the directory stayed undotted
(`agent-runs`), on a premise recorded in the design spec §9.4 and in
`run_log.py`'s own comments that was CORRECT when it was written:
`glob_files`/`grep_files` globbed `_tool_sandbox_root()` alone and could not
reach a workspace folder root at all.

TASK-850 ("Scope glob_files and grep_files to workspace folder roots")
invalidated that premise: both tools now resolve every root
`allowed_file_roots()` returns -- the sandbox AND every bound workspace
folder -- via `_iter_candidates_across_roots`
(`tldw_chatbook/Tools/file_operation_tools.py`, `GlobFiles.execute`/
`GrepFiles.execute`). An undotted workspace-folder log became reachable by
a sub-agent (which inherits its parent's tool allow-list, `spawn`'s default
in `agent_service.py`) through them -- the exact disclosure the
sandbox-fallback dotting was introduced to prevent, reopened for the
workspace case by an unrelated change landing on `dev`.

FIX (TASK-1270, 2026-07-28): `bind()` now dots the directory name
UNCONDITIONALLY -- in both the sandbox-fallback and the bound-workspace
case -- deleting the sandbox-vs-workspace conditional entirely, so the
tests below assert the invariant as ordinary (non-`xfail`) tests. See the
module-level comment above `DEFAULT_DIR_NAME` in `run_log.py` for the full
history, and `task-1270-report.md` at the repository root for the record
of the blocked interim state this superseded.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunBudget,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
from tldw_chatbook.Agents.run_log_search import load_records
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import GlobFiles, GrepFiles


class _AllowGate:
    """Bypasses BuiltinToolProvider's approval machinery for these tests.

    `grep_files`/`glob_files` carry the "reads" risk tag, which floors them
    to `ask` under the real gate. These tests care about path-level
    containment (can the tool even SEE the log directory), not the
    approval round trip.
    """

    def check(self, tool):
        return None


def _workspace_seams(monkeypatch, sandbox: Path, workspace: Path) -> None:
    """Simulate a genuinely bound READ-WRITE workspace folder, OUTSIDE the
    sandbox root -- the common, real-world binding. (The case where a
    bound folder happens to nest INSIDE the sandbox root is a separate,
    already-fixed edge case covered by
    `test_run_log_sandbox_isolation.py::test_workspace_folder_inside_the_
    sandbox_is_dotted_and_hidden_from_grep`.)

    `resolve_log_root()` resolves `allowed_file_roots` via a LOCAL import
    inside its own function body, so patching `workspace_file_roots`'s
    module attribute is enough to redirect it. `GlobFiles.execute`/
    `GrepFiles.execute` instead call a NAME bound once at
    `file_operation_tools` IMPORT time (`from .workspace_file_roots import
    allowed_file_roots` at the top of that module) -- patching only
    `workspace_file_roots.allowed_file_roots` leaves that early-bound
    reference untouched (confirmed empirically: the two names are
    different objects after patching only one), so this ALSO patches
    `file_operation_tools.allowed_file_roots` directly. Both tools resolve
    `_tool_sandbox_root` as a same-module global lookup, so patching that
    one attribute (on `file_operation_tools`) does redirect every
    consumer, unlike `allowed_file_roots`.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox)
    fake_roots = lambda write=False, sandbox_root=None: (sandbox, workspace)
    monkeypatch.setattr(ws_roots, "allowed_file_roots", fake_roots)
    monkeypatch.setattr(file_tools, "allowed_file_roots", fake_roots)


def _grep(pattern: str) -> list[dict]:
    result = asyncio.run(GrepFiles().execute(pattern=pattern))
    return result.get("matches", [])


def _glob(pattern: str) -> list[str]:
    result = asyncio.run(GlobFiles().execute(pattern=pattern))
    return result.get("matches", [])


def _fence(name: str, args: dict) -> str:
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _plant_legacy_record(run_dir: Path, run_id: str, content: str) -> None:
    """Write one on-disk segment, in the writer's own wire format.

    Simulates what an EARLIER version of `RunLogWriter` (pre-migration,
    undotted directory) already left on disk -- a single-segment run
    directory with one record -- without going through `RunLogWriter`
    itself, so the fixture stays independent of the code under test.

    Args:
        run_dir: The run's log directory to create (e.g.
            ``<root>/agent-runs/<run_id>``).
        run_id: The run id recorded in the planted record's header.
        content: The record's full content (e.g. a planted secret).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    record = RunLogRecord(
        number=1,
        run_id=run_id,
        kind="primary",
        type="model",
        ts="2026-01-01T00:00:00.000000Z",
        content=content,
    )
    (run_dir / "logs.0001.txt").write_bytes(encode_record(record))


def test_bound_workspace_folder_log_is_hidden_from_grep_and_glob(tmp_path, monkeypatch):
    """AC #1 / AC #6: a sub-agent must not be able to reach its parent's
    run log via `grep_files` (content) or `glob_files` (path) once the log
    lands in a genuinely bound workspace folder.

    The assertions below check only the tools' OUTCOME (can the
    secret/path be recovered), not any internal detail of how
    `resolve_log_root`/`bind` picked the directory name -- so this keeps
    pinning the invariant even if root resolution is reimplemented
    entirely.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    # Positive control FIRST: prove grep_files/glob_files genuinely scan
    # this workspace folder before trusting a "no matches" result below.
    (workspace / "control.txt").write_text("CONTROL_MARKER_7d2a91\n", encoding="utf-8")
    assert _grep("CONTROL_MARKER_7d2a91"), (
        "positive control failed: grep_files must find visible workspace content"
    )
    assert any("control.txt" in m for m in _glob("**/*")), (
        "positive control failed: glob_files must find visible workspace content"
    )

    writer = RunLogWriter()
    writer.bind("run-secret")
    assert writer.is_active, "writer must activate for a bound workspace folder"
    assert writer.log_dir is not None

    secret = "PARENT_SECRET_API_KEY=sk-live-workspace789"
    writer.append(run_id="run-secret", kind="primary", type="model", content=secret)

    assert _grep("PARENT_SECRET_API_KEY") == [], (
        "grep_files must not be able to read the run log through a "
        "genuinely bound workspace folder"
    )
    leaked_paths = [m for m in _glob("**/*") if str(writer.log_dir) in m]
    assert leaked_paths == [], (
        f"glob_files must not be able to enumerate the run log's files "
        f"through a genuinely bound workspace folder; got {leaked_paths!r}"
    )


def test_spawned_subagent_cannot_read_parents_log_via_grep_files_in_bound_workspace(
    tmp_path, monkeypatch
):
    """AC #1: reproduces the disclosure through a live run, mirroring
    `test_run_log_sandbox_isolation.py::test_spawned_subagent_cannot_read_
    parents_log_via_grep_files` but through a genuinely bound WORKSPACE
    folder instead of the sandbox fallback.

    The PARENT's own turn embeds a secret (captured verbatim into its
    "model" log record before any tool dispatch), it spawns a child, and
    the child tries to read the secret back out through `grep_files` -- a
    tool it inherits through the ordinary allow-list, independent of
    `search_run_log`'s own primary-only gate.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots and to stub `grep_files`'s config gate.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "tools" and key == "grep_files_enabled":
            return True
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider(gate=_AllowGate()))

    secret = "PARENT_SECRET_API_KEY=sk-live-workspace321"
    task = "search the workspace for anything interesting"
    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            f"Noting {secret} before delegating.\n"
                            + _fence(SPAWN_TOOL_NAME, {"task": task})
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": _fence(
                            "grep_files", {"pattern": "PARENT_SECRET"}
                        )
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "found nothing"}}]},  # child's answer
        {"choices": [{"message": {"content": "done"}}]},  # parent's answer
    ]

    def chat(**kwargs):
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", SPAWN_TOOL_NAME, "grep_files"),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert tool_results, "expected the child's grep_files tool_result step"
    assert not any(secret in r or "PARENT_SECRET" in r for r in tool_results)


def test_search_run_log_reads_the_log_in_bound_workspace_configuration(
    tmp_path, monkeypatch
):
    """AC #4: `search_run_log`'s own reader (`run_log_search.load_records`)
    must keep reading the log when it lands in a bound workspace folder.

    It globs `writer.log_dir` directly and never routes through
    `validate_path`/`_is_hidden_within`, so it is unaffected by whichever
    way the directory-naming question above is eventually resolved. This
    passes today and must keep passing after any fix.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active
    writer.append(run_id="run-abc", kind="primary", type="model", content="hello")

    records = load_records(writer.log_dir)
    assert [r.content for r in records] == ["hello"]


def test_search_run_log_reads_the_log_in_sandbox_fallback_configuration(
    tmp_path, monkeypatch
):
    """AC #4, sandbox-fallback sibling of the test above.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox root.
        monkeypatch: Pytest fixture used to redirect root resolution to
            that fake root.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox,),
    )

    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active
    writer.append(run_id="run-abc", kind="primary", type="model", content="hello")

    records = load_records(writer.log_dir)
    assert [r.content for r in records] == ["hello"]


def test_legacy_undotted_log_directory_is_migrated_and_hidden_on_bind(
    tmp_path, monkeypatch
):
    """Upgrade-safety regression: a pre-existing UNDOTTED ``agent-runs``
    tree (left behind by a version predating the unconditional-dotting
    fix) must be migrated under the dotted name the first time `bind()`
    runs against it -- otherwise every historical run log an install has
    ever written stays reachable through `grep_files`/`glob_files` even
    after upgrading to the fixed code.

    Covers the "only the legacy directory exists" shape: `bind()` should
    move it wholesale (a single rename), so the planted secret becomes (a)
    still readable through the app's own reader (`load_records`, which
    globs `log_dir` directly and never consults `_is_hidden_within`), and
    (b) unreachable through `grep_files`, exactly like a freshly-written
    record.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    legacy_root = workspace / "agent-runs"
    secret = "LEGACY_SECRET_API_KEY=sk-live-legacy111"
    _plant_legacy_record(legacy_root / "legacy-run-1", "legacy-run-1", secret)

    writer = RunLogWriter()
    writer.bind("run-new")
    assert writer.is_active, "writer must still activate for the new run"

    # The undotted tree is gone -- moved, not copied and not left behind.
    assert not legacy_root.exists()
    migrated_dir = workspace / ".agent-runs" / "legacy-run-1"
    assert migrated_dir.is_dir()

    # (a) still readable through the app's own reader.
    records = load_records(migrated_dir)
    assert [r.content for r in records] == [secret]

    # (b) unreachable through the generic file tools available to a
    # sub-agent.
    assert _grep("LEGACY_SECRET_API_KEY") == [], (
        "grep_files must not recover a migrated legacy secret"
    )
    leaked_paths = [m for m in _glob("**/*") if "legacy-run-1" in m]
    assert leaked_paths == [], (
        f"glob_files must not enumerate the migrated legacy run directory; "
        f"got {leaked_paths!r}"
    )


def test_legacy_and_dotted_log_directories_both_merge_without_clobbering(
    tmp_path, monkeypatch
):
    """Upgrade-safety regression, "both exist" shape: an install that has
    run both an old (undotted) and a new (dotted) version has BOTH
    `agent-runs` and `.agent-runs` on disk. `bind()` must merge the legacy
    entries into the dotted tree without touching anything already there,
    and the planted legacy secret must end up just as unreachable as in
    the "only legacy exists" case.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    legacy_root = workspace / "agent-runs"
    dotted_root = workspace / ".agent-runs"
    secret = "LEGACY_SECRET_API_KEY=sk-live-legacy222"
    _plant_legacy_record(legacy_root / "legacy-run-2", "legacy-run-2", secret)
    _plant_legacy_record(dotted_root / "existing-run", "existing-run", "EXISTING_DOTTED_CONTENT")

    writer = RunLogWriter()
    writer.bind("run-new")
    assert writer.is_active

    # Legacy tree fully merged away (no collision in this fixture).
    assert not legacy_root.exists()

    # Pre-existing dotted content is untouched.
    existing_records = load_records(dotted_root / "existing-run")
    assert [r.content for r in existing_records] == ["EXISTING_DOTTED_CONTENT"]

    # (a) migrated legacy content still readable through the app's reader.
    migrated_records = load_records(dotted_root / "legacy-run-2")
    assert [r.content for r in migrated_records] == [secret]

    # (b) unreachable through the generic file tools.
    assert _grep("LEGACY_SECRET_API_KEY") == [], (
        "grep_files must not recover a merged legacy secret"
    )
    leaked_paths = [m for m in _glob("**/*") if "legacy-run-2" in m]
    assert leaked_paths == [], (
        f"glob_files must not enumerate the merged legacy run directory; "
        f"got {leaked_paths!r}"
    )


def test_legacy_migration_skips_a_colliding_run_id_without_overwriting(
    tmp_path, monkeypatch
):
    """A same-named run directory on both sides (a `uuid4` collision, not
    expected in practice) must never be silently merged or overwritten --
    `_migrate_legacy_dir` skips it and leaves BOTH copies exactly as they
    were, prioritising "never lose or clobber user data" over fully
    closing the disclosure for this one pathological case.

    Args:
        tmp_path: Pytest-provided temporary directory, used as the parent
            of the fake sandbox and workspace roots.
        monkeypatch: Pytest fixture used to redirect root resolution to
            those fake roots.
    """
    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "genuine-workspace"
    _workspace_seams(monkeypatch, sandbox, workspace)

    legacy_root = workspace / "agent-runs"
    dotted_root = workspace / ".agent-runs"
    _plant_legacy_record(legacy_root / "same-id", "same-id", "LEGACY_COLLIDING_CONTENT")
    _plant_legacy_record(dotted_root / "same-id", "same-id", "DOTTED_ORIGINAL_CONTENT")

    writer = RunLogWriter()
    writer.bind("run-new")
    assert writer.is_active, "a collision during migration must not abort bind()"

    # Neither copy was overwritten or deleted.
    dotted_records = load_records(dotted_root / "same-id")
    assert [r.content for r in dotted_records] == ["DOTTED_ORIGINAL_CONTENT"]
    legacy_records = load_records(legacy_root / "same-id")
    assert [r.content for r in legacy_records] == ["LEGACY_COLLIDING_CONTENT"]

    # The new run's own logging still works despite the collision.
    writer.append(run_id="run-new", kind="primary", type="model", content="hello")
    assert [r.content for r in load_records(writer.log_dir)] == ["hello"]
