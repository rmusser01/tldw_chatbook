# Tests/Agents/test_run_log_workspace_isolation.py
"""TASK-1270: the run log in a BOUND WORKSPACE folder is readable by
sub-agents through `glob_files`/`grep_files`.

`RunLogWriter.bind()` only dots the log directory name (`.agent-runs`,
excluded from `glob_files`/`grep_files` by `_is_hidden_within`) when
`resolve_log_root()` resolved via the SANDBOX FALLBACK -- see
`test_run_log_sandbox_isolation.py`. When a read-write workspace folder is
bound instead, the directory stays undotted (`agent-runs`), on the premise
recorded in the design spec §9.4 and in `run_log.py`'s own comments that
`glob_files`/`grep_files` glob `_tool_sandbox_root()` alone and can never
reach a workspace folder root.

**That premise is false as of TASK-850** ("Scope glob_files and grep_files
to workspace folder roots"): both tools now resolve every root
`allowed_file_roots()` returns -- the sandbox AND every bound workspace
folder -- via `_iter_candidates_across_roots`
(`tldw_chatbook/Tools/file_operation_tools.py`, `GlobFiles.execute`/
`GrepFiles.execute`). A workspace-folder log is therefore undotted inside
a root those tools now search, and a sub-agent (which inherits its
parent's tool allow-list, `spawn`'s default in `agent_service.py`) can
`grep_files`/`glob_files` its way to the parent's entire log -- the exact
disclosure the sandbox-fallback dotting was introduced to prevent,
reopened for the workspace case by an unrelated change landing on `dev`.

STATUS: reproduced below, NOT YET FIXED. The designed remedy -- dot the
directory name unconditionally in `bind()`, deleting the
sandbox-fallback-only conditional entirely -- also flips the literal
directory-name string asserted by roughly two dozen PRE-EXISTING tests
across `Tests/Agents/test_run_log_writer.py` and
`Tests/Agents/test_run_log_service_wiring.py` (every one of them drives
the writer through a bare `resolve_log_root` monkeypatch, which reads back
as "not the sandbox fallback" i.e. undotted under the CURRENT
conditional), plus `test_bound_workspace_folder_keeps_the_undotted_name`
and `test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name`
in this file's sibling suites -- both of which explicitly frame themselves
as a "regression guard" against dotting every workspace folder. Landing
the fix therefore requires updating that whole pre-existing set, which
this pass is not authorized to do (pre-existing tests must not be edited
to make something pass; a test that fails as a result of the change must
be reported, not silently rewritten). See `task-1270-report.md` at the
repository root for the ready-to-apply diff, the full list of tests it
flips, and the recommended next step.

The tests below are marked `xfail(strict=True)` rather than left as bare
failures so the suite stays green while the gap remains open, and so
these tests loudly XPASS (itself a failure under `strict=True`) the
moment someone applies a fix without also revisiting this file --
catching exactly the "quietly stopped being true" failure mode this task
exists to close.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunBudget,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_search import load_records
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import GlobFiles, GrepFiles

_XFAIL_REASON = (
    "TASK-1270: designed fix (dot the log directory name unconditionally) "
    "is blocked -- it flips ~22 pre-existing assertions in "
    "test_run_log_writer.py/test_run_log_service_wiring.py that this pass "
    "is not authorized to edit. See task-1270-report.md."
)


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


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_bound_workspace_folder_log_is_hidden_from_grep_and_glob(tmp_path, monkeypatch):
    """AC #1 / AC #6: a sub-agent must not be able to reach its parent's
    run log via `grep_files` (content) or `glob_files` (path) once the log
    lands in a genuinely bound workspace folder. The assertions below
    check only the tools' OUTCOME (can the secret/path be recovered), not
    any internal detail of how `resolve_log_root`/`bind` picked the
    directory name -- so this keeps pinning the invariant even if root
    resolution is reimplemented entirely.
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


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_spawned_subagent_cannot_read_parents_log_via_grep_files_in_bound_workspace(
    tmp_path, monkeypatch
):
    """AC #1: reproduces the disclosure through a live run, mirroring
    `test_run_log_sandbox_isolation.py::test_spawned_subagent_cannot_read_
    parents_log_via_grep_files` but through a genuinely bound WORKSPACE
    folder instead of the sandbox fallback. The PARENT's own turn embeds a
    secret (captured verbatim into its "model" log record before any tool
    dispatch), it spawns a child, and the child tries to read the secret
    back out through `grep_files` -- a tool it inherits through the
    ordinary allow-list, independent of `search_run_log`'s own
    primary-only gate.
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
    must keep reading the log when it lands in a bound workspace folder --
    it globs `writer.log_dir` directly and never routes through
    `validate_path`/`_is_hidden_within`, so it is unaffected by whichever
    way the directory-naming question above is eventually resolved. This
    passes today and must keep passing after any fix.
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
    """AC #4, sandbox-fallback sibling of the test above."""
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
