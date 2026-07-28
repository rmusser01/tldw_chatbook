# Tests/Agents/test_run_log_sandbox_isolation.py
"""Final-review CRITICAL 2: the sandbox-fallback log must stay invisible to
the generic file tools.

When no read-write workspace folder is bound -- confirmed the REAL default
by a live probe (see Task 8's live-probe finding in the SDD progress log) --
`resolve_log_root()` falls back to `_tool_sandbox_root()`, which is EXACTLY
the root `glob_files`/`grep_files` are rooted at (spec §9.4: those two tools
glob/grep the sandbox root directly and never consult `allowed_file_roots`,
unlike `read_file`). Because §9.1 deliberately made the log directory name
undotted (`agent-runs`) so it reads as a user-visible artifact inside a
BOUND WORKSPACE folder, `_is_hidden_within` does not exclude it -- and that
same undotted name, reached via the sandbox fallback instead, is a plain
directory those two tools happily enumerate and read. A spawned sub-agent
inherits its parent's tool allow-list (`spawn`'s default in
agent_service.py), so a child running `grep_files` could read its PARENT's
entire log, directly contradicting `spawn_subagent`'s promise ("It sees
only the task text you pass") and bypassing `search_run_log`'s own
primary-only gate (Task 6) entirely -- through a completely different pair
of tools.

Fix: `bind()` dots the directory name (`.agent-runs`) specifically in the
sandbox-fallback case, reported by `resolve_log_root()` itself via a
thread-local side channel (see `run_log._root_kind`) rather than guessed by
inspecting the resolved path's name. Dotting does not break
`search_run_log`'s own reader (`run_log_search.load_records` globs the
directory directly, never through `validate_path`/`_is_hidden_within`) --
it only removes the directory from what `glob_files`/`grep_files`/
`read_file` can see, which in the app-internal sandbox case is exactly the
intent.
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
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import GrepFiles


class _AllowGate:
    """Bypasses BuiltinToolProvider's approval machinery for these tests.

    `grep_files` carries the "reads" risk tag, which floors it to `ask`
    under the real gate. These tests care about path-level containment
    (can `grep_files` even SEE the log directory), not the approval round
    trip, so the provider is handed a gate that always allows.
    """

    def check(self, tool):
        return None


def _fallback_seams(monkeypatch, sandbox: Path):
    """Simulate the REAL default configuration: no rw workspace folder bound.

    Both `resolve_log_root()` (via its own local imports) and
    `GrepFiles.execute` (a bare module-level reference) resolve
    `_tool_sandbox_root` dynamically from `file_operation_tools` at call
    time, so patching the one module attribute redirects both consumers to
    the SAME sandbox directory -- exactly the real-world condition this
    finding depends on.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox,),
    )


def _grep(pattern: str) -> list[dict]:
    result = asyncio.run(GrepFiles().execute(pattern=pattern))
    return result.get("matches", [])


def test_sandbox_fallback_directory_is_dotted_and_hidden_from_grep(
    tmp_path, monkeypatch
):
    sandbox = tmp_path / "sandbox"
    _fallback_seams(monkeypatch, sandbox)

    # Positive control FIRST: prove grep_files genuinely scans this sandbox
    # before trusting a "no matches" result below for the log directory --
    # a plain, non-hidden file's content IS found.
    (sandbox / "control.txt").write_text("CONTROL_MARKER_9f1c\n", encoding="utf-8")
    assert _grep("CONTROL_MARKER_9f1c"), (
        "positive control failed: grep_files must find visible sandbox content"
    )

    # The real writer, default dir_name, REAL bind() naming logic -- no
    # dir_name override, so this exercises exactly what a production run
    # does under the sandbox fallback.
    writer = RunLogWriter()
    writer.bind("run-secret")
    assert writer.is_active, "writer must still activate under the fallback"
    assert writer.log_dir is not None
    assert writer.log_dir.parent.name == ".agent-runs", (
        f"expected the dotted fallback directory name, got "
        f"{writer.log_dir.parent.name!r}"
    )
    writer.append(
        run_id="run-secret",
        kind="primary",
        type="model",
        content="PARENT_SECRET_API_KEY=sk-live-abc123",
    )

    assert _grep("PARENT_SECRET_API_KEY") == [], (
        "grep_files must not be able to read the sandbox-fallback run log"
    )


def test_bound_workspace_folder_keeps_the_undotted_name(tmp_path, monkeypatch):
    """Sibling case: when `resolve_log_root()` reports a bound workspace
    folder (not the fallback), the directory must stay undotted -- it is
    meant to be a user-visible artifact there (spec §3.3), and the fix must
    not regress that by dotting unconditionally.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox = tmp_path / "sandbox"
    workspace = tmp_path / "workspace"
    sandbox.mkdir()
    workspace.mkdir()
    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox, workspace),
    )

    writer = RunLogWriter()
    writer.bind("run-1")
    assert writer.is_active
    assert writer.log_dir.parent.name == "agent-runs"
    assert writer.log_dir.parent.parent == workspace


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_spawned_subagent_cannot_read_parents_log_via_grep_files(tmp_path, monkeypatch):
    """Reproduces the reviewer's finding directly through a live run: the
    PARENT's own turn embeds a secret (captured verbatim into its "model"
    log record before any tool dispatch), it spawns a child, and the child
    tries to read the secret back out through `grep_files` -- a tool it
    inherits through the ordinary allow-list, completely independent of
    `search_run_log`'s own primary-only gate.
    """
    sandbox = tmp_path / "sandbox"
    _fallback_seams(monkeypatch, sandbox)

    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "tools" and key == "grep_files_enabled":
            return True
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider(gate=_AllowGate()))

    secret = "PARENT_SECRET_API_KEY=sk-live-abc123"
    task = "search the sandbox for anything interesting"
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
        _svc_fence("grep_files", {"pattern": "PARENT_SECRET"}),  # child tries
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

    # The child's own persisted tool_result for its grep_files call must
    # never contain the parent's secret.
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert tool_results, "expected the child's grep_files tool_result step"
    assert not any(secret in r or "PARENT_SECRET" in r for r in tool_results)
