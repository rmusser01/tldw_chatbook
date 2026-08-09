"""TASK-696: coverage for two ACs previously satisfied only by inspection.

Both behaviors were verified manually during TASK-545 P2's review and never
protected by the suite -- a regression would pass CI. Both tests here drive
REAL objects (the real tool, the real gate, the real DB) rather than the
synthetic doubles the neighbouring suites use, because "the double behaves"
is exactly what those reviews found insufficient.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY


class _SessionApprovalService:
    """`unified_mcp_service`-shaped double: the REAL gate's read/write seam."""

    def __init__(self) -> None:
        self._approved: set[tuple[str, str]] = set()

    def get_kill_switch(self) -> bool:
        return False

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self._approved.add((server_key, tool_name))

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self._approved


class _RealToolProvider:
    """Provider double that serves REAL Tool objects by name."""

    def __init__(self, *tools) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def tool_for(self, name: str):
        return self._tools.get(name)


@pytest.mark.unit
def test_a_real_reads_tagged_tool_reaches_the_approval_card_not_silence(
    tmp_path, monkeypatch
):
    """AC#1: enabling `read_file` produces a PROMPT, not a silent execution.

    Drives the real `ReadFileTool` (risk_tags `("reads",)`) through the real
    `BuiltinToolGate` and `build_tool_review_hook`. The chain under test:
    empty permission payload -> inherited allow -> `resolve_builtin_state`
    floors a high-risk tag to "ask" -> the hook emits a pending approval row
    for the card. The only prior hook-level risk coverage used the synthetic
    `_FakeMutatingRiskyTool`; a regression in the REAL tool's tags (or in
    the floor consulting them) kept the suite green while `read_file` ran
    without asking.
    """
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Tools.file_operation_tools import ReadFileTool

    # Hermetic (review finding): the hook's path precheck would otherwise
    # touch the real sandbox root and workspace-registry DB from a unit
    # test. Same seams the sibling precheck test patches.
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    def _no_registry():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _no_registry)

    tool = ReadFileTool()
    assert "reads" in tool.risk_tags, "precondition: the real tag set"

    gate = BuiltinToolGate(_SessionApprovalService())
    rows = []

    def request_approvals(pending):
        rows.extend(pending)
        return {row.call_id or row.llm_name: "deny" for row in pending}

    hook = build_tool_review_hook(
        gate, _RealToolProvider(tool), None, request_approvals
    )
    verdicts = hook(
        [ToolCall(name="read_file", args={"file_path": "notes.md"}, call_id="c1")],
        "run-1",
    )

    assert [row.llm_name for row in rows] == ["read_file"], (
        "the real reads-tagged tool never reached the approval card -- it "
        f"would have executed silently: {rows}"
    )
    assert rows[0].server_key == BUILTIN_TOOL_SERVER_KEY
    assert rows[0].reason == "risk_floored", rows[0]
    assert verdicts.get("c1") not in (None, "proceed"), (
        f"the user's refusal did not reach the runtime: {verdicts}"
    )


@pytest.mark.integration
def test_create_note_persists_through_a_real_db_on_a_worker_thread(
    tmp_path, monkeypatch
):
    """AC#2: the design-spec claim the nominal test never proved.

    `Tests/Agents/test_builtin_gate_live_tools.py` monkeypatches
    `NotesInteropService` away, proving only that `asyncio.run` works off
    the main thread. This runs the REAL `CreateNoteTool.execute` on a real
    non-main thread against a real `CharactersRAGDB` file, twice over:

    * the PRODUCTION write path -- `_get_db` constructs its own
      `CharactersRAGDB` (same file, `client_id=user_id`) ON the worker
      thread, exactly as a live agent run does, and the row must be durable
      to a main-thread reader of the same file; and
    * the SINGLE-INSTANCE seam the spec named -- `threading.local`
      connections + `check_same_thread=False` -- by reading through THIS
      test's own instance from BOTH threads.

    A review finding caught the first version claiming the second while
    exercising only the first: `_get_db` does not reuse `global_db_to_use`,
    it re-instantiates against its path -- so "same instance across
    threads" must be asserted on an instance this test actually holds.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Tools import note_management_tools as nmt

    db = CharactersRAGDB(tmp_path / "worker.db", "task-696-test")
    monkeypatch.setattr("tldw_chatbook.config.chachanotes_db", db, raising=False)
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "worker_user")
    monkeypatch.setattr(nmt, "_notes_db_base_dir", lambda: str(tmp_path))

    result: dict = {}
    error: list[BaseException] = []

    def _worker() -> None:
        assert threading.current_thread() is not threading.main_thread()
        try:
            result.update(
                asyncio.run(
                    nmt.CreateNoteTool().execute(
                        title="worker-thread note",
                        content="written off the main thread",
                    )
                )
            )
            # The single-instance cross-thread seam: THIS test's instance,
            # used on the worker thread. `threading.local` must mint this
            # thread its own connection rather than raise or return the
            # main thread's.
            note_id = result.get("note_id")
            if note_id:
                result["worker_read"] = db.get_note_by_id(str(note_id))
        except BaseException as exc:  # noqa: BLE001 -- surfaced below
            error.append(exc)

    thread = threading.Thread(target=_worker, name="agent-worker")
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "worker thread hung"
    assert not error, f"execute raised on the worker thread: {error!r}"
    assert "error" not in result, result
    note_id = result.get("note_id")
    assert note_id, result

    worker_row = result.get("worker_read")
    assert worker_row is not None, (
        "the shared instance could not read on the worker thread -- the "
        "threading.local seam the spec flagged"
    )
    assert worker_row["title"] == "worker-thread note"

    # Main thread, same instance again: durable and readable both sides.
    row = db.get_note_by_id(str(note_id))
    assert row is not None, "the note vanished across threads"
    assert row["title"] == "worker-thread note"
    assert row["content"] == "written off the main thread"
