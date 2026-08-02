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
def test_a_real_reads_tagged_tool_reaches_the_approval_card_not_silence():
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
    from tldw_chatbook.Tools.file_operation_tools import ReadFileTool

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
        [ToolCall(name="read_file", args={"file_path": "notes.md"}, call_id="c1")]
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
    `NotesInteropService` away, proving only that `asyncio.run` works off the
    main thread. The spec's actual question: does `CharactersRAGDB` -- built
    for cross-thread use via `threading.local` connections and
    `check_same_thread=False` -- really work on the agent's worker thread?
    This runs the REAL `CreateNoteTool.execute` on a real non-main thread
    against a real `CharactersRAGDB` at a temp path, then reads the row back
    on the MAIN thread through the same instance -- both directions of the
    cross-thread contract.
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

    # Main thread, same instance: the row must be durable and readable.
    row = db.get_note_by_id(str(note_id))
    assert row is not None, "the note vanished across threads"
    assert row["title"] == "worker-thread note"
    assert row["content"] == "written off the main thread"
