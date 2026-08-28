"""TASK-1366: WriteFileTool diffs reach the Console TOOL marker rows.

End-to-end through ``ConsoleAgentBridge.run_reply``: the raw before/after
contents captured at the provider's strip seam land on the TOOL marker
message as the session-only ``tool_diff`` field, while the marker text,
its full output, and the persisted run record carry only the stripped
result (AC1 data path + AC3).
"""

import json
from collections import deque

import pytest

import tldw_chatbook.config as config_module
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge, _pair_step_diff
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import WriteFileTool


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class _AllowAllBuiltinGate:
    """Minimal ``BuiltinToolGate`` double that permits every tool."""

    def check(self, tool, run_id):
        return None


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _enable_write_file(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if section == "tools" and key == "write_file_enabled"
            else default
        ),
    )


@pytest.fixture
def diff_execute(monkeypatch):
    """WriteFileTool.execute stub returning a diff-carrying result."""

    async def _fake_execute(self, **kwargs):
        return {
            "action": "overwritten",
            "file_path": "/tmp/note.txt",
            "lines_written": 1,
            "old_content": "raw before text\n",
            "new_content": "raw after text\n",
        }

    monkeypatch.setattr(WriteFileTool, "execute", _fake_execute)


def _make_bridge(tmp_path, scripts):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, store, session, assistant_id):
    _run_id, outcome = bridge.run_reply(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="",
            model="test-model",
            ready=True,
            execution_key="llama_cpp",
        ),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
        builtin_gate=_AllowAllBuiltinGate(),
    )
    return outcome


def _tool_rows(store, session):
    return [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
        and m.activity_presentation is not None
        and m.activity_presentation.kind == "tool"
    ]


def test_write_file_marker_carries_session_only_diff(
    tmp_path, monkeypatch, diff_execute
):
    """The TOOL marker gets (path, old, new); every string form stays stripped."""
    _enable_write_file(monkeypatch)
    scripts = [
        [_fence("write_file", {"file_path": "note.txt", "content": "x"})],
        ["Saved."],
    ]
    bridge, db, store, session, aid = _make_bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    tool_rows = _tool_rows(store, session)
    assert tool_rows, "a write_file turn must drop a TOOL marker"
    marker = tool_rows[0]
    # AC1 data path: the raw diff reached the UI-side message, in memory only.
    assert marker.tool_diff == (
        "/tmp/note.txt",
        "raw before text\n",
        "raw after text\n",
    )
    # AC3: neither the preview nor the expanded full output carries raw
    # contents (both are built from the post-strip result text).
    assert "raw before text" not in marker.content
    assert "raw after text" not in marker.content
    assert "old_content" not in marker.content
    full = marker.tool_output_full or ""
    assert "raw before text" not in full
    assert "raw after text" not in full
    assert "old_content" not in full
    # AC3: the persisted run record (run-log-bound steps) is stripped too.
    runs = db.list_runs("conv-1", include_superseded=True)
    persisted = json.dumps(runs, default=str)
    assert "raw before text" not in persisted
    assert "raw after text" not in persisted


def test_non_diff_tool_marker_has_no_diff(tmp_path, monkeypatch, diff_execute):
    """AC4: ordinary tool results render exactly as before (no tool_diff)."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is 42."],
    ]
    bridge, _db, store, session, aid = _make_bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    tool_rows = _tool_rows(store, session)
    assert tool_rows, "a tool turn must drop a TOOL marker"
    assert tool_rows[0].tool_diff is None
    assert "calculator" in tool_rows[0].content


def test_plain_write_result_without_capture_has_no_diff(
    tmp_path, monkeypatch
):
    """A write result that did not capture contents yields no diff row data."""
    _enable_write_file(monkeypatch)

    async def _plain_execute(self, **kwargs):
        return {"action": "created", "file_path": "/tmp/plain.txt", "lines_written": 1}

    monkeypatch.setattr(WriteFileTool, "execute", _plain_execute)
    scripts = [
        [_fence("write_file", {"file_path": "plain.txt", "content": "x"})],
        ["Saved."],
    ]
    bridge, _db, store, session, aid = _make_bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    tool_rows = _tool_rows(store, session)
    assert tool_rows
    assert tool_rows[0].tool_diff is None


class TestPairStepDiff:
    """_pair_step_diff: capture/step pairing under the real threading model.

    invoke() runs on a per-call daemon thread (AgentService
    ._call_with_timeout) -- joined before the result step normally,
    abandoned unjoined on timeout/cancel, so a late capture can land
    after its own step passed. The pairing rule (most-recent name match;
    everything older is stale) must keep such a capture from pairing with
    a later write.
    """

    @staticmethod
    def _capture(name, path):
        return (name, path, f"old {path}\n", f"new {path}\n")

    def test_normal_case_pairs_the_only_capture(self):
        queue = deque([self._capture("write_file", "/tmp/a.py")])
        assert _pair_step_diff(queue, "write_file") == (
            "/tmp/a.py",
            "old /tmp/a.py\n",
            "new /tmp/a.py\n",
        )
        assert not queue

    def test_stale_capture_from_abandoned_call_does_not_pair_with_next_write(self):
        """A timed-out write's capture lands late (after its result step
        passed); the NEXT write's result must pair with its OWN capture,
        and the stale one is dropped."""
        queue = deque(
            [
                self._capture("write_file", "/tmp/stale.py"),
                self._capture("write_file", "/tmp/fresh.py"),
            ]
        )
        assert _pair_step_diff(queue, "write_file") == (
            "/tmp/fresh.py",
            "old /tmp/fresh.py\n",
            "new /tmp/fresh.py\n",
        )
        assert not queue, "the stale capture must be dropped with the pair"

    def test_older_capture_for_other_tool_is_stale_and_dropped(self):
        """An unmatched older entry had its own result step pass already;
        it must not survive to pair with a later same-name step."""
        queue = deque(
            [
                self._capture("read_file", "/tmp/unrelated.py"),
                self._capture("write_file", "/tmp/b.py"),
            ]
        )
        assert _pair_step_diff(queue, "write_file") is not None
        assert not queue

    def test_no_match_clears_stale_queue(self):
        """A captureless result step (plain/oversized write, non-diff
        tool) leaves nothing behind for a later step to mis-pair."""
        queue = deque([self._capture("write_file", "/tmp/stale.py")])
        assert _pair_step_diff(queue, "calculator") is None
        assert not queue

    def test_no_match_on_empty_queue_is_a_noop(self):
        queue = deque()
        assert _pair_step_diff(queue, "write_file") is None
        assert not queue

    def test_same_name_race_pairs_most_recent_and_leaves_no_residue(self):
        """A cross-thread append that lands AFTER the current call's
        capture (newer, same tool name) is indistinguishable from the
        current call's own capture -- the documented residual mis-pair.
        The important half: nothing older survives to cascade into
        FUTURE steps."""
        queue = deque(
            [
                self._capture("write_file", "/tmp/current.py"),
                self._capture("write_file", "/tmp/late.py"),
            ]
        )
        paired = _pair_step_diff(queue, "write_file")
        assert paired[0] == "/tmp/late.py"
        assert not queue

    def test_sequential_writes_each_pair_with_their_own_capture(self):
        queue = deque([self._capture("write_file", "/tmp/one.py")])
        assert _pair_step_diff(queue, "write_file")[0] == "/tmp/one.py"
        queue.append(self._capture("write_file", "/tmp/two.py"))
        assert _pair_step_diff(queue, "write_file")[0] == "/tmp/two.py"
        assert not queue
