"""Task 5 (console-turn-file-annotate, spec §4): diff-feedback delivery.

Covers the auto-attach + exact-id stamp + live disclosure seam in
``ConsoleAgentBridge.run_reply``: pending ``change_notes`` are appended to
the next reply's OUTBOUND (provider-facing) payload only, and stamped
delivered at run completion by exactly the ids that made it into that
payload -- never a blanket "all pending for the conversation" stamp.

The bridge-level harness (``_ChunkGateway`` / ``_bridge_with_gateway`` /
``_run_kwargs``, and the ``patch.object(AgentService, "run_turn", ...)``
spy for inspecting the exact outbound ``messages`` payload) mirrors the
established shape already used by ``Tests/Chat/test_console_agent_bridge.py``
(see ``test_run_reply_appends_bundle_block_copy_safely`` there for the
precedent this file's copy-safety assertions are modeled on) -- reused,
not invented.

``AgentRunsDB`` is FILE-BACKED throughout (``tmp_path / "runs.db"``),
never ``:memory:`` -- the V1 thread-affinity lesson (Task 1's own test
suite, carried forward here).
"""
from __future__ import annotations

from unittest.mock import patch

from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_display_state import (
    format_diff_feedback_disclosure,
    render_diff_feedback_block,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


# -- harness (mirrors test_console_agent_bridge.py's _ChunkGateway /
# _bridge_with_gateway / _run shape) -----------------------------------


class _ChunkGateway:
    """Replays one scripted list of chunks per call, in call order."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        chunks = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class _ExplodingGateway:
    """Streams a partial reply, then raises -- never reaches a final answer.

    Mirrors ``test_console_agent_bridge.py``'s ``_ExplodingGateway`` (a
    genuine provider failure, not a usage-accounting hiccup).
    """

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        yield "Wor"
        raise RuntimeError("provider connection dropped")


def _bridge_with_gateway(tmp_path, gateway, *, db=None):
    db = db if db is not None else AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )
    return bridge, db, store, session, assistant.id


def _resolution(**over):
    fields = dict(provider="TestProvider", base_url="", model=None, ready=True)
    fields.update(over)
    return ConsoleProviderResolution(**fields)


def _run_kwargs(session, assistant_id, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=_resolution(),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    kwargs.update(over)
    return kwargs


def _spy_run_turn(captured):
    """Records the exact ``messages=`` payload every ``run_turn`` call
    received -- the truest "outbound copy", since it is literally what the
    bridge handed to the run loop (not a re-derivation)."""
    real_run_turn = AgentService.run_turn

    def spy(self, **kwargs):
        captured.setdefault("messages_by_call", []).append(kwargs.get("messages"))
        return real_run_turn(self, **kwargs)

    return spy


def _add_note(db, run_id, *, path="a.py", header="@@ -1,1 +1,1 @@", excerpt="+x", note="n"):
    return db.add_change_note(
        run_id=run_id,
        root="/workspace",
        path=path,
        hunk_index=0,
        hunk_header=header,
        hunk_excerpt=excerpt,
        note=note,
    )


# -- (a) happy path: attach + stamp + disclosure -------------------------


def test_pending_notes_attach_stamp_and_disclose_on_success(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(
        db,
        earlier_run,
        path="a.py",
        header="@@ -1,2 +1,3 @@",
        excerpt="+cache = {}\n def f():\n+    return cache",
        note="use the cached value here",
    )

    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured: dict = {}

    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(session, aid, agent_messages=agent_messages)
        )

    assert outcome.status == "done"

    # The OUTBOUND copy carries the block on the last user message.
    sent = captured["messages_by_call"][-1]
    assert sent is not agent_messages
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"].startswith("hi\n\n## Diff feedback from the user")
    assert "use the cached value here" in sent[-1]["content"]
    assert "a.py" in sent[-1]["content"]

    # The caller's OWN list/dict were never mutated (turn_bundle_block's
    # copy-safety contract, extended to this seam).
    assert agent_messages == [original_user_message]
    assert agent_messages[0] is original_user_message
    assert original_user_message["content"] == "hi"

    # The STORED (persisted transcript) user message is unchanged -- the
    # block only ever lived in the ephemeral LLM payload.
    stored_user = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.USER
    ][0]
    assert stored_user.content == "hi"

    # Stamped delivered by exact id.
    assert db.pending_notes_for_conversation("conv-1") == []
    delivered = db.notes_for_run(earlier_run)
    assert len(delivered) == 1
    assert delivered[0]["delivered_at"] is not None

    # Disclosure row: TOOL role, note content, no change_review_run_id.
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert tool_rows[0].content == format_diff_feedback_disclosure(delivered)
    assert "use the cached value here" in tool_rows[0].content
    assert tool_rows[0].change_review_run_id is None


# -- (b) failed run: still pending, no disclosure -------------------------


def test_failed_run_leaves_notes_pending_and_appends_no_disclosure(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ExplodingGateway()
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    note_id = _add_note(db, earlier_run, note="fix this")

    captured: dict = {}
    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "error"
    assert outcome.final_text == ""

    # The block WAS in the outbound payload (attach isn't gated on the
    # run's eventual outcome)...
    sent = captured["messages_by_call"][-1]
    assert "## Diff feedback from the user" in sent[-1]["content"]

    # ...but nothing was stamped and no disclosure was emitted: the block
    # only ever lived in a copy nobody persisted, so nothing was lost by
    # leaving the note pending for the retry.
    pending = db.pending_notes_for_conversation("conv-1")
    assert [n["id"] for n in pending] == [note_id]

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows == []


# -- (c) mid-run race: a note created while the run is in flight ----------


def test_note_created_mid_run_is_not_stamped_at_that_runs_completion(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    early_note_id = _add_note(
        db, earlier_run, path="a.py", note="captured before the run started"
    )
    late_note_ids: list[int] = []

    class _RacingGateway:
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            # This fires AFTER run_reply's attach seam already captured
            # its id list (it runs before `run_turn`, which is what
            # eventually drives this call) -- so a note created here is
            # exactly the "mid-run" race the stamp must not sweep up.
            late_note_ids.append(
                _add_note(
                    db,
                    earlier_run,
                    path="b.py",
                    note="added while the run was in flight",
                )
            )
            yield "Done."

    bridge, _db, store, session, aid = _bridge_with_gateway(
        tmp_path, _RacingGateway(), db=db
    )

    run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"
    assert len(late_note_ids) == 1

    pending = db.pending_notes_for_conversation("conv-1")
    assert [n["id"] for n in pending] == late_note_ids

    delivered_ids = {
        n["id"] for n in db.notes_for_run(earlier_run) if n["delivered_at"]
    }
    assert delivered_ids == {early_note_id}


# -- (d) over-cap: only included ids stamped; holdover rides the next send -


def test_over_cap_only_included_stamped_and_second_run_delivers_the_rest(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    small_id = _add_note(db, earlier_run, path="a.py", excerpt="+small", note="small note")
    # Sized so its OWN entry fits comfortably under the 16 KiB default cap
    # alone (so it must deliver on a second, uncontested run), but combined
    # with the first note's entry it pushes the block over the cap (so it
    # must be excluded -- and NOT stamped -- on this run).
    big_excerpt = "x" * 16_200
    big_id = _add_note(db, earlier_run, path="b.py", excerpt=big_excerpt, note="big note")

    # Self-check the fixture actually exercises the cap branch (this is
    # the SAME function production calls -- not a re-implementation).
    pending = db.pending_notes_for_conversation("conv-1")
    assert [n["id"] for n in pending] == [small_id, big_id]
    expected_block, expected_included = render_diff_feedback_block(pending)
    assert expected_included == [small_id]
    assert "more notes held for the next message" in expected_block

    gateway1 = _ChunkGateway([["ack one."]])
    bridge1, _db1, store, session, aid1 = _bridge_with_gateway(
        tmp_path, gateway1, db=db
    )
    run_id1, outcome1 = bridge1.run_reply(**_run_kwargs(session, aid1))
    assert outcome1.status == "done"

    remaining = db.pending_notes_for_conversation("conv-1")
    assert [n["id"] for n in remaining] == [big_id]

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert "small note" in tool_rows[0].content
    assert "big note" not in tool_rows[0].content

    # Second run: the held-over note is now the only pending one, and it
    # fits alone -- it must deliver.
    aid2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
    gateway2 = _ChunkGateway([["ack two."]])
    bridge2 = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway2)
    run_id2, outcome2 = bridge2.run_reply(**_run_kwargs(session, aid2))

    assert outcome2.status == "done"
    assert db.pending_notes_for_conversation("conv-1") == []
    tool_rows_after = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows_after) == 2
    assert "big note" in tool_rows_after[1].content


# -- (e) no pending notes: byte-identical payload --------------------------


def test_no_pending_notes_leaves_payload_byte_identical(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured: dict = {}

    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(session, aid, agent_messages=agent_messages)
        )

    assert outcome.status == "done"
    sent = captured["messages_by_call"][-1]
    # Byte-identical to pre-feature behavior: the very same list object,
    # not merely equal content -- guards against an unconditional mutation
    # seam that would just happen to be a no-op today.
    assert sent is agent_messages
    assert sent[-1]["content"] == "hi"

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows == []


# -- protective posture: notes subsystem failures never break the reply ---


def test_attach_query_failure_never_breaks_the_reply(tmp_path, monkeypatch):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Fine."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    def _boom(self, conversation_id):
        raise RuntimeError("notes DB is on fire")

    monkeypatch.setattr(AgentRunsDB, "pending_notes_for_conversation", _boom)

    run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"
    assert outcome.final_text.strip() == "Fine."


def test_stamp_failure_never_breaks_the_reply_and_leaves_note_pending(
    tmp_path, monkeypatch
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Fine."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(db, earlier_run, note="n")

    def _boom(self, note_ids):
        raise RuntimeError("stamp DB is on fire")

    monkeypatch.setattr(AgentRunsDB, "mark_notes_delivered", _boom)

    run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"
    assert outcome.final_text.strip() == "Fine."
    # The stamp never landed, so the note correctly stays pending rather
    # than being silently lost.
    assert len(db.pending_notes_for_conversation("conv-1")) == 1


# -- fallback hunk (hunk_header == "") renders sanely (Task 4 carried minor)


def test_fallback_hunk_with_empty_header_renders_sanely_in_block_and_disclosure():
    """A note anchored to a fallback (no-``@@``) hunk stores
    ``hunk_header == ""`` (binary files / clean renames -- Task 2's
    ``split_unified_diff`` fallback). Task 2's formatters already accept
    it; this pins that the OUTPUT stays coherent -- no dangling ``@@``
    artifact, no broken line shape -- not merely "doesn't crash"."""
    note = {
        "id": 1,
        "run_id": "run-xyz12345",
        "root": "/workspace",
        "path": "assets/logo.png",
        "hunk_index": 0,
        "hunk_header": "",
        "hunk_excerpt": "Binary files differ",
        "note": "please regenerate this at 2x",
        "created_at": "2026-08-17T00:00:00",
        "delivered_at": None,
    }

    block, included_ids = render_diff_feedback_block([note])
    assert included_ids == [1]
    assert "@@" not in block
    assert "assets/logo.png" in block
    assert "please regenerate this at 2x" in block
    assert "[run run-xyz1]" in block  # short_id = run_id[:8]

    disclosure = format_diff_feedback_disclosure([note])
    assert "\n" not in disclosure
    assert "@@" not in disclosure
    assert disclosure.startswith("📝 Diff feedback attached — assets/logo.png ")
    assert '"please regenerate this at 2x"' in disclosure
