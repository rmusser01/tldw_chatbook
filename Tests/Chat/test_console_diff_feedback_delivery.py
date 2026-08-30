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

from tldw_chatbook.Agents import agent_service
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
        captured.setdefault("plans_by_call", []).append(
            kwargs.get("first_request_schema_plan")
        )
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


def _add_file_note(db, run_id, *, path="c.py", note="n"):
    """A `file`-kind note (TASK-18060 Task 8, spec §4/§5): `hunk_index=-1`,
    `hunk_header=''`, `hunk_excerpt=''` sentinels."""
    return db.add_change_note(
        run_id=run_id,
        root="/workspace",
        path=path,
        hunk_index=-1,
        hunk_header="",
        hunk_excerpt="",
        note=note,
        anchor_kind="file",
    )


def _add_diff_line_note(
    db,
    run_id,
    *,
    path="b.py",
    header="@@ -5,3 +5,4 @@",
    excerpt="+line5\n+line6",
    note="n",
    diff_line_index=6,
    diff_line_text="+line6",
):
    """A `diff_line`-kind note (TASK-18060 Task 8, spec §4/§5): the hunk
    fields are ALSO populated (the hunk the line falls in), plus the
    line-specific fields."""
    return db.add_change_note(
        run_id=run_id,
        root="/workspace",
        path=path,
        hunk_index=1,
        hunk_header=header,
        hunk_excerpt=excerpt,
        note=note,
        anchor_kind="diff_line",
        diff_line_index=diff_line_index,
        diff_line_text=diff_line_text,
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


# -- (a2) lost race: a captured note stamped by a rival BEFORE this run's
# own completion gets to it -- disclosure must not claim it (Qodo #4) -----


def test_lost_race_pre_stamped_note_is_excluded_from_disclosure(tmp_path):
    """Qodo #4 (PR #1779 fix round): the attach seam captures BOTH ids
    before `run_turn` is even called (see the seam's own comment); this
    simulates a concurrent delivery elsewhere stamping ONE of them first,
    between capture and this run's own completion -- `mark_notes_
    delivered`'s own `delivered_at IS NULL` guard correctly skips
    re-stamping it, and the completion seam must disclose only the note
    it ACTUALLY (verifiably) delivered, not the whole captured set.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    note_a = _add_note(db, earlier_run, path="a.py", note="note a survives")
    note_b = _add_note(db, earlier_run, path="b.py", note="note b stolen")

    real_run_turn = AgentService.run_turn

    def _steal_note_b(self, **kwargs):
        # Simulate a concurrent delivery elsewhere stamping note_b BEFORE
        # this run's own completion runs its own mark_notes_delivered --
        # after the attach seam above already captured both ids.
        db.mark_notes_delivered([note_b], delivered_by_run_id="rival-run")
        return real_run_turn(self, **kwargs)

    with patch.object(AgentService, "run_turn", _steal_note_b):
        run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"

    delivered = {n["id"]: n for n in db.notes_for_run(earlier_run)}
    assert delivered[note_a]["delivered_at"] is not None
    assert delivered[note_a]["delivered_by_run_id"] == run_id
    # note_b keeps the rival's stamp -- this run's own stamp call must not
    # have overwritten it.
    assert delivered[note_b]["delivered_by_run_id"] == "rival-run"

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert "note a survives" in tool_rows[0].content
    assert "note b stolen" not in tool_rows[0].content


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


# -- (f) attach found no carrier: never stamp/disclose what the payload
# could not carry (CRITICAL fix-round finding) ----------------------------


def test_no_user_message_leaves_notes_pending_and_appends_no_disclosure(tmp_path):
    """Reviewer-found critical: the attach loop's backward scan can find NO
    ``role=="user"`` string-content message to carry the block -- e.g. a
    payload with no user message at all. ``render_diff_feedback_block`` ran
    and returned included ids/notes, but the block never actually reached
    the outbound payload -- completion must not stamp/disclose feedback the
    model never saw.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Fine."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    note_id = _add_note(db, earlier_run, note="never reaches the model")

    captured: dict = {}
    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(
                session,
                aid,
                agent_messages=[{"role": "system", "content": "no user turn here"}],
            )
        )

    assert outcome.status == "done"
    assert outcome.final_text.strip() == "Fine."

    # Confirm the block was never silently injected anywhere in the
    # outbound payload either.
    sent = captured["messages_by_call"][-1]
    assert all("Diff feedback" not in str(m.get("content")) for m in sent)

    pending = db.pending_notes_for_conversation("conv-1")
    assert [n["id"] for n in pending] == [note_id]

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows == []


def test_list_content_user_message_leaves_notes_pending_and_appends_no_disclosure(
    tmp_path,
):
    """Same failure family: the ONLY/last user message has LIST content (a
    vision/attachment turn shape, e.g. ``[{"type": "text", "text": "hi"}]``),
    which ``isinstance(content, str)`` correctly excludes as a carrier -- so
    again nothing was actually attached, and nothing may be stamped.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Fine."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    note_id = _add_note(db, earlier_run, note="never reaches the model")

    vision_message = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    captured: dict = {}
    # TASK-17610: this test used to force a usage value to dodge the
    # no-usage token-estimate path, which crashed on LIST content. The
    # estimator now normalizes part-list content, so the real fallback
    # path runs here unpatched.
    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(session, aid, agent_messages=[vision_message])
        )

    assert outcome.status == "done"
    assert outcome.final_text.strip() == "Fine."

    sent = captured["messages_by_call"][-1]
    assert sent[-1]["content"] == [{"type": "text", "text": "hi"}]

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
    # Byte-identical provider content with the caller's payload left untouched.
    # The bridge may copy the outer list for other first-request transforms.
    assert sent == agent_messages
    assert agent_messages == [original_user_message]
    assert agent_messages[0] is original_user_message
    assert sent[-1]["content"] == "hi"

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows == []


def test_pending_notes_participate_in_first_request_fit(tmp_path, monkeypatch):
    """A review-note rider can move the exact first request to discovery."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, _store, session, aid = _bridge_with_gateway(
        tmp_path, gateway, db=db
    )
    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(db, earlier_run, note="review feedback pushes this request over")

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 10_000)
    monkeypatch.setattr(
        agent_service, "catalog_schema_tokens", lambda *_a, **_k: 1
    )

    def count_messages(messages, *_args, **_kwargs):
        rendered = "\n".join(str(row.get("content", "")) for row in messages)
        if "## Diff feedback from the user" not in rendered:
            return 8_000
        return 8_500 if "find_tools" in rendered else 9_000

    monkeypatch.setattr(agent_service, "_count_model_messages", count_messages)
    captured: dict = {}

    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        _run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"
    assert "## Diff feedback from the user" in captured["messages_by_call"][-1][-1][
        "content"
    ]
    assert captured["plans_by_call"][-1].offer_find_load is True


# -- kill switch: presentation OFF must not affect bridge-level delivery --


def test_kill_switch_off_does_not_prevent_note_delivery(tmp_path, monkeypatch):
    """AC#5 / the spec's "Kill switch" section: `[console] turn_file_cards
    = false` keeps the CARD off -- pinned separately (byte-identical plain
    marker, no card mounts) by
    `Tests/UI/test_console_turn_file_card_factory.py`'s
    `test_summary_row_stays_plain_marker_when_disabled`, using the same
    monkeypatch shape as here. This is the OTHER half of AC#5: the
    delivery seam lives in ``ConsoleAgentBridge.run_reply`` and never
    consults that presentation switch at all, so a pending note must
    still auto-attach, stamp, and disclose exactly like every other run in
    this file even while the switch is OFF -- "no note UI" must never mean
    "notes silently vanish".

    Patches ``tldw_chatbook.config.get_cli_setting`` itself (not a
    module-local re-export) because every ``get_cli_setting`` call this
    exercises -- including the bridge's own -- is a LOCAL `from
    tldw_chatbook.config import get_cli_setting` inside a function body;
    patching the defining module is the only patch point every one of
    those local imports actually resolves through.
    """
    import tldw_chatbook.config as config_module

    real_get_cli_setting = config_module.get_cli_setting

    def _turn_file_cards_off(section, key, default=None):
        if (section, key) == ("console", "turn_file_cards"):
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", _turn_file_cards_off)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(db, earlier_run, note="fix this even with the card switched off")

    run_id, outcome = bridge.run_reply(**_run_kwargs(session, aid))

    assert outcome.status == "done"
    assert db.pending_notes_for_conversation("conv-1") == []
    delivered = db.notes_for_run(earlier_run)
    assert len(delivered) == 1
    assert delivered[0]["delivered_at"] is not None

    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert "fix this even with the card switched off" in tool_rows[0].content


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


# -- minor: RUN_CANCELLED with truthy final_text still stamps (deliberate) -


def test_cancelled_run_with_partial_final_text_still_stamps_and_discloses(tmp_path):
    """Deliberate, not a gap: completion gates on ``outcome.final_text``
    being truthy, not on ``outcome.status == "done"``. ``agent_runtime``
    checks ``should_cancel()`` again AFTER a tool-call-free turn has
    already produced its full text (see the ``if not calls:`` branch),
    returning ``RUN_CANCELLED`` with that text attached as
    ``final_text``. The outbound payload -- notes included -- genuinely
    reached the model in that case, so stamping/disclosing here is
    correct: "produced assistant output" is the spec's actual gate, and a
    cancelled-but-answered turn satisfies it.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Cancelled but answered."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(db, earlier_run, note="n")

    class _CancelAfterFirstProbe:
        """False on the pre-model-call check, True on the post-turn one --
        so the model's own turn completes before cancellation is observed."""

        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return self.calls > 1

    run_id, outcome = bridge.run_reply(
        **_run_kwargs(session, aid, should_cancel=_CancelAfterFirstProbe())
    )

    assert outcome.status == "cancelled"
    assert outcome.final_text.strip() == "Cancelled but answered."

    assert db.pending_notes_for_conversation("conv-1") == []
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert "n" in tool_rows[0].content


# -- minor: turn_bundle_block + diff-feedback block stack on one message --


def test_bundle_block_and_diff_feedback_block_stack_on_the_same_message(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    _add_note(db, earlier_run, note="fix the bundled file too")

    bundle_block = "Bundled files (readable via skill_file): notes.md (1 bytes)"
    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured: dict = {}

    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(
                session,
                aid,
                agent_messages=agent_messages,
                turn_bundle_block=bundle_block,
            )
        )

    assert outcome.status == "done"

    # Bundle block first, feedback block appended after -- same message,
    # same order the two seams run in.
    sent = captured["messages_by_call"][-1]
    assert sent[-1]["content"].startswith(
        f"hi\n\n{bundle_block}\n\n## Diff feedback from the user"
    )
    assert "fix the bundled file too" in sent[-1]["content"]

    # Caller's own list/dict untouched.
    assert agent_messages == [original_user_message]
    assert agent_messages[0] is original_user_message
    assert original_user_message["content"] == "hi"

    # Stored transcript message unchanged.
    stored_user = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.USER
    ][0]
    assert stored_user.content == "hi"

    # Stamped + disclosed correctly despite the stacked bundle block.
    assert db.pending_notes_for_conversation("conv-1") == []
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    assert "fix the bundled file too" in tool_rows[0].content


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


# -- mixed-batch end-to-end (TASK-18060 Task 8, spec §5): one hunk + one
# file + one diff_line note, through the real bridge harness -------------


def test_mixed_kind_batch_attaches_one_block_stamps_and_discloses_all_three_kinds(
    tmp_path,
):
    """A pending batch of one `hunk`, one `file`, and one `diff_line` note
    -- `run_reply` must attach exactly ONE block containing all three,
    each correctly rendered per its own kind, stamp all three by their
    exact ids, and disclose with the kind-aware lines. This is the
    mixed-batch proof the delivery MECHANICS (attach/stamp/cap/resume)
    never needed to change -- only the two shared formatters learned
    kinds."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    gateway = _ChunkGateway([["Done."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway, db=db)

    earlier_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    hunk_id = _add_note(
        db,
        earlier_run,
        path="a.py",
        header="@@ -1,2 +1,3 @@",
        excerpt="+cache = {}",
        note="hunk note",
    )
    file_id = _add_file_note(
        db, earlier_run, path="c.py", note="please clean this whole file"
    )
    line_id = _add_diff_line_note(
        db, earlier_run, path="b.py", note="fix this line"
    )

    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured: dict = {}

    with patch.object(AgentService, "run_turn", _spy_run_turn(captured)):
        run_id, outcome = bridge.run_reply(
            **_run_kwargs(session, aid, agent_messages=agent_messages)
        )

    assert outcome.status == "done"

    # -- one block, all three kinds correctly rendered ---------------------
    sent = captured["messages_by_call"][-1]
    block_text = sent[-1]["content"]
    assert block_text.count("## Diff feedback from the user") == 1
    assert "### a.py — @@ -1,2 +1,3 @@   [run " in block_text
    assert "### c.py — whole file   [run " in block_text
    assert "### b.py — @@ -5,3 +5,4 @@   [run " in block_text
    assert "> on line: +line6" in block_text
    assert "hunk note" in block_text
    assert "please clean this whole file" in block_text
    assert "fix this line" in block_text
    # File-note entry carries no dangling `@@`/fence.
    file_entry_start = block_text.index("### c.py — whole file")
    file_entry_end = block_text.index(
        "### b.py", file_entry_start
    )  # next heading -- brittle to order but ids are inserted oldest-first
    file_entry = block_text[file_entry_start:file_entry_end]
    assert "@@" not in file_entry
    assert "````" not in file_entry

    # -- exact-id stamping ---------------------------------------------------
    assert db.pending_notes_for_conversation("conv-1") == []
    delivered = db.notes_for_run(earlier_run)
    assert {n["id"] for n in delivered} == {hunk_id, file_id, line_id}
    for delivered_note in delivered:
        assert delivered_note["delivered_at"] is not None

    # -- kind-aware disclosure -----------------------------------------------
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert len(tool_rows) == 1
    disclosure = tool_rows[0].content
    assert disclosure == format_diff_feedback_disclosure(delivered)
    assert "(whole file):" in disclosure
    assert " line: " in disclosure
    disclosure_lines = disclosure.splitlines()
    assert len(disclosure_lines) == 3

    # -- resume re-derives byte-identical to the live disclosure -------------
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    blocks = fresh_bridge.resume_marker_messages("conv-1")
    resumed_disclosure_msgs = [
        m
        for _anchor, block in blocks
        for m in block
        if "Diff feedback attached" in m.content
    ]
    assert len(resumed_disclosure_msgs) == 1
    assert resumed_disclosure_msgs[0].content == disclosure
