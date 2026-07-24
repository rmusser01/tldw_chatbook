"""Controller-level tests for `/rewind` "Summarize up to here" (SP2, Task 3).

Covers ``ConsoleChatController.summarize_up_to`` (gates, span construction,
rolling re-summarize, provider call, storage) and the dispatch-choke-point
``_apply_context_summary_compaction`` (the leak rule: compact only when the
boundary message is present in the payload). Reuses the fake-gateway harness
shape from ``test_console_regenerate_branching.py``.
"""

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_history_budget import bound_messages_to_window


class SummaryGateway:
    """Fake gateway that returns a fixed summary and captures the sent payload."""

    def __init__(self, summary: str = "SUMMARY TEXT", ready: bool = True) -> None:
        self.summary = summary
        self.ready = ready
        self.captured_messages = None

    async def resolve_for_send(self, selection):
        ready = self.ready
        return type(
            "Resolution",
            (),
            {
                "ready": ready,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "max_tokens": 512,
                "visible_copy": "" if ready else "Provider blocked: no key.",
            },
        )()

    async def stream_chat(self, resolution, messages):
        self.captured_messages = messages
        for chunk in _as_chunks(self.summary):
            if chunk:
                yield chunk


def _as_chunks(text: str):
    # Emit in two pieces to exercise chunk accumulation (mirrors a real stream).
    if not text:
        return []
    mid = max(1, len(text) // 2)
    return [text[:mid], text[mid:]]


def _seed_conversation(store, session_id):
    """Append U1/A1/U2/A2/U3/A3 and return the six messages."""
    u1 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q1")
    a1 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a1"
    )
    u2 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q2")
    a2 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )
    u3 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q3")
    a3 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a3"
    )
    return u1, a1, u2, a2, u3, a3


# --------------------------------------------------------------------------
# summarize_up_to gates + storage
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_up_to_stores_summary_and_boundary():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is True
    assert store.session_context_summary(session.id) == ("SUMMARY TEXT", u2.id)
    assert "Summarized" in result.visible_copy


@pytest.mark.asyncio
async def test_summarize_span_is_pre_boundary_user_assistant_only():
    store = ConsoleChatStore()
    gateway = SummaryGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    _seed_conversation(store, session.id)
    u2 = store.messages_for_session(session.id)[2]

    await controller.summarize_up_to(u2.id)

    # The provider saw the internal prompt as system + the pre-boundary span.
    assert gateway.captured_messages[0]["role"] == "system"
    span_text = gateway.captured_messages[1]["content"]
    assert "User: q1" in span_text
    assert "Assistant: a1" in span_text
    # The boundary turn (q2) and everything after it are NOT summarized.
    assert "q2" not in span_text
    assert "a2" not in span_text


@pytest.mark.asyncio
async def test_summarize_provider_not_ready_blocks_and_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=SummaryGateway(ready=False)
    )
    session = store.ensure_session()
    _u1, _a1, u2, *_rest = _seed_conversation(store, session.id)

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_non_user_target_blocks_and_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    _u1, a1, *_rest = _seed_conversation(store, session.id)

    result = await controller.summarize_up_to(a1.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_off_path_target_blocks_and_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    # Move the active leaf back so u2 falls off the active path.
    store.set_active_leaf(session.id, a1.id)
    assert u2.id not in store.active_path_message_ids(session.id)

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_nothing_before_first_prompt_blocks():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="only prompt"
    )

    result = await controller.summarize_up_to(u1.id)

    assert result.accepted is False
    assert "Nothing to summarize" in result.visible_copy
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_empty_reply_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=SummaryGateway(summary="")
    )
    session = store.ensure_session()
    _u1, _a1, u2, *_rest = _seed_conversation(store, session.id)

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_rolling_includes_prior_summary_and_moves_boundary():
    store = ConsoleChatStore()
    gateway = SummaryGateway(summary="S1")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    # First summarize up to u2 -> boundary=u2, summary S1.
    first = await controller.summarize_up_to(u2.id)
    assert first.accepted is True
    assert store.session_context_summary(session.id) == ("S1", u2.id)

    # Second summarize up to u3 rolls: it prepends S1 and covers u2..a2.
    gateway.summary = "S2"
    second = await controller.summarize_up_to(u3.id)
    assert second.accepted is True

    rolling_span = gateway.captured_messages[1]["content"]
    assert "[Previous summary]" in rolling_span
    assert "S1" in rolling_span
    # The un-summarized region since the old boundary (q2/a2) is included.
    assert "q2" in rolling_span
    assert "a2" in rolling_span
    # Turns already folded into S1 (q1/a1) are NOT re-sent raw.
    assert "User: q1" not in rolling_span

    assert store.session_context_summary(session.id) == ("S2", u3.id)


# --------------------------------------------------------------------------
# choke-point compaction + THE LEAK RULE
# --------------------------------------------------------------------------


def _payload_texts(messages):
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(
                "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            )
    return texts


@pytest.mark.asyncio
async def test_compaction_folds_summary_and_drops_pre_boundary_rows():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    controller.system_prompt = "You are helpful."
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    # Compaction anchors the boundary by native id, so the payload must be
    # built id-annotated (as every real send path does).
    payload = controller._provider_messages_for_session(
        session.id, annotate_ids=True
    )
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    texts = _payload_texts(compacted)
    # Pre-boundary turns gone, boundary + tail kept.
    assert "q1" not in texts and "a1" not in texts
    assert "q2" not in texts and "a2" not in texts
    assert "q3" in texts and "a3" in texts
    # Summary folded into the leading system prefix.
    assert compacted[0]["role"] == "system"
    assert "You are helpful." in compacted[0]["content"]
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]

    # The trimmer preserves the leading system prefix (summary survives).
    bound = bound_messages_to_window(
        compacted, model="test-model", provider="llama_cpp", response_reservation=256
    )
    assert bound.messages[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in bound.messages[0]["content"]


@pytest.mark.asyncio
async def test_compaction_creates_system_message_when_payload_has_none():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(
        session.id, annotate_ids=True
    )
    assert payload[0]["role"] != "system"  # no system prompt set

    compacted = controller._apply_context_summary_compaction(session.id, payload)

    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]
    texts = _payload_texts(compacted)
    assert "q3" in texts and "q1" not in texts


@pytest.mark.asyncio
async def test_leak_rule_pre_boundary_payload_is_byte_identical():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    # Payload for regenerating a PRE-boundary message ends before the boundary.
    pre_boundary_payload = controller._provider_messages_for_session(
        session.id, before_message_id=a1.id, annotate_ids=True
    )

    store.set_session_context_summary(session.id, "S", u3.id)
    compacted = controller._apply_context_summary_compaction(
        session.id, controller._provider_messages_for_session(
            session.id, before_message_id=a1.id, annotate_ids=True
        )
    )

    # The boundary (u3) id is absent from this ancestors-only payload, so
    # compaction is a no-op -- byte-identical to the no-summary payload.
    assert compacted == pre_boundary_payload


@pytest.mark.asyncio
async def test_dangling_boundary_leaves_payload_untouched():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    _seed_conversation(store, session.id)
    # A boundary id that is not a live message (branch switch / deletion).
    store.set_session_context_summary(session.id, "S", "ghost-native-id")

    payload = controller._provider_messages_for_session(
        session.id, annotate_ids=True
    )
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    assert compacted == payload


# --------------------------------------------------------------------------
# duplicate-content leak (reviewer repro) + id-anchoring + key stripping
# --------------------------------------------------------------------------


def _seed_duplicate_content(store, session_id):
    """U1/A1/U2/A2/U3(/A3) where U1 and U3 share the exact text "continue"."""
    u1 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="continue"
    )
    a1 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a1"
    )
    u2 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="different"
    )
    a2 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )
    u3 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="continue"
    )
    a3 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a3"
    )
    return u1, a1, u2, a2, u3, a3


@pytest.mark.asyncio
async def test_leak_rule_duplicate_content_pre_boundary_no_false_fire():
    """Reviewer repro: a byte-identical EARLIER duplicate of the boundary's
    text must NOT false-fire compaction on a pre-boundary payload.

    U1 and the boundary U3 both say "continue". Regenerating pre-boundary A1
    builds an ancestors-only ``[U1]`` payload where the boundary U3 is ABSENT.
    First-occurrence content matching wrongly anchored on U1 and injected the
    summary of LATER turns; id-anchored compaction leaves the payload
    byte-identical to the no-summary payload.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_duplicate_content(store, session.id)

    baseline = controller._provider_messages_for_session(
        session.id, before_message_id=a1.id, annotate_ids=True
    )
    store.set_session_context_summary(session.id, "S", u3.id)
    compacted = controller._apply_context_summary_compaction(
        session.id,
        controller._provider_messages_for_session(
            session.id, before_message_id=a1.id, annotate_ids=True
        ),
    )

    # No summary folded, no rows dropped -- the LATER-turn summary never reaches
    # this EARLIER point's context.
    assert compacted == baseline
    assert not any(
        "[Summary of earlier conversation]" in text
        for text in _payload_texts(compacted)
    )


@pytest.mark.asyncio
async def test_compaction_anchors_on_boundary_id_not_duplicate_text():
    """Same duplicate-text tree, but the FULL active-path payload DOES contain
    the real boundary U3. Compaction must anchor on U3 by native id (dropping
    U1/A1/U2/A2) even though the earlier U1 shares U3's exact text -- content
    matching would wrongly anchor on U1 and drop nothing.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_duplicate_content(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(
        session.id, annotate_ids=True
    )
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    texts = _payload_texts(compacted)
    # Everything strictly before the real boundary U3 is dropped: the earlier
    # duplicate "continue" (U1) and the intervening turns are gone.
    assert "different" not in texts
    assert "a1" not in texts and "a2" not in texts
    assert texts.count("continue") == 1  # only the boundary U3 survives
    assert "a3" in texts
    # Summary folded into a leading system row.
    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]


class _SkillsFake:
    """Minimal fake skills service: resolves `$do-it` to a fixed inline
    render. Mirrors the shape of `test_console_skill_substitution.py`'s
    `_Skills` fake, trimmed to only what this regression needs.
    """

    async def get_context(self, *, mode="local"):
        return {
            "available_skills": [
                {
                    "name": "do-it",
                    "description": "d",
                    "user_invocable": True,
                    "trust_blocked": False,
                }
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        return {
            "skill_name": name,
            "rendered_prompt": f"RENDERED[{args}]",
            "allowed_tools": None,
            "execution_mode": "inline",
            "fork_output": None,
        }


@pytest.mark.asyncio
async def test_compaction_anchors_after_skill_substitution_inline_rewrite():
    """Regression (review finding): `_apply_skill_substitution`'s non-fork
    rewrite paths must preserve the original row's private keys (via a
    ``{**row, ...}`` spread), exactly like chat-dictionary/world-info do --
    otherwise, when the compaction boundary IS the final user row AND its
    content also resolves to a skill, the inline rewrite silently drops
    ``NATIVE_MESSAGE_ID_KEY`` and the choke point's id match misses (fails
    SAFE to full history, but compaction never applies).
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SummaryGateway(),
        provider="llama_cpp",
        model="test-model",
        skills_service=_SkillsFake(),
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a1"
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )
    # The boundary is the final user row, and its content resolves to a
    # skill -- the exact overlap the review finding calls out.
    u3 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="$do-it go"
    )
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(
        session.id, annotate_ids=True
    )
    substituted, refuse, notes, bindings, block = (
        await controller._apply_skill_substitution(payload)
    )
    assert refuse is None
    assert bindings == ("do-it",)
    assert substituted[-1]["content"] == "RENDERED[go]"

    compacted = controller._apply_context_summary_compaction(session.id, substituted)

    texts = _payload_texts(compacted)
    # Compaction anchored on the (id-preserved) boundary row: pre-boundary
    # turns are dropped and the summary is folded in.
    assert "q1" not in texts and "a1" not in texts
    assert "q2" not in texts and "a2" not in texts
    assert "RENDERED[go]" in texts
    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]


@pytest.mark.asyncio
async def test_native_message_id_key_stripped_before_provider():
    """The private id-threading key must never reach the provider: after a
    normal compacted send, no captured gateway payload row carries it.
    """
    store = ConsoleChatStore()
    gateway = SummaryGateway(summary="reply")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )
    session = store.ensure_session()
    _u1, _a1, _u2, _a2, u3, _a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    result = await controller.submit_draft("next question")
    assert result.accepted is True

    assert gateway.captured_messages is not None
    assert all(
        "_native_message_id" not in row for row in gateway.captured_messages
    )
    # Sanity: compaction genuinely ran on this send (summary folded), so the
    # strip assertion above is not vacuous.
    assert any(
        row["role"] == "system"
        and "[Summary of earlier conversation]" in row.get("content", "")
        for row in gateway.captured_messages
    )


# ---------------------------------------------------------------------------
# task-548: the inspector next-send preview mirrors boundary compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_reflects_boundary_compaction():
    """With an active summary, build_context_snapshot compacts like a real send."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")

    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="old-q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="old-a")
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="new-q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="new-a")
    store.set_session_context_summary(session.id, "COMPACT-SUMMARY", u2.id)

    snapshot = await controller.build_context_snapshot(draft="")
    rows = snapshot.next_send_payload["messages"]

    # Pre-boundary turns replaced; boundary tail intact.
    contents = [row.get("content") or "" for row in rows]
    assert not any("old-q" in c or "old-a" in c for c in contents)
    assert any("new-q" in c for c in contents)
    assert any("new-a" in c for c in contents)
    # Summary folded into the leading system row AND the duplicated field.
    assert rows[0]["role"] == "system"
    assert "COMPACT-SUMMARY" in rows[0]["content"]
    assert any(
        "COMPACT-SUMMARY" in (row.get("content") or "")
        for row in snapshot.next_send_payload["system"]
    )
    # AC #2: the private id-threading key never reaches the preview.
    assert not any("_native_message_id" in row for row in rows)
    _ = u1


@pytest.mark.asyncio
async def test_snapshot_without_summary_unchanged_and_key_free():
    """No stored summary: preview shows full history and no private keys."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a1")

    snapshot = await controller.build_context_snapshot(draft="next")
    rows = snapshot.next_send_payload["messages"]

    contents = [row.get("content") or "" for row in rows]
    assert any("q1" in c for c in contents)
    assert any("a1" in c for c in contents)
    assert not any("_native_message_id" in row for row in rows)


@pytest.mark.asyncio
async def test_snapshot_with_dangling_boundary_shows_full_history():
    """A dangling boundary leaves the preview un-compacted (leak rule parity)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a1")
    store.set_session_context_summary(session.id, "GHOST-SUMMARY", "ghost-native-id")

    snapshot = await controller.build_context_snapshot(draft="")
    rows = snapshot.next_send_payload["messages"]

    contents = [row.get("content") or "" for row in rows]
    assert any("q1" in c for c in contents)
    assert any("a1" in c for c in contents)
    assert not any("GHOST-SUMMARY" in c for c in contents)
