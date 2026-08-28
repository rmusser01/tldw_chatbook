"""Regenerate is unified onto the streaming reply engine (Task 1).

TASK-6 (Phase A): ``regenerate_message`` no longer streams a replacement
*variant* into the anchor message in place -- it forks a persisted SIBLING
node under the anchor's own parent and streams into that NEW node
(``variant_mode=False``). The ``begin_variant_stream``/``finalize_variant_
stream``/``add_variant`` store primitives below are exercised directly
(store-level, not through the controller) and remain valid, but the
controller-driven tests further down were rewritten to assert against the
new sibling node rather than the untouched anchor -- see
``Tests/Chat/test_console_regenerate_branching.py`` for the full branching
contract.
"""

import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariant,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleThinkingCompatibilityError,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
)
from Tests.console_provider_doubles import provider_resolution


def _store_with_answer():
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    # Non-empty content already yields status "complete" via _initial_status;
    # this mirrors the existing regenerate tests that seed via append_message alone.
    return store, session, assistant.id


def _thinking(text: str, *, block_id: str = "thinking-1") -> ThinkingEnvelope:
    return ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id=block_id,
                round_ordinal=0,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status="complete",
                text=text,
            ),
        )
    )


def test_begin_variant_stream_resets_buffer_and_keeps_base():
    store, _session, mid = _store_with_answer()
    streaming = store.begin_variant_stream(mid)
    assert streaming.status == "streaming"
    assert streaming.content == ""  # visible row cleared for the new take
    store.append_stream_chunk(mid, "re")
    store.append_stream_chunk(mid, "generated")
    final = store.finalize_variant_stream(mid)
    assert final.status == "complete"
    assert final.content == "regenerated"  # new variant selected
    assert final.variants is not None
    contents = [v.content for v in final.variants.variants]
    assert contents == ["original", "regenerated"]  # base preserved, no concat
    assert final.variants.selected_index == 1


def test_finalize_variant_stream_preserves_attached_usage():
    """PR1 Task 6: ``set_message_usage`` mutates the SAME in-memory node
    ``finalize_variant_stream`` later reads -- both resolve the message via
    ``_message_or_raise`` off the shared node tree, never a copy -- so an
    attach ordered right before the finalize call (mirroring
    ``_attach_stream_usage``'s placement immediately before the controller's
    own ``finalize_variant_stream``/``mark_message_complete`` call in
    ``_run_direct_provider_reply``) survives the variant-selection mutation
    and lands on the returned snapshot. Store-level only: no live controller
    call site currently sets ``variant_mode=True`` (see module docstring and
    ``test_regenerate_new_sibling_carries_its_own_generation_usage`` below),
    but the primitive itself remains correct should that path be reactivated.
    """
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store, _session, mid = _store_with_answer()
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "second take")
    usage = ProviderUsage(
        uncached_input=42, output=7, provider="anthropic", model="claude"
    )
    store.set_message_usage(mid, usage)

    final = store.finalize_variant_stream(mid)

    assert final.usage == usage
    assert store.get_message(mid).usage == usage


def test_finalize_variant_stream_appends_to_existing_set():
    store, _session, mid = _store_with_answer()
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "second")
    store.finalize_variant_stream(mid)
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "third")
    final = store.finalize_variant_stream(mid)
    assert [v.content for v in final.variants.variants] == [
        "original",
        "second",
        "third",
    ]
    assert final.variants.selected_index == 2


def test_message_repr_hides_displayable_and_opaque_thinking() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        thinking=_thinking("secret-visible-reasoning"),
        opaque_thinking_json='{"version":99,"secret":"opaque-secret"}',
    )

    rendered = repr(message)

    assert "secret-visible-reasoning" not in rendered
    assert "opaque-secret" not in rendered


def test_select_variant_swaps_complete_generation() -> None:
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationRound,
        ProviderContinuationCheckpoint,
    )
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store, _session, mid = _store_with_answer()
    original = store._message_or_raise(mid)
    original.thinking = _thinking("original thinking", block_id="original")
    original.usage = ProviderUsage(uncached_input=2, output=3)
    original.provider_continuation = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="original",
                reasoning_blocks=("private reasoning",),
                calls=(),
            ),
        ),
    )
    original.assistant_generation_state = "complete"
    original_snapshot = store.get_message(mid)

    store.begin_variant_stream(mid)
    assert store.get_message(mid).thinking is None
    store.replace_message_thinking(mid, _thinking("new thinking", block_id="new"))
    store.append_stream_chunk(mid, "new answer")
    new_usage = ProviderUsage(uncached_input=5, output=8)
    store.set_message_usage(mid, new_usage)
    finalized = store.finalize_variant_stream(mid)

    assert finalized.thinking == _thinking("new thinking", block_id="new")
    assert finalized.variants.current.assistant_generation_state == "complete"
    restored = store.select_variant(mid, 0)
    assert restored.content == "original"
    assert restored.thinking == _thinking("original thinking", block_id="original")
    assert restored.usage == original_snapshot.usage
    assert restored.provider_continuation == original_snapshot.provider_continuation
    assert restored.assistant_generation_state == "complete"
    selected_again = store.select_variant(mid, 1)
    assert selected_again.thinking == _thinking("new thinking", block_id="new")
    assert selected_again.assistant_generation_state == "complete"


def test_add_variant_keeps_original_generation_owned_by_original_variant() -> None:
    store, _session, mid = _store_with_answer()
    original = store._message_or_raise(mid)
    original.thinking = _thinking("original thinking")
    original.assistant_generation_state = "complete"

    added = store.add_variant(mid, "manual alternative")

    assert added.thinking is None
    assert added.assistant_generation_state == "complete"
    restored = store.select_variant(mid, 0)
    assert restored.thinking == _thinking("original thinking")
    assert restored.assistant_generation_state == "complete"


@pytest.mark.parametrize("blocked_owner", [False, True])
def test_select_variant_rejects_unpersistable_generation_before_live_mutation(
    blocked_owner: bool,
) -> None:
    store, _session, mid = _store_with_answer()
    message = store._message_or_raise(mid)
    message.thinking_actions_enabled = not blocked_owner
    message.variants = ConsoleVariantSet.from_generations(
        turn_id=message.turn_id or message.id,
        generations=[
            ConsoleVariant(content="original"),
            ConsoleVariant(
                content="future",
                opaque_thinking_json='{"version":99,"secret":"future"}',
                thinking_actions_enabled=False,
            ),
        ],
    )

    with pytest.raises(ConsoleThinkingCompatibilityError):
        store.select_variant(mid, 1)

    unchanged = store.get_message(mid)
    assert unchanged.content == "original"
    assert unchanged.variants is not None
    assert unchanged.variants.selected_index == 0
    assert mid not in store._failed_retry_message_ids


@pytest.mark.parametrize("terminal", ["mark_message_stopped", "mark_message_failed"])
def test_abandoned_variant_restores_complete_generation(terminal: str) -> None:
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store, _session, mid = _store_with_answer()
    original = store._message_or_raise(mid)
    original.thinking = _thinking("original thinking")
    original.usage = ProviderUsage(uncached_input=1, output=2)
    original.assistant_generation_state = "complete"

    store.begin_variant_stream(mid)
    store.replace_message_thinking(mid, _thinking("abandoned thinking", block_id="new"))
    store.append_stream_chunk(mid, "abandoned answer")
    getattr(store, terminal)(mid)
    restored = store.get_message(mid)

    assert restored.content == "original"
    assert restored.thinking == _thinking("original thinking")
    assert restored.usage == ProviderUsage(uncached_input=1, output=2)
    assert restored.assistant_generation_state == "complete"


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [("mark_message_stopped", "stopped"), ("mark_message_failed", "failed")],
)
def test_normal_terminal_updates_thinking_envelope_status(
    terminal: str, expected_status: str
) -> None:
    store, _session, mid = _store_with_answer()
    live = store._message_or_raise(mid)
    live.thinking = _thinking("partial thinking")
    live.status = "streaming"

    settled = getattr(store, terminal)(mid)

    assert settled.thinking is not None
    assert {block.status for block in settled.thinking.blocks} == {expected_status}


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [
        ("mark_message_complete", "complete"),
        ("mark_message_stopped", "stopped"),
        ("mark_message_failed", "failed"),
    ],
)
def test_terminal_settlement_only_updates_current_thinking_round(
    terminal: str, expected_status: str
) -> None:
    store, _session, mid = _store_with_answer()
    live = store._message_or_raise(mid)
    live.thinking = ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id="earlier",
                round_ordinal=0,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status="failed",
                text="earlier terminal reasoning",
            ),
            DisplayableThinkingBlock(
                block_id="current",
                round_ordinal=1,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status="complete",
                text="current reasoning",
            ),
        )
    )
    live.status = "streaming"

    settled = getattr(store, terminal)(mid)

    assert settled.thinking is not None
    assert [block.status for block in settled.thinking.blocks] == [
        "failed",
        expected_status,
    ]


class _ScriptedGateway:
    """Async stream_chat that yields scripted chunks; resolve_for_send ready."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def resolve_for_send(self, selection):
        return provider_resolution()

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_regenerate_delegates_and_streams_incrementally():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway(["Paris", " is", " the", " answer."]),
        provider="llama_cpp",
        model="test-model",
    )
    result = await controller.regenerate_message(mid)
    assert result.accepted is True

    # The anchor is untouched and off the active path; a NEW sibling node
    # streamed incrementally and carries the fresh answer.
    unchanged = store.get_message(mid)
    assert unchanged.content == "original"
    assert unchanged.variants is None
    assert mid not in store.active_path_message_ids(session.id)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid
    message = store.get_message(new_leaf_id)
    assert message.content == "Paris is the answer."
    assert message.variants is None


@pytest.mark.asyncio
async def test_regenerate_empty_stream_retains_failed_sibling_and_restores_anchor():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway([]),  # zero chunks: empty stream
        provider="llama_cpp",
        model="test-model",
    )

    result = await controller.regenerate_message(mid)

    assert result.accepted is True
    unchanged = store.get_message(mid)
    assert unchanged.status == "complete"
    assert unchanged.content == "original"
    assert store.active_leaf(session.id) == mid
    assert mid in store.active_path_message_ids(session.id)

    siblings, _index, count = store.siblings_at(mid)
    assert count == 2
    new_sibling = next(sibling for sibling in siblings if sibling.id != mid)
    assert new_sibling.status == "failed"
    assert new_sibling.content == ""

    provider_messages = controller._provider_messages_for_session(session.id)
    assert {"role": "assistant", "content": "original"} in provider_messages
    assert {"role": "assistant", "content": ""} not in provider_messages


@pytest.mark.asyncio
async def test_regenerate_stop_mid_stream_leaves_anchor_untouched_new_sibling_stopped():
    """Plan-B final-review Medium-2, superseded by TASK-6's branching model:
    stopping a regenerate mid-stream must not touch the anchor's own
    pre-regenerate answer at all -- it is a completely separate node. The
    NEW sibling that was streaming into is the one left "stopped" with
    whatever partial buffer it had accumulated; the anchor stays "complete"
    with its original content throughout, on the active path only for as
    long as the stop leaves the new (stopped) sibling's `set_active_leaf`
    untouched -- i.e. the new sibling remains the active leaf, and the
    anchor is reachable by swiping back.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    class WaitingGateway:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve_for_send(self, selection):
            return provider_resolution()

        async def stream_chat(self, resolution, messages, **kwargs):
            self.started.set()
            yield "partial regen "
            await self.release.wait()
            yield "ignored"

    gateway = WaitingGateway()
    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model"
    )

    task = asyncio.create_task(controller.regenerate_message(mid))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid

    assert controller.stop_active_run() is True
    # The anchor was never touched by the regenerate attempt at all.
    anchor = store.get_message(mid)
    assert anchor.content == "original"
    assert anchor.status == "complete"
    assert anchor.variants is None
    assert mid not in store._variant_stream_bases
    assert new_leaf_id not in store._variant_stream_bases

    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.accepted is True

    # The anchor is still untouched; the NEW sibling carries the partial
    # buffer and is marked "stopped" as the active leaf.
    anchor = store.get_message(mid)
    assert anchor.content == "original"
    assert anchor.status == "complete"
    stopped_sibling = store.get_message(new_leaf_id)
    assert stopped_sibling.content == "partial regen "
    assert stopped_sibling.status == "stopped"
    active_path = store.active_path_message_ids(session.id)
    assert new_leaf_id in active_path
    assert mid not in active_path
    # (stop_active_run's own "Response stopped by user." system row becomes
    # the new active leaf, parented under the stopped sibling above --
    # pre-existing behavior, unrelated to Task 6, not asserted here.)


class _UsageEmittingScriptedGateway(_ScriptedGateway):
    """``_ScriptedGateway`` plus a signals-recorded usage payload and a
    resolution carrying ``provider``/``model`` (the base stub's bare ``_R``
    has neither, which ``ProviderUsage.from_provider_payload`` needs for
    attribution)."""

    async def resolve_for_send(self, selection):
        return provider_resolution()

    async def stream_chat(self, resolution, messages, **kwargs):
        signals = kwargs.get("signals")
        for chunk in self._chunks:
            yield chunk
        if signals is not None:
            signals.record_usage_payload(
                {"prompt_tokens": 100, "completion_tokens": 20}
            )


@pytest.mark.asyncio
async def test_regenerate_new_sibling_carries_its_own_generation_usage():
    """PR1 Task 6, Step 5 finding: TASK-6 (Phase A, see module docstring)
    already forks ``regenerate_message`` onto a brand-new SIBLING node
    streamed with ``variant_mode=False`` -- so the LIVE regenerate path
    never reaches ``finalize_variant_stream`` at all. It goes through the
    exact same ``mark_message_complete`` attach-then-flush ordering as an
    ordinary send (``_run_direct_provider_reply``'s success block calls
    ``_attach_stream_usage(..., partial=False)`` immediately before
    ``mark_message_complete``/``finalize_variant_stream``, keyed off
    ``variant_mode`` -- which is always False here).

    What this test actually proves: usage attaches to the NEW sibling node
    (this generation's own id), not the anchor -- correct per-generation
    attribution. A regenerate must never clobber the anchor's own usage
    (or, as here, its absence) with the new generation's numbers, since
    the anchor remains a separate, independently reachable node.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_UsageEmittingScriptedGateway(
            ["Paris", " is", " the", " answer."]
        ),
        provider="llama_cpp",
        model="test-model",
    )

    result = await controller.regenerate_message(mid)
    assert result.accepted is True

    anchor = store.get_message(mid)
    assert anchor.usage is None  # prior generation recorded no usage

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid
    new_sibling = store.get_message(new_leaf_id)
    assert new_sibling.usage is not None
    assert new_sibling.usage.uncached_input == 100
    assert new_sibling.usage.output == 20
    assert new_sibling.usage.partial is False
    assert new_sibling.usage.provider == "llama_cpp"
