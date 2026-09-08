"""Capture mode must preserve selected private provider history at dispatch."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_prepared_request import thaw_json
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    parse_thinking_blocks_json,
)


@pytest.fixture
async def replay_harness(tmp_path, monkeypatch, request):
    capture_on, kind = (
        request.param
        if isinstance(request.param, tuple)
        else (request.param, "continuation")
    )
    settings = ConsoleSessionSettings(
        provider="moonshot" if kind == "continuation" else "vllm",
        model="kimi-k3" if kind == "continuation" else "local-thinking-model",
        base_url="https://api.moonshot.ai/v1"
        if kind == "continuation"
        else "http://127.0.0.1:9099/v1",
        streaming=False,
    )
    database, store, controller, _ = _controller(tmp_path, initial_settings=settings)
    assert store.persist_session_if_needed("session-1")
    store.append_message(
        "session-1",
        role=ConsoleMessageRole.USER,
        content="prior question",
        persist=True,
    )
    prior = store.append_message(
        "session-1",
        role=ConsoleMessageRole.ASSISTANT,
        content="prior answer",
        persist=True,
    )
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider=settings.provider,
        protocol="chat_completions",
        model=settings.model,
        api_base_url=settings.base_url,
        state="complete",
        rounds=(ContinuationRound("prior answer", ("REPLAY_REASONING_CANARY",), ()),),
    )
    if kind == "continuation":
        assert database.update_provider_continuation(
            message_id=prior.persisted_message_id,
            expected_message_version=database.get_connection()
            .execute(
                "SELECT version FROM messages WHERE id = ?",
                (prior.persisted_message_id,),
            )
            .fetchone()[0],
            provider_continuation_json=dump_provider_continuation_json(checkpoint),
            content="prior answer",
        )
        store._message_or_raise(prior.id).provider_continuation = checkpoint
    else:
        envelope = ThinkingEnvelope(
            (
                DisplayableThinkingBlock(
                    block_id="saved-thinking",
                    round_ordinal=0,
                    provider=settings.provider,
                    model=settings.model,
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                    status="complete",
                    text="REPLAY_THINKING_CANARY",
                ),
            )
        )
        store.replace_message_thinking(prior.id, envelope)
        assert store.persist_selected_generation(prior.id)
    original_preparation = controller_module.ConsoleTurnPreparation

    def preparation(**kwargs):
        kwargs["capture_mode"] = (
            ConsoleTraceCaptureMode.CAPTURE_ON
            if capture_on
            else ConsoleTraceCaptureMode.CAPTURE_OFF
        )
        return original_preparation(**kwargs)

    monkeypatch.setattr(controller_module, "ConsoleTurnPreparation", preparation)
    harness = SimpleNamespace(
        database=database,
        store=store,
        controller=controller,
        checkpoint=checkpoint,
        prior=prior,
        capture_on=capture_on,
        entries=[],
        failures=[],
        build_errors=[],
        prepared=[],
        reply_text="answer",
        factory=ConsoleTraceBoundaryFactory(database),
    )
    build_trace = controller._build_durable_trace_request

    def record_build(*args, **kwargs):
        try:
            return build_trace(*args, **kwargs)
        except Exception as error:
            harness.build_errors.append(error)
            raise

    monkeypatch.setattr(controller, "_build_durable_trace_request", record_build)

    def adapter(**kwargs):
        harness.entries.append(kwargs)
        return {"choices": [{"message": {"content": harness.reply_text}}]}

    def boundary(prepared, resolution, route):
        try:
            return harness.factory(prepared, resolution, route)
        except ValueError as error:
            harness.failures.append(str(error))
            raise

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter, trace_call_boundary_factory=boundary
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=settings.provider,
        execution_key=settings.provider,
        model=settings.model,
        base_url=settings.base_url,
        streaming=False,
        continuation_protocol="chat_completions",
        thinking_stream_disposition="displayable" if kind == "thinking" else "ignored",
        thinking_round_trip_version=1 if kind == "thinking" else None,
    )
    resolution = replace(
        resolution, resolved_destination=resolve_console_destination(resolution)
    )

    async def resolve(_selection):
        return resolution

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    prepare = gateway.prepare_chat_request

    def record_preparation(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        harness.prepared.append(prepared)
        return prepared

    monkeypatch.setattr(gateway, "prepare_chat_request", record_preparation)
    controller.provider_gateway = gateway
    controller._chat_dictionary_applier = lambda _conversation, text: text.replace(
        "alias", "expanded"
    )
    try:
        yield harness
    finally:
        await gateway.aclose()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [False, True], indirect=True)
@pytest.mark.parametrize("transformed", [False, True])
async def test_capture_preserves_selected_continuation_and_successor(
    replay_harness, transformed
):
    harness = replay_harness
    first_text = "alias request" if transformed else "ordinary request"
    old_traces = []
    for text in (first_text, "ordinary successor"):
        harness.factory = ConsoleTraceBoundaryFactory(harness.database)
        result = await harness.controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (
            result,
            harness.failures,
            harness.build_errors,
        )
        call = harness.entries[-1]
        assert tuple(call.get("provider_continuations", ())) == (harness.checkpoint,)
        assert len(harness.prepared[-1].continuation_groups) == 1
        visible = thaw_json(call["messages_payload"])
        assert visible[-1]["content"] == text.replace("alias", "expanded")
        assert "REPLAY_REASONING_CANARY" not in repr(visible)
        saved = harness.store.get_message(result.user_message_id)
        assert saved.content == text
        if harness.capture_on:
            reader = ConsoleTraceNativeReader(harness.database)
            trace = reader.read_calls(saved.persisted_message_id)
            assert len(trace) == 1
            assert trace[0].capture.request["messages_payload"] == visible
            assert "REPLAY_REASONING_CANARY" in repr(trace[0].capture.request)
            old_traces.append((saved.persisted_message_id, trace))
            for message_id, before in old_traces:
                assert reader.read_calls(message_id) == before
    assert not harness.failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_harness", [(False, "thinking"), (True, "thinking")], indirect=True
)
@pytest.mark.parametrize("transformed", [False, True])
async def test_capture_preserves_selected_displayable_thinking(
    replay_harness, transformed
):
    harness = replay_harness
    old_traces = []
    for text in (
        "alias request" if transformed else "ordinary request",
        "ordinary successor",
    ):
        harness.factory = ConsoleTraceBoundaryFactory(harness.database)
        result = await harness.controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (result, harness.failures)
        visible = thaw_json(harness.entries[-1]["messages_payload"])
        assert (
            visible[1]["content"]
            == "<think>REPLAY_THINKING_CANARY</think>\nprior answer"
        )
        assert visible[-1]["content"] == text.replace("alias", "expanded")
        assert harness.store.get_message(harness.prior.id).content == "prior answer"
        if harness.capture_on:
            saved = harness.store.get_message(result.user_message_id)
            reader = ConsoleTraceNativeReader(harness.database)
            trace = reader.read_calls(saved.persisted_message_id)
            assert len(trace) == 1
            assert trace[0].capture.request["messages_payload"] == visible
            old_traces.append((saved.persisted_message_id, trace))
            for message_id, before in old_traces:
                assert reader.read_calls(message_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_harness", [(False, "thinking"), (True, "thinking")], indirect=True
)
async def test_transformed_send_replays_new_assistant_thinking_on_cold_successor(
    replay_harness,
):
    harness = replay_harness
    harness.reply_text = "<think>NEW_ASSISTANT_THINKING_CANARY</think>answer"
    first = await harness.controller.submit_draft(
        "alias request", session_id="session-1"
    )
    assert first.accepted and first.provider_started, (first, harness.failures)
    assistant = harness.store.get_message(first.assistant_message_id)
    assert assistant.content == "answer"
    assert assistant.thinking.blocks[0].text == "NEW_ASSISTANT_THINKING_CANARY"
    row = harness.database.get_message_by_id(assistant.persisted_message_id)
    assert parse_thinking_blocks_json(row["thinking_blocks_json"]) == assistant.thinking
    saved_user = harness.store.get_message(first.user_message_id)
    reader = ConsoleTraceNativeReader(harness.database)
    previous_trace = reader.read_calls(saved_user.persisted_message_id)

    harness.factory = ConsoleTraceBoundaryFactory(harness.database)
    result = await harness.controller.submit_draft(
        "ordinary successor", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.failures)
    messages = thaw_json(harness.entries[-1]["messages_payload"])
    assert (
        messages[-2]["content"]
        == "<think>NEW_ASSISTANT_THINKING_CANARY</think>\nanswer"
    )
    if harness.capture_on:
        saved = harness.store.get_message(result.user_message_id)
        trace = reader.read_calls(saved.persisted_message_id)
        assert len(trace) == 1
        assert trace[0].capture.request["messages_payload"] == messages
        assert reader.read_calls(saved_user.persisted_message_id) == previous_trace


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [(True, "thinking")], indirect=True)
@pytest.mark.parametrize("credential_equivalent", [False, True])
async def test_capture_rejects_unsaved_thinking_under_saved_owner(
    replay_harness, credential_equivalent
):
    harness = replay_harness
    canonical = harness.store.get_message(harness.prior.id).thinking
    if credential_equivalent:
        canonical = replace(
            canonical,
            blocks=(replace(canonical.blocks[0], text="Bearer sk-proj-" + "x" * 48),),
        )
        harness.store.replace_message_thinking(harness.prior.id, canonical)
        assert harness.store.persist_selected_generation(harness.prior.id)
    forged = replace(
        canonical,
        blocks=(
            replace(
                canonical.blocks[0],
                text="Bearer sk-proj-" + "y" * 48
                if credential_equivalent
                else "FORGED_THINKING_OWNER_CANARY",
            ),
        ),
    )
    harness.store.replace_message_thinking(harness.prior.id, forged)
    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert result.accepted and not result.provider_started
    assert harness.entries == []
    row = harness.database.get_message_by_id(harness.prior.persisted_message_id)
    assert parse_thinking_blocks_json(row["thinking_blocks_json"]) == canonical


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_harness", [(False, "thinking"), (True, "thinking")], indirect=True
)
async def test_capture_keeps_excluded_thinking_out_of_request(replay_harness):
    harness = replay_harness
    _, persisted = harness.store.set_session_thinking_history_policy(
        "session-1", "exclude"
    )
    assert persisted
    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.failures)
    assert "REPLAY_THINKING_CANARY" not in repr(harness.entries[-1])
    assert (
        harness.store.get_message(harness.prior.id).thinking.blocks[0].text
        == "REPLAY_THINKING_CANARY"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [False, True], indirect=True)
@pytest.mark.parametrize("mismatch", ["model", "endpoint"])
async def test_capture_keeps_incompatible_continuation_out_of_request(
    replay_harness, mismatch
):
    harness = replay_harness
    checkpoint = replace(
        harness.checkpoint,
        **(
            {"model": "kimi-k2"}
            if mismatch == "model"
            else {"api_base_url": "https://different.invalid/v1"}
        ),
    )
    row = harness.database.get_message_by_id(harness.prior.persisted_message_id)
    assert harness.database.update_provider_continuation(
        message_id=harness.prior.persisted_message_id,
        expected_message_version=row["version"],
        provider_continuation_json=dump_provider_continuation_json(checkpoint),
        content="prior answer",
    )
    harness.store._message_or_raise(harness.prior.id).provider_continuation = checkpoint
    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.failures)
    assert not harness.entries[-1].get("provider_continuations")
    assert "REPLAY_REASONING_CANARY" not in repr(harness.entries[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [False, True], indirect=True)
async def test_capture_does_not_attach_continuation_for_missing_owner(
    replay_harness, monkeypatch
):
    harness = replay_harness
    read_sidecars = harness.controller._provider_continuation_sidecar_for_session
    extra = ProviderContinuationSidecar(
        "owner-outside-this-conversation",
        replace(
            harness.checkpoint,
            rounds=(
                ContinuationRound("foreign answer", ("FOREIGN_REASONING_CANARY",), ()),
            ),
        ),
    )
    monkeypatch.setattr(
        harness.controller,
        "_provider_continuation_sidecar_for_session",
        lambda session_id: (*read_sidecars(session_id), extra),
    )
    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.failures)
    assert tuple(harness.entries[-1].get("provider_continuations", ())) == (
        harness.checkpoint,
    )
    assert harness.prepared[-1].continuation_groups[0].checkpoint == harness.checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [True], indirect=True)
async def test_capture_rejects_private_sidecar_value_that_disagrees_with_saved_source(
    replay_harness,
):
    harness = replay_harness
    forged = replace(
        harness.checkpoint,
        rounds=(ContinuationRound("prior answer", ("UNSAVED_PRIVATE_CANARY",), ()),),
    )
    harness.store._message_or_raise(harness.prior.id).provider_continuation = forged
    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert result.accepted and not result.provider_started
    assert not harness.entries
    assert harness.prepared[-1].continuation_groups[0].checkpoint == forged
