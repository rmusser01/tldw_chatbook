"""Real-boundary coverage for admitted ephemeral current-user transforms."""

import asyncio
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Chat.test_console_trace_first_send_atomicity import _force_capture_on
from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as make_database,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as make_gateway,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_runtime import (
    test_completed_tool_turn_compound_admission_checks_durable_proof as _tool_successor_case,
)
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_prepared_request import thaw_json
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    SavedRevisionTraceProvenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory


@pytest.fixture
async def trace_harness(tmp_path, monkeypatch, request):
    settings = ConsoleSessionSettings(
        provider="vllm",
        model="test-model",
        base_url="http://127.0.0.1:9099/v1",
        streaming=bool(getattr(request, "param", False)),
    )
    db, store, controller, _ = _controller(tmp_path, initial_settings=settings)
    assert store.persist_session_if_needed("session-1")
    entries = []
    harness = SimpleNamespace(
        db=db,
        store=store,
        controller=controller,
        entries=entries,
        factory=ConsoleTraceBoundaryFactory(db),
        boundaries=[],
        adapter_error=False,
        failures=[],
        partial_received=Event(),
        release_stream=Event(),
    )

    def adapter(**kwargs):
        entries.append(thaw_json(kwargs["messages_payload"]))
        if settings.streaming:

            def chunks():
                yield {
                    "choices": [
                        {
                            "delta": {
                                "content": "partial answer"
                                if harness.adapter_error == "stopped_partial"
                                else "answer"
                            }
                        }
                    ]
                }
                if harness.adapter_error == "stopped_partial":
                    assert harness.release_stream.wait(5)

            return chunks()
        if harness.adapter_error == "stopped":
            controller._signal_stop(session_id="session-1")
        elif harness.adapter_error:
            raise RuntimeError("synthetic provider failure")
        return {"choices": [{"message": {"content": "answer"}}]}

    def boundary(request, resolution, route):
        try:
            result = harness.factory(request, resolution, route)
        except ValueError as exc:
            harness.failures.append(str(exc))
            raise
        harness.boundaries.append(result)
        return result

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter, trace_call_boundary_factory=boundary
    )

    async def resolve(_selection):
        resolution = ConsoleProviderResolution(
            ready=True,
            provider=settings.provider,
            model=settings.model,
            base_url=settings.base_url,
            execution_key="vllm",
            streaming=settings.streaming,
        )
        return replace(
            resolution, resolved_destination=resolve_console_destination(resolution)
        )

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    if settings.streaming:
        append = store.append_stream_chunk

        def record_partial(message_id, chunk):
            result = append(message_id, chunk)
            harness.partial_received.set()
            return result

        monkeypatch.setattr(store, "append_stream_chunk", record_partial)
    controller.provider_gateway = gateway
    controller._chat_dictionary_applier = lambda _conversation, text: text.replace(
        "alias", "expanded"
    )
    _force_capture_on(monkeypatch)
    harness.gateway = gateway
    try:
        yield harness
    finally:
        harness.release_stream.set()
        await gateway.aclose()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cold_factory", [False, True])
async def test_dictionary_send_and_successors_keep_exact_saved_source(
    trace_harness, cold_factory
):
    db, store, controller, entries = (
        trace_harness.db,
        trace_harness.store,
        trace_harness.controller,
        trace_harness.entries,
    )
    saved_ids = []
    original_traces = []
    for text in ("ordinary first", "alias second", "alias third", "ordinary last"):
        if cold_factory:
            trace_harness.factory = ConsoleTraceBoundaryFactory(db)
        result = await controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (text, result, len(entries))
        saved = store.get_message(result.user_message_id)
        assert saved.content == text
        saved_ids.append(saved.persisted_message_id)
        assert entries[-1][-1] == {
            "role": "user",
            "content": text.replace("alias", "expanded"),
        }
        reader = ConsoleTraceNativeReader(db)
        trace = reader.read_calls(saved.persisted_message_id)
        assert len(trace) == 1
        assert trace[0].capture.request["messages_payload"] == entries[-1]
        original_traces.append(trace)
        for previous_id, previous_trace in zip(saved_ids, original_traces, strict=True):
            assert reader.read_calls(previous_id) == previous_trace

    assert entries[-1] == [
        {"role": "user", "content": "ordinary first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "alias second"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "alias third"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "ordinary last"},
    ]
    rows = (
        db.get_connection()
        .execute(
            "SELECT c.turn_id, r.source_message_id FROM console_trace_calls c "
            "JOIN console_trace_events e ON e.call_id = c.call_id "
            "AND e.event_type = 'call_boundary' "
            "JOIN console_trace_semantic_revisions r ON r.revision_id = e.semantic_revision_id "
            "ORDER BY e.sequence"
        )
        .fetchall()
    )
    assert [tuple(row) for row in rows] == [
        (identity, identity) for identity in saved_ids
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["historical_text", "unknown_current", "current_role", "current_image"]
)
async def test_source_pin_never_authorizes_changed_history(
    trace_harness, monkeypatch, change
):
    harness = trace_harness
    first = await harness.controller.submit_draft("alias first", session_id="session-1")
    assert first.provider_started
    assert harness.entries[-1][-1]["content"] == "expanded first"
    persisted_id = harness.store.get_message(first.user_message_id).persisted_message_id
    original = ConsoleTraceNativeReader(harness.db).read_calls(persisted_id)
    original_head = (
        harness.db.get_connection()
        .execute("SELECT surface_node_id FROM console_trace_calls")
        .fetchone()[0]
    )
    substitute = harness.controller._apply_skill_substitution

    async def tamper(rows):
        result = await substitute(rows)
        if change == "historical_text":
            result[0][0] = {**result[0][0], "content": "changed history"}
        elif change == "current_role":
            result[0][-1] = {**result[0][-1], "role": "assistant"}
        elif change == "current_image":
            result[0][-1] = {
                **result[0][-1],
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,Ynl0ZXM="},
                    }
                ],
            }
        else:
            from tldw_chatbook.Chat.console_chat_controller import NATIVE_MESSAGE_ID_KEY

            result[0][-1] = {**result[0][-1], NATIVE_MESSAGE_ID_KEY: "unknown"}
        return result

    monkeypatch.setattr(harness.controller, "_apply_skill_substitution", tamper)
    harness.factory = ConsoleTraceBoundaryFactory(harness.db)
    following = await harness.controller.submit_draft(
        "ordinary next", session_id="session-1"
    )
    assert following.accepted and not following.provider_started
    assert len(harness.entries) == 1
    assert harness.failures
    rows = (
        harness.db.get_connection()
        .execute("SELECT surface_node_id FROM console_trace_calls")
        .fetchall()
    )
    assert len(rows) == 1 and rows[0][0] == original_head
    assert ConsoleTraceNativeReader(harness.db).read_calls(persisted_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "stopped"])
async def test_next_send_after_failed_transformed_call(trace_harness, outcome):
    harness = trace_harness
    harness.adapter_error = outcome
    first = await harness.controller.submit_draft("alias first", session_id="session-1")
    assert first.accepted and len(harness.entries) == 1
    assert harness.entries[0][-1]["content"] == "expanded first"
    assert (
        harness.db.get_connection()
        .execute("SELECT state FROM console_trace_calls")
        .fetchone()[0]
        == outcome
    )
    harness.adapter_error = False
    harness.factory = ConsoleTraceBoundaryFactory(harness.db)
    following = await harness.controller.submit_draft(
        "ordinary next", session_id="session-1"
    )
    assert following.accepted and following.provider_started, (
        following,
        harness.failures,
    )
    assert len(harness.entries) == 2
    assert harness.entries[-1][0] == {"role": "user", "content": "alias first"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "cold",
        "equal_policy_fresh_id",
        "credential_value",
        "wrong_policy",
        "wrong_pii_enabled",
        "wrong_ruleset_revision",
        "wrong_response_revision",
        "incomplete_terminal",
        "unrelated_reservation",
        "non_tool_artifact",
        "changed_prefix",
        "changed_assistant_envelope",
        "extra_new_item",
        "unsupported_route",
        "foreign_assistant",
        "rollback_replace",
        "rollback_append",
        "rollback_bind",
        "cold_fresh_reserved",
        "postcommit_commit",
        "postcommit_cancel",
        "postcommit_agent_new_run",
        "stopped_terminal",
        "changed_loop_source",
        "reverted_loop_source",
    ],
)
async def test_transformed_source_tool_successor_checks_durable_proof(
    tmp_path, make_database, make_gateway, monkeypatch, scenario
):
    await _tool_successor_case(
        tmp_path,
        make_database,
        make_gateway,
        monkeypatch,
        scenario,
        current_transform=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["different_message", "edited_revision", "equal_text_revision"]
)
async def test_cold_successor_requires_the_pinned_revision_identity(
    trace_harness, change
):
    harness = trace_harness
    first = await harness.controller.submit_draft("alias first", session_id="session-1")
    assert first.provider_started
    assert harness.entries[0][-1]["content"] == "expanded first"
    source_id = harness.store.get_message(first.user_message_id).persisted_message_id
    answer_id = harness.store.get_message(
        first.assistant_message_id
    ).persisted_message_id
    database = harness.db
    source_row = database.get_message_by_id(source_id)
    conversation = source_row["conversation_id"]
    source_text = "alias first"
    if change == "different_message":
        _, source = _saved_message(database, conversation, source_text)
    else:
        assert database.update_message(
            source_id,
            {"content": "edited alias first"},
            source_row["version"],
            preserve_descendants=True,
        )
        source_text = "edited alias first"
        if change == "equal_text_revision":
            source_row = database.get_message_by_id(source_id)
            assert database.update_message(
                source_id,
                {"content": "alias first"},
                source_row["version"],
                preserve_descendants=True,
            )
            source_text = "alias first"
        revision = (
            database.get_connection()
            .execute(
                "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1",
                (source_id,),
            )
            .fetchone()[0]
        )
        source = SavedRevisionTraceProvenance(revision)
    answer_revision = (
        database.get_connection()
        .execute(
            "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1",
            (answer_id,),
        )
        .fetchone()[0]
    )
    _, next_user = _saved_message(database, conversation, "next")
    boundary = harness.boundaries[-1]
    policy = boundary._request.semantic.provenance.capture_policy
    request = harness.gateway.prepare_chat_request(
        boundary._resolution,
        _semantic_request(
            [
                {"role": "user", "content": source_text},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "next"},
            ],
            [source, SavedRevisionTraceProvenance(answer_revision), next_user],
            policy,
        ),
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    with pytest.raises(ValueError):
        ConsoleTraceBoundaryFactory(database)(
            request, boundary._resolution, ConsoleRequestRoute.FRESH
        )
    assert len(harness.entries) == 1


@pytest.mark.asyncio
async def test_current_transform_rechecks_its_exact_pretransform_source(
    trace_harness, monkeypatch
):
    harness = trace_harness
    build = harness.controller._build_durable_trace_request

    def change_saved_source(**kwargs):
        source_id = kwargs["committed_user_id"]
        source = harness.db.get_message_by_id(source_id)
        assert harness.db.update_message(
            source_id,
            {"content": "a different saved input"},
            source["version"],
            preserve_descendants=True,
        )
        return build(**kwargs)

    monkeypatch.setattr(
        harness.controller, "_build_durable_trace_request", change_saved_source
    )
    result = await harness.controller.submit_draft(
        "alias first", session_id="session-1"
    )
    assert result.accepted and not result.provider_started
    assert harness.entries == []


@pytest.mark.asyncio
async def test_transformed_successor_owned_retry_reuses_exact_reservation(
    trace_harness, monkeypatch
):
    harness = trace_harness
    first = await harness.controller.submit_draft("alias first", session_id="session-1")
    assert first.provider_started
    assert harness.entries[0][-1]["content"] == "expanded first"
    bind = harness.factory.repository.bind_call
    fail_once = True

    def rollback_once(*args, **kwargs):
        nonlocal fail_once
        result = bind(*args, **kwargs)
        if fail_once:
            fail_once = False
            raise RuntimeError("synthetic bind rollback")
        return result

    monkeypatch.setattr(harness.factory.repository, "bind_call", rollback_once)
    following = await harness.controller.submit_draft("next", session_id="session-1")
    assert following.accepted and not following.provider_started
    assert len(harness.entries) == 1
    pending_id = harness.boundaries[-1].reserve().call_id
    retried = await harness.controller.retry_library_preparation(
        following.preparation_id
    )
    assert retried.accepted and retried.provider_started, retried
    assert len(harness.entries) == 2
    rows = (
        harness.db.get_connection()
        .execute("SELECT call_id, state FROM console_trace_calls")
        .fetchall()
    )
    assert len(rows) == 2
    assert (pending_id, "complete") in [tuple(row) for row in rows]


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_harness", [True], indirect=True)
async def test_streaming_partial_stop_keeps_verified_assistant_on_next_send(
    trace_harness,
):
    harness = trace_harness
    harness.adapter_error = "stopped_partial"
    sending = asyncio.create_task(
        harness.controller.submit_draft("alias first", session_id="session-1")
    )
    try:
        assert await asyncio.to_thread(harness.partial_received.wait, 5)
        assert harness.controller.stop_active_run()
        first = await asyncio.wait_for(sending, 5)
    finally:
        harness.release_stream.set()
    assert first.accepted and first.provider_started
    assert harness.entries[0][-1]["content"] == "expanded first"
    saved_assistant = harness.store.get_message(first.assistant_message_id)
    assert saved_assistant.content == "partial answer"
    row = (
        harness.db.get_connection()
        .execute(
            "SELECT c.state, r.verification_outcome FROM console_trace_calls c "
            "JOIN console_trace_response_links r ON r.call_id = c.call_id"
        )
        .fetchone()
    )
    assert tuple(row) == ("stopped", "verified_equal")
    harness.adapter_error = False
    harness.factory = ConsoleTraceBoundaryFactory(harness.db)
    following = await harness.controller.submit_draft("next", session_id="session-1")
    assert following.accepted and following.provider_started, (
        following,
        harness.failures,
    )
    assert harness.entries[-1] == [
        {"role": "user", "content": "alias first"},
        {"role": "assistant", "content": "partial answer"},
        {"role": "user", "content": "next"},
    ]
