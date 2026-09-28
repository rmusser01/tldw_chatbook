"""Captured-send ownership and refusal regressions for GitHub #2829."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_trace_provenance import SavedRevisionTraceProvenance


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [(True, "thinking")], indirect=True)
async def test_thinking_history_keeps_saved_revision_owner(replay_harness, monkeypatch):
    harness = replay_harness
    requests = []
    build = harness.controller._build_durable_trace_request

    def capture(**kwargs):
        request = build(**kwargs)
        requests.append((kwargs, request))
        return request

    monkeypatch.setattr(harness.controller, "_build_durable_trace_request", capture)
    result = await harness.controller.submit_draft(
        "next request", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.failures)
    kwargs, request = requests[0]
    descriptor = request.provenance.compactable[0].messages[1]
    assert type(descriptor) is SavedRevisionTraceProvenance, (
        descriptor,
        kwargs["provider_messages"],
        kwargs["trace_source_messages"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [(True, "thinking")], indirect=True)
async def test_unsaved_effective_system_prompt_can_be_captured(replay_harness):
    harness = replay_harness
    harness.store.replace_session_settings(
        "session-1",
        replace(
            harness.store.session_settings("session-1"),
            system_prompt="SYSTEM_PROMPT_CAPTURE_CANARY",
        ),
    )
    result = await harness.controller.submit_draft(
        "next request", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (result, harness.build_errors)
    assert harness.entries[-1]["system_message"] == "SYSTEM_PROMPT_CAPTURE_CANARY"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [(True, "thinking")], indirect=True)
async def test_lease_free_rag_capture_passes_an_explicit_empty_launch(
    replay_harness, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events

    captures = []

    async def capture(app, launch, *, user_message):
        captures.append((app, launch, user_message))

    monkeypatch.setattr(
        chat_rag_events, "capture_console_staged_evidence_for_chat", capture
    )
    owner = SimpleNamespace(_app=object())
    provider = ConsoleRuntime._capture_frozen_console_staged_rag.__get__(owner)
    replay_harness.controller._rag_capture_provider = provider
    assert await replay_harness.controller._capture_rag_context("private draft") == (
        None,
        None,
        None,
        None,
    )
    assert captures == [(owner._app, None, "private draft")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("closure", ["failed", "budget_stopped"])
async def test_agent_thinking_history_retains_revision_after_closed_tool_suffix(
    tmp_path, monkeypatch, cold, closure
):
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from Tests.Chat.test_console_trace_discarded_tool_run import _reload_console
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentTraceRequestFactory
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.thinking_blocks import (
        DisplayableThinkingBlock,
        ThinkingEnvelope,
    )

    if closure == "budget_stopped":
        budget = bridge_module.console_run_budget()
        monkeypatch.setattr(
            bridge_module,
            "console_run_budget",
            lambda: replace(budget, max_model_turns=2),
        )
    async with _project_console(tmp_path, monkeypatch, tools=2) as app:
        app.store.set_session_project_instruction_state(
            app.session.id,
            replace(
                app.session.project_instruction_state,
                project_instructions_enabled=False,
            ),
        )
        assert app.store.persist_session_if_needed(app.session.id)
        app.store.append_message(
            app.session.id,
            role=ConsoleMessageRole.USER,
            content="earlier question",
            persist=True,
        )
        prior = app.store.append_message(
            app.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="earlier answer",
            persist=True,
        )
        app.store.replace_message_thinking(
            prior.id,
            ThinkingEnvelope(
                (
                    DisplayableThinkingBlock(
                        block_id="earlier-thinking",
                        round_ordinal=0,
                        provider="llama_cpp",
                        model="qwen3.7-27b",
                        protocol="chat_completions",
                        source_format="start_anchored_think",
                        status="complete",
                        text="EARLIER_THINKING_CANARY",
                    ),
                )
            ),
        )
        assert app.store.persist_selected_generation(prior.id)
        original_resolve = app.gateway.resolve_for_send

        async def resolve(selection):
            return replace(
                await original_resolve(selection),
                thinking_stream_disposition="displayable",
                thinking_round_trip_version=1,
            )

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        built = []
        build = ConsoleAgentTraceRequestFactory.build

        def fail_third(factory, *args, **kwargs):
            if len(app.http_payloads) == 2:
                raise ValueError("synthetic stopped tool run")
            request = build(factory, *args, **kwargs)
            built.append((factory.admitted_request, args, request, kwargs))
            return request

        with monkeypatch.context() as fault:
            fault.setattr(ConsoleAgentTraceRequestFactory, "build", fail_third)
            first = await app.controller.submit_draft(
                "Use calculator twice", session_id=app.session.id
            )
        assert first.accepted and len(app.http_payloads) == 2, (
            first,
            app.reservation_errors,
        )
        admitted, args, request, build_kwargs = built[0]
        assert (
            type(request.provenance.compactable[0].messages[1])
            is SavedRevisionTraceProvenance
        ), (dict(admitted.flattened_messages()[1]), args[0][2])
        with app.db.transaction() as cursor:
            nodes = list(
                cursor.execute(
                    "SELECT component_kind, reference_kind FROM console_trace_surface_nodes ORDER BY sequence"
                )
            )
        assert ("active_request", "artifact") not in [tuple(row) for row in nodes], [
            tuple(row) for row in nodes
        ]
        original_user = app.store.get_message(
            first.user_message_id
        ).persisted_message_id
        reader = ConsoleTraceNativeReader(app.db)
        original_trace = reader.read_calls(original_user)
        assert len(original_trace) == 2
        assert "EARLIER_THINKING_CANARY" in repr(original_trace[0].capture.request)
        if closure == "budget_stopped":
            assert (
                app.runs.list_runs(app.session.persisted_conversation_id)[0]["status"]
                == "stuck"
            )
        if cold:
            await _reload_console(app)
        successor = await app.controller.submit_draft(
            "Continue", session_id=app.session.id
        )
        assert successor.accepted and successor.provider_started, (
            successor,
            app.reservation_errors,
        )
        assert reader.read_calls(original_user) == original_trace
        forged = [dict(row) for row in args[0]]
        forged[2]["content"] = "different assistant content"
        changed = ConsoleAgentTraceRequestFactory(admitted).build(
            forged, **build_kwargs
        )
        assert (
            type(changed.provenance.compactable[0].messages[1])
            is not SavedRevisionTraceProvenance
        )
        relabeled = [dict(row) for row in args[0]]
        relabeled[2]["_tldw_call_thinking_owner"] = "foreign-thinking-owner"
        changed_owner = ConsoleAgentTraceRequestFactory(admitted).build(
            relabeled, **build_kwargs
        )
        assert (
            type(changed_owner.provenance.compactable[0].messages[1])
            is not SavedRevisionTraceProvenance
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["send_without_capture", "cancel"])
async def test_legacy_artifact_history_remains_frozen_and_has_explicit_recovery(
    tmp_path, monkeypatch, action
):
    """Old wrong-owner ledgers are not migrated or implicitly trusted."""
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentTraceRequestFactory
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderArtifactTraceProvenance,
        TraceProvenanceSource,
    )

    budget = bridge_module.console_run_budget()
    monkeypatch.setattr(
        bridge_module, "console_run_budget", lambda: replace(budget, max_model_turns=2)
    )
    async with _project_console(tmp_path, monkeypatch, tools=2) as app:
        app.store.set_session_project_instruction_state(
            app.session.id,
            replace(
                app.session.project_instruction_state,
                project_instructions_enabled=False,
            ),
        )
        assert app.store.persist_session_if_needed(app.session.id)
        app.store.append_message(
            app.session.id,
            role=ConsoleMessageRole.USER,
            content="earlier question",
            persist=True,
        )
        app.store.append_message(
            app.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="earlier answer",
            persist=True,
        )
        build = ConsoleAgentTraceRequestFactory.build

        def legacy_artifact_owner(factory, *args, **kwargs):
            if len(app.http_payloads) == 2:
                raise ValueError("synthetic stopped tool run")
            request = build(factory, *args, **kwargs)
            provenance = request.provenance
            # Persist the exact old ACTIVE_REQUEST owner for unchanged saved
            # assistant bytes, using the real ledger and provider boundary.
            unit = provenance.compactable[0]
            descriptors = list(unit.messages)
            assert type(descriptors[1]) is SavedRevisionTraceProvenance
            descriptors[1] = ProviderArtifactTraceProvenance(
                TraceProvenanceSource.ACTIVE_REQUEST, provenance.capture_policy
            )
            return replace(
                request,
                provenance=replace(
                    provenance,
                    compactable=(
                        replace(unit, messages=tuple(descriptors)),
                        *provenance.compactable[1:],
                    ),
                ),
            )

        with monkeypatch.context() as legacy:
            legacy.setattr(
                ConsoleAgentTraceRequestFactory, "build", legacy_artifact_owner
            )
            first = await app.controller.submit_draft(
                "Use calculator twice", session_id=app.session.id
            )
        assert first.accepted and len(app.http_payloads) == 2
        reader = ConsoleTraceNativeReader(app.db)
        first_id = app.store.get_message(first.user_message_id).persisted_message_id
        prior = reader.read_calls(first_id)
        assert len(prior) == 2
        refused = await app.controller.submit_draft(
            "Continue", session_id=app.session.id
        )
        assert refused.accepted and len(app.http_payloads) == 2
        preparation = app.controller.trace_call_recovery_preparation()
        assert preparation is not None, (refused, app.reservation_errors)
        user_id = app.store.get_message(refused.user_message_id).persisted_message_id
        # Retrying an unchanged legacy ledger refuses again, but keeps the
        # same accepted turn and a fresh exact-live recovery opportunity.
        retried = await app.controller.retry_library_preparation(
            preparation.preparation_id
        )
        assert retried.accepted and len(app.http_payloads) == 2
        preparation = app.controller.trace_call_recovery_preparation()
        assert preparation is not None, (retried, app.reservation_errors)
        assert (
            app.store.get_message(retried.user_message_id).persisted_message_id
            == user_id
        )
        with app.db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_calls WHERE turn_id = ?",
                    (user_id,),
                ).fetchone()[0]
                == 0
            )
        if action == "send_without_capture":
            recovered = await app.controller.send_without_capture(
                preparation.preparation_id
            )
            assert recovered.accepted and recovered.provider_started
            assert len(app.http_payloads) == 3
        else:
            app.controller.cancel_library_preparation(preparation.preparation_id)
            assert len(app.http_payloads) == 2
        assert app.controller.trace_call_recovery_preparation() is None
        with app.db.transaction() as cursor:
            rows = cursor.execute(
                "SELECT id, content, deleted FROM messages WHERE content = ?",
                ("Continue",),
            ).fetchall()
        assert [tuple(row) for row in rows] == [
            (user_id, "Continue", int(action == "cancel"))
        ]
        assert reader.read_calls(first_id) == prior
