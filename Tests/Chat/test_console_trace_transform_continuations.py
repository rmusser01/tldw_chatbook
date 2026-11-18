"""Current-user transforms retain an unchanged saved continuation domain."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_prepared_request import thaw_json
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [True], indirect=True)
@pytest.mark.parametrize("cold_factory", [False, True])
@pytest.mark.parametrize("bound", [False, True])
async def test_transformed_successors_preserve_saved_continuation_domain(
    replay_harness, cold_factory, bound
):
    harness = replay_harness
    # Current sends carry the frozen prompt inputs as the third argument.
    harness.controller._chat_dictionary_applier = (
        lambda _conversation, text, _inputs=None: text.replace("alias", "expanded")
    )
    if bound:
        harness.checkpoint = replace(
            harness.checkpoint, schema_version=3, selected_provider="moonshot"
        )
        owner = harness.prior.persisted_message_id
        version = harness.database.get_message_by_id(owner)["version"]
        assert harness.database.update_provider_continuation(
            message_id=owner,
            expected_message_version=version,
            provider_continuation_json=dump_provider_continuation_json(
                harness.checkpoint
            ),
            content="prior answer",
        )
        harness.store._message_or_raise(
            harness.prior.id
        ).provider_continuation = harness.checkpoint
    reader = ConsoleTraceNativeReader(harness.database)
    previous = []
    for text in ("alias first", "ordinary next", "alias third", "ordinary last"):
        if cold_factory:
            harness.factory = ConsoleTraceBoundaryFactory(harness.database)
        result = await harness.controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (
            text,
            result,
            harness.failures,
            len(harness.entries),
        )
        call = harness.entries[-1]
        assert tuple(call["provider_continuations"]) == (harness.checkpoint,)
        visible = thaw_json(call["messages_payload"])
        assert visible[-1]["content"] == text.replace("alias", "expanded")
        saved = harness.store.get_message(result.user_message_id)
        assert saved.content == text
        trace = reader.read_calls(saved.persisted_message_id)
        assert len(trace) == 1
        assert trace[0].capture.request["messages_payload"] == visible
        assert "REPLAY_REASONING_CANARY" in repr(trace[0].capture.request)
        previous.append((saved.persisted_message_id, trace))
        for message_id, original in previous:
            assert reader.read_calls(message_id) == original
    assert [row["content"] for row in visible if row["role"] == "user"] == [
        "prior question",
        "alias first",
        "ordinary next",
        "alias third",
        "ordinary last",
    ]
