"""Real captured agent retries must outlive an intermediate provider failure."""

import json

import pytest

from Tests.Chat.test_console_project_instruction_traces import _project_console
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError, ChatRateLimitError
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("status", [429, 500])
@pytest.mark.parametrize("after_tool", [False, True])
async def test_captured_model_retry_settles_prior_failure(
    tmp_path, monkeypatch, status, after_tool
):
    async with _project_console(tmp_path, monkeypatch, tools=int(after_tool)) as app:
        attempts = []

        def fail_once(**kwargs):
            attempts.append(kwargs["messages_payload"])
            if len(attempts) == (2 if after_tool else 1):
                if status == 429:
                    raise ChatRateLimitError(
                        "fixture provider rejection", provider="deepseek"
                    )
                raise ChatProviderError(
                    "fixture provider rejection",
                    provider="deepseek",
                    status_code=status,
                )
            content = "Fixture reply"
            if after_tool and len(attempts) == 1:
                content = (
                    "```tool_call\n"
                    + json.dumps(
                        {"name": "calculator", "arguments": {"expression": "6*7"}}
                    )
                    + "\n```"
                )
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

        async def resolve(_selection):
            return ConsoleProviderResolution(
                ready=True,
                provider="deepseek",
                execution_key="deepseek",
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="fixture",
                streaming=False,
                resolved_destination=ConsoleResolvedDestination(
                    provider="deepseek",
                    model="deepseek-chat",
                    endpoint_identity="https://api.deepseek.com",
                    egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
                ),
            )

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", fail_once)
        result = await app.controller.submit_draft(
            "Use calculator once" if after_tool else "Hello",
            session_id=app.session.id,
        )
        assert result.accepted
        assistant = app.store.get_message(result.assistant_message_id)
        assert attempts, (result, assistant)
        assert assistant.content == "Fixture reply", app.reservation_errors
        assert assistant.status == "complete"
        assert len(attempts) == (3 if after_tool else 2)
        assert app.reservation_errors == []
        with app.db.transaction() as cursor:
            calls = cursor.execute(
                "SELECT run_id, call_sequence, state, outcome FROM console_trace_calls "
                "ORDER BY call_sequence"
            ).fetchall()
        assert len({row[0] for row in calls}) == 1
        assert [row[1] for row in calls] == list(range(len(attempts)))
        assert [row[2] for row in calls] == (
            ["complete", "error", "complete"] if after_tool else ["error", "complete"]
        )
        assert app.store.dispatch_recovery_for_session(app.session.id) is None
