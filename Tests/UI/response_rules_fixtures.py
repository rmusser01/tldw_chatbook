"""Actual Console owners with only provider transport/resolution scripted."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from Tests.Chat.test_response_rules_builder import Transport
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


@asynccontextmanager
async def mounted_rules_console(tmp_path, size=(80, 24)):
    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(tmp_path / "console-rules.sqlite", "rules-ui")
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    app.local_chat_conversation_service = ChatConversationService(app.chachanotes_db)
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )

    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=app.local_chat_conversation_service,
        server_service=None,
    )
    _configure_native_ready_console(app, model="test-model")
    from tldw_chatbook.config import save_setting_to_cli_config

    assert save_setting_to_cli_config("console", "agent_runtime", False)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        owner = console._console_runtime()
        rules = owner.ensure_response_rules()
        gateway = controller.provider_gateway
        transport = Transport()
        case = SimpleNamespace(
            app=app,
            host=host,
            pilot=pilot,
            console=console,
            controller=controller,
            owner=owner,
            rules=rules,
            chats=controller.store,
            session_id=controller.store.active_session_id,
            gateway=gateway,
            transport=transport,
            reply="Evidence: corrected answer",
            requests=[],
            failures=[],
            primary_entered=None,
            primary_release=None,
        )

        import json
        import httpx

        async def http_transport(request):
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "data": [{"id": "test-model"}],
                        "default_generation_settings": {"n_ctx": 32768},
                    },
                )
            envelope = json.loads(request.content)
            payload = envelope.get("messages", ())
            try:
                wire = json.loads(payload[-1]["content"])
            except (ValueError, KeyError, IndexError, TypeError):
                wire = None
            if isinstance(wire, dict) and any(
                k in wire for k in ("complaint", "rules", "reviewed_candidate")
            ):
                transport.requests.append(wire)
                if "rules" in wire:
                    checks = []
                    for i, value in enumerate(wire["cases"]):
                        rule = wire["rules"][0]
                        checks.append(
                            {
                                "case_id": value["case_id"],
                                "rule_id": rule["rule_id"],
                                "revision": rule["revision"],
                                "applicability": "applicable",
                                "verdict": transport.judge_verdicts[i],
                                "basis": "response",
                                "reason": "fixture",
                                "references": [
                                    {
                                        "ref": "response",
                                        "excerpt": value["response_text"],
                                    }
                                ],
                            }
                        )
                    content = json.dumps({"checks": checks})
                else:
                    content = json.dumps(transport.answer)
            else:
                case.requests.append(tuple(payload))
                if case.primary_entered is not None:
                    case.primary_entered.set()
                if case.primary_release is not None:
                    await case.primary_release.wait()
                content = case.reply
            if envelope.get("stream"):
                delta = json.dumps(
                    {
                        "choices": [
                            {"delta": {"content": content}, "finish_reason": None}
                        ]
                    }
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content="data: " + delta + "\n\ndata: [DONE]\n\n",
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        await gateway.aclose()
        gateway._new_owned_http_client = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(http_transport), trust_env=False
        )
        case.dispatches = []
        case.learning_results = []
        learn = rules.learn

        async def observed_learn(*args, **kwargs):
            result = await learn(*args, **kwargs)
            case.learning_results.append((result.state, result.reason))
            return result

        rules.learn = observed_learn
        effect = controller._run_durable_postcommit_effect

        async def observed_effect(*args, **kwargs):
            try:
                return await effect(*args, **kwargs)
            except Exception as exc:
                import traceback

                cause = exc
                seen = set()
                while (cause.__cause__ or cause.__context__) is not None and id(
                    cause
                ) not in seen:
                    seen.add(id(cause))
                    cause = cause.__cause__ or cause.__context__
                case.failures.append(
                    (
                        args[1],
                        type(exc).__name__,
                        str(exc),
                        [
                            (f.name, f.lineno)
                            for f in traceback.extract_tb(cause.__traceback__)[-5:]
                        ],
                    )
                )
                raise

        controller._run_durable_postcommit_effect = observed_effect
        commit = case.chats.commit_durable_turn

        def observed_commit(*args, **kwargs):
            try:
                return commit(*args, **kwargs)
            except Exception as exc:
                case.failures.append((type(exc).__name__, str(exc)))
                raise

        case.chats.commit_durable_turn = observed_commit
        submit = rules.queue._submit_queued

        async def observed_submit(*args, **kwargs):
            result = await submit(*args, **kwargs)
            case.dispatches.append(result)
            return result

        rules.queue._submit_queued = observed_submit
        case.composer = console.query_one(
            "#console-native-composer", ConsoleComposerBar
        )
        try:
            yield case
        finally:
            await owner.dispose()


def seed_answer(case):
    user = case.chats.append_message(
        case.session_id,
        role=ConsoleMessageRole.USER,
        content="Explain result",
        persist=True,
    )
    case.chats.persist_message_if_needed(user.id)
    answer = case.chats.append_message(
        case.session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Missing proof",
        persist=True,
    )
    return case.chats.persist_message_if_needed(answer.id)
