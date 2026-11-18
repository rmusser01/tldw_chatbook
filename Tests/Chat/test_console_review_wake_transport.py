"""Accepted review wakes through the real gateway and physical fake HTTP."""

import asyncio
import io
import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
import urllib3

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderSelection,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.owned_http_client]

PROVIDER = "custom-ep:review-wake"
MODEL = "synthetic-review-model"
BASE_URL = "http://127.0.0.1:8010/v1/chat/completions"


def profile():
    return {
        "custom_endpoints": {
            "review-wake": {
                "display_name": "Review wake fixture",
                "family": "openai_compatible",
                "base_url": BASE_URL,
                "models": [MODEL],
            }
        }
    }


def selection():
    return ConsoleProviderSelection(
        provider=PROVIDER, explicit_model=MODEL, max_tokens=1024, streaming=False
    )


@pytest.mark.asyncio
async def test_pinned_resolution_metadata_joins_send_probe_without_config_reresolution():
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []
    configuration = profile()

    async def physical(request):
        requests.append(str(request.url))
        entered.set()
        await release.wait()
        return httpx.Response(
            200, json={"data": [{"id": MODEL, "context_length": 32000}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(physical)) as client:
        gateway = ConsoleProviderGateway(
            config_provider=lambda: configuration, environ={}, http_client=client
        )
        try:
            resolution = await gateway.resolve_for_send(selection())
            assert not resolution.context_window.verified
            await entered.wait()
            configuration["custom_endpoints"]["review-wake"]["base_url"] = (
                "http://127.0.0.1:9999/v1"
            )
            pending = asyncio.create_task(gateway.resolve_context_window(resolution))
            await asyncio.sleep(0)
            assert not pending.done()
            release.set()
            window = await pending
            assert window.verified and window.tokens == 32000
            assert requests == ["http://127.0.0.1:8010/v1/models"]
        finally:
            release.set()
            await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unready", "endpoint", "missing_destination"])
async def test_pinned_metadata_rejects_invalid_resolution_before_probe(change):
    probes = []

    async def metadata(request):
        probes.append(str(request.url))
        return httpx.Response(
            200, json={"data": [{"id": MODEL, "context_length": 32000}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        gateway = ConsoleProviderGateway(
            config_provider=profile, environ={}, http_client=client
        )
        try:
            resolution = await gateway.resolve_for_send(selection())
            await asyncio.gather(*tuple(gateway._context_window_refreshes))
            changes = {
                "unready": {"ready": False},
                "endpoint": {"base_url": "http://127.0.0.1:9999/v1"},
                "missing_destination": {"resolved_destination": None},
            }
            with pytest.raises(ValueError, match="review_metadata_target_changed"):
                await gateway.resolve_context_window(
                    replace(resolution, **changes[change])
                )
            assert probes == ["http://127.0.0.1:8010/v1/models"]
        finally:
            await gateway.aclose()


class PhysicalReviewTransport:
    reviewers = (("openai", "gpt-6-luna"), ("deepseek", "deepseek-flash"))
    verifier = ("anthropic", "claude-haiku-5-5")

    def __init__(self, monkeypatch):
        self.shadows = []
        self.monkeypatch = monkeypatch
        self.wire = []
        self.metadata = []
        self.boundary = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.wait_observed = False
        self.prepared = []
        self.runtime_errors = []
        self.resolutions = []

    def create_console(self, tmp_path):
        import os
        from pathlib import Path

        from tldw_chatbook import config
        from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        # conftest installs a permanent real-profile write guard and redirects
        # all config before importing production modules. Refuse outside it.
        assert config.get_cli_config_path().is_relative_to(
            Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
        )
        for provider in ("OPENAI", "DEEPSEEK", "ANTHROPIC"):
            self.monkeypatch.setenv(
                provider + "_API_KEY", "synthetic-" + provider.lower()
            )
        from tldw_chatbook.model_capabilities import reload_capabilities

        self.old_capabilities = config.get_cli_setting(
            "model_capabilities", "models", {}
        )
        documented = json.loads(
            (
                Path(__file__).parents[2]
                / "Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/capacity-evidence.json"
            ).read_text()
        )
        caps = {
            row["model"]: {
                key: row[key]
                for key in ("context_window", "max_input_tokens", "max_output_tokens")
                if key in row
            }
            for row in documented
        }
        assert MODEL not in caps
        config.save_setting_to_cli_config("model_capabilities", "models", caps)
        reload_capabilities()
        self.old_endpoint = config.get_cli_setting(
            "custom_endpoints", "review-wake", {}
        )
        config.save_setting_to_cli_config(
            "custom_endpoints",
            "review-wake",
            profile()["custom_endpoints"]["review-wake"],
        )
        self.receipt_path = tmp_path / "physical-receipt.json"
        chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="physical-review")
        self.gateway = ConsoleProviderGateway(
            config_provider=profile,
            trace_shadow_sink=self.shadows.append,
            environ={
                "OPENAI_API_KEY": "synthetic-openai",
                "DEEPSEEK_API_KEY": "synthetic-deepseek",
                "ANTHROPIC_API_KEY": "synthetic-anthropic",
            },
        )
        store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
        runs = AgentRunsDB(tmp_path / "runs.db", client_id="physical-review")
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs, store=store, provider_gateway=self.gateway
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=self.gateway,
            provider=PROVIDER,
            model=MODEL,
            agent_bridge=bridge,
            agent_runtime_enabled=True,
        )
        return controller, store, runs

    async def install_script(self, controller, session, script):
        from tldw_chatbook.Agents.run_context import current_run_actor
        from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

        async def metadata(transport, request):
            assert str(request.url) == "http://127.0.0.1:8010/v1/models"
            self.metadata.append(str(request.url))
            if self.boundary is not None:
                self.entered.set()
                await self.release.wait()
            return httpx.Response(
                200, json={"data": [{"id": MODEL, "context_length": 32000}]}
            )

        self.monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", metadata
        )
        from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
        from tldw_chatbook.LLM_Calls import hosted_provider_engine

        original_boundary = ConsoleTraceBoundaryFactory.__call__

        def observe_boundary(factory, *args, **kwargs):
            try:
                return original_boundary(factory, *args, **kwargs)
            except Exception as exc:
                self.runtime_errors.append(
                    "boundary:" + type(exc).__name__ + ":" + str(exc)
                )
                raise

        self.monkeypatch.setattr(
            ConsoleTraceBoundaryFactory, "__call__", observe_boundary
        )
        original_payload = hosted_provider_engine.build_hosted_chat_payload

        def observe_payload(*args, **kwargs):
            try:
                return original_payload(*args, **kwargs)
            except Exception as exc:
                self.runtime_errors.append(type(exc).__name__ + ":" + str(exc))
                raise

        self.monkeypatch.setattr(
            hosted_provider_engine, "build_hosted_chat_payload", observe_payload
        )
        from tldw_chatbook.Agents import agent_service

        original_error = agent_service.AgentService._service_error_step

        def observe_error(owner, run_id, summary):
            self.runtime_errors.append(summary)
            return original_error(owner, run_id, summary)

        self.monkeypatch.setattr(
            agent_service.AgentService, "_service_error_step", observe_error
        )
        original_loop = agent_service.run_agent_loop

        def observed_loop(*args, **kwargs):
            outcome = original_loop(*args, **kwargs)
            for step in outcome.steps:
                if step.kind == "error":
                    self.runtime_errors.append(step.summary)
            return outcome

        self.monkeypatch.setattr(agent_service, "run_agent_loop", observed_loop)
        original_resolve = self.gateway.resolve_for_send

        async def observe_resolution(*args, **kwargs):
            result = await original_resolve(*args, **kwargs)
            self.resolutions.append(
                (result.provider, result.model, result.ready, result.visible_copy)
            )
            return result

        self.monkeypatch.setattr(self.gateway, "resolve_for_send", observe_resolution)
        native_boundary = controller.code_review._on_boundary

        def boundary(receipt):
            self.boundary = receipt
            with self.gateway._context_windows._lock:
                self.gateway._context_windows._cache.clear()
            native_boundary(receipt)

        controller.code_review._on_boundary = boundary
        # An ordinary initial send can use current real metadata; the next wake
        # must independently resolve its deliberately expired primary target.
        window = await self.gateway.resolve_context_window(
            ConsoleSessionSettings(provider=PROVIDER, model=MODEL, base_url=BASE_URL)
        )
        assert window.verified
        original_prepare = self.gateway.prepare_chat_request

        def observe(*args, **kwargs):
            prepared = original_prepare(*args, **kwargs)
            self.prepared.append((current_run_actor(), prepared))
            return prepared

        self.monkeypatch.setattr(self.gateway, "prepare_chat_request", observe)

        async def observe_wait():
            await asyncio.wait_for(self.entered.wait(), 15)
            before = len(self.wire)
            await asyncio.sleep(0.03)
            assert len(self.wire) == before
            assert not any(row[0].kind == "primary" and row[2] for row in self.wire)
            self.wait_observed = True
            self.release.set()

        self.waiter = asyncio.create_task(observe_wait())

        def physical(pool, connection, method, url, *args, **kwargs):
            actor = current_run_actor()
            assert actor is not None
            assert method == "POST"
            if actor.kind == "primary":
                assert (pool.host, pool.port, url) == (
                    "127.0.0.1",
                    8010,
                    "/v1/chat/completions",
                )
            else:
                assert pool.host in {
                    "api.openai.com",
                    "api.deepseek.com",
                    "api.anthropic.com",
                }
            request = json.loads(kwargs["body"])
            wake = any(
                row.get("role") == "user"
                and "[Accepted review continuation]" in str(row.get("content"))
                for row in request["messages"]
            )
            self.wire.append((actor, request, wake))
            messages = deepcopy(request["messages"])
            if pool.host == "api.anthropic.com":
                native_calls = {
                    block["id"]: block["name"]
                    for row in messages
                    for block in row.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                }
                for row in messages:
                    for block in row.get("content", []):
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            block["content"] = (
                                "Tool result for "
                                + native_calls[block["tool_use_id"]]
                                + ":"
                                + str(block.get("content", ""))
                            )
                for index, row in enumerate(messages):
                    if isinstance(row.get("content"), list):
                        messages[index] = {
                            **row,
                            "content": "\n".join(
                                part.get("text", str(part.get("content", "")))
                                for part in row["content"]
                            ),
                        }
            if messages[-1]["role"] == "tool":
                tool_id = messages[-1]["tool_call_id"]
                call = next(
                    call
                    for row in messages
                    for call in row.get("tool_calls", [])
                    if call["id"] == tool_id
                )
                messages[-1] = {
                    "role": "user",
                    "content": "Tool result for "
                    + call["function"]["name"]
                    + ":"
                    + messages[-1]["content"],
                }

            async def output():
                return await anext(script(None, messages))

            text = asyncio.run(output())
            if text.startswith("```"):
                call = json.loads(text.split("\n", 1)[1].rsplit("```", 1)[0])
                assert call["name"] in {
                    tool.get("function", tool)["name"]
                    for tool in request.get("tools", [])
                }, (
                    call["name"],
                    [
                        tool.get("function", tool)["name"]
                        for tool in request.get("tools", [])
                    ],
                )
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_" + str(len(self.wire)),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                    ]
                }
                finish = "tool_calls"
            else:
                delta, finish = {"content": text}, "stop"
            events = [
                {
                    "id": "synthetic",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": request["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "synthetic",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": request["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            ]
            body = (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            )
            content_type = "text/event-stream"
            if not request.get("stream"):
                message = {"role": "assistant", **delta}
                for call in message.get("tool_calls", []):
                    call.pop("index", None)
                body = json.dumps(
                    {
                        "id": "synthetic",
                        "model": request["model"],
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": finish}
                        ],
                        "usage": events[-1]["usage"],
                    }
                )
                content_type = "application/json"
            if pool.host == "api.anthropic.com":
                native_call = delta.get("tool_calls", [None])[0]
                block = (
                    {
                        "type": "tool_use",
                        "id": native_call["id"],
                        "name": native_call["function"]["name"],
                        "input": json.loads(native_call["function"]["arguments"]),
                    }
                    if native_call
                    else {"type": "text", "text": text}
                )
                stop_reason = "tool_use" if native_call else "end_turn"
                response = {
                    "id": "msg_synthetic",
                    "type": "message",
                    "role": "assistant",
                    "model": request["model"],
                    "content": [block],
                    "stop_reason": stop_reason,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
                if request.get("stream"):
                    chunks = [
                        {
                            "type": "message_start",
                            "message": {**response, "content": []},
                        },
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": (
                                {**block, "input": {}}
                                if native_call
                                else {"type": "text", "text": ""}
                            ),
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": (
                                {
                                    "type": "input_json_delta",
                                    "partial_json": native_call["function"][
                                        "arguments"
                                    ],
                                }
                                if native_call
                                else {"type": "text_delta", "text": text}
                            ),
                        },
                        {"type": "content_block_stop", "index": 0},
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason},
                            "usage": {"output_tokens": 5},
                        },
                        {"type": "message_stop"},
                    ]
                    body = "".join(
                        "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n"
                        for e in chunks
                    )
                else:
                    body, content_type = json.dumps(response), "application/json"
            return urllib3.response.HTTPResponse(
                body=io.BytesIO(body.encode()),
                status=200,
                headers={"Content-Type": content_type},
                preload_content=False,
            )

        self.monkeypatch.setattr(
            urllib3.connectionpool.HTTPConnectionPool, "_make_request", physical
        )

    async def close(self):
        from tldw_chatbook import config

        config.save_setting_to_cli_config(
            "custom_endpoints", "review-wake", self.old_endpoint
        )
        config.save_setting_to_cli_config(
            "model_capabilities", "models", self.old_capabilities
        )
        from tldw_chatbook.model_capabilities import reload_capabilities

        reload_capabilities()
        self.receipt_path.write_text(
            json.dumps(
                {
                    "posts": [
                        {
                            "kind": a.kind,
                            "run_id": a.run_id,
                            "wake": w,
                            "tools": [
                                t.get("function", t)["name"] for t in r.get("tools", [])
                            ],
                            "roles": [m["role"] for m in r["messages"]],
                        }
                        for a, r, w in self.wire
                    ],
                    "prepared": [
                        {
                            "actor": a.kind if a else None,
                            "provider": r.provider,
                            "wire_style": r.wire_style,
                            "roles": [m["role"] for m in r.messages_payload],
                            "semantic_roles": [
                                m["role"] for m in r.semantic.flattened_messages()
                            ],
                        }
                        for a, r in self.prepared
                    ],
                    "boundary": self.boundary.boundary_id if self.boundary else None,
                    "metadata": self.metadata,
                    "runtime_errors": self.runtime_errors,
                    "resolutions": self.resolutions,
                    "waiter_error": str(self.waiter.exception())
                    if self.waiter.done() and not self.waiter.cancelled()
                    else None,
                },
                indent=2,
            )
        )
        self.release.set()
        self.waiter.cancel()
        await asyncio.gather(self.waiter, return_exceptions=True)
        await self.gateway.aclose()


@pytest.mark.asyncio
async def test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_code_review_continuation import (
        test_default_two_child_schedule,
    )
    from Tests.Chat.test_console_multi_model_review_journey import ReviewRunObservation

    observed = ReviewRunObservation()
    transport = PhysicalReviewTransport(monkeypatch)
    await test_default_two_child_schedule(
        tmp_path,
        2,
        True,
        [2, 1],
        journey=observed,
        after_schedule=observed.collect,
        physical_transport=transport,
    )
    assert transport.wait_observed
    assert len(transport.metadata) == 2
    assert observed.launches_per_primary_turn == [2, 1]
    assert observed.original_reports_presented_before_verifier
    assert len(observed.chain_ids) == 1
    assert observed.verifier_run_id not in observed.reviewer_run_ids
    assert observed.original_ids == set(observed.final_dispositions)
    wake_posts = [
        request
        for actor, request, wake in transport.wire
        if actor.kind == "primary" and wake
    ]
    assert len(wake_posts) > 1
    assert wake_posts[0]["messages"][-1]["role"] == "user"
    assert not any(row["role"] == "system" for row in wake_posts[0]["messages"][1:])
    first_primary = next(
        request
        for actor, request, wake in reversed(transport.wire)
        if actor.kind == "primary" and not wake
    )
    from tldw_chatbook.Chat.provider_continuation import project_provider_continuation

    completed = next(
        request
        for actor, request in transport.prepared
        if actor is not None
        and actor.kind == "primary"
        and any(
            group.checkpoint.state == "complete"
            for group in request.continuation_groups
        )
    )
    canonical = [
        row
        for group in completed.continuation_groups
        if group.checkpoint.state == "complete"
        for row in project_provider_continuation(group.checkpoint)
    ]
    replay = [
        row
        for row in wake_posts[0]["messages"]
        if row.get("tool_calls") or row["role"] == "tool"
    ]
    assert canonical and replay == canonical
    assert [call["id"] for row in replay for call in row.get("tool_calls", ())] == [
        call["id"]
        for row in first_primary["messages"]
        for call in row.get("tool_calls", ())
    ]
    assert "spawn_subagent" in {
        tool["function"]["name"] for tool in wake_posts[0]["tools"]
    }
    assert any(row["role"] == "tool" for row in wake_posts[-1]["messages"])
