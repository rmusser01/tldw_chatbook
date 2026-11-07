"""V13: actual complete Console workload with an owned scripted HTTP provider.

ConsoleProviderGateway.stream_chat and all its admission logic are unchanged.
Only an explicitly selected owned loopback HTTP endpoint returns synthetic planning.
No fake ready flags, private owner-map writes, direct verdicts, fake starts,
backend output, or render facts. Actual guards may refuse; preserve refusal.
This is prepared QA code, not a passing product timing qualification.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from approval_fixture_driver import ApprovalFixtureDriver as _RequestOnlyDriver

APPROVAL_MODULES = (
    "tldw_chatbook.Chat.console_approval_feedback",
    "tldw_chatbook.UI.Console_Modules.approval_feedback",
    "tldw_chatbook.UI.Console_Modules.approval_controls",
    "tldw_chatbook.UI.Console_Modules.approval_details",
    "tldw_chatbook.Chat.approval_presentation",
    "tldw_chatbook.Agents.approval_observation",
)


class ScriptPreconditionRefused(ValueError):
    def __init__(self, code, correlation):
        super().__init__(code)
        self.code, self.correlation = code, correlation


class ScriptedProvider:
    """Bounded private HTTP planning endpoint, never an external/model client.

    Construct/start before seeding config or constructing TldwCli. Only the
    genuine product gateway and its HTTP adapter consume these scripted turns.
    All request/response bodies remain in memory; observations are content-free.
    """

    model = "gpt-4o-mini"
    provider_key = "openai"
    synthetic_key = "sk-private-fixture-not-a-real-key"

    def __init__(self):
        self._lock = threading.Lock()
        self._turns = []
        self.scripted_turns = 0
        self.chat_post_entries = 0
        self.observations = []
        self.fixture = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def reply(self, status, payload, content_type="application/json"):
                raw = (
                    payload
                    if isinstance(payload, bytes)
                    else json.dumps(payload).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
                self.close_connection = True

            def do_GET(self):
                if self.path not in {"/health", "/props", "/v1/models", "/models"}:
                    self.reply(404, {"error": "private_route_refused"})
                    return
                owner.note("scripted_readiness_request", endpoint=self.path)
                if self.path in {"/models", "/v1/models"}:
                    payload = {
                        "object": "list",
                        "data": [
                            {
                                "id": owner.model,
                                "object": "model",
                                "owned_by": "private-fixture",
                            }
                        ],
                    }
                elif self.path == "/props":
                    payload = {
                        "default_generation_settings": {"n_ctx": 32768},
                        "total_slots": 1,
                        "model_path": owner.model,
                    }
                else:
                    payload = {"status": "ok"}
                self.reply(200, payload)

            def do_POST(self):
                if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                    self.reply(404, {"error": "private_route_refused"})
                    return
                # Count physical adapter entries BEFORE parse or script admission.
                # An empty queue returning409 must still be visible to negatives.
                with owner._lock:
                    owner.chat_post_entries += 1
                owner.note("scripted_chat_post_entered")
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4 * 1024 * 1024:
                        raise ValueError("body_bound")
                    request = json.loads(self.rfile.read(size))
                    if not isinstance(request, dict):
                        raise ValueError("body_type")
                    if (
                        request.get("model") != owner.model
                        or type(request.get("stream", False)) is not bool  # noqa: E721 - Exact boundary types reject bool/subclasses.
                    ):
                        raise ValueError("unowned_request")
                    message, correlation, ordinal = owner.consume(request)
                    if message.get("tool_calls") and not request.get("tools"):
                        raise ValueError("native_tool_definitions_required")
                except ScriptPreconditionRefused as exc:
                    owner.note(
                        "scripted_precondition_refused",
                        correlation=exc.correlation,
                        code=exc.code,
                    )
                    self.reply(
                        409,
                        {
                            "error": "private_script_precondition_refused",
                            "code": exc.code,
                        },
                    )
                    return
                except (ValueError, RuntimeError, json.JSONDecodeError):
                    self.reply(409, {"error": "private_script_admission_refused"})
                    return
                # This timestamp is actual scripted transport entry, NOT a
                # parsed tool request, model output, provider latency, or paint.
                owner.note(
                    "scripted_transport_request",
                    correlation=correlation,
                    ordinal=ordinal,
                    request_bytes=size,
                )
                owner.note(
                    "scripted_output_start", correlation=correlation, ordinal=ordinal
                )
                calls = message.get("tool_calls")
                finish = "tool_calls" if calls else "stop"
                if request.get("stream"):
                    delta = {"role": "assistant"}
                    if calls:
                        delta["tool_calls"] = [
                            {"index": index, **call} for index, call in enumerate(calls)
                        ]
                    else:
                        delta["content"] = message["content"]
                    chunk = {
                        "id": "private-fixture-" + str(ordinal),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": owner.model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                    end = {
                        "id": chunk["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": owner.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                    raw = (
                        "data: "
                        + json.dumps(chunk)
                        + "\n\ndata: "
                        + json.dumps(end)
                        + "\n\ndata: [DONE]\n\n"
                    ).encode("utf-8")
                    self.reply(200, raw, "text/event-stream")
                else:
                    self.reply(
                        200,
                        {
                            "id": "private-fixture-" + str(ordinal),
                            "object": "chat.completion",
                            "created": 0,
                            "model": owner.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", **message},
                                    "finish_reason": finish,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        },
                    )
                owner.note(
                    "scripted_response_written",
                    correlation=correlation,
                    ordinal=ordinal,
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="private-scripted-provider",
            daemon=True,
        )
        self.url = "http://127.0.0.1:" + str(self.server.server_port)
        self.base_url = self.url + "/v1"

    def start(self):
        self.thread.start()
        return self

    def offline_settings(self):
        """Canonical writer mapping; apply BEFORE app construction/import."""
        return {
            "general": {"default_tab": "chat"},
            "first_run": {"setup_started": True, "setup_completed": True},
            "splash_screen": {"enabled": False},
            "model_catalog": {"auto_refresh_enabled": False},
            "permission_summary": {"enabled": False},
            "tools": {"full_agent_tools_enabled": True},
            "console": {
                "agent_runtime": True,
                "native_tool_calls": True,
                "local_tools_enabled": True,
                "raw_cli_permitted": True,
            },
            "chat_defaults": {
                "provider": self.provider_key,
                "model": self.model,
                "streaming": True,
                "max_tokens": 4096,
            },
            "providers": {"OpenAI": [self.model]},
            "api_settings.openai": {
                "api_base_url": self.base_url,
                "model": self.model,
                "api_key": self.synthetic_key,
            },
        }

    def note(self, stage, **facts):
        row = {"stage": stage, "timestamp_ns": time.perf_counter_ns(), **facts}
        with self._lock:
            if len(self.observations) >= 8192:
                raise RuntimeError("Private HTTP observation bound exceeded")
            self.observations.append(row)
        fixture = self.fixture
        if fixture is not None and facts.get("correlation"):
            values = dict(facts)
            correlation = values.pop("correlation")
            fixture._record(
                stage, correlation, timestamp_ns=row["timestamp_ns"], **values
            )

    def plan(self, turns, correlation):
        # Serialize entirely before the actual submit. This fixture cost is
        # explicit and is never mislabeled completed parsed-tool input.
        if (
            len(turns) != 4
            or not isinstance(turns[0], list)
            or len(turns[0]) != 1
            or turns[0][0][0] != "find_tools"
            or not isinstance(turns[1], list)
            or len(turns[1]) != 1
            or turns[1][0][0] != "load_tools"
            or not isinstance(turns[2], list)
            or not turns[2]
            or not isinstance(turns[3], str)
        ):
            raise ValueError("Exact discovery/load/target/completion plan required")
        tool = turns[0][0][1]["query"]
        ids = turns[1][0][1]["ids"]
        if len(ids) != 1 or any(name != tool for name, _args in turns[2]):
            raise ValueError("Exact single target schema required")
        catalog_id = ids[0]
        values = []
        for ordinal, turn in enumerate(turns):
            if isinstance(turn, list):
                calls = [
                    {
                        "id": "qa-"
                        + correlation[:12]
                        + "-"
                        + str(ordinal)
                        + "-"
                        + str(index),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for index, (name, args) in enumerate(turn)
                ]
                values.append({"content": None, "tool_calls": calls})
            else:
                values.append({"content": turn})
        with self._lock:
            if self._turns:
                raise RuntimeError("Previous planning turns remain")
            self._turns = [
                (value, correlation, phase, tool, catalog_id)
                for phase, value in enumerate(values)
            ]

    @staticmethod
    def _validate_prior_results(request, correlation, phase, tool, catalog_id):
        if phase not in (1, 2):
            return
        messages = request.get("messages", [])
        if not isinstance(messages, list):
            raise ScriptPreconditionRefused("prior_messages_missing", correlation)

        def prior_content(ordinal):
            call_id = "qa-" + correlation[:12] + "-" + str(ordinal) + "-0"
            found = [
                row.get("content")
                for row in messages
                if isinstance(row, dict)
                and row.get("role") == "tool"
                and row.get("tool_call_id") == call_id
            ]
            return found[0] if len(found) == 1 and isinstance(found[0], str) else None

        discovery = prior_content(0)
        expected = catalog_id + " — " + tool + ": "
        if discovery is None or not any(
            line.startswith(expected) for line in discovery.splitlines()
        ):
            raise ScriptPreconditionRefused(
                "actual_discovery_target_missing", correlation
            )
        if phase == 2:
            if prior_content(1) != "loaded: " + tool:
                raise ScriptPreconditionRefused(
                    "actual_load_target_not_accepted", correlation
                )
            schemas = request.get("tools", [])
            if not isinstance(schemas, list) or not any(
                isinstance(schema, dict)
                and schema.get("type") == "function"
                and isinstance(schema.get("function"), dict)
                and schema["function"].get("name") == tool
                for schema in schemas
            ):
                raise ScriptPreconditionRefused(
                    "actual_native_target_schema_missing", correlation
                )

    def consume(self, request):
        with self._lock:
            if not self._turns or self.scripted_turns >= 1024:
                raise RuntimeError("No admitted private scripted turn")
            value, correlation, phase, tool, catalog_id = self._turns[0]
            self._validate_prior_results(request, correlation, phase, tool, catalog_id)
            self._turns.pop(0)
            self.scripted_turns += 1
            return value, correlation, self.scripted_turns

    def pending_count(self):
        with self._lock:
            return len(self._turns)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Private scripted provider failed bounded cleanup")


class ApprovalFixtureDriver(_RequestOnlyDriver):
    def __init__(self, app, private_root, emit=None, *, provider):
        super().__init__(app, private_root, emit)
        if not isinstance(provider, ScriptedProvider) or not provider.thread.is_alive():
            raise RuntimeError("Owned running scripted provider required")
        self.provider = provider
        provider.fixture = self
        self._original_bridge_run_reply = None
        self._bridge = None
        self.workspace_id = None
        self._closed = False
        self.gateway = None
        self._original_methods = {}
        self._owner_observations = []
        if getattr(app, "console_provider_gateway_factory", None) is not None:
            raise RuntimeError(
                "Actual stock Console gateway required; factory override refused"
            )

    def _observation_failed(self, stage, error):
        # Diagnostics never replace an original owner result or exception.
        try:
            with self._lock:
                current = self.current
                if current is not None:
                    failures = current.setdefault("observation_failures", [])
                    if len(failures) < 16:
                        failures.append(
                            {
                                "stage": stage,
                                "exception_type": type(error).__module__
                                + "."
                                + type(error).__qualname__,
                            }
                        )
                    current["protocol_error"] = (
                        "QA observation failed; qualification rejected"
                    )
        except Exception:
            pass

    def _attach_marker(self, stage):
        with (self.root.parent / "attach-stages.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write(
                json.dumps(
                    {
                        "stage": stage,
                        "timestamp_ns": time.perf_counter_ns(),
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                    }
                )
                + "\n"
            )

    async def attach(self):
        self._attach_marker("attach_entered")
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        if not isinstance(self.app, TldwCli):
            raise RuntimeError("Actual TldwCli required")
        self._attach_marker("ui_ready_wait_before")
        deadline = time.monotonic() + 35
        while not getattr(self.app, "_ui_ready", False):
            if time.monotonic() > deadline:
                raise RuntimeError("Actual app startup not ready; preserve guards")
            await asyncio.sleep(0.02)
        self._attach_marker("ui_ready_wait_after")
        if not self.app._instance_lock_status.acquired:
            raise RuntimeError("Actual private instance lock not acquired")
        self._attach_marker("navigate_chat_before")
        await self.app.handle_screen_navigation(NavigateToScreen("chat"))
        self._attach_marker("navigate_chat_after")
        if getattr(self.app.screen, "screen_name", None) != "chat":
            raise RuntimeError(
                "Console navigation/modal blocked; no dismissal shortcut"
            )
        self._attach_marker("ensure_controller_before")
        self.controller = self.app.screen._ensure_console_chat_controller()
        self._attach_marker("ensure_controller_after")
        from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

        self._attach_marker("gateway_peek_before")
        self.gateway = self.app.console_runtime.provider_gateway
        self._attach_marker("gateway_peek_after")
        if (
            type(self.gateway) is not ConsoleProviderGateway
            or self.gateway._chat_api_call_fn is not None
        ):
            raise RuntimeError("Stock genuine gateway/adapter required")
        self._attach_marker("fixture_directory_before")
        self.root.mkdir(parents=True, exist_ok=True)
        self._attach_marker("fixture_directory_after")
        self._attach_marker("registry_peek_before")
        registry = getattr(self.app, "workspace_registry_service", None)
        self._attach_marker("registry_peek_after")
        if registry is None:
            raise RuntimeError("Actual app workspace registry unavailable")
        from tldw_chatbook.DB.base_db import run_owned_db_call

        workspace_id = "qa-approval-" + self.new_correlation()  # Mint on the loop.

        def setup_private_workspace():
            # One serialized finite owner callback, existing operation-owned
            # DB connection + maintenance admission + original observers.
            self._attach_marker("workspace_owned_worker_entered")
            self._attach_marker("create_workspace_before")
            workspace = registry.create_workspace(
                workspace_id=workspace_id,
                name="Private approval qualification",
                assistant_defaults=None,
            )
            self._attach_marker("create_workspace_after")
            self._attach_marker("add_folder_binding_before")
            registry.add_folder_binding(
                workspace.workspace_id, self.root, allow_write=True
            )
            self._attach_marker("add_folder_binding_after")
            self._attach_marker("set_workspace_before")
            registry.set_active_workspace(workspace.workspace_id)
            self._attach_marker("set_workspace_after")
            self._attach_marker("workspace_owned_worker_returned")
            return workspace

        self._attach_marker("workspace_owned_call_before")
        workspace = await run_owned_db_call(registry.db, setup_private_workspace)
        self._attach_marker("workspace_owned_call_after")
        self.workspace_id = workspace.workspace_id
        self._attach_marker("new_session_before")
        self.session = self.controller.new_session(
            title="Private approval qualification",
            workspace_id=workspace.workspace_id,
            ephemeral=False,
            assistant_kind=None,
        )
        self._attach_marker("new_session_after")
        self._attach_marker("service_peek_before")
        service = getattr(self.app, "unified_mcp_service", None)
        self._attach_marker("service_peek_after")
        if service is None:
            raise RuntimeError("Actual app permission service unavailable")
        self._attach_marker("global_ask_before")
        service.set_global_default("ask")
        self._attach_marker("global_ask_after")
        self._attach_marker("bridge_observer_before")
        self._bridge = self.controller._agent_bridge
        self._original_bridge_run_reply = self._bridge.run_reply
        run_reply = self._original_bridge_run_reply

        def observe_run_reply(*args, **kwargs):
            review = kwargs.get("review_tool_calls")
            if not callable(review):
                raise RuntimeError("Actual whole-tool review hook unavailable")

            def observe_review(calls, run_id):
                entered = time.perf_counter_ns()
                try:
                    current = self.current
                    if current and any(
                        call.name not in {"find_tools", "load_tools"} for call in calls
                    ):
                        with self._lock:
                            current["run_id"] = run_id
                        current.setdefault("actual_review_call_counts", []).append(
                            len(calls)
                        )
                        if len(calls) != (4 if current["shape"] == "batch" else 1):
                            current["protocol_error"] = (
                                "Actual parsed target call count differs from requested shape"
                            )
                        if current["shape"] in {"large", "raw-deny"}:
                            key = (
                                "content" if current["shape"] == "large" else "command"
                            )
                            expected = 1048576 if key == "content" else 16384
                            sizes = [
                                len(call.args[key].encode("utf-8"))
                                if isinstance(call.args.get(key), str)
                                else None
                                for call in calls
                            ]
                            current["actual_review_argument_bytes"] = sizes
                            if sizes != [expected]:
                                current["protocol_error"] = (
                                    "Actual decoded argument byte count differs from required shape boundary"
                                )
                        self._record(
                            "request_complete",
                            current["correlation"],
                            timestamp_ns=entered,
                            call_count=len(calls),
                            boundary="actual_complete_ToolCall_batch_before_original_combined_review",
                        )
                except Exception as error:
                    self._observation_failed("whole_review", error)
                return review(calls, run_id)

            kwargs["review_tool_calls"] = observe_review
            return run_reply(*args, **kwargs)

        self._bridge.run_reply = observe_run_reply
        self._attach_marker("bridge_observer_after")
        # Observe live controller methods without accessing the lazy feedback
        # property. Do not import feedback/Details/draft modules on attach.
        for name in (
            "request_mcp_approvals",
            "resolve_pending_approval",
            "_publish_approval_observation",
            "_provider_selection_for_session",
        ):
            original = getattr(self.controller, name, None)
            if callable(original):
                self._original_methods[name] = original
        selection_reader = self._original_methods.get("_provider_selection_for_session")
        if selection_reader is not None:

            def observe_selection(*args, **kwargs):
                actual = selection_reader(
                    *args, **kwargs
                )  # Only when PRODUCT asks; no prewarm/readiness force.
                try:
                    current = self.current
                    owner = args[0] if args else kwargs.get("session_id")
                    if (
                        current
                        and self.session is not None
                        and owner == self.session.id
                    ):
                        proof = {
                            "provider_matches": str(actual.provider or "").lower()
                            == self.provider.provider_key,
                            "model_matches": (
                                actual.explicit_model or actual.configured_model
                            )
                            == self.provider.model,
                            "explicit_base_url_matches": str(
                                actual.base_url or ""
                            ).rstrip("/")
                            == self.provider.base_url.rstrip("/"),
                            "base_url_present": actual.base_url is not None,
                        }
                        with self._lock:
                            current["actual_selection_equality"] = proof
                        self._record(
                            "actual_selection_observed", current["correlation"], **proof
                        )
                except Exception as error:
                    self._observation_failed("selection", error)
                return actual

            self.controller._provider_selection_for_session = observe_selection
        request = self._original_methods["request_mcp_approvals"]

        def observe_request(pending, *args, **kwargs):
            from tldw_chatbook.Agents.run_context import current_run_id

            current = self.current
            if current is None:
                return request(pending, *args, **kwargs)
            with self._lock:
                current["run_id"] = current_run_id()
            current.setdefault("actual_pending_call_counts", []).append(len(pending))
            if len(pending) != (4 if current["shape"] == "batch" else 1):
                current["protocol_error"] = (
                    "Actual pending approval call count differs from requested shape"
                )
            self._record(
                "local_checks_complete", current["correlation"], call_count=len(pending)
            )
            decisions = request(pending, *args, **kwargs)
            with self._lock:
                current["actual_decisions"] = dict(decisions)
            self._record(
                "worker_released", current["correlation"], decision_count=len(decisions)
            )
            return decisions

        self.controller.request_mcp_approvals = observe_request
        resolve = self._original_methods["resolve_pending_approval"]

        def observe_resolve(*args, **kwargs):
            current = self.current
            if current:
                self._record("resolver_entered", current["correlation"])
            # A wrong raw choice must NEVER launch a shell in this deny-only
            # fixture. Use the genuine Disarm owner BEFORE resolver release;
            # do not fabricate verdicts or alter a stale/different round.
            choices = args[0] if args else kwargs.get("decisions", {})
            wrong_raw = (
                current
                and current["shape"] == "raw-deny"
                and any(value != "deny" for value in choices.values())
            )
            if wrong_raw:
                with self.controller._approval_state_lock:
                    state = self.controller._pending_approval_rounds.get(
                        kwargs.get("round_id")
                    )
                    same_round = state is not None and state.get(
                        "run_id"
                    ) == current.get("run_id")
                if same_round:
                    self.app.raw_cli_runtime.disarm()
                    with self._lock:
                        current["protocol_error"] = (
                            "Non-Deny raw gesture: actual Disarm revoked authority before release"
                        )
            return resolve(*args, **kwargs)

        self.controller.resolve_pending_approval = observe_resolve
        publish = self._original_methods.get("_publish_approval_observation")

        def observe_publish(observation):
            published = publish(observation)
            current = self.current
            if current and observation.identity.run_id == current.get("run_id"):
                now = time.perf_counter_ns()
                with self._lock:
                    self._owner_observations.append(
                        {
                            "correlation": current["correlation"],
                            "kind": observation.kind,
                            "outcome": observation.outcome,
                            "timestamp_ns": now,
                        }
                    )
                stage = {
                    "settled": "host_outcome",
                    "dispatch_started": "dispatch",
                    "backend_started": "backend_started",
                    "tool_completed": "result",
                }.get(observation.kind)
                if stage:
                    self._record(
                        stage,
                        current["correlation"],
                        timestamp_ns=now,
                        outcome=observation.outcome,
                    )
                elif observation.kind == "grant" and observation.outcome == "applied":
                    self._record(
                        "grant_applied", current["correlation"], timestamp_ns=now
                    )
            return published

        if publish is not None:
            self.controller._publish_approval_observation = observe_publish
        self._attach_marker("attach_snapshot_before")
        value = self.snapshot()
        self._attach_marker("attach_snapshot_after")
        return value

    async def stage(self, shape, correlation):
        if (
            shape not in ("single", "batch", "raw-deny", "large")
            or correlation not in self._correlations
        ):
            raise ValueError("Internally minted correlation/shape required")
        if self.worker is not None and not self.worker.done():
            raise RuntimeError("Previous actual submission remains active")
        if self.controller is None:
            raise RuntimeError("Attach to actual Console first")
        if self.controller.run_state.status.value in {
            "validating",
            "streaming",
            "checking_citations",
            "retrying",
        }:
            raise RuntimeError("Actual Console is busy")
        current = {
            "shape": shape,
            "correlation": correlation,
            "first_use": self._sample_count[shape] == 0,
            "first_shape_use": self._sample_count[shape] == 0,
            "process_cold_first_use": not any(self._sample_count.values()),
            "resident_approval_modules_before": [
                name for name in APPROVAL_MODULES if name in sys.modules
            ],
            "run_id": None,
            "actual_decisions": None,
            "actual_selection_equality": None,
            "unavailable": [
                "real_provider_argument_generation",
                "external_provider_network_dispatch",
                "real_provider_response",
                "real_model_continuation",
            ],
        }
        self._sample_count[shape] += 1
        self.current = current
        # Fixed empty history per workload; use public session creation and
        # real durable ownership, not private state mutation.
        self.session = self.controller.new_session(
            title="QA-" + correlation[:12],
            workspace_id=self.workspace_id,
            ephemeral=False,
            assistant_kind=None,
        )
        from tldw_chatbook.Chat.console_chat_controller import (
            list_project_instruction_bindings,
            commit_project_instruction_setup_decision,
            commit_project_instruction_dispatch_decision,
            project_instruction_notice_key,
        )

        registry = self.app.workspace_registry_service
        options = tuple(list_project_instruction_bindings(self.session, registry))
        if len(options) != 1 or Path(options[0].root).resolve() != self.root:
            raise RuntimeError("Exact sole private fixture binding required")
        action, selection = commit_project_instruction_setup_decision(
            store=self.controller.store,
            session_id=self.session.id,
            registry=registry,
            expected_state=self.session.project_instruction_state,
            expected_options=options,
            action="select",
            binding_id=options[0].binding.binding_id,
        )
        self._record(
            "fixture_setup",
            correlation,
            boundary="real_project_binding_commit",
            outcome=action,
        )
        if action != "select" or selection is None:
            raise RuntimeError("Actual private project binding selection refused")
        self.session = next(
            item
            for item in self.controller.store.sessions()
            if item.id == self.session.id
        )
        notice = project_instruction_notice_key(
            selection.locator_fingerprint,
            self.provider.provider_key,
            self.provider.base_url,
        )
        consent = commit_project_instruction_dispatch_decision(
            store=self.controller.store,
            session_id=self.session.id,
            registry=registry,
            expected_state=self.session.project_instruction_state,
            expected_selection=selection,
            notice_key=notice,
            decision="proceed",
        )
        self._record(
            "fixture_setup",
            correlation,
            boundary="real_private_project_dispatch_consent",
            outcome=consent,
        )
        if consent != "proceed":
            raise RuntimeError("Actual private project dispatch consent refused")
        if shape == "raw-deny":
            runtime = getattr(self.app, "raw_cli_runtime", None)
            if (
                runtime is None
                or runtime.permitted is not True
                or runtime.arm().armed is not True
            ):
                raise RuntimeError("Actual raw unlock/arm refused")
            tool = "shell_exec"
            catalog_id = "raw_shell:shell_exec"
            command_prefix = "echo QA" + correlation + " "
            command = command_prefix + "R" * (
                16384 - len(command_prefix.encode("utf-8"))
            )
            if len(command.encode("utf-8")) != 16384:
                raise RuntimeError(
                    "Exact synthetic raw boundary fixture length mismatch"
                )
            current["fixture_command_bytes"] = 16384
            targets = [
                (
                    tool,
                    {
                        "command": command,
                        "shell": "powershell",
                        "initial_directory": str(self.root),
                        "timeout_seconds": 5,
                    },
                )
            ]
        else:
            tool = "fs_write" if shape == "large" else "fs_read"
            catalog_id = "local:" + tool
            targets = []
            for index in range(4 if shape == "batch" else 1):
                relative = "fixture-" + correlation + "-" + str(index) + ".txt"
                if shape == "large":
                    prefix = "Synthetic QA-" + correlation + "\n"
                    content = prefix + "L" * (1048576 - len(prefix.encode("utf-8")))
                    if len(content.encode("utf-8")) != 1048576:
                        raise RuntimeError(
                            "Exact1MiB synthetic fixture length mismatch"
                        )
                    current["fixture_payload_bytes"] = 1048576
                    args = {"path": relative, "content": content, "dry_run": True}
                else:
                    (self.root / relative).write_text(
                        "Synthetic private read fixture.\n", encoding="utf-8"
                    )
                    args = {"path": relative}
                targets.append((tool, args))
        self.provider.plan(
            [
                [("find_tools", {"query": tool})],
                [("load_tools", {"ids": [catalog_id]})],
                targets,
                "Private scripted planning complete.",
            ],
            correlation,
        )
        self._record("fixture_prepared", correlation, shape=shape)
        # Real accepted send, durability, config, runtime, discovery, tool
        # preparation and approval owners; no directly injected verdicts.
        self.worker = asyncio.create_task(
            self.controller.submit_draft(
                "QA-" + correlation[:12] + " inspect the private fixture.",
                session_id=self.session.id,
            )
        )
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            current = dict(self.current or {})
        result = {
            "current": current,
            "semantic_readiness_only": True,
            "paint_observed": False,
            "product_samples": 0,
            "driver": type(getattr(self.app, "_driver", None)).__name__,
            "size": [self.app.size.width, self.app.size.height],
            "theme": self.app.theme,
            "controls": [],
            "feedback": [],
            "injection_boundary": "owned scripted HTTP responses only; stock gateway/submit/AgentService/ConsoleToolActivity",
        }
        result["submission_task"] = None
        task = self.worker
        if task is not None:
            task_state = {
                "done": task.done(),
                "cancelled": task.cancelled(),
                "exception_type": None,
                "result": None,
            }
            if task.done() and not task.cancelled():
                exception = task.exception()
                if exception is not None:
                    task_state["exception_type"] = (
                        type(exception).__module__ + "." + type(exception).__qualname__
                    )
                else:
                    submitted = task.result()
                    status = getattr(submitted, "terminal_status", None)
                    blocker = getattr(submitted, "blocker", None)
                    task_state["result"] = {
                        "type": type(submitted).__module__
                        + "."
                        + type(submitted).__qualname__,
                        "accepted": getattr(submitted, "accepted", None),
                        "should_clear_draft": getattr(
                            submitted, "should_clear_draft", None
                        ),
                        "session_closed": getattr(submitted, "session_closed", None),
                        "provider_started": getattr(
                            submitted, "provider_started", None
                        ),
                        "terminal_status": getattr(status, "value", None),
                        "blocker": getattr(blocker, "value", None),
                    }
            elif not task.done():
                task_state["stack"] = [
                    {
                        "file": frame.f_code.co_filename,
                        "function": frame.f_code.co_name,
                        "line": frame.f_lineno,
                    }
                    for frame in task.get_stack(limit=12)
                ]
                try:
                    # Read-only await ownership chain. No locals, args, repr,
                    # Future private fields, sends, iteration or advancement.
                    chain = []
                    seen = set()
                    awaited = task.get_coro()
                    for _ in range(24):
                        if awaited is None or id(awaited) in seen:
                            break
                        seen.add(id(awaited))
                        frame = (
                            getattr(awaited, "cr_frame", None)
                            or getattr(awaited, "gi_frame", None)
                            or getattr(awaited, "ag_frame", None)
                        )
                        if frame is None:
                            chain.append(
                                {
                                    "opaque_type": type(awaited).__module__
                                    + "."
                                    + type(awaited).__qualname__
                                }
                            )
                            break
                        chain.append(
                            {
                                "file": frame.f_code.co_filename,
                                "qualname": frame.f_code.co_qualname,
                                "line": frame.f_lineno,
                            }
                        )
                        awaited = (
                            getattr(awaited, "cr_await", None)
                            or getattr(awaited, "gi_yieldfrom", None)
                            or getattr(awaited, "ag_await", None)
                        )
                    task_state["await_chain"] = chain
                except Exception as error:
                    task_state["await_chain_error_type"] = (
                        type(error).__module__ + "." + type(error).__qualname__
                    )
                    self._observation_failed("await_chain", error)
            result["submission_task"] = task_state
        if self.controller is None:
            return result
        result["run_status"] = self.controller.run_state.status.value
        result["actual_agent_runtime_enabled"] = self.controller._agent_runtime_enabled
        result["actual_chat_post_entries"] = self.provider.chat_post_entries
        try:
            card = self.app.screen.query_one("#chat-approval-card")
        except Exception:
            return result
        identity_reader = getattr(card, "feedback_identity", None)
        identity = identity_reader() if callable(identity_reader) else None
        identity_source = "feature_feedback_identity" if identity is not None else None
        if identity is None and card.display:
            actual_round = getattr(card, "_batch_round_id", None)
            actual_generation = getattr(card, "_batch_generation", None)
            if actual_round is not None and actual_generation is not None:
                identity = (actual_round, actual_generation)
                identity_source = "actual_round_and_batch_generation"
        result["card_identity_source"] = identity_source
        result["feature_presentation_revision"] = getattr(
            card, "_presentation_revision", None
        )
        result["feature_feedback_publisher_missing"] = not callable(
            getattr(self.controller, "_publish_approval_observation", None)
        )
        result.update(
            card_identity=list(identity) if identity else None,
            phase=getattr(card, "_batch_phase", None),
            card_display=card.display,
            card_attached=card.is_attached,
            information_ready=getattr(card, "_information_ready", None),
            submitted=getattr(card, "_batch_submitted", None),
        )
        visible = self.app.screen._compositor.visible_widgets

        def region(widget):
            if widget not in visible:
                return None
            bounds, clip = visible[widget]
            return {
                "region": [bounds.x, bounds.y, bounds.width, bounds.height],
                "clip": [clip.x, clip.y, clip.width, clip.height],
                "fully_visible": bounds.width > 0
                and bounds.height > 0
                and bounds.intersection(clip) == bounds,
            }

        result["card_region"] = region(card)
        from textual.widgets import Button

        for button in card.query(Button):
            bounds = region(button)
            if bounds is not None:
                result["controls"].append(
                    {
                        "id": button.id,
                        "disabled": button.disabled,
                        "display": button.display,
                        "focused": button.has_focus,
                        **bounds,
                    }
                )
        # Read only an ALREADY CREATED store; never force lazy first use.
        feedback = self.controller.__dict__.get("_approval_feedback")
        if feedback is not None and current.get("run_id"):
            for fact in feedback.snapshot(self.session.id, current["run_id"]):
                result["feedback"].append(
                    {
                        key: getattr(fact, key)
                        for key in (
                            "sequence",
                            "decision_state",
                            "selected_scope",
                            "grant_state",
                            "applied_scope",
                            "execution_state",
                            "terminal_outcome",
                            "confirmed_backend_start",
                        )
                    }
                )
        with self._lock:
            result["stages"] = [
                dict(row)
                for row in self.events
                if row["correlation"] == current.get("correlation")
            ]
            result["owner_observations"] = [
                dict(row)
                for row in self._owner_observations
                if row["correlation"] == current.get("correlation")
            ]
        return result

    async def collect(self):
        if self.worker is None:
            raise RuntimeError("No actual submission staged")
        submitted = await asyncio.wait_for(asyncio.shield(self.worker), timeout=35)
        if not submitted.accepted:
            result = self.snapshot()
            result["submission_refused"] = True
            result["actual_model_api_calls"] = 0
            return result
        deadline = time.monotonic() + 35
        while self.controller.run_state.status.value in {
            "validating",
            "streaming",
            "checking_citations",
            "retrying",
        }:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Actual runtime did not settle; missing sample retained"
                )
            await asyncio.sleep(0.02)
        result = self.snapshot()
        result["actual_model_api_calls"] = 0
        result["scripted_planning_turns"] = self.provider.scripted_turns
        decisions = result["current"].get("actual_decisions")
        if decisions is None:
            result["protocol_error"] = "No actual approval round reached"
        if (
            result["current"]["shape"] == "raw-deny"
            and decisions is not None
            and any(value != "deny" for value in decisions.values())
        ):
            result["protocol_error"] = (
                "Raw fixture received a non-Deny choice; actual runtime facts must be reviewed"
            )
        return result

    async def trace_admission_negatives(self):
        """Dedicated control process only; never precede a cold metric sample."""
        if self.current is not None or any(self._sample_count.values()):
            raise RuntimeError("Use a separate fresh negative-control process")
        from tldw_chatbook.Chat.console_provider_gateway import (
            ConsoleProviderSelection,
            TemporaryCaptureRequiresSave,
            ConsoleTraceCaptureMode,
            ConsoleRequestRoute,
            TraceCallPersistenceError,
            TraceProvenanceAlignmentError,
        )

        selection = ConsoleProviderSelection(
            provider=self.provider.provider_key,
            base_url=self.provider.base_url,
            explicit_model=self.provider.model,
        )
        resolution = await self.gateway.resolve_for_send(selection)
        if not resolution.ready:
            raise RuntimeError("Actual owned HTTP readiness refused")
        before = self.provider.scripted_turns
        post_before = self.provider.chat_post_entries
        pending_before = self.provider.pending_count()
        results = []
        self._trace_negative_results = results
        for label, ephemeral, expected in [
            ("temporary_capture_on", True, (TemporaryCaptureRequiresSave,)),
            (
                "durable_capture_on_missing_provenance",
                False,
                (TraceCallPersistenceError, TraceProvenanceAlignmentError),
            ),
        ]:
            caught = None
            try:
                async for _ in self.gateway.stream_chat(
                    resolution,
                    [{"role": "user", "content": "Private negative control."}],
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                    ephemeral=ephemeral,
                    route=ConsoleRequestRoute.FRESH,
                ):
                    pass
            except expected as exc:
                caught = type(exc).__name__
            if caught is None:
                raise RuntimeError(
                    "Actual trace admission negative failed closed expectation: "
                    + label
                )
            results.append(
                {
                    "control": label,
                    "actual_exception": caught,
                    "scripted_turns_consumed": self.provider.scripted_turns - before,
                    "chat_post_entry_delta": self.provider.chat_post_entries
                    - post_before,
                }
            )
            if self.provider.chat_post_entries != post_before:
                raise RuntimeError("Trace refusal reached actual HTTP adapter entry")
        if (
            self.provider.scripted_turns != before
            or self.provider.pending_count() != pending_before
        ):
            raise RuntimeError("Trace refusal reached private provider adapter")
        return {
            "qualified": False,
            "product_samples": 0,
            "trace_admission_negatives": results,
            "adapter_turns_consumed": 0,
            "chat_post_entry_delta": 0,
        }

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self.controller is not None:
            await self.controller.shutdown()
            for name, original in self._original_methods.items():
                setattr(self.controller, name, original)
        if self._bridge is not None and self._original_bridge_run_reply is not None:
            self._bridge.run_reply = self._original_bridge_run_reply
        if self.worker is not None and not self.worker.done():
            await asyncio.wait_for(asyncio.shield(self.worker), timeout=5)
        self.provider.fixture = None
        await asyncio.to_thread(self.provider.close)

    async def cleanup(self):
        return await self.close()
