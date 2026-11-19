"""Shared PRIVATE real-TldwCli approval fixture. Source-only preparation.

No model/provider transport, tool dispatch, fabricated paint, Pilot gestures,
Button.press, or direct resolver calls. The caller owns private bootstrap,
source freeze, transport, physical input, independent frame observation and
clock calibration. Snapshot readiness is semantic ONLY, never a paint fact.
All requests are fixture-generated, pass real provider pending-gate checks,
and settle through the existing controller/UI/InterruptRoundHost owners.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import secrets
import threading
import time

SHAPES = ("single", "batch", "raw-deny", "large")
UNAVAILABLE = (
    "provider_argument_generation",
    "provider_response",
    "model_continuation",
    "tool_dispatch",
    "backend_started",
)


class ApprovalFixtureDriver:
    def __init__(self, app, private_root, emit=None):
        self.app = app
        self.root = Path(private_root).resolve()
        self.emit = emit
        self.controller = None
        self.session = None
        self.worker = None
        self.current = None
        self.events = []
        self._lock = threading.Lock()
        self._correlations = set()
        self._sample_count = {shape: 0 for shape in SHAPES}
        self._original_publish = None

    def new_correlation(self):
        value = secrets.token_hex(16)
        self._correlations.add(value)
        return value

    def _record(self, stage, correlation, *, timestamp_ns=None, **facts):
        # Only code-owned stage/fact values; no args, paths, text, or commands.
        row = {
            "stage": stage,
            "correlation": correlation,
            "clock": "python_perf_counter",
            "timestamp_ns": timestamp_ns
            if timestamp_ns is not None
            else time.perf_counter_ns(),
            **facts,
        }
        with self._lock:
            if len(self.events) >= 4096:
                raise RuntimeError("Bounded fixture stage capacity exceeded")
            self.events.append(row)
        if self.emit is not None:
            try:
                self.emit(dict(row))
            except Exception:
                pass  # Observational sink never controls decisions.

    async def attach(self):
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        if not isinstance(self.app, TldwCli):
            raise RuntimeError("Actual TldwCli required")
        deadline = time.monotonic() + 35
        while not getattr(self.app, "_ui_ready", False):
            if time.monotonic() > deadline:
                raise RuntimeError("Actual app startup not ready; no guard bypass")
            await asyncio.sleep(0.02)
        if not self.app._instance_lock_status.acquired:
            raise RuntimeError("Actual private app instance lock not acquired")
        await self.app.handle_screen_navigation(NavigateToScreen("chat"))
        deadline = time.monotonic() + 20
        while getattr(self.app.screen, "screen_name", None) != "chat":
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Console blocked by startup/navigation/modal; preserve it"
                )
            await asyncio.sleep(0.02)
        self.controller = self.app.screen._ensure_console_chat_controller()
        self.session = self.controller.new_session(
            title="Private approval qualification", ephemeral=True
        )
        self.root.mkdir(parents=True, exist_ok=True)
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            raise RuntimeError("Real app MCP permission service unavailable")
        service.set_global_default("ask")  # Actual writer, disposable private profile.
        original = self.controller.approval_feedback.publish
        self._original_publish = original

        def observe(observation):
            answer = original(observation)  # Original reducer and return contract.
            current = self.current
            if current and observation.identity.run_id == current["run_id"]:
                if observation.kind == "received":
                    self._record("controller_receipt", current["correlation"])
                elif observation.kind == "settled":
                    self._record(
                        "host_outcome",
                        current["correlation"],
                        outcome=observation.outcome,
                    )
            return answer

        self.controller.approval_feedback.publish = observe
        return self.snapshot()

    async def configure(self, *, theme, inspect_open):
        if theme not in {"textual-dark", "textual-light"}:
            raise ValueError("Unsupported controlled theme")
        self.app.theme = theme
        self.app.screen._set_console_rail_preference(right_open=bool(inspect_open))
        await asyncio.sleep(0)
        return self.snapshot()

    async def stage(self, shape, correlation):
        if shape not in SHAPES or correlation not in self._correlations:
            raise ValueError("Fixture-owned shape/correlation required")
        if self.worker is not None and not self.worker.done():
            raise RuntimeError("Previous real approval is still pending")
        if self.controller is None:
            raise RuntimeError("Attach to real Console first")
        first_use = self._sample_count[shape] == 0
        self._sample_count[shape] += 1
        run_id = "qa-approval-" + correlation
        self.current = {
            "shape": shape,
            "correlation": correlation,
            "run_id": run_id,
            "first_use": first_use,
            "unavailable": list(UNAVAILABLE),
        }
        self._record("request_complete", correlation, shape=shape, first_use=first_use)
        # Compose through the actual controller and use its exact current guards.
        # The trusted fixture root is explicitly supplied just as in the existing
        # native fs_read journey; no root guard/stat identity is replaced.
        from tldw_chatbook.Chat.console_chat_controller import (
            _capture_project_root_identity,
        )

        identity = _capture_project_root_identity(self.root)
        if identity is None:
            raise RuntimeError("Real fixture-root identity refused")
        if shape == "raw-deny":
            runtime = getattr(self.app, "raw_cli_runtime", None)
            if (
                runtime is None
                or runtime.permitted is not True
                or runtime.arm().armed is not True
            ):
                raise RuntimeError(
                    "Actual raw permission/arm gate refused; no synthetic raw row"
                )
            provider, _ = self.controller._compose_raw_shell_provider(
                self.session.id, project_root=self.root
            )
            if provider is None:
                raise RuntimeError("Actual raw provider unavailable")
        else:
            provider, _ = self.controller._compose_local_provider(
                self.session.id,
                project_root=self.root,
                allow_write=True,
                project_root_identity=identity,
            )
            if provider is None:
                raise RuntimeError("Actual local provider unavailable")

        def request():
            from tldw_chatbook.Agents.agent_models import ToolCall
            from tldw_chatbook.Agents.run_context import use_run_id

            with use_run_id(run_id):
                if shape == "raw-deny":
                    # Validated/displayed by the real raw owner; NEVER invoked.
                    calls = [
                        ToolCall(
                            "shell_exec",
                            {
                                "command": "echo PRIVATE_QUALIFICATION_ONLY",
                                "shell": "powershell",
                                "initial_directory": str(self.root),
                                "timeout_seconds": 5,
                            },
                            call_id="qa-" + correlation,
                        )
                    ]
                    pending = [provider.pending_gate_for(calls[0])]
                else:
                    count = 4 if shape == "batch" else 1
                    calls = []
                    for index in range(count):
                        # Internally minted suffix is independently visible for
                        # real-frame currentness. Never copy args into diagnostics.
                        relative = "fixture-" + correlation + "-" + str(index) + ".txt"
                        if shape == "large":
                            args = {
                                "path": relative,
                                "content": "Synthetic private payload.\n" * 4096,
                                "dry_run": True,
                            }
                            name = "fs_write"
                        else:
                            (self.root / relative).write_text(
                                "Synthetic private read fixture.\n", encoding="utf-8"
                            )
                            args = {"path": relative}
                            name = "fs_read"
                        calls.append(
                            ToolCall(
                                name,
                                args,
                                call_id="qa-" + correlation + "-" + str(index),
                            )
                        )
                    pending = [
                        provider.pending_gate_for(
                            call.name, call.args, call.call_id, run_id=run_id
                        )
                        for call in calls
                    ]
                if any(row is None for row in pending):
                    raise RuntimeError(
                        "Real owner did not return every required Ask gate"
                    )
                self._record(
                    "local_checks_complete", correlation, call_count=len(pending)
                )
                decisions = self.controller.request_mcp_approvals(
                    pending, session_id=self.session.id
                )
                self._record(
                    "worker_released", correlation, decision_count=len(decisions)
                )
                return decisions

        self.worker = asyncio.create_task(asyncio.to_thread(request))
        return dict(self.current)

    def snapshot(self):
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
        }
        if self.controller is None:
            return result
        try:
            card = self.app.screen.query_one("#chat-approval-card")
        except Exception:
            return result
        identity = card.feedback_identity()
        result.update(
            card_identity=list(identity) if identity else None,
            phase=card._batch_phase,
            card_display=card.display,
            card_attached=card.is_attached,
            information_ready=card._information_ready,
            submitted=card._batch_submitted,
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
            if bounds is None:
                continue
            result["controls"].append(
                {
                    "id": button.id,
                    "disabled": button.disabled,
                    "display": button.display,
                    "focused": button.has_focus,
                    **bounds,
                }
            )
        if current:
            for fact in self.controller.approval_feedback.snapshot(
                self.session.id, current["run_id"]
            ):
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
        return result

    async def collect(self):
        if self.worker is None:
            raise RuntimeError("No real round staged")
        decisions = await asyncio.wait_for(asyncio.shield(self.worker), timeout=5)
        result = self.snapshot()
        result["actual_decisions"] = dict(decisions)
        if self.current["shape"] == "raw-deny" and any(
            value != "deny" for value in decisions.values()
        ):
            result["protocol_error"] = (
                "Raw-deny fixture received a non-Deny gesture; no shell was dispatched"
            )
        result["tool_dispatches"] = 0
        result["model_calls"] = 0
        return result

    async def close(self):
        if self.controller is not None:
            # Actual owner shutdown/cancellation; no direct fabricated verdict.
            await self.controller.shutdown()
            if self._original_publish is not None:
                self.controller.approval_feedback.publish = self._original_publish
        if self.worker is not None and not self.worker.done():
            await asyncio.wait_for(asyncio.shield(self.worker), timeout=5)
