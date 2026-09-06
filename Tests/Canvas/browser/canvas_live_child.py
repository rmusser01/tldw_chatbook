"""Deterministic subprocess used by the served Canvas outer-path test."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Canvas.control_protocol import CanvasControlClient
from tldw_chatbook.Canvas.gateway import ServedCanvasControlHandler
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController


def _document(marker: str, version: str) -> str:
    return (
        "<!doctype html><html><head>"
        f"<title>Live {marker}</title></head><body>"
        f'<h1 id="profile-identity">{marker}</h1>'
        f'<p id="revision-marker">{version}</p>'
        '<button id="submit-result">Send result</button>'
        '<button id="download-result">Download result</button>'
        "<script>"
        "document.getElementById('submit-result').addEventListener('click',()=>"
        f"canvas.submit({{profile:'{marker}',revision:'{version}'}}));"
        "document.getElementById('download-result').addEventListener('click',()=>"
        f"canvas.download({{filename:'{marker}.txt',mime_type:'text/plain',"
        f"data:'{marker}:{version}'}}));"
        "</script></body></html>"
    )


class ChildRuntime:
    def __init__(
        self, client: CanvasControlClient, handler: ServedCanvasControlHandler
    ) -> None:
        self.client = client
        self.marker = f"child-{client.child_id[-8:]}"
        self.session_id = f"session-{client.child_id}"
        self.controller = ConsoleCanvasController()
        self.controller.activate_session(self.session_id)
        self.handler = handler
        self.scope = CanvasScope(
            session_id=self.session_id,
            conversation_id=self.session_id,
            active_message_ids=("user-root", "assistant-create"),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="run-create",
        )
        self.authority = NativeConsoleCanvasAuthority(
            scope_resolver=self._resolve_scope,
            canvas_controller=self.controller,
            bridge_prepare=lambda _target: self._drafts.append,
            auto_open=self._bind_created,
        )
        self.controller.add_settlement_listener(
            self.authority.on_settlement_publication
        )
        self._drafts: list[str] = []
        self.root_revision_id = ""
        self.canvas_id = ""

    def _resolve_scope(self, requested: str) -> CanvasScope:
        if requested != self.session_id:
            raise RuntimeError("canvas_scope_unavailable")
        return self.scope

    def _bind_created(self, _session_id: str, info) -> None:
        scope = self.authority.gateway_scope(
            session_id=self.session_id,
            browser_session_id=self.client.child_id,
            canvas_id=info.canvas_id,
            revision_id=info.revision_id,
            follow_latest=True,
        )
        self.handler.bind(self.authority, scope)

    def _invoke_create(self) -> None:
        run = self.controller.register_run(
            self.scope, assistant_message_id="assistant-create", temporary=True
        )
        provider = CanvasToolProvider(run, scope=self.scope)
        with use_run_id(self.scope.run_id), use_tool_call_id("tool-create"):
            result = provider.invoke(
                "canvas:canvas_create",
                {"title": f"Live {self.marker}", "html": _document(self.marker, "v1")},
            )
        if not result.ok:
            raise RuntimeError("canvas_create_failed")
        settlement = run.finish_assistant_run(
            "assistant-create", actual_run_id=self.scope.run_id, terminal_status="done"
        )
        if settlement is None or not self.controller.confirm_exact_settlement(
            settlement
        ):
            raise RuntimeError("canvas_create_settlement_failed")
        selected = self.handler.scope
        if selected is None:
            raise RuntimeError("canvas_create_not_bound")
        self.canvas_id = selected.canvas_id
        self.root_revision_id = selected.revision_id
        self.scope = replace(
            self.scope,
            selected_canvas_id=self.canvas_id,
            selected_revision_id=self.root_revision_id,
        )

    def update(
        self,
        *,
        branch: bool = False,
        source: str | None = None,
        suffix: str | None = None,
    ) -> bool:
        previous_scope = self.scope
        suffix = suffix or ("branch" if branch else "v2")
        assistant = f"assistant-{suffix}"
        parent = self.root_revision_id if branch else self.scope.selected_revision_id
        active = (
            ("user-root", "assistant-create", assistant)
            if branch
            else (*self.scope.active_message_ids, assistant)
        )
        next_scope = replace(
            self.scope,
            active_message_ids=active,
            selected_canvas_id=self.canvas_id,
            selected_revision_id=parent,
            run_id=f"run-{suffix}",
        )
        run = self.controller.register_run(
            next_scope, assistant_message_id=assistant, temporary=True
        )
        self.scope = next_scope
        provider = CanvasToolProvider(run, scope=next_scope)
        with use_run_id(next_scope.run_id), use_tool_call_id(f"tool-{suffix}"):
            result = provider.invoke(
                "canvas:canvas_update",
                {
                    "canvas_id": self.canvas_id,
                    "expected_parent_revision_id": parent,
                    "html": source or _document(self.marker, suffix),
                },
            )
        if not result.ok:
            run.finish_assistant_run(
                assistant,
                actual_run_id=next_scope.run_id,
                terminal_status="failed",
            )
            self.scope = previous_scope
            return False
        settlement = run.finish_assistant_run(
            assistant, actual_run_id=next_scope.run_id, terminal_status="done"
        )
        if settlement is None or not self.controller.confirm_exact_settlement(
            settlement
        ):
            raise RuntimeError("canvas_update_settlement_failed")
        selected = self.authority.control_scope_snapshot(self.handler.scope)
        self.scope = replace(
            next_scope,
            selected_canvas_id=selected.selected_canvas_id,
            selected_revision_id=selected.selected_revision_id,
        )
        return True

    def reopen_root(self) -> None:
        scope = self.authority.gateway_scope(
            session_id=self.session_id,
            browser_session_id=self.client.child_id,
            canvas_id=self.canvas_id,
            revision_id=self.root_revision_id,
            follow_latest=False,
        )
        self.handler.bind(self.authority, scope)
        self.scope = replace(
            self.scope,
            selected_canvas_id=self.canvas_id,
            selected_revision_id=self.root_revision_id,
        )


class CanvasLiveApp(App[None]):
    """Minimal real Textual child with deterministic Canvas tool commands."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("u", "live_update", "Update", priority=True),
        Binding("b", "live_branch", "Branch", priority=True),
        Binding("r", "live_reopen", "Reopen root", priority=True),
        Binding("p", "live_ping", "Ping", priority=True),
        Binding("n", "live_next_adversarial", "Next adversarial", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        handler = ServedCanvasControlHandler()
        client = CanvasControlClient.from_environment(
            os.environ, handler=handler.handle
        )
        if client is None:
            raise RuntimeError("served_control_environment_required")
        self.runtime = ChildRuntime(client, handler)
        start_index = os.environ.get("TLDW_CANVAS_ADVERSARIAL_START_INDEX", "0")
        if not start_index.isdecimal():
            raise RuntimeError("invalid_adversarial_start_index")
        self._adversarial_index = int(start_index) - 1
        self._adversarial_cases: list[dict[str, object]] = []
        probe_origin = os.environ.get("TLDW_CANVAS_EGRESS_PROBE_ORIGIN")
        if probe_origin:
            fixture = Path(__file__).with_name("fixtures") / "adversarial_scripts.json"
            cases = json.loads(fixture.read_text(encoding="utf-8"))
            websocket = probe_origin.replace("http://", "ws://", 1)
            self._adversarial_cases = [
                {
                    **case,
                    "script": str(case["script"])
                    .replace("__EGRESS__", probe_origin)
                    .replace("__WEBSOCKET__", websocket),
                }
                for case in cases
            ]
            canary_path = os.environ.get("TLDW_CANVAS_SAME_ORIGIN_CANARY_PATH")
            if canary_path:
                self._adversarial_cases.append(
                    {
                        "name": "same_origin_relative_request",
                        "script": (
                            f"try {{ fetch('{canary_path}'); }} catch (_error) {{}}"
                        ),
                        "expected": "ready",
                    }
                )
            if not 0 <= self._adversarial_index + 1 < len(self._adversarial_cases):
                raise RuntimeError("invalid_adversarial_start_index")

    def compose(self) -> ComposeResult:
        yield Static(
            "Canvas live child · U update · B branch · R reopen", id="child-status"
        )

    async def on_mount(self) -> None:
        await self.runtime.client.start()
        self.runtime._invoke_create()

    def action_live_update(self) -> None:
        self.runtime.update()
        self.query_one("#child-status", Static).update("CANVAS_LIVE_UPDATED")

    def action_live_branch(self) -> None:
        self.runtime.update(branch=True)
        self.query_one("#child-status", Static).update("CANVAS_LIVE_BRANCHED")

    def action_live_reopen(self) -> None:
        self.runtime.reopen_root()
        self.query_one("#child-status", Static).update("CANVAS_LIVE_REOPENED")

    def action_live_ping(self) -> None:
        self.query_one("#child-status", Static).update("CANVAS_LIVE_ACK")

    def action_live_next_adversarial(self) -> None:
        self._adversarial_index += 1
        case = self._adversarial_cases[self._adversarial_index]
        script = str(case["script"])
        clobbering_ids = "".join(
            f'<span id="{name}">{name}</span>'
            for name in ("fetch", "location", "parent", "postMessage", "Worker")
        )
        source = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>attack</title></head><body>"
            f'<h1 id="adversarial-marker">{self._adversarial_index}</h1>'
            f"{clobbering_ids}"
            '<button id="attack" type="button">attack</button>'
            '<output id="status">static</output>'
            f"<script>{script}</script></body></html>"
        )
        accepted = self.runtime.update(
            source=source,
            suffix=f"adversarial-{self._adversarial_index}",
        )
        self.query_one("#child-status", Static).update(
            f"CANVAS_LIVE_{'ADVERSARIAL' if accepted else 'REJECTED'}_"
            f"{self._adversarial_index}"
        )

    async def on_unmount(self) -> None:
        await self.runtime.client.aclose()
        self.runtime.authority.dispose()
        self.runtime.controller.close_runtime()


if __name__ == "__main__":
    CanvasLiveApp().run()
