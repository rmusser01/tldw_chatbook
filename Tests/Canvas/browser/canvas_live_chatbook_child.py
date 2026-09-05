"""Actual TldwCli child driven by a deterministic local provider double."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from textual.widgets import Button

from Tests.UI.app_factory import (
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook import config as config_module
from tldw_chatbook.Agents.agent_models import FENCE_TOOL_RESULT_PREFIX
from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_canvas_card import ConsoleCanvasCard


def _document(version: str) -> str:
    return (
        "<!doctype html><html><body>"
        '<h1 id="chatbook-app-canvas">CHATBOOK_APP_CANVAS</h1>'
        f'<p id="chatbook-app-revision">{version}</p>'
        "</body></html>"
    )


class _ScriptedCanvasGateway:
    """Replay two genuine agent/tool cycles without contacting a provider."""

    def __init__(self) -> None:
        self.calls = 0
        self._run_phase = "initial"
        self._discovery = False
        self._canvas_id: str | None = None
        self._revision_id: str | None = None
        self._call_count_path = (
            Path(os.environ["XDG_DATA_HOME"]) / "canvas-live-gateway-calls"
        )
        self._tool_status_path = (
            Path(os.environ["XDG_DATA_HOME"]) / "canvas-live-tool-status"
        )
        self._disclosure_path = (
            Path(os.environ["XDG_DATA_HOME"]) / "canvas-live-tool-disclosure"
        )
        self._call_count_path.write_text("0", encoding="ascii")
        self._tool_status_path.write_text("pending", encoding="ascii")
        self._disclosure_path.write_text("pending", encoding="ascii")

    async def resolve_for_send(self, selection):
        resolution = SimpleNamespace(
            provider=selection.provider,
            base_url=selection.base_url or "http://127.0.0.1:9099",
            model=selection.explicit_model
            or selection.configured_model
            or "canvas-live-model",
            ready=True,
            visible_copy="",
        )
        resolution.resolved_destination = resolve_console_destination(resolution)
        return resolution

    @staticmethod
    def _json_content(content: str):
        candidate = content
        if content.startswith(FENCE_TOOL_RESULT_PREFIX):
            separator = content.find(": ")
            candidate = content[separator + 2 :] if separator >= 0 else ""
        return json.loads(candidate)

    def _latest_canvas(self, messages) -> tuple[str, str]:
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            try:
                value = self._json_content(content)
            except json.JSONDecodeError:
                continue
            canvas = value.get("canvas") if isinstance(value, dict) else None
            if (
                isinstance(canvas, dict)
                and isinstance(canvas.get("canvas_id"), str)
                and isinstance(canvas.get("revision_id"), str)
            ):
                return canvas["canvas_id"], canvas["revision_id"]
        raise RuntimeError("canvas_tool_result_missing")

    def _tool_result_status(self, messages, name: str) -> str:
        prefix = f"{FENCE_TOOL_RESULT_PREFIX}{name}: "
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.startswith(prefix):
                continue
            candidate = content[len(prefix) :]
            if candidate.startswith("ERROR: "):
                try:
                    error = json.loads(candidate.removeprefix("ERROR: "))
                except json.JSONDecodeError:
                    return "error:nonjson"
                return f"error:{error.get('code', 'unknown')}"
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                return "result:nonjson"
            return (
                "staged"
                if isinstance(value, dict)
                and value.get("status") == "staged"
                and isinstance(value.get("canvas"), dict)
                else "result:unexpected"
            )
        return "missing"

    @staticmethod
    def _system_prompt(messages) -> str:
        return "\n".join(
            str(message.get("content"))
            for message in messages
            if isinstance(message, Mapping)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
        )

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.calls += 1
        self._call_count_path.write_text(str(self.calls), encoding="ascii")
        system_prompt = self._system_prompt(messages)
        if self._run_phase in {"initial", "update_initial"}:
            self._discovery = "use find_tools, then load_tools" in system_prompt
            self._disclosure_path.write_text(
                "mode="
                + ("discovery" if self._discovery else "direct")
                + f";find_tools={'find_tools' in system_prompt}"
                + f";load_tools={'load_tools' in system_prompt}"
                + f";canvas_create={'canvas_create' in system_prompt}",
                encoding="ascii",
            )
            if self._discovery:
                self._run_phase = (
                    "update_find" if self._run_phase == "update_initial" else "find"
                )
                yield '```tool_call\n{"name":"find_tools","arguments":{"query":"canvas"}}\n```'
                return
            self._run_phase = (
                "update_create" if self._run_phase == "update_initial" else "create"
            )

        if self._run_phase in {"find", "update_find"}:
            self._run_phase = (
                "update_load" if self._run_phase == "update_find" else "load"
            )
            yield (
                '```tool_call\n{"name":"load_tools","arguments":{"ids":'
                '["canvas:canvas_create","canvas:canvas_update"]}}\n```'
            )
            return

        if self._run_phase in {"load", "update_load"}:
            self._run_phase = (
                "update_create" if self._run_phase == "update_load" else "create"
            )

        if self._run_phase == "create":
            arguments = {
                "title": "Actual Chatbook Canvas",
                "html": _document("v1"),
            }
            self._run_phase = "create_result"
            yield (
                "```tool_call\n"
                + json.dumps({"name": "canvas_create", "arguments": arguments})
                + "\n```"
            )
            return

        if self._run_phase == "update_create":
            if self._canvas_id is None or self._revision_id is None:
                raise RuntimeError("canvas_tool_result_missing")
            arguments = {
                "canvas_id": self._canvas_id,
                "expected_parent_revision_id": self._revision_id,
                "html": _document("v2"),
            }
            self._run_phase = "update_result"
            yield (
                "```tool_call\n"
                + json.dumps({"name": "canvas_update", "arguments": arguments})
                + "\n```"
            )
            return

        if self._run_phase == "create_result":
            self._canvas_id, self._revision_id = self._latest_canvas(messages)
            status = "missing"
            roles: list[str] = []
            for message in reversed(messages):
                if isinstance(message, Mapping):
                    roles.append(str(message.get("role", "unknown")))
                    content = message.get("content")
                else:
                    roles.append(type(message).__name__)
                    content = None
                if not isinstance(content, str):
                    continue
                try:
                    value = self._json_content(content)
                except json.JSONDecodeError:
                    if content.startswith(FENCE_TOOL_RESULT_PREFIX):
                        lowered = content.lower()
                        labels = [
                            label
                            for label in (
                                "unknown",
                                "unavailable",
                                "not found",
                                "scope",
                                "conversation",
                                "session",
                                "authority",
                                "disabled",
                                "temporary",
                                "loaded",
                                "registered",
                                "allowed",
                                "approval",
                                "invalid",
                                "failed",
                            )
                            if label in lowered
                        ]
                        status = (
                            "tool_result_error:" + ",".join(labels or ["other"])
                            if ": error:" in lowered
                            else "tool_result_nonjson"
                        )
                        break
                    continue
                if isinstance(value, dict):
                    status = (
                        "canvas"
                        if isinstance(value.get("canvas"), dict)
                        else str(value.get("error", "non_canvas"))
                    )
                    break
            if status == "missing":
                status = "roles:" + ",".join(roles)
            self._tool_status_path.write_text(status, encoding="utf-8")
            self._run_phase = "update_initial"
            yield "CHATBOOK_CANVAS_CREATED"
            return

        if self._run_phase == "update_result":
            self._tool_status_path.write_text(
                "canvas_create," + self._tool_result_status(messages, "canvas_update"),
                encoding="ascii",
            )
            self._run_phase = "complete"
            yield "CHATBOOK_CANVAS_UPDATED"
            return
        yield "CHATBOOK_CANVAS_UPDATED"


def main() -> None:
    app = None
    database = None
    try:
        app = _build_test_app(configured_default="chat")
        data_root = Path(os.environ["XDG_DATA_HOME"])
        data_root.mkdir(parents=True, exist_ok=True)
        database = CharactersRAGDB(
            data_root / "canvas-live-chatbook.sqlite", "canvas-live-chatbook"
        )
        app.chachanotes_db = database
        # The app factory initially wires services with no DB. Bind the normal
        # saved-conversation reader to this same owned durable database as well.
        app._wire_chat_conversation_services()
        _configure_native_ready_console(app, model="gpt-4o")
        if not config_module.save_settings_to_cli_config(
            {
                "first_run": {"setup_completed": True},
                "model_catalog": {
                    "auto_refresh_enabled": False,
                    "refresh_consent_recorded": True,
                },
            }
        ):
            raise RuntimeError("canvas_live_first_run_config_failed")
        app.app_config = config_module.load_settings(force_reload=True)
        app.chat_api_provider_value = "llama_cpp"
        app.chat_api_model_value = "gpt-4o"
        gateway = _ScriptedCanvasGateway()
        app.console_provider_gateway_factory = lambda: gateway
        recovered_root_revision = None

        async def load_saved_conversation():
            nonlocal recovered_root_revision
            path = data_root / "canvas-live-chatbook.sqlite"
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as saved:
                roots = saved.execute(
                    "SELECT d.conversation_id, r.id FROM canvas_documents d "
                    "JOIN canvas_revisions r ON r.canvas_id=d.id "
                    "WHERE d.deleted_at IS NULL AND r.deleted_at IS NULL AND r.sequence=1"
                ).fetchall()
            if len(roots) != 1:
                raise RuntimeError("expected one owned saved Canvas root")
            conversation_id, recovered_root_revision = roots[0]
            loaded = await app.screen._workspace.open_console_workspace_conversation(
                conversation_id
            )
            (data_root / "canvas-live-saved-loaded").write_text(
                "loaded-without-provider"
                if loaded is True and gateway.calls == 0
                else "provider-called"
                if gateway.calls
                else "load-false"
                if loaded is False
                else "load-none",
                encoding="ascii",
            )

        app.action_canvas_fixture_load_saved = load_saved_conversation
        app._bindings.bind("f10", "canvas_fixture_load_saved", priority=True)

        def reopen_exact_created_card():
            # Test keyboard adapter presses the real transcript-card button;
            # routing, selected revision and authority remain production-owned.
            target_revision = recovered_root_revision or gateway._revision_id
            card = next(
                card
                for card in app.screen.query(ConsoleCanvasCard)
                if card.presentation.revision_id == target_revision
            )
            screen = app.screen
            original_open = screen._open_console_canvas_selection

            def acknowledge_applied_selection():
                handler = app.served_canvas_handler
                scope = handler.scope
                exact = scope is not None and scope.revision_id == target_revision
                pinned = (
                    exact and not handler._authority.describe_selection(scope).following
                )
                if recovered_root_revision is not None:
                    (data_root / "canvas-live-restored-provider-calls").write_text(
                        str(gateway.calls), encoding="ascii"
                    )
                (data_root / "canvas-live-card-pressed").write_text(
                    "selected-pinned" if pinned else "selection-not-applied",
                    encoding="ascii",
                )

            async def observe_open_completion(**kwargs):
                screen._open_console_canvas_selection = original_open
                result = await original_open(**kwargs)
                # The real card handler has no further await after this call;
                # acknowledge on the next refresh, after its dispatch returns.
                app.call_after_refresh(acknowledge_applied_selection)
                return result

            screen._open_console_canvas_selection = observe_open_completion
            card.query_one("Button", Button).press()

        app.action_canvas_fixture_reopen = reopen_exact_created_card
        app._bindings.bind("f12", "canvas_fixture_reopen", priority=True)

        def acknowledge_composer_focus():
            focused = app.focused
            while focused is not None and focused.id != "console-native-composer":
                focused = focused.parent
            (data_root / "canvas-live-composer-focused").write_text(
                "focused" if focused is not None else "other", encoding="ascii"
            )

        app.action_canvas_fixture_focus_ack = acknowledge_composer_focus
        app._bindings.bind("f11", "canvas_fixture_focus_ack", priority=True)
        app.run()
    finally:
        if database is not None:
            database.close()
        drain_active_service_patches()
        drain_created_dirs()


if __name__ == "__main__":
    main()
