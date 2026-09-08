"""Actual TldwCli child driven by a deterministic local provider double."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


async def _wait_for_exact_canvas_card(
    screen: Any,
    *,
    target_revision: str | None,
    attempts: int,
    interval: float,
) -> tuple[ConsoleCanvasCard, Button]:
    """Wait for the exact mounted card action in the current Console session."""

    cards: list[ConsoleCanvasCard] = []
    revision_matches: list[ConsoleCanvasCard] = []
    mounted_matches: list[ConsoleCanvasCard] = []
    session_matches: list[ConsoleCanvasCard] = []
    exact_pairs: list[tuple[ConsoleCanvasCard, Button]] = []
    enabled_pairs: list[tuple[ConsoleCanvasCard, Button]] = []
    store = None
    active_session = None
    for attempt in range(attempts):
        store = getattr(screen, "_console_chat_store", None)
        active_session = getattr(store, "active_session_id", None)
        cards = list(screen.query(ConsoleCanvasCard))
        revision_matches = [
            card for card in cards if card.presentation.revision_id == target_revision
        ]
        if type(target_revision) is not str or not target_revision:
            revision_matches = []
        mounted_matches = [card for card in revision_matches if card.is_mounted]
        session_matches = (
            [card for card in mounted_matches if card.session_id == active_session]
            if type(active_session) is str and active_session
            else []
        )
        exact_pairs = [
            (card, button)
            for card in session_matches
            for button in card.query(Button)
            if button.id == f"canvas-open-revision-{card._id_suffix}"
        ]
        enabled_pairs = [
            (card, button)
            for card, button in exact_pairs
            if button.is_mounted and not button.disabled
        ]
        if enabled_pairs:
            return enabled_pairs[0]
        if attempt + 1 < attempts:
            await asyncio.sleep(interval)

    raise RuntimeError(
        "canvas_card_not_ready("
        f"cards={min(len(cards), 32)},"
        f"revision_matches={min(len(revision_matches), 32)},"
        f"mounted_matches={min(len(mounted_matches), 32)},"
        f"store_present={'true' if store is not None else 'false'},"
        "active_session="
        f"{'true' if type(active_session) is str and bool(active_session) else 'false'},"
        f"session_matches={min(len(session_matches), 32)},"
        f"buttons={min(len(exact_pairs), 32)},"
        f"enabled_buttons={min(len(enabled_pairs), 32)})"
    )


def _document(version: str) -> str:
    return (
        "<!doctype html><html><body>"
        '<h1 id="chatbook-app-canvas">CHATBOOK_APP_CANVAS</h1>'
        f'<p id="chatbook-app-revision">{version}</p>'
        + (
            (
                '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]'
                + (
                    " --> A"
                    if version == "v2"
                    and os.environ.get("TLDW_CANVAS_TEST_PREVIEW_FAILURE") == "1"
                    else ""
                )
                + "</pre>"
            )
            if os.environ.get("TLDW_CANVAS_TEST_CANDIDATE") == "1"
            or os.environ.get("TLDW_CANVAS_RELEASE_POLICY") == "candidate"
            else ""
        )
        + (
            '<button id="release-submit">Send result</button><script>document.getElementById("release-submit").addEventListener("click",()=>canvas.submit({result:"owned pending receipt"}));</script>'
            if os.environ.get("TLDW_CANVAS_RELEASE_POLICY")
            else ""
        )
        + "</body></html>"
    )


def _publish_counter(path: Path, value: int) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(str(value), encoding="ascii")
    staged.replace(path)


def _publish_owner_receipt(path: Path, value: dict) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(json.dumps(value), encoding="ascii")
    staged.replace(path)


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
        _publish_counter(self._call_count_path, 0)
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
        _publish_counter(self._call_count_path, self.calls)
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
    diagnostic_root = os.environ.get("TLDW_CANVAS_TEST_DB_DIAGNOSTICS")
    if diagnostic_root:
        import faulthandler
        import threading
        import time

        diagnostic_path = Path(diagnostic_root)
        diagnostic_path.mkdir(parents=True, exist_ok=True)
        fault_file = (diagnostic_path / f"child-{os.getpid()}-fault.txt").open("w")
        faulthandler.enable(file=fault_file, all_threads=True)
        connect = sqlite3.connect
        rows = []
        diagnostic_lock = threading.Lock()
        owned_root = Path(os.environ["XDG_DATA_HOME"]).resolve()

        def observed_connect(database, *args, **kwargs):
            value = os.fspath(database)
            if value.startswith("file:"):
                value = value[5:].split("?", 1)[0]
            relative = (
                ":memory:"
                if value == ":memory:"
                else str(Path(value).resolve().relative_to(owned_root))
            )
            connection = connect(database, *args, **kwargs)

            def trace(statement):
                # No SQL text, arguments, or generated source crosses this seam.
                operation = statement.lstrip().split(None, 1)[0].upper()
                if operation not in {
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "PRAGMA",
                    "BEGIN",
                    "COMMIT",
                    "ROLLBACK",
                    "CREATE",
                    "ALTER",
                    "DROP",
                }:
                    operation = "OTHER"
                with diagnostic_lock:
                    rows.append(
                        {
                            "time": time.monotonic(),
                            "thread": threading.get_ident(),
                            "database": relative,
                            "operation": operation,
                        }
                    )
                    (diagnostic_path / f"child-{os.getpid()}-db.json").write_text(
                        json.dumps(rows[-80:]), encoding="utf-8"
                    )

            connection.set_trace_callback(trace)
            return connection

        sqlite3.connect = observed_connect
    if os.environ.get("TLDW_CANVAS_RELEASE_POLICY"):
        from Tests.Canvas.browser.canvas_release_policy import release_snapshot
        from tldw_chatbook.Canvas import profiles

        snapshot = release_snapshot(os.environ["TLDW_CANVAS_RELEASE_POLICY"])
        profiles.load_profile_snapshot = lambda: snapshot
    elif os.environ.get("TLDW_CANVAS_TEST_CANDIDATE") == "1":
        from dataclasses import replace

        from tldw_chatbook.Canvas import profiles

        base = profiles.load_profile_snapshot()
        snapshot = replace(
            base,
            profiles=tuple(
                replace(row, executable=True, reason=None)
                if row.profile_id == "canvas-v2-mermaid-1"
                else row
                for row in base.profiles
            ),
            default_diagram_profile="canvas-v2-mermaid-1",
        )
        profiles.load_profile_snapshot = lambda: snapshot
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

        async def reopen_exact_created_card():
            # Test keyboard adapter presses the real transcript-card button;
            # routing, selected revision and authority remain production-owned.
            target_revision = recovered_root_revision or gateway._revision_id
            _card, button = await _wait_for_exact_canvas_card(
                app.screen,
                target_revision=target_revision,
                attempts=300 if recovered_root_revision is not None else 100,
                interval=0.05 if recovered_root_revision is not None else 0.02,
            )
            screen = app.screen
            original_open = screen._message._open_console_canvas_selection

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
                screen._message._open_console_canvas_selection = original_open
                result = await original_open(**kwargs)
                # The real card handler has no further await after this call;
                # acknowledge on the next refresh, after its dispatch returns.
                app.call_after_refresh(acknowledge_applied_selection)
                return result

            screen._message._open_console_canvas_selection = observe_open_completion
            button.press()

        app.action_canvas_fixture_reopen = reopen_exact_created_card
        app._bindings.bind("f12", "canvas_fixture_reopen", priority=True)

        def acknowledge_composer_focus():
            if os.environ.get("TLDW_CANVAS_TEST_PREVIEW_FAILURE") == "1":
                draft = app.screen.query_one("#console-native-composer").draft_text()
                _publish_owner_receipt(
                    data_root / "canvas-live-repair-receipt",
                    {
                        "draft_sha256": hashlib.sha256(draft.encode()).hexdigest(),
                        "draft_bytes": len(draft.encode()),
                        "provider_calls": gateway.calls,
                    },
                )
            _publish_owner_receipt(
                data_root / "canvas-live-delivery-owner",
                {
                    "served": app._served_canvas_mode,
                    "native_gateway": app.screen._console_runtime().canvas_gateway
                    is not None,
                    "enabled": app.screen._console_runtime()._canvas_enabled(),
                    "control": app.served_canvas_control is not None,
                },
            )
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
