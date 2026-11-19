"""Serve the real app with isolated state and a loopback HTTP test provider.

Run with the repository's Python and --profile pointing at a private disposable
directory. Browser interactions after setup exercise the production UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
MODELS = ["uat-llama-a", "uat-llama-b"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    root = args.profile.resolve()
    for name in ("home", "config", "data"):
        (root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        TLDW_CONFIG_PATH=str(root / "config/config.toml"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        LOGURU_LEVEL="WARNING",
    )
    from tldw_chatbook.Backup_Recovery.launcher import startup_unlock

    stopped = startup_unlock()
    if stopped is not None:
        raise RuntimeError(f"Startup refused: {type(stopped).__name__}")
    if args.child:
        run_child(root)
        return

    from tldw_chatbook.config import save_settings_to_cli_config

    save_settings_to_cli_config(
        {
            "general": {"default_tab": "chat"},
            "first_run": {"setup_started": True, "setup_completed": True},
            "splash_screen": {"enabled": False},
            "model_catalog": {"enabled": False},
            "console": {"agent_runtime": False},
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": MODELS[0],
                "streaming": False,
                "max_tokens": 256,
            },
            "providers": {"llama_cpp": MODELS},
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:19877", "model": MODELS[0]},
                "vllm": {
                    "api_url": "http://127.0.0.1:19878/v1",
                    "model": "inactive-model",
                },
            },
        }
    )

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "owned_by": "uat"}
                        for model in MODELS
                    ],
                    "default_generation_settings": {"n_ctx": 32768},
                    "status": "ok",
                }
            )

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with (root / "provider-requests.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"path": self.path, "body": request}) + "\n")
            self.respond(
                {
                    "id": "uat-reply",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "PIXEL_MIGU_UAT_OK: I am still in this character chat.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 16,
                        "total_tokens": 46,
                    },
                }
            )

    provider = ThreadingHTTPServer(("127.0.0.1", 19877), Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    from tldw_chatbook.Web_Server.serve import create_server

    server = create_server(host="127.0.0.1", port=19876, title="Character chat UAT")
    server.command = (
        f'"{sys.executable}" "{Path(__file__).resolve()}" --profile "{root}" --child'
    )
    try:
        server.serve()
    finally:
        provider.shutdown()


def run_child(root: Path) -> None:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.builtin_pixel_migu import (
        ensure_builtin_pixel_migu,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.UI.Console_Modules.wiring import (
        _apply_first_chat_control_selection,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = TldwCli()

    async def prepare(_pilot) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if isinstance(app.screen, ChatScreen) and app.chachanotes_db is not None:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Console did not become ready")
        screen = app.screen
        await asyncio.to_thread(ensure_builtin_pixel_migu, app.chachanotes_db)
        registry = app.workspace_registry_service
        if registry.get_workspace("uat-workspace") is None:
            registry.create_workspace(
                workspace_id="uat-workspace", name="UAT Workspace"
            )
        registry.set_active_workspace("uat-workspace")
        store = screen._ensure_console_chat_store()
        source = store.create_session(
            title="UAT source workspace chat",
            workspace_id="uat-workspace",
            settings=ConsoleSessionSettings(
                provider="llama_cpp", model=MODELS[0], streaming=False
            ),
        )
        store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="Existing workspace conversation.",
            persist=True,
        )
        store.append_message(
            source.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="This source chat must remain unchanged.",
            persist=True,
        )
        _apply_first_chat_control_selection(screen, "vllm", "inactive-model")
        await screen._session._activate_native_console_session(source.id)

        def snapshot() -> None:
            active = store.ensure_session()
            state = {
                "source_session": source.id,
                "source_messages": [
                    m.content for m in store.messages_for_session(source.id)
                ],
                "active_session": active.id,
                "title": active.title,
                "workspace": active.workspace_id,
                "registry_workspace": registry.get_active_workspace().workspace_id,
                "provider": active.settings.provider,
                "model": active.settings.model,
                "stale_controls": screen._console_control_provider,
                "screen": type(app.screen).__name__,
                "messages": [m.content for m in store.messages_for_session(active.id)],
                "code_root": str(REPO),
            }
            (root / "state.json").write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )

        snapshot()
        app.set_interval(0.5, snapshot)

    app.run(auto_pilot=prepare)


if __name__ == "__main__":
    main()
