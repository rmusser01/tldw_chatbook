"""TASK-214 native saved-default return and draft ownership, with a loopback catalog.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Default changes use the real config adapter and reload; model listing uses real
HTTP against a task-owned fixture, not a model server or generation backend.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import threading
import tomllib
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    socket, terminal = sys.argv[2:4]
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    for key in ("NO_COLOR", "OPENAI_API_KEY", "LLAMA_CPP_API_KEY"):
        os.environ.pop(key, None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Input, OptionList, Static
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    requests = []

    class Catalog(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            payload = json.dumps(
                {"data": [{"id": f"local-{i}"} for i in range(1, 5)]}
            ).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            requests.append(("POST", self.path))
            self.send_error(405)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Catalog)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    probe_terminal()
    app = TldwCli()
    category = SettingsCategoryId.PROVIDERS_MODELS
    adapter = SettingsConfigAdapter()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings_source_sha256": hashlib.sha256(
            (
                Path(app_module.__file__).parent / "UI/Screens/settings_screen.py"
            ).read_bytes()
        ).hexdigest(),
        "cells": [],
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    def painted(widget):
        geometry = app.screen._compositor.visible_widgets.get(widget)
        return (
            geometry is not None
            and geometry[0].intersection(geometry[1]) == geometry[0]
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 30
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def tab_to(selector):
            target = app.screen.query_one(selector)
            for _ in range(100):
                if app.screen.focused is target:
                    assert painted(target), selector
                    return target
                await pilot.press("tab")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            raise AssertionError(f"Tab did not reach {selector}")

        async def console():
            await pilot.press("ctrl+2")
            await wait_for(lambda: type(app.screen).__name__ == "ChatScreen", "Console")

        async def settings():
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            if app.screen.active_category != category.value:
                await pilot.press("escape", "/", *"Providers & Models", "enter")
            await wait_for(
                lambda: (
                    bool(app.screen.query("#settings-providers-models-card"))
                    and not app.screen._category_pane_swap_pending
                ),
                "Provider form",
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def save_defaults(provider, model):
            assert await asyncio.to_thread(
                adapter.save_sections,
                {
                    "chat_defaults": {"provider": provider, "model": model},
                    "api_settings.llama_cpp": {"api_url": endpoint},
                },
            )
            app.app_config = await asyncio.to_thread(load_settings, force_reload=True)
            assert (
                tomllib.loads((root / "config.toml").read_text())["chat_defaults"][
                    "provider"
                ]
                == provider
            )

        def projection(provider, model):
            screen = app.screen
            assert screen._provider_widget_value() == provider
            assert screen._provider_setting_values()["provider"] == provider
            assert screen.query_one("#settings-model-value", Input).value == model
            picker = screen.query_one("#settings-provider-picker", OptionList)
            assert (
                picker.get_option_at_index(picker.highlighted).provider_id == provider
            )
            readiness = str(
                screen.query_one("#settings-provider-readiness", Static).renderable
            )
            assert (
                screen._provider_display_name(provider) in readiness
                and model in readiness
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        terminal,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(
                        lambda size=size: tuple(app.size) == size, "Terminal size"
                    )
                    await console()
                    await save_defaults("openai", "boot-model")
                    await settings()
                    projection("openai", "boot-model")
                    clean_screen = app.screen
                    await console()
                    model = f"local-{len(result['cells']) + 1}"
                    await save_defaults("llama_cpp", model)
                    # New-chat defaults follow the published config. Existing
                    # user-owned or already-ready sessions have separate authority.
                    defaults = app.screen._session._blank_console_session_settings()
                    assert (defaults.provider, defaults.model) == ("llama_cpp", model)
                    await settings()
                    projection("llama_cpp", model)
                    clean_screen_retained = app.screen is clean_screen
                    assert (
                        app.screen.query_one(
                            "#settings-provider-endpoint-value", Input
                        ).value
                        == endpoint
                    )
                    assert not app.screen._category_has_unsaved_changes(category)
                    await tab_to("#settings-model-value")
                    await pilot.press("escape", "t")
                    await wait_for(
                        lambda: (
                            "model listing reached"
                            in str(
                                app.screen.query_one(
                                    "#settings-provider-test-result", Static
                                ).renderable
                            ).lower()
                        ),
                        "Real loopback catalog result",
                    )
                    test_result = str(
                        app.screen.query_one(
                            "#settings-provider-test-result", Static
                        ).renderable
                    )
                    assert model in test_result
                    await tab_to("#settings-model-value")
                    stem = f"{theme}-{size[0]}"
                    app.save_screenshot(
                        stem + "-saved-provider.svg", path=str(evidence)
                    )
                    await pilot.press("home", "shift+end", "backspace", *"draft-model")
                    await wait_for(
                        lambda: app.screen._category_has_unsaved_changes(category),
                        "Model draft",
                    )
                    draft_screen = app.screen
                    await console()
                    await save_defaults("openai", "boot-model")
                    await settings()
                    projection("llama_cpp", "draft-model")
                    draft_screen_retained = app.screen is draft_screen
                    assert (
                        app.screen.query_one(
                            "#settings-provider-endpoint-value", Input
                        ).value
                        == endpoint
                    )
                    await tab_to("#settings-model-value")
                    app.save_screenshot(stem + "-owned-draft.svg", path=str(evidence))
                    await pilot.press("escape", "r")
                    await wait_for(
                        lambda: bool(app.screen.query("#confirm-button")),
                        "Discard confirmation",
                    )
                    await tab_to("#confirm-button")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: (
                            type(app.screen).__name__ == "SettingsScreen"
                            and not app.screen._category_has_unsaved_changes(category)
                        ),
                        "Discard complete",
                    )
                    projection("openai", "boot-model")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "saved_model": model,
                            "clean_screen_retained": clean_screen_retained,
                            "draft_screen_retained": draft_screen_retained,
                            "draft_preserved": True,
                            "discard_followed_new_default": True,
                            "catalog_result": test_result,
                        }
                    )
                    record()
            assert len(requests) == 4 and all(
                request == ("GET", "/v1/models") for request in requests
            ), requests
            result.update(passed=True, http_requests=requests, generation_requests=0)
        except Exception:  # noqa: BLE001 - retain failure evidence and exit normally
            result.update(
                passed=False, error=traceback.format_exc(), http_requests=requests
            )
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", terminal, "C-q")

    try:
        app.run(auto_pilot=journey, size=(170, 48))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    result.update(app_run_returned=True, server_closed=not server_thread.is_alive())
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
