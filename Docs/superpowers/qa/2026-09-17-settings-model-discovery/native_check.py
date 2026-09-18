"""TASK-32739 native model discovery, selected save and clear, with a loopback catalog.

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
    from textual.widgets import Input, SelectionList, Static
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
    fixture = {"fail": True, "models": []}

    class Catalog(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            payload = json.dumps(
                {"data": [{"id": name} for name in fixture["models"]]}
            ).encode()
            self.send_response(503 if fixture["fail"] else 200)
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
    assert SettingsConfigAdapter().save_sections(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
            "api_settings.llama_cpp": {"api_url": endpoint},
            "providers": {"Llama_cpp": []},
        }
    )
    load_settings(force_reload=True)
    app = TldwCli()
    category = SettingsCategoryId.PROVIDERS_MODELS
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
            for _ in range(100):
                target = app.screen.query_one(selector)
                if app.screen.focused is target:
                    assert painted(target), selector
                    return target
                await pilot.press("tab")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            raise AssertionError(f"Tab did not reach {selector}")

        async def category_search(name):
            await pilot.press("escape", "/", *name, "enter")
            await wait_for(
                lambda: not app.screen._category_pane_swap_pending,
                "Category pane",
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        def listing():
            return app.screen.query_one(
                "#settings-discovered-models-list", SelectionList
            )

        def status():
            return str(
                app.screen.query_one(
                    "#settings-model-discovery-status", Static
                ).renderable
            )

        def config():
            return tomllib.loads((root / "config.toml").read_text())

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            await category_search("Providers & Models")
            await wait_for(
                lambda: bool(app.screen.query("#settings-discover-provider-models")),
                "Discovery form",
            )
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
                    # Begin each viewport journey from category navigation,
                    # rather than inheriting focus from a now-disabled Clear.
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    cell = len(result["cells"]) + 1
                    models = [f"audit-{cell}-alpha", f"audit-{cell}-beta"]
                    fixture["models"] = models
                    baseline = (root / "config.toml").read_bytes()
                    await tab_to("#settings-discover-provider-models")
                    await pilot.press("enter")
                    if cell == 1:
                        await wait_for(
                            lambda: (
                                status() != "Model discovery: running"
                                and bool(requests)
                            ),
                            "HTTP failure",
                        )
                        assert listing().option_count == 0
                        assert "503" in status() or "failed" in status().lower(), (
                            status()
                        )
                        result["discovery_failure"] = status()
                        fixture["fail"] = False
                        await wait_for(
                            lambda: (
                                not app.screen.query_one(
                                    "#settings-discover-provider-models"
                                ).has_class("-active")
                            ),
                            "Retry button ready",
                        )
                        await pilot.press("enter")
                    await wait_for(
                        lambda: listing().option_count == 2, "Discovered rows"
                    )
                    assert (root / "config.toml").read_bytes() == baseline
                    await tab_to("#settings-discovered-models-list")
                    await pilot.press("home", "down", "space")
                    await pilot.pause()
                    assert listing().selected == [models[1]]
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    assert listing().selected == [models[1]]
                    await tab_to("#settings-discovered-models-list")
                    stem = f"{theme}-{size[0]}"
                    app.save_screenshot(stem + "-selection.svg", path=str(evidence))
                    await tab_to("#settings-save-discovered-provider-models")
                    await pilot.press("enter")
                    await wait_for(lambda: "Saved 1" in status(), "Saved model receipt")
                    saved = config()["providers"]["Llama_cpp"]
                    assert saved == [f"audit-{i}-beta" for i in range(1, cell + 1)]
                    assert app.providers_models["Llama_cpp"] == saved
                    # Save selected fills an empty field only, without saving defaults.
                    assert (
                        app.screen.query_one("#settings-model-value", Input).value
                        == "audit-1-beta"
                    )
                    assert config()["chat_defaults"]["model"] == ""
                    assert app.screen._category_has_unsaved_changes(category)
                    app.save_screenshot(stem + "-saved.svg", path=str(evidence))
                    await tab_to("#settings-clear-discovered-provider-models")
                    await pilot.press("enter")
                    await wait_for(lambda: "cache cleared" in status(), "Cleared cache")
                    assert listing().option_count == 0
                    assert (
                        app.screen.query_one("#settings-model-value", Input).suggester
                        is None
                    )
                    assert config()["providers"]["Llama_cpp"] == saved
                    assert not await app.llm_provider_catalog_scope_service.list_discovered_models(
                        mode="local", provider="llama_cpp"
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "selected": models[1],
                            "saved": saved,
                            "rebuild_preserved_selection": True,
                            "clear_preserved_saved_list": True,
                        }
                    )
                    record()
            assert len(requests) >= 5 and all(
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
