"""TASK-32753 native Settings review with private config and real loopback search.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
The loopback fixture emits one HTTP 401 then real JSON results per cell. No
external provider, LLM, backup/restore or manual sync operation is invoked.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import toml


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session = sys.argv[2:4]
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)
    requests = []
    fixture = {"reject": False}

    class SearchServer(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            valid = (
                parsed.path == "/search"
                and query.get("q") == ["tldw chatbook"]
                and query.get("format") == ["json"]
            )
            status = 401 if fixture["reject"] else 200
            assert valid, self.path
            requests.append({"query": query["q"][0], "status": status})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "results": [
                            {
                                "title": "Local review result",
                                "url": "https://example.invalid/review",
                                "content": "Synthetic loopback search result.",
                            }
                        ]
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/search"
    config_path = root / "config.toml"
    raw = toml.loads(config_path.read_text())
    raw["SearchSettings"] = {"search_provider_default": "searx"}
    raw["SearchEngines"] = {"searx_search_api_url": endpoint}
    config_path.write_text(toml.dumps(raw))

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input, Select, Static
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    source = Path(__file__).resolve().parents[4]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "requests": requests,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in (
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/UI/Screens/settings_web_search.py",
                "tldw_chatbook/Widgets/settings_web_search_panel.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
            )
        },
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 35
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def settle():
            await wait_for(
                lambda: not getattr(app.screen, "_category_pane_swap_pending", False),
                "Category settled",
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def category(name):
            await pilot.press("escape", "/", *name, "enter")
            await settle()

        async def tab_to(selector):
            for _ in range(100):
                target = app.screen.query_one(selector)
                if app.screen.focused is target:
                    geometry = app.screen._compositor.visible_widgets.get(target)
                    assert (
                        geometry
                        and geometry[0].intersection(geometry[1]) == geometry[0]
                    ), selector
                    if isinstance(target, Button):
                        region = target.region
                        strips = app.screen._compositor.render_strips()
                        painted = "\n".join(
                            strips[y].crop(region.x, region.right).text
                            for y in range(
                                max(0, region.y), min(len(strips), region.bottom)
                            )
                        )
                        assert str(target.label) in painted, (selector, painted)
                    return target
                await pilot.press("tab")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            raise AssertionError("Tab did not reach " + selector)

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

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
            baseline = toml.loads(config_path.read_text())
            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(
                        lambda size=size: tuple(app.size) == size, "Terminal size"
                    )
                    await category("Overview")
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    assert "Not ready" in str(
                        app.screen.query_one(
                            "#settings-overview-configuration", Static
                        ).renderable
                    )
                    await tab_to("#settings-overview-open-providers-models")
                    await capture(stem + "-overview")
                    for target, expected in (
                        ("providers-models", "providers-models"),
                        ("storage", "storage"),
                        ("privacy-security", "privacy-security"),
                    ):
                        await tab_to("#settings-overview-open-" + target)
                        await pilot.press("enter")
                        await settle()
                        assert app.screen.active_category == expected
                        await category("Overview")
                    await category("Web Search")
                    model = app.screen._web_search_model()
                    await tab_to("#web-search-backend")
                    await pilot.press("enter", "home", "down", "down", "enter")
                    await settle()
                    assert (
                        model.backend == "serper" and model.default_backend == "searx"
                    )
                    field = await tab_to("#web-search-serper_search_api_key")
                    await pilot.press(*"synthetic-draft")
                    await settle()
                    assert field.password and field.value == "synthetic-draft"
                    await category("Overview")
                    await category("Web Search")
                    assert (
                        app.screen.query_one(
                            "#web-search-serper_search_api_key", Input
                        ).value
                        == "synthetic-draft"
                    )
                    await pilot.press("escape", "r")
                    await wait_for(
                        lambda: bool(app.screen.query("#confirm-button")),
                        "Revert confirmation",
                    )
                    await tab_to("#confirm-button")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Settings return",
                    )
                    await settle()
                    assert not model.draft.is_dirty
                    assert toml.loads(config_path.read_text()) == baseline
                    await tab_to("#web-search-backend")
                    await pilot.press("enter", "home", *(["down"] * 6), "enter")
                    await settle()
                    assert model.backend == "searx"
                    field = await tab_to("#web-search-searx_search_api_url")
                    await pilot.press(
                        "home",
                        "shift+end",
                        "backspace",
                        *(endpoint + "?review=" + stem),
                    )
                    await settle()
                    assert model.draft.is_dirty and not model.can_test
                    prior_requests = len(requests)
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda model=model: (
                            not model.saving and not model.draft.is_dirty
                        ),
                        "Save completed",
                    )
                    assert "Saved." in model.save_status
                    saved = toml.loads(config_path.read_text())
                    expected = dict(baseline)
                    expected["SearchEngines"] = {
                        "searx_search_api_url": endpoint + "?review=" + stem
                    }
                    assert saved == expected
                    assert len(requests) == prior_requests
                    baseline = saved
                    await tab_to("#web-search-test")
                    fixture["reject"] = True
                    await pilot.press("enter")
                    await wait_for(
                        lambda model=model, prior_requests=prior_requests: (
                            len(requests) > prior_requests and not model.testing
                        ),
                        "Rejected HTTP request",
                    )
                    assert "success" not in model.test_status.lower()
                    fixture["reject"] = False
                    await tab_to("#web-search-test")
                    await pilot.press("enter")
                    await wait_for(
                        lambda model=model: (
                            "successfully" in model.test_status and not model.testing
                        ),
                        "Successful HTTP retry",
                    )
                    assert len(requests) == prior_requests + 2
                    assert not app.screen.query_one("#web-search-test", Button).disabled
                    await tab_to("#web-search-test")
                    await capture(stem + "-search")
                    assert (
                        app.screen.query_one("#web-search-default", Select).value
                        == "searx"
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "overview_links_keyboard": True,
                            "masked_draft_navigation_and_revert": True,
                            "exact_saved_delta": True,
                            "explicit_http_failure_and_retry": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run diagnostics
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    result.update(app_run_returned=True, loopback_server_closed=not thread.is_alive())
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
