"""TASK-32591: Models source controls after the integration CI repair.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
No provider request is sent. This qualifies integrated rendering and lifecycle,
not model launch, download, or inference.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
    socket, session = sys.argv[2:4]
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
    for key in ("NO_COLOR", "OPENAI_API_KEY", "LLAMA_CPP_API_KEY"):
        os.environ.pop(key, None)

    from textual.css.query import NoMatches, QueryError
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    source = Path(__file__).resolve().parents[4]
    assert Path(app_module.__file__).resolve() == source / "tldw_chatbook/app.py"
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            path: hashlib.sha256((source / path).read_bytes()).hexdigest()
            for path in (
                "tldw_chatbook/app.py",
                "tldw_chatbook/UI/LLM_Management_Window.py",
                "tldw_chatbook/css/features/_llm-management.tcss",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
            )
        },
        "cells": [],
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
            deadline = asyncio.get_running_loop().time() + 40
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            await pilot.wait_for_scheduled_animations()
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            from textual.widgets import Select

            from tldw_chatbook.app import TabNavigationProvider
            from tldw_chatbook.Constants import TAB_LLM
            from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

            TabNavigationProvider(screen=app.screen).switch_tab(TAB_LLM)
            await wait_for(lambda: type(app.screen).__name__ == "LLMScreen", "Models")
            await wait_for(
                lambda: bool(app.screen.query(LLMManagementWindow)), "Models controls"
            )
            window = app.screen.query_one(LLMManagementWindow)
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                await tmux("resize-window", "-t", session, "-x", "80", "-y", "24")
                await wait_for(lambda: tuple(app.size) == (80, 24), "Resize")
                for provider, view in (
                    ("llamacpp", "llama-cpp"),
                    ("llamafile", "llamafile"),
                ):
                    window.active_view = view
                    await wait_for(
                        lambda provider=provider: bool(
                            window.query(f"#{provider}-gguf-source-mode")
                        ),
                        "Source control",
                    )
                    mode = window.query_one(f"#{provider}-gguf-source-mode", Select)
                    mode.scroll_visible(animate=False)
                    mode.focus()
                    await pilot.pause()
                    await pilot.press("enter", "home")
                    if provider == "llamafile":
                        await pilot.press("down")
                    await pilot.press("enter")
                    await wait_for(
                        lambda mode=mode: mode.value == "managed",
                        "Managed mode selected",
                    )
                    managed = window.query_one(
                        f"#{provider}-gguf-managed-select", Select
                    )
                    await wait_for(
                        lambda managed=managed: (
                            managed in app.screen._compositor.visible_widgets
                        ),
                        "Managed selector visible",
                    )
                    await pilot.press("tab")
                    await wait_for(
                        lambda provider=provider: (
                            app.screen.focused
                            is window.query_one(f"#{provider}-gguf-refresh-button")
                        ),
                        "Refresh focus after managed mode",
                    )
                    assert (
                        managed.region
                        in window.query_one(f"#llm-view-{view}").content_region
                    )
                    stem = f"{theme}-80x24-{provider}"
                    await capture(stem)
                    result["cells"].append(
                        {
                            "theme": theme,
                            "provider": provider,
                            "mode": str(mode.value),
                            "mode_region": list(mode.region),
                            "managed_region": list(managed.region),
                            "managed_disabled": managed.disabled,
                            "inventory": "empty private profile; no server started",
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run evidence
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await pilot.press("ctrl+q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
