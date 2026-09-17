"""TASK-32746 native instant-save recovery with real successful config writes.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Each cell injects one false return at the config writer; Retry and subsequent
edits call the real writer. This does not simulate a filesystem permission error.
"""

import asyncio
import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tomllib
import traceback
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
    from textual.widgets import Checkbox, Input, Static
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens import settings_screen as settings_module

    source_root = Path(__file__).resolve().parents[4]
    assert Path(app_module.__file__).resolve() == source_root / "tldw_chatbook/app.py"
    probe_terminal()
    real_save = settings_module.save_settings_to_cli_config
    fixture = {"fail_next": False}
    writes = []

    def writer(sections):
        assert set(sections) == {"model_catalog"}
        assert "refresh_consent_recorded" not in sections["model_catalog"]
        injected = fixture["fail_next"]
        fixture["fail_next"] = False
        saved = False if injected else real_save(sections)
        writes.append(
            {"sections": copy.deepcopy(sections), "injected": injected, "saved": saved}
        )
        return saved

    settings_module.save_settings_to_cli_config = writer
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in (
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
            )
        },
        "cells": [],
        "writes": writes,
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

    def config():
        return tomllib.loads((root / "config.toml").read_text())

    def status():
        return str(
            app.screen.query_one(
                "#settings-model-catalog-save-status", Static
            ).renderable
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
                    geometry = app.screen._compositor.visible_widgets.get(target)
                    assert (
                        geometry
                        and geometry[0].intersection(geometry[1]) == geometry[0]
                    ), selector
                    return target
                await pilot.press("tab")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            raise AssertionError(f"Tab did not reach {selector}")

        async def category_search(name):
            await pilot.press("escape", "/", *name, "enter")
            await wait_for(
                lambda: not app.screen._category_pane_swap_pending, "Category pane"
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def saved():
            await wait_for(
                lambda: (
                    not app.screen._model_catalog_save_running
                    and status().startswith("Saved.")
                ),
                "Saved receipt and idle writer",
            )

        async def capture(name):
            await wait_for(
                lambda: not app.screen.query("Toast"), "Notifications dismissed"
            )
            app.save_screenshot(name, path=str(evidence))
            await tmux("capture-pane", "-p", "-t", terminal)

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
            assert not writes
            baseline = config()
            consent = baseline["model_catalog"]["refresh_consent_recorded"]
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
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    before = config()["model_catalog"]
                    master = await tab_to("#settings-model-catalog-auto-refresh")
                    expected = not master.value
                    fixture["fail_next"] = True
                    await pilot.press("space")
                    await wait_for(
                        lambda: "not saved" in status().lower(), "Failure receipt"
                    )
                    assert config()["model_catalog"] == before
                    assert master.value == expected
                    writes_before = len(writes)
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    assert (
                        app.screen.query_one(
                            "#settings-model-catalog-auto-refresh", Checkbox
                        ).value
                        == expected
                    )
                    assert "not saved" in status().lower()
                    assert len(writes) == writes_before
                    retry = await tab_to("#settings-model-catalog-retry")
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await capture(f"{stem}-failed.svg")
                    assert not retry.has_class("-active")
                    await pilot.press("enter")
                    await saved()
                    assert config()["model_catalog"]["auto_refresh_enabled"] == expected
                    await tab_to("#settings-model-catalog-stale-hours")
                    await pilot.press("home", "shift+end", "backspace", *"0.5")
                    await saved()
                    assert config()["model_catalog"]["stale_after_hours"] == 0.5
                    for selector, key in (
                        ("#settings-mc-auto-openai", "auto_refresh_disabled"),
                        ("#settings-mc-write-openai", "write_to_config"),
                    ):
                        widget = await tab_to(selector)
                        wanted = not widget.value
                        await pilot.press("space")
                        await saved()
                        assert ("OpenAI" in config()["model_catalog"][key]) == (
                            not wanted if key == "auto_refresh_disabled" else wanted
                        )
                    current = config()
                    assert (
                        current["model_catalog"]["refresh_consent_recorded"] == consent
                    )
                    assert {
                        k: v for k, v in current.items() if k != "model_catalog"
                    } == {k: v for k, v in baseline.items() if k != "model_catalog"}
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    await tab_to("#settings-model-catalog-stale-hours")
                    assert (
                        app.screen.query_one(
                            "#settings-model-catalog-stale-hours", Input
                        ).value
                        == "0.5"
                    )
                    await capture(f"{stem}-saved.svg")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "saved": current["model_catalog"],
                            "rebuild_preserved": True,
                        }
                    )
                    record()
            result.update(
                passed=True, consent_unchanged=True, unrelated_config_unchanged=True
            )
        except Exception:  # noqa: BLE001 - preserve failed-run diagnostics
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", terminal, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
