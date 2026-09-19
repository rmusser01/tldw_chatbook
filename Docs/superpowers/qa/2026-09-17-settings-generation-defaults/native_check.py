"""TASK-32748 native per-model generation profile recovery with real successful config writes.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Each cell injects one failed mutation result; the next Save and blank override
removal use the real mutation writer. This does not simulate a filesystem permission error.
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
    from textual.widgets import Collapsible, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import provider_setup_persistence as persistence
    from tldw_chatbook.config import ConfigMutationResult

    source_root = Path(__file__).resolve().parents[4]
    assert Path(app_module.__file__).resolve() == source_root / "tldw_chatbook/app.py"
    probe_terminal()
    real_save = persistence.apply_settings_mutation_to_cli_config
    fixture = {"fail_next": False}
    writes = []

    def writer(sections, *, delete_keys=None):
        assert set(sections) == {"api_settings.openai"}
        assert set(sections["api_settings.openai"]) == {"model_defaults"}
        assert not delete_keys
        injected = fixture["fail_next"]
        fixture["fail_next"] = False
        outcome = (
            ConfigMutationResult(False, False, "before_replace")
            if injected
            else real_save(sections, delete_keys=delete_keys)
        )
        writes.append({"sections": copy.deepcopy(sections), "injected": injected})
        return outcome

    persistence.apply_settings_mutation_to_cli_config = writer
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

        async def edit(selector, value):
            field = await tab_to(selector)
            await pilot.press("home", "shift+end", "backspace", *value)
            await pilot.pause()
            assert field.value == value

        async def generation():
            disclosure = app.screen.query_one(
                "#settings-generation-defaults", Collapsible
            )
            if disclosure.collapsed:
                await tab_to("#settings-generation-defaults CollapsibleTitle")
                await pilot.press("enter")
                await pilot.wait_for_scheduled_animations()
            assert not disclosure.collapsed

        async def revert(discard):
            await pilot.press("escape", "r")
            await wait_for(
                lambda: bool(app.screen.query("#confirm-button")), "Revert confirmation"
            )
            await tab_to("#confirm-button" if discard else "#cancel-button")
            await pilot.press("enter")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Revert return"
            )

        async def save():
            await pilot.press("escape", "s")
            await wait_for(
                lambda: (
                    "saved" in app.screen._provider_save_result.lower()
                    and not app.screen._category_has_unsaved_changes(
                        app.screen._active_category_id()
                    )
                ),
                "Profile saved receipt",
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
            temperature = "#settings-model-profile-temperature"
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
                    await generation()
                    before = config()
                    old_value = str(
                        before["api_settings"]["openai"]["model_defaults"][
                            "model-a"
                        ].get("temperature", "")
                    )
                    await edit(temperature, "nan")
                    count = len(writes)
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda: (
                            "Temperature must be between"
                            in app.screen._provider_save_result
                        ),
                        "Invalid temperature recovery",
                    )
                    assert len(writes) == count and config() == before
                    await edit(temperature, "0.3")
                    await revert(False)
                    assert app.screen.query_one(temperature, Input).value == "0.3"
                    await revert(True)
                    assert app.screen.query_one(temperature, Input).value == old_value
                    await generation()
                    await edit(temperature, "0.35")
                    field = app.screen.query_one(temperature, Input)
                    for resized in ((80, 24), (170, 48), size):
                        await tmux(
                            "resize-window",
                            "-t",
                            terminal,
                            "-x",
                            str(resized[0]),
                            "-y",
                            str(resized[1]),
                        )
                        await wait_for(
                            lambda resized=resized: tuple(app.size) == resized,
                            "Focused field resize",
                        )
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert app.screen.focused is field and field.value == "0.35"
                        geometry = app.screen._compositor.visible_widgets.get(field)
                        assert (
                            geometry
                            and geometry[0].intersection(geometry[1]) == geometry[0]
                        )
                        region = field.region
                        strips = app.screen._compositor.render_strips()
                        painted = "\n".join(
                            strips[y].crop(region.x, region.right).text
                            for y in range(region.y, region.bottom)
                        )
                        assert "0.35" in painted
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    assert not app.screen.query_one(
                        "#settings-generation-defaults", Collapsible
                    ).collapsed
                    assert app.screen.query_one(temperature, Input).value == "0.35"
                    fixture["fail_next"] = True
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda count=count: len(writes) == count + 1, "Injected failure"
                    )
                    assert config() == before
                    assert "file was not written" in app.screen._provider_save_result
                    assert app.screen._category_has_unsaved_changes(
                        app.screen._active_category_id()
                    )
                    assert app.screen.query_one(temperature, Input).value == "0.35"
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await tab_to(temperature)
                    await capture(f"{stem}-pending.svg")
                    await save()
                    expected = copy.deepcopy(before)
                    expected["api_settings"]["openai"]["model_defaults"]["model-a"][
                        "temperature"
                    ] = 0.35
                    assert config() == expected
                    await generation()
                    await edit(temperature, "")
                    await save()
                    expected["api_settings"]["openai"]["model_defaults"]["model-a"].pop(
                        "temperature"
                    )
                    assert config() == expected
                    await category_search("Appearance")
                    await category_search("Providers & Models")
                    assert not app.screen.query_one(
                        "#settings-generation-defaults", Collapsible
                    ).collapsed
                    await tab_to(temperature)
                    assert app.screen.query_one(temperature, Input).value == ""
                    await capture(f"{stem}-saved.svg")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "exact_profile_mutation": True,
                            "same_field_resize_painted": True,
                            "invalid_rejected": True,
                            "failed_save_recovered": True,
                            "blank_removes_override": True,
                            "revert_cancel_and_discard": True,
                        }
                    )
                    record()
            expected_baseline = copy.deepcopy(baseline)
            expected_baseline["api_settings"]["openai"]["model_defaults"][
                "model-a"
            ].pop("temperature", None)
            assert config() == expected_baseline
            result.update(passed=True, unrelated_config_unchanged=True)
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
