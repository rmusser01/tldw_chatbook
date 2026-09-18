"""Keyboard provider drafts, Revert and real Save in a private native profile.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Only endpoint testing is replaced by a fail-on-call sentinel; persistence is real.
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
from unittest.mock import AsyncMock


def main():
    root = Path(sys.argv[1]).resolve()
    socket, terminal = sys.argv[2:4]
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    seed = tomllib.loads((root / "config.toml").read_text())
    if (
        seed.get("chat_defaults", {}).get("provider") != "llama_cpp"
        or seed.get("chat_defaults", {}).get("model") != "model-a"
        or seed.get("api_settings", {}).get("llama_cpp", {}).get("api_url")
        != "http://127.0.0.1:9099"
    ):
        raise ValueError(
            "Seed the private profile with llama_cpp, model-a and http://127.0.0.1:9099"
        )
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Input, Static
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens import settings_endpoint_probe
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    probe = AsyncMock(side_effect=AssertionError("Unexpected endpoint test"))
    settings_endpoint_probe.probe_settings_endpoint = probe
    app = TldwCli()
    category = SettingsCategoryId.PROVIDERS_MODELS
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": [],
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def disk():
        return tomllib.loads((root / "config.toml").read_text())

    def painted(widget):
        geometry = app.screen._compositor.visible_widgets.get(widget)
        return (
            geometry is not None
            and geometry[0].intersection(geometry[1]) == geometry[0]
        )

    async def tmux(*args):
        await asyncio.to_thread(
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
            deadline = asyncio.get_running_loop().time() + 30
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(f"{label}: focus={app.screen.focused!r}")

        async def settle():
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def settings():
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            if app.screen.active_category != category.value:
                await pilot.press("escape", "/", *"Providers & Models", "enter")
            await wait_for(
                lambda: bool(app.screen.query("#settings-providers-models-card")),
                "Providers & Models card",
            )
            await settle()

        async def tab_to(selector):
            target = app.screen.query_one(selector)
            for _ in range(100):
                if app.screen.focused is target:
                    assert painted(target), selector
                    return target
                await pilot.press("tab")
                await settle()
            raise AssertionError(f"Keyboard focus never reached {selector}")

        async def edit(selector, value):
            field = await tab_to(selector)
            await pilot.press("home", "shift+end", "backspace", *value)
            await wait_for(lambda: field.value == value, "exact input")
            assert painted(field)

        async def revert(discard):
            await pilot.press("escape", "r")
            selector = "#confirm-button" if discard else "#cancel-button"
            await wait_for(
                lambda: bool(app.screen.query(selector)), "Revert confirmation"
            )
            await tab_to(selector)
            await pilot.press("enter")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Revert return"
            )
            await settle()

        def values(model, endpoint):
            assert app.screen.query_one("#settings-model-value", Input).value == model
            assert (
                app.screen.query_one("#settings-provider-endpoint-value", Input).value
                == endpoint
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            await settings()
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
                        lambda size=size: tuple(app.size) == size, "terminal size"
                    )
                    await settle()
                    before = disk()
                    old_model = before["chat_defaults"]["model"]
                    old_endpoint = before["api_settings"]["llama_cpp"]["api_url"]
                    model = f"review-{len(result['cells']) + 1}"
                    endpoint = f"http://127.0.0.1:{9100 + len(result['cells'])}"
                    values(old_model, old_endpoint)
                    assert not app.screen._category_has_unsaved_changes(category)
                    await edit("#settings-model-value", model)
                    await pilot.press("tab")
                    await settle()
                    assert app.screen.focused is app.screen.query_one(
                        "#settings-provider-endpoint-value"
                    )
                    assert painted(app.screen.focused)
                    await edit("#settings-provider-endpoint-value", endpoint)
                    assert app.screen._category_has_unsaved_changes(category)
                    assert disk() == before
                    stem = f"{theme}-{size[0]}"
                    app.save_screenshot(stem + "-draft.svg", path=str(evidence))

                    # A real destination departure must preserve the unsaved draft.
                    await pilot.press("ctrl+2")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ChatScreen",
                        "Console return",
                    )
                    await settings()
                    values(model, endpoint)
                    assert app.screen._category_has_unsaved_changes(category)
                    await revert(False)
                    values(model, endpoint)
                    assert disk() == before
                    await revert(True)
                    await wait_for(
                        lambda: not app.screen._category_has_unsaved_changes(category),
                        "Revert clean",
                    )
                    values(old_model, old_endpoint)
                    assert disk() == before

                    await edit("#settings-model-value", model)
                    await edit("#settings-provider-endpoint-value", endpoint)
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda: not app.screen._category_has_unsaved_changes(category),
                        "Save clean",
                    )
                    saved = disk()
                    expected_defaults = dict(
                        before["chat_defaults"], provider="llama_cpp", model=model
                    )
                    assert saved["chat_defaults"] == expected_defaults
                    assert saved["api_settings"]["llama_cpp"]["api_url"] == endpoint
                    expected = copy.deepcopy(before)
                    expected["chat_defaults"] = expected_defaults
                    expected["api_settings"]["llama_cpp"].update(
                        model=model,
                        api_url=endpoint,
                        credential_source="environment",
                        api_key_env_var="LLAMA_CPP_API_KEY",
                    )
                    expected.setdefault("provider_setup", {}).setdefault(
                        "confirmed", {}
                    )["llama_cpp"] = True
                    assert saved == expected, "Unrelated configuration changed"
                    status = app.screen.query_one(
                        "#settings-provider-save-result", Static
                    )
                    assert "saved" in str(status.renderable).lower()
                    status.scroll_visible(animate=False)
                    await wait_for(
                        lambda status=status: painted(status), "visible Save feedback"
                    )
                    app.save_screenshot(stem + "-saved.svg", path=str(evidence))
                    await pilot.press("ctrl+2")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ChatScreen",
                        "Console after Save",
                    )
                    await settings()
                    values(model, endpoint)
                    assert not app.screen._category_has_unsaved_changes(category)
                    assert disk() == saved
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "draft_retained": True,
                            "cancel_retained": True,
                            "revert_restored": True,
                            "saved_model": model,
                            "saved_endpoint": endpoint,
                            "saved_return_clean": True,
                        }
                    )
                    record()
            probe.assert_not_called()
            result.update(passed=True, endpoint_test_calls=probe.call_count)
        except Exception:  # noqa: BLE001 - preserve evidence and quit normally
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
