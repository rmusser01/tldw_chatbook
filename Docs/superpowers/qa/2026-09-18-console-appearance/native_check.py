"""Native Appearance layout and cancellation with a private profile and real catalog."""

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
    socket, session = sys.argv[2:4]
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    runpy.run_path(str(here.parent / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
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
    sys.path.insert(0, str(repo))
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Widgets.Console import console_appearance_picker_modal as picker
    from tldw_chatbook.Widgets.emoji_picker import EmojiButton, load_recent_emojis

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    sources = [
        "tldw_chatbook/Widgets/Console/console_appearance_picker_modal.py",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
        "tldw_chatbook/css/build_css.py",
        "tldw_chatbook/css/widget_css.py",
        str(Path(__file__).relative_to(repo)),
        "Tests/UI/test_console_appearance_compact.py",
    ]
    assert all((repo / p).is_file() for p in sources), sources
    result = {
        "pid": os.getpid(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "cells": [],
        "fixture_scope": "Real TldwCli and catalog; directly opened modal with fixture title/initial appearance. Drafts are cancelled; no conversation or appearance save is submitted.",
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

    def visible(modal, widget):
        region, clip = modal._compositor.visible_widgets[widget]
        assert region.intersection(clip) == region, (widget.id, region, clip)

    def painted(modal, widget):
        r = widget.content_region
        return "\n".join(
            s.crop(r.x, r.right).text
            for s in modal._compositor.render_strips()[r.y : r.bottom]
        )

    async def journey(pilot):
        modal = None

        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            async with asyncio.timeout(35):
                while not predicate():
                    await pilot.pause(0.03)
            await settle()

        async def capture(stem):
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
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "startup")
            recents = load_recent_emojis()
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((80, 24), (170, 48)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(width),
                        "-y",
                        str(height),
                    )
                    await wait_for(
                        lambda width=width, height=height: (
                            (app.size.width, app.size.height) == (width, height)
                        ),
                        "resize",
                    )
                    outcomes = []
                    modal = picker.ConsoleAppearancePickerModal(
                        conversation_id="appearance-review",
                        conversation_title="Research conversation",
                        icon="🎨",
                        color="#22d3ee",
                    )
                    app.push_screen(modal, outcomes.append)
                    await settle()
                    await wait_for(
                        lambda modal=modal: not modal.query("Toast"),
                        "notifications cleared",
                    )
                    for action in (picker.APPLY_ID, picker.CLEAR_ID, picker.CANCEL_ID):
                        button = modal.query_one(f"#{action}", Button)
                        visible(modal, button)
                        assert str(button.label) in painted(modal, button)
                    none = modal.query_one(f"#{picker.SWATCH_NONE_ID}", Button)
                    visible(modal, none)
                    assert "none" in painted(modal, none)
                    icons = list(modal.query(EmojiButton))
                    assert len(icons) >= 12
                    for icon in icons[:12]:
                        visible(modal, icon)
                    stem = f"{theme}-{width}x{height}"
                    await capture(stem + "-icons")
                    field = modal.query_one(f"#{picker.FILTER_INPUT_ID}", Input)
                    assert app.focused is field
                    await pilot.press(*list("rocket"))
                    assert field.value == "rocket", field.value
                    await wait_for(
                        lambda modal=modal, icons=icons: (
                            modal._filter_timer is None
                            and 0 < len(modal.query(EmojiButton)) < len(icons)
                        ),
                        "filtered icons",
                    )
                    await pilot.press("enter")
                    await settle()
                    assert modal._selected_icon == "🚀"
                    await pilot.press(*(["shift+tab"] * 5))
                    await settle()
                    last = list(modal.query(f".{picker.SWATCH_CLASS}"))[-1]
                    assert app.focused is last
                    visible(modal, last)
                    assert "■" in painted(modal, last)
                    assert modal.query_one(f"#{picker.COLORS_ID}").scroll_x > 0
                    await capture(stem + "-palette")
                    assert await pilot.click(last)
                    await settle()
                    assert modal._selected_color == last.hex_color
                    assert (
                        modal.query_one(f"#{picker.HEX_INPUT_ID}", Input).value
                        == last.hex_color
                    )
                    assert app.focused is field
                    await pilot.press("shift+tab")
                    cancel = modal.query_one(f"#{picker.CANCEL_ID}", Button)
                    assert app.focused is cancel
                    visible(modal, cancel)
                    await pilot.press("enter")
                    await settle()
                    assert modal not in app.screen_stack and outcomes == [None]
                    assert load_recent_emojis() == recents
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "actions_fully_visible": True,
                            "none_label_unwrapped": True,
                            "first_icon_row_visible": True,
                            "keyboard_palette_reveal": True,
                            "filter_and_pointer_selection": True,
                            "cancelled_without_commit": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - record unexpected native journey failures
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            if modal is not None and modal in app.screen_stack:
                await modal.request_safe_cancel(source="qa-failure-cleanup")
                await settle()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__
        if app._exception is not None
        else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
