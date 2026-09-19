"""Exercise filtered picker controls in an isolated native TldwCli profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
os.environ.pop("NO_COLOR", None)

from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input
from textual_image._terminal import probe_terminal

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileFilter
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

probe_terminal()
app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def painted(widget):
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


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
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            try:
                if predicate():
                    await pilot.pause()
                    return
            except (NoMatches, QueryError):
                pass
            await pilot.pause(0.03)
        raise AssertionError(f"{label}: focus={app.screen.focused!r}")

    async def tab_to(widget):
        for _ in range(24):
            if app.screen.focused is widget:
                break
            await pilot.press("tab")
        await wait_for(lambda: app.screen.focused is widget, "focus")

    async def resize(width, height):
        await tmux("resize-window", "-t", session, "-x", str(width), "-y", str(height))
        await wait_for(lambda: tuple(app.size) == (width, height), "native resize")
        await pilot.pause(0.15)

    async def capture(name):
        await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    def visible(dialog, nav):
        bounds = dialog.query_one(Dialog).content_region
        bar = dialog.query_one(InputBar)
        field = bar.query_one(Input)
        assert "File name:" in painted(dialog.query_one("#file-name-label"))
        assert field.content_region.width >= 30
        for widget in [field, bar.query_one(FileFilter), *bar.query(Button)]:
            assert bounds.contains_region(widget.region), (
                widget,
                widget.region,
                bounds,
            )
        for button in bar.query(Button):
            assert str(button.label) in painted(button)
        assert nav.content_region.height >= 3
        assert "model.gguf" in painted(nav)

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        assert type(app._driver).__name__ == "LinuxDriver"
        result["driver"] = type(app._driver).__name__
        source = root / "sources"
        for theme in ("textual-dark", "textual-light"):
            app.theme = theme
            for kind in ("open", "folder", "save"):
                await resize(80, 24)
                filters = Filters(
                    ("GGUF model files", lambda p: p.suffix == ".gguf"),
                    ("All files", lambda p: True),
                )
                if kind == "save":
                    dialog = FileSave(
                        source, filters=filters, default_file="new-model.gguf"
                    )
                else:
                    dialog = FileOpen(
                        source, filters=filters, offer_select_folder=kind == "folder"
                    )
                returns = []
                await app.push_screen(dialog, returns.append)
                nav = dialog.query_one(DirectoryNavigation)
                await wait_for(
                    lambda nav=nav: (
                        nav.listing_status.startswith("Loaded")
                        and nav.option_count == 2
                    ),
                    "filtered listing",
                )
                await wait_for(
                    lambda dialog=dialog: not dialog.query("Toast"), "notices expired"
                )
                visible(dialog, nav)
                field = dialog.query_one(InputBar).query_one(Input)
                select = dialog.query_one(FileFilter)
                await tab_to(select)
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("end", "enter")
                await wait_for(
                    lambda select=select, nav=nav: (
                        select.value == 1 and nav.option_count == 3
                    ),
                    "changed filter",
                )
                await pilot.press("end")
                highlighted = nav.get_option_at_index(nav.highlighted).location
                assert {
                    nav.get_option_at_index(i).location.resolve()
                    for i in range(nav.option_count)
                } == {source.parent, source / "model.gguf", source / "notes.txt"}
                assert highlighted.name in {"model.gguf", "notes.txt"}
                await tab_to(field)
                await pilot.press("ctrl+a")
                # Real bracketed-paste input delivered through the owned terminal.
                await tmux("set-buffer", "model.gguf")
                await tmux("paste-buffer", "-p", "-t", session)
                await wait_for(
                    lambda field=field: field.value == "model.gguf", "terminal paste"
                )
                await pilot.press("shift+left", "shift+left")
                selection = field.selection
                for width, height in ((170, 48), (80, 24)):
                    await resize(width, height)
                    assert dialog.query_one(InputBar).query_one(Input) is field
                    assert dialog.query_one(FileFilter) is select
                    assert dialog.focused is field
                    assert field.value == "model.gguf" and field.selection == selection
                    assert select.value == 1
                    assert (
                        nav.get_option_at_index(nav.highlighted).location == highlighted
                    )
                    visible(dialog, nav)
                    assert "model.gguf" in painted(field)
                    if kind == "open":
                        await capture(f"open-{theme}-{width}.svg")
                if kind == "save":
                    await pilot.press("ctrl+a", *"new-model.gguf")
                    await capture(f"save-{theme}-80.svg")
                elif kind == "folder":
                    await tab_to(dialog.query_one("#select-current-folder"))
                    await capture(f"folder-{theme}-80.svg")
                await tab_to(dialog.query_one("#select", Button))
                await pilot.press("enter")
                await wait_for(lambda returns=returns: bool(returns), "picker result")
                expected = source / (
                    "new-model.gguf" if kind == "save" else "model.gguf"
                )
                assert returns == [expected]
                assert not (source / "new-model.gguf").exists()
                result["steps"].append(
                    {
                        "theme": theme,
                        "kind": kind,
                        "paste_filter_resize_select": True,
                        "state_and_focus_retained": True,
                    }
                )
                record()
            returns = []
            dialog = FileOpen(source, filters=filters)
            await app.push_screen(dialog, returns.append)
            await tab_to(dialog.query_one("#cancel", Button))
            assert "Cancel" in painted(dialog.query_one("#cancel"))
            await pilot.press("enter")
            await wait_for(lambda returns=returns: bool(returns), "cancel result")
            assert returns == [None]
            result["steps"].append({"theme": theme, "cancel": True})
            record()
        result["passed"] = True
        record()
        await tmux("send-keys", "-t", session, "C-q")
    except Exception:  # noqa: BLE001 - preserve the failure before shutdown
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        await tmux("send-keys", "-t", session, "C-q")


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
