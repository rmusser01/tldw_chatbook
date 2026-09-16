"""Inspect compact folder listing, validation, navigation and resize in an isolated native TldwCli profile."""

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
from textual.widgets import Button, Collapsible, Input, Select, Static

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
from tldw_chatbook.UI.Screens import library_screen as screen_module
from tldw_chatbook.Widgets.Library import library_ingest_canvas as canvas_module

real_installed = canvas_module._is_installed
screen_module.SelectDirectory = lambda _location, **kwargs: SelectDirectory(
    root / "sources", **kwargs
)
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

    async def focus_is(selector, label):
        await wait_for(
            lambda: (
                app.screen.focused is app.screen.query_one(selector)
                and label in painted(app.screen.focused)
            ),
            f"visible focus {selector}",
        )

    async def screenshot(name):
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    async def stage(path, predicate):
        field = app.screen.query_one("#library-ingest-path", Input)
        field.focus()
        field.value = str(path)
        await wait_for(
            lambda: (
                app.screen._ingest_state.form.path == str(path)
                and app.screen._ingest_state.form.preflight is not None
                and not app.screen._ingest_state.form.preflight_checking
                and predicate(app.screen._ingest_state.form.preflight)
            ),
            f"preflight {path.name}",
        )

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        assert type(app._driver).__name__ == "LinuxDriver"
        result["driver"] = type(app._driver).__name__
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "resize",
            )
            app.theme = theme
            await pilot.press("i")
            await wait_for(lambda: bool(screen.query("#library-ingest-path")), "Import")
            entry = screen.query_one("#library-ingest-path", Input)
            await focus_is(
                "#library-ingest-path",
                Path(entry.value).name if entry.value else "Path to a local",
            )
            await stage(
                root / "sources/audio.wav",
                lambda p: bool(p.type_groups.get("audio_video")),
            )
            await wait_for(
                lambda: bool(screen.query("#type-group-audio_video")), "audio options"
            )
            screen.query_one("#type-group-audio_video", Collapsible).collapsed = False
            directory = "#opt-audio_video-transcription_model_dir"
            browse_id = f"{directory}-browse"
            if width == 170:
                result["real_availability"] = {
                    name: real_installed(name)
                    for name in ("audio_processing", "parakeet_onnx")
                }
                canvas_module._is_installed = lambda feature: (
                    feature in {"audio_processing", "parakeet_onnx"}
                    or real_installed(feature)
                )
                result["ui_availability_simulated"] = [
                    "audio_processing",
                    "parakeet_onnx",
                ]
            screen.query_one(
                "#opt-audio_video-transcription_provider", Select
            ).value = "default"
            await pilot.pause()
            screen.query_one(
                "#opt-audio_video-transcription_provider", Select
            ).value = "parakeet-onnx"
            await wait_for(
                lambda browse_id=browse_id: (
                    not screen.query_one(browse_id, Button).disabled
                ),
                "enabled folder",
            )
            title = screen.query_one("#library-ingest-title", Input)
            title.value = "Unsaved title"
            title.cursor_position = 4
            field = screen.query_one(directory, Input)
            field.value = "prior-model"
            field.focus()
            await focus_is(directory, "prior-model")
            await pilot.press("tab")
            await focus_is(browse_id, "Browse")
            browse = screen.query_one(browse_id, Button)
            assert field.region.right <= browse.region.x
            assert browse.region.right <= field.parent.region.right <= width

            async def loaded(location):
                nav = app.screen.query_one(DirectoryNavigation)
                await wait_for(
                    lambda: (
                        nav.location == location
                        and nav.listing_status.startswith("Loaded")
                    ),
                    "listing loaded",
                )
                return nav

            async def tab_to(selector, label):
                for _ in range(24):
                    if app.screen.focused is app.screen.query_one(selector):
                        break
                    await pilot.press("tab")
                await focus_is(selector, label)

            await pilot.press("enter")
            await wait_for(
                lambda: isinstance(app.screen, SelectDirectory), "directory picker"
            )
            modal = app.screen
            nav = await loaded(root / "sources")
            assert modal.focused is nav
            assert nav.content_region.height >= 3
            await wait_for(
                lambda nav=nav: "model folder" in painted(nav), "visible folder"
            )
            await screenshot(f"listing-{width}.svg")
            await pilot.press("home", "down", "enter")
            selected = root / "sources/model folder"
            await loaded(selected)
            await pilot.press("home", "enter")
            await loaded(root / "sources")
            await tab_to("#path_input", "sources")
            path = modal.query_one("#path_input", Input)
            path.value = "missing-folder"
            await tab_to("#select", "Select")
            await pilot.press("enter")
            await focus_is("#path_input", "missing-folder")
            error = modal.query_one("#picker-error-line", Static)
            await wait_for(
                lambda error=error: "Path not found: missing-folder" in painted(error),
                "readable error",
            )
            selection = path.selection
            highlighted = nav.get_option_at_index(nav.highlighted).location
            if width == 80:
                for resize_width, resize_height in ((170, 48), (80, 24)):
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(resize_width),
                        "-y",
                        str(resize_height),
                    )
                    await wait_for(
                        lambda resize_width=resize_width, resize_height=resize_height: (
                            tuple(app.size) == (resize_width, resize_height)
                        ),
                        "picker resize",
                    )
                    await focus_is("#path_input", "missing-folder")
                    assert path.selection == selection
                    assert (
                        nav.get_option_at_index(nav.highlighted).location == highlighted
                    )
                    assert modal.query_one("#path_input") is path
                    assert "Path not found: missing-folder" in painted(error)
            await screenshot(f"error-{width}.svg")
            if width == 80:
                path.value = "/".join(["missing-folder"] * 36)
                await pilot.press("enter")
                await wait_for(
                    lambda error=error: "Path not found" in painted(error), "long error"
                )
                assert nav.content_region.height >= 3
                assert "model folder" in painted(nav)
                await screenshot("long-error-80.svg")
                path.value = "x" * 260
                await pilot.press("enter")
                await wait_for(
                    lambda error=error: "too long" in painted(error),
                    "overlong path validation",
                )
                assert nav.content_region.height >= 3
                await screenshot("overlong-80.svg")
            path.value = str(root / "sources")
            await pilot.press("enter")
            await wait_for(
                lambda error=error: not error.display, "same-folder error recovery"
            )
            assert nav.location == root / "sources"
            path.value = str(selected)
            await pilot.press("enter")
            await loaded(selected)
            await pilot.press("shift+tab")
            nav.focus()
            await pilot.press("home")
            assert modal.focused is nav
            assert ".." in painted(nav)
            assert nav.option_count == 1
            await screenshot(f"empty-{width}.svg")
            await tab_to("#cancel", "Cancel")
            await pilot.press("enter")
            await wait_for(lambda: app.screen is screen, "cancel returned")
            await focus_is(browse_id, "Browse")
            assert field.value == "prior-model"
            await pilot.press("enter")
            await wait_for(
                lambda: isinstance(app.screen, SelectDirectory), "second picker"
            )
            await loaded(root / "sources")
            await pilot.press("home", "down", "enter")
            await loaded(selected)
            await tab_to("#select", "Select")
            await pilot.press("enter")
            await wait_for(lambda: app.screen is screen, "selection returned")
            await focus_is(browse_id, "Browse")
            assert field.value == str(selected)
            assert screen.query_one("#library-ingest-title") is title
            assert title.value == "Unsaved title" and title.cursor_position == 4
            assert not app.library_ingest_jobs.jobs()
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "listing_and_parent_navigation_visible": True,
                    "typed_path_validation_and_recovery": True,
                    "empty_listing_parent_visible": True,
                    "select_cancel_and_import_context_retained": True,
                    "resize_preserves_path_selection_and_focus": width == 80,
                    "same_folder_error_clears": True,
                    "long_path_checks": width == 80,
                    "jobs_queued": 0,
                }
            )
            record()
            await pilot.press("escape")
            await wait_for(
                lambda: bool(screen.query("#library-hub-action-import")), "hub"
            )
        result["passed"] = True
        record()
        await tmux("send-keys", "-t", session, "C-q")
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
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
