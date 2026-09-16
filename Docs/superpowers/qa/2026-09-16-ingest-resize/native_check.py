"""Inspect Import focus during native terminal resizing in an isolated native TldwCli profile."""

import asyncio
import hashlib
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
from textual.widgets import Collapsible, Input

# Cache the optional image library's real terminal query before Textual owns
# stdin; querying after launch can leak the terminal reply into a focused Input.
from textual_image._terminal import probe_terminal

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry

probe_terminal()
app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def painted(widget):
    region = widget.content_region
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
        await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    async def stage(path, predicate):
        field = app.screen.query_one("#library-ingest-path", Input)
        field.focus()
        await pilot.pause()
        # Start each theme's journey with a fresh edit and normal blur preflight.
        field.value = ""
        await pilot.pause()
        field.value = str(path)
        # A restored source may already equal this value and emit no Changed.
        # Leaving the field runs the normal explicit blur preflight.
        await pilot.press("tab")
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
        # UI projection only: the fresh registry has no store or runner.
        registry = LibraryIngestJobRegistry()
        app.library_ingest_jobs = registry
        result["synthetic_registry_without_store"] = True
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        source = root / "sources" / "notes.txt"
        for theme in ("textual-dark", "textual-light"):
            await tmux("resize-window", "-t", session, "-x", "170", "-y", "48")
            await wait_for(lambda: tuple(app.size) == (170, 48), "initial wide size")
            app.theme = theme
            await pilot.press("i")
            await wait_for(lambda: bool(screen.query("#library-ingest-path")), "Import")
            await stage(source, lambda p: p.total_files == 1)
            screen.query_one("#type-group-generic").collapsed = False
            title = screen.query_one("#library-ingest-title", Input)
            title.value = "Unsaved resize draft"
            title.selection = type(title.selection)(2, 7)
            failed = registry.submit(source_path=str(root / "sources" / "failed.txt"))
            registry.mark_failed(failed.job_id, error="Fixture failure")
            await wait_for(
                lambda failed=failed: bool(
                    screen.query(f"#library-ingest-details-{failed.job_id}")
                ),
                "failed row",
            )
            await pilot.pause()
            for control, selector, label in (
                ("title", "#library-ingest-title", "Unsaved resize draft"),
                ("encoding", "#opt-generic-encoding", "Auto-detect"),
                ("details", f"#library-ingest-details-{failed.job_id}", "Hide details"),
                (
                    "recent",
                    "#library-ingest-recent > CollapsibleTitle",
                    "Recent imports",
                ),
            ):
                for _ in range(48):
                    if screen.focused is screen.query_one(selector):
                        break
                    await pilot.press("tab")
                if control in {"details", "recent"}:
                    await pilot.press("enter")
                await focus_is(selector, label)
                field = screen.query_one(selector)
                # Input focus selects text; snapshot only after reaching it.
                selection = title.selection
                config_hash = hashlib.sha256(
                    (root / "config.toml").read_bytes()
                ).hexdigest()
                compose_generation = screen._library_compose_generation
                snapshot_generation = screen._library_snapshot_state_generation
                jobs = registry.jobs()
                preflight = screen._ingest_state.form.preflight
                for width, height in ((80, 24), (170, 48), (170, 24), (170, 48)):
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
                            tuple(app.size) == (width, height)
                        ),
                        "resize",
                    )
                    await pilot.pause(0.15)
                    await focus_is(selector, label)
                    assert screen.query_one(selector) is field
                    assert title is screen.query_one("#library-ingest-title")
                    assert title.value == "Unsaved resize draft"
                    assert title.selection == selection
                    assert screen._ingest_state.form.path == str(source)
                    assert screen._ingest_state.form.preflight is preflight
                    assert screen._library_compose_generation == compose_generation
                    assert (
                        screen._library_snapshot_state_generation == snapshot_generation
                    )
                    assert (
                        hashlib.sha256((root / "config.toml").read_bytes()).hexdigest()
                        == config_hash
                    )
                    assert registry.jobs() == jobs
                    if control == "recent":
                        assert not screen.query_one(
                            "#library-ingest-recent", Collapsible
                        ).collapsed
                    if (control == "title" and width == 80) or (
                        control == "recent" and (width == 80 or height == 48)
                    ):
                        await screenshot(f"{control}-{theme}-{width}.svg")
                    result["steps"].append(
                        {
                            "control": control,
                            "theme": theme,
                            "size": [width, height],
                            "same_focused_widget_painted": True,
                            "draft_and_selection_preserved": True,
                            "source_snapshot_preflight_and_compose_unchanged": True,
                            "config_unchanged": True,
                        }
                    )
                    record()
            registry.clear_finished()
            await pilot.press("escape")
            await wait_for(
                lambda: bool(screen.query("#library-hub-action-import")), "hub"
            )
        assert registry._store is None and not registry.runner_active
        assert not registry.jobs()
        result["passed"] = True
        record()
        await tmux("send-keys", "-t", session, "C-q")
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
        result["passed"] = False
        result["error"] = traceback.format_exc()
        focused = app.screen.focused
        if focused is not None:
            result["focus_geometry"] = [
                {
                    "widget": str(widget),
                    "region": list(widget.region),
                    "virtual_region": list(widget.virtual_region),
                    "virtual_size": list(widget.virtual_size),
                    "container_size": list(widget.container_size),
                    "dock_gutter": list(widget.dock_gutter),
                    "scroll_y": widget.scroll_y,
                    "max_scroll_y": widget.max_scroll_y,
                }
                for widget in (focused, *focused.ancestors)
                if hasattr(widget, "region")
            ]
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        await tmux("send-keys", "-t", session, "C-q")


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
