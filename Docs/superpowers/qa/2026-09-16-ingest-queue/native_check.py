"""Inspect queue focus, consent and activity in an isolated native TldwCli profile."""

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

# Cache the optional image library's real terminal query before Textual owns
# stdin; querying after launch can leak the terminal reply into a focused Input.
from textual_image._terminal import probe_terminal

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Library.library_ingest_state import LibraryIngestLastSubmission

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
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    async def stage(path, predicate):
        field = app.screen.query_one("#library-ingest-path", Input)
        field.focus()
        await pilot.pause()
        # Start each size's journey with a fresh edit, disarming the previous
        # journey's intentionally unaccepted Retry confirmation.
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
            await stage(source, lambda p: p.total_files == 1)
            screen.query_one("#type-group-generic").collapsed = False
            title = screen.query_one("#library-ingest-title", Input)
            title.value = "Unsaved next import"
            title.selection = type(title.selection)(2, 7)
            await pilot.pause()
            screen._ingest_state.last_submission = LibraryIngestLastSubmission(
                source=str(source), title="Previous import"
            )
            screen._update_library_ingest_dynamic_regions()
            await pilot.pause()
            await wait_for(
                lambda: not screen.query_one("#library-ingest-start", Button).disabled,
                "Start enabled after preflight",
            )
            screen.query_one("#library-ingest-start", Button).focus()
            await focus_is("#library-ingest-start", "Start import")
            for _ in range(48):
                if screen.focused is screen.query_one("#library-ingest-retry-last"):
                    break
                await pilot.press("tab")
            await focus_is("#library-ingest-retry-last", "Retry this batch")
            await screenshot(f"retry-focus-{width}.svg")
            await pilot.press("enter")
            await focus_is("#library-ingest-retry-last", "Press again to replace form")
            assert screen._ingest_state.retry_confirm_armed
            assert title.value == "Unsaved next import"
            await screenshot(f"retry-consent-{width}.svg")
            # Leave the confirmation armed without accepting it. Fixture-only
            # registry events exercise the real queue projection and controls.
            failed = registry.submit(source_path=str(root / "sources" / "failed.txt"))
            registry.mark_failed(
                failed.job_id, error="Fixture failure: dependency unavailable"
            )
            active = registry.submit(source_path=str(root / "sources" / "active.txt"))
            registry.mark_parsing(active.job_id)
            details = f"#library-ingest-details-{failed.job_id}"
            await wait_for(
                lambda details=details: bool(screen.query(details)), "failed row"
            )
            screen.query_one("#library-ingest-start", Button).focus()
            await focus_is("#library-ingest-start", "Start import")
            for _ in range(48):
                await wait_for(
                    lambda details=details: bool(screen.query(details)),
                    "current queue row",
                )
                if screen.focused is screen.query_one(details):
                    break
                await pilot.press("tab")
            await focus_is(details, "Show details")
            await pilot.press("enter")
            await focus_is(details, "Hide details")
            registry.update_progress(
                active.job_id, progress={"message": "Parsing fixture"}
            )
            await pilot.pause()
            await focus_is(details, "Hide details")
            old_details = screen.query_one(details)
            registry.mark_writing(active.job_id)
            await wait_for(
                lambda details=details, old_details=old_details: (
                    screen.query_one(details) is not old_details
                ),
                "queue transition",
            )
            await focus_is(details, "Hide details")
            await screenshot(f"queue-details-{width}.svg")
            await pilot.press("tab")
            await focus_is(f"#library-ingest-retry-{failed.job_id}", "Retry")
            await pilot.press("tab")
            await focus_is(f"#library-ingest-dismiss-{failed.job_id}", "Dismiss")
            await pilot.press("shift+tab")
            await focus_is(f"#library-ingest-retry-{failed.job_id}", "Retry")
            assert screen.query_one("#library-ingest-title") is title
            assert title.value == "Unsaved next import"
            assert screen._ingest_state.form.path == str(source)
            assert len(registry.jobs()) == 2
            assert registry._store is None and not registry.runner_active
            # Remove fixture projection only; no app submission/cancel path.
            registry.mark_failed(active.job_id, error="Fixture complete")
            registry.clear_finished()
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "retry_focus_and_full_consent_painted": True,
                    "queue_tick_and_disclosure_focus_preserved": True,
                    "forward_reverse_recovery_focus_painted": True,
                    "draft_preserved": True,
                    "synthetic_jobs_only": 2,
                    "registry_empty_after_fixture_cleanup": not registry.jobs(),
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
