"""Inspect provider recovery continuity in an isolated native TldwCli profile."""

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
from textual.widgets import Input, Select, Static

# Cache the optional image library's real terminal query before Textual owns
# stdin; querying after launch can leak the terminal reply into a focused Input.
from textual_image._terminal import probe_terminal

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar
from tldw_chatbook.UI.Screens import library_screen as library_module

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
        await wait_for(
            lambda: (
                app.screen._ingest_state.form.path == ""
                and app.screen.query_one("#library-ingest-path", Input).value == ""
            ),
            "cleared source",
        )
        # Clearing audio removes its option group and can replace the form.
        field = app.screen.query_one("#library-ingest-path", Input)
        field.focus()
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
        source = root / "sources" / "audio.wav"
        chosen = root / "sources" / "model.gguf"
        fixture = {"reject": False, "calls": 0, "applied": 0, "dispatches": 0}

        def configure(path):
            assert path == chosen
            fixture["calls"] += 1
            if fixture["reject"]:
                raise ValueError("Fixture rejection")

        library_module.configure_transcribe_cpp_model_path = configure
        apply_result = screen._apply_transcribe_cpp_gguf_result

        def apply(configured, job_id):
            apply_result(configured, job_id)
            fixture["applied"] += 1

        screen._apply_transcribe_cpp_gguf_result = apply

        def dispatch():
            fixture["dispatches"] += 1

        app._top_up_ingest_parse_pool = dispatch
        result["external_boundaries_replaced"] = [
            "model admission/persistence",
            "parse-pool dispatch",
        ]
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "terminal size",
            )
            app.theme = theme
            await pilot.press("i")
            await wait_for(lambda: bool(screen.query("#library-ingest-path")), "Import")
            await stage(source, lambda p: p.total_files == 1)
            screen.query_one("#type-group-audio_video").collapsed = False
            screen.query_one(
                "#opt-audio_video-transcription_provider", Select
            ).value = "transcribe-cpp"
            title = screen.query_one("#library-ingest-title", Input)
            title.value = "Unsaved recovery draft"
            await pilot.pause()
            for outcome, entry in (
                ("cancel", "row"),
                ("reject", "row"),
                ("success", "form"),
                ("success", "row"),
                ("faster", "row"),
            ):
                failed = registry.submit(
                    source_path=str(root / "sources" / "failed.wav"),
                    ingest_options={
                        "audio_video": {"transcription_provider": "transcribe-cpp"}
                    },
                )
                registry.mark_failed(
                    failed.job_id,
                    error="Local transcription model unavailable.",
                    error_detail={
                        "category": "stt_failure",
                        "actions": ["choose_another_gguf", "retry_faster_whisper"],
                    },
                )
                suffix = (
                    "retry-faster-whisper" if outcome == "faster" else "choose-gguf"
                )
                selector = (
                    f"#library-ingest-{suffix}-{failed.job_id}"
                    if entry == "row"
                    else "#opt-audio_video-choose-transcribe-cpp-gguf"
                )
                await wait_for(
                    lambda selector=selector: bool(screen.query(selector)),
                    "recovery control",
                )
                for _ in range(64):
                    if screen.focused is screen.query_one(selector):
                        break
                    await pilot.press("tab")
                await focus_is(
                    selector,
                    "Retry with faster-whisper" if outcome == "faster" else "GGUF",
                )
                title.selection = type(title.selection)(2, 7)
                selection = title.selection
                generation = screen._library_compose_generation
                status_before = screen._transcribe_cpp_configured
                calls, applied, dispatches = (
                    fixture["calls"],
                    fixture["applied"],
                    fixture["dispatches"],
                )
                fixture["reject"] = outcome == "reject"
                await pilot.press("enter")
                if outcome != "faster":
                    await wait_for(
                        lambda: isinstance(app.screen, FileOpen), "GGUF picker"
                    )
                    if outcome == "cancel":
                        await screenshot(f"picker-{width}.svg")
                        await pilot.press("escape")
                    else:
                        field = app.screen.query_one(InputBar).query_one(Input)
                        field.focus()
                        field.value = str(chosen)
                        await pilot.pause()
                        await pilot.press("enter")
                        await wait_for(
                            lambda applied=applied: fixture["applied"] > applied,
                            "admission result",
                        )
                await wait_for(lambda: app.screen is screen, "Import returned")
                await pilot.pause()
                assert screen._library_compose_generation == generation
                assert screen.query_one("#library-ingest-title") is title
                assert (
                    title.value == "Unsaved recovery draft"
                    and title.selection == selection
                )
                assert screen._ingest_state.form.path == str(source)
                assert fixture["calls"] == calls + (outcome in {"reject", "success"})
                assert screen._transcribe_cpp_configured == (
                    True if outcome == "success" else status_before
                )
                if outcome in {"success", "faster"} and entry == "row":
                    await wait_for(
                        lambda: (
                            screen.focused is screen.query_one("#library-ingest-path")
                            and not screen.query("Toast")
                            and bool(painted(screen.focused).strip())
                        ),
                        "visible path focus",
                    )
                    assert painted(screen.focused).strip() in str(source)
                    assert fixture["dispatches"] == dispatches + 1
                    jobs = registry.jobs()
                    assert len(jobs) == 1 and jobs[0].job_id != failed.job_id
                    expected = (
                        "faster-whisper" if outcome == "faster" else "transcribe-cpp"
                    )
                    assert (
                        jobs[0].ingest_options["audio_video"]["transcription_provider"]
                        == expected
                    )
                    registry.mark_failed(
                        jobs[0].job_id, error="Fixture complete; no dispatch"
                    )
                    if outcome == "faster":
                        title.focus()
                        await focus_is("#library-ingest-title", "recovery draft")
                        await screenshot(f"draft-after-recovery-{width}.svg")
                else:
                    await focus_is(selector, "GGUF")
                    assert fixture["dispatches"] == dispatches
                    if outcome == "success":
                        status = screen.query_one(
                            "#opt-audio_video-transcribe-cpp-status", Static
                        )
                        assert "Local GGUF configured." in painted(status)
                        await screenshot(f"configured-{width}.svg")
                registry.clear_finished()
                await pilot.pause()
                result["steps"].append(
                    {
                        "size": [width, height],
                        "theme": theme,
                        "outcome": outcome,
                        "entry": entry,
                        "draft_selection_and_identity_preserved": True,
                        "correct_focus_and_provider_routing": True,
                    }
                )
                record()
            await pilot.press("escape")
            await wait_for(
                lambda: bool(screen.query("#library-hub-action-import")), "hub"
            )
        assert (
            registry._store is None
            and not registry.runner_active
            and not registry.jobs()
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
