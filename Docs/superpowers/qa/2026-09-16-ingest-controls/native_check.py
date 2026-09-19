"""Inspect per-type option context, dependencies and wrapping in an isolated native TldwCli profile."""

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
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Library import library_ingest_canvas as canvas_module

real_installed = canvas_module._is_installed
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
            await stage(root / "sources", lambda p: len(p.type_groups) == 6)
            groups = ("pdf", "document", "audio_video", "ebook", "image", "generic")
            for group in groups:
                await wait_for(
                    lambda group=group: bool(screen.query(f"#type-group-{group}")),
                    group,
                )
                screen.query_one(f"#type-group-{group}", Collapsible).collapsed = False
            await pilot.pause()
            # Record actual unavailable-state copy before the explicit UI seam.
            canvas_module._is_installed = real_installed
            screen._update_library_ingest_option_group("audio_video")
            # Let its keyboard-focus reveal settle before a separate visual
            # inspection scroll to this nonfocused availability explanation.
            await pilot.pause()
            recursive = screen.query_one(
                "#opt-audio_video-summarize_recursively", Checkbox
            )
            recursive.scroll_visible(animate=False)
            await wait_for(
                lambda recursive=recursive: (
                    str(recursive.label) in " ".join(painted(recursive).split())
                ),
                "wrapped real availability",
            )
            await screenshot(f"availability-{width}.svg")
            install = screen.query_one("#opt-audio_video-install-parakeet-v2", Button)
            install.scroll_visible(animate=False)
            await wait_for(
                lambda install=install: (
                    str(install.label) in " ".join(painted(install).split())
                ),
                "wrapped install explanation in Library shell",
            )
            await screenshot(f"install-reason-{width}.svg")
            result["real_availability"] = {
                name: real_installed(name)
                for name in ("pdf_processing", "audio_processing", "parakeet_onnx")
            }
            # Only control availability is simulated. No processing path is invoked.
            canvas_module._is_installed = lambda _feature: True
            result["ui_availability_simulated"] = True
            for group in groups:
                screen.query_one(f"#opt-{group}-reset", Button).press()
                await pilot.pause()
            analyze = screen.query_one("#opt-generic-analyze", Checkbox)
            analyze.focus()
            await focus_is("#opt-generic-analyze", "Analyze after import")
            await pilot.press("space")
            prompt = screen.query_one("#opt-generic-custom_prompt", TextArea)
            await wait_for(lambda prompt=prompt: not prompt.disabled, "prompt enabled")
            title = screen.query_one("#library-ingest-title", Input)
            title.value = "Unsaved metadata"
            title.selection = type(title.selection)(2, 7)
            prompt.load_text("First line\nSecond line")
            prompt.selection = type(prompt.selection)((0, 2), (1, 4))
            title_selection, prompt_selection = title.selection, prompt.selection
            await pilot.pause()

            def retained(
                title=title,
                prompt=prompt,
                title_selection=title_selection,
                prompt_selection=prompt_selection,
            ):
                assert screen.query_one("#library-ingest-title") is title
                assert (
                    title.value == "Unsaved metadata"
                    and title.selection == title_selection
                )
                assert screen.query_one("#opt-generic-custom_prompt") is prompt
                assert (
                    prompt.text == "First line\nSecond line"
                    and prompt.selection == prompt_selection
                )

            engine = screen.query_one("#opt-pdf-pdf_engine", Select)
            engine.focus()
            await focus_is("#opt-pdf-pdf_engine", "PyMuPDF4LLM")
            await pilot.press("enter", "end", "enter")
            await wait_for(lambda engine=engine: engine.value == "docext", "PDF choice")
            ocr = screen.query_one("#opt-pdf-ocr", Checkbox)
            assert not ocr.disabled
            ocr.focus()
            await focus_is("#opt-pdf-ocr", "Enable OCR")
            await pilot.press("space")
            language = screen.query_one("#opt-pdf-ocr_language", Input)
            await wait_for(
                lambda language=language: not language.disabled, "OCR language enabled"
            )
            language.value = "fr"
            size_input = screen.query_one("#opt-generic-chunk_size", Input)
            size_input.value = "abc"
            await pilot.pause()
            error = screen.query_one("#opt-generic-chunk_size-error", Static)
            assert error.display
            chunk = screen.query_one("#opt-generic-chunk", Checkbox)
            chunk.focus()
            await focus_is("#opt-generic-chunk", "Chunk content")
            await pilot.press("space")
            await wait_for(
                lambda size_input=size_input, error=error: (
                    size_input.disabled and not error.display
                ),
                "chunk gate disabled",
            )
            assert screen.query_one("#opt-generic-chunk_template", Select).disabled
            await focus_is("#opt-generic-chunk", "Chunk content")
            retained()
            await screenshot(f"chunk-gate-{width}.svg")
            await pilot.press("space")
            await wait_for(
                lambda size_input=size_input, error=error: (
                    not size_input.disabled and error.display
                ),
                "chunk error returns",
            )
            encoding = screen.query_one("#opt-generic-encoding", Select)
            encoding.focus()
            await focus_is("#opt-generic-encoding", "UTF-8")
            await pilot.press("enter", "home", "down", "enter")
            await wait_for(
                lambda encoding=encoding: encoding.value == "utf-8", "encoding choice"
            )
            retained()
            reset = screen.query_one("#opt-pdf-reset", Button)
            reset.focus()
            await focus_is("#opt-pdf-reset", "Reset to defaults")
            await pilot.press("enter")
            await wait_for(
                lambda engine=engine, language=language: (
                    engine.value == "pymupdf4llm" and language.value == "en"
                ),
                "PDF reset",
            )
            await focus_is("#opt-pdf-reset", "Reset to defaults")
            assert language.disabled and size_input.value == "abc" and error.display
            retained()
            await screenshot(f"pdf-reset-{width}.svg")
            provider = screen.query_one(
                "#opt-audio_video-transcription_provider", Select
            )
            provider.focus()
            await focus_is("#opt-audio_video-transcription_provider", "Auto")
            await pilot.press("enter", "home", "down", "enter")
            await wait_for(
                lambda provider=provider: provider.value == "parakeet-onnx",
                "Parakeet choice",
            )
            folder = screen.query_one("#opt-audio_video-transcription_model_dir", Input)
            assert not folder.disabled
            folder.value = "retained-folder"
            await pilot.press("enter", "end", "enter")
            await wait_for(
                lambda provider=provider: provider.value == "transcribe-cpp",
                "GGUF provider",
            )
            chooser = screen.query_one(
                "#opt-audio_video-choose-transcribe-cpp-gguf", Button
            )
            assert (
                chooser.display
                and folder.disabled
                and folder.value == "retained-folder"
            )
            # Traverse to the actual optional action without invoking its picker.
            for _ in range(32):
                if screen.focused is chooser:
                    break
                await pilot.press("tab")
            await focus_is("#opt-audio_video-choose-transcribe-cpp-gguf", "Choose GGUF")
            retained()
            await screenshot(f"gguf-control-{width}.svg")
            title.focus()
            await focus_is("#library-ingest-title", "Unsaved metadata")
            # Input's normal focus policy selects all on re-entry; the retained
            # off-focus selection was checked immediately before moving here.
            assert title.selection == type(title.selection)(0, len(title.value))
            await pilot.press("x")
            assert title.value == "x"
            assert prompt.selection == prompt_selection
            assert screen._ingest_state.form.path == str(root / "sources")
            assert not app.library_ingest_jobs.jobs()
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "checkbox_select_reset_preserve_editors": True,
                    "dependency_and_error_state_current": True,
                    "optional_action_visibility_current": True,
                    "wrapped_availability_reason_visible": True,
                    "keyboard_reentry_uses_normal_select_all": True,
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
