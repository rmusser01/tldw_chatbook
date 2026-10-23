"""Mounted managed-install publication must retain newer Import intent."""

import asyncio
import json
import threading
import time
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Collapsible, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_parakeet_v2_install_ui import _report
from tldw_chatbook.Library.ingest_preflight import analyze_path
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Screens import library_screen as library_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import library_ingest_canvas as canvas_module
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas


async def _until(check, label: str, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, label
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice",
    [
        "ordinary",
        "ordinary_override",
        "provider",
        "precision",
        "folder",
        "source",
        "batch_provider",
        "provider_aba",
        "queued_folder",
    ],
)
@private_profile_test
async def test_install_retains_later_selection(request, monkeypatch, tmp_path, choice):
    """Unconditional callback writes fail all newer-choice controls.

    Replaced boundaries are install-plan acquisition, artifact provisioning,
    durable source preference and optional-runtime availability. UI handlers, preflight,
    the worker and call_from_thread publication all run normally.
    """
    sources = [tmp_path / "batch-a.wav", tmp_path / "batch-b.wav"]
    for source in sources:
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\0\0" * 80)
    plan = _report(destination=tmp_path / "installed", download_bytes=1)
    entered, release = threading.Event(), threading.Event()
    preference_calls, provision_threads = [], []

    async def preflight():
        return plan

    async def provision(report, *, progress):
        assert report is plan
        provision_threads.append(threading.get_ident())
        entered.set()
        await _until(release.is_set, "provision gate", timeout=40)
        return report.destination

    monkeypatch.setattr(library_module, "run_parakeet_v2_preflight", preflight)
    monkeypatch.setattr(library_module, "run_parakeet_v2_provision", provision)
    # Feature discovery is qualified: enable controls without installing runtimes.
    monkeypatch.setattr(canvas_module, "_is_installed", lambda *_args: True)
    app = _build_test_app()
    _seed_conversations(app, ())
    app._ensure_parakeet_source_service = lambda: SimpleNamespace(
        prefer_managed=lambda key: preference_calls.append(
            (key.value, threading.get_ident())
        )
    )
    app.notify = MagicMock()
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)
    queue_release = asyncio.Event()
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
            await _wait_for_selector(screen, pilot, "#library-ingest-path")
            form = screen._ingest_state.form
            form.path = str(sources[0])
            form.preflight = analyze_path(str(sources[0]), probe_url=False)
            form.type_options["audio_video"] = {
                "transcription_provider": "parakeet-onnx",
                "transcription_precision": "int8",
            }
            if choice == "ordinary_override":
                form.type_options["audio_video"].update(
                    transcription_precision="f32",
                    transcription_model_dir=str(tmp_path / "prior-external-model"),
                )
            screen._update_library_ingest_dynamic_regions()
            await _wait_for_selector(
                screen, pilot, "#opt-audio_video-transcription_provider"
            )
            screen.query_one("#type-group-audio_video", Collapsible).collapsed = False
            await pilot.pause()
            install = screen.query_one("#opt-audio_video-install-parakeet-v2", Button)
            assert not install.disabled
            install.press()
            await _until(
                lambda: bool(host.screen.query("#parakeet-v2-install-confirm")),
                "consent modal",
            )
            host.screen.query_one("#parakeet-v2-install-confirm", Button).press()
            await _until(entered.is_set, "real background installer")
            await _until(lambda: host.screen is screen, "consent modal dismissed")
            await pilot.pause()

            async def select(name, value):
                control = screen.query_one(f"#opt-audio_video-{name}", Select)
                assert not control.disabled
                screen.set_focus(control)
                await pilot.press("enter", "home")
                options = [item[1] for item in control._options]
                await pilot.press(*(["down"] * options.index(value)), "enter")
                await _until(
                    lambda: form.type_options["audio_video"].get(name) == value,
                    f"accepted {name}",
                )
                assert (
                    screen.query_one(f"#opt-audio_video-{name}", Select).value == value
                )

            if choice in {"source", "batch_provider"}:
                screen.query_one("#library-ingest-path", Input).value = str(sources[1])
                await _until(
                    lambda: form.path == str(sources[1]), "accepted batch B source"
                )
            if choice in {"provider", "batch_provider", "provider_aba"}:
                await select("transcription_provider", "faster-whisper")
                if choice == "provider_aba":
                    await select("transcription_provider", "parakeet-onnx")
            if choice == "precision":
                await select("transcription_precision", "f32")
            folder = str(tmp_path / "later-external-model")
            queued = choice == "queued_folder"
            if queued:
                # Hold the canvas pump at an ordinary next-callback boundary.
                # Input.Changed stays genuinely queued; no event is suppressed.
                canvas = screen.query_one(LibraryIngestCanvas)
                queue_entered = asyncio.Event()

                async def hold_canvas_queue():
                    queue_entered.set()
                    await queue_release.wait()

                canvas.call_next(hold_canvas_queue)
                await _until(queue_entered.is_set, "canvas queue gate")
            if choice in {"folder", "queued_folder"}:
                control = screen.query_one(
                    "#opt-audio_video-transcription_model_dir", Input
                )
                assert not control.disabled
                control.value = folder
                if not queued:
                    await _until(
                        lambda: form.type_options["audio_video"].get(
                            "transcription_model_dir"
                        )
                        == folder,
                        "accepted folder",
                    )
                else:
                    assert (
                        form.type_options["audio_video"].get(
                            "transcription_model_dir", ""
                        )
                        != folder
                    )
            before = dict(form.type_options["audio_video"])
            before_path = form.path
            release.set()
            await _until(
                lambda: any(
                    "installed" in str(call.args[0])
                    for call in app.notify.call_args_list
                ),
                "original UI completion",
            )
            if queued:
                (tmp_path / "completion-before-changed.json").write_text(
                    json.dumps(
                        {
                            "form_options": dict(form.type_options["audio_video"]),
                            "raw_folder": control.value,
                            "control_attached": control.is_attached,
                            "notifications": [
                                str(call.args[0]) for call in app.notify.call_args_list
                            ],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                queue_release.set()
                await _until(
                    lambda: form.type_options["audio_video"].get(
                        "transcription_model_dir"
                    )
                    == folder,
                    "queued Changed delivered to original form owner",
                )
                assert (
                    canvas._reported_option_values[
                        ("audio_video", "transcription_model_dir")
                    ]
                    == folder
                )
                before["transcription_model_dir"] = folder
            await pilot.pause()
            after = dict(screen._ingest_state.form.type_options["audio_video"])
            visible = {
                name: screen.query_one(f"#opt-audio_video-{name}").value
                for name in (
                    "transcription_provider",
                    "transcription_precision",
                    "transcription_model_dir",
                )
            }
            notifications = [str(call.args[0]) for call in app.notify.call_args_list]
            (tmp_path / "publication.json").write_text(
                json.dumps(
                    {
                        "choice": choice,
                        "before": before,
                        "after": after,
                        "visible": visible,
                        "notifications": notifications,
                        "path_before": before_path,
                        "path_after": form.path,
                        "preference_calls": preference_calls,
                        "provision_threads": provision_threads,
                        "ui_thread": threading.get_ident(),
                        "queued_changed_delivered": queued,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            assert provision_threads[0] != threading.get_ident()
            assert preference_calls == [("v2_int8", threading.get_ident())]
            assert form.path == before_path
            if choice in {"ordinary", "ordinary_override"}:
                assert (
                    after["transcription_provider"]
                    == visible["transcription_provider"]
                    == "parakeet-onnx"
                )
                assert (
                    after["transcription_precision"]
                    == visible["transcription_precision"]
                    == "int8"
                )
                assert not visible["transcription_model_dir"]
                assert "selected for this batch" in notifications[-1]
            else:
                assert after == before
                for name, value in visible.items():
                    assert value == before.get(name, "")
                assert "selected for this batch" not in notifications[-1]
                assert "retained" in notifications[-1]
    finally:
        release.set()
        queue_release.set()
