"""Privacy and Start equivalence for the side-effect-free launch projection."""

from __future__ import annotations

import builtins
import importlib
import os
import shlex
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.LLM_Management.test_gguf_server_sources import (
    _App,
    _InputWidget,
    _LogWidget,
    _Window,
    _write_sparse_gguf,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events as events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    server_lifecycle,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.gguf_source_modes import (
    GGUFSourceMode,
    GGUFSourceSelection,
)
from tldw_chatbook.LLM_Management import snapshot_settings
from tldw_chatbook.LLM_Management.llamacpp_profiles import LlamaCppTuning


def _preview_module():
    return importlib.import_module(
        "tldw_chatbook.LLM_Management.llamacpp_launch_preview"
    )


def test_preview_keeps_defaults_and_typed_options_equivalent_to_actual_argv():
    preview = _preview_module()
    options = preview.prepare_launch_options(
        "",
        "",
        '--mmproj "/private/projector name.gguf" --threads=6',
        LlamaCppTuning(context_size=4096, gpu_layers=-1),
        snapshots_enabled=False,
        environment={},
    )
    assert (options.host, options.port) == ("127.0.0.1", "8080")
    assert options.additional_args == (
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--mmproj",
        "/private/projector name.gguf",
        "--threads=6",
    )
    command = events._build_gguf_server_command(
        "llamacpp",
        "/private/runtime",
        Path("/private/model.gguf"),
        options.host,
        options.port,
        options.additional_args,
    )
    assert command == [
        "/private/runtime",
        "--model",
        "/private/model.gguf",
        "--alias",
        "chatbook-llamacpp",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--mmproj",
        "/private/projector name.gguf",
        "--threads=6",
    ]
    assert "--ctx-size 4096" in options.preview
    assert "--n-gpu-layers -1" in options.preview
    assert "runtime defaults" in options.preview
    assert "not a shell command" in options.preview
    assert "checked at Start" in options.preview
    assert "--threads" not in options.preview
    assert "/private" not in repr(options)


@pytest.mark.parametrize(
    "raw",
    [
        "--api-key SECRET_TOKEN --mmproj /private/projector.gguf",
        "--api-key=SECRET_TOKEN --mmproj=/private/projector.gguf",
        "--unknown-SECRET_TOKEN=/private/projector.gguf",
        '--chat-template "--threads SECRET_TOKEN"',
        "--threads=SECRET_TOKEN",
        '--api-key "--slots"',
    ],
)
def test_every_expert_token_is_hidden_even_when_it_looks_like_a_typed_flag(raw):
    preview = _preview_module()
    options = preview.prepare_launch_options(
        "127.0.0.1",
        "9099",
        raw,
        LlamaCppTuning(),
        snapshots_enabled=False,
        environment={},
    )
    assert options.additional_args == tuple(shlex.split(raw))
    for surface in (options.preview, repr(options)):
        assert "SECRET_TOKEN" not in surface
        assert "/private" not in surface
        assert "--api-key" not in surface
        assert "--mmproj" not in surface
        assert "--threads" not in surface
        assert "--slots" not in surface
    assert "hidden" in options.preview


@pytest.mark.parametrize(
    "host,port,raw,tuning,enabled,environment",
    [
        ("127.0.0.1", "8080", '--api-key "SECRET_TOKEN', LlamaCppTuning(), False, {}),
        (
            "127.0.0.1",
            "8080",
            "--ctx-size=8192",
            LlamaCppTuning(context_size=4096),
            False,
            {},
        ),
        ("127.0.0.1", "8080", "--alias=SECRET_TOKEN", LlamaCppTuning(), False, {}),
        (
            "127.0.0.1",
            "8080",
            "--models-dir=/private/models",
            LlamaCppTuning(),
            False,
            {},
        ),
        (
            "127.0.0.1",
            "8080",
            "--slot-save-path=/private/slots",
            LlamaCppTuning(),
            True,
            {},
        ),
        ("127.0.0.1", "8080", "--no-slots", LlamaCppTuning(), True, {}),
        (
            "127.0.0.1",
            "8080",
            "",
            LlamaCppTuning(),
            True,
            {"LLAMA_ARG_ENDPOINT_SLOTS": "SECRET_TOKEN"},
        ),
        (
            "127.0.0.1",
            "8080",
            "",
            LlamaCppTuning(),
            True,
            {"LLAMA_ARG_SLOT_SAVE_PATH": "/private/slots"},
        ),
        ("SECRET_TOKEN", "8080", "", LlamaCppTuning(), False, {}),
        ("fe80::1%SECRET_TOKEN", "8080", "", LlamaCppTuning(), False, {}),
        ("127.0.0.1", "SECRET_TOKEN", "", LlamaCppTuning(), False, {}),
        ("127.0.0.1", "0", "", LlamaCppTuning(), False, {}),
        ("127.0.0.1", "8080", "SECRET_TOKEN" * 10000, LlamaCppTuning(), False, {}),
    ],
)
def test_invalid_options_raise_only_bounded_recoverable_messages(
    host, port, raw, tuning, enabled, environment
):
    preview = _preview_module()
    with pytest.raises(preview.LlamaCppLaunchValidationError) as caught:
        preview.prepare_launch_options(
            host, port, raw, tuning, snapshots_enabled=enabled, environment=environment
        )
    assert 0 < len(str(caught.value)) < 300
    assert "SECRET_TOKEN" not in repr(caught.value)
    assert "/private" not in repr(caught.value)


def test_snapshot_preview_is_conditional_and_pure(monkeypatch):
    preview = _preview_module()
    # Import before guarding file access: the preparation itself must be pure.
    import tldw_chatbook.LLM_Management.snapshot_admission  # noqa: F401

    def forbidden(*args, **kwargs):
        pytest.fail("Launch preview attempted an external side effect")

    with monkeypatch.context() as patch:
        for owner, name in (
            (builtins, "open"),
            (os, "open"),
            (Path, "stat"),
            (Path, "mkdir"),
            (socket, "socket"),
            (subprocess, "Popen"),
            (events, "acquire_managed_gguf"),
            (events, "reserve_server_launch"),
        ):
            patch.setattr(owner, name, forbidden)
        options = preview.prepare_launch_options(
            "localhost",
            "09099",
            '--api-key "--slots"',
            LlamaCppTuning(),
            snapshots_enabled=True,
            environment={},
        )
    assert options.port == "9099"
    assert "conditional" in options.preview.lower()
    assert "--slots --slot-save-path <private directory>" in options.preview
    assert options.additional_args == ("--api-key", "--slots")


def _start_fixture(tmp_path, *, raw="", host="", port="", tuning=None):
    executable = tmp_path / "PRIVATE_EXECUTABLE"
    executable.touch()
    model = tmp_path / "PRIVATE_MODEL.gguf"
    _write_sparse_gguf(model)
    window = _Window(
        {
            "#llamacpp-exec-path": _InputWidget(value=str(executable)),
            "#llamacpp-host": _InputWidget(value=host),
            "#llamacpp-port": _InputWidget(value=port),
            "#llamacpp-additional-args": _InputWidget(value=raw),
            "#llamacpp-log-output": _LogWidget(),
        }
    )
    window.gguf_source_snapshot = lambda _provider: GGUFSourceSelection(
        GGUFSourceMode.EXTERNAL, external_path=model
    )
    window.llamacpp_tuning = lambda: tuning or LlamaCppTuning()
    return window, _App()


@pytest.mark.parametrize(
    "raw,host,port,enabled,environment",
    [
        ('--api-key "SECRET_TOKEN', "", "", False, {}),
        ("--alias=SECRET_TOKEN", "", "", False, {}),
        ("--ctx-size=32", "", "", False, {}),
        ("--slots", "", "", True, {}),
        ("", "", "", True, {"LLAMA_ARG_SLOT_SAVE_PATH": "/private/slots"}),
        ("", "SECRET_TOKEN", "", False, {}),
        ("", "", "65536", False, {}),
    ],
)
async def test_start_matches_preview_errors_before_source_checks_or_worker(
    monkeypatch, tmp_path, raw, host, port, enabled, environment
):
    preview = _preview_module()
    tuning = LlamaCppTuning(context_size=4096)
    window, app = _start_fixture(tmp_path, raw=raw, host=host, port=port, tuning=tuning)
    monkeypatch.setattr(
        snapshot_settings,
        "load_snapshot_preferences",
        lambda: snapshot_settings.SnapshotPreferences(enabled=enabled),
    )
    monkeypatch.setattr(events.os, "environ", environment)
    with pytest.raises(preview.LlamaCppLaunchValidationError) as expected:
        preview.prepare_launch_options(
            host, port, raw, tuning, snapshots_enabled=enabled, environment=environment
        )
    monkeypatch.setattr(
        events,
        "_source_snapshot",
        lambda *args: pytest.fail("Invalid options reached source validation"),
    )
    await events.handle_start_llamacpp_server_button_pressed(
        window, app, Button.Pressed(Button("Start"))
    )
    assert app.notifications == [(str(expected.value), "error")]
    assert not app.workers
    assert not app._llm_server_launch_claims


@pytest.mark.parametrize("snapshot_active", [False, True])
async def test_start_freezes_safe_actual_launch_after_snapshot_admission(
    monkeypatch, tmp_path, snapshot_active
):
    preview = _preview_module()
    tuning = LlamaCppTuning(context_size=4096)
    raw = '--api-key=SECRET_TOKEN --mmproj="/private/projector path.gguf"'
    window, app = _start_fixture(tmp_path, raw=raw, tuning=tuning)
    monkeypatch.setattr(
        snapshot_settings,
        "load_snapshot_preferences",
        lambda: snapshot_settings.SnapshotPreferences(enabled=True),
    )
    monkeypatch.setattr(events.os, "environ", {})
    monkeypatch.setattr(events, "_preflight_llamacpp_listener", lambda *_args: None)
    captured = []

    def prepare(_app, command, claim):
        if snapshot_active:
            claim._snapshot_context = server_lifecycle.SnapshotLaunchContext(
                SimpleNamespace(disabled_reason=None), Path("/private/snapshots")
            )
            return [*command, "--slots", "--slot-save-path", "/private/snapshots/"], {}
        claim._snapshot_context = server_lifecycle.SnapshotLaunchContext(
            SimpleNamespace(disabled_reason="SECRET_TOKEN"), None
        )
        return command, {}

    monkeypatch.setattr(events, "_prepare_snapshot_launch", prepare)
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda _app, _provider, command, *args, **kwargs: (
            captured.append(command) or "captured"
        ),
    )
    await events.handle_start_llamacpp_server_button_pressed(
        window, app, Button.Pressed(Button("Start"))
    )
    claim = server_lifecycle.current_server_claim(app, "llamacpp")
    expected = preview.prepare_launch_options(
        "", "", raw, tuning, snapshots_enabled=True, environment={}
    )
    assert claim._launch_preview == expected.preview
    work, _ = app.workers[0]
    assert work() == "captured"
    assert "conditional" not in claim._launch_preview.lower()
    assert (
        "--slots --slot-save-path <private directory>" in claim._launch_preview
    ) is snapshot_active
    assert "--api-key=SECRET_TOKEN" in captured[0]
    assert "--mmproj=/private/projector path.gguf" in captured[0]
    assert ("--slots" in captured[0]) is snapshot_active
    assert captured[0][captured[0].index("--ctx-size") + 1] == "4096"
    assert "SECRET_TOKEN" not in claim._launch_preview
    assert "/private" not in claim._launch_preview
    assert "PRIVATE_" not in claim._launch_preview
    assert claim._launch_preview not in repr(claim)
    assert server_lifecycle.release_server_claim(app, "llamacpp", claim)
