"""Source-only entry point for an unqualified speculative-voice checkout."""

from __future__ import annotations

from importlib import import_module, util
import subprocess
import sys
from typing import Any

import pytest

from Tests.UI.test_console_dictation import _mounted_console
from Tests.UI.test_console_hands_free_wiring import _ready_host
from tldw_chatbook.Chat.console_hands_free import HandsFreeController
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.UI.Console_Modules.hands_free import (
    ConsoleSpeculativeHandsFreeSession,
)


def _entry():
    spec = util.find_spec("scripts.run_speculative_voice_dev")
    assert spec is not None, "source-only speculative voice entry is unavailable"
    return import_module("scripts.run_speculative_voice_dev")


def _head(dev_entry) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=dev_entry.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _QualifiedSession:
    def __init__(self) -> None:
        self.entered_with: list[bool] = []
        self.close_reasons: list[ControlKind] = []

    def enter(self, *, capture_live: bool) -> None:
        self.entered_with.append(capture_live)

    def fence_and_close(self, reason: ControlKind) -> None:
        self.close_reasons.append(reason)


def test_development_entry_refuses_a_non_checkout_before_enabling_the_gate(
    tmp_path,
) -> None:
    dev_entry = _entry()
    package_root = tmp_path / "tldw_chatbook"
    package_root.mkdir()

    with pytest.raises(dev_entry.DevelopmentVoiceEntryError, match="Git worktree"):
        dev_entry.validate_source_checkout(
            repo_root=tmp_path,
            package_root=package_root,
            expected_head="0" * 40,
        )


@pytest.mark.parametrize(
    "invalid", ["abbreviated", "mismatch", "foreign_package", "detached"]
)
def test_development_entry_rejects_inexact_checkout_authority(
    monkeypatch, tmp_path, invalid
):
    dev_entry = _entry()
    head = _head(dev_entry)
    package = dev_entry.REPO_ROOT / "tldw_chatbook"
    if invalid == "abbreviated":
        head = head[:12]
    elif invalid == "mismatch":
        head = "0" * 40
    elif invalid == "foreign_package":
        package = tmp_path / "tldw_chatbook"
    else:
        git = dev_entry._git
        monkeypatch.setattr(
            dev_entry,
            "_git",
            lambda root, *args: (
                "" if args == ("branch", "--show-current") else git(root, *args)
            ),
        )
    with pytest.raises(dev_entry.DevelopmentVoiceEntryError):
        dev_entry.validate_source_checkout(
            repo_root=dev_entry.REPO_ROOT,
            package_root=package,
            expected_head=head,
        )


def test_check_mode_reports_exact_checkout_without_starting_the_app(capsys) -> None:
    dev_entry = _entry()
    assert dev_entry.main(["--expect-head", _head(dev_entry), "--check"]) == 0

    output = capsys.readouterr().out
    assert f"head={_head(dev_entry)}" in output
    assert "mode=source-checkout-only" in output


def test_development_entry_forces_pipeline_only_inside_its_process(monkeypatch) -> None:
    dev_entry = _entry()
    from tldw_chatbook.UI.Console_Modules import hands_free

    original_gate = hands_free.speculative_voice_qualified
    original_resolver = hands_free.resolve_handsfree_engine

    def realtime() -> str:
        return "realtime"

    monkeypatch.setattr(hands_free, "resolve_handsfree_engine", realtime)

    with dev_entry.development_voice_enabled(expected_head=_head(dev_entry)):
        assert hands_free.speculative_voice_qualified() is True
        assert hands_free.resolve_handsfree_engine() == "pipeline"

    assert hands_free.speculative_voice_qualified is original_gate
    assert hands_free.resolve_handsfree_engine is realtime
    monkeypatch.setattr(hands_free, "resolve_handsfree_engine", original_resolver)


def test_development_entry_uses_commit_scoped_data_profile() -> None:
    dev_entry = _entry()
    from tldw_chatbook import config

    original_profile = config.get_user_folder_name
    head = _head(dev_entry)

    with dev_entry.development_voice_enabled(expected_head=head) as identity:
        assert identity.profile == f"speculative_voice_dev_{head[:12]}"
        assert config.get_user_folder_name() == identity.profile

    assert config.get_user_folder_name is original_profile


@pytest.mark.asyncio
async def test_macos_say_tts_override_is_scoped_to_launcher_process(
    monkeypatch,
) -> None:
    dev_entry = _entry()
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        _LazyHandsFreeTts,
    )

    original_synthesize = _LazyHandsFreeTts.synthesize_hands_free

    async def render(text: str) -> bytes:
        assert text == "Voice system check."
        return b"RIFF-local-development-wav"

    monkeypatch.setattr(dev_entry, "_render_macos_say_wav", render)

    with dev_entry.development_voice_enabled(
        expected_head=_head(dev_entry),
        macos_say_tts=True,
    ):
        response = await _LazyHandsFreeTts().synthesize_hands_free(
            text="Voice system check."
        )
        chunks = [chunk async for chunk in response.byte_stream]
        await response.aclose()

        assert response.provider_id == "macos_say_development"
        assert response.audio_format == "wav"
        assert response.sample_rate == 24_000
        assert response.metadata == {"channels": 1}
        assert chunks == [b"RIFF-local-development-wav"]

    assert _LazyHandsFreeTts.synthesize_hands_free is original_synthesize


def test_check_mode_runs_as_a_direct_source_script(tmp_path) -> None:
    dev_entry = _entry()
    result = subprocess.run(
        (
            sys.executable,
            str(dev_entry.__file__),
            "--expect-head",
            _head(dev_entry),
            "--check",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"root={dev_entry.REPO_ROOT}" in result.stdout


def test_app_launch_does_not_forward_development_flags(monkeypatch) -> None:
    dev_entry = _entry()
    from tldw_chatbook import cli

    seen_argv: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["launcher", "--outer-argument"])
    monkeypatch.setattr(
        cli,
        "main_cli_runner",
        lambda: seen_argv.append(list(sys.argv)) or 0,
    )

    assert dev_entry.main(["--expect-head", _head(dev_entry)]) == 0

    assert seen_argv == [["launcher"]]
    assert sys.argv == ["launcher", "--outer-argument"]


@pytest.mark.asyncio
async def test_visible_hands_free_control_creates_speculative_session(
    monkeypatch,
) -> None:
    from Tests.UI.test_console_native_chat_flow import _ReadyResolutionGateway

    dev_entry = _entry()
    session = _QualifiedSession()

    with dev_entry.development_voice_enabled(expected_head=_head(dev_entry)):
        app, host = _ready_host()
        app.console_speculative_voice_session_factory = lambda **_kwargs: session

        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            monkeypatch.setattr(
                console._ensure_console_chat_controller().provider_gateway,
                "resolve_for_send",
                _ReadyResolutionGateway().resolve_for_send,
            )
            await pilot.click("#console-hands-free-switch")
            await pilot.pause()

            selected: Any = console._console_hands_free
            assert type(selected) is ConsoleSpeculativeHandsFreeSession
            assert not isinstance(selected.controller, HandsFreeController)
            assert session.entered_with == [False]

            await pilot.click("#console-hands-free-switch")
            await pilot.pause()

    assert session.close_reasons == [ControlKind.HANDS_FREE_EXIT]
