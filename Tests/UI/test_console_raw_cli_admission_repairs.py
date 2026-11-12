"""Raw command restoration, bounded insertion and Arm admission contracts."""

from __future__ import annotations

import pytest
from textual.events import Key

from Tests.UI.test_console_raw_cli_arm_in_place import (
    COMMAND,
    _dialog,
    _install_runtime,
    _typed_raw_draft,
)
from Tests.UI.test_console_raw_cli_send import _Runtime, _controller
from Tests.UI.test_console_send_acknowledgement import build, eager_tasks, press, until
from tldw_chatbook.UI.Console_Modules.raw_cli import (
    prepare_visible_send,
    restore_refused_raw_cli_stash,
)
from tldw_chatbook.UI.Console_Modules.send_acknowledgement import request_visible_send
from tldw_chatbook.Utils.input_validation import RAW_CLI_COMMAND_MAX_BYTES
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ConsoleComposerBar,
    classify_console_raw_draft,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def owned_controller():
    databases = []

    def create(*args, **kwargs):
        result = _controller(*args, **kwargs)
        databases.append(result[0]._agent_runs_db())
        return result

    yield create
    for database in databases:
        database.close()
        assert not database._maintenance_participant.connections


def _raw_composer():
    composer = ConsoleComposerBar()
    composer.handle_console_key(Key("exclamation_mark", "!"))
    composer.handle_console_key(Key("space", " "))
    composer.insert_pasted_text("echo first")
    return composer


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("later_kind", ["paste", "file", "confirm", "typed"])
def test_delayed_refusal_never_trusts_a_combined_newer_draft(
    tmp_path, owned_controller, banked, later_kind
):
    composer = _raw_composer()
    original = composer.draft_text()
    runtime = _Runtime()

    def restore(session_id, stash):
        return (
            False
            if banked
            else restore_refused_raw_cli_stash(
                session_id,
                stash,
                composer=composer,
                active_session_id="session-1",
                visible_session_id="session-1",
            )
        )

    controller, _, errors, workers, _, _ = owned_controller(
        tmp_path, runtime, restore_stash=restore
    )
    # The actual worker's first persistence boundary refuses before execution.
    controller._persist_session_if_needed = lambda _session_id: None
    stash = composer.capture_draft_for_send()
    assert stash is not None
    prepare_visible_send(
        stash,
        lambda: composer,
        controller.start_user_command,
        composer.commit_captured_draft,
        captured=True,
    )
    assert composer.draft_text() == ""
    assert len(workers) == 1
    later = "; echo " + "independent-input-" * 8
    if later_kind == "file":
        composer.insert_file_segment(later, "later.sh")
    elif later_kind == "typed":
        composer.insert_text("later typed input")
    else:
        composer.insert_pasted_text(later)
        if later_kind == "confirm":
            next(
                s for s in composer._segments if s.collapse_state == "collapsed"
            ).collapse_state = "confirm"
    later_text = composer.draft_text()
    workers[0][0]()
    assert runtime.requests == []
    assert errors
    if banked:
        assert composer.draft_text() == later_text
        assert controller.restore_banked_stashes("session-1", composer) == 1
    restored = composer.capture_draft_for_send()
    assert restored.text == original + later_text
    assert not restored.raw_cli_prefix_typed
    assert classify_console_raw_draft(restored).kind == "chat"


@pytest.mark.parametrize("kind", ["paste", "file"])
@pytest.mark.parametrize("character", ["x", "é"])
def test_oversized_raw_insertion_is_refused_before_render_or_mutation(
    monkeypatch, kind, character
):
    composer = _raw_composer()
    before = composer.capture_draft_for_send()
    notices, rendered = [], []
    monkeypatch.setattr(
        composer, "notify", lambda message, **kwargs: notices.append(message)
    )
    monkeypatch.setattr(
        composer,
        "_refresh_visible_draft",
        lambda: rendered.append(composer.draft_text()),
    )
    payload = character * (
        RAW_CLI_COMMAND_MAX_BYTES // len(character.encode("utf-8")) + 1
    )
    if kind == "paste":
        composer.insert_pasted_text(payload)
    else:
        composer.insert_file_segment(payload, "large.sh")
    assert rendered == [], "Oversized raw input reached the literal renderer"
    assert composer.capture_draft_for_send() == before
    assert notices and "16 KiB" in notices[0]


@pytest.mark.parametrize("kind", ["paste", "file"])
def test_exact_utf8_raw_limit_remains_literal_and_trusted(monkeypatch, kind):
    composer = _raw_composer()
    before = composer.draft_text()
    remaining = RAW_CLI_COMMAND_MAX_BYTES - len(before[2:].encode("utf-8"))
    payload = "é" * (remaining // 2) + "x" * (remaining % 2)
    rendered = []
    monkeypatch.setattr(
        composer,
        "_refresh_visible_draft",
        lambda: rendered.append(composer.draft_text()),
    )
    if kind == "paste":
        composer.insert_pasted_text(payload)
    else:
        composer.insert_file_segment(payload, "boundary.sh")
    stash = composer.capture_draft_for_send()
    assert stash.text == before + payload
    assert stash.raw_cli_prefix_typed
    assert all(s.collapse_state == "literal" for s in stash.segments)
    assert (
        len(classify_console_raw_draft(stash).text.encode("utf-8"))
        == RAW_CLI_COMMAND_MAX_BYTES
    )
    assert rendered == [stash.text]


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["guard", "session"])
async def test_arm_respects_visible_send_admission(monkeypatch, unavailable):
    host, gateway, _ = build()
    runtime = _install_runtime(host, permitted=True)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await _typed_raw_draft(host, pilot, gateway)
            before = composer.capture_draft_for_send()
            notices = []
            monkeypatch.setattr(
                host.app_instance,
                "notify",
                lambda message, **kwargs: notices.append(message),
            )
            if unavailable == "session":
                monkeypatch.setattr(
                    console, "_console_visible_send_session_id", lambda: None
                )
            request_visible_send(console, guard=lambda: unavailable != "guard")
            await pilot.pause(0.2)
            assert not isinstance(host.screen, ConfirmationDialog)
            assert not runtime.armed
            assert composer.capture_draft_for_send() == before
            assert notices


@pytest.mark.asyncio
async def test_repeat_arm_and_failed_worker_report_without_touching_draft(monkeypatch):
    host, gateway, _ = build()
    runtime = _install_runtime(host, permitted=True)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await _typed_raw_draft(host, pilot, gateway)
            request_visible_send(console)
            dialog = await _dialog(host)
            notices = []
            monkeypatch.setattr(
                console._raw_cli,
                "_notify",
                lambda message, severity: notices.append(message),
            )
            assert console._raw_cli.request_arm() is False
            assert notices and "already" in notices[-1].lower()
            assert host.screen is dialog
            press(host, "escape")
            await until(lambda: not isinstance(host.screen, ConfirmationDialog))
            await until(lambda: not console._raw_cli._arm_in_flight)

            def failed_start(*args, **kwargs):
                raise RuntimeError("fixture worker refused")

            monkeypatch.setattr(console._raw_cli, "_start_worker", failed_start)
            assert console._raw_cli.request_arm() is False
            assert "could not" in notices[-1].lower()
            assert not runtime.armed
            assert composer.draft_text() == COMMAND
