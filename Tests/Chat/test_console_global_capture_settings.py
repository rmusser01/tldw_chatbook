"""Global capture saves preserve owner-thread state and durable settlement."""

import asyncio
import threading
import tomllib
from pathlib import Path

import pytest

from Tests.Chat.test_console_chat_controller_exchanges import _new_controller
from Tests.private_profile import private_profile_test
from tldw_chatbook import config
from tldw_chatbook.Chat import console_chat_controller as module
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail


def _setup():
    config.apply_settings_mutation_to_cli_config(
        {
            "console": {
                "exchange_capture": False,
                "trace_viewer_profile": "safe",
                "trace_viewer_profile_version": 1,
            }
        }
    )
    controller = _new_controller()
    session = controller.store.ensure_session()
    return controller, session.id, Path(config.get_cli_config_path())


def _apply(controller, session_id):
    snapshot = controller.capture_policy_snapshot(session_id)
    return controller.apply_global_capture_settings_async(
        enabled=False,
        detail=CaptureDetail.SAFE,
        viewer_profile="full",
        expected_config_generation=snapshot.config_generation,
        expected_policy_revision=snapshot.policy_revision,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["saved", "failed", "exception"])
@pytest.mark.parametrize("cancel", [False, True])
@private_profile_test
async def test_cancelled_global_capture_save_holds_reservation_until_writer_settles(
    request, monkeypatch, outcome, cancel
):
    controller, session_id, path = _setup()
    before = path.read_bytes()
    entered, release = threading.Event(), threading.Event()
    writer = module.apply_console_capture_settings
    owner_thread = threading.get_ident()

    def held_writer(**kwargs):
        assert threading.get_ident() != owner_thread
        entered.set()
        assert release.wait(8)
        if outcome == "exception":
            raise OSError("test unavailable writer")
        if outcome == "failed":
            return config.ConfigMutationResult(False, False, "before_replace")
        return writer(**kwargs)

    monkeypatch.setattr(module, "apply_console_capture_settings", held_writer)
    pending = asyncio.create_task(_apply(controller, session_id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        for _ in range(2):
            if cancel:
                pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            sibling = await _apply(controller, session_id)
            assert sibling.status is module.CapturePolicyMutationStatus.STALE
        assert path.read_bytes() == before
    finally:
        release.set()
    if cancel or outcome == "exception":
        with pytest.raises(asyncio.CancelledError if cancel else OSError):
            await asyncio.wait_for(pending, 5)
    else:
        result = await asyncio.wait_for(pending, 5)
        assert result.status is (
            module.CapturePolicyMutationStatus.APPLIED
            if outcome == "saved"
            else module.CapturePolicyMutationStatus.FAILED
        )
    assert controller.store._capture_policy_mutation is None
    assert tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == (
        "full" if outcome == "saved" else "safe"
    )
    monkeypatch.setattr(module, "apply_console_capture_settings", writer)
    retry = await _apply(controller, session_id)
    assert retry.status is module.CapturePolicyMutationStatus.APPLIED


@pytest.mark.asyncio
@private_profile_test
async def test_closing_opening_session_does_not_misreport_a_saved_global_policy(
    request, monkeypatch
):
    controller, session_id, path = _setup()
    entered, release = threading.Event(), threading.Event()
    writer = module.apply_console_capture_settings

    def held_writer(**kwargs):
        entered.set()
        assert release.wait(8)
        return writer(**kwargs)

    monkeypatch.setattr(module, "apply_console_capture_settings", held_writer)
    pending = asyncio.create_task(_apply(controller, session_id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        controller.store.close_session(session_id)
    finally:
        release.set()
    result = await asyncio.wait_for(pending, 5)
    assert result.status is module.CapturePolicyMutationStatus.APPLIED
    assert result.config_result.file_replaced
    assert controller.store._capture_policy_mutation is None
    assert tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == "full"
