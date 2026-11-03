"""TASK-33620.15.22: one rule decides whether a staged attachment makes Send ready.

Send's readiness is synced from two places: the screen's sync
(``ChatScreen._sync_console_composer_action_state``) and the composer's own
resync on keystrokes and after a Send's acceptance
(``ConsoleComposerBar._sync_current_action_state``). On dev 3be493413a the
screen counted a staged image that carries bytes, the one staged item its send
gate (``_send_console_message_from_visible_action_observed``) sends with an
empty draft, while the composer counted any attachment label. Whichever ran
last decided Send's state, so a staged item the gate refuses flipped Send
between ready and disabled, and a label the screen had not yet cleared kept
Send ready after its image had left the staging list.

Both sites now read the composer's ``has_sendable_payload``, and the screen
answers its attachment half with the send gate's own predicate. So does the
Workbench's own ``send`` action, which used to derive a third answer. A ready
Send is one whose press sends.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_native_chat_flow import (
    DUMMY_OPENAI_API_KEY,
    _build_console_send_test_app,
    _persist_console_provider_config,
    _sync_console_control_bar_now,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

# The harness mounts the real ChatScreen, controller and store.
pytestmark = pytest.mark.bootstrap_profile

#: Every kind of item the store accepts for staging, and whether the send gate
#: sends it with an empty draft. Only the image is produced by today's staging
#: paths (file picker, clipboard, refused-echo resend); a text, code, data, PDF
#: or document file is inserted into the draft instead. The others are kinds
#: the store would hold, so Send must not claim to send them either.
_STAGED_KINDS = (
    (
        "image",
        {"file_type": "image", "insert_mode": "attachment", "data": b"\x89PNG"},
        True,
    ),
    (
        "document-with-bytes",
        {"file_type": "document", "insert_mode": "attachment", "data": b"%PDF-1.4"},
        False,
    ),
    (
        "image-without-bytes",
        {"file_type": "image", "insert_mode": "attachment", "data": None},
        False,
    ),
    (
        "inline-text",
        {
            "file_type": "text",
            "insert_mode": "inline",
            "data": None,
            "text_content": "hello",
        },
        False,
    ),
)


def _attachment(name: str, fields: dict) -> PendingAttachment:
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        original_size=8,
        processed_size=8,
        **fields,
    )


def _send_state(send: Button) -> tuple[bool, bool]:
    """Send's readiness as both sites paint it: (disabled, inactive)."""
    return send.disabled, send.has_class("console-send-inactive")


async def _ready_console(host, pilot, monkeypatch):
    # A staged image must not be refused for vision here, so its readiness is
    # decided by the payload rule alone.
    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *_a, **_k: None)
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    send = composer.query_one("#console-send-message", Button)
    store = console._ensure_console_chat_store()
    session_id = store.ensure_session().id
    composer.load_draft("")
    await pilot.pause()
    return console, composer, send, store, session_id


# Retries the bare sync without the config locks: the projection test needs a
# sync that can be refused. Tests that read Send straight after the screen's
# sync use _sync_console_control_bar_now, which returns with no await after
# rendering; the pause here could let another site repaint first.
async def _sync_controls(console, pilot, *, timeout: float = 4.0) -> None:
    """Wait for a checked projection, including its existing deferred replay."""
    deadline = time.monotonic() + timeout
    while not console._sync_console_control_bar():
        assert time.monotonic() < deadline, (
            "Console control projection did not complete"
        )
        await pilot.pause(0.01)
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_staged_attachment_gets_the_same_send_readiness_from_either_sync_site(
    monkeypatch,
):
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        for name, fields, sendable in _STAGED_KINDS:
            composer.load_draft("")
            store.clear_pending_attachments(session_id)
            assert store.add_pending_attachment(session_id, _attachment(name, fields))

            composer._sync_current_action_state()
            await _sync_console_control_bar_now(console)
            screen_last = _send_state(send)
            composer._sync_current_action_state()
            composer_last = _send_state(send)

            assert composer._pending_attachment_label is not None, name
            assert screen_last == composer_last, name
            assert composer_last == (not sendable, not sendable), name
            if sendable:
                continue
            # A disabled Send is what pressing it does: nothing is sent and
            # the item stays staged.
            before = len(store.messages_for_session(session_id))
            assert await console._send_console_message_from_visible_action() is False
            assert len(store.messages_for_session(session_id)) == before, name
            assert len(store.pending_attachments(session_id)) == 1, name


@pytest.mark.asyncio
async def test_the_composer_resync_reads_what_is_staged_not_the_painted_label(
    monkeypatch,
):
    """A Send's custody takes the image before the screen repaints its label."""
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        image = _attachment("photo.png", _STAGED_KINDS[0][1])
        assert store.add_pending_attachment(session_id, image)
        await _sync_console_control_bar_now(console)
        assert _send_state(send) == (False, False)
        assert composer._pending_attachment_label is not None

        store.transfer_pending_attachments_to_turn(
            session_id, "turn-1", (image.attachment_id,)
        )
        composer._sync_current_action_state()

        assert _send_state(send) == (True, True)
        assert await console._send_console_message_from_visible_action() is False


@pytest.mark.asyncio
async def test_the_workbench_send_action_agrees_with_the_composer_send(monkeypatch):
    """The Workbench's own ``send`` action is a third readiness site.

    ``_build_console_workbench_state`` derived ``can_send`` from draft text
    alone and ignored the vision block, so for an image-only draft it read
    disabled while the composer's Send was ready, and for a vision-blocked
    image under a typed draft it read ready while Send was blocked.
    """
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        workbench_send = console.query_one("#workbench-action-send", Button)
        for name, fields, sendable in _STAGED_KINDS:
            store.clear_pending_attachments(session_id)
            assert store.add_pending_attachment(session_id, _attachment(name, fields))
            await _sync_controls(console, pilot)
            assert (workbench_send.disabled, send.disabled) == (
                not sendable,
                not sendable,
            ), name

        # A staged image the model can't accept blocks Send even under a
        # typed draft; the Workbench action must not read ready over it.
        monkeypatch.setattr(
            attachment_core,
            "vision_block_reason",
            lambda *_a, **_k: "Console send blocked: no image support",
        )
        composer.load_draft("describe this")
        store.clear_pending_attachments(session_id)
        image = _attachment("photo.png", _STAGED_KINDS[0][1])
        assert store.add_pending_attachment(session_id, image)
        await _sync_controls(console, pilot)
        assert (workbench_send.disabled, send.disabled) == (True, True)


@pytest.mark.asyncio
async def test_a_ready_send_with_only_an_image_staged_sends_the_image(monkeypatch):
    """The positive half of "a ready Send is one whose press sends".

    No vision stub here: a vision-capable model passes both the screen's
    and the controller's own image checks, and the provider call is faked
    at ``chat_api_call``.
    """
    provider_calls: list[dict] = []

    def fake_chat_api_call(**kwargs):
        provider_calls.append(kwargs)
        return "A small red square."

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call", fake_chat_api_call
    )
    # A completed send needs the durable-turn DB the factory app lacks, and
    # (as the other full-send tests do) ordinary sending, capture off.
    app = _build_console_send_test_app()
    assert config_module.save_setting_to_cli_config(
        "console", "exchange_capture", False
    )
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4o",
        provider_settings={"api_key": DUMMY_OPENAI_API_KEY},
    )
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send = composer.query_one("#console-send-message", Button)
        store = console._ensure_console_chat_store()
        session_id = store.ensure_session().id
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, target_session_id: str) -> str:
            turn_id = original_launch(draft, target_session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)
        composer.load_draft("")
        image = _attachment("photo.png", _STAGED_KINDS[0][1])
        assert store.add_pending_attachment(session_id, image)
        await _sync_console_control_bar_now(console)
        composer._sync_current_action_state()
        assert _send_state(send) == (False, False)

        assert await console._send_console_message_from_visible_action() is True
        assert turn_ids
        await console._console_runtime().wait_for_turn(turn_ids[0])

        assert store.pending_attachments(session_id) == []
        users = [m for m in store.messages_for_session(session_id) if m.role == "user"]
        assert [m.image_data for m in users] == [b"\x89PNG"]
        # The model received the image, as the only content of the turn.
        (call,) = provider_calls
        parts = call["messages_payload"][-1]["content"]
        assert [p["image_url"]["url"] for p in parts if p["type"] == "image_url"] == [
            "data:image/png;base64,iVBORw=="
        ]


@pytest.mark.asyncio
async def test_attachment_readiness_waits_for_checked_config_projection(monkeypatch):
    """A real config reader holds the projection; releasing it permits replay."""
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        composer.load_draft("describe this")
        await _sync_controls(console, pilot)
        workbench_send = console.query_one("#workbench-action-send", Button)
        assert (workbench_send.disabled, send.disabled) == (False, False)
        entered, release = threading.Event(), threading.Event()

        def hold_config():
            with config_module._settings_rebuild_lock():
                entered.set()
                release.wait(5.0)

        holder = threading.Thread(target=hold_config)
        holder.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            # Existing checked display proofs expire after one real second.
            # The native refresh cannot publish a replacement under this lock.
            await asyncio.sleep(1.05)
            monkeypatch.setattr(
                attachment_core,
                "vision_block_reason",
                lambda *_a, **_k: "Console send blocked: no image support",
            )
            assert store.add_pending_attachment(
                session_id, _attachment("photo.png", _STAGED_KINDS[0][1])
            )
            assert console._sync_console_control_bar() is False
            with pytest.raises(AssertionError, match="projection did not complete"):
                await _sync_controls(console, pilot, timeout=0.05)
            assert not release.is_set()
            assert (workbench_send.disabled, send.disabled) == (False, False)
        finally:
            release.set()
            await asyncio.to_thread(holder.join, 1.0)
        assert not holder.is_alive()
        # Only the already-requested production replay may publish this state.
        deadline = time.monotonic() + 4.0
        while (workbench_send.disabled, send.disabled) != (True, True):
            assert time.monotonic() < deadline, (
                "Deferred control replay did not complete"
            )
            await pilot.pause(0.01)
        assert (workbench_send.disabled, send.disabled) == (True, True)
        assert await console._send_console_message_from_visible_action() is False
        assert len(store.pending_attachments(session_id)) == 1


#: TASK-38649: each return a cold readiness refresh can take without
#: publishing, made once inside the real refresh of a mounted Console.
_COLD_REFRESH_RETURNS = (
    "setup_raises",  # the entry-time session lookup raises (a closed session)
    "read_raises",  # the checked native read raises
    "runtime_rebound",  # another runtime owns the Console when the read returns
    "not_a_read",  # the reader returned no checked read
    "source_moved",  # the config source moved during the read
    "owner_moved",  # the session's settings moved during the read
    "pricing_moved",  # the pricing display sources moved during the read
    "handoff_raises",  # the pristine-default handoff after the read raises
    "owner_moved_in_handoff",  # the config source moved during that handoff
)
_CONSOLE_SYNC_GROUPS = {
    "console-sync",
    "console-readiness-config",
    "console-readiness-publication",
}


@pytest.mark.asyncio
async def test_a_cold_refresh_that_returns_early_still_converges_the_controls(
    monkeypatch,
):
    """TASK-38649: a lone control-bar sync on a cold display defers to the
    refresh it starts. A refresh that publishes nothing must still leave the
    controls converging on their own, and a disposed runtime schedules
    nothing."""
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import console_preparation_reads as reads
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    state = {"path": None, "after_read": False, "disposed": False}
    fired, requests = [], []

    def fire(path, *, after_read=True, depth=2):
        """Take the armed fault once, only inside the readiness refresh."""
        if (
            state["path"] == path
            and state["after_read"] == after_read
            and sys._getframe(depth).f_code.co_name == "_refresh"
        ):
            state["path"] = None
            fired.append(path)
            return True
        return False

    real_read = reads.run_preparation_read

    async def read(callback, **kwargs):
        if getattr(callback, "__name__", "") != "read_current":
            return await real_read(callback, **kwargs)
        if fire("read_raises", after_read=False):
            raise RuntimeError("TASK-38649 injected read failure")
        result = await real_read(callback, **kwargs)
        state["after_read"] = True
        if fire("not_a_read"):
            return None
        if fire("source_moved"):
            return dataclasses.replace(result, source_before=(-1, "moved"))
        if fire("owner_moved"):
            _change_settings(store, session_id)
        if state["path"] == "pricing_moved" and result.pricing is None:
            result = dataclasses.replace(result, pricing=object())
        return result

    real_runtime = spend._readiness_runtime

    def runtime(screen):
        if state["disposed"]:
            return SimpleNamespace(_disposed=True)
        if fire("runtime_rebound"):
            return SimpleNamespace(_disposed=False)
        return real_runtime(screen)

    real_pricing = spend._pricing_display_sources_current
    real_identity = config_module.current_config_identity
    identity_moved = []

    def identity():
        if identity_moved and sys._getframe(2).f_code.co_name == "_refresh":
            identity_moved.clear()
            return (-1, "moved")
        return real_identity()

    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        real_settings = store.session_settings
        real_handoff = console._session._maybe_refresh_stale_default_console_settings
        real_request = console._request_console_control_bar_sync
        real_publish = console._sync_native_console_chat_ui

        def settings(target):
            if fire("setup_raises", after_read=False):
                raise KeyError(target)
            return real_settings(target)

        def handoff(*args, **kwargs):
            if fire("handoff_raises"):
                raise RuntimeError("TASK-38649 injected handoff failure")
            if fire("owner_moved_in_handoff"):
                identity_moved.append(True)
            return real_handoff(*args, **kwargs)

        def request(**kwargs):
            if sys._getframe(1).f_code.co_name == "_refresh":
                requests.append(("retry", state["disposed"]))
            return real_request(**kwargs)

        def publish(*args, **kwargs):
            if sys._getframe(1).f_code.co_name == "_refresh":
                requests.append(("publish", state["disposed"]))
            return real_publish(*args, **kwargs)

        monkeypatch.setattr(reads, "run_preparation_read", read)
        monkeypatch.setattr(spend, "_readiness_runtime", runtime)
        monkeypatch.setattr(
            spend,
            "_pricing_display_sources_current",
            lambda: not fire("pricing_moved") and real_pricing(),
        )
        monkeypatch.setattr(config_module, "current_config_identity", identity)
        monkeypatch.setattr(store, "session_settings", settings)
        monkeypatch.setattr(
            console._session, "_maybe_refresh_stale_default_console_settings", handoff
        )
        monkeypatch.setattr(console, "_request_console_control_bar_sync", request)
        monkeypatch.setattr(console, "_sync_native_console_chat_ui", publish)

        composer.load_draft("describe this")
        image = _attachment("photo.png", _STAGED_KINDS[0][1])
        assert store.add_pending_attachment(session_id, image)
        await _sync_controls(console, pilot)
        workbench_send = console.query_one("#workbench-action-send", Button)
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        failures = {}

        async def settled():
            deadline = time.monotonic() + 4.0
            while projection.pending:
                assert time.monotonic() < deadline, (
                    "the readiness refresh never settled"
                )
                await pilot.pause(0.01)

        async def quiet():
            """Let the last reset's own publication and replays finish first."""
            deadline = time.monotonic() + 8.0
            while (
                projection.pending
                or console._console_control_bar_sync_scheduled
                or any(
                    worker.group in _CONSOLE_SYNC_GROUPS and not worker.is_finished
                    for worker in host.workers
                )
            ):
                assert time.monotonic() < deadline, "the Console never went quiet"
                await pilot.pause(0.02)

        async def converged(blocked, timeout):
            deadline = time.monotonic() + timeout
            while (workbench_send.disabled, send.disabled) != (blocked, blocked):
                if time.monotonic() >= deadline:
                    return False
                await pilot.pause(0.05)
            return True

        for index, path in enumerate((*_COLD_REFRESH_RETURNS, "disposed")):
            blocked = index % 2 == 0
            await quiet()
            assert (workbench_send.disabled, send.disabled) == (not blocked,) * 2, path
            reason = "Console send blocked: no image support" if blocked else None
            monkeypatch.setattr(
                attachment_core, "vision_block_reason", lambda *_a, _r=reason, **_k: _r
            )
            # A settings Apply moves the owner key, so the display is cold.
            _change_settings(store, session_id)
            requests.clear()
            state.update(path=None if path == "disposed" else path, after_read=False)
            assert console._sync_console_control_bar() is False, path
            if path == "disposed":
                # Disposed after the worker was scheduled, before it entered.
                state["disposed"] = True
                # A retry would be asked for in the step that clears pending.
                await settled()
                assert requests == [], "a disposed runtime scheduled more work"
                state["disposed"] = False
            else:
                await settled()
                assert path in fired, path
                # No further trigger: the refresh itself must ask for the retry.
                if (
                    not await converged(blocked, 3.0)
                    or ("retry", False) not in requests
                ):
                    failures[path] = list(requests)
            await _sync_controls(console, pilot)
        assert failures == {}, f"stale after a cold early return: {failures}"


def _change_settings(store, session_id) -> None:
    settings = store.session_settings(session_id)
    store.replace_session_settings(
        session_id,
        dataclasses.replace(
            settings, temperature=round(settings.temperature + 0.01, 2)
        ),
    )
