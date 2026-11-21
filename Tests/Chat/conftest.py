"""Shared fixtures for Tests/Chat (parallel-agents spec).

``controller_with_two_sessions`` originated in Task 1's
``test_console_run_state_per_session.py`` and is reused by Task 7's
``test_console_run_markers.py`` -- moved here per the Task 7 brief's note
so both files (and any future Chat test module) can consume it without
duplicating the fixture or importing across test modules.
"""

from __future__ import annotations

import asyncio
import functools
import warnings

import pytest
import pytest_asyncio

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

from Tests.console_provider_doubles import (
    persisted_console_store,
    provider_resolution,
)


@pytest_asyncio.fixture
async def owned_console_databases():
    """Retire explicitly registered test owners after supported shutdown.

    Yields:
        Registration for an exact database and its optional controller.
    """
    owners = []

    def register(database, controller=None):
        owners.append((database, controller))

    try:
        yield register
    finally:
        errors = []
        for database, controller in reversed(owners):
            try:
                if controller is not None:
                    await asyncio.wait_for(controller.shutdown(), 5)
                with database.quiesce_connections(timeout_seconds=5):
                    pass
                assert database.registered_connection_count() == 0
            except BaseException as error:  # noqa: BLE001 - retain undrained owners
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Console test-owner retirement failed", errors)


@pytest_asyncio.fixture
async def shut_down_console_controllers():
    """Shut down every Console controller the test builds (TASK-42155).

    A rig that builds a ``ConsoleChatController`` and never shuts it down
    leaves its scratch directory and owned work behind. This records each
    controller built while the test runs, shuts down the ones still open at
    teardown, and fails the test if one is still open after that.

    A shutdown that raises after it fenced the controller is reported as a
    warning, not a failure: a test that leaves a dispatch checkpoint pending
    on purpose makes ``_finish_begin_shutdown`` refuse to delete its
    preparation's row ("Resolve pending dispatch before changing
    conversation branches"), and that refusal is the product's.

    Yields:
        The controllers built so far, in build order.
    """
    built = []
    real_init = ConsoleChatController.__init__

    @functools.wraps(real_init)
    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ConsoleChatController, "__init__", init)
        yield built
    for controller in reversed(built):
        if controller._disposed:
            continue
        try:
            await asyncio.wait_for(controller.shutdown(), 5)
        except Exception as error:  # noqa: BLE001 - reported; the check is below
            warnings.warn(
                f"Console controller shutdown raised {type(error).__name__}: {error}",
                stacklevel=1,
            )
    running = [controller for controller in built if not controller._disposed]
    assert not running, f"{len(running)} Console controller(s) left running"


class StreamingGateway:
    """Minimal provider gateway stub -- copied from test_console_chat_controller.py's
    idiom (no network I/O, `ready=True` resolution) since fixture consumers
    typically drive run-state bookkeeping directly rather than actually
    running a send/stream."""

    async def resolve_for_send(self, selection):
        return provider_resolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


@pytest.fixture
def controller_with_two_sessions():
    store = persisted_console_store()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    # `store.new_session` does not exist (verified by grep) -- the real
    # session-creation surface is `store.ensure_session`/`store.create_session`
    # and `controller.new_session`. `controller.new_session()` also activates
    # the session it creates (`ConsoleChatStore.create_session` sets
    # `active_session_id`), matching how `test_controller_creates_and_
    # switches_sessions` in test_console_chat_controller.py builds two
    # sessions. Net effect: session_b (not session_a) is the ACTIVE session
    # once this fixture returns.
    session_a = store.ensure_session(title="Session A")
    session_b = controller.new_session(title="Session B")
    return controller, session_a.id, session_b.id
