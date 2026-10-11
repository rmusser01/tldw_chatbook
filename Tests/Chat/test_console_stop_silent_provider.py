"""TASK-33666: a Stop releases the run thread while the provider is silent.

The hang this reproduces: with a reply waiting on a provider that accepted the
request and never streamed, Quit settled the reply as stopped but the process
stayed alive until the exit watchdog ended it (120 s, status 1). The thread
the exit was joining was the default-executor worker running
``ConsoleAgentBridge.run_reply``, blocked in the model adapter's wait for the
provider turn: the cancel flag was only read when a chunk arrived, so the wait
ran for the first-token window (300 s for a self-hosted provider).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run

#: Far above the stop poll and the model-loop join, far below the stall and
#: first-token windows: a pass means the cancel ended the wait, not a timeout.
_RELEASE_BOUND_SECONDS = 15.0


class _SilentGateway:
    """Accepts the request and never streams, until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.closed = threading.Event()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.entered.set()
        try:
            while not self.released.is_set():
                await asyncio.sleep(0.02)
        finally:
            self.closed.set()
        yield "too late"


# The run builds its tool catalog from real config; keep the collection-time
# profile, as the other config-reading bridge tests do.
@pytest.mark.bootstrap_profile
def test_stop_during_a_silent_provider_releases_the_run_thread(
    tmp_path, monkeypatch
) -> None:
    # Neither window may be what ends the wait: the stall window, and the
    # first-token window at the 300 s a self-hosted provider gets by default.
    monkeypatch.setattr(bridge_module, "_stall_timeout_seconds", lambda: 120.0)
    monkeypatch.setenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", "300")
    gateway = _SilentGateway()
    bridge, db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "silent", gateway
    )
    cancel = threading.Event()
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["outcome"] = _run(
                bridge, store, session, assistant_id, should_cancel=cancel.is_set
            )
        except BaseException as exc:  # noqa: BLE001 -- reported by the asserts
            result["error"] = exc

    thread = threading.Thread(target=run, name="silent-provider-run", daemon=True)
    thread.start()
    try:
        assert gateway.entered.wait(30.0), result.get("error")
        # The state ConsoleChatController.stop_active_run and the quit-time
        # shutdown() both leave: the reply settled, the cancel flag raised.
        store.mark_message_stopped(assistant_id)
        cancel.set()
        thread.join(timeout=_RELEASE_BOUND_SECONDS)
        still_blocked = thread.is_alive()
    finally:
        gateway.released.set()
        thread.join(timeout=30.0)

    assert not still_blocked, (
        "run_reply stayed blocked on a silent provider after the cancel flag was set"
    )
    assert "error" not in result, result.get("error")
    # The provider request was torn down, not left to run out its window.
    assert gateway.closed.is_set()
    outcome = result["outcome"]
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
    stored = store.get_message(assistant_id)
    assert stored.status == "stopped"
    assert stored.content == ""
