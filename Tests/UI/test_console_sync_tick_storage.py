"""A Console sync tick during a run does not re-read the Character scope (TASK-33620.15.1).

Measured on the base build 46c3959526 (live, Anthropic haiku, 160x45, all
threads sampled): from a send's acceptance to its provider call, the busiest
worker work was the Character browser's scope check, run on every 0.2 s sync
tick. Each check made two worker DB calls (``get_local_authority_id`` and the
character-conversation revision, before and after the ambient reads), and a
worker that held no connection opened a fresh one per call -- a private
SQLite helper subprocess each time. Those workers held the GIL long enough to
stretch the UI loop's input waits.

During a run the check now runs on the run's first sync and then at most once
a second while the ambient scope (database, current character, open chat)
is unchanged; any ambient change, a forced refresh, an invalidated scope or a
sync outside a run checks at once, as before.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    REPLY,
    _painted_lines,
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)

pytestmark = pytest.mark.bootstrap_profile


async def _tick(console) -> None:
    """One whole sync tick, after any in-flight (or re-armed) tick settles."""
    for _ in range(3):
        await until(lambda: not console._console_sync_in_progress, timeout=30)
        await asyncio.sleep(0.01)
    await console._sync_native_console_chat_ui()


def _count_scope_reads(console) -> list[int]:
    """Count the Character scope's database reads; settle the projection.

    The harness's in-memory database cannot serve the browser's worker reads
    (each worker connection to ``:memory:`` is a new, empty database), so the
    projection would never settle. The metadata read is replaced by a counted
    constant read and the reload publishes the captured fingerprint, as a
    successful load does: what remains is how often a tick reads.
    """
    from dataclasses import replace

    context = console._character_context
    reads = [0]

    def counting(_database):
        reads[0] += 1
        return ("harness-authority", 1)

    async def settled_refresh():
        snapshot = await context._capture_scope()
        context._publish(replace(context.state, scope_fingerprint=snapshot.fingerprint))

    context._read_database_scope_metadata = counting
    context.refresh = settled_refresh
    return reads


@pytest.mark.asyncio
async def test_sync_ticks_during_a_run_check_the_character_scope_once_a_second(
    monkeypatch,
):
    """AC#2: settled ticks in a run make no DB scope reads; changes still do.

    The recheck interval is pinned long for the settled phase (a loaded host
    must not let a real second pass between ticks) and then zeroed to stand
    for "a second later".
    """
    from tldw_chatbook.UI.Console_Modules import character_context as module

    monkeypatch.setattr(module, "SCOPE_RECHECK_DURING_RUN_SECONDS", 60.0, raising=False)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await console._sync_native_console_chat_ui()
            press(host, "enter", "\r")
            await until(gateway.validation_started.is_set, timeout=30)
            assert console._console_run_active()
            console._stop_console_transcript_sync_timer()
            reads = _count_scope_reads(console)
            try:
                await _tick(console)
                assert reads[0] >= 1, "the run's first sync did not check the scope"
                await _tick(console)  # The first check after a reload settles.
                first = reads[0]
                for _ in range(5):
                    await _tick(console)
                assert reads[0] == first, (
                    f"{reads[0] - first} scope reads in 5 settled run ticks"
                )
                # An ambient change (another open chat) is checked at once.
                context = console._character_context
                real_open = context._open_conversation_accessor
                context._open_conversation_accessor = lambda: "another-chat"
                await _tick(console)
                assert reads[0] > first
                context._open_conversation_accessor = real_open
                await _tick(console)
                settled = reads[0]
                # A second later the run's tick checks the database again.
                monkeypatch.setattr(module, "SCOPE_RECHECK_DURING_RUN_SECONDS", 0.0)
                await _tick(console)
                assert reads[0] > settled
            finally:
                gateway.validation_release.set()
                console._start_console_transcript_sync_timer()
            await until(lambda: REPLY in "\n".join(_painted_lines(host)), timeout=30)
            await until(lambda: not console._console_run_active(), timeout=30)
            # Outside a run every sync checks, as before.
            idle = reads[0]
            await _tick(console)
            await _tick(console)
            assert reads[0] >= idle + 2
