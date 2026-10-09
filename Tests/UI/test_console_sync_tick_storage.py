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


def _count_scope_reads(console) -> dict[str, int]:
    """Count the Character scope's database reads; let the projection load.

    The harness's in-memory database cannot serve the browser's worker reads
    (each worker connection to ``:memory:`` is a new, empty database), so the
    projection would never load. The metadata read is replaced by a counted
    read of a revision the test controls (live, every message write bumps
    it), and the reload publishes the captured fingerprint, as a successful
    load does: what remains is how often a tick reads.
    """
    from dataclasses import replace

    context = console._character_context
    state = {"reads": 0, "revision": 1}

    def counting(_database):
        state["reads"] += 1
        state["revision"] += state.get("writes_per_read", 0)
        return ("harness-authority", state["revision"])

    async def loading_refresh():
        from tldw_chatbook.UI.Console_Modules.character_context import (
            _ConsoleCharacterScopeChanged,
        )

        try:
            snapshot = await context._capture_scope()
        except _ConsoleCharacterScopeChanged:
            return  # As refresh() does when the scope never settles.
        context._publish(replace(context.state, scope_fingerprint=snapshot.fingerprint))

    context._read_database_scope_metadata = counting
    context.refresh = loading_refresh
    return state


@pytest.mark.asyncio
async def test_sync_ticks_during_a_run_check_the_character_scope_once_a_second(
    monkeypatch,
):
    """AC#2: run ticks make no DB scope reads between checks; changes still load.

    The send's own message writes bump the revision, so the ticks of a run
    each reloaded the browser. The recheck interval is pinned long (a loaded
    host must not let a real second pass between ticks) and then zeroed to
    stand for "a second later".
    """
    from tldw_chatbook.UI.Console_Modules import character_context as module

    # raising=False: the base build has no interval to pin (and fails below).
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
            scope = _count_scope_reads(console)
            context = console._character_context
            try:
                context.invalidate_scope()  # An invalidated scope checks at once.
                await _tick(console)
                assert scope["reads"] >= 1, "the invalidated scope was not checked"
                first = scope["reads"]
                for _ in range(5):
                    scope["revision"] += 1  # The send writes a message.
                    await _tick(console)
                assert scope["reads"] == first, (
                    f"{scope['reads'] - first} scope reads in 5 run ticks"
                )
                # Live, writes land between a check's two reads, so it never
                # settles; an unsettled check still counts for the interval.
                scope["writes_per_read"] = 1
                context.invalidate_scope()
                await _tick(console)
                unsettled = scope["reads"]
                assert unsettled > first
                for _ in range(5):
                    await _tick(console)
                assert scope["reads"] == unsettled, (
                    f"{scope['reads'] - unsettled} reads in 5 ticks after an "
                    "unsettled check"
                )
                scope["writes_per_read"] = 0
                first = scope["reads"]
                # An ambient change (another open chat) is checked at once.
                real_open = context._open_conversation_accessor
                context._open_conversation_accessor = lambda: "another-chat"
                await _tick(console)
                assert scope["reads"] > first
                context._open_conversation_accessor = real_open
                await _tick(console)
                # A second later the run's tick reads and loads the change.
                scope["revision"] += 1
                checked = scope["reads"]
                monkeypatch.setattr(module, "SCOPE_RECHECK_DURING_RUN_SECONDS", 0.0)
                await _tick(console)
                assert scope["reads"] > checked
                assert (
                    context.state.scope_fingerprint.data_revision == scope["revision"]
                )
            finally:
                gateway.validation_release.set()
                console._start_console_transcript_sync_timer()
            await until(lambda: REPLY in "\n".join(_painted_lines(host)), timeout=30)
            await until(lambda: not console._console_run_active(), timeout=30)
            # Outside a run every sync checks, as before.
            monkeypatch.setattr(module, "SCOPE_RECHECK_DURING_RUN_SECONDS", 60.0)
            idle = scope["reads"]
            await _tick(console)
            await _tick(console)
            assert scope["reads"] >= idle + 2
