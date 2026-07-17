"""TASK-251: gate the Console 0.2s sync tick.

Covers the fixes described in ``Docs/Design/2026-07-16-performance-audit.md``
(P1 B1):

1. TTL/signature-gated persisted conversation-browser rows (no DB query per
   tick when nothing changed).
2. Equality no-op guards on the settings-summary / workspace-context-tray
   sub-syncs (no forced ``Static.update()``/``refresh(recompose=True)`` when
   the built state is unchanged).
3. The run inspector renders a static "Streaming..." excerpt for the
   currently-streaming message instead of re-deriving live content every
   tick, so it stops full-recomposing 5x/second during a stream.

DEVIATION from the task-251 brief (documented in the task-251 report): the
brief's Change 3 additionally asked to skip the inspector-state build
entirely while the right rail is hidden. Measured against the real test
suite, that broke existing, intentional behavior -- Console keeps the
inspector's mounted content fresh in the background regardless of paint
visibility (`Tests/UI/test_console_native_chat_flow.py`'s
``test_console_selected_message_updates_inspector_action_guidance`` and
several workspace-conversation-resume tests assert exactly that). The
audit's actual measured complaint ("streaming-excerpt selection = 5
teardowns/s") is fully addressed by the static-excerpt fix above: the built
state stops changing tick-to-tick while streaming, so
``ConsoleRunInspector.sync_state``'s own equality guard (pre-existing,
mirrored onto the other widgets below) already skips the recompose
regardless of visibility. The hidden-skip was not implemented; see
``test_console_inspector_content_stays_fresh_while_right_rail_hidden`` below
for a regression test protecting that deliberate choice.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Screens.chat_screen import CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS
from tldw_chatbook.Widgets.Console import (
    ConsoleRunInspector,
    ConsoleSettingsSummary,
    ConsoleTranscript,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)


class _CountingConversationService:
    """Bare-bones synchronous local conversation-listing seam that counts calls."""

    def __init__(self) -> None:
        self.list_calls: list[dict[str, object]] = []

    def list_conversations(
        self,
        *,
        query: str = "",
        scope_type: str = "",
        workspace_id: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, object]:
        self.list_calls.append(
            {
                "query": query,
                "scope_type": scope_type,
                "workspace_id": workspace_id,
            }
        )
        return {
            "items": [],
            "pagination": {"total": 0, "limit": limit, "offset": offset},
        }


@pytest.mark.asyncio
async def test_console_persisted_rows_cache_gates_list_conversations_calls():
    """RED-first: repeated ticks must not re-query the DB while nothing changed.

    Drives ``_sync_native_console_chat_ui`` three times back-to-back: the
    persisted-rows query count must rise after the first call only, then stay
    flat. Explicit invalidation and TTL expiry must each force exactly one
    more fresh query.
    """
    app = _build_test_app()
    service = _CountingConversationService()
    app.local_chat_conversation_service = service
    app.chat_conversation_scope_service = None
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await pilot.pause()

        # Start from a known-clean cache regardless of what mount-time syncs
        # already did.
        console._invalidate_console_persisted_rows_cache()
        baseline = len(service.list_calls)

        await console._sync_native_console_chat_ui()
        after_first = len(service.list_calls)
        assert after_first > baseline, "first sync after invalidation must query the DB"

        await console._sync_native_console_chat_ui()
        after_second = len(service.list_calls)
        assert after_second == after_first, (
            "second back-to-back sync must be served from the TTL cache, not "
            "re-query the DB"
        )

        await console._sync_native_console_chat_ui()
        after_third = len(service.list_calls)
        assert after_third == after_first, (
            "third back-to-back sync must still be served from the TTL cache"
        )

        # Explicit invalidation forces exactly one more fresh query.
        console._invalidate_console_persisted_rows_cache()
        await console._sync_native_console_chat_ui()
        after_invalidate = len(service.list_calls)
        assert after_invalidate > after_third, (
            "explicit cache invalidation must force a fresh DB query"
        )

        # TTL expiry (equivalent to letting monotonic() advance past the TTL)
        # forces exactly one more fresh query even without explicit
        # invalidation.
        console._console_persisted_rows_cache_at -= (
            CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS + 0.5
        )
        await console._sync_native_console_chat_ui()
        after_ttl = len(service.list_calls)
        assert after_ttl > after_invalidate, (
            "a stale (TTL-expired) cache entry must force a fresh DB query"
        )


@pytest.mark.asyncio
async def test_console_settings_summary_sync_state_is_noop_when_state_unchanged():
    """Settings-summary sub-sync must not touch Static.update() when state is equal."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        summary = console.query_one("#console-settings-summary", ConsoleSettingsSummary)
        # Two separately-constructed, value-equal state snapshots (frozen
        # dataclass equality), mirroring how the screen rebuilds state fresh
        # every tick.
        state_a = console._build_console_settings_summary_state()
        state_b = console._build_console_settings_summary_state()
        assert state_a == state_b

        update_calls: list[str] = []
        original_update = Static.update

        def counting_update(self, *args, **kwargs):
            update_calls.append(str(self.id))
            return original_update(self, *args, **kwargs)

        with patch.object(Static, "update", counting_update):
            summary.sync_state(state_a)
            first_call_count = len(update_calls)
            summary.sync_state(state_b)
            assert len(update_calls) == first_call_count, (
                "re-syncing an equal-value state must not call Static.update()"
            )


@pytest.mark.asyncio
async def test_console_workspace_context_tray_sync_state_always_recomposes():
    """Regression test protecting a second deliberate Change-2 deviation.

    The brief's Change 2 asked `ConsoleWorkspaceContextTray.sync_state` to
    mirror `ConsoleRunInspector`'s `if state == self.state: return` guard.
    Measured against the real test suite, that broke click targeting on
    grouped browser rows: skipping `refresh(recompose=True)` also skips
    rebuilding the row children, and this widget's own scroll/fit-pass
    (`_schedule_recomposed_content_fit`) does not settle correct on-screen
    regions for children that were never rebuilt --
    ``test_console_workspace_conversation_search_selection_keeps_query_active``
    and ``..._invalidates_pending_worker`` both failed with "row not found"
    errors. The guard was not implemented; this locks in the (still real)
    screen-side optimization instead (see the next test) and protects
    against a future reintroduction of the widget-level guard.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)

        refresh_calls: list[int] = []
        original_refresh = ConsoleWorkspaceContextTray.refresh

        def counting_refresh(self, *args, **kwargs):
            refresh_calls.append(1)
            return original_refresh(self, *args, **kwargs)

        with patch.object(ConsoleWorkspaceContextTray, "refresh", counting_refresh):
            same_value_state = replace(tray.state)
            assert same_value_state == tray.state
            tray.sync_state(same_value_state)
            assert refresh_calls == [1], (
                "ConsoleWorkspaceContextTray.sync_state must always recompose "
                "-- an equality guard here breaks click targeting (see "
                "docstring)"
            )


@pytest.mark.asyncio
async def test_console_workspace_context_legacy_alias_kick_skipped_when_state_unchanged():
    """Screen-side no-op: the legacy-alias worker only kicks off on a real change.

    `_sync_console_workspace_context` compares the freshly-built state
    against the mounted tray's own current `.state` *before* pushing it, and
    only schedules `_sync_console_legacy_workspace_context_aliases` (a
    `run_worker` kick) when that comparison shows a real change -- reading
    off the widget's own state (rather than a screen-level cache) keeps this
    safe across a full-screen recompose, unlike the widget-level guard
    above.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        worker_calls: list[int] = []
        original_run_worker = console.run_worker

        def counting_run_worker(work, *args, **kwargs):
            if kwargs.get("group") == "console-workspace-context-legacy-aliases":
                worker_calls.append(1)
            return original_run_worker(work, *args, **kwargs)

        # Pin the built state to one fixed (but distinct-object-each-call)
        # value so two consecutive syncs are a genuine "state unchanged"
        # comparison rather than incidentally differing mount-time state.
        original_build = console._build_console_workspace_context_state

        def pinned_build(*args, **kwargs):
            return replace(original_build(*args, **kwargs), heading="Pinned heading")

        with (
            patch.object(console, "run_worker", counting_run_worker),
            patch.object(
                console, "_build_console_workspace_context_state", pinned_build
            ),
        ):
            console._sync_console_workspace_context()
            await pilot.pause()
            assert worker_calls == [1], "the first (changed) sync must kick the worker"

            console._sync_console_workspace_context()
            await pilot.pause()
            assert worker_calls == [1], (
                "re-syncing with unchanged workspace-context state must not "
                "kick the legacy-alias worker again"
            )


@pytest.mark.asyncio
async def test_console_inspector_content_stays_fresh_while_right_rail_hidden():
    """Regression test protecting the deliberate Change-3 deviation.

    The brief's Change 3 asked `_sync_console_control_bar` to skip building
    and pushing inspector state entirely while the right rail is hidden.
    That broke real, tested behavior: Console keeps the inspector's mounted
    content in sync in the background regardless of paint visibility, so a
    later reopen (or a `query_one` assertion, as several existing tests do)
    sees current data instead of stale content from before the rail was
    collapsed. This test locks that in: selecting a message while the right
    rail is closed must still update the mounted inspector's rows.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-right-rail")

        console._set_console_rail_preference(right_open=False)
        await pilot.pause()
        right_rail = console.query_one("#console-right-rail")
        assert right_rail.styles.display == "none"

        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="hidden-rail reply"
        )
        # The transcript must render the message as a row before it will
        # accept a selection against it -- sync once first (mirrors
        # `test_console_selected_message_updates_inspector_action_guidance`).
        await console._sync_native_console_chat_ui()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        assert transcript.selected_message_id == message.id
        await pilot.pause()
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        inspector = console.query_one("#console-run-inspector-state", ConsoleRunInspector)
        selected_row = next(
            (row for row in inspector.state.rows if row.label == "Selected message"),
            None,
        )
        assert selected_row is not None, (
            "inspector state must still reflect the selected message even "
            "while the right rail is hidden"
        )


@pytest.mark.asyncio
async def test_console_streaming_message_excerpt_is_static_placeholder():
    """A selected streaming message must show a static excerpt, not live content.

    Rendering the live streamed text in the inspector's Excerpt row would
    force a full inspector recompose on every 0.2s tick for the duration of
    the stream (5 teardowns/s); the transcript already shows the live text,
    so the inspector shows a stable placeholder instead.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        store.append_stream_chunk(message.id, "partial streamed response text")
        assert store.get_message(message.id).status == "streaming"

        await console._sync_native_console_chat_ui()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        # `select_message` schedules its own row refresh via `call_later`;
        # let it flush before asserting so it doesn't race the harness
        # teardown.
        await pilot.pause()

        rows = console._selected_console_message_inspector_rows()
        excerpt_row = next(row for row in rows if row.label == "Excerpt")
        assert excerpt_row.value == "Streaming…"
