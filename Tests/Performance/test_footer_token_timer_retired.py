"""task-21133: the 10 s footer token-count producer stays retired.

task-17653 removed the counter's entire consumer surface -- no screen
composes an armed ``AppFooterStatus`` (``BaseAppScreen`` is the package's one
construction site and passes ``show_token_count=False``), so
``update_token_count`` is a no-op everywhere -- but the producer kept
ticking. Measured on the pin (mounted ChatScreen, isolated profile): every
tick resolved the active footer, then attempted three ``query_one``
selectors that no live screen composes (``#chat-api-provider``,
``#chat-log``, ``#chat-custom-token-limit``, each raising ``NoMatches``
after a full subtree walk), ran the estimator over the empty history that
the missing ``#chat-log`` guarantees, and threw the answer into a debug log
-- 6 times a minute, forever, for a widget that cannot appear.

These tests pin the retirement at each layer it could be reintroduced:
the scheduling site, the app/manager methods, the deleted producer module,
and the mounted footer chip itself.
"""

from __future__ import annotations

import importlib.util

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.app import TldwCli


def test_no_token_count_display_entry_points_remain():
    """The timer's call chain is gone at both of its seams."""
    assert not hasattr(TldwCli, "update_token_count_display")
    assert not hasattr(TldwCli, "_token_count_update_timer")
    assert not hasattr(DBStatusManager, "update_token_count_display")


def test_periodic_token_producer_module_is_gone():
    """`chat_token_events` had no production caller left once the timer went.

    Its four public functions were reachable only from the periodic path
    (``update_chat_token_counter``) or from two handlers nothing dispatches
    (``handle_chat_input_changed`` / ``handle_model_or_provider_changed`` --
    zero references anywhere in the package, verified before deletion), and
    its two private helpers only from those.
    """
    assert (
        importlib.util.find_spec(
            "tldw_chatbook.Event_Handlers.Chat_Events.chat_token_events"
        )
        is None
    )


def test_footer_status_scheduling_arms_only_the_db_size_timer():
    """`_schedule_footer_status_updates` no longer arms an app-level interval.

    The DB-size interval is owned by ``DBStatusManager``; the app's own
    ``set_interval`` was used by the token timer alone.
    """
    from types import SimpleNamespace

    scheduled_once: list[tuple[float, object]] = []
    scheduled_periodic: list[tuple[str, float]] = []

    def refuse_set_interval(*_args, **_kwargs):
        raise AssertionError("no App-level interval may be armed here")

    fake_app = SimpleNamespace(
        ui_responsiveness_monitor=None,
        query_one=lambda _selector: object(),
        loguru_logger=SimpleNamespace(
            info=lambda *_a, **_k: None,
            debug=lambda *_a, **_k: None,
        ),
        db_status_manager=SimpleNamespace(
            start_periodic_updates=lambda interval: scheduled_periodic.append(
                ("db", interval)
            )
        ),
        update_db_sizes=lambda: None,
        call_after_refresh=lambda callback: callback,
        set_timer=lambda delay, callback: scheduled_once.append((delay, callback)),
        set_interval=refuse_set_interval,
        _record_footer_timer_created=lambda _name: None,
    )

    TldwCli._schedule_footer_status_updates(fake_app)

    assert scheduled_periodic == [("db", 120)]
    assert [callback for _delay, callback in scheduled_once] == [
        fake_app.update_db_sizes
    ]


@pytest.mark.asyncio
async def test_booted_app_arms_no_token_timer_and_shows_no_token_chip():
    """A real boot: no token timer, and the footer chip stays hidden."""
    app = _build_test_app("chat")

    async with app.run_test(size=(160, 44)) as pilot:
        footer = None
        for _ in range(60):
            await pilot.pause(0.05)
            footer = app._active_footer_status()
            if footer is not None:
                break
        assert footer is not None, "no AppFooterStatus resolved on the booted app"

        # Force the footer timers to be wired now rather than waiting on the
        # deferred-startup timer, then assert what they armed.
        app._schedule_footer_status_updates()
        await pilot.pause()

        assert not hasattr(app, "_token_count_update_timer")

        assert isinstance(footer, AppFooterStatus)
        assert footer._show_token_count is False
        token_chip = footer.query_one("#footer-token-count")
        assert token_chip.display is False
        assert str(token_chip.renderable) == ""

        # Shutdown walk: both quit hooks call this pair, and it must stay
        # safe to call with no timer handle left to stop -- and twice, since
        # `on_shutdown_request` and `on_unmount` both run on a normal quit.
        app.db_status_manager.stop_periodic_updates()
        app._stop_footer_status_timers()
        app._stop_footer_status_timers()
        await pilot.pause()
