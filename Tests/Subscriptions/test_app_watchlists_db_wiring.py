"""The one-SubscriptionsDB wiring invariant the in-flight guard rests on.

task-16838 (review F2). The per-(subscription, url) in-flight guard
(`local_watchlists_service._IN_FLIGHT_URL_CHECKS`) keys its claims on
`id(db)` — so the scheduler's checks and the UI's Check Now contend with
each other **only because** `app.py`'s
`_wire_watchlists_and_notifications_services` hands the SAME
`SubscriptionsDB` object to both services (task-15463's "ONE SubscriptionsDB
for this whole wiring"). Nothing else enforced that: if the wiring ever
regressed to the pre-task-15463 per-call `lambda: SubscriptionsDB(...)`
factory — or the handler grew its own instance — the guard would silently
stop guarding across entrants **while every guard test stayed green**,
because those tests construct their own shared object. This pin is what
makes that regression loud.
"""

from __future__ import annotations

from Tests.UI.app_factory import _build_test_app


def test_scheduler_and_ui_watchlists_services_share_one_subscriptions_db():
    app = _build_test_app()

    # The UI-facing service resolves to the app's single held instance...
    assert app.local_watchlists_service._db() is app.subscriptions_db

    # ...and so does the scheduled-check handler's default-constructed
    # service. The handler must exist at all for the invariant to be
    # testable — if the harness config ever disables watchlist checks, this
    # pin must fail loudly rather than pass vacuously.
    handler = app.scheduler_loop.handlers.get("watchlist_job")
    assert handler is not None, (
        "the scheduled watchlist check handler is not wired; the guard's "
        "cross-entrant invariant cannot be pinned"
    )
    assert handler.subscriptions_db is app.subscriptions_db
    assert handler.watchlists_service._db() is app.subscriptions_db, (
        "the scheduler's service and the UI's service must resolve to the "
        "SAME SubscriptionsDB object — the in-flight guard keys on id(db), "
        "so distinct instances would never contend and concurrent checks "
        "of one source could double-report again"
    )
