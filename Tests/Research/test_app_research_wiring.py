"""App-level research service wiring (task-16332).

The research wiring used to exist VERBATIM TWICE in app.py: once inside
`_wire_watchlists_and_notifications_services` (the broad parity bootstrap,
which runs first at startup) and once in `_wire_research_services` (which
then early-returned via its already-wired guard). task-16332 replaced the
embedded copy with a call to the method. These tests pin the contract that
made that replacement safe: the boot path wires the full research service
set exactly once, and a second `_wire_research_services()` call never
reconstructs them (the guard holds -- no torn/duplicate wiring).
"""

from Tests.UI.app_factory import _build_test_app


def test_boot_wires_full_research_service_set():
    app = _build_test_app()

    assert app.local_research_service is not None
    assert app.server_research_service is not None
    assert app.research_scope_service is not None
    assert app.local_research_search_service is not None
    assert app.server_research_search_service is not None
    assert app.research_search_scope_service is not None


def test_second_wire_research_services_call_reuses_existing_services():
    app = _build_test_app()

    scope_before = app.research_scope_service
    search_scope_before = app.research_search_scope_service
    local_before = app.local_research_service

    app._wire_research_services()

    # The guard must treat the already-wired state as done: same instances,
    # not fresh reconstructions (a duplicate wiring would silently detach
    # every consumer holding the originals).
    assert app.research_scope_service is scope_before
    assert app.research_search_scope_service is search_scope_before
    assert app.local_research_service is local_before
