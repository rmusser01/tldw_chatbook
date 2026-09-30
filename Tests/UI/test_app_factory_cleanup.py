"""Real factory-owned handle retirement without releasing app references."""

import sqlite3

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_factory_cleanup_closes_constructor_database_handles():
    from Tests.UI import app_factory

    apps = [app_factory._build_test_app() for _ in range(3)]
    owners = [
        owner
        for app in apps
        for owner in (
            app.local_library_collections_db,
            app.local_workspace_db,
            app.subscriptions_db,
            app.evaluation_orchestrator.db,
        )
    ]
    connections = [
        connection
        for owner in owners
        for connection in owner._maintenance_participant.connections
    ]
    assert len(connections) == 12
    for connection in connections:
        assert connection.execute("SELECT 1").fetchone()[0] == 1

    app_factory.drain_active_service_patches()
    assert app_factory.drain_created_dirs() == 3

    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    assert all(not owner._maintenance_participant.connections for owner in owners)
    assert len(apps) == 3


def test_factory_cleanup_keeps_replacement_database_caller_owned():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    original = app.local_workspace_db
    connection = next(iter(original._maintenance_participant.connections))
    replacement_close = Mock()
    app.local_workspace_db = SimpleNamespace(close=replacement_close)

    assert app_factory.drain_created_dirs() == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    replacement_close.assert_not_called()
    assert app_factory.drain_created_dirs() == 0


def test_factory_cleanup_propagates_database_close_failure():
    from unittest.mock import patch

    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    database = app.evaluation_orchestrator.db
    directory = app_factory._created_dirs[-1]
    error = RuntimeError("injected constructor-owned database close failure")
    with (
        patch.object(database, "close", side_effect=error),
        pytest.raises(RuntimeError) as raised,
    ):
        app_factory.drain_created_dirs()
    assert raised.value is error
    assert directory.exists()

    assert app_factory.drain_created_dirs() == 1
    assert not directory.exists()
