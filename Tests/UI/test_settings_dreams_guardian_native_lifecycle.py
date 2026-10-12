"""Actual Settings getter construction retires only finite worker acquisitions."""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import Tests.UI.app_factory  # noqa: F401 - source-bound config admission
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_settings_dreams import _read_toml, _seed_config
from tldw_chatbook.Backup_Recovery.participants import _repository_participant
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    "kind,enabled", [("guardian", True), ("dreams", True), ("dreams", False)]
)
@pytest.mark.parametrize("mode", ["new", "borrowed", "cancelled"])
async def test_settings_real_getter_constructor_lifetime(
    tmp_path, monkeypatch, kind, enabled, mode
):
    path = tmp_path / f"{kind}.sqlite"
    initial = not enabled or mode == "borrowed"
    config = _seed_config(
        f'[general]\nusers_name = "t"\n[{kind}]\nenabled = {str(initial).lower()}\n'
    )
    if kind == "guardian":
        monkeypatch.setattr(
            "tldw_chatbook.Guardian.settings.guardian_db_path", lambda: path
        )
        cls = GuardianDB
    else:
        monkeypatch.setattr(
            "tldw_chatbook.app_service_wiring.get_dreams_db_path", lambda: path
        )
        cls = DreamsDB
    app = _build_test_app()
    screen = SettingsScreen(app)
    callback = getattr(screen, f"_persist_{kind}_toggle")
    getter = getattr(app, f"get_{kind}_db")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    captured = []
    owners = []
    original = cls._initialize_schema
    get_connection = cls._get_connection

    def acquire(owner):
        connection = get_connection(owner)
        captured.append(connection)
        return connection

    monkeypatch.setattr(cls, "_get_connection", acquire)

    def initialize(owner):
        original(owner)
        captured.append(owner._held_connection())
        owners.append(owner)
        assert threading.current_thread() is not threading.main_thread()
        if mode == "cancelled":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)

    monkeypatch.setattr(cls, "_initialize_schema", initialize)
    pool = ThreadPoolExecutor(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(pool)
    operation = None
    try:
        if mode == "borrowed":
            await loop.run_in_executor(pool, getter)
        operation = asyncio.create_task(callback(enabled))
        if mode == "cancelled":
            await asyncio.wait_for(entered.wait(), 3)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            participant = _repository_participant(owners[0])
            participant.close_admission()
            assert not participant.drain(time.monotonic() + 0.02)
            participant.resume()
            release.set()
        else:
            await asyncio.wait_for(operation, 5)

        def probe():
            assert captured and owners
            for connection in captured:
                if mode == "borrowed":
                    assert connection.execute("SELECT 1").fetchone()[0] == 1
                else:
                    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                        connection.execute("SELECT 1")
            return threading.get_ident()

        worker = await loop.run_in_executor(pool, probe)
        assert worker == await loop.run_in_executor(pool, threading.get_ident)
        owner = getattr(app, f"{kind}_db")
        assert owner is owners[0]  # The app's cached object survives retirement.
        assert _read_toml(config)[kind]["enabled"] is (
            True if mode == "cancelled" else enabled
        )
        # An ordinary UI-thread read may own a separate handle. Retire only that
        # handle before inspecting the installed recovery participant.
        owner._held_connection().execute("SELECT 1")
        owner.close()
        participant = _repository_participant(owner)
        participant.close_admission()
        assert participant.drain(time.monotonic() + 0.02) is (mode != "borrowed")
        participant.resume()
        if mode == "borrowed":
            await loop.run_in_executor(pool, owner.close)
            participant.close_admission()
            assert participant.drain(time.monotonic() + 0.1)
            participant.resume()
    finally:
        release.set()
        if operation is not None:
            await asyncio.gather(operation, return_exceptions=True)
        for owner in owners:
            _repository_participant(owner).resume()
            await loop.run_in_executor(pool, owner.close)
            owner.close()
        loop._default_executor = previous
        pool.shutdown(wait=True)
