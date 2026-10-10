"""Unit mechanisms compiled from installed source; native UI tests are separate.

Each case has its own asyncio.run lifetime, including unconditional cancellation
of any still-pending in-memory Task on failure. No App/Textual/native module is
imported here, and no config mapping, permission or display proof is fabricated.
"""

import ast
import asyncio
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


source = Path(__file__).resolve().parents[2] / "tldw_chatbook/UI/Screens/chat_screen.py"
source_bytes = source.read_bytes()
source_digest = hashlib.sha256(source_bytes).hexdigest()
tree = ast.parse(source_bytes)
screen = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChatScreen"
)
guard = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_console_screen_is_torn_down"
)
methods = {
    n.name: n
    for n in screen.body
    if isinstance(n, ast.AsyncFunctionDef)
    and n.name
    in {
        "_sync_native_console_chat_ui",
        "_sync_console_native_session_tabs",
        "_console_sync_maintenance_drain",
    }
}
minimal = ast.Module(
    body=[
        guard,
        ast.ClassDef(
            name="CompiledSourceScreen",
            bases=[],
            keywords=[],
            body=list(methods.values()),
            decorator_list=[],
        ),
    ],
    type_ignores=[],
)
ast.fix_missing_locations(minimal)
namespace = {
    "asyncio": asyncio,
    "time": time,
    "Any": Any,
    "QueryError": LookupError,
    "ConsoleSessionSurface": object,
}
exec(compile(minimal, str(source), "exec", dont_inherit=True), namespace)
SourceScreen = namespace["CompiledSourceScreen"]
SourceScreen._console_chat_store = property(lambda owner: owner.store)


class Surface:
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls, self.concurrent, self.high_water, self.error = [], 0, 0, None

    async def sync_sessions(self, **values):
        self.concurrent += 1
        self.high_water = max(self.high_water, self.concurrent)
        self.calls.append(values)
        self.entered.set()
        try:
            await self.release.wait()
            if self.error is not None:
                raise self.error
        finally:
            self.concurrent -= 1


def make_screen():
    owner = SourceScreen()
    owner._console_sync_in_progress = True  # An already admitted older pass.
    owner._console_sync_requested = False
    owner._console_sync_maintenance_paused = False
    owner._console_control_bar_replay_whole_sync = False
    owner._console_session_tabs_sync_lock = asyncio.Lock()
    owner._console_session_tabs_sync_calls = 0
    owner._closing = owner._closed = False
    owner.app_instance = SimpleNamespace(_console_runtime_shutdown_task=None)
    owner._console_chat_controller = None
    owner.store = make_store("first")
    owner.surface = Surface()
    owner.query_one = lambda *_: owner.surface
    owner._ensure_console_chat_store = lambda: owner.store
    owner.ensure_calls = []
    owner._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: owner.ensure_calls.append(True)
    )
    owner._fleet = SimpleNamespace(prepare_session_run_markers=lambda *_: {})
    owner._maybe_show_fleet_coachmark = lambda *_: None
    return owner


async def cancel(task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def coalesced_completion():
    owner = make_screen()
    task = asyncio.create_task(owner._sync_native_console_chat_ui())
    await asyncio.sleep(0)
    assert owner.surface.entered.is_set() and not task.done()
    assert owner._console_sync_requested and owner._console_session_tabs_sync_calls == 1
    owner.surface.release.set()
    await task
    assert (
        owner._console_session_tabs_sync_calls == 0
        and not owner._console_session_tabs_sync_lock.locked()
    )


async def fresh_after_serial_wait():
    owner = make_screen()
    first = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    second = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    assert len(owner.surface.calls) == 1 and owner._console_session_tabs_sync_calls == 2
    owner.store.current, owner.store.active_session_id = (
        [SimpleNamespace(id="new-owner", settings=object())],
        "new-owner",
    )
    owner.surface.release.set()
    await asyncio.gather(first, second)
    assert owner.surface.high_water == 1
    assert owner.surface.calls[0]["active_session_id"] == "first"
    assert owner.surface.calls[-1]["active_session_id"] == "new-owner"
    assert [item.id for item in owner.surface.calls[-1]["sessions"]] == ["new-owner"]
    assert owner._console_session_tabs_sync_calls == 0


async def drain_without_whole_boolean():
    owner = make_screen()
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    owner._console_sync_in_progress = False
    owner._console_sync_maintenance_paused = True
    drain = asyncio.create_task(
        owner._console_sync_maintenance_drain(time.monotonic() + 1.0)
    )
    await asyncio.sleep(0)
    try:
        assert not drain.done() and owner._console_session_tabs_sync_calls == 1
    finally:
        owner.surface.release.set()
        await task
    assert await drain is True and owner._console_session_tabs_sync_calls == 0


async def queued_cancellation():
    owner = make_screen()
    first = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    queued = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    assert owner._console_session_tabs_sync_calls == 2
    await cancel(queued)
    assert owner._console_session_tabs_sync_calls == 1 and not first.done()
    assert len(owner.surface.calls) == 1
    owner.surface.release.set()
    await first
    assert owner._console_session_tabs_sync_calls == 0


async def publication_cancellation():
    owner = make_screen()
    task = asyncio.create_task(owner._sync_native_console_chat_ui())
    await asyncio.sleep(0)
    assert owner.surface.entered.is_set()
    await cancel(task)
    assert (
        owner._console_session_tabs_sync_calls == 0
        and not owner._console_session_tabs_sync_lock.locked()
    )
    assert owner.surface.concurrent == 0


async def live_error():
    owner = make_screen()
    error = ValueError("original-surface-error")
    owner.surface.error = error
    owner.surface.release.set()
    try:
        await owner._sync_native_console_chat_ui()
    except ValueError as actual:
        assert actual is error
    else:
        raise AssertionError("live surface error was suppressed")
    assert (
        owner._console_session_tabs_sync_calls == 0
        and not owner._console_session_tabs_sync_lock.locked()
    )


async def teardown_error():
    owner = make_screen()
    owner.surface.error = ValueError("original-teardown-error")
    task = asyncio.create_task(owner._sync_native_console_chat_ui())
    await asyncio.sleep(0)
    assert owner.surface.entered.is_set()
    owner._closing = True
    owner.surface.release.set()
    await task
    assert (
        owner._console_session_tabs_sync_calls == 0
        and not owner._console_session_tabs_sync_lock.locked()
    )


async def maintenance_gate():
    owner = make_screen()
    owner._console_sync_maintenance_paused = True
    await owner._sync_native_console_chat_ui()
    assert owner._console_sync_requested and not owner.surface.calls


async def replay_gate():
    owner = make_screen()
    owner._console_control_bar_replay_whole_sync = True
    await owner._sync_native_console_chat_ui()
    assert owner._console_sync_requested and not owner.surface.calls


async def dead_before_lock():
    owner = make_screen()
    await owner._console_session_tabs_sync_lock.acquire()
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await asyncio.sleep(0)
    assert owner._console_session_tabs_sync_calls == 1
    owner._closing = True
    owner._console_session_tabs_sync_lock.release()
    await task
    assert not owner.surface.calls and owner._console_session_tabs_sync_calls == 0


def make_store(active_id):
    rows = (
        [] if active_id is None else [SimpleNamespace(id=active_id, settings=object())]
    )
    store = SimpleNamespace(active_session_id=active_id, current=rows)
    store.sessions = lambda: list(store.current)
    return store


async def established_settings_reuse():
    owner = make_screen()
    existing = owner.store.current[0].settings
    owner.surface.release.set()
    await owner._sync_console_native_session_tabs()
    assert not owner.ensure_calls
    assert owner.store.current[0].settings is existing
    assert len(owner.surface.calls) == 1


async def missing_settings_call_through():
    owner = make_screen()
    row = owner.store.current[0]
    row.settings = None
    supplied = object()

    def ensure():
        owner.ensure_calls.append(True)
        row.settings = supplied

    owner._session._ensure_active_console_session_settings = ensure
    owner.surface.release.set()
    await owner._sync_console_native_session_tabs()
    assert owner.ensure_calls == [True] and row.settings is supplied
    assert owner.surface.calls[0]["sessions"][0] is row


async def blank_creation_call_through():
    owner = make_screen()
    owner.store = make_store(None)
    created = SimpleNamespace(id="created", settings=object())

    def ensure():
        owner.ensure_calls.append(True)
        owner.store.current = [created]
        owner.store.active_session_id = created.id

    owner._session._ensure_active_console_session_settings = ensure
    owner.surface.release.set()
    await owner._sync_console_native_session_tabs()
    assert owner.ensure_calls == [True]
    assert owner.surface.calls[0]["active_session_id"] == created.id
    assert owner.surface.calls[0]["sessions"][0] is created


async def ordered_resume_no_creation():
    owner = make_screen()
    owner.store = make_store(None)
    owner.surface.release.set()
    await owner._sync_console_native_session_tabs()
    assert owner.ensure_calls == [True]
    assert owner.surface.calls[0]["sessions"] == []
    assert owner.surface.calls[0]["active_session_id"] is None


async def ensure_owner_replacement():
    owner = make_screen()
    owner.store.current[0].settings = None
    replacement = make_store("replacement")

    def ensure():
        owner.ensure_calls.append(True)
        owner.store = replacement

    owner._session._ensure_active_console_session_settings = ensure
    owner.surface.release.set()
    await owner._sync_console_native_session_tabs()
    assert owner.ensure_calls == [True]
    assert owner.surface.calls[0]["sessions"][0] is replacement.current[0]
    assert owner.surface.calls[0]["active_session_id"] == "replacement"


async def same_id_session_replacement_during_surface_await():
    owner = make_screen()
    previous = owner.store.current[0]
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    replacement = SimpleNamespace(id=previous.id, settings=previous.settings)
    assert replacement == previous and replacement is not previous
    owner.store.current = [replacement]
    owner.surface.release.set()
    await task
    assert len(owner.surface.calls) == 2
    assert owner.surface.calls[0]["sessions"][0] is previous
    assert owner.surface.calls[-1]["sessions"][0] is replacement
    assert not owner.ensure_calls


async def store_replacement_during_surface_await():
    owner = make_screen()
    previous = owner.store
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner.store = make_store("first")
    owner.surface.release.set()
    await task
    assert len(owner.surface.calls) == 2
    assert owner.surface.calls[0]["sessions"][0] is previous.current[0]
    assert owner.surface.calls[-1]["sessions"][0] is owner.store.current[0]


async def ordered_membership_change_during_surface_await():
    owner = make_screen()
    second = SimpleNamespace(id="second", settings=object())
    owner.store.current.append(second)
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner.store.current.reverse()
    owner.surface.release.set()
    await task
    assert len(owner.surface.calls) == 2
    assert owner.surface.calls[0]["sessions"][0].id == "first"
    assert owner.surface.calls[-1]["sessions"][0] is second


async def active_change_during_surface_await():
    owner = make_screen()
    owner.store.current.append(SimpleNamespace(id="second", settings=object()))
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner.store.active_session_id = "second"
    owner.surface.release.set()
    await task
    assert len(owner.surface.calls) == 2
    assert owner.surface.calls[0]["active_session_id"] == "first"
    assert owner.surface.calls[-1]["active_session_id"] == "second"


async def surface_replacement_during_surface_await():
    owner = make_screen()
    previous = owner.surface
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await previous.entered.wait()
    owner.surface = Surface()
    owner.surface.release.set()
    previous.release.set()
    await task
    assert len(previous.calls) == 1 and len(owner.surface.calls) == 1
    assert owner._console_session_tabs_sync_calls == 0


async def teardown_during_surface_await():
    owner = make_screen()
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner._closing = True

    def unavailable(*_):
        raise AssertionError("query after teardown")

    owner.query_one = unavailable
    owner.surface.release.set()
    await task
    assert owner._console_session_tabs_sync_calls == 0


class ReplaySurface(Surface):
    def __init__(self):
        super().__init__()
        self.replay_entered, self.replay_release = asyncio.Event(), asyncio.Event()

    async def sync_sessions(self, **values):
        self.concurrent += 1
        self.high_water = max(self.high_water, self.concurrent)
        self.calls.append(values)
        try:
            if len(self.calls) == 1:
                self.entered.set()
                await self.release.wait()
            else:
                self.replay_entered.set()
                await self.replay_release.wait()
        finally:
            self.concurrent -= 1


async def replay_cancellation_retires_one_count():
    owner = make_screen()
    owner.surface = ReplaySurface()
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner.store.current.append(SimpleNamespace(id="new", settings=object()))
    owner.surface.release.set()
    await owner.surface.replay_entered.wait()
    assert owner._console_session_tabs_sync_calls == 1
    await cancel(task)
    assert owner._console_session_tabs_sync_calls == 0
    assert not owner._console_session_tabs_sync_lock.locked()
    assert owner.surface.concurrent == 0


async def replay_keeps_drain_owned():
    owner = make_screen()
    owner.surface = ReplaySurface()
    task = asyncio.create_task(owner._sync_console_native_session_tabs())
    await owner.surface.entered.wait()
    owner.store.current.append(SimpleNamespace(id="new", settings=object()))
    owner.surface.release.set()
    await owner.surface.replay_entered.wait()
    owner._console_sync_in_progress = False
    owner._console_sync_maintenance_paused = True
    drain = asyncio.create_task(
        owner._console_sync_maintenance_drain(time.monotonic() + 1)
    )
    await asyncio.sleep(0)
    assert not drain.done() and owner._console_session_tabs_sync_calls == 1
    owner.surface.replay_release.set()
    await task
    assert await drain is True
    assert owner._console_session_tabs_sync_calls == 0


_CONTROLS = (
    coalesced_completion,
    fresh_after_serial_wait,
    drain_without_whole_boolean,
    queued_cancellation,
    publication_cancellation,
    live_error,
    teardown_error,
    maintenance_gate,
    replay_gate,
    dead_before_lock,
    established_settings_reuse,
    missing_settings_call_through,
    blank_creation_call_through,
    ordered_resume_no_creation,
    ensure_owner_replacement,
    same_id_session_replacement_during_surface_await,
    store_replacement_during_surface_await,
    ordered_membership_change_during_surface_await,
    active_change_during_surface_await,
    surface_replacement_during_surface_await,
    teardown_during_surface_await,
    replay_cancellation_retires_one_count,
    replay_keeps_drain_owned,
)


@pytest.mark.unit
@pytest.mark.parametrize("control", _CONTROLS, ids=lambda control: control.__name__)
def test_console_tab_publication_source_ownership(control):
    """Run original method/guard bodies against explicit in-memory owners."""
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    before = set(sys.modules)
    try:
        asyncio.run(asyncio.wait_for(control(), timeout=1))
    finally:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
        assert not (set(sys.modules) - before).intersection(
            {"tldw_chatbook.app", "tldw_chatbook.UI.Screens.chat_screen", "textual"}
        )
