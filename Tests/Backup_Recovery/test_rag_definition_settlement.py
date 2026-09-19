"""Accepted profile/config sequences settle before ordinary storage closes."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, threading, time, sys
from pathlib import Path
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant, retained_issues
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager, ExperimentConfig
from tldw_chatbook.RAG_Search.simplified import active_config as active
from tldw_chatbook.UI.Screens import settings_rag_profile_adapter as adapter
from tldw_chatbook import config
route, outcome = sys.argv[1:]
manager = get_profile_manager()
profile = manager.clone_profile('hybrid_basic', 'Private test')
active.set_active_profile(profile.id)
app = SimpleNamespace(_rag_service=None)

if route == 'probe':
    before = manager._experiment_results
    if outcome == 'active':
        experiment = ExperimentConfig(results_dir=manager.profiles_dir/'experiments'/'mine')
        manager._current_experiment = experiment
        manager._experiment_results[experiment.experiment_id] = [{'query':'unsaved private bytes'}]
    elif outcome == 'alternate':
        manager.profiles_dir = Path.home()/'alternate'
    elif outcome == 'unknown':
        app._rag_service = SimpleNamespace(profile_manager=object())
    elif outcome == 'unknown_slots':
        app._rag_service = object()
    issues = retained_issues(app)
    assert bool(issues) is (outcome != 'clean')
    assert manager._experiment_results is before
    if outcome == 'active':
        assert manager._current_experiment is experiment
        assert before[experiment.experiment_id][0]['query'] == 'unsaved private bytes'
    print('retired and reopened')
    raise SystemExit

entered, release = threading.Event(), threading.Event()
def block():
    entered.set()
    assert release.wait(5)

if route == 'delete':
    original = adapter.activate_profile
    def second(value):
        block()
        return original(value)
    adapter.activate_profile = second
    call = lambda: adapter.delete_user_profile(profile.id)
    def check():
        assert not manager._profile_path(profile.id).exists()
        assert active._active_profile_id() == 'hybrid_basic'
elif route == 'import':
    active.set_active_profile('hybrid_basic')
    config.save_setting_to_cli_config('AppRAGSearchConfig.rag', 'top_k', 12)
    original = active.set_active_profile
    def second(value):
        block()
        return original(value)
    active.set_active_profile = second
    call = active.ensure_imported_profile
    def check():
        assert manager._profile_path('imported_settings').is_file()
        assert active._active_profile_id() == 'imported_settings'
        assert active._first_run_import_done()
else:
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from dataclasses import replace
    values = replace(adapter.load_rag_defaults_from_active_profile(), default_top_k=13)
    original = SettingsConfigAdapter.save_sections
    def second(self, sections):
        block()
        return original(self, sections)
    SettingsConfigAdapter.save_sections = second
    call = lambda: SettingsScreen._persist_library_rag_save(object(), values, {'console': {'show_timestamps': False}})
    def check():
        assert adapter.load_rag_defaults_from_active_profile().default_top_k == 13
        assert config.get_cli_setting('console', 'show_timestamps') is False

async def main():
    worker = asyncio.create_task(asyncio.to_thread(call))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        participant._maintenance_close_admission()
        drain = asyncio.create_task(participant._maintenance_drain(time.monotonic()+5))
        await asyncio.sleep(0)
        assert not drain.done(), 'accepted between-write work was not retained'
        try:
            call()
        except RecoveryRequired:
            pass
        else:
            raise AssertionError('new operation was admitted')
        release.set()
        await worker
        assert await drain
        check()
    finally:
        release.set()
        await worker
        participant._maintenance_resume()
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["delete", "import", "save"])
def test_exact_accepted_sequence_finishes_before_pause(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)


@pytest.mark.parametrize(
    "outcome", ["clean", "active", "alternate", "unknown", "unknown_slots"]
)
def test_retained_state_probe_is_read_only(tmp_path, outcome):
    _run(tmp_path, "probe", outcome, script=_SCRIPT)


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_native_sequence_retained(tmp_path):
    import asyncio
    import threading
    import time

    from tldw_chatbook.Backup_Recovery.rag_definition_participant import (
        DefinitionParticipant,
    )

    owner = DefinitionParticipant()
    entered, release = threading.Event(), threading.Event()

    @owner.operation
    def write():
        entered.set()
        assert release.wait(3)
        (tmp_path / "finished").write_text("accepted")

    waiter = asyncio.create_task(asyncio.to_thread(write))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        owner._maintenance_close_admission()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not await owner._maintenance_drain(time.monotonic() + 0.03)
        release.set()
        assert await owner._maintenance_drain(time.monotonic() + 3)
        assert (tmp_path / "finished").read_text() == "accepted"
    finally:
        release.set()
        await owner._maintenance_drain(time.monotonic() + 3)
        owner._maintenance_resume()


@pytest.mark.asyncio
async def test_nested_accepted_call_and_failure_release_intake():
    import time

    from tldw_chatbook.Backup_Recovery.rag_definition_participant import (
        DefinitionParticipant,
    )

    owner = DefinitionParticipant()
    error = RuntimeError("primary failure")

    @owner.operation
    def nested():
        raise error

    @owner.operation
    def outer():
        owner._maintenance_close_admission()
        nested()

    with pytest.raises(RuntimeError) as raised:
        outer()
    assert raised.value is error
    assert await owner._maintenance_drain(time.monotonic() + 1)
    owner._maintenance_resume()


@pytest.mark.asyncio
async def test_runtime_closes_definition_intake_before_draft_probe(monkeypatch):
    import asyncio
    import time

    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant

    coordinator = object.__new__(runtime.RuntimeMaintenance)
    coordinator.app = object()
    coordinator.task = asyncio.current_task()
    coordinator.closed = []
    coordinator._settled = False
    # Isolate the definition-to-probe boundary from unrelated navigation owners.
    monkeypatch.setattr(runtime, "_app_hook", lambda app, prefix: None)

    @participant.operation
    def mutate():
        raise AssertionError("mutation admitted during draft probe")

    def probe():
        with pytest.raises(RecoveryRequired, match="rag_definition_operations_paused"):
            mutate()
        return ("unsaved",)

    coordinator.unsaved_editors = probe
    try:
        with pytest.raises(RecoveryRequired, match="needs_user_save_discard"):
            await coordinator.settle_producers(time.monotonic() + 1)
    finally:
        await runtime._resume_hooks(coordinator.closed)
