"""Textual cancellation cannot retire accepted native definition/UI work."""

import asyncio
import threading
import time

import pytest
from textual import work
from textual.app import App
from textual.worker import WorkerCancelled

from tldw_chatbook.Backup_Recovery.rag_definition_participant import (
    DefinitionParticipant,
)


@pytest.mark.asyncio
async def test_cancelled_unstarted_textual_action_cannot_enter_later(tmp_path):
    from tldw_chatbook.UI.Screens.settings_rag_definition_actions import (
        QueuedDefinitionAction,
    )

    cohort = DefinitionParticipant()
    calls = []

    class TestApp(App):
        @work(thread=True, exclusive=True, group="settings-rag-profile-crud")
        def launch(self, action):
            action.execute(lambda: calls.append("native"))

    async with TestApp().run_test() as pilot:
        action = QueuedDefinitionAction(cohort=cohort)
        worker = pilot.app.launch(action)
        action.attach(worker)
        worker.cancel()
        with pytest.raises(WorkerCancelled):
            await worker.wait()
        await asyncio.sleep(0)
        cohort._maintenance_close_admission()
        assert await cohort._maintenance_drain(time.monotonic() + 1)
        action.execute(lambda: calls.append("late executor entry"))
        assert calls == []


@pytest.mark.parametrize("descendant", [False, True])
@pytest.mark.asyncio
async def test_cancelled_running_textual_action_retains_native_and_ui_delivery(
    tmp_path, descendant
):
    from tldw_chatbook.UI.Screens.settings_rag_definition_actions import (
        QueuedDefinitionAction,
    )

    cohort = DefinitionParticipant()
    entered, release = threading.Event(), threading.Event()
    child_entered, child_release = threading.Event(), threading.Event()
    saved, delivered = tmp_path / "saved", tmp_path / "delivered"

    class TestApp(App):
        @work(thread=True, exclusive=True, group="settings-rag-save")
        def launch(self, action):
            def native():
                entered.set()
                assert release.wait(5)
                cohort.operation(lambda: saved.write_text("saved bytes"))()
                action.deliver(self, self.delivery, action)

            action.execute(native)

        def delivery(self, action):
            cohort.operation(lambda: delivered.write_text("delivered result"))()
            if descendant:
                child = QueuedDefinitionAction(cohort=cohort, parent=action)
                child.attach(self.activate(child))

        @work(thread=True, exclusive=True, group="settings-rag-set-active")
        def activate(self, action):
            def native():
                child_entered.set()
                assert child_release.wait(5)
                cohort.operation(lambda: saved.write_text("activated bytes"))()

            action.execute(native)

    try:
        async with TestApp().run_test() as pilot:
            action = QueuedDefinitionAction(cohort=cohort)
            worker = pilot.app.launch(action)
            action.attach(worker)
            assert await asyncio.to_thread(entered.wait, 5)
            cohort._maintenance_close_admission()
            worker.cancel()
            with pytest.raises(WorkerCancelled):
                await worker.wait()
            assert not await cohort._maintenance_drain(time.monotonic() + 0.05)
            release.set()
            if descendant:
                assert await asyncio.to_thread(child_entered.wait, 5)
                assert not await cohort._maintenance_drain(time.monotonic() + 0.05)
                child_release.set()
            assert await cohort._maintenance_drain(time.monotonic() + 5)
            assert saved.read_text() == (
                "activated bytes" if descendant else "saved bytes"
            )
            assert delivered.read_text() == "delivered result"
    finally:
        release.set()
        child_release.set()


@pytest.mark.parametrize(
    "route",
    [
        "crud",
        "activate",
        "save",
        "confirmation",
        "save_entry",
        "switch_save",
        "switch_discard",
    ],
)
def test_paused_settings_dispatch_preserves_result_and_pending_state(tmp_path, route):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        route,
        "paused",
        script=r"""
import sys
from types import SimpleNamespace, MethodType
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import load_rag_defaults_from_active_profile
values = load_rag_defaults_from_active_profile()
def unexpected(*args, **kwargs):
    raise AssertionError('dispatch mutated or queued before admission')
screen = SimpleNamespace(
    _library_rag_profile_result='unchanged profile result',
    _library_rag_result='unchanged save result',
    _rag_profile_pending_activate='preserved pending',
    _rag_reindex_confirm_in_flight=True,
    _set_static_text=unexpected,
    _rag_profile_action_worker=unexpected,
    _rag_set_active_worker=unexpected,
    _settings_save_library_rag_worker=unexpected,
    _app_config_mapping=lambda: {},
    _rag_preview_profile_id=None,
    _category_has_unsaved_changes=lambda category: False,
    _active_category_id=lambda: SettingsCategoryId.LIBRARY_RAG,
    _sync_rag_editor_display=unexpected,
    app=SimpleNamespace(notify=unexpected),
)
screen._dispatch_library_rag_save = MethodType(SettingsScreen._dispatch_library_rag_save, screen)
participant._maintenance_close_admission()
try:
    route = sys.argv[1]
    if route == 'crud':
        SettingsScreen._dispatch_rag_profile_action(screen, 'rename', 'id', 'new')
    elif route == 'activate':
        SettingsScreen._dispatch_rag_set_active(screen, 'id')
    elif route == 'save':
        screen._dispatch_library_rag_save(values, False, 'next')
    elif route == 'confirmation':
        SettingsScreen._handle_reindex_confirmation_result(screen, True, values, 'next')
    elif route == 'save_entry':
        SettingsScreen.action_settings_save_category(screen, allow_text_entry_focus=True)
    else:
        SettingsScreen._handle_rag_profile_switch_confirm(screen, route.removeprefix('switch_'), 'next')
except RecoveryRequired:
    pass
else:
    raise AssertionError('closed dispatch admitted')
assert screen._library_rag_profile_result == 'unchanged profile result'
assert screen._library_rag_result == 'unchanged save result'
assert screen._rag_profile_pending_activate == 'preserved pending'
assert screen._rag_reindex_confirm_in_flight is True
print('retired and reopened')
""",
    )


_SETTINGS_SCRIPT = r"""
import asyncio, json, sys, threading, time
from dataclasses import replace
from pathlib import Path
from Tests import network_guard
network_guard.install()
from textual.app import App
from textual.screen import Screen
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant
from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager
from tldw_chatbook.RAG_Search.simplified import active_config as active
from tldw_chatbook.UI.Screens import settings_screen as module
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens import settings_rag_profile_adapter as adapter
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

route = sys.argv[1]
manager = get_profile_manager()
first = manager.clone_profile('bm25_only', 'Original')
second = manager.clone_profile('bm25_only', 'Next')
active.set_active_profile(first.id)
values = replace(adapter.load_rag_defaults_from_active_profile(), default_top_k=13,
                 chunk_size=777, direct_library_tools=False)
entered, release = threading.Event(), threading.Event()
def block():
    entered.set()
    assert release.wait(5)

if route == 'save':
    original = SettingsConfigAdapter.save_sections
    def save(self, sections):
        block()
        return original(self, sections)
    SettingsConfigAdapter.save_sections = save
elif route == 'crud':
    original = module.clone_profile_as
    def clone(*args):
        block()
        return original(*args)
    module.clone_profile_as = clone
else:
    original = module.fetch_index_status
    calls = []
    def status():
        calls.append(True)
        if len(calls) == 1:
            block()
        return {'state': 'built', 'count': 3} if route == 'modal' else original()
    module.fetch_index_status = status

class Harness(Screen):
    def __init__(self):
        super().__init__()
        self._rag_preview_profile_id = None
        self._rag_profile_pending_activate = None
        self._settings_drafts = {SettingsCategoryId.LIBRARY_RAG: 'unsaved draft'}
        self._library_rag_index_status_cache = None
        self._rag_reindex_confirm_in_flight = False
        self._library_rag_result = 'unchanged'
        self._library_rag_profile_result = 'unchanged'
    def _set_static_text(self, *args): pass
    def _app_config_mapping(self): return config._CONFIG_CACHE
    def _app_config_update_target(self): return config._CONFIG_CACHE
    def _active_category_id(self): return SettingsCategoryId.LIBRARY_RAG
    def _apply_library_rag_index_status(self, status): self._library_rag_index_status_cache = status
    def _library_rag_profile_name(self, profile_id): return manager.get_profile(profile_id).name

for name in ('_dispatch_library_rag_save', '_settings_save_library_rag_worker',
             '_persist_library_rag_save', '_apply_library_rag_save_result',
             '_dispatch_rag_set_active', '_rag_set_active_worker', '_rag_after_set_active',
             '_confirm_reindex_then_save', '_decide_reindex_confirmation',
             '_rag_reindex_confirm_status_worker', '_clear_rag_reindex_confirm_in_flight',
             '_handle_reindex_confirmation_result', '_dispatch_rag_profile_action',
             '_rag_profile_action_worker', '_rag_after_profile_action'):
    setattr(Harness, name, getattr(module.SettingsScreen, name))
for name in ('_sync_library_rag_widgets', '_sync_library_rag_profile_widgets',
             '_update_library_rag_editor_title', '_set_library_rag_preview_banner',
             '_update_draft_status_widgets', '_refresh_library_rag_index_status',
             '_refresh_rag_first_run_panel_state'):
    setattr(Harness, name, lambda self, *args, **kwargs: None)

class TestApp(App):
    def on_mount(self): self.push_screen(Harness())

async def main():
    try:
        async with TestApp().run_test() as pilot:
            screen = pilot.app.screen
            before = manager._profile_path(first.id).read_bytes()
            if route == 'save':
                screen._dispatch_library_rag_save(values, False, second.id)
            elif route == 'crud':
                screen._dispatch_rag_profile_action('clone', first.id, 'Accepted clone')
            else:
                assert module.index_change_pending(values)
                screen._confirm_reindex_then_save(values, second.id)
            assert await asyncio.to_thread(entered.wait, 5)
            participant._maintenance_close_admission()
            for worker in list(pilot.app.workers):
                worker.cancel()
            assert not await participant._maintenance_drain(time.monotonic()+0.05)
            release.set()
            assert await participant._maintenance_drain(time.monotonic()+5)
            if route == 'modal':
                assert isinstance(pilot.app.screen, ConfirmationDialog)
                assert screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] == 'unsaved draft'
                assert manager._profile_path(first.id).read_bytes() == before
                assert screen._rag_reindex_confirm_in_flight is False
                previous = (screen._library_rag_result, screen._rag_profile_pending_activate)
                try:
                    screen._handle_reindex_confirmation_result(True, values, second.id)
                except RecoveryRequired:
                    pass
                else:
                    raise AssertionError('confirmation inherited modal acceptance')
                assert previous == (screen._library_rag_result, screen._rag_profile_pending_activate)
            elif route == 'crud':
                assert any(profile.name == 'Accepted clone' for profile in manager._profiles.values())
                assert "Cloned to 'Accepted clone'" in screen._library_rag_profile_result
            else:
                data = json.loads(manager._profile_path(first.id).read_bytes())
                assert data['rag_config']['search']['default_top_k'] == 13
                assert config.get_cli_setting('console', 'direct_library_tools') is False
                assert active._active_profile_id() == second.id
                assert SettingsCategoryId.LIBRARY_RAG not in screen._settings_drafts
                assert screen._rag_profile_pending_activate is None
                assert 'Active profile: Next' in screen._library_rag_profile_result
    finally:
        release.set()
        participant._maintenance_resume()
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["save", "status", "modal", "crud"])
def test_actual_settings_native_write_and_delivery_chains(tmp_path, route):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, route, "success", script=_SETTINGS_SCRIPT)


@pytest.mark.asyncio
async def test_tokens_reject_foreign_or_retired_parent_and_hold_entered_scope():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    cohort, other = DefinitionParticipant(), DefinitionParticipant()
    token, foreign = cohort.reserve(), other.reserve()
    cohort._maintenance_close_admission()
    with pytest.raises(RecoveryRequired):
        cohort.reserve()
    with pytest.raises(RecoveryRequired):
        cohort.reserve(foreign)
    child = cohort.reserve(token)
    token.close()
    with pytest.raises(RecoveryRequired):
        cohort.reserve(token)
    with child.scope():
        child.close()
        assert not await cohort._maintenance_drain(time.monotonic() + 0.02)
        assert cohort.operation(lambda: "accepted")() == "accepted"
    assert await cohort._maintenance_drain(time.monotonic() + 1)
    with pytest.raises(RecoveryRequired):
        cohort.operation(lambda: "fresh")()
    foreign.close()


@pytest.mark.asyncio
async def test_enqueue_failure_retires_without_cancellation_delivery():
    from tldw_chatbook.UI.Screens.settings_rag_definition_actions import (
        QueuedDefinitionAction,
    )

    cohort = DefinitionParticipant()
    changes = []
    action = QueuedDefinitionAction(
        cohort=cohort, on_cancel=lambda: changes.append(True)
    )

    def failed(*args, **kwargs):
        raise OSError("enqueue failed")

    with pytest.raises(OSError, match="enqueue failed"):
        action.enqueue(failed)
    cohort._maintenance_close_admission()
    assert await cohort._maintenance_drain(time.monotonic() + 1)
    assert changes == []


@pytest.mark.asyncio
async def test_exclusive_replacement_cancels_only_not_started_action():
    from tldw_chatbook.UI.Screens.settings_rag_definition_actions import (
        QueuedDefinitionAction,
    )

    cohort = DefinitionParticipant()
    calls = []

    class TestApp(App):
        @work(thread=True, exclusive=True, group="settings-rag-index-status")
        def launch(self, action, name):
            action.execute(lambda: calls.append(name))

    async with TestApp().run_test() as pilot:
        first = QueuedDefinitionAction(
            cohort=cohort, on_cancel=lambda: calls.append("cleared")
        )
        first.attach(pilot.app.launch(first, "cancelled native"))
        second = QueuedDefinitionAction(cohort=cohort)
        worker = second.attach(pilot.app.launch(second, "replacement native"))
        await worker.wait()
        await asyncio.sleep(0)
        cohort._maintenance_close_admission()
        assert await cohort._maintenance_drain(time.monotonic() + 1)
        assert sorted(calls) == ["cleared", "replacement native"]
