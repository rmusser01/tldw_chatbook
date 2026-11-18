"""The original Settings read does not provision an unused Windows index."""

import asyncio
from collections import Counter
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_console_unused_project_index import (
    _console_index,
    _observe_preparation,
    _assert_retired,
    _index_batch_open,
    pytestmark as pytestmark,
)
from Tests.Backup_Recovery.test_permission_read_operation import _observe
from Tests.private_profile import private_profile_test
from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Widgets.Settings_Widgets.project_index_binding_card import (
    ProjectIndexBindingCard,
    index_status_copy,
)
from tldw_chatbook.Workspaces.project_index import runtime
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexPolicy,
    ProjectIndexRetirementUnproven,
)


def _card(root):
    return ProjectIndexBindingCard(
        root.binding_id, workspace_id=root.workspace_id, root=str(root.root)
    )


@pytest.mark.parametrize("action", ["enable", "clear"])
@private_profile_test
async def test_passive_settings_preserves_absence_until_explicit_action(
    tmp_path, request, action
):
    async with _console_index(tmp_path) as (controller, roots, base):
        (root,) = roots
        card = _card(root)
        counts, pins = Counter(), []
        codes = {
            WorkspaceProjectIndexDB.__init__.__code__: "constructor",
            BindingLifetime.acquire.__func__.__code__: "lifetime",
        }

        def observe(frame, event, result):
            if event == "call" and frame.f_code in codes:
                counts[codes[frame.f_code]] += 1
            if (
                event == "return"
                and frame.f_code is runtime._AdmittedStoreFactory.retain.__code__
                and result is not None
            ):
                pins.append(result)

        def load():
            with _observe(observe):
                return card._load(controller.app)

        status, policy, upgrade = await asyncio.to_thread(load)
        assert status.state == "Off" and not upgrade
        assert policy == ProjectIndexPolicy((), ())
        assert index_status_copy(status) == (
            "Keyword index: Off · 0 discovered · 0 indexed · ~0 KiB · "
            "Last scan activity: never"
        )
        assert pins and all(pin.closed for pin in pins)
        assert not counts and not base.exists()
        observed = await asyncio.to_thread(_observe_preparation, controller, roots)
        _assert_retired(observed)
        assert not observed.counts["index_constructor"] and not base.exists()
        event = card.ActionRequested(
            root.binding_id, root.workspace_id, status, action, policy
        )
        result, _, _ = await asyncio.to_thread(card._execute, controller.app, event)
        assert result.state == ("Paused" if action == "enable" else "Off")
        assert result.scope.epoch == 1 and base.is_dir()
        assert result.scope.absence_authority_revision is None
        loaded, _, upgrade = await asyncio.to_thread(load)
        assert loaded.scope == result.scope and not upgrade
        assert counts["constructor"] == 1


@pytest.mark.parametrize(
    "change", ["binding", "removed", "readded", "index", "identity"]
)
@private_profile_test
async def test_absent_management_action_keeps_captured_scope(tmp_path, request, change):
    async with _console_index(tmp_path) as (controller, roots, base):
        (root,) = roots
        card = _card(root)
        status, policy, _ = await asyncio.to_thread(card._load, controller.app)
        event = card.ActionRequested(
            root.binding_id, root.workspace_id, status, "enable", policy
        )
        registry = controller.app.workspace_registry_service
        binding = await asyncio.to_thread(registry.get_runtime_binding, root.binding_id)

        def change_state():
            if change == "binding":
                registry.set_folder_binding_access(root.binding_id, allow_write=True)
            elif change in {"removed", "readded"}:
                registry.remove_runtime_binding(root.binding_id)
                if change == "readded":
                    registry.save_runtime_binding(binding)
            elif change == "identity":
                event.expected_scope = replace(
                    event.expected_scope,
                    key=replace(
                        event.expected_scope.key, data_authority_id="different"
                    ),
                )
            else:
                card._execute(controller.app, event)

        await asyncio.to_thread(change_state)
        with pytest.raises(
            ValueError,
            match="stale_scope|authority_unavailable|index_data_authority_changed",
        ):
            await asyncio.to_thread(card._execute, controller.app, event)


@private_profile_test
async def test_custom_management_reader_keeps_original_dispatch(
    tmp_path, request, monkeypatch
):
    async with _console_index(tmp_path) as (controller, roots, base):
        service = await asyncio.to_thread(
            runtime.get_project_index_service, controller.app
        )
        original = service.management_snapshot
        calls = []

        def custom(key):
            calls.append(key)
            return original(key)

        with monkeypatch.context() as patch:
            patch.setattr(service, "management_snapshot", custom)
            result = await asyncio.to_thread(_card(roots[0])._load, controller.app)
        assert len(calls) == 1 and result[0].state == "Off" and base.is_dir()


@private_profile_test
async def test_absent_management_refuses_source_change_during_observation(
    tmp_path, request
):
    async with _console_index(tmp_path) as (controller, roots, base):
        service = await asyncio.to_thread(
            runtime.get_project_index_service, controller.app
        )
        original = service._reader
        changed = []

        def observe(frame, event, result):
            if not changed and event == "return" and frame.f_code is original.__code__:
                # Only the reader inside the actual absence operation, not key creation.
                if frame.f_back.f_code is runtime._console_missing_index_base.__code__:
                    changed.append(True)
                    service._reader = lambda _key: result

        def load():
            try:
                with (
                    _observe(observe),
                    pytest.raises(ValueError, match="index_preparation_source_changed"),
                ):
                    _card(roots[0])._load(controller.app)
            finally:
                service._reader = original

        await asyncio.to_thread(load)
        assert changed and not service._operations and not base.exists()


@private_profile_test
async def test_absent_management_retains_actual_uncertain_metadata_close(
    tmp_path, request
):
    async with _console_index(tmp_path) as (controller, roots, base):
        native = windows_files._native()
        close = native.kernel.CloseHandle
        held = []

        def fail_before_native(handle):
            if held and handle == held[0].handle:
                raise OSError("fixture retains the actual metadata handle")
            return close(handle)

        def observe(frame, event, result):
            if held or event != "return" or result is None:
                return
            record = _index_batch_open(frame, base)
            if record is not None and record.path == base.parent:
                held.append(record)
                native.kernel.CloseHandle = fail_before_native

        def load_and_clean():
            service = runtime.get_project_index_service(controller.app)
            try:
                with _observe(observe), pytest.raises(ProjectIndexRetirementUnproven):
                    _card(roots[0])._load(controller.app)
                (operation,) = service._operations
                assert operation.retirement_unproven and operation.done.is_set()
                (pin,) = [
                    item
                    for item in operation.retained
                    if type(item) is runtime._DataAuthorityPin
                ]
                assert not pin.closed
                assert native.info(held[0].handle).volume == held[0].identity[0]
                with pytest.raises(ProjectIndexRetirementUnproven):
                    _card(roots[0])._load(controller.app)
                assert not base.exists()
            finally:
                native.kernel.CloseHandle = close
                if held:
                    assert close(held[0].handle)
                for operation in tuple(service._operations):
                    for retained in operation.retained:
                        if type(retained) is runtime._DataAuthorityPin:
                            retained.close()
                    service._operations.discard(operation)

        await asyncio.to_thread(load_and_clean)
