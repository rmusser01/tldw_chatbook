"""Physical work and custody for an unused Windows Console index cache."""

import asyncio
import errno
import inspect
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_permission_read_operation import _observe
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import core_cached_connection
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    capture_run_admitted_workspace_roots,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.private_paths import secure_private_directory
from tldw_chatbook.Workspaces.project_index import runtime
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexAuthority,
    ProjectIndexPolicy,
    ProjectIndexRetirementUnproven,
)
from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows index admission"
)


@asynccontextmanager
async def _console_index(tmp_path, *, folders=1):
    from tldw_chatbook.config import get_user_data_dir

    data = get_user_data_dir()
    assert secure_private_directory(
        data, create=False, application_owned=True
    ).verified_private
    db = WorkspaceDB(data / "workspaces.db")
    registry = LocalWorkspaceRegistryService(db)
    registry.create_workspace(workspace_id="index-preparation", name="Project")
    bindings = []
    for index in range(folders):
        project = tmp_path / f"workspace-{index}"
        project.mkdir()
        bindings.append(registry.add_folder_binding("index-preparation", project))
    session = ConsoleChatStore().create_session(workspace_id="index-preparation")
    roots = capture_run_admitted_workspace_roots(session=session, registry=registry)
    assert {root.binding_id for root in roots} == {
        binding.binding_id for binding in bindings
    }
    assert roots and all(not root.allow_write for root in roots)
    app = SimpleNamespace(workspace_registry_service=registry)
    controller = SimpleNamespace(app=app)
    base = data / "project_index"
    with pytest.raises(FileNotFoundError):
        base.lstat()
    assert not hasattr(app, "_project_index_service")
    try:
        yield controller, roots, base
    finally:
        await runtime.close_project_index_service(app)
        db.close()


def _observe_preparation(controller, roots, *, fault=None):
    app = controller.app
    db = app.workspace_registry_service.db
    original_leases = set(storage._live_leases)
    counts, handles, leases, keys, pins, authorities = Counter(), [], [], [], [], []
    adapter_code = (
        LocalWorkspaceRegistryService.project_index_authority_adapters.__code__
    )
    reader_code, fence_wrapper = (
        next(
            code
            for code in adapter_code.co_consts
            if getattr(code, "co_name", None) == name
        )
        for name in ("reader", "fence")
    )
    codes = {
        **{root.guard.__code__: "root_guard" for root in roots},
        runtime.project_index_key.__code__: "key",
        reader_code: "reader",
        ProjectIndexService.status.__code__: "public_status",
        WorkspaceProjectIndexDB.__init__.__code__: "index_constructor",
        BindingLifetime.acquire.__func__.__code__: "lifetime",
        runtime._AdmittedStoreFactory.retain.__code__: "pin",
        windows_files._Native.open_handle.__code__: "native_open",
        windows_files._Native.security.__code__: "native_security",
    }
    open_code = inspect.unwrap(SQLiteStoreKernel._get_connection).__code__
    status_code = ProjectIndexService._status_from_store.__code__
    statuses = []

    def observe(frame, event, result):
        if fault is not None:
            fault(frame, event, result)
        code = frame.f_code
        if event == "call" and code in codes:
            counts[codes[code]] += 1
        if event != "return" or result is None:
            return
        if code is fence_wrapper and type(result) is ProjectIndexAuthority:
            counts["live_fence"] += 1
            authorities.append(result)
        if code is runtime.project_index_key.__code__:
            keys.append(result)
        elif code is reader_code:
            authorities.append(result)
        elif code is status_code:
            statuses.append(result.state)
        elif code is runtime._AdmittedStoreFactory.retain.__code__:
            pins.append(result)
        elif code is storage.acquire_storage.__code__:
            leases.append(result)
        elif code is open_code and isinstance(result, sqlite3.Connection):
            owner = frame.f_locals["self"]
            if owner is db or type(owner) is WorkspaceProjectIndexDB:
                handles.append((owner, result))

    with _observe(observe):
        service, scopes = ConsoleChatController._prepare_project_index_scopes(
            controller, roots
        )

    assert not storage._pending_acquisitions
    assert any(owner is db for owner, _handle in handles)
    for owner, handle in handles:
        assert core_cached_connection(owner, handle) is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            sqlite3.Connection.in_transaction.__get__(handle)
    return SimpleNamespace(
        service=service,
        scopes=scopes,
        counts=counts,
        handles=handles,
        leases=leases,
        keys=keys,
        pins=pins,
        authorities=authorities,
        statuses=statuses,
        original_leases=original_leases,
    )


def _assert_retired(result):
    assert result.pins and all(pin.closed for pin in result.pins)
    assert result.leases and all(
        lease not in storage._live_leases for lease in result.leases
    )
    assert set(storage._live_leases) == result.original_leases
    assert result.service is None or not result.service._operations


def _status_with_fault(controller, root, fault):
    service = runtime.get_project_index_service(controller.app)
    key = runtime.project_index_key(
        controller.app,
        root.workspace_id,
        root.binding_id,
        root.root,
    )
    with _observe(fault):
        return runtime.project_index_status_for_console(service, key)


@private_profile_test
async def test_console_prepares_existing_binding_without_creating_empty_index(
    tmp_path, request
):
    async with _console_index(tmp_path) as (controller, roots, base):
        observed = await asyncio.to_thread(_observe_preparation, controller, roots)
        service, scopes = observed.service, observed.scopes
        counts, handles, leases, keys, pins, authorities, statuses = (
            observed.counts,
            observed.handles,
            observed.leases,
            observed.keys,
            observed.pins,
            observed.authorities,
            observed.statuses,
        )
        request.node.user_properties.extend(
            (
                ("index_preparation_counts", json.dumps(counts)),
                ("index_statuses", json.dumps(statuses)),
                ("retained_pins", len(pins)),
                ("retired_leases", len(leases)),
                (
                    "registry_sqlite_births",
                    sum(
                        owner is controller.app.workspace_registry_service.db
                        for owner, _handle in handles
                    ),
                ),
                (
                    "index_sqlite_births",
                    sum(
                        type(owner) is WorkspaceProjectIndexDB
                        for owner, _handle in handles
                    ),
                ),
            )
        )
        assert service is controller.app._project_index_service and not scopes
        assert counts["root_guard"] == counts["key"] == 1
        assert len(keys) == 1 and any(
            authority.key == keys[0] for authority in authorities
        )
        assert counts["live_fence"] == 2
        assert counts["native_open"] and counts["native_security"]
        _assert_retired(observed)
        if base.is_dir():
            created = tuple(base.rglob("workspace_project_index.db"))
            assert len(created) == counts["index_constructor"] == 1
            assert statuses == ["Off"]
            assert created[0].stat().st_size > 0
            request.node.user_properties.append(
                ("created_index_bytes", created[0].stat().st_size)
            )
        assert counts["index_constructor"] == counts["lifetime"] == 0
        with pytest.raises(FileNotFoundError):
            base.lstat()


@pytest.mark.parametrize("descriptor", ["normal", "unsupported"])
@private_profile_test
async def test_existing_stock_index_scope_is_still_advertised(
    tmp_path, request, descriptor
):
    from Tests.Workspaces.project_index_support import page

    async with _console_index(tmp_path) as (controller, roots, base):
        (root,) = roots

        def ready():
            service = runtime.get_project_index_service(controller.app)
            key = runtime.project_index_key(
                controller.app, root.workspace_id, root.binding_id, root.root
            )
            off = service.status(key)
            assert off.state == "Off" and base.is_dir()
            enabled = service.enable(
                key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
            )
            with service._store(key) as store, service._admitted(store, key) as live:
                assert live is not None
                claim = store.claim_job(
                    enabled.scope,
                    owner_token=f"native-v1:{key.binding_namespace}:fixture",
                    authority_revision=live.authority_revision,
                    expected_job_revision=enabled.job_revision,
                )
                assert store.commit_scan_page(
                    enabled.scope,
                    claim.job_id,
                    page(path="seed.txt"),
                    expected_job_revision=claim.job_revision,
                    authority_revision=live.authority_revision,
                )
                assert store.retire_job(
                    enabled.scope,
                    claim.job_id,
                    owner_token=claim.owner_token,
                    expected_job_revision=claim.job_revision,
                    state="Ready",
                )
            return enabled.scope

        expected = await asyncio.to_thread(ready)
        native = windows_files._native()
        length = native.advapi.GetSecurityDescriptorLength
        faults = []

        def oversized(value):
            length(value)
            native.advapi.GetSecurityDescriptorLength = length
            faults.append(True)
            return 4097

        def fault(frame, event, _result):
            if (
                descriptor == "unsupported"
                and event == "call"
                and not faults
                and frame.f_code is native.security.__code__
                and frame.f_locals["with_descriptor"]
            ):
                batch = frame.f_back.f_back
                if (
                    batch.f_code
                    is windows_files.WindowsOS.stat_many_for_admission.__code__
                    and base in batch.f_locals["paths"]
                    and batch.f_locals["node"] == base
                ):
                    # Fault the reported length of a real fresh native
                    # descriptor. Restore the API before legacy status.
                    native.advapi.GetSecurityDescriptorLength = oversized

        try:
            result = await asyncio.to_thread(
                _observe_preparation, controller, roots, fault=fault
            )
        finally:
            native.advapi.GetSecurityDescriptorLength = length
        assert faults == ([True] if descriptor == "unsupported" else [])
        _assert_retired(result)
        assert result.scopes == {root.alias: expected}
        assert result.statuses == ["Ready"]
        assert result.counts["index_constructor"] == 1


@pytest.mark.parametrize("component", ["status", "factory", "reader"])
@private_profile_test
async def test_custom_index_sources_keep_original_status_sequence(
    tmp_path, request, monkeypatch, component
):
    async with _console_index(tmp_path) as (controller, roots, base):
        service = await asyncio.to_thread(
            runtime.get_project_index_service, controller.app
        )
        target = {"status": "status", "factory": "_factory", "reader": "_reader"}[
            component
        ]
        original = getattr(service, target)
        calls = []

        class Factory:
            def __call__(self, key):
                calls.append(key)
                return original(key)

            def __getattr__(self, name):
                return getattr(original, name)

        def custom(key):
            calls.append(key)
            return original(key)

        with monkeypatch.context() as patch:
            patch.setattr(
                service, target, Factory() if component == "factory" else custom
            )
            result = await asyncio.to_thread(_observe_preparation, controller, roots)
        _assert_retired(result)
        assert result.service is service and not result.scopes
        assert calls == result.keys
        assert result.counts["public_status"] == result.counts["index_constructor"] == 1
        assert result.statuses == ["Off"] and base.is_dir()


@private_profile_test
async def test_multiple_missing_index_roots_create_no_cache(tmp_path, request):
    async with _console_index(tmp_path, folders=2) as (controller, roots, base):
        result = await asyncio.to_thread(_observe_preparation, controller, roots)
        _assert_retired(result)
        assert (
            result.service is controller.app._project_index_service
            and not result.scopes
        )
        assert result.counts["root_guard"] == result.counts["key"] == 2
        assert {key.binding_id for key in result.keys} == {
            root.binding_id for root in roots
        }
        assert result.counts["live_fence"] == 4
        assert not result.counts["index_constructor"] and not result.counts["lifetime"]
        with pytest.raises(FileNotFoundError):
            base.lstat()


def _index_batch_open(frame, base):
    if frame.f_code is not windows_files._Native.open_handle.__code__:
        return None
    named = frame.f_back
    batch = named.f_back if named is not None else None
    if (
        batch is not None
        and batch.f_code is windows_files.WindowsOS.stat_many_for_admission.__code__
        and base in batch.f_locals["paths"]
    ):
        return frame.f_locals["_metadata_record"]
    return None


@pytest.mark.parametrize("change", ["appearance", "root", "acl", "source"])
@private_profile_test
async def test_missing_index_observation_does_not_ignore_late_changes(
    tmp_path, request, change
):
    async with _console_index(tmp_path) as (controller, roots, base):
        (root,) = roots
        changed, calls_to_custom, root_refused = [], [], []
        service = await asyncio.to_thread(
            runtime.get_project_index_service, controller.app
        )
        reader_code = service._reader.__code__
        original_status = service.status

        def custom(_key):
            calls_to_custom.append(True)
            pytest.fail("a selected source change replayed the status command")

        def fault(frame, event, result):
            if (
                changed
                and change == "root"
                and event == "return"
                and frame.f_code is reader_code
            ):
                root_refused.append(result is None)
            if changed:
                return
            record = _index_batch_open(frame, base)
            if (
                change == "appearance"
                and event == "return"
                and record is not None
                and record.path == base
            ):
                assert result is None and record.handle is None
                changed.append(True)
                windows_files.WindowsOS().mkdir(base, 0o700)
            elif (
                change == "acl"
                and event == "call"
                and record is not None
                and record.path == base
            ):
                changed.append(True)
                subprocess.run(
                    ["icacls", str(base.parent), "/grant", "*S-1-1-0:(R)"],
                    check=True,
                    capture_output=True,
                )
            elif (
                change in {"root", "source"}
                and event == "return"
                and frame.f_code
                is windows_files.WindowsOS.stat_many_for_admission.__code__
                and base in frame.f_locals["paths"]
            ):
                assert result[base] is None
                changed.append(True)
                if change == "root":
                    root.root.rename(root.root.with_name("retired"))
                    root.root.mkdir()
                else:
                    service.status = custom

        before = set(storage._live_leases)
        try:
            expected = OSError if change == "appearance" else ValueError
            with pytest.raises(expected) as caught:
                await asyncio.to_thread(_status_with_fault, controller, root, fault)
        finally:
            if change == "source" and changed:
                del service.status
            if change == "acl" and changed:
                secure_private_directory(
                    base.parent, create=False, application_owned=True
                )
        assert changed == [True] and not calls_to_custom
        assert set(storage._live_leases) == before
        assert not storage._pending_acquisitions and not service._operations
        assert service.status == original_status
        if change == "appearance":
            assert caught.value.errno == errno.ESTALE
            assert base.is_dir() and not tuple(base.iterdir())
        else:
            with pytest.raises(FileNotFoundError):
                base.lstat()
        if change == "root":
            assert root_refused == [True]


@private_profile_test
async def test_uncertain_missing_index_close_retains_pin_and_prevents_reuse(
    tmp_path, request
):
    async with _console_index(tmp_path, folders=2) as (controller, roots, base):
        native = windows_files._native()
        close = native.kernel.CloseHandle
        held, closes = [], []

        def fail_before_native(handle):
            if held and handle == held[0].handle:
                closes.append(handle)
                raise OSError("fixture leaves the native metadata handle open")
            return close(handle)

        def fault(frame, event, result):
            if (
                held
                or event != "return"
                or frame.f_code is not native.open_handle.__code__
                or result is None
            ):
                return
            record = _index_batch_open(frame, base)
            if record is not None and record.path == base.parent:
                held.append(record)
                native.kernel.CloseHandle = fail_before_native

        def prepare_and_clean():
            try:
                result = _observe_preparation(controller, roots, fault=fault)
                service = controller.app._project_index_service
                (operation,) = service._operations
                (error,) = [
                    item
                    for item in operation.retained
                    if type(item) is windows_files._AdmissionMetadataCloseError
                ]
                (pin,) = [
                    item
                    for item in operation.retained
                    if type(item) is runtime._DataAuthorityPin
                ]
                assert operation.retirement_unproven and operation.done.is_set()
                assert type(error) is windows_files._AdmissionMetadataCloseError
                assert error.failed_handles == tuple(held)
                assert closes == [held[0].handle]
                assert not pin.closed
                assert len(set(storage._live_leases) - result.original_leases) == 1
                assert native.info(held[0].handle).volume == held[0].identity[0]
                assert result.service is None and not result.scopes
                assert (
                    result.counts["key"] == 1 and not result.counts["index_constructor"]
                )
                with pytest.raises(ProjectIndexRetirementUnproven):
                    with service.registry_connection_scope():
                        pytest.fail("retained uncertain native owner was reused")
                with pytest.raises(FileNotFoundError):
                    base.lstat()
            finally:
                native.kernel.CloseHandle = close
                if held:
                    # The fault never reached the native close. Only the fixture
                    # can recover that exact still-live handle before its pin.
                    assert close(held[0].handle)
                service = getattr(controller.app, "_project_index_service", None)
                if service is not None:
                    for operation in tuple(service._operations):
                        for retained in operation.retained:
                            if type(retained) is runtime._DataAuthorityPin:
                                retained.close()
                        service._operations.discard(operation)

        await asyncio.to_thread(prepare_and_clean)
