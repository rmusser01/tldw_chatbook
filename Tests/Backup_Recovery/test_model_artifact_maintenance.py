"""Actual model-store native owners remain counted through maintenance."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time, threading
from pathlib import Path
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService
from tldw_chatbook.Model_Artifacts.leases import ArtifactOperationLease, ArtifactLeaseKey, LeaseMode
from tldw_chatbook.Backup_Recovery import storage_admission as storage, bootstrap
case = sys.argv[1]
root = Path.home() / 'models'
async def main():
    service = ModelArtifactService(root)
    if case == 'lease':
        lease = ArtifactOperationLease(service.locks_path, ArtifactLeaseKey('a','b','c'), LeaseMode.SHARED).acquire()
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic())
            lease.release()
            assert pause.drain(time.monotonic()+1)
        finally:
            lease.release(); pause.resume()
    elif case in ('handle', 'removal'):
        from Tests.Model_Artifacts.test_provision_install import _descriptor
        from tldw_chatbook.Model_Artifacts import ArtifactRef
        ref = ArtifactRef('test', 'rev', 'fp32')
        source = Path.home() / 'source'; source.mkdir()
        (source/'model.bin').write_bytes(b'x')
        service.install(_descriptor(ref), source)
        owner = service.acquire_installed_root(ref) if case == 'handle' else service.acquire_removal_authority(ref)
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic())
            if case == 'removal': owner.commit()
            owner.close()
            assert pause.drain(time.monotonic()+1)
        finally: owner.close(); pause.resume()
    elif case in ('acquire_error', 'close_error'):
        from tldw_chatbook.Model_Artifacts import leases as leases_module
        lease = ArtifactOperationLease(service.locks_path, ArtifactLeaseKey('a','b','c'), LeaseMode.SHARED)
        if case == 'acquire_error':
            original = leases_module.portalocker.lock
            def fail(*args): raise OSError('injected lock failure')
            leases_module.portalocker.lock = fail
            try:
                try: lease.acquire()
                except leases_module.ArtifactLeaseError: pass
                else: raise AssertionError('lock failure lost')
            finally: leases_module.portalocker.lock = original
        else:
            lease.acquire()
            original = lease._handle
            class UncertainClose:
                def fileno(self): return original.fileno()
                def close(self):
                    original.close()
                    raise OSError('injected uncertain close')
            lease._handle = UncertainClose()
            try: lease.release()
            except leases_module.ArtifactLeaseError: pass
            else: raise AssertionError('close failure lost')
        pause = storage._begin_local_pause()
        try: assert pause.drain(time.monotonic()) == (case == 'acquire_error')
        finally: pause.resume()
    elif case == 'intake':
        pause = storage._begin_local_pause()
        try:
            for call in (service.disk_usage, lambda: ModelArtifactService(root/'new')):
                try: call()
                except bootstrap.RecoveryRequired: pass
                else: raise AssertionError('model work admitted while paused')
            assert not (root/'new').exists()
        finally: pause.resume()
    elif case in ('provision', 'cancel', 'hash_cancel', 'staged_read'):
        from Tests.Model_Artifacts.test_provision_install import _descriptor
        from Tests.Model_Artifacts.test_acquisition_types import DictCatalog
        from Tests.Model_Artifacts.acquisition_test_helpers import grant_consent
        from tldw_chatbook.Model_Artifacts import ArtifactRef
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService
        ref = ArtifactRef('test', 'rev', 'fp32')
        descriptor = _descriptor(ref)
        catalog = DictCatalog({ref: descriptor})
        acquisition = ArtifactAcquisitionService(service)
        consent = grant_consent(acquisition, ref, catalog)
        fetched, finish = asyncio.Event(), asyncio.Event()
        async def fetch(descriptor, staging, progress, sources):
            (staging / 'model.bin').write_bytes(b'x')
            fetched.set()
            await finish.wait()
        acquisition._fetch_artifact = fetch
        entered, release = threading.Event(), threading.Event()
        original = service._verify_payload
        def verify(*args, **kwargs):
            entered.set(); assert release.wait(3)
            return original(*args, **kwargs)
        if case == 'cancel': service._verify_payload = verify
        if case == 'hash_cancel':
            original = acquisition._hash_staged_file
            acquisition._hash_staged_file = verify
        if case == 'staged_read':
            stage = service._download_stage_for(descriptor, create=True)
            from tldw_chatbook.Model_Artifacts.acquisition import _fetch_sidecar_path
            sidecar = _fetch_sidecar_path(stage.payload)
            sidecar.write_text('{}')
            original_read = Path.read_text
            def read(path, *args, **kwargs):
                if path == sidecar:
                    with path.open('r') as handle:
                        entered.set(); assert release.wait(3)
                        return handle.read()
                return original_read(path, *args, **kwargs)
            Path.read_text = read
            task = asyncio.create_task(asyncio.to_thread(acquisition._staged_bytes_for, descriptor))
            assert await asyncio.to_thread(entered.wait, 2)
            pause = storage._begin_local_pause()
            try:
                assert not pause.drain(time.monotonic())
                release.set(); assert await task == 0
                assert pause.drain(time.monotonic()+1)
            finally: release.set(); pause.resume()
            print('retired and reopened')
            return
        task = asyncio.create_task(acquisition.provision(ref, consent, catalog))
        await asyncio.wait_for(fetched.wait(), 2)
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic())
            finish.set()
            if case in ('cancel', 'hash_cancel'):
                assert await asyncio.to_thread(entered.wait, 2)
                task.cancel(); await asyncio.sleep(.05)
                assert not task.done()
                assert not pause.drain(time.monotonic())
                task.cancel(); release.set()
                try: await task
                except asyncio.CancelledError: pass
                else: raise AssertionError('cancel lost')
            else: assert await task == ref
            assert pause.drain(time.monotonic()+1)
        finally:
            finish.set(); release.set(); pause.resume()
        if case != 'hash_cancel': assert service.artifact_path(ref).is_dir()
    else:
        entered, release = threading.Event(), threading.Event()
        original = service._regular_tree_bytes
        def read(path):
            entered.set(); assert release.wait(3)
            return original(path)
        service._regular_tree_bytes = read
        task=asyncio.create_task(asyncio.to_thread(service.disk_usage))
        assert await asyncio.to_thread(entered.wait,2)
        pause=storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic())
            release.set(); await task
            assert pause.drain(time.monotonic()+1)
        finally:
            release.set(); await task; pause.resume()
    print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "case",
    [
        "lease",
        "intake",
        "read",
        "provision",
        "cancel",
        "hash_cancel",
        "staged_read",
        "handle",
        "removal",
        "acquire_error",
        "close_error",
    ],
)
def test_model_native_maintenance(tmp_path, case):
    _run(tmp_path, case, "model", script=_SCRIPT)
