"""A live parent waits for isolated publication before checking startup evidence."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, os, sys, threading
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice','pyaudio'): sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import bootstrap, publication, storage_admission as storage
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
outcome = sys.argv[2]
def config_manifest(doc):
    doc['owners'][0]['owner_id']='config'
    doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
    doc['dependency_groups'][0]['members']=['profile:profile:config']
entered, release = threading.Event(), threading.Event()
finalize = publication.finalize_candidate
def held_finalize(*args, **kwargs):
    entered.set()
    assert release.wait(10)
    if outcome == 'pending_failure':
        raise RuntimeError('fixture fails before actual finalization')
    return finalize(*args, **kwargs)
publication.finalize_candidate = held_finalize
async def main():
    app = TldwCli()
    selector = Path(os.environ['TLDW_CONFIG_PATH'])
    before = selector.read_bytes()
    base = Path.home()
    archive = sealed(base, mutate=config_manifest, data=b'[general]\nusers_name="source"\n')
    parent = base/'destinations';parent.mkdir(mode=0o700)
    control=base/'control'
    plan=plan_restore(archive,mode='isolated',destinations={'root':parent/'config','profile:profile:paths.data_dir':parent/'data'},target=None,profile_names={'profile':'Recovered'})
    monitoring=asyncio.create_task(monitor_app(app))
    restoring=asyncio.create_task(asyncio.to_thread(restore_isolated,archive,plan,control,threading.Event()))
    try:
        async with asyncio.timeout(20):
            while not entered.is_set():
                if restoring.done(): await restoring
                await asyncio.sleep(.01)
            while app._backup_runtime_maintenance.pause._startup_thread is None:
                await asyncio.sleep(.01)
        runtime=app._backup_runtime_maintenance
        pause=runtime.pause
        await asyncio.sleep(.1)
        assert pause._startup_error is None, ('premature startup failure',pause._startup_error)
        assert pause._startup_thread.is_alive(), 'startup did not wait for native finalization gate'
        assert not storage._startups
        assert bootstrap.startup_permission(selector,bootstrap.default_bootstrap_root()) == (False,'recovery_scope_uncertain')
        release.set()
        if outcome == 'pending_failure':
            try: await restoring
            except RuntimeError as error:
                assert str(error)=='fixture fails before actual finalization'
            else: raise AssertionError('failed finalization was accepted')
            result=await asyncio.gather(monitoring,return_exceptions=True)
            assert isinstance(result[0],bootstrap.RecoveryRequired)
            assert result[0].args==('startup_reacquisition_failed',)
            assert pause._startup_error.args==('recovery_scope_uncertain',)
            assert storage._pause is pause and not storage._startups
            assert bootstrap._records(bootstrap.default_bootstrap_root())[0]
        else:
            profile=await restoring
            async with asyncio.timeout(10):
                while app._backup_runtime_maintenance is not None:
                    if monitoring.done(): await monitoring
                    await asyncio.sleep(.01)
            assert storage._pause is None and len(storage._startups)==1
            assert selector.read_bytes()==before
            note=app.chachanotes_db.add_note('After isolated restore','Parent ordinary writer resumed.')
            assert note
            config,data=ProfileCatalog(control).resolve(profile)
            assert config==parent/'config'/'config.toml' and data==parent/'data'
            assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
    finally:
        release.set()
        await asyncio.gather(restoring,return_exceptions=True)
        monitoring.cancel()
        await asyncio.gather(monitoring,return_exceptions=True)
        publication.finalize_candidate=finalize
        if storage._pause is None:
            await app._shutdown_app_owned_lifecycles()
            await app.tts_service.close()
            await app.tts_service.wait_closed()
        # Persistent-pending failure deliberately keeps parent admission fenced;
        # this isolated process exits without inventing a release or recovery.
asyncio.run(main())
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize("outcome", ("success", "pending_failure"))
def test_parent_readmission_waits_for_actual_isolated_finalization(tmp_path, outcome):
    _run(tmp_path, "isolated", outcome, script=_SCRIPT)
