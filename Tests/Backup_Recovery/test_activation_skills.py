"""Restored skill trust is inert until owner review, including actual workers."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys, types
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
sys.modules.setdefault('parakeet_mlx', types.ModuleType('parakeet_mlx'))
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore, FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from Tests.Skills.test_skill_trust_service import FakeSecureKeyring
from tldw_chatbook.Skills_Interop.skill_trust_store import KeyringSkillTrustKeyCache
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
base = selector.parent.parent
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(base/'data') + '"\n')
selector.chmod(0o600)
store = base / 'data' / 'skills-store'
skills = store / 'skills'
trust_root = base / 'data' / 'independent-trust'
marker_root = base / 'data' / 'independent-marker'
marker_root.mkdir(mode=0o700)
marker = FileSkillTrustGenerationMarkerStore(marker_root/'marker.json', store_dir=marker_root)
keyring = FakeSecureKeyring()
cache = KeyringSkillTrustKeyCache(keyring_backend=keyring)
trust = SkillTrustService(skills_dir=skills, trust_store=SkillTrustStore(trust_root, marker), key_cache=cache)
trust.unlock_with_passphrase('test-only-passphrase', salt=b'6'*32)
local = LocalSkillsService(store_dir=store, trust_service=trust)
asyncio.run(local.create_skill(name='demo', content='---\nname: demo\ndescription: test\n---\n# Demo\n', supporting_files={'scripts/demo.py': "print('real-script-effect')\n"}))
trust.bootstrap_trust()
trust.grant_script_execution('demo')
trust.enable_keyring_convenience()
original = (skills/'demo'/'SKILL.md').read_text()
from tldw_chatbook import config
config.get_user_data_dir()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
authority = admission_authority(root)
if state == 'trust_skills_source':
    import shutil
    other_skills=base/'data'/'independent-skills'
    shutil.copytree(skills, other_skills)
    trust.skills_dir=other_skills
restored = trust.skills_dir if state == 'trust_skills_source' else trust_root if state == 'trust_source' else marker_root if state == 'marker_source' else skills
# Independent-source cases use an unrelated current selector, so only actual
# trust/marker path admission can discover the restored namespace.
authority.register('profile', (selector.parent, restored) if state in ('shared','trust_source','marker_source','trust_skills_source') else (selector.parent, base/'data'))
control = base / 'operation'
control.mkdir(mode=0o700)
owners = ('config', 'skills', 'mcp.local')
if state not in ('ordinary', 'unqualified'):
    register_pending(root, 'restore', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore', selector, 'generation', owners, session=session)
    (root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
    activation = ActivationStore(control/'activation')
    for owner in owners:
        if state == 'approved' or (state == 'config_only' and owner == 'config'):
            activation.approve('generation', owner)
    if state == 'missing':
        (activation._generation('generation')/'required.json').unlink()
    if state == 'corrupt':
        (activation._generation('generation')/'required.json').write_bytes(b'{')
if state in ('shared', 'trust_source', 'marker_source', 'trust_skills_source'):
    other = base/'other.toml'
    other.write_text('[general]\n')
    other.chmod(0o600)
    os.environ['TLDW_CONFIG_PATH'] = str(other)
if state == 'unqualified':
    storage.qualified_for = lambda *args: (False, 'native_unqualified')
denied = state not in ('ordinary', 'approved', 'unqualified')
probes=[]
old_load = marker.load_marker
# Keep marker implementation real; a probe sentry records unexpected automatic IO.
def counted_marker():
    probes.append('marker')
    return old_load()
# File marker is slotted: instrument its class only after setup.
marker_type = type(marker)
marker_type.load_marker = lambda self: counted_marker() if self is marker else None
old_keys = cache.load_keys
cache_type = type(cache)
cache_type.load_keys = lambda self, **kwargs: (probes.append('key'), old_keys(**kwargs))[1]
events=[]
def sentry(event,args):
    if event in ('subprocess.Popen','os.posix_spawn','os.system'):
        events.append(event)
        if denied: raise AssertionError('inactive skill spawned process')
sys.addaudithook(sentry)
if route == 'posture':
    trust._keys = None
    assert trust.trust_posture() == ('locked' if denied else 'ready')
    assert (not probes) if denied else probes
elif route == 'inspection':
    result = asyncio.run(local.list_skills())
    assert result
    assert original == (skills/'demo'/'SKILL.md').read_text()
    if denied: assert not probes, probes
elif route == 'review':
    trust._keys = None
    assert trust.unlock_from_keyring_convenience()
    review = trust.capture_review('demo')
    assert 'SKILL.md' in review['current_files']
    trust.trust_reviewed_snapshot(review['review_id'])
    assert denied
    assert not activation.allowed('generation', 'skills')
    assert not trust.script_execution_granted('demo')
else:
    try:
        if route == 'ensure': trust.ensure_skill_trusted('demo')
        elif route == 'verify': trust.verify_skill_content('demo', skill_content=original, supporting_files={'scripts/demo.py': "print('real-script-effect')\n"})
        elif route == 'grant':
            assert trust.script_execution_granted('demo') is (not denied)
        elif route == 'execute': asyncio.run(local.execute_skill('demo'))
        elif route == 'read': asyncio.run(local.read_skill_file('demo','SKILL.md'))
        elif route == 'run':
            result = asyncio.run(local.run_skill_script('demo','scripts/demo.py',[]))
            assert result.exit_code == 0 and 'real-script-effect' in result.stdout
        else: raise AssertionError(route)
    except SkillTrustBlockedError:
        assert denied
    else:
        assert not denied or route == 'grant', 'inactive skill use succeeded'
    if denied: assert not probes, probes
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "ensure",
        "verify",
        "grant",
        "execute",
        "read",
        "run",
        "posture",
        "inspection",
        "review",
    ],
)
def test_restored_skill_use_and_cached_credentials_stay_inactive(tmp_path, route):
    _run(tmp_path, route, "inactive", script=_SCRIPT)


@pytest.mark.parametrize(
    "state",
    [
        "ordinary",
        "approved",
        "unqualified",
        "config_only",
        "missing",
        "corrupt",
        "shared",
        "trust_source",
        "marker_source",
        "trust_skills_source",
    ],
)
def test_skill_execution_uses_paired_actual_source_authority(tmp_path, state):
    _run(tmp_path, "run", state, script=_SCRIPT)


_RETENTION = (
    _SCRIPT.split("probes=[]")[0]
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Skills_Interop import skill_script_runner as runner
from tldw_chatbook.Skills_Interop.skill_trust_service import _execution_scope

def assert_held():
    try:
        with authority.maintenance(('profile',), .03):
            raise AssertionError('accepted skills lost native admission')
    except AdmissionTimeout:
        pass

if route == 'generation':
    register_pending(root, 'restore-again', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore-again', selector, 'generation-new', owners, session=session)
    (root/('pending-'+bootstrap._key('restore-again')+'.json')).unlink()
    try: trust.ensure_skill_trusted('demo')
    except SkillTrustBlockedError: pass
    else: raise AssertionError('prior generation approval carried forward')
    assert not trust.script_execution_granted('demo')
elif route == 'pid':
    with _execution_scope(local) as allowed:
        assert allowed
        original_pid=os.getpid
        try:
            os.getpid=lambda: original_pid()+1
            try: trust.ensure_skill_trusted('demo')
            except SkillTrustBlockedError: pass
            else: raise AssertionError('foreign PID borrowed accepted lease')
        finally: os.getpid=original_pid
elif route == 'nested':
    with _execution_scope(local) as allowed:
        assert allowed
        pause=storage._begin_local_pause()
        try:
            trust.ensure_skill_trusted('demo')
            assert trust.script_execution_granted('demo')
            assert trust.trust_posture() == 'ready'
            assert_held()
        finally: pause.resume()
elif route == 'copied':
    async def probe():
        with _execution_scope(local) as allowed:
            assert allowed
            pause=storage._begin_local_pause()
            try:
                async def child():
                    try: trust.ensure_skill_trusted('demo')
                    except SkillTrustBlockedError: return
                    raise AssertionError('copied task borrowed lease')
                await asyncio.create_task(child())
                try: await asyncio.to_thread(trust.ensure_skill_trusted, 'demo')
                except SkillTrustBlockedError: pass
                else: raise AssertionError('copied thread borrowed lease')
                assert_held()
            finally: pause.resume()
    asyncio.run(probe())
else:
    entered=threading.Event()
    release=threading.Event()
    finished=threading.Event()
    effects=[]
    original=runner.run_script_subprocess
    def native(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        # Intake has paused since acceptance. Nested checks must reuse each
        # actual source lease rather than trying to join new work.
        trust.ensure_skill_trusted('demo')
        assert trust.script_execution_granted('demo')
        try:
            result=original(*args, **kwargs)
            assert result.exit_code == 0 and 'real-script-effect' in result.stdout
            effects.append('actual subprocess')
            assert_held()
            return result
        finally: finished.set()
    runner.run_script_subprocess=native
    async def run():
        waiter=asyncio.create_task(local.run_skill_script('demo','scripts/demo.py',[]))
        assert await asyncio.to_thread(entered.wait,2)
        pause=storage._begin_local_pause()
        try:
            waiter.cancel()
            try: await waiter
            except asyncio.CancelledError: pass
            assert not finished.is_set()
            assert_held()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait,3)
            pause.resume()
    asyncio.run(run())
    assert effects == ['actual subprocess']
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["nested", "copied", "pid", "cancel", "generation"])
def test_skills_keep_exact_accepted_native_lifetime(tmp_path, route):
    _run(tmp_path, route, "approved", script=_RETENTION)
