"""Isolated execution retains authenticated inputs and durable launch authority."""

from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import archive as zip_archive
from tldw_chatbook.Backup_Recovery import archive_reader, crypto
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits


def test_encrypted_acquisition_binds_retained_ciphertext(
    tmp_path, helper_resource_root, monkeypatch
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source = zip_archive(tmp_path)
    ciphertext = tmp_path / "archive.age"
    crypto.transform(
        source, ciphertext, password=b"pass", decrypt=False, cancel=Event()
    )
    expected = ciphertext.read_bytes()
    sealed = archive_reader.acquire(
        ciphertext, tmp_path / "acquired", ArchiveLimits(), b"pass", Event()
    )
    assert getattr(sealed, "encrypted_source", None) is not None
    ciphertext.write_bytes(b"original replaced after acquisition")
    retained = tmp_path / "retained.age"
    proof = archive_reader.retain_encrypted(sealed, retained, Event())
    assert retained.read_bytes() == expected
    assert proof.plaintext_digest == sealed.digest
    sealed.encrypted_source.path.write_bytes(b"changed private input")
    with pytest.raises(ValueError, match="encrypted_source_changed"):
        archive_reader.retain_encrypted(sealed, tmp_path / "other.age", Event())
    import shutil

    shutil.rmtree(sealed.path.parent)
    recovered = archive_reader.acquire(
        retained, tmp_path / "reopened", ArchiveLimits(), b"pass", Event()
    )
    assert recovered.digest == sealed.digest


def test_streamed_ciphertext_digest_checked_before_plaintext_publication(
    tmp_path, helper_resource_root, monkeypatch
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source = zip_archive(tmp_path)
    ciphertext = tmp_path / "archive.age"
    crypto.transform(
        source, ciphertext, password=b"pass", decrypt=False, cancel=Event()
    )
    target = tmp_path / "plaintext.zip"
    with pytest.raises(crypto.CryptoError, match="input_changed"):
        crypto.transform(
            ciphertext,
            target,
            password=b"pass",
            decrypt=True,
            cancel=Event(),
            expected_input_sha256="0" * 64,
        )
    assert not target.exists()


def test_plain_acquisition_cannot_supply_encrypted_retention(tmp_path):
    sealed = archive_reader.acquire(
        zip_archive(tmp_path), tmp_path / "acquired", ArchiveLimits(), None, Event()
    )
    with pytest.raises(ValueError, match="encrypted_acquisition_required"):
        archive_reader.retain_encrypted(sealed, tmp_path / "retained.age", Event())


_RESTORE = r"""
import json,os,sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor,_finish_isolated
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.journal import Journal
base=Path.home()
ambient=Path(os.environ['TLDW_CONFIG_PATH']);ambient.write_bytes(b'broken = [current config')
broken=base/'corrupt.db';broken.write_bytes(b'broken current database')
before=(ambient.read_bytes(),broken.read_bytes())
def config_manifest(doc):
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(base,mutate=config_manifest,data=b'[general]\nusers_name="original"\n'.replace(b'\\n',b'\n'),partial=sys.argv[1]=='partial')
dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'custom-control'
if sys.argv[1]=='catalog_failure':
 original=ProfileCatalog.register
 def failed(self,*args):
  original(self,*args)
  raise OSError('catalog barrier interruption')
 ProfileCatalog.register=failed
 try:restore_isolated(archive,plan,control,Event())
 except OSError as error:assert str(error)=='catalog barrier interruption'
 else:raise AssertionError('catalog interruption ignored')
 ProfileCatalog.register=original
 pending,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
 assert len(pending)==1 and pending[0]['control_root']==str(control)
 record=pending[0];journal=Journal(control,record['operation_id'])
 with journal._locked(exclusive=False) as parent:rows=journal._records(parent)
 assert rows[-1].event=='activation_recorded'
 candidate=Path(rows[0].evidence['stage']['path'])
 descriptor=json.loads((candidate/'candidate.json').read_text())
 profile_id=descriptor['isolated_profiles'][0]['profile_id']
 try:_launch_descriptor(profile_id,control)
 except ValueError as error:assert str(error)=='recovery_pending'
 else:raise AssertionError('incomplete catalog launched')
 _finish_isolated(candidate,plan,journal,tuple(record['namespaces']),record['operation_id'],Event())
else:profile_id=restore_isolated(archive,plan,control,Event())
config,data=ProfileCatalog(control).resolve(profile_id)
assert config==dest/'config'/'config.toml' and data==dest/'data'
entry=_launch_descriptor(profile_id,control)
assert entry.installation_id and entry.profile_id==profile_id
pending,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
assert not pending and profiles[0]['activation']['store_root']==str(control/'activation')
journal=Journal(control,profiles[0]['activation']['operation_id'])
with journal._locked(exclusive=False) as parent:rows=journal._records(parent)
assert [row.event for row in rows][-3:]==['activation_recorded','catalog_registered','committed']
assert before==(ambient.read_bytes(),broken.read_bytes())
if sys.argv[1]=='catalog_changed':
 from tldw_chatbook.Backup_Recovery.profile_catalog import _name
 path=ProfileCatalog(control).root/_name(profile_id)
 path.write_text(path.read_text()+' ')
 try:_launch_descriptor(profile_id,control)
 except ValueError as error:assert str(error)=='isolated_catalog_changed'
 else:raise AssertionError('changed catalog launched')
if sys.argv[1]=='data_alias':
 data.rename(data.with_name('saved-data'));data.symlink_to(data.with_name('saved-data'),target_is_directory=True)
 try:_launch_descriptor(profile_id,control)
 except ValueError as error:assert str(error)=='catalog_target_invalid'
 else:raise AssertionError('data alias launched')
if sys.argv[1]=='activation_missing':
 import shutil
 shutil.rmtree(control/'activation')
 try:_launch_descriptor(profile_id,control)
 except (ValueError,OSError):pass
 else:raise AssertionError('missing paired activation launched')
if sys.argv[1]=='public_catalog':
 try:journal.record('catalog_registered',rows[-2].evidence)
 except ValueError as error:assert str(error)=='catalog_finalization_required'
 else:raise AssertionError('caller catalog proof accepted')

assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "complete",
        "partial",
        "catalog_failure",
        "catalog_changed",
        "data_alias",
        "activation_missing",
        "public_catalog",
    ],
)
def test_archive_only_restore_and_catalog_reconciliation(tmp_path, route):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, route, "isolated", script=_RESTORE)


_SEED = r"""
import json,os
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
root=Path.home()/'seed';root.mkdir(mode=0o700)
notes=CharactersRAGDB(root/'notes.db','original-installation')
note=notes.add_note('Recovered note','Exact saved note content')
conversation=notes.add_conversation({'title':'Recovered conversation'})
message=notes.add_message({'conversation_id':conversation,'sender':'user','content':'Exact saved chat content'})
media=MediaDatabase(root/'media.db','original-installation')
media_id,_,_=media.add_media_with_keywords(title='Recovered media',media_type='document',content='Exact saved media content',keywords=['recovery'])
(root/'identities.json').write_text(json.dumps(dict(note=note,message=message,media=media_id,notes_version=notes._CURRENT_SCHEMA_VERSION,media_version=media._CURRENT_SCHEMA_VERSION)))
notes.close();media.close()
assert not blocked_attempts()
"""

_READ_CHILD = r"""
import os,sys,json
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
profile,control,ids=sys.argv[1:]
from tldw_chatbook.Backup_Recovery.isolated_restore import launch_profile
import subprocess
original_call=subprocess.call
# Exercise the real launcher and fresh CLI with a noninteractive help request.
# The wrapper supplies only the terminal-mode argument; no authority is replaced.
def help_child(argv,**kwargs):
 result=subprocess.run([*argv,'--help'],**kwargs,capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr[-4000:]
 return result.returncode
subprocess.call=help_child
assert launch_profile(profile,Path(control))==0
subprocess.call=original_call
sys.argv=['tldw-cli','--recovery-profile',profile,'--recovery-control-root',control,'--help']
from tldw_chatbook.cli import main_cli_runner
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0),error
from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor
from tldw_chatbook.Backup_Recovery.activation import activation_permission
from tldw_chatbook import config
entry=_launch_descriptor(profile,Path(control))
assert config.CLI_APP_CLIENT_ID==entry.installation_id
assert os.environ['TLDW_CONFIG_PATH']==entry.config
assert 'OPENAI_API_KEY' not in os.environ and 'TLDW_MEDIA_DB_PATH' not in os.environ
import keyring
assert keyring.get_password('isolated-fixture','ambient') is None
assert not activation_permission('runtime.sync_state')
identities=json.loads(Path(ids).read_text())
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
notes=CharactersRAGDB(config.get_chachanotes_db_path(),config.CLI_APP_CLIENT_ID)
assert notes.get_note_by_id(identities['note'])['content']=='Exact saved note content'
assert notes.get_message_by_id(identities['message'])['content']=='Exact saved chat content'
new=notes.add_note('Fresh local identity','New saved row')
assert notes.get_note_by_id(new)['client_id']==entry.installation_id
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
media=MediaDatabase(config.get_media_db_path(),config.CLI_APP_CLIENT_ID)
rows,count=media.search_media_db('Exact saved media content')
assert count==1 and rows[0]['id']==identities['media'],(rows,count)
assert media.get_media_by_id(identities['media'])['content']=='Exact saved media content'
media.close()
notes.close()
assert not blocked_attempts()
print('isolated read services verified')
"""

_DATA_RESTORE = r"""
import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
base=Path.home();ambient=Path(os.environ['TLDW_CONFIG_PATH'])
ambient.write_text('[general]\nusers_name="original"\n'.replace('\\n','\n'))
seed=subprocess.run([sys.executable,'-c',SEED],capture_output=True,text=True,timeout=30)
assert seed.returncode==0,seed.stderr[-4000:]
source=base/'seed';identities=json.loads((source/'identities.json').read_text())
config_bytes=b'[general]\nusers_name="original"\n'.replace(b'\\n',b'\n')
doc=manifest(config_bytes);doc['owners']=[dict(owner_id='config',schema_version=1,capabilities=[])]
doc['files'][0].update(logical_id='profile:profile:config',owner_id='config',relative_path='config.toml')
doc['directories'][0]['synthetic']=True
doc['directories'].append(dict(doc['directories'][0],logical_id='data',root_id='data'))
payloads={'payload/1':config_bytes}
for kind,owner,leaf,version in [('notes','db.chachanotes.primary','tldw_chatbook_ChaChaNotes.db',identities['notes_version']),('media','db.media.primary','tldw_chatbook_media_v2.db',identities['media_version'])]:
 data=(source/(kind+'.db')).read_bytes();payload='payload/'+kind
 payloads[payload]=data
 doc['owners'].append(dict(owner_id=owner,schema_version=version,capabilities=[]))
 doc['files'].append(dict(logical_id='profile:profile:'+owner,root_id='data',parent_id='data',relative_path=leaf,owner_id=owner,payload=payload,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
doc['dependency_groups'][0]['members']=[row['logical_id'] for row in doc['files']]
doc['producer_inventory']=[dict(logical_id=row['logical_id'],owner_id=row['owner_id'],status='included',dependencies=[] if row['owner_id']=='config' else ['profile:profile:config'],shared_group=None) for row in doc['files']]
doc['producer_inventory'] += [dict(logical_id=row['logical_id'],owner_id='config' if row['root_id']=='root' else 'db.chachanotes.primary',status='included_directory',dependencies=[],shared_group=None) for row in doc['directories']]
if sys.argv[1]=='multi':
 doc['profile_ids'].append('other')
 originals=list(doc['files'])
 for directory in list(doc['directories']):
  alias=dict(directory,logical_id='alias-'+directory['logical_id'],root_id='alias-'+directory['root_id'])
  doc['directories'].append(alias)
  doc['producer_inventory'].append(dict(logical_id=alias['logical_id'],owner_id='config' if directory['root_id']=='root' else 'db.chachanotes.primary',status='included_directory',dependencies=[],shared_group=None))
 for row in originals:
  alias=dict(row,logical_id=row['logical_id'].replace('profile:profile:','profile:other:'),root_id='alias-'+row['root_id'],parent_id='alias-'+row['parent_id'],payload=row['payload']+'-alias')
  doc['files'].append(alias);payloads[alias['payload']]=payloads[row['payload']]
  shared=None if row['owner_id']=='config' else row['owner_id']
  next(item for item in doc['producer_inventory'] if item['logical_id']==row['logical_id'])['shared_group']=shared
  doc['producer_inventory'].append(dict(logical_id=alias['logical_id'],owner_id=row['owner_id'],status='included',dependencies=[] if row['owner_id']=='config' else ['profile:other:config'],shared_group=shared))
  doc['dependency_groups'][0]['members'].append(alias['logical_id'])
archive_path=base/'source.zip'
with zipfile.ZipFile(archive_path,'w') as output:
 output.writestr('manifest.json',json.dumps(doc))
 for name,data in payloads.items():output.writestr(name,data)
archive=acquire(archive_path,base/'acquired',ArchiveLimits(),None,Event())
ambient.write_bytes(b'broken = [config')
original={path:path.read_bytes() for path in (ambient,source/'notes.db',source/'media.db')}
dest=base/'destinations';dest.mkdir(mode=0o700)
destinations={'root':dest/'config','data':dest/'data'/'recovered','profile:profile:paths.data_dir':dest/'data'}
names={'profile':'recovered'}
if sys.argv[1]=='multi':
 destinations.update({'alias-root':dest/'other-config','alias-data':dest/'data'/'recovered','profile:other:paths.data_dir':dest/'data'})
 names['other']='recovered'
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names=names)
control=base/'custom-control';profile=restore_isolated(archive,plan,control,Event())
if sys.argv[1]=='multi':
 from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor
 from tldw_chatbook.Backup_Recovery.journal import Journal
 from tldw_chatbook.Backup_Recovery import bootstrap
 _,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
 assert len(profiles)==2
 journal=Journal(control,profiles[0]['activation']['operation_id'])
 with journal._locked(exclusive=False) as parent:records=journal._records(parent)
 entries=next(row.evidence['isolated_profiles'] for row in records if row.event=='prepared')
 assert len({row['profile_id'] for row in entries})==2
 assert len({row['installation_id'] for row in entries})==2
 mapped=dict(plan.restore)
 for owner in ('db.chachanotes.primary','db.media.primary'):
  a=mapped['profile:profile:'+owner];b=mapped['profile:other:'+owner]
  assert a==b and a.stat().st_ino!= (source/('notes.db' if 'chachanotes' in owner else 'media.db')).stat().st_ino
 for row in entries:assert _launch_descriptor(row['profile_id'],control).data==str(dest/'data')
# The restoring parent exits before the independent launch child runs.
(base/'launch.json').write_text(json.dumps(dict(profile=profile,control=str(control),ids=str(source/'identities.json'))))
assert all(path.read_bytes()==data for path,data in original.items())
assert not blocked_attempts()
print('retired and reopened')
"""


def test_fresh_process_reopens_real_notes_chat_media_and_local_identity(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "data",
        "isolated",
        script="SEED=" + repr(_SEED) + "\n" + _DATA_RESTORE,
    )
    root = tmp_path.resolve()
    descriptor = json.loads((root / "home" / "launch.json").read_text())
    # Installed-style private package copy prevents the source CLI's automatic
    # CSS build from editing the shared checkout before it parses --help.
    code = root / "launch-code"
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "tldw_chatbook",
        code / "tldw_chatbook",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (code / "Tests").mkdir()
    for name in ("__init__.py", "network_guard.py"):
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / name, code / "Tests" / name
        )
    (code / "sitecustomize.py").write_text(
        "import atexit, os\n"
        "from Tests.network_guard import install, blocked_attempts\n"
        "install()\n"
        "atexit.register(lambda: os._exit(73) if blocked_attempts() else None)\n"
    )
    environment = dict(
        os.environ,
        HOME=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config" / "config.toml"),
        OPENAI_API_KEY="ambient-secret-must-not-be-used",
        TLDW_MEDIA_DB_PATH="/original/custom.db",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _READ_CHILD,
            descriptor["profile"],
            descriptor["control"],
            descriptor["ids"],
        ],
        cwd=code,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "isolated read services verified" in result.stdout


_CREDENTIALS = r"""
import hashlib,json,os,shutil,sys,tomllib,zipfile
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery import archive_reader as reader,crypto,credentials
from tldw_chatbook.Backup_Recovery.models import Inventory,StorageItem
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
base=Path.home();ambient=Path(os.environ['TLDW_CONFIG_PATH']);ambient.write_bytes(b'corrupt = [ambient')
crypto._package_resource_root=lambda:Path(HELPER_ROOT)
def forbidden():raise AssertionError('ambient credential store consulted')
from keyring.backends.null import Keyring
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store=KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store=lambda:store
source=base/'owned-export';source.mkdir(mode=0o700);(source/'payload').mkdir(mode=0o700)
config=source/'payload'/'config'
config.write_text('[general]\nusers_name="original"\n[llm]\napi_key="actual-retained-secret"\n'.replace('\\n','\n'));config.chmod(0o600)
with _preview_reads():
 issues=credentials.process_credentials(source,Inventory((StorageItem('config','profile:profile:config',config,'included',()),),True,'fixture',()),mode='include',encrypted=True)
assert issues==(),issues
credentials._credential_store=forbidden
config_bytes=config.read_bytes();material=(source/'credential-recovery.json').read_bytes()
doc=manifest(config_bytes);doc['credential_policy']='include';doc['owners'][0]['owner_id']='config'
doc['owners'].append(dict(owner_id='recovery.credentials',schema_version=1,capabilities=[]))
doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
doc['dependency_groups'][0]['members']=['profile:profile:config']
doc['directories'].append(dict(doc['directories'][0],logical_id='secrets',root_id='secrets',synthetic=True))
doc['dependency_groups'].append(dict(group_id='credentials',members=['credentials'],complete=True))
doc['files'].append(dict(logical_id='credentials',root_id='secrets',parent_id='secrets',relative_path='credential-recovery.json',owner_id='recovery.credentials',payload='payload/credential-recovery.json',size=len(material),sha256=hashlib.sha256(material).hexdigest()))
zip_path=base/'input.zip'
with zipfile.ZipFile(zip_path,'w') as output:
 output.writestr('manifest.json',json.dumps(doc));output.writestr('payload/1',config_bytes);output.writestr('payload/credential-recovery.json',material)
cipher=base/'input.age';crypto.transform(zip_path,cipher,password=b'pass',decrypt=False,cancel=Event())
expected=cipher.read_bytes();archive=reader.acquire(cipher,base/'acquired',ArchiveLimits(),b'pass',Event())
cipher.write_bytes(b'original source gone');zip_path.unlink()
dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'control'
from tldw_chatbook.Backup_Recovery import space
capacity=space.require_capacity
requests=[]
def checked(requirements):
 requests.append(dict(requirements));return capacity(requirements)
space.require_capacity=checked
if sys.argv[1]=='retention_changed':
 retain=reader.retain_encrypted
 def changed(*args):
  proof=retain(*args);proof.path.write_bytes(b'replaced retained bytes');return proof
 reader.retain_encrypted=changed
 try:restore_isolated(archive,plan,control,Event())
 except ValueError as error:assert str(error)=='encrypted_retention_changed'
 else:raise AssertionError('retained digest lost acquisition binding')
 assert not (dest/'config').exists() and not (dest/'data').exists()
 print('retired and reopened');sys.exit(0)
if sys.argv[1]=='capacity':
 usage=space.shutil.disk_usage
 space.shutil.disk_usage=lambda path:usage(path)._replace(free=space._MARGIN+len(expected)-1)
 try:restore_isolated(archive,plan,control,Event())
 except ValueError as error:assert str(error)=='insufficient_space'
 else:raise AssertionError('encrypted retention capacity was ignored')
 assert requests[0]=={control:len(expected)}
 assert not (dest/'config').exists() and not (dest/'data').exists()
 print('retired and reopened');sys.exit(0)
profile=restore_isolated(archive,plan,control,Event())
assert requests[0]=={control:len(expected)}
installed,_=ProfileCatalog(control).resolve(profile)
assert 'actual-retained-secret' not in installed.read_text()
assert 'api_key' not in tomllib.loads(installed.read_text())['llm']
retained=next(control.glob('isolated-*/credentials.age'))
assert retained.read_bytes()==expected
shutil.rmtree(archive.path.parent)
assert _launch_descriptor(profile,control).profile_id==profile
reopened=reader.acquire(retained,base/'readback',ArchiveLimits(),b'pass',Event())
with zipfile.ZipFile(reopened.path) as recovered:assert recovered.read('payload/1')==config_bytes
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["credentials", "capacity", "retention_changed"])
def test_isolated_credentials_retained_encrypted_without_shared_keyring(
    tmp_path, helper_resource_root, route
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        route,
        "isolated",
        script="HELPER_ROOT=" + repr(str(helper_resource_root)) + "\n" + _CREDENTIALS,
    )


def test_multiple_profiles_publish_shared_databases_once(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "multi",
        "isolated",
        script="SEED=" + repr(_SEED) + "\n" + _DATA_RESTORE,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--recovery-profile", "one"],
        [
            "--recovery-profile",
            "one",
            "--recovery-profile=two",
            "--recovery-control-root",
            "/tmp/control",
        ],
    ],
)
def test_cli_rejects_incomplete_or_duplicate_selectors(monkeypatch, args):
    import sys

    from tldw_chatbook.cli import main_cli_runner

    monkeypatch.setattr(sys, "argv", ["tldw-cli", *args])
    with pytest.raises(SystemExit) as error:
        main_cli_runner()
    assert error.value.code == 2


def test_isolated_missing_owner_capability_refuses_before_targets(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, unknown=True)
    destination = tmp_path / "new-profile"
    with pytest.raises(ValueError, match="unsupported_owner"):
        plan_restore(
            archive, mode="isolated", destinations={"root": destination}, target=None
        )
    assert not destination.exists()
