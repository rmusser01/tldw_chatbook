"""Whole installed Chroma roots are copied and validated without live reopening."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = (
    r"""
import asyncio, hashlib, json, os, sqlite3, subprocess, sys, threading, time
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture, capture
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME'])/'fixture'
source.write_text('[paths]\ndata_dir='+json.dumps(str(data))+'\n')
source.chmod(0o600)
root = data/'default_user'/'chromadb'
root.parent.mkdir(parents=True,mode=0o700)
route = sys.argv[1]
if route == 'empty':
 root.mkdir(parents=True, mode=0o700)
else:
 from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
 store = ChromaVectorStore(root, collection_name='first')
 store.add([str(i) for i in range(1100)],[[1.,0.]]*1100,['kept']*1100,[{'doc_id':str(i)} for i in range(1100)])
 second = ChromaVectorStore(root,collection_name='second')
 second.add(['one'],[[0.,1.]],['other'],[{'doc_id':'one'}])
 assert store.client._system is second.client._system
 from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
 async def drain():
  participant._maintenance_close_admission()
  assert await participant._maintenance_drain(time.monotonic()+5)
 asyncio.run(drain())
 if route == 'missing':
  next(root.glob('*/header.bin')).unlink()
 elif route == 'missing_directory':
  import shutil
  shutil.rmtree(next(p for p in root.iterdir() if p.is_dir()))
 elif route == 'foreign':
  db=sqlite3.connect(root/'chroma.sqlite3')
  db.execute('UPDATE collections SET config_json_str=?', (json.dumps({'embedding_function':{'type':'known','name':'openai','config':{'api_key':'synthetic-managed-secret'}}}),))
  db.commit(); db.close()
 elif route == 'corrupt':
  (root/'chroma.sqlite3').write_bytes(b'not a sqlite database')
 elif route == 'identifier':
  db=sqlite3.connect(root/'chroma.sqlite3')
  db.execute("UPDATE segments SET id='../outside' WHERE id=(SELECT id FROM segments WHERE scope='VECTOR' LIMIT 1)")
  db.commit(); db.close()
 elif route == 'pickle':
  next(root.glob('*/index_metadata.pickle')).write_bytes(b"cos\nsystem\n(S'touch outside'\ntR.")
original={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
admission_authority(bootstrap.default_bootstrap_root())
producer="""
    + '"""'
    + r"""
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_populate_required_dependencies(preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True}))
"""
    + '"""'
    + r"""
prepared=subprocess.run([sys.executable,'-c',producer],capture_output=True,text=True,timeout=30)
assert prepared.returncode==0,prepared.stderr[-1000:]
options={'allow_partial':True,'staging_parent':Path.home()}
async def live_capture():
 for name in ('sounddevice','pyaudio'): sys.modules[name]=None
 import keyring
 from keyring.backends.null import Keyring
 keyring.set_keyring(Keyring())
 from tldw_chatbook.app import TldwCli
 from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
 from tldw_chatbook.Backup_Recovery import storage_admission as storage
 app=TldwCli()
 runtime=RuntimeMaintenance(app)
 app._backup_runtime_maintenance=runtime
 try:
  await participant._maintenance_resume()
  await runtime.settle_producers(time.monotonic()+10)
  runtime.retire_local_caches()
  assert runtime.pause.drain(time.monotonic()+2)
  reviewed=preview_capture((source,),options=options)
  pending=asyncio.create_task(asyncio.to_thread(capture,(source,),reviewed.scope_digest,Path.home()/'backup.tldw-backup.zip',options=options,cancel=Event()))
  while not storage._local_pause_requested():
   if pending.done(): return pending.result()
   await asyncio.sleep(.01)
  runtime.pause.retire_startup(runtime)
  return await pending
 finally:
  await runtime.resume()
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
preview=preview_capture((source,),options=options)
items=[i for i in preview.items if i.owner=='rag.projections']
assert all(i.status in {'included','included_directory'} for i in items), [(str(i.path),i.status) for i in items]
try:
 result=(asyncio.run(live_capture()) if route!='empty' else capture((source,),preview.scope_digest,Path.home()/'backup.tldw-backup.zip',options=options,cancel=Event()))
except ValueError as error:
 assert route in {'missing','missing_directory','foreign','corrupt','identifier','pickle'}, str(error)
 assert 'rag_projection' in str(error), str(error)
else:
 assert route in {'native','empty'}, 'invalid native candidate accepted'
 doc=json.loads(result.manifest_bytes)
 files=[f for f in doc['files'] if f['owner_id']=='rag.projections']
 assert {f['relative_path']:hashlib.sha256((result.root/f['payload']).read_bytes()).hexdigest() for f in files}==original
 if route=='empty':
  assert not list(root.iterdir())
 else:
  # Reassemble archive bytes, never alias payload or live source files.
  reopened=Path.home()/'reopened'; reopened.mkdir(mode=0o700)
  for f in files:
   target=reopened/f['relative_path']; target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
   target.write_bytes((result.root/f['payload']).read_bytes())
  import chromadb
  from chromadb.config import Settings
  client=chromadb.PersistentClient(path=str(reopened),settings=Settings(anonymized_telemetry=False,migrations='validate'))
  try:
   first=client.get_collection('first',embedding_function=None)
   assert first.count()==1100
   row=first.get(ids=['1'],include=['documents','embeddings'])
   assert row['documents']==['kept'] and row['embeddings'].tolist()==[[1.,0.]]
   assert client.get_collection('second',embedding_function=None).get()['documents']==['other']
  finally: client.close()
  from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
  from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore, _document
  from tldw_chatbook.Backup_Recovery.staging import stage_restore, _items
  from tldw_chatbook.Backup_Recovery.publication import _validate_installed_copies
  from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
  # Fresh standalone source scope avoids unrelated app-owner restore work.
  from tldw_chatbook.Backup_Recovery.capture import CaptureResult, _manifest_for
  from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections
  from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
  from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
  from tldw_chatbook.Backup_Recovery.inventory import classify_entries
  from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
  fixture=Path.home()/'stage-source'
  fixture_root=fixture/'default_user'/'chromadb'
  fixture_root.mkdir(parents=True,mode=0o700)
  for f in files:
   target=fixture_root/f['relative_path']; target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
   target.write_bytes((result.root/f['payload']).read_bytes()); target.chmod(0o600)
  selector=Path.home()/'stage-config.toml'
  selector.write_text('[paths]\ndata_dir='+json.dumps(str(fixture))+'\n'); selector.chmod(0o600)
  configuration={'paths':{'data_dir':str(fixture)},DISCOVERY_CONTEXT_KEY:DiscoveryContext(selector,'stage')}
  projection=_Projections('rag.projections')
  scope=classify_entries((*projection.discover(configuration),_RawDeclaration('config')._item(configuration,selector)))
  assert scope.complete,scope.issues
  payload_root=Path.home()/'stage-payloads'; payload_root.mkdir(mode=0o700)
  (payload_root/'payload').mkdir(mode=0o700)
  payloads=[]
  for item in scope.items:
   if item.status=='included':
    path=payload_root/'payload'/hashlib.sha256(item.logical_id.encode()).hexdigest()
    path.write_bytes(item.path.read_bytes()); path.chmod(0o600)
    payloads.append((item,path))
  encoded=_manifest_for(scope,payloads,{},dict(root=payload_root,versions={},mode='exclude',encrypted=False,limits=ArchiveLimits(),cancel=Event()),())
  fixture_capture=CaptureResult(payload_root,scope,encoded)
  archive_area=Path.home()/'archive-area'; archive_area.mkdir(mode=0o700)
  archive=write_archive(fixture_capture,archive_area/'packaged.tldw-backup.zip',password=None,cancel=Event())
  fixture_doc=json.loads(encoded)
  isolated=Path.home()/'isolated'
  destinations={}
  for row in fixture_doc['directories']:
   if row['parent_id'] is None:
    destinations[row['logical_id']]=(isolated/'config' if row.get('synthetic') else isolated/'data'/'restored'/'chromadb')
  destinations['profile:stage:paths.data_dir']=isolated/'data'
  plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={'stage':'restored'})
  candidate=stage_restore(archive,plan,Path.home()/'restore-work',Event())
  assert candidate.is_dir() and not isolated.exists()
  # Exercise the exact installed-copy callback using checked private copies.
  document=_document(archive)
  private_items={key:item for key,item in _items(document,plan).items() if item.owner=='rag.projections'}
  paths={key:reopened/item.metadata.relative_path for key,item in private_items.items()}
  topology={key:(item.metadata.root_id,item.metadata.parent_id,item.metadata.relative_path,item.metadata.kind) for key,item in private_items.items()}
  owners={owner.owner_id:owner for owner in install_adapters()}
  _validate_installed_copies(private_items,paths,topology,set(),owners)
assert {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}==original
if route!='empty':
 asyncio.run(participant._maintenance_resume())
 if route=='native':
  assert store.search([1.,0.],top_k=1)[0].document=='kept'
  store.close(); second.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route",
    [
        "native",
        "empty",
        "missing",
        "missing_directory",
        "foreign",
        "corrupt",
        "identifier",
        "pickle",
    ],
)
def test_public_native_projection_capture(tmp_path, route):
    _run(tmp_path, route, "capture", script=_SCRIPT)


def test_cancel_reaps_native_child_before_removing_private_root(tmp_path, monkeypatch):
    import subprocess
    import sys
    from threading import Event

    from tldw_chatbook.Backup_Recovery import rag_projection_validation as validation
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration

    root = tmp_path / "original"
    root.mkdir(mode=0o700)
    database = root / "chroma.sqlite3"
    database.write_bytes(b"copied opaque candidate")
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "test")}
    items = _RawDeclaration("rag.projections")._tree(config, root)
    candidates = {i.logical_id: i.path for i in items if i.status == "included"}
    cancel = Event()
    original = subprocess.Popen
    processes = []

    def paused_child(arguments, **kwargs):
        # A real owned process, stopped before any candidate engine open.
        child = original(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        processes.append(child)
        cancel.set()
        return child

    monkeypatch.setattr(validation.subprocess, "Popen", paused_child)
    with pytest.raises(InterruptedError):
        validation.validate_groups(
            items, candidates, tmp_path, cancel, ArchiveLimits(), 1024
        )
    assert processes[0].poll() is not None
    assert not list(tmp_path.glob("rag-validation-*"))
    assert database.read_bytes() == b"copied opaque candidate"


def test_shared_roots_declare_aliases_and_complete_member_dependencies(
    tmp_path, monkeypatch
):
    import json

    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.delenv("RAG_VECTOR_STORE", raising=False)
    root = tmp_path / "shared"
    root.mkdir(mode=0o700)
    (root / "chroma.sqlite3").write_bytes(b"inert preview")
    configs = []
    for name in ("first", "second"):
        config = tmp_path / (name + ".toml")
        config.write_text(
            "[general]\nusers_name="
            + json.dumps(name)
            + "\n[paths]\ndata_dir="
            + json.dumps(str(tmp_path / name))
            + "\n[AppRAGSearchConfig.rag.vector_store]\npersist_directory="
            + json.dumps(str(root))
            + "\n"
        )
        config.chmod(0o600)
        configs.append(config)
    result = discover(tuple(configs))
    members = [
        i
        for i in result.items
        if i.owner == "rag.projections" and i.path in {root, root / "chroma.sqlite3"}
    ]
    assert len(members) == 4
    assert not {
        "undeclared_alias",
        "shared_identity_mismatch",
        "overlapping_owner_roots",
    } & set(result.issues)
    aliases = {
        i.path: {j.shared_group for j in members if j.path == i.path} for i in members
    }
    assert all(len(groups) == 1 and None not in groups for groups in aliases.values())
    assert aliases[root] != aliases[root / "chroma.sqlite3"]
    for item in members:
        if item.path == root:
            assert {
                i.logical_id for i in members if i.metadata.root_id == item.logical_id
            } <= set(item.dependencies)


def test_private_validation_refuses_omitted_root_member_before_child(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from threading import Event

    from tldw_chatbook.Backup_Recovery import rag_projection_validation as validation
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration

    root = tmp_path / "original"
    root.mkdir(mode=0o700)
    (root / "chroma.sqlite3").write_bytes(b"candidate")
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "test")}
    items = _RawDeclaration("rag.projections")._tree(config, root)
    parent = next(i for i in items if i.path == root)
    parent = replace(parent, dependencies=tuple(i.logical_id for i in items))
    with pytest.raises(ValueError, match="rag_projection_root_incomplete"):
        validation.validate_groups(
            (parent,), {}, tmp_path, Event(), ArchiveLimits(), 1024
        )
    assert not list(tmp_path.glob("rag-validation-*"))


def test_fast_child_exit_still_checks_regular_file_growth(tmp_path, monkeypatch):
    import subprocess
    import sys
    from threading import Event

    from tldw_chatbook.Backup_Recovery import rag_projection_validation as validation

    original = subprocess.Popen
    processes = []

    def fast_child(arguments, **kwargs):
        child = original(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; (Path(sys.argv[1])/'growth').write_bytes(b'x'*32)",
                arguments[3],
            ],
            **kwargs,
        )
        child.wait()
        processes.append(child)
        return child

    monkeypatch.setattr(validation.subprocess, "Popen", fast_child)
    with pytest.raises(ValueError, match="rag_projection_validation_budget"):
        validation._run_validator(tmp_path, Event(), byte_budget=16)
    assert processes[0].returncode == 0


def test_private_native_candidate_reads_copied_wal_without_runtime_enrollment(tmp_path):
    script = r"""
import hashlib, shutil, sqlite3
from pathlib import Path
from Tests import network_guard
network_guard.install()
import chromadb
from chromadb.config import Settings
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Backup_Recovery.rag_projection_validation import _preflight, _native_validate
root = Path.home()/'original-chroma'
client = chromadb.PersistentClient(path=str(root), settings=Settings(anonymized_telemetry=False))
client.create_collection('before', embedding_function=None)
client.close()
writer = sqlite3.connect(root/'chroma.sqlite3')
writer.execute('PRAGMA journal_mode=WAL')
writer.execute('PRAGMA wal_autocheckpoint=0')
writer.execute("UPDATE collections SET name='committed-in-wal'")
writer.commit()
assert (root/'chroma.sqlite3-wal').stat().st_size > 0
copy = Path.home()/'private-candidate'
copy.mkdir(mode=0o700)
for source in root.iterdir():
    target = copy/source.name
    shutil.copyfile(source, target)
    target.chmod(0o600)
original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
immutable = sqlite3.connect((copy/'chroma.sqlite3').as_uri()+'?immutable=1', uri=True)
assert immutable.execute('SELECT name FROM collections').fetchone() == ('before',)
immutable.close()
def no_runtime(*args, **kwargs):
    raise AssertionError('disposable validation enrolled ordinary runtime')
storage_admission.acquire_storage = no_runtime
assert _preflight(copy)[0][1] == 'committed-in-wal'
_native_validate(copy)
assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
writer.close()
print('retired and reopened')
"""
    _run(tmp_path, "wal", "validation", script=script)


@pytest.mark.parametrize("selector", ["custom", "default"])
def test_isolated_profile_selector_is_refused_without_rewriting_capture(
    tmp_path, selector
):
    script = r"""
import hashlib, json, sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.capture import CaptureResult, _manifest_for
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.Backup_Recovery.rag_inventory import _Definitions, _Projections
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
install_adapters()
root=Path.home(); data=root/'source-data'; profiles=data/'default_user'/'rag_profiles'
profiles.mkdir(parents=True,mode=0o700)
index=root/'original-custom-index'; index.mkdir(mode=0o700)
selector=index if sys.argv[1]=='custom' else None
profile=profiles/'retained.json'
original=json.dumps({'name':'persist_directory history', 'description':str(index), 'rag_config':{'vector_store':{'persist_directory':str(selector) if selector else None}}}).encode()
profile.write_bytes(original); profile.chmod(0o600)
config=root/'source.toml'; config.write_text('[paths]\ndata_dir='+json.dumps(str(data))+'\n'); config.chmod(0o600)
configuration={'paths':{'data_dir':str(data)},DISCOVERY_CONTEXT_KEY:DiscoveryContext(config,'source')}
definitions=_Definitions('rag.definitions')
scope=classify_entries((*definitions.discover(configuration),*_Projections('rag.projections').discover(configuration),_RawDeclaration('config')._item(configuration,config)))
assert scope.complete,scope.issues
payload_root=root/'payloads'; payload_root.mkdir(mode=0o700); (payload_root/'payload').mkdir(mode=0o700)
payloads=[]
for item in scope.items:
 if item.status=='included':
  path=payload_root/'payload'/hashlib.sha256(item.logical_id.encode()).hexdigest()
  path.write_bytes(item.path.read_bytes()); path.chmod(0o600); payloads.append((item,path))
encoded=_manifest_for(scope,payloads,{},dict(root=payload_root,versions={},mode='exclude',encrypted=False,limits=ArchiveLimits(),cancel=Event()),())
area=root/'archives'; area.mkdir(mode=0o700)
archive=write_archive(CaptureResult(payload_root,scope,encoded),area/'backup.tldw-backup.zip',password=None,cancel=Event())
document=json.loads(encoded); isolated=root/'isolated'; destinations={}
for row in document['directories']:
 if row['parent_id'] is None:
  destinations[row['logical_id']]=isolated/('config' if row.get('synthetic') else ('data/restored/rag_profiles' if ':rag.definitions:' in row['logical_id'] else 'data/restored/chromadb'))
destinations['profile:source:paths.data_dir']=isolated/'data'
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={'source':'restored'})
try:
 candidate=stage_restore(archive,plan,root/'work',Event())
except ValueError as error:
 assert selector is not None and str(error)=='rag_definition_root_mapping_required',str(error)
else:
 assert selector is None,'unmapped original live selector accepted'
assert profile.read_bytes()==original and not list(index.iterdir())
assert next(path for item,path in payloads if item.path==profile).read_bytes()==original
print('retired and reopened')
"""
    _run(tmp_path, selector, "restore", script=script)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("rag_projection_record_limit", "rag_projection_validation_budget"),
        ("rag_projection_version_unsupported", "rag_projection_format_unsupported"),
        ("rag_projection_schema_unsupported", "rag_projection_format_unsupported"),
        (
            "rag_projection_foreign_execution_configuration",
            "rag_projection_foreign_execution_configuration",
        ),
        ("untrusted arbitrary error secret", "rag_projection_candidate_invalid"),
    ],
)
def test_native_child_known_refusals_use_fixed_statuses(
    tmp_path, monkeypatch, failure, expected
):
    import subprocess
    import sys
    from threading import Event

    from tldw_chatbook.Backup_Recovery import rag_projection_validation as validation

    original = subprocess.Popen

    def child(arguments, **kwargs):
        return original(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from tldw_chatbook.Backup_Recovery import rag_projection_validation as m; m._native_validate=lambda root: (_ for _ in ()).throw(ValueError(sys.argv[1])); sys.exit(m._main(Path(sys.argv[2])))",
                failure,
                arguments[3],
            ],
            **kwargs,
        )

    monkeypatch.setattr(validation.subprocess, "Popen", child)
    with pytest.raises(ValueError, match="^" + expected + "$"):
        validation._run_validator(tmp_path, Event(), byte_budget=1024)
