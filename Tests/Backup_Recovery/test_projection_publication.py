"""Dependent projections retire through reviewed, held publication operations."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections


def omitted_projection(live):
    """Declare exact omitted topology; planning never needs to open engine bytes."""
    root = live / "chromadb"
    root.mkdir(mode=0o700)
    (root / "chroma.sqlite3").write_bytes(b"opaque planning fixture")
    configuration = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(live / "config.toml", "profile")
    }
    items = _Projections("rag.projections")._tree(configuration, root)
    return tuple(
        replace(
            item,
            status="intentionally_excluded",
            dependencies=(*item.dependencies, "profile:profile:research.local"),
        )
        for item in items
    )


def test_omitted_projection_is_retired_when_authoritative_source_replaced(
    tmp_path, monkeypatch
):
    with replacement_case(tmp_path, monkeypatch, extras=omitted_projection) as case:
        plan = case[1]
        projections = {
            (item.logical_id, item.path)
            for item in plan.target.items
            if item.owner == "rag.projections"
        }
        assert projections <= set(plan.retire)
        assert not projections.intersection(plan.preserve)
        assert any(
            issue.startswith("projection_reconciliation_required:")
            for issue in plan.issues
        )


_NATIVE_SCRIPT = r"""
import asyncio, hashlib, json, os, sqlite3, subprocess, sys, time, zipfile
from dataclasses import replace
from pathlib import Path
from threading import Event
import pytest
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case, run_capture
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
case_root=Path.home()/'case'; case_root.mkdir(mode=0o700)
route=sys.argv[1]
original={}
def extras(live):
 root=live/'chromadb'
 store=ChromaVectorStore(root,collection_name='retained')
 store.add([str(i) for i in range(1100)],[[1.,0.]]*1100,['original']*1100,[{'doc_id':str(i)} for i in range(1100)])
 async def drain():
  participant._maintenance_close_admission()
  assert await participant._maintenance_drain(time.monotonic()+10)
 asyncio.run(drain())
 # Establish real committed WAL-only state during fixture construction, before
 # admission/snapshot proof; production proof never opens these live originals.
 subprocess.run([sys.executable,'-c',"import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0'); c.execute(\"UPDATE collections SET name='wal-retained'\"); c.commit(); os._exit(0)",str(root/'chroma.sqlite3')],check=True,timeout=15)
 if route=='corrupt': next(root.glob('*/header.bin')).unlink()
 original.update({str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
 assert 'chroma.sqlite3-wal' in original and original['chroma.sqlite3-wal']
 config={DISCOVERY_CONTEXT_KEY:DiscoveryContext(live/'config.toml','profile')}
 items=_Projections('rag.projections')._tree(config,root)
 return tuple(replace(item,status='intentionally_excluded',dependencies=(*item.dependencies,'profile:profile:research.local')) for item in items)
with pytest.MonkeyPatch.context() as patch:
 patch.setattr(crypto,'_package_resource_root',lambda:Path(HELPER_ROOT))
 with replacement_case(case_root,patch,extras=extras) as case:
  candidate,plan,journal,session,source,selector=case
  root=source.parent/'chromadb'
  if route=='corrupt':
   try: run_capture(case,case_root)
   except ValueError as e: assert 'rag_projection' in str(e),str(e)
   else: raise AssertionError('incomplete native group authorized rollback')
   with journal._locked(exclusive=False) as fd: assert journal._records(fd)[-1].event=='prepared'
   assert not bootstrap.startup_permission(selector,case_root/'bootstrap')[0]
  else:
   with journal._locked(exclusive=False) as fd:
    prepared=publication._Prepared.model_validate(journal._records(fd)[-1].evidence)
   roots=[r for a in prepared.artifacts for r in a.rollback_projection_roots]
   assert len(roots)==1 and roots[0].path==str(root)
   coverage={a.logical_id:a.logical_id for a in prepared.artifacts if a.previous is not None}
   try: journal.verify_rollback(source,password=b'test-only-password',work_root=case_root/'forbidden',cancel=Event(),coverage=coverage)
   except ValueError as e: assert str(e)=='rollback_projection_held_capture_required',str(e)
   else: raise AssertionError('caller supplied group receipt accepted')
   archive_path=run_capture(case,case_root)
   with journal._locked(exclusive=False) as fd:
    proof=journal._records(fd)[-1]
   assert proof.event=='rollback_verified' and proof.evidence['projection_groups']==[r.model_dump() for r in roots]
   from tldw_chatbook.Backup_Recovery.journal import _validate
   try: journal.record('rollback_verified',proof.evidence)
   except ValueError as error: assert str(error)=='rollback_held_capture_required'
   else: raise AssertionError('public held projection receipt accepted')
   with journal._locked(exclusive=False) as fd: prior=journal._records(fd)[:-1]
   for damaged in ([],[dict(roots[0].model_dump(),path=str(root.parent/'foreign'))],[roots[0].model_dump(),roots[0].model_dump()]):
    try: _validate('rollback_verified',dict(proof.evidence,projection_groups=damaged),prior)
    except ValueError as error: assert str(error)=='journal_evidence_invalid'
    else: raise AssertionError('incomplete or foreign projection group accepted')
   assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==original
   archive=acquire(archive_path,case_root/'reopen',ArchiveLimits(),b'test-only-password',Event())
   doc=verify_sealed(archive)
   reopened=case_root/'reassembled'; reopened.mkdir(mode=0o700)
   with zipfile.ZipFile(archive.path) as packed:
    projection_files=[f for f in doc.files if f.owner_id=='rag.projections']
    assert {f.relative_path:packed.read(f.payload) for f in projection_files}==original
    for f in projection_files:
     path=reopened/f.relative_path; path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
     path.write_bytes(packed.read(f.payload));path.chmod(0o600)
   import chromadb
   from chromadb.config import Settings
   client=chromadb.PersistentClient(path=str(reopened),settings=Settings(anonymized_telemetry=False,migrations='validate'))
   try:
    collection=client.get_collection('wal-retained',embedding_function=None)
    assert collection.count()==1100
    row=collection.get(ids=['17'],include=['embeddings','documents'])
    assert row['documents']==['original'] and row['embeddings'].tolist()==[[1.,0.]]
   finally: client.close()
   if route=='interrupted':
    retire=publication._retire
    def fail(item):
     if item.target==str(root): raise InterruptedError('fixture interruption')
     retire(item)
    patch.setattr(publication,'_retire',fail)
    try: publication.publish_candidate(candidate,plan,journal,archive_path)
    except InterruptedError: pass
    else: raise AssertionError('interruption ignored')
    assert not bootstrap.startup_permission(selector,case_root/'bootstrap')[0]
    assert root.exists()
    patch.setattr(publication,'_retire',retire)
   publication.publish_candidate(candidate,plan,journal,archive_path)
   assert not root.exists()
   assert all(state in {'retired','published'} for state in journal.artifact_states().values())
   generation=publication.finalize_candidate(candidate,plan,journal,session=session)
   assert generation=='new'
   from tldw_chatbook.Backup_Recovery.activation import activation_permission
   assert not activation_permission('rag.projections',config_selector=selector,bootstrap_root=case_root/'bootstrap')
 asyncio.run(participant._maintenance_resume())
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["native", "corrupt", "interrupted"])
def test_held_native_projection_rollback(tmp_path, helper_resource_root, route):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = "HELPER_ROOT=" + repr(str(helper_resource_root)) + "\n" + _NATIVE_SCRIPT
    _run(tmp_path, route, "rollback", script=script)


def test_independent_projection_remains_preserved(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.projection_publication import (
        dependent_retirements,
    )

    live = tmp_path / "live"
    live.mkdir()
    projections = omitted_projection(live)
    source = StorageItem(
        "research.local",
        "profile:profile:research.local",
        live / "source.db",
        "included",
        (),
    )
    config = StorageItem(
        "config", "profile:profile:config", live / "config.toml", "included", ()
    )
    unrelated = live / "unrelated.db"
    preserved = [(item.logical_id, item.path) for item in projections]
    retired, kept, issues = dependent_retirements(
        Inventory((*projections, source, config), True, "scope", ()),
        (("other", unrelated),),
        (),
        tuple(preserved),
    )
    assert (retired, kept, issues) == ([], sorted(preserved), ())


@pytest.mark.parametrize(
    "kind",
    [
        "shared",
        "unknown_source",
        "unselected_source",
        "unknown_file",
        "missing_member",
        "unsupported",
        "occupied_unused",
    ],
)
def test_unreviewed_projection_scope_refuses(tmp_path, monkeypatch, kind):
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    def extras(live):
        items = omitted_projection(live)
        if kind == "shared":
            config = {
                DISCOVERY_CONTEXT_KEY: DiscoveryContext(live / "other.toml", "other")
            }
            other = _Projections("rag.projections")._tree(config, live / "chromadb")
            return (
                *items,
                *(replace(item, status="intentionally_excluded") for item in other),
            )
        if kind == "unselected_source":
            other = live / "unselected.db"
            other.write_bytes(b"preserved source")
            return (
                *(
                    replace(item, dependencies=(*item.dependencies, "unselected"))
                    for item in items
                ),
                StorageItem(
                    "research.local", "unselected", other, "intentionally_excluded", ()
                ),
            )
        if kind == "occupied_unused":
            return tuple(
                replace(item, status="unused", metadata=None) for item in items
            )
        if kind == "unknown_source":
            return tuple(
                replace(item, dependencies=(*item.dependencies, "unmapped-owner"))
                for item in items
            )
        if kind == "unknown_file":
            (live / "chromadb" / "undeclared.bin").write_bytes(b"unclassified")
        if kind == "missing_member":
            return items[:1]
        if kind == "unsupported":
            return (
                *items,
                StorageItem(
                    "rag.projections",
                    "selector-unverified",
                    None,
                    "unsupported",
                    ("profile:profile:research.local",),
                ),
            )
        return items

    with (
        pytest.raises(ValueError, match="projection_|shared_scope_|target_unverified"),
        replacement_case(tmp_path, monkeypatch, extras=extras),
    ):
        pytest.fail("unreviewed group accepted")


def test_projection_payload_requires_owned_whole_root(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def mutate(doc):
        doc["owners"][0]["owner_id"] = "rag.projections"
        doc["files"][0]["owner_id"] = "rag.projections"

    archive = sealed(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="projection_root_incomplete"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "isolated"},
            target=None,
        )


_RESTORED_SCRIPT = r"""
import asyncio, hashlib, json, os, sys, time
from pathlib import Path
from threading import Event
import pytest
from Tests import network_guard
network_guard.install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication
from tldw_chatbook.Backup_Recovery.capture import CaptureResult, _manifest_for
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, bind_profile, register_pending
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
base=Path.home()
live=base/'live';live.mkdir(mode=0o700)
data=live/'data';data.mkdir(mode=0o700)
source_root=data/'default_user'/'chromadb'
(live/'config').mkdir(mode=0o700)
selector=live/'config'/'config.toml'
selector.write_text('[paths]\ndata_dir='+json.dumps(str(data))+'\n');selector.chmod(0o600)
config={'paths':{'data_dir':str(data)},DISCOVERY_CONTEXT_KEY:DiscoveryContext(selector,'profile')}
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
services=[RAGService(RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'memory'},'search':{'semantic_cache_ttl':3600}})) for _ in range(2)]
for service in services:
 service.cache.put('old','semantic',1,['stale-result'],'old-context')
 assert service.cache.get('old','semantic',1) is not None
store=ChromaVectorStore(source_root,collection_name='source')
store.add(['one'],[[1.,0.]],['incoming'],[{'doc_id':'one'}])
async def drain():
 participant._maintenance_close_admission()
 assert await participant._maintenance_drain(time.monotonic()+10)
asyncio.run(drain())
projection=_Projections('rag.projections')
raw=_RawDeclaration('config')
scope=classify_entries((*projection.discover(config),raw._item(config,selector)))
assert scope.complete,scope.issues
payload_root=base/'payloads';payload_root.mkdir(mode=0o700)
(payload_root/'payload').mkdir(mode=0o700)
payloads=[]
for item in scope.items:
 if item.status=='included':
  path=payload_root/'payload'/hashlib.sha256(item.logical_id.encode()).hexdigest()
  path.write_bytes(item.path.read_bytes());path.chmod(0o600);payloads.append((item,path))
encoded=_manifest_for(scope,payloads,{},dict(root=payload_root,versions={},mode='exclude',encrypted=False,limits=ArchiveLimits(),cancel=Event()),())
output=base/'archive';output.mkdir(mode=0o700)
archive=write_archive(CaptureResult(payload_root,scope,encoded),output/'source.tldw-backup.zip',password=None,cancel=Event())
doc=json.loads(encoded)
mode=sys.argv[1]
if mode=='replace':
 asyncio.run(participant._maintenance_resume())
 store.add(['target-only'],[[0.,1.]],['discarded target row'],[{'doc_id':'target-only'}])
 asyncio.run(drain())
 target=classify_entries((*projection.discover(config),raw._item(config,selector)))
 destination_root=source_root;destination_selector=selector;destination_data=data
else:
 target=None
 isolated_container=base/'destination';isolated_container.mkdir(mode=0o700)
 isolated=isolated_container/'isolated'
 destination_root=isolated/'data'/'default_user'/'chromadb'
 destination_selector=isolated/'config'/'config.toml';destination_data=isolated/'data'
destinations={row['logical_id']:(destination_selector.parent if row.get('synthetic') else destination_root) for row in doc['directories'] if row['parent_id'] is None}
destinations['profile:profile:paths.data_dir']=destination_data
plan=plan_restore(archive,mode=mode,destinations=destinations,target=target,profile_names={'profile':'default_user'})
assert any(issue.startswith('projection_reconciliation_required:') for issue in plan.issues)
control=base/'control';control.mkdir(mode=0o700)
journal=Journal(control,'projection-publication')
candidate=stage_restore(archive,plan,base/'stage',Event(),journal=journal)
root=base/'bootstrap'
authority=admission_authority(root)
authority.register('profile',(live,) if mode=='replace' else (isolated_container,))
if mode=='replace':bind_profile(root,selector,('profile',),root/'admission')
with pytest.MonkeyPatch.context() as patch:
 patch.setattr(bootstrap,'default_bootstrap_root',lambda:root)
 patch.setattr(crypto,'_package_resource_root',lambda:Path(HELPER_ROOT))
 from tldw_chatbook.Backup_Recovery import credentials
 from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
 credential_store=KeyringServerCredentialStore(keyring_backend=Keyring())
 patch.setattr(credentials,'_credential_store',lambda:credential_store)
 register_pending(root,journal.operation_id,('profile',),control,(destination_selector,))
 with authority.maintenance(('profile','bootstrap.unbound'),5) as session:
  journal.prepare_publication(candidate,plan,bootstrap_root=root,namespaces=('profile',),selectors=(destination_selector,),generation='restored')
  rollback=None
  if mode=='replace':
   from tldw_chatbook.Backup_Recovery.replacement import capture_verify_rollback
   rollback=capture_verify_rollback(candidate,plan,journal,base/'rollback.tldw-backup.zip.age',session=session,password=b'fixture-password',work_root=base/'rollback-work',cancel=Event(),acknowledged_credential_issues=())
  else:assert not destination_root.exists() and not destination_selector.exists()
  publication.publish_candidate(candidate,plan,journal,rollback)
  assert all(service.cache.get('old','semantic',1) is None for service in services)
  publication.finalize_candidate(candidate,plan,journal,session=session)
  for item,payload in payloads:
   if item.owner=='rag.projections':assert (destination_root/item.metadata.relative_path).read_bytes()==payload.read_bytes()
 asyncio.run(participant._maintenance_resume())
 # A published structural projection is blocked, even after generic runtime approval.
 from tldw_chatbook.Backup_Recovery.activation import ActivationStore
 ActivationStore(control/'activation').approve('restored','rag.projections')
 selected=ChromaVectorStore(destination_root,collection_name='source')
 try:
  try:selected.search([1.,0.])
  except (ValueError,RuntimeError) as error:assert 'projection' in str(error),str(error)
  else:raise AssertionError('imported projection became retrieval ready')
 finally:selected.close()
print('retired and reopened')
"""


@pytest.mark.parametrize("mode", ["isolated", "replace"])
def test_restored_whole_root_requires_reconciliation(
    tmp_path, helper_resource_root, mode
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = "HELPER_ROOT=" + repr(str(helper_resource_root)) + "\n" + _RESTORED_SCRIPT
    _run(tmp_path, mode, "publication", script=script)
