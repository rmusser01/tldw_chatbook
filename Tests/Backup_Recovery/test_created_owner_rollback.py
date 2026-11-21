"""A later rollback retires what a cross-profile restore created (TASK-41521).

Each case backs up a source profile that holds files a populated target lacks,
restores that backup into the target (another profile ID, so every created
file goes through ``later_rollback._created_destination_target``), and rolls
the restore back later. The created single files and databases must be gone
again, with their SQLite sidecars; a refusal case must refuse by name and leave
its file. One orchestrating child per profile, as in
``test_created_persona_subtree_rollback.py``; the producers run in their own
processes so the orchestrator never imports config.
"""

import sys

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

# name: groups, operational stores (test_operational_owners.STORES), other
# databases (module, class), files {label: (folder, leaf, text)} where folder is
# data or config, databases the target deletes first, created databases a
# writer then leaves -wal/-shm files beside ("wal"), whether a second
# profile's config joins the config folder after the restore
# ("second_profile"), and the refusal a later rollback must give (None: it
# retires everything).
_CASES = {
    # Databases a populated target lacks or has deleted, the TTS profile store
    # with its shared reference audio, the dated config copy and a chatbook
    # registry with its archive folder.
    "databases": {
        "groups": (
            "audio",
            "automation",
            "chatbooks",
            "conversations",
            "prompts",
            "research",
            "settings",
            "workspaces",
        ),
        "stores": (
            "kanban",
            "subscriptions",
            "scheduled_tasks",
            "notifications",
            "events",
            "sync",
        ),
        "extra": (
            ("db.dreams", "DB.Dreams_DB", "DreamsDB"),
            ("db.guardian", "DB.Guardian_DB", "GuardianDB"),
        ),
        "files": {
            "config.history": (
                "config",
                "config.toml.bak",
                '[general]\nusers_name="old"\n',
            ),
            "chatbooks.archives": (
                "data",
                "chatbooks/saved.zip",
                "saved chatbook bytes",
            ),
            "chatbooks.registry": ("data", "tldw_chatbook_chatbooks.json", None),
        },
        "delete": (
            "db.prompts.primary",
            "db.agent_runs",
            "notes.file_notes",
            "tts.profile_store",
        ),
        # After the restore a writer adds a board to this created database and
        # exits without a checkpoint: the board is only in its -wal. The files
        # retire together, and the new safety copy must hold the board.
        "wal": ("kanban.local",),
        "refusal": None,
    },
    "library": {
        "groups": ("conversations", "library", "prompts", "tools", "workspaces"),
        "stores": (),
        "extra": (
            (
                "db.library_collections",
                "DB.Library_Collections_DB",
                "LibraryCollectionsDB",
            ),
            (
                "db.library_ingest_jobs",
                "DB.Library_Ingest_Jobs_DB",
                "LibraryIngestJobsDB",
            ),
        ),
        "files": {},
        "delete": ("db.media.primary",),
        "refusal": None,
    },
    "projects": {
        "groups": ("conversations", "evaluations", "research", "workspaces", "writing"),
        "stores": ("subscriptions",),
        "extra": (
            ("db.evals", "DB.Evals_DB", "EvalsDB"),
            (
                "research.local",
                "Research_Interop.local_research_service",
                "LocalResearchService",
            ),
            (
                "writing.local",
                "Writing_Interop.local_writing_service",
                "LocalWritingService",
            ),
        ),
        "files": {},
        "delete": (),
        "refusal": None,
    },
    # The MCP stores with their retained siblings, which keep their own IDs,
    # the RAG pipeline file beside the config, and an indexing database made by
    # RAGIndexingDB, as the app makes it (TASK-41718: capture refused its
    # embedding_cache table as unsupported_schema).
    "tools": {
        # Indexing records depend on the Media, ChaChaNotes and Prompts
        # databases, so Library and Prompts come too.
        "groups": (
            "conversations",
            "library",
            "prompts",
            "retrieval",
            "settings",
            "tools",
            "workspaces",
        ),
        "stores": (),
        # The Library group requires its collections database.
        "extra": (
            (
                "db.library_collections",
                "DB.Library_Collections_DB",
                "LibraryCollectionsDB",
            ),
            ("db.rag_indexing", "DB.RAG_Indexing_DB", "RAGIndexingDB"),
        ),
        "files": {
            "mcp.local": ("data", "local_mcp_store.json", '{"profiles": []}'),
            "mcp.targets": ("data", "mcp_server_targets.json", '{"targets": []}'),
            "mcp.context": ("data", "unified_mcp_context.json", "{}"),
            "mcp.permissions": ("data", "mcp_permissions.json", "{}"),
            "mcp.permissions.bak": ("data", "mcp_permissions.json.bak", "{}"),
            "mcp.history": ("data", "mcp_execution_log.jsonl", '{"event": "newer"}\n'),
            "mcp.history.1": (
                "data",
                "mcp_execution_log.jsonl.1",
                '{"event": "older"}\n',
            ),
            "config.history.dated": (
                "config",
                "config_backup_20260101_000000.toml",
                '[general]\nusers_name="older"\n',
            ),
            "rag.definitions": (
                "config",
                "rag_pipelines.toml",
                '[pipelines.saved]\ndescription = "saved"\n',
            ),
        },
        "delete": (),
        "refusal": None,
    },
    # A created file beside the config, after a second profile's config has
    # joined that folder: it belongs to both, so the rollback refuses it.
    "shared-config-folder": {
        "groups": ("conversations", "workspaces"),
        "stores": (),
        "extra": (),
        "files": {"notes.templates": ("config", "note_templates.json", "{}")},
        "delete": (),
        "second_profile": True,
        "refusal": "local_snapshot_created_config_folder_shared",
    },
}

# TASK-42000 (TASK-41719's cases): pet files sit where the installed pet owner
# discovers them ("pets": the profile's config folder; before TASK-42000 the
# old shared ~/.config/tldw_chatbook). "shared" files go into the target's old
# shared folder, which must never change; "target_files" are the target's own
# files, "retired" the ones the restore must retire and the rollback bring
# back; "pets_after_restore" is what the target's pet store lists once
# started after the restore (no copy from the old shared folder);
# "uncaptured" are source files a backup leaves out, so the restore never
# creates them (TASK-42114: the store's own rotating dated copies).
_PET = '{"bit": {"name": "Bit", "happiness": 70}}'
_PET_GROUPS = ("conversations", "settings", "workspaces")
_CASES.update(
    {
        "pets-missing": {
            "start": True,
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {
                "pets": ("pets", "tamagotchi_pets.json", _PET),
                "pets.dated": (
                    "pets",
                    "tamagotchi_pets.backup_20260101_000000.json",
                    "{}",
                ),
            },
            "uncaptured": ("pets.dated",),
            "delete": (),
            "extract": True,
            "refusal": None,
        },
        # Archives made before TASK-42114 hold the store's dated copies. A
        # restore never recreates them (the current pet file still restores),
        # and a target's own same-named copy is left as it is (TASK-42114
        # review).
        "pets-legacy-dated": {
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {
                "pets": ("pets", "tamagotchi_pets.json", _PET),
                "pets.dated": (
                    "pets",
                    "tamagotchi_pets.backup_20260101_000000.json",
                    '{"dated": {"name": "Dated"}}',
                ),
            },
            "legacy_dated_capture": True,
            "uncaptured": ("pets.dated",),
            "then": "later_rollback",
            "delete": (),
            "refusal": None,
        },
        "pets-legacy-dated-collision": {
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {
                "pets": ("pets", "tamagotchi_pets.json", _PET),
                "pets.dated": (
                    "pets",
                    "tamagotchi_pets.backup_20260101_000000.json",
                    '{"dated": {"name": "Dated"}}',
                ),
            },
            "target_files": {
                "own.dated": (
                    "pets",
                    "tamagotchi_pets.backup_20260101_000000.json",
                    '{"own": {"name": "Own dated"}}',
                ),
            },
            "legacy_dated_capture": True,
            "uncaptured": ("pets.dated",),
            "then": "later_rollback",
            "delete": (),
            "refusal": None,
        },
        "pets-present": {
            "start": True,
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {"pets": ("pets", "tamagotchi_pets.json", _PET)},
            "shared": {
                "tamagotchi_pets.backup_20250101_000000.json": '{"old": {"name": "Old"}}'
            },
            "delete": (),
            "refusal": None,
        },
        "pets-dated-only": {
            "start": True,
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {
                "pets.dated": (
                    "pets",
                    "tamagotchi_pets.backup_20260101_000000.json",
                    _PET,
                )
            },
            "target_files": {
                "own": ("pets", "tamagotchi_pets.json", '{"own": {"name": "Own"}}')
            },
            # A backup with only a dated copy holds no pets (TASK-42114), so
            # the Replace retires the target's pets as for "pets-none".
            "uncaptured": ("pets.dated",),
            "retired": ("own",),
            "delete": (),
            "refusal": None,
        },
        # A Replace from a backup with no pets retires the target's pets, and
        # the next start must not copy the old shared pets back (seed once).
        # The retired file's root stays bound and storage still works
        # (TASK-42022).
        "pets-none": {
            "start": True,
            "groups": _PET_GROUPS,
            "stores": (),
            "extra": (),
            "files": {},
            "target_files": {
                "own": ("pets", "tamagotchi_pets.json", '{"own": {"name": "Own"}}'),
                "seeded": ("pets", "tamagotchi_pets.seeded", "{}\n"),
            },
            "shared": {"tamagotchi_pets.json": '{"old": {"name": "Old"}}'},
            "retired": ("own",),
            "pets_after_restore": [],
            "delete": (),
            "refusal": None,
        },
        # An ungrouped backup (format 1 with no data_groups) Replaces the
        # whole profile; its pet rows are the rows older backups hold.
        "pets-ungrouped": {
            "start": True,
            "groups": None,
            "stores": (),
            "extra": (),
            "common_stores": ("subscriptions", "scheduled_tasks"),
            "common_extra": (
                ("db.evals", "DB.Evals_DB", "EvalsDB"),
                (
                    "db.library_collections",
                    "DB.Library_Collections_DB",
                    "LibraryCollectionsDB",
                ),
            ),
            "files": {"pets": ("pets", "tamagotchi_pets.json", _PET)},
            "shared": {
                "tamagotchi_pets.backup_20250101_000000.json": '{"old": {"name": "Old"}}'
            },
            "delete": (),
            "refusal": None,
        },
    }
)

# The row the [databases] writer adds after the restore, only in the WAL.
_BOARD_INSERT = (
    "INSERT INTO kanban_boards(uuid,name,created_at,updated_at) VALUES (?,?,?,?)"
)

_SCENARIO = r"""
import json,os,subprocess,sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap,credentials
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery.restore_plan import required_rollback_dependencies
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
credential_store=KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store=lambda:credential_store
name,role=sys.argv[1:3]
case=CASES[name]
groups=None if case['groups'] is None else tuple(case['groups'])
os.environ['APPDATA']=str(Path.home()/'AppData'/'Roaming')
backup_options={} if groups is None else {'data_groups':groups}
if LOCATION=='default':
    # The default config folder holds recovery-bootstrap, so it is never a
    # root, and each file beside config.toml is registered on its own.
    os.environ['TLDW_CONFIG_PATH']=str(Path.home()/'.config'/'tldw_cli'/'config.toml')
source=Path(os.environ['TLDW_CONFIG_PATH'])
if role!='crash':
    # The crash role runs inside the target's restored profile (TASK-42115).
    source.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    data=Path(os.environ['XDG_DATA_HOME'])/'fixture';data.mkdir(mode=0o700)
    source.write_text('[general]\nusers_name="default_user"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n')
    source.chmod(0o600)
admission_authority(bootstrap.default_bootstrap_root())
producer='''
import importlib,json,os,sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
case=json.loads(os.environ['CREATED_CASE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
if sys.argv[1]=='admit':
    # A fresh start: startup permission, then startup admission.
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
    permission=bootstrap.startup_permission(selector,bootstrap.default_bootstrap_root())
    with acquire_storage():pass
    print(json.dumps({'permission':list(permission)}))
    raise SystemExit(0)
if sys.argv[1]=='start':
    # TASK-42022: storage access after the restore, as a fresh app start
    # makes it: admission of the pet path, containment over the profile's
    # declared roots, and the pet store's own read.
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.storage_admission import _contains_capture_path,_contains_owned_path
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
    roots=next(row['roots'] for row in bootstrap._records(bootstrap.default_bootstrap_root())[1] if row['selector']==str(selector))
    pets=selector.parent/'tamagotchi_pets.json'
    # Before the store recreates a retired pet file.
    absent=[root for root in roots if not os.path.lexists(root)]
    # An absent root holds nothing else (TASK-42000); a deleted file root
    # scopes only its own path (TASK-42115).
    for root in absent:assert _contains_owned_path(Path(root),selector.parent/'other.json') is False,absent
    contained=_contains_capture_path(tuple(map(Path,roots)),pets)
    store=ConfigFileStorage()
    listed=sorted(store.list_pets())
    # TASK-42115: the pet also saves (its dated copy under the config).
    saved=store.save('saved',{'name':'Saved','happiness':60,'hunger':30,'energy':80,'health':90,'age':0})
    print(json.dumps({'pets':listed,'saved':saved,'after_save':sorted(ConfigFileStorage().list_pets()),'absent_roots':absent,'contained':contained}))
    raise SystemExit(0)
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
preview=preview_capture((selector,),options={'allow_partial':True})
_populate_required_dependencies(preview)
from tldw_chatbook.config import get_user_data_dir
root=get_user_data_dir()
paths={item.owner:item.path for item in preview.items if item.path is not None and item.logical_id.endswith(':'+item.owner)}
folders={'data':root,'config':selector.parent,'pets':paths['tamagotchi.config'].parent}
made={}
def write(folder,leaf,text):
    path=folders[folder]/leaf
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.write_text(text);path.chmod(0o600)
def build(stores,extra):
    from Tests.Backup_Recovery.test_operational_owners import STORES
    built={}
    for store_name in stores:
        module,symbol,_,owner,_=STORES[store_name]
        constructor=getattr(importlib.import_module('tldw_chatbook.'+module),symbol)
        store=constructor(db_path=paths[owner]) if store_name=='kanban' else constructor(paths[owner])
        if store_name in {'kanban','notifications','events','sync'}:store._ensure_schema()
        getattr(store,'close',lambda:None)()
        built[owner]=('data',paths[owner].relative_to(root).as_posix())
    for owner,module,symbol in extra:
        store=getattr(importlib.import_module('tldw_chatbook.'+module),symbol)(paths[owner])
        if hasattr(store,'_ensure_schema'):store._ensure_schema()
        for closer in ('close','close_connection'):
            if hasattr(store,closer):getattr(store,closer)();break
        built[owner]=('data',paths[owner].relative_to(root).as_posix())
    return built
# Stores both profiles hold ("common"): a whole-profile Replace needs them.
build(case.get('common_stores',()),case.get('common_extra',()))
if sys.argv[1]=='source':
    made.update(build(case['stores'],case['extra']))
    for label,(folder,leaf,text) in case['files'].items():
        if text is None:
            # A chatbook registry naming the archive written above, in the
            # layout capture rewrites it to.
            text=json.dumps({'records':[{'title':'saved','file_path':str(root/'chatbooks'/'saved.zip')}]},ensure_ascii=False,indent=2)
        write(folder,leaf,text)
        if label not in case.get('uncaptured',()):made[label]=(folder,leaf)
else:
    for owner in case['delete']:
        for suffix in ('','-wal','-shm','-journal'):
            Path(str(paths[owner])+suffix).unlink(missing_ok=True)
        made[owner]=('data',paths[owner].relative_to(root).as_posix())
    shared=Path(os.environ['APPDATA']) if os.name=='nt' else Path.home()/'.config'
    for leaf,text in case.get('shared',{}).items():
        path=shared/'tldw_chatbook'/leaf
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.write_text(text);path.chmod(0o600)
    for label,(folder,leaf,text) in case.get('target_files',{}).items():
        write(folder,leaf,text)
print(json.dumps({'root':str(root),'made':made,'folders':{k:str(v) for k,v in folders.items()}}))
'''

def produce(role):
    env=dict(os.environ,CREATED_CASE=json.dumps(case))
    result=subprocess.run([sys.executable,'-c',producer,role],capture_output=True,text=True,env=env,timeout=900 if sys.platform=='win32' else 240)
    assert result.returncode==0,result.stderr[-4000:]+result.stdout[-1000:]
    return json.loads(result.stdout.strip().splitlines()[-1])

def succeeded(service,operation):
    result=service.wait(operation)
    assert result['state']=='succeeded',dict(result)
    return result

ABORTS=[]
def reviewed(service,start,retry,expect='succeeded'):
    result=service.wait(start())
    if result['state']=='recovery_required' and result['review_issues']:
        issues=tuple(result['review_issues'])
        assert all(code.startswith('credential_') for code in issues),issues
        ABORTS.append(result['result']['journal_operation_id'])
        succeeded(service,service.start_recovery(result['result']['journal_operation_id'],action='abort'))
        result=service.wait(retry(issues))
    assert result['state']==expect,dict(result)
    return result

class RestoreOnly(Exception):
    # Ends a case before its later rollback (TASK-42115's chain).
    pass

def located(folders,made):
    return {label:Path(folders[folder])/leaf for label,(folder,leaf) in made.items()}

def snapshot(folder):
    return {p.name:p.read_bytes() for p in folder.iterdir()} if folder.exists() else None

def present(paths):
    return {label for label,path in paths.items() if path.exists()}

def board_only_in_wal(paths,stage):
    # The main file alone lacks the post-restore board: nothing has
    # checkpointed the -wal by this stage.
    import shutil,sqlite3
    from contextlib import closing
    for label in case.get('wal',()):
        main_only=Path.home()/(label+'-'+stage+'-main-only.db')
        shutil.copyfile(paths[label],main_only)
        with closing(sqlite3.connect(main_only)) as db:
            assert not db.execute("SELECT name FROM kanban_boards WHERE uuid='post-restore-board'").fetchall(),(label,stage)

shared=Path.home().parent.parent/'source'/'home'
shared_pets=(Path(os.environ['APPDATA']) if os.name=='nt' else Path.home()/'.config')/'tldw_chatbook'
service=RecoveryService(default_control_root() if LOCATION=='default' else Path.home()/'recovery-control')
def plan_for(inspection,issues=()):
    profile=service.summary(inspection)['profile_ids'][0]
    choices=dict(mode='replace',profile_bases={},target_configs={profile:source},external_destinations={},
        profile_names={},target=service.preview_backup((source,),options={}),data_groups=groups,
        acknowledged_credential_issues=issues)
    first=service.preview_restore(inspection,**choices)
    required=required_rollback_dependencies(first)
    return service.preview_restore(inspection,**choices,safety_scope=required) if required else first
def inspected():
    inspection=service.start_inspection(shared/'created.tldw-backup.zip',password=None)
    succeeded(service,inspection)
    return inspection
def crash_at_finalize():
    # The one crash point (TASK-42154, TASK-42115): this process exits 17 at
    # finalize_candidate, after publication, and a fresh process recovers it.
    from tldw_chatbook.Backup_Recovery import publication as crashed
    crashed.finalize_candidate=lambda *args,**kwargs:os._exit(17)
try:
    if role=='crash':
        # TASK-42115: the chain's next Replace, killed at that crash point.
        inspection=inspected()
        crash_at_finalize()
        reviewed(service,lambda:service.start_restore(inspection,plan_for(inspection),rollback_password=b'rollback-test-password'),
            lambda issues:service.start_restore(inspection,plan_for(inspection,issues),rollback_password=b'rollback-test-password'))
        raise AssertionError('the Replace never reached finalize_candidate')
    if role=='source':
        made=produce('source')['made']
        if case.get('legacy_dated_capture'):
            # Capture as before TASK-42114: the store's dated copies included.
            from dataclasses import replace as replaced
            from tldw_chatbook.Widgets.Tamagotchi import recovery as pet_recovery
            current=pet_recovery._ConfiguredPets.discover
            pet_recovery._ConfiguredPets.discover=lambda self,config:tuple(
                replaced(item,status='included') if item.owner=='tamagotchi.config' and item.status=='intentionally_excluded' else item
                for item in current(self,config))
        preview=service.preview_backup((source,),options=backup_options)
        assert preview.complete,[(row.owner,row.status,row.logical_id) for row in preview.items if row.status in {'missing_required','unsupported','unavailable'}]
        succeeded(service,service.start_backup((source,),preview.scope_digest,Path.home()/'created.tldw-backup.zip',options=backup_options,password=None))
        (Path.home()/'created.json').write_text(json.dumps(made))
    else:
        produced=produce('target')
        folders=produced['folders']
        made=json.loads((shared/'created.json').read_text())
        made.update(produced['made'])
        paths=located(folders,made)
        assert not present(paths),sorted(present(paths))
        own={label:(Path(folders[folder])/leaf,text) for label,(folder,leaf,text) in case.get('target_files',{}).items()}
        shared_before=snapshot(shared_pets)
        inspection=inspected()
        def plan(issues=()):
            return plan_for(inspection,issues)
        # Name any refusal behind backup_operation_failed: activation's, which
        # replacement_rolled_back hides, or the service's own (TASK-42113).
        refusals=[]
        from tldw_chatbook.Backup_Recovery import publication,recovery_service as service_module
        finalize,issue=publication.finalize_candidate,service_module.issue_code
        def observed_finalize(*args,**kwargs):
            try:return finalize(*args,**kwargs)
            except (OSError,ValueError,RuntimeError) as error:
                refusals.append(str(error));raise
        def observed_issue(error,**kwargs):
            refusals.append(str(error));return issue(error,**kwargs)
        publication.finalize_candidate,service_module.issue_code=observed_finalize,observed_issue
        def named(step):
            # Each step's failure carries the refusals seen so far.
            try:return step()
            except AssertionError as error:
                raise AssertionError(str(error)+' refusals='+repr(refusals)) from None
        restored=named(lambda:reviewed(service,lambda:service.start_restore(inspection,plan(),rollback_password=b'rollback-test-password'),
            lambda issues:service.start_restore(inspection,plan(issues),rollback_password=b'rollback-test-password')))
        operation=restored['result']['journal_operation_id']
        assert present(paths)==set(paths),sorted(set(paths)-present(paths))
        if case.get('admit_after_restore'):
            # The next start after the restore (TASK-42151).
            assert produce('admit')['permission']==[True,'startup_allowed']
        captured={label:value for label,value in case['files'].items() if label not in case.get('uncaptured',())}
        for label,(folder,leaf,text) in captured.items():
            # The pets restore with the source's bytes.
            if folder=='pets':assert paths[label].read_text()==text,label
        for label in case.get('uncaptured',()):
            folder,leaf,_=case['files'][label]
            kept=[text for own_folder,own_leaf,text in case.get('target_files',{}).values() if own_leaf==leaf]
            if kept:
                assert (Path(folders[folder])/leaf).read_text()==kept[0],'restore replaced the profile\'s own '+leaf
            else:
                assert not (Path(folders[folder])/leaf).exists(),'restore created '+label
        retired=[label for label in case.get('retired',()) if own[label][0].exists()]
        assert not retired,'restore kept '+repr(retired)
        assert snapshot(shared_pets)==shared_before,'restore changed the old shared pet folder'
        if case.get('then')=='chain':
            # TASK-42115: every operation after the Replace retired the pet
            # file, whose root stays bound and gone (at the default location a
            # deleted file root), each followed by a fresh start.
            assert produce('admit')['permission'][0] is True
            # A backup: capture naming, its session and publication roots.
            preview=service.preview_backup((source,),options=backup_options)
            assert preview.complete,[(row.owner,row.status) for row in preview.items if row.status in {'missing_required','unsupported','unavailable'}]
            named(lambda:succeeded(service,service.start_backup((source,),preview.scope_digest,Path.home()/'after.tldw-backup.zip',options=backup_options,password=None)))
            assert produce('admit')['permission'][0] is True
            # A second Replace killed at finalize_candidate, then rollback in
            # this fresh process: the recovery journal meets the deleted root.
            crashed=subprocess.run([sys.executable,'-c',Path(os.environ['CREATED_SCENARIO']).read_text(),name,'crash'],
                capture_output=True,text=True,env=dict(os.environ),timeout=900 if sys.platform=='win32' else 300)
            assert crashed.returncode==17,crashed.stderr[-4000:]+crashed.stdout[-1000:]
            pending,_=bootstrap._records(bootstrap.default_bootstrap_root())
            assert len(pending)==1,pending
            crash=pending[0]['operation_id']
            assert 'rollback' in service.status(crash)['actions'],service.status(crash)
            named(lambda:succeeded(service,service.start_recovery(crash,action='rollback',rollback_password=b'rollback-test-password')))
            assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
            assert produce('admit')['permission'][0] is True
            # The next Replace completes over the deleted root.
            second=named(lambda:reviewed(service,lambda:service.start_restore(inspection,plan(),rollback_password=b'rollback-test-password'),
                lambda issues:service.start_restore(inspection,plan(issues),rollback_password=b'rollback-test-password')))['result']['journal_operation_id']
            assert produce('admit')['permission'][0] is True
            # Its later rollback: stopped for review and aborted, then done.
            first=service.preview_rollback(second,old_password=b'rollback-test-password',target=service.preview_backup((source,),options={}))
            def later_second(issues=()):
                approved=first if not issues else service.preview_rollback(second,old_password=b'rollback-test-password',
                    target=service.preview_backup((source,),options={}),acknowledged_credential_issues=issues)
                return service.start_rollback(second,approved,old_password=b'rollback-test-password',new_password=b'second-safety-password')
            aborted=len(ABORTS)
            named(lambda:reviewed(service,later_second,later_second))
            assert len(ABORTS)==aborted+1,'the later rollback never stopped for review'
            assert produce('admit')['permission'][0] is True
            assert not any(own[label][0].exists() for label in case['retired']),'the retire did not stay in place'
        if case.get('start'):
            started=produce('start')
            if 'pets_after_restore' in case:
                assert started['pets']==case['pets_after_restore']
            # TASK-42115: the footer pet starts and saves.
            assert started['saved'] is True and 'saved' in started['after_save'],started
            # TASK-42022: the retired pet file keeps its root in the profile
            # binding (the registry keeps root history by design), and the
            # profile's storage still admits, contains and reads.
            assert [Path(root) for root in started['absent_roots']]==[own[label][0] for label in case.get('retired',())],started
            if case.get('retired'):assert started['contained'] is True,started
            assert snapshot(shared_pets)==shared_before,'pet start changed the old shared pet folder'
            assert produce('admit')['permission'][0] is True
        if case.get('then')=='chain':
            raise RestoreOnly
        if case.get('extract'):
            # Manual extraction of the backup's pet rows (TASK-42000 AC#3).
            from tldw_chatbook.Backup_Recovery import archive_reader
            document=archive_reader.verify_sealed(service.inspection(inspection))
            pets={row.logical_id:row for row in document.files if row.owner_id=='tamagotchi.config'}
            assert len(pets)==len(captured),sorted(pets)
            group_ids=tuple(group.group_id for group in document.dependency_groups if set(group.members)&set(pets))
            reviewed_extraction=service.wait(service.start_extraction_preview(inspection,group_ids=group_ids,destination=Path.home()/'extracted'))
            assert reviewed_extraction['state']=='succeeded',dict(reviewed_extraction)
            succeeded(service,service.start_extraction(inspection,reviewed_extraction['result']['plan']))
            report=json.loads((Path.home()/'extracted'/'inert-mapping.json').read_text())
            rows={row['archive_relative_path']:row for row in report['files'] if row['owner_id']=='tamagotchi.config'}
            assert set(rows)=={leaf for folder,leaf,_ in captured.values()},sorted(rows)
            for folder,leaf,text in captured.values():
                assert (Path.home()/'extracted'/rows[leaf]['output']).read_text()==text,leaf
        for label in case.get('wal',()):
            writer=("import os,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);c.execute('PRAGMA journal_mode=WAL');"
                "c.execute('PRAGMA wal_autocheckpoint=0');c.execute(sys.argv[2],('post-restore-board','Post-restore board','2026-11-18T00:00:00Z','2026-11-18T00:00:00Z'));c.commit();os._exit(0)")
            subprocess.run([sys.executable,'-c',writer,str(paths[label]),BOARD_INSERT],check=True,timeout=60)
            assert Path(str(paths[label])+'-wal').stat().st_size>0,label
            assert Path(str(paths[label])+'-shm').exists(),label
        board_only_in_wal(paths,'written')
        if case.get('second_profile'):
            from tldw_chatbook.Backup_Recovery.control_records import bind_profile
            other=source.parent/'other.toml'
            other.write_text('[general]\nusers_name="other"\n');other.chmod(0o600)
            control=bootstrap.default_bootstrap_root()
            own=next(row for row in bootstrap._records(control)[1] if row['selector']==str(source))
            bind_profile(control,other,tuple(own['namespaces']),control/'admission')
        if case['refusal']:
            try:
                preview_rollback(operation,control_root=service.control_root,old_password=b'rollback-test-password',
                    target=service.preview_backup((source,),options={}),cancel=Event())
            except ValueError as error:
                assert str(error)==case['refusal'],error
            else:
                raise AssertionError('later rollback accepted '+name)
            assert present(paths)==set(paths),'refused rollback removed a file'
        else:
            rollback=preview_rollback(operation,control_root=service.control_root,old_password=b'rollback-test-password',
                target=service.preview_backup((source,),options={}),cancel=Event())
            # Neither the inventory nor the rollback preview checkpointed it.
            board_only_in_wal(paths,'previewed')
            def later(issues=()):
                approved=rollback if not issues else service.preview_rollback(operation,old_password=b'rollback-test-password',
                    target=service.preview_backup((source,),options={}),acknowledged_credential_issues=issues)
                return service.start_rollback(operation,approved,old_password=b'rollback-test-password',new_password=b'second-safety-password')
            rolled=named(lambda:reviewed(service,later,later))
            kept=present(paths)
            assert not kept,'rollback kept created files: '+repr(sorted(kept))
            for label in case.get('retired',()):
                assert own[label][0].read_text()==own[label][1],'rollback did not bring back '+label
            for label,(path,text) in own.items():
                if label not in case.get('retired',()):
                    assert path.read_text()==text,'rollback changed the profile\'s own '+label
            assert snapshot(shared_pets)==shared_before,'rollback changed the old shared pet folder'
            sidecars=[str(path)+suffix for path in paths.values() for suffix in ('-wal','-shm') if Path(str(path)+suffix).exists()]
            assert not sidecars,sidecars
            if case.get('then')=='later_rollback':
                assert produce('admit')['permission'][0] is True
            if case.get('wal'):
                # The post-restore board reached the new safety copy before the
                # database retired (as test_later_created_destinations checks
                # for created edits).
                import sqlite3,zipfile
                from contextlib import closing
                from tldw_chatbook.Backup_Recovery import archive_reader
                from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
                from tldw_chatbook.Backup_Recovery.recovery_copies import list_recovery_copies
                saved=next(row for row in list_recovery_copies(service.control_root)
                    if row.operation_id==rolled['result']['journal_operation_id'])
                acquired=archive_reader.acquire(saved.path,Path.home()/'new-safety-readback',ArchiveLimits(),b'second-safety-password',Event())
                document=archive_reader.verify_sealed(acquired)
                for label in case['wal']:
                    payload=next(row for row in document.files if row.owner_id==label)
                    copy=Path.home()/(label+'-readback.db')
                    with zipfile.ZipFile(acquired.path) as packed:copy.write_bytes(packed.read(payload.payload))
                    with closing(sqlite3.connect(copy)) as db:
                        found=db.execute("SELECT name FROM kanban_boards WHERE uuid='post-restore-board'").fetchall()
                    assert found==[('Post-restore board',)],(label,found)
except RestoreOnly:
    pass
finally:
    service.close()
assert not network_guard.blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.timeout(4200 if sys.platform == "win32" else 1500)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_later_rollback_retires_the_files_a_restore_created(tmp_path, case):
    """TASK-41521: every file the case's restore created is gone after the
    later rollback, as it was before the restore."""
    script = (
        "CASES="
        + repr(_CASES)
        + "\nBOARD_INSERT="
        + repr(_BOARD_INSERT)
        + "\nLOCATION='separate'\n"
        + _SCENARIO
    )
    for name in ("source", "target"):
        (tmp_path / name).mkdir(mode=0o700)
    _run(
        tmp_path / "source",
        case,
        "source",
        script=script,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "target",
        case,
        "target",
        script=script,
        timeout=2500 if sys.platform == "win32" else 900,
    )


# TASK-42114: after a backup, the pet store rotates its dated copies (three
# kept); the app must still start and save. A backup used to register the
# dated copy as its own admission root, and once rotation deleted it every
# admission raised FileNotFoundError, so the app refused to start.
_ROTATION_CASE = {
    "groups": _PET_GROUPS,
    "stores": (),
    "extra": (),
    "files": {
        "pets": ("pets", "tamagotchi_pets.json", _PET),
        "pets.dated": ("pets", "tamagotchi_pets.backup_20260101_000000.json", _PET),
    },
    "delete": (),
    "refusal": None,
}

_ROTATION = r"""
SAVE = r'''
import json,sys,time
from Tests import network_guard
network_guard.install()
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
with acquire_storage():pass
store=ConfigFileStorage()
saved=[]
for index in range(int(sys.argv[1])):
    time.sleep(1.1)  # dated copies are named to the second
    saved.append(store.save('bit',{'name':'Bit','happiness':60+index,'hunger':30,'energy':80,'health':90,'age':index}))
print(json.dumps({'saved':saved,'pets':sorted(store.list_pets())}))
'''
def start(saves):
    result=subprocess.run([sys.executable,'-c',SAVE,str(saves)],capture_output=True,text=True,env=dict(os.environ),timeout=900 if sys.platform=='win32' else 240)
    assert result.returncode==0,result.stderr[-4000:]+result.stdout[-1000:]
    return json.loads(result.stdout.strip().splitlines()[-1])
backed_up=set(source.parent.glob('tamagotchi_pets.backup_*.json'))
assert backed_up
# Four saves rotate the backed-up copy away, then the app starts again.
assert start(4)=={'saved':[True]*4,'pets':['bit']}
assert not backed_up&set(source.parent.glob('tamagotchi_pets.backup_*.json'))
assert start(1)=={'saved':[True],'pets':['bit']}
# The store's own rotating copies are not registered as roots.
registry=bootstrap._registry(bootstrap.default_bootstrap_root())
roots={Path(root) for entry in registry.values() for root in entry['roots']}
assert not any(root.name.startswith('tamagotchi_pets.backup_') for root in roots),sorted(map(str,roots))
print('rotated and restarted')
"""


@pytest.mark.timeout(1500 if sys.platform == "win32" else 600)
@pytest.mark.parametrize("location", ["separate", "default"])
def test_a_backup_then_pet_rotation_keeps_the_app_starting(tmp_path, location):
    """TASK-42114: a backup, then the pet store's dated-copy rotation, then the
    next start and save, at the default config location and on a separate
    config folder."""
    script = (
        "CASES="
        + repr({"rotation": _ROTATION_CASE})
        + "\nBOARD_INSERT="
        + repr(_BOARD_INSERT)
        + "\nLOCATION="
        + repr(location)
        + "\n"
        + _SCENARIO
        + _ROTATION
    )
    (tmp_path / "source").mkdir(mode=0o700)
    _run(
        tmp_path / "source",
        "rotation",
        "source",
        script=script,
        timeout=1500 if sys.platform == "win32" else 300,
    )


# TASK-42115: the five pet cases the separate-folder test runs, at the default
# config location, where the config folder holds recovery-bootstrap and is
# never a root: the Replace, the pet's start and save, and the later rollback.
_DEFAULT_PET_CASES = (
    "pets-dated-only",
    "pets-missing",
    "pets-none",
    "pets-present",
    "pets-ungrouped",
)


def _scenario(tmp_path, case, spec, location):
    script = (
        "CASES="
        + repr({case: spec})
        + "\nBOARD_INSERT="
        + repr(_BOARD_INSERT)
        + "\nLOCATION="
        + repr(location)
        + "\n"
        + _SCENARIO
    )
    for name in ("source", "target"):
        (tmp_path / name).mkdir(mode=0o700)
    return script


@pytest.mark.timeout(2500 if sys.platform == "win32" else 900)
@pytest.mark.parametrize("case", _DEFAULT_PET_CASES)
def test_a_default_location_pet_replace_completes_and_rolls_back(tmp_path, case):
    """TASK-42115: at the default config location a pet Replace completes with
    the same outcome as on a separate config folder, the pet then starts and
    saves, and the later rollback retires what the Replace created and brings
    back what it retired."""
    script = _scenario(tmp_path, case, _CASES[case], "default")
    _run(
        tmp_path / "source",
        case,
        "source",
        script=script,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "target",
        case,
        "target",
        script=script,
        timeout=2500 if sys.platform == "win32" else 900,
    )


# TASK-42154: a Replace killed at finalize_candidate, after publication has
# moved the candidate's files into place, is recovered in a fresh process.
# finish used to re-prove the archive from the stage (TASK-40135's key
# install), so it raised FileNotFoundError and startup stayed refused.
_CRASH_CASE = {
    "groups": _PET_GROUPS,
    "stores": (),
    "extra": (),
    "files": {},
    "delete": (),
    "refusal": None,
}
_CRASH_POINT = "        restored=named(lambda:reviewed(service,"
# A missing anchor would leave the scenario uncrashed; fail at import instead.
assert _SCENARIO.count(_CRASH_POINT) == 1
# The scenario's own crash point, shared with the chain's crash role
# (TASK-42115), armed just before the target's restore.
_CRASH_SCENARIO = _SCENARIO.replace(
    _CRASH_POINT, "        crash_at_finalize()\n" + _CRASH_POINT
)

# Each child is a fresh process: the crashed Replace, each recovery step, and
# every start (startup permission, then startup admission).
_CRASH_RECOVERY = r"""
RECOVER = r'''
import json,os,sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
credential_store=KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store=lambda:credential_store
step,location=sys.argv[1:3]
selector=Path(os.environ['TLDW_CONFIG_PATH'])
service=RecoveryService(default_control_root() if location=='default' else Path.home()/'recovery-control')
try:
    if step=='later_rollback':
        # The finished restore can still be rolled back (after credential
        # review, as the scenario's reviewed() does).
        operation=json.loads((Path.home()/'recovered.json').read_text())
        def later(issues=()):
            approved=service.preview_rollback(operation,old_password=b'rollback-test-password',
                target=service.preview_backup((selector,),options={}),acknowledged_credential_issues=issues)
            return service.wait(service.start_rollback(operation,approved,old_password=b'rollback-test-password',new_password=b'second-safety-password'))
        result=later()
        if result['state']=='recovery_required' and result['review_issues']:
            issues=tuple(result['review_issues'])
            assert all(code.startswith('credential_') for code in issues),issues
            aborted=service.wait(service.start_recovery(result['result']['journal_operation_id'],action='abort'))
            assert aborted['state']=='succeeded',dict(aborted)
            result=later(issues)
    else:
        (pending,)=service.pending_operations()
        operation=pending['operation_id']
        journal=_journal(service.control_root,operation)
        with journal._locked(exclusive=False) as parent:events=[row.event for row in journal._records(parent)]
        # Killed after publication, before the commit.
        assert 'artifact_published' in events and 'committed' not in events,events
        assert service.status(operation)['actions']==('finish','rollback'),dict(service.status(operation))
        result=service.wait(service.start_recovery(operation,action=step,rollback_password=b'rollback-test-password'))
        (Path.home()/'recovered.json').write_text(json.dumps(operation))
    assert result['state']=='succeeded',dict(result)
finally:
    service.close()
print(json.dumps(result['phase']))
'''
ADMIT = r'''
import json,os,sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
permission=bootstrap.startup_permission(Path(os.environ['TLDW_CONFIG_PATH']),bootstrap.default_bootstrap_root())
if permission[0]:
    with acquire_storage():pass
print(json.dumps(list(permission)))
'''
import json,os,subprocess,sys
from pathlib import Path
name,action=sys.argv[1:3]
env=dict(os.environ)
if LOCATION=='default':
    env['TLDW_CONFIG_PATH']=str(Path.home()/'.config'/'tldw_cli'/'config.toml')
def child(code,*args,expect=0):
    result=subprocess.run([sys.executable,'-X','utf8','-c',code,*args],env=env,capture_output=True,text=True,timeout=900 if sys.platform=='win32' else 300)
    assert result.returncode==expect,(result.returncode,result.stdout[-3000:],result.stderr[-6000:])
    return json.loads(result.stdout.strip().splitlines()[-1]) if expect==0 else None
child(SCENARIO,name,'target',expect=17)
assert child(ADMIT)==[False,'recovery_pending']
phases={'finish':'committed','rollback':'rolled_back','later_rollback':'restoration_validated'}
for step in (action,'later_rollback') if action=='finish' else (action,):
    phase=child(RECOVER,step,LOCATION)
    assert phase==phases[step],(step,phase)
    assert child(ADMIT)==[True,'startup_allowed'],step
print('retired and reopened')
"""


@pytest.mark.timeout(4200 if sys.platform == "win32" else 1500)
@pytest.mark.parametrize("location", ["separate", "default"])
def test_a_replace_killed_after_publication_recovers_by_finish_or_rollback(
    tmp_path, location
):
    """TASK-42154: a Replace killed at finalize_candidate, after publication,
    recovers in a fresh process by finish (a later rollback still undoes it)
    and by rollback, at the default config location and on a separate config
    folder. The app starts after each."""
    scenario = (
        "CASES="
        + repr({"control": _CRASH_CASE})
        + "\nBOARD_INSERT="
        + repr(_BOARD_INSERT)
        + "\nLOCATION="
        + repr(location)
        + "\n"
    )
    (tmp_path / "source").mkdir(mode=0o700)
    _run(
        tmp_path / "source",
        "control",
        "source",
        script=scenario + _SCENARIO,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    driver = (
        scenario
        + "SCENARIO="
        + repr(scenario + _CRASH_SCENARIO)
        + "\n"
        + _CRASH_RECOVERY
    )
    for action in ("finish", "rollback"):
        (tmp_path / action).mkdir(mode=0o700)
        _run(
            tmp_path / action,
            "control",
            action,
            script=driver,
            timeout=2500 if sys.platform == "win32" else 900,
        )


# TASK-42151: at the default config location the config folder holds the
# fixed recovery controls. A Replace whose groups leave out Settings maps
# nothing into that folder, and its first binding refused the folder
# (replacement_config_container_unverified) because the controls were only
# recognized beside a synthetic restore destination.
@pytest.mark.timeout(4200 if sys.platform == "win32" else 1500)
@pytest.mark.parametrize("case", ["library", "projects"])
def test_a_default_location_replace_without_settings_completes(tmp_path, case):
    """TASK-42151: the library and projects cases (no Settings group) at the
    default config location: the Replace completes, the later rollback
    retires what it created, and the app starts afterwards."""
    script = (
        "CASES="
        + repr(
            {case: dict(_CASES[case], then="later_rollback", admit_after_restore=True)}
        )
        + "\nBOARD_INSERT="
        + repr(_BOARD_INSERT)
        + "\nLOCATION='default'\n"
        + _SCENARIO
    )
    for name in ("source", "target"):
        (tmp_path / name).mkdir(mode=0o700)
    _run(
        tmp_path / "source",
        case,
        "source",
        script=script,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "target",
        case,
        "target",
        script=script,
        timeout=2500 if sys.platform == "win32" else 900,
    )


@pytest.mark.timeout(4200 if sys.platform == "win32" else 1500)
@pytest.mark.parametrize("location", ["default", "separate"])
def test_the_operations_after_a_retired_pet_root(tmp_path, monkeypatch, location):
    """TASK-42115 AC#6: after a Replace retires the pet file, its root stays
    bound and gone (at the default location a deleted file root, on a
    separate folder an alias its folder root covers). A backup, a second
    Replace killed at finalize_candidate and then rolled back, the next
    Replace, and that Replace's later rollback (stopped for review, aborted,
    then done) all complete, each followed by a fresh start; then the pet
    starts and saves."""
    spec = dict(_CASES["pets-none"], then="chain")
    script = _scenario(tmp_path, "pets-none", spec, location)
    scenario = tmp_path / "scenario.py"
    scenario.write_text(script)
    monkeypatch.setenv("CREATED_SCENARIO", str(scenario))
    _run(
        tmp_path / "source",
        "pets-none",
        "source",
        script=script,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "target",
        "pets-none",
        "target",
        script=script,
        timeout=2500 if sys.platform == "win32" else 900,
    )
