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
# databases (module, class), databases built from their owner's installed
# schema ("schema_dbs"), files {label: (folder, leaf, text)} where folder is
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
        # After the restore a writer leaves this created database's -wal and
        # -shm files behind; they retire with it.
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
    # the RAG pipeline file beside the config, and an indexing database built
    # from the owner's installed schema (the current constructor adds a table
    # that schema refuses, so capture would stop at unsupported_schema).
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
        ),
        "schema_dbs": ("db.rag_indexing",),
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
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.restore_plan import required_rollback_dependencies
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
credential_store=KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store=lambda:credential_store
name,role=sys.argv[1:3]
case=CASES[name]
groups=tuple(case['groups'])
source=Path(os.environ['TLDW_CONFIG_PATH'])
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
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
preview=preview_capture((selector,),options={'allow_partial':True})
_populate_required_dependencies(preview)
from tldw_chatbook.config import get_user_data_dir
root=get_user_data_dir()
paths={item.owner:item.path for item in preview.items if item.path is not None and item.logical_id.endswith(':'+item.owner)}
folders={'data':root,'config':selector.parent}
made={}
if sys.argv[1]=='source':
    from Tests.Backup_Recovery.test_operational_owners import STORES
    for store_name in case['stores']:
        module,symbol,_,owner,_=STORES[store_name]
        constructor=getattr(importlib.import_module('tldw_chatbook.'+module),symbol)
        store=constructor(db_path=paths[owner]) if store_name=='kanban' else constructor(paths[owner])
        if store_name in {'kanban','notifications','events','sync'}:store._ensure_schema()
        getattr(store,'close',lambda:None)()
        made[owner]=('data',paths[owner].relative_to(root).as_posix())
    for owner,module,symbol in case['extra']:
        store=getattr(importlib.import_module('tldw_chatbook.'+module),symbol)(paths[owner])
        if hasattr(store,'_ensure_schema'):store._ensure_schema()
        for closer in ('close','close_connection'):
            if hasattr(store,closer):getattr(store,closer)();break
        made[owner]=('data',paths[owner].relative_to(root).as_posix())
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    adapters={adapter.owner_id:adapter for adapter in install_adapters()}
    for owner in case.get('schema_dbs',()):
        import sqlite3
        from contextlib import closing
        statements=adapters[owner].schema_policy().schema_sql[0][1]
        with closing(sqlite3.connect(paths[owner])) as db:
            for sql in sorted(statements,key=lambda sql:not sql.startswith('CREATE TABLE')):db.execute(sql)
            db.commit()
        paths[owner].chmod(0o600)
        made[owner]=('data',paths[owner].relative_to(root).as_posix())
    for label,(folder,leaf,text) in case['files'].items():
        path=folders[folder]/leaf
        if text is None:
            # A chatbook registry naming the archive written above, in the
            # layout capture rewrites it to.
            text=json.dumps({'records':[{'title':'saved','file_path':str(root/'chatbooks'/'saved.zip')}]},ensure_ascii=False,indent=2)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);path.chmod(0o600)
        made[label]=(folder,leaf)
else:
    for owner in case['delete']:
        for suffix in ('','-wal','-shm','-journal'):
            Path(str(paths[owner])+suffix).unlink(missing_ok=True)
        made[owner]=('data',paths[owner].relative_to(root).as_posix())
print(json.dumps({'root':str(root),'made':made}))
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

def reviewed(service,start,retry):
    result=service.wait(start())
    if result['state']=='recovery_required' and result['review_issues']:
        issues=tuple(result['review_issues'])
        assert all(code.startswith('credential_') for code in issues),issues
        succeeded(service,service.start_recovery(result['result']['journal_operation_id'],action='abort'))
        result=service.wait(retry(issues))
    assert result['state']=='succeeded',dict(result)
    return result

def located(root,made):
    folders={'data':root,'config':source.parent}
    return {label:folders[folder]/leaf for label,(folder,leaf) in made.items()}

def present(paths):
    return {label for label,path in paths.items() if path.exists()}

shared=Path.home().parent.parent/'source'/'home'
service=RecoveryService(Path.home()/'recovery-control')
try:
    if role=='source':
        made=produce('source')['made']
        preview=service.preview_backup((source,),options={'data_groups':groups})
        assert preview.complete,[(row.owner,row.status,row.logical_id) for row in preview.items if row.status in {'missing_required','unsupported','unavailable'}]
        succeeded(service,service.start_backup((source,),preview.scope_digest,Path.home()/'created.tldw-backup.zip',options={'data_groups':groups},password=None))
        (Path.home()/'created.json').write_text(json.dumps(made))
    else:
        produced=produce('target')
        root=Path(produced['root'])
        made=json.loads((shared/'created.json').read_text())
        made.update(produced['made'])
        paths=located(root,made)
        assert not present(paths),sorted(present(paths))
        inspection=service.start_inspection(shared/'created.tldw-backup.zip',password=None)
        succeeded(service,inspection)
        profile=service.summary(inspection)['profile_ids'][0]
        def plan(issues=()):
            choices=dict(mode='replace',profile_bases={},target_configs={profile:source},external_destinations={},
                profile_names={},target=service.preview_backup((source,),options={}),data_groups=groups,
                acknowledged_credential_issues=issues)
            first=service.preview_restore(inspection,**choices)
            required=required_rollback_dependencies(first)
            return service.preview_restore(inspection,**choices,safety_scope=required) if required else first
        restored=reviewed(service,lambda:service.start_restore(inspection,plan(),rollback_password=b'rollback-test-password'),
            lambda issues:service.start_restore(inspection,plan(issues),rollback_password=b'rollback-test-password'))
        operation=restored['result']['journal_operation_id']
        assert present(paths)==set(paths),sorted(set(paths)-present(paths))
        for label in case.get('wal',()):
            writer="import os,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA wal_autocheckpoint=0');c.execute('SELECT count(*) FROM sqlite_master').fetchall();os._exit(0)"
            subprocess.run([sys.executable,'-c',writer,str(paths[label])],check=True,timeout=60)
            assert all(Path(str(paths[label])+suffix).exists() for suffix in ('-wal','-shm')),label
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
            def later(issues=()):
                approved=rollback if not issues else service.preview_rollback(operation,old_password=b'rollback-test-password',
                    target=service.preview_backup((source,),options={}),acknowledged_credential_issues=issues)
                return service.start_rollback(operation,approved,old_password=b'rollback-test-password',new_password=b'second-safety-password')
            reviewed(service,later,later)
            kept=present(paths)
            assert not kept,'rollback kept created files: '+repr(sorted(kept))
            sidecars=[str(path)+suffix for path in paths.values() for suffix in ('-wal','-shm') if Path(str(path)+suffix).exists()]
            assert not sidecars,sidecars
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
    script = "CASES=" + repr(_CASES) + "\n" + _SCENARIO
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
