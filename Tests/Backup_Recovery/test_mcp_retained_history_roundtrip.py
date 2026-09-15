"""Retain native MCP target, context, and metadata-only history semantics."""

import hashlib
import json
import os
from pathlib import Path

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _isolated_environment,
    _run_profile_child,
)

_PRIVATE = r"""
import asyncio,hashlib,json,os,stat,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['MCP_RETAINED_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(path):
 info=path.stat(follow_symlinks=False)
 return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns]
def snapshot(paths):return {str(path):{'sha256':digest(path),'identity':identity(path)} for path in paths}
def unchanged(receipt):
 for raw,evidence in receipt.items():
  path=Path(raw);assert digest(path)==evidence['sha256'] and identity(path)==evidence['identity']
"""


_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.MCP.execution_log import build_record
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget,UnifiedMCPContext
async def main():
 app=TldwCli()
 try:
  targets=app.unified_mcp_target_store;context_store=app.unified_mcp_context_store
  existing=targets.list_targets();assert existing and any(target.is_default for target in existing)
  defaults=[target.to_dict() for target in existing]
  retained=ConfiguredServerTarget(server_id='retained-loopback',label='Retained inert loopback',base_url='http://127.0.0.1:9/mcp',auth_mode='api_key',auth_reference=None,is_default=False,last_known_server_label='Historical fixture',last_known_reachability='unknown',last_known_auth_state='unknown')
  assert retained.auth_mode=='api_key' and retained.auth_reference is None
  assert retained.authority_scope_id
  targets.save_targets([*existing,retained])
  assert [row.to_dict() for row in targets.list_targets()]==[*defaults,retained.to_dict()]
  assert targets.get_target(retained.server_id)==retained
  context=UnifiedMCPContext(selected_source='local',selected_active_server_id=retained.server_id,selected_section='inventory')
  context_store.save(context)
  assert context_store.load().selected_active_server_id==retained.server_id
  history=app.unified_mcp_service.execution_log;assert history is not None
  assert history.path==app.local_mcp_store.path.parent/'mcp_execution_log.jsonl'
  history.max_records_per_file=1
  first=build_record(server_key='local:retained-history',tool_name='historical-denied',initiator='test',decision='denied',ok=False,status='policy_denied',duration_ms=7,error_category='permission',arguments={'query':'SECRET-FIRST','unknown':'SECRET-UNKNOWN'},registered_argument_names={'query'},result={'body':'SECRET-RESULT'})
  second=build_record(server_key='local:retained-history',tool_name='historical-unavailable',initiator='test',decision='denied',ok=False,status='unavailable',duration_ms=11,error_category='inactive',arguments={'path':'SECRET-SECOND'},registered_argument_names={'path'},result=['SECRET-RESULT-2'])
  history.append(first);history.append(second)
  rows=history.read_recent();assert [row['tool_name'] for row in rows]==['historical-unavailable','historical-denied']
  assert rows[0]['argument_names']==['path'] and rows[0]['result_type']=='list' and rows[0]['result_size']==1
  assert rows[1]['argument_names']==['query'] and rows[1]['unknown_argument_count']==1 and rows[1]['result_type']=='dict'
  current=history.path;rotated=current.with_name(current.name+'.1')
  forbidden=('SECRET-FIRST','SECRET-UNKNOWN','SECRET-SECOND','SECRET-RESULT')
  assert all(token not in (current.read_text()+rotated.read_text()) for token in forbidden)
  sources=snapshot((selector,targets.path,context_store.path,current,rotated))
  seed={'selector':str(selector),'user_root':str(targets.path.parent),'targets_path':str(targets.path),'context_path':str(context_store.path),'history_path':str(current),'defaults':defaults,'target':retained.to_dict(),'context':context.to_dict(),'rows':rows,'sources':sources}
  (fixture/'seed.json').write_text(json.dumps(seed,indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_CAPTURE = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.runtime_policy.server_credentials import RECOVERY_SETUP_REQUIRED
async def main():
 seed=json.loads((fixture/'seed.json').read_text());unchanged(seed['sources'])
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  unchanged(seed['sources'])
  target=app.unified_mcp_target_store.get_target(seed['target']['server_id']);assert target and target.to_dict()==seed['target']
  assert [row['tool_name'] for row in app.unified_mcp_service.execution_log.read_recent()]==['historical-unavailable','historical-denied']
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues},indent=2));assert preview.complete,preview.issues
  destination=fixture/'mcp-retained.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent';source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id'] in {'mcp.targets','mcp.context','mcp.history'}]
  assert [row['owner_id'] for row in rows].count('mcp.targets')==1
  assert [row['owner_id'] for row in rows].count('mcp.context')==1
  history=[row for row in rows if row['owner_id']=='mcp.history'];assert len(history)==2
  assert {source[row['logical_id']].path.name for row in history}=={'mcp_execution_log.jsonl','mcp_execution_log.jsonl.1'}
  receipts={}
  for row in rows:
   payload=captured.root/row['payload'];assert digest(payload)==row['sha256'];receipts[row['logical_id']]={'owner_id':row['owner_id'],'relative_path':row['relative_path'],'sha256':row['sha256'],'size':row['size']}
  target_row=next(row for row in rows if row['owner_id']=='mcp.targets');target_payload=json.loads((captured.root/target_row['payload']).read_text())
  restored_target=next(item for item in target_payload['targets'] if item['server_id']==seed['target']['server_id'])
  expected=dict(seed['target']);expected['auth_reference']=RECOVERY_SETUP_REQUIRED;assert restored_target==expected
  context_row=next(row for row in rows if row['owner_id']=='mcp.context');context_payload=json.loads((captured.root/context_row['payload']).read_text())
  assert {key:context_payload[key] for key in seed['context']}==seed['context']
  for row in history:
   source_path=source[row['logical_id']].path;assert (captured.root/row['payload']).read_bytes()==source_path.read_bytes()
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  selected=fixture/'restore-destinations';mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if original==Path(seed['user_root']) or Path(seed['user_root']) in original.parents:target=selected/'data'/'restored-mcp-retained'/original.relative_to(Path(seed['user_root']))
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'capture.json').write_text(json.dumps({'archive':str(destination),'archive_sha256':written.digest,'manifest_sha256':hashlib.sha256(captured.manifest_bytes).hexdigest(),'receipts':receipts,'mapping':mapping,'source_profile':profile},indent=2))
  unchanged(seed['sources']);assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_RESTORE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
seed=json.loads((fixture/'seed.json').read_text());receipt=json.loads((fixture/'capture.json').read_text());unchanged(seed['sources'])
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event());doc=verify_sealed(archive)
assert doc.consistency=='coherent' and archive.digest==receipt['archive_sha256']
with zipfile.ZipFile(archive.path) as zipped:
 assert hashlib.sha256(zipped.read('manifest.json')).hexdigest()==receipt['manifest_sha256']
 for row in doc.files:
  payload=zipped.read(row.payload);assert len(payload)==row.size and hashlib.sha256(payload).hexdigest()==row.sha256
 for logical_id,evidence in receipt['receipts'].items():
  row=next(row for row in doc.files if row.logical_id==logical_id);assert row.owner_id==evidence['owner_id'] and row.sha256==evidence['sha256'] and row.size==evidence['size']
destinations={key:Path(path) for key,path in receipt['mapping'].items()}
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={receipt['source_profile']:'restored-mcp-retained'})
profile=restore_isolated(archive,plan,fixture/'control',threading.Event());requirements=profile_requirements(profile,fixture/'control')
mcp_owners={evidence['owner_id'] for evidence in receipt['receipts'].values()}
assert requirements['requirements_checked'] and requirements['needs_setup'] and mcp_owners<=set(requirements['pending_owners'])
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements,'mcp_owners':sorted(mcp_owners)},indent=2))
unchanged(seed['sources']);assert not blocked_attempts(),blocked_attempts()
"""
)


_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'restored.json').read_text());select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.runtime_policy.server_credentials import RECOVERY_SETUP_REQUIRED
async def main():
 seed=json.loads((fixture/'seed.json').read_text());unchanged(seed['sources'])
 root=get_user_data_dir();assert root!=Path(seed['user_root'])
 targets_path=root/'mcp_server_targets.json';context_path=root/'unified_mcp_context.json';history_path=root/'mcp_execution_log.jsonl'
 before=snapshot((targets_path,context_path,history_path,history_path.with_name(history_path.name+'.1')))
 app=TldwCli()
 try:
  targets=app.unified_mcp_target_store;target=targets.get_target(seed['target']['server_id']);assert target is not None
  expected=dict(seed['target']);expected['auth_reference']=RECOVERY_SETUP_REQUIRED;assert target.to_dict()==expected
  listed=targets.list_targets();assert not target.is_default
  assert {item.server_id for item in listed}=={row['server_id'] for row in [*seed['defaults'],seed['target']]}
  expected_defaults=[]
  for row in seed['defaults']:
   portable=dict(row);portable['auth_reference']=RECOVERY_SETUP_REQUIRED;expected_defaults.append(portable)
  assert [item.to_dict() for item in listed if item.server_id!=target.server_id]==expected_defaults
  runtime=app.unified_mcp_context_store.load();assert runtime==UnifiedMCPContext() and runtime.selected_source=='local'
  retained=UnifiedMCPContext.from_dict(json.loads(context_path.read_text()));assert retained.selected_active_server_id==target.server_id and retained.selected_source=='local'
  history=MCPExecutionLog(history_path);rows=history.read_recent();assert rows==seed['rows']
  assert snapshot((targets_path,context_path,history_path,history_path.with_name(history_path.name+'.1')))==before
  requirements=profile_requirements(restored['profile'],fixture/'control');assert json.loads(json.dumps(requirements))==restored['requirements'] and set(restored['mcp_owners'])<=set(requirements['pending_owners'])
  unchanged(seed['sources']);assert not blocked_attempts(),blocked_attempts()
  (fixture/'readback.json').write_text(json.dumps({'archive_sha256':json.loads((fixture/'capture.json').read_text())['archive_sha256'],'manifest_sha256':json.loads((fixture/'capture.json').read_text())['manifest_sha256'],'source_preserved':True,'restored_preserved':True,'target_server_id':target.server_id,'authority_scope_id':target.authority_scope_id,'runtime_context':runtime.to_dict(),'retained_context':retained.to_dict(),'history_tools':[row['tool_name'] for row in rows],'history_paths':[str(history_path),str(history_path.with_name(history_path.name+'.1'))],'pending_mcp_owners':sorted(set(restored['mcp_owners'])&set(requirements['pending_owners'])),'blocked_network_attempts':len(blocked_attempts())},indent=2))
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def test_native_mcp_retained_history_target_and_context_roundtrip(tmp_path):
    root = tmp_path.resolve()
    for name in ("home", "xdg-config", "xdg-data", "cache", "tmp", "profile"):
        (root / name).mkdir(mode=0o700)
    profile = root / "profile"
    (profile / "custom").mkdir(mode=0o700)
    (profile / "data").mkdir(mode=0o700)
    selector = profile / "config.toml"
    selector.write_text(
        '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
        f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
        + "".join(
            f"{key}={json.dumps(str(profile / 'custom' / leaf))}\n"
            for key, leaf in (
                ("chachanotes_db_path", "notes.db"),
                ("media_db_path", "media.db"),
                ("research_db_path", "research.db"),
                ("prompts_db_path", "prompts.db"),
            )
        )
    )
    selector.chmod(0o600)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
        if key in os.environ
    }
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "xdg-config"),
        XDG_DATA_HOME=str(root / "xdg-data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        MCP_RETAINED_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    _run_profile_child(root, "seed", _SEED, environment)
    _run_profile_child(root, "capture", _CAPTURE, environment)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "restore", _RESTORE, restored_environment)
    _run_profile_child(root, "read", _READ, restored_environment)
    evidence = json.loads((root / "readback.json").read_text())
    assert evidence["source_preserved"] and evidence["restored_preserved"]
    assert evidence["blocked_network_attempts"] == 0
    seed = json.loads((root / "seed.json").read_text())
    captured = json.loads((root / "capture.json").read_text())
    journals = sorted(str(path) for path in (root / "control").glob("operation-*"))
    assert len(journals) == 1
    print(
        "MCP_RETAINED_EVIDENCE",
        json.dumps(
            {
                "fixture_root": str(root),
                "test_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "source_sha256": {
                    Path(path).name: receipt["sha256"]
                    for path, receipt in seed["sources"].items()
                },
                "archive_sha256": evidence["archive_sha256"],
                "manifest_sha256": evidence["manifest_sha256"],
                "payload_receipts": captured["receipts"],
                "authority_scope_id": evidence["authority_scope_id"],
                "history_tools": evidence["history_tools"],
                "source_history_path": seed["history_path"],
                "restored_history_paths": evidence["history_paths"],
                "pending_mcp_owners": evidence["pending_mcp_owners"],
                "journal_paths": journals,
                "blocked_network_attempts": evidence["blocked_network_attempts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
