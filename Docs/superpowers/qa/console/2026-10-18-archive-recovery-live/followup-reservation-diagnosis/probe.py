"""Actual source20f factory/projector diagnosis against private SQLite backup."""
import ast,hashlib,json,os,shutil,socket,sqlite3,subprocess,sys,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).parent.resolve()
LIVE=Path('/private/tmp/archive-recovery-live-20261018/direct-bare-persona')
WT=Path('/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
SOURCE=LIVE/'data/task7_live/tldw_chatbook_ChaChaNotes.db'
before=json.loads((ROOT/'inputs-before.json').read_text())
assert all(hashlib.sha256((LIVE/k).read_bytes()).hexdigest()==v for k,v in before.items())
COPY=ROOT/'diagnostic-copy.sqlite'
src=sqlite3.connect(SOURCE.as_uri()+'?mode=ro',uri=True)
dst=sqlite3.connect(COPY)
src.backup(dst);dst.close();src.close()
shutil.copyfile(LIVE/'config.toml',ROOT/'config-read-copy.toml')
os.environ['TLDW_CONFIG_PATH']=str(ROOT/'config-read-copy.toml')
os.environ['TMPDIR']=str(ROOT)
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
network=[]
def deny(*a,**kw):
 network.append('blocked');raise RuntimeError('offline-only probe')
socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny;socket.getaddrinfo=deny

def audit(event,args):
 if event=='sqlite3.connect':
  path=Path(str(args[0])).resolve()
  if not path.is_relative_to(ROOT):raise RuntimeError('DB access restricted to private backup')
 if event=='open':
  path,mode,flags=args
  if (any(c in (mode or '') for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) and isinstance(path,(str,bytes,os.PathLike)):
   if Path(os.fsdecode(path)).resolve()!=Path('/dev/null') and not Path(os.fsdecode(path)).resolve().is_relative_to(ROOT):raise RuntimeError('external write forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(WT))
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.console_prepared_request import build_console_request,prepare_provider_request,resolve_request_capacity
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute,ConsoleTraceCaptureMode,ProviderArtifactTraceProvenance,SavedRevisionTraceProvenance,TraceProvenanceSource,request_route_provenance
from tldw_chatbook.Chat.console_semantic_revision import project_semantic_revision_provider_message,_project_revision_response_envelope,SemanticRevisionCoordinator

db=CharactersRAGDB(COPY,client_id='offline-reservation-diagnosis')
factory=ConsoleTraceBoundaryFactory(db)
results={}; exceptions=[]
def observe(frame,event,arg):
 if event=='exception' and frame.f_code.co_filename.startswith(str(WT/'tldw_chatbook/Chat')):
  typ,val,tb=arg
  if issubclass(typ,ValueError):exceptions.append({'function':frame.f_code.co_name,'line':frame.f_lineno,'exception_type':typ.__name__,'message':str(val)})
 return observe
with db.transaction() as cur:
 owner=factory.repository.get_attached_owner_by_conversation(cur,'3c3c55a8-3e67-4ff7-ad82-c1f5c6f79fc4')
 calls=cur.execute('SELECT call_id FROM console_trace_calls ORDER BY created_at,call_sequence').fetchall()
 origin=factory.repository.get_call(cur,calls[0][0]);terminal=factory.repository.get_call(cur,calls[-1][0])
 policy=factory.repository.get_policy(cur,terminal.policy_id)
 header=factory.repository.get_request_header(cur,terminal.request_header_id)
 source_revisions=('a10abb4e-bb99-44b8-8b11-1b7b386c57eb','79aba5f6-f434-4d95-93b1-fca57a610e71','9f02dc4a-108d-4bdb-8d48-c3f9c7f0a092')
 assistant=factory.repository.get_semantic_revision(cur,source_revisions[1])
 envelope=SemanticRevisionCoordinator._message_envelope(cur,assistant.source_message_id)
 blocks=envelope['thinking_blocks']['blocks']
 results['prior_response_evidence']={'terminal_call_sequence':terminal.call_sequence,'profile':dict(header.adapter_defaults['response_projection']),'saved_blocks':[{'round_ordinal':b['round_ordinal'],'source_format':b['source_format'],'status':b['status']} for b in blocks],'terminal_link':str(factory.repository.get_response_link(cur,terminal.call_id))}
 try:
  project_semantic_revision_provider_message(cur,revision_id=assistant.revision_id,expected_conversation_id=owner.conversation_id,response_call=terminal)
 except Exception as exc:
  results['terminal_saved_projection']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 # No policy/profile or durable rows are changed. A detached terminal-only
 # envelope proves aggregate round0 is not the source of this refusal.
 final_envelope={**envelope,'thinking_blocks':{'version':1,'blocks':[b for b in blocks if b['round_ordinal']==terminal.call_sequence]}}
 try:
  _project_revision_response_envelope(cur,assistant,final_envelope,terminal)
 except Exception as exc:
  results['terminal_only_detached_projection']={'type':type(exc).__name__,'message':str(exc)}
 # Detached nonthinking historical control: the existing profile accepts a
 # canonical response without a thinking sidecar, without changing stored data.
 no_thinking_envelope={**envelope,'thinking_blocks':None}
 plain=_project_revision_response_envelope(cur,assistant,no_thinking_envelope,terminal)
 results['detached_nonthinking_projection']={'built':True,'keys':sorted(plain),'content_equals_saved':plain['content']==envelope['content']}
 values=[]
 for rid in source_revisions:
  values.append(project_semantic_revision_provider_message(cur,revision_id=rid,expected_conversation_id=owner.conversation_id))
 system_node=cur.execute("SELECT artifact_id FROM console_trace_surface_nodes WHERE sequence=0").fetchone()
 system=json.loads(factory.repository.get_artifact(cur,system_node[0]).sanitized_bytes)
 msgs=[system,*values]
 descriptors=(ProviderArtifactTraceProvenance(TraceProvenanceSource.RENDERED_SYSTEM,policy),*(SavedRevisionTraceProvenance(r) for r in source_revisions))
 before_calls=cur.execute('SELECT COUNT(*) FROM console_trace_calls').fetchone()[0]
semantic=build_console_request(msgs,message_provenance=descriptors,memory_provenance=(),mandatory_provenance=(),tool_provenance=(),capture_policy=policy,capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,metadata_provenance=(request_route_provenance(ConsoleRequestRoute.AGENT_FIRST,actor_id='34c1f65a-11f7-48af-a9c7-bef60b565127',chain_id='619ba1c8-75c6-4141-b297-30b69b45145a'),))
prepared=prepare_provider_request(semantic,wire_style='distinct_roles',provider='llama_cpp',model=terminal.model_name,capacity=resolve_request_capacity(context_window_tokens=None),count_fn=lambda msgs,model:sum(len(str(m)) for m in msgs)//4)
sys.settrace(observe)
try:
 factory(prepared,SimpleNamespace(api_key=None),ConsoleRequestRoute.AGENT_FIRST)
except Exception as exc:
 results['actual_factory_boundary']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:sys.settrace(None)
with db.transaction() as cur:
 after_calls=cur.execute('SELECT COUNT(*) FROM console_trace_calls').fetchone()[0]
 results['copy_calls_before_after']=[before_calls,after_calls]
db.close()
results['observed_value_errors']=exceptions
results['network_attempts']=network
results['actual_inputs_unchanged']=all(hashlib.sha256((LIVE/k).read_bytes()).hexdigest()==v for k,v in before.items())
# Pin source bytes, not moving metadata HEAD.
paths=['tldw_chatbook/Chat/console_trace_runtime.py','tldw_chatbook/Chat/console_trace_service.py','tldw_chatbook/Chat/console_trace_settlement.py','tldw_chatbook/Chat/console_semantic_revision.py']
results['source_sha256']={p:hashlib.sha256((WT/p).read_bytes()).hexdigest() for p in paths}
results['matches_source20f']=all((WT/p).read_bytes()==subprocess.check_output(['git','show',f'20f3f3e24b5c9ba5a58f6793af1cad0ad76034ce:{p}'],cwd=WT) for p in paths)
results['limits']='Actual factory on private SQLite backup, no gateway/transport. Ordinary canonical history values preserve saved text/revision IDs; detached thinking-free control is a projector-only comparison, not a native send or full service success. Actual live prepared follow-up payload/exception traceback was not retained. No archive metadata is added to reconstructed current user; this qualifies an independent pre-existing response-link guard and does not claim a byte-for-byte live successor replay.'
(ROOT/'result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps(results,indent=2,sort_keys=True))
assert results['actual_factory_boundary']['message']=='completed_tool_turn_revision'
assert results['terminal_saved_projection']['message']=='response_thinking_source_mismatch'
assert results['terminal_only_detached_projection']['message']=='response_thinking_source_mismatch'
assert results['actual_inputs_unchanged'] and not network and before_calls==after_calls
