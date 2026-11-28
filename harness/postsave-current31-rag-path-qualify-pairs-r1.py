"""Root read-only qualification, then paired statistics; never launches an app."""
from pathlib import Path
import hashlib,json,re,socket,statistics,subprocess,tomllib
E=Path(__file__).resolve().parent
P=E/'paired-postsave-current31-rag-path-firstsend-r1'
M=json.loads((E/'postsave-current31-rag-path-paired-frozen-chain-r1.json').read_text())
plan=json.loads((P/'plan-closed.json').read_text())
reference=json.loads((E/'expected-send.json').read_text())
def normalize(v,root):
    if isinstance(v,str):return v.replace(str(root),'<owned-profile>')
    if isinstance(v,list):return [normalize(x,root) for x in v]
    if isinstance(v,dict):return {k:normalize(x,root) for k,x in v.items()}
    return v
import runpy
assert hashlib.sha256((E/'postsave-current31-config-guard-r4.py').read_bytes()).hexdigest()=='ce6ea09787f733a59f06c72267b6d86ea4453f7be0d9c8094c6ebc9b9e391f2d'
guard=runpy.run_path(str(E/'postsave-current31-config-guard-r4.py'),run_name='complete_config_guard')
guard['self_check']()
refconfig=guard['reference']()['configuration']
assert plan['order']==['baseline','candidate','candidate','baseline','baseline','candidate','candidate','baseline']
assert plan['heads']=={'baseline':M['baseline_head'],'candidate':M['candidate_head']} and not plan['profiler_enabled']
for name,digest in M['frozen_helpers'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest
pids={x.strip() for x in subprocess.check_output(['ps','-axo','pid='],text=True).splitlines()}
qualified=[]
for number,r in enumerate(plan['runs'],1):
    out=P/f'run{number}';a=json.loads((out/'closed-audit.json').read_text());root=Path(a['profile'])
    assert a['head']==plan['heads'][r['role']] and a['geometry']==[120,35]
    schemas=list(a['tool_schemas'].values());assert len(schemas)==1 and len(schemas[0])==27 and schemas==list(reference['tool_schemas'].values())
    assert a['headers']==reference['headers'] and r['settings']==reference['settings']
    assert normalize(tomllib.loads((root/'config.toml').read_text()),root)==refconfig
    assert json.loads((out/'exit-codes.json').read_text())=={'driver':0,'server':0}
    assert not set(json.loads((out/'owned-processes-before.json').read_text())['owned_observed'])&pids
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',r['port']))!=0
    assert a['source_map_unchanged'] and a['pending_checkpoints']==0
    assert sorted(m[1:3] for m in a['messages'])==[['assistant','CEDAR READY'],['user','Remember the marker CEDAR. Reply exactly CEDAR READY.']]
    assert len(a['traces'])==len(a['links'])==1 and a['traces'][0][1:3]==['complete','complete'] and a['links'][0][1]=='verified_equal'
    assert [x['event'] for x in a['http']]==['send','response_open','response_closed'] and a['http'][1]['status']==200
    assert not (root/'postsave-spans.json').exists() and not (root/'postsave-stacks.json').exists()
    log=(root/'data/matched-verification/tldw_cli_app.log').read_text()
    assert all(s in log for s in ['Application quit initiated','Screen chat unmounted','--- App Unmounting ---'])
    errors=[l for l in log.splitlines() if '[ERROR' in l];assert len(errors)==1 and 'Failed to find #app-log-display widget' in errors[0]
    qualified.append((r,a,log,errors))
assert len(qualified)==8
G=Path(plan['runs'][0]['root'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==M['candidate_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
sources={role:json.loads((P/f'{role}-sources-before.json').read_text()) for role in ('baseline','candidate')}
changed=[n for n in sorted(sources['baseline'].keys()|sources['candidate'].keys()) if sources['baseline'].get(n)!=sources['candidate'].get(n)]
assert changed==['tldw_chatbook/RAG_Search/config_profiles.py', 'tldw_chatbook/RAG_Search/simplified/config.py', 'tldw_chatbook/RAG_Search/storage_paths.py', 'tldw_chatbook/Tools/remote_worker_bundle.py', 'tldw_chatbook/Utils/sensitive_paths.py']
assert all(hashlib.sha256((G/n).read_bytes()).hexdigest()==h for n,h in sources['candidate'].items())
# Do not compute any statistics until every complete packet qualified above.
rows=[]
for number,(r,a,log,errors) in enumerate(qualified,1):
    stage={}
    for line in log.splitlines():
        if 'event=console_send_stage' in line:
            fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
            key=(fields.get('phase'),fields.get('status'))
            assert key not in stage
            stage[key]=int(fields['duration_ms'])
    post=stage['trace_reservation','entered']-stage['durable_commit','succeeded'];assert post>=0
    rows.append({'run':number,'role':r['role'],'head':r['head'],'postsave_ms':post,'input_to_http_seconds':a['samples'][0]['input_to_post_seconds'],'host_load':r['host_load'],'profile':r['profile']})
stats={role:{metric:{'median':statistics.median(v[metric] for v in rows if v['role']==role),'minimum':min(v[metric] for v in rows if v['role']==role),'maximum':max(v[metric] for v in rows if v['role']==role)} for metric in ('postsave_ms','input_to_http_seconds')} for role in ('baseline','candidate')}
pairs=[]
for b,c in [(1,2),(4,3),(5,6),(8,7)]:
    x,y=rows[b-1],rows[c-1]
    pairs.append({'baseline_run':b,'candidate_run':c,'baseline_minus_candidate_postsave_ms':x['postsave_ms']-y['postsave_ms'],'baseline_minus_candidate_input_to_http_seconds':x['input_to_http_seconds']-y['input_to_http_seconds']})
result={'heads':plan['heads'],'all8qualified':True,'exact_ordered27_schemas_headers_full_supplied_and_expanded_settings_equal':True,'normal_quit_chat_app_unmount_all8':True,'actual_driver_server_exits0_and_observed_pids_ports_closed_all8':True,'whole_python_source_maps_stable':{k:len(v) for k,v in sources.items()},'only_changed_python_paths':changed,'no_profiler_sampler_all8':True,'rows':rows,'statistics':stats,'pairs':pairs,'inherited_errors':[{'run':i,'errors':v[3]} for i,v in enumerate(qualified,1)],'limits':'Four pairs on one host, no p95/OS-cold-cache/native paint/Windows/later-send/large-history/broad-provider claim. No separate real-home audit hook/census, separate app-child exit status, or arbitrary physical native-worker retirement. Post-save and Enter-toHTTP boundaries differ.'}
result['release_gate_met'] = stats['candidate']['input_to_http_seconds']['median'] < stats['baseline']['input_to_http_seconds']['median'] and sum(p['baseline_minus_candidate_input_to_http_seconds'] > 0 for p in pairs) == 4
target=E/'postsave-current31-rag-path-paired-qualified-r1.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':str(target),'rows':rows,'statistics':stats,'pairs':pairs},indent=2))
