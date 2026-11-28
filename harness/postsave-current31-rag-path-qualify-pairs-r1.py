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
assert hashlib.sha256((E/'postsave-current31-config-guard-r4.py').read_bytes()).hexdigest()=='c8d5d50f1b881920ee8724b69c05877d24fe25283528ba540b34868a7d52b622'
guard=runpy.run_path(str(E/'postsave-current31-config-guard-r4.py'),run_name='complete_config_guard')
guard['self_check']()
refconfig=guard['reference']()['configuration']
assert plan['order']==['baseline'] and plan['planned_posts']==1
assert plan['heads']=={'baseline':M['baseline_head'],'candidate':M['candidate_head']} and plan['profiler_enabled']
for name,digest in M['frozen_helpers'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest
pids={x.strip() for x in subprocess.check_output(['ps','-axo','pid='],text=True).splitlines()}
fixture=json.loads((E/'loopback-http.json').read_text())
assert len(fixture)==1 and fixture[0]['status']==200
qualified=[]
for number,r in enumerate(plan['runs'],1):
    out=P/f'run{number}';a=json.loads((out/'closed-audit.json').read_text());root=Path(a['profile'])
    assert a['head']==plan['heads'][r['role']] and a['geometry']==[120,35]
    schemas=list(a['tool_schemas'].values());assert len(schemas)==1 and 0<len(schemas[0])<=128 and schemas==[fixture[0]['tool_schemas']]
    assert len(a['headers'])==1 and a['headers'][0][:2]==['deepseek','deepseek-flash'] and r['settings']==reference['settings']
    assert normalize(tomllib.loads((root/'config.toml').read_text()),root)==refconfig
    assert json.loads((out/'exit-codes.json').read_text())=={'driver':0,'server':0}
    assert not set(json.loads((out/'owned-processes-before.json').read_text())['owned_observed'])&pids
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',r['port']))!=0
    assert a['source_map_unchanged'] and a['pending_checkpoints']==0
    assert sorted(m[1:3] for m in a['messages'])==[['assistant','CEDAR READY'],['user','Remember the marker CEDAR. Reply exactly CEDAR READY.']]
    assert len(a['traces'])==len(a['links'])==1 and a['traces'][0][1:3]==['complete','complete'] and a['links'][0][1]=='verified_equal'
    assert [x['event'] for x in a['http']]==['send','response_open','response_closed'] and a['http'][1]['status']==200
    assert (root/'postsave-spans.json').is_file() and not (root/'postsave-stacks.json').exists()
    log=(root/'data/matched-verification/tldw_cli_app.log').read_text()
    assert all(s in log for s in ['Application quit initiated','Screen chat unmounted','--- App Unmounting ---'])
    errors=[l for l in log.splitlines() if '[ERROR' in l];assert len(errors)==1 and 'Failed to find #app-log-display widget' in errors[0]
    qualified.append((r,a,log,errors))
assert len(qualified)==1
G=Path(plan['runs'][0]['root'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==M['candidate_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
sources=json.loads((P/'baseline-sources-before.json').read_text())
pinned=json.loads((E/'postsave-current31-rag-path-python-prelaunch-r1.json').read_text())
assert sources==pinned
assert all(hashlib.sha256((G/n).read_bytes()).hexdigest()==h for n,h in sources.items())

from active_segments import segments, self_check
self_check()
r,a,log,errors=qualified[0]
data=json.loads((Path(a['profile'])/'postsave-spans.json').read_text())
assert not data['overflow'] and data['observer_sha256']==M['frozen_helpers']['postsave-observer.py']
rows=data['rows']
assert 2<len(rows)<=32768 and rows[0]['event']=='durable_commit_succeeded' and rows[-1]['event']=='trace_reservation_entered'
begin,end=rows[0]['ns'],rows[-1]['ns']
assert 0<begin<end<a['http'][0]['monotonic_ns']
complete,clipped,pending=segments(rows)
stage={}
for line in log.splitlines():
    if 'event=console_send_stage' in line:
        fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
        key=(fields.get('phase'),fields.get('status'))
        assert key not in stage
        stage[key]=int(fields['duration_ms'])
post=stage['trace_reservation','entered']-stage['durable_commit','succeeded']
assert post>=0
result={'head':M['baseline_head'],'qualified':True,'posts':1,'observed_rows':len(rows),
        'instrumented_postsave_ms':(end-begin)/1e6,
        'app_log_postsave_ms':post,'instrumented_input_to_http_seconds':a['samples'][0]['input_to_post_seconds'],
        'clipped':len(clipped),'pending':len(pending),
        'complete_activations':sorted(complete,key=lambda row:-row['active_thread_cpu_ms']),
        'benchmark_reference':{'headers':a['headers'],'tool_schemas':{'expected':fixture[0]['tool_schemas']},'settings':r['settings']},
        'scope':'Instrumented first send, inclusive and overlapping thread CPU or wall time, one loopback stub cohort.'}
target=E/'postsave-current31-rag-path-diagnostic-qualified.json'
with target.open('x') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in {'complete_activations','benchmark_reference'}}))
