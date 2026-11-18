import collections,hashlib,json,os,pathlib,runpy,sys,threading,time,urllib.parse
R=pathlib.Path(__file__).parent
WT='/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook'
for name,path in {'HOME':R/'home','XDG_CONFIG_HOME':R/'xdg-config','XDG_DATA_HOME':R/'xdg-data','XDG_CACHE_HOME':R/'xdg-cache','TLDW_CONFIG_PATH':R/'config.toml'}.items():os.environ[name]=str(path)
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,WT)
import keyring,requests
assert type(keyring.get_keyring()).__module__=='keyring.backends.null'
lock=threading.Lock();count=sum(1 for p in R.parent.glob('*/wire-metadata.jsonl') for line in p.read_text().splitlines() if json.loads(line).get('event')=='request')

def record(row):
 with lock:
  with (R/'wire-metadata.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')

def category(error):
 s=str(error).lower()
 for name,words in [('schema',('schema','json_schema','additionalproperties')),('template',('template','jinja')),('context',('context','n_ctx','too long')),('timeout',('timeout','timed out')),('connection',('connect','connection')),('cancelled',('cancel','abort'))]:
  if any(w in s for w in words):return name
 return 'unclassified'

original_post=requests.Session.post

def observed_post(session,url,*args,**kw):
 global count
 u=urllib.parse.urlparse(url)
 if not u.path.endswith('/chat/completions'):return original_post(session,url,*args,**kw)
 if u.hostname not in ('127.0.0.1','localhost') or u.port!=9099:raise RuntimeError('probe disallows non-local inference')
 with lock:
  if count>=10:raise RuntimeError('whole-probe model-attempt cap reached')
  count+=1;i=count
 payload=kw.get('json') or {}
 schemas=payload.get('tools') or []
 row={'event':'request','attempt':i,'time':time.time(),'max_tokens':payload.get('max_tokens'),'stream':payload.get('stream'),'role_counts':dict(collections.Counter(m.get('role','unknown') for m in payload.get('messages',[]))),'tool_names':[t.get('function',{}).get('name') for t in schemas],'schema_sha256':hashlib.sha256(json.dumps(schemas,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'reasoning_controls':{k:payload[k] for k in ('reasoning_effort','enable_thinking','chat_template_kwargs') if k in payload},'transport_retry_total':session.get_adapter(url).max_retries.total}
 # Record structural counts only, without retaining selection metadata or text.
 handoffs=[]
 for m in payload.get('messages',[]):
  content=m.get('content')
  if not isinstance(content,str) or 'Previously model-observed archive selection metadata.' not in content:continue
  start=content.find('\n{"source_session_id"')
  if start<0:continue
  try:envelope,_=json.JSONDecoder().raw_decode(content[start+1:])
  except (ValueError,TypeError):continue
  pages=envelope.get('archive_recovery_results',[])
  handoffs.append({'pages':len(pages),'rows':sum(len(page.get('items',[])) for page in pages),'omitted_pages':envelope.get('omitted_pages')})
 row['archive_followup_handoffs']=handoffs
 record(row)
 try:resp=original_post(session,url,*args,**kw)
 except Exception as e:
  record({'event':'transport_exception','attempt':i,'exception_type':type(e).__name__,'category':category(e)});raise
 record({'event':'http','attempt':i,'status':resp.status_code,'content_type':resp.headers.get('content-type','').split(';')[0]})
 if resp.status_code>=400:
  try:err=resp.json().get('error',{})
  except Exception:err={}
  record({'event':'http_error','attempt':i,'category':category(err)})
 original_lines=resp.iter_lines
 def observed_lines(*a,**k):
  stats={'event':'completion','attempt':i,'content_bytes':0,'reasoning_bytes':0,'named_call_count':0,'tool_names':[],'finish_reasons':[],'usage':None,'sse_error':False,'invalid_json_lines':0,'done':False}
  names=set();calls=set()
  try:
   for line in original_lines(*a,**k):
    s=line.decode('utf-8',errors='replace') if isinstance(line,bytes) else line
    if s and s.startswith('data:'):
     body=s[5:].strip()
     if body=='[DONE]':stats['done']=True
     elif body:
      try:data=json.loads(body)
      except Exception:stats['invalid_json_lines']+=1;data={}
      if 'error' in data:stats['sse_error']=True;stats['error_category']=category(data['error'])
      if isinstance(data.get('usage'),dict):stats['usage']={k:v for k,v in data['usage'].items() if k in ('prompt_tokens','completion_tokens','total_tokens')}
      for choice in data.get('choices',[]):
       if choice.get('finish_reason') is not None:stats['finish_reasons'].append(choice['finish_reason'])
       delta=choice.get('delta') or choice.get('message') or {}
       for key,dest in (('content','content_bytes'),('reasoning_content','reasoning_bytes'),('reasoning','reasoning_bytes')):
        if isinstance(delta.get(key),str):stats[dest]+=len(delta[key].encode())
       for t in delta.get('tool_calls',[]):
        name=(t.get('function') or {}).get('name')
        if name:names.add(name);calls.add((choice.get('index',0),t.get('index',0)))
    yield line
  except Exception as e:stats.update(exception_type=type(e).__name__,error_category=category(e));raise
  finally:
   stats['named_call_count']=len(calls);stats['tool_names']=sorted(names);record(stats)
 resp.iter_lines=observed_lines
 return resp
requests.Session.post=observed_post
from tldw_chatbook import config
from tldw_chatbook.Chat.console_agent_bridge import console_run_budget
budget=console_run_budget()
record({'event':'provenance','source':config.__file__,'config':str(config.get_cli_config_path()),'data':str(config.get_user_data_dir()),'keyring':type(keyring.get_keyring()).__module__+'.'+type(keyring.get_keyring()).__name__,'runtime_budget':{k:getattr(budget,k) for k in ('max_model_turns','max_steps','max_wall_seconds','max_model_retries')}})
assert budget.max_model_retries==0
runpy.run_module('tldw_chatbook.app',run_name='__main__')
