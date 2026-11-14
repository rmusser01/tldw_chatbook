import ast,collections,hashlib,json,pathlib,subprocess
p=pathlib.Path(__file__).parent;files=['Tests/UI/library_note_links_support.py','Tests/UI/test_library_note_links_readiness_diagnostics.py'];exe='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';out={}
for f in files:
 old=subprocess.run(['git','show','f09597cdd5:'+f],capture_output=True,text=True).stdout;new=pathlib.Path(f).read_text()
 def check(s):
  r=subprocess.run([exe,'check','--no-cache','--output-format','json','--stdin-filename',f,'-'],input=s,capture_output=True,text=True);return [(x['code'],x['message']) for x in json.loads(r.stdout)]
 before,after=check(old),check(new);out[f]={'before_lint':before,'after_lint':after,'added_lint':list((collections.Counter(after)-collections.Counter(before)).elements()),'sha256':hashlib.sha256(new.encode()).hexdigest()}
 if 'support.py' in f:
  find=lambda s:[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_until' and any(isinstance(a,ast.Constant) and a.value=='the healthy Library source DOM' for a in n.args)][0]
  out[f]['exact_wait_call_and_predicate_ast_unchanged']=ast.dump(find(old))==ast.dump(find(new));assert out[f]['exact_wait_call_and_predicate_ast_unchanged']
 assert not out[f]['added_lint']
(p/'static-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
