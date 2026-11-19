import ast,copy,subprocess,json,difflib
from pathlib import Path
base='ba1cafc55d';pairs=[('console_chat_controller','console_agent_reply','ConsoleChatController'),('console_chat_store','console_settings_persistence','ConsoleChatStore'),('chat_persistence_service','console_chat_fork','ChatPersistenceService'),('console_trace_service','console_trace_models',None),('console_display_state','console_memory_banner',None)]
class Norm(ast.NodeTransformer):
 def visit_Attribute(self,n):
  n=self.generic_visit(n)
  if isinstance(n.value,ast.Name) and n.value.id=='host':return ast.Name(n.attr,n.ctx)
  return n
 def visit_ImportFrom(self,n):
  if any(a.asname=='host' for a in n.names):return None
  return n
rows=[];diffs=[]
for owner,helper,cls in pairs:
 p=f'tldw_chatbook/Chat/{owner}.py';old=ast.parse(subprocess.check_output(['git','show',base+':'+p],text=True));current=ast.parse(Path(p).read_text());new=ast.parse(Path(f'tldw_chatbook/Chat/{helper}.py').read_text());oldbody=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else old.body
 curbody=next(n for n in current.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else current.body
 names={n.name:n for n in oldbody if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))};cur={n.name:n for n in curbody if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
 for n in new.body:
  if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) or n.name not in names:continue
  orig=copy.deepcopy(names[n.name]);norm=Norm().visit(copy.deepcopy(n));decorators=ast.dump(ast.Module(orig.decorator_list,[]));orig.decorator_list=[];norm.decorator_list=[]
  same=ast.dump(orig,include_attributes=False)==ast.dump(norm,include_attributes=False)
  r={'owner':owner,'helper':helper,'name':n.name,'normalized_equal':same,'original_decorators':decorators};rows.append(r)
  if not same:diffs.append('\n'+owner+'.'+n.name+'\n'+'\n'.join(difflib.unified_diff(ast.unparse(orig).splitlines(),ast.unparse(norm).splitlines(),n=2)))
  if cls and n.name in cur:
   wrap=cur[n.name];r['wrapper_arguments_equal']=ast.dump(wrap.args)==ast.dump(orig.args);r['wrapper_decorators_equal']=ast.dump(ast.Module(wrap.decorator_list,[]))==decorators
   r['wrapper_body']=ast.unparse(wrap)
Path('/private/tmp/mixins-core-review-ast.json').write_text(json.dumps(rows,indent=2));Path('/private/tmp/mixins-core-review-ast-differences.txt').write_text('\n'.join(diffs));print('MOVED',len(rows),'EQUAL',sum(r['normalized_equal'] for r in rows));print('\n'.join(diffs)[:10000]);print('WRAPPER-SIGNATURE-DIFFERENCES',[r['name'] for r in rows if r.get('wrapper_arguments_equal') is False or r.get('wrapper_decorators_equal') is False])
