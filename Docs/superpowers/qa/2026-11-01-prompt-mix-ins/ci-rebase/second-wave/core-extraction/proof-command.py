import ast,collections,copy,hashlib,inspect,json,pathlib,subprocess
p=pathlib.Path('/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci3346-core-extract-i4xaa7gn'); repo=pathlib.Path.cwd(); specs=json.loads((p/'move-source-proof.json').read_text()); oldfiles=json.loads((p/'files.json').read_text()); results=[]
def tree_old(f): return ast.parse((p/f'before-{oldfiles.index(f)}.py').read_text())
def node(tree,name): return next(x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and x.name==name)
def dump(x): return ast.dump(x,include_attributes=False)
class Unhost(ast.NodeTransformer):
 def visit_Attribute(self,x):
  self.generic_visit(x)
  return ast.copy_location(ast.Name(id=x.attr,ctx=x.ctx),x) if isinstance(x.value,ast.Name) and x.value.id=='host' else x

def docnorm(x):
 for n in ast.walk(x):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str): n.body[0].value.value=inspect.cleandoc(n.body[0].value.value)
 return x
for spec in specs:
 if 'method' in spec:
  name=spec['method']; old=node(tree_old(spec['source']),name); wrapper=node(ast.parse(pathlib.Path(spec['source']).read_text()),name); moved=node(ast.parse(pathlib.Path(spec['destination']).read_text()),name)
  norm=Unhost().visit(copy.deepcopy(moved)); norm.decorator_list=copy.deepcopy(old.decorator_list)
  if spec['local_host_import']: norm.body=[x for x in norm.body if not(isinstance(x,ast.ImportFrom) and any(a.asname=='host' for a in x.names))]
  if name=='_run_agent_reply':
   idx=next(i for i,x in enumerate(old.args.kwonlyargs) if x.arg=='work_origin'); assert norm.args.kw_defaults[idx] is None; norm.args.kw_defaults[idx]=copy.deepcopy(old.args.kw_defaults[idx])
  same=dump(docnorm(norm))==dump(docnorm(copy.deepcopy(old)))
  states=lambda n:collections.Counter((x.attr,type(x.ctx).__name__) for x in ast.walk(n) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self')
  result={'name':name,'implementation_ast_equal_after_documented_normalization':same,'wrapper_arguments_equal':dump(wrapper.args)==dump(old.args),'wrapper_decorators_equal':[dump(x) for x in wrapper.decorator_list]==[dump(x) for x in old.decorator_list],'self_state_accesses_equal':states(old)==states(moved),'destination':spec['destination']}
  assert all(v for k,v in result.items() if k.endswith(('equal','normalization'))),result
  results.append(result)
 else:
  dest=ast.parse(pathlib.Path(spec['pure_destination']).read_text()); old=tree_old(spec['pure_source'])
  def decl(t,name):
   return next(n for n in t.body if getattr(n,'name',None)==name or isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets) or isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id==name)
  for name in spec['names']:
   same=dump(decl(dest,name))==dump(decl(old,name)); assert same,name; results.append({'name':name,'pure_ast_equal':same,'destination':spec['pure_destination']})
(p/'move-check-final.json').write_text(json.dumps(results,indent=2)+'\n')
files=oldfiles+['tldw_chatbook/Chat/'+f+'.py' for f in ['console_agent_reply','console_settings_persistence','console_memory_banner']]+['Tests/Tools/test_admitted_root_consumer_census.py','Tests/Chat/test_console_auxiliary_routing.py','Tests/Chat/test_console_chat_fork_persistence.py','Tests/Chat/test_console_chat_controller.py','Tests/Chat/test_console_prompt_mixin_retry.py']
(p/'source-inventory-final.json').write_text(json.dumps([{'path':f,'lines':len(pathlib.Path(f).read_text().splitlines()),'sha256':hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()} for f in files],indent=2)+'\n')
# Evidence text from actual current stock tables and preparation functions.
controller=ast.parse(pathlib.Path(oldfiles[0]).read_text()); source=pathlib.Path(oldfiles[0]).read_text(); names=['_CONSOLE_TOOL_COMPOSITION_METHODS','_CONTEXT_DISPLAY_CALLBACKS','_CONSOLE_PENDING_FACTS_READERS','_CONSOLE_PREPARING_CONTROLLER_READERS','_CONSOLE_CONTEXT_PRESENTATION_READERS']; chunks=[]
for n in controller.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):chunks.append(ast.get_source_segment(source,n))
prep='tldw_chatbook/Chat/console_configuration_preparation.py'; s=pathlib.Path(prep).read_text(); t=ast.parse(s)
for name in ['_source_references','standard_console_configuration_sources','_standard_configuration_references','_skill_catalog_sources_current']:
 try: chunks.append(ast.get_source_segment(s,node(t,name)))
 except StopIteration: pass
moved=[a['method'] for a in specs if 'method' in a]; witnessfiles=['console_configuration_preparation.py','console_attention_preparation.py','console_turn_preparation.py','console_received_dispatch.py']; matches={f:[n for n in moved if n in pathlib.Path('tldw_chatbook/Chat',f).read_text()] for f in witnessfiles}
(p/'source-fence-audit.md').write_text('# Core extraction source-fence audit\n\nBaseline ba1cafc55dd5ac9b6b1f066c936206e652ed6b65. This records the concrete stock predicates audited, not a claim that every future dependency is known. None of the 16 moved execution/settings/fork methods is captured by these preparation stock tables. The execution/settings delegates are invoked at their existing execution/persistence sites, not used as stock qualifications for FULL preparation skipping. Existing source readers, guards and revision checks remain in their old owners. Pure banner/trace exports bind the actual moved function/class objects, not wrappers hiding their implementation. Live host module lookup preserves the original callback dependencies at use time.\n\nExact moved-name matches in preparation owners: `'+json.dumps(matches)+'`.\n\nActual current stock table and configuration predicate source:\n\n```python\n'+'\n\n'.join(chunks)+'\n```\n')
print('Verified definitions/declarations:',len(results)); print('Inventory files:',len(files)); print('Preparation moved-name matches:',matches)
