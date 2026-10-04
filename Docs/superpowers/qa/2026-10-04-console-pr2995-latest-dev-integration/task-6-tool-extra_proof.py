exec(open('/private/tmp/pr2995-task6/prove_integration.py').read().split('a,b,c=')[0])
a,c=tree(old),tree('HEAD');paths=[p for p in a if p.startswith(('qa/','Docs/superpowers/qa/'))];rows=[{'path':p,'old':a[p],'head':c.get(p),'equal':a[p]==c.get(p)} for p in sorted(paths)];(O/'task-6-historical-artifacts.json').write_text(json.dumps(rows,indent=2)+'\n')
def enrollment(ref):
 t=ast.parse(blob(ref,'Tests/conftest.py'));return max((set(ast.literal_eval(n)) for n in ast.walk(t) if isinstance(n,ast.Set) and all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in n.elts)),key=len)
x,y,z=[enrollment(ref) for ref in (old,dev,'HEAD')];enroll={'old_count':len(x),'dev_count':len(y),'head_count':len(z),'exact_union':z==x|y,'upstream_new':sorted(y-x),'feature_only':sorted(x-y)}
budget=[]
for p in ['Tests/Performance/test_app_import_weight.py','Tests/Performance/test_ui_ready_module_census.py','Tests/Performance/test_boot_css_byte_budget.py','Tests/Performance/test_screen_preimport_payload_budget.py']:
 def constants(ref):
  t=ast.parse(blob(ref,p));return {n.targets[0].id:ast.dump(n.value) for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith(('MAX_','MIN_'))}
 budget.append({'path':p,'dev_constants':constants(dev),'head_constants':constants('HEAD'),'unchanged':constants(dev)==constants('HEAD')})
(O/'task-6-profile-and-budgets.json').write_text(json.dumps({'enrollment':enroll,'budgets':budget},indent=2)+'\n')
print(json.dumps({'historical_count':len(rows),'historical_exact':all(x['equal'] for x in rows),'enrollment':enroll,'budgets_unchanged':all(x['unchanged'] for x in budget)},indent=2))
