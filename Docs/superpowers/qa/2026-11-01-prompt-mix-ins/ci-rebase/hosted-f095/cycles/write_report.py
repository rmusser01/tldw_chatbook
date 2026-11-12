import ast,collections,hashlib,importlib.metadata,importlib.util,json,subprocess
from pathlib import Path
root=Path('/private/tmp/mixins-f095-cycles');p=Path('Tests/UI/test_console_first_send_import_cycles.py');old=subprocess.check_output(['git','show','f09597c:'+str(p)],text=True);new=p.read_text();ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';counts=[]
for source in (old,new):
 r=subprocess.run([ruff,'check','--no-cache','--output-format=json','--stdin-filename',str(p),'-'],input=source,capture_output=True,text=True);counts.append(collections.Counter((d['code'],d['message']) for d in json.loads(r.stdout)))
prior=ast.parse(old);now=ast.parse(new);functions={n.name:n for n in now.body if isinstance(n,ast.FunctionDef)};unchanged=[]
for node in prior.body:
 if isinstance(node,ast.FunctionDef) and node.name!='warm_up_modules':
  assert ast.dump(node,include_attributes=False)==ast.dump(functions[node.name],include_attributes=False),node.name;unchanged.append(node.name)
spec=importlib.util.find_spec('textual_serve');assert spec is not None
proof={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'unchanged_original_graph_oracle_asts':unchanged,'lint':{'before':sum(counts[0].values()),'after':sum(counts[1].values()),'added':[(str(k),v) for k,v in (counts[1]-counts[0]).items()]},'local_optional_package':{'distribution':'textual-serve','version':importlib.metadata.version('textual-serve'),'spec_origin':spec.origin}}
(root/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
(root/'report.md').write_text('''# f095 UI29 optional warmup cycle fixture repair

Starting f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80. Scope: only Tests/UI/test_console_first_send_import_cycles.py. No product/optional dependency installation, threshold, skip, detector or warmup-list change. Parent approved and recorded plan before edits.

## Cause and behavior

The raw fixture loop aborted when textual_serve was not installed in CI's core extras, although pyproject.toml assigns textual-serve to [web] and optional_deps explicitly classifies that feature as optional. Production warmup catches speculative import errors; the test should tolerate only this known absent root while staying stricter about broken imports.

A self-contained helper is shared by unit controls and the actual fresh-process fixture via inspect.getsource. It catches only ModuleNotFoundError when both requested name and exception.name equal textual_serve. Required/transitive/child import failures and ordinary ImportError/RuntimeError propagate as the original exception. It records the missing name and continues the full unchanged module list. No optional availability boolean gates coverage.

The two original graph tests are parametrized for the real environment and explicitly absent optional root. The absent branch uses a meta_path finder before app imports that raises ModuleNotFoundError only for textual_serve itself. Both branches import all other actual warmup modules and execute unchanged graph/cycle/dynamic oracles, >60 module/edge anti-vacuity, and required hooks-package presence. The real environment here has textual-serve installed; source-proof.json records version/spec origin. CI can exercise its actual absent environment plus the explicit absence branch without an eager install.

## Evidence

Focused whole cycle file:11passed in6.38s, tests.log/xml. Four are the two original graph guards across both environments, one is original cycle-finder positive control, six are new optional-continuation/strict-failure controls. Required/transitive/child names and ordinary ImportError/RuntimeError all prove failure preservation, including original exception identity. No full suite/app latency tests. Core confirmed no app tests before these fresh subprocesses and was notified immediately after completion.

Original hosted failure is in /private/tmp/mixins-ci-f095-ui29-red.log. This supplied missing-package setup error is the red evidence; no redundant local rerun of the original raw loop was needed. Both actual installed and blocked-absent full guards now pass.

Post-run formatting only wrapped Python expressions; final format-check and git diff --check pass. Default lint1->1 with zero added (existing SIM114); all original graph/oracle function ASTs identical, recorded in source-proof.json. No product bytes changed, so five performance budgets were not repeated. No owned processes remain active.

Independent review is parent-assigned. Exact artifacts with hashes are in artifact-files.json; scratch subprocess profiles excluded.
''')
paths=[root/n for n in ['report.md','tests.log','tests.xml','patch.diff','source-proof.json','lint-current.json','format-check.log','write_report.py']]+[Path('/private/tmp/mixins-ci-f095-ui29-red.log')]
(root/'artifact-files.json').write_text(json.dumps({'files':[{'path':str(x),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in paths]},indent=2)+'\n')
print(json.dumps(proof,indent=2));print(root/'artifact-files.json')
