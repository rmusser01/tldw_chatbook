"""Extract actual Git call expressions; no subprocess/product/runtime imports."""
import ast, argparse, json, types
from pathlib import Path, PureWindowsPath
parser=argparse.ArgumentParser()
parser.add_argument('--source-root',required=True,type=Path)
parser.add_argument('--output-root',required=True,type=Path)
args=parser.parse_args()
source=args.source_root.resolve(); output=args.output_root.resolve()
if not output.is_relative_to(source/'.superpowers'): raise RuntimeError('Owned_control_output_required')
output.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
receipts=[]
wrong_paths=[]
for name,binding in [('prepare_browser_run.py','source'),('qa_process_guard.py','REPO')]:
 tree=ast.parse((here/name).read_text(encoding='utf-8'))
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='run' and n.args and isinstance(n.args[0],ast.List) and any(isinstance(x,ast.Constant) and x.value=='rev-parse' for x in n.args[0].elts)]
 assert len(calls)==1
 seen=[]
 result=types.SimpleNamespace(stdout='  selected-revision  ')
 def run(*a,**kw): seen.append((a,kw)); return result
 path=PureWindowsPath('C:/owned/source root')
 actual=eval(compile(ast.Expression(calls[0]),name,'eval'),{'subprocess':types.SimpleNamespace(run=run),binding:path,'str':str})
 assert actual is result and len(seen)==1
 positional,kw=seen[0]
 if positional!=(['git','-c','safe.directory=C:/owned/source root','-C',str(path),'rev-parse','HEAD'],): wrong_paths.append(name)
 assert kw=={'check':True,'capture_output':True,'text':True}
 assert actual.stdout.strip()=='selected-revision'
 receipts.append({'source':name,'actual_AST_call_once':True,'posix_safe_directory':True,'original_C_path_unchanged':True,'original_check_capture_text_flags':True})
assert not wrong_paths, wrong_paths
(output/'git-path-controls.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print(json.dumps({'passed':len(receipts),'App_product_browser_subprocess_calls':0}))
