from pathlib import Path
import ast, hashlib,json,subprocess,sys
p=Path('tldw_chatbook/Chat/console_runtime.py');before=p.read_bytes()
argv=[sys.executable,'-B','-m','ruff','format','--range','2304-2323',str(p)]
r=subprocess.run(argv,capture_output=True,text=True);assert r.returncode==0
assert ast.dump(ast.parse(before))==ast.dump(ast.parse(p.read_bytes()))
record={'argv':argv,'returncode':r.returncode,'output':r.stdout+r.stderr,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ast_equal':True}
(Path(__file__).parent/'latest-runtime-format-detail.json').write_text(json.dumps(record,indent=2));print(record)
