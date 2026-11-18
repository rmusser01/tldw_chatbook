from pathlib import Path
import ast,hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
p=Path("Tests/UI/test_console_runtime_ownership.py")
before=p.read_bytes()
argv=[sys.executable,"-B","-m","ruff","format","--range","3130-3170",str(p)]
r=subprocess.run(argv,capture_output=True,text=True)
assert r.returncode==0,r.stdout+r.stderr
after=p.read_bytes()
assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after))
(root/"finalfix-manual-format-detail.json").write_text(json.dumps({"argv":argv,"returncode":r.returncode,"output":r.stdout+r.stderr,"before_sha256":hashlib.sha256(before).hexdigest(),"after_sha256":hashlib.sha256(after).hexdigest(),"ast_equal":True},indent=2))
print(r.stdout+r.stderr,end="")
print("Exact manual-currentness hunk formatted; AST equality confirmed.")
