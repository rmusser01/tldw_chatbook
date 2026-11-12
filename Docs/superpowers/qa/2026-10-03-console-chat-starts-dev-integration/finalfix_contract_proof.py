from pathlib import Path
import ast, hashlib, json, subprocess
root=Path(__file__).resolve().parent
base="473c7b26eccf1e7b77c4c8e2d4c7f57ccb840afe"
paths=json.loads((root/"finalfix-working-source-detail.json").read_text())["checks"]
owned=subprocess.check_output(["git","diff","--name-only",base,"--","tldw_chatbook","Tests"],text=True).splitlines()
def exclusions(data):
 tree=ast.parse(data)
 result=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   for dec in node.decorator_list:
    value=ast.unparse(dec)
    if "pytest.mark.skip" in value or "pytest.mark.xfail" in value:
     result.append((node.name,value))
 return sorted(result)
records=[]
for path in owned:
 before=subprocess.check_output(["git","show",base+":"+path])
 after=Path(path).read_bytes()
 assert exclusions(before)==exclusions(after),path
 records.append({"path":path,"before_sha256":hashlib.sha256(before).hexdigest(),"after_sha256":hashlib.sha256(after).hexdigest(),"unchanged_skip_xfail_decorators":exclusions(after)})
(root/"finalfix-owned-source-manifest.json").write_text(json.dumps({"base":base,"owned":records},indent=2))
print(f"{len(records)} exact owned paths: final hashes captured; all inherited skip/xfail decorators equal FIX_BASE; no added exclusions.")
