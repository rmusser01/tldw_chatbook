import ast, copy, hashlib, sys, pathlib, collections
LIFECYCLE = {"compose", "render", "__init__", "on_mount", "on_unmount"}
by_name, by_body, by_shape = (collections.defaultdict(list) for _ in range(3))
class Anon(ast.NodeTransformer):
    def visit_Name(self, n): n.id = "_"; return n
    def visit_arg(self, n): n.arg = "_"; return n
    def visit_Attribute(self, n):
        self.generic_visit(n)
        if not n.attr.startswith("__"): n.attr = "_"
        return n
    def visit_Constant(self, n): n.value = type(n.value).__name__; return n
def _docstring(body):
    return body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
roots = [pathlib.Path(a) for a in sys.argv[1].split(",")]
files = []
for r in roots:
    files += [r] if r.is_file() else list(r.rglob("*.py"))
for p in files:
    if any(x in p.parts for x in (".venv", "Third_Party", "Tests", "__pycache__", "Splash_Screens")): continue
    try: tree = ast.parse(p.read_text(encoding="utf-8"))
    except Exception: continue
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        name = node.name
        if name in LIFECYCLE or name.startswith(("on_", "watch_", "action_", "_on_")): continue
        loc = f"{p}:{node.lineno}"
        by_name[name].append(loc)
        body = node.body[1:] if _docstring(node.body) else node.body
        if len(body) < 2: continue
        raw = hashlib.sha1(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()[:12]
        shape = hashlib.sha1(ast.dump(Anon().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))).encode()).hexdigest()[:12]
        by_body[raw].append(f"{name}@{loc}"); by_shape[shape].append(f"{name}@{loc}")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "candidates"); out.mkdir(exist_ok=True)
def emit(fn, groups, min_files):
    n=0
    with open(out / fn, "w") as f:
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            fs = {x.split("@")[-1].rsplit(":", 1)[0] for x in v}
            if len(fs) >= min_files: f.write(f"{len(fs)}\t{k}\t{' '.join(v)}\n"); n+=1
    print(fn, n, "rows")
emit("dup_by_name.tsv", by_name, 3)
emit("dup_verbatim.tsv", by_body, 2)
emit("dup_shape.tsv", by_shape, 2)
print("files scanned:", len(files))
