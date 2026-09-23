"""For each cluster name: group definitions by exact body (docstring stripped) and by anonymized shape; print distinct variants."""
import ast, pathlib, sys, collections, copy, hashlib, textwrap
USAGE = "usage: variants.py <comma-separated-symbol-names> [repo-root]"
if len(sys.argv) not in (2, 3):
    sys.exit(USAGE)
# The repo root used to be hard-coded to one reviewer's worktree, so this script
# only ran on that machine and silently analysed a stale tree elsewhere. Default
# to the repo this file lives in; allow an explicit override.
WT = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else pathlib.Path(__file__).resolve().parents[3]
PKG = WT / "tldw_chatbook"
if not PKG.is_dir():
    sys.exit(f"{USAGE}\n  no tldw_chatbook/ under: {WT}")
names = [n for n in (x.strip() for x in sys.argv[1].split(",")) if n]
if not names:
    sys.exit(f"{USAGE}\n  no symbol names given")
class Anon(ast.NodeTransformer):
    def visit_Name(self, n): n.id="_"; return n
    def visit_arg(self, n): n.arg="_"; return n
    def visit_Attribute(self, n):
        self.generic_visit(n)
        if not n.attr.startswith("__"): n.attr="_"
        return n
    def visit_Constant(self, n): n.value=type(n.value).__name__; return n
def body_of(n):
    b = n.body
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str): b = b[1:]
    return b
defs = collections.defaultdict(list)
for p in sorted(PKG.rglob("*.py")):
    if any(x in p.parts for x in (".venv","Third_Party","__pycache__","Splash_Screens")): continue
    try: src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names:
            b = body_of(n)
            raw = hashlib.sha1(ast.dump(ast.Module(body=b, type_ignores=[])).encode()).hexdigest()[:8]
            shape = hashlib.sha1(ast.dump(Anon().visit(ast.Module(body=copy.deepcopy(b), type_ignores=[]))).encode()).hexdigest()[:8]
            defs[n.name].append((raw, shape, str(p.relative_to(PKG)), n.lineno, ast.get_source_segment(src, n) or "", isinstance(n, ast.AsyncFunctionDef)))
for name in names:
    items = defs.get(name, [])
    byraw = collections.defaultdict(list)
    for it in items: byraw[it[0]].append(it)
    print(f"\n{'='*100}\n# {name}: {len(items)} defs, {len(byraw)} distinct bodies, {len({i[1] for i in items})} distinct shapes")
    for raw, its in sorted(byraw.items(), key=lambda kv: -len(kv[1])):
        interop = sum(1 for i in its if "_Interop" in i[2]); core = len(its)-interop
        print(f"\n--- body {raw} (shape {its[0][1]}): {len(its)} copies (core {core}, interop {interop})")
        print("   files: " + ", ".join(f"{i[2]}:{i[3]}" for i in its[:8]) + (f" ... +{len(its)-8}" if len(its)>8 else ""))
        src = its[0][4]
        lines = src.splitlines()
        print(textwrap.indent("\n".join(lines[:28]) + ("\n   ..." if len(lines)>28 else ""), "   | "))
