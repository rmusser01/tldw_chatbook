"""Phase 1b: count importing files for every public def/class in the shared-helper modules."""
import ast, sys, pathlib, collections, re
root = pathlib.Path(sys.argv[1])  # repo root
pkg = root / "tldw_chatbook"
HELPER_MODULES = sorted([p for p in (pkg/"Utils").glob("*.py") if p.name != "__init__.py"]) + [
    pkg/"DB/base_db.py", pkg/"DB/sql_validation.py", pkg/"Widgets/form_components.py", pkg/"Widgets/base_components.py"]
def modname(p): return ".".join(p.relative_to(root).with_suffix("").parts)
helpers = {}  # modname -> {symbol: kind}
for p in HELPER_MODULES:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    syms = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
            syms[n.name] = "class" if isinstance(n, ast.ClassDef) else "def"
    helpers[modname(p)] = syms
def resolve(cur_mod, node):
    if node.level == 0: return node.module or ""
    parts = cur_mod.split(".")[:-1]  # package of current module
    parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join(parts + ([node.module] if node.module else []))
mod_importers = collections.defaultdict(set)   # helper mod -> files
sym_importers = collections.defaultdict(set)   # (mod, sym) -> files
attr_re_cache = {}
files = [p for p in pkg.rglob("*.py") if not any(x in p.parts for x in (".venv","Third_Party","__pycache__","Splash_Screens"))]
for p in files:
    try: src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception: continue
    cur = modname(p)
    aliases = {}  # local name -> helper mod
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            m = resolve(cur, n)
            if m in helpers:
                mod_importers[m].add(str(p))
                for a in n.names:
                    if a.name == "*":
                        for s in helpers[m]: sym_importers[(m,s)].add(str(p))
                    elif a.name in helpers[m]: sym_importers[(m,a.name)].add(str(p))
            else:
                # from ..Utils import X  (module import)
                for a in n.names:
                    cand = f"{m}.{a.name}" if m else a.name
                    if cand in helpers:
                        mod_importers[cand].add(str(p)); aliases[a.asname or a.name] = cand
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in helpers:
                    mod_importers[a.name].add(str(p)); aliases[a.asname or a.name.split(".")[-1]] = a.name
    for local, m in aliases.items():
        for s in helpers[m]:
            if re.search(rf"\b{re.escape(local)}\.{re.escape(s)}\b", src): sym_importers[(m,s)].add(str(p))
out = pathlib.Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
rows = []
for m, syms in helpers.items():
    for s, kind in syms.items():
        rows.append((len(sym_importers[(m,s)]), len(mod_importers[m]), m, s, kind))
rows.sort()
with open(out, "w") as f:
    f.write("sym_importers\tmod_importers\tmodule\tsymbol\tkind\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
zero = [r for r in rows if r[0] == 0]
print("helpers:", len(rows), "zero-importer symbols:", len(zero))
mods = sorted(((len(mod_importers[m]), m, len(s)) for m, s in helpers.items()))
print("modules with 0 importers:", [m for c, m, _ in mods if c == 0])
print("modules with <=2 importers:", [(m, c) for c, m, _ in mods if 0 < c <= 2])
