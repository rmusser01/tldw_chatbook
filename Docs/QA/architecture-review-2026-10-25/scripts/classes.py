"""Class sizes, function complexity, app-attribute reach, DB attribute calls (AST only)."""
import ast, json, sys, re
from collections import Counter, defaultdict

g = json.load(open(sys.argv[1])); mods = g["mods"]
PRES_PREFIX = ("tldw_chatbook.UI", "tldw_chatbook.Widgets", "tldw_chatbook.Event_Handlers", "tldw_chatbook.app")
DBATTR = re.compile(r"(_db|_database|db)$")

classes = []; funcs = []
app_attr = Counter(); app_attr_mods = defaultdict(set)
dbcalls = Counter(); dbcall_mods = defaultdict(set)
app_reach_mods = Counter()  # module -> count of self.app.X / self.app_instance.X
cfg_calls = Counter(); cfg_mods = defaultdict(set)
globals_stmt = Counter()


def cc(fn):
    n = 1
    for x in ast.walk(fn):
        if isinstance(x, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler,
                          ast.With, ast.AsyncWith, ast.Assert, ast.comprehension, ast.match_case)):
            n += 1
        elif isinstance(x, ast.BoolOp):
            n += len(x.values) - 1
    return n


def pkg(m):
    p = m.split(".")
    return p[1] if len(p) > 1 else "<root>"


for m, p in mods.items():
    try:
        tree = ast.parse(open(p, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        continue
    pres = m.startswith(PRES_PREFIX)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            meths = [b for b in node.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append((node.end_lineno - node.lineno + 1, len(meths), m, node.name,
                            [ast.unparse(b) for b in node.bases][:4]))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((cc(node), node.end_lineno - node.lineno + 1, m, node.name, node.lineno))
        elif isinstance(node, ast.Global):
            globals_stmt[m] += 1
        elif isinstance(node, ast.Attribute):
            v = node.value
            # self.app.X  /  self.app_instance.X / app_instance.X
            if isinstance(v, ast.Attribute) and v.attr in ("app", "app_instance") and isinstance(v.value, ast.Name) and v.value.id == "self":
                app_attr[node.attr] += 1; app_attr_mods[node.attr].add(m); app_reach_mods[m] += 1
            elif isinstance(v, ast.Name) and v.id == "app_instance":
                app_attr[node.attr] += 1; app_attr_mods[node.attr].add(m); app_reach_mods[m] += 1
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in ("get_cli_setting", "load_settings", "load_cli_config_and_ensure_existence", "get_cli_providers_and_models", "save_setting_to_cli_config"):
                cfg_calls[name] += 1; cfg_mods[name].add(m)
            if pres and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and DBATTR.search(f.value.attr):
                dbcalls[f.value.attr] += 1; dbcall_mods[f.value.attr].add(m)

classes.sort(reverse=True); funcs.sort(reverse=True)
sec = sys.argv[2]
if sec == "classes":
    print("== largest classes (loc, methods)")
    for l, n, m, c, b in classes[:40]:
        print(f"  {l:6} {n:4}  {m.replace('tldw_chatbook.','')}.{c}  bases={b}")
    print("classes >2000 LOC:", sum(1 for c in classes if c[0] > 2000), " >1000:", sum(1 for c in classes if c[0] > 1000),
          " >500:", sum(1 for c in classes if c[0] > 500), " >100 methods:", sum(1 for c in classes if c[1] > 100))
if sec == "funcs":
    print("== most complex functions (approx CC, loc)")
    for c, l, m, n, line in funcs[:35]:
        print(f"  CC{c:5} {l:5}L  {m.replace('tldw_chatbook.','')}:{line} {n}")
    print("funcs CC>15:", sum(1 for f in funcs if f[0] > 15), " CC>50:", sum(1 for f in funcs if f[0] > 50),
          " CC>100:", sum(1 for f in funcs if f[0] > 100), " total funcs:", len(funcs))
if sec == "app":
    print("== self.app.X / app_instance.X attribute reach")
    print("modules reaching into app:", len(app_reach_mods), " total accesses:", sum(app_reach_mods.values()),
          " distinct attrs:", len(app_attr))
    bypkg = Counter()
    for m, n in app_reach_mods.items(): bypkg[pkg(m)] += n
    print("  by pkg:", bypkg.most_common(20))
    print("  top attrs (count, #modules):")
    for a, n in app_attr.most_common(45):
        print(f"    {n:5} {len(app_attr_mods[a]):4}  {a}")
    print("== config lookups:", {k: (v, len(cfg_mods[k])) for k, v in cfg_calls.items()})
    cp = Counter()
    for k in cfg_mods:
        for m in cfg_mods[k]: cp[pkg(m)] += 1
    print("  modules calling config getters by pkg:", cp.most_common(25), "packages:", len(cp))
    print("== presentation calls on *_db attributes:", sum(dbcalls.values()), "modules:", len(set().union(*dbcall_mods.values())) if dbcall_mods else 0)
    for a, n in dbcalls.most_common(15): print(f"    {n:5} {len(dbcall_mods[a]):4} {a}")
    print("== modules with `global` statements:", len(globals_stmt), " total:", sum(globals_stmt.values()))
    for m, n in globals_stmt.most_common(12): print("   ", n, m)
