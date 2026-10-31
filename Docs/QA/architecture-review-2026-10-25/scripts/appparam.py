import ast, json, sys
from collections import Counter
g = json.load(open(sys.argv[1])); mods = g["mods"]
PRES = ("UI", "Widgets", "css")
by = Counter(); fn = 0; files = set(); typed = 0; attrs = Counter()
for m, p in mods.items():
    seg = m.split(".")[1] if "." in m else ""
    if seg in PRES: continue
    try: t = ast.parse(open(p).read())
    except SyntaxError: continue
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in n.args.args + n.args.kwonlyargs:
                if a.arg in ("app", "app_instance", "tldw_app"):
                    fn += 1; files.add(m); by[seg] += 1
                    if a.annotation is not None and "TldwCli" in ast.unparse(a.annotation): typed += 1
                    for x in ast.walk(n):
                        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == a.arg:
                            attrs[x.attr] += 1
                        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in ("getattr", "hasattr") and x.args and isinstance(x.args[0], ast.Name) and x.args[0].id == a.arg and len(x.args) > 1 and isinstance(x.args[1], ast.Constant):
                            attrs["getattr:" + str(x.args[1].value)] += 1
                    break
print("non-UI functions taking app/app_instance:", fn, "in", len(files), "modules; annotated TldwCli:", typed)
print("by package:", by.most_common(20))
print("attributes read off that app param (top 40):")
for a, n in attrs.most_common(40): print(f"   {n:4} {a}")
print("distinct attrs:", len(attrs), " getattr-string lookups:", sum(v for k, v in attrs.items() if k.startswith('getattr:')))
