import ast, sys
for path, cls in [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]:
    t = ast.parse(open(path).read())
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            meths = [b for b in n.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
            deleg = 0; deleg_loc = 0; big = []
            for m in meths:
                body = [s for s in m.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                if len(body) == 1:
                    s = body[0]; v = getattr(s, "value", None)
                    if isinstance(v, ast.Await): v = v.value
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute):
                        tgt = ast.unparse(v.func.value)
                        if tgt.startswith("self._") or "controller" in tgt:
                            deleg += 1; deleg_loc += m.end_lineno - m.lineno + 1; continue
                big.append((m.end_lineno - m.lineno + 1, m.name))
            props = sum(1 for m in meths if any(ast.unparse(d).endswith(("property", "setter")) for d in m.decorator_list))
            big.sort(reverse=True)
            print(f"{cls}: methods={len(meths)} one-line delegators={deleg} ({deleg_loc} lines) property/setters={props} class_loc={n.end_lineno-n.lineno+1}")
            print("   largest methods:", big[:8])
