"""Phase 1c: mechanical pattern candidates over the Tier-2 files. One TSV per pattern.

The scanned path list is READ FROM ``slice_paths.txt`` rather than hard-coded.
It used to be a literal list, which silently drifted out of date: the coverage
fix that added ``Backup_Recovery``, ``Workflows`` and ``UI/Workflows_Modules``
to ``slice_paths.txt`` never reached this file, so every candidate TSV it
produced was missing those packages while the report presented the tables as
covering the whole Tier-2 surface. Deriving the list makes that divergence
impossible rather than merely fixed once.
"""
import ast, re, sys, pathlib, collections

USAGE = "usage: pattern_greps.py <repo-root> <output-dir>"
if len(sys.argv) != 3:
    sys.exit(USAGE)
root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
if not (root / "tldw_chatbook").is_dir():
    sys.exit(f"{USAGE}\n  <repo-root> has no tldw_chatbook/: {root}")
out.mkdir(parents=True, exist_ok=True)
pkg = root / "tldw_chatbook"

SLICE_PATHS = root / "qa" / "tier2-code-review-2026-09-21" / "slice_paths.txt"
if not SLICE_PATHS.is_file():
    sys.exit(f"{USAGE}\n  missing the authoritative path list: {SLICE_PATHS}")


def _tier2_paths() -> list[str]:
    """Every path claimed by a Tier-2 slice, i.e. everything after the tier-1 block."""
    lines = SLICE_PATHS.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("# S01"))
    except StopIteration:
        sys.exit(f"{SLICE_PATHS} has no '# S01...' marker; cannot tell tier 1 from tier 2")
    paths = []
    for line in lines[start:]:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # slice_paths.txt is repo-relative and prefixed with the package name
        paths.append(line[len("tldw_chatbook/"):] if line.startswith("tldw_chatbook/") else line)
    return paths


T2 = _tier2_paths()
files = []
for t in T2:
    p = pkg / t
    if p.is_file(): files += [p]
    elif p.is_dir(): files += [f for f in p.rglob("*.py") if "Splash_Screens" not in f.parts]
files = sorted(set(files))
rows = collections.defaultdict(list)
def rel(p): return str(p.relative_to(root))
def enclosing_funcs(tree):
    """map node -> enclosing FunctionDef chain, plus Try-nesting and loop-nesting flags."""
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n): parents[c] = n
    return parents
def chain(parents, n, kinds):
    res = []; cur = parents.get(n)
    while cur is not None:
        if isinstance(cur, kinds): res.append(cur)
        cur = parents.get(cur)
    return res
def src_of(src, node):
    try: return ast.get_source_segment(src, node) or ""
    except Exception: return ""
strftime_formats = collections.Counter()
for p in files:
    try: src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception as e: rows["parse_errors"].append((rel(p), 0, str(e)[:80])); continue
    parents = enclosing_funcs(tree)
    r = rel(p)
    imports_optional_deps = "optional_deps" in src
    imports_atomic = "atomic_file_ops" in src
    imports_secure_tmp = "secure_temp_files" in src
    has_loguru = bool(re.search(r"^from loguru import logger", src, re.M))
    has_stdlog = bool(re.search(r"^import logging\b", src, re.M) or re.search(r"^from logging import", src, re.M))
    if has_loguru and has_stdlog: rows["loguru_and_logging"].append((r, 0, ""))
    has_lock = bool(re.search(r"threading\.(R)?Lock\(\)", src)); has_execute = ".execute(" in src
    if has_lock and has_execute: rows["lock_and_execute"].append((r, 0, f"locks={len(re.findall(r'threading[.](R)?Lock[(][)]', src))} executes={src.count('.execute(')}"))
    for m in re.finditer(r"mkdir\(\s*parents=True,\s*exist_ok=True\s*\)", src): rows["raw_mkdir"].append((r, src.count("\n",0,m.start())+1, ""))
    for m in re.finditer(r"\bos\.replace\(", src):
        if not imports_atomic: rows["os_replace_no_atomic"].append((r, src.count("\n",0,m.start())+1, ""))
    for m in re.finditer(r"\btempfile\.", src):
        if not imports_secure_tmp: rows["tempfile_no_secure"].append((r, src.count("\n",0,m.start())+1, ""))
    for m in re.finditer(r"\[:\s*\d+\s*\]\s*\+\s*[\"'](\.\.\.|…)", src): rows["inline_truncate"].append((r, src.count("\n",0,m.start())+1, m.group(0)))
    for m in re.finditer(r"1024\s*\*\s*1024", src): rows["raw_1024x1024"].append((r, src.count("\n",0,m.start())+1, ""))
    for m in re.finditer(r"len\([^()]*\)\s*(/|//)\s*4\b", src): rows["token_est_len_div4"].append((r, src.count("\n",0,m.start())+1, m.group(0)))
    for m in re.finditer(r"\b(is_relative_to|commonpath)\(", src):
        if "path_validation" not in src: rows["inline_path_check_no_pv"].append((r, src.count("\n",0,m.start())+1, m.group(1)))
    for m in re.finditer(r"\b(DEPRECATED|Deprecated|deprecated|legacy|Legacy|LEGACY|retired|Retired)\b", src): rows["legacy_markers"].append((r, src.count("\n",0,m.start())+1, m.group(1)))
    for m in re.finditer(r"get_cli_setting\(\s*[\"']([A-Za-z_]+\.[A-Za-z_.]+)[\"']", src): rows["dotted_section_setting"].append((r, src.count("\n",0,m.start())+1, m.group(1)))
    for m in re.finditer(r"\[\s*id\(", src): rows["id_keyed_dict"].append((r, src.count("\n",0,m.start())+1, ""))
    for m in re.finditer(r"\.label\.plain|str\(\s*[\w.]*\.label\s*\)|\.renderable\.plain|\.plain\b", src): rows["plain_readback"].append((r, src.count("\n",0,m.start())+1, m.group(0)))
    for m in re.finditer(r"strftime\(\s*[\"']([^\"']+)[\"']", src): strftime_formats[m.group(1)] += 1; rows["strftime"].append((r, src.count("\n",0,m.start())+1, m.group(1)))
    for m in re.finditer(r"\bsys\.path\.(insert|append)\(", src): rows["sys_path_mutation"].append((r, src.count("\n",0,m.start())+1, ""))
    for node in ast.walk(tree):
        # re.compile inside a def
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile" and isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
            fs = chain(parents, node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if fs:
                deco = [src_of(src, d) for d in fs[0].decorator_list]
                rows["re_compile_in_def"].append((r, node.lineno, f"{fs[0].name} deco={deco}"))
        # get_cli_setting inside compose / loops / retry
        if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "get_cli_setting") or (isinstance(node.func, ast.Attribute) and node.func.attr == "get_cli_setting")):
            fs = chain(parents, node, (ast.FunctionDef, ast.AsyncFunctionDef)); loops = chain(parents, node, (ast.For, ast.While, ast.AsyncFor))
            fname = fs[0].name if fs else "<module>"
            if fname == "compose" or loops or "retry" in fname.lower() or fname in ("render",):
                rows["get_cli_setting_hot"].append((r, node.lineno, f"{fname} loop={bool(loops)}"))
        # fetchall without LIMIT in the same function's preceding execute
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fetchall":
            fs = chain(parents, node, (ast.FunctionDef, ast.AsyncFunctionDef))
            sql = None; dyn = False
            if fs:
                for sub in ast.walk(fs[0]):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in ("execute","execute_query","execute_many") and sub.args and sub.lineno <= node.lineno:
                        a = sub.args[0]
                        if isinstance(a, ast.Constant) and isinstance(a.value, str): sql = a.value
                        elif isinstance(a, ast.JoinedStr): sql = "".join(v.value for v in a.values if isinstance(v, ast.Constant) and isinstance(v.value,str)); dyn = True
                        else: sql = None; dyn = True
            if sql is None: rows["fetchall_dynamic_sql"].append((r, node.lineno, fs[0].name if fs else "<module>"))
            elif "limit" not in sql.lower(): rows["fetchall_no_limit"].append((r, node.lineno, (fs[0].name if fs else "<module>") + (" (fstring)" if dyn else "") + " :: " + " ".join(sql.split())[:90]))
        # run_worker exclusive without group
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run_worker":
            kws = {k.arg for k in node.keywords}
            excl = any(k.arg == "exclusive" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords)
            if excl and "group" not in kws: rows["run_worker_exclusive_no_group"].append((r, node.lineno, ""))
            # coroutine passed (a Call node) -> not a thread
            if node.args and isinstance(node.args[0], ast.Call) and not any(k.arg=="thread" for k in node.keywords): rows["run_worker_coroutine"].append((r, node.lineno, src_of(src, node.args[0])[:60]))
        # except Exception: pass
        if isinstance(node, ast.ExceptHandler):
            t = node.type
            broad = t is None or (isinstance(t, ast.Name) and t.id in ("Exception","BaseException"))
            if broad and len(node.body) == 1 and isinstance(node.body[0], (ast.Pass,)) : rows["except_exception_pass"].append((r, node.lineno, ""))
            if broad and len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value is Ellipsis: rows["except_exception_pass"].append((r, node.lineno, "..."))
            if broad and len(node.body) == 1 and isinstance(node.body[0], ast.Return): rows["except_exception_return"].append((r, node.lineno, src_of(src, node.body[0])[:40]))
        # try: import guards without optional_deps
        if isinstance(node, ast.Try) and any(isinstance(b, (ast.Import, ast.ImportFrom)) for b in node.body):
            hs = [src_of(src, h.type) if h.type else "<bare>" for h in node.handlers]
            if any(x in " ".join(hs) for x in ("ImportError","ModuleNotFoundError","Exception","<bare>")):
                fs = chain(parents, node, (ast.FunctionDef, ast.AsyncFunctionDef))
                rows["try_import_guard"].append((r, node.lineno, f"{'in '+fs[0].name if fs else 'module'} handlers={hs} optional_deps_imported={imports_optional_deps}"))
        # function-body imports (non-guarded)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            fs = chain(parents, node, (ast.FunctionDef, ast.AsyncFunctionDef)); trys = chain(parents, node, (ast.Try,))
            if fs and not trys:
                names = [a.name for a in node.names]; mod = getattr(node, "module", None) or ",".join(names)
                rows["function_body_import"].append((r, node.lineno, f"{fs[0].name} <- {'.'*getattr(node,'level',0)}{mod}"))
        # mutable class attributes
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                tgt = None; val = None
                if isinstance(b, ast.Assign) and len(b.targets)==1 and isinstance(b.targets[0], ast.Name): tgt, val = b.targets[0].id, b.value
                elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) and b.value is not None: tgt, val = b.target.id, b.value
                if tgt is None or tgt.isupper(): continue
                mut = isinstance(val, (ast.List, ast.Dict, ast.Set)) or (isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in ("list","dict","set","defaultdict","deque","OrderedDict"))
                if mut:
                    bases = [src_of(src, x) for x in node.bases]
                    rows["mutable_class_attr"].append((r, b.lineno, f"{node.name}.{tgt} bases={bases}"))
        # query_one inside timer callbacks without try
        if isinstance(node, ast.ClassDef):
            methods = {m.name: m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in ("set_interval","set_timer") and sub.args:
                    cb = sub.args[1] if len(sub.args) > 1 else None
                    if cb is None:
                        for k in sub.keywords:
                            if k.arg == "callback": cb = k.value
                    name = None
                    if isinstance(cb, ast.Attribute) and isinstance(cb.value, ast.Name) and cb.value.id == "self": name = cb.attr
                    if name and name in methods:
                        m = methods[name]; mp = {}
                        for n2 in ast.walk(m):
                            for c in ast.iter_child_nodes(n2): mp[c] = n2
                        for q in ast.walk(m):
                            if isinstance(q, ast.Call) and isinstance(q.func, ast.Attribute) and q.func.attr in ("query_one","query_exactly_one"):
                                intry = False; cur = mp.get(q)
                                while cur is not None and cur is not m:
                                    if isinstance(cur, ast.Try): intry = True; break
                                    cur = mp.get(cur)
                                if not intry: rows["query_one_in_timer_no_try"].append((r, q.lineno, f"{node.name}.{name} (timer at {sub.lineno})"))
# emit
summary = []
for k, v in sorted(rows.items()):
    with open(out / f"{k}.tsv", "w") as f:
        for a, b, c in sorted(v): f.write(f"{a}\t{b}\t{c}\n")
    summary.append((k, len(v), len({a for a,_,_ in v})))
with open(out / "strftime_formats.tsv", "w") as f:
    for fmt, n in strftime_formats.most_common(): f.write(f"{n}\t{fmt}\n")
print("files scanned:", len(files))
for k, n, nf in summary: print(f"{k:35s} {n:6d} rows in {nf:4d} files")
print("distinct strftime formats:", len(strftime_formats))
