"""Dump every definition of the seed cluster names (whole package) with source, grouped per name."""
import ast, pathlib, sys, collections
WT = pathlib.Path("/Users/macbook-dev/Documents/GitHub/tldw-review"); PKG = WT/"tldw_chatbook"; OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
NAMES = set("_maybe_await _utc_now _utc_now_iso _now _now_iso _now_ns _datetime_to_iso _iso_now _utcnow _timestamp _strict_json_loads _reject_json_constant _json_shape_is_safe _format_size _format_bytes _human_size _humanize_bytes _format_file_size _safe_filename _sanitize_filename _coerce_bool _coerce_int _safe_text _resolve_api_key _resolve_base_url _normalize_keywords _truncate _truncate_text _set_status _get_connection _identity _initialize_schema _dump _enforce_policy _normalize_mode _require_client _clean_text _json_safe _load_json _toast _cancel_safe _perform_safe_cancel _cancel _next_request_token _held_connection".split())
groups = collections.defaultdict(list)
for p in sorted(PKG.rglob("*.py")):
    if any(x in p.parts for x in (".venv","Third_Party","__pycache__","Splash_Screens")): continue
    try: src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in NAMES:
            seg = ast.get_source_segment(src, n) or ""
            groups[n.name].append((str(p.relative_to(WT)), n.lineno, seg))
for name, items in groups.items():
    with open(OUT/f"{name}.txt","w") as f:
        f.write(f"# {name}: {len(items)} definitions in {len({i[0] for i in items})} files\n")
        for path, ln, seg in items:
            f.write(f"\n### {path}:{ln}\n{seg}\n")
    print(f"{name:24s} {len(items):3d} defs {len({i[0] for i in items}):3d} files")
