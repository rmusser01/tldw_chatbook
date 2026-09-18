"""Build per-slice candidate excerpts + partitions.json from the Phase-1 TSVs."""
import json, pathlib, collections, sys
SP = pathlib.Path(sys.argv[1]); WT = pathlib.Path("/Users/macbook-dev/Documents/GitHub/tldw-review"); PKG = WT/"tldw_chatbook"
def lines(p): return sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
def files_under(*dirs, exclude=()):
    out=[]
    for d in dirs:
        p = PKG/d
        out += [p] if p.is_file() else [f for f in sorted(p.rglob("*.py")) if not any(e in f.parts for e in exclude)]
    return out
chat_all = [f for f in files_under("Chat")]
chat_big = {"console_chat_controller.py","console_chat_store.py","console_agent_bridge.py","console_provider_gateway.py","console_trace_service.py","console_runtime.py"}
chat_rest = [f for f in chat_all if f.name not in chat_big]
# split chat_rest into 3 ~equal buckets by lines (sorted by name)
def buckets(fs, n):
    total = sum(lines(f) for f in fs); tgt = total/n; out=[[]]; acc=0
    for f in fs:
        if acc >= tgt and len(out) < n: out.append([]); acc=0
        out[-1].append(f); acc += lines(f)
    return out
cr = buckets(chat_rest, 3)
wc_all = files_under("Widgets/Console"); wc = buckets(wc_all, 2)
db_all = files_under("DB"); db_rest = [f for f in db_all if f.name not in {"ChaChaNotes_DB.py","Client_Media_DB_v2.py","base_db.py","sql_validation.py"}]
parts = {
 "ENTRY-app": files_under("app.py"),
 "ENTRY-config": files_under("config.py","Constants.py","model_capabilities.py","Logging_Config.py","emergency_stop.py","runtime_policy"),
 "CHAT-controller": files_under("Chat/console_chat_controller.py"),
 "CHAT-store": files_under("Chat/console_chat_store.py"),
 "CHAT-bridge": files_under("Chat/console_agent_bridge.py","Chat/console_provider_gateway.py","Chat/console_trace_service.py","Chat/console_runtime.py"),
 "CHAT-rest-1": cr[0], "CHAT-rest-2": cr[1], "CHAT-rest-3": cr[2],
 "AGENTS": files_under("Agents"),
 "TOOLS-MCP": files_under("Tools","MCP"),
 "DB-chacha": files_under("DB/ChaChaNotes_DB.py"),
 "DB-media-base": files_under("DB/Client_Media_DB_v2.py","DB/base_db.py","DB/sql_validation.py"),
 "DB-rest": db_rest,
 "LLM": files_under("LLM_Calls"),
 "EVENTS": files_under("Event_Handlers"),
 "UTILS": files_under("Utils", exclude=("Splash_Screens",)),
 "RAG": files_under("RAG_Search"),
 "UI-library": files_under("UI/Screens/library_screen.py"),
 "UI-settings": files_under("UI/Screens/settings_screen.py"),
 "UI-chat": files_under("UI/Screens/chat_screen.py"),
 "UI-personas": files_under("UI/Screens/personas_screen.py"),
 "UIM-console": files_under("UI/Console_Modules"),
 "UIM-library": files_under("UI/Library_Modules"),
 "UIM-nav-mcp-persona": files_under("UI/Navigation","UI/MCP_Modules","UI/Persona_Modules"),
 "W-top": [f for f in sorted((PKG/"Widgets").glob("*.py"))],
 "W-console-1": wc[0], "W-console-2": wc[1],
 "W-library": files_under("Widgets/Library"),
 "W-persona-settings-chat": files_under("Widgets/Persona_Widgets","Widgets/Settings_Widgets","Widgets/Chat_Widgets"),
}
rel = lambda f: str(f.relative_to(WT))
file2part = {}
for k, fs in parts.items():
    for f in fs: file2part[rel(f)] = k
json.dump({k: [rel(f) for f in fs] for k, fs in parts.items()}, open(SP/"partitions.json","w"), indent=1)
# gather candidate rows
pat_dir = SP/"candidates/patterns"; tier = SP/"candidates/tier1"
per = collections.defaultdict(lambda: collections.defaultdict(list))
for tsv in sorted(pat_dir.glob("*.tsv")):
    if tsv.name in ("strftime_formats.tsv","legacy_markers.tsv","function_body_import.tsv","except_exception_return.tsv","run_worker_coroutine.tsv"): continue  # too noisy; summarized separately
    for l in open(tsv):
        f, ln, extra = l.rstrip("\n").split("\t")
        p = file2part.get(f)
        if p: per[p][tsv.stem].append(f"{f}:{ln} {extra}".strip())
# noisy ones: per-file counts only
for tsv in ("function_body_import","except_exception_return","run_worker_coroutine","legacy_markers"):
    cnt = collections.Counter()
    for l in open(pat_dir/f"{tsv}.tsv"): cnt[l.split("\t")[0]] += 1
    for f, n in cnt.items():
        p = file2part.get(f)
        if p: per[p][tsv+"_per_file"].append(f"{f}: {n}")
for name, minf in (("dup_verbatim", 2), ("dup_shape", 3)):
    for l in open(tier/f"{name}.tsv"):
        nf, h, copies = l.rstrip("\n").split("\t"); copies = copies.split(" ")
        if int(nf) < minf: continue
        touched = set()
        for c in copies:
            f = "tldw_chatbook/" + c.split("@")[-1].rsplit(":",1)[0]
            p = file2part.get(f)
            if p: touched.add(p)
        for p in touched: per[p][name].append(f"[{nf} files, {len(copies)} copies] " + " ".join(copies[:12]) + (" ..." if len(copies)>12 else ""))
SEED = "_maybe_await _enforce_policy _enforce _cancel _cancel_safe _perform_safe_cancel _dump _utc_now _utc_now_iso _set_status _get_connection _normalize_mode _require_client _identity _initialize_schema _now _now_iso _now_ns _truncate _truncate_text _reject_json_constant _strict_json_loads _json_shape_is_safe _coerce_bool _coerce_int _safe_text _resolve_api_key _resolve_base_url _format_size _format_bytes _normalize_keywords _json_safe _load_json _safe_filename _clean_text _datetime_to_iso _next_request_token _toast _notify _held_connection _mapping_value".split()
for l in open(tier/"dup_by_name.tsv"):
    nf, name, locs = l.rstrip("\n").split("\t")
    if name not in SEED: continue
    for loc in locs.split(" "):
        f = "tldw_chatbook/" + loc.rsplit(":",1)[0]; p = file2part.get(f)
        if p: per[p]["seed_name_"+name].append(f"{f}:{loc.rsplit(':',1)[1]}")
(SP/"excerpts").mkdir(exist_ok=True)
for k, fs in parts.items():
    total = sum(lines(f) for f in fs)
    with open(SP/"excerpts"/f"{k}.md","w") as o:
        o.write(f"# Candidate excerpt — {k} ({len(fs)} files, {total} lines)\n\n## Files in this slice\n")
        for f in fs: o.write(f"- {rel(f)} ({lines(f)})\n")
        o.write("\n## Mechanical candidates (hints — read the code)\n")
        for cat, rows_ in sorted(per[k].items()):
            o.write(f"\n### {cat} ({len(rows_)})\n")
            for r in rows_[:200]: o.write(f"- {r}\n")
            if len(rows_) > 200: o.write(f"- ... {len(rows_)-200} more in candidates/patterns/{cat}.tsv\n")
    print(f"{k:26s} {len(fs):4d} files {total:7d} lines  candidate-cats={len(per[k])}")
