"""Assign every audit finding to a PR group (first matching rule wins)."""
import json, re, os, collections, sys
D = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(D, 'findings.json')))
# (pr_key, predicate over (title_lower, path, cat, known))
R = []
def rule(key, title=None, path=None, cat=None, known=None):
    R.append((key, re.compile(title, re.I) if title else None, re.compile(path) if path else None,
              set(cat.split(',')) if cat else None, re.compile(known, re.I) if known else None))

# --- precise overrides (before everything)
rule('K5-trace-maintenance', title=r'(legacy trace maintenance|trace maintenance|trace GC|trace-maintenance|re-marks the entire)')
rule('C1-send-path', title=r'skill(s)?.{0,40}(trust|fingerprint|get_context)|per-skill')
rule('C2-console-idle-tick', title=r'character (context )?search|video card spec')
rule('S1-library', title=r'(ingest|folder.import|folder submit).{0,80}(registry|queue|O\(N)|Search/RAG (query|full panel)|Library RAG')
rule('P1-personal-context', path=r'(Personal_Context/|app_destinations\.py:9[0-9]{2})')
rule('B3-server-client', path=r'(tldw_api/|runtime_policy/)')
rule('C3-agent-runtime', title=r'agent tool call runs on a new bare thread')
# --- quick wins / precise items first
rule('Q2-conv-search', title=r'(search_conversations|conversation (text |content )?search|messages_fts MATCH|correlated (FTS )?EXISTS)', path=r'DB/')
rule('Q3-trace-callback', title=r'trace.?callback|set_trace_callback|expands every bound|render every bound')
rule('Q1-perf-guards', title=r'(ratchet|census cannot|guard (is|went)|perf-guard|mount.?profile|keystroke census|pre-import payload|blinded|stale meta)')
rule('Q4-logging', cat='logging')
rule('Q4-logging', title=r'\blog(ging|uru|ger)?\b.*(sink|redact|format|volume|INFO|pipeline)|redact(ed|ion) three')
rule('W1-gc-leaks', title=r'\bgc\b|gen.?2|garbage|gc\.freeze|leaks? (its|the|each|every)|pinned by|pins departed|un-unsubscribed|theme_changed_signal|active_worker')
# --- keystone: admission
rule('K5-trace-maintenance', title=r'(legacy trace maintenance|trace maintenance|trace GC|trace-maintenance|re-marks the entire)')
rule('K1-config-fastpath', title=r'(load_settings|get_runtime_config_snapshot|runtime config accessors|guarded config helpers|warm config|config (read|admission)|derivation scope|get_model_cache_dir)')
rule('K2-user-data-dir', title=r'(get_user_data_dir|db.path accessor|get_\*_db_path|DB-path|sensitive.?path|resolve_sensitive_context|emergency.?stop)')
rule('K3-admission-core', title=r'(storage.?admission|admission handshake|recovery.?admission|acquire_storage|admission (scope|storm|probe|path walk)|content_call|@_chat_sources\.guarded|guarded (MCP|persona)|re-walks directories|pinned.?director|backup.?maintenance|native pause|execution_allowed|provider.?guard|recovery guard|witness check|per-transaction (backup )?storage|pays? .{0,40}admission)')
rule('K3-admission-core', path=r'Backup_Recovery/')
# --- connection churn
rule('K4-conn-churn', title=r'(helper (sub)?process|helper spawn|private.?sqlite|connect_private|close[sd]? (the )?(connection|handle)|close.per.call|reopens?|re-opens?|fresh (hardened |thread-local |private-SQLite |SQLite )?connection|new connection|connection per (op|call)|per-operation connection|owned.?(db.?call|connection)|run_owned|finite.?worker|list_and_close|wal_checkpoint|db_offload|run_db_off_loop|hardened connects?|connects? per)')
# --- console
rule('C3-agent-runtime', path=r'(Agents/|Chat/console_agent_bridge|Chat/console_trace|Tools/workspace_tool|Tools/local_tool|MCP/unified_control_plane|Chat/console_trace_redaction)')
rule('C1-send-path', title=r'(per.?send|every send|each send|on (every )?send|first send|send path|submit snapshot|turn.?(configuration|snapshot|terminal)|skill(s)? (trust|context)|world.?(info|book)|lorebook|credential sanitizer|sanitiz|mcp catalog|compose_catalog|terminal persist|trace reservation|provenance)')
rule('C2-console-idle-tick', title=r'(poll|tick|4 Hz|5 Hz|2 Hz|idle|per.?frame|streaming tick|sync pass|class sync|relayout|restyle|resume|derivation storm|post-_?ui_ready|first (console )?paint|launch.?wake|keystroke|typing|debounce|composer|character search|cost chip|tray)', path=r'(UI/Screens/chat_screen|UI/Console_Modules|Widgets/Console|Chat/)')
rule('C1-send-path', path=r'(Chat/console_chat_controller|Chat/console_provider_gateway|Character_Chat/)')
rule('C2-console-idle-tick', path=r'(UI/Screens/chat_screen|UI/Console_Modules|Widgets/Console|Chat/)')
# --- boot
rule('B1-boot-init', title=r'(TldwCli\.__init__|before first paint|pre-?paint|first.paint|at boot|boot (path|runs|imports|critical)|eager(ly)? (built|build|construct|wire|wiring)|wiring|constructed eagerly|_ui_ready)', path=r'(app\.py|app_entry|app_destinations)')
rule('B2-import-diet', cat='startup-import')
rule('B2-import-diet', title=r'(import(s|ed)? .{0,60}(at boot|module scope|eager|on the loop|first paint|app import)|pydantic|themes?\.py|AA (text|hue)|hue pinning|splash imports|\bcss\b|stylesheet bundle)')
rule('B1-boot-init', path=r'(^|/)app\.py$')
rule('B3-server-client', path=r'(tldw_api/|runtime_policy/)')
# --- network
rule('N1-network', cat='network')
rule('N1-network', title=r'(httpx|requests\.Session|SSLContext|ssl context|TLS|connection pool|AsyncClient|webbrowser|open_url|chat_api_call)')
# --- screens
rule('S1-library', path=r'(UI/Screens/library_screen|UI/Library_Modules|Widgets/Library|Library/|Workspaces/display_state)')
rule('S2-settings', path=r'(UI/Screens/settings_screen|Widgets/Settings_Widgets|UI/Wizards)')
rule('S3-personas', path=r'(UI/Screens/personas_screen|Widgets/Persona_Widgets|UI/CCP_Modules|UI/Persona_Modules)')
rule('S4-mcp', path=r'(UI/MCP_Modules|MCP/)')
rule('S5-watchlists', path=r'(Watchlists|watchlists|Subscriptions/)')
rule('S6-schedules', path=r'(Scheduling/|UI/Screens/scheduling)')
rule('S7-other-screens', path=r'(UI/|Widgets/)')
# --- feature scoped
rule('F1-terminal', path=r'(Terminal/|Console_Modules/terminal)')
rule('F2-notes', path=r'(Notes/|Sync_Interop/notes)')
rule('F3-db-queries', cat='db-query')
rule('F4-memory', cat='memory')
rule('F5-feature-algorithms', cat='algorithmic')
rule('F6-cold-features', path=r'(TTS/|Audio/|STT/|Evals/|Chunking/|Local_Ingestion/|Chatbooks/|Image_Generation|Video_Generation|Persona_Visual|Research|Web_Scraping|Actor_Packs|Tool_Packs|Model_Artifacts|_Interop/|Media|Meetings|Utils/Splash|Personal_Context|Workspaces/|Canvas/|RAG_Search/|LLM_Calls/|LLM_Provider_Catalog|Kanban|Workflows|Third_Party|Petdex|Persona_Buddy|Prompt_Management|DB/|Utils/|Sync_Interop|Skills_Interop)')
rule('Z-structural', cat='structural')
rule('Z-misc', title=r'.')

def classify(r):
    t = r['title']; p = r['locs'][0] if r['locs'] else ''; c = r['cat']; k = r['known_task']
    for key, tr, pr, cr, kr in R:
        if tr and not tr.search(t): continue
        if pr and not pr.search(p): continue
        if cr and c not in cr: continue
        if kr and not kr.search(k): continue
        return key
    return 'Z-misc'
for r in rows: r['pr'] = classify(r)
json.dump(rows, open(os.path.join(D, 'findings.json'), 'w'), indent=1)
live = [r for r in rows if r['status'] in ('confirmed', 'unverified', 'known')]
cnt = collections.defaultdict(collections.Counter)
for r in live: cnt[r['pr']][r['sev']] += 1
for k in sorted(cnt): print(f"{k:24} total={sum(cnt[k].values()):4}  " + ' '.join(f"{s}={cnt[k][s]}" for s in ('P0','P1','P2','P3')))
if len(sys.argv) > 1:
    for r in live:
        if r['pr'] == sys.argv[1]: print(r['sev'], r['cat'], r['locs'][0] if r['locs'] else '', '|', r['title'][:120])
