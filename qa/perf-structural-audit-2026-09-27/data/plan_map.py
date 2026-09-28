"""Map classifier groups -> numbered PRs, write appendix.md + pr_stats.json."""
import json, os, re, collections
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D)
iss = json.load(open(os.path.join(D, 'issues.json')))
PRS = {
 'PERF-01': ('Wave 0', 'Perf guards that can see admissions, helper spawns and pre-import payload'),
 'PERF-02': ('Wave 0', 'Conversation search: undo the correlated FTS EXISTS (TASK-278 regression)'),
 'PERF-03': ('Wave 0', 'Logging pipeline: level gate, single-pass redaction, off-loop handlers'),
 'PERF-04': ('Wave 0', 'ChaChaNotes trace callback: stop SQLite expanding BLOB parameters'),
 'PERF-05': ('Wave 0', 'Screen leaks: Settings signal, Personas worker pin, Home re-compose pin'),
 'PERF-06': ('Wave 1', 'Config warm-hit fast paths + Console derivation scopes (extends TASK-32804.1)'),
 'PERF-07': ('Wave 1', 'Memoize get_user_data_dir / DB paths / sensitive-path context per config generation'),
 'PERF-08': ('Wave 1', 'Amortize Backup_Recovery storage admission (ADR-126 amendment)'),
 'PERF-09': ('Wave 1', 'Private-SQLite connection lifecycle: stop close-per-call, reuse per-thread handles'),
 'PERF-10': ('Wave 1', 'Legacy trace maintenance: park when complete, incremental GC (TASK-31501)'),
 'PERF-11': ('Wave 1', 'GC policy: gc.freeze after ready + thresholds (TASK-31966, needs ADR)'),
 'PERF-12': ('Wave 2', 'Console send path: one off-loop turn snapshot'),
 'PERF-13': ('Wave 2', 'Console idle, tick and streaming render'),
 'PERF-14': ('Wave 2', 'Console typing, first paint and resume'),
 'PERF-15': ('Wave 2', 'Agent runtime: persistent worker, pooled client, fewer hydrations'),
 'PERF-16': ('Wave 3', 'TldwCli.__init__ diet: defer feature services/DBs, drop dead boot work'),
 'PERF-17': ('Wave 3', 'Boot import diet + pre-import ratchet paydown'),
 'PERF-18': ('Wave 3', 'Server-mode TLDWAPIClient off the loop'),
 'PERF-19': ('Wave 3', 'Boot CSS paydown (bytes + bare-type ratchets)'),
 'PERF-20': ('Wave 4', 'HTTP client reuse: pooled sessions, cached SSLContext, nothing on the loop'),
 'PERF-21': ('Wave 5', 'Library screen: targeted updates, ingest O(N^2), Folder Files poll'),
 'PERF-22': ('Wave 5', 'Settings + Personas: reusable routes, targeted category/list updates'),
 'PERF-23': ('Wave 5', 'MCP workbench: split _sync_children, debounce, cache store reads'),
 'PERF-24': ('Wave 5', 'Watchlists + Schedules: incremental panes, DB off the loop'),
 'PERF-25': ('Wave 5', 'Other screens: Home, Evals, Change Review, Speech, video, splash'),
 'PERF-26': ('Wave 6', 'Terminal: dirty-line projection, event-driven polls (TASK-31503)'),
 'PERF-27': ('Wave 6', 'Notes sync + Personal Context data paths'),
 'PERF-28': ('Wave 6', 'DB query hygiene: N+1s, unbounded scans, planner traps'),
 'PERF-29': ('Wave 6', 'Memory growth + data-scaled algorithms'),
 'PERF-30': ('Wave 6', 'Cold-feature and hygiene sweep (P2/P3)'),
}
def pr_of(x):
    g, t = x['pr'], x['title']
    if g == 'W1-gc-leaks': return 'PERF-11' if re.search(r'\bgc\b|gen.?2|GC|garbage', t) else 'PERF-05'
    if g == 'S7-other-screens' and 'home_screen' in x['loc']: return 'PERF-05'
    if g == 'C2-console-idle-tick':
        return 'PERF-14' if re.search(r'(keystroke|typing|composer|first (console )?paint|mount|resume|post-_?ui_ready|launch|canvas machinery|character search|class-dance)', t, re.I) else 'PERF-13'
    if g == 'B2-import-diet' and re.search(r'(\bcss\b|stylesheet|tcss|bare-type)', t, re.I): return 'PERF-19'
    return {'Q1-perf-guards':'PERF-01','Q2-conv-search':'PERF-02','Q4-logging':'PERF-03','Q3-trace-callback':'PERF-04',
      'K1-config-fastpath':'PERF-06','K2-user-data-dir':'PERF-07','K3-admission-core':'PERF-08','K4-conn-churn':'PERF-09',
      'K5-trace-maintenance':'PERF-10','C1-send-path':'PERF-12','C3-agent-runtime':'PERF-15','B1-boot-init':'PERF-16',
      'Z-structural':'PERF-16','B2-import-diet':'PERF-17','B3-server-client':'PERF-18','N1-network':'PERF-20',
      'S1-library':'PERF-21','S2-settings':'PERF-22','S3-personas':'PERF-22','S4-mcp':'PERF-23','S5-watchlists':'PERF-24',
      'S6-schedules':'PERF-24','S7-other-screens':'PERF-25','F1-terminal':'PERF-26','F2-notes':'PERF-27',
      'P1-personal-context':'PERF-27','F3-db-queries':'PERF-28','F4-memory':'PERF-29','F5-feature-algorithms':'PERF-29',
      'F6-cold-features':'PERF-30','Z-misc':'PERF-30'}[g]
for x in iss: x['perf'] = pr_of(x)
json.dump(iss, open(os.path.join(D, 'issues.json'), 'w'), indent=1)
stats = {}
known = collections.defaultdict(set)
for k in PRS:
    xs = [x for x in iss if x['perf'] == k]
    c = collections.Counter(x['sev'] for x in xs)
    stats[k] = {'n': len(xs), **{s: c[s] for s in ('P0','P1','P2','P3')}, 'findings': sum(x['n'] for x in xs),
                'unverified': sum(1 for x in xs if x['status'] == 'unverified')}
    for x in xs:
        for t in x['known']:
            if re.match(r'TASK-\d', t): known[t].add(k)
json.dump({'stats': stats, 'known': {t: sorted(v) for t, v in known.items()}}, open(os.path.join(D, 'pr_stats.json'), 'w'), indent=1)
esc = lambda s: s.replace('|', '\\|').replace('\n', ' ')
with open(os.path.join(ROOT, 'appendix-issues-by-pr.md'), 'w') as fh:
    fh.write('# Appendix: every unique issue, grouped by PR\n\n'
             'Generated by `data/plan_map.py` from `data/issues.json` (905 unique issues deduplicated from 1,021 live findings). '
             'Columns: severity after adversarial verification; status (`unverified` = the verifier pass did not complete, see report §7); '
             'first location (repo-relative); open tasks the finding overlaps; `n` = independent agents that reported it; `m` = cost was measured by at least one agent.\n')
    for k, (wave, name) in PRS.items():
        s = stats[k]
        fh.write(f"\n## {k} — {name}\n\n{wave} · {s['n']} issues (P0 {s['P0']}, P1 {s['P1']}, P2 {s['P2']}, P3 {s['P3']}) from {s['findings']} findings"
                 + (f" · {s['unverified']} unverified" if s['unverified'] else '') + "\n\n| sev | status | location | issue | tasks | n | m |\n|---|---|---|---|---|---|---|\n")
        for x in sorted([x for x in iss if x['perf'] == k], key=lambda x: ({'P0':0,'P1':1,'P2':2,'P3':3}[x['sev']], -x['n'])):
            fh.write(f"| {x['sev']} | {x['status']} | `{esc(x['loc'].replace('tldw_chatbook/',''))}` | {esc(x['title'])} | {esc(', '.join(x['known']))} | {x['n']} | {'✓' if x['measured'] else ''} |\n")
for k, (w, n) in PRS.items():
    s = stats[k]; print(f"{k} {w} n={s['n']:3} P0={s['P0']} P1={s['P1']:2} P2={s['P2']:2} P3={s['P3']:2} unv={s['unverified']:2}  {n[:70]}")
print('known tasks touched:', len(known))
for t in sorted(known, key=lambda t: [int(p) for p in re.findall(r'\d+', t)]): print(' ', t, sorted(known[t]))
