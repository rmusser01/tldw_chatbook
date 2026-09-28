"""Cluster duplicate findings within each PR group; write issues.json (one row per unique issue)."""
import json, os, re, collections
D = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in json.load(open(os.path.join(D, 'findings.json'))) if r['status'] in ('confirmed', 'unverified', 'known')]
SEV = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3}
STOP = set('the a an of on in to and or per every each is are for with by at from its it that this as be not only still'.split())
def toks(t): return {w for w in re.findall(r'[a-z_][a-z0-9_]+', t.lower()) if w not in STOP and len(w) > 2}
def loc(r):
    l = r['locs'][0] if r['locs'] else ':0'
    p, _, n = l.rpartition(':')
    return p, int(n) if n.isdigit() else 0
parent = list(range(len(rows)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(a, b): parent[find(a)] = find(b)
T = [toks(r['title']) for r in rows]
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if rows[i]['pr'] != rows[j]['pr']: continue
        pi, li = loc(rows[i]); pj, lj = loc(rows[j])
        jac = len(T[i] & T[j]) / max(1, len(T[i] | T[j]))
        same_file = pi == pj
        if (same_file and abs(li - lj) <= 40 and jac >= 0.15) or (same_file and jac >= 0.35) or jac >= 0.55:
            union(i, j)
groups = collections.defaultdict(list)
for i, r in enumerate(rows): groups[find(i)].append(r)
issues = []
for g in groups.values():
    g.sort(key=lambda r: (SEV[r['sev']], r['status'] != 'confirmed', r['cost_basis'] != 'measured'))
    rep = g[0]
    issues.append({
        'pr': rep['pr'], 'sev': rep['sev'], 'status': 'confirmed' if any(x['status'] == 'confirmed' for x in g) else rep['status'],
        'title': rep['title'], 'loc': rep['locs'][0] if rep['locs'] else '', 'cat': rep['cat'],
        'known': sorted({x['known_task'].split(' ')[0].strip('(;,') for x in g if x['known_task'].strip() and x['known_task'].strip().lower() not in ('none', 'related:')})[:4],
        'measured': any(x['cost_basis'] == 'measured' for x in g), 'n': len(g), 'sources': sorted({x['source'] for x in g}),
        'cost': rep['cost'][:400], 'fix': rep['fix'][:500], 'fix_notes': rep['fix_notes'][:300], 'ids': [x['id'] for x in g],
    })
issues.sort(key=lambda x: (x['pr'], SEV[x['sev']], -x['n']))
json.dump(issues, open(os.path.join(D, 'issues.json'), 'w'), indent=1)
c = collections.defaultdict(collections.Counter)
for x in issues: c[x['pr']][x['sev']] += 1
print('findings', len(rows), '-> unique issues', len(issues))
print('by sev', collections.Counter(x['sev'] for x in issues))
for k in sorted(c): print(f"{k:24} {sum(c[k].values()):4}  " + ' '.join(f"{s}={c[k][s]}" for s in ('P0','P1','P2','P3')))
