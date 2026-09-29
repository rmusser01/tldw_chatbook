"""Step 3 of 4 (see extract_journal.py): cluster duplicate findings per PR group into issues.json.

issues.json is generated output and is not committed; re-run this script to recreate it.
"""
import json, os, re, collections, sys
D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(D, 'findings.json'), encoding='utf-8') as fh:
    rows = [r for r in json.load(fh) if r['status'] in ('confirmed', 'unverified', 'known')]
if rows and 'pr' not in rows[0]:
    sys.exit("findings.json has no PR groups yet: run classify.py first")
# Two findings are the same issue when they share a file and sit within
# NEAR_LINES lines with some title overlap, share a file with strong overlap,
# or have near-identical titles anywhere (token Jaccard similarity).
NEAR_LINES = 40
NEAR_JACCARD = 0.15
SAME_FILE_JACCARD = 0.35
ANY_FILE_JACCARD = 0.55
SEV = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3}
STOP = set('the a an of on in to and or per every each is are for with by at from its it that this as be not only still'.split())
def toks(t: str) -> set[str]:
    """Return a title's significant lowercase word tokens.

    Args:
        t: Finding title.

    Returns:
        Tokens of three or more characters, minus stop words.
    """
    return {w for w in re.findall(r'[a-z_][a-z0-9_]+', t.lower()) if w not in STOP and len(w) > 2}


def loc(r: dict) -> tuple[str, int]:
    """Return a finding's first location as ``(path, line)``.

    Args:
        r: Finding record with a ``locs`` list of ``"path:line"`` strings.

    Returns:
        The path and line number; ``("", 0)`` when the finding has no location.
    """
    l = r['locs'][0] if r['locs'] else ':0'
    p, _, n = l.rpartition(':')
    return p, int(n) if n.isdigit() else 0


parent = list(range(len(rows)))


def find(i: int) -> int:
    """Return the union-find root of row ``i``, compressing the path.

    Args:
        i: Row index.

    Returns:
        Index of the representative row of ``i``'s cluster.
    """
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i


def union(a: int, b: int) -> None:
    """Merge the clusters that contain rows ``a`` and ``b``.

    Args:
        a: Row index.
        b: Row index.
    """
    parent[find(a)] = find(b)


T = [toks(r['title']) for r in rows]
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if rows[i]['pr'] != rows[j]['pr']: continue
        pi, li = loc(rows[i]); pj, lj = loc(rows[j])
        jac = len(T[i] & T[j]) / max(1, len(T[i] | T[j]))
        same_file = pi == pj
        if (same_file and abs(li - lj) <= NEAR_LINES and jac >= NEAR_JACCARD) or (same_file and jac >= SAME_FILE_JACCARD) or jac >= ANY_FILE_JACCARD:
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
with open(os.path.join(D, 'issues.json'), 'w', encoding='utf-8') as fh:
    json.dump(issues, fh, indent=1)
c = collections.defaultdict(collections.Counter)
for x in issues: c[x['pr']][x['sev']] += 1
print('findings', len(rows), '-> unique issues', len(issues))
print('by sev', collections.Counter(x['sev'] for x in issues))
for k in sorted(c): print(f"{k:24} {sum(c[k].values()):4}  " + ' '.join(f"{s}={c[k][s]}" for s in ('P0','P1','P2','P3')))
