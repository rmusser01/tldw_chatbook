"""Rebuild merged audit findings from the workflow journal (covers resumed runs).

usage: python3 extract_journal.py <journal.jsonl>  -> writes findings.json + items.md next to this script
"""
import json, sys, os, collections, re
OUT = os.path.dirname(os.path.abspath(__file__))
label = {}; results = {}
for l in open(sys.argv[1]):
    try: o = json.loads(l)
    except Exception: continue
    if o.get('type') == 'started': label[o['key']] = o.get('label', '')
    elif o.get('type') == 'result': results[o['key']] = o['result']
by_label = {label.get(k, k): r for k, r in results.items()}
def item_of_find(l):
    m = re.match(r'find:slice(\d+):', l)
    return f'slice-{m.group(1)}' if m else l.split(':', 1)[1]
finders = {item_of_find(l): r for l, r in by_label.items() if l.startswith('find:')}
verd, meas = {}, {}
for l, r in by_label.items():
    if l.startswith('verify:'):
        for v in (r or {}).get('verdicts', []): verd[v['id']] = v
    elif l.startswith('measure:'):
        for m in (r or {}).get('results', []): meas[m['id']] = m
rows, items = [], []
for item, fd in sorted(finders.items()):
    items.append({'item': item, 'summary': fd.get('summary', ''), 'clean': fd.get('clean_areas', []), 'census': fd.get('census', '')})
    for f in fd.get('findings', []):
        fid = f"{item}:{f['id']}"; v = verd.get(fid); m = meas.get(fid)
        if m: status = 'refuted' if m['verdict'] == 'refute' else 'confirmed'; sev = m['final_severity']
        elif v: status = v['verdict']; sev = v['final_severity']
        else: status = 'unverified'; sev = f['severity']
        rows.append({'id': fid, 'source': item, 'status': status, 'sev': sev, 'orig_sev': f['severity'],
            'cat': (v or {}).get('final_category') or f['category'], 'title': f['title'],
            'locs': [f"{x['path']}:{x['line']}" for x in f['locations']], 'corrected': (v or {}).get('corrected_locations', ''),
            'known_task': (v or {}).get('known_task') or f.get('known_task', ''), 'trigger': (v or {}).get('trigger_verified') or f['trigger'],
            'cost': (m or {}).get('evidence') or (v or {}).get('cost_verified') or f['cost'],
            'cost_basis': 'measured' if (m and m.get('method') == 'measured') else f['cost_basis'],
            'fix': f['fix'], 'fix_notes': (v or {}).get('fix_notes', ''), 'fix_risk': f['fix_risk'], 'evidence': f['evidence'],
            'reason': (v or {}).get('reason', ''), 'measure_verdict': (m or {}).get('verdict', ''), 'has_measure': bool(m)})
json.dump(rows, open(os.path.join(OUT, 'findings.json'), 'w'), indent=1)
with open(os.path.join(OUT, 'items.md'), 'w') as fh:
    for it in items:
        fh.write(f"\n\n# {it['item']}\n\n## summary\n{it['summary']}\n\n## clean areas\n" + '\n'.join('- ' + c for c in it['clean']) + f"\n\n## census\n{it['census']}\n")
c = collections.Counter((r['status'], r['sev']) for r in rows)
print('finders', len(finders), 'rows', len(rows), 'verify ids', len(verd), 'measure ids', len(meas))
for k in sorted(c): print(k, c[k])
hi = [r for r in rows if r['status'] == 'confirmed' and r['sev'] in ('P0', 'P1')]
print('confirmed P0/P1', len(hi), 'with measure pass', sum(r['has_measure'] for r in hi))
print('unverified by source', collections.Counter(r['source'] for r in rows if r['status'] == 'unverified').most_common())
print('measure verdicts', collections.Counter(r['measure_verdict'] for r in rows if r['measure_verdict']))
