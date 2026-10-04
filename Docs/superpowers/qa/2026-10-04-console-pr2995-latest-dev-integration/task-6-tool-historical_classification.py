exec(open('/private/tmp/pr2995-task6/prove_integration.py').read().split('a,b,c=')[0])
a,b,c=tree(old),tree(dev),tree('HEAD');owned=changed(base,old);rows=[]
for p in sorted(a):
 if not p.startswith(('qa/','Docs/superpowers/qa/')):continue
 rows.append({'path':p,'old':a[p],'dev':b.get(p),'head':c.get(p),'owned':p in owned,'old_equal':a[p]==c.get(p),'upstream_change_preserved':p not in owned and b.get(p)==c.get(p)})
assert all(x['old_equal'] if x['owned'] else x['old_equal'] or x['upstream_change_preserved'] for x in rows)
(O/'task-6-historical-artifacts.json').write_text(json.dumps(rows,indent=2)+'\n')
print({'historical':len(rows),'owned_exact':sum(x['owned'] for x in rows),'unrelated_upstream_changes':sum(not x['old_equal'] for x in rows)})
