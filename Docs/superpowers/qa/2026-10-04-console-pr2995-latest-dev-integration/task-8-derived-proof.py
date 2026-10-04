import ast,collections,hashlib,json,subprocess
from pathlib import Path
out=Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge');refs={'prior':'df2ba424de63576d36d9c3e38387c84285303f1c','feature':'edaeececfb3733885bb5fec3ea7b5823266dff23','upstream':'5e0341d1ec701865e019eb2fd8a5e2028ab2d474'}
def read(label,p):
 if label=='final':return Path(p).read_bytes()
 return subprocess.check_output(['git','-c','gc.auto=0','show',refs[label]+':'+p])
def sha(b):return hashlib.sha256(b).hexdigest()
import sys
sys.path.insert(0,str(Path.cwd()))
from scripts import check_persistent_diagnostic_inventory as inv
path='Docs/security/production-diagnostic-inventory.json';inventories={k:json.loads(read(k,path)) for k in [*refs,'final']};print(inventories['final'].keys())
owners={k:{r['path']:r for r in v['owners']} for k,v in inventories.items()}
rows=[]
for p in sorted(set().union(*(d.keys() for d in owners.values()))):
 vals={k:d.get(p) for k,d in owners.items()}
 if vals['feature']==vals['prior']:expected=vals['upstream'];kind='upstream'
 elif vals['upstream']==vals['prior']:expected=vals['feature'];kind='feature'
 elif vals['feature']==vals['upstream']:expected=vals['feature'];kind='identical'
 else:expected=None;kind='shared'
 row={'path':p,'kind':kind,'rows':vals,'exact_owner_union':expected==vals['final']}
 if kind=='shared':
  counters={}
  for label in [*refs,'final']:
   source=read(label,p).decode();tree=ast.parse(source);symbols=inv._logger_symbols(tree)
   counters[label]=collections.Counter(ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and inv._is_diagnostic_call(n,symbols))
  expected=collections.Counter(counters['feature']);expected.update(counters['upstream']);expected.subtract(counters['prior']);expected=+expected
  row['statement_union_exact']=expected==counters['final'];row['statement_counts']={k:sum(v.values()) for k,v in counters.items()};row['extra']=list((counters['final']-expected).elements());row['missing']=list((expected-counters['final']).elements())
 rows.append(row)
result={'revisions':refs,'inventory_sha256':{k:sha(read(k,path)) for k in [*refs,'final']},'owner_union':rows,'topology':{k:v['persistent_sink_topology'] for k,v in inventories.items()}}
(out/'task-8-diagnostic-union-proof.json').write_text(json.dumps(result,indent=2)+'\n');print('unexpected',[(x['path'],x['kind']) for x in rows if not x['exact_owner_union'] and not x.get('statement_union_exact')])
