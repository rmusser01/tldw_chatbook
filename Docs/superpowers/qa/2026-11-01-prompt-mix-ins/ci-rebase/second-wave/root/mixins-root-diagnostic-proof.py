import ast,collections,json,subprocess
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from scripts.check_persistent_diagnostic_inventory import _statement_entries
base='ba1cafc55dd5ac9b6b1f066c936206e652ed6b65'
paths=set(subprocess.check_output(['git','diff','--name-only',base],text=True).splitlines())|set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines())
paths=sorted(p for p in paths if p.startswith('tldw_chatbook/') and p.endswith('.py'))
class Unqualify(ast.NodeTransformer):
 def visit_Attribute(self,n):
  n=self.generic_visit(n)
  if isinstance(n.value,ast.Name) and n.value.id in {'_source','_owner','host'}:return ast.copy_location(ast.Name(n.attr,ast.Load()),n)
  return n
before=collections.Counter();after=collections.Counter();records={}
for p in paths:
 r=subprocess.run(['git','show',f'{base}:{p}'],text=True,capture_output=True)
 old=_statement_entries(r.stdout,p) if r.returncode==0 else []
 new=_statement_entries(Path(p).read_text(),p)
 def sig(entry):return ast.dump(Unqualify().visit(ast.parse(entry['text'])),include_attributes=False)
 before.update(map(sig,old));after.update(map(sig,new))
 if old or new:records[p]={'before':len(old),'after':len(new)}
result={'base':base,'owners':records,'before_calls':sum(before.values()),'after_calls':sum(after.values()),'removed':list((before-after).elements()),'added':list((after-before).elements()),'qualification':'AST call multiset after replacing only actual defining-module qualifiers with original global names. Location and raw-format hashes may change; call arguments, levels and messages must remain identical.'}
Path('/private/tmp/mixins-root-diagnostic-proof.json').write_text(json.dumps(result,indent=2)+'\n')
assert not result['removed'] and not result['added'],(len(result['removed']),len(result['added']))
print('Verified unchanged diagnostic call ASTs:',result['before_calls'],'across',len(records),'owners.')
