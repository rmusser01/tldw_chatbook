import ast,difflib
from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
changes={}
view='tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py'
def edit(path): return changes.setdefault(path,(root/path).read_text())
def tag(path,choose):
 s=edit(path); lines=s.splitlines(keepends=True); offsets=[0]
 for line in lines: offsets.append(offsets[-1]+len(line))
 def pos(line,col): return offsets[line-1]+len(lines[line-1].encode()[:col].decode())
 replacements=[]
 for n in ast.walk(ast.parse(s)):
  if not isinstance(n,ast.Call):continue
  name=n.func.id if isinstance(n.func,ast.Name) else ''
  kwargs={k.arg:k for k in n.keywords}
  identity=kwargs.get('id'); identity=identity.value.value if identity and isinstance(identity.value,ast.Constant) else None
  cls=choose(name,identity)
  if not cls:continue
  if 'classes' in kwargs:
   v=kwargs['classes'].value
   assert isinstance(v,ast.Constant) and isinstance(v.value,str)
   a=pos(v.lineno,v.col_offset);b=pos(v.end_lineno,v.end_col_offset)
   replacements.append((a,b,repr(v.value+' '+cls)))
  else:
   b=pos(n.end_lineno,n.end_col_offset)-1
   assert s[b]==')'
   replacements.append((b,b,('' if s[:b].rstrip().endswith(',') else ',')+f' classes="{cls}"'))
 for a,b,value in sorted(replacements,reverse=True):s=s[:a]+value+s[b:]
 changes[path]=s
 return len(replacements)
print('view',tag(view,lambda name,identity: 'console-mixins-copy' if name=='Static' else 'console-mixins-action' if name=='Button' and identity!='console-mixins-retry' else 'console-mixin-toggle' if name=='Checkbox' else None))
for path,prefixes in [
 ('tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py',{'global-full':{'global-full-cancel','global-full-confirm'},'capture-policy':{'capture-policy-cancel','capture-policy-apply'},'trace-privacy':{'trace-privacy-cancel','trace-privacy-apply'}}),
 ('tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py',{'exchange-export':{'exchange-export-cancel','exchange-export-submit'}}),
 ('tldw_chatbook/Widgets/Console/trace_export_dialog.py',{'trace-export':{'trace-export-browse','trace-export-cancel','trace-export-submit'}}),
]:
 print(path,tag(path,lambda name,identity: next((prefix+'-action' for prefix,ids in prefixes.items() if name=='Button' and identity in ids),None)))
 for prefix in prefixes:
  old=f'#{prefix}-actions Button';new=old+'.'+prefix+'-action'
  assert old in changes[path];changes[path]=changes[path].replace(old,new)
p='tldw_chatbook/css/features/_console_prompts.tcss';s=edit(p)
for old,new in [('ConsolePromptMixinsView Static','ConsolePromptMixinsView Static.console-mixins-copy'),('.console-mixins-actions Button','.console-mixins-actions Button.console-mixins-action'),('.console-mixin-row Checkbox','.console-mixin-row Checkbox.console-mixin-toggle')]:
 assert old in s;s=s.replace(old,new)
changes[p]=s
out=['*** Begin Patch']
for p,new in changes.items():
 old=(root/p).read_text()
 out.append('*** Update File: '+str(root/p))
 out.extend(list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3,lineterm=""))[2:])
out.append('*** End Patch')
Path('/private/tmp/mixins-classkey.patch').write_text('\n'.join(out)+'\n')
