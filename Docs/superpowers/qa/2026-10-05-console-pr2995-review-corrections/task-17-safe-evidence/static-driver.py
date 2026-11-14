import difflib,hashlib,json,re,subprocess
from pathlib import Path
wt=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');out=wt/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-17-safe-evidence';s=json.loads((out.parent/'task-17-latest-dev-preflight-selection.json').read_text());base='13d1668d4eedbdfcab50fff28a47a8a68346f033';python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python';paths=[p['path'] for p in s['incoming_paths'] if p['path'].endswith('.py')]
records=[]
def run(name,argv):
 p=subprocess.run(argv,cwd=wt,capture_output=True,text=True);(out/f'{name}.stdout').write_text(p.stdout);(out/f'{name}.stderr').write_text(p.stderr);records.append({'name':name,'argv':argv,'exit':p.returncode,'stdout':f'{name}.stdout','stderr':f'{name}.stderr'});print(name,'exit',p.returncode,'stdoutbytes',len(p.stdout),'stderrbytes',len(p.stderr));return p
lint=run('fatal-ruff',[python,'-m','ruff','check','--select','E9,F63,F7,F82',*paths]);assert lint.returncode==0
fmt=run('formatter-assessment',[python,'-m','ruff','format','--diff',*paths]);assert fmt.returncode in (0,1)
ws=run('whitespace-incoming',['git','diff','--check',f'{base}..HEAD']);assert ws.returncode==0
format_assessment={}
for path in paths:
 current=(wt/path).read_text();p=subprocess.run([python,'-m','ruff','format','--stdin-filename',path,'-'],input=current,cwd=wt,text=True,capture_output=True);assert p.returncode==0
 try:original=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=wt,text=True,stderr=subprocess.DEVNULL)
 except subprocess.CalledProcessError:original=''
 added=set()
 for tag,a,b,c,d in difflib.SequenceMatcher(None,original.splitlines(),current.splitlines(),autojunk=False).get_opcodes():
  if tag in ('replace','insert'):added.update(range(c+1,d+1))
 changes=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(None,current.splitlines(),p.stdout.splitlines(),autojunk=False).get_opcodes():
  if tag!='equal':changes.append({'current_start':a+1,'current_end':b,'formatted_start':c+1,'formatted_end':d,'overlaps_incoming_added_lines':bool(added.intersection(range(a+1,b+1)))})
 format_assessment[path]={'would_reformat':bool(changes),'incoming_added_line_count':len(added),'formatter_change_ranges':changes,'policy':'All actual bytes remain exact selected-dev/candidate pins. Formatting suggestions are inherited selected-source evidence; no owner reflow.'}
(out/'static-assessment.json').write_text(json.dumps({'commands':records,'fatal_paths':paths,'added_hunk_formatter_assessment':format_assessment,'source_mutations':False},indent=2)+'\n')
print('formatter paths',sum(v['would_reformat'] for v in format_assessment.values()),'added-overlap paths',sum(any(x['overlaps_incoming_added_lines'] for x in v['formatter_change_ranges']) for v in format_assessment.values()))
