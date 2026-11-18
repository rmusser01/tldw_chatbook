from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections,json,re,subprocess,time,sys
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
OUT=WT/'.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration'
def git(*args):
 return subprocess.check_output(['git',*args],cwd=WT)
refs=sorted(set(git('for-each-ref','--format=%(objectname)').decode().splitlines()+[git('rev-parse','HEAD').decode().strip()]))
pattern=re.compile(r'(?:^|/)task-(\d+)(?:\.\d+)*(?: - |%20|\.md)')
def tree(sha):
 return [p.decode('utf-8') for p in git('ls-tree','-r','-z','--name-only',sha,'--','backlog/tasks','backlog/drafts').split(b'\0') if p]
ids=set()
selected_roots={34213,34214,34215}
expected_names={p.name for p in (WT/'backlog/tasks').glob('task-3421*.md') if (m:=pattern.search(str(p))) and int(m.group(1)) in selected_roots}
foreign_names=set()
with ThreadPoolExecutor(max_workers=8) as pool:
 for paths in pool.map(tree,refs):
  ids.update(int(m.group(1)) for p in paths if (m:=pattern.search(p)))
  foreign_names.update(Path(p).name for p in paths if (m:=pattern.search(p)) and int(m.group(1)) in selected_roots and Path(p).name not in expected_names)
worktrees=[line[9:] for line in git('worktree','list','--porcelain').decode().splitlines() if line.startswith('worktree ')]
for path in worktrees:
 for subdir in ('tasks','drafts'):
  d=Path(path)/'backlog'/subdir
  if d.is_dir():
   ids.update(int(m.group(1)) for p in d.glob('task-*.md') if (m:=pattern.search(str(p))))
   foreign_names.update(p.name for p in d.glob('task-*.md') if (m:=pattern.search(str(p))) and int(m.group(1)) in selected_roots and p.name not in expected_names)
current=collections.defaultdict(list)
for p in (WT/'backlog/tasks').glob('task-*.md'):
 match=re.match(r'task-(\d+(?:\.\d+)*) - ',p.name)
 if match: current[match.group(1)].append(p.name)
historical_ids=set()
process=subprocess.Popen(['git','rev-list','--objects','--all','--','backlog/tasks','backlog/drafts'],cwd=WT,stdout=subprocess.PIPE,text=True)
for line in process.stdout:
 match=pattern.search(line)
 if match: historical_ids.add(int(match.group(1)))
assert process.wait()==0
ids.update(historical_ids)
first=max(ids)+1
record={'time':time.time(),'argv':['git','for-each-ref','--format=%(objectname)'],'unique_ref_trees':len(refs),'worktrees':len(worktrees),'head':git('rev-parse','HEAD').decode().strip(),'origin_dev':git('rev-parse','origin/dev').decode().strip(),'max_task_id_all_refs_worktrees':max(ids),'max_historical_task_id':max(historical_ids),'provisional_block':[first,first+1,first+2],'duplicates':{k:v for k,v in current.items() if len(v)>1},'selected_roots':sorted(selected_roots),'selected_expected_filenames':sorted(expected_names),'selected_foreign_filenames':sorted(foreign_names)}
(OUT/(sys.argv[1] if len(sys.argv)>1 else 'task-id-final-sweep.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
