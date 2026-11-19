from pathlib import Path
import hashlib,json,os,subprocess,sys,time
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
OUT=WT/'.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration'
label=sys.argv[1]
assert label.replace('-','').isalnum()
argv=sys.argv[2:]
assert argv
selected={k:os.environ[k] for k in ('PYTHON','TLDW_TEST_GC_EVERY','TLDW_CANVAS_MERMAID_INPUT_DIR') if k in os.environ}
start=time.time()
p=subprocess.run(argv,cwd=WT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
record={'argv':argv,'cwd':str(WT),'env':selected,'started_at':start,'elapsed_seconds':time.time()-start,'returncode':p.returncode,'stdout':p.stdout,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
(OUT/(label+'.log')).write_text(p.stdout)
print(p.stdout,end='')
print('CHECK_RECEIPT',str(OUT/(label+'.json')),'RETURN',p.returncode)
sys.exit(p.returncode)
