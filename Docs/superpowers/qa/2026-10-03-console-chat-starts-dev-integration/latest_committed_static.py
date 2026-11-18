from pathlib import Path
import hashlib, json, os, subprocess, sys
scratch=Path(__file__).resolve().parent
head=sys.argv[1]
paths=json.loads((scratch/'format-paths.json').read_text())+['Tests/Chat/test_console_personal_context_snapshot.py']
checks=[]
for path in sorted(set(paths)):
 data=subprocess.check_output(['git','show',head+':'+path])
 assert data==Path(path).read_bytes(),path
 argv=[sys.executable,'-m','ruff','check','--select','E9,F63,F7,F82','--stdin-filename',path,'-']
 result=subprocess.run(argv,input=data,capture_output=True)
 checks.append({'argv':argv,'returncode':result.returncode,'output':(result.stdout+result.stderr).decode(),'source_sha256':hashlib.sha256(data).hexdigest()})
 assert result.returncode==0,(path,result.stderr)
for path in json.loads((scratch/'owned-stage-paths.json').read_text()):
 if Path(path).is_file():assert subprocess.check_output(['git','show',head+':'+path])==Path(path).read_bytes(),path
obsolete=['Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py','tldw_chatbook/DB/migrations/agent_runs_v18_to_v19_chat_starts.sql','tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_agent_chat_starts.sql']
for path in obsolete:
 assert not Path(path).exists()
 assert subprocess.run(['git','cat-file','-e',head+':'+path],capture_output=True).returncode!=0,path
(scratch/'latest-committed-head-detail.json').write_text(json.dumps({'head':head,'checks':checks,'obsolete_absent':obsolete},indent=2))
print(f'HEAD{head}: {len(checks)} owned Python files match committed bytes and pass fatal Ruff; all owned stage paths match; three obsolete additions absent.')
