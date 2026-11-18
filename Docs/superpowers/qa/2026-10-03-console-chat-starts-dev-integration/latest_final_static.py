from pathlib import Path
import hashlib, json, subprocess,sys
root=Path(__file__).resolve().parent
head=sys.argv[1]
paths=sorted(set(json.loads((root/'format-paths.json').read_text())+[
 'Tests/Chat/test_console_personal_context_snapshot.py','Tests/Chat/test_console_provider_gateway.py',
 'Tests/UI/test_console_cost_chip_screen.py','Tests/UI/test_console_spend_projection.py','Tests/Utils/test_egress.py']))
checks=[]
for path in paths:
 argv=['git','show',head+':'+path]
 result=subprocess.run(argv,capture_output=True)
 assert result.returncode==0,path
 data=result.stdout
 assert data==Path(path).read_bytes(),path
 compile(data,path,'exec')
 argv=[sys.executable,'-B','-m','ruff','check','--select','E9,F63,F7,F82','--stdin-filename',path,'-']
 result=subprocess.run(argv,input=data,capture_output=True)
 checks.append({'argv':argv,'returncode':result.returncode,'output':(result.stdout+result.stderr).decode(),'source_sha256':hashlib.sha256(data).hexdigest()})
 assert result.returncode==0,checks[-1]
obsolete=['Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py','tldw_chatbook/DB/migrations/agent_runs_v18_to_v19_chat_starts.sql','tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_agent_chat_starts.sql']
for path in obsolete:
 assert not Path(path).exists()
 assert subprocess.run(['git','cat-file','-e',head+':'+path],capture_output=True).returncode!=0
(root/'latest-final-committed-source-detail.json').write_text(json.dumps({'head':head,'checks':checks,'obsolete_absent':obsolete},indent=2))
print(f'HEAD {head}: {len(checks)} Python files (original64 plusfouramendedtestowners) match committed bytes, compile and pass fatal Ruff; obsolete additions absent.')
