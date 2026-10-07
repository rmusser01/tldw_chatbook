from pathlib import Path
import difflib,json,os,subprocess,sys
scratch=Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration')
paths=['Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py','Tests/Chat/test_console_chat_create_confirm.py','tldw_chatbook/Chat/console_chat_controller.py']
records=[]
for path in paths:
 source=Path(path).read_text();argv=[sys.executable,'-m','ruff','format','--stdin-filename',path,'-'];p=subprocess.run(argv,input=source,capture_output=True,text=True)
 difference=''.join(difflib.unified_diff(source.splitlines(True),p.stdout.splitlines(True),fromfile=path,tofile=path))
 records.append({'argv':argv,'cwd':str(Path.cwd()),'env':{'TLDW_TEST_GC_EVERY':os.environ.get('TLDW_TEST_GC_EVERY')},'returncode':p.returncode,'output':difference})
 print(difference[-18000:])
(scratch/'review1-format-proof-detail.json').write_text(json.dumps(records,indent=2))
