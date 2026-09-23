import hashlib,json,os,runpy,shlex,shutil,subprocess,sys
from pathlib import Path
repo=Path.cwd();root=Path('<tmp>/tldw-32836-approved-001');socket='tldw32836approved001';session='review'
os.umask(0o077)
sys.argv=['prepare_profile.py',str(root)]
runpy.run_path(str(repo/'Docs/superpowers/qa/2026-09-16-handoff-excerpts/prepare_profile.py'),run_name='__main__')
(root/'home').mkdir();sentinels=root/'sentinels';sentinels.mkdir()
for name in ['fixture-trace.jsonl','fixture-result.json']:(sentinels/name).write_text('Pre-existing unrelated fixture sentinel.\n')
(root/'sentinel-before.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sentinels.iterdir()},indent=2)+'\n')
tmux=shutil.which('tmux');assert tmux
assert subprocess.run([tmux,'-L',socket,'has-session','-t',session],capture_output=True).returncode!=0
subprocess.run([tmux,'-L',socket,'new-session','-d','-s',session,'-x','170','-y','48','-c',str(repo)],check=True)
runner=repo/'Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/native_check.py'
code='import runpy; runpy.run_path('+repr(str(runner))+')["main"]()'
cmd=shlex.join([str(repo/'.venv/bin/python'),'-c',code,str(root),socket,session])
cmd+='; native_exit_code=$?; printf "%s\\n" "$native_exit_code" > '+shlex.quote(str(root/'exit-code'))
(root/'launch-command.txt').write_text(cmd+'\n')
subprocess.run([tmux,'-L',socket,'send-keys','-t',session,'-l',cmd],check=True)
subprocess.run([tmux,'-L',socket,'send-keys','-t',session,'Enter'],check=True)
print('Fresh private profile:',root)
