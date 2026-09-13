"""Task-owned mount-only negative/positive control; no models or speech."""
import json
from pathlib import Path
import subprocess
import uuid

root = Path(__file__).resolve().parent
image = json.loads((root / 'image-inspect.json').read_text())[0]['Id']
payload = '''import ctypes,json,shutil,tempfile
from pathlib import Path
print(json.dumps({"tmp_mount":[line for line in Path("/proc/mounts").read_text().splitlines() if line.split()[1]=="/tmp"]}),flush=True)
with tempfile.TemporaryDirectory() as directory:
    source=Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1")
    target=Path(directory)/source.name
    shutil.copyfile(source,target)
    ctypes.CDLL(str(target))
    print("COPIED_LIBRARY_LOADED",flush=True)
'''
results=[]
for mode in ('default', 'exec'):
    options='/tmp:rw,nosuid,nodev,size=67108864' + (',exec' if mode=='exec' else '')
    command=['docker','run','--rm','--name','tldw-task32153-tmpfs-'+uuid.uuid4().hex[:12],
             '--label','tldw.validation.task=32153','--network','none','--read-only',
             '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','1',
             '--memory','256m','--pids-limit','32','--tmpfs',options,image,'python','-B','-c',payload]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    row={'mode':mode,'argv':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    results.append(row)
    (root/'tmpfs-library-controls.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(row),flush=True)
assert results[0]['exit_code'] != 0 and 'failed to map segment' in results[0]['stderr']
assert 'noexec' in results[0]['stdout']
assert results[1]['exit_code'] == 0 and 'COPIED_LIBRARY_LOADED' in results[1]['stdout']
print('PASS: default tmpfs noexec reproduces loader failure; explicit exec alone fixes copied-library loading')
