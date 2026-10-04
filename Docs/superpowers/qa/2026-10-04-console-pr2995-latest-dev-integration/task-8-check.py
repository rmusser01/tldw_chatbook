import hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
root=Path.cwd();out=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';label=sys.argv[1];argv=sys.argv[2:];profile=Path(tempfile.mkdtemp(prefix='task-8-check-'+label+'-',dir=out));env=dict(os.environ);env.update(TLDW_TEST_CONFIG_ROOT=str(profile),HOME=str(profile/'home'),USERPROFILE=str(profile/'home'),XDG_CONFIG_HOME=str(profile/'config'),XDG_DATA_HOME=str(profile/'data'),TLDW_CONFIG_PATH=str(profile/'config/config.toml'),PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring',HF_HUB_OFFLINE='1')
start=time.monotonic()
with (out/('task-8-'+label+'.log')).open('w') as log:
 r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
d={'argv':argv,'exit':r.returncode,'seconds':time.monotonic()-start};(out/('task-8-'+label+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(d)
