import hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
root=Path.cwd(); out=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'
label=sys.argv[1]; nodes=sys.argv[2:]; profile=Path(tempfile.mkdtemp(prefix='task-8-'+label+'-',dir=out)); env=dict(os.environ)
env.update(TLDW_TEST_CONFIG_ROOT=str(profile), HOME=str(profile/'home'), USERPROFILE=str(profile/'home'), XDG_CONFIG_HOME=str(profile/'config'), XDG_DATA_HOME=str(profile/'data'),TLDW_CONFIG_PATH=str(profile/'config/config.toml'), PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring',HF_HUB_OFFLINE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
for k in ['TLDW_TEST_CONFIG_ROOT_OWNER','PYTEST_ADDOPTS']:env.pop(k,None)
argv=[sys.executable,'-m','pytest','-p','pytest_asyncio.plugin','-p','pytest_timeout','-p','pytest_mock','-o','addopts=','--timeout=45' if label.startswith('red') else '--timeout=300','--timeout-method=thread','-q','--tb=short','--basetemp='+str(profile/'pytest'),'--junitxml='+str(out/('task-8-'+label+'.xml')),*nodes]
paths=sorted({n.split('::')[0] for n in nodes}|{'Tests/conftest.py','Tests/private_profile.py','Tests/real_profile_guard.py','tldw_chatbook/Chat/console_chat_controller.py'})
def hashes():return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).is_file()}
before=hashes(); start=time.monotonic()
with (out/('task-8-'+label+'.log')).open('w') as log:
 try: result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200); code=result.returncode; terminated=False
 except subprocess.TimeoutExpired: code=124; terminated=True
receipt=dict(argv=argv,exit=code,terminated=terminated,seconds=time.monotonic()-start,source_before=before,source_after=hashes())
(out/('task-8-'+label+'.json')).write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps({'exit':code,'seconds':receipt['seconds'],'receipt':str(out/('task-8-'+label+'.json'))}))
