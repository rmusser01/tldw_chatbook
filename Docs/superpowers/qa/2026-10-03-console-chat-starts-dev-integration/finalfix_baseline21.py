from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
source=root/'baseline21-final.json'
r=json.loads(source.read_text());argv=r['argv']
assert len([v for v in argv if '::test_' in v])==21
new=[('--basetemp=/private/tmp/console-finalfix-baseline21-01a0fa6c' if v.startswith('--basetemp=') else v) for v in argv]
assert sum(a!=b for a,b in zip(argv,new))==1
(root/'finalfix-baseline21-argv-proof.json').write_text(json.dumps({'source_receipt':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_argv':argv,'final_argv':new,'only_basetemp_changed':True},indent=2))
r=subprocess.run([sys.executable,'-B',str(root/'run_check.py'),'finalfix-baseline21',*new]);sys.exit(r.returncode)
