import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
python='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
probe=Path('/private/tmp/mixins-pr3346-frozen-final-five/probe.py')
roots=['tldw_chatbook','Tests/Performance','Tests/conftest.py','conftest.py','pyproject.toml','Tests/UI/test_library_rail_switch_keeps_reader_and_note.py','Tests/UI/test_response_rules_end_to_end.py','Tests/UI/test_library_landing_recompose_publication.py']
def snapshot(phase):
 paths=sorted(set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',*roots],text=True).splitlines()))
 rows=[{'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths if Path(p).is_file()]
 encoded=json.dumps(rows,sort_keys=True).encode()
 result={'phase':phase,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cwd':str(Path.cwd()),'path_roots':roots,'paths':rows,'manifest_sha256':hashlib.sha256(encoded).hexdigest(),'product_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--',*roots])).hexdigest(),'observing_driver_sha256':hashlib.sha256(probe.read_bytes()).hexdigest()}
 (root/f'perf-source-{phase}.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
before=snapshot('before')
assert before['observing_driver_sha256']=='0e671fdc3e15b81d7d06a185610de510d45640aedfa8c195131656e4c0d4dd1c'
argv=[python,str(probe),'pr3346-landing-final']
started=time.monotonic()
with (root/'final-five.log').open('w') as output:rc=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT).returncode
after=snapshot('after')
checks={k:before[k]==after[k] for k in ['head','cwd','path_roots','paths','manifest_sha256','product_diff_sha256','observing_driver_sha256']}
result={'argv':argv,'cwd':str(Path.cwd()),'exit_code':rc,'seconds':round(time.monotonic()-started,3),'source_path_count':len(before['paths']),'checks':checks,'observer_unchanged':True}
(root/'final-five-proof.json').write_text(json.dumps(result,indent=2)+'\n')
assert all(checks.values()),checks
print(json.dumps(result))
raise SystemExit(rc)
