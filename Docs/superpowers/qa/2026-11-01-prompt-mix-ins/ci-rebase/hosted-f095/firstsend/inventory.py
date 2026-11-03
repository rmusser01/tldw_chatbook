import hashlib,json
from pathlib import Path
root=Path('/private/tmp/mixins-f095-firstsend');names=['report.md','red.log','green.log','green.xml','run_controls.py','static.py','lint-source.json','unchanged-web-tool-bodies.json','lean_html_probe.py','lean-html-order.json','patch.diff','format.log','source-final.json','inventory.py']
paths=[root/n for n in names]+[Path('/private/tmp/mixins-ci-f095-ui16-red.log')]
for n in ['mixins-perf-f095-firstsend.log','mixins-perf-f095-firstsend-test_ui_ready_module_census-_boot_and_census.json','mixins-perf-f095-firstsend-test_boot_worker_census-_boot_and_census.json','mixins-perf-f095-firstsend-test_screen_preimport_payload_budget-_run_census.json','mixins-perf-f095-firstsend-test_screen_preimport_payload_budget-warmup.json']:
 p=Path('/private/tmp')/n
 if p.exists():paths.append(p)
p=Path('/private/tmp/mixins-perf-probe.py');paths.append(p)
(root/'artifact-files.json').write_text(json.dumps({'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},indent=2)+'\n')
print(len(paths))
