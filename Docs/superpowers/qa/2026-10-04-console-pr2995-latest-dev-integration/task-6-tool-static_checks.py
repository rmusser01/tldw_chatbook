import subprocess,json,pathlib
root=pathlib.Path.cwd();sdd=root/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';py='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python';run='/private/tmp/console-pr2995-merge/run_final_qualification.py'
proof=json.loads((sdd/'task-6-preservation.json').read_text()); paths=[x['path'] for x in proof['owned_python']]
groups=[('task-6-ruff-fatal',[py,'-m','ruff','check','--select','E9,F63,F7,F82',*paths]),('task-6-format-check',[py,'-m','ruff','format','--check',*paths]),('task-6-whitespace',['git','diff','--check','ca2992cb10..HEAD'])]
scripts=['tldw_chatbook/css/check_bundle_sync.py','scripts/check_profile_owned_path_inventory.py','scripts/check_persistent_diagnostic_inventory.py','scripts/check_backlog_task_ids.py','scripts/check_backlog_task_files.py','scripts/check_schema_table_allowlist.py','scripts/check_index_plan_pins.py','scripts/check_textual_worker_contract.py','scripts/check_timestamp_writers.py','scripts/check_ui_pr_gate_census.py','scripts/check_canvas_mermaid_assets.py']
groups.extend(('task-6-derived-'+pathlib.Path(p).stem,[py,p]) for p in scripts)
for label,args in groups:
 print('START '+label,flush=True);p=subprocess.run([py,run,label,*args]);print('END '+label+' exit='+str(p.returncode),flush=True)
