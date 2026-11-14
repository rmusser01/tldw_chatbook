import collections,json,subprocess,re
from pathlib import Path
root=Path.cwd();baseline=Path('/private/tmp/mixins-size-ba1-baseline');owners=['app.py','app_config_notifications.py','app_service_wiring.py','Notes/notes_sync_startup.py','app_command_providers.py','UI/Workbench/help.py','Agents/agent_service.py','Chat/provider_readiness.py','DB/ChaChaNotes_DB.py','DB/sql_validation.py','DB/Prompts_DB.py','DB/prompt_export.py','Chat/console_provider_gateway.py','Chat/refused_stream.py','Chat/console_runtime.py','Chat/console_trace_maintenance_runtime.py','Chat/citation_trace_repository.py','Chat/citation_artifact_ownership.py']
allrows={}
for label,tree in [('before',baseline),('after',root)]:
 paths=[str(tree/'tldw_chatbook'/p) for p in owners if (tree/'tldw_chatbook'/p).exists()];r=subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','ruff','check','--output-format','json',*paths],capture_output=True,text=True);assert r.returncode in (0,1),r.stderr;rows=json.loads(r.stdout);allrows[label]=rows
key=lambda x:(x['code'],re.sub(r'from line [0-9]+:', 'from line <position>:',x['message']))
before=collections.Counter(map(key,allrows['before']));after=collections.Counter(map(key,allrows['after']));added=after-before
out={'before':len(allrows['before']),'after':len(allrows['after']),'added':[{'code':x[0],'message':x[1],'count':v} for x,v in added.items()],'removed':[{'code':x[0],'message':x[1],'count':v} for x,v in (before-after).items()],'qualification':'Aggregate code/message multiset over 18 explicit root extraction owners and destinations, preserving moved inherited diagnostics.'}
Path('/private/tmp/mixins-root-lint-comparison.json').write_text(json.dumps(out,indent=2)+'\n');Path('/private/tmp/mixins-root-lint-details.json').write_text(json.dumps(allrows,indent=2)+'\n');print(json.dumps(out,indent=2));assert not added
