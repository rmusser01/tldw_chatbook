import ast,subprocess,json
from pathlib import Path
from collections import Counter
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
files=['Tests/Performance/test_boot_worker_census.py','Tests/Performance/test_textual_css_fastpath.py','Tests/Performance/test_ui_ready_module_census.py','Tests/Tools/test_remote_session_registry.py','Tests/UI/test_console_first_send_preimport.py','Tests/UI/test_console_preparing_current_sources.py','Tests/UI/test_console_recovered_image_shutdown.py','tldw_chatbook/Chat/console_first_send_modules.py','tldw_chatbook/Tools/remote_session_registry.py','tldw_chatbook/UI/Console_Modules/image.py','tldw_chatbook/UI/Console_Modules/prompts.py','tldw_chatbook/UI/Library_Modules/library_prompts_controller.py','tldw_chatbook/UI/Screens/library_screen.py','tldw_chatbook/Widgets/Console/__init__.py','tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py','tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py','tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py','tldw_chatbook/Widgets/Console/trace_export_dialog.py']
css=['tldw_chatbook/css/features/_console_prompts.tcss','tldw_chatbook/css/screen_css_scoped.tcss','tldw_chatbook/css/tldw_cli_modular.tcss','tldw_chatbook/css/widget_defaults_scoped.tcss']
Path('/private/tmp/mixins-perf-owned-files.txt').write_text('\n'.join(files+css)+'\n')
ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff';total=Counter();new=[];report={}
for f in files:
 current=(root/f).read_text();old=subprocess.check_output(['git','show','HEAD:'+f],cwd=root,text=True)
 def lint(s):
  r=subprocess.run([ruff,'check','--no-cache','--output-format=json','--stdin-filename',f,'-'],cwd=root,input=s,text=True,capture_output=True);return json.loads(r.stdout)
 a=lint(current);b=lint(old);ac=Counter((x['code'],x['message']) for x in a);bc=Counter((x['code'],x['message']) for x in b);delta=ac-bc
 total.update(before=len(b),after=len(a));report[f]={'before':len(b),'after':len(a)}
 for x in a:
  key=x['code'],x['message']
  if delta[key]:new.append((f,x['location'],*key));delta[key]-=1
 ast.parse(current)
result={'totals':dict(total),'new':new,'per_file':report};Path('/private/tmp/mixins-perf-lint.json').write_text(json.dumps(result,indent=2));print(json.dumps({'totals':dict(total),'new':new},indent=2))
