import json,sys,os
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
label=sys.argv[1]
rows={}; current=''
class Probe:
 def pytest_runtest_setup(self,item):
  global current
  current=item.nodeid
 def pytest_sessionstart(self,session):
  from textual.pilot import Pilot
  original=Pilot.pause
  async def pause(self,*args,**kwargs):
   result=await original(self,*args,**kwargs)
   app=self.app
   targets=('global-full-','capture-policy-','trace-privacy-','exchange-export-','trace-export-','console-mixin')
   for widget in app.screen.query('*'):
    if not widget.id or not widget.id.startswith(targets): continue
    key=json.dumps([current,str(app.size),type(app.screen).__name__,widget.id,str(getattr(widget,'label','')),widget.disabled,getattr(app.focused,'id',None)])
    rows[key]={'region':list(widget.region),'content_region':list(widget.content_region),'styles':str(widget.styles),'display':widget.display}
   return result
  Pilot.pause=pause
 def pytest_sessionfinish(self,*args):
  Path(f'/private/tmp/mixins-css-styles-{label}.json').write_text(json.dumps(rows,indent=2))
nodes=['Tests/UI/test_console_capture_policy_dialog.py::test_trace_privacy_dialog_separates_capture_pii_and_viewer','Tests/UI/test_console_capture_policy_dialog.py::test_global_full_requires_restart_aware_distinct_acknowledgement','Tests/UI/test_console_capture_policy_dialog.py::test_compact_dialog_keeps_status_and_fixed_actions_visible_and_escape_restores_focus','Tests/UI/test_console_exchange_export_dialog.py::test_every_full_clipboard_action_requires_fresh_confirmation','Tests/UI/test_console_exchange_export_dialog.py::test_escape_cancels_and_returns_to_previous_focus','Tests/UI/test_trace_export_ui.py::test_export_dialog_composites_with_production_css_at_supported_widths','Tests/UI/test_console_prompt_mixins.py::test_real_console_prompts_action_retains_browse_and_controls_runtime']
raise SystemExit(pytest.main(nodes+['-q','--tb=short','-p','no:cacheprovider',f'--basetemp=/private/tmp/mixins-css-styles-{label}-tmp'],plugins=[Probe()]))
