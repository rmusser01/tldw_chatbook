import sys,os,json
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
arm=sys.argv[1];rows=[]
class Probe:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
  from tldw_chatbook.Widgets.Console import console_composer_bar as owner
  if arm=='original':
   originals=json.load(open('/private/tmp/mixins-extraction-original-functions.json'))
   for name in ('_voice_chip_width_cap','set_voice_status','_sync_full_width_voice_presentation'):
    local={};exec(compile(originals[name],'<unaltered-original-presentation>','exec'),vars(owner),local);setattr(owner.ConsoleComposerBar,name,local[name])
  code=owner.ConsoleComposerBar.set_voice_status.__code__;mon=sys.monitoring;tool=next(i for i in range(6) if mon.get_tool(i) is None);self.tool=tool;self.code=code;mon.use_tool_id(tool,'presentation-caller-observation')
  def observe(code,offset):
   if len(rows)>=200:return
   frame=sys._getframe(1);stack=[];parent=frame.f_back
   for _ in range(5):
    if parent is None:break
    stack.append(parent.f_code.co_name);parent=parent.f_back
   rows.append({'state':frame.f_locals.get('state'),'message':frame.f_locals.get('message'),'callers':stack})
  mon.register_callback(tool,mon.events.PY_START,observe);mon.set_local_events(tool,code,mon.events.PY_START)
 def pytest_sessionfinish(self,*args):
  mon=sys.monitoring;mon.set_local_events(self.tool,self.code,0);mon.register_callback(self.tool,mon.events.PY_START,None);mon.free_tool_id(self.tool);Path(f'/private/tmp/mixins-extraction-pair-{arm}.json').write_text(json.dumps(rows,indent=2))
nodes=['Tests/UI/test_console_composer_reason_width.py::test_chip_going_idle_restores_the_reason_strip_budget','Tests/UI/test_console_composer_reason_width.py::test_preparing_renders_whole_in_the_intermediate_band']
raise SystemExit(pytest.main(nodes+['-q','--tb=short','-p','no:cacheprovider',f'--basetemp=/private/tmp/mixins-extraction-pair-{arm}-tmp'],plugins=[Probe()]))
