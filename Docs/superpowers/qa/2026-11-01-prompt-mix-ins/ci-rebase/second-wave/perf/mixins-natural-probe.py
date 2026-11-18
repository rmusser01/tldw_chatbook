import sys,os,json,functools,asyncio
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
rows=[]
class Probe:
 def pytest_collection_modifyitems(self,items):
  import Tests.UI.test_console_poll_reconciliation as m
  original=m._original_transition_events
  def wrap(bindings,observe):
   def observed(kind,label,frame,value):
    if len(rows)<300:
     owner=frame.f_locals.get('self')
     full=m._transition_parent(frame, m.ChatScreen._sync_native_console_chat_ui.__code__,getattr(owner,'screen',None)) if label=='surface' else frame
     stack=[]
     ancestor=frame.f_back
     while ancestor is not None and len(stack)<12:
      stack.append((ancestor.f_code.co_filename.rsplit('/',1)[-1],ancestor.f_code.co_name,ancestor.f_lineno)); ancestor=ancestor.f_back
     rows.append({'stack':stack,'kind':kind,'label':label,'task':asyncio.current_task().get_name(),'poll':bool(full and full.f_locals.get('poll_record')),'value':str(type(value)),'result':value if type(value) is bool else None,'requested':getattr(owner,'_console_sync_requested',None),'progress':getattr(owner,'_console_sync_in_progress',None)})
    observe(kind,label,frame,value)
   return original(bindings,observed)
  m._original_transition_events=wrap
 def pytest_sessionfinish(self,*args):
  Path('/private/tmp/mixins-natural-probe.json').write_text(json.dumps(rows,indent=2))
raise SystemExit(pytest.main(['Tests/UI/test_console_preparing_current_sources.py::test_preparing_changed_source_after_original_tab_await_replays_full[natural]','-q','--tb=short','-p','no:cacheprovider','--basetemp=/private/tmp/mixins-natural-probe-tmp'],plugins=[Probe()]))
