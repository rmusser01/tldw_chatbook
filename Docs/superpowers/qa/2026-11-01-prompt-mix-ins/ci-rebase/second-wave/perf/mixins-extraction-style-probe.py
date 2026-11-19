import os,sys,json,re
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
label=sys.argv[1];rows={};current=''
class Probe:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
 def pytest_runtest_setup(self,item):
  global current
  current=item.nodeid
 def pytest_sessionstart(self,session):
  from textual.pilot import Pilot
  original=Pilot.pause
  async def pause(self,*args,**kwargs):
   result=await original(self,*args,**kwargs)
   from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
   for composer in self.app.query(ConsoleComposerBar):
    state=getattr(composer,'_voice_status_last',None)
    for node in (composer,*composer.query('*')):
     if not node.id:continue
     key=json.dumps([current,str(self.app.size),state,node.id,getattr(self.app.focused,'id',None)],default=str)
     rows[key]={'region':list(node.region),'content_region':list(node.content_region),'display':node.display,'styles':re.sub(r'^RenderStyles\(.*?\), ','',str(node.styles))}
   return result
  Pilot.pause=pause
 def pytest_sessionfinish(self,*args):
  Path(f'/private/tmp/mixins-extraction-styles-{label}.json').write_text(json.dumps(rows,indent=2))
nodes=['Tests/UI/test_console_voice_chip.py','Tests/UI/test_console_composer_reason_width.py','Tests/Chat/test_openai_streaming_usage.py::test_responses_stream_close_retires_transport_without_recording_completion','Tests/Chat/test_openai_streaming_usage.py::test_responses_adapter_reads_current_owner_clock_ledger_and_error_handler','Tests/Chat/test_openai_streaming_usage.py::test_usage_line_adapter_reads_the_current_owner_json_parser']
raise SystemExit(pytest.main(nodes+['-q','--tb=short','-p','no:cacheprovider',f'--basetemp=/private/tmp/mixins-extraction-styles-{label}-tmp'],plugins=[Probe()]))
