import functools,json,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
root=Path('/private/tmp/mixins-f095-perf-contracts')
class Observe:
 def pytest_collection_modifyitems(self,items):
  seen=set()
  for item in items:
   m=item.module
   if m.__name__ in seen:continue
   seen.add(m.__name__)
   original=getattr(m,'run_completed_pair_history_gate',None)
   if original:
    @functools.wraps(original)
    def record(*args,**kwargs):
     result=original(*args,**kwargs);(root/'history-report.json').write_text(json.dumps(result,indent=2)+'\n');return result
    m.run_completed_pair_history_gate=record
raise SystemExit(pytest.main(['Tests/Performance/test_console_tab_sync_source_ownership.py','Tests/Performance/test_speculative_voice_latency.py','-v','--timeout=300','-p','no:cacheprovider','--tb=short','-rx','--basetemp='+str(root/'final-tmp'),'--junitxml='+str(root/'final.xml')],plugins=[Observe()]))
