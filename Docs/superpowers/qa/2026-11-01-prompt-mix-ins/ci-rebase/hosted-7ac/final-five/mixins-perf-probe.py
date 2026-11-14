import functools,json,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
import pytest
label=sys.argv[1]
class Probe:
 def pytest_collection_modifyitems(self,items):
  seen=set()
  for item in items:
   module=item.module
   if module.__name__ in seen: continue
   seen.add(module.__name__)
   for name in ('_run_census','_boot_and_census'):
    original=getattr(module,name,None)
    if original is None: continue
    def wrap(original=original,module=module,name=name):
     @functools.wraps(original)
     def observe(*args,**kwargs):
      result=original(*args,**kwargs)
      kind='warmup' if isinstance(result,dict) and 'warmup_added_modules' in result else name
      Path(f'/private/tmp/mixins-perf-{label}-{module.__name__.split(".")[-1]}-{kind}.json').write_text(json.dumps(result,indent=2)+'\n')
      return result
     return observe
    setattr(module,name,wrap())
nodes=['Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size','Tests/Performance/test_boot_worker_census.py::test_boot_worker_and_thread_starts_stay_within_the_allowlist','Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet','Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget','Tests/Performance/test_screen_preimport_payload_budget.py::test_first_send_warmup_payload_stays_within_budget']
raise SystemExit(pytest.main(nodes+['-q','--tb=short','-p','no:cacheprovider',f'--basetemp=/private/tmp/mixins-perf-{label}-tmp'],plugins=[Probe()]))
