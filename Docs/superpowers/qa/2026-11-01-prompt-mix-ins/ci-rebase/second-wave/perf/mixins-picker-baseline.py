import os,sys
sys.path.insert(0,os.getcwd())
import pytest
class Profile:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
raise SystemExit(pytest.main(['Tests/UI/test_console_prompt_picker.py::test_console_picker_captures_ordinary_segment_draft_at_opening','-q','--tb=short','-p','no:cacheprovider','--basetemp=/private/tmp/mixins-picker-baseline-tmp'],plugins=[Profile()]))
