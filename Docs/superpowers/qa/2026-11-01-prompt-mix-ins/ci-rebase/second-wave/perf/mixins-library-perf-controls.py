import os,sys
sys.path.insert(0,os.getcwd())
import pytest
class Profile:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
raise SystemExit(pytest.main(['Tests/UI/test_library_prompts_canvas.py::test_library_prompt_delete_modal_cancel_preserves_dirty_editor_and_request','Tests/UI/test_library_prompts_canvas.py::test_library_prompt_copy_uses_live_unsaved_legacy_lane_markdown','-q','--tb=short','-p','no:cacheprovider','--basetemp=/private/tmp/mixins-library-perf-controls-tmp'],plugins=[Profile()]))
