import os,sys
sys.path.insert(0,os.getcwd())
import pytest
class Profile:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
raise SystemExit(pytest.main(['Tests/UI/test_library_notes_characterization.py::test_notes_new_folder_opens_the_folder_name_dialog','-q','--tb=short','-p','no:cacheprovider','--basetemp=/private/tmp/mixins-folder-route-profile-tmp'],plugins=[Profile()]))
