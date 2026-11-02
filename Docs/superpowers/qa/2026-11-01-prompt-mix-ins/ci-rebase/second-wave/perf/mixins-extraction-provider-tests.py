import os,sys
sys.path.insert(0,os.getcwd())
import pytest
class Profile:
 def pytest_collection_modifyitems(self,items):
  for item in items:item.add_marker(pytest.mark.bootstrap_profile)
nodes=['Tests/Chat/test_openai_streaming_usage.py','Tests/Chat/test_session_usage_taps.py::test_openai_nonstreaming_records_exact_usage','Tests/Chat/test_session_usage_taps.py::test_openai_nonstreaming_without_usage_records_estimate','Tests/Chat/test_session_usage_taps.py::test_openai_chat_completions_stream_records_exact_once','Tests/Chat/test_session_usage_taps.py::test_openai_stream_cumulative_usage_chunks_record_once','Tests/Chat/test_session_usage_taps.py::test_responses_stream_records_completed_usage_once','Tests/Chat/test_chat_functions.py::TestProviderRequestPayloads::test_openai_reasoning_uses_responses_api_and_normalizes_output','Tests/Chat/test_chat_functions.py::TestProviderRequestPayloads::test_openai_reasoning_stream_normalizes_responses_events']
raise SystemExit(pytest.main(nodes+['-q','--tb=short','-p','no:cacheprovider','--basetemp=/private/tmp/mixins-extraction-providers-tmp'],plugins=[Profile()]))
