import pytest
import subprocess
from unittest.mock import patch, MagicMock, ANY

from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import chat_with_mlx_lm
from tldw_chatbook.Local_Inference.mlx_lm_inference_local import (
    start_mlx_lm_server,
    stop_mlx_lm_server
)
from tldw_chatbook.config import settings  # Assuming settings is already loaded or mutable for tests
from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError, ChatProviderError

# Helper to reset settings if modified directly (use with caution or preferably mock settings.get)
def_mlx_settings = {
    "model_path": "mlx-community/test-model",
    "host": "127.0.0.1",
    "port": 8080,
    "temperature": 0.6,
    "max_tokens": 1024,
    "streaming": False,
    "top_p": 0.9,
    "api_timeout": 120,
    "api_retries": 1,
    "api_retry_delay": 1
}


@pytest.fixture(autouse=True)
def mock_mlx_settings():
    # This fixture will automatically apply to all tests in this module
    # It mocks settings.get('api_settings', {}).get('mlx_lm', {})
    # to return a controlled dictionary.
    original_api_settings = settings.get('api_settings', {})

    # Ensure mlx_lm key exists under api_settings for the duration of the test
    # by creating a copy and modifying it.
    # This approach is safer if settings is a shared global object.
    # If settings is a DotMap or similar, direct modification might be okay for tests,
    # but patching `settings.get` is often cleaner.

    mocked_api_settings = original_api_settings.copy() if original_api_settings else {}
    if 'mlx_lm' not in mocked_api_settings:
        mocked_api_settings['mlx_lm'] = {}  # Ensure mlx_lm key exists

    # Patch settings.get specifically for the 'api_settings' key
    with patch.object(settings, 'get') as mock_settings_get:

        def side_effect_for_get(key, default=None):
            if key == 'api_settings':
                # Further refine to control what .get('mlx_lm') on this returns
                mock_mlx_lm_config_dict = MagicMock()
                mock_mlx_lm_config_dict.get.side_effect = lambda k, v=None: def_mlx_settings.get(k,
                                                                                                 v) if k in def_mlx_settings else v

                mock_api_settings_dict = MagicMock()
                mock_api_settings_dict.get.side_effect = lambda k, v=None: mock_mlx_lm_config_dict if k == 'mlx_lm' else (
                    original_api_settings.get(k, v) if original_api_settings else v)

                return mock_api_settings_dict
            return original_api_settings.get(key, default) if original_api_settings else default

        mock_settings_get.side_effect = side_effect_for_get
        yield  # Test runs with patched settings


# --- Tests for start_mlx_lm_server ---

@patch('subprocess.Popen')
def test_start_mlx_lm_server_success(mock_popen):
    """Test successful server start."""
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 1234
    mock_popen.return_value = mock_process

    model_path = "mlx-community/test-model"
    host = "127.0.0.1"
    port = 8080

    process = start_mlx_lm_server(model_path, host, port)

    expected_command = [
        "python", "-m", "mlx_lm.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port)
    ]
    mock_popen.assert_called_once_with(
        expected_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=ANY  # Check that env was passed, content checked separately if needed
    )
    assert process == mock_process
    assert process.pid == 1234


@patch('subprocess.Popen')
def test_start_mlx_lm_server_with_additional_args(mock_popen):
    """Test server start with additional arguments."""
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_popen.return_value = mock_process

    model_path = "mlx-community/test-model"
    host = "127.0.0.1"
    port = 8080
    additional_args = "--num-threads 4 --no-cache"

    start_mlx_lm_server(model_path, host, port, additional_args)

    expected_command = [
        "python", "-m", "mlx_lm.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--num-threads", "4", "--no-cache"
    ]
    mock_popen.assert_called_once_with(
        expected_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=ANY
    )


@patch('subprocess.Popen', side_effect=FileNotFoundError("python not found"))
def test_start_mlx_lm_server_file_not_found(mock_popen_error):
    """Test FileNotFoundError when starting server."""
    process = start_mlx_lm_server("model", "host", 8080)
    assert process is None  # Function should catch FileNotFoundError and return None


@patch('subprocess.Popen', side_effect=Exception("Some other error"))
def test_start_mlx_lm_server_other_exception(mock_popen_exception):
    """Test other exceptions during Popen are caught."""
    process = start_mlx_lm_server("model", "host", 8080)
    assert process is None  # Function should catch general exceptions and return None


# --- Tests for stop_mlx_lm_server ---

def test_stop_mlx_lm_server_graceful_termination():
    """Test graceful server stop (terminate then wait)."""
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 123
    mock_process.poll.return_value = None  # Initially running

    stop_mlx_lm_server(mock_process)

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=10)
    mock_process.kill.assert_not_called()


def test_stop_mlx_lm_server_force_kill_on_timeout():
    """Test server kill if terminate/wait times out."""
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 123
    mock_process.poll.return_value = None  # Initially running
    # Simulate wait() timing out
    mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)

    stop_mlx_lm_server(mock_process)

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_any_call(timeout=10)  # First call for terminate
    mock_process.kill.assert_called_once()
    # Check if wait was called again after kill
    assert any(call_args[1].get('timeout') == 5 for call_args in mock_process.wait.call_args_list if call_args[1])


def test_stop_mlx_lm_server_already_terminated():
    """Test stopping an already terminated process."""
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 123
    mock_process.poll.return_value = 0  # Not None, so already terminated

    stop_mlx_lm_server(mock_process)

    mock_process.terminate.assert_not_called()
    mock_process.kill.assert_not_called()


def test_stop_mlx_lm_server_no_process():
    """Test stopping when no process is provided."""
    # This should log a warning but not raise an error.
    # We can't easily check logs here without more setup, so just ensure it runs.
    stop_mlx_lm_server(None)
    # No assertions needed, just checking it doesn't crash


# --- Tests for chat_with_mlx_lm ---

@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server')
def test_chat_with_mlx_lm_success_with_config(mock_openai_call, mock_mlx_settings):
    """Test successful chat using primarily config values."""
    input_data = [{"role": "user", "content": "Hello"}]

    # Values from mock_mlx_settings (def_mlx_settings)
    expected_model = def_mlx_settings["model_path"]
    expected_host = def_mlx_settings["host"]
    expected_port = def_mlx_settings["port"]
    expected_temp = def_mlx_settings["temperature"]
    expected_api_base_url = f"http://{expected_host}:{expected_port}/v1"

    chat_with_mlx_lm(input_data=input_data)

    mock_openai_call.assert_called_once_with(
        api_base_url=expected_api_base_url,
        model_name=expected_model,
        input_data=input_data,
        api_key=None,
        temp=expected_temp,
        system_message=None,  # Not provided in this call
        streaming=def_mlx_settings["streaming"],  # from config
        max_tokens=def_mlx_settings["max_tokens"],  # from config
        top_p=def_mlx_settings["top_p"],  # from config
        top_k=None,  # Not in default config for this test
        min_p=None,  # Not in default config
        n=None,  # Not in default config
        stop=None,  # Not in default config
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        seed=None,  # Not in default config
        response_format=None,
        tools=None,
        tool_choice=None,
        logprobs=None,
        top_logprobs=None,
        user_identifier=None,
        provider_name="MLX-LM",
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"]
    )


@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server')
def test_chat_with_mlx_lm_args_override_config(mock_openai_call, mock_mlx_settings):
    """Test that function arguments override config values."""
    input_data = [{"role": "user", "content": "Override test"}]
    custom_model = "override/model"
    custom_temp = 0.99
    custom_max_tokens = 512
    custom_host = "192.168.1.100"  # This won't be used by chat_with_mlx_lm if api_url is formed by its own host/port config
    custom_port = 9090  # Same as above

    # To test host/port override, we would typically pass api_url directly if chat_with_mlx_lm supported it
    # OR, we modify the settings for host/port for this specific test case.
    # For now, let's assume chat_with_mlx_lm always uses its configured host/port for URL.
    # The `api_url` param in chat_with_mlx_lm is for overriding the *entire* URL.

    expected_api_base_url = f"http://{def_mlx_settings['host']}:{def_mlx_settings['port']}/v1"  # Uses config host/port

    chat_with_mlx_lm(
        input_data=input_data,
        model=custom_model,
        temp=custom_temp,
        max_tokens=custom_max_tokens
        # Not passing host/port directly to chat_with_mlx_lm, as it relies on config or a full api_url override
    )

    mock_openai_call.assert_called_once_with(
        api_base_url=expected_api_base_url,  # Still from config
        model_name=custom_model,  # Overridden
        input_data=input_data,
        api_key=None,
        temp=custom_temp,  # Overridden
        system_message=None,
        streaming=def_mlx_settings["streaming"],  # From config
        max_tokens=custom_max_tokens,  # Overridden
        top_p=def_mlx_settings["top_p"],  # From config
        top_k=None,
        min_p=None,
        n=None,
        stop=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        seed=None,
        response_format=None,
        tools=None,
        tool_choice=None,
        logprobs=None,
        top_logprobs=None,
        user_identifier=None,
        provider_name="MLX-LM",
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"]
    )


@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server')
def test_chat_with_mlx_lm_api_url_override(mock_openai_call, mock_mlx_settings):
    """Test that api_url argument overrides config host/port for base URL."""
    input_data = [{"role": "user", "content": "API URL override"}]
    custom_api_url = "http://custom.server:1234/custom_v1_path"  # Full URL

    chat_with_mlx_lm(
        input_data=input_data,
        api_url=custom_api_url
    )

    mock_openai_call.assert_called_once_with(
        api_base_url=custom_api_url.rstrip('/'),  # api_url is passed directly
        model_name=def_mlx_settings["model_path"],  # From config
        input_data=input_data,
        api_key=None,
        temp=def_mlx_settings["temperature"],  # From config
        system_message=None,
        streaming=def_mlx_settings["streaming"],
        max_tokens=def_mlx_settings["max_tokens"],
        top_p=def_mlx_settings["top_p"],
        # ... other params from config or defaults ...
        top_k=None, min_p=None, n=None, stop=None, presence_penalty=None, frequency_penalty=None,
        logit_bias=None, seed=None, response_format=None, tools=None, tool_choice=None,
        logprobs=None, top_logprobs=None, user_identifier=None,
        provider_name="MLX-LM",
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"]
    )


def test_chat_with_mlx_lm_missing_model_config():
    """Test ChatConfigurationError if model path is missing."""
    input_data = [{"role": "user", "content": "Test"}]

    # Mock settings to have an empty mlx_lm config for this test
    with patch.object(settings, 'get') as mock_settings_get:
        mock_api_settings_dict = MagicMock()
        mock_api_settings_dict.get.return_value = {}  # Empty mlx_lm config
        mock_settings_get.return_value = mock_api_settings_dict  # For 'api_settings'

        with pytest.raises(ChatConfigurationError, match="MLX-LM model path .* is required"):
            chat_with_mlx_lm(input_data=input_data)


@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server',
       side_effect=ChatProviderError("MLX-LM", "Network Error", 503))
def test_chat_with_mlx_lm_provider_error(mock_openai_call_error, mock_mlx_settings):
    """Test that ChatProviderError from underlying call propagates."""
    input_data = [{"role": "user", "content": "Test provider error"}]
    with pytest.raises(ChatProviderError, match="Network Error"):
        chat_with_mlx_lm(input_data=input_data)


# Add more tests as needed, e.g., for streaming, different combinations of parameters, etc.

# Example of how to test if specific kwargs are passed through
@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server')
def test_chat_with_mlx_lm_kwargs_passthrough(mock_openai_call, mock_mlx_settings):
    input_data = [{"role": "user", "content": "Test kwargs"}]
    custom_seed = 12345
    custom_stop_seq = ["\nUser:", "###"]

    chat_with_mlx_lm(
        input_data=input_data,
        seed=custom_seed,  # Passed as kwarg
        stop=custom_stop_seq  # Passed as kwarg
    )

    expected_api_base_url = f"http://{def_mlx_settings['host']}:{def_mlx_settings['port']}/v1"

    mock_openai_call.assert_called_once_with(
        api_base_url=expected_api_base_url,
        model_name=def_mlx_settings["model_path"],
        input_data=input_data,
        api_key=None,
        temp=def_mlx_settings["temperature"],
        system_message=None,
        streaming=def_mlx_settings["streaming"],
        max_tokens=def_mlx_settings["max_tokens"],
        top_p=def_mlx_settings["top_p"],
        top_k=None,
        min_p=None,
        n=None,
        stop=custom_stop_seq,  # Check if passed
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        seed=custom_seed,  # Check if passed
        response_format=None,
        tools=None,
        tool_choice=None,
        logprobs=None,
        top_logprobs=None,
        user_identifier=None,
        provider_name="MLX-LM",
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"]
    )


# Note on settings mocking:
# The fixture `mock_mlx_settings` provides a basic way to mock `settings.get('api_settings', {}).get('mlx_lm', {})`.
# For tests requiring different mlx_lm configurations (e.g., missing host/port),
# you might need to refine the fixture or use `@patch.dict` or more specific `patch.object`
# within those individual tests if the global fixture isn't suitable.
# The current fixture is a bit complex due to nested `get` calls.
# A simpler approach if `settings` is a DotMap or a direct dict:
# @patch.dict(settings, {"api_settings": {"mlx_lm": def_mlx_settings.copy()}}, clear=True)
# However, `settings` is imported as a module/object, so `patch.object` or patching its methods is more common.
# The current fixture attempts to mock `settings.get().get()` behavior.

# Consider testing the case where `start_mlx_lm_server` returns None (e.g. FileNotFoundError)
# and how `chat_with_mlx_lm` would behave. Currently, `chat_with_mlx_lm` doesn't start the server;
# it assumes the server is already running at the specified host/port or api_url.
# So, `start_mlx_lm_server` tests are about the server process, and `chat_with_mlx_lm` tests
# are about correctly calling the OpenAI-compatible endpoint.
# Testing the UI event handlers that use these functions together would be integration testing.
# (e.g., in `test_llm_management_events.py`)

# Test for host/port missing from config and not overridden by api_url in chat_with_mlx_lm
def test_chat_with_mlx_lm_missing_host_port_config():
    input_data = [{"role": "user", "content": "Test"}]

    # Mock settings to have mlx_lm config missing host/port
    with patch.object(settings, 'get') as mock_settings_get:
        incomplete_mlx_config = {"model_path": "some/model"}  # Missing host/port

        def side_effect_for_get(key, default=None):
            if key == 'api_settings':
                mock_api_settings_dict = MagicMock()
                mock_api_settings_dict.get.return_value = incomplete_mlx_config  # for 'mlx_lm'
                return mock_api_settings_dict
            return default

        mock_settings_get.side_effect = side_effect_for_get

        # Since host defaults to 127.0.0.1 and port to 8080 in chat_with_mlx_lm if not in config,
        # this test won't raise ChatConfigurationError unless those defaults are also removed from the function.
        # Instead, it should use the defaults. Let's verify that.
        with patch(
                'tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server') as mock_openai_call:
            chat_with_mlx_lm(input_data=input_data)

            args, kwargs = mock_openai_call.call_args
            assert kwargs['api_base_url'] == "http://127.0.0.1:8080/v1"  # Default host/port used
            assert kwargs['model_name'] == "some/model"


# Test for model path missing but provided as argument
@patch('tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server')
def test_chat_with_mlx_lm_model_arg_overrides_missing_config(mock_openai_call):
    input_data = [{"role": "user", "content": "Test"}]
    model_arg = "specific/model_via_arg"

    with patch.object(settings, 'get') as mock_settings_get:
        # Config for mlx_lm exists but 'model_path' or 'model' is missing
        mlx_config_no_model = {"host": "127.0.0.1", "port": 8080}

        def side_effect_for_get(key, default=None):
            if key == 'api_settings':
                mock_api_settings_dict = MagicMock()
                mock_api_settings_dict.get.return_value = mlx_config_no_model  # for 'mlx_lm'
                return mock_api_settings_dict
            return default

        mock_settings_get.side_effect = side_effect_for_get

        chat_with_mlx_lm(input_data=input_data, model=model_arg)

        args, kwargs = mock_openai_call.call_args
        assert kwargs['model_name'] == model_arg  # Model from argument is used
        assert kwargs['api_base_url'] == "http://127.0.0.1:8080/v1"


# Test for empty additional_args in start_mlx_lm_server
@patch('subprocess.Popen')
def test_start_mlx_lm_server_empty_additional_args(mock_popen):
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_popen.return_value = mock_process
    model_path = "mlx-community/test-model"
    host = "127.0.0.1"
    port = 8080

    start_mlx_lm_server(model_path, host, port, additional_args="")  # Empty string
    expected_command = ["python", "-m", "mlx_lm.server", "--model", model_path, "--host", host, "--port", str(port)]
    mock_popen.assert_called_once_with(expected_command, stdout=ANY, stderr=ANY, text=True, bufsize=ANY,
                                       universal_newlines=ANY, env=ANY)

    mock_popen.reset_mock()
    start_mlx_lm_server(model_path, host, port, additional_args="   ")  # Whitespace string
    mock_popen.assert_called_once_with(expected_command, stdout=ANY, stderr=ANY, text=True, bufsize=ANY,
                                       universal_newlines=ANY, env=ANY)

    mock_popen.reset_mock()
    start_mlx_lm_server(model_path, host, port, additional_args=None)  # None
    mock_popen.assert_called_once_with(expected_command, stdout=ANY, stderr=ANY, text=True, bufsize=ANY,
                                       universal_newlines=ANY, env=ANY)


# Test stop_mlx_lm_server when process.terminate() raises an exception
def test_stop_mlx_lm_server_terminate_exception():
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 456
    mock_process.poll.return_value = None  # Running
    mock_process.terminate.side_effect = ProcessLookupError("Process already terminated")
    # Even if terminate says it's gone, kill might be tried if poll says otherwise.
    # Or, if terminate fails, kill path is taken. Let's assume poll is reliable.

    # If terminate raises ProcessLookupError, it means the process is already gone.
    # The function should catch this and proceed as if terminated.
    stop_mlx_lm_server(mock_process)
    mock_process.terminate.assert_called_once()
    mock_process.kill.assert_not_called()  # kill should not be called if terminate confirms it's gone via exception


def test_stop_mlx_lm_server_kill_exception_after_timeout():
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 789
    mock_process.poll.return_value = None  # Running
    mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=10),
                                     None]  # First wait times out, second (after kill) succeeds
    mock_process.kill.side_effect = ProcessLookupError("Process died before kill")  # Kill itself fails

    # Even if kill fails with ProcessLookupError, wait() after kill should be called.
    # The main thing is that an attempt was made.
    stop_mlx_lm_server(mock_process)

    mock_process.terminate.assert_called_once()
    assert mock_process.wait.call_count == 2  # Once after terminate, once after kill
    mock_process.kill.assert_called_once()
    # No crash expected
