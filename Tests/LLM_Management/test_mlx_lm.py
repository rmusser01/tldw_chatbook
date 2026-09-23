from unittest.mock import patch

import pytest

from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import chat_with_mlx_lm
from tldw_chatbook.config import RuntimeConfigSnapshot

# Define exception classes if they don't exist in Chat_Deps
try:
    from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError, ChatProviderError
except ImportError:
    # Define minimal exception classes for testing
    class ChatConfigurationError(Exception):
        pass

    class ChatProviderError(Exception):
        def __init__(self, provider, message, status_code=None):
            self.provider = provider
            self.message = message
            self.status_code = status_code
            super().__init__(f"{provider}: {message}")


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

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
    "api_retry_delay": 1,
}


@pytest.fixture(autouse=True)
def mock_mlx_settings():
    # Patch the public runtime snapshot seam used by local provider calls.
    mock_settings = {"api_settings": {"mlx_lm": def_mlx_settings.copy()}}
    snapshot = RuntimeConfigSnapshot(generation=0, values=mock_settings)
    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.get_runtime_config_snapshot",
        return_value=snapshot,
    ):
        yield mock_settings



# --- Tests for chat_with_mlx_lm ---


@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
)
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
        reasoning_effort=None,
        thinking_budget_tokens=None,
        thinking_wire_key="local_mlx_lm",
        provider_name=None,
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"],
    )


@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
)
def test_chat_with_mlx_lm_args_override_config(mock_openai_call, mock_mlx_settings):
    """Test that function arguments override config values."""
    input_data = [{"role": "user", "content": "Override test"}]
    custom_model = "override/model"
    custom_temp = 0.99
    custom_max_tokens = 512

    # To test host/port override, we would typically pass api_url directly if chat_with_mlx_lm supported it
    # OR, we modify the settings for host/port for this specific test case.
    # For now, let's assume chat_with_mlx_lm always uses its configured host/port for URL.
    # The `api_url` param in chat_with_mlx_lm is for overriding the *entire* URL.

    expected_api_base_url = f"http://{def_mlx_settings['host']}:{def_mlx_settings['port']}/v1"  # Uses config host/port

    chat_with_mlx_lm(
        input_data=input_data,
        model=custom_model,
        temp=custom_temp,
        max_tokens=custom_max_tokens,
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
        reasoning_effort=None,
        thinking_budget_tokens=None,
        thinking_wire_key="local_mlx_lm",
        provider_name=None,
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"],
    )


@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
)
def test_chat_with_mlx_lm_api_url_override(mock_openai_call, mock_mlx_settings):
    """Test that api_url argument overrides config host/port for base URL."""
    input_data = [{"role": "user", "content": "API URL override"}]
    custom_api_url = "http://custom.server:1234/custom_v1_path"  # Full URL

    chat_with_mlx_lm(input_data=input_data, api_url=custom_api_url)

    mock_openai_call.assert_called_once_with(
        api_base_url=custom_api_url.rstrip("/"),  # api_url is passed directly
        model_name=def_mlx_settings["model_path"],  # From config
        input_data=input_data,
        api_key=None,
        temp=def_mlx_settings["temperature"],  # From config
        system_message=None,
        streaming=def_mlx_settings["streaming"],
        max_tokens=def_mlx_settings["max_tokens"],
        top_p=def_mlx_settings["top_p"],
        # ... other params from config or defaults ...
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
        reasoning_effort=None,
        thinking_budget_tokens=None,
        thinking_wire_key="local_mlx_lm",
        provider_name=None,
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"],
    )


def test_chat_with_mlx_lm_missing_model_config():
    """Test ChatConfigurationError if model path is missing."""
    input_data = [{"role": "user", "content": "Test"}]

    # Mock settings to have an empty mlx_lm config for this test
    mock_settings = {
        "api_settings": {
            "mlx_lm": {}  # Empty mlx_lm config
        }
    }

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.get_runtime_config_snapshot",
        return_value=RuntimeConfigSnapshot(generation=0, values=mock_settings),
    ):
        with pytest.raises(
            ChatConfigurationError, match="MLX-LM model path .* is required"
        ):
            chat_with_mlx_lm(input_data=input_data)


@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server",
    side_effect=ChatProviderError("MLX-LM", "Network Error", 503),
)
def test_chat_with_mlx_lm_provider_error(mock_openai_call_error, mock_mlx_settings):
    """Test that ChatProviderError from underlying call propagates."""
    input_data = [{"role": "user", "content": "Test provider error"}]
    with pytest.raises(ChatProviderError) as exc_info:
        chat_with_mlx_lm(input_data=input_data)
    # The ChatProviderError is being raised, which is what we want to test
    # The exact error attributes may vary between our mock and the real implementation
    assert isinstance(exc_info.value, ChatProviderError)


# Add more tests as needed, e.g., for streaming, different combinations of parameters, etc.


# Example of how to test if specific kwargs are passed through
@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
)
def test_chat_with_mlx_lm_kwargs_passthrough(mock_openai_call, mock_mlx_settings):
    input_data = [{"role": "user", "content": "Test kwargs"}]
    custom_seed = 12345
    custom_stop_seq = ["\nUser:", "###"]

    chat_with_mlx_lm(
        input_data=input_data,
        seed=custom_seed,  # Passed as kwarg
        stop=custom_stop_seq,  # Passed as kwarg
    )

    expected_api_base_url = (
        f"http://{def_mlx_settings['host']}:{def_mlx_settings['port']}/v1"
    )

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
        reasoning_effort=None,
        thinking_budget_tokens=None,
        thinking_wire_key="local_mlx_lm",
        provider_name=None,
        timeout=def_mlx_settings["api_timeout"],
        api_retries=def_mlx_settings["api_retries"],
        api_retry_delay=def_mlx_settings["api_retry_delay"],
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

# `chat_with_mlx_lm` does not start the server; it assumes one is already
# running at the configured host/port or api_url. The MLX server lifecycle is
# owned by `Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py`
# and covered there; the duplicate `Local_Inference/mlx_lm_inference_local.py`
# implementation and the eleven tests that pinned it were deleted in
# TASK-32901 (unreachable, and already drifted from the live path).


# Test for host/port missing from config and not overridden by api_url in chat_with_mlx_lm
def test_chat_with_mlx_lm_missing_host_port_config():
    input_data = [{"role": "user", "content": "Test"}]

    # Mock settings to have mlx_lm config missing host/port
    mock_settings = {
        "api_settings": {
            "mlx_lm": {"model_path": "some/model"}  # Missing host/port
        }
    }

    # Since host defaults to 127.0.0.1 and port to 8080 in chat_with_mlx_lm if not in config,
    # this test won't raise ChatConfigurationError unless those defaults are also removed from the function.
    # Instead, it should use the defaults. Let's verify that.
    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.get_runtime_config_snapshot",
        return_value=RuntimeConfigSnapshot(generation=0, values=mock_settings),
    ):
        with patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
        ) as mock_openai_call:
            chat_with_mlx_lm(input_data=input_data)

            args, kwargs = mock_openai_call.call_args
            assert (
                kwargs["api_base_url"] == "http://127.0.0.1:8080/v1"
            )  # Default host/port used
            assert kwargs["model_name"] == "some/model"


# Test for model path missing but provided as argument
@patch(
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local._chat_with_openai_compatible_local_server"
)
def test_chat_with_mlx_lm_model_arg_overrides_missing_config(mock_openai_call):
    input_data = [{"role": "user", "content": "Test"}]
    model_arg = "specific/model_via_arg"

    # Config for mlx_lm exists but 'model_path' or 'model' is missing
    mock_settings = {
        "api_settings": {
            "mlx_lm": {"host": "127.0.0.1", "port": 8080}  # No model_path
        }
    }

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.get_runtime_config_snapshot",
        return_value=RuntimeConfigSnapshot(generation=0, values=mock_settings),
    ):
        chat_with_mlx_lm(input_data=input_data, model=model_arg)

        args, kwargs = mock_openai_call.call_args
        assert kwargs["model_name"] == model_arg  # Model from argument is used
        assert kwargs["api_base_url"] == "http://127.0.0.1:8080/v1"

