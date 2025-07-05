# tests/integration/test_chat_integrations.py
from unittest.mock import patch

import pytest
import os
import time  # For potential delays or checking stream behavior
from typing import Dict, Any, List, Generator

import requests

# Ensure this path is correct based on your project structure
from tldw_chatbook.Chat.Chat_Functions import chat_api_call, chat, PROVIDER_PARAM_MAP, API_CALL_HANDLERS
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatProviderError, ChatAPIError, ChatConfigurationError
)

# Import mock fixtures
from Tests.fixtures.llm_provider_mocks import (
    mock_koboldcpp_server, mock_koboldcpp_unavailable,
    skip_if_koboldcpp_unavailable, mock_all_providers
)

# Documentation for test dependencies
"""
Integration Test Dependencies:
============================

These tests require actual API services to be available. They will be skipped
if the required environment variables are not set.

Required environment variables by provider:
- OpenAI: OPENAI_API_KEY, OPENAI_TEST_MODEL
- Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_TEST_MODEL  
- Ollama: OLLAMA_TEST_URL, OLLAMA_TEST_MODEL
- KoboldCPP: KOBOLDCPP_TEST_URL
- Other providers: See get_provider_env_vars() function

For local services (Ollama, KoboldCPP, etc.), the service must be running
at the specified URL for tests to pass.

To run only integration tests that require external services:
    pytest -m integration

To run unit tests with mocked services (no external dependencies):
    pytest Tests/Chat/test_chat_unit_mocked_APIs.py
"""


# HOW TO USE THIS TEST:
# # Set environment variables first (example for bash/zsh)
# export OPENAI_API_KEY="your_openai_key"
# export OPENAI_TEST_MODEL="gpt-3.5-turbo"
# export OLLAMA_TEST_URL="http://localhost:11434"
# export OLLAMA_TEST_MODEL="llama3" # make sure this model is pulled
# export PRIMARY_INTEGRATION_TEST_PROVIDER="ollama"
# # ... etc. for other providers
#
# # Run only integration tests
# pytest -m integration
#
# # Run all tests (unit and integration)
# pytest



# --- Helper Fixtures for Configuration ---

PROVIDERS_TO_TEST = list(API_CALL_HANDLERS.keys())  # Get all configured providers


def get_provider_env_vars(provider_name: str) -> Dict[str, Any]:
    """
    Fetches common configuration for a provider from environment variables.
    Adjust environment variable names as per your conventions.
    """
    config = {
        "api_key": os.getenv(f"{provider_name.upper().replace('-', '_')}_API_KEY"),
        "model": os.getenv(f"{provider_name.upper().replace('-', '_')}_TEST_MODEL"),
        "api_url": os.getenv(f"{provider_name.upper().replace('-', '_')}_TEST_URL"),  # For local/self-hosted
        "api_base": os.getenv(f"{provider_name.upper().replace('-', '_')}_TEST_API_BASE"),  # OpenAI compatible base
    }
    return config


@pytest.fixture(scope="module", params=PROVIDERS_TO_TEST)
def provider_config(request):
    """
    Provides configuration for each provider. Skips if essential config is missing.
    This fixture will run tests that use it once for each provider in PROVIDERS_TO_TEST.
    """
    provider_name = request.param
    env_vars = get_provider_env_vars(provider_name)
    params_map = PROVIDER_PARAM_MAP.get(provider_name, {})

    # Basic requirement: model name (unless provider doesn't need it, e.g. some fixed local models)
    # For cloud services, API key is essential.
    # For local services, URL is essential.

    is_local_type = provider_name in ["ollama", "llama_cpp", "ooba", "koboldcpp", "local-llm", "vllm",
                                      "tabbyapi"]  # Add more if needed

    if not env_vars.get("model") and "model" in params_map:  # If model is an expected param for the handler
        # Some local models might have the model fixed by the server, so model in payload is optional
        # For now, let's assume test model should be specified for clarity.
        if not (is_local_type and provider_name == "koboldcpp"):  # Kobold native doesn't take model in payload
            pytest.skip(f"{provider_name.upper()}_TEST_MODEL not set. Skipping integration test for {provider_name}.")

    if not is_local_type and not env_vars.get("api_key") and "api_key" in params_map:
        pytest.skip(f"{provider_name.upper()}_API_KEY not set. Skipping integration test for {provider_name}.")

    # For OpenAI-compatible local servers, api_base (or api_url) is needed
    if provider_name in ["vllm", "local-llm", "custom-openai-api", "custom-openai-api-2", "aphrodite"] and not (
            env_vars.get("api_url") or env_vars.get("api_base")):
        pytest.skip(f"{provider_name.upper()}_TEST_URL or _TEST_API_BASE not set. Skipping for {provider_name}.")
    if provider_name == "ollama" and not env_vars.get("api_url"):
        pytest.skip(f"OLLAMA_TEST_URL not set. Skipping for {provider_name}.")
    if provider_name == "llama_cpp" and not env_vars.get(
            "api_url"):  # Assuming llama_cpp handler uses api_url from config
        pytest.skip(f"LLAMA_CPP_TEST_URL not set. Skipping for {provider_name}.")
    # Add more specific checks as needed

    # Add a simple connectivity check for local servers
    if is_local_type:
        import requests
        url_to_check = env_vars.get("api_url") or env_vars.get("api_base")
        
        # If no URL is set but it's koboldcpp, check default localhost:5001
        if not url_to_check and provider_name == "koboldcpp":
            url_to_check = "http://localhost:5001/api/v1/generate"
        
        if url_to_check:
            try:
                # For OpenAI compatible, check the root or a common path. /v1/models is often available.
                # Adjust if your local servers have a specific health endpoint.
                health_url = url_to_check.rstrip('/') + "/v1/models" if "openai" in provider_name or provider_name in [
                    "vllm", "local-llm", "ollama", "llama_cpp"] else url_to_check
                if provider_name == "kobold_cpp":  # Kobold native API doesn't have /v1/models
                    health_url = url_to_check.replace("/api/v1/generate", "/api/v1/model")  # Check model endpoint
                elif provider_name == "koboldcpp":  # Fix for koboldcpp (different from kobold_cpp)
                    health_url = url_to_check.rstrip('/') + "/api/v1/model" if "/api/v1" not in url_to_check else url_to_check.replace("/generate", "/model")

                if health_url:
                    requests.get(health_url, timeout=5)
            except requests.exceptions.ConnectionError:
                pytest.skip(f"Could not connect to {provider_name} at {url_to_check}. Skipping integration test.")
            except requests.exceptions.Timeout:
                pytest.skip(f"Connection to {provider_name} at {url_to_check} timed out. Skipping integration test.")
            except Exception as e:
                # Catch any other connection errors
                pytest.skip(f"Connection error for {provider_name} at {url_to_check}: {e}. Skipping integration test.")

    return {"provider_name": provider_name, **env_vars}


# --- Integration Tests ---

@pytest.mark.integration
def test_provider_api_basic_call(provider_config: Dict[str, Any]):
    """
    Tests a basic non-streaming call to the provider.
    """
    provider = provider_config["provider_name"]
    api_key = provider_config.get("api_key")
    model = provider_config.get("model")
    # Use api_url for providers that expect it directly (like ollama, llama_cpp in your handlers)
    # or let chat_api_call resolve it via load_settings if the handler uses that.
    # For this integration test, we can be explicit if the handler takes it.
    # This part depends on how your specific handlers (e.g. chat_with_ollama) get their URL.
    # For now, we assume chat_api_call will handle it or the handler will load it via mocked load_settings
    # (if settings point to the TEST_URL env var).

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "Hello! Respond with just 'Hello'."}]

    # Parameters that are generally safe and low-cost
    params_for_call = {
        "api_endpoint": provider,
        "messages_payload": messages,
        "api_key": api_key,
        "model": model,
        "temp": 0.7,  # Using a common default
        "max_tokens": 20,  # Keep it very short
        "streaming": False,
        "seed": 42,  # For providers that support it
    }

    # Some providers might require specific params or have different names
    # The PROVIDER_PARAM_MAP should handle the name mapping internally in chat_api_call
    # However, if a provider *requires* a param not in the common set above,
    # we might need to add it here conditionally.
    # Example: Anthropic might require max_tokens.
    if provider == "anthropic" and not params_for_call.get("max_tokens"):
        params_for_call["max_tokens"] = 20  # Ensure it's set

    print(f"\nTesting provider: {provider} with model: {model}")

    try:
        response = chat_api_call(**params_for_call)
    except ChatConfigurationError as e:
        # This might happen if the test env var for URL/API base isn't picked up correctly by the handler
        # and load_settings returns a default non-functional one.
        pytest.fail(
            f"ChatConfigurationError for {provider}: {e}. Check if handler is using test URL/Base from env via load_settings if applicable.")
    except ChatProviderError as e:
        # Allow provider errors (5xx) but fail otherwise, could indicate real API issue
        if e.status_code and 500 <= e.status_code <= 599:
            pytest.warning(f"Provider {provider} returned a server error {e.status_code}: {e.message}")
            return  # Consider this a pass with warning for integration, or a skip
        raise  # Re-raise other provider errors
    except ChatRateLimitError:
        pytest.skip(f"Rate limit hit for {provider}. Skipping.")
    except ChatAuthenticationError:
        pytest.fail(f"Authentication failed for {provider}. Check API key: {api_key[:5]}...")

    assert response is not None
    if provider_config["provider_name"] == "kobold_cpp":  # Kobold native format is different
        assert "choices" in response and len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        content = response["choices"][0]["message"].get("content")
        assert isinstance(content, str) and len(content) > 0
        print(f"kobold_cpp Response: {content[:100]}")
    else:  # Assuming OpenAI-compatible response structure for others
        assert "choices" in response and len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        message_obj = response["choices"][0]["message"]
        assert "role" in message_obj and message_obj["role"] == "assistant"
        assert "content" in message_obj
        content = message_obj["content"]
        assert isinstance(content, str) and len(content) > 0
        # For stricter test:
        # assert "hello" in content.lower() # If we asked it to say Hello
        print(f"Provider {provider} Response: {content[:100]}")


@pytest.mark.integration
def test_provider_api_streaming_call(provider_config: Dict[str, Any]):
    """
    Tests a basic streaming call to the provider, if supported.
    """
    provider = provider_config["provider_name"]
    api_key = provider_config.get("api_key")
    model = provider_config.get("model")

    # Some providers/handlers might not support streaming or have issues with it
    if provider == "kobold_cpp":  # Your native Kobold handler forces streaming=False
        pytest.skip(
            f"Streaming test skipped for {provider} as native handler may not support it well via chat_api_call.")
    # Add other providers here if their streaming is known to be problematic or not supported by your handlers

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "Stream three words."}]
    params_for_call = {
        "api_endpoint": provider,
        "messages_payload": messages,
        "api_key": api_key,
        "model": model,
        "temp": 0.7,
        "max_tokens": 15,
        "streaming": True,
        "seed": 43,
    }
    if provider == "anthropic" and not params_for_call.get("max_tokens"):
        params_for_call["max_tokens"] = 15

    print(f"\nTesting STREAMING for provider: {provider} with model: {model}")

    try:
        response_generator = chat_api_call(**params_for_call)
    except ChatConfigurationError as e:
        pytest.fail(f"ChatConfigurationError for {provider} (streaming): {e}.")
    except ChatProviderError as e:
        if e.status_code and 500 <= e.status_code <= 599:
            pytest.warning(f"Provider {provider} (streaming) returned server error {e.status_code}: {e.message}")
            return
        raise
    except ChatRateLimitError:
        pytest.skip(f"Rate limit hit for {provider} (streaming). Skipping.")
    except ChatAuthenticationError:
        pytest.fail(f"Authentication failed for {provider} (streaming). Check API key.")

    assert isinstance(response_generator, Generator), "Response should be a generator for streaming"

    received_content = []
    try:
        for i, chunk_str in enumerate(response_generator):
            assert isinstance(chunk_str, str)
            received_content.append(chunk_str)
            if i >= 5 and provider not in ["openai", "anthropic", "google", "mistral", "groq", "deepseek",
                                           "openrouter"]:  # Limit chunks for very verbose local models unless they are known to be openai-like sse
                # Some local model streams might not have a clear [DONE] or might be too verbose
                print(f"Provider {provider} (streaming): Received {i + 1} chunks, stopping test iteration early.")
                break
            if i >= 20:  # Absolute safety break
                print(f"Provider {provider} (streaming): Received {i + 1} chunks, safety break.")
                break
        # print(f"Provider {provider} (streaming) full content: {''.join(received_content)}")
    except requests.exceptions.ChunkedEncodingError as e:
        pytest.fail(f"ChunkedEncodingError during stream for {provider}: {e}")
    except Exception as e:
        pytest.fail(f"Error consuming stream for {provider}: {e}")

    assert len(received_content) > 0, "Should have received at least one chunk for streaming"

    # For OpenAI-compatible streams, the last meaningful chunk (before potential errors or empty lines)
    # is often 'data: [DONE]\n\n'
    # This check depends on how your _chat_with_openai_compatible_local_server formats the yield
    is_openai_like_stream = provider not in ["kobold_cpp"]  # Add others that don't use OpenAI SSE
    if is_openai_like_stream:
        # Find the last non-empty line
        last_meaningful_chunk = ""
        for chunk in reversed(received_content):
            if chunk.strip():
                last_meaningful_chunk = chunk
                break
        assert "data: [DONE]" in last_meaningful_chunk, f"Stream for {provider} did not end with [DONE] marker. Last chunk: {last_meaningful_chunk}"


# --- Test using the main `chat` coordinator with one provider ---
# This tests the full pipeline including message formatting from `chat`

@pytest.mark.integration
def test_main_chat_function_integration_one_provider():
    # Pick one well-behaved provider that is likely to be configured for testing
    # e.g., Ollama if you have a local server, or OpenAI if key is available
    provider_to_test = os.getenv("PRIMARY_INTEGRATION_TEST_PROVIDER", "ollama")  # Default to ollama

    config = get_provider_env_vars(provider_to_test)
    api_key = config.get("api_key")
    model = config.get("model")
    api_url = config.get("api_url")  # Specific for Ollama if it's the one

    if not model:
        pytest.skip(
            f"PRIMARY_INTEGRATION_TEST_PROVIDER ({provider_to_test}) or its _TEST_MODEL not set. Skipping main chat integration.")
    if provider_to_test != "ollama" and not api_key:  # Assuming cloud providers need a key
        pytest.skip(f"API key for {provider_to_test} not set. Skipping main chat integration.")
    if provider_to_test == "ollama" and not api_url:
        pytest.skip(f"OLLAMA_TEST_URL for {provider_to_test} not set. Skipping main chat integration.")

    print(f"\nTesting main 'chat' function with provider: {provider_to_test}, model: {model}")

    history = [{"role": "user", "content": "What was my first question?"}]
    # For local Ollama, ensure the model is pulled and server is running at OLLAMA_TEST_URL
    # The chat_with_ollama handler would internally use load_settings to get api_url.
    # For integration, ensure load_settings is NOT mocked, or if it is, that it returns the test URL.
    # Better: If chat_with_ollama accepts api_url directly, pass it.
    # Your current chat_api_call doesn't easily pass provider-specific URLs like that.
    # So, this test relies on the handler (e.g., chat_with_ollama) correctly picking up
    # its URL from config, which should ideally be influenced by OLLAMA_TEST_URL.
    # For true integration, it's better if load_settings reads actual env vars for these tests.
    # One way is to NOT mock load_settings for integration tests.

    # To ensure load_settings gets the right test URL for Ollama if it's used:
    # Temporarily override the env var that load_settings might read, if different from OLLAMA_TEST_URL
    # This is a bit complex. Simpler: ensure your actual load_settings prefers OLLAMA_TEST_URL if set.

    with patch.dict(os.environ, {"OLLAMA_API_URL": api_url} if provider_to_test == "ollama" and api_url else {}):
        # This patch.dict assumes your load_settings for ollama reads OLLAMA_API_URL. Adjust if needed.
        try:
            response = chat(
                message="This is my second question. Repeat 'second'.",
                history=history,
                media_content=None,
                selected_parts=[],
                api_endpoint=provider_to_test,
                api_key=api_key,
                model=model,  # Pass model through chat function's model param
                custom_prompt=None,
                temperature=0.5,
                llm_max_tokens=15,
                llm_seed=44,
                streaming=False
            )
        except ChatRateLimitError:
            pytest.skip(f"Rate limit hit for {provider_to_test} in main chat integration. Skipping.")
        except ChatAuthenticationError:
            pytest.fail(f"Authentication failed for {provider_to_test} in main chat integration.")
        except ChatProviderError as e:
            if e.status_code and 500 <= e.status_code <= 599:
                pytest.warning(
                    f"Provider {provider_to_test} (main chat) returned server error {e.status_code}: {e.message}")
                return
            raise

    assert isinstance(response, str)
    assert len(response) > 0
    # assert "second" in response.lower() # This can be flaky
    print(f"Main 'chat' function response from {provider_to_test}: {response[:100]}")