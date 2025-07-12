"""
api_test_utilities.py
---------------------

API testing utilities for mocking LLM providers and HTTP responses.
Provides comprehensive mocking patterns for all supported LLM APIs.
"""

import pytest
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import httpx


# ===========================================
# LLM Provider Response Mocks
# ===========================================

class LLMProviderMocks:
    """Mock responses for different LLM providers."""
    
    @staticmethod
    def openai_chat_response(
        content: str = "Test response",
        model: str = "gpt-3.5-turbo",
        finish_reason: str = "stop",
        tokens: Dict[str, int] = None
    ) -> Dict:
        """Create OpenAI chat completion response."""
        if tokens is None:
            tokens = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }],
            "usage": tokens
        }
    
    @staticmethod
    def openai_stream_chunk(
        content: str,
        finish_reason: Optional[str] = None
    ) -> Dict:
        """Create OpenAI streaming chunk."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }]
        }
    
    @staticmethod
    def anthropic_response(
        content: str = "Test response",
        model: str = "claude-3-opus-20240229",
        stop_reason: str = "end_turn"
    ) -> Dict:
        """Create Anthropic response."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": content
            }],
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }
    
    @staticmethod
    def anthropic_stream_chunk(content: str) -> Dict:
        """Create Anthropic streaming chunk."""
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": content
            }
        }
    
    @staticmethod
    def google_response(
        content: str = "Test response",
        model: str = "gemini-pro"
    ) -> Dict:
        """Create Google Gemini response."""
        return {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": content
                    }],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }],
            "promptFeedback": {
                "safetyRatings": []
            }
        }
    
    @staticmethod
    def cohere_response(
        content: str = "Test response",
        model: str = "command"
    ) -> Dict:
        """Create Cohere response."""
        return {
            "id": "123",
            "text": content,
            "generation_id": "gen_123",
            "model": model,
            "citations": [],
            "documents": [],
            "search_queries": [],
            "search_results": [],
            "finish_reason": "COMPLETE",
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": 10, "output_tokens": 20}
            }
        }
    
    @staticmethod
    def local_openai_response(
        content: str = "Test response",
        model: str = "local-model"
    ) -> Dict:
        """Create response for local OpenAI-compatible APIs."""
        return {
            "id": "local-123",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }


@pytest.fixture
def llm_provider_mocks():
    """Provide LLM provider mock factory."""
    return LLMProviderMocks()


# ===========================================
# Streaming Response Mocks
# ===========================================

class StreamingMocks:
    """Mock streaming responses for different providers."""
    
    @staticmethod
    async def openai_stream(
        text: str,
        chunk_size: int = 10,
        delay: float = 0.01
    ) -> AsyncGenerator[bytes, None]:
        """Generate OpenAI streaming response."""
        # First chunk with role
        first_chunk = LLMProviderMocks.openai_stream_chunk("")
        first_chunk["choices"][0]["delta"]["role"] = "assistant"
        yield f"data: {json.dumps(first_chunk)}\n\n".encode()
        await asyncio.sleep(delay)
        
        # Content chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunk = LLMProviderMocks.openai_stream_chunk(chunk_text)
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            await asyncio.sleep(delay)
        
        # Final chunk
        final_chunk = LLMProviderMocks.openai_stream_chunk("", "stop")
        yield f"data: {json.dumps(final_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    
    @staticmethod
    async def anthropic_stream(
        text: str,
        chunk_size: int = 10,
        delay: float = 0.01
    ) -> AsyncGenerator[bytes, None]:
        """Generate Anthropic streaming response."""
        # Start event
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_123', 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'claude-3-opus-20240229'}})}\n\n".encode()
        await asyncio.sleep(delay)
        
        # Content block start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()
        await asyncio.sleep(delay)
        
        # Content chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunk = LLMProviderMocks.anthropic_stream_chunk(chunk_text)
            yield f"event: content_block_delta\ndata: {json.dumps(chunk)}\n\n".encode()
            await asyncio.sleep(delay)
        
        # End events
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}})}\n\n".encode()
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()


@pytest.fixture
def streaming_mocks():
    """Provide streaming mock factory."""
    return StreamingMocks()


# ===========================================
# HTTP Client Mocks
# ===========================================

class MockHTTPXResponse:
    """Enhanced mock httpx response with streaming support."""
    
    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict] = None,
        text: str = "",
        headers: Optional[Dict] = None,
        stream_data: Optional[AsyncGenerator] = None
    ):
        self.status_code = status_code
        self.json_data = json_data
        self.text = text
        self.headers = headers or {}
        self.stream_data = stream_data
        self.is_closed = False
    
    def json(self) -> Dict:
        if self.json_data is not None:
            return self.json_data
        raise ValueError("No JSON data")
    
    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=self
            )
    
    async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        """Async iteration over stream data."""
        if self.stream_data:
            async for chunk in self.stream_data:
                yield chunk
        else:
            yield self.text.encode()
    
    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        """Async iteration over lines."""
        buffer = ""
        async for chunk in self.aiter_bytes():
            buffer += chunk.decode()
            lines = buffer.split('\n')
            buffer = lines[-1]
            for line in lines[:-1]:
                yield line
        if buffer:
            yield buffer
    
    async def aclose(self):
        """Close the response."""
        self.is_closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.aclose()


@pytest.fixture
def mock_httpx_factory():
    """Factory for creating mock httpx clients."""
    def _create_client(default_responses: Optional[Dict] = None):
        client = AsyncMock()
        
        # Setup context manager
        async def async_enter():
            return client
        
        async def async_exit(*args):
            pass
        
        client.__aenter__ = async_enter
        client.__aexit__ = async_exit
        
        # Default responses
        responses = default_responses or {}
        
        # Setup response routing
        async def mock_post(url: str, **kwargs):
            # Route based on URL
            if "openai.com" in url:
                if "stream" in kwargs.get("json", {}) and kwargs["json"]["stream"]:
                    text = responses.get("openai_stream_text", "Streaming response")
                    return MockHTTPXResponse(
                        stream_data=StreamingMocks.openai_stream(text)
                    )
                else:
                    return MockHTTPXResponse(
                        json_data=responses.get(
                            "openai",
                            LLMProviderMocks.openai_chat_response()
                        )
                    )
            elif "anthropic.com" in url:
                if kwargs.get("json", {}).get("stream"):
                    text = responses.get("anthropic_stream_text", "Streaming response")
                    return MockHTTPXResponse(
                        stream_data=StreamingMocks.anthropic_stream(text)
                    )
                else:
                    return MockHTTPXResponse(
                        json_data=responses.get(
                            "anthropic",
                            LLMProviderMocks.anthropic_response()
                        )
                    )
            # Default response
            return MockHTTPXResponse(json_data={"result": "success"})
        
        client.post = mock_post
        client.get = AsyncMock(return_value=MockHTTPXResponse(text="OK"))
        
        return client
    
    return _create_client


# ===========================================
# Error Response Mocks
# ===========================================

class APIErrorMocks:
    """Mock error responses for different scenarios."""
    
    @staticmethod
    def rate_limit_error(provider: str = "openai") -> Dict:
        """Create rate limit error response."""
        if provider == "openai":
            return {
                "error": {
                    "message": "Rate limit reached for gpt-3.5-turbo",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded"
                }
            }
        elif provider == "anthropic":
            return {
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded"
                }
            }
        return {"error": "Rate limit exceeded"}
    
    @staticmethod
    def auth_error(provider: str = "openai") -> Dict:
        """Create authentication error response."""
        if provider == "openai":
            return {
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        elif provider == "anthropic":
            return {
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            }
        return {"error": "Authentication failed"}
    
    @staticmethod
    def model_not_found_error(model: str, provider: str = "openai") -> Dict:
        """Create model not found error."""
        if provider == "openai":
            return {
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        return {"error": f"Model '{model}' not found"}
    
    @staticmethod
    def context_length_error(provider: str = "openai") -> Dict:
        """Create context length exceeded error."""
        if provider == "openai":
            return {
                "error": {
                    "message": "This model's maximum context length is 4097 tokens",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded"
                }
            }
        return {"error": "Context length exceeded"}


@pytest.fixture
def api_error_mocks():
    """Provide API error mock factory."""
    return APIErrorMocks()


# ===========================================
# Request/Response Interceptors
# ===========================================

@pytest.fixture
def api_request_interceptor():
    """Intercept and validate API requests."""
    class RequestInterceptor:
        def __init__(self):
            self.requests = []
        
        def intercept(self, url: str, **kwargs):
            """Record request details."""
            request = {
                "url": url,
                "method": kwargs.get("method", "POST"),
                "headers": kwargs.get("headers", {}),
                "json": kwargs.get("json"),
                "params": kwargs.get("params"),
                "timestamp": datetime.now().isoformat()
            }
            self.requests.append(request)
            return request
        
        def assert_request_made(self, url_pattern: str, json_contains: Optional[Dict] = None):
            """Assert a request was made matching criteria."""
            matching = [r for r in self.requests if url_pattern in r["url"]]
            assert matching, f"No request made to URL containing '{url_pattern}'"
            
            if json_contains:
                for req in matching:
                    if req.get("json"):
                        for key, value in json_contains.items():
                            if req["json"].get(key) == value:
                                return
                assert False, f"No request found with JSON containing {json_contains}"
        
        def get_last_request(self) -> Optional[Dict]:
            """Get the most recent request."""
            return self.requests[-1] if self.requests else None
        
        def clear(self):
            """Clear recorded requests."""
            self.requests.clear()
    
    return RequestInterceptor()


# ===========================================
# Mock API Test Scenarios
# ===========================================

@pytest.fixture
def mock_api_scenarios():
    """Common API test scenarios."""
    class APIScenarios:
        @staticmethod
        def success_flow(provider: str = "openai"):
            """Successful API call scenario."""
            if provider == "openai":
                return LLMProviderMocks.openai_chat_response("Success!")
            elif provider == "anthropic":
                return LLMProviderMocks.anthropic_response("Success!")
            return {"response": "Success!"}
        
        @staticmethod
        def retry_then_success(provider: str = "openai", failures: int = 2):
            """Fail N times then succeed."""
            attempt = 0
            
            def response_func():
                nonlocal attempt
                attempt += 1
                if attempt <= failures:
                    raise httpx.HTTPStatusError(
                        "Server error",
                        request=MagicMock(),
                        response=MockHTTPXResponse(status_code=500)
                    )
                return APIScenarios.success_flow(provider)
            
            return response_func
        
        @staticmethod
        def gradual_degradation():
            """Simulate gradual service degradation."""
            delays = [0.1, 0.5, 1.0, 2.0, 5.0]
            attempt = 0
            
            async def response_func():
                nonlocal attempt
                if attempt < len(delays):
                    await asyncio.sleep(delays[attempt])
                    attempt += 1
                return APIScenarios.success_flow()
            
            return response_func
    
    return APIScenarios()


# ===========================================
# Integration Test Helpers
# ===========================================

@pytest.fixture
def api_integration_helper():
    """Helper for API integration testing."""
    class APIIntegrationHelper:
        def __init__(self):
            self.mock_responses = {}
        
        def setup_provider(self, provider: str, responses: List[Dict]):
            """Setup mock responses for a provider."""
            self.mock_responses[provider] = responses
        
        @contextmanager
        def mock_all_providers(self):
            """Mock all LLM provider HTTP calls."""
            with patch('httpx.AsyncClient') as mock_client:
                client_instance = mock_httpx_factory()(self.mock_responses)
                mock_client.return_value = client_instance
                yield client_instance
        
        async def test_provider_call(
            self,
            provider_func: Callable,
            expected_content: str,
            **kwargs
        ):
            """Test a provider function call."""
            with self.mock_all_providers():
                result = await provider_func(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                assert expected_content in str(result)
    
    return APIIntegrationHelper()


# ===========================================
# Example Usage
# ===========================================

"""
Example Usage:

1. Basic Provider Mocking:
   ```python
   def test_openai_call(llm_provider_mocks):
       response = llm_provider_mocks.openai_chat_response("Hello, world!")
       assert response["choices"][0]["message"]["content"] == "Hello, world!"
   ```

2. Streaming Response Testing:
   ```python
   async def test_streaming(streaming_mocks):
       text = "This is a streaming response"
       async for chunk in streaming_mocks.openai_stream(text):
           # Process chunk
   ```

3. Error Handling:
   ```python
   def test_rate_limit(api_error_mocks, mock_httpx_factory):
       client = mock_httpx_factory({
           "openai": api_error_mocks.rate_limit_error("openai")
       })
       # Test rate limit handling
   ```

4. Request Validation:
   ```python
   def test_api_request(api_request_interceptor):
       # Make API call
       api_request_interceptor.assert_request_made(
           "openai.com",
           json_contains={"model": "gpt-3.5-turbo"}
       )
   ```

5. Integration Testing:
   ```python
   async def test_integration(api_integration_helper):
       api_integration_helper.setup_provider(
           "openai",
           [llm_provider_mocks.openai_chat_response()]
       )
       await api_integration_helper.test_provider_call(
           my_provider_func,
           expected_content="Test response"
       )
   ```
"""