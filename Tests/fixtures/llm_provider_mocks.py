# Tests/fixtures/llm_provider_mocks.py
# Description: Mock fixtures for LLM providers and external services
#
"""
LLM Provider Mock Fixtures
==========================

Provides mock fixtures for external LLM services to enable isolated testing.
Includes mocks for KoboldCPP, sync server, and various LLM providers.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
import httpx
import aiohttp
from typing import Dict, Any, List, AsyncGenerator


# ========== KoboldCPP Mock Fixtures ==========

@pytest.fixture
def mock_koboldcpp_server():
    """Mock KoboldCPP server responses."""
    
    class MockKoboldCPPResponse:
        def __init__(self, status_code=200, json_data=None, text_data=None):
            self.status_code = status_code
            self._json_data = json_data or {}
            self._text_data = text_data or ""
            
        def json(self):
            return self._json_data
            
        @property
        def text(self):
            return self._text_data
            
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    message=f"HTTP {self.status_code}",
                    request=MagicMock(),
                    response=self
                )
    
    # Default responses
    responses = {
        '/api/v1/generate': MockKoboldCPPResponse(
            json_data={"results": [{"text": "Test response from KoboldCPP"}]}
        ),
        '/api/extra/version': MockKoboldCPPResponse(
            json_data={"version": "1.2.3", "model": "test-model"}
        ),
        '/api/v1/model': MockKoboldCPPResponse(
            json_data={"result": "test-model-7b"}
        )
    }
    
    def mock_request(method, url, **kwargs):
        path = url.replace("http://localhost:5001", "")
        if path in responses:
            return responses[path]
        return MockKoboldCPPResponse(status_code=404)
    
    with patch('httpx.request', side_effect=mock_request):
        with patch('httpx.get', side_effect=lambda url, **kwargs: mock_request('GET', url, **kwargs)):
            with patch('httpx.post', side_effect=lambda url, **kwargs: mock_request('POST', url, **kwargs)):
                yield responses


@pytest.fixture
def mock_koboldcpp_unavailable():
    """Mock KoboldCPP server being unavailable."""
    
    def mock_request(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")
    
    with patch('httpx.request', side_effect=mock_request):
        with patch('httpx.get', side_effect=mock_request):
            with patch('httpx.post', side_effect=mock_request):
                yield


@pytest.fixture
def skip_if_koboldcpp_unavailable():
    """Skip test if KoboldCPP server is not available."""
    import socket
    
    def is_server_available(host="localhost", port=5001):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    if not is_server_available():
        pytest.skip("KoboldCPP server not available at localhost:5001")


# ========== Sync Server Mock Fixtures ==========

@pytest.fixture
def mock_sync_server():
    """Mock sync server responses."""
    
    class MockSyncResponse:
        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json_data = json_data or {}
            self.headers = {"content-type": "application/json"}
            
        def json(self):
            return self._json_data
            
        async def json(self):  # Async version
            return self._json_data
            
        def raise_for_status(self):
            if self.status_code >= 400:
                raise aiohttp.ClientError(f"HTTP {self.status_code}")
    
    # Mock sync endpoints
    endpoints = {
        '/api/v1/sync/pull': MockSyncResponse(
            json_data={
                "changes": [],
                "last_sync_version": 1,
                "server_time": "2025-01-01T00:00:00Z"
            }
        ),
        '/api/v1/sync/push': MockSyncResponse(
            json_data={
                "accepted": True,
                "conflicts": [],
                "new_version": 2
            }
        ),
        '/api/v1/sync/status': MockSyncResponse(
            json_data={
                "server_version": 1,
                "client_registered": True
            }
        )
    }
    
    async def mock_request(method, url, **kwargs):
        path = url.replace("http://localhost:8000", "")
        if path in endpoints:
            return endpoints[path]
        return MockSyncResponse(status_code=404)
    
    with patch('aiohttp.ClientSession.request', side_effect=mock_request):
        yield endpoints


@pytest.fixture
def skip_sync_server_tests():
    """Skip tests that require sync server."""
    pytest.skip("Sync server integration tests require running sync server")


# ========== LLM Provider Mock Fixtures ==========

@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    
    def create_chat_response(content="Test response"):
        return {
            "choices": [{
                "message": {"content": content, "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 100}
        }
    
    def create_stream_response(content="Test streaming response"):
        chunks = content.split()
        for i, chunk in enumerate(chunks):
            yield {
                "choices": [{
                    "delta": {"content": chunk + " "},
                    "finish_reason": None if i < len(chunks) - 1 else "stop"
                }]
            }
    
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = create_chat_response()
        mock_create.stream = create_stream_response
        yield mock_create


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API responses."""
    
    class MockMessage:
        def __init__(self, content="Test response from Claude"):
            self.content = [{"text": content, "type": "text"}]
            self.usage = {"total_tokens": 100}
    
    with patch('anthropic.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = MockMessage()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_ollama_api():
    """Mock Ollama API responses."""
    
    responses = {
        '/api/chat': {
            "message": {"content": "Test response from Ollama", "role": "assistant"},
            "done": True
        },
        '/api/tags': {
            "models": [
                {"name": "llama3:latest", "size": 4000000000},
                {"name": "mistral:latest", "size": 7000000000}
            ]
        }
    }
    
    def mock_request(method, url, **kwargs):
        path = url.replace("http://localhost:11434", "")
        if path in responses:
            return MagicMock(
                status_code=200,
                json=lambda: responses[path]
            )
        return MagicMock(status_code=404)
    
    with patch('httpx.post', side_effect=lambda url, **kwargs: mock_request('POST', url, **kwargs)):
        with patch('httpx.get', side_effect=lambda url, **kwargs: mock_request('GET', url, **kwargs)):
            yield


# ========== Skip Decorators ==========

def skip_without_api_key(provider: str):
    """Skip test if API key for provider is not available."""
    import os
    
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'groq': 'GROQ_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'google': 'GOOGLE_API_KEY'
    }
    
    env_var = key_mapping.get(provider.lower())
    if env_var and not os.getenv(env_var):
        return pytest.mark.skip(reason=f"{env_var} not set")
    return lambda f: f


# ========== Unified Provider Mock ==========

@pytest.fixture
def mock_all_providers():
    """Mock all LLM providers at once."""
    with patch('httpx.post') as mock_post, \
         patch('httpx.get') as mock_get, \
         patch('aiohttp.ClientSession.request') as mock_aio_request:
        
        # Setup default responses
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "Mocked response"}}
        )
        
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok"}
        )
        
        mock_aio_request.return_value = AsyncMock(
            status=200,
            json=AsyncMock(return_value={"status": "ok"})
        )
        
        yield {
            'post': mock_post,
            'get': mock_get,
            'aio_request': mock_aio_request
        }