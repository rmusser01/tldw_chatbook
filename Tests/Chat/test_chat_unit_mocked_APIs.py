# test_chat_unit_mocked_APIs.py
# Unit tests with mocked external services for chat API functionality
#
# These tests run alongside integration tests to verify functionality without
# requiring actual API connections. They mock external services while testing
# the internal logic of the chat functions.

import pytest
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List, Generator
import json

from tldw_chatbook.Chat.Chat_Functions import chat_api_call, chat, PROVIDER_PARAM_MAP, API_CALL_HANDLERS
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatProviderError, ChatAPIError, ChatConfigurationError
)


class TestMockedChatAPIs:
    """Unit tests for chat API functions with mocked external services"""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response"""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm a helpful assistant."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response"""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello! I'm Claude."
            }],
            "model": "claude-3-opus",
            "stop_reason": "stop",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 15
            }
        }
    
    @pytest.fixture
    def mock_kobold_response(self):
        """Mock KoboldCPP API response"""
        return {
            "results": [{
                "text": "Hello from Kobold!"
            }]
        }
    
    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama API response"""
        return {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Hello from Ollama!",
            "done": True
        }
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_openai_chat_mocked(self, mock_openai_response):
        """Test OpenAI chat with mocked response"""
        mock_chat_openai = Mock(return_value=json.dumps(mock_openai_response))
        mock_chat_openai.__name__ = 'chat_with_openai'  # Add __name__ attribute for logging
        API_CALL_HANDLERS['openai'] = mock_chat_openai
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = chat_api_call(
            api_endpoint="OpenAI",
            messages_payload=messages,
            api_key="test_key",
            model="gpt-3.5-turbo",
            temp=0.7,
            max_tokens=50,
            streaming=False
        )
        
        # Verify the response
        response_dict = json.loads(response)
        assert response_dict["choices"][0]["message"]["content"] == "Hello! I'm a helpful assistant."
        assert response_dict["model"] == "gpt-3.5-turbo"
        
        # Verify the function was called with correct parameters
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args[1]
        assert call_args["model"] == "gpt-3.5-turbo"
        assert call_args["temp"] == 0.7  # Parameter is 'temp' not 'temperature'
        assert call_args["max_tokens"] == 50
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_anthropic_chat_mocked(self, mock_anthropic_response):
        """Test Anthropic chat with mocked response"""
        mock_chat_anthropic = Mock(return_value=json.dumps(mock_anthropic_response))
        mock_chat_anthropic.__name__ = 'chat_with_anthropic'
        API_CALL_HANDLERS['anthropic'] = mock_chat_anthropic
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = chat_api_call(
            api_endpoint="Anthropic",
            messages_payload=messages,
            api_key="test_key",
            model="claude-3-opus",
            temp=0.7,
            max_tokens=50,
            streaming=False
        )
        
        # Verify the response
        response_dict = json.loads(response)
        assert response_dict["content"][0]["text"] == "Hello! I'm Claude."
        assert response_dict["model"] == "claude-3-opus"
        
        # Verify the function was called
        mock_chat_anthropic.assert_called_once()
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_kobold_chat_mocked(self, mock_kobold_response):
        """Test KoboldCPP chat with mocked response"""
        # KoboldCPP returns a different format
        mock_chat_kobold = Mock(return_value=json.dumps({
            "choices": [{
                "message": {
                    "content": mock_kobold_response["results"][0]["text"]
                }
            }]
        }))
        mock_chat_kobold.__name__ = 'chat_with_kobold'
        API_CALL_HANDLERS['koboldcpp'] = mock_chat_kobold
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = chat_api_call(
            api_endpoint="koboldcpp",
            messages_payload=messages,
            api_key=None,
            model=None,
            temp=0.7,
            max_tokens=50,
            streaming=False
        )
        
        # Verify the response
        response_dict = json.loads(response)
        assert response_dict["choices"][0]["message"]["content"] == "Hello from Kobold!"
        
        # Verify the function was called
        mock_chat_kobold.assert_called_once()
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_ollama_chat_mocked(self, mock_ollama_response):
        """Test Ollama chat with mocked response"""
        # Convert Ollama format to OpenAI format
        mock_chat_ollama = Mock(return_value=json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": mock_ollama_response["response"]
                }
            }],
            "model": mock_ollama_response["model"]
        }))
        mock_chat_ollama.__name__ = 'chat_with_ollama'
        API_CALL_HANDLERS['ollama'] = mock_chat_ollama
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = chat_api_call(
            api_endpoint="Ollama",
            messages_payload=messages,
            api_key=None,
            model="llama3",
            temp=0.7,
            max_tokens=50,
            streaming=False
        )
        
        # Verify the response
        response_dict = json.loads(response)
        assert response_dict["choices"][0]["message"]["content"] == "Hello from Ollama!"
        assert response_dict["model"] == "llama3"
        
        # Verify the function was called
        mock_chat_ollama.assert_called_once()
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_streaming_response_mocked(self):
        """Test streaming response with mocked generator"""
        def mock_stream_generator():
            chunks = [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" from"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" OpenAI!"}}]}\n\n',
                'data: [DONE]\n\n'
            ]
            for chunk in chunks:
                yield chunk
        
        mock_chat_openai = Mock(return_value=mock_stream_generator())
        mock_chat_openai.__name__ = 'chat_with_openai'
        API_CALL_HANDLERS['openai'] = mock_chat_openai
        
        messages = [{"role": "user", "content": "Hello!"}]
        response_gen = chat_api_call(
            api_endpoint="OpenAI",
            messages_payload=messages,
            api_key="test_key",
            model="gpt-3.5-turbo",
            temp=0.7,
            max_tokens=50,
            streaming=True
        )
        
        # Collect streaming response
        chunks = list(response_gen)
        assert len(chunks) == 4
        assert 'Hello' in chunks[0]
        assert '[DONE]' in chunks[3]
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_error_handling_mocked(self):
        """Test error handling with mocked errors"""
        # Test authentication error
        mock_chat_openai = Mock(side_effect=ChatAuthenticationError("Invalid API key", provider="openai"))
        mock_chat_openai.__name__ = 'chat_with_openai'
        API_CALL_HANDLERS['openai'] = mock_chat_openai
        
        with pytest.raises(ChatAuthenticationError) as exc_info:
            chat_api_call(
                api_endpoint="OpenAI",
                messages_payload=[{"role": "user", "content": "Hello!"}],
                api_key="invalid_key",
                model="gpt-3.5-turbo",
                temp=0.7,
                max_tokens=50,
                streaming=False
            )
        assert exc_info.value.status_code == 401  # status_code is set by the class
        
        # Test rate limit error
        mock_chat_openai.side_effect = ChatRateLimitError("Rate limit exceeded", provider="openai")
        
        with pytest.raises(ChatRateLimitError) as exc_info:
            chat_api_call(
                api_endpoint="OpenAI",
                messages_payload=[{"role": "user", "content": "Hello!"}],
                api_key="test_key",
                model="gpt-3.5-turbo",
                temp=0.7,
                max_tokens=50,
                streaming=False
            )
        assert exc_info.value.status_code == 429  # status_code is set by the class
    
    def test_all_providers_have_handlers(self):
        """Test that all providers in PROVIDER_PARAM_MAP have handlers"""
        for provider in PROVIDER_PARAM_MAP:
            assert provider in API_CALL_HANDLERS, f"Provider {provider} missing handler"
    
    def test_provider_param_mapping(self):
        """Test that provider parameter mappings exist and are not empty"""
        # Just verify that each provider has a mapping and it's a dictionary
        for provider in API_CALL_HANDLERS:
            assert provider in PROVIDER_PARAM_MAP, f"Provider {provider} missing from PROVIDER_PARAM_MAP"
            param_map = PROVIDER_PARAM_MAP[provider]
            assert isinstance(param_map, dict), f"Provider {provider} param map is not a dictionary"
            assert len(param_map) > 0, f"Provider {provider} param map is empty"


class TestMockedChatUnit:
    """Unit tests with mocked external services to verify error handling"""
    
    @patch('requests.post')
    def test_local_service_connection_handling(self, mock_post):
        """Test handling of local service connection errors"""
        # Mock connection refused error
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # This should raise a configuration error or provider error
        with pytest.raises((ChatConfigurationError, ChatProviderError)):
            chat_api_call(
                api_endpoint="koboldcpp",
                messages_payload=[{"role": "user", "content": "Hello!"}],
                api_key=None,
                model=None,
                temp=0.7,
                max_tokens=50,
                streaming=False
            )
    
    @patch.dict('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_retry_logic_mocked(self):
        """Test that provider errors are properly raised"""
        # Test that server errors are properly raised
        mock_chat_openai = Mock(side_effect=ChatProviderError("Server error", status_code=500, provider="openai"))
        mock_chat_openai.__name__ = 'chat_with_openai'
        API_CALL_HANDLERS['openai'] = mock_chat_openai
        
        # This should raise the provider error
        with pytest.raises(ChatProviderError) as exc_info:
            chat_api_call(
                api_endpoint="OpenAI",
                messages_payload=[{"role": "user", "content": "Hello!"}],
                api_key="test_key",
                model="gpt-3.5-turbo",
                temp=0.7,
                max_tokens=50,
                streaming=False
            )
        
        assert exc_info.value.status_code == 500
        assert "Server error" in str(exc_info.value)


# Test Documentation
"""
Test Dependencies and Requirements:
=================================

This test module tests chat API functionality with mocked external services.

Dependencies:
- Core pytest
- unittest.mock (standard library)
- No external API connections required
- No special environment variables needed

The tests mock:
1. OpenAI API responses
2. Anthropic API responses  
3. Local service responses (KoboldCPP, Ollama)
4. Network errors and API errors
5. Streaming responses

These tests complement the integration tests by:
- Verifying internal logic without external dependencies
- Testing error handling paths
- Ensuring consistent behavior across providers
- Testing edge cases that are hard to reproduce with real APIs

To run only these unit tests:
    pytest Tests/Chat/test_chat_unit_mocked_APIs.py

To run alongside integration tests:
    pytest Tests/Chat/ -k "test_chat"
"""