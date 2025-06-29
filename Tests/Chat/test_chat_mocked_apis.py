"""
Unit tests for Chat API functions using mock responses.

These tests don't require actual API keys or external services.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from typing import List, Dict, Any

# Import the functions we're testing
from tldw_chatbook.Chat.Chat_Functions import chat_api_call, chat
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatProviderError, ChatAPIError
)

# Import our mock responses
from .mock_api_responses import (
    get_mock_response, mock_api_call, mock_streaming_call,
    MockAPIResponse, MockStreamResponse,
    OPENAI_CHAT_RESPONSE, ANTHROPIC_CHAT_RESPONSE
)


@pytest.mark.unit
class TestMockedChatAPIs:
    """Test chat functions with mocked API responses."""
    
    @pytest.fixture
    def mock_messages(self):
        """Standard test messages."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
    
    @patch('requests.Session.post')  # Patch the session.post method
    def test_openai_chat_mocked(self, mock_post, mock_messages):
        """Test OpenAI chat with mocked response."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = OPENAI_CHAT_RESPONSE
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Make the call
        result = chat_api_call(
            "openai",
            mock_messages,
            api_key="test-key",
            model="gpt-3.5-turbo"
        )
        
        # Verify - chat_api_call returns the full response object
        assert result == OPENAI_CHAT_RESPONSE
        mock_post.assert_called_once()
        
        # Check the call arguments
        call_args = mock_post.call_args
        assert call_args[1]['headers']['Authorization'] == 'Bearer test-key'
        assert call_args[1]['json']['model'] == 'gpt-3.5-turbo'
        assert call_args[1]['json']['messages'] == mock_messages
    
    @patch('requests.Session.post')
    def test_anthropic_chat_mocked(self, mock_post, mock_messages):
        """Test Anthropic chat with mocked response."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = ANTHROPIC_CHAT_RESPONSE
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Make the call
        result = chat_api_call(
            "anthropic",
            mock_messages,
            api_key="test-key",
            model="claude-3-opus-20240229"
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "This is a test response from Anthropic."
        mock_post.assert_called_once()
        
        # Check headers
        call_args = mock_post.call_args
        assert call_args[1]['headers']['x-api-key'] == 'test-key'
        assert call_args[1]['headers']['anthropic-version'] == '2023-06-01'
    
    @patch('requests.Session')
    def test_openai_streaming_mocked(self, mock_session_class, mock_messages):
        """Test OpenAI streaming with mocked response."""
        # Setup mock session
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Setup streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Make the call
        result = chat_api_call(
            "openai",
            mock_messages,
            api_key="test-key",
            model="gpt-3.5-turbo",
            streaming=True
        )
        
        # Collect streamed content
        content = []
        for chunk in result:
            # The streaming response includes the full SSE lines
            chunk_str = chunk.strip() if isinstance(chunk, str) else chunk
            if chunk_str and chunk_str.startswith('data: ') and chunk_str != 'data: [DONE]':
                try:
                    import json
                    json_str = chunk_str[6:]  # Remove 'data: ' prefix
                    if json_str:
                        data = json.loads(json_str)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content.append(delta['content'])
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON
        
        assert content == ["Hello", " world"]
    
    @patch('requests.Session.post')
    def test_api_error_handling(self, mock_post):
        """Test error handling for API failures."""
        # Test 401 Unauthorized
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")
        mock_post.return_value = mock_response
        
        with pytest.raises(ChatAuthenticationError):
            chat_api_call(
                "openai",
                [{"role": "user", "content": "test"}],
                api_key="invalid-key"
            )
        
        # Test 429 Rate Limit
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("Rate limited")
        
        with pytest.raises(ChatRateLimitError):
            chat_api_call(
                "openai",
                [{"role": "user", "content": "test"}],
                api_key="test-key"
            )
    
    @patch('tldw_chatbook.LLM_Calls.LLM_API_Calls.chat_with_openai')
    def test_chat_function_with_mock(self, mock_openai, mock_messages):
        """Test the main chat function with mocked provider."""
        # Setup mock
        mock_openai.return_value = "Mocked response"
        
        # Make the call
        result = chat(
            mock_messages,
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_endpoint="openai"
        )
        
        # Verify
        assert result == "Mocked response"
        mock_openai.assert_called_once_with(
            mock_messages,
            "test-key",
            "gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            streaming=False
        )
    
    def test_provider_parameter_mapping(self):
        """Test that provider parameters are correctly mapped."""
        from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP
        
        # Check that major providers have mappings
        assert "openai" in PROVIDER_PARAM_MAP
        assert "anthropic" in PROVIDER_PARAM_MAP
        assert "google" in PROVIDER_PARAM_MAP
        
        # Check specific mappings
        openai_map = PROVIDER_PARAM_MAP["openai"]
        assert openai_map["max_tokens"] == "max_tokens"
        assert openai_map["temperature"] == "temperature"
        
        anthropic_map = PROVIDER_PARAM_MAP["anthropic"]
        assert anthropic_map["max_tokens"] == "max_tokens"
        assert anthropic_map["temperature"] == "temperature"


@pytest.mark.unit
class TestMockedStreamingAPIs:
    """Test streaming functionality with mocks."""
    
    @patch('requests.Session')
    def test_streaming_chunk_processing(self, mock_session_class):
        """Test that streaming chunks are processed correctly."""
        # Setup
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Create a more complex streaming response
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            'data: {"choices":[{"delta":{"content":"The"}}]}',
            'data: {"choices":[{"delta":{"content":" answer"}}]}',
            'data: {"choices":[{"delta":{"content":" is"}}]}',
            'data: {"choices":[{"delta":{"content":" 42"}}]}',
            'data: {"choices":[{"delta":{"content":"."}}]}',
            'data: [DONE]'
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = chunks
        # Mock decode_unicode parameter
        mock_response.iter_lines.side_effect = lambda decode_unicode=False: chunks
        mock_session.post.return_value = mock_response
        
        # Make the call
        result = chat_api_call(
            "openai",
            [{"role": "user", "content": "What is the answer?"}],
            api_key="test-key",
            model="gpt-3.5-turbo",
            streaming=True
        )
        
        # Collect and verify
        content = []
        for chunk in result:
            # The streaming response includes the full SSE lines
            chunk_str = chunk.strip() if isinstance(chunk, str) else chunk
            if chunk_str and chunk_str.startswith('data: ') and chunk_str != 'data: [DONE]':
                try:
                    import json
                    json_str = chunk_str[6:]  # Remove 'data: ' prefix
                    if json_str:
                        data = json.loads(json_str)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content.append(delta['content'])
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON
        
        assert "".join(content) == "The answer is 42."
    
    @patch('requests.Session')
    def test_streaming_error_handling(self, mock_client_class):
        """Test error handling during streaming."""
        # Setup
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        # Simulate error in stream
        def error_generator():
            yield 'data: {"choices":[{"delta":{"content":"Start"}}]}'
            raise Exception("Stream interrupted")
        
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = error_generator()
        mock_client.post.return_value = mock_response
        
        # Make the call and expect error
        with pytest.raises(ChatAPIError):
            result = chat_api_call(
                "openai",
                [{"role": "user", "content": "test"}],
                api_key="test-key",
                streaming=True
            )
            # Consume the generator to trigger the error
            list(result)