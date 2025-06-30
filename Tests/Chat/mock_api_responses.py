"""
Mock API responses for Chat tests.

This module provides mock responses for various LLM providers to enable
testing without actual API calls.
"""

from typing import Dict, Any, List, Generator
import json
import time


# OpenAI Mock Responses
OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-3.5-turbo",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "This is a test response from OpenAI."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}

OPENAI_STREAM_CHUNKS = [
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {"content": "This "},
            "finish_reason": None
        }]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {"content": "is "},
            "finish_reason": None
        }]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {"content": "streaming."},
            "finish_reason": None
        }]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
]


# Anthropic Mock Responses
ANTHROPIC_CHAT_RESPONSE = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "content": [{
        "type": "text",
        "text": "This is a test response from Anthropic."
    }],
    "model": "claude-3-opus-20240229",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8
    }
}

ANTHROPIC_STREAM_CHUNKS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-opus-20240229",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0}
        }
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "This "}
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "is "}
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "streaming."}
    },
    {
        "type": "content_block_stop",
        "index": 0
    },
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 8}
    },
    {
        "type": "message_stop"
    }
]


# Google Gemini Mock Responses
GOOGLE_CHAT_RESPONSE = {
    "candidates": [{
        "content": {
            "parts": [{
                "text": "This is a test response from Google Gemini."
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


# Mock Response Factory
class MockAPIResponse:
    """Mock HTTP response object."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
    
    def json(self) -> Dict[str, Any]:
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} error")


class MockStreamResponse:
    """Mock streaming response."""
    
    def __init__(self, chunks: List[Dict[str, Any]], provider: str = "openai"):
        self.chunks = chunks
        self.provider = provider
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.chunks:
            raise StopIteration
        
        chunk = self.chunks.pop(0)
        
        # Format based on provider
        if self.provider == "openai":
            return f"data: {json.dumps(chunk)}\n\n"
        elif self.provider == "anthropic":
            return f"data: {json.dumps(chunk)}\n\n"
        else:
            return json.dumps(chunk)
    
    def iter_lines(self):
        """Iterate over lines for OpenAI-style streaming."""
        for chunk in self.chunks:
            if chunk:  # Skip empty chunks
                yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


def get_mock_response(provider: str, streaming: bool = False) -> Any:
    """
    Get a mock response for the specified provider.
    
    Args:
        provider: The LLM provider name
        streaming: Whether to return a streaming response
        
    Returns:
        Mock response object
    """
    if provider == "openai":
        if streaming:
            return MockStreamResponse(OPENAI_STREAM_CHUNKS.copy(), "openai")
        return MockAPIResponse(OPENAI_CHAT_RESPONSE)
    
    elif provider == "anthropic":
        if streaming:
            return MockStreamResponse(ANTHROPIC_STREAM_CHUNKS.copy(), "anthropic")
        return MockAPIResponse(ANTHROPIC_CHAT_RESPONSE)
    
    elif provider == "google":
        return MockAPIResponse(GOOGLE_CHAT_RESPONSE)
    
    else:
        # Generic response for other providers
        generic_response = {
            "response": "This is a test response.",
            "model": "test-model",
            "usage": {"tokens": 10}
        }
        return MockAPIResponse(generic_response)


def mock_api_call(provider: str, messages: List[Dict], **kwargs) -> str:
    """
    Mock API call that returns appropriate response based on provider.
    
    Args:
        provider: The LLM provider name
        messages: The chat messages
        **kwargs: Additional parameters
        
    Returns:
        Mock response string
    """
    if provider == "openai":
        return OPENAI_CHAT_RESPONSE["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        return ANTHROPIC_CHAT_RESPONSE["content"][0]["text"]
    elif provider == "google":
        return GOOGLE_CHAT_RESPONSE["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "This is a test response."


def mock_streaming_call(provider: str, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
    """
    Mock streaming API call that yields chunks.
    
    Args:
        provider: The LLM provider name
        messages: The chat messages
        **kwargs: Additional parameters
        
    Yields:
        Response chunks
    """
    chunks = []
    
    if provider == "openai":
        chunks = ["This ", "is ", "streaming."]
    elif provider == "anthropic":
        chunks = ["This ", "is ", "streaming."]
    else:
        chunks = ["This ", "is ", "a ", "test."]
    
    for chunk in chunks:
        time.sleep(0.01)  # Simulate network delay
        yield chunk