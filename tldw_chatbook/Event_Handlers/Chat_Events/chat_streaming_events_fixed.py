"""
Fixed streaming event handlers that follow Textual best practices.

This module handles streaming LLM responses using Textual's reactive
system and messages instead of direct widget manipulation.
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from textual import work
from textual.worker import get_current_worker
from textual.reactive import reactive
from loguru import logger as loguru_logger

# Import message types
from .chat_messages import (
    LLMResponseChunk,
    LLMResponseCompleted,
    LLMResponseError,
    ChatError,
    ToolCallRequested,
    ToolCallCompleted,
    ToolCallFailed
)

# Import business logic (keep using it)
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.Chat.Chat_Functions import parse_tool_calls_from_response
from tldw_chatbook.Tools import get_tool_executor


# ==================== STREAMING HANDLERS ====================

async def handle_streaming_chunk(app: 'TldwCli', chunk: str, session_id: Optional[str] = None) -> None:
    """
    FIXED: Handle streaming chunk using messages.
    
    This posts a message that widgets will react to, instead of
    directly manipulating widgets.
    """
    # Simply post the chunk message
    app.post_message(LLMResponseChunk(chunk, session_id))
    
    loguru_logger.debug(f"Posted streaming chunk: {len(chunk)} chars")


async def handle_stream_done(
    app: 'TldwCli',
    full_text: str,
    error: Optional[str] = None,
    response_data: Optional[Any] = None,
    session_id: Optional[str] = None
) -> None:
    """
    FIXED: Handle stream completion using messages.
    
    Processes the final response, strips tags if needed,
    handles tool calls, and posts appropriate messages.
    """
    loguru_logger.info(f"Stream done. Text length: {len(full_text)}, Error: {error}")
    
    if error:
        # Post error message
        app.post_message(LLMResponseError(error, session_id))
        return
    
    # Apply thinking tag stripping if enabled
    processed_text = await strip_thinking_tags(app, full_text)
    
    # Check for tool calls
    tool_calls = parse_tool_calls_from_response(response_data) if response_data else None
    
    if tool_calls:
        loguru_logger.info(f"Detected {len(tool_calls)} tool call(s)")
        
        # Post tool call requests
        for tool_call in tool_calls:
            app.post_message(ToolCallRequested(
                tool_call.get('name', 'unknown'),
                tool_call.get('arguments', {})
            ))
        
        # Execute tools in worker
        app.run_worker(
            execute_tools(app, tool_calls, session_id),
            name="tool_executor",
            exclusive=False
        )
    
    # Post completion message
    app.post_message(LLMResponseCompleted(processed_text, session_id))
    
    # Save to database if needed
    if should_save_to_db(app, session_id):
        app.run_worker(
            save_stream_to_db(app, processed_text, session_id),
            name="db_saver",
            exclusive=False
        )


# ==================== TOOL EXECUTION ====================

@work(thread=True)
def execute_tools(
    app: 'TldwCli',
    tool_calls: List[Dict[str, Any]],
    session_id: Optional[str] = None
) -> None:
    """
    Execute tools in a background worker.
    
    Posts messages for results instead of direct manipulation.
    """
    executor = get_tool_executor()
    
    try:
        results = executor.execute_tool_calls_sync(tool_calls)
        loguru_logger.info(f"Tool execution completed with {len(results)} result(s)")
        
        # Post results
        for i, result in enumerate(results):
            if i < len(tool_calls):
                tool_name = tool_calls[i].get('name', 'unknown')
                
                if result.get('error'):
                    app.post_message(ToolCallFailed(
                        tool_name,
                        result['error']
                    ))
                else:
                    app.post_message(ToolCallCompleted(
                        tool_name,
                        result.get('result')
                    ))
        
        # Save tool messages to DB if applicable
        if should_save_to_db(app, session_id):
            save_tool_messages_to_db(app, tool_calls, results, session_id)
            
    except Exception as e:
        loguru_logger.error(f"Error executing tools: {e}", exc_info=True)
        app.post_message(ChatError(f"Tool execution error: {str(e)}"))


# ==================== DATABASE OPERATIONS ====================

@work(thread=True)
def save_stream_to_db(
    app: 'TldwCli',
    text: str,
    session_id: Optional[str] = None
) -> None:
    """
    Save streamed response to database in background.
    
    Runs in thread worker to avoid blocking.
    """
    if not hasattr(app, 'chachanotes_db') or not app.chachanotes_db:
        return
    
    if not hasattr(app, 'current_chat_conversation_id') or not app.current_chat_conversation_id:
        return
    
    if getattr(app, 'current_chat_is_ephemeral', False):
        loguru_logger.debug("Chat is ephemeral, not saving to DB")
        return
    
    try:
        # Determine sender name
        sender_name = "AI"
        if hasattr(app, 'active_character_data') and app.active_character_data:
            sender_name = app.active_character_data.get('name', 'AI')
        
        # Save message
        msg_id = ccl.add_message_to_conversation(
            app.chachanotes_db,
            app.current_chat_conversation_id,
            sender_name,
            text
        )
        
        if msg_id:
            loguru_logger.info(f"Saved stream to DB with ID: {msg_id}")
        else:
            loguru_logger.warning("Failed to save stream to DB")
            
    except Exception as e:
        loguru_logger.error(f"Error saving stream to DB: {e}", exc_info=True)


def save_tool_messages_to_db(
    app: 'TldwCli',
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    session_id: Optional[str] = None
) -> None:
    """
    Save tool call and result messages to database.
    """
    if not hasattr(app, 'chachanotes_db') or not app.chachanotes_db:
        return
    
    if not hasattr(app, 'current_chat_conversation_id') or not app.current_chat_conversation_id:
        return
    
    if getattr(app, 'current_chat_is_ephemeral', False):
        return
    
    try:
        # Save tool call message
        tool_call_msg = f"Tool Calls:\n{json.dumps(tool_calls, indent=2)}"
        tool_call_id = ccl.add_message_to_conversation(
            app.chachanotes_db,
            app.current_chat_conversation_id,
            "tool",
            tool_call_msg
        )
        loguru_logger.debug(f"Saved tool call to DB: {tool_call_id}")
        
        # Save tool results message
        tool_results_msg = f"Tool Results:\n{json.dumps(results, indent=2)}"
        tool_result_id = ccl.add_message_to_conversation(
            app.chachanotes_db,
            app.current_chat_conversation_id,
            "tool",
            tool_results_msg
        )
        loguru_logger.debug(f"Saved tool results to DB: {tool_result_id}")
        
    except Exception as e:
        loguru_logger.error(f"Error saving tool messages to DB: {e}", exc_info=True)


# ==================== HELPER FUNCTIONS ====================

async def strip_thinking_tags(app: 'TldwCli', text: str) -> str:
    """
    Strip thinking tags from response if configured.
    
    Removes <think> and <thinking> blocks except the last one.
    """
    if not text:
        return text
    
    # Check configuration
    strip_tags = app.app_config.get("chat_defaults", {}).get("strip_thinking_tags", True)
    
    if not strip_tags:
        loguru_logger.debug("Tag stripping disabled in config")
        return text
    
    # Find all thinking blocks
    think_blocks = list(re.finditer(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        text,
        re.DOTALL
    ))
    
    if len(think_blocks) <= 1:
        loguru_logger.debug(f"Found {len(think_blocks)} thinking block(s), not stripping")
        return text
    
    loguru_logger.debug(f"Stripping {len(think_blocks) - 1} thinking blocks")
    
    # Keep text between blocks and after last block
    text_parts = []
    last_kept_block_end = 0
    
    for i, block in enumerate(think_blocks):
        if i < len(think_blocks) - 1:  # Remove this block
            text_parts.append(text[last_kept_block_end:block.start()])
            last_kept_block_end = block.end()
    
    # Add remaining text after last removed block
    text_parts.append(text[last_kept_block_end:])
    
    return "".join(text_parts)


def should_save_to_db(app: 'TldwCli', session_id: Optional[str] = None) -> bool:
    """
    Check if we should save to database.
    """
    if not hasattr(app, 'chachanotes_db') or not app.chachanotes_db:
        return False
    
    if not hasattr(app, 'current_chat_conversation_id') or not app.current_chat_conversation_id:
        return False
    
    if getattr(app, 'current_chat_is_ephemeral', False):
        return False
    
    return True


# ==================== CONTINUATION HANDLING ====================

async def handle_continue_streaming(
    app: 'TldwCli',
    message_id: str,
    partial_text: str,
    session_id: Optional[str] = None
) -> None:
    """
    FIXED: Handle continuation of a partial response.
    
    Posts messages instead of direct manipulation.
    """
    from .chat_messages import ContinueResponseRequested
    
    # Post continuation request
    app.post_message(ContinueResponseRequested(message_id, partial_text))
    
    loguru_logger.info(f"Continuation requested for message {message_id}")


# ==================== EXPORT WRAPPER ====================

class StreamingEventsNamespace:
    """Namespace for backward compatibility."""
    
    handle_streaming_chunk = staticmethod(handle_streaming_chunk)
    handle_stream_done = staticmethod(handle_stream_done)
    handle_continue_streaming = staticmethod(handle_continue_streaming)
    execute_tools = staticmethod(execute_tools)
    strip_thinking_tags = staticmethod(strip_thinking_tags)

# Export for backward compatibility
streaming_events = StreamingEventsNamespace()