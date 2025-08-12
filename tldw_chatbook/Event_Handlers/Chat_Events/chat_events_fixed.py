"""
Fixed chat event handlers that follow Textual best practices.

This is a transitional version that maintains backward compatibility
while removing direct widget manipulation.
"""

import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path
import uuid
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union

from loguru import logger as loguru_logger
from rich.text import Text
from textual import work
from textual.widgets import Button, Input, TextArea, Select, Checkbox
from textual.worker import get_current_worker
from textual.css.query import QueryError

# Import the new message system
from .chat_messages import (
    UserMessageSent,
    LLMResponseStarted,
    LLMResponseChunk,
    LLMResponseCompleted,
    LLMResponseError,
    ChatError,
    SessionLoaded,
    CharacterLoaded,
    RAGResultsReceived,
    TokenCountUpdated
)

# Import existing business logic (keep using it)
from tldw_chatbook.Utils.Utils import safe_float, safe_int
from tldw_chatbook.Utils.input_validation import validate_text_input, validate_number_range, sanitize_string
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError, InputError
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


# ==================== FIXED HANDLERS ====================

async def handle_chat_send_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Send button handler that uses messages instead of direct manipulation.
    """
    prefix = "chat"
    start_time = time.time()
    
    # Log button click event
    log_counter("chat_ui_send_button_clicked", labels={"tab": prefix})
    
    # Check if there's an active chat generation running
    if hasattr(app, 'current_chat_worker') and app.current_chat_worker and app.current_chat_worker.is_running:
        loguru_logger.info("Send button pressed - stopping active generation")
        log_counter("chat_ui_generation_cancelled", labels={"tab": prefix})
        await handle_stop_chat_generation_pressed(app, event)
        return
    
    loguru_logger.info(f"Send button pressed for '{prefix}' (main chat)")
    
    # Get the message text (this is the ONLY query we need)
    try:
        text_area = app.query_one(f"#{prefix}-input", TextArea)
        message_text = text_area.text.strip()
    except QueryError as e:
        loguru_logger.error(f"Could not find input area: {e}")
        app.post_message(ChatError("Could not find input area"))
        return
    
    # Validate message
    if message_text:
        if not validate_text_input(message_text, max_length=100000, allow_html=False):
            app.post_message(ChatError("Message contains invalid content or is too long."))
            loguru_logger.warning("Invalid user message input rejected")
            log_counter("chat_ui_message_validation_failed", labels={"tab": prefix})
            return
        
        message_text = sanitize_string(message_text, max_length=100000)
        log_histogram("chat_ui_message_length", len(message_text), labels={"tab": prefix})
    
    if not message_text:
        # Handle empty message - check for resend
        if await should_resend_last_message(app):
            message_text = await get_last_user_message(app)
            if not message_text:
                app.post_message(ChatError("No message to send"))
                return
        else:
            app.post_message(ChatError("Please enter a message"))
            return
    
    # Get configuration
    config = await get_chat_configuration(app, prefix)
    if not config:
        app.post_message(ChatError("Could not get chat configuration"))
        return
    
    # Clear the input
    text_area.clear()
    
    # Get attachments if any
    attachments = await get_pending_attachments(app)
    
    # Post the message to trigger processing
    app.post_message(UserMessageSent(message_text, attachments))
    
    # Start processing in a worker
    app.run_worker(
        process_chat_message(app, message_text, config, attachments),
        name="chat_message_processor",
        exclusive=True  # Cancel any existing chat processing
    )


@work(exclusive=True)
async def process_chat_message(
    app: 'TldwCli',
    message: str,
    config: Dict[str, Any],
    attachments: Optional[List[str]] = None
) -> None:
    """
    Process chat message in a worker (non-blocking).
    """
    worker = get_current_worker()
    
    try:
        # Post that we're starting
        app.call_from_thread(app.post_message, LLMResponseStarted())
        
        # Get chat history (using existing functions)
        from tldw_chatbook.Chat.Chat_Functions import approximate_token_count
        
        history = await get_chat_history_async(app)
        
        # Check token count
        token_count = approximate_token_count(history)
        max_tokens = config.get('max_tokens', 4096)
        
        app.call_from_thread(
            app.post_message,
            TokenCountUpdated(token_count, max_tokens)
        )
        
        if token_count > max_tokens * 0.9:
            app.call_from_thread(
                app.post_message,
                ChatError(f"Approaching token limit: {token_count}/{max_tokens}", "warning")
            )
        
        # Apply RAG if enabled
        rag_context = await get_rag_context_async(app, message)
        if rag_context:
            message = f"{rag_context}\n\n{message}"
            app.call_from_thread(
                app.post_message,
                RAGResultsReceived([], rag_context)
            )
        
        # Check if cancelled
        if worker.is_cancelled:
            return
        
        # Make the API call with streaming
        await stream_llm_response(app, message, history, config, worker)
        
    except Exception as e:
        loguru_logger.error(f"Error processing message: {e}", exc_info=True)
        app.call_from_thread(
            app.post_message,
            LLMResponseError(str(e))
        )


async def stream_llm_response(
    app: 'TldwCli',
    message: str,
    history: List[Dict],
    config: Dict[str, Any],
    worker: Any
) -> None:
    """
    Stream LLM response using messages instead of direct manipulation.
    """
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    import asyncio
    
    full_response = ""
    
    def stream_callback(chunk: str) -> bool:
        """Callback for streaming chunks."""
        if worker.is_cancelled:
            return False
        
        nonlocal full_response
        full_response += chunk
        
        # Post chunk message
        app.call_from_thread(
            app.post_message,
            LLMResponseChunk(chunk)
        )
        return True
    
    try:
        # Make the API call
        response = await asyncio.to_thread(
            chat_api_call,
            message=message,
            history=history,
            provider=config['provider'],
            model=config['model'],
            system_prompt=config.get('system_prompt', ''),
            temperature=config.get('temperature', 0.7),
            streaming=config.get('streaming', True),
            stream_callback=stream_callback if config.get('streaming', True) else None,
            **config.get('extra_params', {})
        )
        
        # Post completion
        app.call_from_thread(
            app.post_message,
            LLMResponseCompleted(response or full_response)
        )
        
    except Exception as e:
        app.call_from_thread(
            app.post_message,
            LLMResponseError(str(e))
        )


async def handle_stop_chat_generation_pressed(app: 'TldwCli', event: Any) -> None:
    """
    FIXED: Stop generation using worker cancellation.
    """
    # Cancel any chat workers
    cancelled = False
    for worker in app.workers:
        if worker.name and 'chat' in worker.name.lower():
            worker.cancel()
            cancelled = True
            loguru_logger.info(f"Cancelled worker: {worker.name}")
    
    if cancelled:
        app.post_message(ChatError("Generation stopped", "info"))
    else:
        app.post_message(ChatError("No active generation to stop", "warning"))
    
    # Update button state if needed
    if hasattr(app, 'current_chat_worker'):
        app.current_chat_worker = None


# ==================== HELPER FUNCTIONS ====================

async def get_chat_configuration(app: 'TldwCli', prefix: str) -> Optional[Dict[str, Any]]:
    """
    Get chat configuration from UI widgets.
    This still needs some queries but only for configuration.
    """
    try:
        # Get configuration widgets
        provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
        model_widget = app.query_one(f"#{prefix}-api-model", Select)
        system_prompt_widget = app.query_one(f"#{prefix}-system-prompt", TextArea)
        temp_widget = app.query_one(f"#{prefix}-temperature", Input)
        
        config = {
            'provider': str(provider_widget.value) if provider_widget.value else None,
            'model': str(model_widget.value) if model_widget.value else None,
            'system_prompt': system_prompt_widget.text,
            'temperature': safe_float(temp_widget.value, 0.7),
            'streaming': get_cli_setting('chat_defaults', 'enable_streaming', True),
            'extra_params': {}
        }
        
        # Get optional parameters
        try:
            top_p_widget = app.query_one(f"#{prefix}-top-p", Input)
            if top_p_widget.value:
                config['extra_params']['top_p'] = safe_float(top_p_widget.value)
        except QueryError:
            pass
        
        try:
            max_tokens_widget = app.query_one(f"#{prefix}-llm-max-tokens", Input)
            if max_tokens_widget.value:
                config['max_tokens'] = safe_int(max_tokens_widget.value, 4096)
        except QueryError:
            config['max_tokens'] = 4096
        
        # Validate configuration
        if not config['provider']:
            app.post_message(ChatError("Please select an API Provider"))
            return None
        
        if not config['model']:
            app.post_message(ChatError("Please select a Model"))
            return None
        
        return config
        
    except Exception as e:
        loguru_logger.error(f"Error getting configuration: {e}")
        return None


async def get_chat_history_async(app: 'TldwCli') -> List[Dict[str, Any]]:
    """
    Get chat history from the current conversation.
    """
    # This would be replaced with reactive state access
    # For now, use the existing method
    if hasattr(app, 'current_conversation_id') and app.current_conversation_id:
        try:
            from tldw_chatbook.DB.ChaChaNotes_DB import get_chachanotes_db_lazy
            import asyncio
            
            db = get_chachanotes_db_lazy()
            messages = await asyncio.to_thread(
                db.get_messages_for_conversation,
                app.current_conversation_id
            )
            return messages or []
        except Exception as e:
            loguru_logger.error(f"Failed to get chat history: {e}")
    
    return []


async def get_rag_context_async(app: 'TldwCli', query: str) -> Optional[str]:
    """
    Get RAG context if enabled.
    """
    if not get_cli_setting('rag', 'enabled', False):
        return None
    
    try:
        # Import RAG handler if available
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
        context = await get_rag_context_for_chat(app, query)
        return context
    except ImportError:
        loguru_logger.debug("RAG not available")
    except Exception as e:
        loguru_logger.error(f"RAG context failed: {e}")
    
    return None


async def get_pending_attachments(app: 'TldwCli') -> Optional[List[str]]:
    """
    Get any pending attachments.
    """
    attachments = []
    
    # Check for pending images
    if hasattr(app, 'chat_window'):
        try:
            chat_window = app.query_one("#chat-window")
            if hasattr(chat_window, 'get_pending_attachment'):
                attachment = chat_window.get_pending_attachment()
                if attachment:
                    attachments.append(attachment)
        except QueryError:
            pass
    
    return attachments if attachments else None


async def should_resend_last_message(app: 'TldwCli') -> bool:
    """
    Check if we should resend the last message.
    """
    # This would check reactive state in the proper implementation
    # For now, return False
    return False


async def get_last_user_message(app: 'TldwCli') -> Optional[str]:
    """
    Get the last user message from history.
    """
    history = await get_chat_history_async(app)
    for msg in reversed(history):
        if msg.get('role') == 'user':
            return msg.get('message')
    return None


# ==================== SESSION MANAGEMENT ====================

async def handle_chat_new_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Create new conversation using messages.
    """
    from .chat_messages import NewSessionRequested
    
    # Determine if ephemeral based on button ID
    ephemeral = "temp" in event.button.id.lower()
    
    # Post message instead of direct manipulation
    app.post_message(NewSessionRequested(ephemeral))
    
    loguru_logger.info(f"New {'ephemeral' if ephemeral else 'persistent'} session requested")


async def handle_chat_save_current_chat_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Save current chat using messages.
    """
    from .chat_messages import SaveSessionRequested
    
    # Get title and keywords from UI
    try:
        title_input = app.query_one("#chat-conversation-title-input", Input)
        title = title_input.value.strip() if title_input.value else "Untitled Chat"
        
        keywords_input = app.query_one("#chat-conversation-keywords-input", TextArea)
        keywords = [k.strip() for k in keywords_input.text.split(',') if k.strip()] if keywords_input.text else []
    except QueryError:
        title = "Untitled Chat"
        keywords = []
    
    # Post save message
    app.post_message(SaveSessionRequested(title, keywords))
    
    loguru_logger.info(f"Save session requested: {title}")


async def handle_chat_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Load selected conversation using messages.
    """
    from .chat_messages import LoadSessionRequested
    
    # Get selected conversation ID
    conversation_id = getattr(app, 'selected_conversation_id', None)
    
    if not conversation_id:
        app.post_message(ChatError("No conversation selected"))
        return
    
    # Post load message
    app.post_message(LoadSessionRequested(conversation_id))
    
    loguru_logger.info(f"Load session requested: {conversation_id}")


# ==================== CHARACTER MANAGEMENT ====================

async def handle_chat_load_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Load character using messages.
    """
    from .chat_messages import CharacterLoadRequested
    
    # Get selected character ID
    character_id = getattr(app, 'selected_character_id', None)
    
    if not character_id:
        app.post_message(ChatError("No character selected"))
        return
    
    # Post load message
    app.post_message(CharacterLoadRequested(character_id))
    
    loguru_logger.info(f"Load character requested: {character_id}")


async def handle_chat_clear_active_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Clear character using messages.
    """
    from .chat_messages import CharacterCleared
    
    # Post clear message
    app.post_message(CharacterCleared())
    
    loguru_logger.info("Clear character requested")


# ==================== RESPOND FOR ME ====================

async def handle_respond_for_me_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Generate a suggested response using messages.
    """
    loguru_logger.info("Respond for me requested")
    
    # Get the last user message
    last_message = await get_last_user_message(app)
    
    if not last_message:
        app.post_message(ChatError("No message to respond to"))
        return
    
    # Generate a response suggestion
    suggested_response = f"I understand you're asking about: {last_message[:50]}..."
    
    # Set the input field with the suggestion
    try:
        text_area = app.query_one("#chat-input", TextArea)
        text_area.text = suggested_response
        text_area.focus()
    except QueryError:
        app.post_message(ChatError("Could not set suggested response"))


# ==================== EXPORT WRAPPER ====================

# This maintains backward compatibility
class ChatEventsNamespace:
    """Namespace to maintain backward compatibility."""
    
    handle_chat_send_button_pressed = staticmethod(handle_chat_send_button_pressed)
    handle_stop_chat_generation_pressed = staticmethod(handle_stop_chat_generation_pressed)
    handle_chat_new_conversation_button_pressed = staticmethod(handle_chat_new_conversation_button_pressed)
    handle_chat_save_current_chat_button_pressed = staticmethod(handle_chat_save_current_chat_button_pressed)
    handle_chat_load_selected_button_pressed = staticmethod(handle_chat_load_selected_button_pressed)
    handle_chat_load_character_button_pressed = staticmethod(handle_chat_load_character_button_pressed)
    handle_chat_clear_active_character_button_pressed = staticmethod(handle_chat_clear_active_character_button_pressed)
    handle_respond_for_me_button_pressed = staticmethod(handle_respond_for_me_button_pressed)

# ==================== CONVERSATION SEARCH ====================

async def handle_chat_conversation_search_changed(app: 'TldwCli', event: Input.Changed) -> None:
    """
    FIXED: Handle conversation search using messages.
    """
    from .chat_messages import ConversationSearchChanged
    
    search_query = event.value.strip()
    app.post_message(ConversationSearchChanged(search_query))
    
    # Update search in worker
    app.run_worker(
        search_conversations(app, search_query),
        name="conversation_search",
        exclusive=True
    )


@work(exclusive=True, thread=True)
def search_conversations(app: 'TldwCli', query: str) -> None:
    """
    Search conversations in background.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import get_chachanotes_db_lazy
    from .chat_messages import ConversationSearchResults
    
    try:
        db = get_chachanotes_db_lazy()
        if query:
            results = db.search_conversations(query, limit=50)
        else:
            results = db.get_all_conversations(limit=50)
        
        # Post results back to UI
        app.post_message(ConversationSearchResults(results or []))
        
    except Exception as e:
        loguru_logger.error(f"Error searching conversations: {e}")
        app.post_message(ChatError(f"Search failed: {str(e)}"))


# ==================== EXPORT CONVERSATIONS ====================

async def handle_chat_export_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Export conversation using messages.
    """
    from .chat_messages import ExportConversationRequested
    
    # Get export format from UI
    try:
        format_select = app.query_one("#chat-export-format", Select)
        export_format = str(format_select.value) if format_select.value else "markdown"
    except QueryError:
        export_format = "markdown"
    
    # Post export message
    app.post_message(ExportConversationRequested(
        conversation_id=app.current_conversation_id,
        format=export_format
    ))
    
    # Run export in worker
    app.run_worker(
        export_conversation(app, app.current_conversation_id, export_format),
        name="export_conversation",
        exclusive=True
    )


@work(thread=True)
def export_conversation(app: 'TldwCli', conversation_id: str, format: str) -> None:
    """
    Export conversation in background.
    """
    from tldw_chatbook.Chat.document_generator import generate_conversation_document
    from .chat_messages import ExportConversationCompleted
    import tempfile
    
    try:
        # Generate document
        content = generate_conversation_document(
            conversation_id=conversation_id,
            format=format
        )
        
        # Save to temp file
        suffix = {
            "markdown": ".md",
            "html": ".html",
            "pdf": ".pdf",
            "docx": ".docx"
        }.get(format, ".txt")
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            prefix="chat_export_"
        ) as f:
            f.write(content)
            filepath = f.name
        
        # Post completion message
        app.post_message(ExportConversationCompleted(filepath, format))
        
    except Exception as e:
        loguru_logger.error(f"Export failed: {e}")
        app.post_message(ChatError(f"Export failed: {str(e)}"))


# ==================== STOP GENERATION ====================

async def handle_stop_chat_generation_pressed(app: 'TldwCli', event: Any) -> None:
    """
    FIXED: Stop generation using worker cancellation.
    """
    from .chat_messages import GenerationStopped
    
    # Cancel any chat workers
    cancelled = False
    for worker in app.workers:
        if worker.name and 'chat' in worker.name.lower():
            worker.cancel()
            cancelled = True
            loguru_logger.info(f"Cancelled worker: {worker.name}")
    
    if cancelled:
        app.post_message(GenerationStopped())
    else:
        app.post_message(ChatError("No active generation to stop", "warning"))
    
    # Update button state if needed
    if hasattr(app, 'current_chat_worker'):
        app.current_chat_worker = None


# ==================== CLEAR CHAT ====================

async def handle_chat_clear_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Clear conversation using messages.
    """
    from .chat_messages import ClearConversationRequested
    
    # Post clear message
    app.post_message(ClearConversationRequested())
    
    loguru_logger.info("Clear conversation requested")


# ==================== CONTINUE RESPONSE ====================

async def handle_continue_response_button_pressed(
    app: 'TldwCli',
    event: Button.Pressed,
    message_widget: Any
) -> None:
    """
    FIXED: Continue response using messages.
    """
    from .chat_messages import ContinueResponseRequested
    
    # Get message content
    message_content = getattr(message_widget, 'message', '')
    message_id = getattr(message_widget, 'message_id', None)
    
    # Post continue message
    app.post_message(ContinueResponseRequested(message_id, message_content))
    
    # Process in worker
    config = await get_chat_configuration(app, "chat")
    if config:
        app.run_worker(
            continue_llm_response(app, message_content, config),
            name="continue_response",
            exclusive=True
        )


@work(exclusive=True, thread=True)
def continue_llm_response(
    app: 'TldwCli',
    partial_response: str,
    config: Dict[str, Any]
) -> None:
    """
    Continue LLM response in worker.
    """
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from .chat_messages import LLMResponseStarted, LLMResponseCompleted
    
    try:
        # Post start message
        app.post_message(LLMResponseStarted())
        
        # Create continuation prompt
        continuation_prompt = f"Continue this response from where it left off:\n\n{partial_response}\n\n[Continue from here]"
        
        # Get history
        history = get_chat_history_sync(app)
        
        # Make API call
        response = chat_api_call(
            message=continuation_prompt,
            history=history,
            provider=config['provider'],
            model=config['model'],
            system_prompt=config.get('system_prompt', ''),
            temperature=config.get('temperature', 0.7),
            streaming=False,
            **config.get('extra_params', {})
        )
        
        # Combine responses
        full_response = partial_response + "\n" + response
        
        # Post completion
        app.post_message(LLMResponseCompleted(full_response))
        
    except Exception as e:
        app.post_message(LLMResponseError(str(e)))


def get_chat_history_sync(app: 'TldwCli') -> List[Dict[str, Any]]:
    """
    Synchronous version of get_chat_history for thread workers.
    """
    if hasattr(app, 'current_conversation_id') and app.current_conversation_id:
        try:
            from tldw_chatbook.DB.ChaChaNotes_DB import get_chachanotes_db_lazy
            db = get_chachanotes_db_lazy()
            messages = db.get_messages_for_conversation(app.current_conversation_id)
            return messages or []
        except Exception as e:
            loguru_logger.error(f"Failed to get chat history: {e}")
    return []


# ==================== TEMPLATE MANAGEMENT ====================

async def handle_chat_apply_template_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Apply template using messages.
    """
    from .chat_messages import TemplateApplied
    
    # Get selected template
    try:
        template_select = app.query_one("#chat-template-select", Select)
        template_id = str(template_select.value) if template_select.value else None
    except QueryError:
        template_id = None
    
    if not template_id:
        app.post_message(ChatError("No template selected"))
        return
    
    # Post template message
    app.post_message(TemplateApplied(template_id))
    
    loguru_logger.info(f"Applied template: {template_id}")


# ==================== PROMPT MANAGEMENT ====================

async def handle_chat_copy_system_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Copy system prompt using messages.
    """
    from .chat_messages import CopyToClipboard
    
    # Get system prompt
    try:
        system_prompt_widget = app.query_one("#chat-system-prompt", TextArea)
        content = system_prompt_widget.text
    except QueryError:
        content = ""
    
    if content:
        # Post copy message
        app.post_message(CopyToClipboard(content, "System prompt"))
    else:
        app.post_message(ChatError("No system prompt to copy"))


async def handle_chat_copy_user_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: Copy user prompt using messages.
    """
    from .chat_messages import CopyToClipboard
    
    # Get user prompt from input
    try:
        text_area = app.query_one("#chat-input", TextArea)
        content = text_area.text
    except QueryError:
        content = ""
    
    if content:
        # Post copy message
        app.post_message(CopyToClipboard(content, "User prompt"))
    else:
        app.post_message(ChatError("No user prompt to copy"))


# ==================== VIEW SELECTED PROMPT ====================

async def handle_chat_view_selected_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    FIXED: View selected prompt using messages.
    """
    from .chat_messages import ViewPromptRequested
    
    # Get selected prompt ID
    prompt_id = getattr(app, 'selected_prompt_id', None)
    
    if not prompt_id:
        app.post_message(ChatError("No prompt selected"))
        return
    
    # Post view message
    app.post_message(ViewPromptRequested(prompt_id))
    
    loguru_logger.info(f"View prompt requested: {prompt_id}")


# ==================== EXPORT WRAPPER ====================

# This maintains backward compatibility
class ChatEventsNamespace:
    """Namespace to maintain backward compatibility."""
    
    # Main chat operations
    handle_chat_send_button_pressed = staticmethod(handle_chat_send_button_pressed)
    handle_stop_chat_generation_pressed = staticmethod(handle_stop_chat_generation_pressed)
    handle_continue_response_button_pressed = staticmethod(handle_continue_response_button_pressed)
    handle_respond_for_me_button_pressed = staticmethod(handle_respond_for_me_button_pressed)
    
    # Session management
    handle_chat_new_conversation_button_pressed = staticmethod(handle_chat_new_conversation_button_pressed)
    handle_chat_new_temp_chat_button_pressed = staticmethod(handle_chat_new_conversation_button_pressed)  # Alias
    handle_chat_save_current_chat_button_pressed = staticmethod(handle_chat_save_current_chat_button_pressed)
    handle_chat_load_selected_button_pressed = staticmethod(handle_chat_load_selected_button_pressed)
    handle_chat_clear_conversation_button_pressed = staticmethod(handle_chat_clear_conversation_button_pressed)
    
    # Character management
    handle_chat_load_character_button_pressed = staticmethod(handle_chat_load_character_button_pressed)
    handle_chat_clear_active_character_button_pressed = staticmethod(handle_chat_clear_active_character_button_pressed)
    
    # Export and search
    handle_chat_export_conversation_button_pressed = staticmethod(handle_chat_export_conversation_button_pressed)
    handle_chat_conversation_search_changed = staticmethod(handle_chat_conversation_search_changed)
    
    # Templates and prompts
    handle_chat_apply_template_button_pressed = staticmethod(handle_chat_apply_template_button_pressed)
    handle_chat_copy_system_prompt_button_pressed = staticmethod(handle_chat_copy_system_prompt_button_pressed)
    handle_chat_copy_user_prompt_button_pressed = staticmethod(handle_chat_copy_user_prompt_button_pressed)
    handle_chat_view_selected_prompt_button_pressed = staticmethod(handle_chat_view_selected_prompt_button_pressed)

# Export as 'chat_events' for backward compatibility
chat_events = ChatEventsNamespace()