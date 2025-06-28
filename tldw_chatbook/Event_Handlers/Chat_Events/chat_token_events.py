# chat_token_events.py
# Description: Token counting and display updates for chat
#
# Imports
from typing import TYPE_CHECKING, List, Dict, Any, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual.widgets import Static, Input, Select
from textual.css.query import QueryError
#
# Local Imports
from ...Utils.token_counter import (
    count_tokens_chat_history, 
    get_model_token_limit,
    estimate_remaining_tokens
)
from ...Widgets.chat_message import ChatMessage
#
if TYPE_CHECKING:
    from ...app import TldwCli
#
########################################################################################################################
#
# Functions:

async def update_chat_token_counter(app: 'TldwCli') -> None:
    """
    Update the token counter display in the chat window.
    
    This function:
    1. Gathers all messages from the chat UI
    2. Counts tokens based on the selected model
    3. Updates the token counter display
    """
    logger.info("update_chat_token_counter called")
    try:
        # Get current model and provider
        try:
            provider_widget = app.query_one("#chat-api-provider", Select)
            model_widget = app.query_one("#chat-api-model", Select)
            max_tokens_widget = app.query_one("#chat-llm-max-tokens", Input)
            system_prompt_widget = app.query_one("#chat-system-prompt")
            
            provider = provider_widget.value or "openai"
            model = model_widget.value or "gpt-3.5-turbo"
            max_tokens_response = int(max_tokens_widget.value or "2048")
            system_prompt = system_prompt_widget.text if hasattr(system_prompt_widget, 'text') else ""
        except (QueryError, ValueError) as e:
            logger.debug(f"Could not get chat settings for token counting: {e}")
            provider = "openai"
            model = "gpt-3.5-turbo"
            max_tokens_response = 2048
            system_prompt = ""
        
        # Get chat history from UI
        chat_history = []
        try:
            chat_log = app.query_one("#chat-log")
            message_widgets = list(chat_log.query(ChatMessage))
            
            for widget in message_widgets:
                if widget.role in ("User", "AI") and widget.generation_complete:
                    role_for_api = "assistant" if widget.role == "AI" else "user"
                    content = widget.message_text
                    if content:
                        chat_history.append({"role": role_for_api, "content": content})
        except QueryError as e:
            logger.debug(f"Could not get chat messages for token counting: {e}")
        
        # Calculate tokens
        used_tokens, total_limit, remaining = estimate_remaining_tokens(
            chat_history,
            model=model,
            provider=provider,
            max_tokens_response=max_tokens_response,
            system_prompt=system_prompt
        )
        
        # Update the display in footer
        try:
            footer = app.query_one("AppFooterStatus")
            from ...Utils.token_counter import format_token_display
            display_text = format_token_display(used_tokens, total_limit)
            logger.info(f"Updating footer with token count: {display_text}")
            footer.update_token_count(display_text)
        except QueryError as e:
            logger.error(f"Footer widget not found: {e}")
                
    except Exception as e:
        logger.error(f"Error updating chat token counter: {e}", exc_info=True)


async def handle_chat_input_changed(app: 'TldwCli', event) -> None:
    """
    Handle changes to the chat input to update token count in real-time.
    
    This provides immediate feedback as the user types.
    """
    try:
        # Get the current input text
        current_input = event.text_area.text if hasattr(event, 'text_area') else ""
        
        if current_input:
            # Update with the pending message included
            await update_chat_token_counter_with_pending(app, current_input)
        else:
            # Just update normally
            await update_chat_token_counter(app)
            
    except Exception as e:
        logger.error(f"Error handling chat input change: {e}")


async def update_chat_token_counter_with_pending(app: 'TldwCli', pending_text: str) -> None:
    """
    Update token counter including a pending message that hasn't been sent yet.
    
    This helps users see what the token count will be after sending their message.
    """
    try:
        # Get current model and provider
        try:
            provider_widget = app.query_one("#chat-api-provider", Select)
            model_widget = app.query_one("#chat-api-model", Select)
            max_tokens_widget = app.query_one("#chat-llm-max-tokens", Input)
            system_prompt_widget = app.query_one("#chat-system-prompt")
            
            provider = provider_widget.value or "openai"
            model = model_widget.value or "gpt-3.5-turbo"
            max_tokens_response = int(max_tokens_widget.value or "2048")
            system_prompt = system_prompt_widget.text if hasattr(system_prompt_widget, 'text') else ""
        except (QueryError, ValueError):
            provider = "openai"
            model = "gpt-3.5-turbo"
            max_tokens_response = 2048
            system_prompt = ""
        
        # Get chat history from UI
        chat_history = []
        try:
            chat_log = app.query_one("#chat-log")
            message_widgets = list(chat_log.query(ChatMessage))
            
            for widget in message_widgets:
                if widget.role in ("User", "AI") and widget.generation_complete:
                    role_for_api = "assistant" if widget.role == "AI" else "user"
                    content = widget.message_text
                    if content:
                        chat_history.append({"role": role_for_api, "content": content})
        except QueryError:
            pass
        
        # Add the pending message
        if pending_text:
            chat_history.append({"role": "user", "content": pending_text})
        
        # Calculate tokens
        used_tokens, total_limit, remaining = estimate_remaining_tokens(
            chat_history,
            model=model,
            provider=provider,
            max_tokens_response=max_tokens_response,
            system_prompt=system_prompt
        )
        
        # Update the display in footer with a pending indicator
        try:
            footer = app.query_one("AppFooterStatus")
            from ...Utils.token_counter import format_token_display
            display_text = format_token_display(used_tokens, total_limit)
            # Add pending indicator
            if pending_text:
                display_text = display_text.replace("Tokens:", "Tokens (typing):")
            footer.update_token_count(display_text)
        except QueryError:
            logger.debug("Footer widget not found")
                
    except Exception as e:
        logger.error(f"Error updating chat token counter with pending: {e}")


async def handle_model_or_provider_changed(app: 'TldwCli', event) -> None:
    """
    Handle changes to model or provider selection to update token limits.
    """
    logger.debug("Model or provider changed, updating token counter")
    await update_chat_token_counter(app)


#
# End of chat_token_events.py
########################################################################################################################