# chat_token_events.py
# Description: Token counting and display updates for chat
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ScreenStackError
from textual.widgets import Input, Select
from textual.css.query import QueryError
#
# Local Imports
from ...Utils.token_counter import (
    estimate_remaining_tokens
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
#
if TYPE_CHECKING:
    from ...app import TldwCli
#
########################################################################################################################
#
# Functions:

def _estimate_tokens_cached(
    app: 'TldwCli',
    chat_history: list,
    *,
    model: str,
    provider: str,
    max_tokens_response: int,
    system_prompt: str,
):
    """Estimate token usage, reusing the last result when the inputs are unchanged.

    task-261 dirty gate: the footer's 10 s interval timer re-ran the full
    tokenizer over the entire visible history every tick even when nothing
    had changed. A cheap signature over every input that influences the
    estimate — settings plus message count and a per-message (role, content)
    hash tuple — is compared against the previous tick's; on a match
    the cached counts are returned without re-tokenizing. The cache lives on
    the app instance (one live history per app), and the caller still
    refreshes the footer widget every tick, so display behavior is
    unchanged.

    Args:
        app: The running app instance the cache is stored on.
        chat_history: Completed messages as ``{"role", "content"}`` dicts.
        model: Model name the tokenizer estimate is computed for.
        provider: Provider name the tokenizer estimate is computed for.
        max_tokens_response: Reserved response-token budget.
        system_prompt: System prompt text included in the estimate.

    Returns:
        The ``(used_tokens, total_limit, remaining)`` tuple from
        ``estimate_remaining_tokens``.
    """
    signature = (
        model,
        provider,
        max_tokens_response,
        system_prompt,
        len(chat_history),
        # Per-message content hashes, not lengths: a same-length edit to an
        # earlier message (or a role flip) must invalidate the cache too
        # (PR #688 review). CPython caches str.__hash__ on the string object,
        # so for an unchanged history this stays O(1) amortized per message.
        tuple(
            (hash(message.get("role", "")), hash(message.get("content", "")))
            for message in chat_history
        ),
    )
    cached = getattr(app, "_footer_token_estimate_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    result = estimate_remaining_tokens(
        chat_history,
        model=model,
        provider=provider,
        max_tokens_response=max_tokens_response,
        system_prompt=system_prompt,
    )
    try:
        app._footer_token_estimate_cache = (signature, result)
    except Exception:
        # A slot-restricted or frozen test double can refuse the attribute;
        # caching is best-effort and never worth failing the update over.
        logger.debug("Footer token-estimate cache not stored on app instance.")
    return result


async def update_chat_token_counter(app: 'TldwCli') -> None:
    """
    Update the token counter display in the chat window.
    
    This function:
    1. Gathers all messages from the chat UI
    2. Counts tokens based on the selected model
    3. Updates the token counter display
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

        # Calculate tokens (task-261: skips re-tokenizing when history and
        # settings are unchanged since the last periodic tick).
        used_tokens, total_limit, remaining = _estimate_tokens_cached(
            app,
            chat_history,
            model=model,
            provider=provider,
            max_tokens_response=max_tokens_response,
            system_prompt=system_prompt
        )

        # Use max_tokens_response as the display limit instead of model's total limit
        # This allows users to see how their conversation measures against their configured limit
        display_limit = max_tokens_response
        
        # Check if there's a custom token limit setting (we'll add this later)
        try:
            custom_limit_widget = app.query_one("#chat-custom-token-limit", Input)
            custom_limit = int(custom_limit_widget.value or "0")
            if custom_limit > 0:
                display_limit = custom_limit
        except (QueryError, ValueError):
            # No custom limit widget or invalid value, use max_tokens_response
            pass
        
        # Update the display in footer
        try:
            # Check if in screen navigation mode
            if hasattr(app, '_use_screen_navigation') and app._use_screen_navigation:
                # In screen mode, footer might not exist or be in a different place
                logger.debug(f"Token count in screen mode: {used_tokens}/{display_limit}")
                # Store for potential screen usage
                app.current_token_count = (used_tokens, display_limit)
            else:
                # Legacy tab mode - update the active screen's footer directly.
                # Resolved via `app.screen` rather than `app.query_one`
                # because BaseAppScreen mounts a per-screen AppFooterStatus
                # (task-264): the default screen's instance is occluded once
                # any screen is pushed, and App.query_one only ever searches
                # the default screen. ScreenStackError is caught alongside
                # QueryError: this can run from interval timers that fire
                # during shutdown, after the screen stack is drained.
                footer = app.screen.query_one("AppFooterStatus")
                from ...Utils.token_counter import format_token_display
                display_text = format_token_display(used_tokens, display_limit)
                footer.update_token_count(display_text)
                logger.debug(f"Token count updated: {used_tokens}/{display_limit} (model limit: {total_limit})")
        except (QueryError, ScreenStackError) as e:
            logger.debug(f"Footer widget not found (may be in screen mode): {e}")
                
    except Exception as e:
        logger.opt(exception=True).error(f"Error updating chat token counter: {e}")


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
        
        # Use max_tokens_response as the display limit instead of model's total limit
        display_limit = max_tokens_response
        
        # Check if there's a custom token limit setting
        try:
            custom_limit_widget = app.query_one("#chat-custom-token-limit", Input)
            custom_limit = int(custom_limit_widget.value or "0")
            if custom_limit > 0:
                display_limit = custom_limit
        except (QueryError, ValueError):
            pass
        
        # Update the display in footer with a pending indicator
        try:
            # Check if in screen navigation mode
            if hasattr(app, '_use_screen_navigation') and app._use_screen_navigation:
                # In screen mode, store for potential screen usage
                logger.debug(f"Pending token count in screen mode: {used_tokens}/{display_limit}")
                app.current_token_count = (used_tokens, display_limit)
                app.token_count_pending = bool(pending_text)
            else:
                # Legacy tab mode - update the active screen's footer directly
                # (see the analogous comment in update_chat_token_counter above;
                # same task-264 active-screen resolution + shutdown guard).
                footer = app.screen.query_one("AppFooterStatus")
                from ...Utils.token_counter import format_token_display
                display_text = format_token_display(used_tokens, display_limit)
                # Add pending indicator
                if pending_text:
                    display_text = display_text.replace("Tokens:", "Tokens (typing):")
                footer.update_token_count(display_text)
        except (QueryError, ScreenStackError):
            logger.debug("Footer widget not found (may be in screen mode)")
                
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