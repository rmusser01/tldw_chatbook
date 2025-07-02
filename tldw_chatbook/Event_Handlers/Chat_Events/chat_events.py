# tldw_app/Event_Handlers/chat_events.py
# Description:
#
# Imports
import logging
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from rich.text import Text
from rich.markup import escape as escape_markup
from textual.widgets import (
    Button, Input, TextArea, Static, Select, Checkbox, ListView, ListItem, Label
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError
#
# Local Imports
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events_sidebar
from tldw_chatbook.Utils.Utils import safe_float, safe_int
from tldw_chatbook.Utils.input_validation import validate_text_input, validate_number_range, sanitize_string
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.titlebar import TitleBar
from tldw_chatbook.Utils.Emoji_Handling import (
    get_char, EMOJI_THINKING, FALLBACK_THINKING, EMOJI_EDIT, FALLBACK_EDIT,
    EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT, EMOJI_COPIED, FALLBACK_COPIED, EMOJI_COPY, FALLBACK_COPY,
    EMOJI_SEND, FALLBACK_SEND, EMOJI_STOP, FALLBACK_STOP
)
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.Character_Chat.Character_Chat_Lib import load_character_and_image
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError, InputError
from tldw_chatbook.Prompt_Management import Prompts_Interop as prompts_interop
#
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
#
########################################################################################################################
#
# Security Functions:

def safe_json_loads(json_str: str, max_size: int = 1024 * 1024) -> Optional[Union[dict, list]]:
    """
    Safely parse JSON with size limits to prevent DoS attacks.
    
    Args:
        json_str: The JSON string to parse
        max_size: Maximum allowed size in bytes (default 1MB)
    
    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not json_str or not json_str.strip():
        return None
    
    # Check size limit
    if len(json_str.encode('utf-8')) > max_size:
        loguru_logger.warning(f"JSON string too large: {len(json_str)} bytes (max {max_size})")
        return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        loguru_logger.warning(f"Invalid JSON: {e}")
        return None
    except Exception as e:
        loguru_logger.error(f"Unexpected error parsing JSON: {e}")
        return None

########################################################################################################################
#
# Functions:

async def handle_chat_tab_sidebar_toggle(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles sidebar toggles specific to the Chat tab."""
    loguru_logger.debug(f"Chat tab sidebar toggle button pressed: {event.button.id}")
    button_id = event.button.id
    if button_id == "toggle-chat-left-sidebar":
        app.chat_sidebar_collapsed = not app.chat_sidebar_collapsed
        loguru_logger.debug("Chat tab settings sidebar (left) now %s", "collapsed" if app.chat_sidebar_collapsed else "expanded")
    elif button_id == "toggle-chat-right-sidebar":
        app.chat_right_sidebar_collapsed = not app.chat_right_sidebar_collapsed
        loguru_logger.debug("Chat tab character sidebar (right) now %s", "collapsed" if app.chat_right_sidebar_collapsed else "expanded")
    else:
        loguru_logger.warning(f"Unhandled sidebar toggle button ID '{button_id}' in Chat tab handler.")

async def handle_chat_send_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the send button press for the main chat tab."""
    prefix = "chat"  # This handler is specific to the main chat tab's send button
    
    # Check if there's an active chat generation running
    if hasattr(app, 'current_chat_worker') and app.current_chat_worker and app.current_chat_worker.is_running:
        # Stop the generation instead of sending
        loguru_logger.info("Send button pressed - stopping active generation")
        await handle_stop_chat_generation_pressed(app, event)
        return
    
    loguru_logger.info(f"Send button pressed for '{prefix}' (main chat)") # Use loguru_logger consistently

    # --- 1. Query UI Widgets ---
    try:
        text_area = app.query_one(f"#{prefix}-input", TextArea)
        chat_container = app.query_one(f"#{prefix}-log", VerticalScroll)
        provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
        model_widget = app.query_one(f"#{prefix}-api-model", Select)
        system_prompt_widget = app.query_one(f"#{prefix}-system-prompt", TextArea)
        temp_widget = app.query_one(f"#{prefix}-temperature", Input)
        top_p_widget = app.query_one(f"#{prefix}-top-p", Input)
        min_p_widget = app.query_one(f"#{prefix}-min-p", Input)
        top_k_widget = app.query_one(f"#{prefix}-top-k", Input)

        llm_max_tokens_widget = app.query_one(f"#{prefix}-llm-max-tokens", Input)
        llm_seed_widget = app.query_one(f"#{prefix}-llm-seed", Input)
        llm_stop_widget = app.query_one(f"#{prefix}-llm-stop", Input)
        llm_response_format_widget = app.query_one(f"#{prefix}-llm-response-format", Select)
        llm_n_widget = app.query_one(f"#{prefix}-llm-n", Input)
        llm_user_identifier_widget = app.query_one(f"#{prefix}-llm-user-identifier", Input)
        llm_logprobs_widget = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
        llm_top_logprobs_widget = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
        llm_logit_bias_widget = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
        llm_presence_penalty_widget = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
        llm_frequency_penalty_widget = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
        llm_tools_widget = app.query_one(f"#{prefix}-llm-tools", TextArea)
        llm_tool_choice_widget = app.query_one(f"#{prefix}-llm-tool-choice", Input)
        llm_fixed_tokens_kobold_widget = app.query_one(f"#{prefix}-llm-fixed-tokens-kobold", Checkbox)
        # Query for the strip thinking tags checkbox
        try:
            strip_tags_checkbox = app.query_one("#chat-strip-thinking-tags-checkbox", Checkbox)
            strip_thinking_tags_value = strip_tags_checkbox.value
        except QueryError:
            loguru_logger.warning("Could not find '#chat-strip-thinking-tags-checkbox'. Defaulting to True for strip_thinking_tags.")
            strip_thinking_tags_value = True

    except QueryError as e:
        loguru_logger.error(f"Send Button: Could not find UI widgets for '{prefix}': {e}")
        try:
            container_for_error = chat_container if 'chat_container' in locals() and chat_container.is_mounted else app.query_one(
                f"#{prefix}-log", VerticalScroll) # Re-query if initial one failed
            await container_for_error.mount(
                ChatMessage(Text.from_markup(f"[bold red]Internal Error:[/]\nMissing UI elements for {prefix}."), role="System", classes="-error"))
        except QueryError:
            loguru_logger.error(f"Send Button: Critical - could not even find chat container #{prefix}-log to display error.")
        return

    # --- 2. Get Message and Parameters from UI ---
    message_text_from_input = text_area.text.strip()
    
    # Validate user message input
    if message_text_from_input:
        if not validate_text_input(message_text_from_input, max_length=100000, allow_html=False):
            await chat_container.mount(ChatMessage(Text.from_markup("Error: Message contains invalid content or is too long."), role="System", classes="-error"))
            loguru_logger.warning(f"Invalid user message input rejected")
            return
        
        # Sanitize the message text to remove dangerous characters
        message_text_from_input = sanitize_string(message_text_from_input, max_length=100000)
    
    reuse_last_user_bubble = False

    if not message_text_from_input: # Try to reuse last user message if input is empty
        try:
            last_msg_widget: Optional[ChatMessage] = None
            # Iterate over a materialized list to avoid issues if querying during modification (though less likely here)
            for widget in reversed(list(chat_container.query(ChatMessage))):
                if widget.role == "User": # Only reuse User messages
                    last_msg_widget = widget
                    break
            if last_msg_widget:
                message_text_from_input = last_msg_widget.message_text
                reuse_last_user_bubble = True
                loguru_logger.debug("Reusing last user message as input is empty.")
        except Exception as exc:
            loguru_logger.error("Failed to inspect last message for reuse: %s", exc, exc_info=True)

    if not message_text_from_input:
        loguru_logger.debug("Send Button: Empty message and no reusable user bubble in '%s'. Focusing input.", prefix)
        text_area.focus()
        return

    selected_provider = str(provider_widget.value) if provider_widget.value != Select.BLANK else None
    selected_model = str(model_widget.value) if model_widget.value != Select.BLANK else None
    system_prompt = system_prompt_widget.text
    
    # Validate system prompt input
    if system_prompt and not validate_text_input(system_prompt, max_length=50000, allow_html=False):
        await chat_container.mount(ChatMessage(Text.from_markup("Error: System prompt contains invalid content or is too long."), role="System", classes="-error"))
        loguru_logger.warning(f"Invalid system prompt input rejected")
        return
    
    # Sanitize system prompt
    if system_prompt:
        system_prompt = sanitize_string(system_prompt, max_length=50000)
    temperature = safe_float(temp_widget.value, 0.7, "temperature") # Use imported safe_float
    top_p = safe_float(top_p_widget.value, 0.95, "top_p")
    min_p = safe_float(min_p_widget.value, 0.05, "min_p")
    top_k = safe_int(top_k_widget.value, 50, "top_k") # Use imported safe_int
    
    # Validate parameter ranges
    if not validate_number_range(temperature, 0.0, 2.0):
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Temperature must be between 0.0 and 2.0."), role="System", classes="-error"))
        return
    
    if not validate_number_range(top_p, 0.0, 1.0):
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Top-p must be between 0.0 and 1.0."), role="System", classes="-error"))
        return
    
    if not validate_number_range(min_p, 0.0, 1.0):
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Min-p must be between 0.0 and 1.0."), role="System", classes="-error"))
        return
    
    if not validate_number_range(top_k, 1, 1000):
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Top-k must be between 1 and 1000."), role="System", classes="-error"))
        return
    custom_prompt = ""  # Assuming this isn't used directly in chat send, but passed

    # Determine if streaming should be enabled based on provider settings
    should_stream = False  # Default to False
    if selected_provider:
        provider_settings_key = selected_provider.lower().replace(" ", "_")
        provider_specific_settings = app.app_config.get("api_settings", {}).get(provider_settings_key, {})
        should_stream = provider_specific_settings.get("streaming", False)
        loguru_logger.debug(f"Streaming for {selected_provider} set to {should_stream} based on config.")
    else:
        loguru_logger.debug("No provider selected, streaming defaults to False for this request.")

    # --- Integration of Active Character Data ---
    system_prompt_from_ui = system_prompt_widget.text # This is the system prompt from the LEFT sidebar
    active_char_data = app.current_chat_active_character_data  # This is from the RIGHT sidebar's loaded char
    final_system_prompt_for_api = system_prompt_from_ui  # Default to UI

    if active_char_data:
        loguru_logger.info(
            f"Active character data found: {active_char_data.get('name', 'Unnamed')}. Checking for system prompt override.")
        # Prioritize system_prompt from active_char_data.
        char_specific_system_prompt = active_char_data.get('system_prompt')  # This comes from the editable fields
        if char_specific_system_prompt is not None and char_specific_system_prompt.strip():  # Check if not None AND not empty/whitespace
            final_system_prompt_for_api = char_specific_system_prompt
            loguru_logger.debug(
                f"System prompt overridden by active character's system prompt: '{final_system_prompt_for_api[:100]}...'")
        else:
            loguru_logger.debug(
                f"Active character has no system_prompt or it's empty. Using system_prompt from left sidebar: '{final_system_prompt_for_api[:100]}...'")
    else:
        loguru_logger.info("No active character data. Using system prompt from left sidebar UI.")

        # Optional: Further persona integration (example)
        # if active_char_data.get('personality'):
        #     system_prompt = f"Personality: {active_char_data['personality']}\n\n{system_prompt}"
        # if active_char_data.get('scenario'):
        #     system_prompt = f"Scenario: {active_char_data['scenario']}\n\n{system_prompt}"
        # else:
        #     loguru_logger.info("No active character data. Using system prompt from UI.")
    # --- End of Integration ---

    llm_max_tokens_value = safe_int(llm_max_tokens_widget.value, 1024, "llm_max_tokens")
    llm_seed_value = safe_int(llm_seed_widget.value, None, "llm_seed") # None is a valid default
    llm_stop_value = [s.strip() for s in llm_stop_widget.value.split(',') if s.strip()] if llm_stop_widget.value.strip() else None
    llm_response_format_value = {"type": str(llm_response_format_widget.value)} if llm_response_format_widget.value != Select.BLANK else {"type": "text"}
    llm_n_value = safe_int(llm_n_widget.value, 1, "llm_n")
    llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
    llm_logprobs_value = llm_logprobs_widget.value
    llm_top_logprobs_value = safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs") if llm_logprobs_value else 0
    llm_presence_penalty_value = safe_float(llm_presence_penalty_widget.value, 0.0, "llm_presence_penalty")
    llm_frequency_penalty_value = safe_float(llm_frequency_penalty_widget.value, 0.0, "llm_frequency_penalty")
    llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None

    # Safely parse logit bias JSON with size limits
    llm_logit_bias_text = llm_logit_bias_widget.text.strip()
    if llm_logit_bias_text and llm_logit_bias_text != "{}":
        llm_logit_bias_value = safe_json_loads(llm_logit_bias_text, max_size=64 * 1024)  # 64KB limit
        if llm_logit_bias_value is None and llm_logit_bias_text:
            await chat_container.mount(ChatMessage(Text.from_markup("Error: Invalid or too large JSON in LLM Logit Bias. Parameter not used."), role="System", classes="-error"))
    else:
        llm_logit_bias_value = None
    # Safely parse tools JSON with size limits
    llm_tools_text = llm_tools_widget.text.strip()
    if llm_tools_text and llm_tools_text != "[]":
        llm_tools_value = safe_json_loads(llm_tools_text, max_size=256 * 1024)  # 256KB limit for tools
        if llm_tools_value is None and llm_tools_text:
            await chat_container.mount(ChatMessage(Text.from_markup("Error: Invalid or too large JSON in LLM Tools. Parameter not used."), role="System", classes="-error"))
    else:
        llm_tools_value = None

    # --- 3. Basic Validation ---
    if not selected_provider:
        await chat_container.mount(ChatMessage(Text.from_markup("Please select an API Provider."), role="System", classes="-error")); return
    if not selected_model:
        await chat_container.mount(ChatMessage(Text.from_markup("Please select a Model."), role="System", classes="-error")); return
    if not app.API_IMPORTS_SUCCESSFUL: # Access as app attribute
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Core API functions failed to load."), role="System", classes="-error"))
        loguru_logger.error("Attempted to send message, but API imports failed.")
        return
    llm_fixed_tokens_kobold_value = llm_fixed_tokens_kobold_widget.value

    # --- 4. Build Chat History for API ---
    # History should contain messages *before* the current user's input.
    # The current user's input (`message_text_from_input`) will be passed as the `message` param to `app.chat_wrapper`.
    chat_history_for_api: List[Dict[str, Any]] = []
    try:
        # Iterate through all messages currently in the UI
        all_ui_messages = list(chat_container.query(ChatMessage))

        # Determine how many messages to actually include in history sent to API
        # (e.g., based on token limits or a fixed number)
        # For now, let's take all completed User/AI messages *before* any reused bubble

        messages_to_process_for_history = all_ui_messages
        if reuse_last_user_bubble and all_ui_messages:
            # If we are reusing the last bubble, it means it's already in the UI.
            # The history should include everything *before* that reused bubble.
            # Find the index of the last_msg_widget (which is the one being reused)
            try:
                # 'last_msg_widget' would have been set if reuse_last_user_bubble is True
                # This assumes last_msg_widget is still a valid reference from the reuse logic block
                idx_of_reused_msg = -1
                # Search for the widget instance if `last_msg_widget` is not directly available
                # or if we need to be more robust:
                temp_last_user_msg_widget = None
                for widget in reversed(all_ui_messages):
                    if widget.role == "User":
                        temp_last_user_msg_widget = widget
                        break
                if temp_last_user_msg_widget:
                    idx_of_reused_msg = all_ui_messages.index(temp_last_user_msg_widget)

                if idx_of_reused_msg != -1:
                    messages_to_process_for_history = all_ui_messages[:idx_of_reused_msg]
            except (ValueError, NameError): # NameError if last_msg_widget wasn't set, ValueError if not found
                 loguru_logger.warning("Could not definitively exclude reused message from history; sending full history.")
                 # Fallback: send all current UI messages as history; API might get duplicate of last user msg.
                 # `app.chat_wrapper` or `chat()` would need to handle this.
                 pass


        for msg_widget in messages_to_process_for_history:
            if msg_widget.role in ("User", "AI") or (app.current_chat_active_character_data and msg_widget.role == app.current_chat_active_character_data.get('name')):
                 if msg_widget.generation_complete: # Only send completed messages
                    # Map UI role to API role (user/assistant)
                    api_role = "user"
                    if msg_widget.role != "User": # Anything not "User" is treated as assistant for API history
                        api_role = "assistant"

                    # Prepare content part(s) - for now, assuming text only
                    content_for_api = msg_widget.message_text
                    # if msg_widget.image_data and msg_widget.image_mime_type: # Future multimodal
                    #     image_url = f"data:{msg_widget.image_mime_type};base64,{base64.b64encode(msg_widget.image_data).decode()}"
                    #     content_for_api = [
                    #         {"type": "text", "text": msg_widget.message_text},
                    #         {"type": "image_url", "image_url": {"url": image_url}}
                    #     ]
                    chat_history_for_api.append({"role": api_role, "content": content_for_api})
        loguru_logger.debug(f"Built chat history for API with {len(chat_history_for_api)} messages.")

    except Exception as e:
        loguru_logger.error(f"Failed to build chat history for API: {e}", exc_info=True)
        await chat_container.mount(ChatMessage(Text.from_markup("Internal Error: Could not retrieve chat history."), role="System", classes="-error"))
        return

    # --- 5. DB and Conversation ID Setup ---
    active_conversation_id = app.current_chat_conversation_id
    db = app.chachanotes_db # Use the correct instance from app
    user_msg_widget_instance: Optional[ChatMessage] = None

    # --- 6. Mount User Message to UI ---
    if not reuse_last_user_bubble:
        user_msg_widget_instance = ChatMessage(message_text_from_input, role="User")
        await chat_container.mount(user_msg_widget_instance)
        loguru_logger.debug(f"Mounted new user message to UI: '{message_text_from_input[:50]}...'")
        
        # Update token counter after adding user message
        try:
            from .chat_token_events import update_chat_token_counter
            await update_chat_token_counter(app)
        except Exception as e:
            loguru_logger.debug(f"Could not update token counter: {e}")

    # --- 7. Save User Message to DB (IF CHAT IS ALREADY PERSISTENT) ---
    if not app.current_chat_is_ephemeral and active_conversation_id and db:
        if not reuse_last_user_bubble and user_msg_widget_instance:
            try:
                loguru_logger.debug(f"Chat is persistent (ID: {active_conversation_id}). Saving user message to DB.")
                user_message_db_id_version_tuple = ccl.add_message_to_conversation(
                    db, conversation_id=active_conversation_id, sender="User", content=message_text_from_input,
                    image_data=None, image_mime_type=None
                )
                # add_message_to_conversation in ccl returns message_id (str). Version is handled by DB.
                # We need to fetch the message to get its version.
                if user_message_db_id_version_tuple: # This is just the ID
                    user_msg_db_id = user_message_db_id_version_tuple
                    saved_user_msg_details = db.get_message_by_id(user_msg_db_id)
                    if saved_user_msg_details:
                        user_msg_widget_instance.message_id_internal = saved_user_msg_details.get('id')
                        user_msg_widget_instance.message_version_internal = saved_user_msg_details.get('version')
                        loguru_logger.debug(f"User message saved to DB. ID: {saved_user_msg_details.get('id')}, Version: {saved_user_msg_details.get('version')}")
                    else:
                        loguru_logger.error(f"Failed to retrieve saved user message details from DB for ID {user_msg_db_id}.")
                else:
                    loguru_logger.error(f"Failed to save user message to DB for conversation {active_conversation_id}.")
            except (CharactersRAGDBError, InputError) as e_add_msg: # Catch specific errors from ccl
                loguru_logger.error(f"Error saving user message to DB: {e_add_msg}", exc_info=True)
            except Exception as e_add_msg_generic:
                 loguru_logger.error(f"Generic error saving user message to DB: {e_add_msg_generic}", exc_info=True)

    elif app.current_chat_is_ephemeral:
        loguru_logger.debug("Chat is ephemeral. User message not saved to DB at this stage.")


    # --- 8. UI Updates (Clear input, scroll, focus) ---
    chat_container.scroll_end(animate=True) # Scroll after mounting user message
    text_area.clear()
    text_area.focus()

    # --- 9. API Key Fetching ---
    api_key_for_call = None
    if selected_provider:
        provider_settings_key = selected_provider.lower().replace(" ", "_")
        provider_config_settings = app.app_config.get("api_settings", {}).get(provider_settings_key, {})

        if "api_key" in provider_config_settings:
            direct_config_key_checked = True
            config_api_key = provider_config_settings.get("api_key", "").strip()
            if config_api_key and config_api_key != "<API_KEY_HERE>":
                api_key_for_call = config_api_key
                loguru_logger.debug(f"Using API key for '{selected_provider}' from config file field.")

        if not api_key_for_call: # If not found in direct 'api_key' field or it was empty
            env_var_name = provider_config_settings.get("api_key_env_var", "").strip()
            if env_var_name:
                env_api_key = os.environ.get(env_var_name, "").strip()
                if env_api_key:
                    api_key_for_call = env_api_key
                    loguru_logger.debug(f"Using API key for '{selected_provider}' from ENV var '{env_var_name}'.")
                else:
                    loguru_logger.debug(f"ENV var '{env_var_name}' for '{selected_provider}' not found or empty.")
            else:
                loguru_logger.debug(f"No 'api_key_env_var' specified for '{selected_provider}' in config.")

    providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]
    if selected_provider in providers_requiring_key and not api_key_for_call:
        loguru_logger.error(f"API Key for '{selected_provider}' is missing and required.")
        error_message_markup = (
            f"API Key for {selected_provider} is missing.\n\n"
            "Please add it to your config file under:\n"
            f"\\[api_settings.{selected_provider.lower().replace(' ', '_')}\\]\n" 
            "api_key = \"YOUR_KEY\"\n\n"
            "Or set the environment variable specified by 'api_key_env_var' in the config for this provider."
        )
        await chat_container.mount(ChatMessage(message=error_message_markup, role="System"))
        if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
            await app.current_ai_message_widget.remove()
            app.current_ai_message_widget = None
        return

    # --- 10. Mount Placeholder AI Message ---
    ai_placeholder_widget = ChatMessage(
        message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}",
        role="AI", generation_complete=False
    )
    await chat_container.mount(ai_placeholder_widget)
    chat_container.scroll_end(animate=False) # Scroll after mounting placeholder
    app.current_ai_message_widget = ai_placeholder_widget

    # --- 10.5. Apply RAG Context if enabled ---
    rag_context = None
    message_text_with_rag = message_text_from_input
    
    try:
        from .chat_rag_events import get_rag_context_for_chat
        
        # Get RAG context for the message
        rag_context = await get_rag_context_for_chat(app, message_text_from_input)
        if rag_context:
            loguru_logger.info(f"RAG context retrieved, length: {len(rag_context)} chars")
            # Prepend RAG context to the user message
            message_text_with_rag = rag_context + message_text_from_input
        else:
            message_text_with_rag = message_text_from_input
    except ImportError:
        loguru_logger.debug("RAG events not available - skipping RAG context")
    except Exception as e:
        loguru_logger.error(f"Error getting RAG context: {e}", exc_info=True)
    
    # --- 11. Prepare and Dispatch API Call via Worker ---
    loguru_logger.debug(f"Dispatching API call to worker. Current message: '{message_text_with_rag[:50]}...', History items: {len(chat_history_for_api)}")

    # Log API parameters for debugging
    api_params = {
        "provider": selected_provider,
        "model": selected_model,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "max_tokens": llm_max_tokens_value,
        "streaming": should_stream,
        "system_prompt_length": len(final_system_prompt_for_api) if final_system_prompt_for_api else 0
    }
    loguru_logger.debug(f"API parameters: {api_params}")

    # Set current_chat_is_streaming before running the worker using thread-safe method
    app.set_current_chat_is_streaming(should_stream)
    loguru_logger.info(f"Set app.current_chat_is_streaming to: {should_stream}")

    worker_target = lambda: app.chat_wrapper(
        message=message_text_with_rag, # Current user utterance with RAG context
        history=chat_history_for_api,    # History *before* current utterance
        api_endpoint=selected_provider,
        api_key=api_key_for_call,
        custom_prompt=custom_prompt,
        temperature=temperature,
        system_message=final_system_prompt_for_api,
        streaming=should_stream,
        minp=min_p,
        model=selected_model,
        topp=top_p,
        topk=top_k,
        llm_max_tokens=llm_max_tokens_value,
        llm_seed=llm_seed_value,
        llm_stop=llm_stop_value,
        llm_response_format=llm_response_format_value,
        llm_n=llm_n_value,
        llm_user_identifier=llm_user_identifier_value,
        llm_logprobs=llm_logprobs_value,
        llm_top_logprobs=llm_top_logprobs_value,
        llm_logit_bias=llm_logit_bias_value,
        llm_presence_penalty=llm_presence_penalty_value,
        llm_frequency_penalty=llm_frequency_penalty_value,
        llm_tools=llm_tools_value,
        llm_tool_choice=llm_tool_choice_value,
        llm_fixed_tokens_kobold=llm_fixed_tokens_kobold_value, # Added new parameter
        media_content={}, # Placeholder for now
        selected_parts=[], # Placeholder for now
        chatdict_entries=None, # Placeholder for now
        max_tokens=500, # This is the existing chatdict max_tokens, distinct from llm_max_tokens
        strategy="sorted_evenly", # Default or get from config/UI
        strip_thinking_tags=strip_thinking_tags_value # Pass the new setting
    )
    worker = app.run_worker(worker_target, name=f"API_Call_{prefix}",
                   group="api_calls",
                   thread=True,
                   description=f"Calling {selected_provider}")
    app.set_current_chat_worker(worker)


async def handle_chat_action_button_pressed(app: 'TldwCli', button: Button, action_widget: ChatMessage) -> None:
    button_classes = button.classes
    message_text = action_widget.message_text  # This is the raw, unescaped text
    message_role = action_widget.role
    db = app.notes_service._get_db(app.notes_user_id) if app.notes_service else None

    if "edit-button" in button_classes:
        loguru_logger.info("Action: Edit clicked for %s message: '%s...'", message_role, message_text[:50])
        is_editing = getattr(action_widget, "_editing", False)
        static_text_widget: Static = action_widget.query_one(".message-text", Static)

        if not is_editing:  # Start editing
            current_text_for_editing = message_text  # Use the internally stored raw text
            static_text_widget.display = False
            editor = TextArea(text=current_text_for_editing, id="edit-area", classes="edit-area")
            editor.styles.width = "100%"
            await action_widget.mount(editor, before=static_text_widget)
            editor.focus()
            action_widget._editing = True
            button.label = get_char(EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT)
            loguru_logger.debug("Editing started.")
        else:  # Stop editing and save
            try:
                editor: TextArea = action_widget.query_one("#edit-area", TextArea)
                new_text = editor.text  # This is plain text from TextArea
                await editor.remove()

                action_widget.message_text = new_text  # Update internal raw text
                # --- DO NOT REMOVE ---
                # When updating the Static widget, explicitly pass the new_text
                # as a plain rich.text.Text object. This tells Textual
                # to render it as is, without trying to parse for markup.
                static_text_widget.update(Text(new_text))
                # --- DO NOT REMOVE ---
                #static_text_widget.update(escape_markup(new_text))  # Update display with escaped text

                static_text_widget.display = True
                action_widget._editing = False
                button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)  # Reset to Edit icon
                loguru_logger.debug("Editing finished. New length: %d", len(new_text))

                # Persist edit to DB if message has an ID
                if db and action_widget.message_id_internal and action_widget.message_version_internal is not None:
                    try:
                        # CORRECTED: Use ccl.edit_message_content
                        success = ccl.edit_message_content(
                            db,
                            action_widget.message_id_internal,
                            new_text,
                            action_widget.message_version_internal  # Pass the expected version
                        )
                        if success:
                            action_widget.message_version_internal += 1  # Increment version on successful update
                            loguru_logger.info(
                                f"Message ID {action_widget.message_id_internal} content updated in DB. New version: {action_widget.message_version_internal}")
                            app.notify("Message edit saved to DB.", severity="information", timeout=2)
                        else:
                            # This path should ideally be covered by exceptions from ccl.edit_message_content
                            loguru_logger.error(
                                f"ccl.edit_message_content returned False for {action_widget.message_id_internal} without raising an exception.")
                            app.notify("Failed to save edit to DB (update operation returned false).", severity="error")
                    except ConflictError as e_conflict:
                        loguru_logger.error(
                            f"Conflict updating message {action_widget.message_id_internal} in DB: {e_conflict}",
                            exc_info=True)
                        app.notify(f"Save conflict: {e_conflict}. Please reload the chat or message.", severity="error",
                                   timeout=7)
                    except (CharactersRAGDBError, InputError) as e_db_update:
                        loguru_logger.error(
                            f"DB/Input error updating message {action_widget.message_id_internal} in DB: {e_db_update}",
                            exc_info=True)
                        app.notify(f"Failed to save edit to DB: {e_db_update}", severity="error")
                    except Exception as e_generic_update:  # Catch any other unexpected error
                        loguru_logger.error(
                            f"Unexpected error updating message {action_widget.message_id_internal} in DB: {e_generic_update}",
                            exc_info=True)
                        app.notify(f"An unexpected error occurred while saving the edit: {e_generic_update}",
                                   severity="error")

            except QueryError:
                loguru_logger.error("Edit TextArea not found when stopping edit. Restoring original.")
                static_text_widget.update(Text(message_text))  # Restore original escaped text
                static_text_widget.display = True
                action_widget._editing = False
                button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)
            except Exception as e_edit_stop:
                loguru_logger.error(f"Error stopping edit: {e_edit_stop}", exc_info=True)
                if 'static_text_widget' in locals() and static_text_widget.is_mounted:
                    static_text_widget.update(Text(message_text))  # Restore with Text()
                    static_text_widget.display = True
                if hasattr(action_widget, '_editing'): action_widget._editing = False
                if 'button' in locals() and button.is_mounted: button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)


    elif "copy-button" in button_classes:
        logging.info("Action: Copy clicked for %s message: '%s...'", message_role, message_text[:50])
        app.copy_to_clipboard(message_text)  # message_text is already the raw, unescaped version
        app.notify("Message content copied to clipboard.", severity="information", timeout=2)
        button.label = get_char(EMOJI_COPIED, FALLBACK_COPIED) + "Copied"
        app.set_timer(1.5, lambda: setattr(button, "label", get_char(EMOJI_COPY, FALLBACK_COPY)))

    elif "speak-button" in button_classes:
        logging.info(f"Action: Speak clicked for {message_role} message: '{message_text[:50]}...'")
        # Actual TTS would go here. For UI feedback:
        try:
            text_widget = action_widget.query_one(".message-text", Static)
            original_display = text_widget.renderable  # Store to restore
            text_widget.update(f"Speaking: {message_text}")
            # After TTS simulation/actual call:
            # app.set_timer(3, lambda: text_widget.update(original_display)) # Example restore
        except QueryError:
            logging.error("Could not find .message-text Static for speak action.")


    elif "thumb-up-button" in button_classes:
        logging.info(f"Action: Thumb Up clicked for {message_role} message.")
        button.label = "👍(OK)"
        # Add DB interaction for feedback if needed

    elif "thumb-down-button" in button_classes:
        logging.info(f"Action: Thumb Down clicked for {message_role} message.")
        button.label = "👎(OK)"
        # Add DB interaction for feedback if needed

    elif "delete-button" in button_classes:
        logging.info("Action: Delete clicked for %s message: '%s...'", message_role, message_text[:50])
        message_id_to_delete = getattr(action_widget, 'message_id_internal', None)
        try:
            await action_widget.remove()
            if action_widget is app.current_ai_message_widget:
                app.current_ai_message_widget = None

            if db and message_id_to_delete:
                try:
                    db.soft_delete_message(message_id_to_delete)  # Assuming soft delete
                    loguru_logger.info(f"Message ID {message_id_to_delete} soft-deleted from DB.")
                    app.notify("Message deleted.", severity="information", timeout=2)
                except Exception as e_db_delete:
                    loguru_logger.error(f"Failed to delete message {message_id_to_delete} from DB: {e_db_delete}",
                                        exc_info=True)
                    app.notify("Failed to delete message from DB.", severity="error")
        except Exception as exc:
            logging.error("Failed to delete message widget: %s", exc, exc_info=True)

    elif "regenerate-button" in button_classes and message_role == "AI":
        loguru_logger.info(
            f"Action: Regenerate clicked for AI message ID: {getattr(action_widget, 'message_id_internal', 'N/A')}")
        prefix = "chat"  # Assuming regeneration only happens in the main chat tab
        try:
            chat_container = app.query_one(f"#{prefix}-log", VerticalScroll)
        except QueryError:
            loguru_logger.error(f"Regenerate: Could not find chat container #{prefix}-log. Aborting.")
            app.notify("Error: Chat log not found for regeneration.", severity="error")
            return

        history_for_regeneration = []
        widgets_to_remove_after_regen_source = []  # Renamed for clarity
        found_target_ai_message_for_regen = False

        all_message_widgets_in_log = list(chat_container.query(ChatMessage))

        for msg_widget_iter in all_message_widgets_in_log:
            if msg_widget_iter is action_widget:  # This is the AI message we're regenerating
                found_target_ai_message_for_regen = True
                widgets_to_remove_after_regen_source.append(msg_widget_iter)
                # Don't add this AI message to history_for_regeneration
                continue

            if found_target_ai_message_for_regen:
                # All messages *after* the AI message being regenerated should also be removed
                widgets_to_remove_after_regen_source.append(msg_widget_iter)
            else:
                # This message is *before* the one we're regenerating
                if msg_widget_iter.role in ("User", "AI") and msg_widget_iter.generation_complete:
                    role_for_api = "assistant" if msg_widget_iter.role == "AI" else "user"
                    history_for_regeneration.append({"role": role_for_api, "content": msg_widget_iter.message_text})

        if not history_for_regeneration:
            loguru_logger.warning("Regenerate: No history found before the target AI message. Cannot regenerate.")
            app.notify("Cannot regenerate: No preceding messages found.", severity="warning")
            return

        loguru_logger.debug(
            f"Regenerate: History for regeneration ({len(history_for_regeneration)} messages): {history_for_regeneration}")

        for widget_to_remove_iter in widgets_to_remove_after_regen_source:
            loguru_logger.debug(
                f"Regenerate: Removing widget {widget_to_remove_iter.id} (Role: {widget_to_remove_iter.role}) from UI.")
            await widget_to_remove_iter.remove()

        if app.current_ai_message_widget in widgets_to_remove_after_regen_source:
            app.current_ai_message_widget = None

        # Fetch current chat settings (same as send-chat button logic)
        try:
            provider_widget_regen = app.query_one(f"#{prefix}-api-provider", Select)
            model_widget_regen = app.query_one(f"#{prefix}-api-model", Select)
            system_prompt_widget_regen = app.query_one(f"#{prefix}-system-prompt", TextArea)
            temp_widget_regen = app.query_one(f"#{prefix}-temperature", Input)
            top_p_widget_regen = app.query_one(f"#{prefix}-top-p", Input)
            min_p_widget_regen = app.query_one(f"#{prefix}-min-p", Input)
            top_k_widget_regen = app.query_one(f"#{prefix}-top-k", Input)
            # Full chat settings
            llm_max_tokens_widget_regen = app.query_one(f"#{prefix}-llm-max-tokens", Input)
            llm_seed_widget_regen = app.query_one(f"#{prefix}-llm-seed", Input)
            llm_stop_widget_regen = app.query_one(f"#{prefix}-llm-stop", Input)
            llm_response_format_widget_regen = app.query_one(f"#{prefix}-llm-response-format", Select)
            llm_n_widget_regen = app.query_one(f"#{prefix}-llm-n", Input)
            llm_user_identifier_widget_regen = app.query_one(f"#{prefix}-llm-user-identifier", Input)
            llm_logprobs_widget_regen = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
            llm_top_logprobs_widget_regen = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
            llm_logit_bias_widget_regen = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
            llm_presence_penalty_widget_regen = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
            llm_frequency_penalty_widget_regen = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
            llm_tools_widget_regen = app.query_one(f"#{prefix}-llm-tools", TextArea)
            llm_tool_choice_widget_regen = app.query_one(f"#{prefix}-llm-tool-choice", Input)
            # Query for the strip thinking tags checkbox for regeneration
            try:
                strip_tags_checkbox_regen = app.query_one("#chat-strip-thinking-tags-checkbox", Checkbox)
                strip_thinking_tags_value_regen = strip_tags_checkbox_regen.value
            except QueryError:
                loguru_logger.warning("Regenerate: Could not find '#chat-strip-thinking-tags-checkbox'. Defaulting to True.")
                strip_thinking_tags_value_regen = True
        except QueryError as e_query_regen:
            loguru_logger.error(f"Regenerate: Could not find UI settings widgets for '{prefix}': {e_query_regen}")
            await chat_container.mount(
                ChatMessage(Text.from_markup("[bold red]Internal Error:[/]\nMissing UI settings for regeneration."),
                            role="System", classes="-error"))
            return

        selected_provider_regen = str(
            provider_widget_regen.value) if provider_widget_regen.value != Select.BLANK else None
        selected_model_regen = str(model_widget_regen.value) if model_widget_regen.value != Select.BLANK else None
        system_prompt_regen = system_prompt_widget_regen.text
        temperature_regen = safe_float(temp_widget_regen.value, 0.7, "temperature")
        top_p_regen = safe_float(top_p_widget_regen.value, 0.95, "top_p")
        min_p_regen = safe_float(min_p_widget_regen.value, 0.05, "min_p")
        top_k_regen = safe_int(top_k_widget_regen.value, 50, "top_k")

        # --- Integration of Active Character Data & Streaming Config for REGENERATION ---
        active_char_data_regen = app.current_chat_active_character_data
        original_system_prompt_from_ui_regen = system_prompt_regen # Keep a reference

        if active_char_data_regen:
            loguru_logger.info(f"Active character data found for REGENERATION: {active_char_data_regen.get('name', 'Unnamed')}. Overriding system prompt.")
            system_prompt_override_regen = active_char_data_regen.get('system_prompt')
            if system_prompt_override_regen is not None:
                system_prompt_regen = system_prompt_override_regen
                loguru_logger.debug(f"System prompt for REGENERATION overridden by active character: '{system_prompt_regen[:100]}...'")
            else:
                loguru_logger.debug(f"Active character data present for REGENERATION, but 'system_prompt' is None or missing. Using: '{system_prompt_regen[:100]}...' (might be from UI or empty).")
        else:
            loguru_logger.info("No active character data for REGENERATION. Using system prompt from UI.")
        should_stream_regen = False  # Default for regen
        if selected_provider_regen:
            provider_settings_key_regen = selected_provider_regen.lower().replace(" ", "_")
            provider_specific_settings_regen = app.app_config.get("api_settings", {}).get(provider_settings_key_regen,
                                                                                          {})
            should_stream_regen = provider_specific_settings_regen.get("streaming", False)
            loguru_logger.debug(
                f"Streaming for REGENERATION with {selected_provider_regen} set to {should_stream_regen} based on config.")
        else:
            loguru_logger.debug("No provider selected for REGENERATION, streaming defaults to False.")
        # --- End of Integration & Streaming Config for REGENERATION ---

        llm_max_tokens_value_regen = safe_int(llm_max_tokens_widget_regen.value, 1024, "llm_max_tokens")
        llm_seed_value_regen = safe_int(llm_seed_widget_regen.value, None, "llm_seed")
        llm_stop_value_regen = [s.strip() for s in
                                llm_stop_widget_regen.value.split(',')] if llm_stop_widget_regen.value.strip() else None
        llm_response_format_value_regen = {"type": str(
            llm_response_format_widget_regen.value)} if llm_response_format_widget_regen.value != Select.BLANK else {
            "type": "text"}
        llm_n_value_regen = safe_int(llm_n_widget_regen.value, 1, "llm_n")
        llm_user_identifier_value_regen = llm_user_identifier_widget_regen.value.strip() or None
        llm_logprobs_value_regen = llm_logprobs_widget_regen.value
        llm_top_logprobs_value_regen = safe_int(llm_top_logprobs_widget_regen.value, 0,
                                                     "llm_top_logprobs") if llm_logprobs_value_regen else 0
        llm_presence_penalty_value_regen = safe_float(llm_presence_penalty_widget_regen.value, 0.0,
                                                           "llm_presence_penalty")
        llm_frequency_penalty_value_regen = safe_float(llm_frequency_penalty_widget_regen.value, 0.0,
                                                            "llm_frequency_penalty")
        llm_tool_choice_value_regen = llm_tool_choice_widget_regen.value.strip() or None
        try:
            llm_logit_bias_text_regen = llm_logit_bias_widget_regen.text.strip()
            llm_logit_bias_value_regen = json.loads(
                llm_logit_bias_text_regen) if llm_logit_bias_text_regen and llm_logit_bias_text_regen != "{}" else None
        except json.JSONDecodeError:
            llm_logit_bias_value_regen = None
        try:
            llm_tools_text_regen = llm_tools_widget_regen.text.strip()
            llm_tools_value_regen = json.loads(
                llm_tools_text_regen) if llm_tools_text_regen and llm_tools_text_regen != "[]" else None
        except json.JSONDecodeError:
            llm_tools_value_regen = None

        if not selected_provider_regen or not selected_model_regen:
            loguru_logger.warning("Regenerate: Provider or model not selected.")
            await chat_container.mount(
                ChatMessage(Text.from_markup("[bold red]Error:[/]\nPlease select provider and model for regeneration."),
                            role="System", classes="-error"))
            return

        api_key_for_regen = None  # API Key fetching logic (same as send-chat)
        provider_settings_key_regen = selected_provider_regen.lower()
        provider_config_settings_regen = app.app_config.get("api_settings", {}).get(provider_settings_key_regen, {})
        if provider_config_settings_regen.get("api_key"):
            api_key_for_regen = provider_config_settings_regen["api_key"]
        elif provider_config_settings_regen.get("api_key_env_var"):
            api_key_for_regen = os.environ.get(provider_config_settings_regen["api_key_env_var"])

        providers_requiring_key_regen = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter",
                                         "HuggingFace", "DeepSeek"]
        if selected_provider_regen in providers_requiring_key_regen and not api_key_for_regen:
            loguru_logger.error(
                f"Regenerate aborted: API Key for required provider '{selected_provider_regen}' is missing.")
            await chat_container.mount(ChatMessage(
                Text.from_markup(f"[bold red]API Key for {selected_provider_regen} is missing for regeneration.[/]"),
                role="System", classes="-error"))
            return

        ai_placeholder_widget_regen = ChatMessage(
            message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)} (Regenerating...)",
            role="AI", generation_complete=False
        )
        await chat_container.mount(ai_placeholder_widget_regen)
        chat_container.scroll_end(animate=False)
        app.current_ai_message_widget = ai_placeholder_widget_regen

        # The "message" to chat_wrapper is empty because we're using the history
        worker_target_regen = lambda: app.chat_wrapper(
            message="", history=history_for_regeneration, api_endpoint=selected_provider_regen,
            api_key=api_key_for_regen,
            custom_prompt="", temperature=temperature_regen, system_message=system_prompt_regen, streaming=should_stream_regen,
            minp=min_p_regen, model=selected_model_regen, topp=top_p_regen, topk=top_k_regen,
            llm_max_tokens=llm_max_tokens_value_regen, llm_seed=llm_seed_value_regen, llm_stop=llm_stop_value_regen,
            llm_response_format=llm_response_format_value_regen, llm_n=llm_n_value_regen,
            llm_user_identifier=llm_user_identifier_value_regen, llm_logprobs=llm_logprobs_value_regen,
            llm_top_logprobs=llm_top_logprobs_value_regen, llm_logit_bias=llm_logit_bias_value_regen,
            llm_presence_penalty=llm_presence_penalty_value_regen,
            llm_frequency_penalty=llm_frequency_penalty_value_regen,
            llm_tools=llm_tools_value_regen, llm_tool_choice=llm_tool_choice_value_regen,
            strip_thinking_tags=strip_thinking_tags_value_regen, # Pass for regeneration
            media_content={}, selected_parts=[], chatdict_entries=None, max_tokens=500, strategy="sorted_evenly"
        )
        worker = app.run_worker(worker_target_regen, name=f"API_Call_{prefix}_regenerate", group="api_calls", thread=True,
                       description=f"Regenerating for {selected_provider_regen}")
        app.set_current_chat_worker(worker)

    elif "continue-button" in button_classes and message_role == "AI":
        loguru_logger.info(
            f"Action: Continue clicked for AI message ID: {getattr(action_widget, 'message_id_internal', 'N/A')}"
        )
        # Create a Button.Pressed event for the continue handler
        button_event = Button.Pressed(button)
        # Call the continue response handler
        await handle_continue_response_button_pressed(app, button_event, action_widget)


async def handle_chat_new_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    loguru_logger.info("New Chat button pressed.")
    try:
        chat_log_widget = app.query_one("#chat-log", VerticalScroll)
        await chat_log_widget.remove_children()
    except QueryError:
        loguru_logger.error("Failed to find #chat-log to clear.")

    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True  # This triggers watcher to update UI elements
    app.current_chat_active_character_data = None
    try:
        default_system_prompt = app.app_config.get("chat_defaults", {}).get("system_prompt", "You are a helpful AI assistant.")
        app.query_one("#chat-system-prompt", TextArea).text = default_system_prompt
        loguru_logger.debug("Reset main system prompt to default on new chat.")
    except QueryError:
        loguru_logger.error("Could not find #chat-system-prompt to reset on new chat.")
    try:
        app.query_one("#chat-character-name-edit", Input).value = ""
        app.query_one("#chat-character-description-edit", TextArea).text = ""
        app.query_one("#chat-character-personality-edit", TextArea).text = ""
        app.query_one("#chat-character-scenario-edit", TextArea).text = ""
        app.query_one("#chat-character-system-prompt-edit", TextArea).text = ""
        app.query_one("#chat-character-first-message-edit", TextArea).text = ""
        # Optionally clear the character search and list
        # app.query_one("#chat-character-search-input", Input).value = ""
        # await app.query_one("#chat-character-search-results-list", ListView).clear()
        loguru_logger.debug("Cleared character editing fields on new chat.")
    except QueryError as e:
        loguru_logger.warning(f"Could not clear all character edit fields on new chat: {e}")
    
    # Update token counter to show empty state
    try:
        from .chat_token_events import update_chat_token_counter
        await update_chat_token_counter(app)
    except Exception as e:
        loguru_logger.debug(f"Could not update token counter: {e}")

    try:
        # Watcher should handle most of this, but explicit clearing is safer
        app.query_one("#chat-conversation-title-input", Input).value = ""
        app.query_one("#chat-conversation-keywords-input", TextArea).text = ""
        app.query_one("#chat-conversation-uuid-display", Input).value = "Ephemeral Chat"
        app.query_one(TitleBar).reset_title()
        app.query_one("#chat-input", TextArea).focus()
    except QueryError as e:
        loguru_logger.error(f"UI component not found during new chat setup: {e}")


async def handle_chat_save_current_chat_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    loguru_logger.info("Save Current Chat button pressed.")
    if not (app.current_chat_is_ephemeral and app.current_chat_conversation_id is None):
        loguru_logger.warning("Chat not eligible for saving (not ephemeral or already has ID).")
        app.notify("This chat is already saved or cannot be saved in its current state.", severity="warning")
        return

    if not app.chachanotes_db: # Use correct DB instance name
        app.notify("Database service not available.", severity="error")
        loguru_logger.error("chachanotes_db not available for saving chat.")
        return

    db = app.chachanotes_db
    try:
        chat_log_widget = app.query_one("#chat-log", VerticalScroll)
    except QueryError:
        app.notify("Chat log not found, cannot save.", severity="error")
        return

    messages_in_log = list(chat_log_widget.query(ChatMessage))

    if not messages_in_log:
        app.notify("Nothing to save in an empty chat.", severity="warning")
        return

    character_id_for_saving = ccl.DEFAULT_CHARACTER_ID
    char_name_for_sender = "AI" # Default sender name for AI messages if no specific character

    if app.current_chat_active_character_data and 'id' in app.current_chat_active_character_data:
        character_id_for_saving = app.current_chat_active_character_data['id']
        char_name_for_sender = app.current_chat_active_character_data.get('name', 'AI') # Use actual char name for sender
        loguru_logger.info(f"Saving chat with active character: {char_name_for_sender} (ID: {character_id_for_saving})")
    else:
        loguru_logger.info(f"Saving chat with default character association (ID: {character_id_for_saving})")


    ui_messages_to_save: List[Dict[str, Any]] = []
    for msg_widget in messages_in_log:
        sender_for_db_initial_msg = "User" if msg_widget.role == "User" else char_name_for_sender

        if msg_widget.generation_complete :
            ui_messages_to_save.append({
                'sender': sender_for_db_initial_msg,
                'content': msg_widget.message_text,
                'image_data': msg_widget.image_data,
                'image_mime_type': msg_widget.image_mime_type,
            })

    new_conv_title_from_ui = app.query_one("#chat-conversation-title-input", Input).value.strip()
    final_title_for_db = new_conv_title_from_ui

    if not final_title_for_db:
        # Use character's name for title generation if a specific character is active
        title_char_name_part = char_name_for_sender if character_id_for_saving != ccl.DEFAULT_CHARACTER_ID else "Assistant"
        if ui_messages_to_save and ui_messages_to_save[0]['sender'] == "User":
            content_preview = ui_messages_to_save[0]['content'][:30].strip()
            if content_preview:
                final_title_for_db = f"Chat: {content_preview}..."
            else:
                final_title_for_db = f"Chat with {title_char_name_part}"
        else:
            final_title_for_db = f"Chat with {title_char_name_part} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"


    keywords_str_from_ui = app.query_one("#chat-conversation-keywords-input", TextArea).text.strip()
    keywords_list_for_db = [kw.strip() for kw in keywords_str_from_ui.split(',') if kw.strip() and not kw.strip().startswith("__")]


    try:
        new_conv_id = ccl.create_conversation(
            db,
            title=final_title_for_db,
            character_id=character_id_for_saving,
            initial_messages=ui_messages_to_save,
            system_keywords=keywords_list_for_db,
            user_name_for_placeholders=app.app_config.get("USERS_NAME", "User")
        )

        if new_conv_id:
            app.current_chat_conversation_id = new_conv_id
            app.current_chat_is_ephemeral = False  # Now it's saved, triggers watcher
            app.notify("Chat saved successfully!", severity="information")

            # After saving, reload the conversation to get all messages with their DB IDs and versions
            await display_conversation_in_chat_tab_ui(app, new_conv_id)

            # The display_conversation_in_chat_tab_ui will populate title, uuid, keywords.
            # It will also set the title bar.

        else:
            app.notify("Failed to save chat (no ID returned).", severity="error")

    except Exception as e_save_chat:
        loguru_logger.error(f"Exception while saving chat: {e_save_chat}", exc_info=True)
        app.notify(f"Error saving chat: {str(e_save_chat)[:100]}", severity="error")


async def handle_chat_save_details_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    loguru_logger.info("Save conversation details button pressed.")
    if app.current_chat_is_ephemeral or not app.current_chat_conversation_id:
        loguru_logger.warning("Cannot save details for an ephemeral or non-existent chat.")
        app.notify("No active saved conversation to update details for.", severity="warning")
        return

    if not app.notes_service:
        loguru_logger.error("Notes service not available for saving chat details.")
        app.notify("Database service not available.", severity="error")
        return

    conversation_id = app.current_chat_conversation_id
    db = app.notes_service._get_db(app.notes_user_id)

    try:
        title_input = app.query_one("#chat-conversation-title-input", Input)
        keywords_input_widget = app.query_one("#chat-conversation-keywords-input", TextArea)

        new_title = title_input.value.strip()
        new_keywords_str = keywords_input_widget.text.strip()

        conv_details = db.get_conversation_by_id(conversation_id)
        if not conv_details:
            loguru_logger.error(f"Conversation {conversation_id} not found in DB for saving details.")
            app.notify("Error: Conversation not found in database.", severity="error")
            return

        current_version = conv_details.get('version')
        if current_version is None:
            loguru_logger.error(f"Conversation {conversation_id} is missing version information.")
            app.notify("Error: Conversation version information is missing.", severity="error")
            return

        title_changed = False
        if new_title != conv_details.get('title', ''):  # Compare with empty string if title is None
            db.update_conversation(conversation_id, {'title': new_title}, current_version)
            current_version += 1  # Version is now incremented for the conversation row
            title_changed = True
            loguru_logger.info(f"Title updated for conversation {conversation_id}. New version: {current_version}")
            try:
                app.query_one(TitleBar).update_title(f"Chat - {new_title}")
            except QueryError:
                loguru_logger.error("Failed to update TitleBar after title save.")

        # Keywords Update (from app.py, adapted)
        all_db_keywords_list = db.get_keywords_for_conversation(conversation_id)
        db_user_keywords_map = {kw['keyword']: kw['id'] for kw in all_db_keywords_list if
                                not kw['keyword'].startswith("__")}
        db_user_keywords_set = set(db_user_keywords_map.keys())
        ui_user_keywords_set = {kw.strip() for kw in new_keywords_str.split(',') if
                                kw.strip() and not kw.strip().startswith("__")}

        keywords_to_add = ui_user_keywords_set - db_user_keywords_set
        keywords_to_remove_text = db_user_keywords_set - ui_user_keywords_set
        keywords_changed = False

        for keyword_text_add in keywords_to_add:
            keyword_detail_add = db.get_keyword_by_text(keyword_text_add)  # Does not take user_id
            keyword_id_to_link = None
            if not keyword_detail_add:  # Keyword doesn't exist globally
                added_kw_id = db.add_keyword(keyword_text_add)  # Takes no user_id, returns int ID
                if isinstance(added_kw_id, int):
                    keyword_id_to_link = added_kw_id
                else:
                    logging.error(f"Failed to add keyword '{keyword_text_add}', received: {added_kw_id}"); continue
            else:
                keyword_id_to_link = keyword_detail_add['id']

            if keyword_id_to_link:
                db.link_conversation_to_keyword(conversation_id, keyword_id_to_link)
                keywords_changed = True

        for keyword_text_remove in keywords_to_remove_text:
            keyword_id_to_unlink = db_user_keywords_map.get(keyword_text_remove)
            if keyword_id_to_unlink:
                db.unlink_conversation_from_keyword(conversation_id, keyword_id_to_unlink)
                keywords_changed = True

        if title_changed or keywords_changed:
            app.notify("Conversation details saved!", severity="information", timeout=3)
            # Refresh keywords in UI to reflect any changes
            final_db_keywords_after_save = db.get_keywords_for_conversation(conversation_id)
            final_user_keywords_after_save = [kw['keyword'] for kw in final_db_keywords_after_save if
                                              not kw['keyword'].startswith("__")]
            keywords_input_widget.text = ", ".join(final_user_keywords_after_save)
        else:
            app.notify("No changes to save.", severity="information", timeout=2)

    except QueryError as e_query:
        loguru_logger.error(f"Save Conversation Details: UI component not found: {e_query}", exc_info=True)
        app.notify("Error accessing UI fields.", severity="error", timeout=3)
    except ConflictError as e_conflict:
        loguru_logger.error(f"Conflict saving conversation details for {conversation_id}: {e_conflict}", exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload.", severity="error", timeout=5)
    except CharactersRAGDBError as e_db:  # More generic DB error
        loguru_logger.error(f"DB error saving conversation details for {conversation_id}: {e_db}", exc_info=True)
        app.notify("Database error saving details.", severity="error", timeout=3)
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error saving conversation details for {conversation_id}: {e_unexp}",
                            exc_info=True)
        app.notify("Unexpected error saving details.", severity="error", timeout=3)


async def handle_chat_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    loguru_logger.info("Load selected chat button pressed.")
    try:
        results_list_view = app.query_one("#chat-conversation-search-results-list", ListView)
        highlighted_widget = results_list_view.highlighted_child

        if not isinstance(highlighted_widget, ListItem): # Check if it's a ListItem
            app.notify("No chat selected to load (not a list item).", severity="warning")
            loguru_logger.info("No conversation selected in the list to load (highlighted_widget is not ListItem).")
            return

        loaded_conversation_id: Optional[str] = getattr(highlighted_widget, 'conversation_id', None)

        if loaded_conversation_id is None:
            app.notify("No chat selected or item is invalid (missing conversation_id).", severity="warning")
            loguru_logger.info("No conversation_id found on the selected ListItem.")
            return

        loguru_logger.info(f"Attempting to load and display conversation ID: {loaded_conversation_id}")

        # _display_conversation_in_chat_tab handles UI updates and history loading
        await display_conversation_in_chat_tab_ui(app, loaded_conversation_id)

        app.current_chat_is_ephemeral = False  # A loaded chat is persistent

        conversation_title = getattr(highlighted_widget, 'conversation_title', 'Untitled')
        app.notify(f"Chat '{conversation_title}' loaded.", severity="information")

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found for loading chat: {e_query}", exc_info=True)
        app.notify("Error accessing UI for loading chat.", severity="error")
    except CharactersRAGDBError as e_db: # Make sure CharactersRAGDBError is imported
        loguru_logger.error(f"Database error loading chat: {e_db}", exc_info=True)
        app.notify("Database error loading chat.", severity="error")
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error loading chat: {e_unexp}", exc_info=True)
        app.notify("Unexpected error loading chat.", severity="error")


async def perform_chat_conversation_search(app: 'TldwCli') -> None:
    loguru_logger.debug("Performing chat conversation search...")
    try:
        search_bar = app.query_one("#chat-conversation-search-bar", Input)
        search_term = search_bar.value.strip()

        include_char_chats_checkbox = app.query_one("#chat-conversation-search-include-character-checkbox", Checkbox)
        include_character_chats = include_char_chats_checkbox.value  # Currently unused in DB query, filtered client side

        all_chars_checkbox = app.query_one("#chat-conversation-search-all-characters-checkbox", Checkbox)
        search_all_characters = all_chars_checkbox.value

        char_filter_select = app.query_one("#chat-conversation-search-character-filter-select", Select)
        selected_character_id_filter = char_filter_select.value if not char_filter_select.disabled and char_filter_select.value != Select.BLANK else None

        results_list_view = app.query_one("#chat-conversation-search-results-list", ListView)
        await results_list_view.clear()

        if not app.notes_service:
            loguru_logger.error("Notes service not available for conversation search.")
            await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
            return

        db = app.notes_service._get_db(app.notes_user_id)
        conversations: List[Dict[str, Any]] = []

        # The DB query `search_conversations_by_title` uses character_id=None for "all characters"
        # or a specific character_id. It doesn't have a separate flag for "regular chats only".
        # The `include_character_chats` checkbox logic might need client-side filtering if
        # "regular chats" mean `character_id IS NULL` specifically.

        # Simplified logic: if a specific character is chosen, filter by it. Otherwise, search all.
        # The `include_character_chats` primarily acts as a conceptual filter for now,
        # unless `search_conversations_by_title` is enhanced.

        effective_character_id_for_search = None
        if include_character_chats:
            if not search_all_characters and selected_character_id_filter:
                effective_character_id_for_search = selected_character_id_filter
            # if search_all_characters or selected_character_id_filter is None, effective_character_id_for_search remains None (search all)
        else:  # Only "regular" (non-character) chats. This requires DB support or client filter.
            loguru_logger.info(
                "Searching for regular chats only (character_id is NULL). This may require DB method enhancement or client-side filter.")
            # For now, let's assume search_conversations_by_title with character_id=ccl.DEFAULT_CHARACTER_ID
            # or some other marker for "regular" if your DB is structured that way.
            # Or if DEFAULT_CHARACTER_ID implies non-specific character chats
            effective_character_id_for_search = ccl.DEFAULT_CHARACTER_ID  # Placeholder assumption
            # A more robust way for "character_id IS NULL" would be a dedicated DB method or flag.

        loguru_logger.debug(
            f"Searching conversations. Term: '{search_term}', CharID for DB: {effective_character_id_for_search}, IncludeCharFlag: {include_character_chats}")
        
        # Handle different search scenarios
        if not search_term:
            # Empty search term - show all conversations based on filters
            if effective_character_id_for_search is not None and effective_character_id_for_search != ccl.DEFAULT_CHARACTER_ID:
                # Specific character selected - show all conversations for that character
                conversations = db.get_conversations_for_character(
                    character_id=effective_character_id_for_search,
                    limit=100
                )
            elif search_all_characters and include_character_chats:
                # "All Characters" checked - get all conversations (both regular and character chats)
                conversations = db.list_all_active_conversations(limit=100)
            elif effective_character_id_for_search == ccl.DEFAULT_CHARACTER_ID:
                # Regular chats only (non-character chats)
                # Get all conversations and filter for those without a character
                all_conversations = db.list_all_active_conversations(limit=100)
                conversations = [conv for conv in all_conversations 
                               if conv.get('character_id') == ccl.DEFAULT_CHARACTER_ID or conv.get('character_id') is None]
            else:
                # No specific filter - still show all conversations
                conversations = db.list_all_active_conversations(limit=100)
        else:
            # Search term provided - use the search function
            conversations = db.search_conversations_by_title(
                title_query=search_term,
                character_id=effective_character_id_for_search,  # This will be None if searching all/all_chars checked
                limit=100
            )

        # If include_character_chats is False, and the DB query couldn't filter by "IS NULL"
        # we might need to filter here:
        if not include_character_chats and conversations:
            # This assumes regular chats are those associated with DEFAULT_CHARACTER_ID
            # or if your DB uses NULL for truly "no character" chats. Adjust as needed.
            # conversations = [conv for conv in conversations if conv.get('character_id') == ccl.DEFAULT_CHARACTER_ID or conv.get('character_id') is None]
            # For now, we assume the DB call with DEFAULT_CHARACTER_ID handled this.
            pass

        if not conversations:
            await results_list_view.append(ListItem(Label("No conversations found.")))
        else:
            for conv_data in conversations:
                title_str = conv_data.get('title') or f"Chat ID: {conv_data['id'][:8]}..."
                # Optionally, prefix with character name if not already part of title logic
                # char_id_of_conv = conv_data.get('character_id')
                # if char_id_of_conv and char_id_of_conv != ccl.DEFAULT_CHARACTER_ID: # Example: don't prefix for default
                #     char_info = db.get_character_card_by_id(char_id_of_conv)
                #     if char_info and char_info.get('name'):
                #         title_str = f"[{char_info['name']}] {title_str}"

                item = ListItem(Label(title_str))
                item.conversation_id = conv_data['id']
                item.conversation_title = conv_data.get('title')  # Store for potential use
                # item.conversation_keywords = conv_data.get('keywords') # Not directly available from search_conversations_by_title
                await results_list_view.append(item)
        loguru_logger.info(f"Conversation search yielded {len(conversations)} results for display.")

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found during conversation search: {e_query}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: UI component missing.")))
            except (QueryError, AttributeError):
                # QueryError if results_list_view is not properly mounted/accessible
                # AttributeError if results_list_view is None or invalid
                pass
    except CharactersRAGDBError as e_db:
        loguru_logger.error(f"Database error during conversation search: {e_db}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: Database search failed.")))
            except (QueryError, AttributeError):
                # QueryError if results_list_view is not properly mounted/accessible
                # AttributeError if results_list_view is None or invalid
                pass
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error during conversation search: {e_unexp}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))
            except (QueryError, AttributeError):
                # QueryError if results_list_view is not properly mounted/accessible
                # AttributeError if results_list_view is None or invalid
                pass


async def handle_chat_conversation_search_bar_changed(app: 'TldwCli', event_value: str) -> None:
    if app._conversation_search_timer:
        app._conversation_search_timer.stop()  # Corrected: Use stop()
    app._conversation_search_timer = app.set_timer(
        0.5,
        lambda: perform_chat_conversation_search(app)
    )


async def handle_chat_search_checkbox_changed(app: 'TldwCli', checkbox_id: str, value: bool) -> None:

    loguru_logger.debug(f"Chat search checkbox '{checkbox_id}' changed to {value}")

    if checkbox_id == "chat-conversation-search-all-characters-checkbox":
        try:
            char_filter_select = app.query_one("#chat-conversation-search-character-filter-select", Select)
            char_filter_select.disabled = value
            if value:
                char_filter_select.value = Select.BLANK  # Clear selection when "All" is checked
        except QueryError as e:
            loguru_logger.error(f"Error accessing character filter select: {e}", exc_info=True)

    # Trigger a new search based on any checkbox change that affects the filter
    await perform_chat_conversation_search(app)


async def display_conversation_in_chat_tab_ui(app: 'TldwCli', conversation_id: str):
    if not app.chachanotes_db: # Use correct DB instance name
        loguru_logger.error("chachanotes_db unavailable, cannot display conversation in chat tab.")
        return

    db = app.chachanotes_db

    full_conv_data = ccl.get_conversation_details_and_messages(db, conversation_id)

    if not full_conv_data or not full_conv_data.get('metadata'):
        loguru_logger.error(f"Cannot display conversation: Details for ID {conversation_id} not found or incomplete.")
        app.notify(f"Error: Could not load chat {conversation_id}.", severity="error")
        # Update UI to reflect error state
        try:
            app.query_one("#chat-conversation-title-input", Input).value = "Error: Not Found"
            app.query_one("#chat-conversation-keywords-input", TextArea).text = ""
            app.query_one("#chat-conversation-uuid-display", Input).value = conversation_id
            app.query_one(TitleBar).update_title(f"Chat - Error Loading")
            chat_log_err = app.query_one("#chat-log", VerticalScroll)
            await chat_log_err.remove_children()
            await chat_log_err.mount(ChatMessage(Text.from_markup("[bold red]Failed to load conversation details.[/]"), role="System", classes="-error"))
        except QueryError as qe_err_disp: loguru_logger.error(f"UI component missing during error display for conv {conversation_id}: {qe_err_disp}")
        return

    conv_metadata = full_conv_data['metadata']
    db_messages = full_conv_data['messages']
    character_name_from_conv_load = full_conv_data.get('character_name', 'AI')

    app.current_chat_conversation_id = conversation_id
    app.current_chat_is_ephemeral = False

    try:
        character_id_from_conv = conv_metadata.get('character_id')
        loaded_char_data_for_ui_fields: Optional[Dict[str, Any]] = None
        current_user_name = app.app_config.get("USERS_NAME", "User")

        if character_id_from_conv and character_id_from_conv != ccl.DEFAULT_CHARACTER_ID:
            loguru_logger.debug(f"Conversation {conversation_id} is associated with char_id: {character_id_from_conv}")
            char_data_for_ui, _, _ = load_character_and_image(db, character_id_from_conv, current_user_name)
            if char_data_for_ui:
                app.current_chat_active_character_data = char_data_for_ui
                loaded_char_data_for_ui_fields = char_data_for_ui
                loguru_logger.info(f"Loaded char data for '{char_data_for_ui.get('name', 'Unknown')}' into app.current_chat_active_character_data.")
                app.query_one("#chat-system-prompt", TextArea).text = char_data_for_ui.get('system_prompt', '')
            else:
                app.current_chat_active_character_data = None
                loguru_logger.warning(f"Could not load char data for char_id: {character_id_from_conv}. Active char set to None.")
                app.query_one("#chat-system-prompt", TextArea).text = app.app_config.get("chat_defaults", {}).get("system_prompt", "You are a helpful AI assistant.")
        else:
            app.current_chat_active_character_data = None
            loguru_logger.debug(f"Conversation {conversation_id} uses default/no character. Active char set to None.")
            app.query_one("#chat-system-prompt", TextArea).text = app.app_config.get("chat_defaults", {}).get("system_prompt", "You are a helpful AI assistant.")

        right_sidebar_chat_tab = app.query_one("#chat-right-sidebar")
        if loaded_char_data_for_ui_fields:
            right_sidebar_chat_tab.query_one("#chat-character-name-edit", Input).value = loaded_char_data_for_ui_fields.get('name') or ''
            right_sidebar_chat_tab.query_one("#chat-character-description-edit", TextArea).text = loaded_char_data_for_ui_fields.get('description') or ''
            right_sidebar_chat_tab.query_one("#chat-character-personality-edit", TextArea).text = loaded_char_data_for_ui_fields.get('personality') or ''
            right_sidebar_chat_tab.query_one("#chat-character-scenario-edit", TextArea).text = loaded_char_data_for_ui_fields.get('scenario') or ''
            right_sidebar_chat_tab.query_one("#chat-character-system-prompt-edit", TextArea).text = loaded_char_data_for_ui_fields.get('system_prompt') or ''
            right_sidebar_chat_tab.query_one("#chat-character-first-message-edit", TextArea).text = loaded_char_data_for_ui_fields.get('first_message') or ''
        else:
            right_sidebar_chat_tab.query_one("#chat-character-name-edit", Input).value = ""
            right_sidebar_chat_tab.query_one("#chat-character-description-edit", TextArea).text = ""
            right_sidebar_chat_tab.query_one("#chat-character-personality-edit", TextArea).text = ""
            right_sidebar_chat_tab.query_one("#chat-character-scenario-edit", TextArea).text = ""
            right_sidebar_chat_tab.query_one("#chat-character-system-prompt-edit", TextArea).text = ""
            right_sidebar_chat_tab.query_one("#chat-character-first-message-edit", TextArea).text = ""

        app.query_one("#chat-conversation-title-input", Input).value = conv_metadata.get('title', '')
        app.query_one("#chat-conversation-uuid-display", Input).value = conversation_id

        keywords_input_disp = app.query_one("#chat-conversation-keywords-input", TextArea)
        keywords_input_disp.text = conv_metadata.get('keywords_display', "")

        app.query_one(TitleBar).update_title(f"Chat - {conv_metadata.get('title', 'Untitled Conversation')}")

        chat_log_widget_disp = app.query_one("#chat-log", VerticalScroll)
        await chat_log_widget_disp.remove_children()
        app.current_ai_message_widget = None

        for msg_data in db_messages:
            content_to_display = ccl.replace_placeholders(
                msg_data.get('content', ''),
                character_name_from_conv_load, # Character name for this specific conversation
                current_user_name
            )

            chat_msg_widget_for_display = ChatMessage(
                message=content_to_display,
                role=msg_data.get('sender', 'Unknown'),
                generation_complete=True,
                message_id=msg_data.get('id'),
                message_version=msg_data.get('version'),
                timestamp=msg_data.get('timestamp'),
                image_data=msg_data.get('image_data'),
                image_mime_type=msg_data.get('image_mime_type')
            )
            # Styling class already handled by ChatMessage constructor based on role "User" or other
            await chat_log_widget_disp.mount(chat_msg_widget_for_display)

        if chat_log_widget_disp.is_mounted:
            chat_log_widget_disp.scroll_end(animate=False)

        app.query_one("#chat-input", TextArea).focus()
        app.notify(f"Chat '{conv_metadata.get('title', 'Untitled')}' loaded.", severity="information", timeout=3)
        
        # Update token counter after loading conversation
        try:
            from .chat_token_events import update_chat_token_counter
            await update_chat_token_counter(app)
        except Exception as e:
            loguru_logger.debug(f"Could not update token counter: {e}")
            
    except QueryError as qe_disp_main:
        loguru_logger.error(f"UI component missing during display_conversation for {conversation_id}: {qe_disp_main}")
        app.notify("Error updating UI for loaded chat.", severity="error")
    loguru_logger.info(f"Displayed conversation '{conv_metadata.get('title', 'Untitled')}' (ID: {conversation_id}) in chat tab.")


async def load_branched_conversation_history_ui(app: 'TldwCli', target_conversation_id: str, chat_log_widget: VerticalScroll) -> None:
    """
    Loads the complete message history for a given conversation_id,
    tracing back through parent branches to the root if necessary.
    """
    if not app.notes_service:
        logging.error("Notes service not available for loading branched history.")
        await chat_log_widget.mount(
            ChatMessage("Error: Notes service unavailable.", role="System", classes="-error"))
        return

    db = app.notes_service._get_db(app.notes_user_id)
    await chat_log_widget.remove_children()
    logging.debug(f"Loading branched history for target_conversation_id: {target_conversation_id}")

    # 1. Trace path from target_conversation_id up to its root,
    #    collecting (conversation_id, fork_message_id_in_parent_that_started_this_segment)
    #    The 'fork_message_id_in_parent' is what we need to stop at when loading the parent's messages.
    path_segments_info = []  # Stores (conv_id, fork_msg_id_in_parent)

    current_conv_id_for_path = target_conversation_id
    while current_conv_id_for_path:
        conv_details = db.get_conversation_by_id(current_conv_id_for_path)
        if not conv_details:
            logging.error(f"Path tracing failed: Conversation {current_conv_id_for_path} not found.")
            await chat_log_widget.mount(
                ChatMessage(f"Error: Conversation segment {current_conv_id_for_path} not found.", role="System",
                            classes="-error"))
            return  # Stop if a segment is missing

        path_segments_info.append({
            "id": conv_details['id'],
            "forked_from_message_id": conv_details.get('forked_from_message_id'),
            # ID of message in PARENT where THIS conv started
            "parent_conversation_id": conv_details.get('parent_conversation_id')
        })
        current_conv_id_for_path = conv_details.get('parent_conversation_id')

    path_segments_info.reverse()  # Now path_segments_info is from root-most to target_conversation_id

    all_messages_to_display = []
    for i, segment_info in enumerate(path_segments_info):
        segment_conv_id = segment_info['id']

        # Get all messages belonging to this specific segment_conv_id
        messages_this_segment = db.get_messages_for_conversation(
            segment_conv_id,
            order_by_timestamp="ASC",
            limit=10000  # Effectively all messages for this segment
        )

        # If this segment is NOT the last one in the path, it means it was forked FROM.
        # We need to know where the NEXT segment (its child) forked from THIS segment.
        # The 'forked_from_message_id' of the *next* segment is the message_id in *this* segment.
        stop_at_message_id_for_this_segment = None
        if (i + 1) < len(path_segments_info):  # If there is a next segment
            next_segment_info = path_segments_info[i + 1]
            # next_segment_info['forked_from_message_id'] is the message in current segment_conv_id
            # from which the next_segment_info['id'] was forked.
            stop_at_message_id_for_this_segment = next_segment_info['forked_from_message_id']

        for msg_data in messages_this_segment:
            all_messages_to_display.append(msg_data)
            if stop_at_message_id_for_this_segment and msg_data['id'] == stop_at_message_id_for_this_segment:
                logging.debug(f"Stopping message load for segment {segment_conv_id} at fork point {msg_data['id']}")
                break  # Stop adding messages from this segment, as the next segment takes over

    # Now mount all collected messages
    logging.debug(f"Total messages collected for display: {len(all_messages_to_display)}")
    for msg_data in all_messages_to_display:
        image_data_for_widget = msg_data.get('image_data')
        chat_message_widget = ChatMessage(
            message=msg_data['content'],
            role=msg_data['sender'],
            timestamp=msg_data.get('timestamp'),
            image_data=image_data_for_widget,
            image_mime_type=msg_data.get('image_mime_type'),
            message_id=msg_data['id']
        )
        await chat_log_widget.mount(chat_message_widget)

    if chat_log_widget.is_mounted:
        chat_log_widget.scroll_end(animate=False)
    logging.info(
        f"Loaded {len(all_messages_to_display)} messages for conversation {target_conversation_id} (including history).")


async def handle_chat_character_search_input_changed(app: 'TldwCli', event: Input.Changed) -> None:
    search_term = event.value.strip()
    try:
        results_list_view = app.query_one("#chat-character-search-results-list", ListView)
        await results_list_view.clear()

        if not search_term:  # If search term is empty, call _populate_chat_character_search_list with no term to show default
            await _populate_chat_character_search_list(app)  # Shows default list
            return

        # If search term is present, call _populate_chat_character_search_list with the term
        await _populate_chat_character_search_list(app, search_term)

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found for character search: {e_query}", exc_info=True)
        # Don't notify here as it's an input change, could be spammy. Log is enough.
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error in character search input change: {e_unexp}", exc_info=True)
        # Don't notify here.


async def handle_chat_load_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    loguru_logger.info("Load Character button pressed.")
    try:
        results_list_view = app.query_one("#chat-character-search-results-list", ListView)
        highlighted_widget = results_list_view.highlighted_child

        # --- Type checking and attribute access fix for highlighted_item ---
        if not isinstance(highlighted_widget, ListItem): # Check if it's a ListItem
            app.notify("No character selected to load (not a list item).", severity="warning")
            loguru_logger.info("No character selected in the list to load (highlighted_widget is not ListItem).")
            return

        # Now that we know it's a ListItem, try to get 'character_id'
        # Use getattr for dynamic attributes to satisfy type checkers and handle missing attribute
        selected_char_id: Optional[str] = getattr(highlighted_widget, 'character_id', None)

        if selected_char_id is None:
            app.notify("No character selected or item is invalid.", severity="warning")
            loguru_logger.info("No character_id found on the selected ListItem.")
            return
        # --- End of fix ---

        loguru_logger.info(f"Attempting to load character ID: {selected_char_id}")

        if not app.notes_service: # This should be app.chachanotes_db for character operations
            app.notify("Database service not available.", severity="error")
            loguru_logger.error("ChaChaNotes DB (via notes_service) not available for loading character.")
            return

        # db = app.notes_service._get_db(app.notes_user_id) # Old way
        # Correct way to get the CharactersRAGDB instance
        if not app.chachanotes_db:
            app.notify("Character database not properly initialized.", severity="error")
            loguru_logger.error("app.chachanotes_db is not initialized.")
            return
        db = app.chachanotes_db


        # Assuming app.notes_user_id is the correct user identifier for character operations.
        # If characters are global or use a different user context, adjust app.notes_user_id.
        character_data_full, _, _ = load_character_and_image(db, selected_char_id, app.notes_user_id)

        if character_data_full is None:
            app.notify(f"Character with ID {selected_char_id} not found in database.", severity="error")
            loguru_logger.error(f"Could not retrieve data for character ID {selected_char_id} from DB (returned None).")
            try:
                # When querying from within an event handler in a separate module,
                # it's safer to query from the app instance.
                app.query_one("#chat-character-name-edit", Input).value = ""
                app.query_one("#chat-character-description-edit", TextArea).text = ""
                app.query_one("#chat-character-personality-edit", TextArea).text = ""
                app.query_one("#chat-character-scenario-edit", TextArea).text = ""
                app.query_one("#chat-character-system-prompt-edit", TextArea).text = ""
                app.query_one("#chat-character-first-message-edit", TextArea).text = ""
            except QueryError as qe_clear:
                loguru_logger.warning(f"Could not clear all character edit fields after failed load: {qe_clear}")
            app.current_chat_active_character_data = None
            return

        # character_data_full is now a dictionary
        app.current_chat_active_character_data = character_data_full

        try:
            app.query_one("#chat-character-name-edit", Input).value = character_data_full.get('name', '')
            app.query_one("#chat-character-description-edit", TextArea).text = character_data_full.get('description', '')
            app.query_one("#chat-character-personality-edit", TextArea).text = character_data_full.get('personality', '')
            app.query_one("#chat-character-scenario-edit", TextArea).text = character_data_full.get('scenario', '')
            app.query_one("#chat-character-system-prompt-edit", TextArea).text = character_data_full.get('system_prompt', '')
            app.query_one("#chat-character-first-message-edit", TextArea).text = character_data_full.get('first_message', '')
        except QueryError as qe_populate:
            loguru_logger.error(f"Error populating character edit fields: {qe_populate}", exc_info=True)
            app.notify("Error updating character display fields.", severity="error")
            # Potentially revert app.current_chat_active_character_data if UI update fails critically
            # app.current_chat_active_character_data = None # Or previous state
            return


        app.notify(f"Character '{character_data_full.get('name', 'Unknown')}' loaded.", severity="information")

        # --- Fix for accessing reactive's value ---
        # When accessing app.current_chat_active_character_data, it *IS* the dictionary (or None)
        # because the reactive attribute itself resolves to its current value when accessed.
        # The type checker error "Unresolved attribute reference 'get' for class 'reactive'"
        # usually happens if you try to do `app.current_chat_active_character_data.get` where
        # `current_chat_active_character_data` is the *descriptor* and not its value.
        # However, in your code, when you assign `app.current_chat_active_character_data = character_data_full`,
        # and then later access `app.current_chat_active_character_data.get('first_message')`,
        # this should work correctly at runtime because `app.current_chat_active_character_data`
        # will return the dictionary `character_data_full`.
        # The type checker might be confused if the type hint for `current_chat_active_character_data` is too broad
        # or if it thinks it's still dealing with the `reactive` object itself.

        # To be absolutely clear for the type checker and ensure runtime correctness:
        active_char_data_dict: Optional[Dict[str, Any]] = app.current_chat_active_character_data
        # Now use active_char_data_dict for .get() calls

        if app.current_chat_is_ephemeral:
            loguru_logger.debug("Chat is ephemeral, checking if greeting is appropriate.")
            if active_char_data_dict: # Check if the dictionary is not None
                try:
                    chat_log_widget = app.query_one("#chat-log", VerticalScroll)
                    messages_in_log = list(chat_log_widget.query(ChatMessage))

                    character_has_spoken = False
                    if not messages_in_log:
                        loguru_logger.debug("Chat log is empty. Greeting is appropriate.")
                    else:
                        for msg_widget in messages_in_log:
                            if msg_widget.role != "User":
                                character_has_spoken = True
                                loguru_logger.debug(f"Found message from role '{msg_widget.role}'. Greeting not appropriate.")
                                break
                        if not character_has_spoken:
                            loguru_logger.debug("No non-User messages found in log. Greeting is appropriate.")

                    if not messages_in_log or not character_has_spoken:
                        # Use active_char_data_dict here
                        first_message_content = active_char_data_dict.get('first_message')
                        character_name = active_char_data_dict.get('name')

                        if first_message_content and character_name:
                            loguru_logger.info(f"Displaying first_message for {character_name}.")
                            greeting_message_widget = ChatMessage(
                                message=first_message_content,
                                role=character_name,
                                generation_complete=True
                            )
                            await chat_log_widget.mount(greeting_message_widget)
                            chat_log_widget.scroll_end(animate=True)
                        elif not first_message_content:
                            loguru_logger.debug(f"Character {character_name} has no first_message defined.")
                        elif not character_name:
                            loguru_logger.debug("Character name not found, cannot display first_message effectively.")
                except QueryError as e_chat_log:
                    loguru_logger.error(f"Could not find #chat-log to check for messages or mount greeting: {e_chat_log}")
                except Exception as e_greeting:
                    loguru_logger.error(f"Error displaying character greeting: {e_greeting}", exc_info=True)
            else:
                loguru_logger.debug("No active character data (active_char_data_dict is None), skipping greeting.")
        # --- End of fix ---

        loguru_logger.info(f"Character ID {selected_char_id} loaded and fields populated.")

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found for loading character: {e_query}", exc_info=True)
        app.notify("Error: Character load UI elements missing.", severity="error")
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error loading character: {e_unexp}", exc_info=True)
        app.notify("Unexpected error during character load.", severity="error")



async def handle_chat_character_attribute_changed(app: 'TldwCli', event: Union[Input.Changed, TextArea.Changed]) -> None:
    if app.current_chat_active_character_data is None:
        # loguru_logger.warning("Attribute changed but no character loaded in current_chat_active_character_data.")
        return

    control_id = event.control.id
    new_value: str = "" # Initialize new_value

    if isinstance(event, Input.Changed):
        new_value = event.value
    elif isinstance(event, TextArea.Changed):
        # For TextArea, the changed text is directly on the control itself
        new_value = event.control.text # Use event.control.text for TextAreas
    else:
        # Fallback or error for unexpected event types, though the handler is specific
        loguru_logger.warning(f"Unhandled event type in handle_chat_character_attribute_changed: {type(event)}")
        return # Or handle error appropriately

    field_map = {
        "chat-character-name-edit": "name",
        "chat-character-description-edit": "description",
        "chat-character-personality-edit": "personality",
        "chat-character-scenario-edit": "scenario",
        "chat-character-system-prompt-edit": "system_prompt",
        "chat-character-first-message-edit": "first_message"
    }

    if control_id in field_map:
        attribute_key = field_map[control_id]
        # Ensure current_chat_active_character_data is not None again, just in case of race conditions (though less likely with async/await)
        if app.current_chat_active_character_data is not None:
            updated_data = app.current_chat_active_character_data.copy()
            updated_data[attribute_key] = new_value
            app.current_chat_active_character_data = updated_data # This updates the reactive variable
            loguru_logger.debug(f"Temporarily updated active character attribute '{attribute_key}' to: '{str(new_value)[:50]}...'")

            # If the character's system_prompt is edited in the right sidebar,
            # also update the main system_prompt in the left sidebar.
            if attribute_key == "system_prompt":
                try:
                    # Ensure querying within the correct sidebar if necessary,
                    # but #chat-system-prompt should be unique.
                    main_system_prompt_ta = app.query_one("#chat-system-prompt", TextArea)
                    main_system_prompt_ta.text = new_value
                    loguru_logger.debug("Updated main system prompt in left sidebar from character edit.")
                except QueryError:
                    loguru_logger.error("Could not find #chat-system-prompt to update from character edit.")
    else:
        loguru_logger.warning(f"Attribute change event from unmapped control_id: {control_id}")


async def handle_chat_clear_active_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Clears the currently active character data and resets related UI fields."""
    loguru_logger.info("Clear Active Character button pressed.")

    app.current_chat_active_character_data = None  # Clear the reactive variable
    try:
        default_system_prompt = app.app_config.get("chat_defaults", {}).get("system_prompt", "You are a helpful AI assistant.")
        app.query_one("#chat-system-prompt", TextArea).text = default_system_prompt
        loguru_logger.debug("Reset main system prompt to default on clear active character.")
    except QueryError:
        loguru_logger.error("Could not find #chat-system-prompt to reset on clear active character.")

    try:
        # Get a reference to the chat tab's right sidebar
        # This sidebar has the ID "chat-right-sidebar"
        right_sidebar = app.query_one("#chat-right-sidebar")

        # Now query within the right_sidebar for the specific character editing fields
        right_sidebar.query_one("#chat-character-name-edit", Input).value = ""
        right_sidebar.query_one("#chat-character-description-edit", TextArea).text = ""
        right_sidebar.query_one("#chat-character-personality-edit", TextArea).text = ""
        right_sidebar.query_one("#chat-character-scenario-edit", TextArea).text = ""
        right_sidebar.query_one("#chat-character-system-prompt-edit", TextArea).text = ""
        right_sidebar.query_one("#chat-character-first-message-edit", TextArea).text = ""

        # Optional: Clear the character search input and list within the right sidebar
        # search_input_char = right_sidebar.query_one("#chat-character-search-input", Input)
        # search_input_char.value = ""
        # results_list_char = right_sidebar.query_one("#chat-character-search-results-list", ListView)
        # await results_list_char.clear()
        # If you clear the list, you might want to repopulate it with the default characters:
        # await _populate_chat_character_search_list(app) # Assuming _populate_chat_character_search_list is defined in this file or imported

        app.notify("Active character cleared. Chat will use default settings.", severity="information")
        loguru_logger.debug("Cleared active character data and UI fields from within #chat-right-sidebar.")

    except QueryError as e:
        loguru_logger.error(
            f"UI component not found when clearing character fields within #chat-right-sidebar. "
            f"Widget ID/Selector: {getattr(e, 'widget_id', getattr(e, 'selector', 'N/A'))}",
            exc_info=True
        )
        app.notify("Error clearing character fields (UI component not found).", severity="error")
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error clearing active character: {e_unexp}", exc_info=True)
        app.notify("Error clearing active character.", severity="error")


async def handle_chat_prompt_search_input_changed(app: 'TldwCli', event_value: str) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    search_term = event_value.strip()
    logger.debug(f"Chat Tab: Prompt search input changed to: '{search_term}'")

    if not app.prompts_service_initialized:
        logger.warning("Chat Tab: Prompts service not available for prompt search.")
        # Optionally notify the user or clear list
        try:
            results_list_view = app.query_one("#chat-prompt-search-results-listview", ListView)
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Prompts service unavailable.")))
        except Exception as e_ui:
            logger.error(f"Chat Tab: Error accessing prompt search listview: {e_ui}")
        return

    if not search_term:  # Clear list if search term is empty
        try:
            results_list_view = app.query_one("#chat-prompt-search-results-listview", ListView)
            await results_list_view.clear()
            logger.debug("Chat Tab: Cleared prompt search results as search term is empty.")
        except Exception as e_ui_clear:
            logger.error(f"Chat Tab: Error clearing prompt search listview: {e_ui_clear}")
        return

    try:
        results_list_view = app.query_one("#chat-prompt-search-results-listview", ListView)
        await results_list_view.clear()

        # Assuming search_prompts returns a tuple: (results_list, total_matches)
        prompt_results, total_matches = prompts_interop.search_prompts(
            search_query=search_term,
            search_fields=["name", "details", "keywords"],  # Or other relevant fields
            page=1,
            results_per_page=50,  # Adjust as needed
            include_deleted=False
        )

        if prompt_results:
            for prompt_data in prompt_results:
                item_label = prompt_data.get('name', 'Unnamed Prompt')
                list_item = ListItem(Label(item_label))
                # Store necessary identifiers on the ListItem itself
                list_item.prompt_id = prompt_data.get('id')
                list_item.prompt_uuid = prompt_data.get('uuid')
                await results_list_view.append(list_item)
            logger.info(f"Chat Tab: Prompt search for '{search_term}' yielded {len(prompt_results)} results.")
        else:
            await results_list_view.append(ListItem(Label("No prompts found.")))
            logger.info(f"Chat Tab: Prompt search for '{search_term}' found no results.")

    except prompts_interop.DatabaseError as e_db:
        logger.error(f"Chat Tab: Database error during prompt search: {e_db}", exc_info=True)
        try:  # Attempt to update UI with error
            results_list_view = app.query_one("#chat-prompt-search-results-listview", ListView)
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("DB error searching.")))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Chat Tab: Unexpected error during prompt search: {e}", exc_info=True)
        try:  # Attempt to update UI with error
            results_list_view = app.query_one("#chat-prompt-search-results-listview", ListView)
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Search error.")))
        except Exception:
            pass


async def perform_chat_prompt_search(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    try:
        search_input_widget = app.query_one("#chat-prompt-search-input",
                                            Input)  # Ensure Input is imported where this is called
        await handle_chat_prompt_search_input_changed(app, search_input_widget.value)
    except Exception as e:
        logger.error(f"Chat Tab: Error performing prompt search via perform_chat_prompt_search: {e}", exc_info=True)


async def handle_chat_view_selected_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Chat Tab: View Selected Prompt button pressed.")

    try:
        results_list_view = app.query_one("#chat-prompts-listview", ListView)
        selected_list_item = results_list_view.highlighted_child

        if not selected_list_item:
            app.notify("No prompt selected in the list.", severity="warning")
            return

        prompt_id_to_load = getattr(selected_list_item, 'prompt_id', None)
        prompt_uuid_to_load = getattr(selected_list_item, 'prompt_uuid', None)

        identifier_to_fetch = prompt_id_to_load if prompt_id_to_load is not None else prompt_uuid_to_load

        if identifier_to_fetch is None:
            app.notify("Selected prompt item is invalid (missing ID/UUID).", severity="error")
            logger.error("Chat Tab: Selected prompt item missing ID and UUID.")
            return

        logger.debug(f"Chat Tab: Fetching details for prompt identifier: {identifier_to_fetch}")
        prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)

        system_display_widget = app.query_one("#chat-prompt-system-display", TextArea)
        user_display_widget = app.query_one("#chat-prompt-user-display", TextArea)
        copy_system_button = app.query_one("#chat-prompt-copy-system-button", Button)
        copy_user_button = app.query_one("#chat-prompt-copy-user-button", Button)

        if prompt_details:
            system_prompt_content = prompt_details.get('system_prompt', '')
            user_prompt_content = prompt_details.get('user_prompt', '')

            system_display_widget.text = system_prompt_content
            user_display_widget.text = user_prompt_content

            # Store the fetched content on the app or widgets for copy buttons
            # If TextAreas are read-only, their .text property is the source of truth
            # No need for app.current_loaded_system_prompt etc. unless used elsewhere

            copy_system_button.disabled = not bool(system_prompt_content)
            copy_user_button.disabled = not bool(user_prompt_content)

            app.notify(f"Prompt '{prompt_details.get('name', 'Selected')}' loaded for viewing.", severity="information")
            logger.info(f"Chat Tab: Displayed prompt '{prompt_details.get('name', 'Unknown')}' for viewing.")
        else:
            system_display_widget.text = "Failed to load prompt details."
            user_display_widget.text = ""
            copy_system_button.disabled = True
            copy_user_button.disabled = True
            app.notify("Failed to load details for the selected prompt.", severity="error")
            logger.error(f"Chat Tab: Failed to fetch details for prompt identifier: {identifier_to_fetch}")

    except prompts_interop.DatabaseError as e_db:
        logger.error(f"Chat Tab: Database error viewing selected prompt: {e_db}", exc_info=True)
        app.notify("Database error loading prompt.", severity="error")
    except Exception as e:
        logger.error(f"Chat Tab: Unexpected error viewing selected prompt: {e}", exc_info=True)
        app.notify("Error loading prompt for viewing.", severity="error")
        # Clear display areas on generic error too
        try:
            app.query_one("#chat-prompt-display-system", TextArea).text = ""
            app.query_one("#chat-prompt-display-user", TextArea).text = ""
            app.query_one("#chat-prompt-copy-system-button", Button).disabled = True
            app.query_one("#chat-prompt-copy-user-button", Button).disabled = True
        except Exception:
            pass  # UI might not be fully available


async def _populate_chat_character_search_list(app: 'TldwCli', search_term: Optional[str] = None) -> None:
    try:
        results_list_view = app.query_one("#chat-character-search-results-list", ListView)
        await results_list_view.clear()

        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            loguru_logger.error("Notes service not available for character list population.")
            await results_list_view.append(ListItem(Label("Error: DB service unavailable.")))
            return

        db = app.notes_service._get_db(app.notes_user_id)
        characters = []
        operation_type = "list_character_cards"  # For logging

        try:
            if search_term:
                operation_type = "search_character_cards"
                loguru_logger.debug(f"Populating character list by searching for: '{search_term}'")
                characters = db.search_character_cards(search_term=search_term, limit=50)
            else:
                loguru_logger.debug("Populating character list with default list (limit 40).")
                characters = db.list_character_cards(limit=40)

            if not characters:
                await results_list_view.append(ListItem(Label("No characters found.")))
            else:
                for char_data in characters:
                    item = ListItem(Label(char_data.get('name', 'Unnamed Character')))
                    item.character_id = char_data.get('id')  # Store ID on the item
                    await results_list_view.append(item)
            loguru_logger.info(f"Character list populated using {operation_type}. Found {len(characters)} characters.")

        except Exception as e_db_call:
            loguru_logger.error(f"Error during DB call ({operation_type}): {e_db_call}", exc_info=True)
            await results_list_view.append(ListItem(Label(f"Error during {operation_type}.")))

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found for character list population: {e_query}", exc_info=True)
        # Avoid app.notify here as this function might be called when tab is not fully visible.
        # Let the calling context (e.g., direct user action) handle user notifications if appropriate.
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error in _populate_chat_character_search_list: {e_unexp}", exc_info=True)
        # Avoid app.notify here as well.


async def handle_chat_copy_system_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Chat Tab: Copy System Prompt button pressed.")
    try:
        system_display_widget = app.query_one("#chat-prompt-system-display", TextArea)
        content_to_copy = system_display_widget.text
        if content_to_copy:
            app.copy_to_clipboard(content_to_copy)
            app.notify("System prompt copied to clipboard!")
            logger.info("Chat Tab: System prompt content copied to clipboard.")
        else:
            app.notify("No system prompt content to copy.", severity="warning")
            logger.warning("Chat Tab: No system prompt content available to copy.")
    except Exception as e:
        logger.error(f"Chat Tab: Error copying system prompt: {e}", exc_info=True)
        app.notify("Error copying system prompt.", severity="error")


async def handle_chat_copy_user_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Chat Tab: Copy User Prompt button pressed.")
    try:
        user_display_widget = app.query_one("#chat-prompt-user-display", TextArea)
        content_to_copy = user_display_widget.text
        if content_to_copy:
            app.copy_to_clipboard(content_to_copy)
            app.notify("User prompt copied to clipboard!")
            logger.info("Chat Tab: User prompt content copied to clipboard.")
        else:
            app.notify("No user prompt content to copy.", severity="warning")
            logger.warning("Chat Tab: No user prompt content available to copy.")
    except Exception as e:
        logger.error(f"Chat Tab: Error copying user prompt: {e}", exc_info=True)
        app.notify("Error copying user prompt.", severity="error")


async def handle_chat_template_search_input_changed(app: 'TldwCli', event_value: str) -> None:
    """Handle changes to the template search input in the Chat tab."""
    from tldw_chatbook.Chat.prompt_template_manager import get_available_templates, load_template

    logger = getattr(app, 'loguru_logger', logging)
    search_term = event_value.strip().lower()
    logger.debug(f"Chat Tab: Template search input changed to: '{search_term}'")

    try:
        template_list_view = app.query_one("#chat-template-list-view", ListView)
        await template_list_view.clear()

        # Get all available templates
        all_templates = get_available_templates()

        if not all_templates:
            await template_list_view.append(ListItem(Label("No templates available.")))
            logger.info("Chat Tab: No templates available.")
            return

        # Filter templates based on search term
        filtered_templates = all_templates
        if search_term:
            filtered_templates = [t for t in all_templates if search_term in t.lower()]

        if filtered_templates:
            for template_name in filtered_templates:
                list_item = ListItem(Label(template_name))
                list_item.template_name = template_name
                await template_list_view.append(list_item)
            logger.info(f"Chat Tab: Template search for '{search_term}' yielded {len(filtered_templates)} results.")
        else:
            await template_list_view.append(ListItem(Label("No matching templates.")))
            logger.info(f"Chat Tab: Template search for '{search_term}' found no results.")

    except Exception as e:
        logger.error(f"Chat Tab: Error during template search: {e}", exc_info=True)
        try:
            template_list_view = app.query_one("#chat-template-list-view", ListView)
            await template_list_view.clear()
            await template_list_view.append(ListItem(Label("Search error.")))
        except Exception:
            pass


async def handle_chat_apply_template_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle the Apply Template button press in the Chat tab."""
    from tldw_chatbook.Chat.prompt_template_manager import load_template

    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Chat Tab: Apply Template button pressed.")

    try:
        template_list_view = app.query_one("#chat-template-list-view", ListView)
        selected_list_item = template_list_view.highlighted_child

        if not selected_list_item:
            app.notify("No template selected in the list.", severity="warning")
            return

        template_name = getattr(selected_list_item, 'template_name', None)

        if template_name is None:
            app.notify("Selected template item is invalid.", severity="error")
            logger.error("Chat Tab: Selected template item missing template_name.")
            return

        logger.debug(f"Chat Tab: Loading template: {template_name}")
        template = load_template(template_name)

        if not template:
            app.notify(f"Failed to load template: {template_name}", severity="error")
            logger.error(f"Chat Tab: Failed to load template: {template_name}")
            return

        # Apply the template to the system prompt and user input
        system_prompt_widget = app.query_one("#chat-system-prompt", TextArea)
        chat_input_widget = app.query_one("#chat-input", TextArea)

        if template.system_message_template:
            system_prompt_widget.text = template.system_message_template

        # If there's text in the chat input, apply the user message template to it
        if chat_input_widget.text.strip() and template.user_message_content_template != "{message_content}":
            # Save the original message content
            original_content = chat_input_widget.text.strip()
            # Apply the template, replacing {message_content} with the original content
            chat_input_widget.text = template.user_message_content_template.replace("{message_content}", original_content)

        app.notify(f"Applied template: {template_name}", severity="information")
        logger.info(f"Chat Tab: Applied template: {template_name}")

    except Exception as e:
        logger.error(f"Chat Tab: Error applying template: {e}", exc_info=True)
        app.notify("Error applying template.", severity="error")


async def handle_chat_sidebar_prompt_search_changed(
    app: "TldwCli",
    new_value: str,
) -> None:
    """
    Populate / update the *Prompts* list that lives in the Chat-tab’s right sidebar.

    Called

        • each time the search-input (#chat-prompt-search-input) changes, and
        • once when the Chat tab first becomes active (app.py calls with an empty string).

    Parameters
    ----------
    app : TldwCli
        The running application instance (passed by `call_later` / the watcher).
    new_value : str
        The raw text currently in the search-input.  Leading / trailing whitespace is ignored.
    """
    logger = getattr(app, "loguru_logger", logging)  # fall back to stdlib if unavailable
    search_term = (new_value or "").strip()
    logger.debug(f"Sidebar-Prompt-Search changed → '{search_term}'")

    # Locate UI elements up-front so we can fail fast.
    try:
        search_input  : Input    = app.query_one("#chat-prompt-search-input", Input)
        results_view  : ListView = app.query_one("#chat-prompts-listview", ListView)
    except QueryError as q_err:
        logger.error(f"[Prompts] UI element(s) missing: {q_err}")
        return

    # Keep the search-box in sync if we were called programmatically (e.g. with "").
    if search_input.value != new_value:
        search_input.value = new_value

    # Always start with a clean slate.
    await results_view.clear()

    # Ensure the prompts subsystem is ready.
    if not getattr(app, "prompts_service_initialized", False):
        await results_view.append(ListItem(Label("Prompt service unavailable.")))
        logger.warning("[Prompts] Service not initialised – cannot search.")
        return

    # === No term supplied → Show a convenient default list (first 100, alpha order). ===
    if not search_term:
        try:
            prompts, _total = prompts_interop.search_prompts(
                search_query   = "",                 # empty → match all
                search_fields  = ["name"],           # cheap field only
                page           = 1,
                results_per_page = 100,
                include_deleted = False,
            )
        except Exception as e:
            logger.error(f"[Prompts] Default-list load failed: {e}", exc_info=True)
            await results_view.append(ListItem(Label("Failed to load prompts.")))
            return
    # === A term is present → Run a full search. ===
    else:
        try:
            prompts, _total = prompts_interop.search_prompts(
                search_query     = search_term,
                search_fields    = ["name", "details", "keywords"],
                page             = 1,
                results_per_page = 100,              # generous but safe
                include_deleted  = False,
            )
        except prompts_interop.DatabaseError as dbe:
            logger.error(f"[Prompts] DB error during search: {dbe}", exc_info=True)
            await results_view.append(ListItem(Label("Database error while searching.")))
            return
        except Exception as ex:
            logger.error(f"[Prompts] Unknown error during search: {ex}", exc_info=True)
            await results_view.append(ListItem(Label("Error during search.")))
            return

    # ----- Render results -----
    if not prompts:
        await results_view.append(ListItem(Label("No prompts found.")))
        logger.info(f"[Prompts] Search '{search_term}' → 0 results.")
        return

    for pr in prompts:
        item = ListItem(Label(pr.get("name", "Unnamed Prompt")))
        # Stash useful identifiers on the ListItem for later pick-up by the “Load Selected Prompt” button.
        item.prompt_id   = pr.get("id")
        item.prompt_uuid = pr.get("uuid")
        await results_view.append(item)

    logger.info(f"[Prompts] Search '{search_term}' → {len(prompts)} results.")


async def handle_continue_response_button_pressed(app: 'TldwCli', event: Button.Pressed, message_widget: ChatMessage) -> None:
    """Handles the 'Continue Response' button press on an AI chat message."""
    loguru_logger.info(f"Continue Response button pressed for message_id: {message_widget.message_id_internal}, current text: '{message_widget.message_text[:50]}...'")
    db = app.chachanotes_db
    prefix = "chat" # Assuming 'chat' is the prefix for UI elements in the main chat window

    continue_button_widget: Optional[Button] = None
    original_button_label: Optional[str] = None
    static_text_widget: Optional[Static] = None
    original_display_text_obj: Optional[Union[str, Text]] = None # renderable can be str or Text

    try:
        button = event.button
        continue_button_widget = button
        original_button_label = continue_button_widget.label
        continue_button_widget.disabled = True
        continue_button_widget.label = get_char(EMOJI_THINKING, FALLBACK_THINKING) # "⏳" or similar

        static_text_widget = message_widget.query_one(".message-text", Static)
        original_display_text_obj = static_text_widget.renderable # Save the original renderable (Text object or str)
    except QueryError as qe:
        loguru_logger.error(f"Error querying essential UI component for continuation: {qe}", exc_info=True)
        app.notify("Error initializing continuation: UI component missing.", severity="error")
        if continue_button_widget and original_button_label: # Attempt to restore button if found
            continue_button_widget.disabled = False
            continue_button_widget.label = original_button_label
        return
    except Exception as e_init: # Catch any other init error
        loguru_logger.error(f"Unexpected error during continue response initialization: {e_init}", exc_info=True)
        app.notify("Unexpected error starting continuation.", severity="error")
        if continue_button_widget and original_button_label:
            continue_button_widget.disabled = False
            continue_button_widget.label = original_button_label
        if static_text_widget and original_display_text_obj: # Restore text if changed
             static_text_widget.update(original_display_text_obj)
        return

    original_message_text = message_widget.message_text # Raw text content
    original_message_version = message_widget.message_version_internal

    # --- 1. Retrieve History for API ---
    # History should include the message being continued, as the LLM needs its content.
    history_for_api: List[Dict[str, Any]] = []
    chat_log: Optional[VerticalScroll] = None
    try:
        chat_log = app.query_one(f"#{prefix}-log", VerticalScroll)
        all_messages_in_log = list(chat_log.query(ChatMessage))

        for msg_w in all_messages_in_log:
            # Map UI role to API role (user/assistant)
            # Allow for character names to be mapped to "assistant"
            api_role = "user" if msg_w.role == "User" else "assistant"

            if msg_w.generation_complete or msg_w is message_widget: # Include incomplete target message
                content_for_api = msg_w.message_text
                history_for_api.append({"role": api_role, "content": content_for_api})

            if msg_w is message_widget: # Stop after adding the target message
                break

        if not any(msg_info['content'] == original_message_text and msg_info['role'] == 'assistant' for msg_info in history_for_api):
             loguru_logger.warning("Target message for continuation not found in constructed history. This is unexpected.")
             # This might indicate an issue with message_widget identity or history construction logic.

        loguru_logger.debug(f"Built history for API continuation with {len(history_for_api)} messages. Last message is the one to continue.")

    except QueryError as e:
        loguru_logger.error(f"Continue Response: Could not find UI elements for history: {e}", exc_info=True)
        app.notify("Error: Chat log or other UI element not found.", severity="error")
        if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
        if static_text_widget: static_text_widget.update(original_display_text_obj)
        return
    except Exception as e_hist:
        loguru_logger.error(f"Error building history for continuation: {e_hist}", exc_info=True)
        app.notify("Error preparing message history for continuation.", severity="error")
        if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
        if static_text_widget: static_text_widget.update(original_display_text_obj)
        return

    # --- 2. LLM Call Preparation ---
    thinking_indicator_suffix = f" ... {get_char(EMOJI_THINKING, FALLBACK_THINKING)}"
    try:
        # Display thinking indicator by updating the Static widget.
        # original_display_text_obj might be a Text object, ensure we append str to str or Text to Text
        if isinstance(original_display_text_obj, Text):
            # Create a new Text object if the original was Text
            text_with_indicator = original_display_text_obj.copy()
            text_with_indicator.append(thinking_indicator_suffix)
            static_text_widget.update(text_with_indicator)
        else: # Assuming str
            static_text_widget.update(original_message_text + thinking_indicator_suffix)

    except Exception as e_indicator: # Non-critical if this fails
        loguru_logger.warning(f"Could not update message with thinking indicator: {e_indicator}", exc_info=True)

    # Prompt for the LLM to continue the last message in the history
    continuation_prompt_instruction = (
        "The last message in this conversation is from you (assistant). "
        "Please continue generating the response for that message. "
        "Only provide the additional text; do not repeat any part of the existing message, "
        "and do not add any conversational filler, apologies, or introductory phrases. "
        "Directly continue from where the last message ended."
    )
    # Note: The actual message to be continued is already the last one in `history_for_api`.
    # The `message` parameter to `chat_wrapper` will be this instruction.

    # --- 3. Fetch Chat Parameters & API Key ---
    try:
        provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
        model_widget = app.query_one(f"#{prefix}-api-model", Select)
        system_prompt_widget = app.query_one(f"#{prefix}-system-prompt", TextArea) # Main system prompt from left sidebar
        temp_widget = app.query_one(f"#{prefix}-temperature", Input)
        top_p_widget = app.query_one(f"#{prefix}-top-p", Input)
        min_p_widget = app.query_one(f"#{prefix}-min-p", Input)
        top_k_widget = app.query_one(f"#{prefix}-top-k", Input)
        llm_max_tokens_widget = app.query_one(f"#{prefix}-llm-max-tokens", Input)
        llm_seed_widget = app.query_one(f"#{prefix}-llm-seed", Input)
        llm_stop_widget = app.query_one(f"#{prefix}-llm-stop", Input)
        llm_response_format_widget = app.query_one(f"#{prefix}-llm-response-format", Select)
        llm_n_widget = app.query_one(f"#{prefix}-llm-n", Input)
        llm_user_identifier_widget = app.query_one(f"#{prefix}-llm-user-identifier", Input)
        llm_logprobs_widget = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
        llm_top_logprobs_widget = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
        llm_logit_bias_widget = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
        llm_presence_penalty_widget = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
        llm_frequency_penalty_widget = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
        llm_tools_widget = app.query_one(f"#{prefix}-llm-tools", TextArea)
        llm_tool_choice_widget = app.query_one(f"#{prefix}-llm-tool-choice", Input)
        llm_fixed_tokens_kobold_widget = app.query_one(f"#{prefix}-llm-fixed-tokens-kobold", Checkbox)
    except QueryError as e:
        loguru_logger.error(f"Continue Response: Could not find UI settings widgets for '{prefix}': {e}", exc_info=True)
        app.notify("Error: Missing UI settings for continuation.", severity="error")
        if static_text_widget: static_text_widget.update(original_display_text_obj) # Restore original text
        if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
        return

    selected_provider = str(provider_widget.value) if provider_widget.value != Select.BLANK else None
    selected_model = str(model_widget.value) if model_widget.value != Select.BLANK else None
    temperature = safe_float(temp_widget.value, 0.7, "temperature")
    top_p = safe_float(top_p_widget.value, 0.95, "top_p")
    min_p = safe_float(min_p_widget.value, 0.05, "min_p")
    top_k = safe_int(top_k_widget.value, 50, "top_k")
    llm_max_tokens_value = safe_int(llm_max_tokens_widget.value, 1024, "llm_max_tokens")
    llm_seed_value = safe_int(llm_seed_widget.value, None, "llm_seed")
    llm_stop_value = [s.strip() for s in llm_stop_widget.value.split(',') if s.strip()] if llm_stop_widget.value.strip() else None
    llm_response_format_value = {"type": str(llm_response_format_widget.value)} if llm_response_format_widget.value != Select.BLANK else {"type": "text"}
    llm_n_value = safe_int(llm_n_widget.value, 1, "llm_n")
    llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
    llm_logprobs_value = llm_logprobs_widget.value
    llm_top_logprobs_value = safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs") if llm_logprobs_value else 0
    llm_presence_penalty_value = safe_float(llm_presence_penalty_widget.value, 0.0, "llm_presence_penalty")
    llm_frequency_penalty_value = safe_float(llm_frequency_penalty_widget.value, 0.0, "llm_frequency_penalty")
    llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None
    llm_fixed_tokens_kobold_value = llm_fixed_tokens_kobold_widget.value
    try:
        llm_logit_bias_text = llm_logit_bias_widget.text.strip()
        llm_logit_bias_value = json.loads(llm_logit_bias_text) if llm_logit_bias_text and llm_logit_bias_text != "{}" else None
    except json.JSONDecodeError: llm_logit_bias_value = None; loguru_logger.warning("Invalid JSON in llm_logit_bias for continuation.")
    try:
        llm_tools_text = llm_tools_widget.text.strip()
        llm_tools_value = json.loads(llm_tools_text) if llm_tools_text and llm_tools_text != "[]" else None
    except json.JSONDecodeError: llm_tools_value = None; loguru_logger.warning("Invalid JSON in llm_tools for continuation.")

    # System Prompt (Active Character > UI)
    final_system_prompt_for_api = system_prompt_widget.text # Default to UI's system prompt
    if app.current_chat_active_character_data:
        char_specific_system_prompt = app.current_chat_active_character_data.get('system_prompt')
        if char_specific_system_prompt and char_specific_system_prompt.strip():
            final_system_prompt_for_api = char_specific_system_prompt
            loguru_logger.debug("Using active character's system prompt for continuation.")
        else:
            loguru_logger.debug("Active character has no system_prompt; using UI system prompt for continuation.")
    else:
        loguru_logger.debug("No active character; using UI system prompt for continuation.")

    should_stream = True # Always stream for continuation for better UX
    if selected_provider: # Log provider's normal streaming setting for info
        provider_settings_key = selected_provider.lower().replace(" ", "_")
        provider_specific_settings = app.app_config.get("api_settings", {}).get(provider_settings_key, {})
        loguru_logger.debug(f"Provider {selected_provider} normally streams: {provider_specific_settings.get('streaming', False)}. Forcing stream for continuation.")

    # API Key Fetching
    api_key_for_call = None
    if selected_provider:
        provider_settings_key = selected_provider.lower().replace(" ", "_")
        provider_config = app.app_config.get("api_settings", {}).get(provider_settings_key, {})
        if "api_key" in provider_config and provider_config["api_key"] and provider_config["api_key"] != "<API_KEY_HERE>":
            api_key_for_call = provider_config["api_key"]
        elif "api_key_env_var" in provider_config and provider_config["api_key_env_var"]:
            api_key_for_call = os.environ.get(provider_config["api_key_env_var"])

    providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]
    if selected_provider in providers_requiring_key and not api_key_for_call:
        loguru_logger.error(f"API Key for '{selected_provider}' is missing for continuation.")
        app.notify(f"API Key for {selected_provider} is missing.", severity="error")
        if static_text_widget: static_text_widget.update(original_display_text_obj)
        if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
        return

    # --- 4. Disable other AI action buttons ---
    other_action_buttons_ids = ["thumb-up", "thumb-down", "regenerate"] # Add other relevant button IDs
    original_button_states: Dict[str, bool] = {}
    try:
        for btn_id in other_action_buttons_ids:
            # Ensure query is specific to the message_widget
            b = message_widget.query_one(f"#{btn_id}", Button)
            original_button_states[btn_id] = b.disabled
            b.disabled = True
    except QueryError as qe:
        loguru_logger.warning(f"Could not find or disable one or more action buttons during continuation: {qe}")


    # --- 5. Streaming LLM Call & UI Update ---
    current_full_text = original_message_text # Start with the original text
    first_chunk_received_flag = False

    try:
        async for chunk_data in app.chat_wrapper(
            message=continuation_prompt_instruction, # The instruction for how to use the history
            history=history_for_api,                 # Contains the actual message to be continued as the last item
            api_endpoint=selected_provider,
            api_key=api_key_for_call,
            system_message=final_system_prompt_for_api,
            temperature=temperature,
            topp=top_p, minp=min_p, topk=top_k,
            llm_max_tokens=llm_max_tokens_value,
            llm_seed=llm_seed_value,
            llm_stop=llm_stop_value,
            llm_response_format=llm_response_format_value,
            llm_n=llm_n_value,
            llm_user_identifier=llm_user_identifier_value,
            llm_logprobs=llm_logprobs_value,
            llm_top_logprobs=llm_top_logprobs_value,
            llm_logit_bias=llm_logit_bias_value,
            llm_presence_penalty=llm_presence_penalty_value,
            llm_frequency_penalty=llm_frequency_penalty_value,
            llm_tools=llm_tools_value,
            llm_tool_choice=llm_tool_choice_value,
            llm_fixed_tokens_kobold=llm_fixed_tokens_kobold_value,
            streaming=should_stream, # Forced True
            # These are older/other params, ensure they are correctly defaulted or excluded if not needed
            custom_prompt="", media_content={}, selected_parts=[], chatdict_entries=None, max_tokens=500, strategy="sorted_evenly"
        ):
            if not first_chunk_received_flag:
                first_chunk_received_flag = True
                # Remove thinking indicator from display.
                # current_full_text already holds original_message_text.
                # Static widget is updated with Text object to prevent markup issues.
                if static_text_widget: static_text_widget.update(Text(current_full_text))

            if isinstance(chunk_data, str):
                current_full_text += chunk_data
                if static_text_widget: static_text_widget.update(Text(current_full_text))
            elif isinstance(chunk_data, dict) and 'error' in chunk_data:
                error_detail = chunk_data['error']
                loguru_logger.error(f"Error chunk received from LLM during continuation: {error_detail}")
                app.notify(f"LLM Error: {str(error_detail)[:100]}", severity="error", timeout=7)
                # Restore original state on error
                message_widget.message_text = original_message_text # Restore internal text
                if static_text_widget: static_text_widget.update(original_display_text_obj)
                if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
                for btn_id, was_disabled in original_button_states.items():
                    try: message_widget.query_one(f"#{btn_id}", Button).disabled = was_disabled
                    except QueryError: pass
                return # Stop processing

            if chat_log: chat_log.scroll_end(animate=False)

        # After successful stream, update the ChatMessage's internal text
        message_widget.message_text = current_full_text

    except Exception as e_llm:
        loguru_logger.error(f"Error during LLM call for continuation: {e_llm}", exc_info=True)
        app.notify(f"LLM call failed: {str(e_llm)[:100]}", severity="error")
        message_widget.message_text = original_message_text # Restore internal text
        if static_text_widget: static_text_widget.update(original_display_text_obj) # Restore display
        if continue_button_widget: continue_button_widget.disabled = False; continue_button_widget.label = original_button_label
        for btn_id, was_disabled in original_button_states.items():
            try: message_widget.query_one(f"#{btn_id}", Button).disabled = was_disabled
            except QueryError: pass
        return

    # --- 6. Post-Stream Processing (DB Update) ---
    message_widget.generation_complete = True # Ensure it's marked complete

    if db and message_widget.message_id_internal and original_message_version is not None:
        try:
            success = ccl.edit_message_content(
                db,
                message_widget.message_id_internal,
                current_full_text, # The new, complete message text
                original_message_version # Expected version before this edit
            )
            if success:
                message_widget.message_version_internal = original_message_version + 1
                loguru_logger.info(f"Continued message ID {message_widget.message_id_internal} updated in DB. New version: {message_widget.message_version_internal}")
                app.notify("Message continuation saved to DB.", severity="information", timeout=2)
            else: # edit_message_content returned False, but no exception
                loguru_logger.error(f"ccl.edit_message_content returned False for continued message {message_widget.message_id_internal} without specific error.")
                app.notify("Failed to save continuation to DB (operation indicated failure).", severity="error")
                # Consider if UI should be reverted here if DB save is critical
        except ccl.ConflictError as e_conflict:
            loguru_logger.error(f"DB conflict updating continued message {message_widget.message_id_internal}: {e_conflict}", exc_info=True)
            app.notify(f"Save conflict: {e_conflict}. Message may be out of sync.", severity="error", timeout=7)
        except (ccl.CharactersRAGDBError, ccl.InputError) as e_db_update:
            loguru_logger.error(f"DB/Input error updating continued message {message_widget.message_id_internal}: {e_db_update}", exc_info=True)
            app.notify(f"Failed to save continuation to DB: {str(e_db_update)[:100]}", severity="error")
        except Exception as e_db_generic:
            loguru_logger.error(f"Unexpected DB error updating continued message {message_widget.message_id_internal}: {e_db_generic}", exc_info=True)
            app.notify(f"Unexpected error saving continuation to DB: {str(e_db_generic)[:100]}", severity="error")

    # --- 7. Button State (Re-enable/Finalize) ---
    if continue_button_widget and original_button_label:
        continue_button_widget.disabled = False # Re-enable for potential further continuation
        continue_button_widget.label = original_button_label

    # Re-enable other action buttons to their original state
    for btn_id, was_disabled_originally in original_button_states.items():
        try:
            # Only re-enable if it was not disabled before we started
            # However, typical flow is they are enabled, we disable, then re-enable.
            # So, setting disabled to 'was_disabled_originally' covers this.
             message_widget.query_one(f"#{btn_id}", Button).disabled = was_disabled_originally
        except QueryError:
            loguru_logger.warning(f"Could not restore state for button {btn_id} post-continuation.")

    loguru_logger.info(f"Continuation process completed for message_id: {message_widget.message_id_internal}. Final text length: {len(current_full_text)}")


async def handle_respond_for_me_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Respond for Me' (Suggest) button press in the chat input area."""
    loguru_logger.info("Enter: handle_respond_for_me_button_pressed")
    loguru_logger.info("Respond for Me button pressed.")
    prefix = "chat" # For querying UI elements like #chat-log, #chat-input, etc.

    respond_button: Optional[Button] = None
    original_button_label: Optional[str] = "💡" # Default/fallback icon

    try:
        respond_button = app.query_one("#respond-for-me-button", Button)
        original_button_label = respond_button.label
        respond_button.disabled = True
        respond_button.label = f"{get_char(EMOJI_THINKING, FALLBACK_THINKING)} Suggesting..."
        app.notify("Generating suggestion...", timeout=2)

        # --- 1. Retrieve History for API ---
        history_for_api: List[Dict[str, Any]] = []
        chat_log_widget: Optional[VerticalScroll] = None
        try:
            chat_log_widget = app.query_one(f"#{prefix}-log", VerticalScroll)
            all_messages_in_log = list(chat_log_widget.query(ChatMessage))

            if not all_messages_in_log:
                app.notify("Cannot generate suggestion: Chat history is empty.", severity="warning", timeout=4)
                loguru_logger.info("Respond for Me: Chat history is empty.")
                # No 'return' here, finally block will re-enable button
                raise ValueError("Empty history") # Raise to go to finally

            for msg_w in all_messages_in_log:
                api_role = "user" if msg_w.role == "User" else "assistant"
                if msg_w.generation_complete: # Only include completed messages
                    history_for_api.append({"role": api_role, "content": msg_w.message_text})

            loguru_logger.debug(f"Built history for suggestion API with {len(history_for_api)} messages.")

        except QueryError as e_hist_query:
            loguru_logger.error(f"Respond for Me: Could not find UI elements for history: {e_hist_query}", exc_info=True)
            app.notify("Error: Chat log not found.", severity="error")
            raise # Re-raise to go to finally
        except ValueError: # Catch empty history explicitly if needed for specific handling before finally
            raise
        except Exception as e_hist_build:
            loguru_logger.error(f"Error building history for suggestion: {e_hist_build}", exc_info=True)
            app.notify("Error preparing message history for suggestion.", severity="error")
            raise # Re-raise to go to finally

        # --- 2. LLM Call Preparation ---
        # Convert history to a string format for the prompt, or pass as structured history if API supports
        conversation_history_str = "\n".join([f"{item['role']}: {item['content']}" for item in history_for_api])

        suggestion_prompt_instruction = (
            "Based on the following conversation, please suggest a concise and relevant response for the user to send next. "
            "Focus on being helpful and natural in the context of the conversation. "
            "Only provide the suggested response text, without any additional explanations, apologies, or conversational filler like 'Sure, here's a suggestion:'. "
            "Directly output the text that the user could send.\n\n"
            "CONVERSATION HISTORY:\n"
            f"{conversation_history_str}"
        )

        # --- 3. Fetch Chat Parameters & API Key (similar to other handlers) ---
        try:
            provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
            model_widget = app.query_one(f"#{prefix}-api-model", Select)
            provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
            model_widget = app.query_one(f"#{prefix}-api-model", Select)
            system_prompt_widget = app.query_one(f"#{prefix}-system-prompt", TextArea) # Main system prompt
            temp_widget = app.query_one(f"#{prefix}-temperature", Input)
            top_p_widget = app.query_one(f"#{prefix}-top-p", Input)
            min_p_widget = app.query_one(f"#{prefix}-min-p", Input)
            top_k_widget = app.query_one(f"#{prefix}-top-k", Input)
            llm_max_tokens_widget = app.query_one(f"#{prefix}-llm-max-tokens", Input)
            llm_seed_widget = app.query_one(f"#{prefix}-llm-seed", Input)
            llm_stop_widget = app.query_one(f"#{prefix}-llm-stop", Input)
            llm_response_format_widget = app.query_one(f"#{prefix}-llm-response-format", Select)
            llm_n_widget = app.query_one(f"#{prefix}-llm-n", Input)
            llm_user_identifier_widget = app.query_one(f"#{prefix}-llm-user-identifier", Input)
            llm_logprobs_widget = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
            llm_top_logprobs_widget = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
            llm_logit_bias_widget = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
            llm_presence_penalty_widget = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
            llm_frequency_penalty_widget = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
            llm_tools_widget = app.query_one(f"#{prefix}-llm-tools", TextArea)
            llm_tool_choice_widget = app.query_one(f"#{prefix}-llm-tool-choice", Input)
            llm_fixed_tokens_kobold_widget = app.query_one(f"#{prefix}-llm-fixed-tokens-kobold", Checkbox)
            # Query for the strip thinking tags checkbox for suggestion
            try:
                strip_tags_checkbox_suggest = app.query_one("#chat-strip-thinking-tags-checkbox", Checkbox)
                strip_thinking_tags_value_suggest = strip_tags_checkbox_suggest.value
            except QueryError:
                loguru_logger.warning("Respond for Me: Could not find '#chat-strip-thinking-tags-checkbox'. Defaulting to True.")
                strip_thinking_tags_value_suggest = True
        except QueryError as e_params_query:
            loguru_logger.error(f"Respond for Me: Could not find UI settings widgets: {e_params_query}", exc_info=True)
            app.notify("Error: Missing UI settings for suggestion.", severity="error")
            raise # Re-raise to go to finally

        selected_provider = str(provider_widget.value) if provider_widget.value != Select.BLANK else None
        selected_model = str(model_widget.value) if model_widget.value != Select.BLANK else None
        temperature = safe_float(temp_widget.value, 0.7, "temperature")
        top_p = safe_float(top_p_widget.value, 0.95, "top_p")
        min_p = safe_float(min_p_widget.value, 0.05, "min_p")
        top_k = safe_int(top_k_widget.value, 50, "top_k")
        llm_max_tokens_value = safe_int(llm_max_tokens_widget.value, 200, "llm_max_tokens_suggestion") # Suggestion max tokens
        llm_seed_value = safe_int(llm_seed_widget.value, None, "llm_seed")
        llm_stop_value = [s.strip() for s in llm_stop_widget.value.split(',') if s.strip()] if llm_stop_widget.value.strip() else None
        llm_response_format_value = {"type": str(llm_response_format_widget.value)} if llm_response_format_widget.value != Select.BLANK else {"type": "text"}
        llm_n_value = safe_int(llm_n_widget.value, 1, "llm_n")
        llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
        llm_logprobs_value = llm_logprobs_widget.value
        llm_top_logprobs_value = safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs") if llm_logprobs_value else 0
        llm_presence_penalty_value = safe_float(llm_presence_penalty_widget.value, 0.0, "llm_presence_penalty")
        llm_frequency_penalty_value = safe_float(llm_frequency_penalty_widget.value, 0.0, "llm_frequency_penalty")
        llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None
        llm_fixed_tokens_kobold_value = llm_fixed_tokens_kobold_widget.value # Added
        try:
            llm_logit_bias_text = llm_logit_bias_widget.text.strip()
            llm_logit_bias_value = json.loads(llm_logit_bias_text) if llm_logit_bias_text and llm_logit_bias_text != "{}" else None
        except json.JSONDecodeError: llm_logit_bias_value = None; loguru_logger.warning("Invalid JSON in llm_logit_bias for suggestion.")
        try:
            llm_tools_text = llm_tools_widget.text.strip()
            llm_tools_value = json.loads(llm_tools_text) if llm_tools_text and llm_tools_text != "[]" else None
        except json.JSONDecodeError: llm_tools_value = None; loguru_logger.warning("Invalid JSON in llm_tools for suggestion.")

        # System Prompt: Use a generic one for suggestion, or allow character's? For now, generic.
        # Or, could use the main chat's system prompt if that makes sense.
        # For this feature, a neutral "you are a helpful assistant suggesting responses" might be better
        # than the character's persona, unless the goal is for the character to suggest *as if they were the user*.
        # Let's use a new, specific system prompt for this feature for now.
        suggestion_system_prompt = "You are an AI assistant helping a user by suggesting potential chat responses based on conversation history."

        # If using the main chat's system prompt:
        # final_system_prompt_for_api = system_prompt_widget.text
        # if app.current_chat_active_character_data:
        #     char_sys_prompt = app.current_chat_active_character_data.get('system_prompt')
        #     if char_sys_prompt and char_sys_prompt.strip():
        #         final_system_prompt_for_api = char_sys_prompt
        final_system_prompt_for_api = suggestion_system_prompt


        # API Key Fetching (copied from continue handler, ensure it's complete)
        api_key_for_call = None
        if selected_provider:
            provider_settings_key = selected_provider.lower().replace(" ", "_")
            provider_config = app.app_config.get("api_settings", {}).get(provider_settings_key, {})
            if "api_key" in provider_config and provider_config["api_key"] and provider_config["api_key"] != "<API_KEY_HERE>":
                api_key_for_call = provider_config["api_key"]
            elif "api_key_env_var" in provider_config and provider_config["api_key_env_var"]:
                api_key_for_call = os.environ.get(provider_config["api_key_env_var"])

        providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]
        if selected_provider in providers_requiring_key and not api_key_for_call:
            loguru_logger.error(f"API Key for '{selected_provider}' is missing for suggestion.")
            app.notify(f"API Key for {selected_provider} is missing.", severity="error")
            raise ApiKeyMissingError(f"API Key for {selected_provider} required.") # Custom exception to catch in finally

        # --- 4. Perform Non-Streaming LLM Call ---
        # For simplicity, the prompt contains the history. Alternatively, pass structured history.
        # The chat_wrapper might need adjustment if it expects history only for streaming.
        # Assuming chat_wrapper can take message + history for non-streaming.
        # If not, history_for_api should be [] and suggestion_prompt_instruction contains all.

        # Forcing non-streaming for a direct suggestion response.
        # The `message` param to chat_wrapper is the main prompt.
        # `history` param is the preceding conversation.

        # Define the target for the worker
        worker_target = lambda: app.chat_wrapper(
            message=suggestion_prompt_instruction, # This is the specific instruction to suggest a response
            history=[], # Full context is in the message for this specific prompt type
            api_endpoint=selected_provider,
            api_key=api_key_for_call,
            system_message=final_system_prompt_for_api, # This is the suggestion_system_prompt
            temperature=temperature,
            topp=top_p, minp=min_p, topk=top_k,
            llm_max_tokens=llm_max_tokens_value,
            llm_seed=llm_seed_value,
            llm_stop=llm_stop_value,
            llm_response_format=llm_response_format_value,
            llm_n=llm_n_value,
            llm_user_identifier=llm_user_identifier_value,
            llm_logprobs=llm_logprobs_value,
            llm_top_logprobs=llm_top_logprobs_value,
            llm_logit_bias=llm_logit_bias_value,
            llm_presence_penalty=llm_presence_penalty_value,
            llm_frequency_penalty=llm_frequency_penalty_value,
            llm_tools=llm_tools_value,
            llm_tool_choice=llm_tool_choice_value,
            llm_fixed_tokens_kobold=llm_fixed_tokens_kobold_value,
            # Ensure custom_prompt, media_content etc. are defaulted if not used for suggestions
            custom_prompt="", media_content={}, selected_parts=[], chatdict_entries=None,
            max_tokens=500, # This is chatdict's max_tokens, distinct from llm_max_tokens. Review if needed here.
            strategy="sorted_evenly", # Default or from config
            strip_thinking_tags=strip_thinking_tags_value_suggest, # Pass for suggestion
            streaming=False # Explicitly non-streaming for suggestions
        )

        # Run the LLM call in a worker
        worker = app.run_worker(
            worker_target,
            name="respond_for_me_worker",
            group="llm_suggestions",
            thread=True,
            description="Generating suggestion for user response..."
        )
        app.set_current_chat_worker(worker)

        # The response will be handled by a worker event (e.g., on_stream_done or a custom one).
        # So, remove direct processing of llm_response_text and UI population here.
        # The notification "Suggestion populated..." will also move to that future event handler.

        loguru_logger.debug(f"Suggestion prompt instruction: {suggestion_prompt_instruction[:500]}...")
        loguru_logger.debug(f"Suggestion params: provider='{selected_provider}', model='{selected_model}', system_prompt (for suggestion)='{final_system_prompt_for_api[:100]}...'")

        loguru_logger.info("Respond for Me worker dispatched. Waiting for suggestion...")

    except ApiKeyMissingError as e_api_key: # Specific catch for API key issues
        # Notification already handled where raised or before.
        loguru_logger.error(f"API Key Error for suggestion: {e_api_key}")
    except ValueError as e_val: # Catch specific ValueErrors like empty history or bad LLM response
        loguru_logger.warning(f"Respond for Me: Value error encountered: {e_val}")
        # Notification for empty history is handled above. Others as they occur.
    except Exception as e_main:
        loguru_logger.error(f"Failed to generate suggestion: {e_main}", exc_info=True)
        app.notify(f"Failed to generate suggestion: {str(e_main)[:100]}", severity="error", timeout=5)
    finally:
        if respond_button:
            respond_button.disabled = False
            respond_button.label = original_button_label
        loguru_logger.debug("Respond for Me button re-enabled.")

class ApiKeyMissingError(Exception): # Custom exception for cleaner handling in try/finally
    pass


async def handle_stop_chat_generation_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Stop Chat Generation' button press."""
    loguru_logger.info("Stop Chat Generation button pressed.")

    worker_cancelled = False
    if app.current_chat_worker and app.current_chat_worker.is_running:
        try:
            app.current_chat_worker.cancel()
            loguru_logger.info(f"Cancellation requested for worker: {app.current_chat_worker.name}")
            worker_cancelled = True # Mark that cancellation was attempted

            if not app.current_chat_is_streaming:
                loguru_logger.debug("Handling cancellation for a non-streaming chat request.")
                if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
                    try:
                        # Update the placeholder message to indicate cancellation
                        static_text_widget = app.current_ai_message_widget.query_one(".message-text", Static)
                        cancelled_text = "[italic]Chat generation cancelled by user.[/]"
                        static_text_widget.update(Text.from_markup(cancelled_text))

                        app.current_ai_message_widget.message_text = "Chat generation cancelled by user." # Update raw text
                        app.current_ai_message_widget.role = "System" # Change role

                        # Update header if it exists
                        try:
                            header_label = app.current_ai_message_widget.query_one(".message-header", Label)
                            header_label.update("System Message")
                        except QueryError:
                            loguru_logger.warning("Could not find .message-header to update for non-streaming cancellation.")

                        app.current_ai_message_widget.mark_generation_complete() # Finalize UI state
                        loguru_logger.info("Non-streaming AI message widget UI updated for cancellation.")
                    except QueryError as qe_widget_update:
                        loguru_logger.error(f"Error updating non-streaming AI message widget UI on cancellation: {qe_widget_update}", exc_info=True)
                else:
                    loguru_logger.warning("Non-streaming cancellation: current_ai_message_widget not found or not mounted.")
            else: # It was a streaming request
                loguru_logger.info("Cancellation for a streaming chat request initiated. Worker will handle stream termination.")
                # For streaming, the worker itself should detect cancellation and stop sending StreamChunks.
                # The on_stream_done event (with error or cancellation status) will then handle UI finalization.

        except Exception as e_cancel:
            loguru_logger.error(f"Error during worker cancellation or UI update: {e_cancel}", exc_info=True)
            app.notify("Error trying to stop generation.", severity="error")
    else:
        loguru_logger.info("No active and running chat worker to stop.")
        if not app.current_chat_worker:
            loguru_logger.debug("current_chat_worker is None.")
        elif not app.current_chat_worker.is_running:
            loguru_logger.debug(f"current_chat_worker ({app.current_chat_worker.name}) is not running (state: {app.current_chat_worker.state}).")


    # Update the send button to change from stop back to send state
    # This provides immediate visual feedback.
    try:
        send_button = app.query_one("#send-chat", Button)
        send_button.label = get_char(EMOJI_SEND, FALLBACK_SEND)
        loguru_logger.debug("Changed send button back to send state.")
    except QueryError:
        loguru_logger.error("Could not find '#send-chat' button to update its state.")


async def populate_chat_conversation_character_filter_select(app: 'TldwCli') -> None:
    """Populates the character filter select in the Chat tab's conversation search."""
    # ... (Keep original implementation as is) ...
    logging.info("Attempting to populate #chat-conversation-search-character-filter-select.")
    if not app.notes_service:
        logging.error("Notes service not available for char filter select (Chat Tab).")
        # Optionally update the select to show an error state
        try:
            char_filter_select_err = app.query_one("#chat-conversation-search-character-filter-select", Select)
            char_filter_select_err.set_options([("Service Offline", Select.BLANK)])
        except QueryError: pass
        return
    try:
        db = app.notes_service._get_db(app.notes_user_id)
        character_cards = db.list_character_cards(limit=1000)
        options = [(char['name'], char['id']) for char in character_cards if char.get('name') and char.get('id')]

        char_filter_select = app.query_one("#chat-conversation-search-character-filter-select", Select)
        char_filter_select.set_options(options if options else [("No characters", Select.BLANK)])
        # Default to BLANK, user must explicitly choose or use "All Characters" checkbox
        char_filter_select.value = Select.BLANK
        logging.info(f"Populated #chat-conversation-search-character-filter-select with {len(options)} chars.")
    except QueryError as e_q:
        logging.error(f"Failed to find #chat-conversation-search-character-filter-select: {e_q}", exc_info=True)
    except CharactersRAGDBError as e_db: # Catch specific DB error
        logging.error(f"DB error populating char filter select (Chat Tab): {e_db}", exc_info=True)
    except Exception as e_unexp:
        logging.error(f"Unexpected error populating char filter select (Chat Tab): {e_unexp}", exc_info=True)


# --- Button Handler Map ---
# This maps button IDs to their async handler functions.
CHAT_BUTTON_HANDLERS = {
    "send-chat": handle_chat_send_button_pressed,
    "respond-for-me-button": handle_respond_for_me_button_pressed,
    "stop-chat-generation": handle_stop_chat_generation_pressed,
    "chat-new-conversation-button": handle_chat_new_conversation_button_pressed,
    "chat-save-current-chat-button": handle_chat_save_current_chat_button_pressed,
    "chat-save-conversation-details-button": handle_chat_save_details_button_pressed,
    "chat-conversation-load-selected-button": handle_chat_load_selected_button_pressed,
    "chat-prompt-load-selected-button": handle_chat_view_selected_prompt_button_pressed,
    "chat-prompt-copy-system-button": handle_chat_copy_system_prompt_button_pressed,
    "chat-prompt-copy-user-button": handle_chat_copy_user_prompt_button_pressed,
    "chat-load-character-button": handle_chat_load_character_button_pressed,
    "chat-clear-active-character-button": handle_chat_clear_active_character_button_pressed,
    "chat-apply-template-button": handle_chat_apply_template_button_pressed,
    "toggle-chat-left-sidebar": handle_chat_tab_sidebar_toggle,
    "toggle-chat-right-sidebar": handle_chat_tab_sidebar_toggle,
    **chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS,
}

#
# End of chat_events.py
########################################################################################################################
