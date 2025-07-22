# chat_streaming_events.py
#
# Imports
import json
import logging
import re
import threading
#
# Third-party Imports
from rich.text import Text
from textual.containers import VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Static, TextArea, Label, Markdown
from rich.markup import escape as escape_markup
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import InputError, CharactersRAGDBError
from tldw_chatbook.Constants import TAB_CHAT, TAB_CCP
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.Chat.Chat_Functions import parse_tool_calls_from_response
from tldw_chatbook.Tools import get_tool_executor
from tldw_chatbook.Widgets.tool_message_widgets import ToolExecutionWidget
#
########################################################################################################################
#
# Event Handlers for Streaming Events

async def handle_streaming_chunk(self, event: StreamingChunk) -> None:
    """Handles incoming chunks of text during streaming."""
    logger = getattr(self, 'loguru_logger', logging)
    
    # Check if this is a continuation (has continue_message_widget) or a new message
    current_widget = None
    is_continuation = False
    
    if hasattr(self, 'continue_message_widget') and self.continue_message_widget:
        # This is a continuation
        current_widget = self.continue_message_widget
        is_continuation = True
    else:
        # Get current widget using thread-safe method
        current_widget = self.get_current_ai_message_widget()
    
    if current_widget and current_widget.is_mounted:
        try:
            markdown_widget = current_widget.query_one(".message-text", Markdown)
            
            # Check if we need to clear the thinking emoji (first real chunk)
            if is_continuation:
                # For continuation, handle the first chunk differently
                if not hasattr(self, 'continue_thinking_removed') or not self.continue_thinking_removed:
                    # First chunk in continuation - append to existing text
                    if hasattr(self, 'continue_original_text'):
                        current_widget.message_text = self.continue_original_text + event.text_chunk
                    else:
                        current_widget.message_text += event.text_chunk
                    self.continue_thinking_removed = True
                else:
                    # Subsequent chunks - append normally
                    current_widget.message_text += event.text_chunk
            else:
                # Regular streaming (not continuation)
                if not hasattr(current_widget, '_streaming_started'):
                    # This is the first chunk - replace any placeholder content
                    current_widget.message_text = event.text_chunk
                    current_widget._streaming_started = True
                else:
                    # Subsequent chunks - append to internal state
                    current_widget.message_text += event.text_chunk
            
            # Always update markdown widget with full text to prevent flickering
            markdown_widget.update(current_widget.message_text)

            # Scroll the chat log to the end, conditionally
            chat_log_id_to_query = None
            if self.current_tab == TAB_CHAT:
                chat_log_id_to_query = "#chat-log"
            elif self.current_tab == TAB_CCP:
                chat_log_id_to_query = "#ccp-conversation-log"  # Ensure this is the correct ID for CCP tab's log

            if chat_log_id_to_query:
                try:
                    chat_log_container = self.query_one(chat_log_id_to_query, VerticalScroll)
                    chat_log_container.scroll_end(animate=False, duration=0.05)
                except QueryError:
                    # This path should ideally not be hit if current_tab is Chat or CCP and their logs exist
                    logger.warning(
                        f"on_streaming_chunk: Could not find chat log container '{chat_log_id_to_query}' even when tab is {self.current_tab}")
            else:
                # This else block will be hit if current_tab is not CHAT or CCP
                logger.debug(
                    f"on_streaming_chunk: Current tab is {self.current_tab}, not attempting to scroll chat log.")

        except QueryError as e:
            logger.error(f"Error accessing UI components during streaming chunk update: {e}", exc_info=True)
        except Exception as e_chunk:  # Catch any other unexpected error
            logger.error("Unexpected error processing streaming chunk: {}", e_chunk, exc_info=True)
    else:
        logger.warning(
            "Received StreamingChunk but no current_ai_message_widget is active/mounted or tab is not Chat/CCP.")


async def handle_stream_done(self, event: StreamDone) -> None:
    """Handles the end of a stream, including errors and successful completion."""
    logger = getattr(self, 'loguru_logger', logging)
    logger.info(f"StreamDone received. Final text length: {len(event.full_text)}. Error: '{event.error}'")

    # Check if this is a continuation or a new message
    ai_widget = None
    is_continuation = False
    
    if hasattr(self, 'continue_message_widget') and self.continue_message_widget:
        # This is a continuation
        ai_widget = self.continue_message_widget
        is_continuation = True
    else:
        # Get current widget using thread-safe method
        ai_widget = self.get_current_ai_message_widget()

    if not ai_widget or not ai_widget.is_mounted:
        logger.warning("Received StreamDone but current_ai_message_widget is missing or not mounted.")
        if event.error:  # If there was an error, at least notify the user
            self.notify(f"Stream error (display widget missing): {event.error}", severity="error", timeout=10)
        # Clear current widget using thread-safe method
        self.set_current_ai_message_widget(None)
        # Attempt to focus input if possible as a fallback
        try:
            if self.current_tab == TAB_CHAT:
                self.query_one("#chat-input", TextArea).focus()
            elif self.current_tab == TAB_CCP:  # Assuming similar input ID convention
                self.query_one("#ccp-chat-input", TextArea).focus()  # Adjust if ID is different
        except QueryError:
            pass  # Ignore if input not found
        return

    try:
        markdown_widget = ai_widget.query_one(".message-text", Markdown)

        if event.error:
            logger.error(f"Stream completed with error: {event.error}")
            # If full_text has content, it means some chunks were received before the error.
            # Display partial text along with the error.
            error_message_content = event.full_text + f"\n\n[bold red]Stream Error:[/]\n{escape_markup(event.error)}"

            ai_widget.message_text = event.full_text + f"\n\nStream Error:\n{event.error}"  # Update internal raw text
            markdown_widget.update(ai_widget.message_text)
            ai_widget.role = "System"  # Change role to "System" or "Error"
            try:
                header_label = ai_widget.query_one(".message-header", Label)
                header_label.update("System Error")  # Update header
            except QueryError:
                logger.warning("Could not update AI message header for stream error display.")
            # Do NOT save to database if there was an error.
        else:  # No error, stream completed successfully
            logger.info("Stream completed successfully.")

            # Apply thinking tag stripping if enabled
            if event.full_text:  # Check if there's any text to process
                strip_tags_setting = self.app_config.get("chat_defaults", {}).get("strip_thinking_tags", True)
                self.loguru_logger.info(f"Strip thinking tags setting: {strip_tags_setting} (from config: {self.app_config.get('chat_defaults', {})})")
                if strip_tags_setting:
                    # Match both <think> and <thinking> tags
                    think_blocks = list(re.finditer(r"<think(?:ing)?>.*?</think(?:ing)?>", event.full_text, re.DOTALL))
                    if len(think_blocks) > 1:
                        self.loguru_logger.debug(
                            f"Stripping thinking tags from streamed response. Found {len(think_blocks)} blocks.")
                        text_parts = []
                        last_kept_block_end = 0
                        for i, block in enumerate(think_blocks):
                            if i < len(think_blocks) - 1:  # This is a block to remove
                                text_parts.append(event.full_text[last_kept_block_end:block.start()])
                                last_kept_block_end = block.end()
                        text_parts.append(event.full_text[last_kept_block_end:])
                        event.full_text = "".join(text_parts)  # Modify the event's full_text
                        self.loguru_logger.debug(f"Streamed response after stripping: {event.full_text[:200]}...")
                    else:
                        self.loguru_logger.info(
                            f"Not stripping tags from stream: {len(think_blocks)} block(s) found (need >1), setting is {strip_tags_setting}.")
                else:
                    self.loguru_logger.debug("Not stripping tags from stream: strip_thinking_tags setting is disabled.")

            ai_widget.message_text = event.full_text  # Ensure internal state has the final, complete text
            markdown_widget.update(event.full_text)  # Update display with final text

            # Check for tool calls in the response
            tool_calls = None
            if hasattr(event, 'response_data') and event.response_data:
                logger.debug(f"Checking for tool calls in response data: {type(event.response_data)}")
                tool_calls = parse_tool_calls_from_response(event.response_data)
                if tool_calls:
                    logger.info(f"Detected {len(tool_calls)} tool call(s) in streaming response")
                    
                    # Get the chat container from the AI widget's parent
                    chat_container = ai_widget.parent
                    if chat_container:
                        # Create and mount tool execution widget
                        tool_widget = ToolExecutionWidget(tool_calls)
                        await chat_container.mount(tool_widget)
                        chat_container.scroll_end(animate=False)
                        
                        # Execute tools asynchronously
                        executor = get_tool_executor()
                        try:
                            results = await executor.execute_tool_calls(tool_calls)
                            logger.info(f"Tool execution completed with {len(results)} result(s)")
                            
                            # Update widget with results
                            tool_widget.update_results(results)
                            
                            # Save tool messages to database if applicable
                            if self.chachanotes_db and self.current_chat_conversation_id and not self.current_chat_is_ephemeral:
                                try:
                                    # Save tool call message
                                    tool_call_msg = f"Tool Calls:\n{json.dumps(tool_calls, indent=2)}"
                                    tool_call_db_id = ccl.add_message_to_conversation(
                                        self.chachanotes_db,
                                        self.current_chat_conversation_id,
                                        "tool",  # This will map to role='tool'
                                        tool_call_msg
                                    )
                                    logger.debug(f"Saved tool call message to DB with ID: {tool_call_db_id}")
                                    
                                    # Save tool results message
                                    tool_results_msg = f"Tool Results:\n{json.dumps(results, indent=2)}"
                                    tool_result_db_id = ccl.add_message_to_conversation(
                                        self.chachanotes_db,
                                        self.current_chat_conversation_id,
                                        "tool",
                                        tool_results_msg
                                    )
                                    logger.debug(f"Saved tool results message to DB with ID: {tool_result_db_id}")
                                except Exception as e:
                                    logger.error(f"Error saving tool messages to DB: {e}", exc_info=True)
                            
                            # Continue conversation with tool results
                            await self._continue_conversation_with_tools(results, tool_calls)
                            
                        except Exception as e:
                            logger.error(f"Error executing tools: {e}", exc_info=True)
                            self.notify(f"Tool execution error: {str(e)}", severity="error")
                    else:
                        logger.warning("Could not find chat container for mounting tool widgets")
                        # Fallback: just append a notice
                        tool_notice = f"\n\n[Tool Calls Detected: {len(tool_calls)} function(s) to execute]"
                        ai_widget.message_text += tool_notice
                        markdown_widget.update(ai_widget.message_text)

            # Determine sender name for DB (already set on widget by handle_api_call_worker_state_changed)
            # This is just to ensure the correct name is used for DB saving if needed.
            ai_sender_name_for_db = ai_widget.role  # Role should be correctly set by now

            # Save to DB if applicable (not ephemeral, not empty, and DB available)
            if self.chachanotes_db and self.current_chat_conversation_id and \
                    not self.current_chat_is_ephemeral and event.full_text.strip():
                
                if is_continuation:
                    # For continuation, update the existing message
                    if ai_widget.message_id_internal and hasattr(ai_widget, 'message_version_internal'):
                        try:
                            logger.debug(
                                f"Attempting to update continued message in DB. MsgID: {ai_widget.message_id_internal}, Version: {ai_widget.message_version_internal}")
                            success = ccl.edit_message_content(
                                self.chachanotes_db,
                                ai_widget.message_id_internal,
                                event.full_text,  # The complete continued text
                                ai_widget.message_version_internal  # Expected version
                            )
                            if success:
                                ai_widget.message_version_internal += 1
                                logger.info(f"Continued message ID {ai_widget.message_id_internal} updated in DB. New version: {ai_widget.message_version_internal}")
                                self.notify("Message continuation saved.", severity="information", timeout=2)
                            else:
                                logger.error(f"Failed to update continued message {ai_widget.message_id_internal} in DB")
                                self.notify("Failed to save continuation to DB", severity="error")
                        except Exception as e_update:
                            logger.error(f"Error updating continued message: {e_update}", exc_info=True)
                            self.notify(f"DB error saving continuation: {str(e_update)[:100]}", severity="error")
                else:
                    # For new messages, add to conversation
                    try:
                        logger.debug(
                            f"Attempting to save streamed AI message to DB. ConvID: {self.current_chat_conversation_id}, Sender: {ai_sender_name_for_db}")
                        ai_msg_db_id = ccl.add_message_to_conversation(
                            self.chachanotes_db,
                            self.current_chat_conversation_id,
                            ai_sender_name_for_db,
                            event.full_text  # Save the clean, full text
                        )
                        if ai_msg_db_id:
                            saved_ai_msg_details = self.chachanotes_db.get_message_by_id(ai_msg_db_id)
                            if saved_ai_msg_details:
                                ai_widget.message_id_internal = saved_ai_msg_details.get('id')
                                ai_widget.message_version_internal = saved_ai_msg_details.get('version')
                                logger.info(
                                    f"Streamed AI message saved to DB. ConvID: {self.current_chat_conversation_id}, MsgID: {saved_ai_msg_details.get('id')}")
                            else:
                                logger.error(
                                    f"Failed to retrieve saved streamed AI message details (ID: {ai_msg_db_id}) from DB.")
                        else:
                            logger.error("Failed to save streamed AI message to DB (no ID returned).")
                    except (CharactersRAGDBError, InputError) as e_save_ai_stream:
                        logger.error(f"DB Error saving streamed AI message: {e_save_ai_stream}", exc_info=True)
                        self.notify(f"DB error saving message: {e_save_ai_stream}", severity="error")
                    except Exception as e_save_unexp:
                        logger.error(f"Unexpected error saving streamed AI message: {e_save_unexp}", exc_info=True)
                        self.notify("Unexpected error saving message.", severity="error")
            elif not event.full_text.strip() and not event.error:
                logger.info("Stream finished with no error but content was empty/whitespace. Not saving to DB.")

        ai_widget.mark_generation_complete()  # Mark as complete in both error/success cases if widget exists
        # Clean up streaming flag
        if hasattr(ai_widget, '_streaming_started'):
            delattr(ai_widget, '_streaming_started')
        
        # Update token counter after AI response is complete
        try:
            from .chat_token_events import update_chat_token_counter
            await update_chat_token_counter(self)
        except Exception as e:
            logger.debug(f"Could not update token counter: {e}")

    except QueryError as e:
        logger.error(f"QueryError during StreamDone UI update (event.error='{event.error}'): {e}", exc_info=True)
        if event.error:  # If there was an underlying stream error, make sure user sees it
            self.notify(f"Stream Error (UI issue): {event.error}", severity="error", timeout=10)
        else:  # If stream was fine, but UI update failed
            self.notify("Error finalizing AI message display.", severity="error")
    except Exception as e_done_unexp:  # Catch any other unexpected error during the try block
        logger.error(f"Unexpected error in on_stream_done (event.error='{event.error}'): {e_done_unexp}", exc_info=True)
        self.notify("Internal error finalizing stream.", severity="error")
    finally:
        # This block executes regardless of exceptions in the try block above.
        # Crucial for resetting state and UI.
        
        # Handle continuation cleanup
        if is_continuation:
            # Clean up continuation-specific attributes
            if hasattr(self, 'continue_message_widget'):
                delattr(self, 'continue_message_widget')
            if hasattr(self, 'continue_markdown_widget'):
                delattr(self, 'continue_markdown_widget')
            if hasattr(self, 'continue_original_text'):
                delattr(self, 'continue_original_text')
            if hasattr(self, 'continue_thinking_removed'):
                delattr(self, 'continue_thinking_removed')
            
            # Re-enable continue button and other buttons
            if ai_widget and ai_widget.is_mounted:
                try:
                    continue_button = ai_widget.query_one("#continue-response-button")
                    continue_button.disabled = False
                    continue_button.label = "↪️"  # Reset to original label
                    
                    # Re-enable other action buttons
                    for btn_id in ["thumb-up", "thumb-down", "regenerate"]:
                        try:
                            button = ai_widget.query_one(f"#{btn_id}")
                            button.disabled = False
                        except QueryError:
                            pass
                            
                except QueryError:
                    logger.debug("Continue button not found during cleanup")
            
            logger.debug("Completed continuation cleanup")
        else:
            # Regular message cleanup
            self.current_ai_message_widget = None  # Clear the reference to the AI message widget
            logger.debug("Cleared current_ai_message_widget in on_stream_done's finally block.")
        
        # Reset streaming state using thread-safe method
        self.set_current_chat_is_streaming(False)
        logger.debug("Reset current_chat_is_streaming to False in on_stream_done's finally block.")

        # Focus the appropriate input based on the current tab
        input_id_to_focus = None
        if self.current_tab == TAB_CHAT:
            input_id_to_focus = "#chat-input"
        elif self.current_tab == TAB_CCP:
            input_id_to_focus = "#ccp-chat-input"  # Adjust if ID is different for CCP tab's input

        if input_id_to_focus:
            try:
                input_widget = self.query_one(input_id_to_focus, TextArea)
                input_widget.focus()
                logger.debug(f"Focused input '{input_id_to_focus}' in on_stream_done.")
            except QueryError:
                logger.warning(f"Could not focus input '{input_id_to_focus}' in on_stream_done (widget not found).")
            except Exception as e_focus_final:
                logger.error(f"Error focusing input '{input_id_to_focus}' in on_stream_done: {e_focus_final}",
                             exc_info=True)
        else:
            logger.debug(f"No specific input to focus for tab {self.current_tab} in on_stream_done.")

async def _continue_conversation_with_tools(self, tool_results, tool_calls):
    """
    Continue the conversation by sending tool results back to the LLM.
    
    Args:
        tool_results: List of tool execution results
        tool_calls: Original tool calls that were executed
    """
    logger = getattr(self, 'loguru_logger', logging)
    
    try:
        # Format tool results for the conversation
        tool_results_text = []
        for i, result in enumerate(tool_results):
            tool_call_id = result.get('tool_call_id', f'call_{i}')
            if 'error' in result:
                tool_results_text.append(f"Tool call {tool_call_id} failed: {result['error']}")
            else:
                tool_result_data = result.get('result', {})
                tool_results_text.append(f"Tool call {tool_call_id} result: {json.dumps(tool_result_data, indent=2)}")
        
        # Join all tool results
        formatted_results = "\n\n".join(tool_results_text)
        
        # Create a new user message with the tool results
        tool_response_message = f"[Tool Results]\n{formatted_results}\n\n[Continue with the original request using these tool results]"
        
        # Find the chat input widget
        try:
            if self.current_tab == TAB_CHAT:
                input_widget = self.query_one("#chat-input", TextArea)
            elif self.current_tab == TAB_CCP:
                input_widget = self.query_one("#ccp-chat-input", TextArea)
            else:
                logger.warning(f"Unknown tab {self.current_tab}, cannot continue conversation")
                return
            
            # Set the input with the tool results
            input_widget.text = tool_response_message
            
            # Trigger send button programmatically
            try:
                if self.current_tab == TAB_CHAT:
                    send_button = self.query_one("#chat-send-button")
                elif self.current_tab == TAB_CCP:
                    send_button = self.query_one("#ccp-send-button")
                
                # Post a button pressed event
                from textual.events import Click
                await send_button.post_message(Click(send_button, 0, 0, 0, 0, 0, False, False, False))
                logger.info("Triggered conversation continuation with tool results")
                
            except QueryError:
                logger.error("Could not find send button to continue conversation")
                self.notify("Could not automatically continue conversation with tool results", severity="warning")
                
        except QueryError as e:
            logger.error(f"Could not find chat input to continue conversation: {e}")
            self.notify("Could not continue conversation with tool results", severity="warning")
            
    except Exception as e:
        logger.error(f"Error continuing conversation with tool results: {e}", exc_info=True)
        self.notify(f"Error continuing conversation: {str(e)}", severity="error")

#
# End of Event Handlers for Streaming Events
########################################################################################################################
