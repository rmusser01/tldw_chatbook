# chat_streaming_events.py
#
# Imports
import logging
import re
import threading
#
# Third-party Imports
from rich.text import Text
from textual.containers import VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Static, TextArea, Label
from rich.markup import escape as escape_markup
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import InputError, CharactersRAGDBError
from tldw_chatbook.Constants import TAB_CHAT, TAB_CCP
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
#
########################################################################################################################
#
# Event Handlers for Streaming Events

async def handle_streaming_chunk(self, event: StreamingChunk) -> None:
    """Handles incoming chunks of text during streaming."""
    logger = getattr(self, 'loguru_logger', logging)
    
    # Get current widget using thread-safe method
    current_widget = self.get_current_ai_message_widget()
    if current_widget and current_widget.is_mounted:
        try:
            # The thinking placeholder should have been cleared when the worker started.
            # The role and header should also have been set at the start of the AI turn.
            static_text_widget = current_widget.query_one(".message-text", Static)

            # Atomically append the clean text chunk and update display
            # This prevents race conditions during concurrent text updates
            new_text = current_widget.message_text + event.text_chunk
            current_widget.message_text = new_text

            # --- Update the display by wrapping the text in a Text object ---
            # This is safer than escape_markup for arbitrary streaming content.
            static_text_widget.update(Text(new_text))

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
        static_text_widget = ai_widget.query_one(".message-text", Static)

        if event.error:
            logger.error(f"Stream completed with error: {event.error}")
            # If full_text has content, it means some chunks were received before the error.
            # Display partial text along with the error.
            error_message_content = event.full_text + f"\n\n[bold red]Stream Error:[/]\n{escape_markup(event.error)}"

            ai_widget.message_text = event.full_text + f"\n\nStream Error:\n{event.error}"  # Update internal raw text
            static_text_widget.update(ai_widget.message_text)
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
                if strip_tags_setting:
                    think_blocks = list(re.finditer(r"<think>.*?</think>", event.full_text, re.DOTALL))
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
                        self.loguru_logger.debug(
                            f"Not stripping tags from stream: {len(think_blocks)} block(s) found (need >1), setting is {strip_tags_setting}.")
                else:
                    self.loguru_logger.debug("Not stripping tags from stream: strip_thinking_tags setting is disabled.")

            ai_widget.message_text = event.full_text  # Ensure internal state has the final, complete text
            static_text_widget.update(event.full_text)  # Update display with final text

            # Determine sender name for DB (already set on widget by handle_api_call_worker_state_changed)
            # This is just to ensure the correct name is used for DB saving if needed.
            ai_sender_name_for_db = ai_widget.role  # Role should be correctly set by now

            # Save to DB if applicable (not ephemeral, not empty, and DB available)
            if self.chachanotes_db and self.current_chat_conversation_id and \
                    not self.current_chat_is_ephemeral and event.full_text.strip():
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
        self.current_ai_message_widget = None  # Clear the reference to the AI message widget
        logger.debug("Cleared current_ai_message_widget in on_stream_done's finally block.")

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

#
# End of Event Handlers for Streaming Events
########################################################################################################################
