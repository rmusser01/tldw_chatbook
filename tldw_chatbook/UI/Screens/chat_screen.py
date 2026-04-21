"""Chat screen implementation with comprehensive state management."""

from typing import TYPE_CHECKING, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import toml
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, TextArea, Select, Collapsible
from textual.events import Key
from textual import on
from textual.reactive import reactive
from textual.css.query import QueryError

from ..Navigation.base_app_screen import BaseAppScreen
from .chat_screen_state import ChatScreenState, TabState, MessageData
from ...Utils.chat_diagnostics import ChatDiagnostics
from ...state.ui_state import UIState

# Import the existing chat window to reuse its functionality
from ..Chat_Window_Enhanced import ChatWindowEnhanced
from ...Widgets.voice_input_widget import VoiceInputMessage

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="ChatScreen")


class ChatScreen(BaseAppScreen):
    """
    Chat screen with comprehensive state management.
    
    This screen preserves all chat state including tabs, messages,
    input text, and UI preferences when navigating away and returning.
    """
    
    @on(Select.Changed, "#chat-api-provider")
    async def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle API provider change and update model dropdown."""
        logger.info(f"API provider changed to: {event.value}")
        
        try:
            from tldw_chatbook.config import get_cli_providers_and_models
            
            # Get the new provider's models
            providers_models = get_cli_providers_and_models()
            new_provider = str(event.value)
            available_models = providers_models.get(new_provider, [])
            logger.info(f"Found {len(available_models)} models for provider {new_provider}")
            
            # Find the model select widget within the chat window
            if self.chat_window:
                try:
                    model_select = self.chat_window.query_one("#chat-api-model", Select)
                    
                    # Update options
                    new_model_options = [(model, model) for model in available_models]
                    model_select.set_options(new_model_options)
                    
                    # Set to first model or blank if no models
                    if available_models:
                        model_select.value = available_models[0]
                        logger.info(f"Set model to: {available_models[0]}")
                    else:
                        model_select.value = Select.BLANK
                        logger.info("No models available, set to BLANK")
                    
                    model_select.prompt = "Select Model..." if available_models else "No models available"
                    logger.info(f"Successfully updated model dropdown with {len(available_models)} models")
                except Exception as e:
                    logger.error(f"Could not find model select widget: {e}")
            else:
                logger.error("chat_window is None")
                
        except Exception as e:
            logger.error(f"Error updating model dropdown: {e}", exc_info=True)
    
    
    # Reactive property for sidebar state persistence
    sidebar_state = reactive({}, layout=False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "chat", **kwargs)
        self.chat_window: Optional[ChatWindowEnhanced] = None
        self.chat_state = ChatScreenState()
        self._state_dirty = False
        self._diagnostics_run = False
        self.ui_state = UIState()
        self._load_sidebar_state()
        
    def compose_content(self) -> ComposeResult:
        """Compose the chat content."""
        # Create and yield the chat window container
        self.chat_window = ChatWindowEnhanced(self.app_instance, id="chat-window", classes="window")
        yield self.chat_window
    
    def on_mount(self) -> None:
        """Run diagnostics when first mounted (only once)."""
        # Call parent's on_mount
        super().on_mount()
        
        if not self._diagnostics_run and self.chat_window:
            self._diagnostics_run = True
            # Run diagnostic in the background
            self.set_timer(0.5, self._run_diagnostic)
        
        # Restore collapsible states after mount
        self.set_timer(0.1, self._restore_collapsible_states)
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save comprehensive chat state.
        
        Captures all tabs, messages, input text, and UI state
        to fully restore the chat experience on return.
        """
        logger.debug("Saving chat screen state")
        state = super().save_state()
        
        try:
            # Create fresh state object
            self.chat_state = ChatScreenState()
            self.chat_state.last_saved = datetime.now()
            
            if self.chat_window:
                # Save UI preferences
                self.chat_state.left_sidebar_collapsed = getattr(
                    self.app_instance, 'chat_sidebar_collapsed', False
                )
                self.chat_state.right_sidebar_collapsed = getattr(
                    self.app_instance, 'chat_right_sidebar_collapsed', False
                )
                
                # Try to detect and save from different chat interface types
                tab_container = self._get_tab_container()
                
                if tab_container and hasattr(tab_container, 'sessions'):
                    # Tabbed interface detected
                    logger.debug(f"Detected tabbed interface with {len(tab_container.sessions)} tabs")
                    
                    # Save all tab sessions
                    self._save_tab_sessions(tab_container)
                    
                    # Save active tab
                    self.chat_state.active_tab_id = tab_container.active_session_id
                    
                    # Save tab order
                    if hasattr(tab_container, 'tab_bar') and tab_container.tab_bar:
                        self.chat_state.tab_order = list(tab_container.tab_bar.tabs.keys())
                    
                    # Also save messages for the active session
                    if tab_container.active_session_id:
                        active_tab = self.chat_state.get_tab_by_id(tab_container.active_session_id)
                        if active_tab:
                            self._extract_and_save_messages(active_tab)
                else:
                    # Non-tabbed interface - try to save single chat state
                    logger.debug("Detected non-tabbed chat interface")
                    self._save_non_tabbed_state()
                
                # Always try to save current input text directly
                self._save_direct_input_text()
                
                # Save sidebar settings (system prompt, temperature, etc.)
                self._save_sidebar_settings()
                
                # Save scroll positions
                self._save_scroll_positions()
                
                # Save pending attachments
                self._save_attachments()
            
            # Convert to dict for storage
            state['chat_state'] = self.chat_state.to_dict()
            state['state_version'] = '1.0'
            state['interface_type'] = 'tabbed' if self.chat_state.tabs else 'single'
            
            logger.info(f"Saved chat state: {len(self.chat_state.tabs)} tabs, interface: {state.get('interface_type')}")
            
        except Exception as e:
            logger.error(f"Error saving chat state: {e}", exc_info=True)
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore comprehensive chat state.
        
        Recreates all tabs, messages, and UI state from saved data.
        """
        logger.debug("Restoring chat screen state")
        super().restore_state(state)
        
        try:
            if 'chat_state' in state:
                # Restore from saved state
                self.chat_state = ChatScreenState.from_dict(state['chat_state'])
                
                logger.debug(f"Restored state has {len(self.chat_state.tabs)} tabs")
                logger.debug(f"Active tab ID: {self.chat_state.active_tab_id}")
                logger.debug(f"Tab order: {self.chat_state.tab_order}")
                
                if self.chat_state.validate():
                    logger.info(f"Restoring {len(self.chat_state.tabs)} tabs")
                    
                    # Schedule restoration after mount
                    self.set_timer(0.1, self._perform_state_restoration)
                else:
                    logger.warning("Chat state validation failed, starting fresh")
                    self.chat_state = ChatScreenState()
            
        except Exception as e:
            logger.error(f"Error restoring chat state: {e}", exc_info=True)
            self.chat_state = ChatScreenState()
    
    async def _perform_state_restoration(self) -> None:
        """Perform actual state restoration after UI is ready."""
        if not self.chat_window:
            logger.warning("Chat window not ready for restoration")
            # Try again in a moment
            self.set_timer(0.2, self._perform_state_restoration)
            return
        
        try:
            logger.info("Starting state restoration...")
            
            # Restore UI preferences
            self.app_instance.chat_sidebar_collapsed = self.chat_state.left_sidebar_collapsed
            self.app_instance.chat_right_sidebar_collapsed = self.chat_state.right_sidebar_collapsed
            
            # Get tab container
            tab_container = self._get_tab_container()
            if tab_container:
                # Tabbed interface - restore tab sessions
                await self._restore_tab_sessions(tab_container)
                
                # Restore active tab
                if self.chat_state.active_tab_id:
                    await tab_container.switch_to_tab(self.chat_state.active_tab_id)
            else:
                # Non-tabbed interface - still need to restore state
                logger.debug("Non-tabbed interface detected, restoring state directly")
            
            # Always restore these regardless of tab container
            # Restore input text
            await self._restore_input_text()
            
            # Restore sidebar settings (system prompt, temperature, etc.)
            await self._restore_sidebar_settings()
            
            # Restore scroll positions
            await self._restore_scroll_positions()
            
            # Restore attachments
            await self._restore_attachments()
            
            # Restore conversation messages
            await self._restore_messages()
            
            logger.info("Chat state restoration complete")
            
        except Exception as e:
            logger.error(f"Error during state restoration: {e}", exc_info=True)
    
    def _get_tab_container(self):
        """Get the ChatTabContainer widget."""
        try:
            if self.chat_window and hasattr(self.chat_window, '_tab_container'):
                return self.chat_window._tab_container
            return self.chat_window.query_one("ChatTabContainer")
        except:
            return None
    
    def _save_tab_sessions(self, tab_container) -> None:
        """Save all tab session states."""
        self.chat_state.tabs.clear()
        
        for session_id, session in tab_container.sessions.items():
            tab_state = TabState(
                tab_id=session_id,
                title=session.session_data.title,
                conversation_id=session.session_data.conversation_id,
                character_id=session.session_data.character_id,
                character_name=session.session_data.character_name,
                is_active=(session_id == tab_container.active_session_id),
                is_ephemeral=session.session_data.is_ephemeral,
                has_unsaved_changes=session.session_data.has_unsaved_changes,
                system_prompt_override=session.session_data.system_prompt_override,
                temperature_override=session.session_data.temperature_override,
                max_tokens_override=session.session_data.max_tokens_override,
            )
            
            # Save input text for this tab
            try:
                input_widget = session.query_one(f"#chat-input-{session_id}", TextArea)
                if input_widget:
                    tab_state.input_text = input_widget.text
                    # TextArea might not have cursor_position, use selection if available
                    if hasattr(input_widget, 'cursor_position'):
                        tab_state.cursor_position = input_widget.cursor_position
                    elif hasattr(input_widget, 'selection'):
                        tab_state.cursor_position = input_widget.selection.end if input_widget.selection else 0
                    else:
                        tab_state.cursor_position = len(input_widget.text)
            except:
                pass
            
            # Save scroll position
            try:
                scroll_widget = session.query_one(f"#chat-log-{session_id}")
                if scroll_widget:
                    tab_state.scroll_position = scroll_widget.scroll_y
            except:
                pass
            
            self.chat_state.tabs.append(tab_state)
    
    async def _restore_tab_sessions(self, tab_container) -> None:
        """Restore all tab sessions."""
        # Clear existing tabs except default
        for session_id in list(tab_container.sessions.keys()):
            if session_id != "default":
                await tab_container.close_tab(session_id)
        
        # Restore saved tabs
        for tab_state in self.chat_state.tabs:
            if tab_state.tab_id == "default" and "default" in tab_container.sessions:
                # Update default tab
                session = tab_container.sessions["default"]
                session.session_data.title = tab_state.title
                session.session_data.conversation_id = tab_state.conversation_id
                session.session_data.character_id = tab_state.character_id
                session.session_data.character_name = tab_state.character_name
            else:
                # Create new tab
                tab_id = await tab_container.create_new_tab(title=tab_state.title)
                if tab_id and tab_id in tab_container.sessions:
                    session = tab_container.sessions[tab_id]
                    session.session_data.conversation_id = tab_state.conversation_id
                    session.session_data.character_id = tab_state.character_id
                    session.session_data.character_name = tab_state.character_name
                    session.session_data.is_ephemeral = tab_state.is_ephemeral
                    session.session_data.has_unsaved_changes = tab_state.has_unsaved_changes
    
    def _save_input_text(self) -> None:
        """Save input text for active tab."""
        try:
            tab_container = self._get_tab_container()
            if tab_container and tab_container.active_session_id:
                active_tab = self.chat_state.get_tab_by_id(tab_container.active_session_id)
                if active_tab:
                    input_widget = self.chat_window.query_one("#chat-input", TextArea)
                    if input_widget:
                        active_tab.input_text = input_widget.text
                        logger.debug(f"Saved input text for tab {tab_container.active_session_id}: '{input_widget.text[:50]}...'")
                        # TextArea might not have cursor_position
                        if hasattr(input_widget, 'cursor_position'):
                            active_tab.cursor_position = input_widget.cursor_position
                        elif hasattr(input_widget, 'selection'):
                            active_tab.cursor_position = input_widget.selection.end if input_widget.selection else 0
                        else:
                            active_tab.cursor_position = len(input_widget.text)
        except Exception as e:
            logger.debug(f"Could not save input text: {e}")
    
    async def _restore_input_text(self) -> None:
        """Restore input text for active tab."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if active_tab and active_tab.input_text:
                logger.info(f"Restoring input text: '{active_tab.input_text[:50]}...'")
                
                # Try to find the input widget
                try:
                    input_widget = self.chat_window.query_one("#chat-input", TextArea)
                except Exception:
                    # Try alternate query
                    input_widget = self.chat_window.query_one("#chat-input")
                
                if input_widget and hasattr(input_widget, 'load_text'):
                    input_widget.load_text(active_tab.input_text)
                    logger.info(f"Successfully restored input text to widget")
                    
                    # Try to restore cursor position
                    if hasattr(input_widget, 'cursor_position'):
                        try:
                            input_widget.cursor_position = active_tab.cursor_position
                        except Exception:
                            pass
                elif input_widget and hasattr(input_widget, 'value'):
                    # Try setting value directly
                    input_widget.value = active_tab.input_text
                    logger.info(f"Restored input text via value property")
                else:
                    logger.warning(f"Could not find suitable method to restore text to widget: {type(input_widget)}")
            else:
                logger.debug("No input text to restore")
        except Exception as e:
            logger.error(f"Error restoring input text: {e}", exc_info=True)
    
    def _save_scroll_positions(self) -> None:
        """Save scroll positions for all tabs."""
        # Implementation depends on tab structure
        pass
    
    async def _restore_scroll_positions(self) -> None:
        """Restore scroll positions for visible tabs."""
        # Implementation depends on tab structure
        pass
    
    def _save_sidebar_settings(self) -> None:
        """Save sidebar settings including system prompt, temperature, etc."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if not active_tab:
                # Create default tab if none exists
                active_tab = TabState(tab_id="default", title="Chat", is_active=True)
                self.chat_state.tabs = [active_tab]
                self.chat_state.active_tab_id = "default"
                self.chat_state.tab_order = ["default"]
            
            logger.debug("Attempting to save sidebar settings...")
            
            # Log widget IDs for debugging (only in debug mode)
            # Note: loguru doesn't have a simple .level property, skip debug logging for now
            # self._log_sidebar_widgets()
            
            # Save system prompt from sidebar
            system_prompt_saved = False
            try:
                system_prompt_widget = self.chat_window.query_one("#chat-system-prompt", TextArea)
                if system_prompt_widget and hasattr(system_prompt_widget, 'text'):
                    active_tab.system_prompt_override = system_prompt_widget.text
                    logger.info(f"✓ Saved system prompt: '{system_prompt_widget.text[:50]}...'")
                    system_prompt_saved = True
            except Exception as e:
                logger.debug(f"Could not find #chat-system-prompt: {e}")
            
            if not system_prompt_saved:
                # Try with all TextAreas and find the system prompt one
                try:
                    text_areas = self.chat_window.query("TextArea")
                    for ta in text_areas:
                        if ta.id and 'system-prompt' in str(ta.id):
                            active_tab.system_prompt_override = ta.text
                            logger.info(f"✓ Saved system prompt from {ta.id}: '{ta.text[:50]}...'")
                            system_prompt_saved = True
                            break
                except Exception as e:
                    logger.debug(f"Could not find system prompt TextArea: {e}")
            
            # Save temperature
            temp_saved = False
            try:
                temp_input = self.chat_window.query_one("#chat-temperature", Input)
                if temp_input and temp_input.value:
                    active_tab.temperature_override = float(temp_input.value)
                    logger.info(f"✓ Saved temperature: {temp_input.value}")
                    temp_saved = True
            except Exception as e:
                logger.debug(f"Could not find #chat-temperature: {e}")
            
            if not temp_saved:
                # Try to find temperature input by searching all inputs
                try:
                    inputs = self.chat_window.query("Input")
                    for inp in inputs:
                        if inp.id and 'temperature' in str(inp.id):
                            if inp.value:
                                active_tab.temperature_override = float(inp.value)
                                logger.info(f"✓ Saved temperature from {inp.id}: {inp.value}")
                                temp_saved = True
                                break
                except Exception as e:
                    logger.debug(f"Could not find temperature Input: {e}")
            
            # Save max tokens
            try:
                max_tokens_input = self.chat_window.query_one("#chat-llm-max-tokens", Input)
                if max_tokens_input and max_tokens_input.value:
                    active_tab.max_tokens_override = int(max_tokens_input.value)
                    logger.info(f"✓ Saved max tokens: {max_tokens_input.value}")
            except Exception:
                # Try alternative ID
                try:
                    max_tokens_input = self.chat_window.query_one("#chat-max-tokens", Input)
                    if max_tokens_input and max_tokens_input.value:
                        active_tab.max_tokens_override = int(max_tokens_input.value)
                        logger.info(f"✓ Saved max tokens: {max_tokens_input.value}")
                except Exception as e:
                    logger.debug(f"Could not find max tokens input: {e}")
            
            logger.debug(f"Sidebar settings saved - System prompt: {bool(active_tab.system_prompt_override)}, "
                        f"Temperature: {active_tab.temperature_override}, Max tokens: {active_tab.max_tokens_override}")
                
        except Exception as e:
            logger.error(f"Error saving sidebar settings: {e}", exc_info=True)
    
    def _save_attachments(self) -> None:
        """Save pending attachment states."""
        if self.chat_window and hasattr(self.chat_window, 'pending_image'):
            active_tab = self.chat_state.get_active_tab()
            if active_tab and self.chat_window.pending_image:
                active_tab.pending_attachments = [self.chat_window.pending_image]
    
    async def _restore_sidebar_settings(self) -> None:
        """Restore sidebar settings including system prompt, temperature, etc."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if not active_tab:
                logger.debug("No active tab to restore sidebar settings from")
                return
            
            logger.debug(f"Attempting to restore sidebar settings - System prompt: {bool(active_tab.system_prompt_override)}, "
                        f"Temperature: {active_tab.temperature_override}, Max tokens: {active_tab.max_tokens_override}")
            
            # Restore system prompt to sidebar
            if active_tab.system_prompt_override is not None:
                system_restored = False
                try:
                    system_prompt_widget = self.chat_window.query_one("#chat-system-prompt", TextArea)
                    if system_prompt_widget:
                        if hasattr(system_prompt_widget, 'load_text'):
                            system_prompt_widget.load_text(active_tab.system_prompt_override)
                        elif hasattr(system_prompt_widget, 'text'):
                            system_prompt_widget.text = active_tab.system_prompt_override
                        else:
                            system_prompt_widget.value = active_tab.system_prompt_override
                        logger.info(f"✓ Restored system prompt to sidebar: '{active_tab.system_prompt_override[:50]}...'")
                        system_restored = True
                except Exception as e:
                    logger.debug(f"Could not restore to #chat-system-prompt: {e}")
                
                if not system_restored:
                    # Try finding any TextArea with system-prompt in ID
                    try:
                        text_areas = self.chat_window.query("TextArea")
                        for ta in text_areas:
                            if ta.id and 'system-prompt' in str(ta.id):
                                if hasattr(ta, 'load_text'):
                                    ta.load_text(active_tab.system_prompt_override)
                                elif hasattr(ta, 'text'):
                                    ta.text = active_tab.system_prompt_override
                                else:
                                    ta.value = active_tab.system_prompt_override
                                logger.info(f"✓ Restored system prompt to {ta.id}")
                                system_restored = True
                                break
                    except Exception as e:
                        logger.debug(f"Could not restore system prompt to any TextArea: {e}")
            
            # Restore temperature
            if active_tab.temperature_override is not None:
                temp_restored = False
                try:
                    temp_input = self.chat_window.query_one("#chat-temperature", Input)
                    if temp_input:
                        temp_input.value = str(active_tab.temperature_override)
                        logger.info(f"✓ Restored temperature: {active_tab.temperature_override}")
                        temp_restored = True
                except Exception as e:
                    logger.debug(f"Could not restore to #chat-temperature: {e}")
                
                if not temp_restored:
                    # Try finding any Input with temperature in ID
                    try:
                        inputs = self.chat_window.query("Input")
                        for inp in inputs:
                            if inp.id and 'temperature' in str(inp.id):
                                inp.value = str(active_tab.temperature_override)
                                logger.info(f"✓ Restored temperature to {inp.id}: {active_tab.temperature_override}")
                                temp_restored = True
                                break
                    except Exception as e:
                        logger.debug(f"Could not restore temperature to any Input: {e}")
            
            # Restore max tokens
            if active_tab.max_tokens_override is not None:
                try:
                    max_tokens_input = self.chat_window.query_one("#chat-llm-max-tokens", Input)
                    if max_tokens_input:
                        max_tokens_input.value = str(active_tab.max_tokens_override)
                        logger.info(f"✓ Restored max tokens: {active_tab.max_tokens_override}")
                except Exception:
                    # Try alternative ID
                    try:
                        max_tokens_input = self.chat_window.query_one("#chat-max-tokens", Input)
                        if max_tokens_input:
                            max_tokens_input.value = str(active_tab.max_tokens_override)
                            logger.info(f"✓ Restored max tokens: {active_tab.max_tokens_override}")
                    except Exception as e:
                        logger.debug(f"Could not restore max tokens: {e}")
                    
        except Exception as e:
            logger.error(f"Error restoring sidebar settings: {e}", exc_info=True)
    
    async def _restore_attachments(self) -> None:
        """Restore pending attachments."""
        active_tab = self.chat_state.get_active_tab()
        if active_tab and active_tab.pending_attachments and self.chat_window:
            # Restore first attachment
            if active_tab.pending_attachments:
                self.chat_window.pending_image = active_tab.pending_attachments[0]
                # Update UI to show attachment indicator
                if hasattr(self.chat_window, 'attachment_handler'):
                    self.chat_window.attachment_handler._update_attachment_indicator()
    
    async def _restore_messages(self) -> None:
        """Restore conversation messages to the chat log."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if not active_tab or not active_tab.messages:
                logger.debug("No messages to restore")
                return
                
            logger.info(f"Restoring {len(active_tab.messages)} messages to chat log")
            
            # Import required classes
            from textual.containers import VerticalScroll
            
            # Find the chat log container (it's a VerticalScroll)
            chat_log = None
            
            # Try the direct approach first
            try:
                chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
                logger.debug("Found chat log for restoration via app_instance")
            except Exception:
                pass
            
            # If not found, try other approaches
            if not chat_log:
                log_selectors = [
                    "#chat-log",
                    ".chat-log"
                ]
            
            for selector in log_selectors:
                try:
                    containers = self.chat_window.query(selector)
                    if containers:
                        chat_log = containers.first()
                        logger.debug(f"Found chat log container for restoration: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"Could not find chat log with {selector}: {e}")
            
            if not chat_log:
                logger.warning("Could not find chat log container to restore messages")
                return
            
            # Import message widget class
            from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            # Clear existing messages (optional - you might want to keep them)
            # await chat_log.remove_children()
            
            # Restore each message
            for i, msg_data in enumerate(active_tab.messages):
                try:
                    # Create a new message widget
                    image_data = None
                    if msg_data.metadata and 'image_data' in msg_data.metadata:
                        image_data = msg_data.metadata['image_data']
                    
                    message_widget = ChatMessageEnhanced(
                        message=msg_data.content,
                        role=msg_data.role,
                        timestamp=msg_data.timestamp,
                        message_id=msg_data.message_id,
                        image_data=image_data,
                        generation_complete=True  # All restored messages are complete
                    )
                    
                    # Mount the message widget to the chat log
                    await chat_log.mount(message_widget)
                    
                    if i < 3:  # Log first few for debugging
                        logger.debug(f"Restored message {i+1}: {msg_data.role} - {msg_data.content[:50]}...")
                        
                except Exception as e:
                    logger.error(f"Error restoring message {i}: {e}")
            
            logger.info(f"Successfully restored {len(active_tab.messages)} messages")
            
            # Scroll to bottom to show latest messages
            chat_log.scroll_end(animate=False)
            
        except Exception as e:
            logger.error(f"Error in _restore_messages: {e}")
    
    def _save_non_tabbed_state(self) -> None:
        """Save state for non-tabbed chat interface."""
        try:
            # Create a single "default" tab to store the state
            default_tab = TabState(
                tab_id="default",
                title="Chat",
                is_active=True
            )
            
            # Try to find and save input text - be specific about chat input only
            input_selectors = [
                "#chat-input",  # Primary chat input ID
                "TextArea#chat-input",  # TextArea with chat-input ID
                ".chat-input",  # Class-based selector
                "#message-input"  # Alternative message input ID
            ]
            
            for selector in input_selectors:
                try:
                    input_widgets = self.chat_window.query(selector)
                    if input_widgets:
                        for widget in input_widgets:
                            # Make sure we're not saving system prompt or other TextAreas
                            if hasattr(widget, 'id') and widget.id:
                                widget_id = str(widget.id).lower()
                                # Skip if it's a system prompt or settings field
                                if any(x in widget_id for x in ['system', 'prompt', 'settings', 'config']):
                                    logger.debug(f"Skipping non-chat input: {widget.id}")
                                    continue
                            
                            if hasattr(widget, 'text'):
                                default_tab.input_text = widget.text
                                logger.info(f"Found chat input text in {selector}: '{widget.text[:50]}...'")
                                break
                        if default_tab.input_text:
                            break
                except Exception as e:
                    logger.debug(f"Could not query {selector}: {e}")
            
            # Save messages from chat log
            self._extract_and_save_messages(default_tab)
            
            self.chat_state.tabs = [default_tab]
            self.chat_state.active_tab_id = "default"
            self.chat_state.tab_order = ["default"]  # Fix validation issue
            
        except Exception as e:
            logger.error(f"Error saving non-tabbed state: {e}")
    
    def _save_direct_input_text(self) -> None:
        """Try to save input text directly from the chat input TextArea only."""
        try:
            # Be specific - only look for the chat input TextArea, not system prompt or other TextAreas
            chat_input = None
            
            # Try to find the specific chat input by ID first
            try:
                chat_input = self.chat_window.query_one("#chat-input", TextArea)
                logger.debug("Found chat input by #chat-input ID")
            except Exception:
                # If not found by ID, try other selectors but be careful
                pass
            
            if not chat_input:
                # Look for TextAreas but filter out system prompt and other non-chat inputs
                text_areas = self.chat_window.query("TextArea")
                logger.debug(f"Found {len(text_areas)} TextArea widgets total")
                
                for text_area in text_areas:
                    # Skip system prompt inputs and other non-chat TextAreas
                    if text_area.id and any(x in str(text_area.id).lower() for x in ['system', 'prompt', 'settings', 'config']):
                        logger.debug(f"Skipping non-chat TextArea: {text_area.id}")
                        continue
                    
                    # Look for chat-related IDs
                    if text_area.id and any(x in str(text_area.id).lower() for x in ['chat-input', 'message', 'input']):
                        chat_input = text_area
                        logger.debug(f"Found likely chat input: {text_area.id}")
                        break
            
            # Save the chat input text if found
            if chat_input and hasattr(chat_input, 'text') and chat_input.text:
                logger.info(f"Saving chat input (id={chat_input.id}): '{chat_input.text[:50]}...'")
                
                # If we have a tab, save to it
                if self.chat_state.tabs:
                    # Save to first/active tab
                    active_tab = self.chat_state.get_active_tab() or self.chat_state.tabs[0]
                    if not active_tab.input_text:  # Don't overwrite if already saved
                        active_tab.input_text = chat_input.text
                        logger.info(f"Saved chat input to tab {active_tab.tab_id}")
                else:
                    # Create a default tab if none exist
                    default_tab = TabState(
                        tab_id="default",
                        title="Chat",
                        input_text=chat_input.text,
                        is_active=True
                    )
                    self.chat_state.tabs = [default_tab]
                    self.chat_state.active_tab_id = "default"
                    logger.info("Created default tab with chat input content")
            else:
                logger.debug("No chat input text to save")
                        
        except Exception as e:
            logger.debug(f"Error in _save_direct_input_text: {e}")
    
    def _extract_and_save_messages(self, tab_state: TabState) -> None:
        """Extract messages from the chat log and save them to the tab state.
        
        Args:
            tab_state: The tab state to save messages to
        """
        try:
            # Import message widget classes
            from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            from textual.containers import VerticalScroll
            
            # Try to find the chat log container (it's a VerticalScroll)
            chat_log = None
            
            # First try the direct approach used in Chat_Window_Enhanced
            try:
                chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
                logger.debug("Found chat log via app_instance.query_one")
            except Exception:
                pass
            
            # If not found, try other selectors
            if not chat_log:
                log_selectors = [
                    "#chat-log",
                    ".chat-log",
                    "#chat-messages-container",
                    ".chat-messages"
                ]
            
            for selector in log_selectors:
                try:
                    containers = self.chat_window.query(selector)
                    if containers:
                        chat_log = containers.first()
                        logger.debug(f"Found chat log container with selector: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"Could not find chat log with {selector}: {e}")
            
            if not chat_log:
                logger.warning("Could not find chat log container to save messages")
                return
            
            # Extract messages from the chat log
            messages_found = 0
            tab_state.messages = []  # Clear existing messages
            
            # Find all message widgets - try different selectors
            try:
                # Try to find ChatMessageEnhanced widgets
                enhanced_messages = list(chat_log.query(ChatMessageEnhanced))
                
                # If no enhanced messages, try generic approach
                if not enhanced_messages:
                    # Look for any widgets with message-like attributes
                    all_widgets = list(chat_log.children)
                    enhanced_messages = [w for w in all_widgets 
                                       if hasattr(w, 'role') and hasattr(w, 'message_text')]
                
                logger.info(f"Found {len(enhanced_messages)} message widgets in chat log")
                
                for msg_widget in enhanced_messages:
                    try:
                        # Extract message data from widget
                        message_data = MessageData(
                            message_id=getattr(msg_widget, 'message_id_internal', f"msg_{messages_found}"),
                            role=getattr(msg_widget, 'role', 'unknown'),
                            content=getattr(msg_widget, 'message_text', ''),
                            timestamp=getattr(msg_widget, 'timestamp', None)
                        )
                        
                        # Save image data if present
                        if hasattr(msg_widget, 'image_data') and msg_widget.image_data:
                            message_data.metadata = {'image_data': msg_widget.image_data}
                        
                        tab_state.messages.append(message_data)
                        messages_found += 1
                        
                        # Log first few messages for debugging
                        if messages_found <= 3:
                            logger.debug(f"Saved message {messages_found}: role={message_data.role}, content={message_data.content[:50]}...")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting message data from widget: {e}")
                        
                logger.info(f"Successfully saved {messages_found} messages to tab state")
                
            except Exception as e:
                logger.error(f"Error querying for message widgets: {e}")
                
        except Exception as e:
            logger.error(f"Error in _extract_and_save_messages: {e}")
    
    def on_screen_suspend(self) -> None:
        """Called when navigating away from this screen."""
        logger.debug("Chat screen suspending - saving state")
        self.save_state()
        # Note: BaseAppScreen doesn't have on_screen_suspend, so no super() call
    
    def on_screen_resume(self) -> None:
        """Called when returning to this screen."""
        logger.debug("Chat screen resuming")
        # Note: BaseAppScreen doesn't have on_screen_resume, so no super() call
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button events at the screen level.
        This ensures buttons work properly with screen-based navigation.
        """
        button_id = event.button.id
        
        # Log for debugging
        logger.info(f"ChatScreen on_button_pressed called with button: {button_id}")
        
        # Sidebar toggle is handled in ChatWindowEnhanced via @on decorator
        
        # Buttons that are handled by @on decorators in ChatWindowEnhanced
        # These should NOT be delegated to avoid double handling
        handled_by_decorators = [
            "send-stop-chat",
            "attach-image",
            "chat-mic"
            # Removed sidebar toggles from here since they're handled above
        ]
        
        if button_id in handled_by_decorators:
            # These are already handled by @on decorators, just stop propagation
            event.stop()
            return
            
        # For remaining buttons that need legacy handling, delegate to ChatWindowEnhanced
        if self.chat_window:
            # The chat window knows how to handle its own buttons
            await self.chat_window.on_button_pressed(event)
            event.stop()  # Prevent bubbling to app level
    
    
    async def _run_diagnostic(self) -> None:
        """Run diagnostic tool on the chat widget structure."""
        try:
            if not self.chat_window:
                return
                
            logger.info("Running chat widget structure diagnostics...")
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(self.chat_window, max_depth=5)
            
            # Log key findings
            logger.info(f"Diagnostic: {report['chat_structure']['type']} interface detected")
            logger.info(f"Found {report['text_areas']['count']} TextArea widgets")
            logger.info(f"Found {report['containers']['chat_containers']} chat containers")
            logger.info(f"Found {report['containers']['tab_containers']} tab containers")
            
            # Log any input widgets found
            if report['input_widgets']:
                for widget in report['input_widgets']:
                    logger.info(f"Input widget: {widget['id']} at {widget['path']}")
            
            # Store report for potential debugging
            self._diagnostic_report = report
            
            # Also log all sidebar-related widgets for debugging
            self._log_sidebar_widgets()
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}", exc_info=True)
    
    def _log_sidebar_widgets(self) -> None:
        """Log all sidebar widgets for debugging state preservation."""
        try:
            logger.info("=== Sidebar Widget IDs ===")
            
            # Find all TextAreas
            text_areas = self.chat_window.query("TextArea")
            for ta in text_areas:
                if ta.id:
                    logger.info(f"TextArea ID: {ta.id}, Has text: {bool(getattr(ta, 'text', None))}")
            
            # Find all Inputs
            inputs = self.chat_window.query("Input")
            for inp in inputs:
                if inp.id:
                    logger.info(f"Input ID: {inp.id}, Value: {getattr(inp, 'value', 'N/A')}")
            
            logger.info("=========================")
        except Exception as e:
            logger.debug(f"Error logging sidebar widgets: {e}")
    
    def watch_sidebar_state(self, new_state: dict) -> None:
        """Auto-save when sidebar state changes."""
        self._save_sidebar_state()
    
    def _load_sidebar_state(self) -> None:
        """Load sidebar state from config file."""
        config_path = Path.home() / ".config" / "tldw_cli" / "ui_state.toml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = toml.load(f)
                    sidebar_data = data.get("sidebar", {})
                    
                    # Load collapsible states into UIState
                    self.ui_state.collapsible_states = sidebar_data.get("collapsible_states", {})
                    self.ui_state.sidebar_search_query = sidebar_data.get("search_query", "")
                    self.ui_state.last_active_section = sidebar_data.get("last_active_section", None)
                    
                    # Update reactive property
                    self.sidebar_state = dict(self.ui_state.collapsible_states)
                    
                    logger.debug(f"Loaded sidebar state with {len(self.ui_state.collapsible_states)} collapsibles")
        except Exception as e:
            logger.error(f"Failed to load sidebar state: {e}")
            self.sidebar_state = {}
    
    def _save_sidebar_state(self) -> None:
        """Save sidebar state to config file."""
        config_path = Path.home() / ".config" / "tldw_cli" / "ui_state.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load existing config or create new
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = toml.load(f)
            else:
                data = {}
            
            # Update sidebar section
            data["sidebar"] = {
                "collapsible_states": dict(self.ui_state.collapsible_states),
                "search_query": self.ui_state.sidebar_search_query,
                "last_active_section": self.ui_state.last_active_section
            }
            
            # Save back to file
            with open(config_path, 'w') as f:
                toml.dump(data, f)
                
            logger.debug(f"Saved sidebar state with {len(self.ui_state.collapsible_states)} collapsibles")
        except Exception as e:
            logger.error(f"Failed to save sidebar state: {e}")
    
    def _restore_collapsible_states(self) -> None:
        """Restore collapsible states from saved state."""
        if not self.ui_state.collapsible_states:
            logger.debug("No collapsible states to restore")
            return
            
        try:
            # Find all collapsibles in the sidebar
            collapsibles = self.query(Collapsible)
            restored_count = 0
            
            for collapsible in collapsibles:
                if collapsible.id and collapsible.id in self.ui_state.collapsible_states:
                    collapsed_state = self.ui_state.collapsible_states[collapsible.id]
                    collapsible.collapsed = collapsed_state
                    restored_count += 1
                    logger.debug(f"Restored {collapsible.id}: collapsed={collapsed_state}")
            
            logger.info(f"Restored {restored_count} collapsible states")
        except Exception as e:
            logger.error(f"Error restoring collapsible states: {e}")
    
    @on(Collapsible.Toggled)
    def handle_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Save collapsible state when toggled."""
        try:
            collapsible_id = event.collapsible.id
            if collapsible_id:
                # Update UIState
                self.ui_state.set_collapsible_state(collapsible_id, event.collapsible.collapsed)
                
                # Update reactive property to trigger watcher
                new_state = dict(self.ui_state.collapsible_states)
                self.sidebar_state = new_state
                
                logger.debug(f"Toggled {collapsible_id}: collapsed={event.collapsible.collapsed}")
        except Exception as e:
            logger.error(f"Error handling collapsible toggle: {e}")
    
    @on(Button.Pressed, "#chat-expand-all")
    def handle_expand_all(self, event: Button.Pressed) -> None:
        """Expand all collapsible sections."""
        try:
            collapsibles = self.query(Collapsible)
            expanded_count = 0
            
            for collapsible in collapsibles:
                if collapsible.collapsed:
                    collapsible.collapsed = False
                    expanded_count += 1
                    if collapsible.id:
                        self.ui_state.set_collapsible_state(collapsible.id, False)
            
            # Update reactive property
            self.sidebar_state = dict(self.ui_state.collapsible_states)
            
            logger.info(f"Expanded {expanded_count} sections")
            self.notify(f"Expanded {expanded_count} sections", severity="information")
        except Exception as e:
            logger.error(f"Error expanding all sections: {e}")
    
    @on(Button.Pressed, "#chat-collapse-all")
    def handle_collapse_all(self, event: Button.Pressed) -> None:
        """Collapse all non-priority collapsible sections."""
        try:
            collapsibles = self.query(Collapsible)
            collapsed_count = 0
            
            for collapsible in collapsibles:
                # Keep priority sections open
                if "priority-high" not in collapsible.classes and not collapsible.collapsed:
                    collapsible.collapsed = True
                    collapsed_count += 1
                    if collapsible.id:
                        self.ui_state.set_collapsible_state(collapsible.id, True)
            
            # Update reactive property
            self.sidebar_state = dict(self.ui_state.collapsible_states)
            
            logger.info(f"Collapsed {collapsed_count} non-essential sections")
            self.notify(f"Collapsed {collapsed_count} sections", severity="information")
        except Exception as e:
            logger.error(f"Error collapsing sections: {e}")
    
    @on(Button.Pressed, "#chat-reset-settings")
    def handle_reset_settings(self, event: Button.Pressed) -> None:
        """Reset settings to defaults."""
        try:
            # Clear all saved collapsible states
            self.ui_state.collapsible_states.clear()
            self.sidebar_state = {}
            
            # Reset collapsibles to default states
            collapsibles = self.query(Collapsible)
            for collapsible in collapsibles:
                # Default state: priority sections open, others closed
                if "priority-high" in collapsible.classes:
                    collapsible.collapsed = False
                else:
                    collapsible.collapsed = True
            
            self._save_sidebar_state()
            logger.info("Reset sidebar to default state")
            self.notify("Settings reset to defaults", severity="success")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")