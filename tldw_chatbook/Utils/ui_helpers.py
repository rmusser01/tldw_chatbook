"""
UI Helpers - Centralized UI manipulation and helper functions.

This module provides common UI operations to reduce code duplication in app.py.
"""

from typing import Optional, List, TYPE_CHECKING
from textual.widgets import Input, TextArea, Select, ListView
from textual.css.query import QueryError
from loguru import logger

if TYPE_CHECKING:
    from textual.app import App


class UIHelpers:
    """Collection of UI helper methods for the application."""
    
    @staticmethod
    def clear_prompt_editor_fields(app: 'App') -> None:
        """
        Clear all prompt editor fields in the center pane.
        
        Args:
            app: The Textual app instance
        """
        field_configs = [
            ("#ccp-editor-prompt-name-input", Input, "value", ""),
            ("#ccp-editor-prompt-author-input", Input, "value", ""),
            ("#ccp-editor-prompt-description-textarea", TextArea, "text", ""),
            ("#ccp-editor-prompt-system-textarea", TextArea, "text", ""),
            ("#ccp-editor-prompt-user-textarea", TextArea, "text", ""),
            ("#ccp-editor-prompt-keywords-textarea", TextArea, "text", "")
        ]
        
        try:
            for field_id, widget_type, attr_name, default_value in field_configs:
                widget = app.query_one(field_id, widget_type)
                setattr(widget, attr_name, default_value)
            logger.debug("Cleared prompt editor fields in center pane.")
        except QueryError as e:
            logger.error(f"Error clearing prompt editor fields in center pane: {e}")
    
    @staticmethod
    def update_model_select(
        app: 'App', 
        id_prefix: str, 
        models: List[str],
        preserve_selection: bool = True
    ) -> None:
        """
        Update a model select widget with new options.
        
        Args:
            app: The Textual app instance
            id_prefix: The prefix for the select widget ID (e.g., 'chat', 'coding')
            models: List of model names to populate the select with
            preserve_selection: Whether to preserve the current selection if still valid
        """
        model_select_id = f"#{id_prefix}-api-model"
        
        try:
            model_select = app.query_one(model_select_id, Select)
            new_model_options = [(model, model) for model in models]
            
            # Store current value if preserve_selection is True
            current_value = model_select.value if preserve_selection else None
            
            # Update options (this might clear the value)
            model_select.set_options(new_model_options)
            
            # Restore or set appropriate value
            if current_value and current_value in models:
                model_select.value = current_value
            elif models:
                model_select.value = models[0]  # Default to first model
            else:
                model_select.value = Select.BLANK  # No models available
            
            # Update prompt text
            model_select.prompt = "Select Model..." if models else "No models available"
            
        except QueryError:
            logger.error(f"Cannot find model select '{model_select_id}'")
        except Exception as e:
            logger.error(f"Error updating model select: {e}")
    
    @staticmethod
    def update_rag_expansion_model_select(app: 'App', models: List[str]) -> None:
        """
        Update the RAG expansion model select widget.
        
        Args:
            app: The Textual app instance
            models: List of model names to populate the select with
        """
        model_select_id = "#chat-rag-expansion-llm-model"
        
        try:
            model_select = app.query_one(model_select_id, Select)
            new_model_options = [(model, model) for model in models]
            
            # Store current value
            current_value = model_select.value
            model_select.set_options(new_model_options)
            
            # Restore value if still valid
            if current_value in models:
                model_select.value = current_value
            elif models:
                model_select.value = models[0]
            else:
                model_select.value = Select.BLANK
            
            model_select.prompt = "Select Model..." if models else "No models available"
            
        except QueryError:
            logger.error(f"Cannot find RAG expansion model select '{model_select_id}'")
        except Exception as e:
            logger.error(f"Error setting RAG expansion model options: {e}")
    
    @staticmethod
    def clear_chat_sidebar_prompt_display(app: 'App') -> None:
        """
        Clear the prompt display areas in the chat sidebar.
        
        Args:
            app: The Textual app instance
        """
        logger.debug("Clearing chat sidebar prompt display areas.")
        
        # Clear the reactive properties (if they exist on the app)
        if hasattr(app, 'chat_sidebar_selected_prompt_id'):
            app.chat_sidebar_selected_prompt_id = None
        if hasattr(app, 'chat_sidebar_selected_prompt_system'):
            app.chat_sidebar_selected_prompt_system = None
        if hasattr(app, 'chat_sidebar_selected_prompt_user'):
            app.chat_sidebar_selected_prompt_user = None
        
        # Clear the prompts listview
        try:
            listview = app.query_one("#chat-sidebar-prompts-listview", ListView)
            listview.clear()
        except QueryError:
            pass  # If not found, it's fine
    
    @staticmethod
    def update_token_count_in_footer(app: 'App', token_count: Optional[int] = None) -> None:
        """
        Update the token count display in the footer widget.
        
        Args:
            app: The Textual app instance
            token_count: The token count to display, or None to clear
        """
        if not hasattr(app, '_db_size_status_widget') or not app._db_size_status_widget:
            return
        
        try:
            if token_count is None:
                app._db_size_status_widget.update_token_count("")
            else:
                app._db_size_status_widget.update_token_count(str(token_count))
        except Exception as e:
            logger.error(f"Error updating token count in footer: {e}")
    
    @staticmethod
    def get_widget_safely(app: 'App', widget_id: str, widget_type: type) -> Optional[object]:
        """
        Safely get a widget by ID, returning None if not found.
        
        Args:
            app: The Textual app instance
            widget_id: The ID of the widget to find
            widget_type: The expected type of the widget
            
        Returns:
            The widget if found, None otherwise
        """
        try:
            return app.query_one(widget_id, widget_type)
        except QueryError:
            return None
    
    @staticmethod
    def set_widget_value(widget: object, value: any) -> bool:
        """
        Set the value/text of a widget based on its type.
        
        Args:
            widget: The widget to update
            value: The value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(widget, Input):
                widget.value = value
            elif isinstance(widget, TextArea):
                widget.text = value
            elif isinstance(widget, Select):
                widget.value = value
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Error setting widget value: {e}")
            return False