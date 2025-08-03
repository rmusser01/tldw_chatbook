"""
AI Generation Worker Handler - Handles AI content generation worker state changes.

This module manages state changes for AI generation workers used in character
creation, including generation of descriptions, personalities, scenarios,
first messages, and system prompts.
"""

from typing import TYPE_CHECKING, Optional, Dict, List
from textual.worker import Worker, WorkerState
from textual.widgets import Button, TextArea
from textual.css.query import QueryError

from .base_handler import BaseWorkerHandler

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class AIGenerationHandler(BaseWorkerHandler):
    """Handles AI generation worker state changes."""
    
    # Mapping of worker names to their UI elements
    GENERATION_CONFIGS: Dict[str, Dict[str, str]] = {
        "ai_generate_description": {
            "textarea_id": "ccp-editor-char-description-textarea",
            "button_id": "ccp-generate-description-button",
            "field_name": "description",
        },
        "ai_generate_personality": {
            "textarea_id": "ccp-editor-char-personality-textarea",
            "button_id": "ccp-generate-personality-button",
            "field_name": "personality",
        },
        "ai_generate_scenario": {
            "textarea_id": "ccp-editor-char-scenario-textarea",
            "button_id": "ccp-generate-scenario-button",
            "field_name": "scenario",
        },
        "ai_generate_first_message": {
            "textarea_id": "ccp-editor-char-first-message-textarea",
            "button_id": "ccp-generate-first-message-button",
            "field_name": "first_message",
        },
        "ai_generate_system_prompt": {
            "textarea_id": "ccp-editor-char-system-prompt-textarea",
            "button_id": "ccp-generate-system-prompt-button",
            "field_name": "system_prompt",
        },
    }
    
    def can_handle(self, worker_name: str, worker_group: Optional[str] = None) -> bool:
        """
        Check if this handler can process the given worker.
        
        Args:
            worker_name: The name attribute of the worker
            worker_group: The group attribute of the worker
            
        Returns:
            True if this is an AI generation worker
        """
        return worker_group == "ai_generation"
    
    async def handle(self, event: Worker.StateChanged) -> None:
        """
        Handle the AI generation worker state change event.
        
        Args:
            event: The worker state changed event
        """
        worker_info = self.get_worker_info(event)
        self.log_state_change(worker_info, "AI Generation: ")
        
        if worker_info['state'] == WorkerState.SUCCESS:
            await self._handle_success_state(event, worker_info)
            
        elif worker_info['state'] == WorkerState.ERROR:
            await self._handle_error_state(event, worker_info)
            
        elif worker_info['state'] == WorkerState.PENDING:
            await self._handle_pending_state(worker_info)
    
    async def _handle_success_state(self, event: Worker.StateChanged, 
                                   worker_info: dict) -> None:
        """Handle the SUCCESS state for AI generation workers."""
        self.logger.info(f"AI generation worker '{worker_info['name']}' completed successfully")
        
        result = event.worker.result
        if not result or not isinstance(result, dict) or 'choices' not in result:
            self.logger.warning(f"AI generation worker returned invalid result: {result}")
            await self._re_enable_button(worker_info['name'])
            return
        
        try:
            content = result['choices'][0]['message']['content']
            
            if worker_info['name'] == "ai_generate_all":
                await self._handle_generate_all(content)
            else:
                await self._handle_single_generation(worker_info['name'], content)
                
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error extracting content from AI response: {e}")
            self.app.notify(
                "Failed to extract content from AI response",
                severity="error"
            )
            await self._re_enable_button(worker_info['name'])
            
        except Exception as e:
            self.logger.error(f"Error handling AI generation success: {e}")
            self.app.notify(
                f"Error updating UI: {str(e)[:100]}",
                severity="error"
            )
            await self._re_enable_button(worker_info['name'])
    
    async def _handle_single_generation(self, worker_name: str, content: str) -> None:
        """
        Handle generation for a single field.
        
        Args:
            worker_name: Name of the worker
            content: Generated content
        """
        config = self.GENERATION_CONFIGS.get(worker_name)
        if not config:
            self.logger.error(f"No configuration found for worker: {worker_name}")
            return
        
        try:
            # Update the textarea
            text_area = self.app.query_one(f"#{config['textarea_id']}", TextArea)
            text_area.text = content
            
            # Re-enable the button
            button = self.app.query_one(f"#{config['button_id']}", Button)
            button.disabled = False
            
            self.logger.info(f"Updated {config['field_name']} field with generated content")
            
        except QueryError as e:
            self.logger.error(f"Failed to update UI for {worker_name}: {e}")
    
    async def _handle_generate_all(self, content: str) -> None:
        """
        Handle generation for all fields at once.
        
        Args:
            content: Generated content containing all sections
        """
        self.logger.info("Processing comprehensive character generation")
        
        # Parse the content into sections
        sections = self._parse_comprehensive_response(content)
        
        # Update each field
        for worker_name, config in self.GENERATION_CONFIGS.items():
            if worker_name == "ai_generate_all":
                continue
                
            field_name = config['field_name']
            if field_name in sections and sections[field_name]:
                try:
                    text_area = self.app.query_one(f"#{config['textarea_id']}", TextArea)
                    text_area.text = sections[field_name]
                    self.logger.debug(f"Updated {field_name} from comprehensive generation")
                except QueryError as e:
                    self.logger.error(f"Failed to update {field_name}: {e}")
        
        # Re-enable the generate all button
        try:
            button = self.app.query_one("#ccp-generate-all-button", Button)
            button.disabled = False
        except QueryError:
            self.logger.error("Failed to re-enable generate all button")
        
        self.app.notify("Character fields generated successfully", severity="information")
    
    def _parse_comprehensive_response(self, content: str) -> Dict[str, str]:
        """
        Parse the comprehensive AI response into sections.
        
        Args:
            content: The full AI response containing all sections
            
        Returns:
            Dictionary mapping field names to their content
        """
        sections = {
            'description': '',
            'personality': '',
            'scenario': '',
            'first_message': '',
            'system_prompt': ''
        }
        
        current_section = None
        current_content: List[str] = []
        
        for line in content.split('\n'):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Remove markdown formatting from potential headers
            line_clean = line_stripped.replace('**', '').replace('##', '').strip()
            line_clean_lower = line_clean.lower()
            
            # Check for section headers
            section_found = False
            for section_name in sections.keys():
                if section_name in line_clean_lower and any(marker in line_stripped 
                                                           for marker in [':', '**', '##']):
                    # Save previous section if exists
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            # If not a section header and we have a current section, add the line
            if not section_found and current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    async def _handle_error_state(self, event: Worker.StateChanged, 
                                 worker_info: dict) -> None:
        """Handle the ERROR state for AI generation workers."""
        error_msg = str(event.worker.error)[:200] if event.worker.error else "Unknown error"
        self.logger.error(f"AI generation worker '{worker_info['name']}' failed: {error_msg}")
        
        self.app.notify(
            f"AI generation failed: {error_msg}",
            title="Generation Error",
            severity="error"
        )
        
        await self._re_enable_button(worker_info['name'])
    
    async def _handle_pending_state(self, worker_info: dict) -> None:
        """Handle the PENDING state for AI generation workers."""
        self.logger.debug(f"AI generation worker '{worker_info['name']}' is pending")
        
        # Disable the corresponding button
        if worker_info['name'] == "ai_generate_all":
            await self.update_button_state("ccp-generate-all-button", disabled=True)
        else:
            config = self.GENERATION_CONFIGS.get(worker_info['name'])
            if config:
                await self.update_button_state(config['button_id'], disabled=True)
    
    async def _re_enable_button(self, worker_name: str) -> None:
        """
        Re-enable the button for a specific worker.
        
        Args:
            worker_name: Name of the worker
        """
        if worker_name == "ai_generate_all":
            await self.update_button_state("ccp-generate-all-button", disabled=False)
        else:
            config = self.GENERATION_CONFIGS.get(worker_name)
            if config:
                await self.update_button_state(config['button_id'], disabled=False)