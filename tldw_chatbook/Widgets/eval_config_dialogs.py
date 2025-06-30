# eval_config_dialogs.py
# Description: Configuration dialogs for evaluation setup
#
"""
Evaluation Configuration Dialogs
-------------------------------

Provides modal dialogs for configuring various aspects of evaluations:
- Model configuration (provider, model ID, parameters)
- Task creation and editing
- Run configuration and parameters
- Template selection and customization
"""

from typing import Dict, List, Any, Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Input, Select, TextArea, Checkbox, 
    Static, ListView, ListItem, Collapsible
)
from textual.validation import Number, Length
from loguru import logger

class ModelConfigDialog(ModalScreen):
    """Dialog for configuring LLM models."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 existing_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.existing_config = existing_config or {}
        self.model_config = {}
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog"):
            yield Label("Model Configuration", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("Name:")
                yield Input(
                    placeholder="e.g., GPT-4 Turbo",
                    value=self.existing_config.get('name', ''),
                    id="model-name"
                )
                
                yield Label("Provider:")
                yield Select(
                    [
                        ("OpenAI", "openai"),
                        ("Anthropic", "anthropic"),
                        ("Cohere", "cohere"),
                        ("Groq", "groq"),
                        ("OpenRouter", "openrouter"),
                        ("HuggingFace", "huggingface"),
                        ("DeepSeek", "deepseek")
                    ],
                    value=self.existing_config.get('provider', 'openai'),
                    id="model-provider"
                )
                
                yield Label("Model ID:")
                yield Input(
                    placeholder="e.g., gpt-4-turbo-preview",
                    value=self.existing_config.get('model_id', ''),
                    id="model-id"
                )
                
                yield Label("API Key:")
                yield Input(
                    placeholder="Leave empty to use config/env",
                    password=True,
                    value=self.existing_config.get('api_key', ''),
                    id="api-key"
                )
            
            with Collapsible(title="Advanced Parameters", collapsed=True):
                with Grid(classes="advanced-grid"):
                    yield Label("Temperature:")
                    yield Input(
                        placeholder="0.0",
                        value=str(self.existing_config.get('temperature', 0.0)),
                        validators=[Number(minimum=0.0, maximum=2.0)],
                        id="temperature"
                    )
                    
                    yield Label("Max Tokens:")
                    yield Input(
                        placeholder="1024",
                        value=str(self.existing_config.get('max_tokens', 1024)),
                        validators=[Number(minimum=1, maximum=8192)],
                        id="max-tokens"
                    )
                    
                    yield Label("Timeout (seconds):")
                    yield Input(
                        placeholder="30",
                        value=str(self.existing_config.get('timeout', 30)),
                        validators=[Number(minimum=1, maximum=300)],
                        id="timeout"
                    )
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Test Connection", id="test-button")
                yield Button("Save", id="save-button", variant="primary")
    
    @on(Button.Pressed, "#test-button")
    async def handle_test_connection(self):
        """Test the model configuration."""
        config = self._collect_config()
        if not config.get('name') or not config.get('model_id'):
            self._show_error("Please fill in name and model ID")
            return
        
        # Update button to show testing state
        try:
            test_button = self.query_one("#test-button")
            test_button.label = "Testing..."
            test_button.disabled = True
        except:
            pass
        
        try:
            # Import here to avoid circular imports
            from tldw_chatbook.Evals import LLMInterface
            
            # Create interface and test
            interface = LLMInterface(
                provider_name=config['provider'],
                model_id=config['model_id'],
                config=config
            )
            
            # Simple test prompt
            result = await interface.generate("Hello, world!")
            
            if result and len(result.strip()) > 0:
                self._show_success("Connection successful!")
            else:
                self._show_error("Connection failed: Empty response")
                
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            self._show_error(f"Connection failed: {str(e)}")
        
        finally:
            # Reset button state
            try:
                test_button = self.query_one("#test-button")
                test_button.label = "Test Connection"
                test_button.disabled = False
            except:
                pass
    
    @on(Button.Pressed, "#save-button")
    def handle_save(self):
        """Save the model configuration."""
        config = self._collect_config()
        
        # Validate required fields
        if not config.get('name') or not config.get('model_id'):
            self._show_error("Name and Model ID are required")
            return
        
        if self.callback:
            self.callback(config)
        self.dismiss(config)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _collect_config(self) -> Dict[str, Any]:
        """Collect configuration from form fields."""
        config = {}
        
        try:
            config['name'] = self.query_one("#model-name").value.strip()
            config['provider'] = self.query_one("#model-provider").value
            config['model_id'] = self.query_one("#model-id").value.strip()
            
            api_key = self.query_one("#api-key").value.strip()
            if api_key:
                config['api_key'] = api_key
            
            # Advanced parameters
            temp_input = self.query_one("#temperature")
            if temp_input.value:
                config['temperature'] = float(temp_input.value)
            
            tokens_input = self.query_one("#max-tokens")
            if tokens_input.value:
                config['max_tokens'] = int(tokens_input.value)
            
            timeout_input = self.query_one("#timeout")
            if timeout_input.value:
                config['timeout'] = int(timeout_input.value)
                
        except Exception as e:
            logger.error(f"Error collecting config: {e}")
        
        return config
    
    def _show_error(self, message: str):
        """Show error message to user."""
        # This would be replaced with proper notification system
        logger.error(message)
        self.app.notify(message, severity="error")
    
    def _show_success(self, message: str):
        """Show success message to user."""
        logger.info(message)
        self.app.notify(message, severity="information")

class TaskConfigDialog(ModalScreen):
    """Dialog for configuring evaluation tasks."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 existing_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.existing_config = existing_config or {}
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog large"):
            yield Label("Task Configuration", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("Task Name:")
                yield Input(
                    placeholder="e.g., Custom Math Evaluation",
                    value=self.existing_config.get('name', ''),
                    validators=[Length(minimum=1)],
                    id="task-name"
                )
                
                yield Label("Task Type:")
                yield Select(
                    [
                        ("Question & Answer", "question_answer"),
                        ("Multiple Choice", "classification"),
                        ("Text Generation", "generation"),
                        ("Log Probability", "logprob")
                    ],
                    value=self.existing_config.get('task_type', 'question_answer'),
                    id="task-type"
                )
                
                yield Label("Dataset:")
                yield Input(
                    placeholder="Dataset name or path",
                    value=self.existing_config.get('dataset_name', ''),
                    id="dataset-name"
                )
                
                yield Label("Description:")
                yield TextArea(
                    text=self.existing_config.get('description', ''),
                    id="task-description"
                )
            
            with Collapsible(title="Generation Parameters", collapsed=True):
                with Grid(classes="advanced-grid"):
                    yield Label("Max Samples:")
                    yield Input(
                        placeholder="100",
                        value=str(self.existing_config.get('max_samples', 100)),
                        validators=[Number(minimum=1)],
                        id="max-samples"
                    )
                    
                    yield Label("Temperature:")
                    yield Input(
                        placeholder="0.0",
                        value=str(self.existing_config.get('temperature', 0.0)),
                        validators=[Number(minimum=0.0, maximum=2.0)],
                        id="gen-temperature"
                    )
                    
                    yield Label("Max Tokens:")
                    yield Input(
                        placeholder="512",
                        value=str(self.existing_config.get('max_tokens', 512)),
                        validators=[Number(minimum=1)],
                        id="gen-max-tokens"
                    )
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Save", id="save-button", variant="primary")
    
    @on(Button.Pressed, "#save-button")
    def handle_save(self):
        """Save the task configuration."""
        config = self._collect_config()
        
        # Validate required fields
        if not config.get('name') or not config.get('dataset_name'):
            self._show_error("Name and Dataset are required")
            return
        
        if self.callback:
            self.callback(config)
        self.dismiss(config)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _collect_config(self) -> Dict[str, Any]:
        """Collect configuration from form fields."""
        config = {}
        
        try:
            config['name'] = self.query_one("#task-name").value.strip()
            config['task_type'] = self.query_one("#task-type").value
            config['dataset_name'] = self.query_one("#dataset-name").value.strip()
            config['description'] = self.query_one("#task-description").text.strip()
            
            # Generation parameters
            max_samples = self.query_one("#max-samples").value
            if max_samples:
                config['max_samples'] = int(max_samples)
            
            temp = self.query_one("#gen-temperature").value
            if temp:
                config['temperature'] = float(temp)
            
            max_tokens = self.query_one("#gen-max-tokens").value
            if max_tokens:
                config['max_tokens'] = int(max_tokens)
                
        except Exception as e:
            logger.error(f"Error collecting task config: {e}")
        
        return config
    
    def _show_error(self, message: str):
        """Show error message to user."""
        logger.error(message)
        self.app.notify(message, severity="error")

class RunConfigDialog(ModalScreen):
    """Dialog for configuring evaluation runs."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 available_tasks: Optional[List[Dict[str, Any]]] = None,
                 available_models: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.available_tasks = available_tasks or []
        self.available_models = available_models or []
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog"):
            yield Label("Evaluation Run Configuration", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("Run Name:")
                yield Input(
                    placeholder="e.g., GPT-4 Math Benchmark",
                    validators=[Length(minimum=1)],
                    id="run-name"
                )
                
                yield Label("Task:")
                yield Select(
                    [(task.get('name', 'Unknown'), task.get('id', '')) 
                     for task in self.available_tasks] or [("No tasks available", "")],
                    id="run-task"
                )
                
                yield Label("Model:")
                yield Select(
                    [(model.get('name', 'Unknown'), model.get('id', '')) 
                     for model in self.available_models] or [("No models available", "")],
                    id="run-model"
                )
                
                yield Label("Max Samples:")
                yield Input(
                    placeholder="100 (0 for all)",
                    value="100",
                    validators=[Number(minimum=0)],
                    id="max-samples"
                )
            
            with Collapsible(title="Advanced Options", collapsed=True):
                yield Checkbox("Enable progress tracking", value=True, id="progress-tracking")
                yield Checkbox("Save detailed results", value=True, id="save-detailed")
                yield Checkbox("Export results automatically", value=False, id="auto-export")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Start Evaluation", id="start-button", variant="primary")
    
    @on(Button.Pressed, "#start-button")
    def handle_start(self):
        """Start the evaluation run."""
        config = self._collect_config()
        
        # Validate required fields
        if not config.get('name') or not config.get('task_id') or not config.get('model_id'):
            self._show_error("Name, Task, and Model are required")
            return
        
        if self.callback:
            self.callback(config)
        self.dismiss(config)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _collect_config(self) -> Dict[str, Any]:
        """Collect configuration from form fields."""
        config = {}
        
        try:
            config['name'] = self.query_one("#run-name").value.strip()
            config['task_id'] = self.query_one("#run-task").value
            config['model_id'] = self.query_one("#run-model").value
            
            max_samples = self.query_one("#max-samples").value
            if max_samples and int(max_samples) > 0:
                config['max_samples'] = int(max_samples)
            
            config['progress_tracking'] = self.query_one("#progress-tracking").value
            config['save_detailed'] = self.query_one("#save-detailed").value
            config['auto_export'] = self.query_one("#auto-export").value
            
        except Exception as e:
            logger.error(f"Error collecting run config: {e}")
        
        return config
    
    def _show_error(self, message: str):
        """Show error message to user."""
        logger.error(message)
        self.app.notify(message, severity="error")