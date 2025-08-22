# swarmui_widget.py
# Description: SwarmUI image generation widget for chat sidebar

from typing import Optional, Dict, Any, List
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, TextArea, Select, Input, Label, LoadingIndicator, Collapsible
from textual.reactive import reactive
from textual.message import Message
from textual import work
from loguru import logger

from ...Media_Creation import ImageGenerationService, get_all_categories, get_templates_by_category, BUILTIN_TEMPLATES
from ...config import load_settings


class ImageGenerationMessage(Message):
    """Message sent when image generation completes."""
    
    def __init__(self, success: bool, images: List[str], error: Optional[str] = None):
        super().__init__()
        self.success = success
        self.images = images
        self.error = error


class SwarmUIWidget(Container):
    """Widget for SwarmUI image generation in chat sidebar."""
    
    # Reactive properties
    is_generating = reactive(False, layout=False)
    server_status = reactive("unknown", layout=False)  # "online", "offline", "unknown"
    current_image = reactive(None, layout=False)
    
    def __init__(self, **kwargs):
        """Initialize the SwarmUI widget."""
        super().__init__(**kwargs)
        self.service = ImageGenerationService()
        self.current_models: List[str] = []
        self.last_result = None
        
    def compose(self) -> ComposeResult:
        """Compose the widget UI."""
        with Container(classes="swarmui-widget"):
            # Status indicator
            with Horizontal(classes="swarmui-status"):
                yield Static("🎨 Image Generation", classes="swarmui-title")
                yield Static("●", id="status-indicator", classes="status-unknown")
            
            # Template selector
            yield Label("Template:", classes="swarmui-label")
            
            categories = get_all_categories()
            template_options = [("Custom", "custom")]
            
            for category in categories:
                templates = get_templates_by_category(category)
                for template in templates:
                    template_options.append((f"{category}: {template.name}", template.id))
            
            yield Select(
                options=template_options,
                value="custom",
                id="template-select",
                allow_blank=False
            )
            
            # Prompt input
            yield Label("Prompt:", classes="swarmui-label")
            yield TextArea(
                id="prompt-input",
                classes="swarmui-textarea"
            )
            
            # Context buttons
            with Horizontal(classes="context-buttons"):
                yield Button(
                    "Use Last Message",
                    id="use-last-message",
                    variant="default",
                    classes="context-button"
                )
                yield Button(
                    "Clear",
                    id="clear-prompt",
                    variant="default",
                    classes="context-button"
                )
            
            # Negative prompt
            yield Label("Negative Prompt:", classes="swarmui-label")
            yield TextArea(
                "blurry, low quality, bad anatomy",
                id="negative-prompt-input",
                classes="swarmui-textarea-small"
            )
            
            # Model selector
            yield Label("Model:", classes="swarmui-label")
            yield Select(
                options=[("Default", "default")],
                value="default",
                id="model-select",
                allow_blank=False
            )
            
            # Size selector
            yield Label("Size:", classes="swarmui-label")
            yield Select(
                options=[
                    ("Square (1024x1024)", "1024x1024"),
                    ("Square (768x768)", "768x768"),
                    ("Square (512x512)", "512x512"),
                    ("Portrait (768x1024)", "768x1024"),
                    ("Landscape (1024x768)", "1024x768"),
                    ("Wide (1344x768)", "1344x768"),
                    ("Tall (768x1344)", "768x1344"),
                ],
                value="1024x1024",
                id="size-select",
                allow_blank=False
            )
            
            # Advanced settings (collapsible)
            with Collapsible(title="Advanced Settings", collapsed=True):
                # Steps
                yield Label("Steps (Quality):", classes="swarmui-label")
                yield Input(
                    value="20",
                    id="steps-input",
                    type="integer",
                    validators=[],
                    classes="swarmui-input"
                )
                
                # CFG Scale
                yield Label("CFG Scale (Prompt Strength):", classes="swarmui-label")
                yield Input(
                    value="7.0",
                    id="cfg-input",
                    type="number",
                    validators=[],
                    classes="swarmui-input"
                )
                
                # Seed
                yield Label("Seed (-1 for random):", classes="swarmui-label")
                yield Input(
                    value="-1",
                    id="seed-input",
                    type="integer",
                    validators=[],
                    classes="swarmui-input"
                )
            
            # Generate button
            with Container(classes="generate-container"):
                yield Button(
                    "Generate Image",
                    id="generate-button",
                    variant="primary",
                    classes="generate-button",
                    disabled=False
                )
                yield LoadingIndicator(id="loading-indicator", classes="hidden")
            
            # Status/error display
            yield Static("", id="status-message", classes="status-message")
            
            # Image preview area
            with Container(id="preview-container", classes="preview-container hidden"):
                yield Static("", id="image-preview", classes="image-preview")
                with Horizontal(classes="preview-actions"):
                    yield Button("Save", id="save-image", variant="success")
                    yield Button("Copy Path", id="copy-path", variant="default")
                    yield Button("Regenerate", id="regenerate", variant="primary")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Check SwarmUI status
        self.check_server_status()
        
        # Load available models
        self.load_models()
    
    @work(exclusive=True, thread=True)
    def check_server_status(self) -> None:
        """Check if SwarmUI server is available."""
        import asyncio
        
        async def check():
            is_online = await self.service.initialize()
            return "online" if is_online else "offline"
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            status = loop.run_until_complete(check())
            self.server_status = status
            
            # Update status indicator
            self.call_from_thread(self.update_status_indicator, status)
            
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
            self.server_status = "offline"
            self.call_from_thread(self.update_status_indicator, "offline")
    
    def update_status_indicator(self, status: str) -> None:
        """Update the status indicator."""
        try:
            indicator = self.query_one("#status-indicator", Static)
            if status == "online":
                indicator.update("● Online")
                indicator.remove_class("status-unknown", "status-offline")
                indicator.add_class("status-online")
            elif status == "offline":
                indicator.update("● Offline")
                indicator.remove_class("status-unknown", "status-online")
                indicator.add_class("status-offline")
                
                # Disable generate button
                self.query_one("#generate-button", Button).disabled = True
                self.show_status_message("SwarmUI server is offline", "error")
            else:
                indicator.update("● Unknown")
                indicator.add_class("status-unknown")
        except Exception as e:
            logger.error(f"Error updating status indicator: {e}")
    
    @work(exclusive=True, thread=True)
    def load_models(self) -> None:
        """Load available models from SwarmUI."""
        import asyncio
        
        async def get_models():
            return await self.service.get_available_models()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models = loop.run_until_complete(get_models())
            
            if models:
                self.current_models = [m.get('name', m) for m in models]
                self.call_from_thread(self.update_model_selector, self.current_models)
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def update_model_selector(self, models: List[str]) -> None:
        """Update the model selector with available models."""
        try:
            model_select = self.query_one("#model-select", Select)
            
            options = [("Default", "default")]
            for model in models:
                # Shorten long model names for display
                display_name = model.split('/')[-1] if '/' in model else model
                options.append((display_name, model))
            
            model_select.set_options(options)
            logger.info(f"Loaded {len(models)} models")
            
        except Exception as e:
            logger.error(f"Error updating model selector: {e}")
    
    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle template selection change."""
        if event.select.id == "template-select":
            template_id = event.value
            
            if template_id != "custom":
                # Load template settings
                template = BUILTIN_TEMPLATES.get(template_id)
                if template:
                    # Update prompt with template base
                    prompt_input = self.query_one("#prompt-input", TextArea)
                    prompt_input.text = template.base_prompt
                    
                    # Update negative prompt
                    neg_prompt_input = self.query_one("#negative-prompt-input", TextArea)
                    neg_prompt_input.text = template.negative_prompt
                    
                    # Update size if specified
                    if 'width' in template.default_params and 'height' in template.default_params:
                        size_str = f"{template.default_params['width']}x{template.default_params['height']}"
                        size_select = self.query_one("#size-select", Select)
                        
                        # Find matching size option
                        for option_text, option_value in size_select._options:
                            if option_value == size_str:
                                size_select.value = option_value
                                break
                    
                    # Update advanced settings
                    if 'steps' in template.default_params:
                        self.query_one("#steps-input", Input).value = str(template.default_params['steps'])
                    if 'cfg_scale' in template.default_params:
                        self.query_one("#cfg-input", Input).value = str(template.default_params['cfg_scale'])
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "generate-button":
            await self.generate_image()
        elif button_id == "use-last-message":
            self.use_last_message()
        elif button_id == "clear-prompt":
            self.query_one("#prompt-input", TextArea).text = ""
        elif button_id == "save-image":
            await self.save_current_image()
        elif button_id == "copy-path":
            self.copy_image_path()
        elif button_id == "regenerate":
            await self.generate_image()
    
    def use_last_message(self) -> None:
        """Use the last chat message as prompt."""
        # This will be connected to chat context later
        # For now, just show a message
        self.show_status_message("Feature coming soon: Will use last chat message", "info")
    
    @work(exclusive=True, thread=True)
    async def generate_image(self) -> None:
        """Generate an image based on current settings."""
        if self.is_generating:
            return
        
        self.is_generating = True
        self.call_from_thread(self.show_generating_ui)
        
        try:
            # Get parameters from UI
            prompt = self.query_one("#prompt-input", TextArea).text.strip()
            if not prompt:
                self.call_from_thread(self.show_status_message, "Please enter a prompt", "error")
                return
            
            negative_prompt = self.query_one("#negative-prompt-input", TextArea).text.strip()
            
            # Get size
            size_str = self.query_one("#size-select", Select).value
            width, height = map(int, size_str.split('x'))
            
            # Get model
            model_value = self.query_one("#model-select", Select).value
            model = None if model_value == "default" else model_value
            
            # Get advanced settings
            steps = int(self.query_one("#steps-input", Input).value or "20")
            cfg_scale = float(self.query_one("#cfg-input", Input).value or "7.0")
            seed = int(self.query_one("#seed-input", Input).value or "-1")
            
            # Generate image
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.service.generate_custom(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed
                )
            )
            
            if result.success and result.images:
                self.last_result = result
                self.current_image = result.images[0]
                self.call_from_thread(self.show_image_preview, result.images[0])
                self.call_from_thread(self.show_status_message, 
                                    f"Image generated in {result.generation_time:.1f}s", "success")
                
                # Post message for other components
                self.post_message(ImageGenerationMessage(True, result.images))
            else:
                error_msg = result.error or "Unknown error"
                self.call_from_thread(self.show_status_message, f"Generation failed: {error_msg}", "error")
                self.post_message(ImageGenerationMessage(False, [], error_msg))
                
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            self.call_from_thread(self.show_status_message, f"Error: {str(e)}", "error")
            self.post_message(ImageGenerationMessage(False, [], str(e)))
            
        finally:
            self.is_generating = False
            self.call_from_thread(self.hide_generating_ui)
    
    def show_generating_ui(self) -> None:
        """Show UI state for generating."""
        self.query_one("#generate-button", Button).disabled = True
        self.query_one("#loading-indicator").remove_class("hidden")
        self.show_status_message("Generating image...", "info")
    
    def hide_generating_ui(self) -> None:
        """Hide generating UI state."""
        self.query_one("#generate-button", Button).disabled = False
        self.query_one("#loading-indicator").add_class("hidden")
    
    def show_status_message(self, message: str, level: str = "info") -> None:
        """Show a status message.
        
        Args:
            message: Message to show
            level: Message level (info, success, error)
        """
        status = self.query_one("#status-message", Static)
        status.update(message)
        
        # Update styling based on level
        status.remove_class("status-info", "status-success", "status-error")
        status.add_class(f"status-{level}")
    
    def show_image_preview(self, image_path: str) -> None:
        """Show generated image preview.
        
        Args:
            image_path: Path to the generated image
        """
        preview_container = self.query_one("#preview-container")
        preview_container.remove_class("hidden")
        
        # For now, just show the path
        # In a real implementation, we'd display the actual image
        preview = self.query_one("#image-preview", Static)
        preview.update(f"Generated: {Path(image_path).name}")
    
    async def save_current_image(self) -> None:
        """Save the current generated image."""
        if self.last_result and self.last_result.images:
            saved_paths = await self.service.save_generation(self.last_result)
            if saved_paths:
                self.show_status_message(f"Saved to: {Path(saved_paths[0]).name}", "success")
            else:
                self.show_status_message("Failed to save image", "error")
    
    def copy_image_path(self) -> None:
        """Copy current image path to clipboard."""
        if self.current_image:
            # This would copy to clipboard in real implementation
            self.show_status_message(f"Path: {self.current_image}", "info")