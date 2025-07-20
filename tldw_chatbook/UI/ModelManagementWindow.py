# ModelManagementWindow.py  
# Description: Window for managing models and providers for evaluations
#
"""
Model Management Window
----------------------

Provides interface for managing evaluation models and providers.
"""

from typing import Dict, Any, Optional, List
from textual import on, work
from textual.app import ComposeResult
from textual.widgets import (
    Button, Label, Static, Select, Input, Switch,
    DataTable, ListView, ListItem, Collapsible
)
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, Grid
from textual.reactive import reactive
from loguru import logger

from .eval_shared_components import (
    BaseEvaluationWindow, RefreshDataRequest,
    format_model_display
)
from ..Event_Handlers.eval_events import (
    get_available_providers, get_available_models,
    test_model_connection, get_model_info
)
# from ..Widgets.eval_config_dialogs import ModelConfigDialog
# TODO: Import when dialog is implemented


class ModelManagementWindow(BaseEvaluationWindow):
    """Window for managing evaluation models and providers."""
    
    # Reactive state
    selected_provider = reactive(None)
    selected_model = reactive(None)
    is_testing = reactive(False)
    provider_filter = reactive("all")
    
    def compose(self) -> ComposeResult:
        """Compose the model management interface."""
        yield from self.compose_header("Model Management")
        
        with Horizontal(classes="eval-content-area"):
            # Left panel - Provider list
            with VerticalScroll(classes="provider-panel", id="provider-panel"):
                yield Static("🔌 Providers", classes="panel-title")
                
                # Filter
                yield Select(
                    [
                        ("all", "All Providers"),
                        ("configured", "Configured Only"),
                        ("available", "Available"),
                        ("favorites", "Favorites")
                    ],
                    id="provider-filter",
                    value="all",
                    classes="filter-select"
                )
                
                # Provider list
                yield ListView(id="provider-list", classes="provider-list")
            
            # Right panel - Model details
            with VerticalScroll(classes="model-panel", id="model-panel"):
                # Provider info section
                with Container(classes="section-container", id="provider-info-section"):
                    yield Static("Provider Information", classes="section-title")
                    
                    with Grid(classes="info-grid"):
                        yield Label("Provider:")
                        yield Static("Select a provider", id="provider-name", classes="info-value")
                        
                        yield Label("Status:")
                        yield Static("Not Selected", id="provider-status", classes="info-value")
                        
                        yield Label("API Configured:")
                        yield Static("N/A", id="api-configured", classes="info-value")
                        
                        yield Label("Total Models:")
                        yield Static("0", id="total-models", classes="info-value")
                    
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Configure API",
                            id="configure-api-btn",
                            classes="action-button",
                            disabled=True
                        )
                        yield Button(
                            "Test Connection",
                            id="test-connection-btn",
                            classes="action-button",
                            disabled=True
                        )
                
                # Available models section
                with Container(classes="section-container", id="models-section"):
                    yield Static("Available Models", classes="section-title")
                    
                    # Model categories
                    with Container(id="model-categories"):
                        with Collapsible(title="💬 Chat Models", collapsed=False):
                            yield Container(id="chat-models", classes="model-category")
                        
                        with Collapsible(title="📝 Completion Models"):
                            yield Container(id="completion-models", classes="model-category")
                        
                        with Collapsible(title="🎯 Specialized Models"):
                            yield Container(id="specialized-models", classes="model-category")
                
                # Model details section (hidden initially)
                with Container(classes="section-container hidden", id="model-details-section"):
                    yield Static("Model Details", classes="section-title")
                    
                    with Grid(classes="info-grid"):
                        yield Label("Model ID:")
                        yield Static("", id="model-id", classes="info-value")
                        
                        yield Label("Context Length:")
                        yield Static("", id="context-length", classes="info-value")
                        
                        yield Label("Pricing:")
                        yield Static("", id="pricing-info", classes="info-value")
                        
                        yield Label("Capabilities:")
                        yield Static("", id="capabilities", classes="info-value")
                    
                    with Container(id="model-description", classes="description-box"):
                        yield Static("Select a model to view details", classes="muted-text")
                    
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Add to Favorites",
                            id="favorite-btn",
                            classes="action-button"
                        )
                        yield Button(
                            "Quick Test",
                            id="quick-test-btn",
                            classes="action-button primary"
                        )
    
    def on_mount(self) -> None:
        """Initialize the model management window."""
        logger.info("ModelManagementWindow mounted")
        self._load_providers()
    
    @work(exclusive=True)
    async def _load_providers(self) -> None:
        """Load available providers."""
        try:
            providers = await get_available_providers(self.app_instance)
            
            provider_list = self.query_one("#provider-list", ListView)
            provider_list.clear()
            
            for provider in providers:
                # Check if configured
                is_configured = await self._check_provider_configured(provider)
                
                item = ListItem(
                    Static(
                        f"{'✅' if is_configured else '⚪'} {provider}",
                        classes="provider-item"
                    ),
                    id=f"provider-{provider}"
                )
                provider_list.append(item)
            
        except Exception as e:
            self.notify_error(f"Failed to load providers: {e}")
    
    async def _check_provider_configured(self, provider: str) -> bool:
        """Check if a provider is configured."""
        try:
            # Check if API key exists in config
            from ..config import load_settings
            config = load_settings()
            api_settings = config.get('api_settings', {})
            provider_config = api_settings.get(provider.lower(), {})
            
            return bool(provider_config.get('api_key'))
        except:
            return False
    
    @on(ListView.Selected, "#provider-list")
    async def handle_provider_selection(self, event: ListView.Selected) -> None:
        """Handle provider selection."""
        if event.item and event.item.id:
            provider = event.item.id.replace("provider-", "")
            self.selected_provider = provider
            await self._load_provider_info(provider)
            await self._load_provider_models(provider)
    
    async def _load_provider_info(self, provider: str) -> None:
        """Load and display provider information."""
        try:
            # Update provider info
            self.query_one("#provider-name").update(provider)
            
            # Check configuration status
            is_configured = await self._check_provider_configured(provider)
            self.query_one("#api-configured").update("Yes" if is_configured else "No")
            self.query_one("#provider-status").update(
                "✅ Ready" if is_configured else "⚠️ Not Configured"
            )
            
            # Enable buttons
            self.query_one("#configure-api-btn").disabled = False
            self.query_one("#test-connection-btn").disabled = not is_configured
            
        except Exception as e:
            logger.error(f"Failed to load provider info: {e}")
    
    @work(exclusive=True)
    async def _load_provider_models(self, provider: str) -> None:
        """Load models for selected provider."""
        try:
            # get_available_models is not async 
            models = get_available_models(self.app_instance)
            
            # Update total count
            self.query_one("#total-models").update(str(len(models)))
            
            # Clear existing models
            for category in ["chat-models", "completion-models", "specialized-models"]:
                container = self.query_one(f"#{category}", Container)
                container.clear()
            
            # Categorize and display models
            for model in models:
                category = self._categorize_model(model)
                container = self.query_one(f"#{category}", Container)
                
                model_item = Container(classes="model-item")
                model_item.mount(Static(model['name'], classes="model-name"))
                model_item.mount(Static(model.get('description', ''), classes="model-desc"))
                model_item.mount(Button(
                    "Details",
                    classes="mini-button",
                    id=f"model-{model['id']}"
                ))
                
                container.mount(model_item)
            
        except Exception as e:
            self.notify_error(f"Failed to load models: {e}")
    
    def _categorize_model(self, model: Dict[str, Any]) -> str:
        """Categorize a model based on its properties."""
        model_id = model.get('id', '').lower()
        
        if 'chat' in model_id or 'turbo' in model_id or 'claude' in model_id:
            return "chat-models"
        elif 'completion' in model_id or 'text-' in model_id:
            return "completion-models"
        else:
            return "specialized-models"
    
    @on(Button.Pressed, "#configure-api-btn")
    async def handle_configure_api(self) -> None:
        """Open API configuration dialog."""
        if not self.selected_provider:
            return
        
        # from ..Widgets.eval_config_dialogs import APIConfigDialog
        # TODO: Implement dialog
        self.notify_error("API config dialog not yet implemented")
        return
        
        def on_config_saved(config):
            if config:
                self.notify_success(f"API configuration saved for {self.selected_provider}")
                self._load_provider_info(self.selected_provider)
        
        dialog = APIConfigDialog(
            provider=self.selected_provider,
            callback=on_config_saved
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#test-connection-btn")
    async def handle_test_connection(self) -> None:
        """Test connection to selected provider."""
        if not self.selected_provider:
            return
        
        self.is_testing = True
        btn = self.query_one("#test-connection-btn", Button)
        btn.label = "Testing..."
        btn.disabled = True
        
        try:
            result = await test_model_connection(
                self.app_instance,
                self.selected_provider
            )
            
            if result['success']:
                self.notify_success(f"Connection successful! Response time: {result['latency']}ms")
            else:
                self.notify_error(f"Connection failed: {result['error']}")
                
        except Exception as e:
            self.notify_error(f"Test failed: {e}")
        finally:
            self.is_testing = False
            btn.label = "Test Connection"
            btn.disabled = False
    
    @on(Button.Pressed)
    async def handle_model_details(self, event: Button.Pressed) -> None:
        """Handle model details button click."""
        if event.button.id and event.button.id.startswith("model-"):
            model_id = event.button.id.replace("model-", "")
            await self._show_model_details(model_id)
    
    async def _show_model_details(self, model_id: str) -> None:
        """Show detailed information for a model."""
        try:
            # Get model info
            model_info = await get_model_info(
                self.app_instance,
                self.selected_provider,
                model_id
            )
            
            # Show details section
            details_section = self.query_one("#model-details-section")
            details_section.remove_class("hidden")
            
            # Update details
            self.query_one("#model-id").update(model_id)
            self.query_one("#context-length").update(
                f"{model_info.get('context_length', 'Unknown'):,} tokens"
            )
            
            # Pricing info
            pricing = model_info.get('pricing', {})
            if pricing:
                pricing_text = (
                    f"Input: ${pricing.get('input', 0)}/1K tokens, "
                    f"Output: ${pricing.get('output', 0)}/1K tokens"
                )
            else:
                pricing_text = "Pricing information not available"
            self.query_one("#pricing-info").update(pricing_text)
            
            # Capabilities
            capabilities = model_info.get('capabilities', [])
            self.query_one("#capabilities").update(", ".join(capabilities) or "Standard")
            
            # Description
            desc_container = self.query_one("#model-description")
            desc_container.clear()
            desc_container.mount(
                Static(
                    model_info.get('description', 'No description available'),
                    classes="model-description-text"
                )
            )
            
            self.selected_model = model_id
            
        except Exception as e:
            self.notify_error(f"Failed to load model details: {e}")
    
    @on(Button.Pressed, "#quick-test-btn")
    async def handle_quick_test(self) -> None:
        """Run a quick test with the selected model."""
        if not self.selected_provider or not self.selected_model:
            self.notify_error("Please select a model first")
            return
        
        # from ..Widgets.eval_config_dialogs import QuickTestDialog
        # TODO: Implement dialog
        self.notify_error("Quick test dialog not yet implemented")
        return
        
        def on_test_complete(result):
            if result:
                self.notify_success("Quick test completed successfully")
                # Could show results in a popup or navigate to results
        
        dialog = QuickTestDialog(
            provider=self.selected_provider,
            model=self.selected_model,
            callback=on_test_complete
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#favorite-btn")
    async def handle_add_favorite(self) -> None:
        """Add model to favorites."""
        if not self.selected_model:
            return
        
        # TODO: Implement favorites functionality
        self.notify_success(f"Added {self.selected_model} to favorites")
    
    @on(Select.Changed, "#provider-filter")
    async def handle_filter_change(self, event: Select.Changed) -> None:
        """Handle provider filter change."""
        self.provider_filter = event.value
        # TODO: Implement filtering logic
        await self._load_providers()
    
    @on(Button.Pressed, "#back-to-main")
    def handle_back(self) -> None:
        """Go back to main evaluation window."""
        self.navigate_to("main")
    
    @on(Button.Pressed, "#refresh-data")
    async def handle_refresh(self) -> None:
        """Refresh all data."""
        await self._load_providers()
        if self.selected_provider:
            await self._load_provider_models(self.selected_provider)
        self.notify_success("Data refreshed")