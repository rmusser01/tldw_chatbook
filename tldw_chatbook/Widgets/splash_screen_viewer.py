"""Splash screen viewer widget for browsing and previewing available splash screens."""

import asyncio
from typing import Optional, Dict, Any, List
import time

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Select, Label, OptionList, Input
from textual.widgets.option_list import Option
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, WorkerState

from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config
from ..Utils.Splash_Screens import load_all_effects, EFFECTS_REGISTRY
from ..Utils.Splash_Screens.card_definitions import get_all_card_definitions
from .splash_screen import SplashScreen


class SplashCardInfo:
    """Container for splash card information."""
    
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.data = data
        self.type = data.get("type", "static")
        self.effect = data.get("effect")
        self.title = data.get("title", name)
        self.description = self._get_description()
    
    def _get_description(self) -> str:
        """Generate a description based on card data."""
        if self.type == "static":
            return "Static ASCII art display"
        elif self.effect:
            return f"Animated with {self.effect.replace('_', ' ').title()} effect"
        return "Custom splash screen"


class SplashPreviewWidget(Container):
    """Widget for previewing splash screen animations."""
    
    DEFAULT_CLASSES = "splash-preview"
    
    # Reactive attributes
    is_playing: reactive[bool] = reactive(False)
    current_card: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.animation_timer = None
        self.effect_instance = None
        self.card_data = None
        
    def compose(self) -> ComposeResult:
        """Compose the preview widget."""
        yield Static(
            "Preview area - select a card to preview",
            id="preview-content",
            classes="preview-content"
        )
    
    def preview_card(self, card_name: str, card_data: Dict[str, Any]) -> None:
        """Start previewing a splash card."""
        self.current_card = card_name
        self.card_data = card_data
        
        # Update preview content
        preview_content = self.query_one("#preview-content", Static)
        
        if card_data.get("type") == "animated":
            # For animated cards, show a placeholder
            preview_content.update(
                f"[bold]{card_data.get('title', card_name)}[/bold]\n\n"
                f"[dim]Animated splash screen\n"
                f"Effect: {card_data.get('effect', 'Unknown')}\n\n"
                f"Click 'Play Selected' to see the animation[/dim]"
            )
        else:
            # For static cards, show the actual content
            content = card_data.get("art", ["No preview available"])
            if isinstance(content, list):
                content = "\n".join(content)
            preview_content.update(content)
    
    def stop_preview(self) -> None:
        """Stop the current preview."""
        self.is_playing = False
        if self.animation_timer:
            self.animation_timer.cancel()
            self.animation_timer = None


class SplashScreenViewer(Container):
    """Main viewer widget for browsing splash screens."""
    
    DEFAULT_CLASSES = "splash-viewer"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splash_cards: Dict[str, SplashCardInfo] = {}
        self._load_splash_cards()
        self.preview_duration = 3.0  # Default preview duration
    
    def _load_splash_cards(self) -> None:
        """Load all available splash cards."""
        # Load all effects first
        load_all_effects()
        
        # Get all card definitions
        all_cards = get_all_card_definitions()
        
        logger.info(f"Loading {len(all_cards)} splash cards")
        
        for card_name, card_data in all_cards.items():
            try:
                self.splash_cards[card_name] = SplashCardInfo(card_name, card_data)
            except Exception as e:
                logger.error(f"Failed to load card {card_name}: {e}")
        
        logger.info(f"Successfully loaded {len(self.splash_cards)} splash cards")
    
    def compose(self) -> ComposeResult:
        """Compose the viewer interface."""
        # Header (will be hidden when embedded via CSS)
        yield Label("🎨 Splash Screen Gallery", classes="viewer-header")
        yield Static("Select a splash screen below to preview it. Double-click to play.", classes="help-text")
        
        # Build the list of available splash screens
        if not self.splash_cards:
            self._load_splash_cards()
        
        # Create options for the list
        options = []
        for card_name, card_info in self.splash_cards.items():
            # If title is generic, show the card name too
            if card_info.title.lower() in ["tldw chatbook", "tldw", "loading tldw chatbook"]:
                option_text = f"{card_info.title} ({card_name}) [{card_info.type}]"
            else:
                option_text = f"{card_info.title} [{card_info.type}]"
            options.append(Option(option_text, id=card_name))
        
        if not options:
            options.append(Option("No splash screens found", id="none"))
        
        # Main list of splash screens
        yield OptionList(*options, id="card-list", classes="card-list")
        
        # Card information display
        yield Static("Select a card to see details", id="card-info", classes="card-info")
        
        # Duration control section
        with Horizontal(classes="duration-control"):
            yield Label("Preview Duration: ", classes="duration-label")
            yield Input(
                value="3.0",
                id="duration-input",
                classes="duration-input",
                placeholder="3.0",
                type="number"
            )
            yield Label(" seconds", classes="duration-suffix")
        
        # Action buttons
        with Horizontal(classes="action-buttons"):
            yield Button("Play Selected", id="play-selected-button", variant="primary")
            yield Button("Set as Default", id="set-default-button", variant="success")
            # Only show close button if we're in a modal (not embedded)
            if not self.has_class("embedded-splash-viewer"):
                yield Button("Close", id="close-button", variant="default")
    
    @on(OptionList.OptionHighlighted)
    def on_card_selected(self, event: OptionList.OptionHighlighted) -> None:
        """Handle card selection from the list."""
        if not event.option_id:
            return
        
        card_name = event.option_id
        if card_name not in self.splash_cards:
            return
        
        card_info = self.splash_cards[card_name]
        
        # Check if this is the current default
        current_default = get_cli_setting("splash_screen", "card_selection", "random")
        is_default = card_name == current_default
        
        # Update card info display
        info_widget = self.query_one("#card-info", Static)
        default_marker = " [green]✓ DEFAULT[/green]" if is_default else ""
        info_text = f"""[bold]{card_info.title}[/bold]{default_marker}
Card ID: {card_name}  |  Type: {card_info.type}  |  Effect: {card_info.effect or 'None'}
{card_info.description}

[dim]Double-click or press "Play Selected" to view this splash screen[/dim]"""
        info_widget.update(info_text)
    
    @on(OptionList.OptionSelected)
    def on_card_double_clicked(self, event: OptionList.OptionSelected) -> None:
        """Handle double-click on a card to play it."""
        if event.option_id:
            # Play the selected card
            self._play_card_async(event.option_id)
    
    def _play_card_async(self, card_id: str) -> None:
        """Start async worker to play a card."""
        from functools import partial
        worker_func = partial(self._play_card_worker, card_id)
        self.run_worker(worker_func, exclusive=True)
    
    async def _play_card_worker(self, card_id: str) -> None:
        """Worker to play a specific splash screen card."""
        if card_id not in self.splash_cards:
            return
            
        # Get duration from input
        try:
            duration_input = self.query_one("#duration-input", Input)
            duration = float(duration_input.value)
            # Clamp duration between 0.5 and 30 seconds
            duration = max(0.5, min(30.0, duration))
        except (ValueError, Exception):
            duration = 3.0  # Default 3 seconds for preview
        
        # Show notification
        card_info = self.splash_cards[card_id]
        self.app.notify(f"Playing '{card_info.title}' splash screen...", severity="information")
        
        # Create splash screen with specific card
        splash = SplashScreen(
            card_name=card_id,
            duration=duration,
            show_progress=True
        )
        
        # Create overlay container
        from textual.containers import Container
        overlay = Container(splash, id="splash-overlay")
        overlay.styles.layer = "overlay"
        overlay.styles.width = "100%"
        overlay.styles.height = "100%"
        overlay.styles.align = ("center", "middle")
        
        # Mount overlay
        await self.app.mount(overlay)
        
        # Wait for splash screen to complete
        await asyncio.sleep(duration)
        
        # Remove overlay
        await overlay.remove()
    
    @on(Button.Pressed, "#play-selected-button")
    def on_play_selected_button(self, event: Button.Pressed) -> None:
        """Play the currently selected splash screen."""
        # Get selected card from option list
        card_list = self.query_one("#card-list", OptionList)
        if card_list.highlighted is None:
            self.app.notify("Please select a splash screen first", severity="warning")
            return
            
        option_id = card_list.get_option_at_index(card_list.highlighted).id
        if not option_id or option_id not in self.splash_cards:
            self.app.notify("Invalid selection", severity="error")
            return
        
        # Play the selected card
        self._play_card_async(option_id)
    
    @on(Button.Pressed, "#set-default-button")
    def on_set_default_button(self, event: Button.Pressed) -> None:
        """Set the currently selected splash screen as the default."""
        # Get selected card from option list
        card_list = self.query_one("#card-list", OptionList)
        if card_list.highlighted is None:
            self.app.notify("Please select a splash screen first", severity="warning")
            return
            
        option_id = card_list.get_option_at_index(card_list.highlighted).id
        if not option_id or option_id not in self.splash_cards:
            self.app.notify("Invalid selection", severity="error")
            return
        
        # Save the selection to config
        try:
            save_setting_to_cli_config("splash_screen", "card_selection", option_id)
            card_info = self.splash_cards[option_id]
            self.app.notify(f"Set '{card_info.title}' as default splash screen", severity="success")
        except Exception as e:
            self.app.notify(f"Failed to save setting: {e}", severity="error")
            logger.error(f"Failed to save default splash screen: {e}")
    
    @on(Button.Pressed, "#close-button")
    def on_close_button(self, event: Button.Pressed) -> None:
        """Handle close button press."""
        # Try to dismiss the parent screen (modal)
        try:
            if hasattr(self.app.screen, 'dismiss'):
                self.app.screen.dismiss()
        except Exception as e:
            logger.error(f"Failed to close: {e}")
    
    @on(Input.Changed, "#duration-input")
    def on_duration_changed(self, event: Input.Changed) -> None:
        """Validate duration input as user types."""
        try:
            value = float(event.value) if event.value else 0
            # Provide feedback if value is out of range
            if value < 0.5:
                event.input.add_class("error")
                self.app.notify("Duration must be at least 0.5 seconds", severity="warning")
            elif value > 30:
                event.input.add_class("error")
                self.app.notify("Duration cannot exceed 30 seconds", severity="warning")
            else:
                event.input.remove_class("error")
        except ValueError:
            if event.value:  # Only show error if there's actual input
                event.input.add_class("error")