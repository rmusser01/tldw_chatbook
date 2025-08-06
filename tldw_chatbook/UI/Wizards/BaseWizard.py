# tldw_chatbook/UI/Wizards/BaseWizard.py
# Description: Base wizard framework for creating step-by-step guided interfaces
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
# from abc import ABC, abstractmethod  # Removed due to metaclass conflict with Textual

# 3rd-Party Imports
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static, Button, Label
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.screen import Screen

# Configure logger
logger = logger.bind(module="BaseWizard")

if TYPE_CHECKING:
    from textual.app import App

########################################################################################################################
#
# Configuration Classes
#
########################################################################################################################

@dataclass
class WizardStepConfig:
    """Configuration for a wizard step."""
    id: str
    title: str
    description: str = ""
    icon: Optional[str] = None
    can_skip: bool = False
    step_number: int = 0

########################################################################################################################
#
# Base Classes
#
########################################################################################################################

class WizardScreen(Screen):
    """Base screen class for wizards."""
    
    DEFAULT_CSS = """
    WizardScreen {
        align: center middle;
        background: $background;
    }
    
    WizardScreen > WizardContainer {
        width: 90%;
        height: 90%;
        max-width: 120;
        background: $surface;
        border: solid $primary;
        padding: 2;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel wizard"),
    ]
    
    def __init__(self, app_instance, **kwargs):
        super().__init__()
        self.app_instance = app_instance
        self.wizard_kwargs = kwargs
        
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)

class WizardStep(Container):
    """Base class for individual wizard steps."""
    
    # Step metadata
    step_number = reactive(0)
    step_title = reactive("")
    step_description = reactive("")
    
    # Step state
    is_active = reactive(False)
    is_complete = reactive(False)
    is_valid = reactive(True)
    
    # Validation
    validation_errors: reactive[List[str]] = reactive([])
    
    def __init__(
        self,
        wizard: Optional['WizardContainer'] = None,  # Make optional for compatibility
        config: Optional[WizardStepConfig] = None,  # Make optional for compatibility
        step_number: int = 0,
        step_title: str = "",
        step_description: str = "",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Handle both old and new initialization patterns
        if wizard and config:
            self.wizard = wizard
            self.config = config
            self.step_number = config.step_number or step_number
            self.step_title = config.title
            self.step_description = config.description
        else:
            # Fallback for old pattern
            self.wizard = None
            self.config = None
            self.step_number = step_number
            self.step_title = step_title
            self.step_description = step_description
            
        self.add_class("wizard-step")
        
    def compose(self) -> ComposeResult:
        """Compose the step's UI elements. Override in subclasses."""
        yield Container()  # Default implementation
        
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the step's data. Override in subclasses.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        return True, []  # Default implementation
        
    def get_data(self) -> Dict[str, Any]:
        """Get the step's collected data. Override in subclasses."""
        return {}  # Default implementation
        
    def get_step_data(self) -> Dict[str, Any]:
        """Compatibility method - calls get_data()."""
        return self.get_data()
        
    def on_show(self) -> None:
        """Called when the step becomes active."""
        self.is_active = True
        self.add_class("active")
        logger.debug(f"Step {self.step_number} '{self.step_title}' activated")
        
    def on_hide(self) -> None:
        """Called when the step becomes inactive."""
        self.is_active = False
        self.remove_class("active")
        logger.debug(f"Step {self.step_number} '{self.step_title}' deactivated")
        
    def reset(self) -> None:
        """Reset the step to initial state."""
        self.is_complete = False
        self.is_valid = True
        self.validation_errors = []


class WizardNavigation(Horizontal):
    """Navigation controls for the wizard."""
    
    can_go_back = reactive(True)
    can_go_forward = reactive(False)
    
    def watch_can_go_forward(self) -> None:
        """React to can_go_forward changes."""
        logger.debug(f"WizardNavigation: can_go_forward changed to {self.can_go_forward}")
        self.update_button_states()
    current_step = reactive(1)
    total_steps = reactive(1)
    
    def compose(self) -> ComposeResult:
        """Compose navigation elements."""
        yield Button("Cancel", id="wizard-cancel", variant="error")
        yield Button("← Back", id="wizard-back", variant="default", disabled=True)
        yield Static("", id="wizard-progress", classes="wizard-progress-text")
        # Start with Next button disabled until validation passes
        yield Button("Next →", id="wizard-next", variant="default", disabled=True)
        
    def on_mount(self) -> None:
        """Update initial state."""
        self.update_progress_text()
        self.update_button_states()
        
    def watch_current_step(self) -> None:
        """React to step changes."""
        logger.debug(f"WizardNavigation: current_step changed to {self.current_step}")
        self.update_progress_text()
        self.update_button_states()
        
    def watch_total_steps(self) -> None:
        """React to total steps changes."""
        self.update_progress_text()
        
    def update_progress_text(self) -> None:
        """Update the progress text."""
        try:
            progress = self.query_one("#wizard-progress", Static)
            progress.update(f"Step {self.current_step} of {self.total_steps}")
        except NoMatches:
            pass
            
    def update_button_states(self) -> None:
        """Update button enabled states."""
        try:
            back_btn = self.query_one("#wizard-back", Button)
            next_btn = self.query_one("#wizard-next", Button)
            
            # Update back button
            back_btn.disabled = not self.can_go_back or self.current_step <= 1
            
            # Update next button - make sure to enable/disable properly
            should_disable = not self.can_go_forward
            if next_btn.disabled != should_disable:
                next_btn.disabled = should_disable
                # Force refresh the button
                next_btn.refresh()
            
            logger.debug(f"WizardNavigation.update_button_states: can_go_forward={self.can_go_forward}, next_btn.disabled={next_btn.disabled}")
            
            # Update next button text for last step
            if self.current_step >= self.total_steps:
                next_btn.label = "Finish"
                next_btn.variant = "success"
            else:
                next_btn.label = "Next →"
                # Only set variant to primary if button is enabled
                if not next_btn.disabled:
                    next_btn.variant = "primary"
        except NoMatches:
            logger.warning("WizardNavigation: Could not find buttons to update")


class WizardProgress(Horizontal):
    """Visual progress indicator for wizard steps."""
    
    DEFAULT_CSS = """
    WizardProgress {
        layout: horizontal;
        align: center middle;
        height: auto;
    }
    
    WizardProgress .step-indicator-container {
        layout: horizontal;
        align: center middle;
        height: auto;
        margin: 0 1;
    }
    
    WizardProgress .step-number {
        width: 4;
        height: 3;
        content-align: center middle;
        text-align: center;
        background: $surface;
        border: solid $primary;
    }
    
    WizardProgress .step-number.active {
        background: $primary;
        color: $background;
        text-style: bold;
    }
    
    WizardProgress .step-number.complete {
        background: $success;
        color: $background;
    }
    
    WizardProgress .step-title {
        margin: 0 1;
    }
    
    WizardProgress .step-title.active {
        text-style: bold;
        color: $primary;
    }
    
    WizardProgress .step-connector {
        width: 4;
        height: 1;
        background: $primary-lighten-2;
    }
    
    WizardProgress .step-connector.complete {
        background: $success;
    }
    """
    
    current_step = reactive(1)
    total_steps = reactive(1)
    step_titles: reactive[List[str]] = reactive([])
    
    def compose(self) -> ComposeResult:
        """Compose progress indicators."""
        for i in range(1, self.total_steps + 1):
            # Step indicator
            is_active = i == self.current_step
            is_complete = i < self.current_step
            
            with Container(classes="step-indicator-container"):
                # Step number circle
                step_classes = "step-number"
                if is_active:
                    step_classes += " active"
                elif is_complete:
                    step_classes += " complete"
                    
                yield Static(
                    "✓" if is_complete else str(i),
                    classes=step_classes
                )
                
                # Step title
                if i <= len(self.step_titles):
                    title_classes = "step-title"
                    if is_active:
                        title_classes += " active"
                    elif is_complete:
                        title_classes += " complete"
                    yield Label(
                        self.step_titles[i-1],
                        classes=title_classes
                    )
                
                # Connector line (except for last step)
                if i < self.total_steps:
                    connector_classes = "step-connector"
                    if is_complete:
                        connector_classes += " complete"
                    yield Static("", classes=connector_classes)


class WizardContainer(Container):
    """Main wizard container that manages steps and navigation."""
    
    DEFAULT_CSS = """
    WizardContainer {
        layout: vertical;
        height: 100%;
        width: 100%;
    }
    
    WizardContainer .wizard-title {
        text-style: bold;
        text-align: center;
        margin: 1 0;
        color: $text;
    }
    
    WizardContainer .wizard-progress {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $boost;
    }
    
    WizardContainer .wizard-steps-container {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
        margin: 1 0;
    }
    
    WizardContainer .wizard-step {
        width: 100%;
        height: 100%;
    }
    
    WizardContainer .wizard-step.hidden {
        display: none;
    }
    
    WizardContainer .wizard-navigation {
        height: auto;
        layout: horizontal;
        align: center middle;
        padding: 1;
        margin-top: 1;
        border-top: solid $primary;
    }
    
    WizardContainer .wizard-navigation Button {
        margin: 0 1;
    }
    
    WizardContainer .wizard-progress-text {
        width: 1fr;
        text-align: center;
        content-align: center middle;
    }
    
    /* Form styling */
    WizardContainer .form-group {
        margin-bottom: 1;
    }
    
    WizardContainer .form-label {
        margin-bottom: 0;
        text-style: bold;
    }
    
    WizardContainer .form-input {
        width: 100%;
        margin-top: 0;
    }
    
    WizardContainer .info-box {
        background: $boost;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel wizard"),
        Binding("ctrl+b", "back", "Previous step"),
        Binding("ctrl+n", "next", "Next step"),
    ]
    
    # Wizard state
    current_step = reactive(0)
    total_steps = reactive(0)
    can_proceed = reactive(False)
    is_complete = reactive(False)
    
    # Callbacks
    on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    on_cancel: Optional[Callable[[], None]] = None
    
    def __init__(
        self,
        app_instance,
        steps: Optional[List[WizardStep]] = None,
        title: str = "Wizard",
        template_data: Optional[Dict[str, Any]] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        *args,
        **kwargs
    ):
        # Remove steps from kwargs if present to avoid duplicate
        kwargs.pop('steps', None)
        super().__init__(*args, **kwargs)
        
        self.app_instance = app_instance
        self.steps = steps or []
        self.title = title
        self.template_data = template_data or {}
        self.wizard_data = {}
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self.total_steps = len(self.steps)
        self.add_class("wizard-container")
        
        # Initialize step numbers and hide all steps
        for i, step in enumerate(self.steps):
            step.step_number = i + 1
            step.add_class("hidden")
            
    def compose(self) -> ComposeResult:
        """Compose the wizard UI."""
        # Title
        yield Label(self.title, classes="wizard-title")
        
        # Progress indicator
        step_titles = [step.step_title for step in self.steps]
        progress = WizardProgress(classes="wizard-progress")
        progress.current_step = self.current_step + 1
        progress.total_steps = self.total_steps
        progress.step_titles = step_titles
        yield progress
        
        # Step container
        with Container(classes="wizard-steps-container"):
            for step in self.steps:
                yield step
                
        # Navigation
        nav = WizardNavigation(classes="wizard-navigation")
        nav.current_step = self.current_step + 1
        nav.total_steps = self.total_steps
        nav.can_go_back = self.current_step > 0
        nav.can_go_forward = self.can_proceed
        yield nav
        
    def on_mount(self) -> None:
        """Initialize wizard on mount."""
        self.show_step(0)
        # Trigger initial validation after a short delay to allow step to fully initialize
        self.set_timer(0.1, self.validate_step)
        
    def show_step(self, step_index: int) -> None:
        """Show a specific step."""
        if 0 <= step_index < len(self.steps):
            # Hide current step
            if 0 <= self.current_step < len(self.steps):
                current = self.steps[self.current_step]
                current.remove_class("active")
                current.add_class("hidden")
                current.on_hide()
                
            # Show new step
            self.current_step = step_index
            new_step = self.steps[step_index]
            new_step.remove_class("hidden")
            new_step.add_class("active")
            new_step.on_show()
            
            # Update progress
            self.update_progress()
            
            # Validate step to update navigation
            self.validate_step()
            
    def update_progress(self) -> None:
        """Update progress indicators."""
        try:
            # Update progress bar
            progress = self.query_one(".wizard-progress", WizardProgress)
            progress.current_step = self.current_step + 1
            
            # Update navigation
            nav = self.query_one(".wizard-navigation", WizardNavigation)
            nav.current_step = self.current_step + 1
            nav.can_go_back = self.current_step > 0
            nav.can_go_forward = self.can_proceed
        except NoMatches:
            pass
            
    def validate_step(self) -> None:
        """Validate the current step and update navigation."""
        logger.debug(f"WizardContainer.validate_step called for step {self.current_step}")
        
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            is_valid, errors = step.validate()
            step.is_valid = is_valid
            step.validation_errors = errors
            self.can_proceed = is_valid
            
            logger.debug(f"Step validation result: valid={is_valid}, errors={errors}")
            logger.debug(f"Setting can_proceed to {is_valid}")
            
            # Update navigation
            try:
                nav = self.query_one(".wizard-navigation", WizardNavigation)
                nav.can_go_forward = is_valid
                logger.debug(f"Updated navigation can_go_forward to {is_valid}")
            except NoMatches:
                logger.warning("Could not find wizard navigation to update")
            
            # Allow proceeding if this is a special step (like ProgressStep)
            if hasattr(step, 'can_proceed'):
                self.can_proceed = step.can_proceed()
                logger.debug(f"Special step can_proceed override: {self.can_proceed}")
                
    @on(Button.Pressed, "#wizard-next")
    def handle_next(self) -> None:
        """Handle next button press."""
        if self.current_step < len(self.steps) - 1:
            # Validate current step
            if self.can_proceed:
                # Save current step data
                current_step = self.steps[self.current_step]
                step_id = current_step.config.id if current_step.config else f"step_{self.current_step}"
                self.wizard_data[step_id] = current_step.get_step_data()
                
                # Mark current step as complete
                current_step.is_complete = True
                
                # Move to next step
                self.show_step(self.current_step + 1)
        else:
            # Last step - complete wizard
            self.complete_wizard()
            
    @on(Button.Pressed, "#wizard-back")
    def handle_back(self) -> None:
        """Handle back button press."""
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
            
    @on(Button.Pressed, "#wizard-cancel")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.action_cancel()
            
    def action_next(self) -> None:
        """Keyboard shortcut for next."""
        self.handle_next()
        
    def action_back(self) -> None:
        """Keyboard shortcut for back."""
        self.handle_back()
        
    def action_cancel(self) -> None:
        """Cancel the wizard."""
        logger.info("Wizard cancelled")
        if self.on_cancel:
            self.on_cancel()
        # Find and dismiss the parent screen
        parent_screen = self.ancestors_with_self[1] if len(self.ancestors_with_self) > 1 else None
        if parent_screen and isinstance(parent_screen, Screen):
            parent_screen.dismiss(None)
            
    def complete_wizard(self) -> None:
        """Complete the wizard and collect all data."""
        # Validate final step
        if not self.can_proceed:
            return
            
        # Mark as complete
        self.is_complete = True
        self.steps[self.current_step].is_complete = True
        
        # Collect all data from each step with its ID
        self.wizard_data[self.steps[self.current_step].config.id if self.steps[self.current_step].config else f"step_{self.current_step}"] = self.steps[self.current_step].get_step_data()
        
        logger.info(f"Wizard completed with data: {self.wizard_data}")
        
        # Call completion callback
        if self.on_complete:
            self.on_complete(self.wizard_data)
            
    def get_all_data(self) -> Dict[str, Any]:
        """Get data from all completed steps."""
        all_data = {}
        for step in self.steps:
            if step.is_complete or step.is_active:
                all_data.update(step.get_data())
        return all_data
    
    def refresh_current_step(self) -> None:
        """Refresh the current step's validation state."""
        logger.debug(f"refresh_current_step called for step {self.current_step}")
        self.validate_step()