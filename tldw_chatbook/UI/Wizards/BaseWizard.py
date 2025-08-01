# BaseWizard.py
# Description: Base class for multi-step wizard screens
#
"""
Base Wizard Framework
--------------------

Provides abstract base classes for creating multi-step wizards with:
- Step management and navigation
- Progress tracking
- State persistence
- Validation between steps
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING, Protocol
from dataclasses import dataclass, field
from enum import Enum

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Static, Button, ProgressBar
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class WizardDirection(Enum):
    """Direction of wizard navigation."""
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class WizardStepConfig:
    """Configuration for a wizard step."""
    id: str
    title: str
    description: str
    can_skip: bool = False
    validator: Optional[Callable[[], tuple[bool, Optional[str]]]] = None


class WizardStepComplete(Message):
    """Message sent when a wizard step is completed."""
    def __init__(self, step_id: str, data: Dict[str, Any]) -> None:
        super().__init__()
        self.step_id = step_id
        self.data = data


class WizardStep(Container):
    """Base class for individual wizard steps."""
    
    def __init__(self, wizard: 'BaseWizard', config: WizardStepConfig, **kwargs):
        """Initialize wizard step."""
        super().__init__(**kwargs)
        self.wizard = wizard
        self.config = config
        self._is_active = False
        
    @abstractmethod
    def compose(self) -> ComposeResult:
        """Compose the step UI."""
        pass
    
    @abstractmethod
    def get_step_data(self) -> Dict[str, Any]:
        """Get the data entered in this step."""
        pass
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate the step data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.config.validator:
            return self.config.validator()
        return True, None
    
    async def on_enter(self) -> None:
        """Called when entering this step."""
        self._is_active = True
        
    async def on_leave(self) -> None:
        """Called when leaving this step."""
        self._is_active = False
        
    def can_proceed(self) -> bool:
        """Check if we can proceed to the next step."""
        is_valid, _ = self.validate()
        return is_valid or self.config.can_skip


class StepProgress(Static):
    """Visual progress indicator for wizard steps."""
    
    DEFAULT_CSS = """
    StepProgress {
        height: 3;
        padding: 1 2;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    .step-indicator {
        layout: horizontal;
        height: 1;
        align: center middle;
    }
    
    .step-dot {
        width: 3;
        height: 1;
        text-align: center;
        margin: 0 1;
    }
    
    .step-dot.completed {
        color: $success;
    }
    
    .step-dot.current {
        color: $primary;
        text-style: bold;
    }
    
    .step-dot.pending {
        color: $text-muted;
    }
    
    .step-connector {
        width: 5;
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    
    .step-label {
        text-align: center;
        color: $text-muted;
        margin-top: 0;
    }
    """
    
    def __init__(self, steps: List[WizardStepConfig], current_step: int = 0, **kwargs):
        """Initialize progress indicator."""
        super().__init__(**kwargs)
        self.steps = steps
        self.current_step = current_step
        
    def compose(self) -> ComposeResult:
        """Compose the progress indicator."""
        with Container(classes="step-indicator"):
            for i, step in enumerate(self.steps):
                if i > 0:
                    yield Static("───", classes="step-connector")
                    
                if i < self.current_step:
                    yield Static("●", classes="step-dot completed")
                elif i == self.current_step:
                    yield Static("●", classes="step-dot current")
                else:
                    yield Static("○", classes="step-dot pending")
        
        yield Static(
            f"Step {self.current_step + 1} of {len(self.steps)}: {self.steps[self.current_step].title}",
            classes="step-label"
        )
    
    def update_step(self, step_index: int) -> None:
        """Update the current step."""
        self.current_step = step_index
        self.refresh()


class BaseWizard(ModalScreen):
    """Base class for multi-step wizards."""
    
    DEFAULT_CSS = """
    BaseWizard {
        align: center middle;
    }
    
    BaseWizard > Container {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 50;
        background: $surface;
        border: thick $primary;
    }
    
    .wizard-header {
        height: auto;
        padding: 1 2;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    .wizard-title {
        text-style: bold;
        color: $text;
        text-align: center;
        margin-bottom: 0;
    }
    
    .wizard-subtitle {
        text-align: center;
        color: $text-muted;
        margin-top: 0;
    }
    
    .wizard-content {
        height: 1fr;
        padding: 2;
        overflow-y: auto;
    }
    
    .wizard-footer {
        dock: bottom;
        height: auto;
        padding: 1 2;
        background: $panel;
        border-top: solid $background-darken-1;
    }
    
    .wizard-buttons {
        layout: horizontal;
        height: auto;
        align: center middle;
    }
    
    .wizard-buttons Button {
        margin: 0 1;
        min-width: 16;
    }
    
    .wizard-buttons .spacer {
        width: 1fr;
    }
    """
    
    # Reactive properties
    current_step_index = reactive(0)
    can_go_back = reactive(False)
    can_go_forward = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize wizard."""
        super().__init__(**kwargs)
        self.app_instance_instance = app_instance
        self.steps: List[WizardStep] = []
        self.step_configs: List[WizardStepConfig] = []
        self.wizard_data: Dict[str, Any] = {}
        self.progress_indicator: Optional[StepProgress] = None
        
    @abstractmethod
    def get_wizard_title(self) -> str:
        """Get the wizard title."""
        pass
    
    @abstractmethod
    def get_wizard_subtitle(self) -> str:
        """Get the wizard subtitle."""
        pass
    
    @abstractmethod
    def create_steps(self) -> List[WizardStep]:
        """Create and return the wizard steps."""
        pass
    
    def compose(self) -> ComposeResult:
        """Compose the wizard UI."""
        # Create steps
        self.steps = self.create_steps()
        self.step_configs = [step.config for step in self.steps]
        
        with Container():
            # Header
            with Container(classes="wizard-header"):
                yield Static(self.get_wizard_title(), classes="wizard-title")
                yield Static(self.get_wizard_subtitle(), classes="wizard-subtitle")
                
                # Progress indicator
                self.progress_indicator = StepProgress(self.step_configs)
                yield self.progress_indicator
            
            # Content area
            with VerticalScroll(classes="wizard-content", id="wizard-content"):
                # All steps are added but only current one is visible
                for i, step in enumerate(self.steps):
                    step.display = i == 0  # Only first step visible initially
                    step.id = f"wizard-step-{i}"
                    yield step
            
            # Footer with navigation
            with Container(classes="wizard-footer"):
                with Horizontal(classes="wizard-buttons"):
                    yield Button("← Back", id="wizard-back", variant="default")
                    yield Button("Cancel", id="wizard-cancel", variant="default")
                    yield Static("", classes="spacer")
                    yield Button("Next →", id="wizard-next", variant="primary")
    
    async def on_mount(self) -> None:
        """Called when wizard is mounted."""
        await self._update_navigation_state()
        # Enter the first step
        if self.steps:
            await self.steps[0].on_enter()
    
    def watch_current_step_index(self, old_value: int, new_value: int) -> None:
        """Handle step changes."""
        if self.progress_indicator:
            self.progress_indicator.update_step(new_value)
        
        # Update navigation button states
        self.can_go_back = new_value > 0
        self.can_go_forward = new_value < len(self.steps) - 1
        
        # Update button labels
        if new_value == len(self.steps) - 1:
            next_button = self.query_one("#wizard-next", Button)
            next_button.label = "Finish"
            next_button.variant = "success"
        else:
            next_button = self.query_one("#wizard-next", Button)
            next_button.label = "Next →"
            next_button.variant = "primary"
    
    async def _update_navigation_state(self) -> None:
        """Update navigation button states."""
        back_button = self.query_one("#wizard-back", Button)
        back_button.disabled = not self.can_go_back
        
        # Check if current step can proceed
        current_step = self.steps[self.current_step_index]
        next_button = self.query_one("#wizard-next", Button)
        next_button.disabled = not current_step.can_proceed()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "wizard-cancel":
            self.dismiss(None)
            
        elif button_id == "wizard-back":
            await self._go_to_previous_step()
            
        elif button_id == "wizard-next":
            if self.current_step_index == len(self.steps) - 1:
                # Finish wizard
                await self._finish_wizard()
            else:
                await self._go_to_next_step()
    
    async def _go_to_previous_step(self) -> None:
        """Navigate to previous step."""
        if self.current_step_index > 0:
            # Leave current step
            current_step = self.steps[self.current_step_index]
            await current_step.on_leave()
            current_step.display = False
            
            # Go to previous step
            self.current_step_index -= 1
            previous_step = self.steps[self.current_step_index]
            previous_step.display = True
            await previous_step.on_enter()
            
            await self._update_navigation_state()
    
    async def _go_to_next_step(self) -> None:
        """Navigate to next step."""
        current_step = self.steps[self.current_step_index]
        
        # Validate current step
        is_valid, error_msg = current_step.validate()
        if not is_valid and not current_step.config.can_skip:
            self.app_instance.notify(error_msg or "Please complete this step before proceeding", severity="warning")
            return
        
        # Save step data
        step_data = current_step.get_step_data()
        self.wizard_data[current_step.config.id] = step_data
        
        # Leave current step
        await current_step.on_leave()
        current_step.display = False
        
        # Go to next step
        self.current_step_index += 1
        next_step = self.steps[self.current_step_index]
        next_step.display = True
        await next_step.on_enter()
        
        await self._update_navigation_state()
    
    async def _finish_wizard(self) -> None:
        """Complete the wizard."""
        # Get final step data
        final_step = self.steps[self.current_step_index]
        is_valid, error_msg = final_step.validate()
        
        if not is_valid:
            self.app_instance.notify(error_msg or "Please complete this step", severity="warning")
            return
        
        # Save final step data
        step_data = final_step.get_step_data()
        self.wizard_data[final_step.config.id] = step_data
        
        # Call the completion handler
        try:
            result = await self.on_wizard_complete(self.wizard_data)
            self.dismiss(result)
        except Exception as e:
            logger.error(f"Error completing wizard: {e}")
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
    
    @abstractmethod
    async def on_wizard_complete(self, wizard_data: Dict[str, Any]) -> Any:
        """
        Called when wizard is completed.
        
        Args:
            wizard_data: All data collected from wizard steps
            
        Returns:
            Result to return when dismissing the wizard
        """
        pass
    
    def refresh_current_step(self) -> None:
        """Refresh the current step's navigation state."""
        self.call_after_refresh(self._update_navigation_state)


class SimpleWizardStep(WizardStep):
    """Simple wizard step with just a container for content."""
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        """Initialize simple step."""
        super().__init__(wizard, config, **kwargs)
        self._content_container: Optional[Container] = None
        
    def compose(self) -> ComposeResult:
        """Compose step with container."""
        self._content_container = Container(id=f"step-content-{self.config.id}")
        yield self._content_container
    
    def get_step_data(self) -> Dict[str, Any]:
        """Default implementation returns empty dict."""
        return {}
    
    def mount_content(self, *widgets) -> None:
        """Mount widgets to the content container."""
        if self._content_container:
            for widget in widgets:
                self._content_container.mount(widget)