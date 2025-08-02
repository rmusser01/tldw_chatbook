# Wizard Widget Fix Plan

## Overview
This document outlines the comprehensive plan to fix the wizard widget implementation in the tldw_chatbook application. The fixes are based on Textual's official documentation and best practices.

## Key Findings from Textual Documentation

### 1. Widget Creation Best Practices
- Custom widgets should extend base classes like `Widget`, `Container`, or `Static`
- Use `DEFAULT_CSS` class variable for widget-specific styles
- Implement `compose()` method to yield child widgets
- Use reactive attributes for automatic UI updates
- Lifecycle methods: `on_mount()` (not `on_enter()`)

### 2. Layout Principles
- Use Container widgets (`Vertical`, `Horizontal`, `Grid`) for structure
- CSS layout properties: `layout: vertical/horizontal/grid`
- Proper nesting with context managers
- Use `fr` units for flexible sizing

### 3. CSS Implementation
- External CSS files with `.tcss` extension
- CSS_PATH class variable for linking stylesheets
- Class selectors with dot notation
- Support for nested CSS rules

### 4. Available Components
- ProgressBar: Built-in progress tracking widget
- ContentSwitcher: For switching between multiple views
- Button, Input, Checkbox, RadioSet: Form controls
- No built-in wizard component - must be custom built

## Identified Issues and Solutions

### Issue 1: Missing WizardStepConfig Class
**Problem**: Referenced but never defined, causing import errors.

**Solution**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class WizardStepConfig:
    id: str
    title: str
    description: str = ""
    icon: Optional[str] = None
    can_skip: bool = False
```

### Issue 2: Constructor Signature Mismatches
**Problem**: Inconsistent step initialization across wizards.

**Solution**: Standardize all wizard steps to use:
```python
def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
    super().__init__()
    self.wizard = wizard
    self.config = config
    self.step_number = 0  # Set by wizard
    self.step_title = config.title
    self.step_description = config.description
```

### Issue 3: Method Signature Conflicts
**Problem**: Different return types for validate() method.

**Solution**: Standardize to return tuple[bool, List[str]]:
```python
def validate(self) -> tuple[bool, List[str]]:
    """Returns (is_valid, list_of_error_messages)"""
    errors = []
    # validation logic
    return len(errors) == 0, errors
```

### Issue 4: Missing Property Access
**Problem**: Steps reference self.wizard but it's not stored.

**Solution**: 
- Store wizard reference in step __init__
- Add wizard_data property to WizardContainer
- Ensure app_instance is accessible through wizard

### Issue 5: Incorrect Method Names
**Problem**: Using on_enter() instead of on_show().

**Solution**: 
- Keep existing on_show()/on_hide() methods
- No async needed for these lifecycle methods
- Remove all async def on_enter() calls

### Issue 6: Missing Components
**Problem**: SmartContentTree imported but doesn't exist.

**Solution**: 
- Create simplified version using Tree widget
- Or replace with existing Tree widget functionality
- Fix import: BaseWizard → WizardContainer

### Issue 7: CSS Implementation
**Problem**: DEFAULT_CSS as class variables doesn't work as intended.

**Solution**:
- Move all CSS to external _wizards.tcss file
- Remove DEFAULT_CSS from Python classes
- Use proper class names in widgets

### Issue 8: Threading/Async Issues
**Problem**: Incorrect worker patterns in EmbeddingsWizard.

**Solution**:
```python
from textual.worker import work

@work(thread=True)
async def process_embeddings(self, config: Dict[str, Any]) -> None:
    # processing logic
    self.app.call_from_thread(self.update_ui, result)
```

### Issue 9: Progress Component
**Problem**: Missing imports and incorrect usage.

**Solution**:
```python
from textual.widgets import ProgressBar

# In compose():
yield ProgressBar(total=100, show_eta=True, show_percentage=True)

# To update:
progress_bar = self.query_one(ProgressBar)
progress_bar.update(progress=50)
```

### Issue 10: Dynamic Step Creation
**Problem**: Mounting steps after composition is fragile.

**Solution**:
- Create all steps upfront in wizard __init__
- Use visibility toggling instead of dynamic creation
- Implement step skipping logic if needed

## Implementation Order

### Phase 1: Core Infrastructure (Priority: Critical)
1. Create WizardStepConfig dataclass
2. Fix BaseWizard.py imports and method signatures
3. Standardize WizardStep base class methods
4. Add wizard_data management to WizardContainer

### Phase 2: Fix Existing Wizards (Priority: High)
1. Update ChatbookCreationWizard constructors
2. Update ChatbookImportWizard constructors
3. Fix validate() return types
4. Replace on_enter() with on_show()
5. Remove DEFAULT_CSS from all steps

### Phase 3: CSS and Layout (Priority: Medium)
1. Consolidate all CSS in _wizards.tcss
2. Update class names in widgets
3. Fix layout containers
4. Test responsive behavior

### Phase 4: Components and Features (Priority: Medium)
1. Implement simplified SmartContentTree
2. Fix ProgressBar usage
3. Update worker/threading patterns
4. Add proper error handling

### Phase 5: Testing and Polish (Priority: Low)
1. Add validation messages display
2. Implement keyboard navigation
3. Add transition animations
4. Test all wizard flows

## Code Templates

### Standard WizardStep Template
```python
class MyWizardStep(WizardStep):
    """Step description."""
    
    def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        # Step-specific initialization
        
    def compose(self) -> ComposeResult:
        """Compose the step UI."""
        with Container(classes="wizard-step-content"):
            yield Label(self.config.title, classes="step-title")
            yield Label(self.config.description, classes="step-description")
            # Step-specific widgets
            
    def validate(self) -> tuple[bool, List[str]]:
        """Validate step data."""
        errors = []
        # Validation logic
        return len(errors) == 0, errors
        
    def get_data(self) -> Dict[str, Any]:
        """Get step data."""
        return {
            # Collect data from widgets
        }
```

### Progress Update Pattern
```python
@work(thread=True)
async def long_running_task(self) -> None:
    """Run task in background."""
    for i in range(100):
        # Do work
        progress = i + 1
        self.app.call_from_thread(self.update_progress, progress)
        
def update_progress(self, value: int) -> None:
    """Update UI from worker thread."""
    progress_bar = self.query_one("#my-progress", ProgressBar)
    progress_bar.update(progress=value)
```

## Next Steps

1. Review this plan with the team
2. Create feature branch for wizard fixes
3. Implement Phase 1 infrastructure changes
4. Test basic wizard functionality
5. Proceed with remaining phases

## Notes

- Keep backward compatibility where possible
- Add deprecation warnings for old patterns
- Document new wizard creation process
- Consider creating example wizard for reference

## Additional Findings from Code Analysis

### Usage Patterns
1. **ChatbookCreationWizard** is instantiated with:
   - `ChatbookCreationWizard(self.app)` - single app parameter
   - `ChatbookCreationWizard(self.app, template_data=result)` - with optional template_data
   - Also used as `ChatbookCreationWizard(self.app_instance)` in older code

2. **ChatbookImportWizard** is instantiated with:
   - `ChatbookImportWizard(self.app)` - single app parameter
   - `ChatbookImportWizard(self.app_instance)` in older code

3. **EmbeddingsWizard** variants:
   - `SimpleEmbeddingsWizard()` - no parameters
   - Used directly in compose() method of Embeddings_Window

### Critical Issues Confirmed
1. **WizardStepConfig Missing**: All chatbook wizard steps expect this class but it doesn't exist
2. **Constructor Mismatch**: Wizards expect app instance, but base class doesn't handle it
3. **Method Name Issues**: Steps use `on_enter()` which should be `on_show()`
4. **Missing SmartContentTree**: Referenced but not implemented
5. **CSS in Python**: DEFAULT_CSS class variables won't work as intended

### Updated Constructor Requirements

For WizardContainer:
```python
def __init__(
    self,
    app_instance,  # Add this parameter
    steps: List[WizardStep] = None,  # Make optional for dynamic creation
    title: str = "Wizard",
    template_data: Optional[Dict] = None,  # For pre-filled wizards
    *args,
    **kwargs
):
    super().__init__(*args, **kwargs)
    self.app_instance = app_instance
    self.template_data = template_data or {}
    self.wizard_data = {}  # Store data between steps
    # ... rest of initialization
```

For WizardStep:
```python
def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
    super().__init__(**kwargs)
    self.wizard = wizard
    self.config = config
    self.step_number = config.step_number if hasattr(config, 'step_number') else 0
    self.step_title = config.title
    self.step_description = config.description
```

### Screen vs Container Issue
The wizards are used as Screens (pushed with `push_screen`), but WizardContainer extends Container.
This needs to be fixed by either:
1. Making WizardContainer extend Screen
2. Creating a WizardScreen wrapper
3. Using ModalScreen as shown in EmbeddingsWizardScreen

### Recommended Approach
Create a WizardScreen base class that wraps WizardContainer:
```python
class WizardScreen(Screen):
    """Screen wrapper for wizards."""
    
    def __init__(self, app_instance, **wizard_kwargs):
        super().__init__()
        self.app_instance = app_instance
        self.wizard_kwargs = wizard_kwargs
        
    def compose(self) -> ComposeResult:
        # Subclasses should override and yield their wizard
        pass
```

## Final Implementation Plan

### 1. Create Core Classes (BaseWizard.py additions)

```python
from dataclasses import dataclass
from typing import Optional
from textual.screen import Screen

@dataclass
class WizardStepConfig:
    """Configuration for a wizard step."""
    id: str
    title: str
    description: str = ""
    icon: Optional[str] = None
    can_skip: bool = False
    step_number: int = 0

class WizardScreen(Screen):
    """Base screen class for wizards."""
    
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
```

### 2. Update WizardContainer Constructor

```python
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
```

### 3. Fix Chatbook Wizards

Update ChatbookCreationWizard and ChatbookImportWizard to extend WizardScreen:

```python
class ChatbookCreationWizard(WizardScreen):
    """Chatbook creation wizard screen."""
    
    def compose(self) -> ComposeResult:
        wizard = ChatbookCreationWizardContainer(
            self.app_instance,
            **self.wizard_kwargs
        )
        yield wizard

class ChatbookCreationWizardContainer(WizardContainer):
    """The actual wizard implementation."""
    
    def __init__(self, app_instance, template_data=None, **kwargs):
        # Create steps with proper config
        steps = self._create_steps()
        
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title="Create New Chatbook",
            template_data=template_data,
            **kwargs
        )
        
    def _create_steps(self) -> List[WizardStep]:
        """Create wizard steps with proper configuration."""
        return [
            BasicInfoStep(
                wizard=self,
                config=WizardStepConfig(
                    id="basic-info",
                    title="Basic Information",
                    description="Enter chatbook details",
                    step_number=1
                )
            ),
            # ... other steps
        ]
```

### 4. Fix Step Base Class

Update WizardStep to properly handle wizard reference:

```python
class WizardStep(Container):
    """Base class for individual wizard steps."""
    
    def __init__(
        self,
        wizard: WizardContainer = None,  # Make optional for compatibility
        config: WizardStepConfig = None,  # Make optional for compatibility
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
```

### 5. Remove CSS from Python Files

Move all DEFAULT_CSS content to _wizards.tcss and remove from Python files.

### 6. Fix Method Names

Search and replace in all wizard files:
- `async def on_enter(` → `def on_show(`
- `await super().on_enter()` → `super().on_show()`
- Remove all `async` keywords from these methods

### 7. Create SmartContentTree Stub

Create a minimal implementation or replace with Tree widget:

```python
# In Widgets/SmartContentTree.py
from textual.widgets import Tree
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    CONVERSATION = "conversation"
    NOTE = "note"
    CHARACTER = "character"
    MEDIA = "media"
    PROMPT = "prompt"

@dataclass
class ContentNodeData:
    type: ContentType
    id: str
    title: str
    subtitle: str = ""

class SmartContentTree(Tree):
    """Tree widget for content selection."""
    
    def __init__(self, load_content=None, **kwargs):
        super().__init__("Content", **kwargs)
        self.load_content = load_content
        self._selections = {}
        
    def get_selections(self):
        """Get selected content."""
        return self._selections
```

## Testing Plan

1. Test WizardStepConfig creation
2. Test WizardScreen instantiation
3. Test basic wizard flow (next/back)
4. Test data collection between steps
5. Test validation errors
6. Test cancel functionality
7. Test complete wizard flow
8. Test with template data

## Migration Notes

- Old wizard instantiation will need updates
- CSS class names may need adjustment
- Some reactive attributes may need fixes
- Worker thread patterns need review