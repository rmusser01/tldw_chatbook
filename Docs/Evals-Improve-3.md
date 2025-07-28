# Evals Window UI Improvement Guide - Version 3

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Understanding Textual Layouts](#understanding-textual-layouts)
3. [Current Implementation Analysis](#current-implementation-analysis)
4. [Architectural Decision Records (ADRs)](#architectural-decision-records)
5. [New Architecture Overview](#new-architecture-overview)
6. [Implementation Details](#implementation-details)
7. [Migration Guide](#migration-guide)

## Executive Summary

This document outlines the complete refactoring of the Evaluation Window UI in the tldw_chatbook application. The current implementation, described as a "hack job" by an intern, suffers from poor separation of concerns, excessive complexity, and maintenance difficulties. This refactoring transforms it into a clean, modular, and maintainable UI following Textual best practices.

### Key Changes
- **Modular Architecture**: Breaking down the 800+ line monolithic file into focused components
- **State Management**: Consolidating 20+ reactive properties into a single, well-structured state object
- **Reusable Components**: Creating a library of common UI patterns
- **Clean Event Handling**: Moving from external delegation to proper message passing
- **Improved Error Handling**: Removing excessive try/except blocks in favor of proper error boundaries

## Understanding Textual Layouts

### What is Textual?

Textual is a Python framework for building sophisticated Terminal User Interfaces (TUIs). It uses a reactive programming model similar to modern web frameworks but renders to the terminal.

### Core Layout Concepts

#### 1. **The Widget Tree**
```
Application
└── Screen
    └── Container (root)
        ├── Header
        ├── Container (main content)
        │   ├── Sidebar
        │   └── ContentArea
        └── Footer
```

Every Textual app is a tree of widgets. Widgets can contain other widgets, creating a hierarchy.

#### 2. **Layout Types**

**Vertical Layout** (default):
```python
with Container():
    yield Widget1()  # Stacks vertically
    yield Widget2()  # Below Widget1
    yield Widget3()  # Below Widget2
```

**Horizontal Layout**:
```python
with Horizontal():
    yield Widget1()  # Side by side
    yield Widget2()  # To the right of Widget1
```

**Grid Layout**:
```python
Container {
    layout: grid;
    grid-size: 3 2;  /* 3 columns, 2 rows */
    grid-columns: 1fr 2fr 1fr;  /* Column sizing */
}
```

**Dock Layout**:
```python
Widget {
    dock: top;  /* Docks to top of parent */
    height: 5;
}
```

#### 3. **The compose() Method**

The `compose()` method is where you declaratively define your UI structure:

```python
def compose(self) -> ComposeResult:
    """Build the widget tree."""
    yield Header()
    with Container(id="main"):
        yield Sidebar()
        yield ContentArea()
    yield Footer()
```

#### 4. **Reactive Properties**

Reactive properties automatically update the UI when changed:

```python
class MyWidget(Widget):
    count = reactive(0)  # When count changes, watchers are called
    
    def watch_count(self, old_value: int, new_value: int):
        """Called when count changes."""
        self.update_display()
```

#### 5. **CSS in Textual (TCSS)**

Textual uses a subset of CSS for styling:

```css
Container {
    background: $surface;
    padding: 1 2;  /* vertical horizontal */
    border: solid $primary;
}

.my-class {
    color: $text;
    text-style: bold italic;
}

#my-id {
    width: 50%;
    height: 10;
}
```

### Layout Best Practices

1. **Use Containers for Organization**: Group related widgets
2. **Leverage CSS Grid**: For complex layouts
3. **Minimize Nesting**: Deep nesting makes code hard to follow
4. **Use Semantic IDs and Classes**: For easier styling and querying
5. **Keep compose() Simple**: Extract complex sections into methods

## Current Implementation Analysis

### Problems Identified

#### 1. **Monolithic compose() Method**
The current compose() method is over 200 lines long, making it difficult to understand the UI structure at a glance.

```python
def compose(self) -> ComposeResult:
    # 200+ lines of nested containers and widgets
    # Impossible to see the overall structure
```

#### 2. **Event Handler Anti-Pattern**
Every button handler just delegates to external functions:

```python
@on(Button.Pressed, "#upload-task-btn")
def handle_upload_task(self, event: Button.Pressed) -> None:
    from ..Event_Handlers.eval_events import handle_upload_task
    handle_upload_task(self.app_instance, event)  # Just delegates!
```

This creates tight coupling and makes testing difficult.

#### 3. **Excessive Try/Except Blocks**
The code is littered with try/except blocks around UI queries:

```python
try:
    status_element = self.query_one(f"#{status_id}")
    status_element.update(message)
except QueryError:
    logger.warning(f"Status element not found: {status_id}")
```

This suggests the UI structure is fragile and timing-dependent.

#### 4. **State Management Chaos**
20+ reactive properties scattered throughout:

```python
evals_sidebar_collapsed: reactive[bool] = reactive(False)
evals_active_view: reactive[Optional[str]] = reactive(None)
current_run_status: reactive[str] = reactive("idle")
active_run_id: reactive[Optional[str]] = reactive(None)
is_loading_results = reactive(False)
is_loading_models = reactive(False)
is_loading_datasets = reactive(False)
# ... and many more
```

#### 5. **Poor Separation of Concerns**
Business logic mixed with UI code:

```python
@work(exclusive=True)
async def _refresh_results_dashboard(self) -> None:
    # UI widget directly calling database operations
    recent_runs = await self.app.run_in_executor(None, get_recent_evaluations, self.app_instance)
```

## Architectural Decision Records (ADRs)

### ADR-001: Component-Based Architecture

**Status**: Accepted

**Context**: The current monolithic structure makes the code difficult to maintain and extend.

**Decision**: Adopt a component-based architecture where each logical section of the UI is a self-contained widget.

**Consequences**:
- ✅ Better code organization
- ✅ Easier testing
- ✅ Reusable components
- ❌ More files to manage
- ❌ Initial refactoring effort

### ADR-002: Centralized State Management

**Status**: Accepted

**Context**: Multiple reactive properties make state management complex and error-prone.

**Decision**: Use a single state object (EvaluationState) to manage all application state.

**Consequences**:
- ✅ Single source of truth
- ✅ Easier debugging
- ✅ Predictable state updates
- ❌ Need to refactor all state access

### ADR-003: Message-Based Communication

**Status**: Accepted

**Context**: Direct method calls between components create tight coupling.

**Decision**: Use Textual's message system for all inter-component communication.

**Consequences**:
- ✅ Loose coupling
- ✅ Better testability
- ✅ Clear data flow
- ❌ Slightly more verbose

### ADR-004: Separation of UI and Business Logic

**Status**: Accepted

**Context**: Business logic embedded in UI components makes testing and maintenance difficult.

**Decision**: Create a clear separation with UI components only handling presentation and user interaction.

**Consequences**:
- ✅ Testable business logic
- ✅ UI components focused on presentation
- ✅ Easier to change either layer independently
- ❌ Need to create service layer

### ADR-005: Reusable Widget Library

**Status**: Accepted

**Context**: Common UI patterns are repeated throughout the codebase.

**Decision**: Create a library of reusable base widgets for common patterns.

**Consequences**:
- ✅ Consistent UI patterns
- ✅ Less code duplication
- ✅ Faster development
- ❌ Need to maintain widget library

## New Architecture Overview

### Component Hierarchy

```
EvalsWindow (Main Container)
├── EvalsSidebar (Navigation)
│   ├── SidebarHeader
│   ├── NavigationButton (Setup)
│   ├── NavigationButton (Results)
│   ├── NavigationButton (Models)
│   └── NavigationButton (Datasets)
├── ContentArea
│   ├── ContentHeader
│   │   ├── SidebarToggle
│   │   └── ViewTitle
│   └── ViewContainer
│       ├── EvaluationSetupView
│       ├── ResultsDashboardView
│       ├── ModelManagementView
│       └── DatasetManagementView
└── StatusBar (Optional)
```

### State Management

```python
@dataclass
class EvaluationState:
    """Centralized state for evaluation system."""
    # Navigation
    active_view: str = "setup"
    sidebar_collapsed: bool = False
    
    # Evaluation Run
    current_run: Optional[EvaluationRun] = None
    run_status: RunStatus = RunStatus.IDLE
    
    # Data
    available_models: List[Model] = field(default_factory=list)
    available_datasets: List[Dataset] = field(default_factory=list)
    recent_results: List[EvaluationResult] = field(default_factory=list)
    
    # UI State
    loading_states: Dict[str, bool] = field(default_factory=dict)
    error_messages: Dict[str, str] = field(default_factory=dict)
```

### Message Flow

```
User clicks "Start Evaluation"
    ↓
Button posts StartEvaluationRequested message
    ↓
EvalsWindow receives message
    ↓
EvalsWindow updates state
    ↓
State change triggers UI update
    ↓
EvalsWindow posts EvaluationStarted message
    ↓
Service layer handles evaluation
```

## Implementation Details

### Base Widget Library

#### SectionContainer
A reusable container for consistent section styling:

```python
class SectionContainer(Container):
    """A styled container for UI sections."""
    
    def __init__(
        self, 
        title: str,
        *children,
        collapsible: bool = False,
        classes: str = "",
        **kwargs
    ):
        super().__init__(classes=f"section-container {classes}", **kwargs)
        self.title = title
        self._children = children
        self.collapsible = collapsible
        self._collapsed = False
    
    def compose(self) -> ComposeResult:
        with Container(classes="section-header"):
            yield Static(self.title, classes="section-title")
            if self.collapsible:
                yield Button("▼", classes="collapse-button")
        
        with Container(classes="section-content"):
            yield from self._children
```

#### ActionButtonRow
Standardized button group layout:

```python
class ActionButtonRow(Horizontal):
    """A row of action buttons with consistent styling."""
    
    def __init__(
        self,
        buttons: List[ButtonConfig],
        classes: str = "",
        **kwargs
    ):
        super().__init__(classes=f"action-button-row {classes}", **kwargs)
        self.buttons = buttons
    
    def compose(self) -> ComposeResult:
        for config in self.buttons:
            yield Button(
                config.label,
                id=config.id,
                variant=config.variant,
                disabled=config.disabled,
                classes="action-button"
            )
```

#### StatusDisplay
Consistent status message display:

```python
class StatusDisplay(Static):
    """Displays status messages with appropriate styling."""
    
    status_text = reactive("")
    status_type = reactive("info")
    
    def __init__(self, initial_text: str = "", **kwargs):
        super().__init__(initial_text, **kwargs)
        self.status_text = initial_text
    
    def watch_status_text(self, text: str):
        self.update(text)
    
    def watch_status_type(self, type: str):
        self.remove_class("info", "success", "warning", "error")
        self.add_class(type)
    
    def set_status(self, text: str, type: str = "info"):
        self.status_text = text
        self.status_type = type
```

### View Components

#### EvaluationSetupView
Handles task and model configuration:

```python
class EvaluationSetupView(Container):
    """View for setting up evaluations."""
    
    def compose(self) -> ComposeResult:
        # Task Configuration
        yield SectionContainer(
            "Task Configuration",
            ActionButtonRow([
                ButtonConfig("Upload Task File", "upload-task"),
                ButtonConfig("Create New Task", "create-task"),
            ]),
            StatusDisplay(id="task-status")
        )
        
        # Model Configuration
        yield SectionContainer(
            "Model Configuration",
            ModelSelector(id="model-selector"),
            StatusDisplay(id="model-status")
        )
        
        # Run Configuration
        yield SectionContainer(
            "Run Configuration",
            ConfigurationForm([
                FormField("max_samples", "Max Samples", "number"),
                FormField("temperature", "Temperature", "number"),
                FormField("timeout", "Timeout (seconds)", "number"),
            ]),
            ActionButton("Start Evaluation", id="start-eval", variant="primary")
        )
        
        # Progress Tracking
        yield SectionContainer(
            "Progress",
            ProgressTracker(id="progress-tracker"),
            collapsible=True
        )
```

### Event Handling

Instead of delegating to external handlers, handle events internally and post messages:

```python
class EvalsWindow(Container):
    """Main evaluation window with clean event handling."""
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses internally."""
        button_id = event.button.id
        
        if button_id == "start-eval":
            self._handle_start_evaluation()
        elif button_id == "upload-task":
            self._handle_upload_task()
        # ... etc
    
    def _handle_start_evaluation(self) -> None:
        """Handle evaluation start request."""
        # Validate state
        if not self._validate_evaluation_config():
            self.post_message(
                ShowError("Please complete configuration")
            )
            return
        
        # Update UI state
        self.state.run_status = RunStatus.STARTING
        
        # Post message for service layer
        self.post_message(
            StartEvaluationRequested(
                config=self._build_evaluation_config()
            )
        )
```

### CSS Organization

The new CSS file is organized into logical sections:

```css
/* ========================================
 * Evaluation System v3
 * ======================================== */

/* --- Layout Structure --- */
.evals-window {
    layout: grid;
    grid-size: 2;
    grid-columns: 250 1fr;
}

/* --- Component Styles --- */

/* Sidebar */
.sidebar {
    background: $surface;
    border-right: solid $primary;
}

.sidebar.collapsed {
    width: 0;
    visibility: hidden;
}

/* Section Containers */
.section-container {
    background: $panel;
    margin: 0 0 1 0;
    padding: 1;
    border: round $secondary;
}

.section-header {
    layout: horizontal;
    height: 3;
    margin-bottom: 1;
}

.section-title {
    text-style: bold;
    width: 1fr;
}

/* Buttons */
.action-button {
    margin: 0 1 0 0;
}

.action-button:last-child {
    margin-right: 0;
}

/* Status Messages */
.status-display {
    padding: 0 1;
    height: 3;
}

.status-display.info {
    background: $primary 20%;
}

.status-display.success {
    background: $success 20%;
}

.status-display.warning {
    background: $warning 20%;
}

.status-display.error {
    background: $error 20%;
}

/* --- Responsive Behavior --- */
.sidebar-collapsed .evals-window {
    grid-columns: 0 1fr;
}
```

## Migration Guide

### Step 1: Create New Widget Library
1. Create `tldw_chatbook/Widgets/base_components.py`
2. Implement base widgets (SectionContainer, ActionButtonRow, etc.)
3. Write unit tests for each widget

### Step 2: Create State Management
1. Create `tldw_chatbook/Models/evaluation_state.py`
2. Define EvaluationState dataclass
3. Create state update methods

### Step 3: Create View Components
1. Create `tldw_chatbook/UI/Views/` directory
2. Implement each view as a separate file
3. Keep views focused on presentation

### Step 4: Refactor Main Window
1. Create `tldw_chatbook/UI/Evals_Window_v2.py`
2. Start with basic structure
3. Gradually migrate functionality

### Step 5: Update Event Handlers
1. Move logic from eval_events.py into window
2. Create message classes for communication
3. Update service layer to use messages

### Step 6: Testing
1. Test each component in isolation
2. Test state management
3. Test message flow
4. Integration testing

### Migration Checklist
- [ ] Base widget library complete
- [ ] State management implemented
- [ ] View components created
- [ ] Main window refactored
- [ ] Event handlers updated
- [ ] CSS migrated and cleaned
- [ ] Unit tests written
- [ ] Integration tests passing
- [ ] Old code removed
- [ ] Documentation updated

## Conclusion

This refactoring transforms the Evals window from a difficult-to-maintain "hack job" into a clean, modular, and extensible UI component. By following Textual best practices and modern UI architecture patterns, the new implementation will be:

1. **Easier to Understand**: Clear component hierarchy and data flow
2. **Easier to Maintain**: Isolated components with single responsibilities
3. **Easier to Extend**: New features can be added without affecting existing code
4. **More Reliable**: Proper error handling and state management
5. **Better Performing**: Efficient updates through proper use of reactive properties

The investment in this refactoring will pay dividends in reduced bugs, faster feature development, and a better developer experience.