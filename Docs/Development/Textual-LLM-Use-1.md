# Textual Framework Reference Guide for LLMs

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Widget System](#widget-system)
4. [Styling with CSS](#styling-with-css)
5. [Layout Systems](#layout-systems)
6. [Event Handling](#event-handling)
7. [Reactive Programming](#reactive-programming)
8. [Screen Management](#screen-management)
9. [Workers and Concurrency](#workers-and-concurrency)
10. [Actions and Input](#actions-and-input)
11. [Testing](#testing)
12. [Built-in Widgets](#built-in-widgets)
13. [Best Practices](#best-practices)

## Introduction

Textual is a Python framework for building sophisticated Terminal User Interfaces (TUIs). It's inspired by modern web development and provides a reactive UI model with CSS-like styling.

### Key Features
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Remote capable**: Can run over SSH
- **Low requirements**: Runs on single-board computers
- **Rich rendering**: Leverages the Rich library for beautiful output
- **Async support**: Built on Python's asyncio
- **CSS styling**: Familiar web-like styling system
- **Testing support**: Built-in testing framework

### Installation
```bash
pip install textual
# For development features:
pip install textual[dev]
```

## Core Concepts

### The App Class

Every Textual application starts with an App class:

```python
from textual.app import App
from textual.widgets import Label

class MyApp(App):
    def compose(self):
        """Define the initial widgets."""
        yield Label("Hello, Textual!")
    
    def on_mount(self):
        """Called after the app is mounted."""
        pass

# Run the app
if __name__ == "__main__":
    app = MyApp()
    app.run()
```

#### Key App Methods
- `compose()`: Define initial widget layout
- `on_mount()`: Lifecycle method called after app starts
- `run()`: Start the application
- `exit(result=None)`: Exit the app with optional result
- `push_screen()`, `pop_screen()`, `switch_screen()`: Screen management

### App Configuration
```python
class MyApp(App):
    CSS_PATH = "styles.tcss"  # External CSS file
    BINDINGS = [("q", "quit", "Quit")]  # Key bindings
    TITLE = "My Application"  # App title
    SUB_TITLE = "Version 1.0"  # App subtitle
```

## Widget System

### Creating Custom Widgets

Widgets are the building blocks of Textual UIs:

```python
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text

class Counter(Widget):
    """A simple counter widget."""
    
    count = reactive(0)  # Reactive attribute
    
    def render(self):
        """Render the widget content."""
        return Text(f"Count: {self.count}")
    
    def on_click(self):
        """Handle click events."""
        self.count += 1
```

### Widget Lifecycle
1. **Creation**: Widget instantiated
2. **Mounting**: Added to the DOM via `mount()` or `compose()`
3. **Rendering**: `render()` called to display content
4. **Updates**: Reactive changes trigger re-renders
5. **Unmounting**: Removed from DOM

### Key Widget Properties
- `id`: Unique identifier for CSS and queries
- `classes`: CSS classes for styling
- `styles`: Inline styles
- `can_focus`: Whether widget can receive keyboard focus
- `disabled`: Whether widget is disabled

### Static Widgets

For simple content, use the `Static` widget:

```python
from textual.widgets import Static

class MyWidget(Static):
    DEFAULT_CSS = """
    MyWidget {
        background: $primary;
        padding: 1;
    }
    """
    
    def compose(self):
        yield Static("Simple content")
```

## Styling with CSS

Textual uses a CSS-like syntax adapted for terminal interfaces:

### Basic Syntax
```css
/* styles.tcss */
Screen {
    background: $surface;
}

#my-id {
    border: solid $primary;
    padding: 1 2;
}

.my-class {
    color: $text;
    text-style: bold;
}

Button {
    width: 16;
    margin: 1 2;
}

Button:hover {
    background: $primary-darken-1;
}

Button:focus {
    text-style: bold reverse;
}
```

### Selectors
- **Type selector**: `Button` (matches widget type)
- **ID selector**: `#submit` (matches widget ID)
- **Class selector**: `.primary` (matches CSS classes)
- **Universal selector**: `*` (matches all widgets)
- **Descendant selector**: `Container Button`
- **Child selector**: `Container > Button`

### Pseudo-classes
- `:hover` - Mouse over widget
- `:focus` - Widget has focus
- `:disabled` - Widget is disabled
- `:enabled` - Widget is enabled

### CSS Variables
```css
$primary: #0066cc;
$background: #1e1e1e;

Button {
    background: $primary;
}
```

### Nested CSS
```css
#container {
    padding: 1;
    
    .button {
        margin: 1;
        
        &:hover {
            background: $accent;
        }
    }
}
```

## Layout Systems

### Vertical Layout (Default)
```python
class MyScreen(Screen):
    def compose(self):
        yield Header()
        yield Label("Top")
        yield Label("Middle")
        yield Label("Bottom")
        yield Footer()
```

### Horizontal Layout
```python
from textual.containers import Horizontal

class MyWidget(Widget):
    def compose(self):
        with Horizontal():
            yield Button("Left")
            yield Button("Center")
            yield Button("Right")
```

### Grid Layout
```python
from textual.containers import Grid

class MyGrid(Widget):
    def compose(self):
        with Grid():
            yield Label("Cell 1")
            yield Label("Cell 2")
            yield Label("Cell 3")
            yield Label("Cell 4")
```

CSS for grid:
```css
Grid {
    grid-size: 2 2;  /* 2 columns, 2 rows */
    grid-columns: 1fr 2fr;  /* Column sizes */
    grid-rows: 1fr 1fr;  /* Row sizes */
    grid-gutter: 1;  /* Gap between cells */
}
```

### Docking
```css
Header {
    dock: top;
    height: 3;
}

Sidebar {
    dock: left;
    width: 30;
}
```

### Flexible Units
- `fr`: Fractional units (e.g., `1fr`, `2fr`)
- `%`: Percentage of parent
- `vh/vw`: Viewport height/width
- Fixed units: Numbers represent character cells

## Event Handling

### Event Methods
```python
class MyWidget(Widget):
    def on_click(self, event):
        """Handle click events."""
        print(f"Clicked at {event.x}, {event.y}")
    
    def on_key(self, event):
        """Handle key events."""
        if event.key == "enter":
            self.submit()
    
    def on_focus(self):
        """Widget gained focus."""
        self.add_class("focused")
    
    def on_blur(self):
        """Widget lost focus."""
        self.remove_class("focused")
```

### Custom Messages
```python
from textual.message import Message

class ColorChanged(Message):
    """Custom message for color changes."""
    def __init__(self, color: str):
        super().__init__()
        self.color = color

class ColorPicker(Widget):
    def select_color(self, color: str):
        # Post message to parent
        self.post_message(ColorChanged(color))

class MyApp(App):
    def on_color_changed(self, message: ColorChanged):
        """Handle custom message."""
        self.styles.background = message.color
```

### Event Bubbling
Events bubble up through the widget hierarchy. Use `stop()` to prevent bubbling:

```python
def on_click(self, event):
    # Handle event
    event.stop()  # Prevent bubbling
```

### Message Handler Decorator
```python
from textual import on

class MyWidget(Widget):
    @on(Button.Pressed, "#submit")
    def handle_submit(self):
        """Handle submit button press."""
        self.submit_form()
```

## Reactive Programming

### Reactive Attributes
```python
from textual.reactive import reactive

class Timer(Widget):
    time = reactive(0.0)
    running = reactive(False)
    
    def watch_time(self, new_time: float):
        """Called when time changes."""
        self.refresh()  # Update display
    
    def validate_time(self, new_time: float) -> float:
        """Validate time before setting."""
        return max(0.0, new_time)
    
    def compute_display(self) -> str:
        """Compute derived value."""
        minutes = int(self.time // 60)
        seconds = int(self.time % 60)
        return f"{minutes:02d}:{seconds:02d}"
```

### Reactive Parameters
- `init=False`: Don't trigger watchers on initialization
- `layout=True`: Trigger layout refresh on change
- `recompose=True`: Rebuild widget tree on change
- `always_update=True`: Always trigger watchers even if value unchanged

### Data Binding
```python
class MyApp(App):
    def compose(self):
        # Bind input value to label
        input = Input(id="name")
        label = Label()
        input.bind("value", label, "content")
        yield input
        yield label
```

## Screen Management

### Creating Screens
```python
from textual.screen import Screen

class MainMenu(Screen):
    def compose(self):
        yield Static("Main Menu", id="title")
        yield Button("New Game", id="new")
        yield Button("Load Game", id="load")
        yield Button("Quit", id="quit")
    
    @on(Button.Pressed, "#new")
    def new_game(self):
        self.app.push_screen(GameScreen())
```

### Modal Screens
```python
from textual.screen import ModalScreen

class Dialog(ModalScreen):
    """A modal dialog."""
    
    DEFAULT_CSS = """
    Dialog {
        align: center middle;
    }
    
    Dialog > Container {
        width: 60;
        height: 20;
        border: thick $primary;
        background: $surface;
    }
    """
    
    def compose(self):
        with Container():
            yield Label("Are you sure?")
            with Horizontal():
                yield Button("Yes", id="yes")
                yield Button("No", id="no")
    
    @on(Button.Pressed)
    def handle_button(self, event):
        self.dismiss(event.button.id == "yes")
```

### Screen Stack Operations
```python
# Push new screen
self.app.push_screen(MyScreen())

# Pop current screen
self.app.pop_screen()

# Switch (replace) current screen
self.app.switch_screen(NewScreen())

# Push and wait for result (in worker)
@work
async def get_user_input(self):
    result = await self.app.push_screen_wait(InputDialog())
    self.process_input(result)
```

### Screen Modes
```python
class MyApp(App):
    MODES = {
        "main": MainScreen,
        "settings": SettingsScreen,
        "help": HelpScreen,
    }
    
    def on_mount(self):
        self.switch_mode("main")
```

## Workers and Concurrency

### Using Workers
```python
from textual.worker import work

class DataLoader(Widget):
    @work(exclusive=True)
    async def load_data(self, url: str):
        """Load data in background."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            self.display_data(data)
    
    def on_mount(self):
        self.load_data("https://api.example.com/data")
```

### Thread Workers
```python
@work(thread=True)
def process_file(self, path: str):
    """CPU-intensive work in thread."""
    with open(path) as f:
        result = expensive_processing(f.read())
    
    # Update UI from thread
    self.call_from_thread(self.update_display, result)
```

### Worker Lifecycle
```python
class MyWidget(Widget):
    def on_worker_state_changed(self, event):
        """Monitor worker state changes."""
        worker = event.worker
        if worker.state == WorkerState.SUCCESS:
            self.notify("Task completed!")
        elif worker.state == WorkerState.ERROR:
            self.notify("Task failed!", severity="error")
```

### Worker Control
```python
# Start worker
worker = self.run_worker(self.background_task())

# Cancel worker
worker.cancel()

# Wait for completion
await worker.wait()

# Check if cancelled
if worker.is_cancelled:
    return
```

## Actions and Input

### Defining Actions
```python
class MyApp(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        Binding("ctrl+s", "save", "Save", show=False),
    ]
    
    def action_save(self):
        """Save action."""
        self.save_data()
    
    def action_toggle_dark(self):
        """Toggle dark mode."""
        self.dark = not self.dark
```

### Dynamic Actions
```python
def check_action(self, action: str, parameters: tuple) -> bool | None:
    """Enable/disable actions dynamically."""
    if action == "save":
        # Only enable if there are changes
        return self.has_changes
    return True
```

### Key Bindings
```python
class MyWidget(Widget):
    BINDINGS = [
        Binding("enter", "select", "Select item"),
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def key_space(self):
        """Handle space key directly."""
        self.toggle()
```

### Focus Management
```python
# Set focus
self.query_one("#input").focus()

# Check focus
if self.has_focus:
    self.highlight()

# Focus next/previous
self.app.action_focus_next()
self.app.action_focus_previous()
```

### Command Palette
```python
from textual.command import Provider

class MyCommands(Provider):
    async def search(self, query: str):
        """Provide searchable commands."""
        matcher = self.matcher(query)
        
        commands = [
            ("Open file", "open_file"),
            ("Save file", "save_file"),
            ("Close file", "close_file"),
        ]
        
        for name, action in commands:
            if matcher.match(name):
                yield (name, action)
```

## Testing

https://github.com/Textualize/pytest-textual-snapshot

Textual provides a comprehensive testing framework built on pytest and async testing patterns. While testing isn't mandatory, it's strongly recommended to catch bugs early and ensure application reliability.

### Testing Setup

First install the required dependencies:
```bash
pip install pytest pytest-asyncio pytest-textual-snapshot
```

### Core Testing Concepts

#### App Testing with run_test()
The `run_test()` method creates a test harness that simulates a running Textual app:

```python
import pytest
from textual.app import App
from textual.widgets import Button, Label

class CounterApp(App):
    def __init__(self):
        super().__init__()
        self.counter = 0
    
    def compose(self):
        yield Label(f"Count: {self.counter}", id="counter")
        yield Button("Increment", id="increment")
    
    def on_button_pressed(self, event):
        if event.button.id == "increment":
            self.counter += 1
            self.query_one("#counter").update(f"Count: {self.counter}")

@pytest.mark.asyncio
async def test_counter_app():
    app = CounterApp()
    async with app.run_test() as pilot:
        # Test initial state
        counter_label = app.query_one("#counter")
        assert counter_label.renderable == "Count: 0"
        
        # Simulate button click
        await pilot.click("#increment")
        
        # Verify state change
        assert app.counter == 1
        assert counter_label.renderable == "Count: 1"
```

#### The Pilot Object
The `pilot` object provides methods to interact with your app during testing:

```python
async def test_pilot_interactions(app):
    async with app.run_test() as pilot:
        # Key presses
        await pilot.press("enter")          # Single key
        await pilot.press("ctrl+c")         # Key combination
        await pilot.press("tab", "tab")     # Multiple keys
        await pilot.press(*"Hello")         # Type text
        
        # Mouse interactions
        await pilot.click("#button-id")     # Click by CSS selector
        await pilot.click("Button")         # Click by widget type
        await pilot.click(10, 5)           # Click coordinates
        await pilot.hover("#widget")        # Hover over widget
        
        # Wait for async operations
        await pilot.pause()                 # Process pending messages
        await pilot.pause(0.1)             # Wait specific time
        
        # Screen size simulation
        pilot.resize_terminal(120, 40)      # Set terminal size
```

### Advanced Testing Patterns

#### Form Testing
Test complex forms with multiple inputs and validation:

```python
@pytest.mark.asyncio
async def test_media_ingestion_form():
    from tldw_chatbook.Widgets.Media_Ingest.Ingest_Local_Video_Window import VideoIngestWindowRedesigned

    class FormTestApp(App):
        def compose(self):
            yield VideoIngestWindowRedesigned(self)

    app = FormTestApp()
    async with app.run_test() as pilot:
        # Test form field inputs
        await pilot.click("#title-input")
        await pilot.press(*"Test Video Title")

        # Verify form state
        title_input = app.query_one("#title-input")
        assert title_input.value == "Test Video Title"

        # Test validation
        await pilot.click("#author-input")
        await pilot.press("a")  # Too short - should trigger validation

        # Check validation error
        author_input = app.query_one("#author-input")
        assert "error" in author_input.classes
```

#### Widget State Testing
Test reactive properties and state changes:

```python
@pytest.mark.asyncio
async def test_reactive_widget():
    class ReactiveTestApp(App):
        counter = reactive(0)
        
        def compose(self):
            yield Label(f"Value: {self.counter}", id="display")
            yield Button("Increment", id="inc")
        
        def watch_counter(self, value):
            self.query_one("#display").update(f"Value: {value}")
        
        def on_button_pressed(self):
            self.counter += 1
    
    app = ReactiveTestApp()
    async with app.run_test() as pilot:
        # Test reactive property updates
        assert app.counter == 0
        
        await pilot.click("#inc")
        await pilot.pause()  # Let reactive system update
        
        assert app.counter == 1
        display = app.query_one("#display")
        assert "Value: 1" in str(display.renderable)
```

#### Async Worker Testing
Test background workers and async operations:

```python
@pytest.mark.asyncio
async def test_background_processing():
    class ProcessingApp(App):
        def __init__(self):
            super().__init__()
            self.result = None
        
        def compose(self):
            yield Button("Start Processing", id="start")
            yield Label("", id="status")
        
        @work(exclusive=True)
        async def process_data(self):
            self.query_one("#status").update("Processing...")
            await asyncio.sleep(0.1)  # Simulate work
            self.result = "Complete"
            self.query_one("#status").update("Done!")
        
        def on_button_pressed(self):
            self.process_data()
    
    app = ProcessingApp()
    async with app.run_test() as pilot:
        await pilot.click("#start")
        
        # Wait for worker to complete
        await pilot.pause(0.2)
        
        assert app.result == "Complete"
        status = app.query_one("#status")
        assert "Done!" in str(status.renderable)
```

### Snapshot Testing

Visual regression testing with snapshots captures the rendered appearance of your app:

```python
# Install: pip install pytest-textual-snapshot

def test_app_appearance(snap_compare):
    """Test that app looks the same as before."""
    # Snapshot of a Python file that creates an app
    assert snap_compare("path/to/my_app.py", terminal_size=(80, 24))

def test_app_with_interaction(snap_compare):
    """Test app appearance after user interaction."""
    from textual.app import App
    from textual.widgets import Button
    
    class SnapApp(App):
        def compose(self):
            yield Button("Click me", id="btn")
        
        def on_button_pressed(self):
            self.query_one("#btn").label = "Clicked!"
    
    async def run_before_snapshot(pilot):
        await pilot.click("#btn")
    
    # Snapshot after interaction
    assert snap_compare(SnapApp(), run_before=run_before_snapshot)
```

### Mocking and Stubbing

Mock external dependencies and services:

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
@patch('tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService')
async def test_video_processing(mock_service):
    # Mock the transcription service
    mock_service.return_value.get_available_providers.return_value = ["whisper"]
    mock_service.return_value.get_available_models.return_value = ["base", "large"]
    
    app = VideoProcessingApp()
    async with app.run_test() as pilot:
        await pilot.click("#transcription-provider")
        
        # Verify mocked service was called
        mock_service.return_value.get_available_providers.assert_called_once()

@pytest.mark.asyncio
async def test_api_call_mocking():
    class ApiApp(App):
        def __init__(self):
            super().__init__()
            self.api_result = None
        
        async def call_api(self):
            # Simulate API call
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.example.com/data")
                self.api_result = response.json()
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"data": "test"}
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        app = ApiApp()
        await app.call_api()
        assert app.api_result == {"data": "test"}
```

### Testing Best Practices

#### Test Structure
```python
# Good: Focused, single-purpose tests
@pytest.mark.asyncio
async def test_button_increments_counter():
    app = CounterApp()
    async with app.run_test() as pilot:
        await pilot.click("#increment")
        assert app.counter == 1

# Good: Clear test names describing behavior
@pytest.mark.asyncio
async def test_form_validation_shows_error_for_empty_required_field():
    # Test implementation...
    pass

# Good: Test setup with fixtures
@pytest.fixture
def sample_app():
    return MyApp()

@pytest.mark.asyncio
async def test_with_fixture(sample_app):
    async with sample_app.run_test() as pilot:
        # Test using the fixture
        pass
```

#### Error Testing
```python
@pytest.mark.asyncio
async def test_error_handling():
    app = MyApp()
    async with app.run_test() as pilot:
        # Test error conditions
        with pytest.raises(ValueError):
            await app.invalid_operation()
        
        # Test error state in UI
        error_widget = app.query_one("#error-display")
        assert "error" in error_widget.classes
```

#### Testing Different Terminal Sizes
```python
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (60, 20)])
@pytest.mark.asyncio
async def test_responsive_layout(size):
    app = ResponsiveApp()
    async with app.run_test() as pilot:
        pilot.resize_terminal(*size)
        await pilot.pause()
        
        # Verify layout adapts to size
        main_container = app.query_one("#main")
        assert main_container.size.width <= size[0]
```

### Integration Testing

Test multiple components working together:

```python
@pytest.mark.asyncio
async def test_full_ingestion_workflow():
    """Test complete media ingestion from file selection to processing."""
    app = TldwCli()
    async with app.run_test(size=(120, 40)) as pilot:
        # Navigate to media ingestion
        await pilot.press("ctrl+i")  # Shortcut to ingestion
        
        # Select video ingestion
        await pilot.click("#video-tab")
        
        # Add test file
        test_file = "test_video.mp4"
        video_window = app.query_one("VideoIngestWindowRedesigned")
        video_window.add_files([Path(test_file)])
        
        # Configure options
        await pilot.click("#extract-audio-only")
        
        # Start processing
        await pilot.click("#process-button")
        
        # Wait for processing to complete
        await pilot.pause(1.0)
        
        # Verify success
        status = video_window.processing_status
        assert status.state == "complete"
```

### Performance Testing

Test app performance and responsiveness:

```python
import time

@pytest.mark.asyncio
async def test_performance_large_dataset():
    """Test app performance with large amounts of data."""
    app = DataApp()
    
    start_time = time.time()
    async with app.run_test() as pilot:
        # Load large dataset
        large_data = list(range(10000))
        app.load_data(large_data)
        
        await pilot.pause()  # Wait for rendering
        
        # Should render within reasonable time
        render_time = time.time() - start_time
        assert render_time < 2.0  # Should render in under 2 seconds
        
        # UI should remain responsive
        await pilot.press("j")  # Scroll down
        await pilot.pause(0.1)
        
        # Verify scroll worked
        scrollview = app.query_one("ScrollView")
        assert scrollview.scroll_y > 0
```

### Testing Utilities and Helpers

Create reusable testing utilities:

```python
# test_helpers.py
async def wait_for_condition(pilot, condition, timeout=1.0):
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return True
        await pilot.pause(0.01)
    return False

async def fill_form(pilot, form_data):
    """Helper to fill multiple form fields."""
    for field_id, value in form_data.items():
        await pilot.click(f"#{field_id}")
        # Clear existing content
        await pilot.press("ctrl+a")
        # Type new value
        await pilot.press(*value)

# Usage in tests
@pytest.mark.asyncio
async def test_with_helpers(sample_app):
    async with sample_app.run_test() as pilot:
        await fill_form(pilot, {
            "title-input": "Test Title",
            "author-input": "Test Author"
        })
        
        # Wait for validation to complete
        await wait_for_condition(
            pilot,
            lambda: sample_app.query_one("#submit-button").disabled == False
        )
```

Testing in Textual is powerful and flexible, allowing you to verify both the behavior and appearance of your TUI applications. The key is to test user interactions, state changes, and visual consistency while keeping tests focused and maintainable.

## Built-in Widgets

### Input Widgets

#### Button
```python
# Basic button
yield Button("Submit", id="submit", variant="primary")

# Button with custom styling
yield Button("Cancel", id="cancel", variant="default", classes="cancel-btn")

# Handle button press
@on(Button.Pressed, "#submit")
def handle_submit(self):
    self.process_form()
```

#### Input (Single-line text)
```python
# Basic input
yield Input(placeholder="Enter your name", id="name")

# Input with validation
yield Input(
    value="",
    placeholder="Email address",
    id="email",
    validators=[Email()]  # Custom validator
)

# Handle input changes
@on(Input.Changed)
def handle_input_change(self, event):
    self.validate_field(event.input.id, event.value)

# CSS for proper visibility
Input {
    height: 3;
    width: 100%;
    margin-bottom: 1;
    border: solid $primary;
}

Input:focus {
    border: solid $accent;
}
```

#### TextArea (Multi-line text)
```python
# Basic textarea
yield TextArea(
    "Default text",
    id="description",
    classes="form-textarea"
)

# Textarea with language support
yield TextArea(
    "",
    language="python",  # Syntax highlighting
    id="code-input",
    soft_wrap=True
)

# CSS for textareas
TextArea {
    min-height: 5;
    max-height: 15;
    width: 100%;
    margin-bottom: 1;
}
```

#### Checkbox
```python
# Basic checkbox
yield Checkbox("Enable notifications", value=True, id="notifications")

# Checkbox with custom styling
yield Checkbox(
    "I agree to terms",
    id="terms",
    classes="required-checkbox"
)

# Handle checkbox changes
@on(Checkbox.Changed)
def handle_checkbox(self, event):
    if event.checkbox.id == "terms":
        self.update_submit_button_state()
```

#### RadioButton and RadioSet
```python
# Radio button group
with RadioSet(id="difficulty"):
    yield RadioButton("Easy", id="easy", value=True)
    yield RadioButton("Medium", id="medium")
    yield RadioButton("Hard", id="hard")

# Handle radio selection
@on(RadioSet.Changed)
def handle_radio_change(self, event):
    selected_value = event.radio_set.pressed_button.id
    self.update_difficulty(selected_value)
```

#### Select (Dropdown)
```python
# Basic select
options = [("option1", "Option 1"), ("option2", "Option 2")]
yield Select(options, id="dropdown", value="option1")

# Dynamic select options
yield Select([], id="dynamic-select")

# Populate select after mount
def on_mount(self):
    select = self.query_one("#dynamic-select")
    select.set_options([("new1", "New Option 1")])

# Handle selection change
@on(Select.Changed)
def handle_selection(self, event):
    self.process_selection(event.value)
```

#### Switch
```python
# Toggle switch
yield Switch(value=False, id="dark-mode")

# Handle switch toggle
@on(Switch.Changed)
def handle_switch(self, event):
    self.toggle_theme(event.value)
```

### Display Widgets
- **Label**: Simple text display
- **Static**: Static content with Rich rendering
- **Markdown**: Render markdown documents
- **MarkdownViewer**: Interactive markdown viewer
- **Pretty**: Display Python objects prettily
- **Log**: Scrolling log display
- **RichLog**: Rich text log display
- **DataTable**: Tabular data display
- **Tree**: Hierarchical tree view
- **DirectoryTree**: File system tree

### Display Widgets
- **Label**: Simple text display
- **Static**: Static content with Rich rendering
- **Markdown**: Render markdown documents
- **MarkdownViewer**: Interactive markdown viewer
- **Pretty**: Display Python objects prettily
- **Log**: Scrolling log display
- **RichLog**: Rich text log display
- **DataTable**: Tabular data display
- **Tree**: Hierarchical tree view
- **DirectoryTree**: File system tree

### Container Widgets
- **Container**: Generic container
- **Horizontal**: Horizontal layout
- **Vertical**: Vertical layout
- **Grid**: Grid layout
- **ScrollableContainer**: Scrollable container
- **Tabs**: Tab container
- **TabbedContent**: Content with tabs
- **Collapsible**: Collapsible content section

### Utility Widgets
- **Header**: App header with title
- **Footer**: App footer with key bindings
- **LoadingIndicator**: Loading animation
- **ProgressBar**: Progress indicator
- **Sparkline**: Mini chart
- **Rule**: Horizontal/vertical separator
- **Placeholder**: Development placeholder

## Best Practices

### Code Organization
```python
# app.py
from textual.app import App
from .screens import MainScreen, SettingsScreen
from .widgets import CustomWidget

class MyApp(App):
    CSS_PATH = "styles.tcss"
    
    def compose(self):
        yield MainScreen()
```

### Performance Tips
1. **Use workers for I/O**: Don't block the main thread
2. **Limit reactive updates**: Use `init=False` when appropriate
3. **Batch updates**: Update multiple attributes together
4. **Profile rendering**: Use `textual run --dev` for performance info
5. **Minimize recomposition**: Prefer `refresh()` over `recompose=True`

### Common Patterns

#### Loading Data
```python
class DataView(Widget):
    data = reactive(None)
    loading = reactive(True)
    
    def on_mount(self):
        self.load_data()
    
    @work
    async def load_data(self):
        try:
            self.data = await fetch_data()
        finally:
            self.loading = False
    
    def render(self):
        if self.loading:
            return "Loading..."
        return Pretty(self.data)
```

#### Form Handling with Validation
```python
class FormData(BaseModel):
    """Pydantic model for form validation."""
    name: str
    email: EmailStr
    age: int = Field(ge=0, le=120)

class AdvancedForm(Container):
    form_data = reactive({})
    errors = reactive({})
    is_valid = reactive(False)
    
    def compose(self):
        with Vertical(classes="form-container"):
            # Form fields
            yield Label("Name:", classes="form-label")
            yield Input(placeholder="Enter your name", id="name", classes="form-input")
            
            yield Label("Email:", classes="form-label") 
            yield Input(placeholder="Enter email", id="email", classes="form-input")
            
            yield Label("Age:", classes="form-label")
            yield Input(placeholder="Enter age", id="age", classes="form-input")
            
            # Error display
            yield Static("", id="form-errors", classes="error-display hidden")
            
            # Submit button
            yield Button("Submit", id="submit", disabled=True, classes="submit-button")
    
    @on(Input.Changed)
    def handle_input_change(self, event):
        """Handle input changes and validate in real-time."""
        field_id = event.input.id
        value = event.value
        
        # Update form data
        self.form_data = {**self.form_data, field_id: value}
        
        # Validate field
        self.validate_field(field_id, value)
        
        # Update submit button state
        self.update_submit_state()
    
    def validate_field(self, field_id: str, value: str):
        """Validate individual field."""
        errors = dict(self.errors)
        
        if field_id == "name":
            if not value.strip():
                errors[field_id] = "Name is required"
            elif len(value) < 2:
                errors[field_id] = "Name must be at least 2 characters"
            else:
                errors.pop(field_id, None)
                
        elif field_id == "email":
            if not value:
                errors[field_id] = "Email is required"
            elif "@" not in value or "." not in value:
                errors[field_id] = "Please enter a valid email"
            else:
                errors.pop(field_id, None)
                
        elif field_id == "age":
            if not value:
                errors[field_id] = "Age is required"
            else:
                try:
                    age = int(value)
                    if age < 0 or age > 120:
                        errors[field_id] = "Age must be between 0 and 120"
                    else:
                        errors.pop(field_id, None)
                except ValueError:
                    errors[field_id] = "Age must be a number"
        
        self.errors = errors
        self.display_errors()
    
    def display_errors(self):
        """Display validation errors."""
        error_widget = self.query_one("#form-errors")
        
        if self.errors:
            error_text = "\n".join(f"• {error}" for error in self.errors.values())
            error_widget.update(error_text)
            error_widget.remove_class("hidden")
            error_widget.add_class("visible")
        else:
            error_widget.add_class("hidden")
            error_widget.remove_class("visible")
    
    def update_submit_state(self):
        """Enable/disable submit button based on validation."""
        submit_button = self.query_one("#submit")
        required_fields = {"name", "email", "age"}
        
        has_all_fields = all(
            field in self.form_data and self.form_data[field].strip()
            for field in required_fields
        )
        
        has_no_errors = not self.errors
        
        submit_button.disabled = not (has_all_fields and has_no_errors)
    
    @on(Button.Pressed, "#submit")
    def submit_form(self):
        """Submit the form."""
        try:
            # Final validation with Pydantic
            validated_data = FormData(**self.form_data)
            self.post_message(FormSubmitted(validated_data.dict()))
            
            # Clear form
            self.clear_form()
            
        except ValidationError as e:
            # Handle Pydantic validation errors
            self.handle_validation_errors(e.errors())
    
    def clear_form(self):
        """Clear the form after successful submission."""
        for field_id in ["name", "email", "age"]:
            input_widget = self.query_one(f"#{field_id}")
            input_widget.value = ""
        
        self.form_data = {}
        self.errors = {}
        self.query_one("#form-errors").add_class("hidden")
```

#### Progressive Disclosure Form
```python
class ProgressiveDisclosureForm(Container):
    """Form with simple/advanced mode toggle."""
    
    advanced_mode = reactive(False)
    
    def compose(self):
        with Vertical(classes="progressive-form"):
            # Mode toggle
            with Horizontal(classes="mode-toggle"):
                yield RadioSet(id="mode-selector"):
                    yield RadioButton("Simple", value=True, id="simple-mode")
                    yield RadioButton("Advanced", id="advanced-mode")
            
            # Essential fields (always visible)
            with Container(classes="essential-fields"):
                yield Label("Essential Information", classes="section-title")
                yield Label("Title:", classes="form-label")
                yield Input(id="title", placeholder="Required title")
                
                yield Label("Description:", classes="form-label")
                yield TextArea(id="description", classes="form-textarea")
            
            # Advanced fields (collapsible)
            with Collapsible(
                "Advanced Options",
                collapsed=True,
                id="advanced-options",
                classes="advanced-section"
            ):
                yield Label("Tags:", classes="form-label")
                yield Input(id="tags", placeholder="Comma-separated tags")
                
                yield Label("Priority:", classes="form-label")
                yield Select([
                    ("low", "Low"),
                    ("medium", "Medium"),
                    ("high", "High")
                ], id="priority")
                
                yield Checkbox("Email notifications", id="notifications")
                yield Checkbox("Public visibility", id="public")
    
    @on(RadioSet.Changed, "#mode-selector")
    def handle_mode_change(self, event):
        """Handle mode toggle."""
        self.advanced_mode = event.pressed.id == "advanced-mode"
    
    def watch_advanced_mode(self, advanced: bool):
        """React to mode changes."""
        collapsible = self.query_one("#advanced-options")
        collapsible.collapsed = not advanced
        
        # Update form styling
        if advanced:
            self.add_class("advanced-mode")
        else:
            self.remove_class("advanced-mode")
```

#### Responsive Layout Pattern
```python
class ResponsiveForm(Container):
    """Form that adapts to terminal size."""
    
    def compose(self):
        with Container(classes="responsive-container"):
            # Header section
            with Container(classes="form-header"):
                yield Static("Media Ingestion", classes="form-title")
                yield Static("Configure your media processing options", classes="form-subtitle")
            
            # Main content - switches between single/double column
            with Container(classes="form-content"):
                # File selection (always full width)
                with Container(classes="file-section"):
                    yield Button("Browse Files", id="browse", classes="file-button")
                    yield Static("No files selected", id="file-status")
                
                # Metadata fields (responsive columns)
                with Container(classes="metadata-section responsive-columns"):
                    with Container(classes="form-column"):
                        yield Label("Title:", classes="form-label")
                        yield Input(id="title", classes="form-input")
                        
                        yield Label("Author:", classes="form-label")
                        yield Input(id="author", classes="form-input")
                    
                    with Container(classes="form-column"):
                        yield Label("Keywords:", classes="form-label")
                        yield TextArea(id="keywords", classes="form-textarea-small")
                
                # Action buttons
                with Container(classes="form-actions"):
                    yield Button("Process", id="process", variant="primary")
                    yield Button("Cancel", id="cancel", variant="default")
    
    def on_mount(self):
        """Adjust layout based on terminal size."""
        self.adjust_layout()
        
    def on_resize(self, event):
        """Handle terminal resize."""
        self.adjust_layout()
    
    def adjust_layout(self):
        """Adjust layout for current terminal size."""
        terminal_size = self.app.size
        
        if terminal_size.width < 100:
            # Narrow terminal - single column
            self.add_class("narrow-layout")
            self.remove_class("wide-layout")
        else:
            # Wide terminal - double column
            self.add_class("wide-layout")
            self.remove_class("narrow-layout")

# Corresponding CSS
"""
.responsive-container {
    width: 100%;
    height: 100%;
    padding: 1;
}

.responsive-columns {
    layout: vertical; /* Default to single column */
}

.wide-layout .responsive-columns {
    layout: horizontal; /* Switch to side-by-side */
}

.form-column {
    width: 1fr;
    padding-right: 2;
}

.narrow-layout .form-column {
    padding-right: 0;
    margin-bottom: 1;
}
"""
```

#### Error Handling
```python
@work
async def risky_operation(self):
    try:
        result = await dangerous_api_call()
        self.display_result(result)
    except Exception as e:
        self.notify(f"Error: {e}", severity="error")
        self.log.error(f"Operation failed: {e}")
```

### Debugging Tips
1. **Use textual console**: `textual console` for print debugging
2. **Dev mode**: `textual run --dev app.py` for live reload
3. **Inspect DOM**: Press Ctrl+D in dev mode
4. **Log messages**: Use `self.log()` for debugging
5. **Query debugging**: `self.query("Button").first()` to find widgets

### Security Considerations
1. **Sanitize user input**: Especially for dynamic CSS/actions
2. **Validate file paths**: When dealing with file operations
3. **Handle sensitive data carefully**: Don't log passwords/tokens
4. **Use HTTPS for API calls**: When fetching remote data

### Accessibility
1. **Provide keyboard navigation**: All features keyboard accessible
2. **Use semantic widgets**: Buttons for actions, not clickable divs
3. **Add helpful tooltips**: `tooltip="Help text"`
4. **Meaningful IDs and classes**: For screen readers
5. **High contrast themes**: Support both light and dark modes

## Common Gotchas

1. **Async context**: Remember Textual runs in async context
2. **Thread safety**: Use `call_from_thread()` when updating from threads
3. **CSS specificity**: More specific selectors override less specific
4. **Widget lifecycle**: Don't access widgets before mounting
5. **Focus handling**: Only one widget can have focus
6. **Event bubbling**: Events bubble up unless stopped
7. **Reactive timing**: Watchers fire after validation
8. **Worker cleanup**: Workers cancelled when widget unmounted

## Troubleshooting Common UI Issues

### Input Widgets Not Visible

**Problem**: Input widgets are present in DOM but not rendering visually
```python
# This may not display properly
yield Input(id="name")
```

**Solutions**:
```python
# 1. Add explicit height and width
yield Input(id="name", classes="form-input")

# CSS:
.form-input {
    height: 3;        # Explicit height required!
    width: 100%;      # Or specific width
    margin-bottom: 1;
}

# 2. Check parent container layout
with Container(classes="input-container"):
    yield Input(id="name")
    
# CSS for container:
.input-container {
    height: auto;     # Allow container to size to content
    width: 100%;
}
```

### Double Scrolling Issues

**Problem**: Nested scrollable containers cause broken scrolling
```python
# WRONG - nested VerticalScroll containers
with VerticalScroll():
    with SomeWidget():  # Also has VerticalScroll internally
        yield content
```

**Solution**:
```python
# RIGHT - only one level of scrolling
with VerticalScroll(classes="main-scroll"):
    # Use regular containers inside
    with Container():
        yield content
```

### Layout Not Updating

**Problem**: Layout doesn't respond to size changes
```python
# Missing reactive updates
class MyWidget(Widget):
    def compose(self):
        yield Static("Fixed content")
```

**Solution**:
```python
# Add reactive updates and proper watchers
class MyWidget(Widget):
    content = reactive("Default")
    
    def compose(self):
        yield Static(self.content, id="dynamic-content")
    
    def watch_content(self, new_content: str):
        """Update display when content changes."""
        self.query_one("#dynamic-content").update(new_content)
        
    def on_resize(self, event):
        """Handle terminal resize."""
        self.refresh_layout()
```

### Form Validation Issues

**Problem**: Form validation not working correctly
```python
# Validation runs but doesn't show feedback
@on(Input.Changed)
def validate_input(self, event):
    if not self.is_valid(event.value):
        # Error not displayed to user
        pass
```

**Solution**:
```python
@on(Input.Changed)
def validate_input(self, event):
    field_id = event.input.id
    value = event.value
    
    # Validate and store errors
    error = self.validate_field(field_id, value)
    
    if error:
        # Show error to user
        self.display_error(field_id, error)
        event.input.add_class("error")
    else:
        # Clear error display
        self.clear_error(field_id)
        event.input.remove_class("error")

def display_error(self, field_id: str, error: str):
    """Show error message to user."""
    error_widget = self.query_one(f"#{field_id}-error", expect_type=Static)
    error_widget.update(f"Error: {error}")
    error_widget.remove_class("hidden")
```

### CSS Not Applying

**Problem**: CSS rules not taking effect
```tcss
/* This might not work */
Input {
    background: red;
}
```

**Common causes and solutions**:

1. **Specificity issues**:
```tcss
/* More specific selector needed */
.form-container Input {
    background: red;
}

/* Or use ID selector */
#my-input {
    background: red;
}
```

2. **CSS file not loaded**:
```python
class MyApp(App):
    CSS_PATH = "styles.tcss"  # Make sure file exists
    
    # Or inline CSS
    CSS = """
    Input {
        height: 3;
        background: $surface;
    }
    """
```

3. **Modular CSS build issues**:
```bash
# Rebuild CSS after changes
./build_css.sh

# Check if your CSS file is included in the build
```

### Widget Query Failures

**Problem**: `query_one()` raising exceptions
```python
# This might fail
widget = self.query_one("#missing-id")
```

**Solutions**:
```python
# 1. Check if widget exists first
try:
    widget = self.query_one("#my-widget")
except NoMatches:
    self.log.warning("Widget #my-widget not found")
    return

# 2. Use optional query
widgets = self.query("#my-widget")
if widgets:
    widget = widgets.first()

# 3. Wait for widget to mount
def on_mount(self):
    # Schedule callback after mount complete
    self.call_after_refresh(self.setup_widgets)

def setup_widgets(self):
    widget = self.query_one("#my-widget")
    # Now safe to access widget
```

### Focus and Keyboard Navigation Issues

**Problem**: Widgets not receiving focus or keyboard events
```python
# Widget exists but can't be focused
yield Input(id="name")
```

**Solution**:
```python
# Ensure widget can receive focus
yield Input(id="name", can_focus=True)  # Usually automatic for Input

# Set initial focus
def on_mount(self):
    self.query_one("#name").focus()

# Handle tab order with explicit focus calls
@on(Key)
def handle_key(self, event):
    if event.key == "tab":
        next_widget = self.get_next_focusable()
        if next_widget:
            next_widget.focus()
            event.prevent_default()
```

### Performance Issues

**Problem**: UI becomes sluggish with many widgets or updates
```python
# Too many reactive updates
class SlowWidget(Widget):
    data = reactive([], recompose=True)  # Expensive!
    
    def update_data(self):
        for i in range(1000):
            self.data.append(i)  # Triggers recompose 1000 times!
```

**Solution**:
```python
# Batch updates
class FastWidget(Widget):
    data = reactive([], recompose=True)
    
    def update_data(self):
        # Build new data first, then update once
        new_data = list(range(1000))
        self.data = new_data  # Single update

# Or use refresh instead of recompose when possible
class EfficientWidget(Widget):
    data = reactive([])  # No recompose
    
    def watch_data(self, new_data):
        # Manual DOM update instead of full recompose
        list_widget = self.query_one("#data-list")
        list_widget.clear()
        for item in new_data:
            list_widget.append(ListItem(Label(str(item))))
```

## Resources

- Official docs: https://textual.textualize.io/
- GitHub: https://github.com/Textualize/textual
- Discord community: https://discord.gg/Enf6Z3qhVr
- Example apps: https://github.com/Textualize/textual/tree/main/examples