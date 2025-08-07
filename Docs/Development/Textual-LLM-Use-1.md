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

### Basic Testing
```python
import pytest
from textual.testing import AppTest

@pytest.mark.asyncio
async def test_app():
    app = MyApp()
    async with app.run_test() as pilot:
        # Test initial state
        assert pilot.app.title == "My App"
        
        # Simulate key press
        await pilot.press("q")
        
        # Check app exited
        assert pilot.app.return_value == 0
```

### Simulating Input
```python
async with app.run_test() as pilot:
    # Key presses
    await pilot.press("tab", "enter")
    await pilot.press("ctrl+s")
    
    # Mouse clicks
    await pilot.click("#button")
    await pilot.click(10, 20)  # Coordinates
    
    # Text input
    await pilot.press(*"Hello World")
    
    # Wait for updates
    await pilot.pause()
```

### Testing Widgets
```python
async def test_counter_widget():
    class TestApp(App):
        def compose(self):
            yield Counter(id="counter")
    
    app = TestApp()
    async with app.run_test() as pilot:
        counter = pilot.app.query_one("#counter")
        
        # Test initial state
        assert counter.count == 0
        
        # Simulate click
        await pilot.click("#counter")
        
        # Verify update
        assert counter.count == 1
```

### Snapshot Testing
```python
# Install: pip install pytest-textual-snapshot

def test_snapshot(snap_compare):
    assert snap_compare("path/to/app.py", terminal_size=(80, 24))
```

## Built-in Widgets

### Input Widgets
- **Button**: Clickable button with various styles
- **Input**: Single-line text input
- **TextArea**: Multi-line text editor with syntax highlighting
- **Checkbox**: Toggle on/off state
- **RadioButton**: Single selection from group
- **RadioSet**: Container for radio buttons
- **Switch**: Toggle switch control
- **Select**: Dropdown selection
- **SelectionList**: Multi-select list

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

#### Form Handling
```python
class Form(Container):
    def compose(self):
        yield Input(placeholder="Name", id="name")
        yield Input(placeholder="Email", id="email")
        yield Button("Submit", id="submit")
    
    @on(Button.Pressed, "#submit")
    def submit_form(self):
        name = self.query_one("#name").value
        email = self.query_one("#email").value
        
        if self.validate(name, email):
            self.post_message(FormSubmitted(name, email))
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

## Resources

- Official docs: https://textual.textualize.io/
- GitHub: https://github.com/Textualize/textual
- Discord community: https://discord.gg/Enf6Z3qhVr
- Example apps: https://github.com/Textualize/textual/tree/main/examples