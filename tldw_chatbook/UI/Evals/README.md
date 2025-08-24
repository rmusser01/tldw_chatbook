# Evaluation Navigation System

## Overview

The new evaluation navigation system provides a modern, intuitive TUI experience for the evaluation lab. It follows Textual best practices and implements a card-based navigation hub with focused workflow screens.

## Architecture

### Directory Structure
```
tldw_chatbook/UI/Evals/
├── navigation/           # Navigation components
│   ├── eval_nav_screen.py    # Main navigation hub
│   ├── nav_bar.py            # Persistent navigation bar
│   └── breadcrumbs.py        # Breadcrumb trail widget
├── screens/             # Workflow-specific screens
│   └── quick_test.py         # Quick test workflow
├── widgets/             # Reusable UI components
│   └── progress_dashboard.py # Enhanced progress tracking
└── evals_window_v3.py   # Main container
```

## Key Features

### 1. Navigation Hub
- **Card-based layout** with 6 main workflows
- **Visual hierarchy** with icons and descriptions
- **Keyboard shortcuts** (1-6) for quick navigation
- **Status bar** with quick actions

### 2. Navigation Bar
- **Breadcrumb trail** for context awareness
- **Quick action buttons** (Run, Stop, Export, Refresh)
- **Live status indicator** with visual states
- **Persistent across all screens**

### 3. Quick Test Screen
- **Streamlined form** for single evaluations
- **Real-time progress** tracking
- **Inline results** display
- **Smart defaults** and validation

### 4. Progress Dashboard
- **Real-time metrics** grid
- **Throughput visualization** with sparkline
- **ETA calculation** and timing
- **Success/error counters**

## Keyboard Shortcuts

### Global
- `Escape` - Go back/Cancel
- `Tab/Shift+Tab` - Focus navigation
- `Enter` - Activate focused element
- `Ctrl+/` - Show shortcuts help

### Navigation Hub
- `1-6` - Quick jump to sections
- `Ctrl+R` - Run last evaluation

### Quick Test Screen
- `Ctrl+R` - Run evaluation
- `Ctrl+S` - Stop evaluation
- `Ctrl+E` - Export results

## Usage Example

```python
from tldw_chatbook.UI.Evals.evals_window_v3 import EvalsWindowV3

# Create the evaluation window
eval_window = EvalsWindowV3(app_instance=app)

# The window starts with the navigation hub
# Users can navigate using keyboard or mouse
```

## Testing

Run the test script to see the navigation in action:
```bash
python test_eval_navigation.py
```

## Extending the System

### Adding a New Screen

1. Create screen in `screens/`:
```python
# screens/my_workflow.py
class MyWorkflowScreen(Screen):
    def compose(self) -> ComposeResult:
        # Add navigation bar
        self.nav_bar = EvalNavigationBar(self.app_instance)
        yield self.nav_bar
        # Add your content
```

2. Add navigation card in `eval_nav_screen.py`:
```python
NavigationCard(
    id="my_workflow",
    title="My Workflow",
    icon="🎯",
    description="Description here",
    shortcut="Press [7]"
)
```

3. Register in `evals_window_v3.py`:
```python
screen_map = {
    "my_workflow": lambda: MyWorkflowScreen(self.app_instance),
}
```

### Adding Keyboard Shortcuts

Add to screen's BINDINGS:
```python
BINDINGS = [
    Binding("ctrl+x", "my_action", "Do Something", show=True),
]
```

## Design Principles

1. **Navigation First** - Clear paths and context
2. **Keyboard Friendly** - All actions accessible via keyboard
3. **Progressive Disclosure** - Show complexity only when needed
4. **Visual Feedback** - Clear status and progress indicators
5. **Consistent Patterns** - Similar workflows across screens

## Status

### Implemented
- ✅ Navigation hub with card layout
- ✅ Navigation bar with breadcrumbs
- ✅ Quick test screen
- ✅ Progress dashboard widget
- ✅ Keyboard shortcuts
- ✅ Status management

### Planned
- ⏳ Comparison mode screen
- ⏳ Batch evaluation screen
- ⏳ Results browser screen
- ⏳ Task manager screen
- ⏳ Model manager screen
- ⏳ Settings integration
- ⏳ Export functionality
- ⏳ Help system

## Benefits

1. **Improved UX** - Clear navigation and focused workflows
2. **Better Discoverability** - All features visible from hub
3. **Faster Access** - Keyboard shortcuts for power users
4. **Context Awareness** - Breadcrumbs show location
5. **Scalable** - Easy to add new workflows
6. **Accessible** - Keyboard-only navigation support