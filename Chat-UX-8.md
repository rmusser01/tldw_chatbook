# Chat-UX-8.md: Sidebar & Menu UX Improvements

## Executive Summary

This document outlines comprehensive UX improvements for the chat sidebar/menu system, following Textual best practices. The improvements focus on visual hierarchy, user efficiency, accessibility, and modern UI patterns while maintaining consistency with the existing codebase architecture.

## Table of Contents

1. [Visual Hierarchy & Organization](#1-visual-hierarchy--organization)
2. [Collapsible Management](#2-collapsible-management)
3. [Search & Filter System](#3-search--filter-system)
4. [Quick Actions Bar](#4-quick-actions-bar)
5. [Visual Feedback & States](#5-visual-feedback--states)
6. [Contextual Help System](#6-contextual-help-system)
7. [Settings Profiles & Presets](#7-settings-profiles--presets)
8. [Keyboard Navigation](#8-keyboard-navigation)
9. [Responsive Design & Resizing](#9-responsive-design--resizing)
10. [Sidebar Display Modes](#10-sidebar-display-modes)
11. [Form Validation & Error Handling](#11-form-validation--error-handling)
12. [CSS & Styling Improvements](#12-css--styling-improvements)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Visual Hierarchy & Organization

### Problem Statement
Current sidebar presents all settings in a flat or loosely grouped structure, making it difficult for users to quickly locate specific settings or understand relationships between options.

### Solution: Semantic Grouping with Visual Indicators

#### Implementation

```python
# In settings_sidebar.py
def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Create an organized, hierarchical sidebar."""
    
    with VerticalScroll(id=f"{id_prefix}-left-sidebar", classes="sidebar"):
        # Quick access toolbar at top
        yield from create_quick_actions_bar(id_prefix)
        
        # Search bar for filtering settings
        yield Input(
            placeholder="🔍 Search settings...",
            id=f"{id_prefix}-settings-search",
            classes="sidebar-search-input"
        )
        
        # Primary Settings Group - Always visible
        with Container(classes="settings-group primary-group"):
            yield Static("ESSENTIAL", classes="group-header")
            
            with Collapsible(
                title="🤖 Model Configuration", 
                collapsed=False,
                id=f"{id_prefix}-model-config",
                classes="settings-collapsible priority-high"
            ):
                yield from create_model_settings(id_prefix, config)
            
            with Collapsible(
                title="💬 Active Conversation",
                collapsed=False,
                id=f"{id_prefix}-conversation",
                classes="settings-collapsible priority-high"
            ):
                yield from create_conversation_controls(id_prefix)
        
        # Secondary Settings Group
        yield Static(classes="sidebar-section-divider")
        
        with Container(classes="settings-group secondary-group"):
            yield Static("FEATURES", classes="group-header")
            
            with Collapsible(
                title="🎨 Character Settings",
                collapsed=True,
                id=f"{id_prefix}-character",
                classes="settings-collapsible"
            ):
                yield from create_character_settings(id_prefix)
            
            with Collapsible(
                title="🔍 RAG & Search",
                collapsed=True,
                id=f"{id_prefix}-rag",
                classes="settings-collapsible"
            ):
                yield from create_rag_settings(id_prefix)
        
        # Advanced Settings Group
        yield Static(classes="sidebar-section-divider")
        
        with Container(classes="settings-group advanced-group"):
            yield Static("ADVANCED", classes="group-header")
            
            with Collapsible(
                title="⚙️ Parameters",
                collapsed=True,
                id=f"{id_prefix}-parameters",
                classes="settings-collapsible advanced-only"
            ):
                yield from create_advanced_parameters(id_prefix)
```

#### Visual Hierarchy CSS (Textual-Compatible)

```css
/* Group headers - Note: font-size and letter-spacing not supported in Textual */
.group-header {
    text-style: bold;
    color: $text-muted;
    margin: 1 0;
    padding: 0 1;
}

/* Section dividers */
.sidebar-section-divider {
    height: 1;
    margin: 2 0;
    border-top: solid $primary-lighten-2;
    opacity: 50%;  /* Textual uses percentages or 0-1 values */
}

/* Priority indicators - Note: ::before pseudo-element not supported */
.priority-high > .collapsible--header {
    text-style: bold;
    border-left: thick $success;  /* Use border for visual indicator */
}

/* Indentation for nested items */
.settings-group {
    padding-left: 1;
}

.settings-collapsible .settings-collapsible {
    margin-left: 2;
}
```

---

## 2. Collapsible Management

### Problem Statement
Users lose their preferred sidebar configuration when navigating between screens or restarting the application.

### Solution: Intelligent State Management

#### Implementation

```python
# In chat_screen_state.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SidebarState:
    """Persistent sidebar state."""
    collapsible_states: Dict[str, bool] = field(default_factory=dict)
    last_active_section: Optional[str] = None
    search_query: str = ""
    sidebar_width: int = 35  # percentage
    compact_mode: bool = False
    
# In chat_screen.py
class ChatScreen(BaseAppScreen):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "chat", **kwargs)
        self.sidebar_state = self._load_sidebar_state()
    
    def _load_sidebar_state(self) -> SidebarState:
        """Load sidebar state from user preferences."""
        try:
            state_data = self.app_instance.user_preferences.get('sidebar_state', {})
            return SidebarState(**state_data)
        except:
            return SidebarState()
    
    def _save_sidebar_state(self):
        """Persist sidebar state to user preferences."""
        self.app_instance.user_preferences['sidebar_state'] = asdict(self.sidebar_state)
        self.app_instance.save_preferences()
    
    @on(Collapsible.Toggled)
    async def handle_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Track collapsible state changes."""
        self.sidebar_state.collapsible_states[event.collapsible.id] = event.collapsed
        self._save_sidebar_state()
    
    def on_mount(self) -> None:
        """Restore sidebar state on mount."""
        super().on_mount()
        self._restore_collapsible_states()
    
    def _restore_collapsible_states(self):
        """Restore all collapsible states."""
        for coll_id, collapsed in self.sidebar_state.collapsible_states.items():
            try:
                collapsible = self.query_one(f"#{coll_id}", Collapsible)
                collapsible.collapsed = collapsed
            except QueryError:
                pass  # Collapsible might not exist in current mode
```

#### Bulk Actions

```python
# In chat_window_enhanced.py
class ChatWindowEnhanced(Container):
    def compose(self) -> ComposeResult:
        # Add control buttons
        with Horizontal(classes="sidebar-controls"):
            yield Button("➕", id="expand-all", tooltip="Expand all sections")
            yield Button("➖", id="collapse-all", tooltip="Collapse all sections")
            yield Button("↺", id="reset-layout", tooltip="Reset to default")
    
    @on(Button.Pressed, "#expand-all")
    async def expand_all_sections(self) -> None:
        """Expand all collapsible sections."""
        for collapsible in self.query(Collapsible).results():
            collapsible.collapsed = False
        self.notify("All sections expanded", severity="information")
    
    @on(Button.Pressed, "#collapse-all")
    async def collapse_all_sections(self) -> None:
        """Collapse all collapsible sections."""
        for collapsible in self.query(Collapsible).results():
            # Keep essential sections open
            if "priority-high" not in collapsible.classes:
                collapsible.collapsed = True
        self.notify("Non-essential sections collapsed", severity="information")
```

---

## 3. Search & Filter System

### Problem Statement
With many settings and options, users need a way to quickly find specific controls without manually expanding sections.

### Solution: Real-time Filtering with Highlighting

#### Implementation

```python
# In sidebar_search.py
from textual.reactive import reactive
from textual.widgets import Input
from fuzzywuzzy import fuzz  # For fuzzy matching

class SidebarSearch(Input):
    """Smart search for sidebar settings."""
    
    search_active = reactive(False)
    matches = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__(
            placeholder="🔍 Search settings... (Ctrl+/)",
            **kwargs
        )
        self.search_cache = {}  # Cache searchable content
    
    async def on_mount(self) -> None:
        """Build search index on mount."""
        await self._build_search_index()
    
    async def _build_search_index(self):
        """Index all searchable elements."""
        self.search_cache.clear()
        
        for collapsible in self.parent.query(Collapsible):
            # Index title
            self.search_cache[collapsible.id] = {
                'element': collapsible,
                'text': collapsible.title.lower(),
                'type': 'section'
            }
            
            # Index contained labels and inputs
            for widget in collapsible.query("Label, Input, Select, TextArea"):
                widget_text = ""
                if hasattr(widget, 'label'):
                    widget_text = widget.label.lower()
                elif hasattr(widget, 'placeholder'):
                    widget_text = widget.placeholder.lower()
                
                if widget_text:
                    self.search_cache[widget.id] = {
                        'element': widget,
                        'text': widget_text,
                        'parent': collapsible,
                        'type': 'control'
                    }
    
    @on(Input.Changed)
    async def filter_sidebar(self, event: Input.Changed) -> None:
        """Filter sidebar content based on search query."""
        query = event.value.lower().strip()
        
        if not query:
            # Reset visibility
            await self._reset_visibility()
            self.search_active = False
            return
        
        self.search_active = True
        matches = 0
        
        # First, hide everything
        for collapsible in self.parent.query(Collapsible):
            collapsible.display = False
        
        # Show matches
        for item_id, item_data in self.search_cache.items():
            # Use fuzzy matching for better UX
            score = fuzz.partial_ratio(query, item_data['text'])
            
            if score > 60:  # Threshold for match
                matches += 1
                
                if item_data['type'] == 'section':
                    item_data['element'].display = True
                    item_data['element'].collapsed = False  # Auto-expand
                    # Highlight match
                    item_data['element'].add_class("search-match")
                
                elif item_data['type'] == 'control':
                    # Show parent section
                    item_data['parent'].display = True
                    item_data['parent'].collapsed = False
                    # Highlight specific control
                    item_data['element'].add_class("search-highlight")
        
        self.matches = matches
        
        # Update placeholder with results
        if matches == 0:
            self.placeholder = "🔍 No matches found..."
        else:
            self.placeholder = f"🔍 {matches} matches found"
    
    async def _reset_visibility(self):
        """Reset all visibility and highlighting."""
        for collapsible in self.parent.query(Collapsible):
            collapsible.display = True
            collapsible.remove_class("search-match")
            # Restore original collapsed state
            if collapsible.id in self.screen.sidebar_state.collapsible_states:
                collapsible.collapsed = self.screen.sidebar_state.collapsible_states[collapsible.id]
        
        for widget in self.parent.query(".search-highlight"):
            widget.remove_class("search-highlight")
        
        self.placeholder = "🔍 Search settings... (Ctrl+/)"
```

#### Search Highlighting CSS (Textual-Compatible)

```css
/* Search matches - Note: Textual doesn't support animations */
.search-match > .collapsible--header {
    background: $accent-lighten-3;
    border-left: thick $accent;
}

.search-highlight {
    background: $warning-lighten-3;
    text-style: bold;  /* Use text styling instead of animation */
}

/* Search active state */
.sidebar.search-active .settings-collapsible {
    opacity: 50%;
}

.sidebar.search-active .search-match {
    opacity: 100%;  /* Full opacity for matches */
}
```

---

## 4. Quick Actions Bar

### Problem Statement
Frequently used actions require multiple clicks or are buried in collapsible sections.

### Solution: Persistent Quick Action Toolbar

#### Implementation

```python
# In quick_actions.py
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button
from textual.binding import Binding

class QuickActionsBar(Horizontal):
    """Quick access toolbar for common actions."""
    
    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("ctrl+s", "save_chat", "Save Chat"),
        Binding("ctrl+shift+s", "save_as", "Save As..."),
        Binding("ctrl+z", "undo_last", "Undo"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the quick actions bar."""
        yield Button(
            "🆕",
            id="quick-new-chat",
            classes="quick-action-btn",
            tooltip="New Chat (Ctrl+N)"
        )
        yield Button(
            "💾",
            id="quick-save",
            classes="quick-action-btn",
            tooltip="Save (Ctrl+S)"
        )
        yield Button(
            "📋",
            id="quick-copy",
            classes="quick-action-btn",
            tooltip="Copy Chat"
        )
        yield Button(
            "📤",
            id="quick-export",
            classes="quick-action-btn",
            tooltip="Export..."
        )
        yield Button(
            "🔄",
            id="quick-reset",
            classes="quick-action-btn warning",
            tooltip="Reset Settings"
        )
        yield Button(
            "⭐",
            id="quick-favorite",
            classes="quick-action-btn",
            tooltip="Add to Favorites"
        )
    
    @on(Button.Pressed, "#quick-new-chat")
    async def handle_new_chat(self) -> None:
        """Create new chat with animation."""
        self.add_class("action-processing")
        await self.app.action_new_chat()
        self.remove_class("action-processing")
        self.notify("New chat created", severity="success")
    
    @on(Button.Pressed, "#quick-save")
    async def handle_quick_save(self) -> None:
        """Quick save current chat."""
        if not self.app.current_chat_conversation_id:
            self.notify("No active conversation to save", severity="warning")
            return
        
        # Animate button during save
        btn = self.query_one("#quick-save")
        btn.label = "⏳"
        
        try:
            await self.app.save_current_conversation()
            btn.label = "✅"
            self.notify("Chat saved successfully", severity="success")
        except Exception as e:
            btn.label = "❌"
            self.notify(f"Save failed: {e}", severity="error")
        finally:
            # Reset after delay
            self.set_timer(2.0, lambda: setattr(btn, 'label', '💾'))
```

#### Enhanced Quick Actions with Context Menu

```python
class ContextualQuickActions(QuickActionsBar):
    """Context-aware quick actions that change based on state."""
    
    def on_mount(self) -> None:
        """Set up reactive watchers."""
        self.watch(self.app, "current_chat_conversation_id", self._update_actions)
        self.watch(self.app, "chat_has_unsaved_changes", self._update_save_button)
    
    def _update_actions(self, conversation_id: Optional[str]) -> None:
        """Update available actions based on context."""
        save_btn = self.query_one("#quick-save")
        export_btn = self.query_one("#quick-export")
        
        if conversation_id:
            save_btn.disabled = False
            export_btn.disabled = False
            save_btn.tooltip = f"Save '{conversation_id[:8]}...'"
        else:
            save_btn.disabled = True
            export_btn.disabled = True
            save_btn.tooltip = "No active conversation"
    
    def _update_save_button(self, has_changes: bool) -> None:
        """Update save button based on unsaved changes."""
        save_btn = self.query_one("#quick-save")
        if has_changes:
            save_btn.add_class("unsaved-changes")
            save_btn.label = "💾*"
        else:
            save_btn.remove_class("unsaved-changes")
            save_btn.label = "💾"
```

---

## 5. Visual Feedback & States

### Problem Statement
Users lack clear feedback about system state, loading operations, and action results.

### Solution: Comprehensive State Indicators

#### Implementation

```python
# In state_indicators.py
from textual.widgets import Static, LoadingIndicator
from textual.reactive import reactive
from enum import Enum

class LoadingState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    ERROR = "error"

class StateIndicator(Static):
    """Visual state indicator with animations."""
    
    state = reactive(LoadingState.IDLE)
    message = reactive("")
    
    STATE_ICONS = {
        LoadingState.IDLE: "○",
        LoadingState.LOADING: "◐",
        LoadingState.SUCCESS: "✅",
        LoadingState.ERROR: "❌"
    }
    
    STATE_CLASSES = {
        LoadingState.IDLE: "state-idle",
        LoadingState.LOADING: "state-loading",
        LoadingState.SUCCESS: "state-success",
        LoadingState.ERROR: "state-error"
    }
    
    def watch_state(self, new_state: LoadingState) -> None:
        """Update visual state."""
        # Remove all state classes
        for cls in self.STATE_CLASSES.values():
            self.remove_class(cls)
        
        # Add new state class
        self.add_class(self.STATE_CLASSES[new_state])
        
        # Update content
        icon = self.STATE_ICONS[new_state]
        self.update(f"{icon} {self.message}")
        
        # Auto-reset after success/error
        if new_state in (LoadingState.SUCCESS, LoadingState.ERROR):
            self.set_timer(3.0, self._reset_state)
    
    def _reset_state(self) -> None:
        """Reset to idle state."""
        self.state = LoadingState.IDLE
        self.message = ""

class SmartListView(ListView):
    """ListView with built-in loading states."""
    
    loading = reactive(False)
    error_message = reactive("")
    empty_message = reactive("No items")
    
    def watch_loading(self, is_loading: bool) -> None:
        """Show/hide loading indicator."""
        if is_loading:
            self.clear()
            self.mount(LoadingIndicator())
        else:
            # Remove loading indicator if present
            for child in self.children:
                if isinstance(child, LoadingIndicator):
                    child.remove()
    
    def watch_error_message(self, message: str) -> None:
        """Display error state."""
        if message:
            self.clear()
            self.append(ListItem(
                Static(f"❌ {message}", classes="error-message")
            ))
    
    async def load_items(self, loader_func):
        """Load items with state management."""
        self.loading = True
        self.error_message = ""
        
        try:
            items = await loader_func()
            self.loading = False
            
            if not items:
                self.append(ListItem(
                    Static(self.empty_message, classes="empty-message")
                ))
            else:
                for item in items:
                    self.append(item)
        
        except Exception as e:
            self.loading = False
            self.error_message = str(e)
```

#### Progress Indicators for Long Operations

```python
from textual.widgets import ProgressBar

class ProgressOperation(Container):
    """Container for operations with progress tracking."""
    
    def compose(self) -> ComposeResult:
        yield Static("", id="progress-label")
        yield ProgressBar(id="progress-bar", total=100)
        yield Button("Cancel", id="cancel-operation", variant="warning")
    
    async def run_operation(self, operation_func, steps: List[str]):
        """Run operation with progress updates."""
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        label = self.query_one("#progress-label", Static)
        
        progress_bar.total = len(steps)
        
        for i, step in enumerate(steps):
            label.update(f"Step {i+1}/{len(steps)}: {step}")
            progress_bar.progress = i
            
            try:
                await operation_func(step)
            except Exception as e:
                label.update(f"❌ Failed at: {step}")
                self.notify(str(e), severity="error")
                break
        
        progress_bar.progress = progress_bar.total
        label.update("✅ Operation complete")
```

---

## 6. Contextual Help System

### Problem Statement
Users need guidance on complex settings without leaving the interface.

### Solution: Integrated Help System

#### Implementation

```python
# In help_system.py
from textual.widgets import Static, Button
from textual.containers import Container
import json

class HelpTooltip(Container):
    """Rich help tooltip with examples."""
    
    HELP_DATABASE = {
        "temperature": {
            "brief": "Controls randomness in responses",
            "detail": "Temperature affects how creative vs deterministic the model's responses are.",
            "range": "0.0 - 2.0",
            "examples": {
                "0.0": "Very consistent, deterministic",
                "0.7": "Balanced creativity (default)",
                "1.0": "Creative, varied responses",
                "2.0": "Very creative, may be inconsistent"
            }
        },
        "max_tokens": {
            "brief": "Maximum response length",
            "detail": "Sets the maximum number of tokens (words/pieces) in the response.",
            "range": "1 - model maximum",
            "examples": {
                "256": "Short responses (~200 words)",
                "1024": "Medium responses (~750 words)",
                "2048": "Long responses (~1500 words)"
            }
        }
    }
    
    def __init__(self, setting_id: str, **kwargs):
        super().__init__(**kwargs)
        self.setting_id = setting_id
        self.help_data = self.HELP_DATABASE.get(setting_id, {})
    
    def compose(self) -> ComposeResult:
        if not self.help_data:
            yield Static("No help available")
            return
        
        with Container(classes="help-content"):
            yield Static(self.help_data.get('brief', ''), classes="help-brief")
            
            if 'detail' in self.help_data:
                yield Static(self.help_data['detail'], classes="help-detail")
            
            if 'range' in self.help_data:
                yield Static(f"Range: {self.help_data['range']}", classes="help-range")
            
            if 'examples' in self.help_data:
                yield Static("Examples:", classes="help-examples-header")
                for value, description in self.help_data['examples'].items():
                    yield Static(f"  {value}: {description}", classes="help-example")

class SettingWithHelp(Horizontal):
    """Setting control with integrated help."""
    
    def __init__(self, label: str, setting_id: str, control_widget, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.setting_id = setting_id
        self.control_widget = control_widget
        self.help_visible = False
    
    def compose(self) -> ComposeResult:
        yield Label(self.label, classes="setting-label")
        yield Button("?", id=f"help-{self.setting_id}", classes="help-button")
        yield self.control_widget
        
        # Hidden help tooltip
        self.help_tooltip = HelpTooltip(self.setting_id, id=f"tooltip-{self.setting_id}")
        self.help_tooltip.display = False
        yield self.help_tooltip
    
    @on(Button.Pressed, ".help-button")
    async def toggle_help(self, event: Button.Pressed) -> None:
        """Toggle help visibility."""
        self.help_visible = not self.help_visible
        self.help_tooltip.display = self.help_visible
        
        if self.help_visible:
            event.button.label = "✕"
            self.add_class("help-active")
        else:
            event.button.label = "?"
            self.remove_class("help-active")
```

#### Interactive Tutorials

```python
class InteractiveTutorial(Container):
    """Step-by-step interactive tutorial system."""
    
    def __init__(self, tutorial_id: str, **kwargs):
        super().__init__(**kwargs)
        self.tutorial_id = tutorial_id
        self.current_step = 0
        self.steps = self._load_tutorial_steps()
    
    def _load_tutorial_steps(self) -> List[Dict]:
        """Load tutorial steps from configuration."""
        tutorials = {
            "first_chat": [
                {
                    "target": "#chat-api-provider",
                    "title": "Select Your AI Provider",
                    "content": "Choose your preferred AI service provider",
                    "action": "highlight"
                },
                {
                    "target": "#chat-api-model",
                    "title": "Choose a Model",
                    "content": "Select the AI model you want to use",
                    "action": "highlight"
                },
                {
                    "target": "#chat-input",
                    "title": "Type Your Message",
                    "content": "Enter your message here and press Enter to send",
                    "action": "focus"
                }
            ]
        }
        return tutorials.get(self.tutorial_id, [])
    
    async def start_tutorial(self) -> None:
        """Start the interactive tutorial."""
        self.current_step = 0
        await self._show_step()
    
    async def _show_step(self) -> None:
        """Display current tutorial step."""
        if self.current_step >= len(self.steps):
            await self._complete_tutorial()
            return
        
        step = self.steps[self.current_step]
        
        # Highlight target element
        try:
            target = self.app.query_one(step['target'])
            target.add_class("tutorial-highlight")
            
            # Show tooltip near target
            self.show_tooltip(
                step['title'],
                step['content'],
                near=target
            )
            
            # Perform action
            if step['action'] == 'focus':
                target.focus()
        
        except QueryError:
            logger.warning(f"Tutorial target not found: {step['target']}")
            await self._next_step()
    
    async def _next_step(self) -> None:
        """Move to next tutorial step."""
        # Clean up current step
        for element in self.app.query(".tutorial-highlight"):
            element.remove_class("tutorial-highlight")
        
        self.current_step += 1
        await self._show_step()
```

---

## 7. Settings Profiles & Presets

### Problem Statement
Users need to switch between different configurations for different use cases.

### Solution: Profile Management System

#### Implementation

```python
# In profiles.py
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class SettingsProfile:
    """Settings profile configuration."""
    name: str
    description: str
    settings: Dict[str, Any]
    is_default: bool = False
    is_builtin: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def apply(self, app: 'TldwCli') -> None:
        """Apply this profile's settings."""
        for key, value in self.settings.items():
            if hasattr(app, key):
                setattr(app, key, value)

class ProfileManager(Container):
    """Manage and apply settings profiles."""
    
    BUILTIN_PROFILES = [
        SettingsProfile(
            name="Creative Writing",
            description="Optimized for creative content generation",
            settings={
                "chat_temperature": 0.9,
                "chat_top_p": 0.95,
                "chat_frequency_penalty": 0.5,
                "chat_presence_penalty": 0.5
            },
            is_builtin=True
        ),
        SettingsProfile(
            name="Code Assistant",
            description="Optimized for code generation and debugging",
            settings={
                "chat_temperature": 0.3,
                "chat_top_p": 0.9,
                "chat_max_tokens": 2048,
                "chat_stop_sequences": ["```", "</code>"]
            },
            is_builtin=True
        ),
        SettingsProfile(
            name="Academic Research",
            description="Factual, citation-friendly responses",
            settings={
                "chat_temperature": 0.2,
                "chat_top_p": 0.85,
                "chat_system_prompt": "You are an academic research assistant. Provide factual, well-researched responses with citations where possible."
            },
            is_builtin=True
        )
    ]
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="profile-selector"):
            yield Select(
                options=self._get_profile_options(),
                id="profile-select",
                value="default",
                prompt="Select Profile..."
            )
            yield Button("💾", id="save-profile", tooltip="Save current as profile")
            yield Button("✏️", id="edit-profile", tooltip="Edit profile")
            yield Button("🗑️", id="delete-profile", tooltip="Delete profile")
    
    def _get_profile_options(self) -> List[Tuple[str, str]]:
        """Get available profiles for dropdown."""
        options = [("default", "Default Settings")]
        
        # Add builtin profiles
        for profile in self.BUILTIN_PROFILES:
            options.append((
                profile.name.lower().replace(" ", "_"),
                f"⭐ {profile.name}"
            ))
        
        # Add user profiles
        user_profiles = self._load_user_profiles()
        for profile in user_profiles:
            options.append((
                profile.name.lower().replace(" ", "_"),
                profile.name
            ))
        
        return options
    
    @on(Select.Changed, "#profile-select")
    async def apply_profile(self, event: Select.Changed) -> None:
        """Apply selected profile."""
        profile_id = event.value
        
        if profile_id == "default":
            await self._apply_default_settings()
        else:
            profile = self._get_profile_by_id(profile_id)
            if profile:
                profile.apply(self.app)
                self.notify(f"Applied profile: {profile.name}", severity="success")
    
    @on(Button.Pressed, "#save-profile")
    async def save_current_as_profile(self) -> None:
        """Save current settings as a new profile."""
        # Show dialog for profile name
        name = await self.app.push_screen(
            InputDialog("Profile Name", "Enter a name for this profile:")
        )
        
        if name:
            profile = SettingsProfile(
                name=name,
                description="User-created profile",
                settings=self._capture_current_settings()
            )
            
            self._save_user_profile(profile)
            self.notify(f"Saved profile: {name}", severity="success")
            
            # Refresh dropdown
            select = self.query_one("#profile-select", Select)
            select.set_options(self._get_profile_options())
    
    def _capture_current_settings(self) -> Dict[str, Any]:
        """Capture current settings state."""
        settings = {}
        
        # List of settings to capture
        setting_ids = [
            "chat_temperature", "chat_top_p", "chat_max_tokens",
            "chat_system_prompt", "chat_api_provider", "chat_api_model"
        ]
        
        for setting_id in setting_ids:
            if hasattr(self.app, setting_id):
                settings[setting_id] = getattr(self.app, setting_id)
        
        return settings
```

---

## 8. Keyboard Navigation

### Problem Statement
Power users need efficient keyboard-only navigation through settings.

### Solution: Comprehensive Keyboard Support

#### Implementation

```python
# In keyboard_navigation.py
from textual.binding import Binding

class KeyboardNavigableSidebar(VerticalScroll):
    """Sidebar with full keyboard navigation support."""
    
    BINDINGS = [
        # Navigation
        Binding("ctrl+\\", "toggle_sidebar", "Toggle Sidebar"),
        Binding("ctrl+/", "focus_search", "Search Settings"),
        Binding("tab", "next_section", "Next Section"),
        Binding("shift+tab", "prev_section", "Previous Section"),
        
        # Section jumps
        Binding("alt+1", "jump_to_model", "Model Settings"),
        Binding("alt+2", "jump_to_conversation", "Conversation"),
        Binding("alt+3", "jump_to_character", "Character"),
        Binding("alt+4", "jump_to_rag", "RAG Settings"),
        Binding("alt+5", "jump_to_advanced", "Advanced"),
        
        # Actions
        Binding("ctrl+enter", "apply_settings", "Apply"),
        Binding("escape", "cancel_changes", "Cancel"),
        Binding("ctrl+r", "reset_section", "Reset Section"),
        
        # Quick settings
        Binding("ctrl+t", "quick_temperature", "Temperature"),
        Binding("ctrl+m", "quick_model", "Model Select"),
        Binding("ctrl+p", "quick_system_prompt", "System Prompt"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.focusable_elements = []
        self.current_focus_index = 0
    
    def on_mount(self) -> None:
        """Build focusable elements list."""
        self._update_focusable_elements()
    
    def _update_focusable_elements(self):
        """Update list of focusable elements."""
        self.focusable_elements = list(
            self.query("Input, Select, TextArea, Button, Checkbox, RadioButton")
        )
    
    def action_next_section(self) -> None:
        """Move to next collapsible section."""
        sections = list(self.query(Collapsible))
        current = self.focused
        
        # Find current section
        current_section = None
        for section in sections:
            if current in section.query("*"):
                current_section = section
                break
        
        if current_section and sections.index(current_section) < len(sections) - 1:
            next_section = sections[sections.index(current_section) + 1]
            next_section.collapsed = False
            
            # Focus first input in next section
            first_input = next_section.query_first("Input, Select, TextArea")
            if first_input:
                first_input.focus()
    
    def action_jump_to_model(self) -> None:
        """Jump directly to model settings."""
        self._jump_to_section("model-config")
    
    def _jump_to_section(self, section_id: str) -> None:
        """Jump to a specific section by ID."""
        try:
            section = self.query_one(f"#{section_id}", Collapsible)
            section.collapsed = False
            section.scroll_visible()
            
            # Focus first input
            first_input = section.query_first("Input, Select, TextArea")
            if first_input:
                first_input.focus()
        except QueryError:
            pass
    
    def action_quick_temperature(self) -> None:
        """Quick jump to temperature setting."""
        try:
            temp_input = self.query_one("#chat-temperature", Input)
            temp_input.focus()
            temp_input.action_select_all()  # Select current value for quick edit
        except QueryError:
            pass
```

#### Vi-style Navigation

```python
class ViStyleNavigation:
    """Vi-style keyboard navigation for power users."""
    
    BINDINGS = [
        Binding("j", "move_down", "Down", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("g g", "move_top", "Top", show=False),
        Binding("G", "move_bottom", "Bottom", show=False),
        Binding("/", "start_search", "Search", show=False),
        Binding("n", "next_match", "Next Match", show=False),
        Binding("N", "prev_match", "Previous Match", show=False),
    ]
    
    def __init__(self):
        self.vi_mode = False
        self.search_matches = []
        self.current_match = 0
    
    def action_move_down(self) -> None:
        """Move to next element (j key)."""
        if self.vi_mode:
            self._move_focus(1)
    
    def action_move_up(self) -> None:
        """Move to previous element (k key)."""
        if self.vi_mode:
            self._move_focus(-1)
    
    def _move_focus(self, direction: int) -> None:
        """Move focus by direction."""
        elements = self.focusable_elements
        if not elements:
            return
        
        current = self.focused
        if current in elements:
            idx = elements.index(current)
            new_idx = (idx + direction) % len(elements)
            elements[new_idx].focus()
```

---

## 9. Responsive Design & Resizing

### Problem Statement
Fixed sidebar width doesn't adapt to different screen sizes and user preferences.

### Solution: Flexible Resizing System

#### Implementation

```python
# In resizable_sidebar.py
from textual.reactive import reactive
from textual.geometry import Offset

class ResizableSidebar(Container):
    """Sidebar with drag-to-resize functionality."""
    
    width_percent = reactive(35)  # Percentage of screen width
    min_width = 20  # Minimum width in cells
    max_width = 60  # Maximum width in cells
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="resizable-container"):
            # Sidebar content
            with VerticalScroll(
                id="sidebar-content",
                classes="sidebar-content"
            ) as sidebar:
                yield from create_settings_sidebar(self.id_prefix, self.config)
            
            # Resize handle
            yield Static(
                "⋮",
                id="resize-handle",
                classes="resize-handle"
            )
    
    def on_mount(self) -> None:
        """Set initial width from saved preferences."""
        saved_width = self.app.user_preferences.get('sidebar_width', 35)
        self.width_percent = saved_width
        self._apply_width()
    
    def watch_width_percent(self, new_width: int) -> None:
        """Update sidebar width when percentage changes."""
        self._apply_width()
        # Save preference
        self.app.user_preferences['sidebar_width'] = new_width
    
    def _apply_width(self) -> None:
        """Apply the current width to the sidebar."""
        sidebar = self.query_one("#sidebar-content")
        screen_width = self.app.size.width
        
        # Calculate actual width in cells
        width = int(screen_width * (self.width_percent / 100))
        width = max(self.min_width, min(width, self.max_width))
        
        sidebar.styles.width = f"{width}"
    
    @on(MouseDown, "#resize-handle")
    def start_resize(self, event: MouseDown) -> None:
        """Start resizing operation."""
        self.capture_mouse()
        self.add_class("resizing")
        self.resize_start_x = event.x
        self.resize_start_width = self.width_percent
    
    @on(MouseMove)
    def handle_resize(self, event: MouseMove) -> None:
        """Handle resize dragging."""
        if self.mouse_captured:
            # Calculate new width based on mouse movement
            delta = event.x - self.resize_start_x
            screen_width = self.app.size.width
            
            # Convert pixel delta to percentage
            percent_delta = (delta / screen_width) * 100
            new_width = self.resize_start_width + percent_delta
            
            # Clamp to valid range
            new_width = max(20, min(60, new_width))
            self.width_percent = new_width
            
            # Show width indicator
            self.show_resize_indicator(new_width)
    
    @on(MouseUp)
    def end_resize(self, event: MouseUp) -> None:
        """End resizing operation."""
        if self.mouse_captured:
            self.release_mouse()
            self.remove_class("resizing")
            self.hide_resize_indicator()
    
    def show_resize_indicator(self, width: int) -> None:
        """Show visual indicator of current width."""
        indicator = self.query_one("#width-indicator", Static)
        indicator.update(f"{width}%")
        indicator.display = True
    
    def hide_resize_indicator(self) -> None:
        """Hide width indicator."""
        indicator = self.query_one("#width-indicator", Static)
        indicator.display = False
```

#### Responsive Breakpoints

```python
class ResponsiveSidebar(ResizableSidebar):
    """Sidebar that adapts to screen size."""
    
    BREAKPOINTS = {
        'mobile': 768,
        'tablet': 1024,
        'desktop': 1440,
        'wide': 1920
    }
    
    def on_resize(self, event: Resize) -> None:
        """Adapt to screen size changes."""
        width = event.size.width
        
        if width < self.BREAKPOINTS['mobile']:
            self.enter_mobile_mode()
        elif width < self.BREAKPOINTS['tablet']:
            self.enter_tablet_mode()
        elif width < self.BREAKPOINTS['desktop']:
            self.enter_desktop_mode()
        else:
            self.enter_wide_mode()
    
    def enter_mobile_mode(self) -> None:
        """Optimize for mobile screens."""
        self.add_class("mobile-mode")
        # Auto-collapse non-essential sections
        for collapsible in self.query(".settings-collapsible"):
            if "priority-high" not in collapsible.classes:
                collapsible.collapsed = True
        
        # Use overlay mode
        self.styles.position = "absolute"
        self.styles.z_index = 100
    
    def enter_tablet_mode(self) -> None:
        """Optimize for tablet screens."""
        self.remove_class("mobile-mode")
        self.add_class("tablet-mode")
        
        # Adjust width
        self.width_percent = 40
        
        # Show essential controls only
        for element in self.query(".advanced-only"):
            element.display = False
    
    def enter_desktop_mode(self) -> None:
        """Standard desktop layout."""
        self.remove_class("tablet-mode")
        self.width_percent = 35
        
        # Show all controls
        for element in self.query(".advanced-only"):
            element.display = True
```

---

## 10. Sidebar Display Modes

### Problem Statement
Different users prefer different levels of detail and screen real estate usage.

### Solution: Multiple Display Modes

#### Implementation

```python
# In sidebar_modes.py
from enum import Enum

class SidebarMode(Enum):
    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"
    FLOATING = "floating"
    HIDDEN = "hidden"

class MultiModeSidebar(Container):
    """Sidebar with multiple display modes."""
    
    mode = reactive(SidebarMode.FULL)
    
    def compose(self) -> ComposeResult:
        # Mode selector
        with Horizontal(classes="mode-selector"):
            yield Button("◧", id="mode-full", tooltip="Full Mode")
            yield Button("▭", id="mode-compact", tooltip="Compact Mode")
            yield Button("⋯", id="mode-minimal", tooltip="Minimal Mode")
            yield Button("⧉", id="mode-floating", tooltip="Floating Mode")
        
        # Sidebar content
        with Container(id="sidebar-content", classes="sidebar-mode-full"):
            yield from self._create_mode_content()
    
    def watch_mode(self, new_mode: SidebarMode) -> None:
        """Switch sidebar mode."""
        content = self.query_one("#sidebar-content")
        
        # Remove all mode classes
        for mode in SidebarMode:
            content.remove_class(f"sidebar-mode-{mode.value}")
        
        # Add new mode class
        content.add_class(f"sidebar-mode-{new_mode.value}")
        
        # Update content based on mode
        if new_mode == SidebarMode.COMPACT:
            self._enter_compact_mode()
        elif new_mode == SidebarMode.MINIMAL:
            self._enter_minimal_mode()
        elif new_mode == SidebarMode.FLOATING:
            self._enter_floating_mode()
        else:
            self._enter_full_mode()
    
    def _enter_compact_mode(self) -> None:
        """Enter compact mode - labels become placeholders."""
        # Hide labels
        for label in self.query(".setting-label"):
            label.display = False
        
        # Move label text to placeholder
        for container in self.query(".setting-row"):
            label = container.query_first(".setting-label")
            input_field = container.query_first("Input, Select, TextArea")
            
            if label and input_field:
                input_field.placeholder = label.text
        
        # Reduce padding
        self.styles.padding = 1
        
        # Collapse section headers
        for header in self.query(".group-header"):
            header.styles.height = "1"
    
    def _enter_minimal_mode(self) -> None:
        """Enter minimal mode - icons only."""
        # Show only icon buttons
        for widget in self.query("*"):
            if isinstance(widget, Button) and len(widget.label) == 1:
                widget.display = True
            elif widget.has_class("essential-control"):
                widget.display = True
            else:
                widget.display = False
        
        # Ultra-compact width
        self.styles.width = "5"
    
    def _enter_floating_mode(self) -> None:
        """Enter floating mode - overlay on demand."""
        self.styles.position = "absolute"
        self.styles.z_index = 1000
        self.styles.background = "$panel-lighten-1"
        self.styles.border = "solid $accent"
        
        # Add close button
        close_btn = Button("✕", id="close-floating", classes="floating-close")
        self.mount(close_btn, before=0)
        
        # Make draggable
        self.add_class("draggable")
```

#### Mode-Specific CSS (Textual-Compatible)

```css
/* Full mode - default */
.sidebar-mode-full {
    width: 35%;
    padding: 2;
}

.sidebar-mode-full .setting-label {
    display: block;
    margin-bottom: 1;
}

/* Compact mode */
.sidebar-mode-compact {
    width: 25%;
    padding: 1;
}

.sidebar-mode-compact .setting-row {
    margin-bottom: 1;
}

.sidebar-mode-compact Input,
.sidebar-mode-compact Select {
    width: 100%;
}

/* Minimal mode */
.sidebar-mode-minimal {
    width: 5;
    padding: 0;
    overflow-y: hidden;
}

.sidebar-mode-minimal Button {
    width: 100%;
    padding: 0;
    text-align: center;
}

/* Floating mode - using Textual's layer system */
.sidebar-mode-floating {
    layer: above;  /* Use layers instead of position: absolute */
    dock: left;
    width: 30%;
    max-height: 80%;
    border: thick $accent;
    background: $panel 95%;
}

/* Visual feedback for draggable elements */
.draggable {
    border: dashed $primary;
}

.draggable:hover {
    background: $primary 10%;
    border: solid $primary;
}
```

---

## 11. Form Validation & Error Handling

### Problem Statement
Users need immediate feedback on invalid inputs and clear error messages.

### Solution: Real-time Validation System

#### Implementation

```python
# In validation.py
from typing import Optional, Callable, Any
from dataclasses import dataclass

@dataclass
class ValidationRule:
    """Validation rule for form inputs."""
    validator: Callable[[Any], bool]
    error_message: str
    warning_threshold: Optional[Callable[[Any], bool]] = None
    warning_message: Optional[str] = None

class ValidatedInput(Input):
    """Input with built-in validation."""
    
    def __init__(
        self,
        validation_rules: List[ValidationRule] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.validation_rules = validation_rules or []
        self.is_valid = True
        self.validation_messages = []
    
    @on(Input.Changed)
    async def validate_input(self, event: Input.Changed) -> None:
        """Validate input on change."""
        value = event.value
        self.validation_messages.clear()
        self.is_valid = True
        
        # Check each validation rule
        for rule in self.validation_rules:
            # Check warning threshold first
            if rule.warning_threshold and rule.warning_threshold(value):
                self.add_class("input-warning")
                self.validation_messages.append(("warning", rule.warning_message))
            
            # Check validation
            if not rule.validator(value):
                self.is_valid = False
                self.add_class("input-error")
                self.remove_class("input-valid")
                self.validation_messages.append(("error", rule.error_message))
                break
        
        if self.is_valid:
            self.remove_class("input-error")
            self.add_class("input-valid")
            
            # Show success briefly
            self.add_class("input-success")
            self.set_timer(1.0, lambda: self.remove_class("input-success"))
        
        # Update tooltip with validation message
        if self.validation_messages:
            level, message = self.validation_messages[0]
            self.tooltip = f"{level.upper()}: {message}"
        else:
            self.tooltip = None
    
    @on(Input.Submitted)
    async def handle_submit(self, event: Input.Submitted) -> None:
        """Prevent submission if invalid."""
        if not self.is_valid:
            event.stop()
            self.add_class("shake")
            self.set_timer(0.5, lambda: self.remove_class("shake"))
            
            # Show error notification
            self.notify(
                f"Invalid input: {self.validation_messages[0][1]}",
                severity="error"
            )

class FormValidator:
    """Centralized form validation."""
    
    VALIDATORS = {
        'temperature': [
            ValidationRule(
                validator=lambda x: x.replace('.', '').isdigit(),
                error_message="Must be a number"
            ),
            ValidationRule(
                validator=lambda x: 0 <= float(x) <= 2,
                error_message="Must be between 0 and 2",
                warning_threshold=lambda x: float(x) > 1.5,
                warning_message="High temperature may produce inconsistent results"
            )
        ],
        'max_tokens': [
            ValidationRule(
                validator=lambda x: x.isdigit(),
                error_message="Must be a positive integer"
            ),
            ValidationRule(
                validator=lambda x: int(x) > 0,
                error_message="Must be greater than 0"
            ),
            ValidationRule(
                validator=lambda x: int(x) <= 32000,
                error_message="Exceeds maximum token limit",
                warning_threshold=lambda x: int(x) > 4096,
                warning_message="Large token counts may be slow and expensive"
            )
        ],
        'api_key': [
            ValidationRule(
                validator=lambda x: len(x) > 20,
                error_message="API key seems too short"
            ),
            ValidationRule(
                validator=lambda x: not ' ' in x,
                error_message="API key contains spaces"
            )
        ]
    }
    
    @classmethod
    def create_validated_input(cls, input_type: str, **kwargs) -> ValidatedInput:
        """Create input with appropriate validation rules."""
        rules = cls.VALIDATORS.get(input_type, [])
        return ValidatedInput(validation_rules=rules, **kwargs)
```

#### Visual Validation Feedback (Textual-Compatible)

```css
/* Validation states */
.input-valid {
    border: solid $success;
}

.input-error {
    border: solid $error;
    background: $error 10%;  /* Textual supports alpha percentages */
}

.input-warning {
    border: solid $warning;
    background: $warning 10%;
}

.input-success {
    background: $success 20%;
    border: solid $success;
}

/* Error state - use visual styling instead of animation */
.shake {
    border: thick $error;
    background: $error 20%;
}

/* Validation messages */
.validation-message {
    margin-top: 1;
    padding: 1;
    border: round $border;
}

.validation-message.error {
    color: $error;
    background: $error 10%;
    border: solid $error;
}

.validation-message.warning {
    color: $warning;
    background: $warning 10%;
    border: solid $warning;
}
```

---

## 12. CSS & Styling Improvements

### Comprehensive Theme System (Textual-Compatible)

```css
/* Note: Textual doesn't support CSS variables, animations, or transforms */
/* Use consistent values directly in styles */

/* Better focus indicators for accessibility */
.sidebar *:focus {
    outline: solid $accent;
}

/* Scrollbar styling - Textual has its own scrollbar implementation */
.sidebar {
    scrollbar-size: 1;  /* Width of scrollbar in cells */
    scrollbar-gutter: stable;
}

/* Floating panel effect - using supported properties */
.floating-panel {
    background: $panel 95%;  /* Background with opacity */
    border: thick $accent 20%;
    dock: top;  /* Position using dock instead of absolute positioning */
}

/* Collapsible styling - Textual handles collapsible internally */
.collapsible.-collapsed > Contents {
    display: none;
}

.collapsible:not(.-collapsed) > Contents {
    display: block;
    height: auto;
}

/* Hover effects for interactive elements */
.sidebar Button:hover {
    background: $primary-lighten-1;
    text-style: bold;
}

.sidebar Button:focus {
    background: $primary-darken-1;
    outline: solid $accent;
}

/* Status indicators - use text styling instead of animations */
.status-indicator.loading {
    color: $warning;
    text-style: italic;
}

.status-indicator.processing {
    color: $primary;
    text-style: bold;
}

/* Typography - using Textual's text-style property */
.sidebar {
    /* Textual doesn't support font-size or line-height */
    padding: 1 2;
}

.sidebar .group-header {
    text-style: bold;
    color: $text-muted;
}

.sidebar .setting-label {
    text-style: bold;
    color: $text;
}

/* Dark mode optimizations using :dark pseudo-class */
.sidebar:dark {
    background: $background-darken-1;
}

.sidebar:dark Input,
.sidebar:dark Select,
.sidebar:dark TextArea {
    background: $background;
    border: solid $primary-lighten-3;
}

.sidebar:dark Input:focus,
.sidebar:dark Select:focus,
.sidebar:dark TextArea:focus {
    border: solid $accent;
    background: $background-lighten-1;
}
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. **Visual Hierarchy Reorganization**
   - Implement semantic grouping
   - Add section headers and dividers
   - Update CSS for improved spacing

2. **State Management**
   - Implement sidebar state persistence
   - Add collapsible state tracking
   - Create preference storage system

### Phase 2: Core Features (Week 3-4)
1. **Search System**
   - Implement search indexing
   - Add real-time filtering
   - Create search highlighting

2. **Quick Actions Bar**
   - Design and implement toolbar
   - Add context-aware actions
   - Implement action animations

### Phase 3: Enhanced UX (Week 5-6)
1. **Visual Feedback**
   - Add loading states
   - Implement progress indicators
   - Create status notifications

2. **Form Validation**
   - Implement validation rules
   - Add real-time feedback
   - Create error handling

### Phase 4: Advanced Features (Week 7-8)
1. **Profiles System**
   - Create profile management
   - Implement preset configurations
   - Add import/export functionality

2. **Keyboard Navigation**
   - Implement comprehensive bindings
   - Add section jumping
   - Create navigation indicators

### Phase 5: Polish & Optimization (Week 9-10)
1. **Responsive Design**
   - Implement resizable sidebar
   - Add breakpoint handling
   - Create mobile optimizations

2. **Display Modes**
   - Implement multiple view modes
   - Add mode transitions
   - Create mode-specific layouts

### Phase 6: Testing & Refinement (Week 11-12)
1. **User Testing**
   - Conduct usability testing
   - Gather feedback
   - Iterate on problem areas

2. **Performance Optimization**
   - Profile rendering performance
   - Optimize search indexing
   - Reduce re-render cycles

## Testing Strategy

### Unit Tests
```python
# test_sidebar_state.py
def test_sidebar_state_persistence():
    """Test that sidebar state persists across sessions."""
    sidebar = SidebarWithState()
    sidebar.collapsible_states['test-section'] = True
    sidebar.save_state()
    
    new_sidebar = SidebarWithState()
    new_sidebar.load_state()
    
    assert new_sidebar.collapsible_states['test-section'] == True

def test_search_filtering():
    """Test search filtering accuracy."""
    search = SidebarSearch()
    search.search_cache = {
        'temp': {'text': 'temperature', 'element': Mock()},
        'model': {'text': 'model selection', 'element': Mock()}
    }
    
    matches = search.filter_items('temp')
    assert len(matches) == 1
    assert matches[0]['text'] == 'temperature'
```

### Integration Tests
```python
# test_sidebar_integration.py
async def test_profile_application():
    """Test that profiles correctly apply settings."""
    app = MockApp()
    profile_manager = ProfileManager(app)
    
    creative_profile = profile_manager.BUILTIN_PROFILES[0]
    await profile_manager.apply_profile(creative_profile)
    
    assert app.chat_temperature == 0.9
    assert app.chat_top_p == 0.95
```

## Performance Considerations

1. **Lazy Loading**: Load sidebar sections on-demand
2. **Virtual Scrolling**: For long lists, implement virtual scrolling
3. **Debounced Search**: Debounce search input to reduce processing
4. **Memoized Computations**: Cache expensive calculations
5. **Optimized Re-renders**: Use reactive properties judiciously

## Accessibility Requirements

1. **Keyboard Navigation**: Full functionality without mouse
2. **Screen Reader Support**: Proper ARIA labels and roles
3. **Focus Management**: Clear focus indicators and logical tab order
4. **Color Contrast**: WCAG AA compliance minimum
5. **Text Scaling**: Support for user font size preferences

## Conclusion

This comprehensive improvement plan transforms the sidebar from a basic settings panel into a powerful, user-friendly control center. By implementing these features following Textual best practices, we create an interface that serves both novice users with its intuitive organization and power users with advanced features like keyboard navigation and profiles.

The phased implementation approach ensures that each feature is properly tested and refined before moving to the next, resulting in a robust and polished user experience.