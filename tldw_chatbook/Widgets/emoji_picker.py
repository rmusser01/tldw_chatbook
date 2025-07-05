# emoji_picker.py
#
# Imports
from typing import List, Dict, Tuple, Optional, Set, Any
from pathlib import Path
#
# 3rd-party Libraries
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError  # Import QueryError
from textual.message import Message
# from textual.message import Message # Not used directly, can remove
# from textual.reactive import reactive # No longer needed if search_results reactive is removed
# from textual.widget import Widget # Not used directly, can remove
from textual.screen import ModalScreen
from textual.widgets import Button, Input, TabbedContent, TabPane, Static, Label
from textual.timer import Timer

# Try to get the richer EMOJI_DATA from unicode_codes if available (recommended)
try:
    from emoji.unicode_codes import EMOJI_DATA as EMOJI_METADATA

    EMOJI_SOURCE_TYPE = "unicode_codes.EMOJI_DATA"
except ImportError:
    import emoji  # Import here if fallback is used

    EMOJI_METADATA = emoji.EMOJI_DATA
    EMOJI_SOURCE_TYPE = "emoji.EMOJI_DATA"
#
# Local Imports
#
########################################################################################################################
#
# Classes:

# --- Emoji Data Loading and Processing ---
PREFERRED_CATEGORY_ORDER = [
    "Recently Used",  # Added recently used as first category
    "Smileys & Emotion", "People & Body", "Animals & Nature", "Food & Drink",
    "Travel & Places", "Activities", "Objects", "Symbols", "Flags",
]
ProcessedEmoji = Dict[str, Any]  # {'char': str, 'name': str, 'category': str, 'aliases': List[str]}

# Storage for recently used emojis
RECENT_EMOJIS_FILE = Path.home() / ".config" / "tldw_cli" / "recent_emojis.json"
MAX_RECENT_EMOJIS = 30


def load_recent_emojis() -> List[str]:
    """Load recently used emojis from file."""
    try:
        if RECENT_EMOJIS_FILE.exists():
            import json
            with open(RECENT_EMOJIS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('recent', [])[:MAX_RECENT_EMOJIS]
    except Exception:
        pass
    return []


def save_recent_emoji(emoji_char: str) -> None:
    """Save an emoji to the recently used list."""
    try:
        # Ensure config directory exists
        RECENT_EMOJIS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing recent emojis
        recent = load_recent_emojis()
        
        # Remove if already exists (to move to front)
        if emoji_char in recent:
            recent.remove(emoji_char)
        
        # Add to front
        recent.insert(0, emoji_char)
        
        # Limit to max
        recent = recent[:MAX_RECENT_EMOJIS]
        
        # Save back
        import json
        with open(RECENT_EMOJIS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'recent': recent}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Fail silently for recent emojis


def _load_emojis() -> Tuple[List[ProcessedEmoji], Dict[str, List[ProcessedEmoji]], List[str]]:
    all_emojis_list: List[ProcessedEmoji] = []
    categorized_emojis: Dict[str, List[ProcessedEmoji]] = {}
    category_names_set: Set[str] = set()

    if EMOJI_SOURCE_TYPE == "unicode_codes.EMOJI_DATA":
        for char, data in EMOJI_METADATA.items():
            # The emoji character is the key now, not in data
            name = data.get('en', '').strip(':').replace('_', ' ')
            if not name:
                continue
            
            # Enhanced category detection with more patterns and Unicode ranges
            name_lower = name.lower()
            
            # Check Unicode ranges first for more accurate categorization
            # Handle multi-character emojis (with variation selectors, etc.)
            try:
                # Get the first character's code point
                char_code = ord(char[0])
                
                # Emoji Unicode blocks (approximate ranges)
                if 0x1F600 <= char_code <= 0x1F64F:  # Emoticons
                    category = "Smileys & Emotion"
                elif 0x1F300 <= char_code <= 0x1F5FF:  # Misc Symbols and Pictographs
                    if any(word in name_lower for word in ['weather', 'sun', 'moon', 'cloud', 'rain', 'snow', 'tree', 'flower', 'plant']):
                        category = "Animals & Nature"
                    elif any(word in name_lower for word in ['food', 'fruit', 'vegetable', 'meal', 'drink']):
                        category = "Food & Drink"
                    else:
                        category = "Objects"
                elif 0x1F680 <= char_code <= 0x1F6FF:  # Transport and Map
                    category = "Travel & Places"
                elif 0x1F1E6 <= char_code <= 0x1F1FF:  # Regional Indicator
                    category = "Flags"
                else:
                    # Unicode range didn't match, fall through to name-based detection
                    char_code = None
            except (TypeError, IndexError):
                # If ord() fails or char is empty, fall through to name-based detection
                char_code = None
            
            if char_code is None:
                # Fallback to name-based detection with expanded patterns
                if any(word in name_lower for word in ['face', 'smile', 'frown', 'cry', 'laugh', 'wink', 'kiss', 'tongue', 'angry', 'sad', 'happy', 'joy', 'tear', 'sweat', 'think', 'neutral', 'expressionless', 'grin']):
                    category = "Smileys & Emotion"
                elif any(word in name_lower for word in ['person', 'man', 'woman', 'boy', 'girl', 'baby', 'hand', 'finger', 'body', 'people', 'family', 'couple', 'skin', 'tone', 'gesture']):
                    category = "People & Body"
                elif any(word in name_lower for word in ['cat', 'dog', 'animal', 'bird', 'fish', 'bug', 'monkey', 'horse', 'cow', 'pig', 'bear', 'panda', 'koala', 'tiger', 'lion', 'whale', 'dolphin', 'nature', 'tree', 'flower', 'plant', 'leaf']):
                    category = "Animals & Nature"
                elif any(word in name_lower for word in ['food', 'fruit', 'vegetable', 'drink', 'coffee', 'tea', 'wine', 'beer', 'pizza', 'burger', 'fries', 'sushi', 'cake', 'candy', 'chocolate', 'ice', 'cream', 'bread', 'cheese', 'meat', 'bacon']):
                    category = "Food & Drink"
                elif any(word in name_lower for word in ['car', 'bus', 'train', 'plane', 'ship', 'travel', 'place', 'building', 'house', 'city', 'mountain', 'beach', 'island', 'bridge', 'statue', 'fountain', 'castle', 'church', 'hospital', 'school', 'office', 'factory']):
                    category = "Travel & Places"
                elif any(word in name_lower for word in ['sport', 'ball', 'game', 'medal', 'trophy', 'music', 'art', 'paint', 'soccer', 'basketball', 'football', 'baseball', 'tennis', 'golf', 'ski', 'surf', 'swim', 'run', 'dance', 'sing', 'guitar', 'piano', 'drum']):
                    category = "Activities"
                elif any(word in name_lower for word in ['heart', 'star', 'circle', 'square', 'flag', 'symbol', 'sign', 'arrow', 'diamond', 'club', 'spade', 'cross', 'check', 'mark', 'x', 'exclamation', 'question', 'warning', 'prohibited']):
                    category = "Symbols"
                elif 'flag' in name_lower:
                    category = "Flags"
                else:
                    category = "Objects"
            
            aliases = data.get('alias', [])
            if isinstance(aliases, str): 
                aliases = [aliases]

            emoji_obj: ProcessedEmoji = {
                'char': char,
                'name': name.replace('_', ' ').title(),
                'category': category,
                'aliases': [a.strip(':') for a in aliases] if aliases else []
            }
            all_emojis_list.append(emoji_obj)

            if category not in categorized_emojis: 
                categorized_emojis[category] = []
            categorized_emojis[category].append(emoji_obj)
            category_names_set.add(category)

    elif EMOJI_SOURCE_TYPE == "emoji.EMOJI_DATA":
        # Fallback: less feature-rich (e.g., no categories from this source directly)
        # import emoji # Already imported at the top if this path is taken
        for char, data_val in EMOJI_METADATA.items():  # Renamed data to data_val to avoid conflict
            # Try to get a name from 'en' field or aliases
            name = data_val.get('en', '').strip(':').replace('_', ' ')
            if not name:
                aliases = data_val.get('alias', [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                if aliases:
                    name = aliases[0].strip(':').replace('_', ' ')
                else:
                    name = 'Emoji'
            
            # Enhanced category detection with more patterns and Unicode ranges
            name_lower = name.lower()
            
            # Check Unicode ranges first for more accurate categorization
            # Handle multi-character emojis (with variation selectors, etc.)
            try:
                # Get the first character's code point
                char_code = ord(char[0])
                
                # Emoji Unicode blocks (approximate ranges)
                if 0x1F600 <= char_code <= 0x1F64F:  # Emoticons
                    category = "Smileys & Emotion"
                elif 0x1F300 <= char_code <= 0x1F5FF:  # Misc Symbols and Pictographs
                    if any(word in name_lower for word in ['weather', 'sun', 'moon', 'cloud', 'rain', 'snow', 'tree', 'flower', 'plant']):
                        category = "Animals & Nature"
                    elif any(word in name_lower for word in ['food', 'fruit', 'vegetable', 'meal', 'drink']):
                        category = "Food & Drink"
                    else:
                        category = "Objects"
                elif 0x1F680 <= char_code <= 0x1F6FF:  # Transport and Map
                    category = "Travel & Places"
                elif 0x1F1E6 <= char_code <= 0x1F1FF:  # Regional Indicator
                    category = "Flags"
                else:
                    # Unicode range didn't match, fall through to name-based detection
                    char_code = None
            except (TypeError, IndexError):
                # If ord() fails or char is empty, fall through to name-based detection
                char_code = None
            
            if char_code is None:
                # Fallback to name-based detection with expanded patterns
                if any(word in name_lower for word in ['face', 'smile', 'frown', 'cry', 'laugh', 'wink', 'kiss', 'tongue', 'angry', 'sad', 'happy', 'joy', 'tear', 'sweat', 'think', 'neutral', 'expressionless', 'grin']):
                    category = "Smileys & Emotion"
                elif any(word in name_lower for word in ['person', 'man', 'woman', 'boy', 'girl', 'baby', 'hand', 'finger', 'body', 'people', 'family', 'couple', 'skin', 'tone', 'gesture']):
                    category = "People & Body"
                elif any(word in name_lower for word in ['cat', 'dog', 'animal', 'bird', 'fish', 'bug', 'monkey', 'horse', 'cow', 'pig', 'bear', 'panda', 'koala', 'tiger', 'lion', 'whale', 'dolphin', 'nature', 'tree', 'flower', 'plant', 'leaf']):
                    category = "Animals & Nature"
                elif any(word in name_lower for word in ['food', 'fruit', 'vegetable', 'drink', 'coffee', 'tea', 'wine', 'beer', 'pizza', 'burger', 'fries', 'sushi', 'cake', 'candy', 'chocolate', 'ice', 'cream', 'bread', 'cheese', 'meat', 'bacon']):
                    category = "Food & Drink"
                elif any(word in name_lower for word in ['car', 'bus', 'train', 'plane', 'ship', 'travel', 'place', 'building', 'house', 'city', 'mountain', 'beach', 'island', 'bridge', 'statue', 'fountain', 'castle', 'church', 'hospital', 'school', 'office', 'factory']):
                    category = "Travel & Places"
                elif any(word in name_lower for word in ['sport', 'ball', 'game', 'medal', 'trophy', 'music', 'art', 'paint', 'soccer', 'basketball', 'football', 'baseball', 'tennis', 'golf', 'ski', 'surf', 'swim', 'run', 'dance', 'sing', 'guitar', 'piano', 'drum']):
                    category = "Activities"
                elif any(word in name_lower for word in ['heart', 'star', 'circle', 'square', 'flag', 'symbol', 'sign', 'arrow', 'diamond', 'club', 'spade', 'cross', 'check', 'mark', 'x', 'exclamation', 'question', 'warning', 'prohibited']):
                    category = "Symbols"
                elif 'flag' in name_lower:
                    category = "Flags"
                else:
                    category = "Objects"
            aliases = data_val.get('alias', [])
            if isinstance(aliases, str): 
                aliases = [aliases]

            emoji_obj: ProcessedEmoji = {
                'char': char,
                'name': name.replace('_', ' ').title(),
                'category': category,
                'aliases': [a.strip(':') for a in aliases] if aliases else []
            }
            all_emojis_list.append(emoji_obj)

            if category not in categorized_emojis: 
                categorized_emojis[category] = []
            categorized_emojis[category].append(emoji_obj)
            category_names_set.add(category)

    # Add recently used emojis as a category
    recent_emoji_chars = load_recent_emojis()
    if recent_emoji_chars:
        recent_emojis = []
        emoji_lookup = {e['char']: e for e in all_emojis_list}
        
        for char in recent_emoji_chars:
            if char in emoji_lookup:
                recent_emojis.append(emoji_lookup[char])
        
        if recent_emojis:
            categorized_emojis["Recently Used"] = recent_emojis
            category_names_set.add("Recently Used")

    sorted_category_names = sorted(
        list(category_names_set),
        key=lambda c: (PREFERRED_CATEGORY_ORDER.index(c) if c in PREFERRED_CATEGORY_ORDER else float('inf'), c)
    )

    for cat_emojis in categorized_emojis.values():
        cat_emojis.sort(key=lambda e: e.get('sort_order', e['name']))  # Use name if sort_order absent
    all_emojis_list.sort(key=lambda e: e.get('sort_order', e['name']))

    return all_emojis_list, categorized_emojis, sorted_category_names


# Lazy loading - will be initialized on first use
_EMOJI_DATA_CACHE = None

def get_emoji_data() -> Tuple[List[ProcessedEmoji], Dict[str, List[ProcessedEmoji]], List[str]]:
    """Get emoji data, loading it lazily on first access."""
    global _EMOJI_DATA_CACHE
    if _EMOJI_DATA_CACHE is None:
        _EMOJI_DATA_CACHE = _load_emojis()
    return _EMOJI_DATA_CACHE


# --- Textual Widgets ---

# Add the new Message class definition here
class EmojiSelected(Message):
    """Message sent when an emoji is selected from the picker."""
    def __init__(self, emoji: str, picker_id: Optional[str] = None) -> None:
        super().__init__()
        self.emoji: str = emoji
        self.picker_id: Optional[str] = picker_id # Optional: if we need to identify the source picker

class EmojiButton(Button):
    def __init__(self, emoji_data: ProcessedEmoji, **kwargs):
        super().__init__(label=emoji_data['char'], **kwargs)
        self.emoji_data = emoji_data
        self.tooltip = emoji_data['name']


class EmojiGrid(VerticalScroll):
    COLUMN_COUNT = 12  # More columns for better use of space
    MAX_DISPLAY = 180  # Limit initial display for performance

    def __init__(self, emojis: List[ProcessedEmoji], **kwargs):
        super().__init__(**kwargs)
        self.emojis = emojis  # Original list of emojis for this grid (used if no specific list passed to populate_grid)

    def on_mount(self) -> None:
        # Populate with its default emojis if not immediately populated by search/category logic
        if not self.children:  # Avoid double-populating if already handled
            self.populate_grid()

    def populate_grid(self, emojis_to_display: Optional[List[ProcessedEmoji]] = None) -> None:
        for child in self.query("Horizontal, EmojiButton, Static.no_emojis_message"):
            child.remove()

        current_emojis = emojis_to_display if emojis_to_display is not None else self.emojis
        
        # Limit the number of emojis displayed for performance
        if len(current_emojis) > self.MAX_DISPLAY:
            current_emojis = current_emojis[:self.MAX_DISPLAY]

        rows_to_mount = []
        row_buttons = []
        
        for i, emoji_data in enumerate(current_emojis):
            if i % self.COLUMN_COUNT == 0 and row_buttons:
                # Create and populate the row container
                row_container = Horizontal(classes="emoji_row")
                rows_to_mount.append((row_container, row_buttons))
                row_buttons = []
            
            button = EmojiButton(emoji_data, classes="emoji_button")
            row_buttons.append(button)
        
        # Handle the last row if it has buttons
        if row_buttons:
            row_container = Horizontal(classes="emoji_row")
            rows_to_mount.append((row_container, row_buttons))
        
        # Now mount all rows with their buttons
        for row_container, buttons in rows_to_mount:
            self.mount(row_container)
            row_container.mount(*buttons)

        if not current_emojis:
            self.mount(Static("No emojis found.", classes="no_emojis_message"))
        else:
            first_button_instance = self.query(EmojiButton).first()
            if first_button_instance:
                try:
                    if self.app.is_mounted(first_button_instance):  # Ensure widget is active
                        first_button_instance.focus()
                except Exception:
                    pass  # Ignore focus errors, e.g. if screen not fully ready or widget not focusable


class EmojiPickerScreen(ModalScreen[str]):
    BINDINGS = [
        Binding("escape", "dismiss_picker", "Close Picker"),
        Binding("1", "select_recent_1", "Recent Emoji 1", show=False),
        Binding("2", "select_recent_2", "Recent Emoji 2", show=False),
        Binding("3", "select_recent_3", "Recent Emoji 3", show=False),
        Binding("4", "select_recent_4", "Recent Emoji 4", show=False),
        Binding("5", "select_recent_5", "Recent Emoji 5", show=False),
        Binding("6", "select_recent_6", "Recent Emoji 6", show=False),
        Binding("7", "select_recent_7", "Recent Emoji 7", show=False),
        Binding("8", "select_recent_8", "Recent Emoji 8", show=False),
        Binding("9", "select_recent_9", "Recent Emoji 9", show=False),
        Binding("ctrl+left", "prev_category", "Previous Category", show=False),
        Binding("ctrl+right", "next_category", "Next Category", show=False),
    ]
    CSS = """
    EmojiPickerScreen { align: center middle; }
    #dialog { 
        width: 80%; 
        max-width: 120; 
        height: 80%; 
        max-height: 40;
        border: thick $primary; 
        background: $surface; 
        padding: 1;
    }
    #search-input { 
        width: 100%; 
        margin-bottom: 1; 
        border: tall $primary-background;
    }
    #search-input:focus {
        border: tall $primary;
    }
    TabbedContent#emoji-tabs { 
        height: 1fr; 
        border: none;
    }
    TabPane { 
        padding: 0 1; 
        height: 100%; 
    }
    EmojiGrid { 
        width: 100%; 
        height: 100%; 
        padding: 0;
    }
    .emoji_row { 
        width: 100%; 
        height: auto; 
        align: left top; 
        margin: 0;
    }
    EmojiButton.emoji_button { 
        width: 4; 
        height: 3; 
        border: none; 
        background: transparent; 
        color: $text; 
        padding: 0;
        text-align: center;
        content-align: center middle;
    }
    EmojiButton.emoji_button:hover { 
        background: $primary-background; 
    }
    EmojiButton.emoji_button:focus { 
        background: $primary-background-lighten-1;
    }
    .no_emojis_message { 
        width: 100%; 
        content-align: center middle; 
        padding: 2; 
        color: $text-muted; 
        text-style: italic;
    }
    #footer { 
        height: auto; 
        width: 100%; 
        dock: bottom; 
        padding-top: 1; 
        align: right middle; 
    }
    .dialog-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    """

    # Removed: search_results: reactive[List[ProcessedEmoji] | None] = reactive(None)

    def __init__(self, name: str | None = None, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(name, id, classes)
        # Load emoji data lazily
        all_emojis, categorized_emojis, category_names = get_emoji_data()
        self._all_emojis: List[ProcessedEmoji] = all_emojis
        self._categorized_emojis: Dict[str, List[ProcessedEmoji]] = categorized_emojis
        self._category_names: List[str] = category_names
        self._search_timer: Optional[Timer] = None
        self._pending_search: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("🎨 Emoji Picker", classes="dialog-title")
            yield Input(placeholder="Search emojis (e.g., smile, cat, :thumbsup:)", id="search-input")

            # Check if we have meaningful categories to create tabs
            if self._category_names and len(self._category_names) > 1:
                with TabbedContent(id="emoji-tabs"):  # ID for TabbedContent
                    for category_name in self._category_names:
                        emojis_in_category = self._categorized_emojis.get(category_name, [])
                        pane_id = f"tab-{category_name.lower().replace(' ', '_').replace('&', 'and')}"
                        grid_id = f"grid-{category_name.lower().replace(' ', '_').replace('&', 'and')}"
                        with TabPane(category_name.replace("_", " ").title(), id=pane_id):
                            yield EmojiGrid(emojis_in_category, id=grid_id)
            else:  # Fallback: no categories or only "All Emojis"
                yield EmojiGrid(self._all_emojis, id="grid-all_emojis")  # ID for the single grid

            # This grid is for search results, initially empty and hidden
            yield EmojiGrid([], id="search-results-grid")

            with Horizontal(id="footer"):
                yield Button("Cancel", variant="error", id="cancel-button")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()
        self.query_one("#search-results-grid", EmojiGrid).display = False  # Ensure it starts hidden

    def _filter_emojis(self, query: str) -> List[ProcessedEmoji]:
        if not query: return []
        query = query.lower()
        results: List[ProcessedEmoji] = []
        for emoji_data in self._all_emojis:
            if (query in emoji_data['name'].lower() or
                    any(query in alias.lower() for alias in emoji_data['aliases']) or
                    (len(query) == 1 and query == emoji_data['char'])):
                results.append(emoji_data)
        return results

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes with debouncing."""
        self._pending_search = event.value.strip()
        
        # Cancel existing timer if any
        if self._search_timer:
            self._search_timer.stop()
        
        # Set up new timer for debouncing (300ms delay)
        self._search_timer = self.set_timer(0.3, self._perform_search)
    
    def _perform_search(self) -> None:
        """Actually perform the search after debounce delay."""
        if self._pending_search is None:
            return
            
        query = self._pending_search
        self._pending_search = None

        search_grid = self.query_one("#search-results-grid", EmojiGrid)

        tab_content: Optional[TabbedContent] = None
        try:
            tab_content = self.query_one("#emoji-tabs", TabbedContent)
        except QueryError:
            pass  # It's okay if tab_content doesn't exist

        main_grid_no_tabs: Optional[EmojiGrid] = None
        if not tab_content:  # If no tabs, there should be a main grid
            try:
                main_grid_no_tabs = self.query_one("#grid-all_emojis", EmojiGrid)
            except QueryError:
                pass  # This would be an unexpected state

        if query:
            filtered_emojis = self._filter_emojis(query)

            search_grid.display = True
            if tab_content:
                tab_content.display = False
            elif main_grid_no_tabs:
                main_grid_no_tabs.display = False

            search_grid.populate_grid(filtered_emojis)  # Pass the direct list
            # Focus is handled by populate_grid if items are found
        else:  # No query, restore tab/main view
            search_grid.display = False  # Hide search results

            if tab_content:
                tab_content.display = True
                active_pane_id = tab_content.active
                if active_pane_id:
                    try:
                        # active_pane_id is like "tab-smileys_&_emotion"
                        active_pane = tab_content.query_one(f"#{active_pane_id}", TabPane)
                        grid_in_tab = active_pane.query_one(EmojiGrid)
                        # Re-populate or ensure it's visible and attempt to focus
                        # grid_in_tab.populate_grid() # Could re-populate if needed
                        first_button_in_tab = grid_in_tab.query(EmojiButton).first()
                        if first_button_in_tab:
                            first_button_in_tab.focus()
                        else:
                            active_pane.focus()  # Focus pane if no buttons
                    except QueryError:
                        tab_content.focus()  # Fallback to tab content
            elif main_grid_no_tabs:
                main_grid_no_tabs.display = True
                # main_grid_no_tabs.populate_grid() # Could re-populate if needed
                try:
                    first_button_main = main_grid_no_tabs.query(EmojiButton).first()
                    if first_button_main:
                        first_button_main.focus()
                    else:
                        self.query_one("#search-input").focus()  # Fallback to search
                except QueryError:
                    self.query_one("#search-input").focus()  # Fallback to search
            else:  # Should not be reached, fallback to search input
                self.query_one("#search-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if isinstance(event.button, EmojiButton):
            emoji_char = event.button.emoji_data['char']
            save_recent_emoji(emoji_char)  # Save to recently used
            self.dismiss(emoji_char)
        elif event.button.id == "cancel-button":
            self.action_dismiss_picker()  # Corrected: call the action method

    def action_dismiss_picker(self) -> None:  # This is the action method bound to "escape"
        self.dismiss("")  # Dismiss with empty string for cancellation
    
    def _select_recent_emoji(self, index: int) -> None:
        """Select a recent emoji by index (0-based)."""
        if "Recently Used" in self._categorized_emojis:
            recent_emojis = self._categorized_emojis["Recently Used"]
            if index < len(recent_emojis):
                emoji_char = recent_emojis[index]['char']
                save_recent_emoji(emoji_char)
                self.dismiss(emoji_char)
    
    def action_select_recent_1(self) -> None:
        self._select_recent_emoji(0)
    
    def action_select_recent_2(self) -> None:
        self._select_recent_emoji(1)
    
    def action_select_recent_3(self) -> None:
        self._select_recent_emoji(2)
    
    def action_select_recent_4(self) -> None:
        self._select_recent_emoji(3)
    
    def action_select_recent_5(self) -> None:
        self._select_recent_emoji(4)
    
    def action_select_recent_6(self) -> None:
        self._select_recent_emoji(5)
    
    def action_select_recent_7(self) -> None:
        self._select_recent_emoji(6)
    
    def action_select_recent_8(self) -> None:
        self._select_recent_emoji(7)
    
    def action_select_recent_9(self) -> None:
        self._select_recent_emoji(8)
    
    def action_prev_category(self) -> None:
        """Navigate to previous category tab."""
        try:
            tab_content = self.query_one("#emoji-tabs", TabbedContent)
            # Get current tab index
            current = tab_content.active
            tabs = list(tab_content.children)
            if tabs and current:
                for i, tab in enumerate(tabs):
                    if tab.id == current:
                        prev_index = (i - 1) % len(tabs)
                        tab_content.active = tabs[prev_index].id
                        break
        except QueryError:
            pass
    
    def action_next_category(self) -> None:
        """Navigate to next category tab."""
        try:
            tab_content = self.query_one("#emoji-tabs", TabbedContent)
            # Get current tab index
            current = tab_content.active
            tabs = list(tab_content.children)
            if tabs and current:
                for i, tab in enumerate(tabs):
                    if tab.id == current:
                        next_index = (i + 1) % len(tabs)
                        tab_content.active = tabs[next_index].id
                        break
        except QueryError:
            pass

#
# End of emoji_picker.py
########################################################################################################################
