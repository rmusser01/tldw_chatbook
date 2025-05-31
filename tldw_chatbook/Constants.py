# Constants.py
# Description: Constants for the application
#
# Imports
#
# 3rd-Party Imports
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Constants ---
TAB_CHAT = "chat"
TAB_CCP = "conversations_characters_prompts"
TAB_NOTES = "notes"
TAB_MEDIA = "media"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_TOOLS_SETTINGS = "tools_settings"
TAB_LLM = "llm_management"
TAB_STATS = "stats"
TAB_LOGS = "logs"
TAB_EVALS = "evals"
ALL_TABS = [TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH, TAB_INGEST,
            TAB_TOOLS_SETTINGS, TAB_LLM, TAB_LOGS, TAB_STATS, TAB_EVALS]


# --- CSS definition ---
# (Keep your CSS content here, make sure IDs match widgets)
css_content = """
Screen { layout: vertical; }
Header { dock: top; height: 1; background: $accent-darken-1; }
Footer { dock: bottom; height: 1; background: $accent-darken-1; }
#tabs { dock: top; height: 3; background: $background; padding: 0 1; }
#tabs Button { width: 1fr; height: 100%; border: none; background: $panel; color: $text-muted; }
#tabs Button:hover { background: $panel-lighten-1; color: $text; }
#tabs Button.-active { background: $accent; color: $text; text-style: bold; border: none; }
#content { height: 1fr; width: 100%; }

/* Base style for ALL windows. The watcher will set display: True/False */
.window {
    height: 100%;
    width: 100%;
    layout: horizontal; /* Or vertical if needed by default */
    overflow: hidden;
}

.placeholder-window { align: center middle; background: $panel; }

/* Sidebar Styling */
/* Generic .sidebar (used by #chat-left-sidebar and potentially others) */
.sidebar {
    dock: left;
    width: 25%; /* <-- CHANGE to percentage (adjust 20% to 35% as needed) */
    min-width: 20; /* <-- ADD a minimum width to prevent it becoming unusable */
    max-width: 80; /* <-- ADD a maximum width (optional) */
    background: $boost;
    padding: 1 2;
    border-right: thick $background-darken-1;
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
}
/* Collapsed state for the existing left sidebar */
.sidebar.collapsed {
    width: 0 !important;
    min-width: 0 !important; /* Ensure min-width is also 0 */
    border-right: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none; /* ensures it doesn’t grab focus */
}

/* Right sidebar (chat-right-sidebar) */
#chat-right-sidebar {
    dock: right;
    /* width: 70;   <-- REMOVE fixed width */
    width: 25%;  /* <-- CHANGE to percentage (match .sidebar or use a different one) */
    min-width: 20; /* <-- ADD a minimum width */
    max-width: 80; /* <-- ADD a maximum width (optional) */
    background: $boost;
    padding: 1 2;
    border-left: thick $background-darken-1; /* Border on the left */
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Collapsed state for the new right sidebar */
#chat-right-sidebar.collapsed {
    width: 0 !important;
    min-width: 0 !important; /* Ensure min-width is also 0 */
    border-left: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none; /* Ensures it doesn't take space or grab focus */
}

/* Common sidebar elements */
.sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
.sidebar-label { margin-top: 1; text-style: bold; }
.sidebar-input { width: 100%; margin-bottom: 1; }
.sidebar-textarea { width: 100%; border: round $surface; margin-bottom: 1; }
.sidebar Select { width: 100%; margin-bottom: 1; }

.prompt-display-textarea {
    height: 7; /* Example height */
    border: round $primary-lighten-2;
    background: $primary-background;
}

.sidebar-listview {
    height: 10; /* Example height for listviews in sidebars */
    border: round $primary-lighten-2;
    background: $primary-background;
}

/* --- Chat Window specific layouts --- */
#chat-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr; /* This is KEY - it takes up the remaining horizontal space */
}
/* VerticalScroll for chat messages */
#chat-log {
    height: 1fr; /* Takes remaining space */
    width: 100%;
    /* border: round $surface; Optional: Add border to scroll area */
    padding: 0 1; /* Padding around messages */
}

/* Input area styling (shared by chat and character) */
#chat-input-area, #conv-char-input-area { /* Updated from #character-input-area */
    height: auto;    /* Allow height to adjust */
    max-height: 12;  /* Limit growth */
    width: 100%;
    align: left top; /* Align children to top-left */
    padding: 1; /* Consistent padding */
    border-top: round $surface;
}
/* Input widget styling (shared) */
.chat-input { /* Targets TextArea */
    width: 1fr;
    height: auto;      /* Allow height to adjust */
    max-height: 100%;  /* Don't overflow parent */
    margin-right: 1; /* Space before button */
    border: round $surface;
}
/* Send button styling (shared) */
.send-button { /* Targets Button */
    width: 5;
    height: 3; /* Fixed height for consistency */
    /* align-self: stretch; REMOVED */
    margin-top: 0;
}

/* --- Conversations, Characters & Prompts Window specific layouts (previously Character Chat) --- */
/* Main container for the three-pane layout */
#conversations_characters_prompts-window {
    layout: horizontal; /* Crucial for side-by-side panes */
    /* Ensure it takes full height if not already by .window */
    height: 100%;
}

/* Left Pane Styling */
.cc-left-pane {
    width: 25%; /* Keep 25% or 30% - adjust as needed */
    min-width: 20; /* ADD a minimum width */
    height: 100%;
    background: $boost;
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Center Pane Styling */
.cc-center-pane {
    width: 1fr; /* Takes remaining space */
    height: 100%;
    padding: 1;
    overflow-y: auto; /* For conversation history */
}

/* Right Pane Styling */
.cc-right-pane {
    width: 25%; /* Keep 25% or 30% - adjust as needed */
    min-width: 20; /* ADD a minimum width */
    height: 100%;
    background: $boost;
    padding: 1;
    border-left: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

/* General styles for elements within these panes (can reuse/adapt from .sidebar styles) */
.cc-left-pane Input, .cc-right-pane Input {
    width: 100%; margin-bottom: 1;
}
.cc-left-pane ListView {
    height: 1fr; /* Make ListView take available space */
    margin-bottom: 1;
    border: round $surface;
}
.cc-left-pane Button, .cc-right_pane Button { /* Typo Fixed */
    width: 100%;
    margin-bottom: 1;
}

/* Specific title style for panes */
.pane-title {
    text-style: bold;
    margin-bottom: 1;
    text-align: center;
    width: 100%; /* Ensure it spans width for centering */
}

/* Specific style for keywords TextArea in the right pane */
.conv-char-keywords-textarea {
    height: 5; /* Example height */
    width: 100%;
    margin-bottom: 1;
    border: round $surface; /* Re-apply border if not inherited */
}

/* Specific style for the "Export Options" label */
.export-label {
    margin-top: 2; /* Add some space above export options */
}


/* Old styles for #conv-char-main-content, #conv-char-top-area etc. are removed */
/* as the structure within #conversations_characters_prompts-window is now different. */
/* Portrait styling - if still needed, would be part of a specific pane's content now */
/* #conv-char-portrait {
    width: 25;
    height: 100%;
    border: round $surface;
    padding: 1;
    margin: 0;
    overflow: hidden;
    align: center top;
}

/* Logs Window adjustments */
#logs-window {
    layout: vertical; /* Override .window's default horizontal layout for this specific window */
    /* The rest of your #logs-window styles (padding, border, height, width) are fine */
    /* E.g., if you had: padding: 0; border: none; height: 100%; width: 100%; those are okay. */
}
#app-log-display {
    border: none;
    height: 1fr;    /* RichLog takes most of the vertical space */
    width: 100%;    /* RichLog takes full width in the vertical layout */
    margin: 0;
    padding: 1;     /* Your existing padding is good */
}

/* Style for the new "Copy All Logs" button */
.logs-action-button {
    width: 100%;     /* Button takes full width */
    height: 3;       /* A standard button height */
    margin-top: 1;   /* Add some space between RichLog and the button */
    /* dock: bottom; /* Optional: If you want it always pinned to the very bottom.
                       If omitted, it will just flow after the RichLog in the vertical layout.
                       For simplicity, let's omit it for now. */
}
/* old #logs-window { padding: 0; border: none; height: 100%; width: 100%; }
#app-log-display { border: none; height: 1fr; width: 1fr; margin: 0; padding: 1; }
*/

/* --- ChatMessage Styling --- */
ChatMessage {
    width: 100%;
    height: auto;
    margin-bottom: 1;
}
ChatMessage > Vertical {
    border: round $surface;
    background: $panel;
    padding: 0 1;
    width: 100%;
    height: auto;
}
ChatMessage.-user > Vertical {
    background: $boost; /* Different background for user */
    border: round $accent;
}
.message-header {
    width: 100%;
    padding: 0 1;
    background: $surface-darken-1;
    text-style: bold;
    height: 1; /* Ensure header is minimal height */
}
.message-text {
    padding: 1; /* Padding around the text itself */
    width: 100%;
    height: auto;
}
.message-actions {
    height: auto;
    width: 100%;
    padding: 1; /* Add padding around buttons */
    /* Use a VALID border type */
    border-top: solid $surface-lighten-1; /* CHANGED thin to solid */
    align: right middle; /* Align buttons to the right */
    display: block; /* Default display state */
}
.message-actions Button {
    min-width: 8;
    height: 1;
    margin: 0 0 0 1; /* Space between buttons */
    border: none;
    background: $surface-lighten-2;
    color: $text-muted;
}
.message-actions Button:hover {
    background: $surface;
    color: $text;
}
/* Initially hide AI actions until generation is complete */
ChatMessage.-ai .message-actions.-generating {
    display: none;
}
/* microphone button – same box as Send but subdued colour */
.mic-button {
    width: 1;
    height: 3;
    margin-right: 1;           /* gap before Send */
    border: none;
    background: $surface-darken-1;
    color: $text-muted;
}
.mic-button:hover {
    background: $surface;
    color: $text;
}
.sidebar-toggle {
    width: 2;                /* tiny square */
    height: 3;
    /* margin-right: 1; Removed default margin, apply specific below */
    border: none;
    background: $surface-darken-1;
    color: $text;
}
.sidebar-toggle:hover { background: $surface; }

/* Specific margins for sidebar toggles based on position */
#toggle-chat-left-sidebar {
    margin-right: 1; /* Original toggle on the left of input area */
}

#toggle-chat-right-sidebar {
    margin-left: 1; /* New toggle on the right of input area */
}

#app-titlebar {
    dock: top;
    height: 1;                 /* single line */
    background: $accent;       /* or any colour */
    color: $text;
    text-align: center;
    text-style: bold;
    padding: 0 1;
}

/* Reduce height of Collapsible headers */
Collapsible > .collapsible--header {
    height: 2;
}

.chat-system-prompt-styling {
    width: 100%;
    height: auto;
    min-height: 3;
    max-height: 10; /* Limit height */
    border: round $surface;
    margin-bottom: 1;
}

/* --- Notes Tab Window --- */
/* (Assuming #notes-window has layout: horizontal; by default from .window or is set in Python) */

#notes-main-content { /* Parent of the editor and controls */
    layout: vertical; /* This is what I inferred based on your Python structure */
    width: 1fr;       /* Takes space between sidebars */
    height: 100%;
}

.notes-editor { /* Targets your #notes-editor-area by class */
    width: 100%;
    height: 1fr; /* This makes it take available vertical space */
}

#notes-controls-area { /* The container for buttons below the editor */
    height: auto;
    width: 100%;
    padding: 1;
    border-top: round $surface;
    align: center middle; /* Aligns buttons horizontally if Horizontal container */
                           /* If this itself is a Vertical container, this might not do much */
}

/* --- Metrics Screen Styling --- */
MetricsScreen {
    padding: 1 2; /* Add some padding around the screen content */
    /* layout: vertical; /* MetricsScreen is a Static, VerticalScroll handles layout */
    /* align: center top; /* If needed, but VerticalScroll might handle this */
}

#metrics-container {
    padding: 1;
    /* border: round $primary-lighten-2; /* Optional: a subtle border */
    /* background: $surface; /* Optional: a slightly different background */
}

/* Styling for individual metric labels within MetricsScreen */
MetricsScreen Label {
    width: 100%;
    margin-bottom: 1; /* Space between metric items */
    padding: 1;       /* Padding inside each label's box */
    background: $panel-lighten-1; /* A slightly lighter background for each item */
    border: round $primary-darken-1; /* Border for each item */
    /* Textual CSS doesn't allow direct styling of parts of a Label's text (like key vs value) */
    /* The Python code uses [b] for keys, which Rich Text handles. */
}

/* Style for the title label: "Application Metrics" */
/* This targets the first Label directly inside the VerticalScroll with ID metrics-container */
#metrics-container > Label:first-of-type {
    text-style: bold underline;
    align: center middle;
    padding: 1 0 2 0; /* More padding below the title */
    background: transparent; /* No specific background for the title itself */
    border: none; /* No border for the title itself */
    margin-bottom: 2; /* Extra space after the title */
}

/* Style for error messages within MetricsScreen */
/* These require the Python code to add the respective class to the Label widget */
MetricsScreen Label.-error-message {
    color: $error; /* Text color for errors */
    background: $error 20%; /* Background for error messages, e.g., light red. USES $error WITH 20% ALPHA */
    /* border: round $error; /* Optional: border for error messages */
    text-style: bold;
}

/* Style for info messages (e.g. "file empty") within MetricsScreen */
MetricsScreen Label.-info-message {
    color: $text-muted; /* Or another color that indicates information */
    background: $panel; /* A more subdued background, or $transparent */
    /* border: round $primary-lighten-1; /* Optional: border for info messages */
    text-style: italic;
}


/* Collapsible Sidebar Toggle Button For Character/Conversation Editing Page */
.cc-sidebar-toggle-button { /* Applied to the "☰" button */
    width: 5; /* Adjust width as needed */
    height: 100%; /* Match parent Horizontal height, or set fixed e.g., 1 or 3 */
    min-width: 0; /* Override other button styles if necessary */
    border: none; /* Style as you like, e.g., remove border */
    background: $surface-darken-1; /* Example background */
    color: $text;
}
.cc-sidebar-toggle-button:hover {
    background: $surface;
}
/* End of Collapsible Sidebar Toggle Button for character/conversation editing */


/* Save Chat Button in Character Sidebar in Chat Tab */
.save-chat-button { /* Class used in character_sidebar.py */
    margin-top: 2;   /* Add 1 cell/unit of space above the button */
    /*width: 100%;      Optional: make it full width like other sidebar buttons */
}



/* Character Sidebar Specific Styles */
#chat-right-sidebar #chat-conversation-title-input { /* Title input */
    /* width: 100%; (from .sidebar-input) */
    /* margin-bottom: 1; (from .sidebar-input) */
}

#chat-right-sidebar .chat-keywords-textarea { /* Keywords TextArea specific class */
    height: 4;  /* Or 3 to 5, adjust as preferred */
    /* width: 100%; (from .sidebar-textarea) */
    /* border: round $surface; (from .sidebar-textarea) */
    /* margin-bottom: 1; (from .sidebar-textarea) */
}

/* Styling for the new "Save Details" button */
#chat-right-sidebar .save-details-button {
    margin-top: 1; /* Space above this button */
    /* width: 100%;    Make it full width */
}

/* Ensure the Save Current Chat button also has clear styling if needed */
#chat-right-sidebar .save-chat-button {
    margin-top: 1; /* Ensure it has some space if it's after keywords */
    /* width: 100%; */
}
/* End of Character Sidebar Specific Styles */


/* --- Prompts Sidebar Vertical --- */
.ccp-prompt-textarea { /* Specific class for prompt textareas if needed */
    height: 5; /* Example height */
    /* width: 100%; (from .sidebar-textarea) */
    /* margin-bottom: 1; (from .sidebar-textarea) */
}

#ccp-prompts-listview { /* ID for the prompt list */
    height: 10; /* Or 1fr if it's the main element in its collapsible */
    border: round $surface;
    margin-bottom: 1;
}

.ccp-prompt-action-buttons Button {
    width: 1fr; /* Make buttons share space */
    margin: 0 1 0 0; /* Small right margin, no top/bottom if already in Horizontal */
}
.ccp-prompt-action-buttons Button:last-of-type { /* Corrected pseudo-class */
    margin-right: 0;
}

/* Ensure Collapsible titles are clear */
#conv-char-right-pane Collapsible > .collapsible--header {
    background: $primary-background-darken-1; /* Example to differentiate */
    color: $text;
}

#conv-char-right-pane Collapsible.-active > .collapsible--header { /* Optional: when expanded */
    background: $primary-background;
}
/* --- End of Prompts Sidebar Vertical --- */

/* Right Pane Styling */
.cc-right-pane {
    width: 25%; /* Keep 25% or 30% - adjust as needed */
    min-width: 20; /* ADD a minimum width */
    height: 100%;
    background: $boost;
    padding: 1;
    border-left: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

/* ADD THIS: Collapsed state for the CCP tab's right pane */
.cc-right-pane.collapsed {
    width: 0 !important;
    min-width: 0 !important;
    border-left: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none !important; /* Ensures it doesn't take space or grab focus */
}

/* Styles for the dynamic view areas within the CCP center pane */
.ccp-view-area {
    width: 100%;
    height: 100%;
    /* overflow: auto; /* If content within might overflow */
}

/* Add this class to hide elements */
.ccp-view-area.hidden,
.ccp-right-pane-section.hidden { /* For sections in the right pane */
    display: none !important;
}

/* By default, let conversation messages be visible, and editor hidden */
#ccp-conversation-messages-view {
    /* display: block; /* or whatever its natural display is, usually block for Container */
}
#ccp-prompt-editor-view {
    display: none; /* Initially hidden by CSS */
}

/* Ensure the right pane sections also respect hidden class */
#ccp-right-pane-llm-settings-container {
    /* display: block; default */
}
#ccp-right-pane-llm-settings-container.hidden {
    display: none !important;
}

/* --- Tools & Settings Tab --- */
#tools_settings-window { /* Matches TAB_TOOLS_SETTINGS */
    layout: horizontal; /* Main layout for this tab */
}

.tools-nav-pane {
    dock: left;
    width: 25%; /* Adjust as needed */
    min-width: 25; /* Example min-width */
    max-width: 60; /* Example max-width */
    height: 100%;
    background: $boost; /* Or $surface-lighten-1 */
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

.tools-nav-pane .ts-nav-button { /* Style for navigation buttons */
    width: 100%;
    margin-bottom: 1;
    border: none; /* Cleaner look for nav buttons */
    height: 3;
}
.tools-nav-pane .ts-nav-button:hover {
    background: $accent 50%;
}
/* Consider an active state style for the selected nav button */
/* .tools-nav-pane .ts-nav-button.-active-view {
    background: $accent;
    color: $text;
} */

.tools-content-pane {
    width: 1fr; /* Takes remaining horizontal space */
    height: 100%;
    padding: 1 2; /* Padding for the content area */
    overflow-y: auto; /* If content within sub-views might scroll */
}

.ts-view-area { /* Class for individual content areas */
    width: 100%;
    height: 100%; /* Or auto if content dictates height */
    /* border: round $surface; /* Optional: border around content views */
    /* padding: 1; /* Optional: padding within content views */
}

/* Container for the HorizontalScroll, this takes the original #tabs styling for docking */
#tabs-outer-container {
    dock: top;
    height: 3; /* Or your desired tab bar height */
    background: $background; /* Or your tab bar background */
    padding: 0 1; /* Padding for the overall bar */
    width: 100%;
}

/* The HorizontalScroll itself, which will contain the buttons */
#tabs {
    width: 100%;
    height: 100%; /* Fill the outer container's height */
    overflow-x: auto !important; /* Ensure horizontal scrolling is enabled */
    /* overflow-y: hidden; /* Usually not needed for a single row of tabs */
}

#tabs Button {
    width: auto;         /* Let button width be determined by content + padding */
    min-width: 10;       /* Minimum width to prevent squishing too much */
    height: 100%;        /* Fill the height of the scrollable area */
    border: none; /* Your existing style */
    background: $panel;  /* Your existing style */
    color: $text-muted;  /* Your existing style */
    padding: 0 2;        /* Add horizontal padding to buttons */
    margin: 0 1 0 0;     /* Small right margin between buttons */
}

#tabs Button:last-of-type { /* No margin for the last button */
    margin-right: 0;
}

#tabs Button:hover {
    background: $panel-lighten-1; /* Your existing style */
    color: $text;                 /* Your existing style */
}

#tabs Button.-active {
    background: $accent;          /* Your existing style */
    color: $text;                 /* Your existing style */
    text-style: bold;             /* Your existing style */
    /* border: none; /* Already set */
}

/* --- Ingest Content Tab --- */
#ingest-window { /* Matches TAB_INGEST */
    layout: horizontal;
}

.ingest-nav-pane { /* Style for the left navigation pane */
    dock: left;
    width: 25%;
    min-width: 25;
    max-width: 60;
    height: 100%;
    background: $boost; /* Or a slightly different shade */
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

.ingest-nav-pane .ingest-nav-button { /* Style for navigation buttons */
    width: 100%;
    margin-bottom: 1;
    border: none;
    height: 3;
}
.ingest-nav-pane .ingest-nav-button:hover {
    background: $accent 60%; /* Slightly different hover potentially */
}
/* Active state for selected ingest nav button (optional) */
/* .ingest-nav-pane .ingest-nav-button.-active-view {
    background: $accent-darken-1;
    color: $text;
} */

.ingest-content-pane { /* Style for the right content display area */
    width: 1fr;
    height: 100%;
    padding: 1 2;
    overflow-y: auto;
}

.ingest-view-area { /* Common class for individual content areas */
    width: 100%;
    height: 100%; /* Or auto */
    /* Example content styling */
    /* align: center middle; */
    /* border: round $primary; */
    /* padding: 2; */
}

/* --- LLM Management Tab --- */
#llm_management-window { /* Matches TAB_LLM ("llm_management") */
    layout: horizontal;
}

.llm-nav-pane { /* Style for the left navigation pane */
    dock: left;
    width: 25%; /* Or your preferred width */
    min-width: 25;
    max-width: 60; /* Example */
    height: 100%;
    background: $boost; /* Or $surface-darken-1 or $surface-lighten-1 */
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

.llm-nav-pane .llm-nav-button { /* Style for navigation buttons */
    width: 100%;
    margin-bottom: 1;
    border: none;
    height: 3;
}
.llm-nav-pane .llm-nav-button:hover {
    background: $accent 70%; /* Example hover */
}
/* Active state for selected llm nav button (optional) */
/* .llm-nav-pane .llm-nav-button.-active-view {
    background: $accent;
    color: $text;
} */

.llm-content-pane { /* Style for the right content display area */
    width: 1fr;
    height: 100%;
    padding: 1 2;
    overflow-y: auto;
}

.llm-view-area { /* Common class for individual content areas */
    width: 100%;
    height: 100%; /* Or auto */
}

/* --- Media Tab --- */
#media-window { /* Matches TAB_MEDIA */
    layout: horizontal; /* Main layout for this tab */
}

.media-nav-pane {
    dock: left;
    width: 25%; /* Adjust as needed */
    min-width: 20; /* Example min-width */
    max-width: 50; /* Example max-width */
    height: 100%;
    background: $boost; /* Or a different background */
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

.media-nav-pane .media-nav-button { /* Style for navigation buttons */
    width: 100%;
    margin-bottom: 1;
    border: none;
    height: 3;
}
.media-nav-pane .media-nav-button:hover {
    background: $accent 75%; /* Example hover, distinct from other navs */
}

.media-content-pane {
    width: 1fr; /* Takes remaining horizontal space */
    height: 100%;
    padding: 1 2; /* Padding for the content area */
    overflow-y: auto; /* If content within sub-views might scroll */
}

.media-view-area { /* Class for individual content areas in Media tab */
    width: 100%;
    height: 100%; /* Or auto if content dictates height */
}
    """


#
# End of Constants.py
########################################################################################################################
