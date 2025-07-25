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
from tldw_chatbook.Third_Party.textual_fspicker import Filters

# --- Constants ---
TAB_CHAT = "chat"
TAB_CCP = "conversations_characters_prompts"
TAB_NOTES = "notes"
TAB_MEDIA = "media"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_EMBEDDINGS = "embeddings"
TAB_EVALS = "evals"
TAB_LLM = "llm_management"
TAB_TOOLS_SETTINGS = "tools_settings"
TAB_STATS = "stats"
TAB_LOGS = "logs"
TAB_CODING = "coding"
TAB_STTS = "stts"
TAB_SUBSCRIPTIONS = "subscriptions"
ALL_TABS = [TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH, TAB_INGEST,
            TAB_EMBEDDINGS, TAB_EVALS, TAB_LLM, TAB_TOOLS_SETTINGS, TAB_LOGS, TAB_STATS, TAB_STTS, TAB_SUBSCRIPTIONS]

# --- TLDW API Form Specific Option Containers (IDs) ---
TLDW_API_VIDEO_OPTIONS_ID = "tldw-api-video-options"
TLDW_API_AUDIO_OPTIONS_ID = "tldw-api-audio-options"
TLDW_API_PDF_OPTIONS_ID = "tldw-api-pdf-options"
TLDW_API_EBOOK_OPTIONS_ID = "tldw-api-ebook-options"
TLDW_API_DOCUMENT_OPTIONS_ID = "tldw-api-document-options"
TLDW_API_XML_OPTIONS_ID = "tldw-api-xml-options"
TLDW_API_MEDIAWIKI_OPTIONS_ID = "tldw-api-mediawiki-options"
TLDW_API_PLAINTEXT_OPTIONS_ID = "tldw-api-plaintext-options"

ALL_TLDW_API_OPTION_CONTAINERS = [
    TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID,
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID,
    TLDW_API_MEDIAWIKI_OPTIONS_ID, TLDW_API_PLAINTEXT_OPTIONS_ID
]


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

/* Sidebar resize buttons */
.sidebar-resize-button {
    min-width: 8;  /* Increased minimum width to 8 cells */
    width: 8;      /* Fixed width for better visibility */
    height: 2;     /* Standard button height */
    margin: 0 1;   /* Small margin on sides */
    padding: 0 1;  /* Padding for text */
    border: none;
    background: $primary;
    color: white;
    text-align: center;
    text-style: bold;
}
.sidebar-resize-button:hover {
    background: $primary-lighten-1;
    color: white;
    text-style: bold;
}
.sidebar-resize-button:focus {
    background: $primary-lighten-2;
    color: white;
    text-style: bold;
}

/* Header container for sidebar with resize controls */
.sidebar-header-with-resize {
    layout: horizontal;
    height: auto;
    width: 100%;
    align: center middle;
    margin-bottom: 1;
}

/* Flex grow utility class */
.flex-grow {
    width: 1fr;  /* Takes up remaining space in horizontal layout */
}

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
/* --- End of Sidebar Styling --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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
    width: 2;
    height: 3; /* Fixed height for consistency */
    /* align-self: stretch; REMOVED */
    margin-top: 0;
    padding: 0; /* No padding for button */
}

.stop-button {
    width: 2;
    height: 3;
    margin-top: 0;
}

/* Save Chat Button in chat-right-sidebar in Chat Tab */
.save-chat-button { /* Class used in character_sidebar.py */
    margin-top: 2;   /* Add 1 cell/unit of space above the button */
    /*width: 100%;      Optional: make it full width like other sidebar buttons */
}

/* chat-right-sidebar Specific Styles */
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

/* Chat Sidebar - Prompts Section */
#chat-prompts-collapsible Input,
#chat-prompts-collapsible ListView,
#chat-prompts-collapsible Button {
    margin-bottom: 1; /* Consistent spacing */
}
#chat-prompt-list-view { /* ID for the prompt list in chat sidebar */
    height: 8; /* Adjust as needed */
    border: round $surface;
    margin-bottom: 1;
}
.loaded-prompt-area { /* Container for loaded prompt details in chat sidebar */
    margin-top: 1;
    padding: 0 1;        /* Padding for content inside */
    background: $surface; /* Slightly different background to group elements */
    /* To get rounded corners for the area itself, apply a border style */
    border: round $primary-background-lighten-1; /* Example: round border with a light color */
                                               /* Or use 'round $surface' if you want it less distinct */
                                               /* Or 'none' if you only want background and no explicit border */
}
/* End of Character Sidebar Specific Styles */

/* --- End of Chat Window specific layouts --- */
/* ----------------------------- ************************* ----------------------------- */



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
.cc-left-pane Button, .cc-right-pane Button { /* Typo in original was .cc-right_pane */
    width: 100%;
    margin-bottom: 1;
}

/* Ensure Select widgets in left and right panes also get full width */
.cc-left-pane Select, .cc-right-pane Select {
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
    height: auto; /* Allow height to be determined by content */
    /* overflow: auto; /* If content within might overflow */
    overflow: auto;
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

#ccp-character-card-view {
    display: none; /* Initially hidden, to be shown by Python logic */
}

#ccp-character-editor-view {
    display: none; /* Initially hidden */
    layout: vertical; /* Important for stacking the scroller and button bar */
    width: 100%;
    height: 100%; /* Fill the .cc-center-pane */
}

/* Ensure the right pane sections also respect hidden class */
#ccp-right-pane-llm-settings-container {
    /* display: block; default */
}
#ccp-right-pane-llm-settings-container.hidden {
    display: none !important;
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

/* --- Prompts Sidebar Vertical --- */
.ccp-prompt-textarea { /* Specific class for prompt textareas if needed */
    height: 20; /* Example height - Increased from 10 */
    /* width: 100%; (from .sidebar-textarea) */
    /* margin-bottom: 1; (from .sidebar-textarea) */
}

#ccp-prompts-listview { /* ID for the prompt list */
    height: 10; /* Or 1fr if it's the main element in its collapsible */
    border: round $surface;
    margin-bottom: 1;
}
.ccp-card-action-buttons {
    height: auto; /* Let it size to content */
    width: 100%;
    margin-top: 1; /* Space above buttons */
    margin-bottom: 2; /* Extra space below to ensure buttons are visible */
}
.ccp-prompt-action-buttons {
    margin-top: 1; /* Add space above the button bar */
    height: auto; /* Allow container height to fit buttons */
    width: 100%; /* Full width for the button bar */
    /* padding-bottom: 1; Removed, parent #ccp-character-editor-view now handles this */
}

.ccp-prompt-action-buttons Button {
    width: 1fr; /* Make buttons share space */
    margin: 0 1 0 0; /* Small right margin for all but last */
    height: auto; /* Let button height fit its content (typically 1 line) */
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

/* TextAreas for Character Card Display */
.ccp-card-textarea {
    height: 15;
    width: 100%;
    margin-bottom: 1;
    border: round $surface; /* Ensuring consistent styling */
}

/* --- End of Prompts Sidebar Vertical --- */
/* --- End of Conversations, Characters & Prompts Window specific layouts --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
/* --- Logs Window --- */
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
/* --- End of Logs Window --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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
/* --- End of Notes Tab Window --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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
/* --- End of Metrics Screen Styling --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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
.tldw-api-media-specific-options { /* Common class for specific option blocks */
    padding: 1;
    border: round $surface;
    margin-top: 1;
    margin-bottom: 1;
}

/* Added to ensure initially hidden specific options are indeed hidden */
.tldw-api-media-specific-options.hidden {
     padding: 1;
     border: round $surface;
     margin-top: 1;
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
.ingest-label {
    margin-top: 1;
    margin-bottom: 0;
}
.ingest-selected-files-list {
    height: 5;
    border: round $primary;
    margin-bottom: 1;
    background: $surface;
}
.ingest-preview-area {
    height: 1fr;
    border: round $primary-lighten-2;
    padding: 1;
    margin-bottom: 1;
    background: $surface;
}
.ingest-preview-area > Static#ingest-prompts-preview-placeholder {
    color: $text-muted;
    width: 100%;
    text-align: center;
    padding: 2 0;
}
.ingest-status-area {
    height: 8;
    margin-top: 1;
}

.prompt-preview-item {
    border: panel $background-lighten-2;
    padding: 1;
    margin-bottom: 1;
}
.prompt-preview-item .prompt-title {
    text-style: bold;
}
.prompt-preview-item .prompt-field-label {
    text-style: italic;
    color: $text-muted;
}
.prompt-preview-item Markdown {
    background: $surface-darken-1;
    padding: 0 1;
    margin-top: 1;
    margin-bottom: 1;
    border: solid $primary-darken-1; /* Use 'solid' instead of 'narrow' */
    max-height: 10;
    overflow-y: auto;
}
.prompt-preview-item .prompt-details-text {
    max-height: 5;
    overflow-y: auto;
    background: $surface;
    padding: 0 1;
    border: dashed $primary-darken-2; /* Use 'dashed' instead of 'dotted' */
    margin-bottom: 1;
}
/* --- END OF INTEGRATED NEW CSS --- */

/* ----------------------------- ************************* ----------------------------- */
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
/* Active state for selected llm nav button */
.llm-nav-pane .llm-nav-button.-active {
    background: $accent;
    color: $text;
}

.llm-content-pane { /* Style for the right content display area */
    width: 1fr;
    height: 100%;
    padding: 1 2;
    overflow-y: auto;
}

.llm-view-area { /* Common class for individual content areas */
    width: 100%;
    height: 100%; /* Or auto */
    display: none; /* Initially hidden until activated */
}


/* Chat Sidebar Prompts Section Specific Styles */
#chat-sidebar-prompts-collapsible { /* The collapsible container itself */
    /* Add any specific styling for the collapsible if needed */
}

#chat-sidebar-prompt-search-input,
#chat-sidebar-prompt-keyword-filter-input {
    margin-bottom: 1; /* Add some space below these inputs */
}

#chat-sidebar-prompts-listview {
    min-height: 5;
    max-height: 15;
    height: auto;
    overflow-y: auto;
    border: round $surface;
    margin-bottom: 1;
}

#chat-sidebar-prompt-system-display,
#chat-sidebar-prompt-user-display {
    min-height: 5;
    max-height: 15;
    height: auto;
    width: 100%; /* Ensure they take full width */
    margin-bottom: 1;
    border: round $surface; /* Standard border like other textareas */
    /* read_only is set in Python, CSS cannot enforce it but can style */
}

#chat-sidebar-copy-system-prompt-button,
#chat-sidebar-copy-user-prompt-button {
    width: 100%; /* Make copy buttons full width */
    margin-top: 0; /* Remove top margin if directly after TextArea */
    margin-bottom: 1; /* Space after copy buttons */
}
/*

/* LLM Management Tab Specific Styles */
#llm_management-window .llm-view-area > VerticalScroll { /* Target the new VS inside each view */
    height: 100%; /* Ensure the VerticalScroll takes full height of its parent view area */
}

#llamacpp-args-help-collapsible,
#llamafile-args-help-collapsible,
#vllm-args-help-collapsible { /* Style for the collapsible itself */
    margin-top: 1;
    margin-bottom: 1;
    border: round $primary-lighten-1; /* Optional border for the collapsible */
}

#llamacpp-args-help-collapsible > .collapsible--header,
#llamafile-args-help-collapsible > .collapsible--header,
#vllm-args-help-collapsible > .collapsible--header { /* Style for the collapsible title */
    background: $surface-darken-1; /* Slightly different background for the header */
}

.help-text-display {
    height: 25; /* Or your desired fixed height for the scrollable help text */
    width: 100%;
    border: round $surface;
    padding: 1;
    margin-top: 1; /* Space between collapsible title and the content */
    background: $panel; /* Background for the help text area */
    overflow-y: scroll !important; /* Ensure vertical scrolling */
}

/* Ensure input fields for additional args are appropriately sized if they were TextAreas */
/* If you changed #llamacpp-additional-args to Input, this might not be needed for it */
#llamacpp-additional-args, /* This is now an Input, so this rule might be less relevant for it */
#llamafile-additional-args,
#vllm-additional-args {
    height: 3; /* For single-line Input or small TextArea */
    /* If it's a TextArea and you want more lines: */
    /* min-height: 3; */
    /* max-height: 7; */
    /* height: auto; */
}

/* Optional: Style for the input_container if needed */
.input_container {
    layout: horizontal;
    height: auto;
    margin-bottom: 1;
}
.input_container Input {
    width: 1fr; /* Input takes available space */
    margin-right: 1; /* Space before browse button */
}
.input_container .browse_button {
    width: auto; /* Let button size itself */
    height: 100%;
}

.button_container {
    layout: horizontal;
    height: auto;
    margin-top: 1;
}
.button_container Button {
    margin-left: 1; /* Space between buttons */
}
/* Ollama Specific Styles within #llm-view-ollama */
#llm-view-ollama {
    layout: vertical; /* Ensures children stack vertically by default */
    /* overflow-y: auto; /* Add if direct children of llm-view-ollama might overflow before VerticalScroll */
}
#llm-view-ollama VerticalScroll { /* Ensure the VerticalScroll takes up space */
    height: 100%;
}

#llm-view-ollama .label { /* General label styling if not already covered */
    margin-top: 1;
    margin-bottom: 0; /* Tighten space below label if input is right after */
    text-style: bold;
}
#llm-view-ollama .section_label { /* For labels that act as section headers */
    margin-top: 2; /* More space above section labels */
    padding: 1 0;
    border: ascii $primary-background-lighten-1; /* Optional: add a border or background */
    text-align: center;
    width: 100%;
}

#llm-view-ollama .input_field,
#llm-view-ollama .input_field_short,
#llm-view-ollama .input_field_long {
    margin-bottom: 1; /* Space after input fields */
}
#llm-view-ollama .input_field_long { /* For paths or longer inputs */
    width: 100%;
}

#llm-view-ollama .input_action_container { /* For Input + Button on same line */
    layout: horizontal;
    height: auto; /* Fit content */
    margin-bottom: 0; /* Status/Output will provide margin */
}
#llm-view-ollama .input_action_container .input_field_short {
    width: 1fr; /* Input takes most space */
    margin-right: 1;
    margin-bottom: 0; /* No bottom margin if button is next to it */
}
#llm-view-ollama .input_action_container .action_button_short {
    width: auto; /* Let button size itself */
    min-width: 10; /* Ensure it's not too small */
    height: 100%; /* Match input height */
}
#llm-view-ollama .input_action_container .browse_button_short {
    width: auto;
    min-width: 15;
    height: 100%;
}


#llm-view-ollama .action_container { /* For standalone buttons */
    width: 100%;
    height: auto;
    margin-bottom: 0;
}
#llm-view-ollama .action_button { /* For buttons that take full width or are main actions */
    width: 100%;
    margin-top: 1; /* If it's after a section label or input */
}
#llm-view-ollama .full_width_button {
    width: 100%;
    margin-top: 1;
}


#llm-view-ollama .delete_button Button { /* Specific styling for delete button if needed */
    background: $error-darken-1;
    color: $text;
}
#llm-view-ollama .delete_button Button:hover {
    background: $error;
}

/* New Status TextArea */
.ollama-status-textarea {
    width: 100%;
    height: 3; /* Small, 1-2 lines of text */
    border: round $primary-darken-1;
    background: $surface;
    margin-top: 1; /* Space above status */
    margin-bottom: 1; /* Space below status, before main output or next section */
    padding: 0 1;
    color: $text-muted; /* Subdued text color for status */
}

/* Output TextAreas */
#llm-view-ollama .output_textarea_medium { /* For JSON lists, model info */
    width: 100%;
    height: 25; /* Or adjust as needed */
    border: round $primary;
    background: $panel;
    margin-bottom: 1; /* Space before next element/spacer */
}
#llm-view-ollama .output_textarea_small { /* For embeddings */
    width: 100%;
    height: 6;
    border: round $primary;
    background: $panel;
    margin-bottom: 1;
}

/* Main RichLog for streaming */
#llm-view-ollama .log_output_large { /* Renamed from .log_output for specificity */
    width: 100%;
    height: 10; /* Make it taller */
    border: panel $primary-darken-2;
    background: $background-darken-1; /* Darker background for log */
}

/* Spacer element */
.ollama-spacer {
    height: 1; /* Creates 1 cell of vertical space */
    width: 100%;
}

/* OLLAMA compact layout grid */
.ollama-button-bar {
    layout: horizontal;
    height: auto;
    width: 100%;
    margin: 1 0;
    align-horizontal: center;
}
.ollama-button-bar Button {
    width: 1fr;
    margin: 0 1;
}

.ollama-actions-grid {
    layout: horizontal;
    height: auto;
    align-vertical: top; /* Align columns to the top */
    margin-bottom: 1;
}

.ollama-actions-column {
    layout: vertical;
    width: 1fr;
    height: 100%; /* Make columns fill parent height */
    padding: 1;
    border: round $panel-lighten-1;
    margin: 0 1;
}
.ollama-actions-column:first-of-type {
    margin-left: 0;
}
.ollama-actions-column:last-of-type {
    margin-right: 0;
}

.column-title {
    text-style: bold underline;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}

/* Override input container margin inside a column */
.ollama-actions-column .input_action_container {
    margin-bottom: 1;
}
.ollama-actions-column .full_width_button {
    margin-top: 0; /* Tighten up space for buttons below inputs */
}


/* --- End of LLM Management Tab --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
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

.media-nav-pane.collapsed {
    width: 0 !important;
    min-width: 0 !important;
    border-right: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none !important; /* Ensures it doesn't take space or grab focus */
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

.media-content-left-pane {
    width: 35%;
}

.media-content-right-pane {
    width: 65%;
}
/* --- End of Media Tab --- */
/* ----------------------------- ************************* ----------------------------- */







/* ----------------------------- ************************* ----------------------------- */
/* --- Evals Tab --- */
#evals-window { /* Matches TAB_EVALS, .window class provides layout: horizontal */
    /* layout: horizontal; /* Provided by .window by default */
}

#evals-sidebar {
    dock: left;
    width: 25%;
    min-width: 20;
    max-width: 50; /* Adjusted max-width */
    height: 100%;
    background: $boost;
    padding: 1; /* Standard padding */
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

#evals-sidebar.collapsed {
    width: 0 !important;
    min-width: 0 !important;
    border-right: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none !important; /* Ensure it's hidden */
}

/* Styles for the main content area within the Evals tab */
#evals-main-content-area {
    width: 1fr; /* Takes remaining horizontal space */
    height: 100%;
    padding: 1 2; /* Padding for the content area */
    /* border: round $primary; /* Optional: for visual debugging */
}

/* Styles for the sidebar toggle button in the Evals tab */
#toggle-evals-sidebar {
    /* Positioned by EvalsWindow's compose, next to the content area */
    /* dock: left; is set in EvalsWindow.py's DEFAULT_CSS for the button */
    width: auto; /* Small width, text will determine */
    height: 3;   /* Standard button height */
    min-width: 0; /* Allow it to be small */
    margin: 0 1 0 0; /* Top, Right, Bottom, Left margin - space it from main content */
    /* color: $text; */
    /* background: $surface-darken-1; */
    /* border: none; */
}
/* Hover state for the toggle button if needed, can inherit from general .sidebar-toggle if class is added */
/* #toggle-evals-sidebar:hover { background: $surface; } */

/* --- End Evals Tab (Old Implementation) --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
/* --- Media Tab Specific Options --- */
.ingest-form-scrollable {
    height: 1fr; /* Allow scrolling within the form area */
}

.ingest-textarea-small {
    height: auto;
    max-height: 10;
    overflow-y: hidden;
    margin-bottom: 1;
}
.ingest-textarea-medium {
    height: auto;
    max-height: 15;
    overflow-y: hidden;
    margin-bottom: 1;
}
.ingest-form-row {
    layout: horizontal;
    width: 100%;
    height: auto;
    margin-bottom: 1;
}
.title-author-row { /* New class for Title/Author row */
    layout: horizontal;
    width: 100%;
    height: auto;
    margin-bottom: 0 !important; /* Override existing margin */
}
.ingest-form-col {
    width: 1fr;
    padding: 0 1;
}
.ingest-form-col:first-of-type {
    padding-left: 0;
}
.ingest-form-col:last-of-type {
    padding-right: 0;
}

.ingest-submit-button {
    width: 100%;
    margin-top: 2;
}

.hidden {
    display: none;
}
.tldw-api-media-specific-options { /* Common class for specific option blocks */
    padding: 1;
    border: round $surface;
    margin-top: 1;
    margin-bottom: 1;
}

/* --- End of Media Tab Specific Options --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
/* --- Conversations, Characters & Prompts Window specific layouts --- */
#send-chat {
    width: 12;
    min-width: 12 !important;
    max-width: 12vh !important; /* Using ch unit and important */
    /* height: 3; /* Already set by .send-button class, but can be reiterated if needed */
    /* margin-top: 0; /* Already set by .send-button class */
}

#stop-chat-generation {
    width: 6;
    min-width: 6 !important;
    max-width: 6vh !important; /* Using ch unit and important */
    margin: 0 1;
    /* height: 3; /* Already set by .stop-button class, but can be reiterated if needed */
    /* margin-top: 0; /* Already set by .stop-button class */
}

.suggest-button {
    width: 8;
    min-width: 8 !important;
    max-width: 8vh !important; /* Using ch unit and important */
    margin: 0 1;
}
/* --- End of Conversations, Characters & Prompts Window specific layouts --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
/* --- LLM Management Tab --- */

.llm-view-area { /* Common class for individual content areas */
    width: 100%;
    height: 100%; /* Or auto */
    display: none; /* Initially hidden until activated */
}
/* --- End of LLM Management Tab --- */
/* ----------------------------- ************************* ----------------------------- */



/* ----------------------------- ************************* ----------------------------- */
/* --- Window Footer Widget --- */

AppFooterStatus {
    dock: bottom;
    height: 1;
    background: $primary-background-darken-1;
    width: 100%;
    layout: horizontal;
    padding: 0 1;
    /* Removed align: right middle; from parent, will control children individually */
}

#footer-key-palette {
    width: auto;
    padding: 0 1; /* Padding around each key binding */
    color: $text-muted;
    dock: left; /* Dock key bindings to the left */
    visibility: visible;
    display: initial;
}

#footer-key-quit {
    width: auto;
    padding: 0 1; /* Padding around each key binding */
    color: $text-muted;
    dock: left; /* Dock key bindings to the left */
}

#footer-spacer {
    width: 1fr; /* Takes up remaining space in the middle */
}

#internal-db-size-indicator { /* This is for the DB sizes */
    width: auto;
    /* content-align: right; Textual doesn't have content-align for Static directly */
    /* dock: right; Docking within Horizontal might be tricky, align on parent is better */
    color: $text-muted;
    dock: right; /* Dock DB sizes to the right */
    padding: 0 1; /* Add padding to the right of DB sizes as well */
    margin-left: 2; /* Add buffer before DB status */
}
/* --- End of Window Footer Widget --- */
/* ----------------------------- ************************* ----------------------------- */

/* ----------------------------- ************************* ----------------------------- */
/* --- Utility/Not-Tab Specific --- */

.disabled {
    /* Make the button semi-transparent. This is the most effective way
       to show it's inactive and works just like web CSS. */
    opacity: 0.6;

    /* To achieve a "grayscale" effect, we manually set the colors
       to be less vibrant, using Textual's design tokens for theming. */
    background: $panel-darken-2;
    color: $text-muted;

    /* You can also ensure the border is muted. */
    border: round $primary-darken-2;
}
/* --- End of Utility/Not-Tab Specific --- */
/* ----------------------------- ************************* ----------------------------- */

/* Chat Sidebar Media Search Section Specific Styles */
#chat-media-collapsible .sidebar-listview {
    min-height: 5;
    max-height: 15;
    height: 10;
    border: round $surface;
    margin-bottom: 1;
}

.pagination-controls {
    layout: horizontal;
    height: auto;
    width: 100%;
    align-horizontal: center;
    margin-bottom: 1;
}
.pagination-controls Button {
    width: 8;
    min-width: 0;
}
.pagination-controls Label {
    width: 1fr;
    text-align: center;
}

.detail-field-container {
    layout: horizontal;
    height: auto;
    align: center middle;
}
.detail-field-container .detail-label { width: 1fr; }
.detail-field-container .copy-button { width: auto; height: 1; border: none; }
.detail-textarea { height: 5; margin-bottom: 1; }
.detail-textarea.content-display { height: 10; }


/* ----------------------------- ************************* ----------------------------- */
/* --- Search Tab (RAG/Embeddings) --- */
#search-window { /* Matches TAB_SEARCH, .window class provides layout: horizontal */
    /* No explicit layout needed here if .window handles it */
}

.search-nav-pane { /* Style for the left navigation pane in Search Tab */
    dock: left;
    width: 25%;
    min-width: 25;
    max-width: 60;
    height: 100%;
    background: $boost;
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

.search-nav-pane .search-nav-button { /* Style for navigation buttons in Search Tab */
    width: 100%;
    margin-bottom: 1;
    border: none;
    height: 3;
}
.search-nav-pane .search-nav-button:hover {
    background: $accent 80%; /* Example: accent color with 80% opacity */
}
/* Active state for selected search nav button */
.search-nav-pane .search-nav-button.-active-search-sub-view {
    background: $accent;
    color: $text;
    text-style: bold;
}
.search-content-pane { /* Style for the right content display area in Search Tab */
    width: 1fr;
    height: 100%;
    padding: 1 2;
    overflow: auto; /* Changed from overflow-y: auto to allow both horizontal and vertical scrolling if needed */
}

/* -------------------------------------------------------------------------------------- */

/* Web Search Specific Styles within Search Tab */
#search-view-web-search {
    /* Overriding the generic .search-view-area Static centering if needed */
    /* For direct children like Input, Button, VerticalScroll, default layout (vertical) should be fine. */
    padding: 1; /* Add some padding inside the web search view area */
}

#search-view-web-search > Input#web-search-input { /* Target Input directly inside */
    margin-bottom: 1; /* Space below the input field */
    width: 100%;
}

/* .search-action-button is used by #web-search-button */
.search-action-button {
    width: 100%;
    margin-bottom: 1; /* Space below the button */
    /* height: 3; /* Optional: Standard button height */
}

#search-view-web-search > VerticalScroll > Markdown#web-search-results { /* Target Markdown inside VS */
    width: 100%; /* Take full width */
    height: 1fr; /* Take remaining vertical space within its parent VerticalScroll */
    border: round $primary-background-lighten-2;
    padding: 1;
    background: $surface; /* A slightly different background for the results area */
}

/* Embeddings Creation View Styles */
#search-view-embeddings-creation {
    padding: 1;
}

.search-form-container {
    width: 100%;
    margin: 0 0;
}

.search-view-title {
    text-style: bold;
    text-align: center;
    background: blue 30%;
    color: $text;
    padding: 1;
    margin-bottom: 2;
    border: round $accent-darken-1;
}

.search-section-title {
    text-style: bold;
    margin-top: 2;
    margin-bottom: 1;
    background: $primary-background-lighten-1;
    padding: 0 1;
    border-left: thick $accent;
}

.search-form-row {
    margin-bottom: 1;
    height: 3;
    align: left middle;
}

.search-form-label {
    width: 30%;
    padding-right: 1;
    text-align: right;
}

/* Embeddings Management View Styles */
#search-view-embeddings-management {
    padding: 1;
}

.search-management-left-pane {
    width: 45%;
    padding-right: 1;
    border-right: solid $background-darken-1;
}

.search-management-right-pane {
    width: 55%;
    padding-left: 1;
}

.search-button-row {
    margin-top: 2;
    align-horizontal: center;
}

.search-button-row Button {
    margin: 0 1;
}

/* Status output styling */
#creation-status-output, #mgmt-status-output {
    margin-top: 2;
    border: round $primary-background-lighten-2;
    padding: 1;
    background: $surface;
}

/* --- End of Search Tab --- */
/* ----------------------------- ************************* ----------------------------- */
    """
#
#
#
##########################################################################################################################




##########################################################################################################################
#
#
#
LLAMA_CPP_SERVER_ARGS_HELP_TEXT = """
[bold cyan]--- Server & Model Params ---[/]

[bold]Simple 'Just Get Me Up And Running': -ngl 99 -fa -c 8192[/]

[bold]-ngl, --gpu-layers, --n-gpu-layers N[/]
  Number of layers to store in VRAM (e.g., [italic]--n-gpu-layers 35[/])
  (env: LLAMA_ARG_N_GPU_LAYERS)

[bold]-fa, --flash-attn[/]
  Enable Flash Attention (default: disabled)
  (env: LLAMA_ARG_FLASH_ATTN)

[bold]-c, --ctx-size N[/]
  Size of the prompt context (default: 4096, 0 = loaded from model)
  (e.g., [italic]-c 2048[/])
  (env: LLAMA_ARG_CTX_SIZE)

[bold]-n, --predict, --n-predict N[/]
  Number of tokens to predict (default: -1, -1 = infinity)
  (e.g., [italic]-n 512[/])
  (env: LLAMA_ARG_N_PREDICT)

[bold]-m, --model FNAME[/]
  Model path (Set via 'Model Path' field above)
  (env: LLAMA_ARG_MODEL)

[bold]-mu, --model-url MODEL_URL[/]
  Model download URL (default: unused)
  (env: LLAMA_ARG_MODEL_URL)

[bold]-hf, -hfr, --hf-repo <user>/<model>[:quant][/]
  Hugging Face model repository.
  (e.g., [italic]--hf-repo unsloth/phi-3-mini-4k-instruct-gguf:Q4_K_M[/])
  (env: LLAMA_ARG_HF_REPO)

[bold]-hfd, -hfrd, --hf-repo-draft <user>/<model>[:quant][/]
  Same as --hf-repo, but for the draft model.
  (env: LLAMA_ARG_HFD_REPO)

[bold]-hft, --hf-token TOKEN[/]
  Hugging Face access token.
  (env: HF_TOKEN)

[bold]-t, --threads N[/]
  Number of threads for generation (default: system dependent)
  (e.g., [italic]-t 8[/])
  (env: LLAMA_ARG_THREADS)

[bold]-b, --batch-size N[/]
  Logical maximum batch size (default: 2048)
  (env: LLAMA_ARG_BATCH)

[bold]-tb, --threads-batch N[/]
  Number of threads for batch/prompt processing (default: same as --threads)
  (env: LLAMA_ARG_THREADS_BATCH)

[bold]-ub, --ubatch-size N[/]
  Physical maximum batch size (default: 512)
  (env: LLAMA_ARG_UBATCH)

[bold]--keep N[/]
  Tokens to keep from initial prompt (default: 0, -1 = all)

[bold]-e, --escape[/]
  Process escape sequences (default: true)

[bold]--no-escape[/]
  Do not process escape sequences

[bold]--lora FNAME[/]
  Path to LoRA adapter (repeatable)

[bold]--lora-scaled FNAME SCALE[/]
  Path to LoRA adapter with scaling (repeatable)

[bold]--control-vector FNAME[/]
  Add a control vector (repeatable)

[bold]--control-vector-scaled FNAME SCALE[/]
  Add a scaled control vector (repeatable)

[bold]--control-vector-layer-range START END[/]
  Layer range for control vector(s)

[bold cyan]--- Sampling Params ---[/]

[bold]--samplers SAMPLERS[/]
  Samplers order, separated by ';' (default: see llama.cpp help)

[bold]-s, --seed SEED[/]
  RNG seed (default: -1, random)
  (e.g., [italic]-s 1234[/])

[bold]--temp N[/]
  Temperature (default: 0.8)
  (e.g., [italic]--temp 0.7[/])

[bold]--top-k N[/]
  Top-k sampling (default: 40, 0 = disabled)
  (e.g., [italic]--top-k 50[/])

[bold]--top-p N[/]
  Top-p sampling (default: 0.9, 1.0 = disabled)
  (e.g., [italic]--top-p 0.95[/])

[bold]--min-p N[/]
  Min-p sampling (default: 0.1, 0.0 = disabled)
  (e.g., [italic]--min-p 0.05[/])

[bold]--typical N[/]
  Locally typical sampling (default: 1.0, 1.0 = disabled)

[bold]--repeat-last-n N[/]
  Last N tokens for penalty (default: 64, 0 = disabled)

[bold]--repeat-penalty N[/]
  Repeat penalty (default: 1.0, 1.0 = disabled)
  (e.g., [italic]--repeat-penalty 1.1[/])

[bold]--presence-penalty N[/]
  Presence penalty (default: 0.0, 0.0 = disabled)

[bold]--frequency-penalty N[/]
  Frequency penalty (default: 0.0, 0.0 = disabled)

[bold]--mirostat N[/]
  Mirostat sampling (0=disabled, 1=Mirostat, 2=Mirostat 2.0)

[bold]--mirostat-lr N[/]
  Mirostat learning rate (default: 0.1)

[bold]--mirostat-ent N[/]
  Mirostat target entropy (default: 5.0)

[bold]-l, --logit-bias TOKEN_ID(+/-)BIAS[/]
  Modify token likelihood (e.g., [italic]--logit-bias 15043+1[/])

[bold]--grammar GRAMMAR[/]
  BNF-like grammar constraint

[bold]--grammar-file FNAME[/]
  File for grammar

[bold]-j, --json-schema SCHEMA[/]
  JSON schema constraint

[bold]-jf, --json-schema-file FILE[/]
  File for JSON schema

[italic]Obtained from: https://github.com/ggml-org/llama.cpp/tree/master/tools/server[/]
"""
#
#
#
##########################################################################################################################

##########################################################################################################################





##########################################################################################################################



##########################################################################################################################
#
#
#
LLAMAFILE_SERVER_ARGS_HELP_TEXT = """
[bold cyan]--- Server & Model Params ---[/]

[bold]Simple 'Just Get Me Up And Running': -ngl 99 -fa -c 8192[/]

--threads N, -t N: Set the number of threads to use during generation.

-tb N, --threads-batch N: Set the number of threads to use during batch and prompt processing. If not specified, the number of threads will be set to the number of threads used for generation.

-m FNAME, --model FNAME: Specify the path to the LLaMA model file (e.g., models/7B/ggml-model.gguf).

-a ALIAS, --alias ALIAS: Set an alias for the model. The alias will be returned in API responses.

-c N, --ctx-size N: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference. The size may differ in other models, for example, baichuan models were build with a context of 4096.

-ngl N, --n-gpu-layers N: When compiled with appropriate support (currently CLBlast or cuBLAS), this option allows offloading some layers to the GPU for computation. Generally results in increased performance.

-mg i, --main-gpu i: When using multiple GPUs this option controls which GPU is used for small tensors for which the overhead of splitting the computation across all GPUs is not worthwhile. The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results. By default GPU 0 is used. Requires cuBLAS.

-ts SPLIT, --tensor-split SPLIT: When using multiple GPUs this option controls how large tensors should be split across all GPUs. SPLIT is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order. For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1. By default the data is split in proportion to VRAM but this may not be optimal for performance. Requires cuBLAS.

-b N, --batch-size N: Set the batch size for prompt processing. Default: 512.

--memory-f32: Use 32-bit floats instead of 16-bit floats for memory key+value. Not recommended.

--mlock: Lock the model in memory, preventing it from being swapped out when memory-mapped.

--no-mmap: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed.

--numa: Attempt optimizations that help on some NUMA systems.

--lora FNAME: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.

--lora-base FNAME: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the --lora flag, and specifies the base model for the adaptation.
-to N, --timeout N: Server read/write timeout in seconds. Default 600.

--host: Set the hostname or ip address to listen. Default 127.0.0.1.

--port: Set the port to listen. Default: 8080.

--path: path from which to serve static files (default examples/server/public)

--api-key: Set an api key for request authorization. By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token. May be used multiple times to enable multiple valid keys.

--api-key-file: path to file containing api keys delimited by new lines. If set, requests must include one of the keys for access. May be used in conjunction with --api-key's.

--embedding: Enable embedding extraction, Default: disabled.
-np N, --parallel N: Set the number of slots for process requests (default: 1)

-cb, --cont-batching: enable continuous batching (a.k.a dynamic batching) (default: disabled)

-spf FNAME, --system-prompt-file FNAME Set a file to load "a system prompt (initial prompt of all slots), this is useful for chat applications. See more

--mmproj MMPROJ_FILE: Path to a multimodal projector file for LLaVA.

--grp-attn-n: Set the group attention factor to extend context size through self-extend(default: 1=disabled), used together with group attention width --grp-attn-w

--grp-attn-w: Set the group attention width to extend context size through self-extend(default: 512), used together with group attention factor --grp-attn-n

[italic]Obtained from: https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md[/]
"""










#
# MLX-LM Server Arguments Help Text
MLX_LM_SERVER_ARGS_HELP_TEXT = """
[bold cyan]--- MLX-LM Server Arguments ---[/]

options:
  --adapter-path ADAPTER_PATH
                        Optional path for the trained adapter weights and
                        config.
  


  --temp TEMP           Default sampling temperature (default: 0.0)
  --top-p TOP_P         Default nucleus sampling top-p (default: 1.0)
  --top-k TOP_K         Default top-k sampling (default: 0, disables top-k)
  --min-p MIN_P         Default min-p sampling (default: 0.0, disables min-p)
  --max-tokens MAX_TOKENS
                        Default maximum number of tokens to generate (default:
                        512)
  --chat-template-args CHAT_TEMPLATE_ARGS
                        A JSON formatted string of arguments for the
                        tokenizer's apply_chat_template, e.g.
                        '{"enable_thinking":false}'

[bold]--model MODEL[/]
  The path to the MLX model weights, tokenizer, and config
  (e.g., [italic]--model mlx-community/Qwen3-30B-A3B-4bit[/])

[bold]--host HOST[/]
  Host address to bind the server to (default: 127.0.0.1)
  (e.g., [italic]--host 0.0.0.0[/])

[bold]--port PORT[/]
  Port to run the server on (default: 8080)
  (e.g., [italic]--port 8000[/])

[bold]--draft-model DRAFT_MODEL[/]
    A model to be used for speculative decoding.
    (e.g., [italic]--draft-model mlx-community/Qwen3-0.6B-8bit[/])

[bold]--num-draft-tokens NUM_DRAFT_TOKENS[/]
    Number of tokens to draft when using speculative decoding.

[bold]--trust-remote-code[/]
  Enable trusting remote code for tokenizer
  
[bold]--chat-template CHAT_TEMPLATE[/]
    Specify a chat template for the tokenizer

[bold]--use-default-chat-template[/]
    Use the default chat template

[bold]--temperature TEMP[/]
  Sampling temperature (default: 0.8)
  (e.g., [italic]--temperature 0.7[/])

[bold]--top-p P[/]
  Top-p sampling (default: 0.9)
  (e.g., [italic]--top-p 0.95[/])

[bold]--top-k K[/]
  Top-k sampling (default: 40)
  (e.g., [italic]--top-k 50[/])

[bold]--min-p MIN_P[/]
    Default min-p sampling (default: 0.0, disables min-p)

[bold]--max-tokens N[/]
  Maximum number of tokens to generate (default: 100)
  (e.g., [italic]--max-tokens 512[/])

[bold]--chat-template-args CHAT_TEMPLATE_ARGS[/]
    A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'
"""

# End of Constants.py
########################################################################################################################
