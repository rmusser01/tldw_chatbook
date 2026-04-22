# Emoji_Handling.py
#
# Imports
import os
import platform
import sys
#
# Third-party Imports
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

#####################################
# --- Emoji Checker/Support ---
#####################################
# Cache the result so we don't re-calculate every time
_emoji_support_cached = None

def supports_emoji() -> bool:
    """
    Detects if the current terminal likely supports emojis.
    This is heuristic-based and not 100% foolproof.
    Caches the result for efficiency.
    """
    global _emoji_support_cached
    if _emoji_support_cached is not None:
        return _emoji_support_cached

    # 1. Must be a TTY
    if not sys.stdout.isatty():
        _emoji_support_cached = False
        return False

    # 2. Encoding should ideally be UTF-8
    # (getattr is used for safety, e.g., if sys.stdout is mocked)
    encoding = getattr(sys.stdout, 'encoding', '').lower()
    if 'utf-8' not in encoding and 'utf8' not in encoding:
        # Some terminals might still render emojis with other encodings,
        # but UTF-8 is the most reliable indicator.
        # For cmd.exe, even with chcp 65001 (UTF-8), font support is the main issue.
        # Don't immediately fail, let OS checks decide more
        #_emoji_support_cached = False
        #return False
        pass

    #####################################
    # 3. OS-specific checks
    #####################################
    os_name = platform.system()

    if os_name == 'Windows':
        # Windows Terminal has good emoji support.
        if 'WT_SESSION' in os.environ or 'TERMINUS_SUBLIME' in os.environ: # WT_SESSION for Windows Terminal, TERMINUS_SUBLIME for Terminus
            _emoji_support_cached = True
            return True
        # For older cmd.exe or PowerShell without Windows Terminal,
        # emoji support is unreliable or poor even with UTF-8 codepage.
        # Check if running in ConEmu, which has better support
        if 'CONEMUBUILD' in os.environ or 'CMDER_ROOT' in os.environ :
             _emoji_support_cached = True
             return True
        # Check if it's Fluent Terminal
        if os.environ.get('FLUENT_TERMINAL_PROFILE_NAME'):
            _emoji_support_cached = True
            return True

        # For standard cmd.exe or older PowerShell, be pessimistic.
        _emoji_support_cached = False
        return False

    # For macOS and Linux:
    # If it's a UTF-8 TTY, support is generally good on modern systems.
    # We can check for TERM=dumb as a negative indicator.
    if os.environ.get('TERM') == 'dumb':
        _emoji_support_cached = False
        return False

    # If encoding wasn't explicitly UTF-8 earlier, but it's Linux/macOS not TERM=dumb,
    # it's still likely okay on modern systems.
    # However, to be safer, if not UTF-8, tend towards no.
    if 'utf-8' not in encoding and 'utf8' not in encoding:
        _emoji_support_cached = False
        return False

    # Default to True for non-Windows UTF-8 (or generally capable) TTYs not being 'dumb'
    _emoji_support_cached = True
    return True

# Define your emoji and fallback pairs
# You can centralize these or define them where needed.
# Example:
# --- Emoji and Fallback Definitions ---
EMOJI_TITLE_BRAIN = "🧠"
FALLBACK_TITLE_BRAIN = "[B]"
EMOJI_TITLE_NOTE = "📝"
FALLBACK_TITLE_NOTE = "[N]"
EMOJI_TITLE_SEARCH = "🔍"
FALLBACK_TITLE_SEARCH = "[S]"

EMOJI_SEND = "▶ Send"
FALLBACK_SEND = "Send"

EMOJI_SIDEBAR_TOGGLE = "☰"
FALLBACK_SIDEBAR_TOGGLE = "Menu"

EMOJI_CHARACTER_ICON = "👤"
FALLBACK_CHARACTER_ICON = "Char"

EMOJI_THINKING = "🤔" # Or "⏳" "💭"
FALLBACK_THINKING = "..."

EMOJI_COPY = "📋"
FALLBACK_COPY = "Copy"
EMOJI_COPIED = "✅" # For feedback
FALLBACK_COPIED = "[OK]"

EMOJI_EDIT = "✏️"
FALLBACK_EDIT = "Edit"
EMOJI_SAVE_EDIT = "💾" # Or use check/OK
FALLBACK_SAVE_EDIT = "Save"

ROCKET_EMOJI = "🚀"
ROCKET_FALLBACK = "[Go!]"

CHECK_EMOJI = "✅"
CHECK_FALLBACK = "[OK]"

CROSS_EMOJI = "❌"
CROSS_FALLBACK = "[FAIL]"

SPARKLES_EMOJI = "✨"
SPARKLES_FALLBACK = "*"

EMOJI_STOP = "⏹ Stop"
FALLBACK_STOP = "Stop"

EMOJI_WRITE_FOR_ME = "💡"  # Suggestion button
FALLBACK_WRITE_FOR_ME = "Suggest"

def get_char(emoji_char: str, fallback_char: str) -> str:
    """Returns the emoji if supported, otherwise the fallback."""
    return emoji_char if supports_emoji() else fallback_char


#
# End of Emoji_Handling.py
#######################################################################################################################
