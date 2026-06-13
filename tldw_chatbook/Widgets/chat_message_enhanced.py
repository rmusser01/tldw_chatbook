"""Compatibility alias for the enhanced chat message widget module."""

import sys

from tldw_chatbook.Widgets.Chat_Widgets import chat_message_enhanced as _impl

sys.modules[__name__] = _impl
