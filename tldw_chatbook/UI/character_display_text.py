"""Compatibility shim: the sanitizer now lives in ``Utils.character_display_text``."""

from tldw_chatbook.Utils.character_display_text import *  # noqa: F401,F403
from tldw_chatbook.Utils.character_display_text import __all__  # noqa: F401
