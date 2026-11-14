"""Compatibility shim: the sanitizer now lives in ``Utils.character_display_text``.

TASK-36944.6 moved every in-repo importer to the Utils path; this shim only keeps
branches written against the old path working. Delete it once nothing imports
``tldw_chatbook.UI.character_display_text``.
"""

from tldw_chatbook.Utils.character_display_text import *  # noqa: F401,F403
from tldw_chatbook.Utils.character_display_text import __all__  # noqa: F401
