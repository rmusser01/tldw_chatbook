"""Internal/system prompt registry.

Import from this package (not submodules) so subsystem prompt modules are
registered: ``from tldw_chatbook.Internal_Prompts import get_internal_prompt``.
"""

from .catalog import CATALOG, PromptSpec, register
from .resolver import get_internal_prompt, render_internal_prompt, safe_substitute

__all__ = [
    "CATALOG",
    "PromptSpec",
    "register",
    "get_internal_prompt",
    "render_internal_prompt",
    "safe_substitute",
]
