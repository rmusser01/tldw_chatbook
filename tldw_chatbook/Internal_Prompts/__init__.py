"""Internal/system prompt registry.

Import from this package (not submodules) so subsystem prompt modules are
registered: ``from tldw_chatbook.Internal_Prompts import get_internal_prompt``.
"""

from .catalog import CATALOG, PromptSpec, register
from . import websearch_prompts  # noqa: F401  (registers specs on import)
from . import rag_reranker_prompts  # noqa: F401  (registers specs on import)
from . import agents_prompts  # noqa: F401  (registers specs on import)
from . import summarization_prompts  # noqa: F401  (registers specs on import)
from .resolver import get_internal_prompt, render_internal_prompt, safe_substitute

__all__ = [
    "CATALOG",
    "PromptSpec",
    "register",
    "get_internal_prompt",
    "render_internal_prompt",
    "safe_substitute",
]
