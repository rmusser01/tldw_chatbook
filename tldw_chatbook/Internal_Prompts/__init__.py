"""Internal/system prompt registry.

Import from this package (not submodules) so subsystem prompt modules are
registered: ``from tldw_chatbook.Internal_Prompts import get_internal_prompt``.
"""

from .catalog import CATALOG, PromptSpec, register
from . import websearch_prompts  # noqa: F401  (registers specs on import)
from . import rag_reranker_prompts  # noqa: F401  (registers specs on import)
from . import agents_prompts  # noqa: F401  (registers specs on import)
from . import console_prompts  # noqa: F401  (registers specs on import)
from . import summarization_prompts  # noqa: F401  (registers specs on import)
from . import document_generation_prompts  # noqa: F401  (registers specs on import)
# subscriptions_prompts was removed in TASK-1220 with the ContentProcessor it
# described. Its five specs were listed in Settings > Internal Prompts, badged as
# customizable, and could not affect anything -- their only consumer had no
# caller. Recover the text from git history if the briefing work needs it.
from . import character_prompts  # noqa: F401  (registers specs on import)
from .resolver import get_internal_prompt, render_internal_prompt, safe_substitute

__all__ = [
    "CATALOG",
    "PromptSpec",
    "register",
    "get_internal_prompt",
    "render_internal_prompt",
    "safe_substitute",
]
