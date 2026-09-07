"""Shared markdown parser factories for preview surfaces (TASK-1993).

Notes synced from files and HuggingFace READMEs commonly start with YAML
front matter. Textual's default gfm-like parser renders the ``---`` block as
a thematic break plus stray list/paragraph noise at the top of the preview.
With ``mdit-py-plugins`` installed, the factory returned here consumes front
matter instead of rendering it; without it, callers fall back to Textual's
default parser (today's behavior).
"""

from typing import Callable, Optional

from tldw_chatbook.Utils.optional_deps import check_dependency


def front_matter_parser_factory() -> Optional[Callable]:
    """Return a gfm-like MarkdownIt factory that consumes YAML front matter.

    Returns:
        A zero-arg factory for ``Markdown(parser_factory=...)``, or ``None``
        when ``mdit-py-plugins`` is not installed — ``None`` selects Textual's
        default parser, so absence of the extra changes nothing.
    """
    if not check_dependency("mdit_py_plugins", "front_matter"):
        return None
    from markdown_it import MarkdownIt
    from mdit_py_plugins import front_matter

    return lambda: MarkdownIt("gfm-like").use(front_matter.front_matter_plugin)
