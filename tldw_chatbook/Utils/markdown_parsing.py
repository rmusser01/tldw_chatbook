"""Shared markdown parser factories for preview surfaces (TASK-1993).

Notes synced from files and HuggingFace READMEs commonly start with YAML
front matter. Textual's default gfm-like parser renders the ``---`` block as
a thematic break plus stray list/paragraph noise at the top of the preview.
With ``mdit-py-plugins`` installed, the factory returned here consumes front
matter instead of rendering it; without it, callers fall back to Textual's
default parser (today's behavior).
"""

import re
from typing import Callable, Optional

from tldw_chatbook.Utils.optional_deps import check_dependency

#: One Obsidian callout header: ``> [!note] Title``, ``> [!warning]-`` etc.
#: Only the header line carries the marker; the body lines under it are
#: ordinary blockquote text and need no rewrite.
_CALLOUT_HEADER_RE = re.compile(
    r"^(?P<quote>[ \t]*>[ \t]*)\[!(?P<kind>[A-Za-z][\w-]*)\][+-]?(?P<title>.*)$",
    re.MULTILINE,
)


def render_obsidian_callouts(text: str) -> str:
    """Rewrite Obsidian callout headers as plain Markdown blockquote headers.

    task-32249: Textual's Markdown has no callout extension, so a note
    written in Obsidian rendered its marker as literal text --
    ``▌ [!note] Obsidian callout`` -- inside the quote bar the callout was
    already drawing. Turning the header into a bold label keeps the quote
    bar (the callout's box), keeps the type word and the author's title,
    and drops the marker that only Obsidian can read.

    Args:
        text: The note's raw Markdown source.

    Returns:
        The same source with every callout header line rewritten. Text
        with no callout in it is returned unchanged.
    """

    def _header(match: re.Match[str]) -> str:
        kind = match.group("kind").replace("-", " ").strip().capitalize()
        title = match.group("title").strip()
        label = f"{kind}: {title}" if title else kind
        return f"{match.group('quote')}**{label}**"

    return _CALLOUT_HEADER_RE.sub(_header, text)


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
