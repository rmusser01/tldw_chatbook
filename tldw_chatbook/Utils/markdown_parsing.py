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

#: One Obsidian callout header: ``> [!note] Title``, ``> [!warning]-``, and
#: nested ``> > [!tip]``. Only the header line carries the marker; the body
#: lines under it are ordinary blockquote text and need no rewrite.
_CALLOUT_HEADER_RE = re.compile(
    r"^(?P<quote>[ \t]*(?:>[ \t]*)+)\[!(?P<kind>[A-Za-z][\w-]*)\][+-]?(?P<title>.*)$"
)

#: A fenced-code delimiter, with the blockquote markers a fence inside a quote
#: carries. Only the RUN matters, not its info string.
_FENCE_RE = re.compile(r"^[ \t]*(?:>[ \t]*)*(?P<fence>```+|~~~+)")


def render_obsidian_callouts(text: str) -> str:
    """Rewrite Obsidian callout headers as plain Markdown blockquote headers.

    task-32249: Textual's Markdown has no callout extension, so a note
    written in Obsidian rendered its marker as literal text --
    ``▌ [!note] Obsidian callout`` -- inside the quote bar the callout was
    already drawing. Turning the header into a bold label keeps the quote
    bar (the callout's box), keeps the type word and the author's title,
    and drops the marker that only Obsidian can read.

    Scanned line by line rather than with one ``re.MULTILINE`` pass (review
    of the wave-3 layout branch, F8): a note that DOCUMENTS callout syntax
    inside a fenced block had its own example rewritten in Preview.

    ponytail: fence tracking is a run-length toggle, not a CommonMark
    parser -- it does not model indented code blocks or a fence opened
    inside one blockquote and closed in another. Both render wrong in
    Textual's Markdown anyway; reach for a real parser only if a note ever
    needs one.

    Args:
        text: The note's raw Markdown source.

    Returns:
        The same source with every callout header line outside a fenced
        block rewritten. Text with no callout in it is returned unchanged.
    """

    def _header(match: re.Match[str]) -> str:
        kind = match.group("kind").replace("-", " ").strip()
        # ``.capitalize()`` lowercases the rest of the word, which turned an
        # acronym type like ``[!TODO]`` into "Todo". A type the author wrote
        # in caps is kept as written.
        if not kind.isupper():
            kind = kind.capitalize()
        title = match.group("title").strip()
        label = f"{kind}: {title}" if title else kind
        return f"{match.group('quote')}**{label}**"

    lines = text.split("\n")
    open_fence: str | None = None
    for index, line in enumerate(lines):
        fence = _FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group("fence")
            if open_fence is None:
                open_fence = marker[0] * len(marker)
            elif marker[0] == open_fence[0] and len(marker) >= len(open_fence):
                open_fence = None
            continue
        if open_fence is None:
            lines[index] = _CALLOUT_HEADER_RE.sub(_header, line)
    return "\n".join(lines)


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
