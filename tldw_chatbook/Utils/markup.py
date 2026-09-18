"""The one escaper that matches the parser this app actually renders through.

``rich.markup.escape`` -- and ``textual.markup.escape``, which shares its
regex -- only escapes tags matching ``\\[[a-z#/@]...]``. Textual 8 renders a
``str`` handed to a widget through ``Content.from_markup``, which consumes
ANY ``[...]`` token, uppercase included. The two disagree on exactly the
tokens users write in titles:

    >>> from textual.content import Content
    >>> from rich.markup import escape
    >>> Content.from_markup(escape("[TODO] Q3 plan")).plain
    ' Q3 plan'
    >>> Content.from_markup(escape("[IMPORTANT]")).plain
    ''

So a note called ``[TODO] Q3 plan`` lost its tag and ``[IMPORTANT]`` rendered
as an empty row, across roughly 40 surfaces. Escaping every bracket is the
only escape that matches this parser, and it is equally correct for Rich's
own parser -- ``\\[`` is a literal ``[`` to both, so a Rich consumer renders
byte-identically to what ``rich.markup.escape`` gave it.

The one place this is NOT right is a markup-OFF sink: a plain ``Text(...)``,
a log line, a ``Static(..., markup=False)``. There the backslash is shown
literally -- but that was already true of ``rich.markup.escape`` for any
lowercase tag, so such a site is a pre-existing double-escape bug, not a
reason to keep the narrow escape.

This function was first written for the Schedules results pane (task 6),
where live verification lost a literal ``[PR-6]`` out of a real result while
that pane's escaping test passed -- the test used ``[bold]``, the one shape
rich does escape. TASK-32802.1 moved it here and adopted it repo-wide.
"""

from __future__ import annotations

__all__ = ["escape_markup"]


def escape_markup(value: object) -> str:
    """Escape EVERY ``[`` so a Textual or Rich surface renders text literally.

    Args:
        value: Text to render literally. Coerced via ``str()``, so a
            non-string stored-JSON value is safe to pass straight in.

    Returns:
        The same text with every ``[`` backslash-escaped.
    """
    return str(value).replace("[", "\\[")
