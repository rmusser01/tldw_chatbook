"""Find a character card inside Markdown text in linear time (TASK-38226).

``load_character_card_from_string_content`` reads two Markdown forms besides
raw JSON: YAML frontmatter and a fenced json block. On dev (29592ecc9b) it
found them with two backtracking regexes:

* frontmatter: ``re.match(r"^\\s*---\\s*\\n(.*?)\\n\\s*---\\s*", text, re.DOTALL)``
* json block: ``re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text,
  re.DOTALL | re.IGNORECASE)``

Their overlapping ``\\s*``, ``.*?`` and ``\\n`` were retried at every
backtrack point and, for the second, at every fence. A 2 KB crafted
frontmatter took 2.7-6.0 s there (cubic) and 48 KB of ``"```{"`` 2.6 s
(quadratic), on the import worker, which cannot be cancelled. This module
replaces both regexes, and the parser now calls it: the same inputs take
under a millisecond, and 10 MB of the worst ones about 0.3 s.

Each function returns exactly what its regex's ``group(1)`` returned, for any
string (``Tests/Character_Chat/test_character_import_markdown.py`` keeps both
regexes as the oracle and compares them on generated and real cards). Each
makes a fixed number of scans, and every scan is linear in the text:

* ``str.find``, ``str.rfind`` and ``str.startswith`` with fixed needles.
* ``_WHITESPACE`` anchored at one position: one run, and nothing after it.
* Three searches whose quantifiers are all possessive (``*+``, ``?+``), so
  no start position backtracks, and whose starts never re-read each other's
  input. ``_FENCE_OPEN`` reads a whitespace run only after a fence, and no
  later fence starts inside that run. ``_FENCE_CLOSE`` reads one only after a
  ``}``. ``_FRONTMATTER_CLOSE``'s lookbehind lets a run be read only from its
  first character. The old frontmatter regex read a run of n newlines once
  from each of them.

Pure and dependency-free, and imported only from inside the parser, so the
module adds nothing to boot or to the screen pre-import.
"""

from __future__ import annotations

import re

#: One whitespace run from a given position (Python's ``\s``, as the old
#: regexes used). Nothing follows it, so it never backtracks.
_WHITESPACE = re.compile(r"\s*")
#: The first newline of a whitespace run that ends in ``---``. The lookbehind
#: lets a match start only at a run's first character.
_FRONTMATTER_CLOSE = re.compile(r"(?<!\s)[^\S\n]*+(\n)\s*+---")
#: A fence, an optional ``json`` tag in any case (``re.IGNORECASE`` lets
#: ``ſ`` match ``s``, as the old regex did), whitespace, then ``{``.
_FENCE_OPEN = re.compile(r"```(?:json)?+\s*+\{", re.IGNORECASE)
#: A ``}`` whose following whitespace run ends in a fence.
_FENCE_CLOSE = re.compile(r"\}\s*+```")


def _skip_whitespace(text: str, pos: int) -> int:
    """Return the first index at or after ``pos`` that is not whitespace."""
    return _WHITESPACE.match(text, pos).end()


def frontmatter_body(text: str) -> str | None:
    """Return the YAML frontmatter body, or None when there is none.

    The result equals dev's ``re.match(r"^\\s*---\\s*\\n(.*?)\\n\\s*---\\s*",
    text, re.DOTALL).group(1)``. Read the regex as the engine runs it:

    1. ``^\\s*---``: the leading whitespace, then ``---``. A ``-`` is not
       whitespace, so the run can end nowhere else.
    2. ``\\s*\\n`` is greedy, so the body starts after the last newline in the
       whitespace run that follows the opening ``---``.
    3. ``(.*?)\\n\\s*---`` is lazy, so the body ends at the first later
       newline whose following whitespace run ends in ``---``.
    4. Only when step 3 finds no such newline does step 2 back off to the
       run's previous newline. The body then ends at the run's last newline,
       when the run itself ends in ``---``. Every earlier newline in the run
       shares that verdict, so backing off further never helps.

    Args:
        text: Any string. The parser passes its BOM-stripped, left-stripped
            content.

    Returns:
        The text between the delimiters, unstripped, or None.
    """
    start = _skip_whitespace(text, 0)
    if not text.startswith("---", start):
        return None
    after_opening = start + 3
    run_end = _skip_whitespace(text, after_opening)
    last_newline = text.rfind("\n", after_opening, run_end)
    if last_newline < 0:
        return None
    body_start = last_newline + 1
    closing = _FRONTMATTER_CLOSE.search(text, body_start)
    if closing is not None:
        return text[body_start : closing.start(1)]
    if text.startswith("---", run_end):
        previous_newline = text.rfind("\n", after_opening, last_newline)
        if previous_newline >= 0:
            return text[previous_newline + 1 : last_newline]
    return None


def fenced_json_object(text: str) -> str | None:
    """Return the first fenced ``{...}`` block, or None when there is none.

    The result equals dev's ``re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```",
    text, re.DOTALL | re.IGNORECASE).group(1)``. Read the regex as the engine
    runs it:

    1. The leftmost fence that opens a ``{``: three backticks, an optional
       ``json`` tag, whitespace, then ``{``. A tag starts with ``j``, which is
       neither whitespace nor ``{``, so a fence opens at most one ``{``.
    2. ``(\\{.*?\\})\\s*```` is lazy, so the block ends at the first later
       ``}`` whose following whitespace run ends in a fence.
    3. When the leftmost opening has no such ``}``, no later opening can: a
       later one's ``{`` lies after this one's, so it could only use some of
       the same ``}`` candidates. The old search retried at every later
       fence anyway, which made it quadratic.

    Args:
        text: Any string.

    Returns:
        The block from its ``{`` to its ``}``, or None.
    """
    opening = _FENCE_OPEN.search(text)
    if opening is None:
        return None
    brace = opening.end() - 1
    closing = _FENCE_CLOSE.search(text, brace + 1)
    if closing is None:
        return None
    return text[brace : closing.start() + 1]


__all__ = ["fenced_json_object", "frontmatter_body"]
