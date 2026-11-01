"""Decide a compose delta that moves no box without composing it.

TASK-36012. A widget that rebuilds itself with ``refresh(recompose=True)``
removes and re-mounts every child; each mount asks Textual for a layout, and
Textual answers any layout request with a reflow of the whole screen. When a
new state changes only what some children *say* -- a label, a tooltip, a
colour class -- and not their boxes, that rebuild is pure cost, and the
widget can write the new values into its mounted children instead.

TASK-33620.15.20. Deciding that by composing the widget twice off the DOM
and diffing the two trees cost as much as the rebuild it replaced: every
widget those composes built was constructed and restyled (each
``add_class``/``set_class`` restyles its node at once, mounted or not), and
then dropped. A widget that patches in place now decides from its state:
compose records the values it wrote that a delta may change, and the patch
compares that record with the values the next state gives. The helpers here
are the parts of that comparison that do not depend on the widget:

* ``ComposeReadView`` answers, before anything else, which state fields the
  last compose read. A delta confined to fields it did not read needs neither
  a patch nor a rebuild.
* ``same_visual`` compares two label, content or tooltip values the way they
  render, style included.
* ``box_ignores_content`` says whether a widget's copy can change without a
  layout: only then does a repaint land the geometry a rebuild would.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.content import Content
from textual.widget import Widget


def _visual_key(value: Any) -> Any:
    if isinstance(value, Content):
        return ("content", value.plain, tuple(value.spans))
    if isinstance(value, Text):
        return ("text", value.plain, tuple(value.spans), str(value.style))
    return value


def same_visual(left: Any, right: Any) -> bool:
    """Return whether two labels, contents or tooltips render the same.

    ``Content.__eq__`` compares text only, and ``Text.__eq__`` ignores the
    base style (a colour given as ``Text(icon, style=color)``), so plain
    equality would call a colour change "unchanged".

    Args:
        left: A ``Content``, ``Text``, string or ``None``.
        right: Another such value.

    Returns:
        True when the two show the same text with the same styling.
    """
    return _visual_key(left) == _visual_key(right)


def box_ignores_content(node: Widget) -> bool:
    """True when neither of ``node``'s box sides is sized by its content.

    An unset side fills its container. An ``fr`` width counts as
    content-sized: Textual resolves it as ``auto`` when an auto-width parent
    arranges for its optimal size.

    Args:
        node: A mounted widget whose label or content may change.

    Returns:
        True when new copy repaints inside the same box.
    """
    width = node.styles.width
    height = node.styles.height
    return (width is None or not (width.is_auto or width.is_fraction)) and (
        height is None or not height.is_auto
    )


class ComposeReadView:
    """Attribute-recording view of a widget's state, live only during compose.

    TASK-26836: the Conversations tray's ``_can_skip_recompose`` used
    whole-state value equality, so a delta in a field the tray's ``content``
    never renders (probe-observed: ``conversation_browser`` pushed into the
    ``content="workspace"`` tray) still forced a full recompose -- and its
    paint-blocking App batch. The tray now records which top-level state
    fields ``compose`` actually reads, exactly as it already records the row
    signature of what it builds, and the guard skips when only unread fields
    changed.

    TASK-36012 gave it a second owner: the Details tray rebuilds only when a
    field its last completed compose read changes.

    The view delegates everything to the real state; reads observed while it
    is live can only ADD fields to the recorded set, so interleaved readers
    during a compose window bias toward recomposing more, never less.
    """

    __slots__ = ("_target", "_reads")

    def __init__(self, target: object, reads: set) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_reads", reads)

    def __getattr__(self, name: str):
        if not name.startswith("_"):
            self._reads.add(name)
        return getattr(self._target, name)

    def __eq__(self, other: object) -> bool:
        return self._target == getattr(other, "_target", other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return self._target is not None

    def __repr__(self) -> str:
        return f"ComposeReadView({self._target!r})"


def composer_extend_fitting_cells(prefix: str, text: str, width: int) -> str:
    """Return the longest prefix of ``text`` that keeps ``prefix + <it>`` within ``width``.

    Measures the actual **joined** candidate (``prefix + text[:k]``)
    directly with `cell_len` at every step -- never a pre-computed
    numeric budget (``width - cell_len(prefix)``) checked against
    `cell_len` of the candidate piece *alone*. Cell width is not
    additive across a join in every case, and not only in the way
    `_cell_wrap_line`'s own docstring already covers (a boundary
    crossing a narrow_to_wide upgrade): `cell_len` has a still sharper
    edge where a **trailing** ZWJ silently absorbs the character that
    would have followed it in a longer string (confirmed:
    ``cell_len("#ZXY b9\\u200d ")`` -- 9 characters, ending right after
    a ZWJ then a space -- is 7, as if the trailing space were never
    there; append even one more character and the space *and* a
    narrow-to-wide upgrade both reappear in the total). A numeric budget
    derived from `cell_len(prefix)` in isolation cannot see that the
    join itself changes what the tail of `prefix` measures as; measuring
    the literal joined string every time sidesteps needing to know why.

    Also why this doesn't delegate to `rich.cells.chop_cells`/
    `split_graphemes`: those compute cell width via Unicode grapheme
    segmentation, which itself disagrees with `cell_len`'s own
    character-scan algorithm on ZWJ-bearing text (confirmed:
    ``cell_len("\\u200d\\u200dbcb3Z33")`` is 7, but `split_graphemes`
    -- what `chop_cells` trusts -- reports 6 for the identical string).
    Recomputing directly with `cell_len` keeps this self-consistent with
    every other width check in the file, all of which measure that way.

    Returns ``""`` when even a single character of ``text`` doesn't fit
    -- an honest "nothing fits" answer, not a forced minimum.
    `_cell_wrap_line` is responsible for its own forward-progress
    guarantee on an empty result; folding a forced minimum in here
    caused it to fire on ordinary partial-budget rows too (an
    already-full row plus a next chunk that simply has no room left is
    not the same situation as a wholly empty row that cannot fit
    *anything*, and conflating them broke the single-width parity
    guarantee -- caught by 100k-trial differential fuzzing against
    `textwrap.wrap` during development).

    Args:
        prefix: Text already committed to the row being built.
        text: Candidate string to take a fitting extension from.
        width: Total cell width the joined row must fit within.

    Returns:
        The longest fitting extension; ``""`` if nothing fits (including
        an empty ``text``).
    """
    from . import console_composer_bar as composer_module

    if not text:
        return text
    end = 0
    while (
        end < len(text) and composer_module.cell_len(prefix + text[: end + 1]) <= width
    ):
        end += 1
    return text[:end]


def composer_cell_wrap_line(cls: Any, line: str, width: int) -> list[str]:
    """Greedy word-wrap ``line`` by terminal cell width, not character count.

    Mirrors ``textwrap.wrap(line, width=width, break_long_words=True,
    break_on_hyphens=False, drop_whitespace=False, replace_whitespace=False)``
    chunk-for-chunk -- same tab expansion, same whitespace-run chunking,
    same greedy fill and long-word hard-break -- except every length
    check measures ``rich.cells.cell_len`` (terminal columns) instead of
    ``len()`` (Python characters). For single-width-only text the two
    measures are identical, so this produces byte-identical output to
    the ``textwrap`` call it replaces (confirmed with 100k-trial
    differential fuzzing against the exact `textwrap.wrap` call above);
    the difference only surfaces on double-width text (CJK, emoji),
    where a character-counted wrap can under-count how many terminal
    columns a row actually occupies and let it silently overflow the
    wrap width at paint time.

    Both the greedy fill and the long-word hard-break measure the
    *actual joined candidate row* directly with `cell_len` -- never a
    running sum or a pre-computed numeric budget checked against a
    piece's own isolated `cell_len` (see `_extend_fitting_cells`'s
    docstring for why cell width is not reliably additive across a
    join). The hard-break takes one fitting bite per call via
    `_extend_fitting_cells`, matching `TextWrapper._handle_long_word`'s
    one-bite-per-line contract, so a chunk that needs several lines to
    exhaust naturally gets the rest on subsequent passes of the outer
    loop below.

    Args:
        line: A single logical line (no newlines) to wrap.
        width: Wrap width in terminal cells.

    Returns:
        Wrapped rows for ``line``; always at least one entry (matching
        ``textwrap.wrap(...) or [""]`` at every call site).
    """
    from . import console_composer_bar as composer_module

    width = max(1, width)
    # `textwrap.wrap` always expands tabs before splitting into chunks
    # (`expand_tabs` defaults to True and is never overridden by any
    # caller in this module) regardless of `replace_whitespace`, which
    # only controls whether *other* whitespace becomes plain spaces.
    chunks = [
        chunk
        for chunk in composer_module._DRAFT_WORD_SPLIT_RE.split(
            line.expandtabs(composer_module._DRAFT_TAB_WIDTH)
        )
        if chunk
    ]
    chunks.reverse()
    lines: list[str] = []
    while chunks:
        current_text = ""
        while chunks:
            candidate = current_text + chunks[-1]
            if composer_module.cell_len(candidate) > width:
                break
            current_text = candidate
            chunks.pop()

        if chunks and composer_module.cell_len(chunks[-1]) > width:
            chunk = chunks[-1]
            piece = cls._extend_fitting_cells(current_text, chunk, width)
            if not piece and not current_text:
                # A wholly fresh row and even the FULL `width` budget
                # can't fit this chunk's first character (only possible
                # at width < 2 with double-width leading content --
                # unreachable in production, where every call site
                # floors width at 8, but no caller here relies on that,
                # so this stays correct if one ever passes a smaller
                # width). Force exactly one character so the outer loop
                # is guaranteed to make progress; the CSS
                # `text_overflow: clip` guard crops the resulting
                # overflow at paint time.
                piece = chunk[:1]
            if piece:
                current_text += piece
                chunks[-1] = chunk[len(piece) :]
                if not chunks[-1]:
                    chunks.pop()
            # else: this row already holds prior content and has no
            # room left for even one more character of the next
            # (individually too-wide) chunk -- leave it untouched for a
            # fresh, full-budget row on the next pass of the outer loop,
            # exactly mirroring `TextWrapper._handle_long_word`'s own
            # `space_left == 0` behavior. Forcing a character onto an
            # already-full row here would silently make that row wider
            # than `width` for perfectly ordinary text, not just the
            # width-1 double-width case above (caught by fuzzing during
            # development).

        # `current_text` is always non-empty here: either the greedy
        # fill above consumed at least one chunk, or the hard-break
        # branch's forced-progress fallback fired (guaranteed whenever
        # `current_text` started this iteration empty and nothing else
        # fit, per the note above). The only way to reach here with an
        # unconsumed too-wide `chunks[-1]` and an empty `current_text`
        # would require that fallback to not fire on an empty row, which
        # cannot happen.
        lines.append(current_text)
    return lines or [""]


def composer_wrap_draft_lines(cls: Any, text: str, width: int) -> list[str]:
    """Return wrapped draft lines for the visible bounded composer."""
    width = max(8, width)
    source_lines = text.splitlines() or [text]
    wrapped_lines: list[str] = []
    for line in source_lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(cls._cell_wrap_line(line, width))
    return wrapped_lines or [""]


def composer_wrap_draft_line_slices(cls: Any, text: str, width: int) -> list[Any]:
    """Return wrapped draft lines with source offsets for style remapping."""
    from . import console_composer_bar as composer_module

    width = max(8, width)
    source_lines = text.splitlines(keepends=True) or [text]
    wrapped_lines: list[composer_module._DraftLineSlice] = []
    source_offset = 0
    for raw_line in source_lines:
        if raw_line.endswith("\r\n"):
            line = raw_line[:-2]
            separator_length = 2
        elif raw_line.endswith(("\n", "\r")):
            line = raw_line[:-1]
            separator_length = 1
        else:
            line = raw_line
            separator_length = 0

        if not line:
            wrapped_lines.append(
                composer_module._DraftLineSlice("", source_offset, source_offset)
            )
            source_offset += separator_length
            continue

        line_offset = 0
        wrapped_segments = cls._cell_wrap_line(line, width)
        for wrapped_segment in wrapped_segments:
            start = source_offset + line_offset
            end = start + len(wrapped_segment)
            wrapped_lines.append(
                composer_module._DraftLineSlice(wrapped_segment, start, end)
            )
            line_offset += len(wrapped_segment)
        source_offset += len(line) + separator_length

    return wrapped_lines or [composer_module._DraftLineSlice("", 0, 0)]
