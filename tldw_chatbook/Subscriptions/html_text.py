"""Turn remote feed bodies into readable, inert plain text.

TASK-2307. Feeds hand us HTML. `item_persist` only knows the formats
``text``/``markdown``/``diff`` (see `_CONTENT_PAIRINGS`), so an RSS entry
whose ``<description>`` is a block of HTML is stored as ``text`` and the
reader showed it verbatim: `<p>Article URL: <a href="https://...">...</a></p>`
literally on screen, one long unreadable line.

**Why this lives here rather than at ingest.** The stored body is the honest
capture of what the feed served, and every item persisted before this task
already holds HTML -- a converter at ingest would fix nothing the user is
currently looking at. So the conversion is a RENDER-time step, the last one
before the reader appends text to a `rich.text.Text`.

**Why stdlib rather than a sanitizer.** Nothing here tries to make HTML safe
to render; it produces text and throws the markup away. `html.parser` is
lenient by construction (malformed nesting yields odd spacing, never an
exception), needs no optional dependency, and -- crucially -- there is no
"allowed tag" list that can fail open on the case it did not anticipate. The
two existing HTML paths in this package were both considered and rejected for
this job: `ContentExtractor.extract_text_from_html` (`monitoring_engine.py`)
collapses an entire document onto ONE line and drops every `href`, and
`html2text` is an optional `[ebook]` extra, absent on a default install.

**Inertness.** The output of every function here is a plain `str` with no
control characters, handed to `Text.append`, which never interprets Rich
markup. A body of ``[bold red]x[/]`` renders as those literal characters --
see `content_pane.render_article`, which states the rule for that whole path.
`strip_control_characters` closes the one hole `Text.append` does NOT close:
Rich writes a segment's text to the terminal verbatim, so a raw ESC in a feed
body would reach the emulator as an escape sequence (an OSC-8 hyperlink whose
label lies about its destination, or a cursor-positioning sequence that
corrupts the frame).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

__all__ = [
    "html_to_display_text",
    "looks_like_html",
    "readable_body_text",
    "strip_control_characters",
]

#: Elements whose *content* is never prose. Dropped wholesale rather than
#: having their text extracted -- a `<script>` body rendered as text is noise
#: at best and an attacker's payload verbatim at worst.
_DROP_CONTENT: frozenset[str] = frozenset(
    {"script", "style", "head", "title", "noscript", "template", "svg", "math"}
)

#: Elements that end the current line. `br` and `hr` are void, so they are
#: handled on the start tag only; the rest break before and after.
_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
        "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr",
        "ul",
    }
)

#: Block tags that also want a BLANK line after them, because they separate
#: units of thought rather than merely rows of one.
_PARAGRAPH_TAGS: frozenset[str] = frozenset(
    {"p", "div", "blockquote", "pre", "section", "article", "h1", "h2", "h3",
     "h4", "h5", "h6", "table", "ul", "ol", "figure", "header", "footer"}
)

#: Control characters that must never reach the terminal. C0 minus tab and
#: newline (a feed body legitimately contains both), DEL, and the C1 range --
#: an 8-bit CSI/OSC introducer is just as capable as `ESC [` / `ESC ]`.
#:
#: Batch-4 review, M1: CR (0x0D) is included -- it is neither tab nor
#: newline, so "C0 minus tab and newline" always meant to cover it, but the
#: range below used to jump `\x0c` -> `\x0e-\x1f` and skip it by one code
#: point, letting a bare CR survive (a weaker primitive than ESC/OSC -- it
#: can only overwrite characters earlier on the same terminal line -- but a
#: real mismatch between what this module documents and what it did, on a
#: module whose whole purpose is inertness). Deleting CR rather than
#: replacing it with anything is what keeps a `\r\n` feed body from becoming
#: a doubled blank line: the `\n` in that pair survives untouched as the one
#: line break, exactly as if the body had used `\n` alone.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

#: A tag-shaped or entity-shaped fragment. Requires a letter (or `/`) directly
#: after the `<` AND a closing `>`, so ordinary prose containing "a < b" is not
#: mistaken for markup and mangled.
_HTML_SHAPED = re.compile(
    r"<\s*/?[a-zA-Z][^<>]*>|<!--|&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#\d{1,7}|#[xX][0-9a-fA-F]{1,6});"
)

#: Runs of spaces/tabs, collapsed inside a line. Newlines are structure here
#: and are handled separately.
_INLINE_SPACE = re.compile(r"[ \t]+")

#: Three or more newlines, i.e. more than one blank line.
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def strip_control_characters(value: Any) -> str:
    """Remove terminal control characters from remote-derived text.

    `Text.append` protects against Rich *markup*; it does nothing about
    control bytes, which Rich writes through to the terminal untouched. A
    feed body carrying ``\\x1b]8;;http://evil\\x07Anthropic docs\\x1b]8;;\\x07``
    would therefore paint a real, clickable hyperlink whose visible label is
    attacker-chosen -- the exact phishing shape `content_pane`'s
    `_MARKDOWN_HYPERLINKS` note rejects for markdown links, arriving by a
    different door.

    Args:
        value: Any value; coerced with `str()`. `None` becomes `""`.

    Returns:
        The text with every C0 control except tab and newline, DEL, and every
        C1 control removed. Nothing is escaped or substituted -- the
        characters simply do not survive.
    """
    if value is None:
        return ""
    return _CONTROL_CHARS.sub("", str(value))


def looks_like_html(value: Any) -> bool:
    """Whether this body should go through the HTML converter.

    Deliberately conservative. A plain-text feed body that happens to contain
    a mathematical ``<`` must be rendered untouched, so a bare ``<`` is not
    enough: the text has to carry a complete tag-shaped fragment, an HTML
    comment opener, or a named/numeric character reference.

    Args:
        value: Any value; coerced with `str()`.

    Returns:
        `True` when `html_to_display_text` should be applied.
    """
    if not value:
        return False
    return _HTML_SHAPED.search(str(value)) is not None


class _DisplayTextExtractor(HTMLParser):
    """Walk an HTML fragment, emitting the text a reader wants to see.

    Not a general-purpose converter and not trying to be: it exists to make a
    feed entry legible in a nine-row terminal pane. Structure is reduced to
    line breaks, list items get a bullet, and a link keeps BOTH halves -- the
    label and, when they differ, the address -- because a terminal reader who
    cannot see the destination cannot judge it.
    """

    def __init__(self) -> None:
        # `convert_charrefs=True` is the default and is what unescapes `&amp;`
        # and friends, exactly once, inside `handle_data`. Doing it ourselves
        # afterwards would double-unescape (`&amp;lt;` -> `<`).
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        #: Depth inside a `_DROP_CONTENT` element. An int, not a bool: nested
        #: `<svg><style>` must not be re-enabled by the inner element's end tag.
        self._drop_depth = 0
        #: One entry per open `<a>`: the href it carried, and the index in
        #: `_parts` where its text began, so `handle_endtag` can tell whether
        #: the label already spells the URL out.
        self._anchors: list[tuple[str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _DROP_CONTENT:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag == "a":
            href = ""
            for name, value in attrs:
                if name.lower() == "href" and value:
                    href = value.strip()
                    break
            self._anchors.append((href, len(self._parts)))
            return
        if tag == "img":
            # An image cannot render here, but its alt text is often the only
            # caption a feed carries. Named so it cannot be mistaken for prose.
            alt = ""
            for name, value in attrs:
                if name.lower() == "alt" and value:
                    alt = value.strip()
                    break
            if alt:
                self._parts.append(f"[image: {alt}]")
            return
        if tag == "li":
            self._parts.append("\n")
            self._parts.append("- ")
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n\n" if tag in _PARAGRAPH_TAGS else "\n")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # `<br/>` arrives here, not at `handle_starttag`. Void elements have no
        # end tag, so this must not touch `_drop_depth`.
        tag = tag.lower()
        if self._drop_depth or tag in _DROP_CONTENT:
            return
        if tag == "img":
            self.handle_starttag(tag, attrs)
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_CONTENT:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag == "a":
            if not self._anchors:
                # A stray `</a>`; malformed input must not raise.
                return
            href, start = self._anchors.pop()
            label = "".join(self._parts[start:]).strip()
            if href and href not in label:
                # The address, as ordinary visible text -- never an OSC-8
                # hyperlink. Same decision, same reason, as
                # `content_pane._MARKDOWN_HYPERLINKS`: a label the feed chose
                # over a destination the reader cannot see is a phishing
                # anchor, and a terminal reader who can read the URL can judge
                # it. `href not in label` covers the common feed shape where
                # the link text IS the URL, which would otherwise print twice.
                self._parts.append(f" ({href})")
            return
        if tag in _PARAGRAPH_TAGS:
            self._parts.append("\n\n")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        """The extracted text, with whitespace normalized for a narrow pane."""
        raw = "".join(self._parts)
        lines = [_INLINE_SPACE.sub(" ", line).strip() for line in raw.split("\n")]
        return _EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def html_to_display_text(value: Any) -> str:
    """Convert an HTML fragment to readable plain text.

    Args:
        value: The raw body; coerced with `str()`.

    Returns:
        Plain text with block structure preserved as line breaks, list items
        bulleted, link destinations kept as visible text, and `script`/`style`
        content dropped. Control characters are removed last, so nothing this
        function produces can be interpreted by a terminal.

        Never raises: `html.parser` tolerates malformed markup, and the one
        thing that can still raise (`AssertionError` from a badly-formed CDATA
        section in some CPython versions) is caught and degrades to the
        control-stripped original -- an exception escaping the reader's
        `compose()` would exit the whole application.
    """
    if value is None:
        return ""
    source = str(value)
    parser = _DisplayTextExtractor()
    try:
        parser.feed(source)
        parser.close()
    except Exception:  # pragma: no cover - defensive, see the docstring
        return strip_control_characters(source)
    return strip_control_characters(parser.text())


def readable_body_text(value: Any) -> str:
    """The last step before a remote body reaches the reader.

    Args:
        value: The stored item body; coerced with `str()`.

    Returns:
        `html_to_display_text` for a body that carries markup, otherwise the
        body with only its control characters stripped -- a plain-text feed
        must survive this function byte-for-byte apart from characters that
        cannot legally be displayed.
    """
    if value is None:
        return ""
    if looks_like_html(value):
        return html_to_display_text(value)
    return strip_control_characters(value)
