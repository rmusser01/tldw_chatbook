"""Plain spoken text derived from a validated, completed Console reply."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from markdown_it import MarkdownIt
from markdown_it.token import Token

_HTML_BREAK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _HTMLSpeechText(HTMLParser):
    """Collect visible words and structural breaks without executing HTML."""

    def __init__(self) -> None:
        """Initialize the collector with visible content enabled."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Keep structural breaks and suppress script/style bodies.

        Args:
            tag: Lowercase HTML tag name.
            attrs: Parsed attributes, which do not contribute spoken text.
        """
        if self.hidden_tag:
            return
        if tag in {"script", "style"}:
            self.hidden_tag = tag
        elif tag in _HTML_BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Resume visible content and separate closing block elements.

        Args:
            tag: Lowercase HTML tag name.
        """
        if self.hidden_tag:
            if tag == self.hidden_tag:
                self.hidden_tag = None
        elif tag in _HTML_BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect visible text, treating source line wraps as whitespace.

        Args:
            data: Decoded HTML data or already parsed Markdown literal text.
        """
        if not self.hidden_tag:
            self.parts.append(data.replace("\n", " ").replace("\r", " "))

    def take_text(self) -> str:
        """Drain collected words while retaining document-wide HTML state."""
        parts, self.parts = self.parts, []
        return _join_blocks("".join(parts).split("\n"))


def _pause(text: str) -> str:
    """Separate blocks even when the author omitted sentence punctuation."""
    ending = text.rstrip("\"'”’)]}")
    return text if not ending or ending[-1] in ".!?:;。！？…" else text + "."


def _join_blocks(blocks: list[str]) -> str:
    blocks = [" ".join(block.split()) for block in blocks if block.strip()]
    return " ".join([*(_pause(block) for block in blocks[:-1]), *blocks[-1:]])


def _inline_text(
    tokens: list[Token],
    html: _HTMLSpeechText,
    task_state: str | None = None,
    *,
    image_alt: bool = False,
) -> str:
    for index, token in enumerate(tokens):
        if token.type in {"text", "code_inline"}:
            # Parsed literals must not be reinterpreted as HTML tags/entities.
            text = token.content
            if index == 0 and task_state:
                text = f"{task_state}: {text[3:].lstrip()}"
            html.handle_data(text)
        elif token.type == "html_inline" and not image_alt:
            html.feed(token.content)
        elif token.type == "image":
            # Alt tags cannot create hidden elements inside the caption.
            html.handle_data(
                _inline_text(token.children or [], _HTMLSpeechText(), image_alt=True)
            )
        elif token.type == "softbreak":
            html.handle_data(" ")
        elif token.type == "hardbreak" and not html.hidden_tag:
            html.parts.append("\n")
    return html.take_text()


def console_markdown_to_speech(content: str) -> str:
    """Render Markdown words and structure for speech without rewriting prose.

    Args:
        content: Original completed reply, already admitted by the caller's
            message validation and raw text length limit.

    Returns:
        Plain speech text with code-block omission announcements, checklist
        states, table column labels and visible HTML words. Empty text means
        there is nothing to speak. Callers must also apply their speech
        length limit to the result.
    """
    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    blocks: list[str] = []
    headers: list[str] = []
    row: list[str] = []
    in_table = False
    in_header = False
    has_table_rows = False
    html = _HTMLSpeechText()
    tokens = parser.parse(content)
    for index, token in enumerate(tokens):
        if token.type in {"fence", "code_block"}:
            if not html.hidden_tag:
                blocks.append("Code block omitted.")
        elif token.type == "html_block":
            html.feed(token.content)
            blocks.append(html.take_text())
        elif token.type == "table_open":
            in_table = True
            headers = []
            has_table_rows = False
        elif token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            row = []
        elif token.type == "inline":
            task_state = None
            if (
                index >= 2
                and tokens[index - 2].type == "list_item_open"
                and tokens[index - 1].type == "paragraph_open"
                and re.match(r"\[[ xX]\]\s+", token.content)
                and token.children
                and token.children[0].type == "text"
                and token.children[0].content.startswith(token.content[:3])
            ):
                task_state = "Unchecked" if token.content[1] == " " else "Checked"
            text = _inline_text(token.children or [], html, task_state)
            if in_table:
                row.append(text)
            else:
                blocks.append(text)
        elif token.type == "tr_close":
            if in_header:
                headers = row
            else:
                has_table_rows = True
                cells = [
                    f"{header}: {cell}" if header and cell else header or cell
                    for header, cell in zip(headers, row)
                ]
                text = "; ".join(cell for cell in cells if cell)
                if text:
                    blocks.append(_pause(text))
        elif token.type == "table_close":
            if not has_table_rows:
                blocks.append("; ".join(header for header in headers if header))
            in_table = False
    html.close()
    blocks.append(html.take_text())
    return _join_blocks(blocks)
