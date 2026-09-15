"""Plain spoken text derived from a validated, completed Console reply."""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token


def _pause(text: str) -> str:
    """Separate blocks even when the author omitted sentence punctuation."""
    ending = text.rstrip("\"'”’)]}")
    return text if not ending or ending[-1] in ".!?:;。！？…" else text + "."


def _join_blocks(blocks: list[str]) -> str:
    blocks = [" ".join(block.split()) for block in blocks if block.strip()]
    return " ".join([*(_pause(block) for block in blocks[:-1]), *blocks[-1:]])


def _inline_text(tokens: list[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif token.type == "image":
            parts.append(_inline_text(token.children or []))
        elif token.type == "softbreak":
            parts.append(" ")
        elif token.type == "hardbreak":
            parts = [_pause("".join(parts).rstrip()), " "]
    return "".join(parts).strip()


def console_markdown_to_speech(content: str) -> str:
    """Render Markdown words and structure for speech without rewriting prose.

    Args:
        content: Original completed reply, already admitted by the caller's
            message validation and raw text length limit.

    Returns:
        Plain speech text, with code-block omission announcements and table
        column labels. Empty text means there is nothing to speak. Callers
        must also apply their speech length limit to the result.
    """
    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    blocks: list[str] = []
    headers: list[str] = []
    row: list[str] = []
    in_table = False
    in_header = False
    has_table_rows = False
    for token in parser.parse(content):
        if token.type in {"fence", "code_block"}:
            blocks.append("Code block omitted.")
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
            text = _inline_text(token.children or [])
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
    return _join_blocks(blocks)
