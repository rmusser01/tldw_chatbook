"""Bounded, inert text preparation for saved Watchlists articles."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from html import escape

from markdown_it import MarkdownIt

from tldw_chatbook.Subscriptions.article_listening_models import MAX_TEXT_BYTES
from tldw_chatbook.Subscriptions.html_text import (
    _DisplayTextExtractor,
    looks_like_html,
    strip_control_characters,
)

CONVERTER = "article-speech-v1"
MAX_CHUNKS = 128


class ArticleSpeechError(ValueError):
    """Bounded preparation failure suitable for the listening service."""


@dataclass(frozen=True, slots=True)
class PreparedArticleSpeech:
    text: str = field(repr=False)
    sha256: str
    converter: str
    is_excerpt: bool


def _bounded_bytes(text: str) -> bytes:
    if type(text) is not str or len(text) > MAX_TEXT_BYTES:
        raise ArticleSpeechError("article_text_limit")
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        raise ArticleSpeechError("invalid_article_text") from None
    if len(encoded) > MAX_TEXT_BYTES:
        raise ArticleSpeechError("article_text_limit")
    return encoded


def _markdown_text(body: str, parser: _DisplayTextExtractor) -> None:
    # Feed raw HTML through the same parser state across tokens. Text/code tokens
    # have already been decoded by CommonMark and must remain literal text.
    for block in MarkdownIt("commonmark").parse(body):
        for token in block.children if block.type == "inline" else (block,):
            if token.type in {"html_inline", "html_block"}:
                parser.feed(token.content)
            elif token.type in {"text", "code_inline", "code_block", "fence"}:
                parser.feed(escape(token.content, quote=False))
            elif token.block or token.type in {"softbreak", "hardbreak"}:
                parser.feed("\n")


def prepare_article_speech(
    body: str, *, content_format: str, is_excerpt: bool = False
) -> PreparedArticleSpeech:
    """Extract visible saved prose. Limits apply both before and after conversion."""
    _bounded_bytes(body)
    if (
        content_format not in ("html", "text", "markdown")
        or type(is_excerpt) is not bool
    ):
        raise ArticleSpeechError("unsupported_article_format")
    if content_format == "text" and not looks_like_html(body):
        text = body
    else:
        parser = _DisplayTextExtractor(speech=True)
        try:
            if content_format == "markdown":
                _markdown_text(body, parser)
            else:
                parser.feed(body)
            parser.close()
        except (AssertionError, ValueError, RecursionError):
            raise ArticleSpeechError("invalid_article_markup") from None
        text = parser.text()
    text = strip_control_characters(text).strip()
    if not text:
        raise ArticleSpeechError("empty_article")
    return PreparedArticleSpeech(
        text, sha256(_bounded_bytes(text)).hexdigest(), CONVERTER, is_excerpt
    )


def article_chunks(text: str) -> list[str]:
    """Split the complete snapshot into the bounded provider submission sequence."""
    from tldw_chatbook.Subscriptions.briefing_audio import (
        _hard_split_piece,
        _split_turn_text,
    )

    _bounded_bytes(text)
    if not text.strip():
        raise ArticleSpeechError("empty_article")
    chunks = _split_turn_text(text)
    # The sentence splitter normalizes repeated punctuation in long clauses.
    # Article reading must retain every non-whitespace character of the source.
    if "".join("".join(chunks).split()) != "".join(text.split()):
        chunks = _hard_split_piece(text)
    if len(chunks) > MAX_CHUNKS:
        raise ArticleSpeechError("article_chunk_limit")
    return chunks


def prepare_saved_text(body: str, kind: str | None, fmt: str | None) -> str:
    """Store preparation callback. Empty prose becomes an unavailable placeholder."""
    try:
        speech = prepare_article_speech(body, content_format=fmt or "text")
    except ValueError as error:
        if error.args == ("empty_article",):
            return ""
        raise
    article_chunks(speech.text)
    return speech.text
