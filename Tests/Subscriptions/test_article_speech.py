"""Saved speech uses the visible complete excerpt, under byte and chunk limits."""

from hashlib import sha256
from importlib import import_module, util

import pytest

from tldw_chatbook.Subscriptions.article_listening_models import MAX_TEXT_BYTES


def api():
    name = "tldw_chatbook.Subscriptions.article_speech"
    assert util.find_spec(name), "missing bounded saved-article speech preparation"
    return import_module(name)


@pytest.mark.parametrize(
    "fmt, body",
    [
        (
            "html",
            '<p>Start <a href="https://private">label</a><img alt="private"><script>private</script><style>private</style><pre>code end</pre>',
        ),
        (
            "markdown",
            "Start [label](https://private) ![private](https://private)\n\n<script>private</script>\n\n`code end`",
        ),
        ("text", "Start label\x00 code end"),
    ],
)
def test_speaks_visible_words_only(fmt, body):
    speech = api().prepare_article_speech(body, content_format=fmt, is_excerpt=True)
    assert speech.text.split() == ["Start", "label", "code", "end"]
    assert speech.sha256 == sha256(speech.text.encode()).hexdigest()
    assert speech.is_excerpt and speech.converter


def test_raw_and_prepared_utf8_boundaries_and_empty():
    mod = api()
    limit = "é" * (MAX_TEXT_BYTES // 2)
    assert mod.prepare_article_speech(limit, content_format="text").text == limit
    for body in [limit + "x", "<!--" + limit + "-->"]:
        with pytest.raises(ValueError, match="limit"):
            mod.prepare_article_speech(body, content_format="text")
    with pytest.raises(ValueError, match="empty"):
        mod.prepare_article_speech("<script>private</script>", content_format="html")
    assert mod.prepare_saved_text("<img alt='private'>", None, None) == ""


def test_speech_chunks_preserve_the_entire_tail_and_bound_provider_calls():
    mod = api()
    prose = "Distinctive phrase. " * 500 + "Final sentence."
    pieces = mod.article_chunks(prose)
    assert " ".join(" ".join(pieces).split()) == " ".join(prose.split())
    assert all(len(piece) <= 1800 for piece in pieces)
    assert len(pieces) > 1
    with pytest.raises(ValueError, match="chunk_limit"):
        mod.article_chunks("x" * (128 * 1800 + 1))


def test_long_dense_prose_preserves_punctuation_and_inline_code():
    prose = "First,,second,,,third; " * 420
    pieces = api().article_chunks(prose)
    assert "".join("".join(pieces).split()) == "".join(prose.split())


def test_html_output_limit_and_markdown_hidden_inline_html():
    mod = api()
    with pytest.raises(ValueError, match="limit"):
        mod.prepare_article_speech(
            "&nLt;" * (MAX_TEXT_BYTES // 5), content_format="html"
        )
    speech = mod.prepare_article_speech(
        "Before <script>secret</script> `<tag>` after", content_format="markdown"
    )
    assert speech.text.split() == ["Before", "<tag>", "after"]
    assert not speech.is_excerpt


def test_speech_list_requires_visible_prose():
    prepare = api().prepare_saved_text
    assert (
        prepare(
            "<ul><li><img alt='caption'></li><li><script>hidden</script></li></ul>",
            None,
            None,
        )
        == ""
    )
    assert prepare("<ol><li>One</li><li>Two</li></ol>", None, None).split() == [
        "One",
        "Two",
    ]


def test_chunk_preparation_uses_bounded_windows(monkeypatch):
    from tldw_chatbook.Subscriptions import briefing_audio

    def unbounded_sentence_matcher(*args):
        pytest.fail("article preparation entered the unbounded sentence matcher")

    monkeypatch.setattr(briefing_audio, "_split_turn_text", unbounded_sentence_matcher)
    text = (("!?." * 32 + "Word. ") * 40) + "Last."
    chunks = api().article_chunks(text)
    assert "".join("".join(chunks).split()) == "".join(text.split())
    assert max(map(len, chunks)) <= 1800


def test_whitespace_between_full_article_windows_spends_no_empty_request():
    first = "x" * 1800
    assert api().article_chunks(first + " " * 2000 + "The end.") == [first, "The end."]
