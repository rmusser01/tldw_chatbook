import pytest

from tldw_chatbook.Chunking.engine.strategies.sentences import (
    SentenceChunkingStrategy,
)
from tldw_chatbook.Chunking.engine.strategies.structure_aware import (
    StructureAwareChunkingStrategy,
)
from tldw_chatbook.Chunking.engine.strategies.words import (
    WordChunkingStrategy,
)


def test_thai_sentence_fallback_does_not_split_on_spaces():
    """Thai fallback segmentation should not split sentences on spaces."""
    s = SentenceChunkingStrategy(language="th")

    # Force fallback path regardless of local environment
    s.pythainlp_available = False
    s._th_sent_tokenize = None

    # Include spaces inside sentences; fallback delimiters are punctuation only
    text = "ประโยคหนึ่ง ทดสอบ! ประโยคสอง ทดสอบ? ประโยคสาม"
    sentences = s._split_sentences(text)

    assert sentences, "Expected sentences from Thai fallback splitter"
    # Expect 3 sentences split on ! and ? only
    assert len(sentences) == 3
    # Ensure boundaries are on punctuation, not spaces
    assert sentences[0].endswith("!")
    assert sentences[1].endswith("?")
    assert not sentences[2].endswith("!") and not sentences[2].endswith("?")


def test_thai_sentence_with_pythainlp_if_available():
    """If PyThaiNLP is available, ensure it integrates and returns non-empty sentences."""
    s = SentenceChunkingStrategy(language="th")
    text = "สวัสดีครับนี่คือประโยคแรก!นี่คือประโยคที่สอง?และนี่คือประโยคที่สาม"

    if s.pythainlp_available and callable(getattr(s, "_th_sent_tokenize", None)):
        sents = s._split_sentences(text)
        assert sents, "PyThaiNLP path should yield sentences"
        # Still should segment on punctuation boundaries for this simple sample
        assert len(sents) >= 2
    else:
        pytest.skip("PyThaiNLP not available")


def test_markdown_table_parsing_preserves_empty_cells():
    """Markdown table parser should preserve empty cells and column counts."""
    sa = StructureAwareChunkingStrategy()
    table_md = (
        "| Col1 | Col2 | Col3 |\n" "|------|------|------|\n" "|  a   |      |  c   |\n" "|      |  b   |      |\n"
    )

    table = sa._parse_markdown_table(table_md)
    assert table is not None
    assert table.headers == ["Col1", "Col2", "Col3"]
    assert len(table.rows) == 2
    # Row 1 should preserve an empty Col2
    assert len(table.rows[0]) == 3
    assert table.rows[0][0] == "a"
    assert table.rows[0][1] == ""
    assert table.rows[0][2] == "c"
    # Row 2 should preserve empty Col1 and Col3
    assert table.rows[1] == ["", "b", ""]


def test_word_spans_monotonic_on_repeated_tokens():
    """Span mapping should be monotonic and finish quickly on repeated tokens."""
    ws = WordChunkingStrategy()
    # Build a moderately large repeated-token payload
    token = "x" * 50
    text = (token + " ") * 500  # 500 repeated tokens with spaces

    tokens = ws._tokenize_text(text)
    assert tokens, "Tokenization produced no tokens"

    toks, spans = ws._tokenize_with_spans(text)
    assert len(toks) == len(spans)

    last_end = -1
    for start, end in spans:
        assert start <= end <= len(text)
        assert start >= last_end
        last_end = end
