from tldw_chatbook.Chunking.engine.strategies.sentences import SentenceChunkingStrategy


def _inputs():

    # Mixed whitespace and punctuation scenarios
    yield "Hello world!  This is a test.\nNew line.  Another sentence?"
    yield "  Leading spaces. Middle   spaces.\n\nParagraph break!  \nTail.  "
    yield "No punctuation here but newlines\nwill be treated as boundaries\n when needed"
    yield "One. Two! Three? Four.  Five."


def test_split_sentences_with_spans_round_trip_like():

    strat = SentenceChunkingStrategy(language="en")
    for text in _inputs():
        spans = strat._split_sentences_with_spans(text)
        # Monotonic non-overlapping and within bounds
        prev_end = 0
        for sent, start, end in spans:
            assert 0 <= start <= end <= len(text)
            assert start >= prev_end
            prev_end = end
            # Sentences produced should match source slices modulo leading/trailing whitespace
            assert sent.strip() == text[start:end].strip()
