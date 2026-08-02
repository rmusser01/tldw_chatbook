"""Tests for `tldw_chatbook.Chat.reply_sentence_sequencer.SentenceSequencer`.

The sequencer is a pure, headless module (no Textual, no TTS imports, no
wall-clock) that turns streamed reply text into gated, sequential
speakable-sentence utterances. `speak`/`stop_speech` are injected callables
so every test below drives the sequencer with plain lists instead of a real
speech path.
"""

from tldw_chatbook.Chat.reply_sentence_sequencer import (
    MAX_UTTERANCE_LENGTH,
    SentenceSequencer,
)


# ---------------------------------------------------------------------------
# Verbatim load-bearing tests (task-2 brief, Step 1)
# ---------------------------------------------------------------------------


def test_sentences_emit_one_at_a_time_gated_on_completion():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("First sentence. Second sentence. ")
    assert spoken == ["First sentence."]
    seq.utterance_finished(ok=True)
    assert spoken == ["First sentence.", "Second sentence."]


def test_abbreviations_and_decimals_do_not_chop():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Dr. Smith measured 3.14 units. Then left. ")
    assert spoken == ["Dr. Smith measured 3.14 units."]


def test_code_fences_are_skipped_entirely():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Here you go:\n```python\nx = 1. Yes.\n```\nDone now. ")
    seq.reply_completed()
    joined = " ".join(spoken)
    assert "x = 1" not in joined and "Done now." in joined


def test_markdown_is_stripped_links_keep_text():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("See **the [docs](https://x.y)** now. ")
    assert spoken == ["See the docs now."]


def test_flush_clears_queue_and_stops_inflight():
    spoken, stops = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: stops.append(1))
    seq.feed("A one. A two. A three. ")
    seq.flush()
    assert stops == [1]
    seq.utterance_finished(ok=False)   # late completion of the stopped utterance
    assert spoken == ["A one."]        # nothing further spoken


def test_reply_completed_flushes_final_partial_and_drains():
    spoken, drained = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)
    seq.feed("Only a fragment with no terminator")
    seq.reply_completed()
    assert spoken == ["Only a fragment with no terminator"]
    seq.utterance_finished(ok=True)
    assert drained == [1]


def test_zero_speakable_reply_drains_immediately():
    drained = []
    seq = SentenceSequencer(speak=lambda t: None, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)
    seq.feed("```\ncode only\n```")
    seq.reply_completed()
    assert drained == [1]


def test_failed_utterance_skips_and_continues():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("A. B. ")
    seq.utterance_finished(ok=False)
    assert spoken == ["A.", "B."]


def test_gate_holds_across_multiple_feed_calls_without_completion():
    """The one-at-a-time gate must survive across separate `feed()` calls,
    not just within a single call: a second `feed()` that arrives before the
    first utterance's `utterance_finished()` must not advance the queue.
    (This is the scenario that actually falsifies a gate mutation -- a
    single-`feed()` test can pass even with the gate deleted, since
    `_maybe_dispatch` is only invoked once per external event either way.)
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("First one. Second one. ")
    assert spoken == ["First one."]
    seq.feed("Third one. ")
    assert spoken == ["First one."]  # still gated; nothing new dispatched
    seq.utterance_finished(ok=True)
    assert spoken == ["First one.", "Second one."]
    seq.utterance_finished(ok=True)
    assert spoken == ["First one.", "Second one.", "Third one."]


# ---------------------------------------------------------------------------
# Additional cases named by the brief
# ---------------------------------------------------------------------------


def test_max_length_force_splits_near_a_whitespace_boundary():
    """A 600+-char terminator-free stream must not grow the buffer forever.

    Each forced chunk lands at or before MAX_UTTERANCE_LENGTH, cuts on a
    whitespace boundary (never mid-word), and no text is lost.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    long_text = "word " * 130  # 650 chars, no '.', '!', or '?' anywhere
    assert len(long_text) > 600

    seq.feed(long_text)
    seq.reply_completed()
    # Drain the whole queue by simulating completion signals.
    while True:
        before = len(spoken)
        seq.utterance_finished(ok=True)
        if len(spoken) == before:
            break

    assert len(spoken) > 1, "650 terminator-free chars must force more than one split"
    for chunk in spoken:
        assert len(chunk) <= MAX_UTTERANCE_LENGTH
        # A mid-word cut would glue the trailing partial word onto the next
        # chunk's first word with no space between them.
        assert not chunk.endswith("wor"), "force-split must not cut mid-word"
    reconstructed = " ".join(spoken)
    assert reconstructed.count("word") == 130


def test_delta_split_across_chunk_boundaries_still_forms_one_sentence():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Half a sen")
    assert spoken == []  # no terminator yet; nothing to speak
    seq.feed("tence. ")
    assert spoken == ["Half a sentence."]


def test_ellipsis_is_not_a_boundary_mid_thought():
    """Pinned rule: a run of 2+ '.' characters ("...") is never a sentence
    boundary by itself -- only a single '.', '!', or '?' (subject to the
    abbreviation/decimal guards) can end a sentence. This keeps a
    trailing-off ellipsis from fragmenting speech mid-thought.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Wait for it... Then go. ")
    assert spoken == ["Wait for it... Then go."]


# ---------------------------------------------------------------------------
# Bonus: fence delimiter split across feed() calls (explicitly flagged by the
# task instructions -- "a fence toggle line ``` may arrive SPLIT across
# deltas — handle the line-buffer accordingly").
# ---------------------------------------------------------------------------


def test_fence_delimiter_split_across_deltas_still_skips_the_fence():
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Before.\n``")       # opening fence marker split mid-backtick-run
    assert spoken == ["Before."]
    seq.feed("`python\nx = 1. Yes.\n``")  # closing fence marker split too
    seq.feed("`\nAfter now. ")
    # "Before." is still in flight (one-at-a-time discipline); advance past it
    # so the queued "After now." actually dispatches.
    seq.utterance_finished(ok=True)
    joined = " ".join(spoken)
    assert "x = 1" not in joined
    assert "After now." in joined


# ---------------------------------------------------------------------------
# Once-per-reply `on_drained` guarantee: not before completion, not twice.
# ---------------------------------------------------------------------------


def test_on_drained_fires_exactly_once_not_before_completion_and_not_twice():
    spoken, drained = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)

    seq.feed("One sentence here. ")
    seq.utterance_finished(ok=True)
    # Transiently drained (queue empty, nothing in flight) but the reply
    # hasn't been marked complete yet -- must not fire.
    assert drained == []

    seq.reply_completed()  # nothing left to flush; already drained
    assert drained == [1]

    seq.reply_completed()  # defensive: calling again must not double-fire
    assert drained == [1]
