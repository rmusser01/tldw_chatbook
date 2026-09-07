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
    seq.utterance_finished(ok=False)  # late completion of the stopped utterance
    assert spoken == ["A one."]  # nothing further spoken


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
    seq.feed("Before.\n``")  # opening fence marker split mid-backtick-run
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


# ---------------------------------------------------------------------------
# task-2-review.md fix wave: F1 (flush latch), F2 (utterance identity),
# F3 (begin_reply lifecycle), F4 (fence line-position tracking),
# F5 (ordered-list guard), F6 (discriminating delta-split test),
# F7 (abbreviation set), F8 (flush drained reconciliation),
# F9 (link-boundary guard), F10 (MIN_SENTENCE_LENGTH coverage).
# ---------------------------------------------------------------------------


def test_flush_latches_suppression_reply_completed_tail_does_not_speak():
    """F1: after flush(), NOTHING may speak again for this reply -- including
    the reply's own later sentences and reply_completed()'s final-partial
    flush. Reviewer's exact resume-after-barge-in sequence.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Speaking now. And a trailing partial")
    assert spoken == ["Speaking now."]
    seq.flush()
    seq.reply_completed()
    assert spoken == ["Speaking now."]  # nothing more, ever, for this reply


def test_late_completion_with_stale_token_does_not_advance_a_newer_utterance():
    """F2: a late utterance_finished() carrying the token of an utterance a
    NEWER one has already superseded must be ignored, not misattributed to
    the current in-flight utterance (the double-voice class). Reviewer's
    within-one-reply sequence: barge-in -> next sentence in flight -> late
    ok=False for the stopped one -> must NOT advance past the current
    utterance.
    """
    spoken, stops = [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: stops.append(1))
    seq.feed("Alpha one. Alpha two. ")
    assert spoken == ["Alpha one."]
    stale_token = seq.current_utterance_token
    seq.flush()
    assert stops == [1]
    seq.begin_reply()
    seq.feed("Bravo one. Bravo two. ")
    assert spoken == ["Alpha one.", "Bravo one."]
    current_token = seq.current_utterance_token
    assert current_token != stale_token

    # The late, stale-token completion for the abandoned "Alpha one." must
    # not advance past the current in-flight "Bravo one.".
    seq.utterance_finished(ok=False, token=stale_token)
    assert spoken == ["Alpha one.", "Bravo one."]

    # The real completion, carrying the current token, advances normally.
    seq.utterance_finished(ok=True, token=current_token)
    assert spoken == ["Alpha one.", "Bravo one.", "Bravo two."]


def test_utterance_finished_with_no_token_keeps_working_for_simple_callers():
    """F2 backward-compat: token=None (the default) means "whatever is in
    flight" -- every existing verbatim test relies on this.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("One. Two. ")
    assert spoken == ["One."]
    seq.utterance_finished(ok=True)  # no token
    assert spoken == ["One.", "Two."]


def test_on_drained_rearms_for_the_next_reply_after_begin_reply():
    """F3, probe 1: on_drained must fire again for reply 2, not just reply 1."""
    drained = []
    seq = SentenceSequencer(speak=lambda t: None, stop_speech=lambda: None)
    seq.on_drained = lambda: drained.append(1)

    seq.feed("Reply one. ")
    seq.reply_completed()
    seq.utterance_finished(ok=True)
    assert drained == [1]

    seq.begin_reply()
    seq.feed("Reply two. ")
    seq.reply_completed()
    seq.utterance_finished(ok=True)
    assert drained == [1, 1]


def test_begin_reply_clears_a_stale_half_sentence_from_an_abandoned_reply():
    """F3, probe 2: an abandoned mid-sentence fragment must not glue onto
    the next reply's content.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Complete one. Half of an abandoned sen")
    seq.flush()
    seq.begin_reply()
    seq.feed("New reply starts here. ")
    seq.reply_completed()
    joined = " ".join(spoken)
    assert "abandoned sen" not in joined
    assert "New reply starts here." in joined


def test_begin_reply_clears_a_stuck_fence_flag_from_an_abandoned_reply():
    """F3, probe 3: an abandoned reply left mid-fence must not silence the
    entire next reply.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Here is code:\n```python\nx = 1\n")
    seq.flush()
    seq.begin_reply()
    seq.feed("Totally new reply here. ")
    seq.reply_completed()
    joined = " ".join(spoken)
    assert "Totally new reply here." in joined


def test_mid_line_fence_marker_does_not_hijack_the_fence_flag():
    """F4: a ``` arriving as its own delta, mid-line (not at the true start
    of a line), must be treated as literal text -- not a fence toggle -- or
    the rest of the reply silently vanishes into a permanently-open fence.
    Reviewer's exact token-shaped partition.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Wrap it in ")
    seq.feed("```")
    seq.feed(" markers. Then continue. ")
    seq.utterance_finished(ok=True)
    joined = " ".join(spoken)
    assert "markers." in joined
    assert "Then continue." in joined


def test_numbered_list_items_are_not_chopped_at_the_ordinal():
    """F5: '<digit>.' at the start of a line is an ordered-list marker, not
    a sentence boundary -- the reviewer's exact repro showed the ordinal
    landing on the wrong neighbour ('Steps: 1.' / 'First item 2.').
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Steps:\n1. First item\n2. Second item\nDone. ")
    # None of the buggy fragments may appear as their own utterance.
    assert "Steps: 1." not in spoken
    assert "First item 2." not in spoken
    assert "Second item Done." not in spoken
    joined = " ".join(spoken)
    assert "First item" in joined
    assert "Second item" in joined
    assert "Done." in joined


def test_terminator_at_chunk_end_waits_for_the_next_delta_before_deciding():
    """F6: a terminator landing as the LAST character of the currently
    buffered content must not be treated as boundary-confirmed until more
    data disambiguates it. The named verbatim delta-split test
    ("Half a sen" + "tence. ") is vacuous for this rule (no terminator ever
    lands at a chunk end); this is the discriminating case the reviewer
    named: a decimal split exactly at the '.'.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Value is 3.")
    assert spoken == []  # must not commit with the digit unconfirmed
    seq.feed("14 done. ")
    assert spoken == ["Value is 3.14 done."]


def test_common_words_that_look_like_abbreviations_are_not_swallowed():
    """F7: 'no', 'st', 'co', 'ft' (etc.) are common standalone words far more
    often than they are abbreviations -- keeping them in the lookbehind
    list swallowed genuine sentence boundaries. Reviewer's exact examples.
    """
    for text, first_utterance in [
        ("The answer is no. Moving on now. ", "The answer is no."),
        ("She came in 1st. Then he did. ", "She came in 1st."),
        ("Ask Smith and co. Then leave. ", "Ask Smith and co."),
    ]:
        spoken = []
        seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
        seq.feed(text)
        assert spoken == [first_utterance], text


def test_flush_after_reply_completed_still_fires_on_drained():
    """F8: a barge-in landing after reply_completed() (while the final
    utterance is still in flight) must still notify the controller the
    reply is done -- otherwise the hands-free loop hangs waiting for a
    drain signal that never arrives.
    """
    spoken, stops, drained = [], [], []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: stops.append(1))
    seq.on_drained = lambda: drained.append(1)
    seq.feed("Only one sentence here")
    seq.reply_completed()
    assert spoken == ["Only one sentence here"]
    assert drained == []  # still in flight; not drained yet
    seq.flush()  # barge-in lands mid-final-utterance
    assert stops == [1]
    assert drained == [1]  # must fire now -- nothing left to wait for


def test_link_text_containing_a_terminator_is_not_split_mid_link():
    """F9: a '.' inside markdown link TEXT must not be treated as a sentence
    boundary -- otherwise the link splits mid-syntax and the raw URL leaks
    into speech as its own utterance.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("Read [Hello. World](https://example.com/page) now. ")
    assert spoken == ["Read Hello. World now."]


def test_pure_punctuation_noise_below_min_length_is_dropped():
    """F10: a confirmed-boundary candidate that normalizes to pure
    punctuation (no letters/digits) shorter than MIN_SENTENCE_LENGTH is
    noise, not a sentence -- dropped rather than spoken.
    """
    spoken = []
    seq = SentenceSequencer(speak=spoken.append, stop_speech=lambda: None)
    seq.feed("?! Really? Yes indeed. ")
    assert spoken == ["Really?"]
    seq.utterance_finished(ok=True)
    assert spoken == ["Really?", "Yes indeed."]
    assert "?!" not in spoken
