"""Exact, privacy-preserving Console composer improvement transactions."""

from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ComposerDraftSegmentSnapshot,
    ComposerTransactionValidationError,
    ConsoleComposerBar,
)


LARGE_PASTE = "ordinary pasted material " * 8
INLINE_BODY = "SECRET INLINE FILE BODY\nwith unicode: 雪"
INLINE_LABEL = "📄 /private/customer-notes.md · 2 KB"


def _mixed_composer(*, second_file: bool = False) -> ConsoleComposerBar:
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_text("Draft ")
    composer.insert_pasted_text("small paste ")
    composer.insert_pasted_text(LARGE_PASTE)
    composer.insert_file_segment(INLINE_BODY, INLINE_LABEL)
    composer.insert_text(" tail Ω")
    if second_file:
        composer.insert_file_segment("SECOND SECRET", "second.txt · 13 B")
        composer.insert_text(" end")
    return composer


def _semantic_snapshot(snapshot):
    """Exclude the staleness-only generation/fingerprint from exact UI state."""
    return (
        snapshot.segments,
        snapshot.cursor_index,
        snapshot.selection,
        snapshot.edit_serial,
    )


def test_capture_snapshot_preserves_explicit_origins_state_cursor_and_selection():
    composer = _mixed_composer()
    composer._segments[1].collapse_state = "expanded"
    composer._segments[2].collapse_state = "confirm"
    composer.move_cursor_home()
    composer.move_cursor_right()
    snapshot = composer.capture_draft_snapshot()

    assert [segment.origin for segment in snapshot.segments] == [
        "literal",
        "paste",
        "paste",
        "inline_file",
        "literal",
    ]
    assert [segment.collapse_state for segment in snapshot.segments] == [
        "literal",
        "expanded",
        "confirm",
        "collapsed",
        "literal",
    ]
    assert snapshot.segments[3].text == INLINE_BODY
    assert snapshot.segments[3].label == INLINE_LABEL
    assert snapshot.cursor_index == 1
    assert snapshot.selection is None
    assert snapshot.edit_serial == composer.edit_serial

    composer.select_all_draft()
    selected = composer.capture_draft_snapshot()
    assert selected.selection == "all"
    assert selected.cursor_index == len(composer.draft_text())


def test_snapshot_is_deeply_immutable_and_fingerprint_is_deterministic():
    composer = _mixed_composer()

    first = composer.capture_draft_snapshot()
    second = composer.capture_draft_snapshot()

    assert first == second
    assert isinstance(first.segments, tuple)
    with pytest.raises(FrozenInstanceError):
        first.cursor_index = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.segments[0].text = "changed"  # type: ignore[misc]


def test_restore_snapshot_round_trips_exact_semantic_state_without_legacy_load(
    monkeypatch,
):
    composer = _mixed_composer()
    composer._segments[1].collapse_state = "expanded"
    composer._segments[2].collapse_state = "confirm"
    composer.select_all_draft()
    before = composer.capture_draft_snapshot()
    pending_before = "photo.png · 240 KB"
    composer.set_pending_attachment_label(pending_before)
    monkeypatch.setattr(
        composer,
        "load_draft",
        lambda _text: pytest.fail("restore_snapshot must not call load_draft"),
    )

    composer.insert_text("replacement")
    composer.restore_snapshot(before)

    after = composer.capture_draft_snapshot()
    assert _semantic_snapshot(after) == _semantic_snapshot(before)
    assert composer._pending_attachment_label == pending_before


def test_prompt_replacement_consumes_all_segments_as_one_paste_and_undo_restores_exactly():
    composer = _mixed_composer()
    composer._segments[1].collapse_state = "expanded"
    composer._segments[2].collapse_state = "confirm"
    composer.select_all_draft()
    before = composer.capture_draft_snapshot()
    composer.set_pending_attachment_label("photo.png · 240 KB")
    replacement = "replacement material " * 8

    applied = composer.replace_snapshot_as_paste(before, replacement)

    assert applied == before
    assert composer.draft_text() == replacement
    after = composer.capture_draft_snapshot()
    assert [(segment.origin, segment.collapse_state) for segment in after.segments] == [
        ("paste", "collapsed")
    ]
    assert after.selection is None
    assert after.cursor_index == len(replacement)
    assert composer._pending_attachment_label == "photo.png · 240 KB"
    assert composer.improvement_undo_available is True

    assert composer.undo_improvement() is True
    assert _semantic_snapshot(composer.capture_draft_snapshot()) == _semantic_snapshot(
        before
    )
    assert composer._pending_attachment_label == "photo.png · 240 KB"


def test_prompt_replacement_can_clear_an_exact_nonempty_snapshot():
    composer = _mixed_composer()
    before = composer.capture_draft_snapshot()

    assert composer.replace_snapshot_as_paste(before, "") == before
    assert composer.capture_draft_snapshot().segments == ()
    assert composer.draft_text() == ""
    assert composer.undo_improvement() is True
    assert _semantic_snapshot(composer.capture_draft_snapshot()) == _semantic_snapshot(
        before
    )


def test_prompt_replacement_empty_to_empty_is_a_noop_without_consuming_prior_undo():
    composer = ConsoleComposerBar()
    original = composer.capture_draft_snapshot()

    assert composer.replace_snapshot_as_paste(original, "") is None
    assert composer.capture_draft_snapshot() == original
    assert composer.improvement_undo_available is False


def test_prompt_replacement_rejects_a_stale_complete_snapshot_without_mutation():
    composer = _mixed_composer()
    stale = composer.capture_draft_snapshot()
    composer.insert_text(" changed")
    live = composer.capture_draft_snapshot()

    with pytest.raises(ComposerTransactionValidationError, match="stale"):
        composer.replace_snapshot_as_paste(stale, "replacement")

    assert composer.capture_draft_snapshot() == live
    assert composer.improvement_undo_available is False


def test_same_text_load_invalidates_snapshot_by_generation():
    composer = ConsoleComposerBar()
    composer.load_draft("same bytes")
    snapshot = composer.capture_draft_snapshot()

    composer.load_draft("same bytes")

    with pytest.raises(ComposerTransactionValidationError, match="stale"):
        composer.apply_improvement(snapshot, "better bytes")
    assert composer.draft_text() == "same bytes"


def test_replaced_composer_owner_rejects_byte_identical_snapshot():
    original = ConsoleComposerBar()
    original.insert_text("same bytes")
    snapshot = original.capture_draft_snapshot()
    replacement = ConsoleComposerBar()
    replacement.insert_text("same bytes")

    with pytest.raises(ComposerTransactionValidationError, match="stale"):
        replacement.apply_improvement(snapshot, "better bytes")

    assert replacement.draft_text() == "same bytes"


def test_same_bytes_after_manual_edit_reject_snapshot_by_edit_serial():
    composer = ConsoleComposerBar()
    composer.insert_text("same bytes")
    snapshot = composer.capture_draft_snapshot()
    composer.insert_text("!")
    composer.delete_left()
    assert composer.draft_text() == "same bytes"

    with pytest.raises(ComposerTransactionValidationError, match="stale"):
        composer.apply_improvement(snapshot, "better bytes")

    assert composer.draft_text() == "same bytes"


def test_projection_keeps_literal_and_paste_but_hides_every_inline_file_value():
    composer = _mixed_composer(second_file=True)
    snapshot = composer.capture_draft_snapshot()

    projection = composer.project_snapshot_for_model(
        snapshot, request_nonce="request-雪-1"
    )

    assert "Draft small paste" in projection.text
    assert LARGE_PASTE in projection.text
    for protected in (
        INLINE_BODY,
        "SECRET INLINE FILE BODY",
        "customer-notes.md",
        "/private/",
        "2 KB",
        "SECOND SECRET",
        "second.txt",
        "13 B",
    ):
        assert protected not in projection.text
    assert projection.placeholder_nonce == "request-雪-1"
    assert len(projection.placeholder_ids) == 2
    assert projection.placeholder_ids[0] != projection.placeholder_ids[1]
    assert projection.placeholder_ids[0] in projection.text
    assert projection.placeholder_ids[1] in projection.text
    assert (
        composer.project_snapshot_for_model(snapshot, request_nonce="request-雪-1")
        == projection
    )


@pytest.mark.parametrize("nonce", ["", " ", "bad\nnonce", "x" * 129, 7])
def test_projection_rejects_empty_or_invalid_nonce(nonce):
    composer = _mixed_composer()

    with pytest.raises(ComposerTransactionValidationError, match="nonce"):
        composer.project_snapshot_for_model(
            composer.capture_draft_snapshot(), request_nonce=nonce
        )


def test_projection_rejects_nonce_collision_with_improvable_source():
    composer = ConsoleComposerBar()
    composer.insert_text("user already wrote nonce-collision here")

    with pytest.raises(ComposerTransactionValidationError, match="collision"):
        composer.project_snapshot_for_model(
            composer.capture_draft_snapshot(), request_nonce="nonce-collision"
        )


@pytest.mark.parametrize(
    "reserved_text",
    [
        "[[TLDW_PROTECTED:user-authored]]",
        "[[TLDW_PROTECTED:not-closed",
    ],
)
def test_projection_rejects_reserved_placeholder_syntax_in_literal_text(
    reserved_text,
):
    composer = ConsoleComposerBar()
    composer.insert_text(f"before {reserved_text} after")
    before = composer.capture_draft_snapshot()

    with pytest.raises(ComposerTransactionValidationError, match="reserved"):
        composer.project_snapshot_for_model(before, request_nonce="reserved-literal-1")

    assert composer.capture_draft_snapshot() == before
    assert composer.improvement_undo_available is False


def test_projection_rejects_reserved_placeholder_candidate_in_paste_with_inline_file():
    composer = ConsoleComposerBar()
    composer.insert_text("before ")
    composer.insert_pasted_text("pasted [[TLDW_PROTECTED:user-candidate]] bytes")
    composer.insert_file_segment(INLINE_BODY, INLINE_LABEL)
    composer.insert_text(" after")
    before = composer.capture_draft_snapshot()

    with pytest.raises(ComposerTransactionValidationError, match="reserved"):
        composer.project_snapshot_for_model(before, request_nonce="reserved-paste-1")

    assert composer.capture_draft_snapshot() == before
    assert composer.improvement_undo_available is False


def test_safe_unicode_near_placeholder_spelling_round_trips_without_mutation():
    safe_text = "雪 [[TLDW_PROTECTED：user-authored]] Ω"
    composer = ConsoleComposerBar()
    composer.insert_text(safe_text)
    before = composer.capture_draft_snapshot()

    projection = composer.project_snapshot_for_model(
        before, request_nonce="safe-unicode-1"
    )
    result = composer.apply_improvement(before, projection.text)

    assert projection.text == safe_text
    assert result is None
    assert composer.capture_draft_snapshot() == before
    assert composer.improvement_undo_available is False


def test_apply_rewrites_only_improvable_spans_and_rehydrates_files_exactly():
    composer = _mixed_composer(second_file=True)
    before = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(before, request_nonce="apply-1")
    pending_before = "2 files"
    composer.set_pending_attachment_label(pending_before, count=2, total=2)
    rewritten = projection.text.replace("Draft ", "Improved ").replace(
        " tail Ω", " concise tail Ω"
    )

    undo = composer.apply_improvement(before, rewritten)
    after = composer.capture_draft_snapshot()

    assert undo == before
    assert composer.draft_text() == (
        rewritten.replace(projection.placeholder_ids[0], INLINE_BODY).replace(
            projection.placeholder_ids[1], "SECOND SECRET"
        )
    )
    protected = [s for s in after.segments if s.origin == "inline_file"]
    assert protected == [before.segments[3], before.segments[5]]
    assert all(
        segment.origin == "literal"
        for segment in after.segments
        if segment.origin != "inline_file"
    )
    assert composer._pending_attachment_label == pending_before
    assert composer.improvement_undo_available is True


def _tampered_rewrites(projection):
    first, second = projection.placeholder_ids
    return {
        "removed": projection.text.replace(first, ""),
        "duplicated": projection.text.replace(first, first + first),
        "edited": projection.text.replace(first, first[:-2] + "x]]"),
        "reordered": projection.text.replace(first, "TEMP", 1)
        .replace(second, first, 1)
        .replace("TEMP", second, 1),
        "extra": projection.text + " [[TLDW_PROTECTED:forged]]",
    }


@pytest.mark.parametrize(
    "tamper_kind", ["removed", "duplicated", "edited", "reordered", "extra"]
)
def test_tampered_placeholders_fail_atomically(tamper_kind):
    composer = _mixed_composer(second_file=True)
    snapshot = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(snapshot, request_nonce="veto-1")
    before = composer.capture_draft_snapshot()
    pending_before = "photo.png · 240 KB"
    composer.set_pending_attachment_label(pending_before)

    with pytest.raises(ComposerTransactionValidationError, match="placeholder"):
        composer.apply_improvement(
            snapshot, _tampered_rewrites(projection)[tamper_kind]
        )

    assert composer.capture_draft_snapshot() == before
    assert composer._pending_attachment_label == pending_before
    assert composer.improvement_undo_available is False


def test_user_edit_and_forged_snapshot_fail_before_mutation():
    composer = _mixed_composer()
    snapshot = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(snapshot, request_nonce="stale-1")
    composer.insert_text(" user edit")
    live = composer.capture_draft_snapshot()

    with pytest.raises(ComposerTransactionValidationError, match="stale"):
        composer.apply_improvement(
            snapshot, projection.text.replace("Draft", "Improved")
        )
    assert composer.capture_draft_snapshot() == live

    forged = replace(live, fingerprint="0" * 64)
    with pytest.raises(ComposerTransactionValidationError, match="fingerprint"):
        composer.restore_snapshot(forged)
    assert composer.capture_draft_snapshot() == live


@pytest.mark.parametrize(
    "segment_values",
    [
        ("x", "bogus", "literal", None),
        ("x", "literal", "bogus", None),
        (7, "literal", "literal", None),
        ("x", "literal", "literal", 7),
    ],
)
def test_invalid_snapshot_segment_shape_fails_closed(segment_values):
    composer = ConsoleComposerBar()
    valid = composer.capture_draft_snapshot()
    segment = ComposerDraftSegmentSnapshot(*segment_values)
    malformed = replace(valid, segments=(segment,))

    with pytest.raises(ComposerTransactionValidationError):
        composer.project_snapshot_for_model(malformed, request_nonce="shape-1")


@pytest.mark.parametrize(
    "changes",
    [
        {"cursor_index": -1},
        {"cursor_index": 99},
        {"selection": (2, 1)},
        {"selection": (0, 99)},
        {"selection": "partial"},
        {"edit_serial": -1},
        {"generation": -1},
    ],
)
def test_invalid_snapshot_cursor_selection_and_counter_shape_fails_closed(changes):
    composer = ConsoleComposerBar()
    composer.load_draft("abc")
    valid = composer.capture_draft_snapshot()
    malformed = replace(valid, **changes)

    with pytest.raises(ComposerTransactionValidationError):
        composer.restore_snapshot(malformed)


def test_no_change_has_no_mutation_serial_bump_or_new_undo():
    composer = _mixed_composer()
    snapshot = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(snapshot, request_nonce="same-1")

    result = composer.apply_improvement(snapshot, projection.text)

    assert result is None
    assert composer.capture_draft_snapshot() == snapshot
    assert composer.improvement_undo_available is False


def test_improvement_undo_restores_exact_state_and_preserves_attachment():
    composer = _mixed_composer()
    composer._segments[1].collapse_state = "expanded"
    composer._segments[2].collapse_state = "confirm"
    composer.select_all_draft()
    before = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(before, request_nonce="undo-1")
    composer.set_pending_attachment_label("photo.png · 240 KB")

    composer.apply_improvement(before, projection.text.replace("Draft", "Better"))
    assert composer.undo_improvement() is True

    restored = composer.capture_draft_snapshot()
    assert _semantic_snapshot(restored) == _semantic_snapshot(before)
    assert composer._pending_attachment_label == "photo.png · 240 KB"
    assert composer.improvement_undo_available is False
    assert composer.undo_improvement() is False


def test_improvement_comparison_masks_inline_file_bytes_but_keeps_labels():
    composer = _mixed_composer()
    before = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(before, request_nonce="review-1")

    composer.apply_improvement(before, projection.text.replace("Draft", "Better"))

    comparison = composer.improvement_comparison()
    assert comparison is not None
    original, improved = comparison
    assert INLINE_BODY not in original
    assert INLINE_BODY not in improved
    assert INLINE_LABEL in original
    assert INLINE_LABEL in improved
    assert "Draft" in original
    assert "Better" in improved


@pytest.mark.parametrize("event", ["manual_edit", "send", "load"])
def test_improvement_undo_expires_on_documented_draft_scope_events(event):
    composer = _mixed_composer()
    snapshot = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(snapshot, request_nonce="expire-1")
    composer.apply_improvement(snapshot, projection.text.replace("Draft", "Better"))
    assert composer.improvement_undo_available is True

    if event == "manual_edit":
        composer.insert_text("!")
    elif event == "send":
        assert composer.stash_draft_for_send() is not None
    else:
        composer.load_draft(composer.draft_text())

    assert composer.improvement_undo_available is False
    assert composer.undo_improvement() is False


def test_empty_stash_after_improvement_to_empty_expires_undo():
    composer = ConsoleComposerBar()
    composer.insert_text("remove this")
    snapshot = composer.capture_draft_snapshot()

    assert composer.apply_improvement(snapshot, "") == snapshot
    assert composer.draft_text() == ""
    assert composer.improvement_undo_available is True

    assert composer.stash_draft_for_send() is None
    assert composer.improvement_undo_available is False
    assert composer.undo_improvement() is False


def test_attachment_only_empty_stash_expires_undo_without_mutating_attachment():
    composer = ConsoleComposerBar()
    composer.insert_text("remove this before image-only send")
    snapshot = composer.capture_draft_snapshot()
    assert composer.apply_improvement(snapshot, "") == snapshot

    store = ConsoleChatStore()
    session = store.ensure_session()
    attachment = PendingAttachment(
        file_path="/tmp/photo.png",
        display_name="photo.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )
    attachment_state = vars(attachment).copy()
    store.set_pending_attachment(session.id, attachment)
    composer.set_pending_attachment_label(attachment.label)

    assert composer.stash_draft_for_send() is None

    assert composer.improvement_undo_available is False
    assert store.pending_attachment(session.id) is attachment
    assert vars(attachment) == attachment_state
    assert composer._pending_attachment_label == attachment.label


def test_later_improvement_replaces_prior_undo_and_undoes_only_latest_apply():
    composer = _mixed_composer()
    first_snapshot = composer.capture_draft_snapshot()
    first_projection = composer.project_snapshot_for_model(
        first_snapshot, request_nonce="later-1"
    )
    composer.apply_improvement(
        first_snapshot, first_projection.text.replace("Draft", "First")
    )
    after_first = composer.capture_draft_snapshot()
    second_projection = composer.project_snapshot_for_model(
        after_first, request_nonce="later-2"
    )

    composer.apply_improvement(
        after_first, second_projection.text.replace("First", "Second")
    )
    assert composer.undo_improvement() is True

    assert _semantic_snapshot(composer.capture_draft_snapshot()) == _semantic_snapshot(
        after_first
    )


def test_failed_and_no_change_attempts_preserve_existing_improvement_undo():
    composer = _mixed_composer()
    original = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(original, request_nonce="keep-1")
    composer.apply_improvement(original, projection.text.replace("Draft", "Better"))
    current = composer.capture_draft_snapshot()
    current_projection = composer.project_snapshot_for_model(
        current, request_nonce="keep-2"
    )

    assert composer.apply_improvement(current, current_projection.text) is None
    assert composer.improvement_undo_available is True
    with pytest.raises(ComposerTransactionValidationError):
        composer.apply_improvement(
            current,
            current_projection.text.replace(
                current_projection.placeholder_ids[0],
                current_projection.placeholder_ids[0] * 2,
            ),
        )
    assert composer.improvement_undo_available is True


def test_take_improvement_undo_snapshot_consumes_the_single_pending_undo():
    composer = _mixed_composer()
    before = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(before, request_nonce="take-1")
    composer.apply_improvement(before, projection.text.replace("Draft", "Better"))

    assert composer.take_improvement_undo_snapshot() == before
    assert composer.take_improvement_undo_snapshot() is None
    assert composer.improvement_undo_available is False
