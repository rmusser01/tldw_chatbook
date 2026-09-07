"""Pure cursor/progress logic for Library review sets (task-28240).

These exercise the tombstone-aware navigation model with NO database: a set is
a tuple of items plus a cursor, and an injected ``is_live`` predicate decides
which pinned items still resolve against the Media DB. Progress and completion
are always computed over LIVE items; the cursor is an absolute position that
never renumbers.
"""

from __future__ import annotations

from tldw_chatbook.Library.review_set_state import (
    ReviewProgress,
    ReviewSet,
    ReviewSetItem,
    advance_cursor,
    build_picker_rows,
    build_pinned_items,
    format_review_progress,
    is_complete,
    is_empty,
    plan_walk,
    resolve_cursor,
    review_progress,
)


def _items(*specs: tuple[int, bool]) -> tuple[ReviewSetItem, ...]:
    """Build items from (backing_media_id, done) pairs; position = index."""
    return tuple(
        ReviewSetItem(
            position=index,
            backing_media_id=media_id,
            title_snapshot=f"Item {media_id}",
            done=done,
        )
        for index, (media_id, done) in enumerate(specs)
    )


def _live(*dead_ids: int):
    """Return an is_live predicate where the named backing ids are tombstones."""
    dead = set(dead_ids)
    return lambda backing_media_id: backing_media_id not in dead


# --- progress over live items ------------------------------------------------


def test_progress_counts_live_items_and_reviewed():
    items = _items((10, True), (11, False), (12, True))
    progress = review_progress(items, cursor=1, is_live=_live())

    assert progress.index == 2  # cursor item is the 2nd live item (1-based)
    assert progress.total == 3
    assert progress.reviewed == 2


def test_progress_excludes_tombstones_from_total_and_ordinal():
    # position 1 (id 11) is deleted; live items are positions 0 and 2.
    items = _items((10, True), (11, False), (12, False))
    progress = review_progress(items, cursor=2, is_live=_live(11))

    assert progress.total == 2  # only 10 and 12 are live
    assert progress.index == 2  # id 12 is the 2nd of the live items
    assert progress.reviewed == 1  # only id 10 is live+done


def test_progress_on_empty_set_is_zero():
    items = _items((10, True), (11, False))
    progress = review_progress(items, cursor=0, is_live=_live(10, 11))

    assert progress == type(progress)(index=0, total=0, reviewed=0)


# --- cursor resolution on a tombstone ---------------------------------------


def test_resolve_cursor_on_a_tombstone_advances_to_next_live():
    items = _items((10, False), (11, False), (12, False))
    # cursor sits on the deleted middle item -> resolves forward to position 2.
    assert resolve_cursor(items, cursor=1, is_live=_live(11)) == 2


def test_resolve_cursor_falls_back_when_no_live_ahead():
    items = _items((10, False), (11, False), (12, False))
    # cursor on the last, which is dead; nothing live ahead -> nearest live back.
    assert resolve_cursor(items, cursor=2, is_live=_live(12)) == 1


def test_resolve_cursor_keeps_a_live_position():
    items = _items((10, False), (11, False))
    assert resolve_cursor(items, cursor=1, is_live=_live()) == 1


# --- advancing, skipping tombstones -----------------------------------------


def test_advance_forward_skips_a_tombstone():
    items = _items((10, False), (11, False), (12, False))
    # from position 0, forward, position 1 is dead -> land on position 2.
    assert advance_cursor(items, cursor=0, step=1, is_live=_live(11)) == 2


def test_advance_back_skips_a_tombstone():
    items = _items((10, False), (11, False), (12, False))
    assert advance_cursor(items, cursor=2, step=-1, is_live=_live(11)) == 0


def test_advance_clamps_at_the_last_live_item():
    items = _items((10, False), (11, False))
    # already on the last live item; forward stays put.
    assert advance_cursor(items, cursor=1, step=1, is_live=_live()) == 1


def test_advance_clamps_at_the_first_live_item():
    items = _items((10, False), (11, False))
    assert advance_cursor(items, cursor=0, step=-1, is_live=_live()) == 0


# --- completion vs empty -----------------------------------------------------


def test_is_complete_when_every_live_item_is_done():
    items = _items((10, True), (11, True))
    assert is_complete(items, is_live=_live()) is True


def test_is_complete_ignores_tombstoned_not_done_items():
    # id 11 is not done, but it is deleted -> the live set (id 10) is all done.
    items = _items((10, True), (11, False))
    assert is_complete(items, is_live=_live(11)) is True


def test_not_complete_while_a_live_item_is_unreviewed():
    items = _items((10, True), (11, False))
    assert is_complete(items, is_live=_live()) is False


def test_all_tombstoned_set_is_empty_not_complete():
    items = _items((10, True), (11, True))
    assert is_empty(items, is_live=_live(10, 11)) is True
    assert is_complete(items, is_live=_live(10, 11)) is False


# --- the walk planner (forward marks-done, back does not) --------------------


def test_plan_walk_forward_marks_current_done_and_targets_next():
    items = _items((10, False), (11, False), (12, False))
    outcome = plan_walk(items, cursor=0, step=1, is_live=_live())

    assert outcome.new_cursor == 1
    assert outcome.mark_done_backing_id == 10  # the item we leave
    assert outcome.target.backing_media_id == 11  # the item we load


def test_plan_walk_forward_skips_a_tombstone_target():
    items = _items((10, False), (11, False), (12, False))
    outcome = plan_walk(items, cursor=0, step=1, is_live=_live(11))

    assert outcome.new_cursor == 2
    assert outcome.mark_done_backing_id == 10
    assert outcome.target.backing_media_id == 12


def test_plan_walk_back_does_not_mark_and_targets_prev():
    items = _items((10, False), (11, False), (12, False))
    outcome = plan_walk(items, cursor=2, step=-1, is_live=_live())

    assert outcome.new_cursor == 1
    assert outcome.mark_done_backing_id is None  # Prev never auto-marks
    assert outcome.target.backing_media_id == 11


def test_plan_walk_forward_on_last_item_marks_done_without_moving():
    items = _items((10, False), (11, False))
    outcome = plan_walk(items, cursor=1, step=1, is_live=_live())

    assert outcome.new_cursor == 1  # clamped
    assert outcome.mark_done_backing_id == 11  # the completion gesture
    assert outcome.target is None  # nothing new to load


# --- the progress readout string --------------------------------------------


def test_format_progress_reads_x_of_m_reviewed_n():
    assert (
        format_review_progress(ReviewProgress(index=12, total=40, reviewed=7))
        == "12 of 40 · 7 reviewed"
    )


def test_format_progress_all_reviewed():
    assert (
        format_review_progress(ReviewProgress(index=40, total=40, reviewed=40))
        == "All 40 reviewed"
    )


def test_format_progress_empty_set():
    assert (
        format_review_progress(ReviewProgress(index=0, total=0, reviewed=0))
        == "No items to review"
    )


# --- building the pinned item list (entry points, task-28242) ----------------


def test_build_pinned_items_dedupes_by_id_keeping_first():
    items, truncated = build_pinned_items([(10, "A"), (11, "B"), (10, "A-again")])
    assert items == [(10, "A"), (11, "B")]
    assert truncated is False


def test_build_pinned_items_caps_and_flags_truncation():
    items, truncated = build_pinned_items(
        [(index, f"t{index}") for index in range(600)], cap=500
    )
    assert len(items) == 500
    assert items[0] == (0, "t0") and items[-1] == (499, "t499")
    assert truncated is True


def test_build_pinned_items_dups_past_cap_do_not_flag_truncation():
    pairs = [(index, f"t{index}") for index in range(500)] + [(0, "dup")] * 50
    items, truncated = build_pinned_items(pairs, cap=500)
    assert len(items) == 500
    assert truncated is False  # only duplicates followed the cap, nothing dropped


def test_build_pinned_items_coerces_types():
    items, _ = build_pinned_items([("10", 42)])
    assert items == [(10, "42")]


# --- picker rows (set picker, task-28243) ------------------------------------


def _review_set(
    set_id: str,
    name: str,
    items: tuple[ReviewSetItem, ...],
    *,
    cursor: int = 0,
    active: bool = False,
) -> ReviewSet:
    return ReviewSet(
        set_id=set_id,
        name=name,
        origin="browse",
        cursor=cursor,
        active=active,
        completed_at=None,
        items=items,
        created_at="2026-09-01T00:00:00Z",
        updated_at="2026-09-01T00:00:00Z",
    )


def test_picker_rows_carry_id_name_progress_and_active_in_order():
    sets = (
        _review_set(
            "s1", "All media", _items((10, True), (11, False)), cursor=1, active=True
        ),
        _review_set("s2", "pdf items", _items((20, False))),
    )
    rows = build_picker_rows(sets, is_live=_live())

    assert rows == [
        ("s1", "All media", "2 of 2 · 1 reviewed · 2026-09-01 00:00", True),
        ("s2", "pdf items", "1 of 1 · 0 reviewed · 2026-09-01 00:00", False),
    ]


def test_picker_rows_progress_is_live_only():
    # id 11 tombstoned: total counts live items, done tombstones don't count.
    sets = (_review_set("s1", "Set", _items((10, False), (11, True))),)
    rows = build_picker_rows(sets, is_live=_live(11))
    assert rows == [("s1", "Set", "1 of 1 · 0 reviewed · 2026-09-01 00:00", False)]


def test_picker_rows_completed_and_empty_labels():
    sets = (
        _review_set("done", "Done set", _items((10, True), (11, True))),
        _review_set("gone", "Gone set", _items((20, False))),
    )
    rows = build_picker_rows(sets, is_live=_live(20))
    assert rows[0][2] == "All 2 reviewed · 2026-09-01 00:00"
    assert rows[1][2] == "No items to review · 2026-09-01 00:00"


def test_picker_rows_same_name_sets_are_distinguishable():
    """task-31238: auto-names ("2 selected items") rendered two selection
    sets as identical picker rows -- the detail label ends with the created
    minute so same-name sets can be told apart even when created the SAME
    DAY (Qodo on #2366: date-only left same-day twins identical)."""
    twin_a = _review_set("a", "2 selected items", _items((10, False)))
    twin_b = ReviewSet(
        set_id="b",
        name="2 selected items",
        origin="selection",
        cursor=0,
        active=False,
        completed_at=None,
        items=_items((20, False)),
        created_at="2026-09-01T12:30:00Z",
        updated_at="2026-09-01T12:30:00Z",
    )
    rows = build_picker_rows((twin_a, twin_b), is_live=_live())

    labels = [(name, detail) for _sid, name, detail, _active in rows]
    assert len(set(labels)) == 2  # not identical any more
    assert rows[0][2].endswith("2026-09-01 00:00")
    assert rows[1][2].endswith("2026-09-01 12:30")


def test_picker_rows_tolerate_a_malformed_created_timestamp():
    """A garbage created_at must not crash the picker; the date part is
    simply omitted."""
    broken = ReviewSet(
        set_id="x",
        name="Set",
        origin="browse",
        cursor=0,
        active=False,
        completed_at=None,
        items=_items((10, False)),
        created_at="",
        updated_at="",
    )
    rows = build_picker_rows((broken,), is_live=_live())
    assert rows == [("x", "Set", "1 of 1 · 0 reviewed", False)]
