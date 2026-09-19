"""conversation_local_marks.updated_at must use one canonical shape (TASK-32803.4).

The service's `_now()` used `isoformat()` (microseconds, 27 chars, or no
fraction when microsecond==0, 20 chars) while the column's other writers use
the canonical 24-char millisecond+Z shape via
`_get_current_utc_timestamp_iso`. The table's only ORDER BY over updated_at is
lexical, and the two shapes sort against each other by string prefix rather than
by instant; the voice-promotion reconciler's canonical-timestamp validator also
rejected the microsecond/no-fraction shape.
"""

from datetime import UTC, datetime

from tldw_chatbook.Chat.chat_persistence_service import _is_canonical_utc_timestamp
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)


def test_now_produces_the_canonical_millisecond_shape():
    now = ConversationLocalMarksService._now()
    assert len(now) == 24 and now.endswith("Z"), now
    # The exact shape the column's other writers produce and the validator
    # accepts (AC#1 + AC#2).
    assert _is_canonical_utc_timestamp(now), now


def test_the_old_microsecond_shape_inverted_the_lexical_ordering():
    """Regression: two marks at the SAME instant, one written in each old
    shape, sorted by string prefix rather than by time. With both writers on the
    canonical shape this cannot happen (AC#3)."""
    instant = datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=UTC)

    old_microsecond = instant.isoformat().replace("+00:00", "Z")  # writer A (old)
    canonical_millis = instant.isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )  # writer B

    # Same instant, but the microsecond form sorts BEFORE the millisecond form
    # ('0' < 'Z' at the position where one has more digits and the other 'Z').
    assert old_microsecond != canonical_millis
    assert old_microsecond < canonical_millis  # the inversion the bug caused

    # The validator rejected the old shape and accepts the canonical one.
    assert not _is_canonical_utc_timestamp(old_microsecond)
    assert _is_canonical_utc_timestamp(canonical_millis)

    # After the fix, `_now()` is the canonical shape, so every mark sorts by
    # instant.
    assert _is_canonical_utc_timestamp(ConversationLocalMarksService._now())
