"""Disposable six-role authoring without storage or provider authority."""

import pytest

from Tests.Chat.test_portrait_performance_state import performance_record
from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
    PortraitPerformanceDraft,
)


def test_incomplete_draft_cannot_publish_and_cancel_releases_artwork():
    record = performance_record()
    draft = PortraitPerformanceDraft(
        record.source, record.frames[0].png, expected_revision=0
    )
    draft.stage(record.frames[0])
    with pytest.raises(ValueError):
        draft.frames()
    for frame in record.frames[1:]:
        draft.stage(frame)
    assert draft.frames() == record.frames
    draft.cancel()
    assert draft.source_bytes == b""
    assert draft.staged_roles == ()
    with pytest.raises(ValueError):
        draft.frames()


def test_replacement_is_bounded_and_bad_geometry_keeps_the_old_role():
    from dataclasses import replace

    record = performance_record()
    draft = PortraitPerformanceDraft(
        record.source, record.frames[0].png, expected_revision=3
    )
    for frame in record.frames:
        draft.stage(frame)
    with pytest.raises(ValueError, match="geometry"):
        draft.stage(replace(record.frames[1], width=13))
    assert draft.frames() == record.frames
    draft.remove("open")
    assert "open" not in draft.staged_roles
    with pytest.raises(ValueError):
        draft.frames()
    draft.stage(record.frames[2])
    assert draft.expected_revision == 3
