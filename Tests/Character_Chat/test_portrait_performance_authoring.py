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


def test_new_composition_replaces_old_slots_and_failed_retry_is_atomic():
    from dataclasses import replace

    from Tests.Character_Chat.test_portrait_compositing import core, scene
    from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
        PortraitCompositionCandidate,
    )

    api = core()
    base, recipe, components = scene()
    record = performance_record()
    draft = PortraitPerformanceDraft(record.source, base, expected_revision=3)
    draft.stage_group(record.frames)
    assert draft.can_apply
    partial = PortraitCompositionCandidate(
        recipe, 1, (), api.assemble_components(base, recipe, {})
    )
    draft.replace_composition(partial)
    assert draft.staged_roles == ("rest",)
    assert not draft.can_apply
    with pytest.raises(ValueError):
        draft.frames()
    complete = PortraitCompositionCandidate(
        recipe,
        1,
        tuple(components.items()),
        api.assemble_components(base, recipe, components),
    )
    draft.replace_composition(complete)
    assert draft.can_apply
    old = draft.frames()
    with pytest.raises(ValueError):
        draft.replace_composition(replace(complete, frames=complete.frames[:-1]))
    assert draft.composition is complete and draft.frames() == old and draft.can_apply
    assert "PNG" not in repr(complete)
    with pytest.raises(ValueError):
        draft.stage(replace(old[0], width=13))
    assert draft.composition is complete
    draft.stage(old[0])
    assert draft.composition is None and draft.can_apply
    draft.replace_composition(complete)
    draft.remove("small")
    assert draft.composition is None and not draft.can_apply
    draft.replace_composition(complete)
    draft.cancel()
    assert (
        draft.composition is None and not draft.can_apply and draft.source_bytes == b""
    )


def test_invalid_manual_actions_do_not_drop_composition():
    from Tests.Character_Chat.test_portrait_compositing import core, scene
    from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
        PortraitCompositionCandidate,
    )

    api = core()
    base, recipe, components = scene()
    record = performance_record()
    draft = PortraitPerformanceDraft(record.source, base, expected_revision=3)
    candidate = PortraitCompositionCandidate(
        recipe,
        1,
        tuple(components.items()),
        api.assemble_components(base, recipe, components),
    )
    draft.replace_composition(candidate)
    for action in (lambda: draft.stage_group(()), lambda: draft.remove("unknown")):
        with pytest.raises(ValueError):
            action()
        assert draft.composition is candidate and draft.can_apply
