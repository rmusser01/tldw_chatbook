"""Focused fixtures for Canvas runtime-profile tests."""

from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.profiles import (
    ProfileRecord,
    ProfileSnapshot,
    load_profile_snapshot,
)


@pytest.fixture
def profile_snapshot():
    return ProfileSnapshot(
        build_id="a" * 64,
        policy_id="b" * 64,
        profiles=(
            ProfileRecord("canvas-v1", "c" * 64, True, None, 0),
            ProfileRecord("canvas-v2-mermaid-1", "d" * 64, True, None, 77817),
        ),
        default_diagram_profile="canvas-v2-mermaid-1",
    )


@pytest.fixture
def candidate_snapshot():
    base = load_profile_snapshot()
    candidate = "canvas-v2-mermaid-1"
    assert any(row.profile_id == candidate for row in base.profiles)
    return replace(
        base,
        profiles=tuple(
            replace(row, executable=True, reason=None)
            if row.profile_id == candidate
            else row
            for row in base.profiles
        ),
        default_diagram_profile=candidate,
    )
