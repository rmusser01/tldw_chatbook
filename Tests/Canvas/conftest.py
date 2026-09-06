"""Focused fixtures for Canvas runtime-profile tests."""

import pytest

from tldw_chatbook.Canvas.profiles import ProfileRecord, ProfileSnapshot


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
