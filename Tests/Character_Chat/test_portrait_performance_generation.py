"""Fixed reference and closed performance-pose directions."""

import hashlib
from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.expression_generation import (
    compose_performance_prompt,
)
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES
from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
    performance_reference,
)


def test_each_pose_preserves_identity_frame_background_and_only_changes_eyes_mouth():
    for role in FRAME_ROLES:
        prompt = compose_performance_prompt(role=role, expression_key="neutral")
        assert role in prompt
        for word in ("identity", "camera", "background", "alignment"):
            assert word in prompt
        assert (
            "closed eyes" in prompt
            if role.startswith("blink_")
            else "open eyes" in prompt
        )
    with pytest.raises(ValueError):
        compose_performance_prompt(role="wink", expression_key="neutral")


def test_animated_source_reference_is_frame_zero_and_original_is_unchanged():
    output = BytesIO()
    with (
        Image.new("RGB", (12, 12), "red") as first,
        Image.new("RGB", (12, 12), "blue") as second,
    ):
        first.save(
            output,
            format="GIF",
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
        )
    raw = output.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    reference = performance_reference(raw, digest)
    assert raw == output.getvalue()
    assert reference.mime_type == "image/png"
    assert reference.file_id == digest
    with Image.open(BytesIO(reference.content)) as still:
        assert still.n_frames == 1
        assert still.getpixel((2, 2)) == (255, 0, 0, 255)


def test_three_component_prompts_limit_edits_and_keep_same_reference_geometry():
    from tldw_chatbook.Character_Chat.expression_generation import (
        compose_performance_component_prompt,
    )
    from tldw_chatbook.Character_Chat.portrait_compositing import COMPONENT_ROLES

    for component in COMPONENT_ROLES:
        prompt = compose_performance_component_prompt(
            component=component, expression_key="neutral"
        )
        assert component in prompt
        assert all(
            word in prompt for word in ("identity", "camera", "background", "alignment")
        )
    with pytest.raises(ValueError):
        compose_performance_component_prompt(
            component="blink_small", expression_key="neutral"
        )
