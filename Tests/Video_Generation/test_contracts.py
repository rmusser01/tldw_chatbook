"""Contracts for the Video_Generation package (task-3401.2)."""

import dataclasses

import pytest


def test_request_defaults_and_frozen():
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenRequest

    req = VideoGenRequest(
        backend="minimax", prompt="a kite over the harbor", negative_prompt=None,
        duration_seconds=None, fps=None, width=None, height=None, ratio=None,
        steps=None, cfg_scale=None, seed=None, sampler=None, model=None,
        format="mp4", extra_params={},
    )
    assert req.request_id is None
    assert req.reference_assets == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.prompt = "mutated"  # type: ignore[misc]


def test_result_defaults_and_resolved_fields():
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    result = VideoGenResult(
        content=b"1234", content_type="video/mp4", container="mp4", bytes_len=4
    )
    assert result.duration_seconds is None
    assert result.fps is None
    assert result.width is None and result.height is None
    # Same contract as the image package's task-558 rule: adapters leave these
    # None unless they can state the value with certainty.
    assert result.resolved_seed is None
    assert result.resolved_model is None
    assert result.container == "mp4"


def test_result_requires_observed_container():
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    with pytest.raises(TypeError, match="container"):
        VideoGenResult(content=b"1234", content_type="video/mp4", bytes_len=4)


def test_reference_asset_shape():
    from tldw_chatbook.Video_Generation.adapters.base import ResolvedReferenceAsset

    asset = ResolvedReferenceAsset(
        kind="first_frame", content=b"\x89PNG", mime_type="image/png", source_name="kept variant 1",
    )
    assert asset.kind == "first_frame"
    assert asset.source_name == "kept variant 1"
    default_named = ResolvedReferenceAsset(kind="reference_video", content=b"x", mime_type="video/mp4")
    assert default_named.source_name == ""


def test_adapter_protocol_conformance():
    from tldw_chatbook.Video_Generation.adapters.base import (
        VideoGenRequest,
        VideoGenResult,
        VideoGenerationAdapter,
    )

    class FakeAdapter:
        name = "fake"
        supported_formats = {"mp4"}

        def generate(self, request: VideoGenRequest) -> VideoGenResult:
            return VideoGenResult(
                content=b"vid", content_type="video/mp4", container="mp4", bytes_len=3
            )

    adapter: VideoGenerationAdapter = FakeAdapter()
    req = VideoGenRequest(
        backend="fake", prompt="p", negative_prompt=None, duration_seconds=5,
        fps=24, width=1280, height=720, ratio="16:9", steps=None, cfg_scale=None,
        seed=-1, sampler=None, model=None, format="mp4", extra_params={},
    )
    result = adapter.generate(req)
    assert result.content == b"vid"
