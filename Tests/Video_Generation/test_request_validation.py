
from tldw_chatbook.Video_Generation.request_validation import (
    REFERENCE_AUDIO_MAX_BYTES,
    REFERENCE_IMAGE_MAX_BYTES,
    REFERENCE_VIDEO_MAX_BYTES,
    validate_video_generation_request,
)


class _FakeConfig:
    """Minimal stand-in for VideoGenerationConfig (only the attrs the
    validator reads)."""

    max_prompt_length = 100
    max_duration_seconds = 15
    max_fps = 30
    max_width = 2560
    max_height = 1440
    max_pixels = 2560 * 1440
    max_steps = 50
    max_reference_assets = 12
    minimax_video_allowed_extra_params: list = ["callback_url"]
    comfyui_allowed_extra_params: list = []
    sd_cpp_allowed_extra_params: list = ["cli_args"]


CONFIG = _FakeConfig()


def _valid_structured(**overrides):
    base = {
        "backend": "minimax",
        "prompt": "a kite over the harbor",
        "duration_seconds": 5,
        "fps": 24,
        "width": 1280,
        "height": 720,
        "ratio": "16:9",
        "steps": None,
        "cfg_scale": None,
        "extra_params": {},
        "reference_assets": (),
    }
    base.update(overrides)
    return base


def _asset(kind, *, content=b"x", mime="image/png"):
    from tldw_chatbook.Video_Generation.adapters.base import ResolvedReferenceAsset
    return ResolvedReferenceAsset(kind=kind, content=content, mime_type=mime)


def test_valid_request_passes():
    assert validate_video_generation_request(_valid_structured(), config=CONFIG) == []


def test_prompt_too_long_rejected():
    issues = validate_video_generation_request(_valid_structured(prompt="x" * 101), config=CONFIG)
    assert [i.path for i in issues] == ["prompt"]


def test_duration_fps_bounds_rejected():
    issues = validate_video_generation_request(
        _valid_structured(duration_seconds=16, fps=0), config=CONFIG,
    )
    paths = {i.path for i in issues}
    assert paths == {"duration_seconds", "fps"}


def test_dimension_and_pixel_bounds_rejected():
    issues = validate_video_generation_request(
        _valid_structured(width=2560, height=1441), config=CONFIG,
    )
    assert [i.path for i in issues] == ["height"]
    issues = validate_video_generation_request(
        _valid_structured(width=2560, height=1440), config=CONFIG,
    )
    assert issues == []  # exactly at the pixel cap is allowed
    issues = validate_video_generation_request(
        _valid_structured(width=1280, height=720, ratio="wide"), config=CONFIG,
    )
    assert [i.path for i in issues] == ["ratio"]


def test_ratio_adaptive_accepted():
    assert validate_video_generation_request(_valid_structured(ratio="adaptive"), config=CONFIG) == []


def test_steps_and_cfg_bounds():
    issues = validate_video_generation_request(
        _valid_structured(steps=51, cfg_scale=-1.0), config=CONFIG,
    )
    paths = {i.path for i in issues}
    assert paths == {"steps", "cfg_scale"}


def test_extra_params_allowlist_enforced():
    issues = validate_video_generation_request(
        _valid_structured(extra_params={"callback_url": "https://example.invalid/hook"}), config=CONFIG,
    )
    assert issues == []
    issues = validate_video_generation_request(
        _valid_structured(extra_params={"not_allowed": 1}), config=CONFIG,
    )
    assert [i.path for i in issues] == ["extra_params.not_allowed"]


def test_cli_args_must_be_list_when_allowlisted():
    issues = validate_video_generation_request(
        _valid_structured(backend="stable_diffusion_cpp", extra_params={"cli_args": "--unsafe"}),
        config=CONFIG,
    )
    assert [i.path for i in issues] == ["extra_params.cli_args"]


def test_reference_assets_valid_passes():
    assets = (
        _asset("first_frame", content=b"png-bytes"),
        _asset("reference_audio", content=b"wav-bytes", mime="audio/wav"),
    )
    assert validate_video_generation_request(
        _valid_structured(reference_assets=assets), config=CONFIG,
    ) == []


def test_reference_asset_mime_rejected():
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("first_frame", mime="image/gif"),)),
        config=CONFIG,
    )
    assert any(i.path == "reference_assets[0].mime_type" for i in issues)


def test_reference_asset_empty_and_oversize_rejected():
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("reference_video", content=b"", mime="video/mp4"),)),
        config=CONFIG,
    )
    assert any("no content bytes" in i.message for i in issues)

    big = b"x" * (REFERENCE_VIDEO_MAX_BYTES + 1)
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("reference_video", content=big, mime="video/mp4"),)),
        config=CONFIG,
    )
    assert any("50MB" in i.message for i in issues)

    big_img = b"x" * (REFERENCE_IMAGE_MAX_BYTES + 1)
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("first_frame", content=big_img),)),
        config=CONFIG,
    )
    assert any("30MB" in i.message for i in issues)

    big_audio = b"x" * (REFERENCE_AUDIO_MAX_BYTES + 1)
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("reference_audio", content=big_audio, mime="audio/mpeg"),)),
        config=CONFIG,
    )
    assert any("15MB" in i.message for i in issues)


def test_reference_asset_kind_counts_enforced():
    two_first_frames = (
        _asset("first_frame"),
        _asset("first_frame"),
    )
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=two_first_frames), config=CONFIG,
    )
    assert any(i.path == "reference_assets.first_frame" for i in issues)


def test_reference_asset_total_cap_enforced():
    class _SmallCap(_FakeConfig):
        max_reference_assets = 1

    assets = (_asset("first_frame"), _asset("last_frame"))
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=assets), config=_SmallCap(),
    )
    assert any(i.path == "reference_assets" and "1-asset limit" in i.message for i in issues)


def test_unknown_asset_kind_rejected():
    issues = validate_video_generation_request(
        _valid_structured(reference_assets=(_asset("storyboard"),)),
        config=CONFIG,
    )
    assert any(i.path == "reference_assets[0].kind" for i in issues)
