from enum import StrEnum

import pytest


def test_supported_video_format_records_are_exact():
    from tldw_chatbook.Video_Generation.video_formats import SUPPORTED_VIDEO_FORMATS

    assert SUPPORTED_VIDEO_FORMATS == frozenset(
        {
            ("mp4", "video/mp4", "mp4"),
            ("webm", "video/webm", "webm"),
        }
    )


@pytest.mark.parametrize(
    ("container", "mime", "extension"),
    [
        ("mp4", "video/mp4", "mp4"),
        ("webm", "video/webm", "webm"),
    ],
)
def test_video_format_helpers_return_exact_canonical_values(container, mime, extension):
    from tldw_chatbook.Video_Generation.video_formats import (
        canonical_video_extension,
        video_container_for_mime,
    )

    assert canonical_video_extension(container) == extension
    assert video_container_for_mime(mime) == container
    assert video_container_for_mime(f"  {mime.upper()} ; codecs=test  ") == container


@pytest.mark.parametrize(
    "value",
    ["mov", "mpeg4", ".mp4", "MP4", "", "mkv", None, b"mp4", ["mp4"], object()],
)
def test_canonical_video_extension_rejects_noncanonical_values(value):
    from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension

    with pytest.raises(ValueError, match="^unsupported video container$"):
        canonical_video_extension(value)


def test_canonical_video_extension_accepts_canonical_strenum_value():
    from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension

    class VideoFormat(StrEnum):
        MP4 = "mp4"

    assert canonical_video_extension(VideoFormat.MP4) == "mp4"


@pytest.mark.parametrize(
    "value",
    [
        "application/octet-stream",
        "video/quicktime",
        "video/x-matroska",
        "video/mp4-alias",
        "",
        None,
        b"video/mp4",
        ["video/mp4"],
        object(),
    ],
)
def test_video_container_for_mime_rejects_unknown_and_malformed_values(value):
    from tldw_chatbook.Video_Generation.video_formats import video_container_for_mime

    with pytest.raises(ValueError, match="^unsupported video MIME$"):
        video_container_for_mime(value)


@pytest.mark.parametrize("value", [None, b"video/mp4", ["video/mp4"], object()])
def test_normalize_video_mime_rejects_malformed_runtime_types(value):
    from tldw_chatbook.Video_Generation.video_formats import normalize_video_mime

    with pytest.raises(ValueError, match="^unsupported video MIME$"):
        normalize_video_mime(value)


def test_canonical_video_extension_rejects_equality_spoofers_before_comparison():
    from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension

    class EqualitySpoof:
        def __eq__(self, _other):
            return True

    with pytest.raises(ValueError, match="^unsupported video container$"):
        canonical_video_extension(EqualitySpoof())


def test_canonical_video_extension_contains_hostile_equality_errors():
    from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension

    class EqualityTrap:
        def __eq__(self, _other):
            raise RuntimeError("PRIVATE-EQUALITY-ERROR")

    with pytest.raises(ValueError, match="^unsupported video container$") as exc_info:
        canonical_video_extension(EqualityTrap())

    assert "PRIVATE-EQUALITY-ERROR" not in str(exc_info.value)
