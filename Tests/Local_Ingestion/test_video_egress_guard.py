"""TASK-19556 (b): the yt-dlp media arm adopts the egress policy.

At this branch's base `tldw_chatbook/Local_Ingestion/video_processing.py`
contained **zero** references to `Utils/egress.py`. Both of its yt-dlp
seams -- the pre-download size probe and the download itself in
`download_video`, and `extract_metadata` -- handed the caller's URL
straight to `yt_dlp.YoutubeDL(...).extract_info(url, ...)`.

The instructive contrast lives one module over:
`audio_processing.download_audio_file` routes its plain-HTTP branch
through `guarded_fetch_requests(url, trusted_origins=origin_set(url), ...)`.
That is the shape adopted here, so the media and article arms of the same
ingest entry point behave identically:

* a URL the user typed may resolve privately (an intranet media server is
  a legitimate ingest source, and `config.py`'s `[web_security]` contract
  says configured URLs may be private) -- so the entry URL is its own
  trusted origin, exactly as in the audio arm;
* a cloud metadata endpoint is refused **regardless** of that trust, which
  is `Utils/egress.py`'s one hard rule;
* a non-http(s) scheme is refused -- yt-dlp itself is happy to hand
  `file://` and dozens of other protocols to its extractors.

WHAT THIS DOES NOT COVER, stated plainly: yt-dlp performs its own HTTP
fetching. This is a pre-check on the entry URL only. It cannot re-validate
yt-dlp's own redirect hops, the per-format media URLs an extractor
discovers inside a page, or a DNS answer that changes between this check
and yt-dlp's own resolution (the same TOCTOU window `Utils/egress.py`
documents as a residual for every consumer). Closing those would require
either a yt-dlp request hook or refusing yt-dlp entirely; neither is in
this task's scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tldw_chatbook.Local_Ingestion import video_processing
from tldw_chatbook.Local_Ingestion.video_processing import (
    LocalVideoProcessor,
    VideoDownloadError,
)

METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
PRIVATE_URL = "http://10.255.255.1:8080/clip.mp4"
FILE_URL = "file:///etc/passwd"


class _RecordingYoutubeDL:
    """Records every URL `extract_info` is asked to fetch."""

    urls: List[str] = []
    constructions: List[Dict[str, Any]] = []

    def __init__(self, opts: Dict[str, Any]):
        self.opts = dict(opts)
        type(self).constructions.append(self.opts)

    def __enter__(self) -> "_RecordingYoutubeDL":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def extract_info(self, url: str, download: bool = False) -> Dict[str, Any]:
        type(self).urls.append(url)
        if download:
            Path(self.opts["_test_output"]).write_bytes(b"\x00" * 8)
        return {"title": "clip", "filesize": 1024, "uploader": "someone"}

    def prepare_filename(self, info: Dict[str, Any]) -> str:
        return str(self.opts["_test_output"])


@pytest.fixture
def recording_ytdlp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> List[str]:
    """Install the recording yt-dlp seam; return the fetched-URL log.

    yt-dlp is an optional dependency and absent from this venv, so the
    module's `yt_dlp` name is the seam. Nothing here opens a socket: the
    assertion is on which URLs *would* have been fetched.
    """
    _RecordingYoutubeDL.urls = []
    _RecordingYoutubeDL.constructions = []
    output = tmp_path / "clip.mp4"

    class _Module:
        @staticmethod
        def YoutubeDL(opts: Dict[str, Any]) -> _RecordingYoutubeDL:  # noqa: N802
            return _RecordingYoutubeDL({**opts, "_test_output": str(output)})

    monkeypatch.setattr(video_processing, "yt_dlp", _Module, raising=False)
    monkeypatch.setattr(video_processing, "YT_DLP_AVAILABLE", True)
    return _RecordingYoutubeDL.urls


# ---------------------------------------------------------------------------
# download_video
# ---------------------------------------------------------------------------


def test_download_video_refuses_a_cloud_metadata_endpoint(
    tmp_path: Path, recording_ytdlp: List[str]
) -> None:
    """The hard rule: metadata IPs are blocked even for a trusted origin."""
    processor = LocalVideoProcessor(None)
    with pytest.raises(VideoDownloadError):
        processor.download_video(METADATA_URL, str(tmp_path))
    assert recording_ytdlp == [], (
        f"an unchecked URL reached yt-dlp: {recording_ytdlp}"
    )


def test_download_video_refuses_a_non_http_scheme(
    tmp_path: Path, recording_ytdlp: List[str]
) -> None:
    """yt-dlp handles many protocols; the ingest entry point should not."""
    processor = LocalVideoProcessor(None)
    with pytest.raises(VideoDownloadError):
        processor.download_video(FILE_URL, str(tmp_path))
    assert recording_ytdlp == []


def test_download_video_still_allows_a_user_typed_private_url(
    tmp_path: Path, recording_ytdlp: List[str]
) -> None:
    """Parity with the audio arm: a typed URL is its own trusted origin.

    Without this the guard would break intranet media ingest, which
    `config.py`'s `[web_security]` contract explicitly permits.
    """
    processor = LocalVideoProcessor(None)
    processor.download_video(PRIVATE_URL, str(tmp_path), download_video_flag=True)
    assert recording_ytdlp == [PRIVATE_URL, PRIVATE_URL]  # probe + download


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


def test_extract_metadata_refuses_a_cloud_metadata_endpoint(
    recording_ytdlp: List[str],
) -> None:
    processor = LocalVideoProcessor(None)
    assert processor.extract_metadata(METADATA_URL) is None
    assert recording_ytdlp == []


def test_extract_metadata_still_allows_a_user_typed_private_url(
    recording_ytdlp: List[str],
) -> None:
    processor = LocalVideoProcessor(None)
    assert processor.extract_metadata(PRIVATE_URL) is not None
    assert recording_ytdlp == [PRIVATE_URL]


# ---------------------------------------------------------------------------
# Adoption pin (mutation guard)
# ---------------------------------------------------------------------------


def test_video_processing_references_the_egress_policy() -> None:
    """Deleting the guard must not merely change behaviour silently.

    The task's own finding was phrased as "contains zero references to the
    egress helpers"; this keeps that phrasing checkable.
    """
    source = Path(video_processing.__file__).read_text(encoding="utf-8")
    assert "check_url_or_raise" in source
    assert "origin_set" in source
