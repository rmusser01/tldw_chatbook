"""Cookie handling in the yt-dlp download path (task-3306 xhigh review round).

Three defects live here, all introduced or exposed when the ingest form's
``Cookies file for gated URLs`` field started routing a USER-OWNED path into
``download_video(cookies=...)``:

* **The cleanup deleted the user's file.** ``download_video``'s ``finally``
  unlinked any ``cookiefile`` whose path merely *started with*
  ``tempfile.gettempdir()``. That heuristic was safe while the key could only
  hold a temp file the function itself had written; once a user path could
  land in the same key, exporting cookies to ``/tmp/cookies.txt`` meant the
  file was destroyed by the first import (the unlink is swallowed by
  ``except (OSError, FileNotFoundError)``), and every later gated import
  failed with "Invalid cookie format".
* **The cookies never reached the probe.** The file-size probe immediately
  before the download built a bare ``YoutubeDL({"quiet": True})``, and
  ``extract_metadata`` ignored the ``use_cookies``/``cookies`` arguments it
  declares -- so an authentication-gated URL, the option's only reason to
  exist, failed before the cookied download ever ran.
* **A missing file degraded silently.** A non-existent path fell through to
  ``json.loads``, logged "Invalid cookie format" at WARNING and continued
  without cookies.

yt-dlp is an optional dependency and is absent from this venv, so the module's
``yt_dlp`` seam is stubbed; every assertion is on the options the real
``YoutubeDL`` would have been constructed with.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tldw_chatbook.Local_Ingestion import video_processing
from tldw_chatbook.Local_Ingestion.video_processing import (
    LocalVideoProcessor,
    VideoDownloadError,
)


class _FakeYoutubeDL:
    """Records the options each ``YoutubeDL(...)`` was constructed with."""

    constructions: List[Dict[str, Any]] = []

    def __init__(self, opts: Dict[str, Any]):
        self.opts = dict(opts)
        type(self).constructions.append(self.opts)

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def extract_info(self, url: str, download: bool = False) -> Dict[str, Any]:
        if download:
            Path(self.opts["_test_output"]).write_bytes(b"\x00" * 8)
        return {"title": "clip", "filesize": 1024, "uploader": "someone"}

    def prepare_filename(self, info: Dict[str, Any]) -> str:
        return str(self.opts["_test_output"])


@pytest.fixture
def fake_ytdlp(monkeypatch, tmp_path: Path):
    """Install the stub yt-dlp seam and return the construction log."""
    _FakeYoutubeDL.constructions = []
    output = tmp_path / "clip.mp4"

    class _Module:
        YoutubeDL = _FakeYoutubeDL

    def _construct(opts):
        return _FakeYoutubeDL({**opts, "_test_output": str(output)})

    _Module.YoutubeDL = _construct  # type: ignore[assignment]
    monkeypatch.setattr(video_processing, "yt_dlp", _Module, raising=False)
    monkeypatch.setattr(video_processing, "YT_DLP_AVAILABLE", True)
    # (TASK-19556) The download seam now consults the egress policy before
    # yt-dlp, which for the `example.com` URL these tests use would mean a
    # real DNS lookup -- and an offline machine would fail them for the
    # wrong reason. These tests are about cookie handling; the guard itself
    # is owned by Tests/Local_Ingestion/test_video_egress_guard.py.
    monkeypatch.setattr(video_processing, "check_url_or_raise", lambda *a, **k: None)
    return _FakeYoutubeDL.constructions


@pytest.fixture
def user_cookiefile_in_tmpdir():
    """A user-owned cookies file living directly in the system temp dir.

    ``/tmp/cookies.txt`` is exactly where a browser cookie exporter drops
    its output, and it is the path the old prefix heuristic destroyed.
    pytest's ``tmp_path`` is NOT usable here: on macOS it resolves under
    ``/private/var/...`` while ``tempfile.gettempdir()`` reports
    ``/var/...``, so the old heuristic would not even have fired and the
    regression test would have been vacuous.
    """
    path = Path(tempfile.gettempdir()) / f"tldw-user-cookies-{uuid.uuid4().hex}.txt"
    path.write_text("# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\n")
    assert str(path).startswith(tempfile.gettempdir())
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def _download(tmp_path: Path, **kwargs: Any) -> Any:
    processor = LocalVideoProcessor(None)
    return processor.download_video(
        "https://example.com/watch?v=x",
        str(tmp_path),
        download_video_flag=True,
        **kwargs,
    )


class TestUserCookiefileIsNotDestroyed:
    def test_user_supplied_cookiefile_in_the_temp_dir_survives(
        self, tmp_path: Path, fake_ytdlp, user_cookiefile_in_tmpdir: Path
    ):
        _download(
            tmp_path, use_cookies=True, cookies=str(user_cookiefile_in_tmpdir)
        )

        assert user_cookiefile_in_tmpdir.exists(), (
            "download_video deleted a cookies file it did not create"
        )
        assert user_cookiefile_in_tmpdir.read_text().startswith("# Netscape")

    def test_user_cookiefile_survives_repeated_imports(
        self, tmp_path: Path, fake_ytdlp, user_cookiefile_in_tmpdir: Path
    ):
        """The reported symptom: the SECOND gated import is the one that
        fails, because the first silently removed the file."""
        for _ in range(2):
            _download(
                tmp_path, use_cookies=True, cookies=str(user_cookiefile_in_tmpdir)
            )
        assert user_cookiefile_in_tmpdir.exists()

    def test_self_created_temp_cookiefile_is_still_cleaned_up(
        self, tmp_path: Path, fake_ytdlp
    ):
        """The cleanup must not be dropped -- only narrowed to our own file."""
        _download(tmp_path, use_cookies=True, cookies={"session": "abc"})

        written = [
            opts["cookiefile"] for opts in fake_ytdlp if "cookiefile" in opts
        ]
        assert written, "a cookiefile was never produced from the cookie dict"
        for path in set(written):
            assert not os.path.exists(path), (
                f"temp cookiefile {path} this function created was left behind"
            )


class TestCookiesReachEveryYtDlpCall:
    def test_cookies_reach_the_file_size_probe(
        self, tmp_path: Path, fake_ytdlp, user_cookiefile_in_tmpdir: Path
    ):
        """The probe runs BEFORE the download; an auth-gated URL fails there
        first, so cookies that only reach the download are useless."""
        _download(
            tmp_path, use_cookies=True, cookies=str(user_cookiefile_in_tmpdir)
        )

        assert fake_ytdlp, "no YoutubeDL was constructed"
        assert all("cookiefile" in opts for opts in fake_ytdlp), (
            "some yt-dlp call ran without the cookies: "
            f"{[sorted(o) for o in fake_ytdlp]}"
        )
        assert fake_ytdlp[0]["cookiefile"] == str(user_cookiefile_in_tmpdir)

    def test_extract_metadata_uses_the_cookies_it_declares(
        self, fake_ytdlp, user_cookiefile_in_tmpdir: Path
    ):
        processor = LocalVideoProcessor(None)
        processor.extract_metadata(
            "https://example.com/watch?v=x",
            use_cookies=True,
            cookies=str(user_cookiefile_in_tmpdir),
        )

        assert fake_ytdlp, "extract_metadata constructed no YoutubeDL"
        assert fake_ytdlp[0].get("cookiefile") == str(user_cookiefile_in_tmpdir)

    def test_extract_metadata_without_cookies_sends_none(self, fake_ytdlp):
        processor = LocalVideoProcessor(None)
        processor.extract_metadata("https://example.com/watch?v=x")

        assert fake_ytdlp
        assert "cookiefile" not in fake_ytdlp[0]

    def test_metadata_temp_cookiefile_is_cleaned_up(self, fake_ytdlp):
        processor = LocalVideoProcessor(None)
        processor.extract_metadata(
            "https://example.com/watch?v=x",
            use_cookies=True,
            cookies={"session": "abc"},
        )

        path = fake_ytdlp[0]["cookiefile"]
        assert not os.path.exists(path)


class TestMissingCookiefileIsHonest:
    def test_nonexistent_cookiefile_names_the_file(
        self, tmp_path: Path, fake_ytdlp
    ):
        """A path that does not exist used to be handed to ``json.loads``,
        logged as "Invalid cookie format" and then ignored -- the download
        proceeded un-authenticated and failed later for an unrelated-looking
        reason."""
        missing = tmp_path / "not-there" / "cookies.txt"

        with pytest.raises(VideoDownloadError) as excinfo:
            _download(tmp_path, use_cookies=True, cookies=str(missing))

        assert "cookies.txt" in str(excinfo.value)
        assert not fake_ytdlp, "yt-dlp ran despite the cookies being unusable"

    def test_cookie_json_string_still_works(self, tmp_path: Path, fake_ytdlp):
        """The JSON-string spelling is a real (if undocumented) input; the
        stricter path must not break it."""
        _download(tmp_path, use_cookies=True, cookies='{"session": "abc"}')

        assert fake_ytdlp
        assert "cookiefile" in fake_ytdlp[0]
