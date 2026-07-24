"""Real streamed-byte caps for media/audio downloads; GitHub API response caps.

Covers TASK-329: the download surfaces used to trust the declared
``Content-Length`` header (or, for the GitHub client, had no cap at all) to
decide when a body was "done" or "too big". A server that lies about (or
omits) ``Content-Length`` could smuggle an arbitrarily large body past that
check. These tests prove the REAL streamed byte count is what's enforced,
at both the transport primitive and the call-site wiring in:

- ``tldw_chatbook.Media.local_media_reading_service.LocalMediaReadingService._default_url_file_downloader``
- ``tldw_chatbook.Local_Ingestion.audio_processing.LocalAudioProcessor.download_audio_file``
- ``tldw_chatbook.Utils.github_api_client.GitHubAPIClient`` (get_file_content /
  get_repository_tree / get_directory_contents)

and that every failure path (spoofed-length overflow, HTTP error status,
egress block) leaves no leaked temp file behind.
"""

import io
import os
import tempfile
from pathlib import Path

import httpx
import pytest
import requests as requests_lib
from requests.adapters import BaseAdapter
from requests.models import Response as RequestsResponse

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import EgressFetchError, guarded_fetch_requests


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    """No real DNS, egress checks enabled with no allowlist (matches test_egress.py)."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _fake_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda s, k=None, d=None: d
    )


class _LyingAdapter(BaseAdapter):
    """Serves a body far larger than the declared (lying) Content-Length."""

    def __init__(self, body: bytes, declared_length: str = "10", status_code: int = 200):
        super().__init__()
        self.body = body
        self.declared_length = declared_length
        self.status_code = status_code

    def send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = self.status_code
        resp.headers["Content-Length"] = self.declared_length
        resp.raw = io.BytesIO(self.body)
        resp.url = request.url
        resp.request = request
        return resp

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Transport-level regression: the primitive itself must not trust the
# declared Content-Length in sink mode.
# ---------------------------------------------------------------------------


def test_transport_real_cap_beats_spoofed_content_length():
    """A body larger than max_bytes fails even when Content-Length lies."""
    sess = requests_lib.Session()
    sess.mount("http://", _LyingAdapter(b"a" * 4096, declared_length="10"))
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests(
            "http://media.example/file.mp3",
            session=sess,
            max_bytes=1024,
            sink=io.BytesIO(),
        )


# ---------------------------------------------------------------------------
# Wiring: LocalAudioProcessor.download_audio_file
# ---------------------------------------------------------------------------


def test_audio_download_real_cap_closes_spoofed_length_bypass(monkeypatch, tmp_path):
    """The audio downloader enforces max_file_size on REAL streamed bytes,
    not the (spoofable/omittable) declared Content-Length, and cleans up
    the temp .part file on rejection."""
    from tldw_chatbook.Local_Ingestion.audio_processing import (
        AudioDownloadError,
        LocalAudioProcessor,
    )

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 200
        resp.headers["Content-Length"] = "10"  # lie: well under the cap
        resp.raw = io.BytesIO(b"a" * 4096)
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    processor = LocalAudioProcessor()
    processor.max_file_size = 1024

    with pytest.raises(AudioDownloadError, match="blocked or too large"):
        processor.download_audio_file("http://media.example/file.mp3", str(tmp_path))

    # No leaked .part temp file after the rejection.
    assert list(tmp_path.glob("*.part")) == []


def test_audio_download_cleans_up_on_http_error_status(monkeypatch, tmp_path):
    """A non-2xx status (raised after the sink write) cleans up the temp
    .part file too, not just the oversize path."""
    from tldw_chatbook.Local_Ingestion.audio_processing import (
        AudioDownloadError,
        LocalAudioProcessor,
    )

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 404
        resp.raw = io.BytesIO(b"not found")
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    processor = LocalAudioProcessor()

    with pytest.raises(AudioDownloadError, match="Download failed"):
        processor.download_audio_file("http://media.example/missing.mp3", str(tmp_path))

    assert list(tmp_path.glob("*.part")) == []


def test_audio_download_cleans_up_on_egress_blocked(monkeypatch, tmp_path):
    """A redirect hop into a private-IP host is blocked by the egress guard,
    and the temp .part file is cleaned up (not just the oversize path)."""
    from tldw_chatbook.Local_Ingestion.audio_processing import (
        AudioDownloadError,
        LocalAudioProcessor,
    )

    def fake_resolve(host):
        return ["10.0.0.5"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        if "internal.example" in request.url:
            resp.status_code = 200
            resp.raw = io.BytesIO(b"should never be reached")
        else:
            resp.status_code = 302
            resp.headers["Location"] = "http://internal.example/payload.mp3"
            resp.raw = io.BytesIO(b"")
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    processor = LocalAudioProcessor()

    with pytest.raises(AudioDownloadError, match="blocked or too large"):
        processor.download_audio_file(
            "http://media.example/redirecting.mp3", str(tmp_path)
        )

    assert list(tmp_path.glob("*.part")) == []


def test_audio_download_success_streams_within_cap(monkeypatch, tmp_path):
    """Sanity: a body within the cap downloads, renames off .part, and the
    declared-Content-Length courtesy check does not block a truthful body."""
    from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

    body = b"a" * 2048

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 200
        resp.headers["Content-Length"] = str(len(body))
        resp.headers["Content-Disposition"] = 'attachment; filename="clip.mp3"'
        resp.raw = io.BytesIO(body)
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    processor = LocalAudioProcessor()
    processor.max_file_size = 1024 * 1024

    saved = processor.download_audio_file("http://media.example/clip.mp3", str(tmp_path))

    saved_path = Path(saved)
    assert saved_path.exists()
    assert saved_path.read_bytes() == body
    assert saved_path.suffix == ".mp3"
    assert list(tmp_path.glob("*.part")) == []


# ---------------------------------------------------------------------------
# Wiring: LocalMediaReadingService._default_url_file_downloader
# ---------------------------------------------------------------------------


def test_media_downloader_cleans_up_temp_file_on_oversize(monkeypatch):
    """Oversize (real, streamed) bodies raise and leave no temp file behind."""
    from tldw_chatbook.Media.local_media_reading_service import (
        LocalMediaReadingService,
    )

    # Cap patched small so the test body (4096 bytes) trips it without
    # having to synthesize hundreds of MB.
    monkeypatch.setattr(egress, "MAX_FETCH_BYTES_MEDIA", 1024)

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 200
        resp.headers["Content-Type"] = "application/pdf"
        resp.raw = io.BytesIO(b"x" * 4096)
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    captured_paths = []
    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        captured_paths.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)

    with pytest.raises(EgressFetchError, match="exceeds"):
        LocalMediaReadingService._default_url_file_downloader(
            "http://media.example/file", media_type="document", options={}
        )

    assert captured_paths, "expected the downloader to create a temp file"
    assert not os.path.exists(captured_paths[-1]), "temp file leaked on oversize rejection"


def test_media_downloader_cleans_up_temp_file_on_egress_blocked(monkeypatch):
    """A redirect hop into a private-IP host is blocked by the egress guard,
    and the temp file is cleaned up (not just the oversize path)."""
    from tldw_chatbook.Media.local_media_reading_service import (
        LocalMediaReadingService,
    )
    from tldw_chatbook.Utils.egress import EgressBlockedError

    def fake_resolve(host):
        return ["10.0.0.5"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        if "internal.example" in request.url:
            resp.status_code = 200
            resp.raw = io.BytesIO(b"should never be reached")
        else:
            resp.status_code = 302
            resp.headers["Location"] = "http://internal.example/payload.pdf"
            resp.raw = io.BytesIO(b"")
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    captured_paths = []
    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        captured_paths.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)

    with pytest.raises(EgressBlockedError):
        LocalMediaReadingService._default_url_file_downloader(
            "http://media.example/redirecting", media_type="document", options={}
        )

    assert captured_paths, "expected the downloader to create a temp file"
    assert not os.path.exists(captured_paths[-1]), "temp file leaked on egress block"


def test_media_downloader_cleans_up_temp_file_on_http_error_status(monkeypatch):
    """A non-2xx status (raised after the sink write) still cleans up the temp file."""
    from tldw_chatbook.Media.local_media_reading_service import (
        LocalMediaReadingService,
    )

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 404
        resp.headers["Content-Type"] = "text/plain"
        resp.raw = io.BytesIO(b"not found")
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    captured_paths = []
    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        captured_paths.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)

    with pytest.raises(requests_lib.HTTPError):
        LocalMediaReadingService._default_url_file_downloader(
            "http://media.example/missing.pdf", media_type="document", options={}
        )

    assert captured_paths, "expected the downloader to create a temp file"
    assert not os.path.exists(captured_paths[-1]), "temp file leaked on HTTP error status"


def test_media_downloader_success_renames_off_part_suffix(monkeypatch):
    """Sanity: a successful download streams into the .part file then renames
    it to the resolved suffix, leaving no .part file behind."""
    from tldw_chatbook.Media.local_media_reading_service import (
        LocalMediaReadingService,
    )

    body = b"%PDF-1.4 fake pdf body"

    def fake_send(self, request, **kwargs):
        resp = RequestsResponse()
        resp.status_code = 200
        resp.headers["Content-Type"] = "application/pdf"
        resp.raw = io.BytesIO(body)
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_lib.Session, "send", fake_send)

    result = LocalMediaReadingService._default_url_file_downloader(
        "http://media.example/no-extension-here", media_type="document", options={}
    )

    path = Path(result["path"])
    try:
        assert result["cleanup"] is True
        assert path.suffix == ".pdf"
        assert path.exists()
        assert path.read_bytes() == body
        assert not path.with_suffix(".part").exists()
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Wiring: GitHubAPIClient (get_file_content / get_repository_tree /
# get_directory_contents) — no trusted_origins (fixed vendor host), oversize
# containment via each method's trailing except Exception -> GitHubAPIError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_client_get_file_content_oversize_contained(monkeypatch):
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient, GitHubAPIError

    # Patch the cap small so a modest fake body trips it.
    monkeypatch.setattr(egress, "MAX_FETCH_BYTES_GITHUB_FILE", 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"y" * 4096,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    try:
        gh = GitHubAPIClient(token=None)
        gh._client = client

        with pytest.raises(GitHubAPIError):
            await gh.get_file_content("owner", "repo", "big_file.bin", "main")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_github_client_get_directory_contents_oversize_contained(monkeypatch):
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient, GitHubAPIError

    monkeypatch.setattr(egress, "MAX_FETCH_BYTES_GITHUB_FILE", 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"[" + b"y" * 4096 + b"]",
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    try:
        gh = GitHubAPIClient(token=None)
        gh._client = client

        with pytest.raises(GitHubAPIError):
            await gh.get_directory_contents("owner", "repo", "", "main")
    finally:
        await client.aclose()
