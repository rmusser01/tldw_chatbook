"""TASK-20973: a media URL vouches for itself only when provenance says so.

TASK-19556 put the egress policy in front of the two yt-dlp seams as
``check_url_or_raise(url, trusted_origins=origin_set(url))`` -- the URL is
its own trusted origin. That is correct **for a URL the user typed into
the ingest form** (an intranet media server is a legitimate source), but
the property held by wiring, not by invariant: two public, test-covered
methods -- ``LocalMediaReadingService.process_video(urls=...)`` and
``MediaReadingScopeService.process_video(urls=...)`` -- accepted
caller-supplied URLs and forwarded them into that self-trusting check, so
any future wiring (config, API payload, feed, agent tool) would silently
convert the guard into no guard.

This module pins the provenance contract:

* the trust decision is an explicit ``UrlProvenance`` parameter at every
  seam, defaulting to ``UNKNOWN`` (fail closed) -- a URL of unknown
  provenance is NOT its own trusted origin;
* both public ``process_video`` seams therefore cannot grant self-trust to
  a caller-supplied URL, which is the property that did not hold before;
* the user-entered ingest path still self-trusts a private/intranet URL
  (pins TASK-19556's legitimate case). ``USER_ENTERED`` is minted where
  user entry is a fact -- the Library ingest queue derives it from the
  job's own lineage -- never from "who happens to call";
* the minted enum rides the pickled parse-``options`` dict across the
  spawn pool and is translated back by ``run_parse_job``, which accepts
  the enum ONLY -- a plain string in ``options`` cannot launder trust.

Every test that executes the real processor/parse chain is wrapped in
``@private_profile_test`` (the ``test_youtube_stt_selection.py`` pattern
for this same chain): these constructs read live config, and under the
default per-test sandbox that trips the profile-selection recovery guard
for environmental reasons unrelated to this task.

WHAT THIS DOES NOT CLOSE (unchanged residuals, stated by TASK-19556 and
still true): yt-dlp performs its own fetching. The egress call is a
pre-check on the entry URL only; it cannot re-validate yt-dlp's own
redirect hops, the per-format media URLs an extractor discovers inside a
page, or a DNS answer that changes between the check and yt-dlp's own
resolution (the resolve-then-connect TOCTOU window ``Utils/egress.py``
documents). This task changes WHO may vouch for the entry URL, not what
happens after yt-dlp starts. The adjacent audio arm
(``audio_processing.download_audio_file``'s own ``origin_set(url)`` call,
reached via the ``process_audio`` seams) keeps its TASK-19556 shape and is
out of scope here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Local_Ingestion import video_processing
from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
)
from tldw_chatbook.Local_Ingestion.video_processing import (
    LocalVideoProcessor,
    VideoDownloadError,
)
from tldw_chatbook.Utils.egress import UrlProvenance

#: RFC1918 IP literal: classifies as "private" by the egress policy with
#: no DNS lookup (the suite forbids real sockets), and its ``.mp4`` path
#: makes ``classify_ingest_source`` route it to the video branch.
PRIVATE_URL = "http://10.255.255.1:8080/clip.mp4"
METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


class _RecordingYoutubeDL:
    """Records every URL ``extract_info`` is asked to fetch."""

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
    module's ``yt_dlp`` name is the seam. Nothing here opens a socket: the
    assertion target is which URLs *would* have been fetched, and every
    blocked target is an IP literal (no DNS).
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


@pytest.fixture
def stub_av_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Make the post-download tail (ffmpeg + STT) succeed without binaries.

    Only needed by the tests that must prove the FULL user-entered chain
    succeeds (the intranet-ingest pin). The blocked-path tests never get
    past ``download_video``, so they do not need this.
    """
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3\x00" + b"\x00" * 32)

    monkeypatch.setattr(
        LocalVideoProcessor,
        "_extract_audio_from_video",
        lambda self, video_path, output_dir, start_time=None, end_time=None: str(audio),
    )

    def fake_single_audio(self, input_item, processing_dir, **kwargs):
        return {
            "status": "Success",
            "input_ref": input_item,
            "content": "transcript of the clip",
            "metadata": {"title": "clip"},
            "segments": [],
            "chunks": [],
            "analysis": "",
            "warnings": [],
        }

    monkeypatch.setattr(
        LocalAudioProcessor, "_process_single_audio", fake_single_audio
    )
    return audio


# ---------------------------------------------------------------------------
# The decision point: check_media_url_egress / download_video / extract_metadata
# ---------------------------------------------------------------------------


@private_profile_test
def test_check_media_url_egress_does_not_self_trust_by_default(
    request, recording_ytdlp: List[str]
) -> None:
    """Unknown provenance must not vouch for itself (the AC-1 core).

    At the branch base this passes the private URL through, because the
    function computed ``trusted_origins=origin_set(url)`` unconditionally.
    """
    with pytest.raises(VideoDownloadError):
        video_processing.check_media_url_egress(PRIVATE_URL)
    assert recording_ytdlp == []


@private_profile_test
def test_check_media_url_egress_self_trusts_only_with_user_entered_provenance(
    request, recording_ytdlp: List[str]
) -> None:
    video_processing.check_media_url_egress(
        PRIVATE_URL, url_provenance=UrlProvenance.USER_ENTERED
    )


@private_profile_test
def test_check_media_url_egress_blocks_metadata_even_when_user_entered(
    request,
) -> None:
    """The policy's one hard rule survives the provenance parameter."""
    with pytest.raises(VideoDownloadError):
        video_processing.check_media_url_egress(
            METADATA_URL, url_provenance=UrlProvenance.USER_ENTERED
        )


@private_profile_test
def test_download_video_default_blocks_private_and_user_entered_allows(
    request, tmp_path: Path, recording_ytdlp: List[str]
) -> None:
    processor = LocalVideoProcessor(None)
    with pytest.raises(VideoDownloadError):
        processor.download_video(PRIVATE_URL, str(tmp_path))
    assert recording_ytdlp == []

    processor.download_video(
        PRIVATE_URL, str(tmp_path), url_provenance=UrlProvenance.USER_ENTERED
    )
    assert recording_ytdlp == [PRIVATE_URL, PRIVATE_URL]  # probe + download


@private_profile_test
def test_extract_metadata_default_blocks_private_and_user_entered_allows(
    request, recording_ytdlp: List[str]
) -> None:
    processor = LocalVideoProcessor(None)
    assert processor.extract_metadata(PRIVATE_URL) is None
    assert recording_ytdlp == []

    metadata = processor.extract_metadata(
        PRIVATE_URL, url_provenance=UrlProvenance.USER_ENTERED
    )
    assert metadata is not None
    assert recording_ytdlp == [PRIVATE_URL]


@private_profile_test
def test_process_videos_threads_provenance_to_the_egress_check(
    request, recording_ytdlp: List[str], stub_av_tail: Path
) -> None:
    """The batch entry the services call honors the same contract."""
    processor = LocalVideoProcessor(None)
    blocked = processor.process_videos(inputs=[PRIVATE_URL])
    assert blocked["results"][0]["status"] == "Error"
    assert "egress" in blocked["results"][0]["error"]
    assert recording_ytdlp == []

    allowed = processor.process_videos(
        inputs=[PRIVATE_URL], url_provenance=UrlProvenance.USER_ENTERED
    )
    assert allowed["results"][0]["status"] != "Error"
    # metadata + the download's probe + the download itself
    assert recording_ytdlp == [PRIVATE_URL] * 3


# ---------------------------------------------------------------------------
# The two public seams (TASK-20973's subject)
# ---------------------------------------------------------------------------


def _real_video_local_service():
    from tldw_chatbook.Media.local_media_reading_service import (
        LocalMediaReadingService,
    )

    return LocalMediaReadingService(
        None,
        video_processor_factory=lambda: LocalVideoProcessor(None),
    )


class _AllowAllPolicy:
    def require_allowed(self, *, action_id: str) -> None:
        return None


@private_profile_test
def test_local_service_process_video_does_not_self_trust_a_caller_supplied_url(
    request, recording_ytdlp: List[str]
) -> None:
    """THE born-red seam test: a hypothetical future caller of
    ``LocalMediaReadingService.process_video(urls=...)`` -- a config, an
    API payload, a feed, an agent tool -- must not get self-trust.

    At the branch base the private URL REACHED yt-dlp through this seam
    (probe + download), which is exactly the property this task exists to
    remove.
    """
    service = _real_video_local_service()
    result = service.process_video(urls=[PRIVATE_URL])
    assert result["results"][0]["status"] == "Error"
    assert "egress" in result["results"][0]["error"]
    assert recording_ytdlp == [], (
        f"a caller-supplied URL of unknown provenance reached yt-dlp: "
        f"{recording_ytdlp}"
    )


@private_profile_test
async def test_scope_service_process_video_does_not_self_trust_a_caller_supplied_url(
    request, recording_ytdlp: List[str]
) -> None:
    """Same property through the outer public seam."""
    from tldw_chatbook.Media.media_reading_scope_service import (
        MediaReadingScopeService,
    )

    scope = MediaReadingScopeService(
        local_service=_real_video_local_service(),
        server_service=None,
        policy_enforcer=_AllowAllPolicy(),
    )
    result = await scope.process_video(mode="local", urls=[PRIVATE_URL])
    assert result["results"][0]["status"] == "Error"
    assert "egress" in result["results"][0]["error"]
    assert recording_ytdlp == [], (
        f"a caller-supplied URL of unknown provenance reached yt-dlp via "
        f"the scope seam: {recording_ytdlp}"
    )


@private_profile_test
def test_local_service_seam_threads_user_entered_provenance_explicitly(
    request, recording_ytdlp: List[str], stub_av_tail: Path
) -> None:
    """A caller MAY establish provenance at the seam -- and the trusted
    variant then reaches the egress check through the whole chain."""
    service = _real_video_local_service()
    result = service.process_video(
        urls=[PRIVATE_URL], url_provenance=UrlProvenance.USER_ENTERED
    )
    assert result["results"][0]["status"] != "Error"
    # metadata + the download's probe + the download itself
    assert recording_ytdlp == [PRIVATE_URL] * 3


@private_profile_test
async def test_scope_service_seam_threads_user_entered_provenance_to_local_only(
    request,
) -> None:
    """Provenance threads through the scope seam to the LOCAL backend; the
    SERVER backend never sees it -- the server fetches in another process
    whose own policy this parameter cannot reach, and a provenance claim
    crossing a process boundary is unverifiable anyway."""
    from tldw_chatbook.Media.media_reading_scope_service import (
        MediaReadingScopeService,
    )

    class _FakeLocal:
        def __init__(self) -> None:
            self.calls: list = []

        def process_video(self, **kwargs):
            self.calls.append(kwargs)
            return {"results": [{"media_type": "video"}]}

    class _FakeServer:
        def __init__(self) -> None:
            self.calls: list = []

        async def process_video(self, request_data=None, *, file_paths=None, **kwargs):
            self.calls.append(kwargs)
            return {"results": [{"media_type": "video"}]}

    local = _FakeLocal()
    server = _FakeServer()
    scope = MediaReadingScopeService(
        local_service=local, server_service=server, policy_enforcer=_AllowAllPolicy()
    )

    await scope.process_video(
        mode="local", urls=[PRIVATE_URL], url_provenance=UrlProvenance.USER_ENTERED
    )
    await scope.process_video(
        mode="server", urls=[PRIVATE_URL], url_provenance=UrlProvenance.USER_ENTERED
    )

    assert local.calls[0]["url_provenance"] is UrlProvenance.USER_ENTERED
    assert "url_provenance" not in server.calls[0], (
        "provenance must not cross into the server backend's request"
    )


# ---------------------------------------------------------------------------
# The ingest boundary: where USER_ENTERED is minted (and nowhere else)
# ---------------------------------------------------------------------------


def _queue_app():
    """A bare app instance for ``_ingest_job_options`` (analysis-off jobs
    read no live app state) -- the ``object.__new__`` pattern from
    ``Tests/App/test_submit_library_ingest_job.py``.

    Imported lazily so this module's collection stays light and the import
    happens inside whichever profile the caller runs under.
    """
    import tldw_chatbook.app as app_module

    return object.__new__(app_module.TldwCli)


@private_profile_test
def test_ingest_queue_mints_user_entered_for_general_submissions(request) -> None:
    """The Library import form's submissions carry USER_ENTERED -- minted
    from the job's own lineage at dispatch, not from who happens to call."""
    from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob

    job = LibraryIngestJob(
        job_id="ingest-job-provenance-general",
        source_path=PRIVATE_URL,
        ingest_options={"generic": {"analyze": False, "chunk": False}},
    )
    options = _queue_app()._ingest_job_options(job)
    assert options["url_provenance"] is UrlProvenance.USER_ENTERED


@private_profile_test
def test_ingest_queue_mints_unknown_for_research_source_jobs(request) -> None:
    """Research-source URLs are agent-discovered catalog content, not user
    entry: they ride the same parse pipeline and must NOT get self-trust
    (the live non-user-entered source that shares the pipeline today)."""
    from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob

    job = LibraryIngestJob(
        job_id="ingest-job-provenance-research",
        source_path=PRIVATE_URL,
        ingest_options={"generic": {"analyze": False, "chunk": False}},
        research_source_operation_id="op-1",
    )
    options = _queue_app()._ingest_job_options(job)
    assert options["url_provenance"] is UrlProvenance.UNKNOWN


@private_profile_test
def test_parse_seam_fails_closed_without_provenance(
    request, recording_ytdlp: List[str]
) -> None:
    """``parse_local_file_for_ingest`` is a public function too: without an
    explicit provenance it must not self-trust the URL it is handed. The
    parse pipeline surfaces the processor's refusal as a
    ``FileIngestionError`` (its per-type error contract), so assert the
    raise plus the yt-dlp log."""
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        FileIngestionError,
    )

    with pytest.raises(FileIngestionError, match="egress"):
        parse_local_file_for_ingest(PRIVATE_URL, {})
    assert recording_ytdlp == []


@private_profile_test
def test_parse_seam_still_ingests_a_user_entered_private_url(
    request, recording_ytdlp: List[str], stub_av_tail: Path
) -> None:
    """Pin: the user-entered ingest of an intranet media URL still works
    end to end -- classify -> metadata -> download -> transcript."""
    payload = parse_local_file_for_ingest(
        PRIVATE_URL, {}, url_provenance=UrlProvenance.USER_ENTERED
    )
    assert payload["content"] == "transcript of the clip"
    # metadata + the download's probe + the download itself
    assert recording_ytdlp == [PRIVATE_URL] * 3


@private_profile_test
def test_run_parse_job_translates_provenance_from_options_as_the_enum_only(
    request, recording_ytdlp: List[str], stub_av_tail: Path
) -> None:
    """The pickled options dict is the transport across the spawn pool;
    ``run_parse_job`` translates it back to the explicit parameter and
    accepts the enum ONLY -- a plain string cannot launder trust."""
    from tldw_chatbook.Local_Ingestion.ingest_parse_worker import run_parse_job

    honored = run_parse_job(PRIVATE_URL, {"url_provenance": UrlProvenance.USER_ENTERED})
    assert honored["ok"] is True
    assert honored["payload"]["content"] == "transcript of the clip"
    # metadata + the download's probe + the download itself
    assert recording_ytdlp == [PRIVATE_URL] * 3

    # In-place clear: reassigning would orphan the fixture's list object.
    del _RecordingYoutubeDL.urls[:]
    refused = run_parse_job(PRIVATE_URL, {"url_provenance": "USER_ENTERED"})
    assert refused["ok"] is False, (
        "a string cannot launder provenance -- the run must refuse the URL"
    )
    assert "egress" in refused["error"]
    assert recording_ytdlp == []


@private_profile_test
def test_run_parse_job_without_provenance_fails_closed(
    request, recording_ytdlp: List[str]
) -> None:
    from tldw_chatbook.Local_Ingestion.ingest_parse_worker import run_parse_job

    result = run_parse_job(PRIVATE_URL, {})
    assert result["ok"] is False
    assert "egress" in result["error"]
    assert recording_ytdlp == []


# ---------------------------------------------------------------------------
# Mutation pins: the decision must stay expressed at the seams
# ---------------------------------------------------------------------------


def test_the_self_trusting_call_is_no_longer_unconditional() -> None:
    """Mutation guard: reverting ``check_media_url_egress`` to the
    unconditional self-trust (``trusted_origins=origin_set(url)``) must
    fail a test, not just change behaviour.

    Structurally checked (AST), not textually: this module's own docstring
    legitimately mentions the historical ``origin_set(url)`` spelling, and
    a text pin would false-red on prose -- the same regex-over-text trap
    the egress-adoption census was rewritten to avoid (TASK-19556).
    """
    import ast

    source = Path(video_processing.__file__).read_text(encoding="utf-8")
    assert "UrlProvenance" in source
    tree = ast.parse(source)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if callee != "check_url_or_raise":
            continue
        checked += 1
        for kw in node.keywords:
            if kw.arg != "trusted_origins":
                continue
            seed = kw.value
            assert isinstance(seed, ast.Call) and (
                getattr(seed.func, "id", None) == "trusted_origins_for"
            ), (
                "the URL is vouching for itself again -- the egress call "
                "must seed trust from trusted_origins_for(url, url_provenance)"
            )
    assert checked >= 1, "the module must still call the egress policy"
