"""Tests for the library ingestion pre-flight analyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from tldw_chatbook.Library import ingest_preflight
from tldw_chatbook.Library.ingest_preflight import (
    _collect_files,
    _probe_url,
    _safe_size,
    analyze_path,
)
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Local_Ingestion.local_file_ingestion import is_http_url


@pytest.fixture
def probing_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt into the URL probe and let the egress policy pass (TASK-19556).

    The probe is OFF by default now -- it used to fire from the ingest
    field's 0.8 s typing debounce, which made it an internal-host scanning
    oracle -- and when on it consults ``check_url_or_raise`` before any
    transport call. These tests are about the probe's TRANSPORT outcomes,
    so both gates are opened here; the gates themselves are owned by
    ``Tests/Library/test_ingest_preflight_egress.py``.
    """
    monkeypatch.setattr(ingest_preflight, "url_probe_enabled", lambda: True)
    monkeypatch.setattr(ingest_preflight, "check_url_or_raise", lambda *a, **k: None)


class TestSafeSize:
    def test_returns_file_size(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello world")
        assert _safe_size(file_path) == 11

    def test_returns_zero_on_os_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.txt"
        assert _safe_size(missing) == 0


class TestCollectFiles:
    def test_collects_files_recursively(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "b.txt").write_text("hello")

        files, truncated, skipped = _collect_files(tmp_path, 1000)
        assert len(files) == 2
        assert {f.name for f in files} == {"a.pdf", "b.txt"}
        assert truncated is False
        assert skipped == 0

    def test_respects_scan_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file{i}.pdf").write_bytes(b"%PDF")

        files, truncated, skipped = _collect_files(tmp_path, 3)
        assert len(files) == 3
        assert truncated is True
        # Files left behind by the LIMIT are truncation, never "skipped".
        assert skipped == 0

    def test_exact_scan_limit_is_not_truncated(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"file{i}.pdf").write_bytes(b"%PDF")

        files, truncated, skipped = _collect_files(tmp_path, 3)
        assert len(files) == 3
        assert truncated is False
        assert skipped == 0

    def test_skips_symlinks(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.pdf"
        real_file.write_bytes(b"%PDF")
        symlink = tmp_path / "link.pdf"
        symlink.symlink_to(real_file)

        files, truncated, skipped = _collect_files(tmp_path, 1000)
        assert len(files) == 1
        assert files[0].name == "real.pdf"
        assert truncated is False
        assert skipped == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        files, truncated, skipped = _collect_files(tmp_path, 1000)
        assert files == []
        assert truncated is False
        # A genuinely empty folder skipped nothing -- the distinction the
        # ingest gate needs to stop calling every 0-file folder "empty".
        assert skipped == 0

    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / "visible.pdf").write_bytes(b"%PDF")
        (tmp_path / ".hidden").write_text("secret")

        files, truncated, skipped = _collect_files(tmp_path, 1000)
        assert len(files) == 1
        assert files[0].name == "visible.pdf"
        assert truncated is False
        assert skipped == 1

    def test_handles_permission_error(self, tmp_path: Path, monkeypatch) -> None:
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "secret.pdf").write_bytes(b"%PDF")

        real_iterdir = Path.iterdir

        def mock_iterdir(self: Path):
            if self.resolve() == locked.resolve():
                raise PermissionError("access denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)

        files, truncated, skipped = _collect_files(tmp_path, 1000)
        assert {f.name for f in files} == set()
        assert truncated is False
        # An unreadable folder is not an empty one.
        assert skipped == 1


class TestProbeUrl:
    def test_returns_none_on_success(self, probing_allowed) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight._open_probe", return_value=mock_response):
            probe = _probe_url("https://example.com/doc.pdf")
        assert probe.error is None
        assert probe.note is None

    def test_returns_error_on_url_error(self, probing_allowed) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=URLError("connection refused"),
        ):
            probe = _probe_url("https://example.com/doc.pdf")
        assert probe.error is not None
        assert "unreachable" in probe.error.lower()

    def test_returns_error_on_timeout(self, probing_allowed) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=TimeoutError(),
        ):
            probe = _probe_url("https://example.com/doc.pdf")
        assert probe.error is not None
        assert "timed out" in probe.error.lower()

    def test_returns_error_on_unexpected_exception(self, probing_allowed) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=ValueError("boom"),
        ):
            probe = _probe_url("https://example.com/doc.pdf")
        assert probe.error is not None
        assert "failed" in probe.error.lower()

    def test_returns_error_on_http_404(self, probing_allowed) -> None:
        error = HTTPError("https://example.com/doc.pdf", 404, "Not Found", {}, None)
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=error,
        ):
            probe = _probe_url("https://example.com/doc.pdf")
        assert probe.error is not None
        assert "unreachable" in probe.error.lower()

    @pytest.mark.parametrize("status", [401, 403, 405, 429, 500])
    def test_a_status_the_probe_cannot_interpret_does_not_veto(self, status: int, probing_allowed) -> None:
        """The probe may report doubt; it may not refuse the source.

        Any HTTP status proves the host resolved and answered. Sites routinely
        refuse HEAD (405) or unrecognised clients (403) while serving the page
        perfectly well to whoever actually fetches it -- verified on a Wikipedia
        article that answers 403 to our client even with a browser User-Agent,
        and that a tldw server clipped at 200 (task-697).
        """
        error = HTTPError("https://example.com/page", status, "Nope", {}, None)
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe", side_effect=error
        ):
            probe = _probe_url("https://example.com/page")

        assert probe.error is None, f"{status} must not block the source"
        assert probe.note is not None and str(status) in probe.note

    def test_a_gone_resource_is_still_refused(self, probing_allowed) -> None:
        """410 is the host stating the resource is not there, like 404."""
        error = HTTPError("https://example.com/page", 410, "Gone", {}, None)
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe", side_effect=error
        ):
            probe = _probe_url("https://example.com/page")
        assert probe.error is not None


class TestAnalyzePath:
    def test_single_pdf_file(self, tmp_path: Path) -> None:
        pdf = tmp_path / "document.pdf"
        content = b"%PDF-1.4 fake"
        pdf.write_bytes(content)

        result = analyze_path(str(pdf))

        assert isinstance(result, PreflightResult)
        assert result.errors == []
        assert result.total_files == 1
        assert result.total_size == len(content)
        assert result.type_groups == {"pdf": [str(pdf)]}
        assert result.truncated is False

    def test_missing_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.pdf"
        result = analyze_path(str(missing))
        assert result.errors == [f"Path not found: {missing}"]
        assert result.total_files == 0
        assert result.total_size == 0

    def test_directory_recursion_and_scan_limit(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"root{i}.pdf").write_bytes(b"%PDF")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        for i in range(3):
            (subdir / f"sub{i}.txt").write_text("hello")

        result = analyze_path(str(tmp_path), scan_limit=4)

        assert result.total_files == 4
        assert result.truncated is True
        assert result.total_size > 0

    def test_directory_type_grouping(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.epub").write_bytes(b"epub")
        (tmp_path / "c.txt").write_text("plain")
        (tmp_path / "d.mp3").write_bytes(b"mp3")

        result = analyze_path(str(tmp_path))

        assert set(result.type_groups.keys()) == {"pdf", "ebook", "generic", "audio_video"}
        assert len(result.type_groups["pdf"]) == 1
        assert len(result.type_groups["ebook"]) == 1
        assert len(result.type_groups["generic"]) == 1
        assert len(result.type_groups["audio_video"]) == 1
        assert result.total_files == 4

    def test_directory_collects_tooling_warnings(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF")

        def fake_warnings(group: str) -> list[dict]:
            return [{"feature": "test", "group": group}]

        monkeypatch.setattr(
            "tldw_chatbook.Library.ingest_preflight.get_tooling_warnings",
            fake_warnings,
        )

        result = analyze_path(str(tmp_path))
        assert result.warnings == [{"feature": "test", "group": "pdf"}]

    def test_single_unsupported_file_is_grouped_not_raised(self, tmp_path: Path) -> None:
        """An unsupported file belongs in its own group, not in an exception.

        ``get_type_group`` returns ``"unsupported"`` by design so the summary
        can surface those files separately, but the capability lookup has no
        such group, so asking it for tooling warnings raised ``KeyError`` and
        replaced the entire pre-flight summary with a raw error string
        (task-674).
        """
        (tmp_path / "notes.xyz").write_text("nope")

        result = analyze_path(str(tmp_path / "notes.xyz"))

        assert result.errors == []
        assert result.type_groups.get("unsupported") == [str(tmp_path / "notes.xyz")]
        assert result.total_files == 1

    def test_directory_with_one_unsupported_file_still_summarises(
        self, tmp_path: Path
    ) -> None:
        """One unsupported file must not destroy the summary for the rest.

        Any real folder is likely to hold a ``.json``, ``.srt`` or ``.tmp``
        alongside the content, and a single one of them used to abort the
        whole analysis -- losing the file count, the size, the type breakdown
        and, critically, the tooling warnings that feed the guardrail.
        (Fixture was ``cover.jpg`` until task-3307 made images a supported
        group.)
        """
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.txt").write_text("plain")
        (tmp_path / "subs.srt").write_bytes(b"1\n00:00 --> 00:01\nhi")

        result = analyze_path(str(tmp_path))

        assert result.errors == []
        assert result.total_files == 3
        assert len(result.type_groups["pdf"]) == 1
        assert len(result.type_groups["generic"]) == 1
        assert result.type_groups["unsupported"] == [str(tmp_path / "subs.srt")]

    def test_unsupported_file_in_directory_does_not_block_tooling_warnings(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Supported groups still report their tooling warnings."""
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "subs.srt").write_bytes(b"1\n00:00 --> 00:01\nhi")

        def fake_warnings(group: str) -> list[dict]:
            return [{"feature": "test", "group": group}]

        monkeypatch.setattr(
            "tldw_chatbook.Library.ingest_preflight.get_tooling_warnings",
            fake_warnings,
        )

        result = analyze_path(str(tmp_path))

        assert {"feature": "test", "group": "pdf"} in result.warnings
        assert not any(w["group"] == "unsupported" for w in result.warnings)

    def test_reachable_url(self, probing_allowed) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight._open_probe", return_value=mock_response):
            result = analyze_path("https://example.com/document.pdf")

        assert result.errors == []
        assert result.total_files == 1
        assert "pdf" in result.type_groups
        assert result.total_size == 0

    def test_unreachable_url(self, probing_allowed) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=URLError("connection refused"),
        ):
            result = analyze_path("https://example.com/document.pdf")

        assert len(result.errors) == 1
        assert result.total_files == 0
        assert result.type_groups == {}

    def test_url_with_video_extension(self, probing_allowed) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight._open_probe", return_value=mock_response):
            result = analyze_path("https://example.com/lecture.mp4")

        assert result.errors == []
        assert result.type_groups == {"audio_video": ["https://example.com/lecture.mp4"]}

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = analyze_path(str(tmp_path))
        assert result.errors == []
        assert result.total_files == 0
        assert result.total_size == 0
        assert result.truncated is False
        assert result.type_groups == {}

    @pytest.mark.parametrize("bad_limit", [0, -1, -100])
    def test_invalid_scan_limit_raises(self, bad_limit: int) -> None:
        with pytest.raises(ValueError, match="scan_limit must be greater than zero"):
            analyze_path("/some/path", scan_limit=bad_limit)


class TestPublicApi:
    def test_uses_public_is_http_url(self) -> None:
        # The preflight module should rely on the public helper rather than
        # importing a private name from local_file_ingestion.
        assert is_http_url("https://example.com") is True
        assert is_http_url("/local/path") is False


class TestPathErrorsAreMarked:
    """A path that cannot be found is not something to retry (task-666)."""

    def test_missing_path_is_flagged_as_a_path_problem(self, tmp_path: Path) -> None:
        result = analyze_path(str(tmp_path / "nope.txt"))
        assert result.errors
        assert result.path_invalid is True

    def test_successful_analysis_is_not_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        result = analyze_path(str(tmp_path / "a.txt"))
        assert result.errors == []
        assert result.path_invalid is False

    def test_unreachable_url_is_not_a_path_problem(self, probing_allowed) -> None:
        """A URL that failed to respond is worth retrying; a typo'd path isn't."""
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=URLError("connection refused"),
        ):
            result = analyze_path("https://example.com/document.pdf")
        assert result.errors
        assert result.path_invalid is False


def test_zero_byte_files_classified_as_empty_not_importable(tmp_path):
    """(task-2160) A 0-byte file leaves its type group at analysis time
    and lands in ``empty_files`` -- the forecast used to promise it would
    import and the pipeline then failed it post-commit."""
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    real = tmp_path / "real.txt"
    real.write_text("content")

    solo = analyze_path(str(empty))
    assert solo.empty_files == (str(empty),)
    assert not solo.type_groups
    assert solo.total_files == 1

    folder = analyze_path(str(tmp_path))
    assert folder.empty_files == (str(empty),)
    assert sorted(
        path for files in folder.type_groups.values() for path in files
    ) == [str(real)]
    assert folder.total_files == 2


def test_unstatable_files_are_not_mislabeled_empty(tmp_path, monkeypatch):
    """(task-2160 Qodo round) A file whose stat raises must stay in its
    type group -- the error fallback of 0 bytes used to classify it as
    "empty" and forecast "is 0 B" for a file nobody measured."""
    import tldw_chatbook.Library.ingest_preflight as preflight_mod

    victim = tmp_path / "unreadable.txt"
    victim.write_text("content")
    real_probe = preflight_mod._statted_size

    def failing_probe(path):
        if path.name == "unreadable.txt":
            return None  # what a raising stat resolves to
        return real_probe(path)

    monkeypatch.setattr(preflight_mod, "_statted_size", failing_probe)
    result = analyze_path(str(victim))
    assert result.empty_files == ()
    assert [
        path for files in result.type_groups.values() for path in files
    ] == [str(victim)]


class TestUrlProbePlainLanguage:
    """(task-3305, MI-13) URL preflight failures must read as plain
    language -- the probe used to surface a raw exception repr
    (``URL unreachable: <urlopen error [Errno 8] nodename nor servname
    provided, or not known>``) as the primary line."""

    @pytest.fixture(autouse=True)
    def _allow_probing(self, probing_allowed) -> None:
        """Every test here drives `_probe_url` directly (TASK-19556 gates)."""

    def _probe_with(self, exc: Exception) -> object:
        with patch(
            "tldw_chatbook.Library.ingest_preflight._open_probe",
            side_effect=exc,
        ):
            return _probe_url("https://no-such-host.example/doc.pdf")

    @staticmethod
    def _assert_no_repr(message: str) -> None:
        assert "<urlopen" not in message
        assert "Errno" not in message
        assert "gaierror" not in message

    def test_dns_failure_names_the_unresolvable_host(self) -> None:
        import socket

        probe = self._probe_with(
            URLError(
                socket.gaierror(8, "nodename nor servname provided, or not known")
            )
        )
        assert probe.error == (
            "URL unreachable — the server name could not be found."
        )

    def test_connection_refused_reads_plain(self) -> None:
        probe = self._probe_with(
            URLError(ConnectionRefusedError(61, "Connection refused"))
        )
        assert probe.error == (
            "URL unreachable — the connection was refused."
        )

    def test_timeout_inside_urlerror_reads_plain(self) -> None:
        probe = self._probe_with(URLError(TimeoutError("timed out")))
        assert probe.error == (
            "URL unreachable — the connection timed out."
        )

    def test_tls_failure_reads_plain(self) -> None:
        import ssl

        probe = self._probe_with(
            URLError(ssl.SSLError(1, "certificate verify failed"))
        )
        assert probe.error == (
            "URL unreachable — the secure connection (TLS) failed."
        )

    def test_http_absent_status_reads_plain(self) -> None:
        error = HTTPError(
            "https://example.com/doc.pdf", 404, "Not Found", {}, None
        )
        probe = self._probe_with(error)
        assert probe.error == (
            "URL unreachable — the server says this page does not exist "
            "(HTTP 404)."
        )
        self._assert_no_repr(probe.error)

    def test_unmapped_url_error_never_leaks_a_repr(self) -> None:
        probe = self._probe_with(URLError(OSError(999, "weird transport")))
        assert probe.error == (
            "URL unreachable — the server could not be contacted."
        )
        self._assert_no_repr(probe.error)

    def test_unexpected_exception_never_leaks_a_repr(self) -> None:
        probe = self._probe_with(ValueError("boom <internal>"))
        assert probe.error is not None
        assert "failed" in probe.error.lower()
        assert "boom" not in probe.error
        assert "<" not in probe.error


def test_xml_file_lands_in_the_unsupported_bucket_task_3308(tmp_path) -> None:
    """task-3308 (defer ruling, task-3310 notes): an ``.xml`` source is
    classified unsupported at pre-flight -- never grouped, never raised --
    so the queue can never hand it to the parse path whose XML branch
    still says "not yet implemented"."""
    xml = tmp_path / "feed.xml"
    xml.write_text("<rss><channel/></rss>")

    result = analyze_path(str(xml))

    assert result.errors == []
    assert result.type_groups.get("unsupported") == [str(xml)]
    assert result.total_files == 1


def test_png_file_lands_in_the_image_group_task_3307(tmp_path) -> None:
    """task-3307 (ship ruling, task-3310 notes): a raster image pre-flights
    into its own ``image`` group -- it used to land in the unsupported
    bucket while ``process_image`` sat unreachable."""
    png = tmp_path / "photo.png"
    # A real minimal PNG header is not needed for pre-flight (extension
    # classification only), but keep the bytes non-empty so the empty-file
    # classifier stays out of the way.
    png.write_bytes(b"\x89PNG\r\n\x1a\n rest")

    result = analyze_path(str(png))

    assert result.errors == []
    assert result.type_groups.get("image") == [str(png)]
    assert "unsupported" not in result.type_groups
    assert result.total_files == 1


def test_preflight_reports_entries_the_scan_skipped(tmp_path: Path) -> None:
    """(xhigh review round, F5) ``total_files == 0`` conflates "this folder
    holds nothing" with "this folder's entries were all skipped" (symlinks
    and dot-entries), and the ingest gate said "This folder is empty" about
    both. The pre-flight is the only layer that can tell them apart."""
    target = tmp_path / "real.txt"
    target.write_text("hello world")
    folder = tmp_path / "links"
    folder.mkdir()
    (folder / "linked.txt").symlink_to(target)
    (folder / ".hidden.txt").write_text("hidden")

    result = analyze_path(str(folder))

    assert result.total_files == 0
    assert result.skipped_entries == 2, (
        "the pre-flight cannot tell an empty folder from one whose entries "
        "the scan skipped"
    )


def test_preflight_reports_no_skipped_entries_for_an_empty_folder(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "nothing"
    folder.mkdir()

    result = analyze_path(str(folder))

    assert (result.total_files, result.skipped_entries) == (0, 0)


def test_preflight_does_not_count_skips_inside_collected_subfolders(
    tmp_path: Path,
) -> None:
    """A folder that DID yield files still reports the entries it passed
    over, but the count never inflates the collected total."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "notes.txt").write_text("kept")
    (folder / ".dotfile").write_text("skipped")

    result = analyze_path(str(folder))

    assert result.total_files == 1
    assert result.skipped_entries == 1
