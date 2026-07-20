"""Tests for the library ingestion pre-flight analyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from tldw_chatbook.Library.ingest_preflight import (
    _collect_files,
    _probe_url,
    _safe_size,
    analyze_path,
)
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Local_Ingestion.local_file_ingestion import is_http_url


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

        files, truncated = _collect_files(tmp_path, 1000)
        assert len(files) == 2
        assert {f.name for f in files} == {"a.pdf", "b.txt"}
        assert truncated is False

    def test_respects_scan_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file{i}.pdf").write_bytes(b"%PDF")

        files, truncated = _collect_files(tmp_path, 3)
        assert len(files) == 3
        assert truncated is True

    def test_exact_scan_limit_is_not_truncated(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"file{i}.pdf").write_bytes(b"%PDF")

        files, truncated = _collect_files(tmp_path, 3)
        assert len(files) == 3
        assert truncated is False

    def test_skips_symlinks(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.pdf"
        real_file.write_bytes(b"%PDF")
        symlink = tmp_path / "link.pdf"
        symlink.symlink_to(real_file)

        files, truncated = _collect_files(tmp_path, 1000)
        assert len(files) == 1
        assert files[0].name == "real.pdf"
        assert truncated is False

    def test_empty_directory(self, tmp_path: Path) -> None:
        files, truncated = _collect_files(tmp_path, 1000)
        assert files == []
        assert truncated is False

    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / "visible.pdf").write_bytes(b"%PDF")
        (tmp_path / ".hidden").write_text("secret")

        files, truncated = _collect_files(tmp_path, 1000)
        assert len(files) == 1
        assert files[0].name == "visible.pdf"
        assert truncated is False

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

        files, truncated = _collect_files(tmp_path, 1000)
        assert {f.name for f in files} == set()
        assert truncated is False


class TestProbeUrl:
    def test_returns_none_on_success(self) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight.urlopen", return_value=mock_response):
            assert _probe_url("https://example.com/doc.pdf") is None

    def test_returns_error_on_url_error(self) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight.urlopen",
            side_effect=URLError("connection refused"),
        ):
            result = _probe_url("https://example.com/doc.pdf")
        assert result is not None
        assert "unreachable" in result.lower()

    def test_returns_error_on_timeout(self) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight.urlopen",
            side_effect=TimeoutError(),
        ):
            result = _probe_url("https://example.com/doc.pdf")
        assert result is not None
        assert "timed out" in result.lower()

    def test_returns_error_on_unexpected_exception(self) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight.urlopen",
            side_effect=ValueError("boom"),
        ):
            result = _probe_url("https://example.com/doc.pdf")
        assert result is not None
        assert "failed" in result.lower()

    def test_returns_error_on_http_404(self) -> None:
        error = HTTPError("https://example.com/doc.pdf", 404, "Not Found", {}, None)
        with patch(
            "tldw_chatbook.Library.ingest_preflight.urlopen",
            side_effect=error,
        ):
            result = _probe_url("https://example.com/doc.pdf")
        assert result is not None
        assert "unreachable" in result.lower()


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

    def test_reachable_url(self) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight.urlopen", return_value=mock_response):
            result = analyze_path("https://example.com/document.pdf")

        assert result.errors == []
        assert result.total_files == 1
        assert "pdf" in result.type_groups
        assert result.total_size == 0

    def test_unreachable_url(self) -> None:
        with patch(
            "tldw_chatbook.Library.ingest_preflight.urlopen",
            side_effect=URLError("connection refused"),
        ):
            result = analyze_path("https://example.com/document.pdf")

        assert len(result.errors) == 1
        assert result.total_files == 0
        assert result.type_groups == {}

    def test_url_with_video_extension(self) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tldw_chatbook.Library.ingest_preflight.urlopen", return_value=mock_response):
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
