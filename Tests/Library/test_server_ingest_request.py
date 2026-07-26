"""Tests for mapping a Library ingest submission onto the server jobs API.

The Library ingest canvas only ever ran locally: ``build_library_ingest_state``
took a ``runtime_source`` and, per its own docstring, used it for nothing but a
"ingest runs on Local" quiet line. Server-backed ingestion lived in a separate
window (task-684). This module is the pure mapping half of bringing it over --
no UI, no I/O, so the shape of the request can be pinned before anything is
wired.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.server_ingest_request import (
    ServerIngestUnsupported,
    build_server_ingest_kwargs,
    server_media_type_for,
)


class TestServerMediaTypeFor:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("/tmp/paper.pdf", "pdf"),
            ("/tmp/report.docx", "document"),
            ("/tmp/book.epub", "ebook"),
            ("/tmp/notes.txt", "plaintext"),
            ("/tmp/notes.md", "plaintext"),
            ("/tmp/talk.mp3", "audio"),
            ("/tmp/talk.mp4", "video"),
            ("https://youtube.com/watch?v=abc", "video"),
            ("https://example.com/talk.mp3", "audio"),
        ],
    )
    def test_maps_local_detection_onto_server_media_types(
        self, source: str, expected: str
    ) -> None:
        assert server_media_type_for(source) == expected

    def test_html_is_a_document_to_the_server(self) -> None:
        """The server has no ``html`` media type; its document extractor takes it."""
        assert server_media_type_for("/tmp/page.html") == "document"

    def test_plain_web_page_is_not_a_jobs_api_source(self) -> None:
        """A web page belongs to the clipper endpoint, not the ingest-jobs API.

        ``classify_ingest_source`` calls a non-media URL an ``article``, and the
        jobs API has no such media type. Refusing it here keeps the boundary
        explicit instead of guessing a type the server would reject.
        """
        with pytest.raises(ServerIngestUnsupported) as excinfo:
            server_media_type_for("https://example.com/some-post")
        assert "web page" in str(excinfo.value).lower()

    def test_unsupported_extension_is_refused(self) -> None:
        with pytest.raises(ServerIngestUnsupported):
            server_media_type_for("/tmp/cover.jpg")


class TestBuildServerIngestKwargs:
    def test_local_file_goes_in_file_paths_not_urls(self) -> None:
        kwargs = build_server_ingest_kwargs("/tmp/notes.txt", options={})

        assert kwargs["media_type"] == "plaintext"
        assert kwargs["file_paths"] == ["/tmp/notes.txt"]
        assert kwargs.get("urls") is None

    def test_url_goes_in_urls_not_file_paths(self) -> None:
        kwargs = build_server_ingest_kwargs(
            "https://youtube.com/watch?v=abc", options={}
        )

        assert kwargs["media_type"] == "video"
        assert kwargs["urls"] == ["https://youtube.com/watch?v=abc"]
        assert kwargs.get("file_paths") is None

    def test_metadata_and_analysis_are_forwarded(self) -> None:
        kwargs = build_server_ingest_kwargs(
            "/tmp/paper.pdf",
            options={},
            title="A title",
            author="An author",
            keywords=("alpha", "beta"),
            perform_analysis=True,
        )

        assert kwargs["title"] == "A title"
        assert kwargs["author"] == "An author"
        assert kwargs["keywords"] == ["alpha", "beta"]
        assert kwargs["perform_analysis"] is True

    def test_chunking_options_come_from_the_generic_group(self) -> None:
        """Chunking is declared once, in the capability schema, for both backends."""
        kwargs = build_server_ingest_kwargs(
            "/tmp/notes.txt",
            options={"generic": {"chunk": True, "chunk_size": 1200, "chunk_overlap": 150}},
        )

        assert kwargs["perform_chunking"] is True
        assert kwargs["chunk_size"] == 1200
        assert kwargs["chunk_overlap"] == 150

    def test_chunking_off_is_forwarded_as_off(self) -> None:
        kwargs = build_server_ingest_kwargs(
            "/tmp/notes.txt", options={"generic": {"chunk": False}}
        )

        assert kwargs["perform_chunking"] is False

    def test_string_chunk_size_from_the_form_echo_is_coerced(self) -> None:
        """The form stores chunk size as display text; the API wants an int."""
        kwargs = build_server_ingest_kwargs(
            "/tmp/notes.txt",
            options={"generic": {"chunk": True, "chunk_size": "900"}},
        )

        assert kwargs["chunk_size"] == 900

    def test_unparseable_chunk_size_falls_back_rather_than_raising(self) -> None:
        kwargs = build_server_ingest_kwargs(
            "/tmp/notes.txt",
            options={"generic": {"chunk": True, "chunk_size": "not-a-number"}},
        )

        assert isinstance(kwargs["chunk_size"], int)
        assert kwargs["chunk_size"] > 0

    def test_per_type_options_are_passed_through_for_the_detected_group(self) -> None:
        """PDF engine/OCR travel with a PDF; unrelated groups' options do not."""
        kwargs = build_server_ingest_kwargs(
            "/tmp/paper.pdf",
            options={
                "pdf": {"pdf_engine": "docling", "ocr": True},
                "audio_video": {"transcription_model": "small"},
            },
        )

        assert kwargs["pdf_engine"] == "docling"
        assert kwargs["ocr"] is True
        assert "transcription_model" not in kwargs

    def test_empty_source_is_refused(self) -> None:
        with pytest.raises(ServerIngestUnsupported):
            build_server_ingest_kwargs("   ", options={})
