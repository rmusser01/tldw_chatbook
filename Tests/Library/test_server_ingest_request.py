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
            ("/tmp/notes.txt", "document"),
            ("/tmp/notes.md", "document"),
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

    def test_every_mapping_target_is_one_the_server_accepts(self) -> None:
        """Only the server's five media types may ever be sent.

        Discovered by submitting to a live server, NOT from its OpenAPI spec --
        the spec types ``media_type`` as a bare string and the real set is
        enforced by a runtime validator, which answered:
        "Input should be 'video', 'audio', 'document', 'pdf' or 'ebook'".

        An earlier version of this mapping sent ``plaintext`` (inferred from the
        legacy ingest window's own form dispatch) and every plain-text server
        ingest would have failed validation.
        """
        from tldw_chatbook.Library.server_ingest_request import (
            SERVER_ACCEPTED_MEDIA_TYPES,
            SERVER_MEDIA_TYPE_BY_LOCAL_TYPE,
        )

        assert SERVER_ACCEPTED_MEDIA_TYPES == frozenset(
            {"video", "audio", "document", "pdf", "ebook"}
        )
        unaccepted = {
            local: sent
            for local, sent in SERVER_MEDIA_TYPE_BY_LOCAL_TYPE.items()
            if sent not in SERVER_ACCEPTED_MEDIA_TYPES
        }
        assert not unaccepted, f"would be rejected by the server: {unaccepted}"

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
    def test_shared_generic_options_are_projected_for_every_server_media_type(
        self,
    ) -> None:
        """Shared controls do not depend on the detected type-group loop."""
        kwargs = build_server_ingest_kwargs(
            "/tmp/paper.pdf",
            options={
                "generic": {
                    "analyze": True,
                    "overwrite_existing": True,
                    "custom_prompt": "Extract decisions.",
                    "system_prompt": "Be concise.",
                    "generate_embeddings": False,
                    "keep_original_file": True,
                }
            },
        )

        assert kwargs["overwrite_existing"] is True
        assert kwargs["custom_prompt"] == "Extract decisions."
        assert kwargs["system_prompt"] == "Be concise."
        assert kwargs["generate_embeddings"] is False
        assert kwargs["keep_original_file"] is True

    def test_server_omits_analysis_prompts_when_analysis_is_off(self) -> None:
        kwargs = build_server_ingest_kwargs(
            "/tmp/notes.txt",
            options={
                "generic": {
                    "analyze": False,
                    "custom_prompt": "Extract decisions.",
                    "system_prompt": "Be concise.",
                }
            },
        )

        assert kwargs["perform_analysis"] is False
        assert "custom_prompt" not in kwargs
        assert "system_prompt" not in kwargs

    def test_local_file_goes_in_file_paths_not_urls(self) -> None:
        kwargs = build_server_ingest_kwargs("/tmp/notes.txt", options={})

        assert kwargs["media_type"] == "document"
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
            options={
                "generic": {"chunk": True, "chunk_size": 1200, "chunk_overlap": 150}
            },
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

        # task-3309: these used to assert the CLIENT's spelling, which the
        # server never declares -- so the fields were dropped in silence and
        # this test was pinning the bug as a requirement. The server calls
        # them `pdf_parsing_engine` and `enable_ocr`.
        assert kwargs["pdf_parsing_engine"] == "docling"
        assert kwargs["enable_ocr"] is True
        assert "pdf_engine" not in kwargs
        assert "ocr" not in kwargs
        assert "transcription_model" not in kwargs

    def test_document_group_options_travel_for_docx(self) -> None:
        """(task-3303) .docx now groups as ``document``, so the document
        panel's options ride a server submission; other groups' options do not.

        (task-3309) ``extra="allow"`` on the request model was never the whole
        story: it lets a field onto the wire, but the endpoint binds its form
        fields explicitly and never reads the raw form, so an undeclared one is
        discarded server-side. ``processing_method`` has no server equivalent
        at all and is therefore no longer sent -- it is reported through
        ``server_unsupported_options`` instead of being lost in transit.
        """
        kwargs = build_server_ingest_kwargs(
            "/tmp/report.docx",
            options={
                "document": {
                    "processing_method": "docling",
                    "ocr": True,
                    "ocr_language": "de",
                },
                "pdf": {"pdf_engine": "docling"},
            },
        )

        assert kwargs["media_type"] == "document"
        assert "processing_method" not in kwargs
        assert kwargs["enable_ocr"] is True
        assert kwargs["ocr_lang"] == "de"
        assert "pdf_engine" not in kwargs
        assert "pdf_parsing_engine" not in kwargs

    def test_empty_source_is_refused(self) -> None:
        with pytest.raises(ServerIngestUnsupported):
            build_server_ingest_kwargs("   ", options={})


# --- task-3307: image files are local-only ----------------------------------


class TestImageSourcesStayLocal:
    """The server ingest-jobs endpoint accepts only video/audio/document/
    pdf/ebook (SERVER_ACCEPTED_MEDIA_TYPES, established live). Images are
    deliberately NOT mapped to a lookalike type: a server-mode submission
    is refused with the honest no-handler reason instead of asking the
    server's document extractor to read pixels."""

    def test_image_file_is_refused_with_honest_reason(self) -> None:
        with pytest.raises(ServerIngestUnsupported) as excinfo:
            server_media_type_for("/tmp/photo.png")
        assert "image" in str(excinfo.value)

    def test_build_kwargs_refuses_image_files_too(self) -> None:
        with pytest.raises(ServerIngestUnsupported):
            build_server_ingest_kwargs("/tmp/photo.png", options={})


# --- task-14827: what the SERVER path refuses, asked as a question -----------


class TestServerIngestRefusal:
    """(task-14827 AC#1) The forecast has to ask the backend it targets.

    ``server_ingest_refusal`` is that question, and it mirrors what
    ``submit_library_ingest_job`` actually does in server mode -- including
    the clipper route, without which every server-mode URL import would be
    forecast as a certain failure.
    """

    def test_a_file_the_server_maps_is_not_refused(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("/tmp/notes.txt") is None
        assert server_ingest_refusal("/tmp/talk.mp3") is None
        assert server_ingest_refusal("/tmp/book.epub") is None

    def test_an_image_file_is_refused_although_local_imports_it(self) -> None:
        """The divergence that made this predicate necessary: an image is a
        real LOCAL capability (the ``image`` group, OCR) with no server
        media type at all."""
        from tldw_chatbook.Library.ingest_capabilities import get_type_group
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert get_type_group("/tmp/photo.png") == "image"
        reason = server_ingest_refusal("/tmp/photo.png")
        assert reason is not None
        assert "image" in reason

    def test_an_unclassifiable_file_is_refused_not_silently_skipped(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("/tmp/weird.xyz") is not None

    def test_a_page_url_is_not_refused_because_it_goes_to_the_clipper(
        self,
    ) -> None:
        """``server_media_type_for`` refuses a page on purpose, but the
        submit path never asks it: ``is_web_clip_source`` routes the page to
        the clipper first. A predicate that ignored that would condemn every
        server-mode URL import."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
            server_media_type_for,
        )

        with pytest.raises(ServerIngestUnsupported):
            server_media_type_for("https://example.com/post")
        assert server_ingest_refusal("https://example.com/post") is None

    def test_a_media_url_is_not_refused_either(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("https://youtube.com/watch?v=abc") is None

    def test_an_empty_source_is_refused(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("   ") is not None


class TestZeroByteSourcesAreRefusedBeforeTheyAreSent:
    """(task-14910) The one claim the SERVER forecast had not earned.

    ``build_ingest_forecast`` counts every 0-byte staged file as a certain
    failure on BOTH backends. Locally that is verified -- the parse chain
    raises ``EmptySourceIngestError`` before any write. On the server path
    nothing verified it: ``build_server_ingest_kwargs`` happily built
    kwargs for a 0-byte file and the app SENT it, handing the outcome to a
    server this process cannot inspect. The forecast was asserting
    knowledge it did not have, one line above copy that says outright
    "server tooling isn't checked from here".

    The resolution is a client-side refusal, which is knowable by
    construction: a 0-byte file is refused here, with the reason, and
    never leaves the machine.
    """

    def test_a_zero_byte_file_is_refused(self, tmp_path) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        empty = tmp_path / "empty.txt"
        empty.write_text("")
        reason = server_ingest_refusal(str(empty))
        assert reason is not None
        assert "empty.txt" in reason
        assert "empty" in reason

    def test_the_submit_seam_refuses_it_with_the_same_reason(
        self, tmp_path
    ) -> None:
        """The predicate the FORECAST reads and the builder the SUBMIT
        path calls must state the same thing about the same file -- a
        forecast that promises a refusal the submit path does not perform
        is the divergence this task exists to remove."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ServerIngestUnsupported) as excinfo:
            build_server_ingest_kwargs(str(empty), options={})
        assert str(excinfo.value) == server_ingest_refusal(str(empty))

    def test_a_zero_byte_file_of_a_mapped_type_is_refused_too(
        self, tmp_path
    ) -> None:
        """The refusal is about the file's CONTENT, not its extension: a
        0-byte .mp3 maps to ``audio`` perfectly well and is still nothing
        to send."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
            server_media_type_for,
        )

        empty = tmp_path / "silence.mp3"
        empty.write_bytes(b"")
        assert server_media_type_for(str(empty)) == "audio"
        assert server_ingest_refusal(str(empty)) is not None

    def test_a_file_with_content_is_still_sent(self, tmp_path) -> None:
        """Guard: the refusal is 0 bytes exactly, not "small"."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        one_byte = tmp_path / "tiny.txt"
        one_byte.write_bytes(b" ")
        assert server_ingest_refusal(str(one_byte)) is None
        assert build_server_ingest_kwargs(str(one_byte), options={})[
            "file_paths"
        ] == [str(one_byte)]

    def test_a_url_is_never_called_empty(self) -> None:
        """Guard: a URL has no local size to measure -- claiming one is
        the same class of fabrication (task-3305, MI-19: "1 file - 0 B"
        for a URL)."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("https://example.com/talk.mp3") is None

    def test_an_unstatable_path_is_not_called_empty(self) -> None:
        """Guard: a path that does not exist is not a 0-byte file. The
        pre-flight makes the same distinction (``_statted_size`` returns
        ``None`` rather than 0 on ``OSError``), and the existing refusal
        tests all use paths that were never created."""
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        assert server_ingest_refusal("/tmp/does-not-exist-14910.txt") is None

    def test_a_directory_is_not_called_an_empty_source(self, tmp_path) -> None:
        """Guard: a directory can stat at 0 bytes on some filesystems, and
        "this folder is empty" is a different diagnosis with a different
        recovery -- the ingest Start gate owns that one."""
        from tldw_chatbook.Library.server_ingest_request import (
            empty_source_refusal,
        )

        folder = tmp_path / "nothing"
        folder.mkdir()
        assert empty_source_refusal(str(folder)) is None
