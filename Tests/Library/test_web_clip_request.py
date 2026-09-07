"""Tests for mapping a Library ingest submission of a web page onto a clip.

The pure mapping half of bringing web clipping into the ingest canvas
(task-684.3). No UI, no I/O, no server -- so the request shape is pinned before
anything is wired, the same way ``test_server_ingest_request`` pins the
ingest-jobs mapping.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.web_clip_request import (
    NotAWebClipSource,
    build_web_clip_kwargs,
    clip_failure_reason,
    is_web_clip_source,
)


class TestIsWebClipSource:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/some-post",
            "https://en.wikipedia.org/wiki/Fort_Sumter",
            "http://example.com",
        ],
    )
    def test_a_page_is_clippable(self, url: str) -> None:
        assert is_web_clip_source(url) is True

    @pytest.mark.parametrize(
        "source",
        [
            # Media URLs go to the ingest-jobs path, which can download and
            # transcode them; the clipper would only scrape the watch page.
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://example.com/talk.mp3",
            # Local files are never clips.
            "/tmp/notes.txt",
            "/tmp/paper.pdf",
            "",
            "   ",
        ],
    )
    def test_everything_else_is_not(self, source: str) -> None:
        assert is_web_clip_source(source) is False


class TestBuildWebClipKwargs:
    def test_a_page_becomes_a_single_url_clip(self) -> None:
        kwargs = build_web_clip_kwargs("https://example.com/post", options={})

        assert kwargs["urls"] == ["https://example.com/post"]
        assert kwargs["scrape_method"] == "individual"

    def test_chunking_comes_from_the_generic_group(self) -> None:
        """Chunking is declared once, for every backend and every source type."""
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={"generic": {"chunk": True, "chunk_size": 1200, "chunk_overlap": 150}},
        )

        assert kwargs["perform_chunking"] is True
        assert kwargs["chunk_size"] == 1200
        assert kwargs["chunk_overlap"] == 150

    def test_chunking_off_sends_no_sizes(self) -> None:
        kwargs = build_web_clip_kwargs(
            "https://example.com/post", options={"generic": {"chunk": False}}
        )

        assert kwargs["perform_chunking"] is False
        assert "chunk_size" not in kwargs

    def test_a_string_chunk_size_from_the_form_is_coerced(self) -> None:
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={"generic": {"chunk": True, "chunk_size": "900"}},
        )

        assert kwargs["chunk_size"] == 900

    def test_an_unparseable_chunk_size_falls_back(self) -> None:
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={"generic": {"chunk": True, "chunk_size": "not-a-number"}},
        )

        assert kwargs["chunk_size"] > 0

    def test_page_and_depth_limits_are_omitted_for_a_single_page_clip(self) -> None:
        """Sending crawl limits for a single page would imply a crawl."""
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={"web": {"scrape_method": "individual", "max_pages": 7}},
        )

        assert "max_pages" not in kwargs
        assert "max_depth" not in kwargs

    def test_page_and_depth_limits_travel_with_a_crawl(self) -> None:
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={
                "web": {
                    "scrape_method": "recursive_scraping",
                    "max_pages": 7,
                    "max_depth": 2,
                }
            },
        )

        assert kwargs["scrape_method"] == "recursive_scraping"
        assert kwargs["max_pages"] == 7
        assert kwargs["max_depth"] == 2

    def test_an_unknown_scrape_method_falls_back_to_the_single_page_one(self) -> None:
        """The server enforces its ScrapeMethod enum with a runtime validator.

        The accepted set is not in its OpenAPI type -- the same trap that made
        every plain-text server ingest send a rejected ``media_type``. Falling
        back beats submitting a value the server will refuse.
        """
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={"web": {"scrape_method": "telepathy"}},
        )

        assert kwargs["scrape_method"] == "individual"

    def test_every_offered_scrape_method_is_one_the_server_accepts(self) -> None:
        """The schema's choices and the server's enum must not drift apart."""
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities
        from tldw_chatbook.Library.web_clip_request import SERVER_SCRAPE_METHODS

        field = next(
            f for f in get_capabilities("web").fields if f.name == "scrape_method"
        )
        unaccepted = set(field.options) - SERVER_SCRAPE_METHODS
        assert not unaccepted, f"the form offers what the server rejects: {unaccepted}"

    def test_metadata_is_forwarded_as_the_lists_the_endpoint_takes(self) -> None:
        kwargs = build_web_clip_kwargs(
            "https://example.com/post",
            options={},
            title="A title",
            author="An author",
            keywords=("alpha", "beta"),
        )

        assert kwargs["titles"] == ["A title"]
        assert kwargs["authors"] == ["An author"]
        assert kwargs["keywords"] == ["alpha", "beta"]

    def test_a_file_is_refused(self) -> None:
        with pytest.raises(NotAWebClipSource):
            build_web_clip_kwargs("/tmp/notes.txt", options={})

    def test_a_media_url_is_refused(self) -> None:
        with pytest.raises(NotAWebClipSource):
            build_web_clip_kwargs(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ", options={}
            )

    def test_an_empty_source_is_refused(self) -> None:
        with pytest.raises(NotAWebClipSource):
            build_web_clip_kwargs("   ", options={})


class TestClipFailureReason:
    def test_a_real_success_payload_is_a_success(self) -> None:
        """Captured verbatim from a live server (2026-07-26)."""
        response = {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": "https://en.wikipedia.org/wiki/Fort_Sumter",
                    "title": "Untitled",
                    "author": "Unknown",
                    "content": "Fort Sumter ...",
                    "extraction_successful": True,
                }
            ],
        }

        assert clip_failure_reason(response) is None

    def test_a_200_that_extracted_nothing_is_a_failure(self) -> None:
        """A successful call is not a captured page.

        The endpoint answers 200 with a body describing the outcome, so a clip
        that extracted nothing arrives as transport-level success. Recording it
        as done would repeat the empty ingest the local pipeline had to be
        guarded against (task-677).
        """
        response = {
            "status": "success",
            "message": "Web content processed",
            "results": [
                {"url": "https://example.com/x", "extraction_successful": False}
            ],
        }

        reason = clip_failure_reason(response)
        assert reason is not None
        assert "example.com/x" in reason

    def test_no_results_is_a_failure(self) -> None:
        assert clip_failure_reason({"status": "success", "results": []}) is not None

    def test_a_non_success_status_carries_the_servers_message(self) -> None:
        reason = clip_failure_reason(
            {"status": "error", "message": "Scraper unavailable"}
        )
        assert reason == "Scraper unavailable"

    def test_no_response_at_all_is_a_failure(self) -> None:
        assert clip_failure_reason(None) is not None

    def test_a_model_shaped_response_is_accepted(self) -> None:
        """The service returns a model; only its own tests pass dicts."""
        from types import SimpleNamespace

        response = SimpleNamespace(
            status="success",
            message="ok",
            results=[SimpleNamespace(url="https://x", extraction_successful=True)],
        )

        assert clip_failure_reason(response) is None


def test_every_kwarg_this_builds_is_one_the_real_service_accepts() -> None:
    """Check the mapping against the REAL signature, not against a fake.

    This is the guard that was missing when the remote-ingest poller sent an
    ``offset`` the service did not accept: the fake declared it because it was
    written to match the call site, so the suite agreed with a wrong assumption
    for as long as no real service was involved. Reading the actual signature
    cannot agree with a mistake.
    """
    import inspect

    from tldw_chatbook.Media.server_media_reading_service import (
        ServerMediaReadingService,
    )

    parameters = inspect.signature(
        ServerMediaReadingService.ingest_web_content
    ).parameters
    accepts_extra = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )

    # Build with every optional branch taken, so no key escapes the check.
    kwargs = build_web_clip_kwargs(
        "https://example.com/post",
        options={
            "generic": {"chunk": True, "chunk_size": 900, "chunk_overlap": 50, "analyze": True},
            "web": {"scrape_method": "recursive_scraping", "max_pages": 5, "max_depth": 2},
        },
        title="T",
        author="A",
        keywords=("k",),
    )

    unknown = sorted(set(kwargs) - set(parameters))
    assert not unknown or accepts_extra, (
        f"ingest_web_content does not accept: {unknown}"
    )
