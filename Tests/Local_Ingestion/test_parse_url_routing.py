from unittest.mock import patch
import pytest

from tldw_chatbook.Local_Ingestion.local_file_ingestion import parse_local_file_for_ingest


def test_article_url_routes_to_extractor_and_sets_real_url():
    fake = {"content": "Body text here", "title": "T", "author": "A", "keywords": [],
            "chunks": [], "analysis": "", "metadata": {}, "url": "https://example.com/post"}
    with patch("tldw_chatbook.Local_Ingestion.web_article_ingestion.extract_article_for_ingest",
               return_value=fake) as ex:
        payload = parse_local_file_for_ingest("https://example.com/post?utm_source=x", {})
    ex.assert_called_once()
    assert payload["media_type"] == "article"
    assert payload["content"] == "Body text here"
    assert payload["url"] == "https://example.com/post"          # NOT file://, NOT .absolute()
    assert not payload["url"].startswith("file://")


def test_video_url_routes_to_audio_video_branch():
    # the audio/video branch calls the processor with the URL as input;
    # mock the processor to return a transcript result.
    fake_result = {"content": "transcript", "title": "Vid", "author": "U", "chunks": [], "analysis": ""}
    with patch("tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalVideoProcessor") as VP:
        VP.return_value.process_videos.return_value = {"results": [{**fake_result, "status": "Success", "metadata": {}}]}
        payload = parse_local_file_for_ingest("https://youtube.com/watch?v=abc", {})
    assert payload["media_type"] == "video"
    assert payload["url"] == "https://youtube.com/watch?v=abc"    # the URL, not file://
    args, kwargs = VP.return_value.process_videos.call_args
    assert kwargs.get("inputs") == ["https://youtube.com/watch?v=abc"]


def test_article_permanent_error_propagates_unwrapped():
    # CRITICAL: a PermanentIngestError from the extractor must NOT be re-wrapped
    # into a plain FileIngestionError by the outer `except Exception` -- else
    # classify_parse_failure sees a bare FileIngestionError and marks it retryable.
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import PermanentIngestError
    from tldw_chatbook.Local_Ingestion.ingest_parse_worker import classify_parse_failure
    with patch("tldw_chatbook.Local_Ingestion.web_article_ingestion.extract_article_for_ingest",
               side_effect=PermanentIngestError("page requires JavaScript")):
        with pytest.raises(PermanentIngestError) as ei:
            parse_local_file_for_ingest("https://example.com/spa", {})
    assert classify_parse_failure(ei.value) is True     # stays permanent through the wrapper
