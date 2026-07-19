"""Regression guards for the Library ingest submit path's URL support.

Task 4 wires a URL into ``submit_library_ingest_job`` (via
``classify_ingest_source``) and ``_submit_library_ingest_form`` (via
``validate_url``). These tests lock the two seams that wiring depends on:
Tasks 1-3 already shipped ``classify_ingest_source``, and
``validate_url`` predates this feature -- both should already pass as
regression guards before any Task 4 wiring lands.
"""

from tldw_chatbook.Local_Ingestion.local_file_ingestion import classify_ingest_source
from tldw_chatbook.Utils.input_validation import validate_url


def test_submit_detected_type_for_url_is_article_or_video():
    assert classify_ingest_source("https://example.com/post") == "article"
    assert classify_ingest_source("https://youtube.com/watch?v=z") == "video"


def test_url_validation_accepts_http_rejects_scheme_tricks():
    assert validate_url("https://example.com/post") is True
    assert validate_url("file:///etc/passwd") is False
    assert validate_url("javascript:alert(1)") is False
