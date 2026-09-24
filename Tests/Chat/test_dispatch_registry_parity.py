# Tests/Chat/test_dispatch_registry_parity.py
"""Dispatch registration parity for engine-driven providers (ADR-179).

Task 8: Databricks joins the dispatch surface through the hosted provider
engine, and the sensitive-audit universe is derived from the provider
registry instead of a hand-maintained literal set.
"""
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)
from tldw_chatbook.provider_registry import AUDITED_ENDPOINT_KEYS


def test_databricks_registered_and_audited():
    assert "databricks" in API_CALL_HANDLERS
    assert callable(API_CALL_HANDLERS["databricks"])
    assert "databricks" in SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS


def test_audited_set_is_registry_derived_and_covers_all_handlers():
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS
    assert set(API_CALL_HANDLERS) <= SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
