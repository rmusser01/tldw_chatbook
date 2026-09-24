# Tests/Chat/test_databricks_continuation_registration.py
"""Continuation registration and generic gateway finish-policy entry (ADR-179).

Task 11: the engine-driven preset joins the canonical continuation format,
and the gateway resolves finish policies through one generic entry point --
map hit for the hand-written policies, registry-driven
``HostedPresetFinishPolicy`` for engine-driven providers -- instead of the
bare provider-class lookup.
"""
import typing

from tldw_chatbook.Chat.console_provider_gateway import resolve_finish_policy
from tldw_chatbook.Chat.provider_continuation import _PAIRINGS, ContinuationProvider


def test_continuation_provider_and_pairings_cover_databricks():
    assert "databricks" in typing.get_args(ContinuationProvider)
    assert ("databricks", "chat_completions") in _PAIRINGS


def test_gateway_finish_policy_resolves_engine_presets():
    assert resolve_finish_policy("databricks") is not None  # engine preset policy
    assert resolve_finish_policy("zai") is not None          # existing behavior
    assert resolve_finish_policy("openai") is None           # non-hosted untouched


def test_gateway_finish_policy_instances_are_cached_per_key():
    first = resolve_finish_policy("databricks")
    assert first is not None
    assert resolve_finish_policy("databricks") is first
    zai_first = resolve_finish_policy("zai")
    assert zai_first is not None
    assert resolve_finish_policy("zai") is zai_first
