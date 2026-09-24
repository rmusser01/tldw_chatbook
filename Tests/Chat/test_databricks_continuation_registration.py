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


def test_trace_projection_routes_through_engine_preset_resolution(monkeypatch):
    """The frozen trace projection consults the generic policy entry.

    Review fix (Task 11 round 1): ``_response_projection_profile`` read the
    static moonshot/zai map directly, so an engine-driven preset with a
    proprietary reasoning disposition would get gateway-side proprietary
    handling with no frozen trace projection. A synthetic engine record
    proves the trace path now resolves through ``resolve_finish_policy``.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_provider_gateway as gateway_module
    from tldw_chatbook.Chat.console_trace_service import (
        _response_projection_profile,
    )
    from tldw_chatbook.provider_registry import DATABRICKS, RECORDS_BY_KEY

    synthetic = replace(
        DATABRICKS,
        key="synthetic-engine-projection",
        display_name="Synthetic Engine Projection",
        reasoning_disposition="proprietary",
    )
    # Swap the gateway's registry view (a copy, so the shared registry dict
    # and the resolver's per-key cache never see the synthetic record).
    monkeypatch.setattr(
        gateway_module,
        "RECORDS_BY_KEY",
        {**RECORDS_BY_KEY, synthetic.key: synthetic},
    )
    monkeypatch.setattr(
        gateway_module,
        "_RESOLVED_FINISH_POLICIES",
        dict(gateway_module._RESOLVED_FINISH_POLICIES),
    )
    resolution = SimpleNamespace(
        execution_key=synthetic.key,
        thinking_stream_disposition="proprietary",
        thinking_round_trip_version=1,
        model="synthetic-model",
        continuation_protocol="chat_completions",
    )

    profile = _response_projection_profile(resolution)

    assert profile is not None
    assert profile["provider"] == synthetic.key
    assert profile["source_format"] == "reasoning_content"
