"""Compat tests for the TASK-16174 Phase K retirement of the parent-inclusion knobs.

`include_parent_docs` / `parent_size_threshold` / `parent_inclusion_strategy`
were shipped, user-switchable and wired to nothing (spec:
`Docs/superpowers/specs/2026-08-15-rag-agentic-expansion-design.md`). Retiring a
dataclass field is not free: `RAGConfig.from_dict` builds `SearchConfig(**search_data)`
straight from a user-editable dict, so a saved TOML that still carries a retired key
would raise `TypeError` on load. These tests pin the retirement AND the survival of
such a saved config.

Loguru does not feed pytest's `caplog` on its own; the sink installed here is the
repo's own `_forward_loguru_to_standard` bridge (precedent:
`Tests/test_persistent_diagnostic_boundary.py:97`).
"""

from __future__ import annotations

import inspect
import logging

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import _forward_loguru_to_standard
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig


@pytest.fixture()
def loguru_caplog(caplog):
    """Make loguru warnings visible to pytest's `caplog`."""
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    caplog.set_level(logging.DEBUG)
    try:
        yield caplog
    finally:
        loguru_logger.remove(sink_id)


def test_saved_config_with_retired_parent_keys_loads(loguru_caplog):
    """A saved config still carrying the three retired keys loads, warns, drops them."""
    data = {
        "search": {
            "include_parent_docs": True,
            "parent_size_threshold": 5000,
            "parent_inclusion_strategy": "size_based",
            "default_top_k": 7,
        }
    }
    cfg = RAGConfig.from_dict(data)  # must NOT raise TypeError
    assert cfg.search.default_top_k == 7
    assert not hasattr(cfg.search, "include_parent_docs")
    assert not hasattr(cfg.search, "parent_size_threshold")
    assert not hasattr(cfg.search, "parent_inclusion_strategy")
    messages = [r.message for r in loguru_caplog.records]
    assert any("include_parent_docs" in m for m in messages), messages
    assert any("parent_size_threshold" in m for m in messages), messages
    assert any("parent_inclusion_strategy" in m for m in messages), messages


def test_unknown_search_key_is_dropped_with_notice(loguru_caplog):
    """Any unknown search key degrades to an ignored key with a logged notice."""
    cfg = RAGConfig.from_dict({"search": {"never_a_field": 1}})
    assert not hasattr(cfg.search, "never_a_field")
    assert any("never_a_field" in r.message for r in loguru_caplog.records)


def test_no_profile_sets_parent_inclusion():
    """No shipped profile switches on the retired parent-inclusion surface."""
    import tldw_chatbook.RAG_Search.config_profiles as m

    src = inspect.getsource(m)
    assert "include_parent_docs" not in src
    assert "parent_size_threshold" not in src
    assert "parent_inclusion_strategy" not in src


def test_saved_profile_json_with_retired_keys_still_loads():
    """The real-world compat path: a user's saved custom profile JSON.

    Profiles round-trip through `ProfileConfig.to_dict()` -> disk ->
    `ProfileConfig.from_dict()`, which hands `rag_config` to `RAGConfig.from_dict`.
    A profile saved before the retirement carries the three keys inside
    `rag_config["search"]`; it must load rather than take profile loading down.
    """
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig

    profile = ProfileConfig.from_dict(
        {
            "id": "legacy_custom",
            "name": "Legacy Custom",
            "description": "Saved before TASK-16174 retired the parent knobs",
            "profile_type": "custom",
            "rag_config": {
                "search": {
                    "include_parent_docs": True,
                    "parent_size_threshold": 8000,
                    "parent_inclusion_strategy": "size_based",
                    "default_top_k": 12,
                }
            },
        }
    )
    assert profile.rag_config.search.default_top_k == 12
    assert not hasattr(profile.rag_config.search, "include_parent_docs")
