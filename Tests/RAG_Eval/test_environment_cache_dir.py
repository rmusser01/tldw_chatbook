# Tests/RAG_Eval/test_environment_cache_dir.py
"""Always-on tests for `_resolve_model_cache_dir`'s env-sourced path validation.

Pure and gate-free: exercises `Tests/RAG_Eval/harness/environment.py`'s
private cache-dir resolver directly, with `os.environ` monkeypatched — no
`RAG_EVAL`, no embeddings extras, no model. Companion to
`Tests/RAG_Search/test_keyword_leg_db_resolution.py`'s traversal test for
`config.search.media_db_path` (Qodo PR #1428 finding 1), the precedent
Qodo PR #1458 finding 3 asked this module to mirror.
"""
from __future__ import annotations

import pytest

from Tests.RAG_Eval.harness import environment


@pytest.fixture(autouse=True)
def _clear_cache_dir_env_vars(monkeypatch):
    """None of these leak from the real environment into an assertion."""
    for var in (
        environment.MODEL_CACHE_ENV_VAR,
        "HF_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)


def test_no_env_vars_set_falls_back_to_the_unsandboxed_default():
    assert environment._resolve_model_cache_dir() == (
        environment._unsandboxed_home() / ".cache" / "huggingface" / "hub"
    )


def test_valid_escape_hatch_override_is_used_as_is(tmp_path, monkeypatch):
    target = tmp_path / "my-cache"
    monkeypatch.setenv(environment.MODEL_CACHE_ENV_VAR, str(target))
    assert environment._resolve_model_cache_dir() == target


def test_valid_hf_home_gets_the_hub_suffix_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert environment._resolve_model_cache_dir() == tmp_path / "hub"


def test_invalid_escape_hatch_override_degrades_to_the_default_with_a_warning(
    monkeypatch,
):
    """Qodo PR #1458 finding 3: `_resolve_model_cache_dir` built a `Path`
    straight out of an environment variable with no traversal/injection
    screen. An invalid value must never raise and must never be used — it
    degrades to the same default a wholly unset environment resolves to,
    with a logged warning naming the rejected value.
    """
    messages: list[str] = []
    sink_id = environment.logger.add(messages.append, level="WARNING", format="{message}")
    monkeypatch.setenv(environment.MODEL_CACHE_ENV_VAR, "/tmp/evil;rm -rf /")
    try:
        resolved = environment._resolve_model_cache_dir()
    finally:
        environment.logger.remove(sink_id)

    assert resolved == environment._unsandboxed_home() / ".cache" / "huggingface" / "hub"
    rendered = "\n".join(messages)
    assert environment.MODEL_CACHE_ENV_VAR in rendered
    assert "/tmp/evil;rm -rf /" in rendered


def test_invalid_override_never_falls_through_to_a_lower_priority_candidate(
    tmp_path, monkeypatch
):
    """An invalid `MODEL_CACHE_ENV_VAR` must degrade straight to the
    default, not silently promote a lower-priority (and unvalidated-at-this-
    point) candidate like `HF_HUB_CACHE` into its place.
    """
    monkeypatch.setenv(environment.MODEL_CACHE_ENV_VAR, "/tmp/evil;rm -rf /")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert environment._resolve_model_cache_dir() == (
        environment._unsandboxed_home() / ".cache" / "huggingface" / "hub"
    )


def test_invalid_hf_home_degrades_without_appending_the_hub_suffix(monkeypatch):
    """The same screen applies to every candidate, not just the escape hatch."""
    monkeypatch.setenv("HF_HOME", "/tmp/evil;rm -rf /")
    assert environment._resolve_model_cache_dir() == (
        environment._unsandboxed_home() / ".cache" / "huggingface" / "hub"
    )
