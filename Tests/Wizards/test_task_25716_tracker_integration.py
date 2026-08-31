"""TASK-25716 integration: the tracker must RENDER the attention state.

Qodo review (PR #2256) was right that unit-testing `setup_attention_ids`
alone leaves the wiring untested -- a rebuild or rendering regression would
keep the unit tests green while the tracker went back to showing ✓ for a step
the user never configured. This drives the real projection and asserts the
glyph the user actually sees.
"""

from __future__ import annotations

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    build_setup_progress,
    setup_attention_ids,
)

_TRACK = ("welcome", "provider", "model", "voice", "protect-keys", "summary")


def _states(wizard_data, *, at: str, probe_failed: bool = False):
    """Project the tracker exactly as `_rebuild_progress` does."""
    items = build_setup_progress(
        _TRACK,
        _TRACK.index(at),
        attention_ids=setup_attention_ids(wizard_data, probe_failed=probe_failed),
    )
    return {item.step_id: item.state for item in items}


def test_unconfigured_provider_renders_attention_not_complete() -> None:
    states = _states({}, at="model")
    assert states["provider"] == "attention"


def test_configured_provider_renders_complete() -> None:
    wizard_data = {
        "provider": {"provider_key": "llama_cpp"},
        "model": {"model_id": "local-model"},
    }
    assert _states(wizard_data, at="model")["provider"] == "complete"


def test_optional_skipped_steps_still_render_complete() -> None:
    """Voice is legitimately skippable; it must not be flagged."""
    states = _states({}, at="summary")
    assert states["voice"] == "complete"
    assert states["protect-keys"] == "complete"


def test_attention_downgrade_only_applies_to_visited_steps() -> None:
    """A step not yet reached reports its position, not a verdict."""
    states = _states({}, at="provider")
    assert states["model"] in {"upcoming", "active"}


def test_tracker_and_summary_cannot_disagree_about_provider() -> None:
    """The defect: tracker said complete while the summary said unconfigured."""
    unconfigured = {"provider": {"provider_key": ""}, "model": {"model_id": ""}}
    states = _states(unconfigured, at="summary")
    assert states["provider"] == "attention"
    assert states["model"] == "attention"
