"""TASK-1260: the suite-wide Hypothesis profile must stay loaded.

`test_safe_paths_always_validate` failed once inside a three-suite run, passed
alone, passed on re-run, and passed on a clean pre-change baseline with the
identical command. It is a property that creates a `TemporaryDirectory` plus up
to four directories per example, and Hypothesis' default per-example deadline is
200ms -- which a machine running 10+ concurrent pytest processes crosses on work
that is not actually slow.

A deadline that fails a property which *holds* is measuring the machine, not the
code. The cost is in attribution, not the failure: establishing that one instance
was not a regression took five runs across two worktrees.

These tests exist so the profile cannot be quietly dropped or "tightened back
up" as an apparent improvement.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")


def test_per_example_deadline_is_disabled() -> None:
    """A loaded machine must not be able to fail a property that holds."""
    assert hypothesis.settings.default.deadline is None, (
        "Hypothesis' per-example deadline is active again; property tests doing "
        "filesystem or database work will fail intermittently under load and "
        "read as regressions (TASK-1260)"
    )


def test_too_slow_health_check_is_suppressed() -> None:
    """`too_slow` fires for the same reason the deadline does: load."""
    suppressed = hypothesis.settings.default.suppress_health_check
    assert hypothesis.HealthCheck.too_slow in suppressed, (
        f"HealthCheck.too_slow is no longer suppressed, got {list(suppressed)}"
    )


def test_the_profile_is_the_one_this_repo_registered() -> None:
    """Guard the guard: a default-settings object would satisfy neither check
    by accident, but naming the profile makes the intent explicit if someone
    later registers a different one."""
    assert "tldw" in hypothesis.settings._profiles, (
        "the 'tldw' Hypothesis profile is not registered; Tests/conftest.py "
        "should register and load it"
    )
