"""Regression tests for the standalone Console mount profiler."""

from __future__ import annotations

import pytest

from Tests.Performance.run_console_mount_profile import (
    _outgoing_detached_elapsed_ms,
)


def test_outgoing_detached_elapsed_uses_unmount_completion_timestamp() -> None:
    assert (
        _outgoing_detached_elapsed_ms(
            {"completed_at": 12.5},
            started=10.0,
        )
        == 2500.0
    )


def test_outgoing_detached_elapsed_rejects_a_missing_unmount_observation() -> None:
    with pytest.raises(RuntimeError, match="outgoing unmount was not observed"):
        _outgoing_detached_elapsed_ms({}, started=10.0)
