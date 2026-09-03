"""TASK-25904: oversized tool results spill to disk instead of truncating.

The 32 KiB ceiling used to cut the tail unrecoverably; now, when a spill
home exists (the Console scratch root doubles as it), the FULL output is
written atomically to a restricted file inside that root and the model gets
a bounded preview naming the pre-truncation size and the read-back path.
Without a spill home (standalone providers), behavior is byte-identical to
before.
"""

from __future__ import annotations

import os
from pathlib import Path

from tldw_chatbook.Agents.local_tool_provider import (
    _MAX_RESULT_BYTES,
    LocalToolExposure,
    LocalToolProvider,
    LocalToolSpec,
    _fit_or_spill_result,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def test_small_results_are_returned_inline_untouched(tmp_path: Path) -> None:
    """AC#6: no new file writes under the ceiling."""
    spill = tmp_path / "spill"
    text = "small output"
    assert _fit_or_spill_result(text, spill_dir=spill, invocation_id="i1") is text
    assert not spill.exists()


def test_oversized_result_spills_in_full_with_a_stating_preview(tmp_path: Path) -> None:
    """AC#1/#2/#3."""
    spill_dir = tmp_path / "spill"
    body = "x" * (_MAX_RESULT_BYTES + 5_000)
    fitted = _fit_or_spill_result(body, spill_dir=spill_dir, invocation_id="call-1")

    files = list(spill_dir.iterdir())
    assert len(files) == 1
    spill = files[0]
    assert spill.read_text() == body, "the FULL output must be recoverable"
    mode = spill.stat().st_mode & 0o777
    assert mode == 0o600, f"restrictive permissions expected, got {oct(mode)}"
    assert f"{len(body.encode('utf-8')):,} bytes total" in fitted
    assert spill.name in fitted, "the preview must name the read-back path"
    assert "fs_read" in fitted
    assert fitted.startswith("x" * 100), "the preview keeps the head"


def test_without_a_spill_home_behavior_is_todays_truncation(tmp_path: Path) -> None:
    body = "y" * (_MAX_RESULT_BYTES + 5_000)
    fitted = _fit_or_spill_result(body, spill_dir=None, invocation_id="i1")
    assert fitted.endswith("… [truncated]")
    assert "bytes total" not in fitted


def test_provider_spills_into_the_redaction_root_with_relative_path(
    tmp_path: Path,
) -> None:
    """AC#4: the model-facing path is relative to the scratch root the
    fs tools already resolve against -- no absolute locator leaks."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    big = "z" * (_MAX_RESULT_BYTES + 1_000)
    spec = LocalToolSpec(
        name="bigtool",
        description="returns a lot",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: big,
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(),
    )
    provider = LocalToolProvider(
        workspace_root=workspace,
        specs=[spec],
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        result_redaction_root=scratch,
    )
    catalog = provider.list_catalog()
    tool_id = next(t.id for t in catalog if t.id.endswith("bigtool"))

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    spill_files = list((scratch / "tool-spill").iterdir())
    assert len(spill_files) == 1
    assert spill_files[0].read_text() == big
    assert str(scratch) not in (result.content or ""), (
        "the absolute scratch locator must never reach the model"
    )
    assert "tool-spill/" in (result.content or "")


def test_per_run_aggregate_budget_spills_even_under_the_ceiling(
    tmp_path: Path,
) -> None:
    """AC#5: once a run's cumulative inline output passes the aggregate
    budget, further large-ish results spill instead of stacking inline."""
    from tldw_chatbook.Agents.local_tool_provider import (
        _AGGREGATE_INLINE_BUDGET_BYTES,
        _SPILL_FLOOR_BYTES,
    )
    from tldw_chatbook.Agents.run_context import use_run_id

    workspace = tmp_path / "ws"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    chunk = "c" * 20_000  # under the 32 KiB ceiling, over the spill floor
    spec = LocalToolSpec(
        name="chunky",
        description="returns 20k",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: chunk,
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(),
    )
    provider = LocalToolProvider(
        workspace_root=workspace,
        specs=[spec],
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        result_redaction_root=scratch,
    )
    tool_id = next(
        t.id for t in provider.list_catalog() if t.id.endswith("chunky")
    )
    calls_to_exceed = _AGGREGATE_INLINE_BUDGET_BYTES // len(chunk) + 2

    with use_run_id("run-agg"):
        results = [provider.invoke(tool_id, {}) for _ in range(calls_to_exceed)]

    assert all(r.ok for r in results)
    spilled = [r for r in results if "bytes total" in (r.content or "")]
    inline = [r for r in results if (r.content or "") == chunk]
    assert inline, "early results stay inline"
    assert spilled, "past the aggregate budget, results must spill"
    assert results[-1] in spilled, "the budget never resets within a run"
    assert len(chunk) > _SPILL_FLOOR_BYTES
