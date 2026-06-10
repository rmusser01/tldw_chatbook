"""Product maturity Phase 6.5 recovery/setup/documentation alignment."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
RECOVERY_DOC = Path("Docs/Development/release-recovery-setup.md")
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-5-recovery-setup-docs-alignment.md"
)
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)
TASK_13_5 = Path(
    "backlog/tasks/task-13.5 - Phase-6.5-Recovery-setup-and-documentation-alignment.md"
)

REQUIRED_RECOVERY_BLOCKERS = {
    "provider-model",
    "server",
    "acp-runtime",
    "mcp-management",
    "optional-dependency",
    "missing-source",
}
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, relative_path in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        target = tmp_path / relative_path
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(target))

    config_path = tmp_path / "xdg-config" / "tldw_cli" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_5_RECOVERY_DOCS_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_5_RECOVERY_DOCS_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _markdown_table_row(markdown: str, first_cell_text: str) -> list[str]:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or first_cell_text not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == first_cell_text:
            return cells
    raise AssertionError(f"Missing markdown table row for {first_cell_text!r}")


def _recovery_matrix_rows(evidence: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] in REQUIRED_RECOVERY_BLOCKERS:
            rows[cells[0]] = cells
    return rows


@pytest.mark.asyncio
async def test_phase6_recovery_copy_is_visible_in_running_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting), patch(
        "tldw_chatbook.config.get_cli_setting", side_effect=_test_cli_setting
    ):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )
            home_text = _screen_text(app)
            assert "Model: Blocked" in home_text
            assert "RAG: Missing sources" in home_text
            assert "Set up Console model" in home_text
            assert "Console needs a working model before live AI tasks." in home_text
            assert "Server sync: Configured; local mode" in home_text
            assert "Local mode is active. Server sync is optional." in home_text

            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            console_text = _screen_text(app)
            assert "Configure provider in Settings." in console_text
            assert "OPENAI_API_KEY" in console_text or "Provider setup needed" in console_text
            assert "RAG/source: not staged" in console_text
            assert "MCP: Not wired - Manage servers in MCP." in console_text
            assert "ACP: Connected - Stage ACP session payloads." in console_text

            await app.handle_screen_navigation(NavigateToScreen("acp"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "acp" and app.screen.__class__.__name__ == "ACPScreen",
            )
            acp_text = _screen_text(app)
            assert "Runtime not configured" in acp_text
            assert "Why: no ACP-compatible runtime is configured." in acp_text
            assert "Next: Configure ACP runtime setup in ACP before launch." in acp_text
            assert "Owner: ACP runtime." in acp_text

            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "mcp" and app.screen.__class__.__name__ == "MCPScreen",
            )
            mcp_text = _screen_text(app)
            assert "Manage MCP servers, scoped tools, permissions, and audit readiness." in mcp_text
            assert "Select Section: Inventory to inspect runnable MCP tools." in mcp_text

            await app.handle_screen_navigation(NavigateToScreen("library"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            library_text = _screen_text(app)
            assert "Library source services unavailable; retry Library later." in library_text
            assert "No source selected." in library_text


def test_phase6_recovery_docs_evidence_and_tracking_are_current() -> None:
    evidence = _text(EVIDENCE)
    readme = _text(PHASE_6_README)
    tracker = _text(TRACKER)
    recovery_doc = _text(RECOVERY_DOC)
    parent_task = _text(TASK_13)
    task = _text(TASK_13_5)
    metadata = _metadata(evidence)

    _assert_no_local_path_prefixes(evidence)
    assert metadata["task"] == "TASK-13.5"
    assert metadata["parent_task"] == "TASK-13"
    assert metadata["decision"] == "recovery_setup_docs_alignment_recorded"
    assert set(metadata["recovery_blockers_checked"]) == REQUIRED_RECOVERY_BLOCKERS
    assert metadata["p0_p1_findings"] == []
    assert metadata["screenshot_gate"] == "not_required_no_visible_ui_changes"
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Recovery Matrix",
        "## Running-App Replay Notes",
        "## Documentation Alignment",
        "## P0/P1 Decision",
        "## Residual Risk",
        "## Verification",
    ):
        assert section in evidence
    assert "running Textual app" in evidence
    assert "Tests/UI/test_product_maturity_phase6_recovery_docs.py" in evidence

    rows = _recovery_matrix_rows(evidence)
    assert set(rows) == REQUIRED_RECOVERY_BLOCKERS
    for blocker, row in rows.items():
        assert len(row) >= 6, f"{blocker} row is missing required columns"
        assert row[1] in {"verified", "accepted-risk"}
        assert row[4] in {"P0", "P1", "P2", "P3", "none"}
        assert RECOVERY_DOC.as_posix() in row[5]

    for required_phrase in (
        "Provider/model setup",
        "Server/local mode",
        "ACP runtime setup",
        "MCP server management",
        "Optional dependency recovery",
        "Local-first baseline",
        "Advanced optional capability groups",
        "Missing optional features do not mean Chatbook is broken",
        "Missing-source recovery",
        "OPENAI_API_KEY",
        "pip install -e \".[embeddings_rag]\"",
        "pip install -e \".[mcp]\"",
        "pip install \"tldw_chatbook[embeddings_rag]\"",
    ):
        assert required_phrase in recovery_doc

    assert EVIDENCE.name in readme
    assert RECOVERY_DOC.as_posix() in readme
    assert "Phase 6.5 Recovery/setup/documentation alignment" in readme
    assert "Status: TASK-13.1 through TASK-13.7 done; Phase 6 verified" in readme

    phase6_row = _markdown_table_row(tracker, "Phase 6: Release Hardening And Documentation")
    assert "verified; TASK-13.1 through TASK-13.7 done" in phase6_row[2]
    assert "TASK-13.5" in phase6_row[3]
    assert EVIDENCE.name in phase6_row[4]
    assert "recovery/setup/documentation alignment" in phase6_row[5].lower()
    assert "release hardening complete" in phase6_row[5].lower()

    qa_row = _markdown_table_row(tracker, "Phase 6.5")
    assert EVIDENCE.as_posix() in qa_row[1]
    assert "verified; TASK-13.5 done" == qa_row[2]

    assert "TASK-13.5" in parent_task
    assert "status: Done" in task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "## Implementation Plan" in task
    assert "## Implementation Notes" in task
