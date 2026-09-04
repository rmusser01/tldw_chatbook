"""Product maturity Phase 6.5 recovery/setup/documentation alignment."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Static

from Tests.UI.app_factory import _build_test_app
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
    assert not leaked_prefixes, (
        f"evidence contains local filesystem prefix(es): {leaked_prefixes}"
    )


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

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting),
        patch("tldw_chatbook.config.get_cli_setting", side_effect=_test_cli_setting),
    ):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "home"
                    and app.screen.__class__.__name__ == "HomeScreen"
                ),
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
                lambda: (
                    app.current_tab == "chat"
                    and app.screen.__class__.__name__ == "ChatScreen"
                ),
            )
            console_text = _screen_text(app)
            assert "Blocked" in console_text
            assert "Provider: OpenAI" in console_text
            assert "Set up provider" in console_text
            assert (
                app.screen._console_provider_blocker_copy()
                == "Provider setup needed: API key missing for OpenAI"
            )

            await app.handle_screen_navigation(NavigateToScreen("acp"))
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "acp"
                    and app.screen.__class__.__name__ == "ACPScreen"
                ),
            )
            acp_text = _screen_text(app)
            assert "Runtime not configured" in acp_text
            assert "Why: no ACP-compatible runtime is configured." in acp_text
            assert "Next: Configure ACP runtime setup in ACP before launch." in acp_text
            assert "Owner: ACP runtime." in acp_text

            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "mcp"
                    and app.screen.__class__.__name__ == "MCPScreen"
                ),
            )
            mcp_text = _screen_text(app)
            # task-2241: the plain-English explainer leads the header (the
            # old jargon purpose line is deleted; the mode chips already
            # enumerate what the screen manages).
            assert (
                "MCP (Model Context Protocol) lets chatbook use external "
                "tools — most people never need to change anything here."
                in mcp_text
            )
            assert "scoped tools" not in mcp_text
            assert (
                "Next: select Inventory to inspect tools and actions." not in mcp_text
            )

            # The legacy Inventory section was retired with the MCP Hub
            # workbench. Recovery now starts in Servers mode: on a fresh
            # install the lone built-in row is pre-selected (task-2240), so
            # its detail view offers the turned-off built-in's opt-in
            # Enabled checkbox directly (F-051) -- not a problem callout --
            # and the overview's primary Add server action is one
            # "← All servers" click away.
            def mcp_recovery_controls_are_ready() -> bool:
                enabled_boxes = list(app.screen.query("#mcp-builtin-enabled"))
                return bool(
                    enabled_boxes
                    and enabled_boxes[0].region.width > 0
                    and enabled_boxes[0].region.height > 0
                )

            await _wait_until(
                pilot,
                mcp_recovery_controls_are_ready,
            )
            builtin_enabled = app.screen.query_one("#mcp-builtin-enabled", Checkbox)
            assert builtin_enabled.display is True
            assert builtin_enabled.disabled is False
            assert builtin_enabled.value is False

            # The overview (Add server + the Enable affordance row) is one
            # breadcrumb click away from the pre-selected detail view.
            def mcp_overview_recovery_controls_are_ready() -> bool:
                add_servers = list(app.screen.query("#mcp-add-server"))
                enable_affordances = list(app.screen.query("#mcp-builtin-enable"))
                return bool(
                    add_servers
                    and enable_affordances
                    and add_servers[0].region.width > 0
                    and add_servers[0].region.height > 0
                    and enable_affordances[0].region.width > 0
                    and enable_affordances[0].region.height > 0
                )

            await pilot.click("#mcp-detail-back")
            await _wait_until(
                pilot,
                mcp_overview_recovery_controls_are_ready,
            )
            add_server = app.screen.query_one("#mcp-add-server", Button)
            builtin_enable = app.screen.query_one("#mcp-builtin-enable", Button)
            assert str(add_server.label).strip() == "Add server"
            assert add_server.display is True
            assert add_server.disabled is False
            assert add_server.region.width > 0
            assert add_server.region.height > 0
            assert builtin_enable.display is True
            assert builtin_enable.disabled is False
            assert builtin_enable.region.width > 0
            assert builtin_enable.region.height > 0
            assert "turned off" in str(builtin_enable.label).lower()
            assert "Enable" in str(builtin_enable.label)

            await app.handle_screen_navigation(NavigateToScreen("library"))
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "library"
                    and app.screen.__class__.__name__ == "LibraryScreen"
                ),
            )
            await _wait_until(
                pilot,
                lambda: (
                    "Library source services unavailable; retry Library later."
                    in _screen_text(app)
                ),
            )
            library_text = _screen_text(app)
            assert (
                "Library source services unavailable; retry Library later."
                in library_text
            )
            # The retired inspector pane's "No source selected." placeholder
            # went away with the legacy workbench chrome; the shell canvas
            # landing copy is the surviving empty/no-source cue (design: the
            # canvas empty state carries the landing-page guidance).
            assert (
                "Search everything, pick a section, or add something new." in library_text
            )


def test_phase6_mcp_recovery_doc_matches_current_hub_workflow() -> None:
    mcp_row = _markdown_table_row(
        _text(RECOVERY_DOC),
        "MCP server management",
    )

    assert len(mcp_row) == 4
    assert "`MCP: Not wired - MCP servers.`" in mcp_row[1]
    assert "`Add server`" in mcp_row[1]
    assert "`Import…`" in mcp_row[1]
    assert "Open MCP → Servers" in mcp_row[2]
    assert "then use Tools" in mcp_row[2]
    assert "Inventory" not in " ".join(mcp_row)
