"""Mounted, disposable-profile UAT for the Console Watchlists loop."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.QA.test_console_watchlists_workflow_uat import (
    BRIEFING_ONLY_MARKER,
    _ScriptedWatchlistsGateway,
    _run_external_mcp_boundary,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    _visible_text,
)
from tldw_chatbook import config as app_config
from tldw_chatbook.Chat.provider_setup_persistence import persist_provider_setup
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Skills_Interop.skill_package_inspection import (
    SkillPackageKind,
    inspect_skill_directory,
)
from tldw_chatbook.Subscriptions import (
    briefing_service,
    watchlists_operation_coordinator as coordinator_module,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


_REPOSITORY_ROOT = Path(__file__).parents[2]
_MOUNTED_CAPTURE_BUNDLE = (
    _REPOSITORY_ROOT
    / "Docs"
    / "superpowers"
    / "qa"
    / "console-watchlists-workflow-2026-08"
)


def _validated_capture_root(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _REPOSITORY_ROOT / candidate
    capture_root = validate_path(
        candidate,
        _MOUNTED_CAPTURE_BUNDLE,
        redact_paths=True,
        allow_hidden=True,
    )
    if not capture_root.is_dir():
        raise ValueError("Capture directory must exist")
    return capture_root


def test_mounted_capture_path_is_confined_to_evidence_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside-capture-root"
    outside.mkdir()

    assert _validated_capture_root(_MOUNTED_CAPTURE_BUNDLE) == (
        _MOUNTED_CAPTURE_BUNDLE
    )
    with pytest.raises(ValueError):
        _validated_capture_root(outside)


async def _wait_for_screen(app, pilot, screen_type, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if isinstance(app.screen, screen_type) and app.screen.is_mounted:
            await pilot.pause()
            return app.screen
        await pilot.pause(0.02)
    raise AssertionError(f"{screen_type.__name__} did not mount")


@pytest.mark.asyncio
async def test_mounted_app_drives_console_approvals_receipts_and_navigation(
    tmp_path, monkeypatch
):
    """Drive the real controller/provider composition without public egress."""
    loaded = app_config.load_cli_config_and_ensure_existence(force_reload=True)
    mutation = wizard_state.build_first_run_provider_commit(
        wizard_state.FirstRunProviderDraft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8791/v1/chat/completions",
            credential=wizard_state.ProviderCredentialDraft("none", "", 0),
        ),
        "mounted-persisted-model",
        loaded,
    )
    assert persist_provider_setup(mutation).fully_applied is True

    feed_urls = [
        f"https://public.example/feed-{index}.xml"
        for index in range(1, 4)
    ]

    real_execute = briefing_service.execute_accepted_briefing

    async def scripted_briefing(db, briefing_id, **kwargs):
        return await real_execute(
            db,
            briefing_id,
            chat=lambda **_chat_kwargs: (
                f"## Daily signals\n\n{BRIEFING_ONLY_MARKER} "
                "links the observed campaigns [item 1] [item 2] [item 3]."
            ),
            **kwargs,
        )

    monkeypatch.setattr(
        coordinator_module, "execute_accepted_briefing", scripted_briefing
    )

    app = _build_test_app(configured_default="chat")
    profile = tmp_path / "profile"
    profile.mkdir()
    app.chachanotes_db = CharactersRAGDB(
        profile / "chachanotes.sqlite", client_id="mounted-uat"
    )
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "active-conversation-conflict"

    def receipt_ready(kind: str) -> bool:
        if kind == "checks":
            runs = app.subscriptions_db.list_operations_for_agent(limit=10)[
                "source_runs"
            ]
            return len(runs) == 3 and all(
                row["status"] == "completed" for row in runs
            )
        rows = app.subscriptions_db.list_briefings(1)
        return bool(rows and rows[0]["status"] == "complete")

    gateway = _ScriptedWatchlistsGateway(
        feed_urls,
        receipt_ready=receipt_ready,
    )
    app.console_provider_gateway_factory = lambda: gateway
    app.unified_mcp_service.set_global_default("allow")
    app.unified_mcp_service.set_tool_state(
        "local:__local__",
        "watchlists_create_collection",
        "ask",
    )

    async def scripted_feed(subscription):
        source_id = int(subscription["id"])
        return {
            "items": [
                {
                    "url": f"https://public.example/item-{source_id}",
                    "title": f"Campaign signal {source_id}",
                    "content": f"Observed indicator family {source_id}.",
                    "content_hash": f"mounted-signal-{source_id}",
                    "content_kind": "article",
                    "content_format": "text",
                    "published_date": "2026-08-29T12:00:00+00:00",
                }
            ],
            "stats": {"fixture": "mounted-no-network"},
        }

    app.local_watchlists_service.run_executor = scripted_feed

    framework = tmp_path / "framework"
    framework.mkdir()
    (framework / "README.md").write_text("# Generic framework\n", encoding="utf-8")
    (framework / "pyproject.toml").write_text(
        "[project]\nname='generic-framework'\n", encoding="utf-8"
    )

    async with app.run_test(size=(180, 50)) as pilot:
        console = await _wait_for_screen(app, pilot, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(
            "Create a daily threat-intel Watchlist, run it, brief it, and read it."
        )
        console.query_one("#console-send-message", Button).press()

        approved_round_ids: set[str] = set()
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            console = app.screen
            pending_approval = (
                console._task_resume_state.pending_approval
                if isinstance(console, ChatScreen)
                else None
            )
            round_id = (
                str(pending_approval.get("round_id", ""))
                if isinstance(pending_approval, dict)
                else ""
            )
            call_names = {
                str(call.get("llm_name", "") or call.get("name", ""))
                for call in (
                    pending_approval.get("calls", [])
                    if isinstance(pending_approval, dict)
                    else []
                )
                if isinstance(call, dict)
            }
            matching_button = None
            if isinstance(console, ChatScreen):
                for row in console.query(".approval-row"):
                    header = " ".join(
                        str(item.renderable)
                        for item in row.query(".approval-row-header")
                    )
                    if any(name and name in header for name in call_names):
                        buttons = list(row.query(".approval-row-fast-approve"))
                        if buttons:
                            matching_button = buttons[0]
                            break
            if (
                round_id
                and round_id not in approved_round_ids
                and matching_button is not None
            ):
                approved_round_ids.add(round_id)
                matching_button.press()
            if gateway.stage >= 12 and BRIEFING_ONLY_MARKER in _visible_text(console):
                break
            await pilot.pause(0.03)
        else:
            raise AssertionError(
                f"mounted Console loop did not settle: stage={gateway.stage}, "
                f"text={_visible_text(app.screen)!r}"
            )

        assert approved_round_ids
        assert app.console_runtime.agent_bridge is not None
        assert app.console_runtime.chat_controller is console._console_chat_controller
        assert app.watchlists_operation_coordinator.active_receipt_ids == ()
        briefings = app.subscriptions_db.list_briefings(1)
        assert briefings[0]["status"] == "complete"
        assert briefings[0]["model_used"] == "llama_cpp/mounted-persisted-model"
        with app.subscriptions_db.transaction() as conn:
            stored = conn.execute(
                "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?",
                (1,),
            ).fetchone()
        assert stored["briefing_cadence_seconds"] == 86_400

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        normal_svg = "\n".join(
            line.rstrip()
            for line in app.export_screenshot(
                title="TASK-22868 mounted Console 180x50", simplify=True
            ).splitlines()
        )
        assert BRIEFING_ONLY_MARKER in normal_svg
        await pilot.resize_terminal(160, 42)
        transcript.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        compact_svg = "\n".join(
            line.rstrip()
            for line in app.export_screenshot(
                title="TASK-22868 mounted Console 160x42", simplify=True
            ).splitlines()
        )
        assert BRIEFING_ONLY_MARKER in compact_svg
        if capture_dir := os.environ.get("TASK22868_MOUNTED_CAPTURE_DIR"):
            capture_root = _validated_capture_root(capture_dir)
            (capture_root / "mounted-console-180x50.svg").write_text(
                normal_svg,
                encoding="utf-8",
            )
            (capture_root / "mounted-console-160x42.svg").write_text(
                compact_svg,
                encoding="utf-8",
            )

        app.post_message(NavigateToScreen("watchlists_collections"))
        watchlists = await _wait_for_screen(app, pilot, WatchlistsCollectionsScreen)
        deadline = time.monotonic() + 5.0
        while (
            "Daily threat intelligence" not in _visible_text(watchlists)
            and time.monotonic() < deadline
        ):
            await pilot.pause(0.03)
        assert "Daily threat intelligence" in _visible_text(watchlists)

        app.post_message(NavigateToScreen("settings"))
        settings = await _wait_for_screen(app, pilot, SettingsScreen)
        await pilot.pause(0.2)
        await _wait_for_selector(settings, pilot, "#settings-category-schedules")
        schedules_button = settings.query_one("#settings-category-schedules", Button)
        if not schedules_button.display:
            settings.query_one(
                "#settings-category-group-domain-defaults", Button
            ).press()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                schedules_button = settings.query_one(
                    "#settings-category-schedules", Button
                )
                if schedules_button.display:
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError("Settings schedules category did not expand")
        schedules_button.press()
        await _wait_for_selector(
            settings,
            pilot,
            "#settings-briefing-schedules-status",
            timeout=5.0,
        )
        schedule_status = settings.query_one(
            "#settings-briefing-schedules-status", Static
        )
        assert "Global briefing schedules: Enabled" in str(schedule_status.renderable)

        app.post_message(NavigateToScreen("library"))
        deadline = time.monotonic() + 10.0
        while (
            type(app.screen).__name__ != "LibraryScreen"
            and time.monotonic() < deadline
        ):
            await pilot.pause(0.03)
        assert type(app.screen).__name__ == "LibraryScreen"
        assert inspect_skill_directory(framework).kind is (
            SkillPackageKind.FRAMEWORK_REPOSITORY
        )
    external_root = tmp_path / "external"
    external_root.mkdir()
    external = await _run_external_mcp_boundary(external_root, monkeypatch)
    assert external["private_marker_absent"] is True
    assert external["database_unchanged"] is True
    assert external["console_only_tools"].isdisjoint(external["published_tools"])
