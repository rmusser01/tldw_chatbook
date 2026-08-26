"""Production-CSS Textual evidence driver for TASK-22033.

The rig mounts the real Library screen against isolated, seeded Prompt services.
It is closeout evidence, not a replacement for the automated regression suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[5]
EVIDENCE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TASK22033_DATA_DIR", ""))
SCRATCH_ROOT = Path(os.environ.get("TASK22033_SCRATCH_ROOT", ""))
REQUIRED_ENV = (
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "TLDW_CONFIG_PATH",
    "TLDW_TEST_MODE",
    "TASK22033_DATA_DIR",
    "TASK22033_SCRATCH_ROOT",
)
missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
if missing:
    raise SystemExit(f"refusing unisolated run; missing: {', '.join(missing)}")
config_path = Path(os.environ["TLDW_CONFIG_PATH"]).resolve()
scratch_root = SCRATCH_ROOT.resolve()
if (
    scratch_root not in config_path.parents
    or scratch_root not in DATA_DIR.resolve().parents
):
    raise SystemExit("config/data must be contained by TASK22033_SCRATCH_ROOT")
app_data_dir = scratch_root / "app-data"
for isolated_dir in (
    scratch_root,
    DATA_DIR.resolve(),
    app_data_dir,
    config_path.parent,
    Path(os.environ["XDG_CONFIG_HOME"]).resolve(),
    Path(os.environ["XDG_DATA_HOME"]).resolve(),
    Path(os.environ["XDG_CACHE_HOME"]).resolve(),
):
    isolated_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    isolated_dir.chmod(0o700)
if not config_path.exists():
    config_path.write_text(
        f'[paths]\ndata_dir = "{app_data_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    config_path.chmod(0o600)

sys.path.insert(0, str(WORKTREE))

from textual.widgets import Button, Collapsible, Input, Static  # noqa: E402

from Tests.UI.test_library_prompts_canvas import (  # noqa: E402
    _FakePromptScopeServiceWithList,
    _build_test_app,
    _open_prompt_editor,
    _open_prompts_list,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library import (  # noqa: E402
    LibraryAdaptiveReaderShell,
    LibraryPromptWorkPane,
    LibraryPromptsListCanvas,
)


SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


def _region(widget) -> dict[str, int]:
    region = widget.region
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
    }


def _screen_text(app) -> str:
    return "\n".join(
        strip.text.rstrip() for strip in app.screen._compositor.render_strips()
    )


def _capture(app, name: str, facts: dict[str, object]) -> None:
    svg_path = Path(app.save_screenshot(filename=f"{name}.svg", path=str(EVIDENCE_DIR)))
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    (EVIDENCE_DIR / f"{name}.txt").write_text(
        _screen_text(app) + "\n", encoding="utf-8"
    )
    facts["svg"] = svg_path.name
    (EVIDENCE_DIR / f"{name}.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seed_real_prompt(case: str):
    case_dir = DATA_DIR / case
    case_dir.mkdir(parents=True, exist_ok=True)
    db, service = _real_prompt_scope_service(case_dir)
    definition = {
        "kind": "block_prompt",
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Release role",
                        "syntax": "xml",
                        "content": "Be exact.",
                        "xml_tag": "release_role",
                        "mapping_hint": "system-role-v1",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "request",
                        "title": "Release request",
                        "syntax": "freeform",
                        "content": "Summarize {changes}.",
                        "mapping_hint": "user-request-v1",
                    }
                ],
            },
        ],
    }
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Release assistant",
        author="Ada",
        details="Prepares release notes",
        system_prompt="Be exact.",
        user_prompt="Summarize {changes}.",
        keywords=["release", "summary"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "basic"
    return app, db, prompt_id, prompt_uuid, definition


async def geometry_matrix(summary: dict[str, object]) -> None:
    matrix: list[dict[str, object]] = []
    for width, height in SIZES:
        app, _db, prompt_id, _uuid, _definition = _seed_real_prompt(
            f"geometry-{width}x{height}"
        )
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(width, height)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_selector(screen, pilot, "#library-prompt-basic-region")
            shell = screen.query_one(
                "#library-prompts-reader-shell", LibraryAdaptiveReaderShell
            )
            items = screen.query_one(
                "#library-prompts-canvas", LibraryPromptsListCanvas
            )
            work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
            facts = {
                "size": [width, height],
                "effective": {
                    "library_open": shell.effective_layout.library_open,
                    "items_open": shell.effective_layout.items_open,
                    "priority_pane": shell.effective_layout.priority_pane,
                },
                "regions": {
                    "shell": _region(shell),
                    "library": _region(shell.library),
                    "library_grip": _region(shell.library_grip),
                    "items": _region(shell.items),
                    "items_grip": _region(shell.items_grip),
                    "work": _region(shell.work),
                },
                "basic_default": work.editor_mode == "basic",
                "retained_roles": {
                    "items": items.is_mounted,
                    "work": work.is_mounted,
                },
                "grip_names": [shell.library_grip.name, shell.items_grip.name],
            }
            assert facts["basic_default"]
            assert facts["retained_roles"] == {"items": True, "work": True}
            assert shell.region.contains_region(shell.work.region)
            assert shell.library_grip.region.width == shell.items_grip.region.width == 5
            if width == 80:
                shell.library_grip.press()
                shell.items_grip.press()
                await pilot.pause()
                facts["collapsed"] = {
                    "library_open": shell.effective_layout.library_open,
                    "items_open": shell.effective_layout.items_open,
                    "work": _region(shell.work),
                    "restore_focusable": [
                        shell.library_grip.can_focus,
                        shell.items_grip.can_focus,
                    ],
                    "restore_names": [
                        shell.library_grip.name,
                        shell.items_grip.name,
                    ],
                }
                assert facts["collapsed"]["restore_focusable"] == [True, True]
                shell.items_grip.press()
                shell.library_grip.press()
                await pilot.pause()
            _capture(host, f"prompts-{width}x{height}", facts)
            matrix.append(facts)
    summary["geometry"] = matrix


async def preservation_history_and_validation(summary: dict[str, object]) -> None:
    app, db, prompt_id, prompt_uuid, definition = _seed_real_prompt("preservation")
    db.update_prompt_by_id(
        prompt_id,
        {"details": "Prepares exact release notes"},
        expected_version=1,
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_editor_armed,
            message="Prompt editor did not arm",
        )
        screen.query_one("#library-prompt-author", Input).value = "Grace"
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_dirty,
            message="Basic edit did not become dirty",
        )
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_prompt_dirty,
            message=lambda: (
                "Prompt save did not settle: "
                f"status={screen._library_prompt_status!r}, "
                f"conflict={screen._library_prompt_conflict_snapshot is not None!r}, "
                f"version={screen._library_prompt_version!r}, "
                f"can_update={screen._library_prompt_can_update_original()!r}, "
                f"reason={screen._library_prompt_basic_unavailable_reason(screen._current_library_prompt_editor_state())!r}"
            ),
        )
        persisted = db.fetch_prompt_details(prompt_id)
        persisted_definition = json.loads(persisted["prompt_definition"])
        persisted_user = persisted_definition["lanes"][1]["blocks"][0]
        source_user = definition["lanes"][1]["blocks"][0]
        for key in ("id", "title", "syntax", "xml_tag", "mapping_hint"):
            assert persisted_user.get(key) == source_user.get(key)
        assert persisted_user["content"] == source_user["content"]
        assert persisted["author"] == "Grace"

        screen.query_one("#library-prompt-mode-info", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-prompt-info-region").display,
            message="Info mode did not settle",
        )
        history = screen.query_one("#library-prompt-history-collapsible", Collapsible)
        history.collapsed = False
        await pilot.pause()
        info_text = str(
            screen.query_one("#library-prompt-info-provenance", Static).renderable
        )
        facts = {
            "prompt_uuid": prompt_uuid,
            "mode": screen._library_prompt_editor_mode,
            "provenance": info_text,
            "history_title": str(history.title),
            "hidden_advanced_fields_preserved": True,
            "persisted_version": persisted["version"],
        }
        assert "local" in info_text.casefold()
        _capture(host, "prompts-preservation-history", facts)
        summary["preservation_history"] = facts


async def validation_focus(summary: dict[str, object]) -> None:
    app, _db, prompt_id, _uuid, _definition = _seed_real_prompt("validation")
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_editor_armed,
            message="Prompt editor did not arm",
        )
        name = screen.query_one("#library-prompt-name", Input)
        name.value = ""
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_dirty,
            message="Invalid Prompt name did not mark the draft dirty",
        )
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is name,
            message="Validation did not focus the owning Name control",
        )
        facts = {
            "mode": screen._library_prompt_editor_mode,
            "focused": getattr(screen.focused, "id", None),
            "status": screen._library_prompt_status,
            "dirty": screen._library_prompt_dirty,
        }
        assert facts["mode"] == "basic"
        assert facts["focused"] == "library-prompt-name"
        _capture(host, "prompts-validation-focus", facts)
        summary["validation"] = facts


async def bulk_preview(summary: dict[str, object]) -> None:
    app, _db, prompt_id, _uuid, _definition = _seed_real_prompt("bulk")
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        items = screen.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        items.query_one("#library-prompts-select", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_select_mode and work.bulk_read_only,
            message="Bulk preview did not become read-only",
        )
        status = await _wait_for_selector(screen, pilot, "#library-prompt-bulk-status")
        assert isinstance(status, Static)
        facts = {
            "status": str(status.renderable),
            "loaded_prompt_id": screen._selected_prompt_id,
            "name_disabled": work.query_one("#library-prompt-name", Input).disabled,
            "save_disabled": work.query_one("#library-prompt-save", Button).disabled,
        }
        assert "Not included" in facts["status"]
        assert facts["name_disabled"] and facts["save_disabled"]
        _capture(host, "prompts-bulk-readonly-preview", facts)
        summary["bulk"] = facts


async def import_and_retry(summary: dict[str, object]) -> None:
    app, _db, prompt_id, _uuid, _definition = _seed_real_prompt("import")
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        items = screen.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        identities = [id(items), id(work)]
        items.query_one("#library-prompts-import", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-import-path")
        facts = {
            "items_retained": screen.query_one("#library-prompts-canvas") is items,
            "work_retained": screen.query_one("#library-prompt-work-pane") is work,
            "identities": identities,
            "opened_from_editor": screen._selected_prompt_id == prompt_id,
            "editor_hidden": not bool(work.query("#library-prompt-name")),
            "import_path_visible": screen.query_one(
                "#library-prompts-import-path", Input
            ).display,
        }
        assert all(
            facts[key]
            for key in (
                "items_retained",
                "work_retained",
                "opened_from_editor",
                "editor_hidden",
                "import_path_visible",
            )
        )
        _capture(host, "prompts-import-work-pane", facts)
        summary["import"] = facts

    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    service = _FakePromptScopeServiceWithList(
        [{"id": 5, "name": "Recovered", "version": 1}],
        browse_failures=1,
    )
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_browse_controller.result.status == "error",
            message="Prompt browse error did not settle",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(screen.query("#library-prompts-retry"))
                and (
                    bool(screen.query("#library-prompts-error"))
                    or bool(screen.query("#library-prompts-page-status"))
                )
            ),
            message=lambda: (
                "Prompt browse error controls did not settle together: "
                f"route={screen._library_selected_row_id!r}, "
                f"view={screen._library_prompts_view!r}, "
                f"status={screen._library_prompt_browse_controller.result.status!r}, "
                f"shells={len(screen.query('#library-prompts-reader-shell'))}, "
                f"canvases={len(screen.query('#library-prompts-canvas'))}, "
                f"retry={len(screen.query('#library-prompts-retry'))}, "
                f"error={len(screen.query('#library-prompts-error'))}, "
                f"projection_depth={screen._library_canvas_projection_depth!r}, "
                f"dirty={screen._library_entry_reconcile_dirty!r}, "
                f"pending={screen._library_entry_reconcile_pending!r}, "
                f"retry_generation={screen._library_entry_reconcile_retry_generation!r}"
            ),
        )
        error_widgets = screen.query("#library-prompts-error")
        error_copy = str(
            (
                error_widgets.first(Static)
                if error_widgets
                else screen.query_one("#library-prompts-page-status", Static)
            ).renderable
        )
        _capture(host, "prompts-browse-error", {"error": error_copy})
        screen.query_one("#library-prompts-retry", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")
        facts = {
            "browse_calls": len(service.browse_calls),
            "status": screen._library_prompt_browse_controller.result.status,
            "recovered_row": bool(screen.query("#library-prompt-row-5")),
        }
        assert facts == {"browse_calls": 2, "status": "ready", "recovered_row": True}
        _capture(host, "prompts-browse-retry", facts)
        summary["retry"] = facts


async def detail_failure_and_retry(summary: dict[str, object]) -> None:
    """Capture truthful selected-versus-loaded detail recovery."""
    app, db, first_id, _uuid, _definition = _seed_real_prompt("detail-retry")
    second_id, _second_uuid, _message = db.add_prompt(
        name="Incident assistant",
        author="Grace",
        details="Prepares incident summaries",
        system_prompt="Preserve verified facts.",
        user_prompt="Summarize {incident}.",
        keywords=["incident", "summary"],
    )
    service = app.prompt_scope_service
    original_get_prompt = service.get_prompt

    async def fail_second_detail(*, prompt_identifier, **kwargs):
        if prompt_identifier == second_id:
            raise RuntimeError("simulated live detail failure")
        return await original_get_prompt(
            prompt_identifier=prompt_identifier,
            **kwargs,
        )

    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, first_id)
        service.get_prompt = fail_second_detail
        screen.query_one(f"#library-prompt-row-{second_id}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-detail-retry")
        status = screen.query_one("#library-prompt-detail-status", Static)
        failure = {
            "selected_id": screen._selected_prompt_id,
            "loaded_id": screen._library_prompt_loaded_id,
            "loaded_name": screen.query_one("#library-prompt-name", Input).value,
            "editor_locked": screen.query_one(
                "#library-prompt-name", Input
            ).disabled,
            "notice": str(status.renderable),
        }
        assert failure["selected_id"] == second_id
        assert failure["loaded_id"] == first_id
        assert failure["loaded_name"] == "Release assistant"
        assert failure["editor_locked"] is True
        assert "remains selected" in failure["notice"]
        _capture(host, "prompts-detail-failure", failure)

        service.get_prompt = original_get_prompt
        screen.query_one("#library-prompt-detail-retry", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_loaded_id == second_id
                and screen.query_one("#library-prompt-name", Input).value
                == "Incident assistant"
            ),
            message="Live Prompt detail retry did not adopt the selection",
        )
        recovered = {
            "selected_id": screen._selected_prompt_id,
            "loaded_id": screen._library_prompt_loaded_id,
            "loaded_name": screen.query_one("#library-prompt-name", Input).value,
            "editor_locked": screen.query_one(
                "#library-prompt-name", Input
            ).disabled,
            "retry_visible": bool(screen.query("#library-prompt-detail-retry")),
        }
        assert recovered["selected_id"] == recovered["loaded_id"] == second_id
        assert recovered["loaded_name"] == "Incident assistant"
        assert recovered["editor_locked"] is False
        assert recovered["retry_visible"] is False
        _capture(host, "prompts-detail-retry", recovered)
        summary["detail_recovery"] = {
            "failure": failure,
            "recovered": recovered,
        }


async def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "package": str(Path(__import__("tldw_chatbook").__file__).resolve()),
        "config": str(config_path),
        "data": str(DATA_DIR.resolve()),
        "sizes": [list(size) for size in SIZES],
    }
    selected = set(sys.argv[1:]) or {
        "geometry",
        "preservation",
        "bulk",
        "import",
        "detail",
    }
    if "geometry" in selected:
        await geometry_matrix(summary)
    if "preservation" in selected:
        await preservation_history_and_validation(summary)
        await validation_focus(summary)
    if "bulk" in selected:
        await bulk_preview(summary)
    if "import" in selected:
        await import_and_retry(summary)
    if "detail" in selected:
        await detail_failure_and_retry(summary)
    (EVIDENCE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
