"""Deterministic full-app QA capture runner for the Console Prompt Workbench.

The runner mounts the real ``TldwCli`` application with its bundled stylesheet.
Only unrelated external services and the provider/network boundary use the
repository's supported test injection seams. Prompt Browse and Library operate
against a real, isolated ``PromptsDatabase``.
"""

# ruff: noqa: E402 -- direct execution must expose the repository test helpers.

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMPORT_SANDBOX = Path(tempfile.mkdtemp(prefix="console-prompt-qa-bootstrap-")).resolve()
(IMPORT_SANDBOX / "config").mkdir(mode=0o700)
(IMPORT_SANDBOX / "data").mkdir(mode=0o700)
IMPORT_CONFIG_PATH = IMPORT_SANDBOX / "config" / "config.toml"
IMPORT_CONFIG_PATH.write_text(
    f'[paths]\ndata_dir = "{IMPORT_SANDBOX / "data"}"\n', encoding="utf-8"
)
os.environ["TLDW_CONFIG_PATH"] = str(IMPORT_CONFIG_PATH)
os.environ["XDG_CONFIG_HOME"] = str(IMPORT_SANDBOX / "config")
os.environ["XDG_DATA_HOME"] = str(IMPORT_SANDBOX / "data")

from loguru import logger
from textual.widgets import Button, Checkbox, Input, Static

from Tests.UI.app_factory import _build_test_app, drain_created_dirs
from Tests.UI.test_library_prompts_canvas import _wire_empty_non_prompt_services
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    ServerPromptService,
    build_prompt_scope_service,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ConsoleComposerMenuModal,
)
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor


HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures"
SIZES = ((140, 40), (100, 30), (80, 24))
PLACEHOLDER = re.compile(r"\[\[TLDW_PROTECTED:[^\]]+\]\]")


def _definition(kind: str, *, block_canary: str = "") -> dict[str, Any]:
    return {
        "kind": kind,
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "markdown",
                        "content": block_canary or "Be exact and concise.",
                        "mapping_hint": "Define the model's function.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "markdown",
                        "content": "Produce a verifiable answer.",
                        "mapping_hint": "State the desired outcome.",
                    },
                    {
                        "id": "output",
                        "title": "Output",
                        "syntax": "xml",
                        "xml_tag": "output",
                        "content": "Return concise Markdown.",
                        "mapping_hint": "Describe the answer shape.",
                    },
                ],
            },
        ],
    }


def _seed(db: PromptsDatabase) -> dict[str, int]:
    ids: dict[str, int] = {}

    def add(key: str, **values: Any) -> None:
        prompt_id, _uuid, _message = db.add_prompt(**values)
        assert prompt_id is not None
        ids[key] = prompt_id

    add(
        "legacy",
        name="Legacy release note",
        author="QA",
        details="Legacy two-lane compatibility prompt",
        system_prompt="Keep claims source-backed.",
        user_prompt="Draft a compact release note.",
        keywords=["legacy", "release"],
    )
    add(
        "foreign_v1",
        name="Foreign v1 workflow",
        author="QA",
        details="Read-only structured v1 compatibility record",
        system_prompt="Compatibility system text.",
        user_prompt="Compatibility user text.",
        keywords=["foreign", "structured"],
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition={"schema_version": 1, "blocks": []},
    )
    add(
        "block_prompt",
        name="Structured answer prompt",
        author="QA",
        details="Editable block Prompt",
        system_prompt="# Role\n\nBe exact and concise.",
        user_prompt="# Goal\n\nProduce a verifiable answer.\n\n<output>\nReturn concise Markdown.\n</output>",
        keywords=["structured", "answer"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=_definition("block_prompt"),
        artifact_type="prompt",
    )
    add(
        "recipe",
        name="Reusable answer recipe",
        author="QA",
        details="Editable block Recipe",
        system_prompt="# Role\n\nBe exact and concise.",
        user_prompt="# Goal\n\nProduce a verifiable answer.\n\n<output>\nReturn concise Markdown.\n</output>",
        keywords=["structured", "recipe"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=_definition("block_recipe"),
        artifact_type="recipe",
    )
    add(
        "malformed",
        name="Malformed future artifact",
        author="QA",
        details="Guarded malformed record",
        system_prompt="Read-only compatibility text.",
        user_prompt="Inspect without mutation.",
        keywords=["malformed"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition="{not-json",
        artifact_type="prompt",
    )
    add(
        "future",
        name="Future schema artifact",
        author="QA",
        details="Guarded unsupported record",
        system_prompt="Future compatibility system.",
        user_prompt="Future compatibility user.",
        keywords=["future"],
        prompt_format="structured",
        prompt_schema_version=99,
        prompt_definition={"kind": "future_prompt", "schema_version": 99},
        artifact_type="prompt",
    )
    for index in range(6):
        add(
            f"filler_{index}",
            name=f"Pagination sample {index + 1:02d}",
            author="QA",
            details="Deterministic pagination fixture",
            system_prompt="",
            user_prompt=f"Synthetic local prompt {index + 1:02d}.",
            keywords=["pagination"],
        )
    return ids


class DeterministicGateway:
    """One-call provider fake injected at the final gateway boundary only."""

    def __init__(self) -> None:
        self.auxiliary_calls = 0
        self.stream_calls = 0
        self.hold_next = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.observed_placeholders: set[str] = set()
        self.response_canary = ""

    async def resolve_for_send(self, selection: Any) -> ConsoleProviderResolution:
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=selection.base_url or "http://127.0.0.1:9099",
            model=selection.explicit_model
            or selection.configured_model
            or "qa-local-model",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        )

    async def complete_auxiliary(self, request: Any) -> AuxiliaryCompletionResult:
        self.auxiliary_calls += 1
        if self.hold_next:
            self.hold_next = False
            self.started.set()
            await self.release.wait()
        payload = json.loads(str(request.messages[-1]["content"]))
        source = str(payload["source_prompt"])
        self.observed_placeholders.update(PLACEHOLDER.findall(source))
        if request.response_format["json_schema"]["name"] == "recipe_fill":
            recipe = payload["recipe"]
            block_ids = [
                str(block["id"]) for lane in recipe["lanes"] for block in lane["blocks"]
            ]
            system_text = str(payload.get("system_context", {}).get("text", ""))
            fills = []
            blocks = {
                str(block["id"]): block
                for lane in recipe["lanes"]
                for block in lane["blocks"]
            }
            for block_id in block_ids:
                content = str(blocks[block_id].get("content", ""))
                if block_id == "role":
                    content = "\n".join(part for part in (content, system_text) if part)
                elif block_id == "goal":
                    content = "\n".join(part for part in (content, source) if part)
                fills.append({"block_id": block_id, "content": content})
            response = {
                "kind": "recipe_fill",
                "recipe_fingerprint": payload["recipe_fingerprint"],
                "fills": fills,
                "additional_context": self.response_canary,
            }
        else:
            if "NO_CHANGE_MARKER" in source:
                rewritten = source
            elif "{{ACCOUNT_ID}}" in source:
                rewritten = "Summarize the account outcome in Markdown."
            elif self.response_canary:
                rewritten = f"{source}\n{self.response_canary}"
            else:
                rewritten = source.replace(
                    "Draft a useful answer.",
                    "Produce a concise answer with verifiable claims and a clear stopping condition.",
                )
            response = {"kind": "prompt_rewrite", "rewritten_prompt": rewritten}
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=str(request.resolution.model),
            text=json.dumps(response, ensure_ascii=False),
        )

    async def stream_chat(self, *_args: Any, **_kwargs: Any) -> None:
        self.stream_calls += 1
        raise AssertionError("Prompt improvement must not use normal Console send")


async def _wait(
    pilot: Any,
    condition: Callable[[], bool],
    *,
    label: str,
    timeout: float = 12.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Timed out waiting for {label}")


def _configure_app(app: Any, service: Any, gateway: DeterministicGateway) -> None:
    _wire_empty_non_prompt_services(app)
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    app.app_config.setdefault("console", {}).setdefault("onboarding", {})[
        "first_send_completed"
    ] = True
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "qa-local-model",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "qa-local-model",
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "qa-local-model"
    app.prompt_scope_service = service
    app.console_provider_gateway_factory = lambda: gateway


def _qa_cli_setting(section: str, key: str | None = None, default: Any = None) -> Any:
    if section == "splash_screen" and key == "enabled":
        return False
    return default


@asynccontextmanager
async def _run_test(app: Any, *, size: tuple[int, int]):
    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_qa_cli_setting):
        async with app.run_test(size=size) as pilot:
            yield pilot


async def _close_modal(pilot: Any, modal: ConsolePromptsModal) -> None:
    """Follow the real Close/Discard/Back lifecycle until the shell dismisses."""

    for _attempt in range(6):
        if not isinstance(modal.app.screen_stack[-1], ConsolePromptsModal):
            return
        modal.query_one("#console-prompts-close", Button).press()
        await pilot.pause()
        discard = modal.query("#console-prompts-discard")
        if discard and discard.first().display:
            discard.first(Button).press()
            await pilot.pause()
            continue
        if not isinstance(modal.app.screen_stack[-1], ConsolePromptsModal):
            return
    raise AssertionError("Prompt Workbench did not dismiss after real Back semantics")


async def _console(app: Any, pilot: Any) -> Any:
    if app.current_tab != "chat" or app.screen.__class__.__name__ != "ChatScreen":
        await app.handle_screen_navigation(NavigateToScreen("chat"))
    await _wait(
        pilot,
        lambda: (
            app.screen.__class__.__name__ == "ChatScreen"
            and bool(app.screen.query("#console-shell"))
        ),
        label="real Console screen",
    )
    return app.screen


def _capture(app: Any, name: str, title: str, forbidden: tuple[str, ...] = ()) -> None:
    svg = app.export_screenshot(title=title, simplify=True)
    assert "<svg" in svg and "</svg>" in svg
    assert "Traceback" not in svg and "Internal Error" not in svg
    assert "/Users/" not in svg and "API_KEY" not in svg
    for value in forbidden:
        assert value not in svg
    (CAPTURES / name).write_text(svg, encoding="utf-8")


def _semantic_snapshot(snapshot: Any) -> tuple[Any, ...]:
    """Compare visible draft state while excluding the stale-result generation."""

    return (
        snapshot.segments,
        snapshot.cursor_index,
        snapshot.selection,
        snapshot.edit_serial,
    )


async def _capture_responsive_surfaces(
    size: tuple[int, int], service: Any, forbidden: tuple[str, ...]
) -> dict[str, Any]:
    gateway = DeterministicGateway()
    gateway.response_canary = "Unmatched evidence retained for explicit review."
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    width, height = size
    prefix = f"{width}x{height}"
    observed: dict[str, Any] = {"size": prefix}
    async with _run_test(app, size=size) as pilot:
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft a useful answer.")
        store = console._ensure_console_chat_store()
        store.set_session_system_prompt(store.active_session_id, "Be concise.")

        visible_ids = [
            button.id
            for button in composer.query(Button)
            if button.display and button.region.width > 0 and button.region.height > 0
        ]
        assert "console-composer-menu" in visible_ids
        assert "console-send-message" in visible_ids
        assert "console-dictation" in visible_ids
        assert "console-attach-context" not in visible_ids
        assert "console-save-chatbook" not in visible_ids
        assert not console.query("#console-control-prompts")

        composer.query_one("#console-composer-menu", Button).press()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsoleComposerMenuModal),
            label="composer hamburger menu",
        )
        menu = app.screen_stack[-1]
        action_ids = [
            str(button.id).removeprefix("console-composer-menu-")
            for button in menu.query(".console-composer-menu-item")
        ]
        normal = (
            action_ids[1:]
            if action_ids and action_ids[0] == "save-chat"
            else action_ids
        )
        assert normal[:3] == ["prompts", "attach-context", "save-chatbook"]
        observed["menu_order"] = action_ids
        _capture(
            app,
            f"{prefix}-composer-menu.svg",
            f"Prompt Workbench composer menu {prefix}",
            forbidden,
        )

        menu.query_one("#console-composer-menu-prompts", Button).press()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="Prompt Workbench modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: len(modal.query(".console-prompts-result")) == 10,
            label="first local Prompt page",
        )
        page = str(modal.query_one("#console-prompts-page", Static).renderable)
        assert page == "Page 1 of 2"
        observed["browse_page"] = page
        _capture(
            app,
            f"{prefix}-browse-page-1.svg",
            f"Prompt Workbench Browse {prefix}",
            forbidden,
        )

        if size == (140, 40):
            await modal.switch_source("server")
            await _wait(
                pilot,
                lambda: modal.query_one("#console-prompts-retry", Button).display,
                label="Server unavailable Retry state",
            )
            observed["server_retry"] = True
            _capture(
                app,
                f"{prefix}-server-unavailable.svg",
                "Prompt Workbench Server unavailable",
                forbidden,
            )
            await modal.switch_source("local")
            await _wait(
                pilot,
                lambda: len(modal.query(".console-prompts-result")) == 10,
                label="local Prompt page after source switch",
            )

        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="Improve choices",
        )
        assert not modal.query_one("#console-prompts-auto-improve", Button).disabled
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-recipe-outcome-first")),
            label="Recipe chooser",
        )
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query(PromptBlockEditor)),
            label="shared Recipe block editor",
        )
        editor = modal.query_one(PromptBlockEditor)
        await editor._change_field("goal", "content", "Produce a measurable outcome.")
        editor.query_one("#prompt-block-duplicate-goal", Button).press()
        await pilot.pause()
        assert (
            modal.query_one("#console-prompts-include-system", Checkbox).value is True
        )
        assert editor.query_one("#prompt-editor-apply-system", Checkbox).value is False
        observed["recipe_blocks"] = sum(
            len(lane.blocks) for lane in editor.state.definition.lanes
        )
        _capture(
            app,
            f"{prefix}-recipe-editor.svg",
            f"Prompt Workbench Recipe editor {prefix}",
            forbidden,
        )

        before_fill = composer.capture_draft_snapshot()
        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await _wait(
            pilot,
            lambda: getattr(modal._editor_state, "artifact_type", None) == "prompt",
            label="Filled Prompt mandatory review",
        )
        editor = modal.query_one(PromptBlockEditor)
        mapped = editor.state.definition.lanes[1].blocks[-1]
        assert mapped.id == "additional-context"
        assert mapped.content == gateway.response_canary
        assert composer.capture_draft_snapshot() == before_fill
        assert not modal.query("#console-prompts-recipe-fill")
        duplicate = editor.query_one(
            "#prompt-block-duplicate-additional-context",
            Button,
        )
        save_recipe = editor.query_one("#prompt-editor-save-recipe", Button)
        assert duplicate.disabled is True
        assert save_recipe.disabled is True
        observed["filled_prompt_review"] = True
        observed["mapped_context_duplicate_disabled"] = True
        observed["mapped_context_recipe_save_disabled"] = True
        _capture(
            app,
            f"{prefix}-filled-prompt-review.svg",
            f"Prompt Workbench Filled Prompt review {prefix}",
            forbidden,
        )

        editor.query_one(
            "#prompt-block-delete-additional-context",
            Button,
        ).press()
        await _wait(
            pilot,
            lambda: all(
                block.id != "additional-context"
                for lane in editor.state.definition.lanes
                for block in lane.blocks
            ),
            label="mapped Additional context deletion",
        )
        assert save_recipe.disabled is False
        observed["recipe_save_recovered_after_mapped_context_delete"] = True
    return observed


async def _exercise_improvement_states(
    service: Any, forbidden: tuple[str, ...]
) -> dict[str, Any]:
    gateway = DeterministicGateway()
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("NO_CHANGE_MARKER")
        before = composer.capture_draft_snapshot()
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="no-change Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="no-change Improve mode",
        )
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await _wait(
            pilot,
            lambda: (
                "already looks good"
                in str(
                    modal.query_one(
                        "#console-prompts-improvement-status", Static
                    ).renderable
                ).lower()
            ),
            label="no-change outcome",
        )
        assert composer.capture_draft_snapshot() == before
        observed["no_change"] = True
        await _close_modal(pilot, modal)

        composer.clear_draft()
        composer.insert_text("Draft a useful answer.")
        success_before = composer.capture_draft_snapshot()
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="success Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="success Improve mode",
        )
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await _wait(
            pilot,
            lambda: app.screen_stack[-1] is console,
            label="Auto success dismissal",
        )
        assert "verifiable claims" in composer.draft_text()
        assert composer.improvement_undo_available
        composer.query_one("#console-composer-menu", Button).press()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsoleComposerMenuModal),
            label="Undo composer menu",
        )
        menu = app.screen_stack[-1]
        assert menu.query_one("#console-composer-menu-undo-prompt-improvement", Button)
        _capture(
            app,
            "140x40-auto-success-undo.svg",
            "Prompt Workbench Auto success and Undo",
            forbidden,
        )
        menu.query_one("#console-composer-menu-undo-prompt-improvement", Button).press()
        await _wait(
            pilot,
            lambda: app.screen_stack[-1] is console,
            label="Undo completion",
        )
        assert _semantic_snapshot(
            composer.capture_draft_snapshot()
        ) == _semantic_snapshot(success_before)
        observed["auto_success_undo"] = True

        composer.clear_draft()
        composer.insert_text("Summarize {{ACCOUNT_ID}} without changing the token.")
        veto_before = composer.capture_draft_snapshot()
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="preservation Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="preservation Improve mode",
        )
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-review-user")),
            label="preservation-veto Review",
        )
        assert composer.capture_draft_snapshot() == veto_before
        status = str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        )
        assert status == "Review required before applying"
        observed["preservation_veto_review"] = True
        _capture(
            app,
            "140x40-preservation-review.svg",
            "Prompt Workbench preservation Review",
            forbidden,
        )
        await _close_modal(pilot, modal)

        composer.clear_draft()
        composer.insert_text("Draft a useful answer.")
        cancel_before = composer.capture_draft_snapshot()
        gateway.hold_next = True
        gateway.started.clear()
        gateway.release.clear()
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="cancel Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="cancel Improve mode",
        )
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await gateway.started.wait()
        modal.query_one("#console-prompts-improvement-cancel", Button).press()
        await pilot.pause()
        gateway.release.set()
        await pilot.pause()
        await pilot.pause()
        assert composer.capture_draft_snapshot() == cancel_before
        observed["cancel_late_discard"] = True
    assert gateway.stream_calls == 0
    return observed


async def _capture_guards_and_library(
    service: Any, ids: dict[str, int], forbidden: tuple[str, ...]
) -> dict[str, Any]:
    gateway = DeterministicGateway()
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="guard Prompt modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: len(modal.query(".console-prompts-result")) == 10,
            label="guard Browse page",
        )
        search = modal.query_one("#console-prompts-search", Input)
        search.value = "Foreign v1 workflow"
        await _wait(
            pilot,
            lambda: len(modal.query(".console-prompts-result")) == 1,
            label="foreign v1 search",
        )
        modal.query_one(".console-prompts-result", Button).press()
        await pilot.pause(0.5)
        if not modal.query("#console-prompts-compatibility"):
            browse_status = modal.query_one(
                "#console-prompts-browse-status", Static
            ).renderable
            row_identifier = (
                modal.browse_result.items[0].get("id")
                if modal.browse_result.items
                else None
            )
            raise AssertionError(
                "Foreign-v1 row did not open its compatibility view: "
                f"row_identifier={row_identifier!r}, mode={modal.state.mode!r}, "
                f"status={str(browse_status)!r}"
            )
        await _wait(
            pilot,
            lambda: (
                bool(modal.query("#console-prompts-compatibility"))
                and "cannot be edited losslessly"
                in str(
                    modal.query_one("#console-prompts-compatibility", Static).renderable
                ).lower()
            ),
            label="foreign v1 guard",
        )
        assert not modal.query(PromptBlockEditor)
        observed["foreign_v1_guard"] = True
        _capture(
            app,
            "140x40-foreign-v1-guard.svg",
            "Prompt Workbench foreign v1 guard",
            forbidden,
        )
        await _close_modal(pilot, modal)

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await _wait(
            pilot,
            lambda: (
                app.screen.__class__.__name__ == "LibraryScreen"
                and bool(app.screen.query("#library-row-browse-prompts"))
            ),
            label="real Library screen",
        )
        library = app.screen
        library.query_one("#library-row-browse-prompts", Button).press()
        await _wait(
            pilot,
            lambda: bool(library.query(".library-prompt-row")),
            label="Library Prompt rows",
        )
        labels = [str(button.label) for button in library.query(".library-prompt-row")]
        assert any("Recipe · Local" in label for label in labels)
        assert any("Prompt · Local" in label for label in labels)
        observed["library_type_labels"] = True
        _capture(
            app,
            "140x40-library-prompt-recipe-labels.svg",
            "Library Prompt and Recipe labels",
            forbidden,
        )

        beta_id = ids["filler_5"]
        library.query_one(f"#library-prompt-row-{beta_id}", Button).press()
        await _wait(
            pilot,
            lambda: bool(library.query("#library-prompt-name")),
            label="Library Prompt editor",
        )
        library.query_one(
            "#library-prompt-name", Input
        ).value = "Reusable answer recipe"
        await pilot.pause()
        library.query_one("#library-prompt-save", Button).press()
        await _wait(
            pilot,
            lambda: (
                "Name already in use"
                in str(
                    library.query_one("#library-prompt-save-status", Static).renderable
                )
            ),
            label="Library save-name conflict",
        )
        observed["library_save_conflict"] = True
        _capture(
            app,
            "140x40-library-save-conflict.svg",
            "Library Prompt save conflict",
            forbidden,
        )
    return observed


async def _canary_run(service: Any, log_path: Path) -> dict[str, Any]:
    category_values = {
        category: f"qa-{category}-{uuid.uuid4().hex}"
        for category in (
            "system",
            "user",
            "block",
            "inline_body",
            "inline_label",
            "response",
        )
    }
    gateway = DeterministicGateway()
    gateway.response_canary = category_values["response"]
    definition = _definition("block_recipe", block_canary=category_values["block"])
    await service.save_prompt(
        mode="local",
        name="QA Canary Recipe",
        author="QA",
        details="Ephemeral privacy audit fixture",
        system_prompt=f"# Role\n\n{category_values['block']}",
        user_prompt=(
            "# Goal\n\nProduce a verifiable answer.\n\n"
            "<output>Return concise Markdown.</output>"
        ),
        keywords=["qa", "canary"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    sink_id = -1
    async with _run_test(app, size=(100, 30)) as pilot:
        console = await _console(app, pilot)
        sink_id = logger.add(log_path, level="DEBUG", enqueue=False)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text(category_values["user"])
        composer.insert_file_segment(
            category_values["inline_body"], category_values["inline_label"]
        )
        store = console._ensure_console_chat_store()
        store.set_session_system_prompt(
            store.active_session_id, category_values["system"]
        )
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="canary Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-review-improve")),
            label="canary Improve mode",
        )
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-recipe-saved")),
            label="canary Recipe choices",
        )
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-search")),
            label="canary saved Recipe chooser",
        )
        modal.query_one("#console-prompts-search", Input).value = "QA Canary Recipe"
        await _wait(
            pilot,
            lambda: len(modal.query(".console-prompts-result")) == 1,
            label="canary Recipe search",
        )
        modal.query_one(".console-prompts-result", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query(PromptBlockEditor)),
            label="canary Recipe editor",
        )
        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await _wait(
            pilot,
            lambda: getattr(modal._editor_state, "artifact_type", None) == "prompt",
            label="canary Filled Prompt",
        )
        assert gateway.auxiliary_calls == 1
        assert gateway.observed_placeholders
        await _close_modal(pilot, modal)
    if sink_id != -1:
        logger.remove(sink_id)
    values = dict(category_values)
    for index, token in enumerate(sorted(gateway.observed_placeholders), start=1):
        values[f"placeholder_{index}"] = token
    return {"values": values, "provider_calls": gateway.auxiliary_calls}


async def main() -> None:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="console-prompt-improvement-qa-")).resolve()
    (profile / "data").mkdir(mode=0o700)
    log_path = profile / "logs" / "qa.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    db = PromptsDatabase(profile / "data" / "prompts.sqlite", "qa-client")
    ids = _seed(db)
    service = build_prompt_scope_service(
        prompt_db=db,
        app_config={},
        policy_enforcer=None,
        server_service=ServerPromptService(client=None),
    )
    generic_forbidden: tuple[str, ...] = ()
    observations: dict[str, Any] = {
        "chatbook_head": "12acb277751ebd3985b768ff8a66605da3ae3818",
        "profile_shape": "<temporary-root>/console-prompt-improvement-qa-*/",
        "seeded_rows": len(ids),
        "sizes": [],
    }
    stage = os.environ.get("TLDW_QA_CAPTURE_STAGE", "all")
    try:
        if stage in {"all", "responsive"}:
            for size in SIZES:
                observations["sizes"].append(
                    await _capture_responsive_surfaces(size, service, generic_forbidden)
                )
        if stage in {"all", "improvement"}:
            observations["improvement"] = await _exercise_improvement_states(
                service, generic_forbidden
            )
        if stage in {"all", "guards"}:
            observations["guards_library"] = await _capture_guards_and_library(
                service, ids, generic_forbidden
            )
        canary = (
            await _canary_run(service, log_path)
            if stage in {"all", "canary"}
            else {"values": {}, "provider_calls": 0}
        )
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        counts = {
            category: log_text.count(value)
            for category, value in canary["values"].items()
        }
        assert stage not in {"all", "canary"} or (
            counts and all(count == 0 for count in counts.values())
        )
        observations["canary_audit"] = {
            "counts": counts,
            "provider_calls": canary["provider_calls"],
            "permitted_metadata": [
                "provider",
                "model",
                "mode",
                "duration",
                "input/output sizes",
                "typed outcome",
            ],
        }
        (CAPTURES / "qa-observations.json").write_text(
            json.dumps(observations, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for svg_path in CAPTURES.glob("*.svg"):
            svg = svg_path.read_text(encoding="utf-8")
            assert str(profile) not in svg
            for value in canary["values"].values():
                assert value not in svg
    finally:
        db.close_connection()
        drain_created_dirs()
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(IMPORT_SANDBOX, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
