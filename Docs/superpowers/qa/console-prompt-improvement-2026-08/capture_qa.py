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
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TextArea

from Tests.UI.app_factory import _build_test_app, drain_created_dirs
from Tests.UI.test_library_prompts_canvas import _wire_empty_non_prompt_services
from tldw_chatbook.Chat.attachment_core import PendingAttachment
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
    add(
        "mismatched",
        name="Mismatched structured artifact",
        author="QA",
        details="Guarded artifact-type and definition-kind mismatch",
        system_prompt="Mismatched compatibility system.",
        user_prompt="Mismatched compatibility user.",
        keywords=["mismatched"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=_definition("block_recipe"),
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
        self.resolution_calls = 0
        self.stream_calls = 0
        self.hold_next = False
        self.provider_unavailable = False
        self.drop_protected_tokens = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.observed_placeholders: set[str] = set()
        self.last_source_prompt = ""
        self.response_canary = ""

    async def resolve_for_send(self, selection: Any) -> ConsoleProviderResolution:
        self.resolution_calls += 1
        if self.provider_unavailable:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=selection.base_url or "http://127.0.0.1:9099",
                model=selection.explicit_model
                or selection.configured_model
                or "qa-local-model",
                ready=False,
                visible_copy="The selected QA provider is unavailable. Configure a ready provider and model.",
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
            )
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
        self.last_source_prompt = source
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
            elif self.drop_protected_tokens:
                rewritten = PLACEHOLDER.sub("", source)
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


def _assert_apply_footer_painted(editor: PromptBlockEditor) -> dict[str, Any]:
    """Prove the live modal paints complete lane choices and safe action geometry."""

    footer = editor.query_one("#prompt-editor-footer")
    outer_footer = editor.app.screen.query_one("#console-prompts-footer")
    lane_options = editor.query_one("#prompt-editor-lane-options")
    actions = editor.query_one("#prompt-editor-actions")
    system = editor.query_one("#prompt-editor-apply-system", Checkbox)
    user = editor.query_one("#prompt-editor-apply-user", Checkbox)
    for checkbox, label in (
        (system, "Replace this session's System prompt"),
        (user, "Apply User"),
    ):
        painted = "\n".join(
            checkbox.render_line(row).text for row in range(checkbox.region.height)
        )
        assert "▐X▌" in painted, (checkbox.id, checkbox.region, painted)
        assert label in painted, (checkbox.id, checkbox.region, painted)
        assert checkbox.is_on_screen
        assert editor.region.contains_region(checkbox.region)

    apply_reason = editor.query_one("#prompt-editor-apply-reason", Static)
    apply_explanation = str(apply_reason.renderable)
    assert "active session" in apply_explanation
    assert "System changes only" in apply_explanation
    painted_apply_explanation = "\n".join(
        apply_reason.render_line(row).text for row in range(apply_reason.region.height)
    )
    assert (
        "System changes only on Apply in this active session"
        in " ".join(painted_apply_explanation.split())
    )

    assert footer.has_class("two-row")
    assert lane_options.region.bottom <= actions.region.y
    assert editor.region.bottom <= outer_footer.region.y
    assert system.region.right <= user.region.x
    action_widgets = [
        editor.query_one("#prompt-editor-apply", Button),
        editor.query_one("#prompt-editor-save-menu", Select),
    ]
    assert not editor.query_one("#prompt-editor-back", Button).is_on_screen
    for action in action_widgets:
        assert action.is_on_screen
        assert action.region.width > 0 and action.region.height > 0
        assert editor.region.contains_region(action.region)
        assert action.region.bottom <= outer_footer.region.y
    for left, right in zip(action_widgets, action_widgets[1:]):
        assert left.region.right <= right.region.x
    return {
        "stacked_footer": True,
        "system_checkbox_glyph_visible": True,
        "system_checkbox_full_label_visible": True,
        "user_checkbox_glyph_visible": True,
        "user_checkbox_full_label_visible": True,
        "lane_and_action_rows_do_not_overlap": True,
        "editor_actions_do_not_overlap_outer_footer": True,
        "action_controls_do_not_overlap_or_clip": True,
        "apply_precedes_save_menu": True,
    }


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


async def _return_to_browse(pilot: Any, modal: ConsolePromptsModal) -> None:
    """Use the modal's real Back control and wait for Browse to remount."""

    if modal.state.mode == "browse":
        return
    modal.query_one("#console-prompts-back", Button).press()
    await _wait(
        pilot,
        lambda: (
            modal.state.mode == "browse"
            and bool(modal.query("#console-prompts-search"))
        ),
        label="Prompt Workbench Browse return",
    )


async def _open_named_artifact(
    pilot: Any,
    modal: ConsolePromptsModal,
    *,
    name: str,
    local_id: int,
) -> dict[str, Any]:
    """Search the real Prompt DB, select its normalized row, and open detail."""

    await _return_to_browse(pilot, modal)
    search = modal.query_one("#console-prompts-search", Input)
    search.value = name
    await _wait(
        pilot,
        lambda: (
            len(modal.query(".console-prompts-result")) == 1
            and len(modal.browse_result.items) == 1
            and modal.browse_result.items[0].get("name") == name
        ),
        label=f"normalized row for {name}",
    )
    row = dict(modal.browse_result.items[0])
    assert row["backend"] == "local"
    assert row["local_id"] == local_id
    assert row["id"] == f"local:prompt:{row['source_id']}"
    modal.query_one(".console-prompts-result", Button).press()
    await _wait(
        pilot,
        lambda: (
            modal.state.mode == "edit"
            and modal.state.selected_identity == str(row["source_id"])
        ),
        label=f"latest detail for {name}",
    )
    return row


async def _capture_artifact_compatibility_states(
    service: Any,
    db: PromptsDatabase,
    ids: dict[str, int],
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    """Exercise real normalized legacy and guarded structured artifact rows."""

    gateway = DeterministicGateway()
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {"compatibility": {}}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="artifact compatibility Prompt modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: bool(modal.query(".console-prompts-result")),
            label="artifact compatibility Browse rows",
        )

        await _open_named_artifact(
            pilot,
            modal,
            name="Legacy release note",
            local_id=ids["legacy"],
        )
        editor = modal.query_one(PromptBlockEditor)
        state = editor.state
        assert modal._decoded is not None and modal._decoded.state == "legacy"
        assert state.system_origin is not None
        assert state.user_origin is not None
        assert state.system_origin.text == "Keep claims source-backed."
        assert state.user_origin.text == "Draft a compact release note."
        assert state.compiled_system == state.system_origin.text
        assert state.compiled_user == state.user_origin.text
        assert [block.id for block in state.definition.lanes[0].blocks] == [
            "legacy-system-1"
        ]
        assert [block.id for block in state.definition.lanes[1].blocks] == [
            "legacy-user-1"
        ]
        assert not editor.query_one(
            "#prompt-block-content-legacy-system-1", TextArea
        ).read_only
        assert not editor.query_one(
            "#prompt-block-content-legacy-user-1", TextArea
        ).read_only
        observed["legacy"] = {
            "normalized_row_verified": True,
            "definition_state": "legacy",
            "editable": True,
            "conservative_lane_origins_retained": True,
            "model_calls": gateway.auxiliary_calls,
        }
        _capture(
            app,
            "140x40-legacy-editable-blocks.svg",
            "Prompt Workbench legacy Prompt editable blocks",
            forbidden,
        )

        guarded_cases = (
            ("malformed", "Malformed future artifact", "malformed"),
            ("future", "Future schema artifact", "unsupported"),
            (
                "mismatched",
                "Mismatched structured artifact",
                "mismatched",
            ),
        )
        for key, name, expected_state in guarded_cases:
            before_record = db.fetch_prompt_details(ids[key])
            await _open_named_artifact(
                pilot,
                modal,
                name=name,
                local_id=ids[key],
            )
            assert modal._decoded is not None
            assert modal._decoded.state == expected_state
            assert not modal.query(PromptBlockEditor)
            compatibility = str(
                modal.query_one("#console-prompts-compatibility", Static).renderable
            )
            assert expected_state in compatibility
            system = modal.query_one("#console-prompts-compat-system", TextArea)
            user = modal.query_one("#console-prompts-compat-user", TextArea)
            convert = modal.query_one("#console-prompts-convert", Button)
            assert system.read_only and user.read_only
            assert convert.label == "Convert and save as new"
            assert convert.disabled is False
            assert gateway.auxiliary_calls == 0
            assert db.fetch_prompt_details(ids[key]) == before_record
            case_observation = {
                "normalized_row_verified": True,
                "definition_state": expected_state,
                "read_only": True,
                "convert_enabled": True,
                "model_calls": 0,
                "record_unchanged": True,
            }
            if key == "malformed":
                _capture(
                    app,
                    "140x40-malformed-compatibility.svg",
                    "Prompt Workbench malformed structured compatibility",
                    forbidden,
                )
                convert.press()
                await _wait(
                    pilot,
                    lambda: bool(modal.query(PromptBlockEditor)),
                    label="malformed compatibility conversion working copy",
                )
                converted = modal.query_one(PromptBlockEditor)
                assert modal.state.working_copy_unsaved
                save_menu = converted.query_one("#prompt-editor-save-menu", Select)
                save_options = [
                    value
                    for _label, value in save_menu._options
                    if value is not Select.NULL
                ]
                assert "update" not in save_options
                assert "prompt" in save_options
                assert converted.query_one("#prompt-editor-apply", Button).disabled
                case_observation.update(
                    {
                        "converted_to_unsaved_copy": True,
                        "update_original_disabled": True,
                        "save_as_new_enabled": True,
                        "apply_guarded_until_saved": True,
                    }
                )
            observed["compatibility"][key] = case_observation

        assert gateway.auxiliary_calls == 0
        assert gateway.stream_calls == 0
    return observed


async def _capture_block_edit_and_system_apply(
    service: Any,
    ids: dict[str, int],
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    """Edit/reorder/validate real blocks, then opt into atomic lane Apply."""

    gateway = DeterministicGateway()
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Original unsent user draft.")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        store.set_session_system_prompt(session_id, "Original live System prompt.")
        staged_attachment = PendingAttachment(
            file_path="qa-evidence.txt",
            display_name="QA evidence.txt",
            file_type="text",
            insert_mode="attachment",
            text_content="This staged attachment must remain untouched.",
            original_size=46,
            processed_size=46,
        )
        assert store.add_pending_attachment(session_id, staged_attachment)
        messages_before = tuple(store.messages_for_session(session_id))
        attachments_before = tuple(store.pending_attachments(session_id))

        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="block edit Prompt modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: bool(modal.query(".console-prompts-result")),
            label="block edit Browse rows",
        )
        row = await _open_named_artifact(
            pilot,
            modal,
            name="Structured answer prompt",
            local_id=ids["block_prompt"],
        )
        editor = modal.query_one(PromptBlockEditor)
        assert modal._decoded is not None
        assert modal._decoded.state == "supported_v2"
        assert row["definition_state"] == "supported_v2"

        goal = editor.query_one("#prompt-block-content-goal", TextArea)
        goal.cursor_location = (0, 8)
        goal.focus()
        await pilot.pause()
        goal_identity = id(goal)
        goal_cursor = goal.cursor_location
        await editor._change_field(
            "role",
            "content",
            "Be exact, concise, and cite the available evidence.",
        )
        await pilot.pause()
        same_goal = editor.query_one("#prompt-block-content-goal", TextArea)
        assert same_goal is goal
        assert id(same_goal) == goal_identity
        assert same_goal.cursor_location == goal_cursor
        assert app.focused is same_goal

        editor.query_one("#prompt-block-move-up-output", Button).press()
        await pilot.pause()
        same_goal = editor.query_one("#prompt-block-content-goal", TextArea)
        assert same_goal is goal
        assert same_goal.cursor_location == goal_cursor
        assert app.focused is same_goal
        assert [block.id for block in editor.state.definition.lanes[1].blocks] == [
            "output",
            "goal",
        ]

        await editor._change_field("output", "xml_tag", "bad tag")
        await pilot.pause()
        assert editor.state.issues
        assert "Invalid" in str(
            editor.query_one("#prompt-editor-validation", Static).renderable
        )
        assert editor.query_one("#prompt-editor-apply", Button).disabled
        assert editor.query_one("#prompt-editor-save-menu", Select).disabled
        assert editor.query_one("#prompt-block-content-goal", TextArea) is goal
        assert app.focused is goal
        _capture(
            app,
            "140x40-block-validation.svg",
            "Prompt Workbench block validation and recovery",
            forbidden,
        )

        await editor._change_field("output", "xml_tag", "result")
        await pilot.pause()
        assert not editor.state.issues
        assert "Valid" in str(
            editor.query_one("#prompt-editor-validation", Static).renderable
        )
        assert editor.query_one("#prompt-block-content-goal", TextArea) is goal
        assert goal.cursor_location == goal_cursor
        assert app.focused is goal
        system_checkbox = editor.query_one("#prompt-editor-apply-system", Checkbox)
        user_checkbox = editor.query_one("#prompt-editor-apply-user", Checkbox)
        assert system_checkbox.value is False
        assert user_checkbox.value is True
        system_checkbox.value = True
        await pilot.pause()
        assert system_checkbox.value is True
        assert not editor.query_one("#prompt-editor-apply", Button).disabled
        expected_system = editor.state.compiled_system
        expected_user = editor.state.compiled_user
        editor.query_one("#prompt-lane-system", Collapsible).collapsed = True
        editor.query_one("#prompt-lane-user", Collapsible).collapsed = True
        await pilot.pause()
        system_checkbox.focus()
        system_checkbox.scroll_visible()
        await pilot.pause()
        apply_button = editor.query_one("#prompt-editor-apply", Button)
        assert system_checkbox.is_on_screen, (
            system_checkbox.region,
            system_checkbox.virtual_region,
        )
        assert user_checkbox.is_on_screen, (
            user_checkbox.region,
            user_checkbox.virtual_region,
        )
        assert apply_button.is_on_screen, (
            apply_button.region,
            apply_button.virtual_region,
        )
        footer_observation = _assert_apply_footer_painted(editor)
        _capture(
            app,
            "140x40-system-user-apply-ready.svg",
            "Prompt Workbench optional System and User Apply ready",
            forbidden,
        )

        auxiliary_before = gateway.auxiliary_calls
        stream_before = gateway.stream_calls
        editor.query_one("#prompt-editor-apply", Button).press()
        await _wait(
            pilot,
            lambda: app.screen_stack[-1] is console,
            label="optional System and User Apply completion",
        )
        live_settings = store.session_settings(session_id)
        assert live_settings is not None
        assert live_settings.system_prompt == expected_system
        assert composer.draft_text() == expected_user
        assert store.session_draft(session_id) == expected_user
        assert tuple(store.messages_for_session(session_id)) == messages_before
        assert tuple(store.pending_attachments(session_id)) == attachments_before
        assert gateway.auxiliary_calls == auxiliary_before == 0
        assert gateway.stream_calls == stream_before == 0
        assert store.persistence is None
        observed["block_editor"] = {
            "normalized_row_verified": True,
            "edited": True,
            "reordered": True,
            "validation_introduced": True,
            "validation_resolved": True,
            "sibling_widget_identity_retained": True,
            "cursor_retained": True,
            "focus_retained": True,
        }
        observed["optional_system_apply"] = {
            "system_default_off": True,
            "user_default_on": True,
            "system_opted_in": True,
            "compiled_system_applied": True,
            "compiled_user_applied": True,
            "persistence_outcome": "not_required_success",
            "transcript_unchanged": True,
            "attachments_unchanged": True,
            "normal_send_calls": 0,
            "auxiliary_calls": 0,
            "visible_footer": footer_observation,
        }
        _capture(
            app,
            "140x40-system-user-applied.svg",
            "Console after optional System and User Prompt Apply",
            forbidden,
        )
    return observed


async def _capture_provider_unavailable_improve(
    service: Any,
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve a real unavailable model target and exercise its recovery route."""

    gateway = DeterministicGateway()
    gateway.provider_unavailable = True
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft a useful answer.")
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="provider-unavailable Prompt modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-improve")),
            label="provider-unavailable Browse",
        )
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: (
                modal.state.mode == "improve"
                and bool(modal.query("#console-prompts-auto-improve"))
            ),
            label="provider-unavailable Improve state",
        )
        auto = modal.query_one("#console-prompts-auto-improve", Button)
        review = modal.query_one("#console-prompts-review-improve", Button)
        assert not auto.disabled and not review.disabled
        auto.press()
        await _wait(
            pilot,
            lambda: (
                gateway.resolution_calls == 1
                and modal.query_one(
                    "#console-prompts-auto-improve", Button
                ).disabled
                and modal.query_one(
                    "#console-prompts-review-improve", Button
                ).disabled
                and "unavailable"
                in str(
                    modal.query_one(
                        "#console-prompts-improvement-status", Static
                    ).renderable
                ).lower()
            ),
            label="provider-unavailable resolved recovery state",
        )
        auto = modal.query_one("#console-prompts-auto-improve", Button)
        review = modal.query_one("#console-prompts-review-improve", Button)
        status = str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        )
        assert auto.disabled and review.disabled
        assert "unavailable" in status.lower()
        assert "qa provider" in status.lower()
        assert gateway.resolution_calls == 1
        assert gateway.auxiliary_calls == 0
        _capture(
            app,
            "140x40-provider-unavailable-improve.svg",
            "Prompt Workbench provider unavailable Improve state",
            forbidden,
        )

        modal.query_one("#console-prompts-back", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-configure-provider")),
            label="provider recovery control",
        )
        configure = modal.query_one("#console-prompts-configure-provider", Button)
        assert configure.disabled is False
        browse_status = str(
            modal.query_one("#console-prompts-model-status", Static).renderable
        )
        assert "Model improvement unavailable" in browse_status
        configure.press()
        await _wait(
            pilot,
            lambda: bool(app.screen_stack[-1].query("#console-settings-modal")),
            label="provider recovery settings modal",
        )
        assert gateway.auxiliary_calls == 0
        assert gateway.stream_calls == 0
        observed = {
            "resolution_calls": gateway.resolution_calls,
            "auxiliary_calls": 0,
            "normal_send_calls": 0,
            "improve_actions_disabled": True,
            "actionable_unavailable_copy": True,
            "configure_control_enabled": True,
            "configure_opened_console_settings": True,
            "distinct_from_server_source_unavailable": True,
        }
    return observed


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
        assert set(visible_ids) == {
            "console-composer-collapse",
            "console-composer-menu",
            "console-send-message",
            "console-dictation",
        }, visible_ids
        observed["entry_point"] = "composer_hamburger"
        observed["top_control_prompts_absent"] = True
        observed["idle_composer_controls"] = visible_ids
        composer.insert_text("Draft a useful answer.")

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
        assert normal[:4] == [
            "improve-current-draft",
            "prompts",
            "attach-context",
            "save-chatbook",
        ]
        observed["improve_first_normal_menu_item"] = True
        observed["prompt_library_second_normal_menu_item"] = True
        observed["menu_order"] = action_ids
        _capture(
            app,
            f"{prefix}-composer-menu.svg",
            f"Prompt Workbench composer menu {prefix}",
            forbidden,
        )

        menu.query_one(
            "#console-composer-menu-improve-current-draft", Button
        ).press()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="direct Prompt improvement modal",
        )
        modal = app.screen_stack[-1]
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-review-improve")),
            label="direct Improve choices",
        )
        assert modal.state.mode == "improve"
        assert app.focused is modal.query_one(
            "#console-prompts-review-improve", Button
        )
        await pilot.pause()
        observed["direct_improve_recommended_focus"] = True
        _capture(
            app,
            f"{prefix}-direct-improve.svg",
            f"Prompt Workbench direct Improve {prefix}",
            forbidden,
        )

        analysis_context = modal.query_one(
            "#console-prompts-include-system", Checkbox
        )
        analysis_disclosure = modal.query_one(
            "#console-prompts-analysis-context-disclosure", Static
        )
        assert str(analysis_context.label) == (
            "Let the improver read the current System prompt"
        )
        assert str(analysis_disclosure.renderable) == (
            "Used only to improve the draft. It does not change this session."
        )
        analysis_context.focus()
        analysis_context.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert analysis_context.is_on_screen
        assert analysis_disclosure.is_on_screen
        painted_analysis = "\n".join(
            analysis_context.render_line(row).text
            for row in range(analysis_context.region.height)
        )
        painted_disclosure = "\n".join(
            analysis_disclosure.render_line(row).text
            for row in range(analysis_disclosure.region.height)
        )
        assert "Let the improver read the current System prompt" in painted_analysis
        assert "Used only to improve the draft" in painted_disclosure
        assert "does not change this session" in painted_disclosure
        observed["system_analysis_choice_painted"] = True
        _capture(
            app,
            f"{prefix}-system-analysis-choice.svg",
            f"Prompt Workbench System analysis choice {prefix}",
            forbidden,
        )
        await _close_modal(pilot, modal)

        composer.query_one("#console-composer-menu", Button).press()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsoleComposerMenuModal),
            label="composer hamburger menu after direct Improve",
        )
        menu = app.screen_stack[-1]
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
        assert str(
            modal.query_one(
                "#console-prompts-recipe-outcome-description", Static
            ).renderable
        ).startswith("Outcome-first starts with Goal")
        assert str(
            modal.query_one(
                "#console-prompts-recipe-saved-description", Static
            ).renderable
        ) == "Saved Recipe reuses a format from Library > Prompts."
        assert str(
            modal.query_one(
                "#console-prompts-recipe-blank-description", Static
            ).renderable
        ).startswith("Blank starts with empty System and User lanes")
        for selector in (
            "#console-prompts-recipe-outcome-first",
            "#console-prompts-recipe-saved",
            "#console-prompts-recipe-blank",
        ):
            choice = modal.query_one(selector, Button)
            choice.focus()
            choice.scroll_visible(
                animate=False, force=True, immediate=True, top=True
            )
            await pilot.pause()
            assert choice.is_on_screen
        recommended_choice = modal.query_one(
            "#console-prompts-recipe-outcome-first", Button
        )
        recommended_choice.focus()
        recommended_choice.scroll_visible(
            animate=False, force=True, immediate=True, top=True
        )
        modal.query_one("#console-prompts-back", Button).focus()
        await pilot.pause()
        observed["recipe_chooser_guidance"] = True
        _capture(
            app,
            f"{prefix}-recipe-chooser.svg",
            f"Prompt Workbench Recipe chooser {prefix}",
            forbidden,
        )
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query(PromptBlockEditor)),
            label="shared Recipe block editor",
        )
        editor = modal.query_one(PromptBlockEditor)
        for block_id in ("goal", "context-evidence", "constraints", "output"):
            assert editor.query_one(f"#prompt-block-{block_id}").display
        for block_id in (
            "role",
            "personality",
            "collaboration-style",
            "success-criteria",
            "stop-rules",
        ):
            assert not editor.query_one(f"#prompt-block-{block_id}").display
        reveal_optional = editor.query_one("#prompt-editor-show-optional", Button)
        assert str(reveal_optional.label) == "Show 5 optional blocks"
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
        reveal_optional.focus()
        reveal_optional.scroll_visible(animate=False, immediate=True)
        await pilot.press("enter")
        await pilot.pause()
        assert all(
            editor.query_one(f"#prompt-block-{block_id}").display
            for block_id in (
                "role",
                "personality",
                "collaboration-style",
                "success-criteria",
                "stop-rules",
            )
        )
        observed["recipe_optional_blocks_keyboard_revealed"] = True

        before_fill = composer.capture_draft_snapshot()
        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await _wait(
            pilot,
            lambda: getattr(modal._editor_state, "artifact_type", None) == "prompt",
            label="Filled Prompt mandatory review",
        )
        editor = modal.query_one(PromptBlockEditor)
        await _wait(
            pilot,
            lambda: (
                editor.query_one("#prompt-editor-apply-system", Checkbox).region.width
                > 0
                and editor.query_one("#prompt-editor-apply-user", Checkbox).region.width
                > 0
                and (
                    editor.query_one("#prompt-editor-footer").has_class("two-row")
                    is (editor.size.width < 120)
                )
            ),
            label=f"Filled Prompt Apply footer at {prefix}",
        )
        observed["visible_apply_footer"] = _assert_apply_footer_painted(editor)
        mapped = editor.state.definition.lanes[1].blocks[-1]
        assert mapped.id == "additional-context"
        assert mapped.content == gateway.response_canary
        assert composer.capture_draft_snapshot() == before_fill
        assert not modal.query("#console-prompts-recipe-fill")
        duplicate = editor.query_one(
            "#prompt-block-duplicate-additional-context",
            Button,
        )
        save_menu = editor.query_one("#prompt-editor-save-menu", Select)
        save_options = [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        assert duplicate.disabled is True
        assert "recipe" not in save_options
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
        assert "recipe" in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        observed["recipe_save_recovered_after_mapped_context_delete"] = True

        await _close_modal(pilot, modal)
        composer.clear_draft()
        composer.insert_text("Draft a useful answer.")
        auto_before = composer.capture_draft_snapshot()
        console._open_console_prompts_modal(initial_mode="improve")
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label=f"responsive Auto Improve modal at {prefix}",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await _wait(
            pilot,
            lambda: (
                app.screen_stack[-1] is console
                and composer.query_one(
                    "#console-prompt-improvement-recovery"
                ).display
            ),
            label=f"responsive Draft improved recovery at {prefix}",
        )
        recovery = composer.query_one("#console-prompt-improvement-recovery")
        assert recovery.region.width > 0 and recovery.region.height == 1
        assert not composer.query_one(
            "#console-prompt-improvement-undo", Button
        ).disabled
        assert not composer.query_one(
            "#console-prompt-improvement-review", Button
        ).disabled
        observed["auto_recovery_visible"] = True
        _capture(
            app,
            f"{prefix}-auto-success-recovery.svg",
            f"Prompt Workbench automatic replacement recovery {prefix}",
            forbidden,
        )

        composer.query_one("#console-prompt-improvement-review", Button).press()
        await _wait(
            pilot,
            lambda: bool(
                app.screen_stack[-1].query("#console-prompt-comparison-modal")
            ),
            label=f"responsive before-after comparison at {prefix}",
        )
        comparison = app.screen_stack[-1]
        assert comparison.query_one(
            "#console-prompt-comparison-before", TextArea
        ).read_only
        assert comparison.query_one(
            "#console-prompt-comparison-after", TextArea
        ).read_only
        assert app.focused is comparison.query_one(
            "#console-prompt-comparison-keep", Button
        )
        observed["before_after_review_visible"] = True
        _capture(
            app,
            f"{prefix}-auto-review-changes.svg",
            f"Prompt Workbench automatic replacement review {prefix}",
            forbidden,
        )
        comparison.query_one("#console-prompt-comparison-keep", Button).press()
        await _wait(
            pilot,
            lambda: app.screen_stack[-1] is console,
            label=f"responsive comparison Keep at {prefix}",
        )
        assert composer.improvement_undo_available
        composer.query_one("#console-prompt-improvement-undo", Button).press()
        await _wait(
            pilot,
            lambda: not composer.improvement_undo_available,
            label=f"responsive visible Undo at {prefix}",
        )
        assert _semantic_snapshot(
            composer.capture_draft_snapshot()
        ) == _semantic_snapshot(auto_before)
        observed["visible_undo_exact"] = True
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
        inline_body = "QA protected inline body that must never reach the model"
        inline_label = "qa-private-inline.txt"
        composer.insert_text("Summarize the protected source: ")
        composer.insert_file_segment(inline_body, inline_label)
        composer.insert_text(" Return only a concise conclusion.")
        veto_before = composer.capture_draft_snapshot()
        gateway.drop_protected_tokens = True
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
        assert len(gateway.observed_placeholders) == 1
        protected_token = next(iter(gateway.observed_placeholders))
        assert protected_token in gateway.last_source_prompt
        assert inline_body not in gateway.last_source_prompt
        assert inline_label not in gateway.last_source_prompt
        candidate = modal.query_one("#console-prompts-review-user", TextArea)
        assert protected_token not in candidate.text
        assert inline_body not in candidate.text
        assert inline_label not in candidate.text
        modal.query_one("#console-prompts-review-apply", Button).press()
        await _wait(
            pilot,
            lambda: (
                "Protected prompt material changed"
                in str(
                    modal.query_one(
                        "#console-prompts-improvement-status", Static
                    ).renderable
                )
            ),
            label="protected inline-file Apply veto",
        )
        after_blocked_apply = composer.capture_draft_snapshot()
        assert after_blocked_apply == veto_before
        protected_segments = [
            segment
            for segment in after_blocked_apply.segments
            if segment.origin == "inline_file"
        ]
        assert len(protected_segments) == 1
        assert protected_segments[0].text == inline_body
        assert protected_segments[0].label == inline_label
        observed["protected_inline_file_veto"] = {
            "generic_review_required_copy": True,
            "apply_blocked": True,
            "placeholder_round_trip_guarded": True,
            "protected_segment_retained": True,
            "provider_received_no_inline_body_or_label": True,
            "composer_unchanged": True,
        }
        _capture(
            app,
            "140x40-protected-inline-review-blocked.svg",
            "Prompt Workbench protected inline-file Review veto",
            (*forbidden, inline_body, inline_label, protected_token),
        )
        await _close_modal(pilot, modal)
        gateway.drop_protected_tokens = False

        composer.clear_draft()
        composer.insert_text("Draft a useful answer.")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        store.set_session_system_prompt(session_id, "Captured System prompt.")
        gateway.hold_next = True
        gateway.started.clear()
        gateway.release.clear()
        calls_before_stale = gateway.auxiliary_calls
        console._open_console_prompts_modal()
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="stale-result Prompt modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-auto-improve")),
            label="stale-result Improve mode",
        )
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await gateway.started.wait()
        composer.insert_text(" Live user edit while waiting.")
        store.set_session_system_prompt(session_id, "Live System edit while waiting.")
        stale_live_draft = composer.capture_draft_snapshot()
        messages_before_stale_release = tuple(store.messages_for_session(session_id))
        attachments_before_stale_release = tuple(store.pending_attachments(session_id))
        gateway.release.set()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-review-user")),
            label="stale result Review state",
        )
        stale_status = str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        )
        assert "System prompt changed" in stale_status
        assert composer.capture_draft_snapshot() == stale_live_draft
        live_settings = store.session_settings(session_id)
        assert live_settings is not None
        assert live_settings.system_prompt == "Live System edit while waiting."
        assert (
            tuple(store.messages_for_session(session_id))
            == messages_before_stale_release
        )
        assert (
            tuple(store.pending_attachments(session_id))
            == attachments_before_stale_release
        )
        assert gateway.auxiliary_calls == calls_before_stale + 1
        await pilot.pause()
        assert gateway.auxiliary_calls == calls_before_stale + 1
        assert gateway.stream_calls == 0
        observed["stale_in_flight_result"] = {
            "live_draft_retained": True,
            "live_system_retained": True,
            "review_state_mounted": True,
            "actionable_stale_copy": True,
            "partial_apply": False,
            "extra_provider_calls": 0,
            "transcript_unchanged": True,
            "attachments_unchanged": True,
        }
        _capture(
            app,
            "140x40-stale-result-review.svg",
            "Prompt Workbench stale in-flight result Review",
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


async def _capture_recipe_library_round_trip(
    service: Any,
    db: PromptsDatabase,
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    """Save, deep-link, edit, reopen, fill, review, and apply one real Recipe."""

    gateway = DeterministicGateway()
    app = _build_test_app(configured_default="chat")
    _configure_app(app, service, gateway)
    observed: dict[str, Any] = {}
    async with _run_test(app, size=(140, 40)) as pilot:
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft a launch brief from the available evidence.")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        store.set_session_system_prompt(session_id, "Keep the launch brief factual.")
        system_before = store.session_settings(session_id).system_prompt

        console._open_console_prompts_modal(initial_mode="improve")
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="round-trip Improve modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-recipe-outcome-first")),
            label="round-trip Recipe chooser",
        )
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query(PromptBlockEditor)),
            label="round-trip Outcome-first editor",
        )
        editor = modal.query_one(PromptBlockEditor)
        await editor._change_field(
            "goal",
            "content",
            "Create a decision-ready launch brief.",
        )
        await editor._change_field(
            "context-evidence",
            "content",
            "Use only the evidence supplied in the current request.",
        )
        initial_definition = editor.state.definition
        save_menu = editor.query_one("#prompt-editor-save-menu", Select)
        save_menu.value = "recipe"
        await _wait(
            pilot,
            lambda: modal.query_one(
                "#console-prompts-recipe-save-confirmation-panel"
            ).display,
            label="Recipe save confirmation",
        )
        target = modal._saved_recipe_library_target
        assert target is not None and target[0] == "local"
        saved_id = int(target[1])
        saved = db.fetch_prompt_details(saved_id)
        assert saved is not None
        assert saved["artifact_type"] == "recipe"
        assert saved["version"] == 1
        saved_definition = saved["prompt_definition"]
        if isinstance(saved_definition, str):
            saved_definition = json.loads(saved_definition)
        assert saved_definition["lanes"][1]["blocks"][0]["id"] == "goal"
        observed["saved_recipe_identity"] = saved_id
        observed["save_confirmation_named_library_prompts"] = True
        _capture(
            app,
            "140x40-recipe-saved-confirmation.svg",
            "Recipe saved to Library confirmation",
            forbidden,
        )

        modal.query_one("#console-prompts-open-saved-recipe", Button).press()
        await _wait(
            pilot,
            lambda: (
                app.screen.__class__.__name__ == "LibraryScreen"
                and bool(app.screen.query("#library-prompt-name"))
            ),
            label="newly saved Recipe deep-link in Library",
        )
        library = app.screen
        assert library._selected_prompt_id == saved_id
        assert "Recipe · Local" in str(
            library.query_one("#library-prompt-artifact-status", Static).renderable
        )
        library_editor = library.query_one(
            "#library-prompt-block-editor", PromptBlockEditor
        )
        assert library_editor.state.definition == initial_definition
        await _wait(
            pilot,
            lambda: not library.query_one("#library-prompt-save", Button).disabled,
            label="saved Recipe Library update capability",
        )
        await library_editor._change_field(
            "goal",
            "content",
            "Create an edited, decision-ready launch brief.",
        )
        library.query_one(
            "#library-prompt-recipe-starter", Checkbox
        ).value = True
        library.query_one("#library-prompt-name", Input).value = (
            "QA round-trip reusable prompt"
        )
        await pilot.pause()
        edited_goal = "Create an edited, decision-ready launch brief."
        assert (
            library_editor.state.definition.lanes[1].blocks[0].content
            == edited_goal
        ), library_editor.state.definition.lanes[1].blocks[0].content
        assert library._library_prompt_block_state is not None
        assert (
            library._library_prompt_block_state.definition.lanes[1]
            .blocks[0]
            .content
            == edited_goal
        ), library._library_prompt_block_state.definition.lanes[1].blocks[0].content
        library.query_one("#library-prompt-save", Button).press()
        await _wait(
            pilot,
            lambda: bool(
                str(
                    library.query_one(
                        "#library-prompt-save-status", Static
                    ).renderable
                )
            ),
            label="edited Recipe save in Library",
        )
        library_save_status = str(
            library.query_one("#library-prompt-save-status", Static).renderable
        )
        assert library_save_status == "Saved.", library_save_status
        assert library._library_prompt_version == 2
        persisted = db.fetch_prompt_details(saved_id)
        assert persisted is not None
        assert persisted["artifact_type"] == "recipe"
        assert persisted["version"] == 2
        persisted_definition = persisted["prompt_definition"]
        if isinstance(persisted_definition, str):
            persisted_definition = json.loads(persisted_definition)
        assert persisted_definition["kind"] == "block_recipe"
        persisted_goal = persisted_definition["lanes"][1]["blocks"][0]["content"]
        assert persisted_goal == (
            "Create an edited, decision-ready launch brief."
        ), persisted_goal
        observed["library_lossless_reopen"] = True
        observed["library_edit_version"] = 2
        _capture(
            app,
            "140x40-library-saved-recipe.svg",
            "Saved Recipe reopened and edited in Library",
            forbidden,
        )

        await app.handle_screen_navigation(NavigateToScreen("chat"))
        console = await _console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        console._open_console_prompts_modal(initial_mode="improve")
        await _wait(
            pilot,
            lambda: isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="round-trip reopen Improve modal",
        )
        modal = app.screen_stack[-1]
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-recipe-saved")),
            label="round-trip saved Recipe choice",
        )
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await _wait(
            pilot,
            lambda: bool(modal.query("#console-prompts-search")),
            label="round-trip saved Recipe browser",
        )
        modal.query_one("#console-prompts-search", Input).value = (
            "QA round-trip reusable prompt"
        )
        await _wait(
            pilot,
            lambda: len(modal.query(".console-prompts-result")) == 1,
            label="round-trip saved Recipe search result",
        )
        result = modal.query_one(".console-prompts-result", Button)
        assert "Recipe" in str(result.label)
        result.press()
        await _wait(
            pilot,
            lambda: bool(modal.query(PromptBlockEditor)),
            label="round-trip saved Recipe Console editor",
        )
        editor = modal.query_one(PromptBlockEditor)
        assert modal._recipe_source == "local"
        assert modal._recipe_source_id
        assert not modal._recipe_source_id.startswith("builtin:")
        assert modal._recipe_version == 2
        assert editor.state.definition.lanes[1].blocks[0].content == (
            "Create an edited, decision-ready launch brief."
        )
        assert tuple(block.id for block in editor.state.definition.lanes[1].blocks) == (
            "goal",
            "context-evidence",
            "constraints",
            "output",
            "success-criteria",
            "stop-rules",
        )
        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await _wait(
            pilot,
            lambda: getattr(modal._editor_state, "artifact_type", None) == "prompt",
            label="round-trip Filled Prompt review",
        )
        review_editor = modal.query_one(PromptBlockEditor)
        assert review_editor.query_one(
            "#prompt-editor-apply-system", Checkbox
        ).value is False
        assert review_editor.query_one(
            "#prompt-editor-apply-user", Checkbox
        ).value is True
        expected_user = review_editor.state.compiled_user
        review_editor.query_one("#prompt-editor-apply", Button).press()
        await _wait(
            pilot,
            lambda: not isinstance(app.screen_stack[-1], ConsolePromptsModal),
            label="round-trip Prompt Apply dismissal",
        )
        assert composer.draft_text() == expected_user
        settings_after = store.session_settings(session_id)
        assert settings_after is not None
        assert settings_after.system_prompt == system_before
        assert gateway.stream_calls == 0
        observed["filled_review_required"] = True
        observed["applied_user_lane"] = True
        observed["system_lane_unchanged"] = True
        observed["normal_send_calls"] = 0
        _capture(
            app,
            "140x40-recipe-roundtrip-applied.svg",
            "Recipe round trip applied to Console",
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
        "chatbook_head": "b856795415cb8f8f6abf9eafeb2f73a7a6bae908",
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
        if stage in {"all", "artifacts"}:
            observations[
                "artifact_live"
            ] = await _capture_artifact_compatibility_states(
                service,
                db,
                ids,
                generic_forbidden,
            )
        if stage in {"all", "blocks"}:
            observations[
                "block_apply_live"
            ] = await _capture_block_edit_and_system_apply(
                service,
                ids,
                generic_forbidden,
            )
        if stage in {"all", "provider"}:
            observations[
                "provider_unavailable_live"
            ] = await _capture_provider_unavailable_improve(
                service,
                generic_forbidden,
            )
        if stage in {"all", "improvement"}:
            observations["improvement"] = await _exercise_improvement_states(
                service, generic_forbidden
            )
        if stage in {"all", "guards"}:
            observations["guards_library"] = await _capture_guards_and_library(
                service, ids, generic_forbidden
            )
        if stage in {"all", "roundtrip"}:
            observations[
                "recipe_library_round_trip"
            ] = await _capture_recipe_library_round_trip(
                service,
                db,
                generic_forbidden,
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
