"""Unified Console Prompt Library Browse/Edit modal contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    outcome_first_recipe,
)
from tldw_chatbook.Prompt_Management.prompt_improvement_models import (
    PromptImprovementOutcome,
    PromptImprovementRequestSnapshot,
    fingerprint_block_definition,
)
from tldw_chatbook.Prompt_Management.prompt_improvement_service import (
    PromptImprovementService,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ComposerTransactionValidationError,
    ConsoleComposerBar,
)

from tldw_chatbook.Widgets.Console.console_prompts_browse import ConsolePromptsBrowse
from tldw_chatbook.Widgets.Console.console_prompt_improve_view import (
    ConsolePromptImprovementContext,
)
from tldw_chatbook.Widgets.Console.console_prompts_modal import (
    ConsolePromptsModal,
    ConsoleRecipeApplyGuard,
)
from tldw_chatbook.Widgets.Console.console_prompts_state import (
    ConsolePromptsState,
    PromptBrowseResult,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    ADDITIONAL_CONTEXT_RESERVED_PREFIX,
)


_BUNDLED_STYLESHEET = (
    Path(__file__).parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
)


def _definition(kind: str = "block_prompt") -> dict[str, Any]:
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
                        "content": "Be exact.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "freeform",
                        "content": "Answer the question.",
                    }
                ],
            },
        ],
    }


def _detail(
    *,
    artifact_type: str = "prompt",
    prompt_format: str = "structured",
    schema_version: int = 2,
    definition: Any | None = None,
    identifier: str = "prompt-1",
    version: int = 4,
) -> dict[str, Any]:
    if definition is None:
        definition = _definition(
            "block_recipe" if artifact_type == "recipe" else "block_prompt"
        )
    return {
        "id": identifier,
        "name": "Precise answer",
        "artifact_type": artifact_type,
        "prompt_format": prompt_format,
        "prompt_schema_version": schema_version,
        "prompt_definition": definition,
        "system_prompt": "# Role\n\nBe exact.",
        "user_prompt": "Answer the question.",
        "version": version,
        "backend": "local",
    }


def _brief(identifier: str = "prompt-1", *, artifact_type: str = "prompt"):
    return {
        "id": identifier,
        "name": "Precise answer",
        "artifact_type": artifact_type,
        "has_system_prompt": True,
        "has_user_prompt": True,
        "updated_at": "2026-08-01T12:00:00Z",
        "backend": "local",
        "version": 4,
    }


def _choose_save_action(root: Any, action: str) -> None:
    """Choose one currently offered action from the editor's Save menu."""
    menu = root.query_one("#prompt-editor-save-menu", Select)
    offered = [
        value for _label, value in menu._options if value is not Select.NULL
    ]
    assert action in offered
    menu.value = action


class _PromptBackend:
    def __init__(self, *, pages: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        self.pages = dict(pages or {})
        self.list_calls: list[tuple[str, int]] = []
        self.search_calls: list[tuple[str, str]] = []
        self.detail_calls: list[tuple[str, str]] = []
        self.save_calls: list[dict[str, Any]] = []
        self.model_calls = 0
        self.usage_mutations = 0
        self.search_result: Any = []
        self.detail_result: Any = _detail()
        self.save_result: Any = None
        self.list_error: Exception | None = None
        self.search_error: Exception | None = None
        self.detail_error: Exception | None = None
        self.capabilities_result: object = SimpleNamespace(
            structured_kinds=frozenset({(2, "block_prompt"), (2, "block_recipe")}),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def capabilities(self, source: str) -> object:
        return self.capabilities_result

    async def list_page(self, source: str, page: int) -> Mapping[str, Any]:
        self.list_calls.append((source, page))
        if self.list_error is not None:
            raise self.list_error
        return self.pages.get(
            page,
            {"items": [], "page": page, "total_pages": 1, "total_items": 0},
        )

    async def search(self, source: str, query: str) -> Any:
        self.search_calls.append((source, query))
        if self.search_error is not None:
            raise self.search_error
        return self.search_result

    async def detail(self, source: str, identifier: str) -> Any:
        self.detail_calls.append((source, identifier))
        if self.detail_error is not None:
            raise self.detail_error
        return self.detail_result

    async def save(self, **payload: Any) -> Any:
        self.save_calls.append(payload)
        return payload if self.save_result is None else self.save_result


class _RealPromptScopeBackend:
    """Console callback adapter over the real local Prompt scope/DB stack."""

    def __init__(self, service: PromptScopeService) -> None:
        self.service = service
        self.detail_calls: list[tuple[str, str]] = []
        self.save_calls: list[dict[str, Any]] = []

    async def capabilities(self, source: str) -> object:
        return await self.service.get_capabilities(mode=source)

    async def list_page(self, source: str, page: int) -> Mapping[str, Any]:
        return await self.service.list_prompts(mode=source, page=page, per_page=10)

    async def search(self, source: str, query: str) -> Any:
        return await self.service.search_prompts(mode=source, query=query, limit=25)

    async def detail(self, source: str, identifier: str) -> Any:
        self.detail_calls.append((source, identifier))
        return await self.service.get_prompt(
            mode=source,
            prompt_identifier=identifier,
        )

    async def save(self, **payload: Any) -> Any:
        self.save_calls.append(dict(payload))
        source = str(payload.pop("source", "local"))
        return await self.service.save_prompt(mode=source, **payload)


class _Harness(App):
    def __init__(
        self,
        backend: _PromptBackend,
        *,
        improve_unavailable_reason: str = "",
        configure_provider: Callable[[], Any] | None = None,
        improvement_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.results: list[object] = []
        self.improve_unavailable_reason = improve_unavailable_reason
        self.configure_provider = configure_provider
        self.improvement_kwargs = dict(improvement_kwargs or {})

    def compose(self) -> ComposeResult:
        yield Input(id="console-native-composer")

    async def on_mount(self) -> None:
        self.query_one("#console-native-composer", Input).focus()
        kwargs: dict[str, Any] = {}
        if self.configure_provider is not None:
            kwargs["configure_provider"] = self.configure_provider
        await self.push_screen(
            ConsolePromptsModal(
                capabilities=self.backend.capabilities,
                list_page=self.backend.list_page,
                search=self.backend.search,
                detail=self.backend.detail,
                save=self.backend.save,
                improve_unavailable_reason=self.improve_unavailable_reason,
                **self.improvement_kwargs,
                **kwargs,
            ),
            callback=self.results.append,
        )


class _StyledHarness(_Harness):
    """Modal harness with the same bundled stylesheet as the real Console."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)


class _ApplyOverlay(ModalScreen[None]):
    """Top-screen sentinel used to prove stale apply completion is harmless."""


class _ImprovementDriver:
    """Small captured-state driver for the Task 12 modal contract."""

    def __init__(self, *outcomes: PromptImprovementOutcome) -> None:
        composer = ConsoleComposerBar()
        composer.insert_text("Draft question ")
        composer.insert_file_segment("PRIVATE BODY", "/private/notes.txt · 12 B")
        composer.insert_text(" tail")
        self.composer = composer
        self.snapshot = composer.capture_draft_snapshot()
        self.preview = composer.project_snapshot_for_model(
            self.snapshot,
            request_nonce="preview-request",
        )
        self.outcomes = list(outcomes)
        self.requests: list[Any] = []
        self.applies: list[tuple[Any, Any]] = []
        self.validate_calls: list[tuple[Any, str]] = []

    @property
    def context(self) -> SimpleNamespace:
        return SimpleNamespace(
            session_id="session-1",
            composer_snapshot=self.snapshot,
            current_user_projection=self.preview,
            current_system_prompt="Be accurate.",
            current_system_fingerprint="system-fingerprint",
            provider_label="OpenAI",
            model_label="gpt-test",
        )

    async def build_snapshot(self, **values: Any) -> Any:
        request_id = str(values["request_id"])
        projection = self.composer.project_snapshot_for_model(
            self.snapshot,
            request_nonce=request_id,
        )
        snapshot = SimpleNamespace(
            request_id=request_id,
            mode=values["mode"],
            composer_snapshot=self.snapshot,
            projection=projection,
            system_prompt=(
                "Be accurate." if values.get("include_system", False) else None
            ),
            system_fingerprint=(
                "system-fingerprint" if values.get("include_system", False) else None
            ),
            recipe_source=values.get("recipe_source"),
            recipe_source_id=values.get("recipe_source_id"),
            recipe_version=values.get("recipe_version"),
            recipe_definition=values.get("recipe_definition"),
            recipe_fingerprint=values.get("recipe_fingerprint"),
        )
        self.requests.append(snapshot)
        return snapshot

    async def improve(self, snapshot: Any) -> PromptImprovementOutcome:
        if not self.outcomes:
            raise AssertionError("Unexpected improvement call")
        outcome = self.outcomes.pop(0)
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind=outcome.kind,
            rewritten_prompt=outcome.rewritten_prompt,
            filled_definition=outcome.filled_definition,
            provider=outcome.provider,
            model=outcome.model,
            user_message=outcome.user_message,
        )

    def validate_candidate(self, snapshot: Any, text: str) -> None:
        self.validate_calls.append((snapshot, text))
        self.composer.validate_improvement(snapshot.composer_snapshot, text)

    async def apply(self, result: Any, snapshot: Any) -> Any:
        self.applies.append((result, snapshot))
        return SimpleNamespace(kind="applied", user_message="")

    async def retry_persistence(self, _result: Any) -> Any:
        return SimpleNamespace(kind="applied", user_message="")

    def kwargs(self) -> dict[str, Any]:
        return {
            "improvement_context": self.context,
            "build_improvement_snapshot": self.build_snapshot,
            "improve": self.improve,
            "validate_improvement": self.validate_candidate,
            "apply_improvement_result": self.apply,
            "retry_improvement_persistence": self.retry_persistence,
        }


class _OneShotAuxiliaryGateway:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[AuxiliaryCompletionRequest] = []

    async def complete_auxiliary(
        self, request: AuxiliaryCompletionRequest
    ) -> AuxiliaryCompletionResult:
        self.requests.append(request)
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=str(request.resolution.model),
            text=self.response,
        )


async def _real_additional_context_recipe_outcome(
    driver: _ImprovementDriver,
) -> PromptImprovementOutcome:
    recipe = outcome_first_recipe()
    request_id = "real-recipe-fill"
    projection = driver.composer.project_snapshot_for_model(
        driver.snapshot,
        request_nonce=request_id,
    )
    recipe_fingerprint = fingerprint_block_definition(recipe)
    response = json.dumps(
        {
            "kind": "recipe_fill",
            "recipe_fingerprint": recipe_fingerprint,
            "fills": [
                {
                    "block_id": block.id,
                    "content": (
                        "Deliver a checked answer." if block.id == "goal" else ""
                    ),
                }
                for lane in recipe.lanes
                for block in lane.blocks
            ],
            "additional_context": (
                f"Unmatched evidence: {projection.placeholder_ids[0]}"
            ),
        }
    )
    gateway = _OneShotAuxiliaryGateway(response)
    resolution = ConsoleProviderResolution(
        provider="OpenAI",
        base_url="https://api.example.test/v1",
        model="gpt-test",
        ready=True,
        readiness_key="openai",
        execution_key="openai",
        max_tokens=777,
        streaming=True,
    )
    snapshot = PromptImprovementRequestSnapshot(
        request_id=request_id,
        mode="recipe",
        session_id="session-1",
        composer_snapshot=driver.snapshot,
        projection=projection,
        system_prompt=None,
        system_fingerprint=None,
        resolution=resolution,
        provider_label="OpenAI",
        model_label="gpt-test",
        recipe_source=None,
        recipe_source_id="builtin:outcome-first",
        recipe_version=0,
        recipe_definition=recipe,
        recipe_fingerprint=recipe_fingerprint,
    )

    outcome = await PromptImprovementService(gateway=gateway).improve(snapshot)

    assert gateway.requests
    assert outcome.kind == "success"
    assert outcome.filled_definition is not None
    return outcome


@pytest.mark.unit
def test_state_owns_navigation_search_identity_and_stale_tokens() -> None:
    state = ConsolePromptsState.initial()
    state = (
        state.with_query("alpha")
        .with_page(3)
        .remember_focus("browse", "console-prompts-search")
    )
    state = state.select(identity="prompt-7", version=12).enter_mode("improve")
    stale_token = state.search_token
    state = state.begin_search().with_source("server")

    assert state.mode == "improve"
    assert state.query == "alpha"
    assert state.page == 1
    assert state.selected_identity == "prompt-7"
    assert state.selected_version == 12
    assert state.focus_for("browse") == "console-prompts-search"
    assert not state.accepts(stale_token, "local")
    assert state.accepts(state.search_token, "server")
    assert state.go_back().mode == "browse"


@pytest.mark.unit
def test_browse_result_is_source_scoped_and_immutable() -> None:
    result = PromptBrowseResult(
        source="local",
        items=(_brief(),),
        page=2,
        total_pages=4,
        total_items=31,
    )

    assert result.source == "local"
    assert result.items[0]["artifact_type"] == "prompt"
    assert result.page == 2
    assert result.total_pages == 4


@pytest.mark.asyncio
async def test_empty_local_library_and_improve_action_are_visible() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app.screen.query_one("#console-prompts-improve", Button)
        empty = app.screen.query_one("#console-prompts-browse-status", Static)
        assert "Local Prompt Library is empty" in str(empty.renderable)
        assert "Create or save a Prompt" in str(empty.renderable)

    assert backend.list_calls == [("local", 1)]


@pytest.mark.asyncio
async def test_improve_and_back_preserve_browse_state_and_focus() -> None:
    backend = _PromptBackend(
        pages={
            2: {
                "items": [_brief()],
                "page": 2,
                "total_pages": 2,
                "total_items": 11,
            }
        }
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.state = (
            modal.state.with_query("kept query")
            .with_page(2)
            .select(identity="prompt-1", version=4)
        )
        modal.query_one("#console-prompts-search", Input).focus()
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        assert modal.state.mode == "improve"

        modal.query_one("#console-prompts-back", Button).press()
        await pilot.pause()

        assert modal.state.mode == "browse"
        assert modal.state.query == "kept query"
        assert modal.state.page == 2
        assert modal.state.selected_identity == "prompt-1"
        assert getattr(app.focused, "id", None) == "console-prompts-search"


@pytest.mark.asyncio
async def test_empty_query_paginates_and_nonempty_query_uses_backend_search() -> None:
    backend = _PromptBackend(
        pages={
            1: {
                "items": [_brief()],
                "page": 1,
                "total_pages": 2,
                "total_items": 12,
            },
            2: {
                "items": [_brief("prompt-2")],
                "page": 2,
                "total_pages": 2,
                "total_items": 12,
            },
        }
    )
    backend.search_result = [_brief("searched")]
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-prompts-next", Button).press()
        await pilot.pause()
        assert backend.list_calls[-1] == ("local", 2)

        search = app.screen.query_one("#console-prompts-search", Input)
        search.focus()
        await pilot.press("a", "l", "p", "h", "a")
        await asyncio.sleep(0.23)
        await pilot.pause()

    assert backend.search_calls == [("local", "alpha")]
    assert backend.list_calls == [("local", 1), ("local", 2)]


@pytest.mark.asyncio
async def test_source_switch_reloads_without_merging_results() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.switch_source("server")
        await pilot.pause()

        assert app.screen.state.source == "server"
        assert backend.list_calls == [("local", 1), ("server", 1)]
        assert app.screen.browse_result.source == "server"


@pytest.mark.asyncio
async def test_late_source_completion_is_rejected() -> None:
    local_started = asyncio.Event()
    release_local = asyncio.Event()

    async def list_page(source: str, page: int) -> Mapping[str, Any]:
        if source == "local":
            local_started.set()
            await release_local.wait()
            return {
                "items": [_brief("late-local")],
                "page": page,
                "total_pages": 1,
                "total_items": 1,
            }
        return {
            "items": [{**_brief("server"), "backend": "server"}],
            "page": page,
            "total_pages": 1,
            "total_items": 1,
        }

    backend = _PromptBackend()
    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        await pilot.pause()
        modal._list_page = list_page
        late_local = asyncio.create_task(modal.reload_browse())
        await wait_for_background_signal(
            local_started, late_local, what="the late local browse reload"
        )
        await modal.switch_source("server")
        release_local.set()
        await late_local
        await pilot.pause()
        await pilot.pause()

        assert modal.browse_result.source == "server"
        assert modal.browse_result.items[0]["id"] == "server"


@pytest.mark.asyncio
async def test_no_matches_retry_and_source_unavailable_are_explicit() -> None:
    backend = _PromptBackend()
    backend.search_result = []
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.set_query("missing")
        await pilot.pause()
        status = app.screen.query_one("#console-prompts-browse-status", Static)
        assert "No matches" in str(status.renderable)
        assert "Change the query or switch source" in str(status.renderable)

        backend.search_error = RuntimeError("offline")
        await app.screen.reload_browse()
        await pilot.pause()
        assert app.screen.query_one("#console-prompts-retry", Button).display
        assert "Search failed" in str(status.renderable)

        backend.search_error = None
        app.screen.query_one("#console-prompts-retry", Button).press()
        await pilot.pause()
        assert backend.search_calls[-1] == ("local", "missing")

        backend.list_error = ValueError("Server prompt backend is unavailable.")
        await app.screen.set_query("")
        await pilot.pause()
        assert "source is unavailable" in str(status.renderable)
        assert "Retry or switch source" in str(status.renderable)


@pytest.mark.asyncio
async def test_selected_row_deleted_before_detail_fetch_stays_in_browse() -> None:
    backend = _PromptBackend(
        pages={1: {"items": [_brief()], "page": 1, "total_pages": 1, "total_items": 1}}
    )
    backend.detail_error = KeyError("deleted")
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-prompts-result-prompt-1", Button).press()
        await pilot.pause()

        assert app.screen.state.mode == "browse"
        assert "changed or deleted" in str(
            app.screen.query_one("#console-prompts-browse-status", Static).renderable
        )
        assert backend.detail_calls == [("local", "prompt-1")]


@pytest.mark.asyncio
async def test_normalized_browse_detail_and_update_use_source_id_not_composite_id() -> (
    None
):
    source_id = "9f4e2f0a-1111-4222-8333-444455556666"
    composite_id = f"local:prompt:{source_id}"
    backend = _PromptBackend(
        pages={
            1: {
                "items": [
                    {
                        **_brief(composite_id),
                        "source_id": source_id,
                    }
                ],
                "page": 1,
                "total_pages": 1,
                "total_items": 1,
            }
        }
    )
    backend.detail_result = {
        **_detail(identifier=composite_id),
        "source_id": source_id,
    }
    backend.save_result = {
        **backend.detail_result,
        "version": 5,
    }
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        result_button = app.screen.query_one(".console-prompts-result", Button)
        assert result_button.id is not None
        assert composite_id.encode("utf-8").hex() in result_button.id

        result_button.press()
        await pilot.pause()

        assert backend.detail_calls == [("local", source_id)]
        assert app.screen.state.selected_identity == source_id

        _choose_save_action(app.screen, "update")
        await pilot.pause()
        await pilot.pause()

    assert backend.save_calls[-1]["prompt_identifier"] == source_id


@pytest.mark.asyncio
async def test_real_normalized_local_prompt_round_trip_opens_updates_and_uses_source_id(
    tmp_path,
) -> None:
    prompt_db = PromptsDatabase(
        tmp_path / "console-prompts.db", client_id="console-test"
    )
    try:
        _prompt_id, prompt_uuid, _message = prompt_db.add_prompt(
            name="Normalized Console Prompt",
            author="Console test",
            details="Real PromptScopeService and PromptsDatabase round trip",
            system_prompt="# Role\n\nBe exact.",
            user_prompt="Answer the question.",
            keywords=["console", "identity"],
            overwrite=False,
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=_definition(),
            artifact_type="prompt",
        )
        assert prompt_uuid
        scope = PromptScopeService(
            local_service=LocalPromptService(prompt_db),
            server_service=None,
        )
        backend = _RealPromptScopeBackend(scope)
        guarded_identities: list[tuple[str, str]] = []

        async def apply_result(_result: Any, captured: Any) -> Any:
            latest = await scope.get_prompt(
                mode=captured.source,
                prompt_identifier=captured.prompt_source_id,
            )
            guarded_identities.append(
                (captured.prompt_source_id, str(latest["source_id"]))
            )
            used = await scope.record_prompt_usage(
                mode=captured.source,
                prompt_identifier=captured.prompt_source_id,
            )
            guarded_identities.append(
                (captured.prompt_source_id, str(used["source_id"]))
            )
            return SimpleNamespace(kind="applied", user_message="")

        app = _Harness(
            backend,  # type: ignore[arg-type]
            improvement_kwargs={
                "improvement_context": SimpleNamespace(
                    composer_snapshot=SimpleNamespace(),
                    current_system_fingerprint="system-fingerprint",
                ),
                "apply_improvement_result": apply_result,
            },
        )

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            row = app.screen.browse_result.items[0]
            assert row["id"] == f"local:prompt:{prompt_uuid}"
            assert row["source_id"] == prompt_uuid

            app.screen.query_one(".console-prompts-result", Button).press()
            await pilot.pause()
            await pilot.pause()

            assert app.screen.state.mode == "edit"
            assert app.screen.state.selected_identity == prompt_uuid
            assert backend.detail_calls == [("local", prompt_uuid)]

            _choose_save_action(app.screen, "update")
            await pilot.pause()
            await pilot.pause()

            assert backend.save_calls[-1]["prompt_identifier"] == prompt_uuid
            assert app.screen.state.selected_identity == prompt_uuid

            app.screen.query_one("#prompt-editor-apply", Button).press()
            await pilot.pause()
            await pilot.pause()

        assert guarded_identities == [
            (prompt_uuid, prompt_uuid),
            (prompt_uuid, prompt_uuid),
        ]
    finally:
        prompt_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detail", "expected_mode", "unsaved"),
    [
        (_detail(), "edit", False),
        (_detail(artifact_type="recipe"), "edit", True),
        (
            _detail(
                prompt_format="legacy",
                schema_version=0,
                definition=None,
            ),
            "edit",
            False,
        ),
    ],
)
async def test_supported_prompt_recipe_copy_and_legacy_open_without_side_effects(
    detail: Mapping[str, Any], expected_mode: str, unsaved: bool
) -> None:
    backend = _PromptBackend(
        pages={1: {"items": [_brief()], "page": 1, "total_pages": 1, "total_items": 1}}
    )
    backend.detail_result = detail
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        assert app.screen.state.mode == expected_mode
        assert app.screen.state.working_copy_unsaved is unsaved
        editor = app.screen.query_one(PromptBlockEditor)
        assert editor.state.artifact_type == "prompt"
        assert app.screen.state.selected_identity == "prompt-1"
        assert app.screen.state.selected_version == 4

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


def _rendered_lines(widget: Checkbox) -> tuple[str, ...]:
    """Return the terminal-cell text actually painted by a mounted widget."""
    return tuple(widget.render_line(row).text for row in range(widget.region.height))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 40), (100, 30), (80, 24)])
async def test_filled_prompt_footer_paints_apply_checkboxes_at_supported_sizes(
    size: tuple[int, int],
) -> None:
    backend = _PromptBackend()
    app = _StyledHarness(backend)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        editor = app.screen.query_one(PromptBlockEditor)
        outer_footer = app.screen.query_one("#console-prompts-footer")
        footer = editor.query_one("#prompt-editor-footer")
        lane_options = editor.query_one("#prompt-editor-lane-options")
        actions = editor.query_one("#prompt-editor-actions")
        system = editor.query_one("#prompt-editor-apply-system", Checkbox)
        user = editor.query_one("#prompt-editor-apply-user", Checkbox)

        assert system.value is False
        assert user.value is True

        for checkbox, label in (
            (system, "Replace this session's System prompt"),
            (user, "Apply User"),
        ):
            rendered = _rendered_lines(checkbox)
            painted = "\n".join(rendered)
            assert "▐X▌" in painted, (
                f"{checkbox.id} has no rendered checkbox glyph at {size}: "
                f"region={checkbox.region!r}, lines={rendered!r}"
            )
            assert label in painted, (
                f"{checkbox.id} has no readable label at {size}: "
                f"region={checkbox.region!r}, lines={rendered!r}"
            )

        assert footer.has_class("two-row")
        apply_reason = editor.query_one("#prompt-editor-apply-reason", Static)
        apply_explanation = str(apply_reason.renderable)
        assert "active session" in apply_explanation
        assert "System changes only" in apply_explanation
        painted_apply_explanation = "\n".join(
            apply_reason.render_line(row).text
            for row in range(apply_reason.region.height)
        )
        assert (
            "System changes only on Apply in this active session"
            in " ".join(painted_apply_explanation.split())
        )
        assert lane_options.region.bottom <= actions.region.y
        assert editor.region.bottom <= outer_footer.region.y

        action_widgets = [
            editor.query_one(selector)
            for selector in (
                "#prompt-editor-apply",
                "#prompt-editor-save-menu",
            )
        ]
        editor_back = editor.query_one("#prompt-editor-back", Button)
        assert not editor_back.is_on_screen, (
            editor_back.styles.display,
            editor_back.region,
            editor_back.display,
        )
        for action in action_widgets:
            assert action.region.width > 0 and action.region.height > 0
            assert editor.region.contains_region(action.region)
            assert action.region.bottom <= outer_footer.region.y
        for left, right in zip(action_widgets, action_widgets[1:]):
            assert left.region.right <= right.region.x

        system.focus()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert system.value is True
        await pilot.press("space")
        await pilot.press("tab")
        await pilot.pause()
        assert system.value is False
        assert app.focused is user
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is action_widgets[0]
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is action_widgets[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail",
    [
        _detail(schema_version=1, definition={"schema_version": 1, "blocks": []}),
        _detail(
            definition={"schema_version": 2, "definition_kind": "single_text_recipe"}
        ),
        _detail(schema_version=99, definition={"schema_version": 99, "kind": "future"}),
        _detail(definition="{not-json"),
        _detail(artifact_type="prompt", definition=_definition("block_recipe")),
        _detail(artifact_type="alien"),
    ],
)
async def test_foreign_future_malformed_and_mismatched_artifacts_are_guarded(
    detail: Mapping[str, Any],
) -> None:
    backend = _PromptBackend()
    backend.detail_result = detail
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        assert app.screen.state.mode == "edit"
        assert not app.screen.query(PromptBlockEditor)
        assert app.screen.query_one("#console-prompts-compatibility", Static)
        convert = app.screen.query_one("#console-prompts-convert", Button)
        assert convert.label == "Convert and save as new"
        assert convert.disabled is False
        assert app.screen.query_one(
            "#console-prompts-compat-system", TextArea
        ).read_only
        assert app.screen.query_one("#console-prompts-compat-user", TextArea).read_only

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


@pytest.mark.asyncio
async def test_guarded_artifact_without_compatibility_text_disables_conversion() -> (
    None
):
    backend = _PromptBackend()
    backend.detail_result = {
        **_detail(schema_version=99, definition={"schema_version": 99}),
        "system_prompt": "",
        "user_prompt": "",
    }
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        button = app.screen.query_one("#console-prompts-convert", Button)
        assert button.disabled is True
        assert "no compatible System or User text" in str(button.tooltip)


@pytest.mark.asyncio
async def test_provider_unavailability_disables_only_improve() -> None:
    backend = _PromptBackend()
    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        improve = app.screen.query_one("#console-prompts-improve", Button)
        assert improve.disabled is True
        assert "No active provider" in str(improve.tooltip)
        assert app.screen.query_one("#console-prompts-search", Input).disabled is False
        assert "Browse and manual editing remain available" in str(
            app.screen.query_one("#console-prompts-model-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_source_capabilities_gate_only_unsupported_structured_saves() -> None:
    backend = _PromptBackend()
    backend.capabilities_result = SimpleNamespace(
        structured_kinds=frozenset({(2, "block_prompt")}),
        artifact_types=frozenset({"prompt"}),
        conditional_update=False,
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        save_menu = app.screen.query_one("#prompt-editor-save-menu", Select)
        assert save_menu.disabled is False
        assert [
            value for _label, value in save_menu._options if value is not Select.NULL
        ] == ["prompt"]
        assert "does not support block_recipe" in str(save_menu.tooltip)
        assert "conditional updates" in str(save_menu.tooltip)


@pytest.mark.asyncio
async def test_dirty_back_offers_only_keep_editing_or_discard() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()
        app.screen.mark_dirty()
        await pilot.press("escape")
        await pilot.pause()

        guard = app.screen.query_one("#console-prompts-dirty-guard")
        assert guard.display
        buttons = list(guard.query(Button))
        assert [str(button.label) for button in buttons] == [
            "Keep editing",
            "Discard changes",
        ]
        assert app.screen.state.mode == "edit"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_modal_geometry_keeps_important_actions_in_bounds(
    size: tuple[int, int],
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shell = app.screen.query_one("#console-prompts-modal")
        improve = app.screen.query_one("#console-prompts-improve")
        close = app.screen.query_one("#console-prompts-close")

        assert 0 <= shell.region.x < size[0]
        assert shell.region.x + shell.region.width <= size[0]
        assert 0 <= shell.region.y < size[1]
        assert shell.region.y + shell.region.height <= size[1]
        for widget in (improve, close):
            assert shell.region.contains_region(widget.region)


@pytest.mark.asyncio
async def test_root_escape_dismisses_and_restores_composer_focus() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert getattr(app.focused, "id", None) == "console-native-composer"


async def _dismissal_gesture(pilot: Any, gesture: str) -> None:
    if gesture == "escape":
        await pilot.press("escape")
    else:
        await pilot.click(offset=(0, 0))


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_clean_root_dismissal_returns_exact_result_once(
    gesture: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()
        await pilot.pause()

        assert app.screen is not modal
        assert app.results == [None]
        assert getattr(app.focused, "id", None) == "console-native-composer"


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_clean_nested_dismissal_closes_whole_workbench(
    gesture: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        back = modal.query_one("#console-prompts-back", Button)
        assert back.display is True
        assert str(back.label) == "Back"

        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()
        await pilot.pause()

        assert app.screen is not modal
        assert app.results == [None]
        assert getattr(app.focused, "id", None) == "console-native-composer"


async def _open_dirty_prompt_mode(
    modal: ConsolePromptsModal, pilot: Any, mode: str
) -> TextArea:
    if mode == "edit":
        await modal.open_artifact("prompt-1")
    else:
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
    await pilot.pause()
    editor = modal.query_one("#prompt-block-content-role", TextArea)
    editor.focus()
    await pilot.pause()
    modal.mark_dirty()
    return editor


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["edit", "recipe"])
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_dirty_dismissal_reveals_guard(
    mode: str,
    gesture: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await _open_dirty_prompt_mode(modal, pilot, mode)

        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()

        guard = modal.query_one("#console-prompts-dirty-guard")
        assert app.screen is modal
        assert guard.display is True
        assert modal.state.mode == mode
        assert app.results == []
        assert getattr(app.focused, "id", None) == "console-prompts-keep-editing"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["edit", "recipe"])
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_guard_visible_cannot_be_bypassed(
    mode: str,
    gesture: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        editor = await _open_dirty_prompt_mode(modal, pilot, mode)
        await pilot.press("escape")
        await pilot.pause()
        guard = modal.query_one("#console-prompts-dirty-guard")
        assert guard.display is True

        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []
        if gesture == "escape":
            assert guard.display is False
            assert app.focused is editor
        else:
            assert guard.display is True
            assert getattr(app.focused, "id", None) == "console-prompts-keep-editing"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["edit", "recipe"])
async def test_prompts_repeated_visible_close_preserves_guard_editor_focus(
    mode: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        editor = await _open_dirty_prompt_mode(modal, pilot, mode)
        close = modal.query_one("#console-prompts-close", Button)
        close.press()
        await pilot.pause()
        guard = modal.query_one("#console-prompts-dirty-guard")
        assert guard.display is True
        assert getattr(app.focused, "id", None) == "console-prompts-keep-editing"

        close.press()
        await pilot.pause()
        assert guard.display is True
        assert getattr(app.focused, "id", None) == "console-prompts-keep-editing"
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()
        assert guard.display is False
        assert app.focused is editor
        assert app.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["edit", "recipe"])
@pytest.mark.parametrize("activation", ["click", "keyboard"])
async def test_prompts_close_activation_restores_last_content_focus(
    mode: str,
    activation: str,
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        editor = await _open_dirty_prompt_mode(modal, pilot, mode)
        close = modal.query_one("#console-prompts-close", Button)
        if activation == "click":
            await pilot.click("#console-prompts-close")
        else:
            close.focus()
            await pilot.pause()
            await pilot.press("enter")
        await pilot.pause()

        guard = modal.query_one("#console-prompts-dirty-guard")
        assert guard.display is True
        assert getattr(app.focused, "id", None) == "console-prompts-keep-editing"
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()
        assert guard.display is False
        assert app.focused is editor
        assert app.results == []


@pytest.mark.asyncio
async def test_prompts_expanded_select_descendant_owns_primary_click() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        source = modal.query_one("#console-prompts-source", Select)
        await pilot.click("#console-prompts-source")
        await pilot.pause()

        assert app.screen is modal
        assert source.expanded is True
        overlay = modal.query_one("SelectOverlay")
        assert overlay.display is True
        await pilot.click(overlay, offset=(1, 1))
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []
        assert app.focused is overlay


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_active_improvement_dismissal_cancels_once(
    gesture: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()
    improve_calls = 0

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        nonlocal improve_calls
        improve_calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id, kind="no_change"
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        auto = modal.query_one("#console-prompts-auto-improve", Button)
        auto.focus()
        auto.press()
        await wait_for_signal(started, what="the prompt improvement starting")
        await pilot.pause()
        focused_before = app.focused

        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()

        assert app.screen is modal
        assert modal.state.mode == "improve"
        assert modal._active_request_id is None
        assert modal._improvement_worker is not None
        assert modal._improvement_worker.is_cancelled
        assert improve_calls == 1
        assert app.results == []
        assert app.focused is focused_before
        assert (
            str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Cancelling..."
        )

        release.set()
        await pilot.pause()
        await pilot.pause()
        assert driver.applies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["escape", "backdrop"])
async def test_prompts_cancelling_improvement_dismissal_stays_in_transaction(
    gesture: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id, kind="no_change"
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(started, what="the prompt improvement starting")
        cancel = modal.query_one("#console-prompts-improvement-cancel", Button)
        cancel.focus()
        await pilot.pause()
        cancel.press()
        await pilot.pause()
        assert modal._improvement_worker is not None
        assert modal._improvement_worker.is_cancelled
        focused_before = app.focused

        await _dismissal_gesture(pilot, gesture)
        await pilot.pause()

        assert app.screen is modal
        assert modal.state.mode == "improve"
        assert app.results == []
        assert app.focused is focused_before
        assert (
            str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Cancelling..."
        )

        release.set()
        await pilot.pause()
        await pilot.pause()
        assert driver.applies == []


@pytest.mark.asyncio
async def test_late_local_detail_cannot_open_after_switching_to_server() -> None:
    backend = _PromptBackend()
    local_started = asyncio.Event()
    release_local = asyncio.Event()

    async def detail(source: str, identifier: str) -> Mapping[str, Any]:
        if source == "local":
            local_started.set()
            await release_local.wait()
        return {
            **_detail(identifier=identifier),
            "backend": source,
        }

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._detail = detail

        late_open = asyncio.create_task(modal.open_artifact("late-local"))
        await wait_for_background_signal(
            local_started, late_open, what="the late local detail open"
        )
        await modal.switch_source("server")
        release_local.set()
        await late_open
        await pilot.pause()

        assert modal.state.source == "server"
        assert modal.state.mode == "browse"
        assert modal.state.selected_identity != "late-local"
        assert modal._selected_record is None


@pytest.mark.asyncio
async def test_late_first_detail_cannot_replace_newer_selection() -> None:
    backend = _PromptBackend()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def detail(source: str, identifier: str) -> Mapping[str, Any]:
        if identifier == "prompt-a":
            first_started.set()
            await release_first.wait()
        return {
            **_detail(identifier=identifier),
            "name": f"Name {identifier}",
            "backend": source,
        }

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._detail = detail

        first_open = asyncio.create_task(modal.open_artifact("prompt-a"))
        await wait_for_background_signal(
            first_started, first_open, what="the first detail open"
        )
        await modal.open_artifact("prompt-b")
        release_first.set()
        await first_open
        await pilot.pause()

        assert modal.state.selected_identity == "prompt-b"
        assert modal._selected_record is not None
        assert modal._selected_record["name"] == "Name prompt-b"
        assert modal.state.mode_stack == ("browse", "edit")


@pytest.mark.asyncio
async def test_source_switch_clears_foreign_rows_before_unavailable_result() -> None:
    backend = _PromptBackend(
        pages={
            1: {
                "items": [_brief("local-only")],
                "page": 1,
                "total_pages": 1,
                "total_items": 1,
            }
        }
    )
    server_started = asyncio.Event()
    release_server = asyncio.Event()

    async def list_page(source: str, page: int) -> Mapping[str, Any]:
        if source == "server":
            server_started.set()
            await release_server.wait()
            raise ValueError("Server Prompt source is unavailable.")
        return backend.pages[page]

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._list_page = list_page
        assert modal.query("#console-prompts-result-local-only")

        switch = asyncio.create_task(modal.switch_source("server"))
        await wait_for_background_signal(
            server_started, switch, what="the server source switch"
        )
        foreign_rows_visible_while_loading = bool(
            modal.query("#console-prompts-result-local-only")
        )
        owner_while_loading = modal.browse_result.source
        release_server.set()
        await switch
        await pilot.pause()

        assert foreign_rows_visible_while_loading is False
        assert owner_while_loading == "server"
        assert modal.browse_result.source == "server"
        assert modal.browse_result.items == ()
        assert not modal.query("#console-prompts-result-local-only")
        assert "Server Prompt source is unavailable" in str(
            modal.query_one("#console-prompts-browse-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_recipe_save_as_prompt_becomes_the_guarded_saved_prompt() -> None:
    backend = _PromptBackend()
    backend.detail_result = _detail(artifact_type="recipe")
    backend.save_result = {
        **_detail(
            artifact_type="prompt",
            identifier="local:prompt:new-77",
            version=9,
        ),
        "source_id": "new-77",
        "name": "Saved Prompt",
        "backend": "local",
    }
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("recipe-1")
        await pilot.pause()
        assert modal.state.working_copy_unsaved is True

        _choose_save_action(modal, "prompt")
        await pilot.pause()
        await pilot.pause()

        assert modal.state.working_copy_unsaved is False
        assert modal.state.selected_identity == "new-77"
        assert modal.state.selected_version == 9
        assert modal.state.selected_source == "local"
        assert modal._selected_record is not None
        assert modal._selected_record["name"] == "Saved Prompt"
        assert "Saved Prompt" in str(
            modal.query_one("#console-prompts-location", Static).renderable
        )
        save_menu = modal.query_one("#prompt-editor-save-menu", Select)
        assert "update" in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        assert (
            str(modal.query_one("#prompt-editor-update-reason", Static).renderable)
            == ""
        )

        _choose_save_action(modal, "update")
        await pilot.pause()
        await pilot.pause()

    assert backend.save_calls[1]["prompt_identifier"] == "new-77"
    assert backend.save_calls[1]["expected_version"] == 9
    assert backend.save_calls[1]["name"] == "Saved Prompt"


@pytest.mark.asyncio
async def test_existing_prompt_save_recipe_does_not_retarget_prompt_working_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PromptBackend()
    backend.detail_result = {
        **_detail(identifier="prompt-original", version=4),
        "system_prompt": "STALE COMPILED SYSTEM",
    }
    backend.save_result = _detail(
        artifact_type="recipe",
        identifier="recipe-new",
        version=1,
    )
    app = _Harness(backend)
    notifications: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("prompt-original")
        await pilot.pause()
        original_record = modal._selected_record
        original_decoded = modal._decoded
        warning = modal.query_one("#console-prompts-compatibility-stale", Static)
        modal.mark_dirty()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notifications.append(str(message)),
        )

        _choose_save_action(modal, "recipe")
        await pilot.pause()
        await pilot.pause()

        assert backend.save_calls[0]["artifact_type"] == "recipe"
        assert modal.state.selected_identity == "prompt-original"
        assert modal.state.selected_version == 4
        assert modal.state.working_copy_unsaved is False
        assert modal.state.dirty is True
        assert modal._selected_record is original_record
        assert modal._decoded is original_decoded
        assert warning.display is True
        save_menu = modal.query_one("#prompt-editor-save-menu", Select)
        assert "update" in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        assert notifications == ["Recipe saved as a new artifact."]

        backend.save_result = _detail(
            identifier="prompt-original",
            version=5,
        )
        _choose_save_action(modal, "update")
        await pilot.pause()
        await pilot.pause()

    assert backend.save_calls[1]["artifact_type"] == "prompt"
    assert backend.save_calls[1]["prompt_identifier"] == "prompt-original"
    assert backend.save_calls[1]["expected_version"] == 4
    assert all(
        call.get("prompt_identifier") != "recipe-new" for call in backend.save_calls
    )


@pytest.mark.asyncio
async def test_recipe_derived_prompt_save_recipe_stays_unsaved_and_not_updatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PromptBackend()
    backend.detail_result = _detail(
        artifact_type="recipe",
        identifier="recipe-source",
        version=3,
    )
    backend.save_result = _detail(
        artifact_type="recipe",
        identifier="recipe-copy",
        version=1,
    )
    app = _Harness(backend)
    notifications: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("recipe-source")
        await pilot.pause()
        original_record = modal._selected_record
        original_decoded = modal._decoded
        modal.mark_dirty()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notifications.append(str(message)),
        )

        _choose_save_action(modal, "recipe")
        await pilot.pause()
        await pilot.pause()

        assert backend.save_calls[0]["artifact_type"] == "recipe"
        assert modal.state.selected_identity == "recipe-source"
        assert modal.state.selected_version == 3
        assert modal.state.working_copy_unsaved is True
        assert modal.state.dirty is True
        assert modal._selected_record is original_record
        assert modal._decoded is original_decoded
        save_menu = modal.query_one("#prompt-editor-save-menu", Select)
        assert "update" not in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        assert notifications == ["Recipe saved as a new artifact."]

    assert len(backend.save_calls) == 1


@pytest.mark.asyncio
async def test_name_only_prompt_save_response_warns_and_keeps_working_copy_unpromoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PromptBackend()
    backend.detail_result = _detail(
        artifact_type="recipe",
        identifier="recipe-source",
        version=3,
    )
    backend.save_result = {
        "name": "Name is not identity",
        "artifact_type": "prompt",
        "version": 9,
    }
    app = _Harness(backend)
    notifications: list[tuple[str, str | None]] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("recipe-source")
        await pilot.pause()
        original_record = modal._selected_record
        modal.mark_dirty()
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **kwargs: notifications.append(
                (str(message), kwargs.get("severity"))
            ),
        )

        _choose_save_action(modal, "prompt")
        await pilot.pause()
        await pilot.pause()

        assert modal.state.selected_identity == "recipe-source"
        assert modal.state.selected_identity != "Name is not identity"
        assert modal.state.selected_version == 3
        assert modal.state.working_copy_unsaved is True
        assert modal.state.dirty is True
        assert modal._selected_record is original_record
        save_menu = modal.query_one("#prompt-editor-save-menu", Select)
        assert "update" not in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]

    assert notifications == [
        (
            "Prompt saved, but its new identity was not returned. Reload the Library before updating it.",
            "warning",
        )
    ]


@pytest.mark.asyncio
async def test_stale_compiled_text_warns_that_definition_wins_and_save_repairs() -> (
    None
):
    backend = _PromptBackend()
    backend.detail_result = {
        **_detail(),
        "system_prompt": "STALE COMPILED SYSTEM",
        "user_prompt": "STALE COMPILED USER",
    }
    backend.save_result = _detail(
        identifier="local:prompt:repaired",
        version=5,
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("prompt-1")
        await pilot.pause()

        warnings = list(modal.query("#console-prompts-compatibility-stale"))
        assert warnings
        warning_copy = str(warnings[0].renderable)
        assert "definition is authoritative" in warning_copy
        assert "Saving repairs" in warning_copy

        _choose_save_action(modal, "prompt")
        await pilot.pause()
        await pilot.pause()

        assert warnings[0].display is False

    assert backend.save_calls[0]["system_prompt"] != "STALE COMPILED SYSTEM"
    assert "Be exact." in backend.save_calls[0]["system_prompt"]
    assert backend.save_calls[0]["user_prompt"] == "Answer the question."


@pytest.mark.asyncio
async def test_host_apply_deferral_replaces_ready_copy_and_has_no_apply_path() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("prompt-1")
        await pilot.pause()

        apply_button = modal.query_one("#prompt-editor-apply", Button)
        apply_copy = str(
            modal.query_one("#prompt-editor-apply-reason", Static).renderable
        )
        assert apply_button.disabled is True
        assert "Apply unavailable" in apply_copy
        assert "save this Prompt" in apply_copy
        assert "Ready" not in apply_copy
        apply_button.press()
        await pilot.pause()

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


@pytest.mark.asyncio
async def test_provider_unavailable_configure_action_is_focusable_and_injected() -> (
    None
):
    backend = _PromptBackend()
    configure_calls: list[bool] = []

    async def configure_provider() -> None:
        configure_calls.append(True)

    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
        configure_provider=configure_provider,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        configure = modal.query_one("#console-prompts-configure-provider", Button)
        assert configure.disabled is False
        configure.focus()
        await pilot.pause()
        assert app.focused is configure

        configure.press()
        await pilot.pause()

        assert configure_calls == [True]
        assert modal.state.mode == "browse"
        assert modal.query_one("#console-prompts-search", Input).disabled is False


@pytest.mark.asyncio
async def test_source_switch_cancels_pending_query_debounce() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._query_requested(ConsolePromptsBrowse.QueryChanged("alpha"))
        await modal.switch_source("server")
        await asyncio.sleep(0.23)
        await pilot.pause()

    assert backend.search_calls.count(("server", "alpha")) == 1


@pytest.mark.asyncio
async def test_improve_surface_names_exact_paths_and_shows_captured_context() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
        improvement_kwargs=driver.kwargs(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        await pilot.pause()

        labels = [
            str(app.screen.query_one(selector, Button).label)
            for selector in (
                "#console-prompts-auto-improve",
                "#console-prompts-review-improve",
                "#console-prompts-structured-recipe",
            )
        ]
        assert labels == [
            "Replace draft automatically",
            "Analyze and user review (Recommended)",
            "Build a reusable prompt",
        ]
        assert app.screen.query_one("#console-prompts-auto-improve", Button).disabled
        assert app.screen.query_one("#console-prompts-review-improve", Button).disabled
        assert not app.screen.query_one(
            "#console-prompts-structured-recipe", Button
        ).disabled
        assert "Be accurate." in str(
            app.screen.query_one("#console-prompts-current-system", TextArea).text
        )
        analysis_context = app.screen.query_one(
            "#console-prompts-include-system", Checkbox
        )
        assert str(analysis_context.label) == (
            "Let the improver read the current System prompt"
        )
        assert analysis_context.value is True
        assert analysis_context.disabled is False
        analysis_explanation = str(
            app.screen.query_one(
                "#console-prompts-analysis-context-disclosure", Static
            ).renderable
        )
        assert analysis_explanation == (
            "Used only to improve the draft. It does not change this session."
        )
        current_user = app.screen.query_one("#console-prompts-current-user", TextArea)
        assert "PRIVATE BODY" not in current_user.text
        assert "/private/notes.txt" not in current_user.text
        assert "[[TLDW_PROTECTED:" in current_user.text
        status = str(
            app.screen.query_one("#console-prompts-provider-summary", Static).renderable
        )
        assert "OpenAI" in status and "gpt-test" in status


@pytest.mark.asyncio
async def test_recipe_chooser_explains_each_starting_point_before_selection() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()

        assert str(
            app.screen.query_one(
                "#console-prompts-recipe-outcome-description", Static
            ).renderable
        ) == (
            "Outcome-first starts with Goal, context and evidence, constraints, "
            "and output; reveal optional guidance when useful."
        )
        assert str(
            app.screen.query_one(
                "#console-prompts-recipe-saved-description", Static
            ).renderable
        ) == "Saved Recipe reuses a format from Library > Prompts."
        assert str(
            app.screen.query_one(
                "#console-prompts-recipe-blank-description", Static
            ).renderable
        ) == "Blank starts with empty System and User lanes for your own blocks."
        outcome = app.screen.query_one(
            "#console-prompts-recipe-outcome-first", Button
        )
        blank = app.screen.query_one("#console-prompts-recipe-blank", Button)
        outcome.focus()
        await pilot.pause()
        assert outcome.region.height >= 3
        assert "Outcome-first" in "".join(_rendered_lines(outcome))
        assert "&#160;Outcome-first&#160;" in app.export_screenshot()
        blank.focus()
        await pilot.pause()
        assert blank.region.height >= 3
        assert "Blank" in "".join(_rendered_lines(blank))

        outcome.press()
        await pilot.pause()

        for block_id in ("goal", "context-evidence", "constraints", "output"):
            assert app.screen.query_one(f"#prompt-block-{block_id}").display
        for block_id in (
            "role",
            "personality",
            "collaboration-style",
            "success-criteria",
            "stop-rules",
        ):
            assert not app.screen.query_one(f"#prompt-block-{block_id}").display
        reveal = app.screen.query_one("#prompt-editor-show-optional", Button)
        assert str(reveal.label) == "Show 5 optional blocks"


@pytest.mark.asyncio
async def test_recipe_save_confirms_library_destination_and_opens_saved_identity() -> None:
    backend = _PromptBackend()
    backend.save_result = {
        **_detail(artifact_type="recipe", identifier="77", version=1),
        "id": "local:prompt:recipe-uuid-77",
        "local_id": 77,
        "source_id": "recipe-uuid-77",
        "name": "Reusable analysis",
    }
    driver = _ImprovementDriver()
    opened: list[tuple[str, str]] = []

    def open_library(source: str, identity: str) -> bool:
        opened.append((source, identity))
        return True

    app = _Harness(
        backend,
        improvement_kwargs={
            **driver.kwargs(),
            "open_library_prompt": open_library,
        },
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one(
            "#console-prompts-recipe-outcome-first", Button
        ).press()
        await pilot.pause()

        _choose_save_action(app.screen, "recipe")
        await pilot.pause()
        await pilot.pause()

        assert len(backend.save_calls) == 1
        confirmation = app.screen.query_one(
            "#console-prompts-recipe-save-confirmation", Static
        )
        assert str(confirmation.renderable) == (
            "Reusable analysis saved to Library > Prompts as a Recipe."
        )
        open_library = app.screen.query_one(
            "#console-prompts-open-saved-recipe", Button
        )
        assert open_library.display is True
        assert open_library.has_focus

        open_library.press()
        await pilot.pause()

    assert opened == [("local", "77")]
    assert app.results == [None]


@pytest.mark.asyncio
async def test_recipe_saved_confirmation_disables_unsupported_library_target() -> None:
    backend = _PromptBackend()
    app = _Harness(
        backend,
        improvement_kwargs={
            "open_library_prompt": lambda _source, _identity: True,
        },
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("recipe")
        modal.query_one(
            "#console-prompts-recipe-outcome-first", Button
        ).press()
        await pilot.pause()
        modal._show_recipe_saved_confirmation(
            {
                "backend": "server",
                "source_id": "recipe-server-77",
                "name": "Server recipe",
            }
        )
        await pilot.pause()

        open_library = modal.query_one(
            "#console-prompts-open-saved-recipe", Button
        )
        assert open_library.display is True
        assert open_library.disabled is True
        assert modal._saved_recipe_library_target is None
        assert "local recipes" in str(open_library.tooltip)
        assert app.results == []


@pytest.mark.asyncio
async def test_recipe_library_open_keeps_modal_when_navigation_declines() -> None:
    backend = _PromptBackend()
    app = _Harness(
        backend,
        improvement_kwargs={
            "open_library_prompt": lambda _source, _identity: False,
        },
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("recipe")
        modal.query_one(
            "#console-prompts-recipe-outcome-first", Button
        ).press()
        await pilot.pause()
        modal._show_recipe_saved_confirmation(
            {
                "backend": "local",
                "local_id": 77,
                "name": "Local recipe",
            }
        )
        await pilot.pause()

        modal.query_one("#console-prompts-open-saved-recipe", Button).press()
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []


@pytest.mark.asyncio
async def test_system_analysis_opt_out_survives_recipe_path_replacement() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(request_id="ignored", kind="no_change"),
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            filled_definition=outcome_first_recipe(),
        ),
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        include_system = app.screen.query_one(
            "#console-prompts-include-system", Checkbox
        )
        include_system.value = False
        await pilot.pause()
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert driver.requests[-1].system_prompt is None
        assert driver.requests[-1].system_fingerprint is None

        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert driver.requests[-1].mode == "recipe"
        assert driver.requests[-1].system_prompt is None
        assert driver.requests[-1].system_fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "review", "recipe"])
@pytest.mark.parametrize("system_case", ["included", "excluded", "absent"])
async def test_improvement_system_read_permission_is_request_only(
    mode: str,
    system_case: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(request_id="ignored", kind="no_change")
    )
    kwargs = driver.kwargs()
    if system_case == "absent":
        kwargs["improvement_context"] = SimpleNamespace(
            **{
                **vars(driver.context),
                "current_system_prompt": "",
                "current_system_fingerprint": None,
            }
        )
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        checkbox = app.screen.query_one("#console-prompts-include-system", Checkbox)
        disclosure = app.screen.query_one(
            "#console-prompts-analysis-context-disclosure", Static
        )

        assert str(checkbox.label) == (
            "Let the improver read the current System prompt"
        )
        if system_case == "absent":
            assert checkbox.disabled is True
            assert checkbox.value is False
            assert str(checkbox.tooltip) == (
                "Unavailable — this session has no current System prompt."
            )
            assert "no current System prompt" in str(disclosure.renderable)
            assert "only the unsent message" in str(disclosure.renderable)
        else:
            assert checkbox.disabled is False
            checkbox.value = system_case == "included"
            await pilot.pause()
            assert str(disclosure.renderable) == (
                "Used only to improve the draft. It does not change this session."
            )

        if mode == "auto":
            app.screen.query_one("#console-prompts-auto-improve", Button).press()
        elif mode == "review":
            app.screen.query_one("#console-prompts-review-improve", Button).press()
        else:
            app.screen.query_one(
                "#console-prompts-structured-recipe", Button
            ).press()
            await pilot.pause()
            app.screen.query_one(
                "#console-prompts-recipe-outcome-first", Button
            ).press()
            await pilot.pause()
            recipe_checkbox = app.screen.query_one(
                "#console-prompts-include-system", Checkbox
            )
            assert recipe_checkbox.value is (system_case == "included")
            app.screen.query_one("#console-prompts-recipe-fill", Button).press()

        await pilot.pause()
        await pilot.pause()

        request = driver.requests[-1]
        assert request.mode == mode
        assert request.system_prompt == (
            "Be accurate." if system_case == "included" else None
        )
        assert request.system_fingerprint == (
            "system-fingerprint" if system_case == "included" else None
        )
        assert driver.applies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("include_system", [True, False])
async def test_recipe_editor_keeps_analysis_disclosure_bound_through_edits_and_fill(
    include_system: bool,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        started.set()
        await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind="no_change",
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()

        editor = modal.query_one(PromptBlockEditor)
        await editor._change_field("goal", "content", "Edited before Fill.")
        await pilot.pause()
        edited_definition = modal.query_one(PromptBlockEditor).state.definition
        assert modal.state.dirty is True

        analysis_context = modal.query_one("#console-prompts-include-system", Checkbox)
        disclosure = str(
            modal.query_one(
                "#console-prompts-recipe-analysis-disclosure", Static
            ).renderable
        )
        assert str(analysis_context.label) == (
            "Let the improver read the current System prompt"
        )
        assert analysis_context.value is True
        assert disclosure == (
            "Used only to improve the draft. It does not change this session."
        )
        assert modal.query_one("#prompt-editor-apply-system", Checkbox).value is False

        analysis_context.value = not include_system
        await pilot.pause()
        analysis_context.value = include_system
        await pilot.pause()

        assert modal._include_system_context is include_system
        assert modal.query_one(PromptBlockEditor).state.definition == edited_definition
        assert modal.state.dirty is True

        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await wait_for_signal(started, what="the recipe-fill improvement starting")
        await pilot.pause()

        assert (
            modal.query_one("#console-prompts-include-system", Checkbox).value
            is include_system
        )
        assert driver.requests[-1].system_prompt == (
            "Be accurate." if include_system else None
        )
        assert driver.requests[-1].system_fingerprint == (
            "system-fingerprint" if include_system else None
        )
        assert driver.requests[-1].recipe_definition == replace(
            edited_definition,
            kind="block_recipe",
        )

        release.set()
        await pilot.pause()
        await pilot.pause()


@pytest.mark.asyncio
async def test_compact_recipe_working_copy_scrolls_to_the_block_editor() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()

        scroll = modal.query_one("#console-prompts-recipe-scroll")
        scroll.scroll_end(animate=False)
        await pilot.pause()

        editor = modal.query_one(PromptBlockEditor)
        footer = modal.query_one("#console-prompts-footer")
        assert editor.region.y < footer.region.y
        assert editor.region.bottom > modal.query_one("#console-prompts-body").region.y


@pytest.mark.asyncio
async def test_recipe_retry_save_is_absent_from_compact_tab_order_until_failure() -> (
    None
):
    backend = _PromptBackend()
    driver = _ImprovementDriver()

    async def persistence_failure(_result: Any, _snapshot: Any) -> Any:
        return SimpleNamespace(kind="persistence_failed", user_message="")

    kwargs = driver.kwargs()
    kwargs["apply_improvement_result"] = persistence_failure
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()

        retry = modal.query_one("#console-prompts-persistence-retry", Button)
        assert retry.display is False
        assert retry.disabled is True
        assert retry.can_focus is False
        assert retry.region.width == 0 and retry.region.height == 0

        focused_ids: list[str | None] = []
        for _ in range(24):
            await pilot.press("tab")
            focused_ids.append(getattr(app.focused, "id", None))
        assert "console-prompts-persistence-retry" not in focused_ids

        captured = SimpleNamespace(
            composer_snapshot=driver.snapshot,
            mode="recipe",
        )
        result = modal._result_for(
            user_text=None,
            system_text="Updated System",
            apply_user=False,
            apply_system=True,
            captured=captured,
        )
        await modal._coordinate_apply(result, captured)
        await pilot.pause()

        assert retry.display is True
        assert retry.disabled is False
        assert retry.can_focus is True
        assert app.focused is retry

        await modal._back_internal(discard=True)
        await pilot.pause()
        assert retry.display is False
        assert retry.disabled is True
        assert retry.can_focus is False
        assert modal._pending_persistence_result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation", ["close", "back", "escape"])
async def test_held_improve_activation_navigation_cancels_and_ignores_late_resolution(
    navigation: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()
    activation_calls = 0

    async def activate() -> Any:
        nonlocal activation_calls
        activation_calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            await release.wait()
        return driver.context

    kwargs = driver.kwargs()
    kwargs["activate_improvement_context"] = activate
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        improve = modal.query_one("#console-prompts-improve", Button)
        improve.press()
        await pilot.pause()
        modal.query_one("#console-prompts-review-improve", Button).press()
        await wait_for_signal(started, what="the held improvement activation starting")
        await asyncio.sleep(0.05)

        if navigation == "close":
            modal.query_one("#console-prompts-close", Button).press()
        elif navigation == "back":
            modal.query_one("#console-prompts-back", Button).press()
        else:
            modal.action_back()
        await asyncio.sleep(0.05)
        dismissed_while_held = app.screen is not modal

        release.set()
        await pilot.pause()
        await pilot.pause()

        assert dismissed_while_held is (navigation == "close")
        expected_mode = "improve" if navigation == "close" else "browse"
        assert modal.state.mode == expected_mode
        assert activation_calls == 1
        assert driver.requests == []


@pytest.mark.asyncio
async def test_held_improve_activation_disables_duplicate_resolution() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()
    activation_calls = 0

    async def activate() -> Any:
        nonlocal activation_calls
        activation_calls += 1
        started.set()
        await release.wait()
        return driver.context

    kwargs = driver.kwargs()
    kwargs["activate_improvement_context"] = activate
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        improve = modal.query_one("#console-prompts-improve", Button)
        improve.press()
        await pilot.pause()
        auto = modal.query_one("#console-prompts-auto-improve", Button)
        review = modal.query_one("#console-prompts-review-improve", Button)
        structured = modal.query_one("#console-prompts-structured-recipe", Button)
        auto.press()
        await wait_for_signal(started, what="the held improvement activation starting")
        await asyncio.sleep(0.05)

        disabled_while_held = all(
            button.disabled for button in (auto, review, structured)
        )
        resolving_copy = str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        )
        review.press()
        await asyncio.sleep(0.05)
        calls_while_held = activation_calls

        release.set()
        await pilot.pause()
        await pilot.pause()

        assert disabled_while_held is True
        assert "Resolving" in resolving_copy
        assert calls_while_held == 1
        assert activation_calls == 1
        assert modal.state.mode == "improve"


@pytest.mark.asyncio
async def test_stale_opening_context_does_not_offer_provider_recovery() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()

    async def activate() -> Any:
        raise ValueError("The Console System prompt changed.")

    async def configure_provider() -> None:
        raise AssertionError("stale context is not repaired in provider settings")

    kwargs = driver.kwargs()
    kwargs["activate_improvement_context"] = activate
    app = _Harness(
        backend,
        improvement_kwargs=kwargs,
        configure_provider=configure_provider,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-review-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        status = modal.query_one("#console-prompts-improvement-status", Static)
        assert str(status.renderable) == "The Console System prompt changed."
        assert list(modal.query("#console-prompts-configure-provider")) == []
        assert app.focused is modal.query_one(
            "#console-prompts-structured-recipe", Button
        )


@pytest.mark.asyncio
async def test_invalidated_activation_token_ignores_late_resolution() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()

    async def activate() -> Any:
        started.set()
        await release.wait()
        return driver.context

    kwargs = driver.kwargs()
    kwargs["activate_improvement_context"] = activate
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        activation_id = modal._next_activation_id()
        modal._active_activation_id = activation_id
        activation = asyncio.create_task(
            modal._run_improvement_activation(activation_id)
        )
        await wait_for_background_signal(
            started, activation, what="the improvement activation"
        )

        modal._active_activation_id = None
        release.set()
        await activation
        await pilot.pause()

        assert modal.state.mode == "browse"
        assert driver.requests == []


@pytest.mark.asyncio
async def test_auto_success_returns_one_apply_transaction_and_closes() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt="Improved question [[TLDW_PROTECTED:kept]]",
        )
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert len(driver.applies) == 1
        result, captured = driver.applies[0]
        assert result.kind == "apply"
        assert result.apply_user is True
        assert result.apply_system is False
        assert result.user_text == "Improved question [[TLDW_PROTECTED:kept]]"
        assert result.composer_snapshot == driver.snapshot
        assert captured.mode == "auto"


@pytest.mark.asyncio
async def test_no_change_keeps_modal_open_without_apply_or_new_usage() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(request_id="ignored", kind="no_change")
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert app.screen.state.mode == "improve"
        assert (
            str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Prompt already looks good"
        )
        assert app.screen._active_request_id is None
        assert driver.applies == []
        assert backend.usage_mutations == 0


@pytest.mark.asyncio
async def test_active_improvement_disables_duplicate_model_launches() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()
    improve_calls: list[str] = []

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        improve_calls.append(snapshot.request_id)
        started.set()
        await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind="no_change",
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        auto = app.screen.query_one("#console-prompts-auto-improve", Button)
        review = app.screen.query_one("#console-prompts-review-improve", Button)
        auto.press()
        await wait_for_signal(started, what="the auto-improvement starting")
        await pilot.pause()

        assert auto.disabled is True
        assert review.disabled is True
        auto.press()
        await pilot.pause()
        assert improve_calls == ["prompt-improvement-1"]

        release.set()
        for _ in range(20):
            await pilot.pause()
            if (
                app.screen._active_activation_id is None
                and app.screen._active_request_id is None
            ):
                break
        assert auto.disabled is False
        assert review.disabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "review"])
@pytest.mark.parametrize("close_source", ["escape", "backdrop", "close", "back"])
async def test_apply_transaction_blocks_every_close_path_until_commit_finishes(
    mode: str,
    close_source: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    driver.outcomes.append(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt=driver.preview.text.replace(
                "Draft question", "Committed candidate"
            ),
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()
    apply_calls = 0

    async def apply(result: Any, snapshot: Any) -> Any:
        nonlocal apply_calls
        apply_calls += 1
        started.set()
        await release.wait()
        return SimpleNamespace(kind="applied", user_message="")

    kwargs = driver.kwargs()
    kwargs["apply_improvement_result"] = apply
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one(
            "#console-prompts-auto-improve"
            if mode == "auto"
            else "#console-prompts-review-improve",
            Button,
        ).press()
        await pilot.pause()
        await pilot.pause()
        if mode == "review":
            modal.query_one("#console-prompts-review-apply", Button).press()
        await wait_for_signal(started, what=f"the {mode} apply commit starting")
        await pilot.pause()

        assert modal._apply_in_progress is True
        assert "Applying" in str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        )
        assert getattr(app.focused, "id", None) == "console-prompts-improvement-status"
        assert modal.query_one("#console-prompts-close", Button).disabled
        assert modal.query_one("#console-prompts-back", Button).disabled
        duplicate_apply = next(
            (
                button
                for selector in (
                    "#console-prompts-review-apply",
                    "#console-prompts-auto-improve",
                )
                for button in modal.query(selector)
            ),
            None,
        )
        assert duplicate_apply is not None and duplicate_apply.disabled
        duplicate_apply.press()
        await pilot.pause()
        assert apply_calls == 1

        if close_source == "escape":
            await pilot.press("escape")
        elif close_source == "backdrop":
            await pilot.click(offset=(0, 0))
        elif close_source == "close":
            modal.query_one("#console-prompts-close", Button).press()
        else:
            modal.query_one("#console-prompts-back", Button).press()
        await pilot.pause()

        assert app.screen is modal
        assert apply_calls == 1
        assert app.results == []

        release.set()
        await pilot.pause()
        await pilot.pause()

        assert apply_calls == 1
        assert len(app.results) == 1
        assert app.results[0].kind == "apply"


@pytest.mark.asyncio
async def test_stale_apply_completion_cannot_dismiss_a_new_top_screen() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt="Committed candidate",
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def apply(_result: Any, _snapshot: Any) -> Any:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            await release.wait()
        return SimpleNamespace(kind="applied", user_message="")

    kwargs = driver.kwargs()
    kwargs["apply_improvement_result"] = apply
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(started, what="the stale apply commit starting")

        modal.dismiss(None)
        await pilot.pause()
        overlay = _ApplyOverlay()
        await app.push_screen(overlay)
        await pilot.pause()

        release.set()
        await pilot.pause()
        await pilot.pause()

        assert app.screen is overlay
        assert app.results == [None]


@pytest.mark.asyncio
async def test_apply_completion_keeps_nested_top_and_reports_committed_state() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt="Committed candidate",
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def apply(_result: Any, _snapshot: Any) -> Any:
        started.set()
        await release.wait()
        return SimpleNamespace(kind="applied", user_message="")

    kwargs = driver.kwargs()
    kwargs["apply_improvement_result"] = apply
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(started, what="the nested apply commit starting")

        overlay = _ApplyOverlay()
        await app.push_screen(overlay)
        release.set()
        await pilot.pause()
        await pilot.pause()

        assert app.screen is overlay
        overlay.dismiss(None)
        await pilot.pause()
        assert app.screen is modal
        assert modal._apply_in_progress is False
        assert (
            str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Applied to the Console."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("validation_result", ["success", "failure"])
async def test_stale_review_validation_cannot_mutate_or_commit_after_remount(
    validation_result: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    driver.outcomes.append(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt=driver.preview.text.replace(
                "Draft question", "Reviewed candidate"
            ),
        )
    )
    validation_started = asyncio.Event()
    validation_finished = asyncio.Event()
    release_validation = asyncio.Event()
    apply_calls = 0

    async def validate(_snapshot: Any, _candidate: str) -> None:
        validation_started.set()
        try:
            await release_validation.wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            await release_validation.wait()
        validation_finished.set()
        if validation_result == "failure":
            raise ValueError("stale validation failure")

    async def apply(_result: Any, _snapshot: Any) -> Any:
        nonlocal apply_calls
        apply_calls += 1
        return SimpleNamespace(kind="applied", user_message="")

    kwargs = driver.kwargs()
    kwargs["validate_improvement"] = validate
    kwargs["apply_improvement_result"] = apply
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-review-improve", Button).press()
        await pilot.pause()
        await pilot.pause()
        modal.query_one("#console-prompts-review-apply", Button).press()
        await wait_for_signal(
            validation_started, what="the stale review validation starting"
        )
        old_generation = modal._safe_mount_generation

        modal.dismiss(None)
        await pilot.pause()
        await app.push_screen(modal)
        await pilot.pause()
        assert modal._safe_mount_generation > old_generation
        await modal.enter_mode("improve")
        assert modal._claim_apply_transaction() == modal._safe_mount_generation
        modal._set_improvement_status("New presentation transaction")

        release_validation.set()
        await wait_for_signal(
            validation_finished, what="the stale review validation finishing"
        )
        await pilot.pause()
        await pilot.pause()

        assert apply_calls == 0
        assert modal._apply_in_progress is True
        assert (
            str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "New presentation transaction"
        )


@pytest.mark.asyncio
async def test_review_success_exposes_exactly_one_editable_user_area_then_applies() -> (
    None
):
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    rewritten = driver.preview.text.replace("Draft question", "Rewritten user message")
    driver.outcomes.append(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            rewritten_prompt=rewritten,
        )
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-review-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        areas = [area for area in app.screen.query(TextArea) if not area.read_only]
        assert len(areas) == 1
        assert areas[0].id == "console-prompts-review-user"
        assert areas[0].text == rewritten
        app.screen.query_one("#console-prompts-review-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert len(driver.applies) == 1
        result, _snapshot = driver.applies[0]
        assert result.user_text == rewritten
        assert result.system_text is None


@pytest.mark.asyncio
async def test_preservation_veto_retains_candidate_in_review_and_tamper_blocks_apply() -> (
    None
):
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="preservation_veto",
            rewritten_prompt="Candidate with protected token removed",
        )
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-review-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert (
            str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Review required before applying"
        )
        area = app.screen.query_one("#console-prompts-review-user", TextArea)
        assert area.text == "Candidate with protected token removed"
        app.screen.query_one("#console-prompts-review-apply", Button).press()
        await pilot.pause()

        assert driver.applies == []
        assert (
            "protected"
            in str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            ).lower()
        )


@pytest.mark.asyncio
async def test_typed_provider_error_keeps_state_and_exposes_retry() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="provider_error",
            user_message="Provider request failed.",
        ),
        PromptImprovementOutcome(request_id="ignored", kind="no_change"),
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        retry = app.screen.query_one("#console-prompts-improvement-retry", Button)
        assert retry.display and not retry.disabled
        assert "Provider request failed" in str(
            app.screen.query_one(
                "#console-prompts-improvement-status", Static
            ).renderable
        )
        retry.press()
        await pilot.pause()
        await pilot.pause()
        assert len(driver.requests) == 2


@pytest.mark.asyncio
async def test_cancel_invalidates_request_and_ignores_late_completion() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind="success",
            rewritten_prompt="Late candidate",
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(started, what="the auto-improvement starting")
        app.screen.query_one("#console-prompts-improvement-cancel", Button).press()
        await pilot.pause()
        assert (
            str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Cancelling..."
        )
        release.set()
        await pilot.pause()
        await pilot.pause()

        assert driver.applies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation", ["back", "escape"])
async def test_back_and_escape_cancel_active_improvement_before_navigation(
    navigation: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    started = asyncio.Event()
    release = asyncio.Event()

    async def improve(snapshot: Any) -> PromptImprovementOutcome:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind="success",
            rewritten_prompt="Late candidate",
        )

    kwargs = driver.kwargs()
    kwargs["improve"] = improve
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(started, what="the auto-improvement starting")

        if navigation == "back":
            app.screen.query_one("#console-prompts-back", Button).press()
        else:
            await pilot.press("escape")
        await pilot.pause()

        assert app.screen.state.mode == "improve"
        assert app.screen._active_request_id is None
        assert (
            str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Cancelling..."
        )

        release.set()
        await pilot.pause()
        await pilot.pause()
        assert driver.applies == []


@pytest.mark.asyncio
async def test_recipe_fill_mounts_service_block_prompt_as_unsaved_prompt_review() -> (
    None
):
    backend = _PromptBackend()
    source_recipe = outcome_first_recipe()
    user_lane = source_recipe.lanes[1]
    filled_blocks = list(user_lane.blocks)
    filled_blocks[0] = replace(filled_blocks[0], content="Deliver a checked answer.")
    filled_prompt = replace(
        source_recipe,
        kind="block_prompt",
        lanes=(source_recipe.lanes[0], replace(user_lane, blocks=tuple(filled_blocks))),
    )
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored",
            kind="success",
            filled_definition=filled_prompt,
        )
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()
        source_fingerprint = app.screen._recipe_source_fingerprint

        app.screen.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        editor = app.screen.query_one(PromptBlockEditor)
        assert editor.state.artifact_type == "prompt"
        assert editor.state.definition == filled_prompt
        assert app.screen.state.working_copy_unsaved is True
        assert driver.requests[-1].recipe_source is None
        assert app.screen._recipe_source_id == "builtin:outcome-first"
        assert app.screen._recipe_source_fingerprint == source_fingerprint
        assert backend.usage_mutations == 0
        assert len(app.screen.query("#console-prompts-recipe-fill")) == 0
        assert len(app.screen.query("#console-prompts-include-system")) == 0
        assert len(app.screen.query("#console-prompts-recipe-analysis-disclosure")) == 0
        assert (
            app.screen.query_one("#prompt-editor-apply-system", Checkbox).value is False
        )
        retry = app.screen.query_one("#console-prompts-persistence-retry", Button)
        assert retry.display is False
        assert retry.disabled is True
        assert retry.can_focus is False


@pytest.mark.asyncio
async def test_real_recipe_fill_with_additional_context_reaches_block_review() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    outcome = await _real_additional_context_recipe_outcome(driver)
    driver.outcomes.append(outcome)
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()

        app.screen.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        editor = app.screen.query_one(PromptBlockEditor)
        mapped = editor.state.definition.lanes[1].blocks[-1]
        assert editor.state.artifact_type == "prompt"
        assert mapped.id == ADDITIONAL_CONTEXT_RESERVED_PREFIX
        assert mapped.content.startswith("Unmatched evidence:")
        assert app.screen.state.working_copy_unsaved is True
        assert len(app.screen.query("#console-prompts-recipe-fill")) == 0


@pytest.mark.asyncio
async def test_mapped_context_blocks_recipe_save_until_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    outcome = await _real_additional_context_recipe_outcome(driver)
    driver.outcomes.append(outcome)
    app = _Harness(backend, improvement_kwargs=driver.kwargs())
    notifications: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, **_kwargs: notifications.append(str(message)),
        )
        modal.state = modal.state.select(
            identity="prompt-7",
            version=4,
            source="local",
            capabilities=backend.capabilities_result,
        )
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        editor = modal.query_one(PromptBlockEditor)
        save_menu = modal.query_one("#prompt-editor-save-menu", Select)
        editor._sync_footer()
        modal._sync_editor_host_gates(editor)
        await modal._save_editor_state(editor.state, artifact_type="recipe")

        assert "recipe" not in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        assert backend.save_calls == []
        assert notifications == [
            "Recipe save unavailable — delete the mapped Additional context block first."
        ]

        modal.query_one(
            "#prompt-block-delete-additional-context",
            Button,
        ).press()
        await pilot.pause()

        assert "recipe" in [
            value
            for _label, value in save_menu._options
            if value is not Select.NULL
        ]
        _choose_save_action(modal, "recipe")
        await pilot.pause()
        await pilot.pause()

    assert len(backend.save_calls) == 1
    assert backend.save_calls[0]["artifact_type"] == "recipe"
    saved_definition = backend.save_calls[0]["prompt_definition"]
    assert saved_definition["kind"] == "block_recipe"
    assert all(
        block["id"] != ADDITIONAL_CONTEXT_RESERVED_PREFIX
        for lane in saved_definition["lanes"]
        for block in lane["blocks"]
    )
    assert notifications[-1] == "Recipe saved as a new artifact."


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe_kind", ["outcome", "blank"])
async def test_recipe_fill_snapshots_current_editor_definition(
    recipe_kind: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver(
        PromptImprovementOutcome(request_id="ignored", kind="no_change")
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        selector = (
            "#console-prompts-recipe-outcome-first"
            if recipe_kind == "outcome"
            else "#console-prompts-recipe-blank"
        )
        app.screen.query_one(selector, Button).press()
        await pilot.pause()
        editor = app.screen.query_one(PromptBlockEditor)
        if recipe_kind == "blank":
            await editor._add_lane_block("user")
            await pilot.pause()
            editor = app.screen.query_one(PromptBlockEditor)
            block_id = editor.state.definition.lanes[1].blocks[0].id
        else:
            block_id = "goal"
        await editor._change_field(block_id, "content", "Edited before Fill.")
        await pilot.pause()
        editor = app.screen.query_one(PromptBlockEditor)
        if recipe_kind == "outcome":
            editor.query_one("#prompt-block-duplicate-goal", Button).press()
            await pilot.pause()
            editor = app.screen.query_one(PromptBlockEditor)
        expected = replace(editor.state.definition, kind="block_recipe")

        app.screen.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert driver.requests[-1].recipe_definition == expected
        assert driver.requests[-1].recipe_fingerprint is not None
        assert driver.requests[-1].recipe_source is None


@pytest.mark.asyncio
async def test_saved_recipe_ai_fill_snapshots_accepted_source_with_raw_identity() -> (
    None
):
    source_id = "saved-recipe-source"
    composite_id = f"local:prompt:{source_id}"
    backend = _PromptBackend(
        pages={
            1: {
                "items": [
                    {
                        **_brief(composite_id, artifact_type="recipe"),
                        "source_id": source_id,
                    }
                ],
                "page": 1,
                "total_pages": 1,
                "total_items": 1,
            }
        }
    )
    backend.detail_result = {
        **_detail(artifact_type="recipe", identifier=composite_id),
        "source_id": source_id,
    }
    driver = _ImprovementDriver(
        PromptImprovementOutcome(request_id="ignored", kind="no_change")
    )
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-saved", Button).press()
        await pilot.pause()
        app.screen.query_one(".console-prompts-result", Button).press()
        await pilot.pause()
        await pilot.pause()

        app.screen.query_one("#console-prompts-recipe-fill", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert driver.requests[-1].recipe_source == "local"
        assert driver.requests[-1].recipe_source_id == source_id
        assert backend.detail_calls == [("local", source_id)]

        app.screen.query_one("#prompt-editor-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

        guard = driver.applies[-1][1]
        assert isinstance(guard, ConsoleRecipeApplyGuard)
        assert guard.recipe_source == "local"
        assert guard.recipe_source_id == source_id


@pytest.mark.asyncio
async def test_builtin_recipe_manual_guard_has_no_saved_source() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        app.screen.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()
        editor = app.screen.query_one(PromptBlockEditor)
        await editor._change_field("goal", "content", "Answer the question.")
        await pilot.pause()

        app.screen.query_one("#prompt-editor-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

    guard = driver.applies[-1][1]
    assert isinstance(guard, ConsoleRecipeApplyGuard)
    assert guard.recipe_source is None
    assert guard.recipe_source_id == "builtin:outcome-first"


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation", ["back", "escape", "close"])
async def test_leaving_saved_recipe_chooser_clears_selection_intent(
    navigation: str,
) -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(backend, improvement_kwargs=driver.kwargs())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await pilot.pause()
        assert modal._recipe_selecting is True

        if navigation == "back":
            modal.query_one("#console-prompts-back", Button).press()
        elif navigation == "escape":
            await pilot.press("escape")
        else:
            modal.query_one("#console-prompts-close", Button).press()
        await pilot.pause()

        assert modal._recipe_selecting is False


@pytest.mark.asyncio
async def test_stale_persistence_retry_shows_host_message_and_stops_retrying() -> None:
    backend = _PromptBackend()
    driver = _ImprovementDriver()

    async def stale_retry(_result: Any) -> Any:
        return SimpleNamespace(
            kind="stale",
            user_message="The live System prompt changed.",
        )

    kwargs = driver.kwargs()
    kwargs["retry_improvement_persistence"] = stale_retry
    app = _Harness(backend, improvement_kwargs=kwargs)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        captured = SimpleNamespace(
            composer_snapshot=driver.snapshot,
            mode="recipe",
        )
        app.screen._pending_persistence_result = app.screen._result_for(
            user_text=None,
            system_text="Updated system",
            apply_user=False,
            apply_system=True,
            captured=captured,
        )
        retry = app.screen.query_one("#console-prompts-persistence-retry", Button)
        retry.display = True
        retry.disabled = False
        retry.can_focus = True

        await app.screen._retry_persistence()
        await pilot.pause()

        assert (
            str(
                app.screen.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "The live System prompt changed."
        )
        assert app.screen._pending_persistence_result is None
        assert retry.display is False
        assert retry.disabled is True
        assert retry.can_focus is False


@pytest.mark.asyncio
async def test_recipe_manual_paths_work_without_provider_and_lane_defaults_are_safe() -> (
    None
):
    backend = _PromptBackend()
    driver = _ImprovementDriver()
    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
        improvement_kwargs=driver.kwargs(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.enter_mode("improve")
        app.screen.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()

        outcome = app.screen.query_one("#console-prompts-recipe-outcome-first", Button)
        blank = app.screen.query_one("#console-prompts-recipe-blank", Button)
        assert not outcome.disabled and not blank.disabled
        assert not app.screen.query("#console-prompts-recipe-fill")
        outcome.press()
        await pilot.pause()

        editor = app.screen.query_one(PromptBlockEditor)
        assert editor.state.definition == outcome_first_recipe()
        assert not editor.query_one("#prompt-editor-apply-system", Checkbox).value
        assert not editor.query_one("#prompt-editor-apply-user", Checkbox).value


def test_composer_public_validation_seam_checks_without_mutating() -> None:
    composer = ConsoleComposerBar()
    composer.insert_text("Draft ")
    composer.insert_file_segment("PRIVATE", "secret.txt")
    before = composer.capture_draft_snapshot()
    projection = composer.project_snapshot_for_model(before, request_nonce="validate-1")

    assert composer.validate_improvement(before, projection.text) is None
    assert composer.capture_draft_snapshot() == before
    assert not composer.improvement_undo_available
    with pytest.raises(ComposerTransactionValidationError, match="placeholder"):
        composer.validate_improvement(
            before,
            projection.text.replace(projection.placeholder_ids[0], ""),
        )
    assert composer.capture_draft_snapshot() == before


def test_captured_improvement_context_repr_hides_prompt_bytes() -> None:
    driver = _ImprovementDriver()
    context = ConsolePromptImprovementContext(
        session_id="session-1",
        composer_snapshot=driver.snapshot,
        current_user_projection=driver.preview,
        current_system_prompt="PRIVATE SYSTEM BYTES",
        current_system_fingerprint="fingerprint",
        provider_label="OpenAI",
        model_label="gpt-test",
    )

    rendered = repr(context)
    assert "PRIVATE BODY" not in rendered
    assert "PRIVATE SYSTEM BYTES" not in rendered
