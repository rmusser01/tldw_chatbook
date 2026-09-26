"""Composer entry points and lossless Draft Shelf save behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App

from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    PromptDraftShelfFullError,
)
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.UI.Console_Modules.prompts import (
    ConsolePromptsController,
    _ConsolePromptSource,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ACTION_SAVE_PROMPT_DRAFT,
    build_composer_menu_entries,
)
from tldw_chatbook.Widgets.Console.console_prompt_draft_save_dialog import (
    ConsolePromptDraftSaveDialog,
)


def test_nonblank_draft_exposes_one_shared_save_destination_in_menu_and_palette():
    entries = build_composer_menu_entries(draft_available=True)
    ids = [entry.action_id for entry in entries]

    assert ids[:3] == [
        "improve-current-draft",
        ACTION_SAVE_PROMPT_DRAFT,
        "prompts",
    ]
    save = entries[1]
    assert save.label == "Save draft to shelf…"
    assert save.description == "Save the unsent message without sending it"

    save_callback = object()

    class _Screen:
        action_save_console_prompt_draft = save_callback

        def __getattr__(self, _name):
            return object()

    screen = _Screen()
    commands = ConsoleCommandProvider._commands(SimpleNamespace(), screen)
    matching = [
        command for command in commands if command[0] == "Console: Save draft to shelf…"
    ]
    assert matching == [
        (
            "Console: Save draft to shelf…",
            save_callback,
            "Save the unsent message without sending it",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "expected"),
    [
        ("console-prompt-draft-save-keep", "keep"),
        ("console-prompt-draft-save-clear", "clear"),
        ("console-prompt-draft-save-cancel", None),
    ],
)
async def test_save_dialog_returns_the_exact_keep_clear_or_cancel_choice(
    button_id, expected
):
    results = []
    app = App()

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(ConsolePromptDraftSaveDialog(), callback=results.append)
        await pilot.pause()
        await pilot.click(f"#{button_id}")
        await pilot.pause()

    assert results == [expected]


class _Composer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.fingerprint = "opening"
        self.clear_calls = 0

    def capture_draft_snapshot(self):
        return SimpleNamespace(
            fingerprint=self.fingerprint,
            segments=(SimpleNamespace(text=self.text),),
        )

    def clear_draft(self) -> None:
        self.clear_calls += 1
        self.text = ""
        self.fingerprint = "cleared"


class _Store:
    def __init__(self) -> None:
        self.active_session_id = "session-1"
        self.drafts = []

    def set_session_draft(self, session_id: str, text: str) -> None:
        self.drafts.append((session_id, text))


class _Scope:
    def __init__(self, *, outcome=None, after_create=None) -> None:
        self.outcome = outcome or {"draft_id": 7}
        self.after_create = after_create
        self.contents = []

    async def create_prompt_draft(self, *, content: str):
        self.contents.append(content)
        if self.after_create is not None:
            self.after_create()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _controller_double(*, composer, store, scope):
    notifications = []
    return SimpleNamespace(
        app_instance=SimpleNamespace(
            prompt_scope_service=scope,
            notify=lambda message, **kwargs: notifications.append((message, kwargs)),
        ),
        _console_composer_or_none=lambda: composer,
        _ensure_console_chat_store=lambda: store,
        _sync_console_command_popup=lambda: None,
        notifications=notifications,
    )


@pytest.mark.asyncio
async def test_save_and_clear_persists_exact_snapshot_before_clearing_session_draft():
    composer = _Composer("first\nfull collapsed paste body\n")
    store = _Store()
    scope = _Scope()
    controller = _controller_double(composer=composer, store=store, scope=scope)

    outcome = await ConsolePromptsController._save_current_prompt_draft(
        controller, clear_after_save=True
    )

    assert outcome == "saved-cleared"
    assert scope.contents == ["first\nfull collapsed paste body\n"]
    assert composer.clear_calls == 1
    assert store.drafts == [("session-1", "")]


@pytest.mark.asyncio
async def test_save_uses_full_canonical_text_behind_a_real_collapsed_paste():
    composer = ConsoleComposerBar(paste_collapse_threshold=20)
    pasted = "hidden paste payload " * 8
    composer.insert_text_as_paste(pasted)
    snapshot = composer.capture_draft_snapshot()
    assert snapshot.segments[0].collapse_state == "collapsed"
    assert composer._display_draft_text() != pasted

    store = _Store()
    scope = _Scope()
    controller = _controller_double(composer=composer, store=store, scope=scope)

    outcome = await ConsolePromptsController._save_current_prompt_draft(
        controller, clear_after_save=False
    )

    assert outcome == "saved-kept"
    assert scope.contents == [pasted]
    assert composer.draft_text() == pasted


@pytest.mark.asyncio
async def test_save_failure_and_concurrent_edit_both_preserve_the_composer():
    composer = _Composer("opening text")
    store = _Store()
    full_scope = _Scope(outcome=PromptDraftShelfFullError())
    full = _controller_double(composer=composer, store=store, scope=full_scope)

    full_outcome = await ConsolePromptsController._save_current_prompt_draft(
        full, clear_after_save=True
    )

    assert full_outcome == "shelf-full"
    assert composer.text == "opening text"
    assert composer.clear_calls == 0
    assert store.drafts == []
    assert "100 of 100" in full.notifications[-1][0]

    def edit_while_saving() -> None:
        composer.text = "newer text"
        composer.fingerprint = "newer"

    edited_scope = _Scope(after_create=edit_while_saving)
    edited = _controller_double(composer=composer, store=store, scope=edited_scope)

    edited_outcome = await ConsolePromptsController._save_current_prompt_draft(
        edited, clear_after_save=True
    )

    assert edited_outcome == "saved-kept-newer"
    assert composer.text == "newer text"
    assert composer.clear_calls == 0
    assert store.drafts == []


class _DraftAdapterScope:
    def __init__(self) -> None:
        self.calls = []

    async def list_prompt_drafts(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"items": [], "page": 1, "total_pages": 1, "total_items": 0}

    async def get_prompt_draft(self, **kwargs):
        self.calls.append(("detail", kwargs))
        return {"draft_id": kwargs["draft_id"]}

    async def update_prompt_draft(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"draft_id": kwargs["draft_id"], "version": 2}

    async def delete_prompt_draft(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return True

    async def list_prompt_collections(self, **kwargs):
        self.calls.append(("collections", kwargs))
        return {"collections": []}

    async def replace_prompt_collection_memberships(self, **kwargs):
        self.calls.append(("memberships", kwargs))
        return kwargs


@pytest.mark.asyncio
async def test_console_source_routes_draft_shelf_only_to_local_draft_contracts():
    scope = _DraftAdapterScope()
    source = _ConsolePromptSource(scope)

    capabilities = await source.capabilities("draft_shelf")
    await source.list_page("draft_shelf", 2)
    await source.search("draft_shelf", "needle")
    await source.detail("draft_shelf", "draft_shelf:7")
    await source.update_draft(draft_id=7, content="edited", expected_version=1)
    await source.delete_draft(draft_id=7, expected_version=2)
    await source.list_draft_collections()
    await source.assign_draft_collection(prompt_id=41, collection_ids=(5,))

    assert capabilities.conditional_update is True
    assert scope.calls == [
        ("list", {"mode": "local", "page": 2, "per_page": 10}),
        (
            "list",
            {"mode": "local", "query": "needle", "page": 1, "per_page": 25},
        ),
        ("detail", {"mode": "local", "draft_id": 7}),
        (
            "update",
            {
                "mode": "local",
                "draft_id": 7,
                "content": "edited",
                "expected_version": 1,
            },
        ),
        ("delete", {"mode": "local", "draft_id": 7, "expected_version": 2}),
        ("collections", {"mode": "local", "limit": 100, "offset": 0}),
        (
            "memberships",
            {"mode": "local", "prompt_id": 41, "collection_ids": (5,)},
        ),
    ]


@pytest.mark.asyncio
async def test_draft_promotion_collection_choices_include_every_bounded_page():
    class _PagedCollectionsScope:
        def __init__(self) -> None:
            self.offsets = []

        async def list_prompt_collections(self, **kwargs):
            offset = kwargs["offset"]
            limit = kwargs["limit"]
            self.offsets.append(offset)
            items = [
                {
                    "id": f"local:prompt_collection:{index}",
                    "backend": "local",
                    "collection_id": index,
                    "name": f"Collection {index:03d}",
                    "display_name": f"Collection {index:03d}",
                    "description": None,
                    "prompt_ids": [],
                }
                for index in range(offset + 1, min(offset + limit, 205) + 1)
            ]
            return {
                "collections": items,
                "limit": limit,
                "offset": offset,
                "total": 205,
            }

    scope = _PagedCollectionsScope()
    result = await _ConsolePromptSource(scope).list_draft_collections()

    assert scope.offsets == [0, 100, 200]
    assert len(result["collections"]) == 205
    assert result["collections"][0]["collection_id"] == 1
    assert result["collections"][-1]["collection_id"] == 205


def test_controller_inserts_shelf_text_at_live_caret_without_replacing_draft():
    composer = ConsoleComposerBar()
    composer.load_draft("alpha omega")
    for _ in range(len("omega")):
        assert composer.move_cursor_left()
    store = _Store()
    popup_syncs = []
    controller = SimpleNamespace(
        _sync_console_command_popup=lambda: popup_syncs.append(True)
    )

    inserted = ConsolePromptsController._insert_prompt_draft_at_caret(
        controller,
        content="saved ",
        composer=composer,
        store=store,
        session_id="session-1",
    )

    assert inserted is True
    assert composer.draft_text() == "alpha saved omega"
    assert store.drafts == [("session-1", "alpha saved omega")]
    assert popup_syncs == [True]

    store.active_session_id = "session-2"
    assert (
        ConsolePromptsController._insert_prompt_draft_at_caret(
            controller,
            content="never",
            composer=composer,
            store=store,
            session_id="session-1",
        )
        is False
    )
    assert composer.draft_text() == "alpha saved omega"
