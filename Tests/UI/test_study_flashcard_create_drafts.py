"""Pending Flashcards creates acknowledge only the submitted raw fields.

Controlled awaits delegate successful creates to real Study services and SQLite.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import LocalStudyService, StudyScopeService
from tldw_chatbook.UI.Study_Window import StudyWindow


class _HeldCreate:
    def __init__(self, delegate, verb):
        self.delegate = delegate
        self.verb = verb
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def _create(self, verb, kwargs):
        if verb == self.verb:
            self.entered.set()
            await self.release.wait()
            if self.fail:
                # Definite pre-write failure, never an uncertain committed write.
                raise OSError("Create backend unavailable")
        return await getattr(self.delegate, verb)(**kwargs)

    async def create_deck(self, **kwargs):
        return await self._create("create_deck", kwargs)

    async def create_flashcard(self, **kwargs):
        return await self._create("create_flashcard", kwargs)


class _StudyHost(ConsolidatedCSSApp):
    def __init__(self, scope):
        super().__init__()
        self.instance = SimpleNamespace(
            study_scope_service=scope,
            study_quiz_scope_service=None,
            current_runtime_backend="local",
            runtime_backend=None,
            app_config={},
            notify=lambda *args, **kwargs: None,
        )

    def compose(self):
        yield StudyWindow(self.instance)


async def _settle(app, pilot):
    async with asyncio.timeout(8):
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()


@asynccontextmanager
async def _mounted_form(verb):
    db = CharactersRAGDB(db_path=":memory:", client_id="flashcard-create-drafts")
    deck_id = db.create_deck("Existing deck")
    scope = _HeldCreate(
        StudyScopeService(local_service=LocalStudyService(db=db), server_service=None),
        verb,
    )
    app = _StudyHost(scope)
    try:
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            app.query_one("#view-flashcards-btn", Button).press()
            await _settle(app, pilot)
            window = app.query_one(StudyWindow)
            if verb == "create_deck":
                fields = (window.query_one("#new-deck-name-input", Input),)
                values = ("  Submitted deck  ",)
                button = window.query_one("#create-deck-button", Button)
            else:
                window.query_one("#deck-select", Select).value = str(deck_id)
                await _settle(app, pilot)
                fields = (
                    window.query_one("#card-front", TextArea),
                    window.query_one("#card-back", TextArea),
                    window.query_one("#card-tags", Input),
                )
                values = (
                    "  Submitted front  ",
                    "  Submitted back  ",
                    "  first-tag   second-tag  ",
                )
                button = window.query_one("#create-card-btn", Button)
            yield app, pilot, db, deck_id, scope, fields, values, button
    finally:
        scope.release.set()
        db.close()


def _seed(fields, values):
    for field, value in zip(fields, values):
        if isinstance(field, TextArea):
            field.load_text(value)
        else:
            field.value = value


def _draft(fields):
    return tuple(
        field.text if isinstance(field, TextArea) else field.value for field in fields
    )


def _stored(db, deck_id, verb):
    if verb == "create_deck":
        return [(deck["name"],) for deck in db.list_decks() if deck["id"] != deck_id]
    return [
        (card["front"], card["back"], card["tags"])
        for card in db.list_flashcards(deck_id=deck_id)
    ]


async def _start(button, scope):
    scope.entered.clear()
    scope.release.clear()
    assert not button.disabled
    button.press()
    async with asyncio.timeout(8):
        await scope.entered.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["create_deck", "create_flashcard"])
@private_profile_test
async def test_create_preserves_newer_raw_fields_and_editing_context(request, verb):
    """Unconditional or normalized comparisons would erase these live drafts."""
    async with _mounted_form(verb) as state:
        app, pilot, db, deck_id, scope, fields, values, button = state
        submitted = (
            ("Submitted deck",)
            if verb == "create_deck"
            else ("Submitted front", "Submitted back", "first-tag second-tag")
        )
        cases = [((i,), "-new", False) for i in range(len(fields))]
        if len(fields) > 1:
            cases.append(((0, 1, 2), "-new", False))
        cases.extend(((i,), " ", False) for i in range(len(fields)))
        cases.append((tuple(range(len(fields))), "-new", True))
        for count, (edited, suffix, move_focus) in enumerate(cases, 1):
            if verb == "create_deck":
                values = (f"  Submitted deck {count}  ",)
            _seed(fields, values)
            await pilot.pause()
            await _start(button, scope)
            assert all(not field.disabled for field in fields)
            for index in edited:
                fields[index].focus()
                await pilot.press("end", *suffix)
            await pilot.press("shift+left")
            await pilot.pause()
            before = _draft(fields)
            assert before == tuple(
                value + suffix if i in edited else value
                for i, value in enumerate(values)
            )
            # Keep every edited field's caret/selection, including off-focus ones.
            selections = {i: fields[i].selection for i in edited}
            focused = fields[edited[-1]]
            if move_focus:
                focused = app.query_one("#flashcard-search-input", Input)
                focused.focus()
                await pilot.pause()
            assert app.focused is focused
            scope.release.set()
            await _settle(app, pilot)
            expected_records = (
                [(f"Submitted deck {i}",) for i in range(1, count + 1)]
                if verb == "create_deck"
                else [submitted] * count
            )
            assert _stored(db, deck_id, verb) == expected_records
            assert _draft(fields) == tuple(
                value if i in edited else "" for i, value in enumerate(before)
            )
            assert all(app.query_one("#" + field.id) is field for field in fields)
            assert app.focused is focused
            assert {i: fields[i].selection for i in edited} == selections


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["create_deck", "create_flashcard"])
@private_profile_test
async def test_unchanged_create_clears_once_and_definite_failure_can_retry(
    request, verb
):
    """Removing clearing or clearing on a failed pre-write attempt breaks this."""
    async with _mounted_form(verb) as state:
        app, pilot, db, deck_id, scope, fields, values, button = state
        submitted = (
            ("Submitted deck",)
            if verb == "create_deck"
            else ("Submitted front", "Submitted back", "first-tag second-tag")
        )
        _seed(fields, values)
        await pilot.pause()
        await _start(button, scope)
        scope.release.set()
        await _settle(app, pilot)
        assert _stored(db, deck_id, verb) == [submitted]
        assert _draft(fields) == tuple("" for _ in fields)

        _seed(fields, values)
        await pilot.pause()
        scope.fail = True
        await _start(button, scope)
        for field in fields:
            field.focus()
            await pilot.press("end", *"-retry")
        await pilot.press("shift+left")
        await pilot.pause()
        retry_draft = _draft(fields)
        selections = tuple(field.selection for field in fields)
        scope.release.set()
        await _settle(app, pilot)
        assert _stored(db, deck_id, verb) == [submitted]
        assert _draft(fields) == retry_draft
        assert app.focused is fields[-1]
        assert tuple(field.selection for field in fields) == selections

        scope.fail = False
        await _start(button, scope)
        scope.release.set()
        await _settle(app, pilot)
        retried = (
            ("Submitted deck  -retry",)
            if verb == "create_deck"
            else (
                "Submitted front  -retry",
                "Submitted back  -retry",
                "first-tag second-tag -retry",
            )
        )
        assert sorted(_stored(db, deck_id, verb)) == sorted([submitted, retried])
        assert _draft(fields) == tuple("" for _ in fields)
