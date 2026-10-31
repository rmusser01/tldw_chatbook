from pathlib import Path
import asyncio, json, threading, time, sys, sqlite3
import pytest
from Tests.private_profile import private_profile_test

OUT = Path(__file__).resolve().parent


async def until(predicate, seconds=12):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["ordinary", "held_service"])
@private_profile_test
async def test_lore_save_queue(request, tmp_path, monkeypatch, scenario):
    from textual.widgets import Button, Input, TabbedContent, Static
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.roleplay_frame_harness import list_state, select_roleplay_item
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
        PersonasLoreDetailWidget,
    )

    app = _build_test_app()
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "lore.db", "audit-client")
    app.chachanotes_db = db
    manager = WorldBookManager(db)
    book_id = manager.create_world_book("Audit Lore")
    entered = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    calls = []
    original = WorldBookManager.update_world_book

    def observed(self, *args, **kwargs):
        calls.append(
            dict(
                thread=threading.get_ident(),
                at=time.monotonic(),
                book_id=args[0],
                name=kwargs.get("name"),
            )
        )
        entered.set()
        if scenario == "held_service":
            assert release.wait(20), "audit hold deadline"
        try:
            return original(self, *args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(WorldBookManager, "update_world_book", observed)
    host = DestinationHarness(app, "personas")
    record = dict(
        scenario=scenario,
        book_id=book_id,
        calls=calls,
        main_thread=threading.get_ident(),
    )
    blocker = None
    try:
        async with host.run_test(size=(220, 60)) as pilot:
            screen = host.screen_stack[-1]
            record["modules"] = {
                type(x).__name__: sys.modules[type(x).__module__].__file__
                for x in (screen, manager, db)
            }
            assert all(
                Path(p).is_relative_to(Path.cwd()) for p in record["modules"].values()
            )
            await until(lambda: bool(screen.query("#personas-mode-lore")))
            screen.query_one("#personas-mode-lore", Button).press()
            await until(
                lambda: screen.state.active_mode == "lore"
                and any(x.name == "Audit Lore" for x in list_state(screen).items)
            )
            await select_roleplay_item(pilot, screen, index=0)
            await until(
                lambda: screen.state.selected_entity_kind == "lore"
                and bool(screen.query(PersonasLoreDetailWidget))
            )
            await pilot.pause()
            detail = screen.query_one(PersonasLoreDetailWidget)
            detail.query_one(TabbedContent).active = "personas-lore-tab-settings"
            await pilot.pause()
            field = screen.query_one("#personas-lore-name", Input)
            field.focus()
            await pilot.press("end", "-", "n", "e", "w")
            await pilot.pause()
            record["submitted"] = field.value
            save = screen.query_one("#personas-lore-settings-save", Button)
            assert not save.disabled
            if scenario == "sqlite_busy":
                blocker = sqlite3.connect(db.db_path_str)
                blocker.execute("BEGIN IMMEDIATE")
            save.press()
            await until(entered.is_set)
            app_callback = []
            screen_callback = []
            host.call_later(lambda: app_callback.append(True))
            screen.call_later(lambda: screen_callback.append(True))
            if scenario == "ordinary":
                await until(finished.is_set)
                # The DB thread can finish before the handler's reloads.
                # Establish a completed screen operation for this control.
                await until(lambda: bool(screen_callback))
            rail = screen.query_one("#personas-library-rail-collapse", Button)
            record["rail_enabled"] = not rail.disabled
            rail.press()
            before = time.monotonic()
            await asyncio.sleep(0.6)
            record.update(
                window=time.monotonic() - before,
                app_callback=bool(app_callback),
                screen_callback=bool(screen_callback),
                rail_collapsed=screen._library_rail_collapsed,
                finished=finished.is_set(),
                save_label=str(save.label),
            )
            release.set()
            if blocker is not None:
                blocker.rollback()
                blocker.close()
                blocker = None
            await until(finished.is_set)
            await until(lambda: screen._library_rail_collapsed)
            await until(lambda: bool(screen_callback))
            record["after_release_collapsed"] = screen._library_rail_collapsed
            record["persisted"] = manager.get_world_book(book_id)["name"]
            await pilot.pause()
            if scenario == "held_service":
                assert record["app_callback"] and not record["screen_callback"]
                assert not record["rail_collapsed"] and not record["finished"]
                assert calls[0]["thread"] != record["main_thread"]
            else:
                assert record["rail_collapsed"] and record["screen_callback"]
            assert record["persisted"] == record["submitted"]
    finally:
        release.set()
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        db.close_connection()
        (OUT / (scenario + ".json")).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
