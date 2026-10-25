from pathlib import Path
import asyncio, json, threading, time, sys
import pytest
from Tests.private_profile import private_profile_test

OUT = Path(__file__).resolve().parent

async def until(predicate, seconds=12):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(.01)

@pytest.mark.asyncio
@pytest.mark.parametrize('scenario', ['ordinary', 'held_selection', 'held_section'])
@private_profile_test
async def test_watchlists_reader_queue(request, tmp_path, monkeypatch, scenario):
    from textual.widgets import ListView, Static, Button
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    app = _build_test_app()
    service = app.local_watchlists_service
    db = service._db()
    source = db.add_subscription(name='Audit feed', type='rss', source='https://audit.invalid/feed')
    ids = []
    with db.transaction() as cursor:
        for letter, day in [('A', '24'), ('B', '23'), ('C', '22')]:
            ids.append(persist_subscription_item(cursor, source, dict(title='Article '+letter, url='https://audit.invalid/'+letter, content='Body '+letter, content_hash=letter, content_kind='article', content_format='text', published_date=f'2026-10-{day}T10:00:00+00:00'), None, '2026-10-25T10:00:00+00:00'))
    release, entered = threading.Event(), threading.Event()
    calls = []
    main_thread = threading.get_ident()
    original = db.get_item_content
    def held_read(item_id):
        calls.append(dict(item_id=item_id, thread=threading.get_ident(), at=time.monotonic()))
        if scenario != 'ordinary' and int(item_id) == ids[1]:
            entered.set()
            assert release.wait(20), 'audit hold safety deadline reached'
        return original(item_id)
    monkeypatch.setattr(db, 'get_item_content', held_read)
    host = DestinationHarness(app, 'watchlists_collections')
    record = dict(scenario=scenario, ids=ids, calls=calls, main_thread=main_thread)
    pending=[]
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            try:
                screen=host.screen_stack[-1]
                record['production_modules']={kind.__name__:sys.modules[kind.__module__].__file__ for kind in (type(screen),type(service),type(db),ArticleListPane,ContentPane)}
                assert all(Path(path).is_relative_to(Path.cwd()) for path in record['production_modules'].values())
                await until(lambda: bool(screen.query('#watchlists-items-pane')) and len(screen.query_one('#watchlists-items-pane', ArticleListPane).items) == 3)
                await pilot.pause()
                pane=screen.query_one('#watchlists-items-pane', ArticleListPane)
                record['page_ids']=[x['id'] for x in pane.items]
                assert all(row.get('content') is None for row in pane.items), 'list projection must stay narrow'
                pane.focus_first_row_without_selecting()
                await pilot.press('enter')
                await until(lambda: screen._selected_content_item is not None)
                assert screen._selected_content_item['title']=='Article A'
                record['initial_body']=str(screen.query_one('#content-body', Static).renderable)
                assert 'Body A' in record['initial_body']
                # Actual Pilot key creates the production ListView/ItemSelected path.
                first=asyncio.create_task(pilot.press('down'));pending.append(first)
                if scenario == 'ordinary':
                    await first
                    await until(lambda: screen._selected_content_item['title']=='Article B')
                    await pilot.press('down')
                    await until(lambda: screen._selected_content_item['title']=='Article C')
                    await pilot.pause()
                    record['final_body']=str(screen.query_one('#content-body', Static).renderable)
                    assert 'Body C' in record['final_body']
                    button=screen.query_one('#wl-tab-sources', Button)
                    assert not button.disabled
                    button.press()
                    await until(lambda: screen.active_section=='sources')
                    record['final_section']=screen.active_section
                else:
                    await until(entered.is_set)
                    record['held_reader_title']=screen._selected_content_item['title']
                    record['held_pane_title']=pane.selected_item['title']
                    app_callback=[];screen_callback=[]
                    host.call_later(lambda: app_callback.append(True))
                    screen.call_later(lambda: screen_callback.append(True))
                    if scenario=='held_selection':
                        # Public production widget selection posts the real ItemSelected
                        # event while the screen's older event is awaiting its real DB read.
                        pane.select_item_by_id(str(pane.items[2]['id']))
                        record['requested_title']=pane.selected_item['title']
                    else:
                        button=screen.query_one('#wl-tab-sources', Button)
                        record['button_enabled']=not button.disabled
                        button.press()
                    started=time.monotonic()
                    await asyncio.sleep(.6)
                    record['held_window_seconds']=time.monotonic()-started
                    record['app_callback_ran']=bool(app_callback)
                    record['screen_callback_ran']=bool(screen_callback)
                    record['before_release_title']=screen._selected_content_item['title']
                    record['before_release_section']=screen.active_section
                    record['before_release_calls']=[c['item_id'] for c in calls]
                    record['hold_released']=release.is_set()
                    assert record['app_callback_ran'] and not record['screen_callback_ran']
                    assert record['before_release_title']=='Article A'
                    assert record['before_release_section']=='items'
                    assert len(calls)==2 and calls[-1]['thread']!=main_thread
                    release.set()
                    await asyncio.gather(*pending)
                    if scenario=='held_selection':
                        await until(lambda: screen._selected_content_item['title']=='Article C')
                        await pilot.pause()
                        record['final_body']=str(screen.query_one('#content-body', Static).renderable)
                        assert 'Body C' in record['final_body']
                    else:
                        await until(lambda: screen.active_section=='sources')
                    record['after_release_section']=screen.active_section
                    record['after_release_title']=screen._selected_content_item['title'] if screen._selected_content_item else None
                    await until(lambda: bool(screen_callback))
                    record['screen_callback_after_release']=bool(screen_callback)
            finally:
                release.set()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                await pilot.pause()
    finally:
        release.set()
        (OUT/(scenario+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
