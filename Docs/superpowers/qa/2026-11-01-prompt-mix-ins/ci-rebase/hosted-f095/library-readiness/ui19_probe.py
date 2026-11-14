import inspect,json,pathlib,time
import pytest
OUT=pathlib.Path(__file__).parent/'predicates.jsonl'
@pytest.fixture(autouse=True)
def record_library_readiness(monkeypatch):
    from Tests.UI import library_note_links_support as support
    original=support._until
    async def observed(pilot,predicate,what,*args,**kwargs):
        if what!='the healthy Library source DOM':
            return await original(pilot,predicate,what,*args,**kwargs)
        screen=inspect.getclosurevars(predicate).nonlocals['screen'];previous=None
        def read():
            nonlocal previous
            result=predicate()
            rows=list(screen.query('#library-row-browse-notes'))
            row=rows[0] if rows else None
            state={'result':result,'loaded':screen._library_loaded,'lookup_error_type':type(screen._library_lookup_error).__name__ if screen._library_lookup_error else None,'rendered_generation':screen._library_snapshot_rendered_generation,'state_generation':screen._library_snapshot_state_generation,'entry_dirty':screen._library_entry_reconcile_dirty,'canvas_error':bool(screen.query('#library-canvas-error')),'row_count':len(rows),'parent_active':row.is_parent_active if row else None,'row_mounted':row.is_mounted if row else None,'row_display':row.display if row else None,'row_disabled':row.disabled if row else None,'screen_current':screen.app.screen is screen,'route':getattr(screen,'id',None),'parents':[(type(a).__name__,a.id,bool(a.display),a.is_mounted if isinstance(a.is_mounted,bool) else "app-method") for a in row.ancestors] if row else []}
            if state!=previous:
                with OUT.open('a') as f:f.write(json.dumps({'monotonic':time.monotonic(),**state})+'\n')
                previous=state
            return result
        return await original(pilot,read,what,*args,**kwargs)
    monkeypatch.setattr(support,'_until',observed)
