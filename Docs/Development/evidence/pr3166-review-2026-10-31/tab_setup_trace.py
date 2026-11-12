"""Original-body tab wait diagnostics; no scheduling or admission replacement."""
import asyncio
import json
import sys
import threading
import time


def pytest_runtest_setup(item):
    from Tests.UI.test_console_session_tab_close import _settle
    from Tests.UI.test_console_tab_publication_read_barrier import _shape, _source_code
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    monitor = sys.monitoring
    tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    reader_code = next(code for code in spend.ConsoleReadinessConfigProjection.for_screen.__func__.__code__.co_consts if getattr(code,"co_name",None) == "read_current")
    rows, starts = [], {}

    def observed(code, offset, value=None):
        frame = sys._getframe(1)
        if code is reader_code:
            if value is None:
                starts[id(frame)] = time.monotonic()
            else:
                rows.append({"reader_seconds":time.monotonic()-starts.pop(id(frame),time.monotonic()),"proof":getattr(value,"_display_proof",None) is not None,"thread":threading.get_ident()})
            return
        if value is not False:
            return
        caller = frame.f_back
        while caller is not None and "projection" not in caller.f_locals:
            caller = caller.f_back
        state={"settle_failed":True,"test_line":None if caller is None else caller.f_lineno}
        if caller is not None:
            projection=caller.f_locals["projection"]
            console=caller.f_locals["console"]
            state.update(pending=projection.pending, read_count=len(projection._reads), sync_running=console._console_sync_in_progress, sync_requested=console._console_sync_requested, replay=console._console_control_bar_replay_whole_sync)
            state["reads"]=[{"task_done":read.task.done(),"producer_done":read._producer is not None and read._producer.done(),"retired":read.retired.done(),"reader_original":read.callback is projection.read_current} for read in projection._reads]
            state["workers"]=[{"name":worker.name,"group":worker.group,"state":worker.state.name,"node":type(worker.node).__name__} for worker in console.workers if not worker.is_finished]
        stacks=[]
        for ident,top in sys._current_frames().items():
            frames=[]
            while top is not None:
                module_name=top.f_globals.get("__name__", "")
                if module_name.startswith("tldw_chatbook"):
                    module=sys.modules.get(module_name)
                    try:
                        original=_shape(top.f_code)==_shape(_source_code(module,top.f_code.co_qualname))
                    except (AttributeError,StopIteration,OSError):
                        original=False
                    frames.append({"module":module_name,"qualname":top.f_code.co_qualname,"line":top.f_lineno,"original_source":original})
                top=top.f_back
            if frames:
                stacks.append({"thread":ident,"frames":frames})
        state["native_stacks"]=stacks
        rows.append(state)
        print("TAB_SETUP_TRACE="+json.dumps(state),flush=True)
    monitor.use_tool_id(tool,"tab-original-setup-owner")
    monitor.register_callback(tool,monitor.events.PY_START,observed)
    monitor.register_callback(tool,monitor.events.PY_RETURN,observed)
    monitor.set_local_events(tool,reader_code,monitor.events.PY_START|monitor.events.PY_RETURN)
    monitor.set_local_events(tool,_settle.__code__,monitor.events.PY_RETURN)
    item._tab_setup_trace=(tool,reader_code,_settle.__code__,rows)


def pytest_runtest_teardown(item):
    tool,reader,settle,rows=item._tab_setup_trace
    for code in (reader,settle):
        sys.monitoring.set_local_events(tool,code,0)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,None)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
    sys.monitoring.free_tool_id(tool)
    print("TAB_READ_TRACE="+json.dumps(rows),flush=True)
