"""Original-body tab wait diagnostics; no scheduling or admission replacement."""
import asyncio
import json
import sys
import threading
import time


def pytest_runtest_setup(item):
    from Tests.UI.test_console_session_tab_close import _settle
    from Tests.UI.test_console_tab_publication_read_barrier import _shape, _source_code, _event_within
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    monitor = sys.monitoring
    tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    reader_code = next(code for code in spend.ConsoleReadinessConfigProjection.for_screen.__func__.__code__.co_consts if getattr(code,"co_name",None) == "read_current")
    rows, starts = [], {}
    dumps = set()

    def held_tasks(gate):
        state = {"held_tasks": True, "gate": gate.reader_diagnostic(), "tasks": []}
        for task in asyncio.all_tasks():
            chain, current = [], task.get_coro()
            for _ in range(30):
                frame = getattr(current, "cr_frame", getattr(current, "gi_frame", None))
                if frame is not None:
                    owner = frame.f_locals.get("self")
                    row = {"function": frame.f_code.co_qualname, "line": frame.f_lineno, "module": frame.f_globals.get("__name__"), "owner": type(owner).__name__}
                    if row["module"].startswith("tldw_chatbook"):
                        try:
                            row["original_source"] = _shape(frame.f_code) == _shape(_source_code(sys.modules[row["module"]], frame.f_code.co_qualname))
                        except (AttributeError, StopIteration, OSError):
                            row["original_source"] = False
                    chain.append(row)
                current = getattr(current, "cr_await", getattr(current, "gi_yieldfrom", None))
                if current is None:
                    break
            if chain and any(row["module"].startswith("tldw_chatbook") or row["function"] in {"_click", "Pilot._wait_for_screen"} for row in chain):
                state["tasks"].append({"name": task.get_name(), "chain": chain})
        print("TAB_HELD_TASKS=" + json.dumps(state), flush=True)

    def observed(code, offset, value=None):
        frame = sys._getframe(1)
        if code is _event_within.__code__:
            caller = frame.f_back
            gate = None if caller is None else caller.f_locals.get("gate")
            if gate is not None and frame.f_locals.get("event") is gate.entered_async and id(gate) not in dumps:
                dumps.add(id(gate))
                asyncio.get_running_loop().call_later(3, held_tasks, gate)
            return
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
        state={"settle_failed":True,"test_line":None if caller is None else caller.f_lineno,"loop_type":type(asyncio.get_running_loop()).__name__,"loop_clock_resolution":getattr(asyncio.get_running_loop(),"_clock_resolution",None)}
        if caller is not None:
            projection=caller.f_locals["projection"]
            console=caller.f_locals["console"]
            state.update(pending=projection.pending, read_count=len(projection._reads), sync_running=console._console_sync_in_progress, sync_requested=console._console_sync_requested, replay=getattr(console,"_console_control_bar_replay_whole_sync",False))
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
    monitor.set_local_events(tool,_event_within.__code__,monitor.events.PY_RETURN)
    item._tab_setup_trace=(tool,reader_code,_settle.__code__,_event_within.__code__,rows)


def pytest_runtest_teardown(item):
    tool,reader,settle,event,rows=item._tab_setup_trace
    for code in (reader,settle,event):
        sys.monitoring.set_local_events(tool,code,0)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,None)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
    sys.monitoring.free_tool_id(tool)
    print("TAB_READ_TRACE="+json.dumps(rows),flush=True)
