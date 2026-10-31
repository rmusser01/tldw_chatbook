"""Record actual cadence decisions and publication sites without replacing bodies."""
import asyncio
import json
import sys
import threading


def pytest_runtest_setup(item):
    from Tests.UI import test_console_config_sync_cadence as fixture
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monitor = sys.monitoring
    tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    rows = []
    own_task = None
    codes = {
        fixture.test_unchanged_poll_ticks_skip_the_operation_but_once_a_second_in_a_run.__code__: "test",
        fixture._quiesce.__code__: "quiet",
        config_sync._projection_inputs.__code__: "inputs",
        config_sync.publish_console_presentation.__code__: "publish",
        ChatScreen._sync_console_rail_and_controls.__code__: "project",
    }

    def record(code, offset, value=None):
        nonlocal own_task
        frame = sys._getframe(1)
        name = codes[code]
        task = asyncio.current_task()
        if name == "test":
            own_task = task
            return
        screen = frame.f_locals.get("screen", frame.f_locals.get("self"))
        if name == "quiet":
            screen = frame.f_locals["tick"].console
        if screen is None:
            return
        cadence = config_sync._cadence(screen)
        row = {"event": name, "own_task": task is own_task, "clock": config_sync._clock(), "age": config_sync._clock()-cadence.at, "requested": cadence.requested, "served": cadence.served, "published": cadence.published, "poll": config_sync._POLL_SYNC.get()}
        if name == "inputs":
            objects, values = value
            row["changed_values"] = [i for i,(a,b) in enumerate(zip(cadence.values or (),values)) if a != b]
            row["changed_owners"] = [i for i,(a,b) in enumerate(zip(cadence.objects,objects)) if a is not b]
            row["cold"] = cadence.values is None
            row["value_changes"] = [(i, repr(a), repr(b)) for i,(a,b) in enumerate(zip(cadence.values or (),values)) if a != b]
            row["owner_changes"] = [(i, id(a), id(b)) for i,(a,b) in enumerate(zip(cadence.objects,objects)) if a is not b]
            row["context_key"] = repr(getattr(getattr(screen, "_console_context_read_snapshot", None), "key", None))
            row["config_key"] = repr(getattr(getattr(screen, "_console_readiness_config_projection", None), "key", None))
            row["live_workers"] = [(worker.group,worker.state.name) for worker in screen.app.workers if worker.node is screen and not worker.is_finished]
        elif name == "quiet":
            row["calls"] = frame.f_locals["seen"].calls
        elif name == "publish":
            sites = []
            caller = frame.f_back
            while caller is not None and len(sites)<9:
                sites.append(caller.f_code.co_qualname)
                caller = caller.f_back
            row["sites"] = sites
            row["group"] = frame.f_locals.get("group")
        elif name == "project":
            row["thread"] = threading.get_ident()
            caller = frame.f_back
            while caller is not None:
                if caller.f_code is config_sync.project_console_tick.__code__:
                    row["decision"] = {key:repr(caller.f_locals.get(key)) for key in ("force","unchanged","requested","values")}
                    break
                caller = caller.f_back
        caller = frame
        while caller is not None:
            if caller.f_code.co_name.startswith("test_unchanged_poll"):
                row["test_line"] = caller.f_lineno
                row["tail"] = caller.f_locals.get("tail")
                seen = caller.f_locals.get("seen")
                row["calls"] = getattr(seen,"calls",None)
                break
            caller = caller.f_back
        rows.append(row)

    monitor.use_tool_id(tool,"cadence-actual-decisions")
    monitor.register_callback(tool,monitor.events.PY_START,record)
    monitor.register_callback(tool,monitor.events.PY_RETURN,record)
    for code,name in codes.items():
        monitor.set_local_events(tool,code,monitor.events.PY_RETURN if name in {"inputs","quiet"} else monitor.events.PY_START)
    item._cadence_trace=(tool,tuple(codes),rows)


def pytest_runtest_teardown(item):
    tool,codes,rows=item._cadence_trace
    for code in codes:
        sys.monitoring.set_local_events(tool,code,0)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,None)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
    sys.monitoring.free_tool_id(tool)
    print("CADENCE_TRACE="+json.dumps(rows),flush=True)
