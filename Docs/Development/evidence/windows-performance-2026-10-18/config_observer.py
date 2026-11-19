"""Diagnostic-only original config-presentation return sites (no replacements)."""

from collections import Counter
import sys
import threading


def install(report, stage):
    from tldw_chatbook.UI.Console_Modules import config_sync, console_spend_projection

    functions = (
        config_sync.run_console_config_sync,
        config_sync.run_console_presentation_sync,
        console_spend_projection._checked_display_status,
        console_spend_projection._checked_pricing_current,
    )
    counts = Counter()
    main = threading.get_ident()
    monitor = sys.monitoring
    tool = next(i for i in range(1, 6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "config-presentation-diagnostic")

    def returned(code, offset, result):
        if threading.get_ident() != main:
            return
        line = next(
            (line for start, end, line in code.co_lines() if start <= offset < end),
            None,
        )
        counts[(stage(), code.co_name, line, str(result))] += 1

    monitor.register_callback(tool, monitor.events.PY_RETURN, returned)
    for function in functions:
        monitor.set_local_events(tool, function.__code__, monitor.events.PY_RETURN)

    def finish():
        for function in functions:
            monitor.set_local_events(tool, function.__code__, 0)
        monitor.register_callback(tool, monitor.events.PY_RETURN, None)
        monitor.free_tool_id(tool)
        report["config_presentation_returns"] = [
            [*key, count]
            for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
        ]

    return finish
