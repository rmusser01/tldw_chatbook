"""Subprocess-only fault instrumentation; never a product wire operation."""

import json
import os
from pathlib import Path
import threading


def run(mode):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP import permission_worker as worker
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery import admission

    path = Path(os.environ["TLDW_CONFIG_PATH"]).with_name("permission-fault.json")
    if mode == "command":
        retirement = process._Command.retirement

        def fail_after_effect(command):
            if command.source is not None and any(
                state.mcp_effects for _, state in command.raw
            ):
                hold = command.startups[0][1]
                hold.error = OSError("injected_retained_hold_error")
                path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "disk_effect": True,
                            "deferred_stamps": sum(
                                len(state.mcp_payload_updates)
                                for state in command.completions
                            ),
                            "hold_alive": hold.thread.is_alive(),
                        }
                    )
                )
            return retirement(command)

        process._Command.retirement = fail_after_effect
    elif mode == "shutdown":
        close = admission.os.close

        def failed_native_close(fd):
            if threading.current_thread().name == "chatbook-storage-admission" and any(
                hold.stop.is_set() for hold in storage._retiring_holds
            ):
                path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "native_fd": fd,
                            "native_close_attempted": True,
                            "thread": threading.current_thread().name,
                        }
                    )
                )
                raise OSError("injected_native_descriptor_close_failure")
            return close(fd)

        admission.os.close = failed_native_close
    else:
        raise ValueError(mode)
    raise SystemExit(worker.main())
