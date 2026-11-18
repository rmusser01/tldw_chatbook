"""Run a frozen candidate's extra Windows writer under the original network guard."""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4


def guarded_serve(profile):
    from task40608_lab_observer import _original_guard, _settings

    directory = Path(os.environ["TLDW_UAT_ARTIFACTS"]) / "writer-guards"
    directory.mkdir(exist_ok=True)
    token = uuid4().hex
    start = {"pid": os.getpid(), "token": token}
    (directory / (token + ".start")).write_text(json.dumps(start), encoding="utf-8")
    executed = False

    def first_command(code, offset):
        nonlocal executed
        executed = True
        return sys.monitoring.DISABLE

    # A writer started before the fixture's server has no TCP destination.
    # The original guard permits OS socketpair IPC and checks both connect APIs.
    port = int(os.environ.get("TLDW_TEST_LAB_PORT", "-1"))
    with _original_guard(_settings())(port) as blocked:
        from tldw_chatbook.DB.profile_writer import _serve
        from tldw_chatbook.DB.profile_writer_commands import execute_profile_writer_command

        monitor = sys.monitoring
        slot = next(i for i in range(6) if monitor.get_tool(i) is None)
        monitor.use_tool_id(slot, "task40608-writer-guard")
        try:
            monitor.register_callback(slot, monitor.events.PY_START, first_command)
            monitor.set_local_events(slot, execute_profile_writer_command.__code__, monitor.events.PY_START)
            status = _serve(profile)
        finally:
            monitor.free_tool_id(slot)
    end = {**start, "status": status, "blocked": len(blocked), "executed": executed}
    (directory / (token + ".end")).write_text(json.dumps(end), encoding="utf-8")
    return status


def verify_writer_guards(run):
    """Require every started writer to have exited cleanly and one to have served."""
    directory = Path(run) / "writer-guards"
    starts = {p.stem: json.loads(p.read_bytes()) for p in directory.glob("*.start")}
    ends = {p.stem: json.loads(p.read_bytes()) for p in directory.glob("*.end")}
    if not starts or starts.keys() != ends.keys():
        raise ValueError("Missing or incomplete profile writer network guards")
    for key, end in ends.items():
        if (
            starts[key] != {"pid": end["pid"], "token": key} or end["token"] != key
            or type(end["status"]) is not int or end["status"] != 0
            or end["blocked"] != 0 or type(end["executed"]) is not bool
        ):
            raise ValueError("Unverified profile writer network guard")
    served = sum(end["executed"] for end in ends.values())
    if not served:
        raise ValueError("No guarded profile writer received a domain command")
    return {"started": len(starts), "retired": len(ends), "served": served, "blocked_destinations": 0}
