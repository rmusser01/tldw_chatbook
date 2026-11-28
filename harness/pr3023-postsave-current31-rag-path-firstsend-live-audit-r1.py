"""Read-only audit of a closed, owned matched live run; never accesses a key."""

import argparse
import hashlib
import json
import socket
import sqlite3
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=int, required=True, choices=range(1, 9))
args = parser.parse_args()
E = Path(__file__).resolve().parent
parent = E / "paired-postsave-current31-rag-path-firstsend-r1"
plan = json.loads((parent / "plan.json").read_text())
run = plan["runs"][args.run - 1]
out = parent / f"run{args.run}"
profile = Path(run["profile"])
owned = json.loads((out / "owned-processes-before.json").read_text())
processes = {
    int(s.split(None, 1)[0]): s.split(None, 1)[1]
    for s in subprocess.check_output(
        ["ps", "-axo", "pid=,comm="], text=True
    ).splitlines()
}
remaining = {
    pid: row
    for pid, row in owned["owned_observed"].items()
    if int(pid) in processes and processes[int(pid)] == row["comm"]
}
assert not remaining, remaining
with socket.socket() as check:
    assert check.connect_ex(("127.0.0.1", run["port"])) != 0
assert (profile / "app-returned.json").exists()
app = json.loads((profile / "app-start.json").read_text())
assert app["head"] == run["head"] and app["role"] == run["role"]
assert Path(app["db_path"]).is_relative_to(profile)
connection = sqlite3.connect(Path(app["db_path"]).as_uri() + "?mode=ro", uri=True)
try:

    def read(sql):
        return connection.execute(sql).fetchall()

    messages = read(
        "select id,role,content,assistant_generation_state,version from messages where deleted=0 order by timestamp,id"
    )
    traces = read(
        "select call_id,state,integrity_state,provider_name,model_name,dispatch_started_at,response_started_at,settled_at,provider_inactive_at from console_trace_calls order by created_at,call_sequence"
    )
    links = read(
        "select l.call_id,l.verification_outcome,r.source_message_id,r.live_message_id from console_trace_response_links l join console_trace_semantic_revisions r on r.revision_id=l.semantic_revision_id"
    )
    headers = read(
        "select provider_name,model_name,generation_parameters_json,reasoning_controls_json from console_trace_request_headers order by created_at,header_id"
    )
    checkpoints = read("select count(*) from console_dispatch_checkpoints")[0][0]
    tool_rows = read(
        "select c.header_id,c.ordinal,a.sanitized_bytes from console_trace_header_components c join console_trace_artifacts a on a.artifact_id=c.artifact_id where c.component_kind='tool_schema' order by c.header_id,c.ordinal"
    )
    tools = {}
    for header, ordinal, payload in tool_rows:
        tools.setdefault(header, []).append(
            {"ordinal": ordinal, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    assert (
        len(messages) == 2
        and len(traces) == len(links) == len(headers) == 1
        and checkpoints == 0
    )
    assert all(x[1:3] == ("complete", "complete") for x in traces)
    assert all(x[3:5] == ("deepseek", "deepseek-flash") for x in traces)
    assert len({x[0] for x in traces}) == 1 and {x[0] for x in traces} == {
        x[0] for x in links
    }
    assert {x[2] for x in links} == {x[0] for x in messages if x[1] == "assistant"}
    assert all(x[1] == "verified_equal" and x[2] == x[3] for x in links)
    assert [x[2] for x in messages if x[1] == "assistant"] == [
        "CEDAR READY",
    ]
    assert [x[2] for x in messages if x[1] == "user"] == plan["prompts"]
    assert len(tools) == 1 and all(len(rows) > 0 for rows in tools.values())
finally:
    connection.close()
http = [json.loads(line) for line in (profile / "http.jsonl").read_text().splitlines()]
assert [x["request"] for x in http if x["event"] == "send"] == [1]
assert all(x["status"] == 200 for x in http if x["event"] == "response_open")
assert {x["request"] for x in http if x["event"] == "response_closed"} == {1}
assert not any(x["event"] == "send_exception" for x in http)
root = Path(run["root"])
names = (
    subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ]
    )
    .decode()
    .split("\0")
)
actual = {
    n: hashlib.sha256((root / n).read_bytes()).hexdigest()
    for n in sorted(set(names) - {""})
    if n.endswith(".py") and (root / n).is_file()
}
assert actual == json.loads(
    (parent / (run["role"] + "-sources-before.json")).read_text()
)
assert (
    subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    == run["head"]
)
assert not subprocess.check_output(
    ["git", "-C", str(root), "status", "--porcelain"], text=True
).strip()
session = json.loads((out / "browser/session.json").read_text())
assert session["operation_errors"] == [] and [session["columns"], session["rows"]] == [
    120,
    35,
]
inputs = [
    action
    for action in session["actions"]
    if (action["action"] == "click_text" and action["text"] == "Send")
    or (action["action"] == "key" and action["key"] == "Enter")
]
assert len(inputs) == 1 and [x["action"] for x in inputs] == [
    "key",
]
ready_waits = [action for action in session["actions"] if action["action"] == "wait_text" and action.get("text") == "Send" and action.get("seconds") == 60]
assert len(ready_waits) == 2 and all("complete" in action and "failed" not in action for action in ready_waits)
typed = [action for action in session["actions"] if action["action"] == "type"]
assert [action["text"] for action in typed] == plan["prompts"]
assert all(action["dispatch_method"] == "Input.insertText" for action in typed)
samples = []
for number, action in enumerate(inputs, 1):
    records = {x["event"]: x for x in http if x["request"] == number}
    assert set(records) == {"send", "response_open", "response_closed"}
    start = action["pre_input"]["host_monotonic_ns"]
    samples.append(
        {
            "send": number,
            "first_app_send": number == 1,
            "input_to_post_seconds": (records["send"]["monotonic_ns"] - start) / 1e9,
            "input_to_http_close_seconds": (
                records["response_closed"]["monotonic_ns"] - start
            )
            / 1e9,
        }
    )
result = {
    "role": run["role"],
    "head": run["head"],
    "profile": str(profile),
    "geometry": [120, 35],
    "messages": messages,
    "traces": traces,
    "links": links,
    "headers": headers,
    "tool_schemas": tools,
    "pending_checkpoints": checkpoints,
    "http": http,
    "samples": samples,
    "app_entry_returned": True,
    "driver_exit": 0,
    "server_exit": 0,
    "observed_owned_identities_remaining": 0,
    "port_closed": True,
    "source_map_unchanged": True,
    "draft_input": "atomic CDP Input.insertText",
    "submit_input": "Enter for every send",
    "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "Tool exit codes are separately observed by Root; PID disappearance is observed only. Normal app return is not arbitrary physical native worker retirement. Input-to-POST is not an isolated adapter/setup span. Browser transport frames are not native OS paint.",
}
target = out / "closed-audit.json"
assert not target.exists()
target.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "run": args.run,
            "result": str(target),
            "samples": samples,
            "complete_traces": 1,
            "tool_counts": [len(x) for x in tools.values()],
            "pending_checkpoints": 0,
            "owned_identities_remaining": 0,
            "port_closed": True,
        }
    )
)
