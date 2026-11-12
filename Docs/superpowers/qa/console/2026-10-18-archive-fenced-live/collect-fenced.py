"""Collect bounded synthetic handoff and actual no-ID dispatch evidence read-only."""

import json
import os
import sqlite3
import sys
from pathlib import Path

root = Path("/private/tmp/archive-recovery-profilev2-live-20261018")
assert sys.argv[1] in {"direct-fenced", "rag-fenced"}
case = root / sys.argv[1]
assert (case / "config.toml").is_file()
for key, name in {
    "HOME": "home",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
    "XDG_CACHE_HOME": "xdg-cache",
    "TLDW_CONFIG_PATH": "config.toml",
}.items():
    os.environ[key] = str(case / name)
os.environ.update(
    PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
sys.path.insert(0, str(root))
from transport_guard import install

install(root, consumed=22, cap=30)
sys.path.insert(
    0, "/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook"
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

fixture = json.loads((case / "fixture.json").read_text())
final = json.loads((case / "final.json").read_text())
user = next(
    m
    for m in final["source_messages"]
    if m["role"] == "user" and m["content"] == "Restore the second one."
)
first = min(
    (c for c in final["calls"] if c["turn_id"] == user["id"]),
    key=lambda c: c["call_sequence"],
)
with sqlite3.connect(
    "file:" + fixture["paths"]["chat_db"] + "?mode=ro", uri=True
) as db:
    reader = ConsoleTraceNativeReader(db)
    cursor = db.cursor()
    call = reader.repository.get_call(cursor, first["call_id"])
    request, omissions = reader._reconstruct_request(cursor, call)
    assert not omissions
    rows = [
        row
        for row in request["messages_payload"]
        if isinstance(row.get("content"), str)
        and row["content"].startswith(
            "Evidence: Previously model-observed archive selection metadata."
        )
    ]
    assert len(rows) == 1
    body = rows[0]["content"]
    metadata, end = json.JSONDecoder().raw_decode(body.partition("\n")[2])
    assert body.partition("\n")[2][end:] == "\n\n---\n\nRestore the second one."
    assert len(body.encode()) <= 32768
with sqlite3.connect(
    "file:" + str(Path(fixture["paths"]["user_data"]) / "agent_runs.db") + "?mode=ro",
    uri=True,
) as db:
    steps = [
        (run, seq, json.loads(payload))
        for run, seq, payload in db.execute(
            "SELECT run_id,seq,payload FROM agent_run_steps ORDER BY created_at,seq"
        )
    ]
fences = []
for run, seq, step in steps:
    text = step.get("summary", "").strip()
    if (
        step.get("kind") == "model"
        and text.startswith("```tool_call\n")
        and text.endswith("\n```")
    ):
        payload = json.loads(text[len("```tool_call\n") : -len("\n```")])
        if payload.get("name") == "request_conversation_restore":
            fences.append({"run_id": run, "seq": seq, "payload": payload})
assert len(fences) == 1
fence = fences[0]
assert "call_id" not in fence["payload"]
items = metadata["archive_recovery_results"][0]["items"]
assert fence["payload"]["arguments"] == {
    "result_ref": items[1]["result_ref"],
    "scope": "workspace_only",
}
selected = [
    {
        "run_id": run,
        "seq": seq,
        **{
            k: step.get(k)
            for k in ("kind", "tool_name", "call_id", "status", "tool_outcome")
        },
    }
    for run, seq, step in steps
    if run == fence["run_id"]
    and step.get("tool_name") == "request_conversation_restore"
]
assert {s["call_id"] for s in selected} == {"turn-1-call-0"}
assert {"tool_proposed", "tool_execution_started", "tool_succeeded", "tool_result"} <= {
    s["kind"] for s in selected
}
assert all(
    s["tool_outcome"] == "success"
    for s in selected
    if s["kind"] in {"tool_succeeded", "tool_result"}
)
output = {
    "call_id": first["call_id"],
    "current_user_message": user["id"],
    "metadata": metadata,
    "wrapper_bytes": len(body.encode()),
    "actual_restore_fence": fence,
    "selected_steps": selected,
    "read_only": True,
}
(case / "handoff-proof.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        {
            "case": case.name,
            "handoffs": 1,
            "actual_no_id_fence": True,
            "selected_ordinal": 2,
            "scope": "workspace_only",
            "dispatch_identity": "turn-1-call-0",
            "tool_outcome": "success",
        }
    )
)
