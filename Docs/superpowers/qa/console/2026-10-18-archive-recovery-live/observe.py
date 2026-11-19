"""Read only synthetic live fixtures and production capture reconstruction."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

root = Path(__file__).parent
case = root / sys.argv[1]
phase = sys.argv[2]
for key, name in {"HOME": "home", "XDG_CONFIG_HOME": "xdg-config", "XDG_DATA_HOME": "xdg-data", "XDG_CACHE_HOME": "xdg-cache", "TLDW_CONFIG_PATH": "config.toml"}.items():
    os.environ[key] = str(case / name)
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
sys.path.insert(0, "/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook")
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

fixture = json.loads((case / "fixture.json").read_text())
result = {"conversations": [], "messages": [], "calls": [], "reconstructions": [], "user_revisions": [], "workspace_records": []}
with sqlite3.connect("file:" + fixture["paths"]["chat_db"] + "?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    for item in fixture["cases"]:
        result["conversations"].extend(dict(row) for row in cursor.execute("SELECT * FROM conversations WHERE id = ?", (item["conversation_id"],)))
        result["messages"].extend(dict(row) for row in cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp,id", (item["conversation_id"],)))
    result["calls"] = [dict(row) for row in cursor.execute("SELECT * FROM console_trace_calls ORDER BY call_sequence,call_id")]
    reader = ConsoleTraceNativeReader(db)
    for row in result["calls"]:
        call = reader.repository.get_call(cursor, row["call_id"])
        if call.surface_node_id is None or call.request_header_id is None:
            continue
        request, omissions = reader._reconstruct_request(cursor, call)
        tail = reader.repository.get_surface_node(cursor, call.surface_node_id)
        projection = reader.service._surface_projection(cursor, call.segment_id, tail)
        revisions = [key[2] for _, key in projection.entries if key[1] == "revision"]
        rows = request.get("messages_payload", [])
        source_pins = [row[0] for row in cursor.execute("SELECT semantic_revision_id FROM console_trace_events WHERE call_id=? AND event_type=?", (call.call_id, "call_boundary"))]
        result["reconstructions"].append({"call_id": call.call_id, "turn_id": call.turn_id, "sha256": hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "omissions": omissions, "pinned_revisions": revisions, "boundary_source_revisions": source_pins, "archive_handoffs": sum(isinstance(row, dict) and "Previously model-observed archive selection metadata." in str(row.get("content", "")) for row in rows)})
    result["user_revisions"] = [dict(row) for row in cursor.execute("SELECT revision_id,source_message_id,source_conversation_id,normalized_role FROM console_trace_semantic_revisions WHERE normalized_role=? ORDER BY created_at", ("user",))]
    fixture_ids = {item["conversation_id"] for item in fixture["cases"]}
    result["source_messages"] = [dict(row) for row in cursor.execute("SELECT id,conversation_id,parent_message_id,role,sender,content FROM messages ORDER BY timestamp,id") if row["conversation_id"] not in fixture_ids]
    result["conversation_count"] = cursor.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
with sqlite3.connect("file:" + fixture["paths"]["workspace_db"] + "?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    result["workspace_records"] = [dict(row) for row in db.execute("SELECT * FROM workspace_records ORDER BY workspace_id")]
(case / (phase + ".json")).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"phase": phase, "calls": len(result["calls"]), "states": [row["state"] for row in result["calls"]], "reconstructed": len(result["reconstructions"]), "user_revisions": len(result["user_revisions"])}))
