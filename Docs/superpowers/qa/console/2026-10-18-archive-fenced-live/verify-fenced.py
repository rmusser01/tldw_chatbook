"""Verify retained synthetic native observations; no app, DB writes or network."""

import json
from pathlib import Path

root = Path(__file__).parent
results = {}
for name in ("direct-fenced", "rag-fenced"):
    case = root / name
    load = lambda stem, case=case: json.loads((case / (stem + ".json")).read_text())
    baseline, search, pending, final = map(
        load, ("baseline", "after-search", "before-confirm", "final")
    )
    for kind in ("conversations", "messages"):
        assert baseline[kind] == search[kind] == pending[kind] == final[kind], kind
    assert (
        baseline["workspace_records"]
        == search["workspace_records"]
        == pending["workspace_records"]
    )
    changes = []
    for before, after in zip(baseline["workspace_records"], final["workspace_records"]):
        diff = {key for key in before if before[key] != after[key]}
        if diff == {"updated_at"}:
            changes.append({"navigation_timestamp_only": before["workspace_id"]})
        elif diff:
            assert before["workspace_id"] == after["workspace_id"] == "atlas"
            assert diff == {"archived", "archived_at", "archive_revision", "updated_at"}
            assert (
                before["archived"] == 1
                and after["archived"] == 0
                and after["archived_at"] is None
            )
            assert after["archive_revision"] == before["archive_revision"] + 1
            changes.append({"restored_workspace": "atlas"})
    assert sum(x.get("restored_workspace") == "atlas" for x in changes) == 1
    assert (
        baseline["conversation_count"] == 3
        and search["conversation_count"] == final["conversation_count"] == 4
    )
    for kind, key in (
        ("calls", "call_id"),
        ("reconstructions", "call_id"),
        ("responses", "call_id"),
        ("thinking_structure", "message_id"),
    ):
        end = {row[key]: row for row in final[kind]}
        assert all(end[row[key]] == row for row in search[kind]), kind
    assert len(final["calls"]) == 4 and all(
        c["state"] == "complete" for c in final["calls"]
    )
    users = [m for m in final["source_messages"] if m["role"] == "user"]
    assert [m["content"] for m in users] == [
        "Find recently archived chats across all workspaces.",
        "Restore the second one.",
    ]
    assert len({m["conversation_id"] for m in users}) == 1
    links = {row["call_id"]: row for row in final["responses"]}
    for user in users:
        calls = [c for c in final["calls"] if c["turn_id"] == user["id"]]
        assert {c["call_sequence"] for c in calls} == {0, 1}
        terminal = links[max(calls, key=lambda c: c["call_sequence"])["call_id"]][
            "link"
        ]
        assert (
            terminal["link_kind"] == "revision"
            and terminal["verification_outcome"] == "verified_equal"
        )
        own = [
            r for r in final["user_revisions"] if r["source_message_id"] == user["id"]
        ]
        assert len(own) == 1
        for request in (
            r for r in final["reconstructions"] if r["turn_id"] == user["id"]
        ):
            assert request["boundary_source_revisions"] == [own[0]["revision_id"]]
            assert not request["omissions"]
            assert request["archive_handoffs"] == (1 if user is users[1] else 0)
    for response in final["responses"]:
        assert response["profile"]["version"] == 2
        assert response["profile"]["source_formats"] == [
            "start_anchored_think",
            "reasoning_content",
        ]
    assert len(final["thinking_structure"]) == 2
    for row in final["thinking_structure"]:
        assert [block["source_format"] for block in row["blocks"]] == [
            "reasoning_content:tool_call",
            "reasoning_content",
        ]
        assert [block["round_ordinal"] for block in row["blocks"]] == [0, 1]
        assert all(
            block["status"] == "complete"
            and block["provider"] == "llama_cpp"
            and block["protocol"] == "chat_completions"
            for block in row["blocks"]
        )
    proof = load("handoff-proof")
    metadata = proof["metadata"]
    assert (
        set(metadata)
        == {"source_session_id", "archive_recovery_results", "omitted_pages"}
        and metadata["omitted_pages"] == 0
    )
    assert (
        len(metadata["archive_recovery_results"]) == 1
        and proof["wrapper_bytes"] <= 32768
    )
    page = metadata["archive_recovery_results"][0]
    assert set(page) == {"result_set_id", "items"}
    assert [x["title"] for x in page["items"]] == [
        "Birch itinerary conversation",
        "Linden planning conversation",
        "Maple archival notes",
    ]
    for ordinal, item in enumerate(page["items"], 1):
        assert (
            set(item)
            == {"ordinal", "result_ref", "title", "workspace_label", "allowed_scopes"}
            and item["ordinal"] == ordinal
        )
    assert page["items"][1]["workspace_label"] == "Atlas Research" and page["items"][1][
        "allowed_scopes"
    ] == ["workspace_only"]
    assert "call_id" not in proof["actual_restore_fence"]["payload"]
    assert proof["actual_restore_fence"]["payload"]["arguments"] == {
        "result_ref": page["items"][1]["result_ref"],
        "scope": "workspace_only",
    }
    assert all(s["call_id"] == "turn-1-call-0" for s in proof["selected_steps"])
    confirm = (case / "restore-awaiting.txt").read_text()
    assert all(
        text in confirm
        for text in (
            "Restore Linden planning conversation",
            "Workspace: Atlas Research",
            "Restore workspace",
            "Confirm restore",
        )
    )
    opened = (case / "opened-original.txt").read_text()
    assert all(
        text in opened
        for text in (
            "Conversation | Linden planning conversation",
            "Original history for Linden planning conversation",
            "Saved reply for Linden planning conversation",
        )
    )
    returned = (case / "source-draft-retained.txt").read_text()
    assert (
        "Conversation | Recovery Check Chat" in returned
        and "Unsent recovery draft stays here." in returned
    )
    assert load("closure-final")["native_quit"]
    results[name] = {
        "complete_calls": 4,
        "terminal_revision_verified_equal": 2,
        "exact_current_user_pins": True,
        "predecessors_unchanged": True,
        "original_history_identity_unchanged": True,
        "workspace_changes": changes,
        "no_restore_before_confirmation": True,
        "exact_second_result_workspace_confirmation": True,
        "no_id_fence_dispatch_succeeded": True,
        "explicit_open_original_history": True,
        "source_draft_preserved": True,
    }
rows = [
    json.loads(line)
    for line in (root / "transport-metadata.jsonl").read_text().splitlines()
]
new = [r for r in rows if r.get("attempt", 0) >= 23]
requests = [r for r in new if r["event"] == "request"]
assert [r["attempt"] for r in requests] == list(range(23, 31))
assert all(
    r["retry_total"] == 0 and r["transport"] == "httpx_async" and not r["tool_names"]
    for r in requests
)
http = [r for r in new if r["event"] == "http"]
observations = [r for r in new if r["event"] == "response_observation"]
assert len(http) == len(observations) == 8 and all(r["status"] == 200 for r in http)
assert all(
    r["sse_done"] and not r["invalid_json"] and not r["observation_truncated"]
    for r in observations
)
assert not any(r["event"] in {"transport_exception", "denied_cap"} for r in rows)
assert json.loads((root / "counter.json").read_text()) == {
    "consumed": 30,
    "authorized_total": 30,
}
assert json.loads((root / "fenced-normal-before.json").read_text()) == json.loads(
    (root / "fenced-normal-after.json").read_text()
)
closure = json.loads((root / "fenced-closure.json").read_text())
assert (
    closure["normal_fingerprints_equal"]
    and closure["health"] == 200
    and closure["no_source_change"]
)
results.update(
    attempts=list(range(23, 31)),
    http200_sse_done=8,
    budget=closure["budget"],
    normal_fingerprints_equal=True,
    source_unchanged=True,
    native_apps_exit0=True,
    server_idle="not measured",
)
(root / "fenced-verification.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
