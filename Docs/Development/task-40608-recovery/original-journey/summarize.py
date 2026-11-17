"""Summarize non-overlapping monotonic timing intervals from the latency pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(rows, turn):
    trace = sorted((r for r in rows if r["turn"] == turn), key=lambda r: r["t"])
    by_name = {}
    for row in trace:
        by_name.setdefault(row["event"], []).append(row)

    def first(name):
        return by_name.get(name, [{}])[0].get("t")

    send = first("send")
    if send is None:
        return None
    done = first("reply_completed")
    if done is None:
        done = first("reply_seen")
    provider_entry = first("http_send")
    if provider_entry is None:
        provider_entry = first("http_request_received")
    paint = first("painted_sending")
    out = {"turn": turn, "events": dict(Counter(r["event"] for r in trace))}

    def offset(label, value):
        if value is not None:
            out[label] = round(value - send, 4)

    offset("first_sending_s", paint)
    offset("first_http_s", provider_entry)
    offset("completed_s", done)
    offset("verified_s", first("reply_verified"))
    offset("disclosure_painted_s", first("painted_disclosure"))
    offset("disclosure_click_s", first("disclosure_click"))
    if first("painted_disclosure") is not None and first("disclosure_click") is not None:
        out["disclosure_wait_s"] = round(first("disclosure_click") - first("painted_disclosure"), 4)
    readys = by_name.get("http_response_sent", [])
    receipts = by_name.get("http_request_received", [])
    if receipts and len(readys) == len(receipts):
        out["loopback_handler_s"] = round(sum(b["t"] - a["t"] for a, b in zip(receipts, readys)), 4)
    posts = by_name.get("http_send", [])
    requests = by_name.get("model_request_started", [])
    responses = by_name.get("model_response_completed", [])
    # AgentStep may be copied for persistence or UI projection. Anchor its
    # interval to a real POST and use the first source observation.
    out["model_slices_s"] = []
    for i, post in enumerate(posts):
        previous = posts[i - 1]["t"] if i else send
        following = posts[i + 1]["t"] if i + 1 < len(posts) else float("inf")
        begin = next((r["t"] for r in requests if previous < r["t"] <= post["t"]), None)
        end = next((r["t"] for r in responses if post["t"] <= r["t"] < following), None)
        out["model_slices_s"].append({
            "app_before_http": round(post["t"] - begin, 4) if begin is not None else None,
            "transport_and_handler": round(readys[i]["t"] - post["t"], 4) if i < len(readys) else None,
            "app_after_http": round(end - readys[i]["t"], 4) if end is not None and i < len(readys) else None,
        })

    def one_per_call(event):
        firsts = {}
        for row in by_name.get(event, []):
            firsts.setdefault(row["call"], row)
        return firsts

    opens = one_per_call("approval_requested")
    observes = by_name.get("approval_observed", [])
    paint_approvals = by_name.get("painted_approval", [])
    clicks = by_name.get("approval_click", [])
    out["approval_requested_to_click_s"] = {
        a["tool"]: round(b["t"] - a["t"], 4)
        for a, b in zip((r for r in opens.values() if any(o["tool"] == r["tool"] for o in observes)), clicks)
    }
    if clicks and len(paint_approvals) == len(clicks):
        out["painted_approval_to_click_s"] = [round(b["t"] - a["t"], 4) for a, b in zip(paint_approvals, clicks)]
    elif clicks and len(clicks) == len(observes):
        out["observed_approval_to_click_s"] = [
            round(b["t"] - a["t"], 4)
            for a, b in zip(observes, clicks)
        ]
    start_exec = one_per_call("tool_execution_started")
    out["tool_execution_s"] = {
        r["tool"]: round(r["t"] - start_exec[r["call"]]["t"], 4)
        for r in one_per_call("tool_succeeded").values() if r["call"] in start_exec
    }
    for key in ("WorkspaceToolExecutor.execute", "WorkspaceToolExecutor._build_request"):
        starts = by_name.get(key + "_begin", [])
        ends = by_name.get(key + "_end", [])
        if len(starts) == len(ends):
            out[key + "_s"] = [round(b["t"] - a["t"], 4) for a, b in zip(starts, ends)]
    out["worker_reported_ms"] = [
        r["worker_ms"] for r in by_name.get("_parse_worker_output_end", []) if r["worker_ms"] is not None
    ]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.trace:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        print(json.dumps({"source": path.parent.name,
                          "turns": [summarize(rows, t) for t in ("readme", "edit")]}, indent=2))

