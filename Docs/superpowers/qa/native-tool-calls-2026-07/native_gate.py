"""task-243 live gate: real-stack native tool-calls vs llama.cpp @127.0.0.1:9099.

Real ConsoleAgentBridge -> real ConsoleProviderGateway -> real chat_api_call ->
real HTTP. The ONLY instrumentation is a recording passthrough around
gateway.stream_chat that logs the kwargs (tools present? fence protocol in the
system prompt?) then delegates unchanged.

Cases (select via argv, default all):
  A. native single tool-call round-trip (custom-openai-api -> llama.cpp)
  B. native multi-call attempt (best-effort; model may serialize)
  C. fence regression (llama_cpp provider branch — must NOT go native)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime")

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway, ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

BASE = "http://127.0.0.1:9099"

def _served_model() -> str:
    import urllib.request
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
        data = json.load(r)
    entries = data.get("models") or data.get("data") or []
    return entries[0].get("model") or entries[0].get("id")

MODEL = _served_model()


class RecordingGateway:
    """Passthrough recorder — delegates every call to the REAL gateway."""

    def __init__(self, real):
        self._real = real
        self.calls = []

    async def stream_chat(self, resolution, messages, tools=None):
        head = ""
        if messages and messages[0].get("role") == "system":
            head = str(messages[0]["content"])
        self.calls.append({
            "tools_names": [t["function"]["name"] for t in tools] if tools else None,
            "fence_protocol_in_system": "tool_call" in head and "MUST START" in head,
        })
        async for chunk in self._real.stream_chat(resolution, messages, tools=tools):
            yield chunk


def resolution(provider, execution_key):
    return ConsoleProviderResolution(
        provider=provider, base_url=BASE, model=MODEL, ready=True,
        readiness_key=provider, execution_key=execution_key,
        api_key="local", streaming=True, max_tokens=768)


def run_case(label, res, question, dbdir):
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content=question)
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    gw = RecordingGateway(ConsoleProviderGateway())
    db = AgentRunsDB(Path(dbdir) / f"{label}.db", client_id="gate")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gw)
    t0 = time.time()
    outcome = bridge.run_reply(
        conversation_id=f"conv-{label}", session_id=session.id, resolution=res,
        assistant_message_id=assistant.id, model=MODEL,
        session_system_prompt="", agent_messages=[{"role": "user", "content": question}],
        should_cancel=lambda: False)
    elapsed = time.time() - t0
    runs = db.list_runs(f"conv-{label}")
    steps = runs[0]["steps"] if runs else []
    final = store.get_message(assistant.id).content
    tool_markers = [m.content for m in store.messages_for_session(session.id)
                    if str(getattr(m.role, "value", m.role)).lower() == "tool"]
    print(f"\n===== {label} ({elapsed:.1f}s) =====")
    print("status:", outcome.status)
    print("final answer:", json.dumps(final[:300]))
    print("provider calls:")
    for i, c in enumerate(gw.calls):
        print(f"  turn {i}: tools={c['tools_names']} fence_in_system={c['fence_protocol_in_system']}")
    print("run steps:")
    for s in steps:
        print(f"  [{s['kind']}] {s.get('tool_name','')} :: {str(s.get('summary') or s.get('result'))[:120]}")
    print("TOOL markers in transcript:", len(tool_markers))
    for m in tool_markers[:4]:
        print("   ", m[:110].replace("\n", " | "))
    return outcome, gw, steps


def main():
    which = set(sys.argv[1:]) or {"A", "B", "C"}
    dbdir = os.environ.get("GATE_DBDIR") or tempfile.mkdtemp(prefix="native-gate-")
    print("evidence db dir:", dbdir)
    results = {}

    if "A" in which:
        out, gw, steps = run_case(
            "A-native-single", resolution("custom", "custom-openai-api"),
            "What is 234*77? Use the calculator tool, then answer with just the number.",
            dbdir)
        native = gw.calls and gw.calls[0]["tools_names"] and not gw.calls[0]["fence_protocol_in_system"]
        roundtrip = out.status == "done" and any(s["kind"] == "tool_result" for s in steps)
        print("A native (tools= sent, no fence protocol):", bool(native))
        print("A tool round-trip done:", roundtrip)
        results["A"] = bool(native and roundtrip)

    if "B" in which:
        out, gw, steps = run_case(
            "B-native-multi", resolution("custom", "custom-openai-api"),
            "Use your tools: get the current date AND calculate 91*7. "
            "You may call both tools at once in a single turn.", dbdir)
        batch = sum(1 for s in steps if s["kind"] == "tool_call")
        print("B tool calls made:", batch, "| status:", out.status)
        results["B"] = out.status == "done" and batch >= 2

    if "C" in which:
        out, gw, steps = run_case(
            "C-fence-llamacpp", resolution("llama_cpp", "llama_cpp"),
            "What is 234*77? Use the calculator tool, then answer with just the number.",
            dbdir)
        fence = all(c["tools_names"] is None for c in gw.calls) and gw.calls[0]["fence_protocol_in_system"]
        roundtrip = out.status == "done" and any(s["kind"] == "tool_result" for s in steps)
        print("C fence path (no tools=, protocol present):", bool(fence))
        print("C tool round-trip done:", roundtrip)
        results["C"] = bool(fence and roundtrip)

    print("\n===== VERDICTS =====")
    print("results:", results)
    ok = all(results.values()) and bool(results)
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
