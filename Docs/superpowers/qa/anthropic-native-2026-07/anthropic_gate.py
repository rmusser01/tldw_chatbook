"""task-263 live gate: Anthropic native tool-calls vs the REAL API.

Real ConsoleAgentBridge -> real ConsoleProviderGateway -> real chat_api_call
-> chat_with_anthropic -> api.anthropic.com. Streaming on (exercises the
input_json_delta reassembly live). The native-set FLIP has not landed yet:
this harness overrides NATIVE_TOOLS_PROVIDERS in-process so the flip commit
only happens after this evidence (plan Task 4, AC #3 ordering).

The API key is read from the git-excluded repo-root anthropic-api-key.txt
and never printed.

Cases: A single calculator round-trip; B multi-tool single-turn attempt.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

WT = "/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime"
sys.path.insert(0, WT)

from tldw_chatbook.Agents import native_tools
# Pre-flip override (gate-only): prove the conversion end-to-end BEFORE
# committing the set change.
native_tools.NATIVE_TOOLS_PROVIDERS = frozenset(
    native_tools.NATIVE_TOOLS_PROVIDERS | {"anthropic"})

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway, ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

MODEL = "claude-haiku-4-5-20251001"
KEY = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook/anthropic-api-key.txt").read_text().strip()


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


def resolution() -> "ConsoleProviderResolution":
    """Build the ready Anthropic resolution the gate cases share.

    Returns:
        A streaming-on resolution routed at execution_key "anthropic".
    """
    return ConsoleProviderResolution(
        provider="anthropic", base_url="", model=MODEL, ready=True,
        readiness_key="anthropic", execution_key="anthropic",
        api_key=KEY, streaming=True, max_tokens=1024)


def run_case(label: str, question: str, dbdir: str) -> tuple:
    """Run one gate case through the real Console reply engine.

    Args:
        label: Case label (used for the evidence DB filename and prints).
        question: The user prompt to send.
        dbdir: Directory for this case's AgentRunsDB evidence file.

    Returns:
        (RunOutcome, RecordingGateway, steps) for verdict computation.
    """
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content=question)
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    gw = RecordingGateway(ConsoleProviderGateway())
    db = AgentRunsDB(Path(dbdir) / f"{label}.db", client_id="gate")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gw)
    t0 = time.time()
    outcome = bridge.run_reply(
        conversation_id=f"conv-{label}", session_id=session.id, resolution=resolution(),
        assistant_message_id=assistant.id, model=MODEL,
        session_system_prompt="", agent_messages=[{"role": "user", "content": question}],
        should_cancel=lambda: False)
    elapsed = time.time() - t0
    runs = db.list_runs(f"conv-{label}")
    steps = runs[0]["steps"] if runs else []
    final = store.get_message(assistant.id).content
    print(f"\n===== {label} ({elapsed:.1f}s) =====")
    print("status:", outcome.status)
    print("final answer:", json.dumps(final[:300]))
    print("provider calls:")
    for i, c in enumerate(gw.calls):
        print(f"  turn {i}: tools={c['tools_names']} fence_in_system={c['fence_protocol_in_system']}")
    print("run steps:")
    for s in steps:
        print(f"  [{s['kind']}] {s.get('tool_name','')} :: {str(s.get('summary') or s.get('result'))[:120]}")
    return outcome, gw, steps


def main() -> int:
    """Run cases A and B and print verdicts.

    Returns:
        Process exit code: 0 on GATE PASS, 1 otherwise.
    """
    dbdir = os.environ.get("GATE_DBDIR") or tempfile.mkdtemp(prefix="anthropic-gate-")
    print("evidence db dir:", dbdir)
    results = {}

    out, gw, steps = run_case(
        "A-anthropic-single",
        "What is 234*77? Use the calculator tool, then answer with just the number.",
        dbdir)
    native = gw.calls and gw.calls[0]["tools_names"] and not gw.calls[0]["fence_protocol_in_system"]
    roundtrip = out.status == "done" and any(s["kind"] == "tool_result" for s in steps)
    second_turn_ok = len(gw.calls) >= 2  # role=tool history accepted (no 400)
    print("A native + no fence:", bool(native), "| round-trip done:", roundtrip,
          "| tool_result turn accepted:", second_turn_ok)
    results["A"] = bool(native and roundtrip and second_turn_ok)

    out, gw, steps = run_case(
        "B-anthropic-multi",
        "Use your tools: get the current date AND calculate 91*7. "
        "Call both tools in parallel in a single reply.", dbdir)
    batch = sum(1 for s in steps if s["kind"] == "tool_call")
    model_turns = sum(1 for s in steps if s["kind"] == "model")
    print("B tool calls:", batch, "| model turns:", model_turns, "| status:", out.status)
    results["B"] = out.status == "done" and batch >= 2

    print("\n===== VERDICTS =====")
    print("results:", results)
    ok = all(results.values()) and bool(results)
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
