"""Live verification: local thinking controls against the real llama-server on :9191.

Exercises OUR code (build_llamacpp_chat_payload, complete_llamacpp_chat,
stream_llamacpp_chat via ConsoleProviderGateway) plus raw wire probes for
observable thinking-depth evidence. TASK-16812 Task 7 Step 2.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    build_llamacpp_chat_payload,
)
from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute

BASE = "http://127.0.0.1:9191"
MODEL = "../../../Downloads/Qwen3.8-27B-UD-Q8_K_XL.gguf"
PROMPT = [{"role": "user", "content": "In one sentence, explain why the sky is blue."}]
MAX_TOKENS = 300

results: list[str] = []


def record(name: str, ok: bool, detail: str) -> None:
    line = f"CHECK {name}: {'PASS' if ok else 'FAIL'} — {detail}"
    results.append(line)
    print(line, flush=True)


def raw_probe(label: str, thinking_fields: dict, prefill: str | None = None) -> dict:
    messages = [dict(m) for m in PROMPT]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    payload.update(thinking_fields)
    t0 = time.time()
    r = httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=600)
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = body.get("usage", {})
    print(
        f"PROBE {label}: dt={dt:.0f}s comp={usage.get('completion_tokens')} "
        f"content_len={len(content)} reasoning_len={len(reasoning)} "
        f"has_think_tag={'<think>' in content}",
        flush=True,
    )
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "dt": dt,
    }


def main() -> None:
    # --- Wire probe 0: server split mode detection (no thinking fields) ---
    base = raw_probe("baseline-default", {})

    # --- A: effort low vs xhigh observably change thinking depth ---
    low = raw_probe("effort-low", {"chat_template_kwargs": {"reasoning_effort": "low"}})
    xhigh = raw_probe(
        "effort-xhigh", {"chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    )
    if base["reasoning"] or low["reasoning"] or xhigh["reasoning"]:
        depth_low, depth_high = len(low["reasoning"]), len(xhigh["reasoning"])
        record(
            "A-effort-changes-depth",
            depth_high > depth_low,
            f"reasoning chars low={depth_low} xhigh={depth_high} (server-split mode)",
        )
    else:
        # unsplit: measure <think> block inside content
        def think_len(probe: dict) -> int:
            c = probe["content"]
            if "</think>" in c:
                return c.index("</think>")
            return len(c) if c.startswith("<think") else 0

        depth_low, depth_high = think_len(low), think_len(xhigh)
        record(
            "A-effort-changes-depth",
            depth_high > depth_low,
            f"think chars low={depth_low} xhigh={depth_high} (unsplit mode)",
        )

    # --- B: reasoning_budget truncates thinking ---
    budget = raw_probe(
        "budget-1024",
        {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
            "reasoning_budget": 1024,
        },
    )
    xhigh_reason = len(xhigh["reasoning"]) or len(
        xhigh["content"].split("</think>")[0]
    )
    budget_reason = len(budget["reasoning"]) or len(
        budget["content"].split("</think>")[0]
    )
    record(
        "B-budget-truncates",
        budget_reason < xhigh_reason,
        f"reasoning chars xhigh={xhigh_reason} budget1024={budget_reason}",
    )

    # --- C: effort none disables thinking ---
    none_p = raw_probe(
        "effort-none", {"chat_template_kwargs": {"enable_thinking": False}}
    )
    if none_p["reasoning"]:
        ok = len(none_p["reasoning"]) == 0
        detail = f"reasoning chars={len(none_p['reasoning'])}"
    else:
        ok = not none_p["content"].lstrip().startswith("<think")
        detail = f"content starts with think tag: {none_p['content'].lstrip()[:20]!r}"
    record("C-none-disables", ok, detail)

    # --- D: prefill + thinking controls must not 400 ---
    try:
        pre = raw_probe("prefill+xhigh", {"chat_template_kwargs": {"reasoning_effort": "xhigh"}}, prefill="The sky appears blue")
        record("D-prefill-no-400", True, f"status ok, content_len={len(pre['content'])}")
    except httpx.HTTPStatusError as exc:
        record("D-prefill-no-400", False, f"HTTP {exc.response.status_code}")

    # --- E: OUR payload builder composes the exact expected wire fields ---
    payload = build_llamacpp_chat_payload(
        model=MODEL,
        messages=PROMPT,
        stream=False,
        reasoning_effort="low",
        thinking_budget_tokens=1024,
    )
    ok = (
        payload.get("chat_template_kwargs") == {"reasoning_effort": "low"}
        and payload.get("reasoning_budget") == 1024
    )
    record("E-builder-composes", ok, json.dumps(payload.get("chat_template_kwargs")) + f" reasoning_budget={payload.get('reasoning_budget')}")

    # --- F: OUR gateway stream filters think text (unsplit) / stays clean (split) ---
    async def stream_check() -> tuple[str, str]:
        gw = ConsoleProviderGateway()

        async def admit():
            return gw._capture_off_admission(None)

        async def admit_fallback(_endpoint, _payload):
            return gw._capture_off_admission(ConsoleRequestRoute.LLAMA_FALLBACK)

        try:
            chunks = []
            async for c in gw.stream_llamacpp_chat(
                base_url=BASE,
                model=MODEL,
                messages=PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                reasoning_effort="xhigh",
                before_adapter=admit,
                before_fallback_adapter=admit_fallback,
            ):
                chunks.append(c)
            visible = "".join(chunks)
        finally:
            await gw.aclose()
        # Also raw non-streamed content for comparison
        return visible, base["content"]

    visible, _ = asyncio.run(stream_check())
    record(
        "F-stream-no-think-soup",
        "<think>" not in visible and "<think" not in visible[:40],
        f"visible starts {visible.lstrip()[:40]!r} len={len(visible)}",
    )

    # --- G: OUR gateway complete path filters too ---
    async def complete_check() -> str:
        gw = ConsoleProviderGateway()
        try:
            return await gw.complete_llamacpp_chat(
                base_url=BASE,
                model=MODEL,
                messages=PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                reasoning_effort="xhigh",
                adapter_admission=gw._capture_off_admission(None),
            )
        finally:
            await gw.aclose()

    visible2 = asyncio.run(complete_check())
    record(
        "G-complete-no-think-soup",
        not visible2.lstrip().startswith("<think"),
        f"visible starts {visible2.lstrip()[:40]!r}",
    )

    print("\n=== SUMMARY ===", flush=True)
    fails = [r for r in results if "FAIL" in r]
    for r in results:
        print(r, flush=True)
    print(f"\n{len(results) - len(fails)}/{len(results)} checks passed", flush=True)


if __name__ == "__main__":
    main()
