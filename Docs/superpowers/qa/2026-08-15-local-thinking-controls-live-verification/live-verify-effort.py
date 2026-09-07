"""Focused re-check of live CHECK A (effort changes thinking depth) and B
(budget truncates), with noise-reduced methodology: temperature 0, a
reasoning-inviting prompt, and two repetitions per arm."""

from __future__ import annotations

import json

import httpx

BASE = "http://127.0.0.1:9191"
MODEL = "../../../Downloads/Qwen3.8-27B-UD-Q8_K_XL.gguf"
PROMPT = [
    {
        "role": "user",
        "content": (
            "A bat and a ball cost $1.10 together. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Reason carefully."
        ),
    }
]


def probe(label: str, fields: dict) -> tuple[int, int]:
    payload = {
        "model": MODEL,
        "messages": PROMPT,
        "max_tokens": 500,
        "temperature": 0.0,
    }
    payload.update(fields)
    r = httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    reasoning = len(msg.get("reasoning_content") or "")
    comp = body.get("usage", {}).get("completion_tokens", 0)
    print(f"PROBE {label}: reasoning_chars={reasoning} completion_tokens={comp}", flush=True)
    return reasoning, comp


def main() -> None:
    arms = {
        "low": {"chat_template_kwargs": {"reasoning_effort": "low"}},
        "medium": {"chat_template_kwargs": {"reasoning_effort": "medium"}},
        "xhigh": {"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
        "budget400": {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
            "reasoning_budget": 400,
        },
    }
    samples: dict[str, list[tuple[int, int]]] = {}
    for rep in range(2):
        for name, fields in arms.items():
            samples.setdefault(name, []).append(probe(f"{name}-rep{rep}", fields))

    print("\n=== A: monotonic depth low <= medium <= xhigh (reasoning chars) ===", flush=True)
    means = {n: sum(r for r, _ in v) / len(v) for n, v in samples.items()}
    print(f"means: {json.dumps({k: int(v) for k, v in means.items()})}", flush=True)
    a_ok = means["low"] <= means["medium"] <= means["xhigh"]
    print(f"CHECK A-effort-monotonic: {'PASS' if a_ok else 'FAIL'}", flush=True)

    print("\n=== B: budget 400 caps below unbounded xhigh ===", flush=True)
    b_ok = means["budget400"] < means["xhigh"]
    print(
        f"CHECK B-budget-truncates: {'PASS' if b_ok else 'FAIL'} "
        f"(budget400={int(means['budget400'])} vs xhigh={int(means['xhigh'])})",
        flush=True,
    )


if __name__ == "__main__":
    main()
