#!/usr/bin/env python3
"""TASK-18602: cost of estimating tokens for a growing conversation.

Every caller of the token estimator re-estimates the WHOLE message list on
each turn, so an N-turn run pays O(N^2) to learn N answers. Two independent
fixes landed; this measures each one's contribution separately, because
they are NOT equal partners and the summary numbers are easy to misread:

* The ASCII fast path in ``_chars_estimate`` does essentially all the work
  on installs WITHOUT a real tokenizer (the shipped default). It removes a
  per-character Python loop.
* The memo in ``estimate_tokens`` adds little on top of that for the chars
  tier -- it matters on installs WITH tiktoken, where the per-turn
  re-encode of the whole history is the dominant cost and no fast path can
  remove it. Those installs cannot be measured here unless tiktoken is
  present, so the third column is the honest floor, not the headline.

Run: python Helper_Scripts/Benchmarks/token_estimate_benchmark.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tldw_chatbook.Utils import token_counter  # noqa: E402
from tldw_chatbook.Utils.token_counter import (  # noqa: E402
    TIKTOKEN_AVAILABLE,
    _chars_estimate,
    _is_cjk,
    clear_estimate_cache,
    count_tokens_messages,
)

TURNS = 400
ROUND_CHARS = 800


def legacy_chars_estimate(text: str, provider: str) -> int:
    """The pre-TASK-18602 implementation, for the 'before' column.

    Calls the REAL ``_is_cjk`` per character rather than a faster lookup:
    that generator-per-character shape is precisely what made the original
    slow, so a tidier reimplementation would understate the baseline by
    more than an order of magnitude and flatter the result.
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, int((other * 0.25 + cjk * 1.0) * 1.1))


def _run(label: str, *, legacy: bool, memo: bool) -> float:
    real = token_counter._chars_estimate
    if legacy:
        token_counter._chars_estimate = legacy_chars_estimate
    clear_estimate_cache()
    try:
        messages = [{"role": "system", "content": "s" * 2000}]
        cumulative = 0.0
        for turn in range(TURNS):
            # DISTINCT content per message, as a real conversation has. An
            # earlier draft appended the same string every turn, which let
            # the memo collapse a whole turn's payload into one entry and
            # made every configuration look identical.
            messages.append(
                {"role": "assistant", "content": f"a{turn} " + "x" * ROUND_CHARS}
            )
            messages.append(
                {"role": "user", "content": f"u{turn} " + "y" * ROUND_CHARS}
            )
            if not memo:
                clear_estimate_cache()
            start = time.perf_counter()
            count_tokens_messages(messages, "gpt-4o-mini", provider="openai")
            cumulative += time.perf_counter() - start
    finally:
        token_counter._chars_estimate = real
        clear_estimate_cache()
    print(f"  {label:<34} {cumulative:8.3f} s")
    return cumulative


def _bench_send_path() -> None:
    """The LIVE path: `bound_messages_to_window` on each Console send.

    `console_history_budget` is imported by `console_chat_controller` and
    tokenizes the whole history to decide what fits in the context window,
    once per send. This is the number that describes real user-facing cost;
    the synthetic agent-loop figure below describes the worst case.
    """
    from tldw_chatbook.Chat.console_history_budget import bound_messages_to_window

    print("\n== per Console send: bound_messages_to_window ==")
    print(f"  {'history':<24}{'before':>12}{'after':>12}")
    for turns in (60, 120, 240):
        history = [{"role": "system", "content": "s" * 1500}]
        for i in range(turns):
            history.append({"role": "user", "content": f"q{i} " + "u" * 1500})
            history.append({"role": "assistant", "content": f"a{i} " + "x" * 1500})
        kilobytes = sum(len(m["content"]) for m in history) / 1024
        timings = []
        for legacy in (True, False):
            real = token_counter._chars_estimate
            if legacy:
                token_counter._chars_estimate = legacy_chars_estimate
            clear_estimate_cache()
            try:
                start = time.perf_counter()
                bound_messages_to_window(
                    history,
                    model="gpt-4o-mini",
                    provider="openai",
                    response_reservation=4096,
                )
                timings.append(time.perf_counter() - start)
            finally:
                token_counter._chars_estimate = real
        label = f"{turns} turns / {kilobytes:.0f} KB"
        print(
            f"  {label:<24}{timings[0] * 1000:>9.1f} ms{timings[1] * 1000:>9.1f} ms"
        )


def main() -> None:
    print(f"tiktoken installed: {TIKTOKEN_AVAILABLE}")
    if TIKTOKEN_AVAILABLE:
        print("  (the chars tier is bypassed; the memo carries the win here)")
    print(f"\n== single call: CJK scan of a 640 KB payload ==")
    big = "x" * 640_000
    start = time.perf_counter()
    legacy_chars_estimate(big, "openai")
    before = time.perf_counter() - start
    start = time.perf_counter()
    _chars_estimate(big, "openai")
    after = time.perf_counter() - start
    print(f"  per-char Python loop               {before * 1000:8.2f} ms")
    print(f"  str.isascii() fast path            {after * 1000:8.4f} ms")
    print(f"  -> {before / after:,.0f}x")

    _bench_send_path()

    print(f"\n== cumulative estimator CPU, {TURNS}-turn agent run (worst case) ==")
    baseline = _run("before (per-char loop, no memo)", legacy=True, memo=False)
    fast = _run("+ ASCII fast path", legacy=False, memo=False)
    both = _run("+ memo (shipped)", legacy=False, memo=True)
    print()
    print(f"  ASCII fast path:  {baseline / max(fast, 1e-9):>8,.0f}x vs before")
    print(f"  memo on top:      {fast / max(both, 1e-9):>8,.1f}x further")
    print(f"  shipped total:    {baseline / max(both, 1e-9):>8,.0f}x vs before")


if __name__ == "__main__":
    main()
