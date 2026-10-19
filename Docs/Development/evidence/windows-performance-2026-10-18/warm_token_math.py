"""Measure warm-backend counting separately from native loading and TUI paint."""

import faulthandler
import json
from pathlib import Path
from statistics import median
import sys
import time

sys.argv.extend(["--tag", "warm-token-math"])
import probe  # Sets up the same protected private profile/native fixture owner.

probe.stop_sampling.set()
probe.sampler.join(1)

try:
    from tldw_chatbook.UI.Console_Modules.tokenizer_display import (
        capture_tokenizer_preparation,
        preparation_current,
    )
    from tldw_chatbook.Utils import token_counter

    request = capture_tokenizer_preparation("local-model", "llama_cpp")
    assert request is not None and request()
    results = {"temporary_profile": str(probe.scratch), "measurements": []}
    for length in (1_000, 10_000, 100_000, 1_000_000):
        payload = ("def hello_world(): return 'hello world'\n" * (length // 30 + 1))[
            :length
        ]
        messages = [{"role": "user", "content": payload}]
        misses, hits = [], []
        for _ in range(7):
            token_counter.clear_estimate_cache()
            started = time.perf_counter()
            count = token_counter.count_tokens_messages(
                messages, "local-model", "llama_cpp"
            )
            misses.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            assert (
                token_counter.count_tokens_messages(
                    messages, "local-model", "llama_cpp"
                )
                == count
            )
            hits.append((time.perf_counter() - started) * 1000)
        results["measurements"].append(
            {
                "characters": length,
                "tokens": count,
                "warm_backend_uncached_median_ms": median(misses),
                "warm_backend_uncached_max_ms": max(misses),
                "memo_hit_median_ms": median(hits),
            }
        )
    started = time.perf_counter()
    for _ in range(1000):
        assert preparation_current(request)
    results["readiness_check_mean_ms"] = time.perf_counter() - started
    results["blocked_network_attempts"] = len(probe.network_guard.blocked_attempts())
    results["real_profile_refusals"] = len(probe.real_profile_guard.take_violations())
    (Path(__file__).parent / "warm-token-math.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
finally:
    faulthandler.cancel_dump_traceback_later()
    probe.stack_log.close()
    probe.fixture_owner.__exit__(None, None, None)
