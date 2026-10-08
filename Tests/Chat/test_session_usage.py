"""SessionUsageLedger unit tests (issue #365).

The ledger is the quit-time summary's single data source; these tests pin
its accumulation rules: exact wins over estimate, malformed payloads never
raise, and recording is thread-safe.
"""

import threading

import pytest

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.session_usage import (
    SessionUsageSnapshot,
    reset_for_tests,
    session_usage,
)
from tldw_chatbook.Chat.usage_recorder import estimate_tokens


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_for_tests()
    yield
    reset_for_tests()


def _openai_usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def test_exact_accumulates():
    ledger = session_usage()
    ledger.record_provider_payload(_openai_usage(10, 5), provider="openai", model="m")
    ledger.record_provider_payload(_openai_usage(1, 2), provider="openai", model="m")
    snap = ledger.snapshot()
    assert snap.exact_tokens == 18
    assert snap.estimated_tokens == 0
    assert snap.calls == 2
    assert snap.total_tokens == 18


def test_estimate_fallback_when_payload_lacks_usage():
    ledger = session_usage()
    ledger.record_provider_payload(
        None, fallback_texts=("hello world, quite long", "ok")
    )
    snap = ledger.snapshot()
    assert snap.exact_tokens == 0
    assert snap.estimated_tokens == (
        estimate_tokens("hello world, quite long") + estimate_tokens("ok")
    )
    assert snap.calls == 1


def test_exact_wins_no_estimate_double_count():
    ledger = session_usage()
    ledger.record_provider_payload(
        _openai_usage(10, 5),
        provider="openai",
        model="m",
        fallback_texts=("a very long prompt text", "a very long reply"),
    )
    snap = ledger.snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_malformed_payloads_never_raise():
    ledger = session_usage()
    ledger.record_provider_payload(object())
    ledger.record_provider_payload({"prompt_tokens": "not-a-number"})
    ledger.record_provider_payload("usage")
    assert ledger.snapshot() == SessionUsageSnapshot(0, 0, 0)


def test_none_payload_without_texts_is_noop():
    session_usage().record_provider_payload(None)
    assert session_usage().snapshot().calls == 0


def test_record_exact_accepts_none():
    session_usage().record_exact(None)
    assert session_usage().snapshot().calls == 0


def test_record_exact_sums_provider_usage_totals():
    usage = ProviderUsage.from_provider_payload(
        {"input_tokens": 7, "output_tokens": 3}, provider="anthropic", model="m"
    )
    assert usage is not None
    session_usage().record_exact(usage)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


def test_concurrent_records_are_thread_safe():
    ledger = session_usage()

    def worker() -> None:
        for _ in range(1000):
            ledger.record_provider_payload(
                _openai_usage(1, 1), provider="openai", model="m"
            )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = ledger.snapshot()
    assert snap.exact_tokens == 8 * 1000 * 2
    assert snap.calls == 8 * 1000


def test_reset_for_tests_gives_fresh_singleton():
    old = session_usage()
    old.record_provider_payload(_openai_usage(1, 1), provider="openai", model="m")
    reset_for_tests()
    assert session_usage() is not old
    assert session_usage().snapshot() == SessionUsageSnapshot(0, 0, 0)
