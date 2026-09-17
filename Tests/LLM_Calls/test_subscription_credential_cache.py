"""Credential I/O stays single-flight and failed reads have a useful TTL."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription


@pytest.fixture
def keychain(monkeypatch):
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(subscription, "_KEYCHAIN_CACHE", None)


def test_timeout_is_cached_from_completion_and_retried_after_ttl(keychain, monkeypatch):
    now = [100.0]
    attempts = []
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]),
    )

    def timeout(command, **kwargs):
        attempts.append(command)
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subscription.subprocess, "run", timeout)
    assert subscription._keychain_credential_raw() is None
    now[0] += 4.9
    assert subscription._keychain_credential_raw() is None
    assert len(attempts) == 1
    now[0] += 0.2
    assert subscription._keychain_credential_raw() is None
    assert len(attempts) == 2


def test_concurrent_keychain_reads_share_one_process(keychain, monkeypatch):
    entered = Event()
    release = Event()
    barrier = Barrier(8)
    attempts = []

    def read(command, **kwargs):
        attempts.append(command)
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(returncode=0, stdout="credential-json")

    def concurrent_read():
        barrier.wait(timeout=3)
        return subscription._keychain_credential_raw()

    monkeypatch.setattr(subscription.subprocess, "run", read)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(concurrent_read) for _ in range(8)]
        try:
            assert entered.wait(3)
        finally:
            release.set()
        assert [future.result(timeout=3) for future in futures] == [
            "credential-json"
        ] * 8
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "token", [{"secret": "not-a-token"}, ["not-a-token"], 123, True]
)
def test_non_string_tokens_do_not_report_a_usable_credential(tmp_path, token):
    path = tmp_path / "credential.json"
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": token}}))
    assert subscription.read_claude_code_credential(path) is None


def test_non_finite_expiry_is_malformed_without_raising(tmp_path):
    path = tmp_path / "credential.json"
    path.write_text(
        '{"claudeAiOauth": {"accessToken": "private-token", "expiresAt": 1e999}}'
    )
    assert subscription.read_claude_code_credential(path) is None
