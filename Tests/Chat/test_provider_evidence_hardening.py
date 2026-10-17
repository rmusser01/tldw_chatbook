"""TASK-33005.10: the Phase 5 evidence paths, measured and hardened.

Begin order is allocated under a lock (a free-threaded build gives
``next()`` on a shared ``count`` no atomicity), a warm readiness build with
shared evidence present does no endpoint parsing the evidence-free build
does not, and Settings' Test rows and the snapshot derive the model facet one
way.
"""

from __future__ import annotations

import sys
import threading

from tldw_chatbook.Chat import provider_endpoint_contract
from tldw_chatbook.Chat import provider_test_evidence as evidence_module
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
    console_send_connection,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
)

_IDENTITY = ProviderDraftIdentity(
    provider_key="openai",
    connection_identity=("openai", "https://api.openai.com/v1/chat/completions"),
    credential_source="none",
    credential_revision=0,
    draft_generation=0,
)


def test_begin_order_allocation_waits_for_its_lock():
    """AC#2: a begin allocates its order under ``_BEGIN_ORDER_LOCK``, so two
    stores on two threads never read the shared count at once -- whether or
    not a GIL makes ``next()`` atomic."""
    identity = _IDENTITY
    began = threading.Event()

    def begin() -> None:
        ProviderTestEvidenceStore().begin(identity)
        began.set()

    with evidence_module._BEGIN_ORDER_LOCK:
        worker = threading.Thread(target=begin)
        worker.start()
        assert not began.wait(0.2)  # Parked on the allocation lock.
    assert began.wait(5)
    worker.join(5)


def test_concurrent_begins_across_stores_never_share_an_order():
    """AC#2: eight threads, each with its own store, begin 400 probes at once;
    every token holds a distinct, gap-free order."""
    identity = _IDENTITY
    orders: list[list[int]] = [[] for _ in range(8)]
    barrier = threading.Barrier(len(orders))

    def run(sink: list[int]) -> None:
        store = ProviderTestEvidenceStore()
        barrier.wait()
        for _ in range(400):
            sink.append(store.begin(identity).order)

    threads = [threading.Thread(target=run, args=(sink,)) for sink in orders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    allocated = sorted(order for sink in orders for order in sink)
    assert len(allocated) == 3200
    assert allocated == list(range(allocated[0], allocated[0] + 3200))


def test_a_warm_readiness_build_with_evidence_parses_no_endpoint():
    """AC#1: evidence present made each warm build key its connection, which
    parsed the endpoint three more times (2.2-3.4x the build). Endpoint
    parsing is memoized, so a warm build parses none, evidence or not."""
    config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    connection = console_send_connection(settings, app_config=config, environ={})
    owner = ProviderConnectionEvidence()
    owner.publish(
        ProviderTestEvidence(connection, "unreachable", (), "timeout"), order=1
    )
    parse = provider_endpoint_contract._resolve_provider_endpoint.__code__

    def parses() -> int:
        calls = 0

        def profile(frame, event, _arg):
            nonlocal calls
            calls += event == "call" and frame.f_code is parse

        build_console_settings_readiness(
            settings, app_config=config, environ={}, connection_evidence=owner
        )  # Warm.
        sys.setprofile(profile)
        try:
            readiness = build_console_settings_readiness(
                settings, app_config=config, environ={}, connection_evidence=owner
            )
        finally:
            sys.setprofile(None)
        assert readiness.blocker == "endpoint_unreachable"
        return calls

    assert parses() == 0


def test_endpoint_parsing_memo_skips_values_that_can_carry_a_secret():
    """A URL with user info, a query or a fragment is refused anyway; it is
    never kept as a cache key."""
    resolve = provider_endpoint_contract.resolve_provider_endpoint
    memo = provider_endpoint_contract._memoized_resolution
    memo.cache_clear()
    for value in (
        "https://user:sk-secret@api.example.test/v1",
        "https://api.example.test/v1?api_key=sk-secret",
        "https://api.example.test/v1#sk-secret",
    ):
        assert resolve("custom", value).errors
    assert memo.cache_info().currsize == 0
    assert resolve("custom", "http://127.0.0.1:8080/v1") is resolve(
        "custom", "http://127.0.0.1:8080/v1"
    )


def test_settings_and_the_snapshot_derive_the_model_facet_one_way():
    """AC#3: Settings' Readiness row and ``ProviderReadiness.snapshot`` read
    one model facet: missing, confirmed only for a listed model of a reached
    listing, else unconfirmed. For a key its listing accepted, the facet is
    the difference between "verified" and "reachable", so Settings' word is
    checked against the snapshot's for a listed, an unlisted and no model."""
    from dataclasses import replace
    from datetime import datetime

    from tldw_chatbook.Chat.console_session_settings import verdict_readiness_words
    from tldw_chatbook.Chat.provider_test_evidence import model_facet
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    assert model_facet("", "reachable", ("model-a",)) == "missing"
    assert model_facet("  ", "reachable", ("model-a",)) == "missing"
    assert model_facet("model-a", "reachable", ("model-a",)) == "confirmed"
    assert model_facet(" model-a ", "reachable", ("model-a",)) == "confirmed"
    assert model_facet("model-b", "reachable", ("model-a",)) == "unconfirmed"
    assert model_facet("model-a", "testing", ("model-a",)) == "unconfirmed"
    assert model_facet("model-a", "unreachable", ()) == "unconfirmed"

    readiness = get_provider_readiness(
        "llama_cpp",
        {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}},
        environ={},
    )
    identity = ProviderDraftIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", "http://127.0.0.1:9099"),
        credential_source="none",
        credential_revision=0,
        draft_generation=0,
    )
    for model, ids in (("model-a", ("model-a",)), ("model-b", ("model-a",)), ("", ())):
        evidence = ProviderTestEvidence(identity, "reachable", ids, None)
        snapshot = readiness.snapshot(
            selected_model=model, evidence=evidence, current_identity=identity
        )
        assert snapshot.model == model_facet(model, "reachable", ids)

    cloud = get_provider_readiness(
        "openai", {"api_settings": {"openai": {"api_key": "fake-test-key"}}}, environ={}
    )
    keyed = replace(_IDENTITY, credential_source="stored", credential_revision=7)
    listing = ProviderTestEvidence(
        keyed,
        "reachable",
        ("gpt-a",),
        credential="listing_accepted",
        observed_at=datetime(2026, 10, 1, 14, 4).astimezone(),
    )
    words = {}
    for model in ("gpt-a", "gpt-b", ""):
        rows = SettingsScreen._provider_test_rows(
            cloud,
            display_name="OpenAI",
            model=model,
            endpoint="https://api.openai.com/v1",
            evidence=listing,
        )
        snapshot = cloud.snapshot(
            selected_model=model, evidence=listing, current_identity=keyed
        )
        words[model] = dict(rows)["Readiness"]
        assert words[model] == verdict_readiness_words(snapshot, listing)
    assert words == {
        "gpt-a": "Ready · verified 14:04",
        "gpt-b": "Ready · reachable 14:04",
        "": "Not ready · no model",
    }
