"""TASK-33005.3: one readiness vocabulary for every model surface (spec §5).

'Ready · not tested', 'Ready · reachable HH:MM', 'Ready · verified HH:MM' or
'Not ready · <reason>' -- one mapping owns the words, and a key accepted by a
model listing is recorded apart from a successful paid generation.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import get_args

import pytest

import tldw_chatbook.Chat.provider_test_evidence as evidence_module
from tldw_chatbook.Chat import console_session_settings as session_settings
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsBlockerCode,
    ConsoleSettingsReadiness,
    build_console_settings_readiness,
    readiness_words,
    verdict_readiness_words,
)
from tldw_chatbook.Chat.provider_endpoint_contract import (
    canonical_connection_identity,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    EndpointFailureCategory,
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderGenerationProbeResult,
    ProviderProbeResult,
    ProviderReadinessSnapshot,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
    ReadinessVerdictCode,
    connection_credential_revision,
    provider_connection_evidence,
)

WORD = re.compile(
    r"Ready · not tested"
    r"|Ready · (reachable|verified) \d\d:\d\d"
    r"|Not ready · [a-z][a-zA-Z :0-9]*"
)
#: An aware LOCAL time, so the expected HH:MM never depends on the machine's zone.
SEEN = datetime(2026, 10, 1, 9, 7).astimezone()
OPENAI_KEY = "sk-test-openai-readiness-words"
OPENAI_CONFIG = {"api_settings": {"openai": {"api_key": OPENAI_KEY}}}
LLAMA_CONFIG = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}


def _identity(provider: str, endpoint: str, key: str | None = None):
    return ProviderDraftIdentity(
        provider_key=provider,
        connection_identity=canonical_connection_identity(provider, endpoint),
        credential_source="stored" if key else "none",
        credential_revision=connection_credential_revision(key),
        draft_generation=0,
    )


LLAMA = _identity("llama_cpp", "http://127.0.0.1:9099")
OPENAI = _identity("openai", "https://api.openai.com/v1", OPENAI_KEY)


def _console(
    provider: str, config, evidence: ProviderTestEvidence
) -> ConsoleSettingsReadiness:
    owner = ProviderConnectionEvidence()
    owner.publish(evidence, order=1)
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider=provider, model="model-a"),
        app_config=config,
        environ={},
        connection_evidence=owner,
    )
    assert readiness.connection == evidence.identity  # The build HIT the record.
    return readiness


def test_every_blocker_failure_and_verdict_code_reads_as_exactly_one_word():
    """AC#7: one mapping owns the words; a new code without one fails here."""
    reasons = session_settings._NOT_READY_REASONS
    failures = session_settings._FAILURE_REASONS
    verdicts = session_settings._VERDICT_READINESS

    assert set(reasons) == set(get_args(ConsoleSettingsBlockerCode))
    assert set(failures) == set(get_args(EndpointFailureCategory))
    assert set(verdicts) == set(get_args(ReadinessVerdictCode))
    for reason in (*reasons.values(), *failures.values()):
        assert WORD.fullmatch(f"Not ready · {reason}"), reason
    for code, reads in verdicts.items():
        assert reads in {"evidence", None} or reads in reasons, code
    # Longest reason (with a 5-digit port) fits the switcher's 28-cell column.
    assert max(len(f"Not ready · {r}") for r in reasons.values()) <= 28
    assert len("Not ready · refused :65535") <= 28


def test_config_only_readiness_reads_not_tested_or_a_blocker():
    ready = ConsoleSettingsReadiness("Ready", "Ready.", True)
    no_key = ConsoleSettingsReadiness(
        "Not ready",
        "API key missing",
        False,
        operability="not_ready",
        blocker="credential_missing",
        recovery_action="configure_credential",
        configuration="incomplete",
        configuration_issue="credential_missing",
        credential="missing",
    )

    assert readiness_words(ready) == "Ready · not tested"
    assert readiness_words(no_key) == "Not ready · no key"
    # Facets without an observed connection prove nothing (P4's pin, kept).
    assert readiness_words(replace(ready, endpoint="reachable")) == (
        "Ready · not tested"
    )


@pytest.mark.parametrize(
    ("category", "word"),
    [
        ("connection_refused", "Not ready · refused :9099"),
        ("timeout", "Not ready · timed out"),
        ("unauthorized", "Not ready · key rejected"),
        ("forbidden", "Not ready · key rejected"),
    ],
)
def test_a_known_failure_names_its_reason_and_only_the_port(category, word):
    readiness = _console(
        "llama_cpp",
        LLAMA_CONFIG,
        ProviderTestEvidence(LLAMA, "unreachable", (), category, observed_at=SEEN),
    )

    assert readiness_words(readiness) == word
    assert "127.0.0.1" not in word


def test_a_local_endpoint_that_listed_models_reads_reachable_at_the_observed_time():
    """AC#3/#9: local endpoint, its listing answered, at the local time seen."""
    readiness = _console(
        "llama_cpp",
        LLAMA_CONFIG,
        ProviderTestEvidence(LLAMA, "reachable", ("model-a",), observed_at=SEEN),
    )

    assert readiness_words(readiness) == "Ready · reachable 09:07"


def test_a_cloud_key_accepted_by_its_listing_reads_verified():
    """AC#2/#6: an authenticated listing verifies the key, not generation."""
    readiness = _console(
        "openai",
        OPENAI_CONFIG,
        ProviderTestEvidence(
            OPENAI,
            "reachable",
            ("model-a",),
            credential="listing_accepted",
            observed_at=SEEN,
        ),
    )

    assert readiness.credential == "listing_accepted"
    assert readiness.generation == "not_tested"
    assert readiness_words(readiness) == "Ready · verified 09:07"


def test_a_public_cloud_listing_without_key_acceptance_stays_not_tested():
    """AC#4: OpenRouter's catalog needs no key, so it proves nothing about one."""
    key = "sk-or-test-readiness-words"
    identity = _identity("openrouter", "https://openrouter.ai/api/v1", key)
    readiness = _console(
        "openrouter",
        {"api_settings": {"openrouter": {"api_key": key}}},
        ProviderTestEvidence(identity, "reachable", ("model-a",), observed_at=SEEN),
    )

    assert readiness.endpoint == "reachable"
    assert readiness_words(readiness) == "Ready · not tested"


def test_a_url_endpoint_reads_reachable_even_when_its_listing_took_a_key():
    """Ruling 9: a self-hosted server may ignore auth, so it is never verified."""
    key = "sk-vllm-readiness-words"
    identity = _identity("vllm", "http://127.0.0.1:8000", key)
    readiness = _console(
        "vllm",
        {"api_settings": {"vllm": {"api_url": "http://127.0.0.1:8000", "api_key": key}}},
        ProviderTestEvidence(
            identity,
            "reachable",
            ("model-a",),
            credential="listing_accepted",
            observed_at=SEEN,
        ),
    )

    assert readiness_words(readiness) == "Ready · reachable 09:07"


def test_a_successful_paid_test_reads_verified():
    """AC#2: a successful paid generation is the other source of 'verified'."""
    readiness = _console(
        "llama_cpp",
        LLAMA_CONFIG,
        # Review round 1 (rewritten on purpose): the paid test's own time.
        # Qodo #2958 (rewritten on purpose): and the model it tested.
        ProviderTestEvidence(
            LLAMA,
            "not_tested",
            (),
            generation="succeeded",
            generation_model="model-a",
            generation_observed_at=SEEN,
        ),
    )

    assert readiness_words(readiness) == "Ready · verified 09:07"


def test_listing_acceptance_is_recorded_apart_from_generation(monkeypatch):
    """AC#6/#9: the listing records 'listing_accepted' at the local settle
    time; only a successful paid generation makes it 'authenticated'."""
    monkeypatch.setattr(evidence_module, "_local_now", lambda: SEEN)
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    owner = provider_connection_evidence(app)

    store.settle(
        store.begin(OPENAI),
        ProviderProbeResult("reachable", ("model-a",), key_accepted=True),
    )
    listed = owner.evidence_for(OPENAI)
    assert (listed.credential, listed.generation) == ("listing_accepted", "not_tested")
    assert listed.observed_at == SEEN

    store.settle_generation(
        store.begin_generation(OPENAI), ProviderGenerationProbeResult("succeeded")
    )
    assert owner.evidence_for(OPENAI).credential == "authenticated"

    # A later refused listing keeps the paid result it did not observe.
    store.settle(store.begin(OPENAI), ProviderProbeResult("unreachable", (), "timeout"))
    assert owner.evidence_for(OPENAI).credential == "authenticated"


def test_a_refused_listing_replaces_an_earlier_acceptance():
    owner = ProviderConnectionEvidence()
    owner.publish(
        ProviderTestEvidence(OPENAI, "reachable", (), credential="listing_accepted"),
        order=1,
    )
    owner.publish(ProviderTestEvidence(OPENAI, "unreachable", (), "unauthorized"), order=2)

    assert owner.evidence_for(OPENAI).credential == "present_unverified"


def test_listing_acceptance_needs_a_key_and_an_answer():
    with pytest.raises(ValueError):
        ProviderProbeResult("unreachable", (), "timeout", key_accepted=True)
    with pytest.raises(ValueError):
        ProviderTestEvidence(
            OPENAI, "unreachable", (), "timeout", credential="listing_accepted"
        )
    keyless = ProviderTestEvidence(LLAMA, "reachable", (), credential="listing_accepted")
    assert keyless.credential == "not_required"


def _snapshot(**facets) -> ProviderReadinessSnapshot:
    return ProviderReadinessSnapshot(
        **{"configuration": "configured", "endpoint": "not_tested", "model": "unconfirmed", **facets}
    )


@pytest.mark.parametrize(
    ("snapshot", "evidence", "word"),
    [
        (_snapshot(), None, "Ready · not tested"),
        (
            _snapshot(configuration="incomplete", configuration_issue="credential_missing"),
            None,
            "Not ready · no key",
        ),
        (_snapshot(model="missing"), None, "Not ready · no model"),
        (
            _snapshot(endpoint="unreachable", category="connection_refused"),
            ProviderTestEvidence(LLAMA, "unreachable", (), "connection_refused"),
            "Not ready · refused :9099",
        ),
        (
            _snapshot(endpoint="reachable", model="confirmed"),
            ProviderTestEvidence(LLAMA, "reachable", ("model-a",), observed_at=SEEN),
            "Ready · reachable 09:07",
        ),
        (
            _snapshot(endpoint="changed_since_test"),
            ProviderTestEvidence(LLAMA, "reachable", ("model-a",), observed_at=SEEN),
            "Ready · not tested",
        ),
    ],
)
def test_setup_verdicts_read_the_same_words(snapshot, evidence, word):
    """AC#7/#8: Settings' test result reads its verdict through the same map."""
    assert verdict_readiness_words(snapshot, evidence) == word


def test_the_settings_verdict_and_the_console_agree_for_one_connection():
    evidence = ProviderTestEvidence(
        LLAMA, "unreachable", (), "connection_refused", observed_at=SEEN
    )
    console = _console("llama_cpp", LLAMA_CONFIG, evidence)
    setup = verdict_readiness_words(
        _snapshot(endpoint="unreachable", category="connection_refused"), evidence
    )

    assert readiness_words(console) == setup == "Not ready · refused :9099"


@pytest.mark.parametrize(
    ("status", "reason"),
    [("pending", "checking login"), ("expired", "login expired"), ("missing", "no login")],
)
def test_a_claude_subscription_names_its_login_not_a_key(status, reason):
    readiness = ConsoleSettingsReadiness(
        "Missing key",
        "",
        False,
        operability="not_ready",
        blocker="credential_missing",
        recovery_action="configure_credential",
        configuration="incomplete",
        configuration_issue="credential_missing",
        credential="missing",
        subscription_status=status,
    )

    assert readiness_words(readiness) == f"Not ready · {reason}"


def test_words_never_claim_generation_success():
    """AC#5: 'verified'/'reachable' are the only qualifiers; nothing says sent."""
    for word in (
        readiness_words(
            _console(
                "openai",
                OPENAI_CONFIG,
                ProviderTestEvidence(
                    OPENAI, "reachable", (), credential="listing_accepted", observed_at=SEEN
                ),
            )
        ),
        readiness_words(
            _console(
                "llama_cpp",
                LLAMA_CONFIG,
                ProviderTestEvidence(LLAMA, "reachable", (), observed_at=SEEN),
            )
        ),
    ):
        assert not re.search(r"generat|sent|works|succe", word, re.IGNORECASE), word


# -- Review round 1 -----------------------------------------------------------


@pytest.mark.parametrize("category", ["connection_refused", "timeout", "unauthorized"])
def test_a_missing_model_outranks_a_failed_listing_on_every_surface(category):
    """AC#8 (review round 1): Settings ranked the failed listing first and the
    Console the missing model, so one connection read two words. Both now
    follow the Console's blocker precedence."""
    evidence = ProviderTestEvidence(LLAMA, "unreachable", (), category, observed_at=SEEN)
    owner = ProviderConnectionEvidence()
    owner.publish(evidence, order=1)
    console = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        app_config=LLAMA_CONFIG,
        environ={},
        connection_evidence=owner,
    )
    setup = verdict_readiness_words(
        _snapshot(endpoint="unreachable", category=category, model="missing"), evidence
    )

    assert console.blocker == "model_missing"
    assert readiness_words(console) == setup == "Not ready · no model"


@pytest.mark.parametrize(
    ("facets", "word"),
    [
        ({"endpoint": "testing", "model": "missing"}, "Not ready · no model"),
        (
            {"endpoint": "model_listing_unavailable", "model": "missing"},
            "Not ready · no model",
        ),
        ({"endpoint": "changed_since_test", "model": "missing"}, "Not ready · no model"),
        (
            {
                "endpoint": "changed_since_test",
                "configuration": "incomplete",
                "configuration_issue": "credential_missing",
            },
            "Not ready · no key",
        ),
    ],
)
def test_a_blocker_is_never_hidden_by_a_running_or_stale_test(facets, word):
    """TASK-30011 AC#2: Ready means no known blocker, whatever the test state."""
    assert verdict_readiness_words(_snapshot(**facets)) == word


def test_reachable_keeps_the_listing_time_when_a_later_generation_fails(monkeypatch):
    """AC#9 (review round 1): a generation settle restamped the one time, so
    a listing answered at 09:00 read 'reachable 09:30' -- the moment a
    generation then failed to connect."""
    times = iter(
        [datetime(2026, 10, 1, 9, 0).astimezone(), datetime(2026, 10, 1, 9, 30).astimezone()]
    )
    monkeypatch.setattr(evidence_module, "_local_now", lambda: next(times))
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    store.settle(store.begin(LLAMA), ProviderProbeResult("reachable", ("model-a",)))
    store.settle_generation(
        store.begin_generation(LLAMA),
        ProviderGenerationProbeResult("failed", "connection_error"),
    )
    shared = provider_connection_evidence(app)
    console = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        app_config=LLAMA_CONFIG,
        environ={},
        connection_evidence=shared,
    )
    setup = verdict_readiness_words(
        _snapshot(endpoint="reachable", model="confirmed"), store.evidence_for(LLAMA)
    )

    assert readiness_words(console) == setup == "Ready · reachable 09:00"


def test_verified_by_a_paid_test_reads_the_time_of_that_test(monkeypatch):
    times = iter(
        [datetime(2026, 10, 1, 9, 0).astimezone(), datetime(2026, 10, 1, 9, 30).astimezone()]
    )
    monkeypatch.setattr(evidence_module, "_local_now", lambda: next(times))
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    store.settle(store.begin(LLAMA), ProviderProbeResult("reachable", ("model-a",)))
    store.settle_generation(
        # Qodo #2958 (rewritten on purpose): the test names its model.
        store.begin_generation(LLAMA, model="model-a"),
        ProviderGenerationProbeResult("succeeded"),
    )
    console = _console("llama_cpp", LLAMA_CONFIG, store.evidence_for(LLAMA))

    assert readiness_words(console) == "Ready · verified 09:30"
    assert provider_connection_evidence(app).evidence_for(LLAMA).observed_at.hour == 9
    assert provider_connection_evidence(app).evidence_for(LLAMA).observed_at.minute == 0


@pytest.mark.parametrize(
    ("provider", "identity", "config", "listing", "other"),
    (
        # Keyless: the listing still reads "reachable" for the untested model.
        ("llama_cpp", LLAMA, LLAMA_CONFIG, {}, "Ready · reachable 09:00"),
        # Keyed: the accepted key listing still reads "verified" at its time.
        ("openai", OPENAI, OPENAI_CONFIG, {"key_accepted": True}, "Ready · verified 09:00"),
    ),
)
def test_a_paid_test_verifies_only_the_model_it_tested(
    monkeypatch, provider, identity, config, listing, other
):
    """Qodo #2958 finding 1: a paid test of model-a read model-b, on the same
    provider, endpoint and key, as "verified" at the test's time. The test
    belongs to its model; the listing and the key check do not."""
    times = iter(
        [datetime(2026, 10, 1, 9, 0).astimezone(), datetime(2026, 10, 1, 9, 30).astimezone()]
    )
    monkeypatch.setattr(evidence_module, "_local_now", lambda: next(times))
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    store.settle(
        store.begin(identity),
        ProviderProbeResult("reachable", ("model-a", "model-b"), **listing),
    )
    store.settle_generation(
        store.begin_generation(identity, model="model-a"),
        ProviderGenerationProbeResult("succeeded"),
    )
    owner = provider_connection_evidence(app)

    def word(model: str) -> str:
        return readiness_words(
            build_console_settings_readiness(
                ConsoleSessionSettings(provider=provider, model=model),
                app_config=config,
                environ={},
                connection_evidence=owner,
            )
        )

    assert word("model-a") == "Ready · verified 09:30"
    assert word("model-b") == other


def test_a_public_listing_never_records_an_accepted_key():
    """AC#4 (review round 1): held by the record, not by producer discipline.
    OpenRouter's /models answers any key, so key_accepted proves nothing."""
    key = "sk-or-test-readiness-words"
    identity = _identity("openrouter", "https://openrouter.ai/api/v1", key)
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    store.settle(
        store.begin(identity),
        ProviderProbeResult("reachable", ("model-a",), key_accepted=True),
    )
    readiness = _console(
        "openrouter",
        {"api_settings": {"openrouter": {"api_key": key}}},
        store.evidence_for(identity),
    )

    assert store.evidence_for(identity).credential == "present_unverified"
    assert readiness.credential == "present_unverified"
    assert readiness_words(readiness) == "Ready · not tested"
