"""TASK-33005.9: the first-run wizard's store publishes to the shared owner.

The wizard's draft identity counts credential edits, so it is no connection:
its store is given ``credential_of`` and publishes each settled result under
the sent key's digest, the connection the Console, Home and the Library
pre-check key, and reads nothing back.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderTestEvidenceStore,
    connection_credential_revision,
    provider_connection_evidence,
)

_OPENAI = ("openai", "https://api.openai.com/v1/chat/completions")
_REFUSED = ProviderProbeResult("unreachable", (), "connection_refused")


def _wizard_identity(source="environment", revision=3, generation=4):
    """A first-run identity: its revision counts edits, it is no key digest."""
    return ProviderDraftIdentity(
        provider_key="openai",
        connection_identity=_OPENAI,
        credential_source=source,
        credential_revision=revision,
        draft_generation=generation,
    )


def test_a_credential_counting_store_publishes_under_the_sent_keys_digest():
    """TASK-33005.9 AC#2: the wizard's identity counts edits, so its facts go
    to the connection the Console keys -- the sent key's digest -- never to
    the counter (whose 0 would read as the keyless connection)."""
    app = SimpleNamespace()
    identity = _wizard_identity()
    store = ProviderTestEvidenceStore(
        lambda: app, credential_of=lambda: "sk-wizard-env"
    )

    assert store.settle(store.begin(identity), _REFUSED)

    owner = provider_connection_evidence(app)
    connection = replace(
        identity,
        credential_revision=connection_credential_revision("sk-wizard-env"),
        draft_generation=0,
    )
    shared = owner.evidence_for(connection)
    assert shared is not None and shared.endpoint == "unreachable"
    assert shared.category == "connection_refused"
    assert owner.evidence_for(identity) is None


def test_a_credential_counting_store_reads_no_shared_facts_back():
    """Its identities are not connections: after an invalidate, the result it
    published is not resurrected for the identity it tested."""
    app = SimpleNamespace()
    identity = _wizard_identity(revision=0)
    store = ProviderTestEvidenceStore(lambda: app, credential_of=lambda: None)
    assert store.settle(store.begin(identity), _REFUSED)
    assert provider_connection_evidence(app).version == 1

    assert store.invalidate()
    assert store.evidence_for(identity) is None


def test_a_credential_counting_store_that_cannot_read_its_key_keeps_it_private():
    app = SimpleNamespace()

    def unreadable():
        raise LookupError("detached")

    store = ProviderTestEvidenceStore(lambda: app, credential_of=unreadable)
    assert store.settle(store.begin(_wizard_identity()), _REFUSED)
    assert provider_connection_evidence(app).version == 0


def test_a_credential_counting_store_carries_a_saved_typed_key_to_stored():
    """A typed key the wizard then saves is the stored key: Save carries the
    result to the stored connection the Console reads."""
    app = SimpleNamespace()
    tested = _wizard_identity(source="draft", revision=2)
    store = ProviderTestEvidenceStore(lambda: app, credential_of=lambda: "sk-typed")
    assert store.settle(store.begin(tested), _REFUSED)
    saved = replace(tested, credential_source="stored")
    result = type("Applied", (), {"fully_applied": True, "conflict": False})()

    assert store.rebase_after_save(
        tested, saved, result, lease=store.begin_save(tested)
    )

    stored = ProviderDraftIdentity(
        provider_key="openai",
        connection_identity=_OPENAI,
        credential_source="stored",
        credential_revision=connection_credential_revision("sk-typed"),
        draft_generation=0,
    )
    carried = provider_connection_evidence(app).evidence_for(stored)
    assert carried is not None and carried.endpoint == "unreachable"
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4o"),
        app_config={"api_settings": {"openai": {"api_key": "sk-typed"}}},
        environ={},
        connection_evidence=provider_connection_evidence(app),
    )
    assert readiness.blocker == "endpoint_unreachable"
