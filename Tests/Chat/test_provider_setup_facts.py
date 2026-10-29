"""TASK-34100.6: the provider setup facts the setup form and Settings share.

Review structural fix SF5 and finding provider-25 ('Where do I get a key'
existed for 8 of 45 keyed providers, in a private wizard dict). Every keyed
catalog row must have a sourced key URL or sit on the explicit allowlist that
falls back to the account-dashboard copy; never an invented link.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_BASE_URL_KEYS
from tldw_chatbook.Chat.provider_setup_facts import (
    KEY_URL_UNCONFIRMED,
    KEY_URLS,
    key_help_text,
    provider_setup_facts,
)

_CATALOG = settings_provider_catalog()


@pytest.mark.parametrize(
    "entry",
    [entry for entry in _CATALOG if entry.requires_api_key],
    ids=lambda entry: entry.readiness_key,
)
def test_every_keyed_row_has_a_sourced_key_url_or_is_on_the_allowlist(entry):
    facts = entry.setup_facts
    sourced = entry.readiness_key in KEY_URLS
    unconfirmed = entry.readiness_key in KEY_URL_UNCONFIRMED
    assert sourced != unconfirmed, entry.readiness_key
    assert (facts.key_url is not None) == sourced
    text = key_help_text(facts, entry.display_name)
    if sourced:
        assert text == f"New keys: {facts.key_url}."
    else:
        assert text == f"Get a key from your {entry.display_name} account dashboard."


def test_the_allowlist_names_only_keyed_catalog_rows():
    keyed = {entry.readiness_key for entry in _CATALOG if entry.requires_api_key}
    assert KEY_URL_UNCONFIRMED <= keyed
    assert not set(KEY_URLS) - keyed


def test_the_catalog_record_carries_the_facts_the_form_needs():
    by_key = {entry.readiness_key: entry for entry in _CATALOG}
    assert by_key["openai"].setup_facts.env_var == "OPENAI_API_KEY"
    assert by_key["openai"].setup_facts.locality == "cloud"
    assert by_key["ollama"].setup_facts.locality == "local"
    assert by_key["ollama"].setup_facts.local_ports == (11434,)
    assert by_key["zai"].setup_facts.route_suffix == "/chat/completions"
    assert by_key["openrouter"].setup_facts.check_kind == "authenticated"
    assert not by_key["openrouter"].setup_facts.listing_requires_auth
    assert by_key["local_ollama"].setup_facts.legacy
    assert by_key["custom_2"].setup_facts.legacy
    assert not by_key["custom"].setup_facts.legacy
    for key in PROVIDERS_REQUIRING_BASE_URL_KEYS:
        facts = by_key[key].setup_facts
        assert facts.base_url_required and facts.base_url_label
        assert facts.base_url_example.startswith("https://")


def test_the_wizard_reads_the_key_url_from_the_catalog(monkeypatch):
    """provider-25: the Provider step's missing-key line names where to get a
    key, from the catalog facts.

    Settings' half of this test read ``SettingsScreen._provider_credential_status``,
    which dev's TASK-33007.2 removed when it rebuilt Connect as one-row
    field-truth rows (merged 2026-10-19); its width-capped help names no URL.
    """
    from tldw_chatbook.UI.Wizards.first_run_provider_key_entry import key_help_line

    readiness = SimpleNamespace(env_var="OPENAI_API_KEY", reason="Missing API key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    line = key_help_line("openai", readiness).plain
    assert "New keys: platform.openai.com/api-keys." in line
    assert provider_setup_facts("openai").key_url == "platform.openai.com/api-keys"


def test_the_route_suffix_has_one_owner_the_endpoint_contract_reads():
    """SF5 'one owner'; review round 1, F11: the endpoint contract kept its
    own hard-coded {'zai'} beside the facts' ROUTE_SUFFIXES, so the two
    could drift. Every preset the facts give a route suffix resolves its chat
    URL as <saved base> + that suffix, and the contract owns no other."""
    from tldw_chatbook.Chat import provider_endpoint_contract as contract
    from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
    from tldw_chatbook.Chat.provider_setup_facts import ROUTE_SUFFIXES
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    assert set(contract._ROUTE_OWNING_PRESETS) == set(ROUTE_SUFFIXES)
    for key, suffix in ROUTE_SUFFIXES.items():
        assert provider_setup_facts(key).route_suffix == suffix
        base = RECORDS_BY_KEY[key].default_base_url
        assert base, key
        resolution = resolve_provider_endpoint(key, base)
        assert resolution.chat_url == base.rstrip("/") + suffix, key


def test_first_run_hides_exactly_the_aliases_the_console_calls_legacy():
    """Review v3r1, R1-09: the wizard's legacy list repeated the catalog's
    alias set (TASK-33007.9) plus the '#2' slot, with nothing tying the two:
    an alias dev adds later would be hidden in one surface and shown in the
    other. This fails until both lists move together."""
    from tldw_chatbook.Chat.provider_catalog import PROVIDER_LEGACY_ALIAS_KEYS
    from tldw_chatbook.Chat.provider_setup_facts import LEGACY_PROVIDER_KEYS

    assert LEGACY_PROVIDER_KEYS == PROVIDER_LEGACY_ALIAS_KEYS | {"custom_2"}
