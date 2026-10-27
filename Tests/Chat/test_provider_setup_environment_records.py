"""TASK-34100.6 AC#6 (coverage-08): exported keys are recorded at setup.

Setup commits one default provider. Every *other* provider whose key comes from
an exported variable -- no explicit decision in config and no saved key -- is
recorded as ``credential_source = "environment"`` with the variable's name, in
the same atomic write, and never with the variable's value. A record that a
concurrent write made stale is dropped under the config lock; the commit itself
never fails because of it.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import provider_setup_persistence as persistence_module
from tldw_chatbook.Chat.provider_setup_persistence import (
    ProviderSetupDraft,
    ProviderSetupMutation,
    build_provider_setup_mutation,
    environment_credential_records,
    persist_provider_setup,
)

_ANTHROPIC = "sk-ant-" + "a" * 40
_GROQ = "gsk_" + "g" * 40
_OPENAI = "sk-proj-" + "o" * 40
_MISTRAL = "m" * 32


def _record(env_var: str) -> dict[str, str]:
    return {"credential_source": "environment", "api_key_env_var": env_var}


def _plain(records) -> dict[str, dict[str, str]]:
    return {section: dict(values) for section, values in records.items()}


def _openai_draft(**overrides) -> ProviderSetupDraft:
    values = {
        "provider": "openai",
        "model": "gpt-test",
        "endpoint": "",
        "credential_source": "draft",
        "credential_revision": 1,
        "draft_generation": 0,
        "credential_value": _OPENAI,
    }
    values.update(overrides)
    return ProviderSetupDraft(**values)


def test_every_other_exported_key_is_recorded_by_name_only():
    records = environment_credential_records(
        {},
        exclude_provider="openai",
        environ={
            "ANTHROPIC_API_KEY": _ANTHROPIC,
            "GROQ_API_KEY": _GROQ,
            "OPENAI_API_KEY": _OPENAI,
        },
    )

    assert _plain(records) == {
        "api_settings.anthropic": _record("ANTHROPIC_API_KEY"),
        "api_settings.groq": _record("GROQ_API_KEY"),
    }, "the chosen default is the main commit's, never a record"
    rendered = repr(records) + str(_plain(records))
    for secret in (_ANTHROPIC, _GROQ, _OPENAI):
        assert secret not in rendered


def test_a_decision_or_a_saved_key_is_never_overwritten():
    config = {
        "api_settings": {
            # An explicit decision stays whatever it is.
            "anthropic": {"credential_source": "stored", "api_key": "<API_KEY_HERE>"},
            # A saved key outranks the variable in legacy mode: recording the
            # environment would switch which key is sent.
            "groq": {"api_key": "gsk_" + "s" * 40},
            # The shipped placeholder is no key: the variable is what is used.
            "openrouter": {"api_key": "<API_KEY_HERE>"},
            # Already recorded: nothing to write.
            "deepseek": {"credential_source": "environment"},
        }
    }
    records = environment_credential_records(
        config,
        exclude_provider="openai",
        environ={
            "ANTHROPIC_API_KEY": _ANTHROPIC,
            "GROQ_API_KEY": _GROQ,
            "OPENROUTER_API_KEY": "sk-or-" + "r" * 40,
            "DEEPSEEK_API_KEY": "sk-" + "d" * 40,
            "MISTRAL_API_KEY": _MISTRAL,
            "COHERE_API_KEY": "<API_KEY_HERE>",
            "TOGETHER_API_KEY": "",
        },
    )

    assert _plain(records) == {
        # 'mistral' is the legacy alias of 'mistralai': one record, not two.
        "api_settings.mistralai": _record("MISTRAL_API_KEY"),
        "api_settings.openrouter": _record("OPENROUTER_API_KEY"),
    }


def test_the_configured_section_spelling_and_variable_name_are_kept():
    config = {"api_settings": {"Anthropic": {"api_key_env_var": "WORK_ANTHROPIC"}}}

    records = environment_credential_records(
        config,
        exclude_provider="openai",
        environ={"WORK_ANTHROPIC": _ANTHROPIC, "ANTHROPIC_API_KEY": ""},
    )

    # A second 'anthropic' table beside 'Anthropic' would make the provider's
    # settings invalid; the variable the user named is the one recorded.
    assert _plain(records) == {"api_settings.Anthropic": _record("WORK_ANTHROPIC")}


@pytest.mark.parametrize(
    "records",
    [
        {"api_settings.openai": _record("OPENAI_API_KEY")},  # the default itself
        {
            "api_settings.anthropic": {
                **_record("ANTHROPIC_API_KEY"),
                "api_key": _ANTHROPIC,
            }
        },
        {
            "api_settings.anthropic": {
                "credential_source": "stored",
                "api_key_env_var": "X",
            }
        },
        {"api_settings.anthropic": _record("not a name")},
        {"chat_defaults": _record("ANTHROPIC_API_KEY")},
        {
            "api_settings.anthropic": _record("ANTHROPIC_API_KEY"),
            "api_settings.Anthropic": _record("ANTHROPIC_API_KEY"),
        },
    ],
    ids=["default", "secret", "stored", "bad-name", "not-a-provider", "twice"],
)
def test_a_mutation_refuses_any_record_that_is_not_a_bare_environment_choice(records):
    valid = build_provider_setup_mutation(_openai_draft(), {})

    with pytest.raises(ValueError):
        ProviderSetupMutation(
            section_values=valid.section_values,
            delete_keys=valid.delete_keys,
            semantic_identity=valid.semantic_identity,
            environment_records=MappingProxyType(
                {key: MappingProxyType(value) for key, value in records.items()}
            ),
        )


def test_only_a_setup_that_asks_carries_records():
    environ = {"ANTHROPIC_API_KEY": _ANTHROPIC}

    plain = build_provider_setup_mutation(_openai_draft(), {})
    recorded = build_provider_setup_mutation(
        _openai_draft(), {}, record_environment_credentials=True, environ=environ
    )

    assert dict(plain.environment_records) == {}, "Settings' saves are unchanged"
    assert _plain(recorded.environment_records) == {
        "api_settings.anthropic": _record("ANTHROPIC_API_KEY")
    }
    assert "api_settings.anthropic" not in recorded.section_values
    assert _ANTHROPIC not in repr(recorded)


def _clear_provider_environment(monkeypatch) -> None:
    from tldw_chatbook.Chat.provider_setup_facts import provider_setup_facts

    for provider_key in persistence_module._CANONICAL_PROVIDER_KEYS:
        env_var = provider_setup_facts(provider_key).env_var
        if env_var:
            monkeypatch.delenv(env_var, raising=False)


@pytest.mark.asyncio
@private_profile_test
def test_the_wizard_commit_records_exported_keys_in_its_one_write(request, monkeypatch):
    import tomllib

    import toml

    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC)
    monkeypatch.setenv("GROQ_API_KEY", _GROQ)
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.write_text(
        toml.dumps({"api_settings": {"groq": {"model": "kept-model"}}}),
        encoding="utf-8",
    )
    writes: list[dict] = []
    real_writer = persistence_module.apply_settings_mutation_to_cli_config

    def counting_writer(section_values, **kwargs):
        writes.append(dict(section_values))
        return real_writer(section_values, **kwargs)

    monkeypatch.setattr(
        persistence_module, "apply_settings_mutation_to_cli_config", counting_writer
    )
    draft = wizard_state.FirstRunProviderDraft(
        "openai", "", wizard_state.ProviderCredentialDraft("draft", _OPENAI, 1)
    )
    mutation = wizard_state.build_first_run_provider_commit(
        draft, "gpt-test", config_module.get_atomic_config_snapshot().values
    )

    result = persist_provider_setup(mutation)

    assert result.fully_applied
    assert len(writes) == 1, "the records ride the commit's one atomic write"
    text = config_path.read_text(encoding="utf-8")
    saved = tomllib.loads(text)["api_settings"]
    assert saved["anthropic"] == _record("ANTHROPIC_API_KEY")
    assert saved["groq"] == {"model": "kept-model", **_record("GROQ_API_KEY")}
    assert saved["openai"]["credential_source"] == "stored"
    assert _ANTHROPIC not in text and _GROQ not in text, "no exported secret is written"


@pytest.mark.asyncio
@private_profile_test
def test_a_record_made_stale_under_the_lock_is_dropped_never_the_commit(
    request, monkeypatch
):
    import tomllib

    import toml

    _clear_provider_environment(monkeypatch)
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.write_text(toml.dumps({"api_settings": {}}), encoding="utf-8")
    mutation = build_provider_setup_mutation(
        _openai_draft(),
        config_module.get_atomic_config_snapshot().values,
        record_environment_credentials=True,
        environ={"ANTHROPIC_API_KEY": _ANTHROPIC},
    )
    assert "api_settings.anthropic" in mutation.environment_records
    # Settings (or another window) saves an Anthropic key before the commit.
    saved_key = "sk-ant-" + "s" * 40
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.anthropic": {"api_key": saved_key}}
    ).fully_applied

    result = persist_provider_setup(mutation)

    assert result.fully_applied, "a stale record never fails the commit"
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))["api_settings"]
    assert saved["anthropic"] == {"api_key": saved_key}, "the saved key still wins"
    assert saved["openai"]["credential_source"] == "stored"
