"""TASK-33001.14: a summary sends the key readiness resolves (ADR-012).

``analyze`` reaches every provider through ``_summarize_via_chat_api``. With
no caller key, its key is the one ``resolve_provider_credential`` returns for
the provider's ``[api_settings]`` table, so ``credential_source`` decides for
a summary as it does for Console readiness. Real loader on a scratch config;
only the HTTP session (KoboldCpp, TabbyAPI), the hosted transport (Moonshot,
Z.ai) or the chat handler (OpenAI, MistralAI) is replaced.
"""

from __future__ import annotations

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Chat import Chat_Functions
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local
from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl
from tldw_chatbook.LLM_Calls import moonshot, zai

_URL = "http://summary.invalid/v1/chat/completions"
_ENV_KEY = "sk-env-canary-33001-14"
#: Per provider: the variable its table names, its legacy ``[API]`` key, the
#: ``analyze`` name, and the header its chat handler sends a key in.
_LOCAL = {
    "koboldcpp": ("KOBOLDCPP_SUMMARY_KEY", "kobold_api_key", "kobold", "X-Api-Key"),
    "tabbyapi": ("TABBYAPI_API_KEY", "tabby_api_key", "tabbyapi", "Authorization"),
}


class _Reply:
    """What both servers answer: Kobold's ``results``, TabbyAPI's ``choices``."""

    status_code = 200
    text = ""

    def json(self):
        return {
            "results": [{"text": "SUMMARY"}],
            "id": "cmpl-test",
            "created": 0,
            "model": "m",
            "object": "chat.completion",
            "usage": {"completion_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "SUMMARY"}}],
        }

    def raise_for_status(self):
        return None


@pytest.fixture
def sent_headers(monkeypatch):
    sent: list[dict] = []

    class Session:
        def mount(self, *_args, **_kwargs):
            return None

        def post(self, url, headers=None, **_kwargs):
            sent.append(dict(headers or {}))
            return _Reply()

    monkeypatch.setattr(local, "create_pooled_default_session", lambda **_kw: Session())
    return sent


def _load(tmp_path, monkeypatch, toml_text: str):
    """Load ``toml_text`` with the real loader; the summarizer and the chat
    handlers then read what it built."""
    path = tmp_path / "summary-credentials.toml"
    path.write_text(toml_text, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    fresh = install_config_source(monkeypatch)
    fresh.load_cli_config_and_ensure_existence(force_reload=True)
    settings = fresh.load_settings(force_reload=True)
    monkeypatch.setattr(sgl, "load_settings", fresh.load_settings, raising=False)
    monkeypatch.setattr(sgl, "get_cli_setting", fresh.get_cli_setting)
    for handler_module in (local, moonshot, zai):
        monkeypatch.setattr(
            handler_module,
            "get_runtime_config_snapshot",
            fresh.get_runtime_config_snapshot,
        )
    return settings


def _local_toml(provider: str, source, table_key: bool, legacy_key: bool) -> str:
    env_var, legacy_name, _, _ = _LOCAL[provider]
    lines = ["[API]"]
    if legacy_key:
        lines.append(f'{legacy_name} = "legacy-key"')
    lines += [
        f"[api_settings.{provider}]",
        f'api_url = "{_URL}"',
        'model = "m"',
        f'api_key_env_var = "{env_var}"',
    ]
    if source:
        lines.append(f'credential_source = "{source}"')
    if table_key:
        lines.append('api_key = "stored-key"')
    # chat_with_tabbyapi reads its URL from the `tabby_api` table spelling.
    lines += ["[api_settings.tabby_api]", f'api_url = "{_URL}"', 'model = "m"']
    return "\n".join(lines) + "\n"


def _summarize_local(provider: str) -> str:
    return sgl.analyze(
        api_name=_LOCAL[provider][2],
        input_data="a chunk of evidence",
        custom_prompt_arg="Summarize.",
        api_key=None,
    )


def _sent_key(headers: list[dict], provider: str):
    assert len(headers) == 1, headers
    value = headers[0].get(_LOCAL[provider][3])
    return value.removeprefix("Bearer ") if value else None


@pytest.mark.parametrize("provider", sorted(_LOCAL))
def test_credential_source_none_sends_no_key_with_the_variable_exported(
    tmp_path, monkeypatch, sent_headers, provider
):
    """AC#1/#3: "none" with the table's variable exported sends no key, as
    readiness resolves none -- fails if the summary's environment rung
    ignores credential_source."""
    env_var = _LOCAL[provider][0]
    settings = _load(tmp_path, monkeypatch, _local_toml(provider, "none", False, False))
    monkeypatch.setenv(env_var, _ENV_KEY)

    result = _summarize_local(provider)

    assert get_provider_readiness(provider, settings).api_key is None
    assert result.strip() == "SUMMARY", result
    assert _sent_key(sent_headers, provider) is None


#: (credential_source, key in the table, key under legacy [API]) -- every
#: pair a Settings save leaves on disk ("none" and "environment" saves
#: delete the table's key; a hand-written one is ADR-012's recorded case).
_ROWS = [
    (None, False, False),
    (None, False, True),
    (None, True, False),
    (None, True, True),
    ("none", False, False),
    ("none", False, True),
    ("stored", False, False),
    ("stored", False, True),
    ("stored", True, False),
    ("stored", True, True),
    ("environment", False, False),
    ("environment", False, True),
]


@pytest.mark.parametrize(
    ("source", "table_key", "legacy_key"),
    _ROWS,
    ids=[
        f"{source or 'no-source'}-{'table' if table else 'no-table'}-"
        f"{'api' if legacy else 'no-api'}"
        for source, table, legacy in _ROWS
    ],
)
@pytest.mark.parametrize("provider", sorted(_LOCAL))
def test_a_summary_sends_the_key_readiness_resolves(
    tmp_path, monkeypatch, sent_headers, provider, source, table_key, legacy_key
):
    """AC#2: for every source and key location, the key a summary puts on the
    wire is the one readiness resolves, with the table's variable unset and
    then exported."""
    env_var = _LOCAL[provider][0]
    settings = _load(
        tmp_path, monkeypatch, _local_toml(provider, source, table_key, legacy_key)
    )
    seen = []
    for exported in (False, True):
        if exported:
            monkeypatch.setenv(env_var, _ENV_KEY)
        else:
            monkeypatch.delenv(env_var, raising=False)
        sent_headers.clear()
        _summarize_local(provider)
        seen.append(
            (
                _sent_key(sent_headers, provider),
                get_provider_readiness(provider, settings).api_key,
            )
        )

    assert [sent for sent, _ in seen] == [ready for _, ready in seen], seen


@pytest.mark.parametrize(
    ("toml_text", "expected"),
    [
        ('[api_settings.openai]\napi_key = "sk-table"\n', "sk-table"),
        ('[API]\nopenai_api_key = "sk-legacy"\n', "sk-legacy"),
        (
            (
                '[API]\nopenai_api_key = "sk-legacy"\n'
                '[api_settings.openai]\ncredential_source = "none"\n'
            ),
            None,
        ),
    ],
    ids=["table-key", "legacy-api-key", "none-with-legacy-api-key"],
)
def test_a_cloud_summary_with_no_caller_key_sends_the_saved_key(
    tmp_path, monkeypatch, toml_text, expected
):
    """A caller that passes no key (web search does) gets the key readiness
    resolves. Before, the summary read only a raw-TOML ``[openai_api]``
    table, which the loader never writes, and refused every such call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _load(tmp_path, monkeypatch, toml_text)
    handed: list = []

    def handler(**kwargs):
        handed.append(kwargs.get("api_key"))
        return {"choices": [{"message": {"content": "SUMMARY"}}]}

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "openai", handler)

    result = sgl.analyze(
        api_name="openai", input_data="body", custom_prompt_arg="Sum.", api_key=None
    )

    assert get_provider_readiness("openai", settings).api_key == expected
    if expected is None:
        assert result == sgl._SUMMARIZER_CHAT_SPECS["openai"].missing_key_message
        assert handed == []
    else:
        assert result == "SUMMARY", result
        assert handed == [expected]


_NAMED_KEY = "sk-named-canary-33001-14"
#: Per hosted provider: the variable its handler falls back to on its own
#: when handed no key.
_HOSTED = {"moonshot": "MOONSHOT_API_KEY", "zai": "ZAI_API_KEY"}


@pytest.fixture
def hosted_wire(monkeypatch):
    """The key each hosted handler hands its transport; nothing is sent."""
    sent: list = []

    def transport(*, config, **_kwargs):
        sent.append(config.api_key)
        raise ChatProviderError(provider=config.provider, message="stubbed")

    monkeypatch.setattr(moonshot, "hosted_chat_request", transport)
    monkeypatch.setattr(zai, "owned_json_post", transport)
    return sent


@pytest.mark.parametrize(
    ("source", "named_exported", "expected"),
    [
        ("none", True, None),
        ("environment", False, None),
        ("environment", True, _NAMED_KEY),
    ],
    ids=["none", "environment-named-unset", "environment-named-exported"],
)
@pytest.mark.parametrize("provider", sorted(_HOSTED))
def test_a_hosted_summary_sends_only_the_key_readiness_resolves(
    tmp_path, monkeypatch, hosted_wire, provider, source, named_exported, expected
):
    """Moonshot and Z.ai handlers search the environment themselves when
    handed no key, so a summary refuses where readiness resolves none: under
    "none", and under "environment" with the named variable unset, the
    conventional variable stays unsent."""
    named = f"{provider.upper()}_SUMMARY_KEY"
    settings = _load(
        tmp_path,
        monkeypatch,
        f'[api_settings.{provider}]\napi_key_env_var = "{named}"\n'
        f'credential_source = "{source}"\n',
    )
    monkeypatch.setenv(_HOSTED[provider], _ENV_KEY)
    if named_exported:
        monkeypatch.setenv(named, _NAMED_KEY)
    else:
        monkeypatch.delenv(named, raising=False)

    result = sgl.analyze(
        api_name=provider, input_data="body", custom_prompt_arg="Sum.", api_key=None
    )

    assert get_provider_readiness(provider, settings).api_key == expected
    assert hosted_wire == ([] if expected is None else [expected])
    if expected is None:
        assert result == sgl._SUMMARIZER_CHAT_SPECS[provider].missing_key_message


@pytest.mark.parametrize(
    ("table", "exported", "expected"),
    [
        ('credential_source = "none"', True, None),
        ('api_key = "sk-mistralai-table"', False, "sk-mistralai-table"),
        ('api_key = "sk-mistralai-table"', True, "sk-mistralai-table"),
    ],
    ids=["none-variable-exported", "stored-key", "stored-key-variable-exported"],
)
def test_a_mistralai_summary_decides_its_key_under_api_settings_mistralai(
    tmp_path, monkeypatch, table, exported, expected
):
    """Settings keeps MistralAI under ``[api_settings.mistralai]``; the
    summary is dispatched to the "mistral" handler, but its key is the one
    readiness resolves for "mistralai"."""
    settings = _load(tmp_path, monkeypatch, f"[api_settings.mistralai]\n{table}\n")
    if exported:
        monkeypatch.setenv("MISTRAL_API_KEY", _ENV_KEY)
    else:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    handed: list = []

    def handler(**kwargs):
        handed.append(kwargs.get("api_key"))
        return {"choices": [{"message": {"content": "SUMMARY"}}]}

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "mistral", handler)

    result = sgl.analyze(
        api_name="mistralai", input_data="body", custom_prompt_arg="Sum.", api_key=None
    )

    assert get_provider_readiness("mistralai", settings).api_key == expected
    if expected is None:
        assert result == sgl._SUMMARIZER_CHAT_SPECS["mistral"].missing_key_message
        assert handed == []
    else:
        assert result == "SUMMARY", result
        assert handed == [expected]
