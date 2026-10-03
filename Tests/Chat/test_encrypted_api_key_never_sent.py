"""An ``enc:`` config value is never sent as a credential (TASK-34100.4).

new-protect-summary-03: a locked session (encryption on, no password) or a
value that did not decrypt keeps its ``enc:`` ciphertext in the loaded config.
Provider handlers whose fallback read ``api_key`` raw from config, and
``chat_api_call`` itself, would hand that ciphertext to the provider as a
bearer token -- giving the operator of a remote OpenAI-compatible host material
to attack the master password offline.

Every transport these handlers use is replaced by a recorder
(``create_default_session`` is patched on EVERY module that binds it -- a
patch on its home module alone misses each ``from``-import), so a test fails
for the right reason: a credential header carrying ciphertext reached the
wire, or the refusal was something other than the missing-key one.
"""

from __future__ import annotations

import importlib
import re
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatAuthenticationError, ChatConfigurationError

CIPHERTEXT = "enc:AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
MESSAGES = [{"role": "user", "content": "hello"}]
REFUSALS = (ChatConfigurationError, ChatAuthenticationError)
MISSING_KEY = re.compile(r"api.?key", re.IGNORECASE)
LOCAL_URL = "http://127.0.0.1:9/v1"

#: Every module that binds ``create_default_session`` by name for a handler
#: under test.
TRANSPORT_MODULES = (
    "tldw_chatbook.LLM_Calls.hosted_chat",
    "tldw_chatbook.LLM_Calls.LLM_API_Calls",
    "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local",
    "tldw_chatbook.LLM_Calls.Local_Summarization_Lib",
    "tldw_chatbook.LLM_Calls.Summarization_General_Lib",
)


class RequestBuilt(Exception):
    """A provider request reached the (recorded) transport."""


@pytest.fixture
def sent(monkeypatch):
    """Record the headers of every request the handlers try to send."""
    requests_sent: list[dict] = []

    class RecordingSession:
        def mount(self, *_args, **_kwargs):
            return None

        def post(self, url, headers=None, **_kwargs):
            requests_sent.append(dict(headers or {}))
            raise RequestBuilt(url)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    for module_name in TRANSPORT_MODULES:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(
            module, "create_default_session", lambda *a, **k: RecordingSession()
        )
    return requests_sent


def _ciphertext_credentials(requests_sent: list[dict]) -> list[str]:
    """Header names whose value carries the ciphertext."""
    return [
        name
        for headers in requests_sent
        for name, value in headers.items()
        if "enc:" in str(value)
    ]


def _outcome(call):
    """Run ``call`` and return what it raised (or None)."""
    try:
        call()
    except Exception as error:  # noqa: BLE001 -- the outcome is asserted
        return error
    return None


def _assert_missing_key_refusal(error, requests_sent):
    assert not _ciphertext_credentials(requests_sent), (
        "ciphertext reached the wire as a credential"
    )
    assert isinstance(error, REFUSALS), repr(error)
    assert MISSING_KEY.search(str(error)), str(error)
    assert not requests_sent


def _snapshot(provider: str):
    return lambda: SimpleNamespace(
        values={"api_settings": {provider: {"api_key": CIPHERTEXT}}}
    )


@pytest.mark.parametrize(
    ("module_name", "function_name", "provider"),
    [
        ("tldw_chatbook.LLM_Calls.deepseek", "chat_with_deepseek", "deepseek"),
        ("tldw_chatbook.LLM_Calls.groq", "chat_with_groq", "groq"),
        ("tldw_chatbook.LLM_Calls.mistral", "chat_with_mistral", "mistral"),
        ("tldw_chatbook.LLM_Calls.openrouter", "chat_with_openrouter", "openrouter"),
        ("tldw_chatbook.LLM_Calls.LLM_API_Calls", "chat_with_cohere", "cohere"),
    ],
)
def test_snapshot_backed_handlers_refuse_ciphertext(
    monkeypatch, sent, module_name, function_name, provider
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "get_runtime_config_snapshot", _snapshot(provider))

    error = _outcome(lambda: getattr(module, function_name)(MESSAGES, api_key=None, model="m"))

    _assert_missing_key_refusal(error, sent)


def test_openai_refuses_ciphertext_from_either_config_table(monkeypatch, sent):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls._provider_recovery, "recovered_settings", lambda: None
    )
    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {
            "openai_api": {"api_key": CIPHERTEXT},
            "api_settings": {"openai": {"api_key": CIPHERTEXT}},
        },
    )

    error = _outcome(lambda: LLM_API_Calls.chat_with_openai.__wrapped__(MESSAGES, api_key=None))

    _assert_missing_key_refusal(error, sent)


def test_anthropic_refuses_ciphertext(monkeypatch, sent):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"anthropic_api": {"api_key": CIPHERTEXT}},
    )

    error = _outcome(lambda: LLM_API_Calls.chat_with_anthropic(MESSAGES, api_key=None, model="m"))

    _assert_missing_key_refusal(error, sent)


def test_huggingface_sends_no_ciphertext(monkeypatch, sent):
    # The key is optional for HuggingFace (public Inference API / open TGI):
    # ciphertext reads as no key, so the request goes out with no
    # Authorization header rather than with the ciphertext in it.
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {
            "huggingface_api": {
                "api_key": CIPHERTEXT,
                "model": "m",
                "api_base_url": LOCAL_URL,
            }
        },
    )

    _outcome(lambda: LLM_API_Calls.chat_with_huggingface(MESSAGES, api_key=None, model="m"))

    assert sent, "the handler never reached its transport"
    assert not _ciphertext_credentials(sent)


def _local_config() -> dict:
    """Every section a local/OpenAI-compatible handler reads, key encrypted."""
    section = {
        "api_key": CIPHERTEXT,
        "api_url": LOCAL_URL,
        "api_ip": LOCAL_URL,
        "model": "m",
    }
    names = (
        "llama_cpp",
        "koboldcpp",
        "ooba_api",
        "tabby_api",
        "vllm_api",
        "aphrodite_api",
        "ollama",
        "custom",
        "custom_openai_api_2",
    )
    return {
        "api_settings": {name: dict(section) for name in names},
        "custom_openai_api_2": dict(section),
    }


@pytest.mark.parametrize(
    "function_name",
    [
        "chat_with_llama",
        "chat_with_oobabooga",
        "chat_with_tabbyapi",
        "chat_with_vllm",
        "chat_with_aphrodite",
        "chat_with_ollama",
    ],
)
def test_local_handlers_never_forward_ciphertext(monkeypatch, sent, function_name):
    # Review round 1 (TASK-34100.4, F1/F2): these handlers' raw
    # `api_key or cfg.get("api_key")` fallback forwarded ciphertext -- and
    # chat_api_call dropping an `enc:` argument only sent them to that
    # fallback. The key is optional for a local server, so ciphertext must
    # read as NO key.
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    config = _local_config()
    monkeypatch.setattr(
        local, "get_runtime_config_snapshot", lambda: SimpleNamespace(values=config)
    )
    monkeypatch.setattr(local, "load_settings", lambda: config)
    forwarded: list = []

    def capture(**kwargs):
        forwarded.append(kwargs.get("api_key"))
        return "captured"

    monkeypatch.setattr(local, "_chat_with_openai_compatible_local_server", capture)

    error = _outcome(
        lambda: getattr(local, function_name)(MESSAGES, api_key=None, model="m")
    )

    assert forwarded, repr(error)
    assert forwarded[0] is None or "enc:" not in str(forwarded[0]), (
        "ciphertext forwarded to the request builder"
    )
    assert not _ciphertext_credentials(sent)


def test_kobold_never_sends_ciphertext_as_x_api_key(monkeypatch, sent):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    config = _local_config()
    monkeypatch.setattr(
        local, "get_runtime_config_snapshot", lambda: SimpleNamespace(values=config)
    )

    _outcome(lambda: local.chat_with_kobold(MESSAGES, api_key=None, model="m"))

    assert sent, "the handler never reached its transport"
    assert not _ciphertext_credentials(sent)
    assert "X-Api-Key" not in sent[0]


def test_custom_openai_2_refuses_ciphertext(monkeypatch, sent):
    # custom_openai_2 often points at a REMOTE OpenAI-compatible host and
    # requires a key: ciphertext is the missing-key refusal.
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    config = _local_config()
    monkeypatch.setattr(local, "load_settings", lambda: config)
    forwarded: list = []
    monkeypatch.setattr(
        local,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: forwarded.append(kwargs.get("api_key")) or "captured",
    )

    error = _outcome(lambda: local.chat_with_custom_openai_2(MESSAGES, api_key=None, model="m"))

    assert not forwarded, "ciphertext forwarded to the request builder"
    _assert_missing_key_refusal(error, sent)


SUMMARIZERS = (
    "openai",
    "anthropic",
    "cohere",
    "google",
    "groq",
    "huggingface",
    "openrouter",
    "deepseek",
    "mistral",
    "llama.cpp",
    "kobold",
    "ooba",
    "tabbyapi",
    "vllm",
    "custom-openai-api",
    "custom-openai-api-2",
    "ollama",
)


def _summarization_config() -> dict:
    # Every field these summarizers index directly, so each one reaches its
    # transport (or its missing-key refusal) instead of dying on a KeyError
    # first -- an early KeyError would make the ciphertext assertion vacuous.
    section = {
        "api_key": CIPHERTEXT,
        "api_url": LOCAL_URL,
        "api_ip": LOCAL_URL,
        "api_base_url": LOCAL_URL,
        "model": "m",
        "api_retries": 0,
        "api_retry_delay": 0,
        "api_timeout": 5,
        "max_tokens": 16,
        "temperature": 0.1,
        "streaming": False,
    }
    legacy = (
        "openai_api",
        "anthropic_api",
        "cohere_api",
        "google_api",
        "groq_api",
        "huggingface_api",
        "openrouter_api",
        "deepseek_api",
        "mistral_api",
        "llama_cpp_api",
        "llama_api",
        "kobold_api",
        "ooba_api",
        "tabby_api",
        "vllm_api",
        "ollama_api",
        "custom_openai_api",
        "custom_openai_api_2",
    )
    modern = (
        "llama_cpp",
        "koboldcpp",
        "tabbyapi",
        "tabby_api",
        "vllm",
        "ollama",
        "custom",
    )
    config = {name: dict(section) for name in legacy}
    config["api_settings"] = {name: dict(section) for name in modern}
    return config


@pytest.mark.parametrize("passed_key", [None, CIPHERTEXT], ids=["config", "argument"])
@pytest.mark.parametrize("api_name", SUMMARIZERS)
def test_summarizers_never_send_ciphertext(monkeypatch, sent, api_name, passed_key):
    # Review round 1: the summarization libraries read keys raw from config
    # (and accept a raw `api_key` argument) on a path chat_api_call never
    # screens. Ciphertext from either source must never reach the wire.
    from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as local_lib
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as general_lib

    config = _summarization_config()

    def setting(section, key=None, default=None):
        value = config.get(section, {}).get(key)
        return default if value is None else value

    monkeypatch.setattr(local_lib, "load_settings", lambda: config)
    for module in (local_lib, general_lib):
        monkeypatch.setattr(module, "get_cli_setting", setting)
    # The TLS policy reads the selected profile; it is not under test here.
    monkeypatch.setattr(general_lib, "requests_verify", lambda: True)

    result = general_lib._dispatch_to_api(
        "text to summarize", "summarize", api_name, passed_key, 0.1, None, False
    )

    assert not _ciphertext_credentials(sent), (
        f"{api_name}: ciphertext reached the wire as a credential"
    )
    # Not vacuous: the summarizer either reached its transport (with no
    # ciphertext, asserted above) or refused for the missing key.
    assert sent or MISSING_KEY.search(str(result)), (
        f"{api_name}: neither sent nor refused for the key: {str(result)[:200]}"
    )


def test_chat_api_call_drops_a_ciphertext_api_key(monkeypatch):
    from tldw_chatbook.Chat import Chat_Functions

    received = {}

    def capture(**kwargs):
        received.update(kwargs)
        return "ok"

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "openai", capture)

    assert (
        Chat_Functions.chat_api_call("openai", MESSAGES, api_key=CIPHERTEXT) == "ok"
    )
    assert received.get("api_key") is None


def test_openai_embeddings_refuse_ciphertext(monkeypatch, sent):
    # Review round 1 survey: the embeddings helper read the key raw.
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls, "load_settings", lambda: {"openai_api": {"api_key": CIPHERTEXT}}
    )

    error = _outcome(lambda: LLM_API_Calls.get_openai_embeddings("text", "m"))

    assert not _ciphertext_credentials(sent), "ciphertext reached the wire"
    assert isinstance(error, ValueError) and MISSING_KEY.search(str(error)), repr(error)
    assert not sent


@pytest.mark.parametrize("prefix_case", ["enc:", "ENC:"])
def test_recovered_openai_selection_refuses_ciphertext(tmp_path, prefix_case):
    # Review round 1 survey: the recovered-settings check refused "ENC:" but
    # not the engine's real lowercase "enc:" prefix.
    import toml

    from tldw_chatbook.LLM_Calls import recovery_review

    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    path = home / "config.toml"
    value = prefix_case + CIPHERTEXT.removeprefix("enc:")
    path.write_text(toml.dumps({"api_settings": {"openai": {"api_key": value}}}))
    path.chmod(0o600)

    with pytest.raises(recovery_review.ProviderReconnectRequired):
        recovery_review._selection(
            path, "config:api_settings.openai.api_key", resolve=True
        )


def test_recovered_openai_selection_still_returns_a_plain_key(tmp_path):
    import toml

    from tldw_chatbook.LLM_Calls import recovery_review

    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    path = home / "config.toml"
    path.write_text(
        toml.dumps({"api_settings": {"openai": {"api_key": "sk-proj-plain-key"}}})
    )
    path.chmod(0o600)

    *_, secret = recovery_review._selection(
        path, "config:api_settings.openai.api_key", resolve=True
    )

    assert secret == "sk-proj-plain-key"
