"""Controller wiring: exchange captures attach alongside usage, and a
config kill-switch gates capture end-to-end (task-7, Console Conversation
Inspector).

Fixture/driver idioms copied mechanically from
Tests/Chat/test_console_chat_controller.py's usage-attach coverage (that
file has NO pytest fixtures at all -- every test builds `store`/`controller`
inline, mirrored here):
  * plain construction: ``ConsoleChatController(store=store,
    provider_gateway=StreamingGateway())`` and its own minimal
    ``StreamingGateway`` stub.
  * direct-call driver for ``_attach_stream_usage``:
    ``test_re_attaching_the_same_signals_is_idempotent`` /
    ``test_stop_path_usage_attach_survives_a_persistence_exception``.
  * monkeypatching a module-level import in the controller's OWN
    namespace (a from-import binds at import time, so the CONSUMER's
    namespace -- not the definition site -- is what must be patched):
    ``monkeypatch.setattr(controller_module, "is_vision_capable", ...)``.
  * swallow-and-log diagnostics capture: mirrors
    Tests/Chat/test_console_chat_store_exchanges.py's
    ``test_persist_exchanges_only_survives_a_serialization_failure``
    (a loguru sink collecting WARNING-level records).
  * per-call signals construction (``new_usage_call()`` then
    ``begin_exchange``/``close_exchange``): the real gateway call site in
    ``console_provider_gateway.py``'s ``stream_chat`` (llama.cpp branch).
  * shipped-default pin against the REAL resolved settings layer, no extra
    setup beyond the autouse config isolation: mirrors
    Tests/test_config_console_defaults.py's own
    ``test_console_sidechat_model_default_is_empty_string`` style.

Review fix round: the kill-switch must coerce ``get_cli_setting``'s RAW
string return through ``coerce_bool_setting`` (not bare ``bool()``), pinned
in both directions by the two ``test_kill_switch_string_*`` tests below.
"""
from __future__ import annotations

from types import SimpleNamespace

from loguru import logger as loguru_logger

from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals


class StreamingGateway:
    """Minimal gateway stub, mirroring test_console_chat_controller.py's own
    ``StreamingGateway`` -- this suite never drives a real send; only the
    controller's construction needs a gateway object to exist."""

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


def _new_controller() -> ConsoleChatController:
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=StreamingGateway())


def _controller_with_placeholder():
    """Real store + a real assistant placeholder message, mirroring the
    setup ``test_re_attaching_the_same_signals_is_idempotent`` builds
    before driving ``_attach_stream_usage`` directly."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    return controller, store, placeholder.id


def _captured_signals() -> ConsoleProviderStreamSignals:
    """One provider call's worth of exchange capture, using the real
    per-call pattern from ``console_provider_gateway.py``'s ``stream_chat``
    (llama.cpp branch): ``new_usage_call()`` for the per-call view, then
    begin/close_exchange on it (the aggregate itself has no begin_exchange
    method -- only the per-call view does)."""
    # Explicit opt-in: the dataclass default is False (review finding I1) --
    # this helper builds a signals object with a real capture in it.
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    call_signals = signals.new_usage_call()
    call_signals.begin_exchange(
        provider="p", model="m", endpoint=None, request={}, omitted_keys=()
    )
    call_signals.close_exchange()
    return signals


def test_signals_created_with_capture_enabled_by_default():
    controller = _new_controller()
    signals = controller._new_run_stream_signals()
    assert signals.exchange_capture_enabled is True


def test_kill_switch_disables_capture(monkeypatch):
    """Patch ``get_cli_setting`` AT THE CONTROLLER'S NAMESPACE (a from-import
    binds at import time -- patch the consumer, prove it with a call
    counter)."""
    controller = _new_controller()
    calls: list[tuple[str, str]] = []

    def fake_get_cli_setting(section, key, default=None):
        calls.append((section, key))
        if (section, key) == ("console", "exchange_capture"):
            return False
        return default

    monkeypatch.setattr(
        controller_module, "get_cli_setting", fake_get_cli_setting
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is False
    assert ("console", "exchange_capture") in calls


def test_kill_switch_string_false_disables_capture(monkeypatch):
    """``get_cli_setting`` returns the RAW TOML value, uncoerced -- a
    hand-typed ``exchange_capture = "false"`` is a non-empty string and
    therefore truthy under bare ``bool()``, which would silently defeat
    the only escape hatch for this privacy-sensitive feature (the arc's
    sixth occurrence of this exact trap; see ``local_tools_enabled``'s own
    read in ``console_chat_controller.py`` for the first).
    ``coerce_bool_setting`` must be applied to the read."""
    controller = _new_controller()

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("console", "exchange_capture"):
            return "false"
        return default

    monkeypatch.setattr(controller_module, "get_cli_setting", fake_get_cli_setting)

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is False


def test_kill_switch_string_true_enables_capture(monkeypatch):
    """Pin the coercion in both directions -- a hand-typed string ``"true"``
    must still resolve to enabled."""
    controller = _new_controller()

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("console", "exchange_capture"):
            return "true"
        return default

    monkeypatch.setattr(controller_module, "get_cli_setting", fake_get_cli_setting)

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is True


def test_shipped_config_default_resolves_exchange_capture_true():
    """Make the shipped [console] default itself load-bearing: read the
    REAL resolved settings layer with no controller-side default masking
    whether the TOML key is actually present (``default=None``, not
    ``True``) -- mirrors Tests/test_config_console_defaults.py's own
    default-pin style (e.g. ``test_console_sidechat_model_default_is_
    empty_string``), which calls ``get_cli_setting`` directly with no
    extra setup beyond the autouse ``isolate_test_environment`` fixture
    (Tests/conftest.py) that already redirects XDG_CONFIG_HOME/
    TLDW_CONFIG_PATH to a per-test temp directory -- this never touches
    the user's real config (a documented incident in this repo)."""
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    assert real_get_cli_setting("console", "exchange_capture", None) is True


def test_attach_site_forwards_captures_to_store():
    """The usage-attach method (``_attach_stream_usage``) forwards BOTH the
    usage payload AND ``signals.exchange_captures()`` to the store from the
    SAME call -- a usage payload is recorded on the call-scoped view before
    closing the exchange, mirroring the real gateway's
    ``record_usage_payload`` + ``begin_exchange``/``close_exchange`` +
    ``close_usage_call`` sequence on one ``new_usage_call()`` view
    (``console_provider_gateway.py``'s ``stream_chat``, llama.cpp branch,
    and its ``finally`` close-out order). Driven the same way
    ``test_re_attaching_the_same_signals_is_idempotent`` drives usage."""
    controller, store, message_id = _controller_with_placeholder()
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    call_signals = signals.new_usage_call()
    call_signals.record_usage_payload(
        {"prompt_tokens": 100, "completion_tokens": 20}
    )
    call_signals.begin_exchange(
        provider="p", model="m", endpoint=None, request={}, omitted_keys=()
    )
    call_signals.close_exchange()
    call_signals.close_usage_call()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    controller._attach_stream_usage(message_id, signals, resolution, partial=False)

    message = store.get_message(message_id)
    assert message.usage is not None
    assert message.usage.total_tokens == 120, "the usage attach must still land"
    assert message.exchanges, "the controller must forward exchange_captures() to the store"
    assert message.exchanges[0].provider == "p"


def test_attach_forwards_captures_even_without_usage():
    """Exchange capture must not be gated on a nonzero usage total -- these
    signals carry an exchange but no usage payload at all (no
    ``record_usage_payload`` call), which would otherwise make
    ``_attach_stream_usage`` return before ever reaching the exchange
    attach if it were nested inside the usage-total branch."""
    controller, store, message_id = _controller_with_placeholder()
    signals = _captured_signals()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    controller._attach_stream_usage(message_id, signals, resolution, partial=False)

    assert store.get_message(message_id).exchanges


def test_attach_skips_the_store_call_when_nothing_was_captured(monkeypatch):
    """No captures -> no store call at all (same shape as usage's own
    "nothing to bill" early return) -- proven via an instance-level
    monkeypatch spy, since ``ConsoleProviderStreamSignals()`` with no
    ``begin_exchange`` produces an empty ``exchange_captures()``."""
    controller, store, message_id = _controller_with_placeholder()
    calls = []
    original = store.attach_message_exchanges

    def _spy(mid, captures):
        calls.append((mid, captures))
        return original(mid, captures)

    monkeypatch.setattr(store, "attach_message_exchanges", _spy)
    signals = ConsoleProviderStreamSignals()  # no exchanges recorded

    controller._attach_stream_usage(message_id, signals, resolution=SimpleNamespace(
        provider="openai", model="gpt-4o"), partial=False)

    assert calls == []


def test_attach_never_fails_the_send(monkeypatch):
    """Store raising from ``attach_message_exchanges`` is swallowed+logged --
    same never-fail contract as ``usage_attach_failed``. Mirrors
    Tests/Chat/test_console_chat_store_exchanges.py's
    ``test_persist_exchanges_only_survives_a_serialization_failure`` for the
    loguru-sink diagnostics-capture idiom."""
    controller, store, message_id = _controller_with_placeholder()

    def _raise(mid, captures):
        raise RuntimeError("simulated store failure")

    monkeypatch.setattr(store, "attach_message_exchanges", _raise)

    signals = _captured_signals()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra[message_id]} {extra[error]} {message}",
    )
    try:
        # Must not raise.
        controller._attach_stream_usage(
            message_id, signals, resolution, partial=False
        )
    finally:
        loguru_logger.remove(sink_id)

    assert any("exchange_attach_failed" in d for d in diagnostics), diagnostics
