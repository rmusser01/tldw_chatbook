"""Cross-pass memo for the Console template defaults (TASK-24301).

`_default_console_session_settings` is a pure function of
(app_config, provider, model): it reads no environment and mutates nothing. The
composer keystroke path reached it 3.25 times per printable key and the answer
is identical between two characters of a word, so the result is memoised across
passes rather than only within one.

A cache is only as good as its invalidation, so each of the three key
components gets a test that fails if that component is dropped from the key.
The config leg is compared by IDENTITY against a retained reference -- not by
`id()`, which can be recycled after GC, and not by value, which would cost more
than the derivation it protects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.UI.Console_Modules import session as session_module
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


class _Recorder:
    """Counts calls to the pure defaults builder the controller delegates to."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None, str | None]] = []

    def __call__(
        self, app_config: Any, provider: str | None, model: str | None
    ) -> object:
        self.calls.append((id(app_config), provider, model))
        return SimpleNamespace(provider=provider or "openai", model=model)


@pytest.fixture()
def controller(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A bare controller with only the seams `_default_console_session_settings` uses.

    Built through `__new__` deliberately: the real `__init__` takes 40+ wired
    callables, none of which this derivation touches.
    """
    obj = ConsoleSessionController.__new__(ConsoleSessionController)
    obj._screen = SimpleNamespace(_console_derivation_memo=None)
    obj._config = {"marker": "first"}
    obj._provider_model = ("openai", "gpt-4")
    # The controller exposes these as read-only properties over the callables
    # bound in `__init__`; set the underlying `_fn` attributes.
    obj._provider_readiness_app_config_fn = lambda: obj._config
    obj._effective_console_provider_model_fn = lambda: obj._provider_model
    return obj


def test_repeated_derivation_with_unchanged_inputs_builds_once(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keystroke case: same config, same provider/model, one build."""
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    first = controller._default_console_session_settings()
    for _ in range(50):
        controller._default_console_session_settings()

    assert len(recorder.calls) == 1, (
        f"{len(recorder.calls)} builds for 51 derivations with unchanged "
        "inputs; the cross-pass memo is not engaging."
    )
    assert controller._default_console_session_settings() is first


def test_a_reloaded_config_object_invalidates_the_memo(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh config mapping is a different answer, not a cache hit.

    `load_settings()` hands back the same mapping until its cache is
    invalidated and a NEW object after a save, so identity is exactly the
    reload signal. Dropping the config leg of the key would serve pre-save
    defaults forever -- the task-177 shape of bug.
    """
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    controller._default_console_session_settings()
    controller._config = {"marker": "second"}  # a save happened; new object
    controller._default_console_session_settings()

    assert len(recorder.calls) == 2, (
        "a reloaded configuration did not invalidate the memo; the derivation "
        "would keep serving pre-save defaults."
    )


def test_an_equal_but_distinct_config_object_still_invalidates(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity, not equality: a new object with identical content re-derives.

    This is the conservative direction. Comparing by value would be both
    slower and, for a nested mapping, unreliable; re-deriving on a fresh
    object that happens to be equal costs one build and can never be stale.
    """
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    controller._default_console_session_settings()
    controller._config = {"marker": "first"}  # equal content, different object
    controller._default_console_session_settings()

    assert len(recorder.calls) == 2


def test_a_changed_provider_invalidates_the_memo(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching provider must re-derive, or the control bar shows the old one."""
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    controller._default_console_session_settings()
    controller._provider_model = ("anthropic", "gpt-4")
    controller._default_console_session_settings()

    assert len(recorder.calls) == 2
    assert recorder.calls[1][1] == "anthropic"


def test_a_changed_model_invalidates_the_memo(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching model must re-derive."""
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    controller._default_console_session_settings()
    controller._provider_model = ("openai", "gpt-4o")
    controller._default_console_session_settings()

    assert len(recorder.calls) == 2
    assert recorder.calls[1][2] == "gpt-4o"


def test_blank_provider_and_model_normalise_to_none(
    controller: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only selections are None, and do not thrash the memo."""
    recorder = _Recorder()
    monkeypatch.setattr(session_module, "default_console_session_settings", recorder)

    controller._provider_model = ("   ", "")
    controller._default_console_session_settings()
    controller._default_console_session_settings()

    assert len(recorder.calls) == 1
    assert recorder.calls[0][1] is None
    assert recorder.calls[0][2] is None
