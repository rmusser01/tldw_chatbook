"""Changed finite-read proof helpers are rejected before their bodies execute."""

import pytest

from tldw_chatbook.Widgets import compact_model_bar as module

pytestmark = pytest.mark.bootstrap_profile


def _foreign_helper(*args):
    _compact_source_test_calls.append(True)  # noqa: F821 - injected into helper globals
    return False


@pytest.mark.parametrize("name", ["_source_current", "_function_current"])
def test_changed_proof_helper_body_is_never_invoked(monkeypatch, name):
    assert module._sources_current()
    calls = []
    monkeypatch.setattr(module, "_compact_source_test_calls", calls, raising=False)
    helper = getattr(module, name)
    assert helper.__closure__ is None and _foreign_helper.__closure__ is None
    monkeypatch.setattr(helper, "__code__", _foreign_helper.__code__)
    assert module._sources_current() is False
    assert not calls, "changed helper executed before its source was checked"
