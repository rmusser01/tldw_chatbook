"""`@on` inside a plain mixin is silently never dispatched.

Textual collects decorated handlers in its metaclass, scanning only each
class's own namespace. A mixin that is not itself a MessagePump never passes
through that metaclass, so an `@on` method defined there is registered
nowhere -- no error, no warning, the handler simply never runs.

That is how provider switching broke when the catalog code moved to a mixin:
every test that switched provider timed out waiting for state that no longer
arrived. These tests assert each host declares the handler itself.
"""

from __future__ import annotations

import pytest
from textual.widgets import Select

from tldw_chatbook.UI.Speech.speech_catalog_mixin import SpeechCatalogMixin
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget

HOSTS = (TTSPlaygroundWidget, SpeechPlaygroundPane)


@pytest.mark.unit
@pytest.mark.parametrize("host", HOSTS, ids=lambda h: h.__name__)
def test_each_host_registers_the_provider_handler_itself(host):
    """The registration must be on the host, not inherited from the mixin."""
    registered = host.__dict__.get("_decorated_handlers", {})
    assert Select.Changed in registered, (
        f"{host.__name__} does not register a Select.Changed handler; "
        "provider switching will silently stop working"
    )


@pytest.mark.unit
@pytest.mark.parametrize("host", HOSTS, ids=lambda h: h.__name__)
def test_the_registered_handler_reaches_the_shared_implementation(host):
    """Registering some other handler would satisfy the check above."""
    handlers = host.__dict__["_decorated_handlers"][Select.Changed]
    names = {function.__name__ for function, _selectors in handlers}
    assert "on_tts_provider_select_changed" in names


@pytest.mark.unit
def test_the_mixin_does_not_carry_a_dead_decorated_handler():
    """A re-decorated mixin method would look right and never fire."""
    method = SpeechCatalogMixin.handle_provider_select_changed
    assert not hasattr(method, "_textual_on"), (
        "handle_provider_select_changed is decorated inside the mixin, where "
        "Textual will never register it"
    )
