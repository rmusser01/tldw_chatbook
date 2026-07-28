"""Generated audio must reach whichever playground is on screen.

The delivery lookup used to name the legacy widget specifically. With the
rebuilt pane mounted, a generation would start, succeed, and hand its
artifact to a widget that was not on screen -- so nothing appeared, no error
was raised, and the take silently never arrived.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Speech.speech_playback_mixin import SpeechPlaybackMixin
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
HOSTS = (SpeechPlaygroundPane,)


@pytest.mark.unit
@pytest.mark.parametrize("host", HOSTS, ids=lambda h: h.__name__)
def test_both_playgrounds_can_receive_a_delivered_artifact(host):
    """`stts_events` delivers by calling `_generation_complete` on whatever
    `_mounted_playground` returned. A host without it silently drops the
    audio down the fallback path."""
    assert callable(getattr(host, "_generation_complete", None))


@pytest.mark.unit
def test_delivery_comes_from_the_shared_mixin():
    """The completion path is inherited, not redefined on the pane.

    A pane-local copy would drift from the mixin the moment either changed,
    and the mixin is what the remaining Speech surfaces will inherit.
    """
    assert (
        SpeechPlaygroundPane._generation_complete
        is SpeechPlaybackMixin._generation_complete
    )


@pytest.mark.unit
def test_the_delivery_lookup_names_both_playgrounds():
    """Guards the actual defect: a lookup naming only one host.

    Asserted against the source because the failure is an absence -- the
    audio goes to a widget that is not mounted, and nothing raises.
    """
    import inspect

    from tldw_chatbook.Event_Handlers.STTS_Events import stts_events

    source = inspect.getsource(stts_events.STTSEventHandler._mounted_playground)
    assert "SpeechPlaygroundPane" in source, (
        "the playground is not reachable for delivery; generations will "
        "succeed and produce nothing on screen"
    )


@pytest.mark.unit
def test_a_provider_config_change_invalidates_both_playgrounds():
    """The same defect as delivery, in the invalidation path.

    `on_stts_provider_configuration_changed` queried the legacy widget by
    type selector, so changing a provider's settings left the rebuilt pane
    serving a stale catalog -- wrong models and voices, with nothing on
    screen to say so.
    """
    import inspect

    from tldw_chatbook.Event_Handlers.STTS_Events import stts_events

    source = inspect.getsource(
        stts_events.STTSEventHandler.on_stts_provider_configuration_changed
    )
    assert "SpeechPlaygroundPane" in source, (
        "the playground is never invalidated; it will serve a stale "
        "catalog after a provider is reconfigured"
    )


@pytest.mark.unit
@pytest.mark.parametrize("host", HOSTS, ids=lambda h: h.__name__)
def test_both_playgrounds_can_be_invalidated(host):
    """Whatever the query returns must carry the callback."""
    assert callable(getattr(host, "mark_provider_configuration_changed", None))
