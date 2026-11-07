"""The stock publisher is one shared marker, including unsupported nesting."""

import pytest


def test_shared_publisher_masks_and_restores_unsupported_nested_exception():
    from tldw_chatbook import Skills_Interop, app_service_wiring

    app_skill_wiring = app_service_wiring._skill_wiring
    from tldw_chatbook.Chat.console_configuration_preparation import (
        native_skill_trust_publication,
    )

    marker = getattr(Skills_Interop, "_skill_trust_publisher", None)
    assert marker is not None, "shared publisher is missing"
    assert marker is app_skill_wiring._skill_trust_publisher
    outer = object()
    token = marker.set(outer)
    try:
        with (
            pytest.raises(RuntimeError, match="original exception"),
            native_skill_trust_publication(object()) as result,
        ):
            assert result is None
            assert marker.get() is None
            raise RuntimeError("original exception")
        assert marker.get() is outer
    finally:
        marker.reset(token)
    assert marker.get() is None
