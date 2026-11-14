"""Settings ▸ Providers & Models ▸ Model: the picker sync ends quietly when
the picker's field is already gone (fix round 1 of the #3057 review,
finding 17).

`_sync_model_picker` is reached after an await (Save selected fills an empty
Model field when its write returns). Teardown prunes the picker's field
before the picker itself, and the sync asked that field for focus, and the
picker for the Custom ID it holds, outside its lookup guard.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _SIZE,
    _app,
    _open_providers,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
async def test_the_picker_sync_ends_quietly_when_the_picker_field_is_gone(request):
    app = _app("openai", "gpt-4.1", catalog=("gpt-4.1", "gpt-4o"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, _adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press("tab", "enter")  # Custom ID
        await pilot.pause()
        assert picker.custom_mode

        await field.remove()  # as teardown does, before the picker goes

        screen._sync_model_picker()  # NoMatches before the fix
