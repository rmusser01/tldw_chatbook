"""Initial-screen completion exists before mounted readiness reads it."""
import os
from pathlib import Path

import pytest

from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_original_app_initial_screen_flag_is_false_before_mount(request, tmp_path):
    from tldw_chatbook import app as app_source

    creators = OriginalPrivateAppCreators(app_source, Path(os.environ["XDG_DATA_HOME"]).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    try:
        assert app._initial_screen_pushed is False
        assert app._ui_ready is False
    finally:
        await runtime.dispose()
        creators.retire()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_original_initial_screen_transition_with_bootstrap_profile(monkeypatch):
    from Tests.UI.test_splash_initial_screen_preimport import (
        _isolate_screen_modules,
        test_initial_screen_module_is_imported_by_the_thread_not_the_loop,
    )

    isolation = _isolate_screen_modules.__wrapped__()
    next(isolation)
    try:
        await test_initial_screen_module_is_imported_by_the_thread_not_the_loop(monkeypatch)
    finally:
        try:
            next(isolation)
        except StopIteration:
            pass
