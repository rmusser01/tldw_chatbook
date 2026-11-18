"""Real rail writes must not silently replace a selected backend's sources."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_external_original_rail_save_invalidates_backend_selection(request):
    """Exercise the actual writer and reader, in separate Python processes."""
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_rail_state import build_console_rail_preference_key
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory,
    )
    original_cache = config._CONFIG_CACHE
    original_identity = config.current_config_identity()
    key = build_console_rail_preference_key(layout_scope="global").value
    try:
        assert profile.selection_is_current()
        # This is the unchanged public writer called by the original rail
        # worker. The separate process cannot mutate the reader's Python cache.
        writer = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", (
                "import sys; from pathlib import Path; "
                "from tldw_chatbook import config; "
                "from tldw_chatbook.Chat.console_rail_state import "
                "build_console_rail_preference_key; "
                "assert config.get_cli_config_path() == Path(sys.argv[1]); "
                "assert config.get_user_data_dir() == Path(sys.argv[2]); "
                "assert config.save_setting_to_cli_config("
                "'console.rail_state', "
                "build_console_rail_preference_key(layout_scope='global').value, "
                "{'workspace_open': True})"
            ), str(path), str(directory)],
            env={**os.environ, "TLDW_CONFIG_PATH": str(path)},
            capture_output=True, check=False,
        )
        assert writer.returncode == 0, "Original external rail save failed"
        assert config._CONFIG_CACHE is original_cache
        assert config.current_config_identity() == original_identity
        assert profile.selection_is_current()

        # Respect the original external-edit sampling interval. No cache flag,
        # loader, file-stamp check or selection predicate is patched.
        await asyncio.sleep(1.05)
        settings = config.load_settings()
        assert settings["console"]["rail_state"][key]["workspace_open"] is True
        assert config._CONFIG_CACHE is not original_cache
        assert not profile.selection_is_current()
    finally:
        await profile.aclose()
