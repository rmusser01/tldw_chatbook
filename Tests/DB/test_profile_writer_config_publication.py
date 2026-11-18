"""A writer admission must preserve the runtime's checked configuration source."""

import os
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
def test_writer_joins_current_config_read_and_detects_same_size_replacement(request):
    from tldw_chatbook import config
    from tldw_chatbook.DB import profile_writer_selection as selection

    profile = config.get_user_data_dir()
    path = config.get_cli_config_path()
    selection.require_writer_repository(profile, "db.agent_runs")
    # The UI atomically saves a rail preference and the original runtime fence
    # rereads it before adopting that exact cache as its configuration source.
    assert config.save_setting_to_cli_config(
        "console.rail_state", "console_rail_state:global:shared-layout-v1",
        {"workspace_open": True},
    )
    assert config.refresh_runtime_config_from_cli_config().caches_reloaded
    published = config._CONFIG_CACHE
    assert config._CONFIG_PINNED_READ[0] is published
    assert config._CONFIG_PINNED_READ[1:] == (path, selection._stamp(path))
    selection.require_writer_repository(profile, "db.agent_runs")
    assert config._CONFIG_CACHE is published

    # A new file with identical bytes, size and write time is still a new
    # source. A writer must perform the original reload in that case.
    replacement = path.with_name("equal-config.toml")
    replacement.write_bytes(path.read_bytes())
    before = path.stat()
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    selection.require_writer_repository(profile, "db.agent_runs")
    assert config._CONFIG_CACHE is not published
