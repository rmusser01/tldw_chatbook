"""Fresh-process prerequisites for the two original Library admission nodes."""

import json
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Performance.test_fresh_config_library_lifecycle import (
    _SCRIPT as _CREATION_SCRIPT,
    _reopen_same_profile,
)


def _compat_script():
    """Keep the qualified original constructor/quit counter and add the rail oracle."""
    old = "    admission_at_creation = config.first_profile_created_this_session()\n"
    new = (
        old
        + """    if route == 'fresh' and stage == 'first':
        assert config.save_setting_to_cli_config('first_run', 'setup_completed', True)
    if route == 'fresh':
        assert config.get_cli_setting('first_run', 'setup_completed', False) is True
"""
    )
    assert _CREATION_SCRIPT.count(old) == 1
    script = _CREATION_SCRIPT.replace(old, new)
    old = "        snapshot_lifecycle = rail(app.app_config).get('lifecycle')\n"
    new = (
        old
        + """        assert app.library_new_profile_admission is (route == 'fresh' and stage == 'first')
        if stage == 'reopen' or route == 'old_missing':
            screen = LibraryScreen(app)
            if route == 'fresh':
                assert screen._library_lifecycle is LibraryLifecycle.UNKNOWN
            else:
                assert 'lifecycle' not in rail(app.app_config)
                assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
"""
    )
    assert script.count(old) == 1
    script = script.replace(old, new)
    old = "    assert counts['library_construction'] == counts['library_mount'] == 0\n"
    new = """    assert counts['library_construction'] == int(stage == 'reopen' or route == 'old_missing')
    assert counts['library_mount'] == 0
"""
    assert script.count(old) == 1
    script = script.replace(old, new)
    old = "    durable = tomllib.loads(selector.read_text(encoding='utf-8'))\n"
    new = (
        old
        + """    if route == 'fresh':
        assert selector.read_text(encoding='utf-8').count('lifecycle = "unknown"') == 1, 'profile creation must stamp the lifecycle into [library.rail_state]'
        assert durable['first_run']['setup_completed'] is True
"""
    )
    assert script.count(old) == 1
    return script.replace(old, new)


def run_original_library_profile_compat(tmp_path, *, existing):
    """Select the real physical document before either child first imports config."""
    script = _compat_script()
    route = "old_missing" if existing else "fresh"
    _run(tmp_path, route, "first", script=script, timeout=240)
    first = json.loads(
        (tmp_path / "library-creation-first.json").read_text(encoding="utf-8")
    )
    assert first["observer"]["complete"] and first["normal_runtime_disposal"]
    assert first["actual_creation_admission"] is (not existing)
    if existing:
        assert (
            not first["app_snapshot_unknown"] and not first["physical_creation_unknown"]
        )
    else:
        # The first process has physically exited before this original helper
        # starts its second process against precisely the same profile path.
        _reopen_same_profile(tmp_path, route, script=script)
        second = json.loads(
            (tmp_path / "library-creation-reopen.json").read_text(encoding="utf-8")
        )
        assert second["observer"]["complete"] and second["normal_runtime_disposal"]
        assert not second["actual_creation_admission"]
        assert second["physical_creation_unknown"] and second["app_snapshot_unknown"]
