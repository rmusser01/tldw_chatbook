# test_eval_orchestrator_db_path.py
# Description: Regression tests for TASK-860 (Evals DB profile isolation)
#
"""
Regression tests for task-860: EvaluationOrchestrator._initialize_database()
must honor the configured profile, not always write to "default_user".

Background: the old code resolved the profile via
``settings.get("user_id", settings.get("username", "default_user"))`` and
the data root via ``settings.get("user_data_dir", ...)``. ``load_settings()``
never publishes either key -- the real profile name is published as
``USERS_NAME`` -- so both lookups silently missed and every profile shared
one ``default_user/evals.db``. The fix resolves the path through
``tldw_chatbook.config.get_user_data_dir()``, the same helper every other DB
in the app uses.
"""

from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook import config as config_module
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Force load_settings()/get_cli_setting() to re-read the (monkeypatched)
    active config in every test here, instead of serving a stale cross-test
    cache (see Tests/test_user_data_dir_isolation.py for the same pattern)."""
    _reset_config_caches()
    yield
    _reset_config_caches()


def _reset_config_caches():
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None


def _isolate_profile(
    monkeypatch,
    tmp_path,
    profile_name: str,
    *,
    home_dir: Path,
    data_dir: Path | None = None,
) -> None:
    """Point HOME at a scratch dir and configure the given profile name.

    No ``[paths] data_dir`` is set unless ``data_dir`` is given, so
    ``get_user_data_dir()`` normally exercises the real default-fallback
    branch (``HOME/.local/share/tldw_cli``).
    """
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    data_dir_line = f'data_dir = "{data_dir}"\n' if data_dir is not None else ""
    scratch_config = tmp_path / f"config_{profile_name}.toml"
    scratch_config.write_text(
        f'[general]\nusers_name = "{profile_name}"\n\n[paths]\n{data_dir_line}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch_config))


def test_evals_db_path_changes_with_the_configured_profile(monkeypatch, tmp_path):
    """Two different profiles under the same HOME must resolve to two
    different evals.db paths, and neither may be hardcoded to
    'default_user' when the configured profile says otherwise."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    _isolate_profile(monkeypatch, tmp_path, "alpha_profile", home_dir=home_dir)
    orchestrator_a = EvaluationOrchestrator(client_id="test_a")
    path_a = Path(orchestrator_a.db.db_path)

    _reset_config_caches()

    _isolate_profile(monkeypatch, tmp_path, "beta_profile", home_dir=home_dir)
    orchestrator_b = EvaluationOrchestrator(client_id="test_b")
    path_b = Path(orchestrator_b.db.db_path)

    assert path_a != path_b, (
        f"Both profiles resolved to the same evals.db path ({path_a}) -- "
        "the profile is not actually being honored."
    )
    assert path_a.name == "evals.db"
    assert path_b.name == "evals.db"
    assert path_a.parent.name == "alpha_profile", path_a
    assert path_b.parent.name == "beta_profile", path_b
    assert "default_user" not in path_a.parts, path_a
    assert "default_user" not in path_b.parts, path_b
    assert str(path_a).startswith(str(home_dir)), path_a
    assert str(path_b).startswith(str(home_dir)), path_b


def test_evals_db_path_still_used_for_the_default_user_profile(monkeypatch, tmp_path):
    """A profile literally named 'default_user' is still a legitimate
    profile -- it must resolve normally, not be treated specially."""
    home_dir = tmp_path / "home_default"
    home_dir.mkdir()
    _isolate_profile(monkeypatch, tmp_path, "default_user", home_dir=home_dir)

    orchestrator = EvaluationOrchestrator(client_id="test_default")
    path = Path(orchestrator.db.db_path)

    assert path.parent.name == "default_user"
    assert path.name == "evals.db"
    assert str(path).startswith(str(home_dir))


def test_legacy_default_user_data_notice_fires_when_legacy_exists(
    monkeypatch, tmp_path
):
    """If the profile-resolved path has no data yet but the legacy
    default_user/evals.db does, a clear one-time warning must fire naming
    both paths -- and the legacy file must be left completely untouched
    (no silent copy or move)."""
    home_dir = tmp_path / "home_legacy"
    home_dir.mkdir()
    legacy_dir = home_dir / ".local" / "share" / "tldw_cli" / "default_user"
    legacy_dir.mkdir(parents=True)
    legacy_db = legacy_dir / "evals.db"
    legacy_db.write_bytes(b"pretend-legacy-sqlite-bytes")
    legacy_mtime = legacy_db.stat().st_mtime

    _isolate_profile(monkeypatch, tmp_path, "gamma_profile", home_dir=home_dir)

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        orchestrator = EvaluationOrchestrator(client_id="test_gamma")
    finally:
        loguru_logger.remove(sink_id)

    resolved_path = Path(orchestrator.db.db_path)
    assert resolved_path.parent.name == "gamma_profile"

    joined = "\n".join(messages)
    assert str(resolved_path) in joined, messages
    assert str(legacy_db) in joined, messages

    # Nothing was copied or moved: the legacy file is unchanged, and the new
    # profile got its own fresh (empty-of-legacy-content) database file.
    assert legacy_db.read_bytes() == b"pretend-legacy-sqlite-bytes"
    assert legacy_db.stat().st_mtime == legacy_mtime
    assert resolved_path != legacy_db
    assert resolved_path.exists()  # EvalsDB creates its own fresh file
    assert resolved_path.stat().st_size != legacy_db.stat().st_size or (
        resolved_path.read_bytes() != legacy_db.read_bytes()
    )


def test_no_legacy_notice_when_no_legacy_data_exists(monkeypatch, tmp_path):
    """A brand-new user with no legacy default_user/evals.db must not be
    warned about data that was never there."""
    home_dir = tmp_path / "home_fresh"
    home_dir.mkdir()
    _isolate_profile(monkeypatch, tmp_path, "fresh_profile", home_dir=home_dir)

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        EvaluationOrchestrator(client_id="test_fresh")
    finally:
        loguru_logger.remove(sink_id)

    assert not any("default_user" in message for message in messages), messages


def test_no_legacy_notice_when_the_new_profile_already_has_its_own_data(
    monkeypatch, tmp_path
):
    """Once a profile has its own evals.db, the legacy notice must stop --
    it already has independent data and does not need the old file."""
    home_dir = tmp_path / "home_has_own_data"
    home_dir.mkdir()
    legacy_dir = home_dir / ".local" / "share" / "tldw_cli" / "default_user"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "evals.db").write_bytes(b"legacy-bytes")

    _isolate_profile(monkeypatch, tmp_path, "delta_profile", home_dir=home_dir)

    # First run creates the profile's own evals.db.
    EvaluationOrchestrator(client_id="test_delta_first")

    _reset_config_caches()
    _isolate_profile(monkeypatch, tmp_path, "delta_profile", home_dir=home_dir)

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        EvaluationOrchestrator(client_id="test_delta_second")
    finally:
        loguru_logger.remove(sink_id)

    assert not any("default_user" in message for message in messages), messages


def test_legacy_notice_uses_the_hardcoded_legacy_location_even_with_custom_data_dir(
    monkeypatch, tmp_path
):
    """The pre-fix code's `user_data_dir` lookup key never existed in
    `load_settings()`, so it ALWAYS fell back to the literal
    "~/.local/share/tldw_cli", even for a user who had configured a custom
    `[paths] data_dir`. The real legacy Evals data for such a user is
    therefore under HOME, not under their custom data root -- the notice
    must look there, or it would miss exactly the users most likely to have
    lost track of an old file."""
    home_dir = tmp_path / "home_custom_data_dir"
    home_dir.mkdir()
    custom_data_dir = tmp_path / "somewhere_else_entirely"
    custom_data_dir.mkdir()

    # The TRUE legacy location the old buggy code always wrote to: hardcoded,
    # under HOME, regardless of the custom data_dir configured below.
    legacy_dir = home_dir / ".local" / "share" / "tldw_cli" / "default_user"
    legacy_dir.mkdir(parents=True)
    legacy_db = legacy_dir / "evals.db"
    legacy_db.write_bytes(b"pretend-legacy-sqlite-bytes")

    _isolate_profile(
        monkeypatch,
        tmp_path,
        "epsilon_profile",
        home_dir=home_dir,
        data_dir=custom_data_dir,
    )

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        orchestrator = EvaluationOrchestrator(client_id="test_epsilon")
    finally:
        loguru_logger.remove(sink_id)

    resolved_path = Path(orchestrator.db.db_path)
    # Sanity: the profile's own DB really did land under the custom data
    # root, not under HOME -- confirms the test is exercising the scenario
    # it claims to.
    assert str(resolved_path).startswith(str(custom_data_dir)), resolved_path
    assert resolved_path.parent.name == "epsilon_profile"

    joined = "\n".join(messages)
    assert str(legacy_db) in joined, messages
    assert legacy_db.read_bytes() == b"pretend-legacy-sqlite-bytes"
