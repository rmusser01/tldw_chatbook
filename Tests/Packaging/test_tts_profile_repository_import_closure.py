"""Import-closure guard for first-use TTS profile-repository construction."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFERRED_REPOSITORY_MODULES = (
    "tldw_chatbook.TTS.profile_repository",
    "tldw_chatbook.TTS.profile_schema",
    "tldw_chatbook.TTS.profile_sqlite_policy",
    "tldw_chatbook.TTS.profile_validation",
    "tldw_chatbook.TTS.profile_store_lock",
    "tldw_chatbook.TTS.profile_reference_storage",
    "tldw_chatbook.TTS.profile_migration_candidate",
    "tldw_chatbook.TTS.profile_migration_journal",
    "tldw_chatbook.TTS.profile_migration_namespace",
    "tldw_chatbook.TTS.profile_migration_publication",
    "tldw_chatbook.TTS.profile_migration_recovery",
    "tldw_chatbook.TTS.migrations",
    "tldw_chatbook.TTS.migrations.v0_to_v1",
    "tldw_chatbook.TTS.migrations.v1_to_v2",
    "tldw_chatbook.TTS.migrations.v2_to_v3",
    "tldw_chatbook.TTS.migrations.v3_to_v4",
)


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    for path in (home, data_home, config_home):
        path.mkdir(parents=True, exist_ok=True)
    config_path = config_home / "config.toml"
    config_path.write_text("[first_run]\nsetup_completed = true\n", encoding="utf-8")
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HOME": str(home),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONPATH": str(REPO_ROOT),
        "TLDW_CONFIG_PATH": str(config_path),
        "TLDW_SCREEN_PREIMPORT": "0",
        "TLDW_TEST_MODE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_DATA_HOME": str(data_home),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


_IMPORT_CLOSURE_SNIPPET = f"""
import sys
import Tests.conftest

import tldw_chatbook.app

forbidden = {DEFERRED_REPOSITORY_MODULES!r}
resident = sorted(module for module in forbidden if sys.modules.get(module) is not None)
assert not resident, "profile repository closure resident at app import: " + repr(resident)
assert "tldw_chatbook.TTS.profile_service" in sys.modules

import tldw_chatbook.UI.Screens.personas_screen
from tldw_chatbook.UI.stts_profile_library import (
    VoiceBundleActionProjection,
    voice_bundle_import_choice,
)

resident = sorted(module for module in forbidden if sys.modules.get(module) is not None)
assert not resident, "profile repository closure resident at personas import: " + repr(resident)

choice = voice_bundle_import_choice(
    VoiceBundleActionProjection(
        operation="import_create",
        label="Create",
        tooltip="Create a profile",
        disabled=False,
    ),
    inactive_consent=True,
)
from tldw_chatbook.TTS.voice_bundle_service import (
    TTSVoiceBundleImportChoice as RealImportChoice,
)

assert type(choice) is RealImportChoice
assert choice.choice == "create"
assert choice.inactive_consent is True

from tldw_chatbook.TTS import TTSProfileRepository
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository as RealOwner

assert TTSProfileRepository is RealOwner
assert sys.modules.get("tldw_chatbook.TTS.profile_repository") is not None
print("TTS_PROFILE_REPOSITORY_CLOSURE_OK")
"""


def test_app_and_personas_import_defer_repository_until_real_first_use(
    tmp_path: Path,
) -> None:
    result = _run_isolated_python(tmp_path, _IMPORT_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        "TTS profile repository must stay off app and personas import closures:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "TTS_PROFILE_REPOSITORY_CLOSURE_OK" in result.stdout
