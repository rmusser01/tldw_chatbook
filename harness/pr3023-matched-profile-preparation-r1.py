"""Prepare matched disposable profiles; never import the app or read a key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import tomllib


MODEL = "deepseek-flash"


def prepare_profile(parent: Path, arm: str, *, tools_enabled: bool = True) -> tuple[Path, Path]:
    """Create a fresh private profile using identical settings for both arms."""
    if arm not in {"baseline", "candidate"}:
        raise ValueError("unknown comparison arm")
    profile = Path(tempfile.mkdtemp(prefix=f"pr3023-{arm}-", dir=parent))
    for leaf in ("home", "config", "cache", "data"):
        (profile / leaf).mkdir(mode=0o700)
    config = profile / "config.toml"
    config.write_text(
        f'''[general]
users_name = "matched-verification"
[paths]
data_dir = "{profile}/data"
[first_run]
setup_completed = true
[chat_defaults]
provider = "deepseek"
model = "{MODEL}"
rag_auto_retrieve_on_send = false
[api_settings.deepseek]
api_key_env_var = "DEEPSEEK_API_KEY"
model = "{MODEL}"
streaming = true
reasoning_effort = "none"
api_retries = 0
max_tokens = 4096
temperature = 0.7
[providers]
DeepSeek = ["{MODEL}"]
[console]
agent_runtime = true
native_tool_calls = {str(tools_enabled).lower()}
local_tools_enabled = false
[model_catalog]
auto_refresh_enabled = false
[splash_screen]
enabled = false
''',
        encoding="utf-8",
    )
    config.chmod(0o600)
    return profile, config


def comparison_settings(config: Path) -> dict:
    """Normalize only the fresh profile's necessarily distinct data path."""
    settings = tomllib.loads(config.read_text(encoding="utf-8"))
    settings["paths"]["data_dir"] = "<owned-profile>/data"
    return settings


def self_check() -> dict:
    """Check actual TOML, isolation and permissions without application imports."""
    with tempfile.TemporaryDirectory(prefix="pr3023-profile-check-", dir="/private/tmp") as temporary:
        parent = Path(temporary)
        for enabled in (True, False):
            left, left_config = prepare_profile(parent, "baseline", tools_enabled=enabled)
            right, right_config = prepare_profile(parent, "candidate", tools_enabled=enabled)
            assert left != right
            settings = comparison_settings(left_config)
            assert settings == comparison_settings(right_config)
            assert settings["api_settings"]["deepseek"]["api_retries"] == 0
            assert settings["console"]["agent_runtime"] is True
            assert settings["console"]["native_tool_calls"] is enabled
            for profile, config in ((left, left_config), (right, right_config)):
                for directory in (profile, *(profile / name for name in ("home", "config", "cache", "data"))):
                    info = directory.lstat()
                    assert stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                    assert stat.S_IMODE(info.st_mode) == 0o700
                assert stat.S_IMODE(config.lstat().st_mode) == 0o600
                assert not list((profile / "data").iterdir())
    assert not parent.exists()
    return {"matched_route_settings": 2, "fresh_profiles_checked": 4, "temporary_parent_removed": True, "application_imported": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("preparation artifact supports only --self-check until final source binding")
    result = self_check()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
