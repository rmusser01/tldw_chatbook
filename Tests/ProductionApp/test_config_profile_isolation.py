from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "TASK15674_RESULT="
EFFECTIVE_SENTINEL = "PRIVATE_EFFECTIVE_CONFIG_VALUE"
DECOY_SENTINEL = "PRIVATE_DECOY_CONFIG_VALUE"


def _write_sparse_config(path: Path, *, data_dir: Path, marker: str) -> bytes:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    serialized = textwrap.dedent(
        f"""
        [general]
        default_tab = "chat"
        profile_marker = {json.dumps(marker)}

        [first_run]
        setup_started = true
        setup_completed = true

        [splash_screen]
        enabled = false

        [paths]
        data_dir = {json.dumps(str(data_dir))}

        [model_catalog]
        auto_refresh_enabled = false
        """
    ).lstrip()
    path.write_text(serialized, encoding="utf-8")
    path.chmod(0o600)
    return serialized.encode("utf-8")


def _result_payload(stdout: str) -> dict[str, bool]:
    records = [
        line.removeprefix(RESULT_PREFIX)
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    unique_result = len(records) == 1
    assert unique_result, "isolated lifecycle result record was not unique"
    return json.loads(records[0])


def test_real_app_quit_persists_only_the_effective_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    xdg_config = tmp_path / "xdg-config"
    xdg_data = tmp_path / "xdg-data"
    xdg_cache = tmp_path / "xdg-cache"
    temp_dir = tmp_path / "tmp"
    effective_data = tmp_path / "effective-data"
    decoy_data = tmp_path / "decoy-data"
    for directory in (
        home,
        xdg_config,
        xdg_data,
        xdg_cache,
        temp_dir,
        effective_data,
        decoy_data,
    ):
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)

    effective_config = tmp_path / "effective" / "config.toml"
    decoy_default = home / ".config" / "tldw_cli" / "config.toml"
    _write_sparse_config(
        effective_config,
        data_dir=effective_data,
        marker=EFFECTIVE_SENTINEL,
    )
    decoy_before = _write_sparse_config(
        decoy_default,
        data_dir=decoy_data,
        marker=DECOY_SENTINEL,
    )

    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TERM", "COLORTERM")
        if key in os.environ
    }
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "XDG_CACHE_HOME": str(xdg_cache),
            "TMPDIR": str(temp_dir),
            "TLDW_CONFIG_PATH": str(effective_config),
            "TLDW_TEST_MODE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )

    snippet = f"""
import asyncio
import json
import os
from pathlib import Path

state = {{"mounted": False, "effective_path_selected": False,
         "persistence_called": False, "persistence_succeeded": False}}
phase = "import"

try:
    import tldw_chatbook.config as config_module
    import tldw_chatbook.app as app_module

    real_persist = config_module.persist_cli_config_for_shutdown

    def observed_persist():
        state["persistence_called"] = True
        state["effective_path_selected"] = (
            config_module.get_cli_config_path()
            == Path(os.environ["TLDW_CONFIG_PATH"])
        )
        result = real_persist()
        state["persistence_succeeded"] = result is True
        return result

    app_module.persist_cli_config_for_shutdown = observed_persist

    async def main():
        app = app_module.TldwCli()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            state["mounted"] = True
            await app._confirm_and_quit()

    phase = "lifecycle"
    asyncio.run(main())
except BaseException as error:
    print({RESULT_PREFIX!r} + json.dumps(
        {{"phase": phase, "error_type": type(error).__name__}},
        sort_keys=True,
    ))
    raise SystemExit(2)

print({RESULT_PREFIX!r} + json.dumps(state, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    marker_exposed = any(
        marker in stream
        for marker in (EFFECTIVE_SENTINEL, DECOY_SENTINEL)
        for stream in (result.stdout, result.stderr)
    )
    child_succeeded = result.returncode == 0
    lifecycle_evidence_complete = child_succeeded and _result_payload(
        result.stdout
    ) == {
        "effective_path_selected": True,
        "mounted": True,
        "persistence_called": True,
        "persistence_succeeded": True,
    }
    try:
        decoy_unchanged = decoy_default.read_bytes() == decoy_before
    except OSError:
        decoy_unchanged = False
    try:
        effective_present = (
            effective_config.is_file() and effective_config.stat().st_size > 0
        )
    except OSError:
        effective_present = False

    assert not marker_exposed, "isolated lifecycle exposed a config marker"
    assert child_succeeded, "isolated lifecycle child failed"
    assert lifecycle_evidence_complete, "isolated lifecycle evidence was incomplete"
    assert decoy_unchanged, "isolated lifecycle changed the decoy config"
    assert effective_present, "isolated lifecycle effective config was unavailable"
