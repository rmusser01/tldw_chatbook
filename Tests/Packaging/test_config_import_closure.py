"""Import-closure guard: config load must not import feature packages.

TASK-22223. `config.py` runs `load_settings()` at module scope, so anything
`_load_settings_uncached` imports is executed on EVERY `import
tldw_chatbook.config` -- inside the app import closure, and inside every
feature module that imports config. Before this guard, `config.py:1802`
imported `Library.library_adaptive_reader_state` for one pure normalization
function under a comment claiming the import was lazy. It was not: the
`Library` package `__init__` chain pulled `library_collections_service ->
library_collections_state -> Sync_Interop (29 modules) -> Chat (10) ->
Skills_Interop (10) -> runtime_policy` through config load. Beyond the
layering cost, that closed a LIVE import cycle: any module that imports
`runtime_policy.bootstrap` BEFORE config (e.g.
`Character_Chat/server_character_persona_service.py`) hit
`bootstrap -> config -> Library -> ... -> Chat ->
server_chat_conversation_service -> bootstrap (partially initialized)` and
died with ImportError at collection time
(`Tests/Character_Chat/test_character_persona_scope_service.py` run solo
reproduced it).

The normalizer now lives in the stdlib-only leaf
`tldw_chatbook/Utils/adaptive_reader_state.py` (`Utils/__init__.py` is
empty), which both config and the Library package import -- one source of
truth, no feature package on the config path.

Subprocess-isolated for the same reason as
`test_app_import_diet_closure.py`, whose pattern this file follows:
`sys.modules` is process-global, so an earlier test that legitimately
imported Library/Chat would false-fail this check in-process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the config import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_CONFIG_CLOSURE_SNIPPET = """
import sys

import tldw_chatbook.config  # noqa: F401

# Feature packages that must never ride config load. Library is the package
# this guard was written for; the other three are the cascade its package
# __init__ dragged in, and runtime_policy is the far side of the import
# cycle config->Library used to close. If config genuinely needs something
# from one of these, that is the layering decision to review -- move the
# needed logic into a config-safe leaf (see Utils/adaptive_reader_state.py
# for the shape), do not delete it from this list.
forbidden_packages = (
    "tldw_chatbook.Library",
    "tldw_chatbook.Chat",
    "tldw_chatbook.Sync_Interop",
    "tldw_chatbook.Skills_Interop",
    "tldw_chatbook.runtime_policy",
)
resident = sorted(
    m for m in sys.modules
    if sys.modules[m] is not None
    and any(m == pkg or m.startswith(pkg + ".") for pkg in forbidden_packages)
)
assert not resident, (
    "feature-package modules resident after import tldw_chatbook.config: "
    + repr(resident)
)

# Anti-vacuity: the config-safe leaf that replaced the Library import must
# still be part of config's closure -- load_settings() normalizes adaptive
# reader preferences through it at module import. If it leaves, the
# assertions above pass without testing the relayering at all.
assert sys.modules.get("tldw_chatbook.Utils.adaptive_reader_state") is not None, (
    "Utils.adaptive_reader_state not resident after config import -- the "
    "normalizer seam moved; update this guard alongside it"
)
print("OK")
"""


def test_config_import_stays_out_of_feature_packages(tmp_path: Path) -> None:
    """A bare `import tldw_chatbook.config` must not pull feature packages.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _CONFIG_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        f"config import-closure guard failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-3000:]}"
    )
    assert "OK" in result.stdout
