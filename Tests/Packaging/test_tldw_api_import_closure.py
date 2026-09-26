"""`tldw_api.client` does not drag the STT subsystem onto the boot path.

Tier-2 review S06, P2 [D2]: `tldw_api/media_reading_schemas.py` imported
four names from `STT.persistence` at module scope for two field
validators, and `client.py` imports `media_reading_schemas` at module
scope. Measured on 2026-09-21: `import tldw_chatbook.STT.persistence`
alone is ~94-99 ms and pulls 9 STT submodules;
`import tldw_chatbook.tldw_api.client` was ~475-630 ms with all 9 present
every time -- ~20% of the client's import cost for two validators that
fire only when a media record carries transcription provenance. ~44
`Server*Service` modules import `TLDWAPIClient` at module level, several
of them from `app.py` module scope.

The fix is the shape TASK-23023 established for
`Research_Workspace/server_adapter.py` (see
`test_research_workspace_import_closure.py`): the import moves into the
function bodies that use it.

Subprocess-isolated because `sys.modules` is process-global: an earlier
test that legitimately imported STT would make an in-process check a
false pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import sys
import tldw_chatbook.tldw_api.client  # noqa: F401
print(sorted(m for m in sys.modules if m.startswith("tldw_chatbook.STT")))
"""

# `_normalize_transcription_provenance(None)` returns on its first line and
# never reaches the deferred import, so the probe it used to run was vacuous:
# it stayed green with the import removed entirely. A non-`None` value is what
# gets past that early return. The document is then rejected by
# `STT.persistence`'s own validator -- which is the point: reaching a
# `ValueError` raised inside that module, with the module now in
# `sys.modules`, is direct evidence the deferred import resolved. An
# `ImportError` propagates and fails the subprocess.
_PROBE_STILL_WORKS = """
import sys
from tldw_chatbook.tldw_api.media_reading_schemas import (
    _normalize_transcription_provenance,
)

assert not [m for m in sys.modules if m.startswith("tldw_chatbook.STT")], (
    "STT was already imported before the validator ran"
)
try:
    _normalize_transcription_provenance({})
except ValueError:
    pass
print(sorted(m for m in sys.modules if m == "tldw_chatbook.STT.persistence"))
"""


def _run(code: str, tmp_path: Path) -> str:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_api_client_does_not_import_the_stt_subsystem(tmp_path):
    assert _run(_PROBE, tmp_path) == "[]", (
        "tldw_api.client pulled STT modules into its import closure again -- "
        "move the STT.persistence import back into the validator bodies of "
        "tldw_api/media_reading_schemas.py"
    )


def test_the_deferred_import_still_resolves_when_the_validator_runs(tmp_path):
    """Anti-vacuity: a deferred import that does not resolve is worse."""
    assert (
        _run(_PROBE_STILL_WORKS, tmp_path) == "['tldw_chatbook.STT.persistence']"
    ), (
        "the validator body ran without resolving its deferred "
        "tldw_chatbook.STT.persistence import"
    )
