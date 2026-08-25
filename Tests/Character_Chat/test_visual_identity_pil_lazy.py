"""Census guard: the warm-boot seeding chain must not import PIL.

TASK-22217: every boot runs ``seed_builtin_content`` ->
``ensure_builtin_samira`` (``config.py`` -> ``Character_Chat/
visual_identity.py``). The preflight exits early on warm boots, but a
module-level ``from PIL import Image, UnidentifiedImageError`` in
``visual_identity.py`` made every boot pay the ~80-module PIL import
before the preflight could run. PIL belongs only to the code paths that
actually decode image bytes (fresh-profile seeding, pack validation).

The guard runs the REAL chain twice, each in a clean interpreter against
one scratch profile:

* Phase 1 (fresh profile): ``seed_builtin_content`` on an empty DB. This
  is the real image-work path -- PIL is expected and allowed. The phase
  asserts the Samira card AND pack were actually created, so phase 2
  cannot silently measure a half-seeded (non-terminal) profile.
* Phase 2 (warm boot): a new interpreter opens the same DB and runs
  ``seed_builtin_content`` again. The chain must import
  ``visual_identity`` (proving the census exercises the real import
  chain, not a stub) while loading no ``PIL*`` module at all.

Scope note (honest census): this guard covers the seeding chain only.
At ``_ui_ready`` in a full app boot PIL is still present via
chat_screen's own pre-first-paint import chains -- finding 22213 in
``Docs/Design/2026-08-24-holistic-perf-review.md``, not this task's
scope -- so a whole-process census at ``_ui_ready`` cannot pass today
and would not isolate a regression in this chain if it could.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PHASE_1_FRESH_SEED = """
import sys
from pathlib import Path

from tldw_chatbook.config import seed_builtin_content
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

db = CharactersRAGDB(db_path=Path(sys.argv[1]), client_id="test-pil-lazy-seed")
try:
    seed_builtin_content(db)

    from tldw_chatbook.Character_Chat.visual_identity import (
        _find_builtin_samira_card,
        _find_builtin_samira_pack,
    )

    assert _find_builtin_samira_card(db) is not None, "fresh seed created no card"
    assert _find_builtin_samira_pack(db) is not None, "fresh seed created no pack"
finally:
    db.close_connection()
print("SEEDED")
"""

_PHASE_2_WARM_CENSUS = """
import sys
from pathlib import Path

preloaded = sorted(n for n in sys.modules if n == "PIL" or n.startswith("PIL."))
assert not preloaded, f"PIL preloaded before the chain ran: {preloaded[:5]}"

from tldw_chatbook.config import seed_builtin_content
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

db = CharactersRAGDB(db_path=Path(sys.argv[1]), client_id="test-pil-lazy-warm")
try:
    seed_builtin_content(db)

    assert "tldw_chatbook.Character_Chat.visual_identity" in sys.modules, (
        "the seeding chain never imported visual_identity -- "
        "this census is not measuring the real chain"
    )
    from tldw_chatbook.Character_Chat.visual_identity import _samira_seed_preflight

    assert _samira_seed_preflight(db)["terminal"] is True, (
        "profile is not terminal after the warm pass -- phase 1 half-seeded"
    )
    offenders = sorted(n for n in sys.modules if n == "PIL" or n.startswith("PIL."))
    assert not offenders, (
        "warm-boot seeding chain imported PIL (module-level import back in "
        f"visual_identity.py, or the profile was not terminal): {offenders[:5]}"
    )
finally:
    db.close_connection()
print("WARM-NO-PIL")
"""


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_DATA_HOME": str(data),
            "XDG_CONFIG_HOME": str(config_dir),
            "TLDW_CONFIG_PATH": str(config_dir / "config.toml"),
            "TLDW_TEST_MODE": "1",
        }
    )
    return env


def _run_phase(script: str, db_path: Path, env: dict[str, str], marker: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script, str(db_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"phase failed (rc={result.returncode}):\n{result.stderr[-4000:]}"
    )
    assert marker in result.stdout, (
        f"phase produced no {marker!r} marker:\n{result.stdout[-2000:]}"
    )


@pytest.mark.integration
def test_warm_seeding_chain_loads_no_pil(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    db_path = tmp_path / "profile" / "chachanotes.db"
    db_path.parent.mkdir(mode=0o700)

    _run_phase(_PHASE_1_FRESH_SEED, db_path, env, "SEEDED")
    _run_phase(_PHASE_2_WARM_CENSUS, db_path, env, "WARM-NO-PIL")
