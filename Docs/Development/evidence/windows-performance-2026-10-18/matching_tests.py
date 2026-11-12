"""Run focused checks with genuine user-owned native Windows fixtures."""

import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Tests.windows_private_fixture_runner import user_fixture_default_owner  # noqa: E402

with user_fixture_default_owner():
    scratch = tempfile.mkdtemp(prefix=".tmp-windows-matching-tests-", dir=Path.home())
    print("Fixture root:", scratch, flush=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = scratch
    tempfile.tempdir = scratch
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
