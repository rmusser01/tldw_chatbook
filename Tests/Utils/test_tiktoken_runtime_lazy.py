"""The bundled tiktoken reader arms lazily and retries a failed first import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_child(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_token_counter_import_leaves_bundled_arming_intact() -> None:
    """A tokenizer availability probe cannot consume the one-shot import hook."""
    result = _run_child(
        "import sys\n"
        "import tldw_chatbook\n"
        "import tldw_chatbook.Utils.token_counter\n"
        "assert 'tiktoken' not in sys.modules\n"
        "from tldw_chatbook.Utils import tiktoken_runtime as runtime\n"
        "assert any(isinstance(finder, runtime._TiktokenRuntimeArmingFinder) "
        "for finder in sys.meta_path)\n"
        "import tiktoken.load\n"
        "assert tiktoken.load.read_file_cached is runtime._read_bundled_file\n"
    )
    assert result.returncode == 0, result.stderr


def test_failed_first_import_rearms_bundled_runtime() -> None:
    """A transient installer failure cannot leave later imports unprotected."""
    result = _run_child(
        "import os, sys\n"
        "os.environ.pop('TIKTOKEN_CACHE_DIR', None)\n"
        "os.environ.pop('DATA_GYM_CACHE_DIR', None)\n"
        "import tldw_chatbook\n"
        "from tldw_chatbook.Utils import tiktoken_runtime as runtime\n"
        "real_install = runtime.install_tiktoken_runtime\n"
        "attempts = []\n"
        "def fail_once():\n"
        "    attempts.append(1)\n"
        "    if len(attempts) == 1:\n"
        "        raise RuntimeError('transient install failure')\n"
        "    real_install()\n"
        "runtime.install_tiktoken_runtime = fail_once\n"
        "try:\n"
        "    import tiktoken\n"
        "except RuntimeError as error:\n"
        "    assert str(error) == 'transient install failure'\n"
        "else:\n"
        "    raise AssertionError('first import unexpectedly succeeded')\n"
        "assert 'tiktoken' not in sys.modules\n"
        "import tiktoken\n"
        "import tiktoken.load\n"
        "assert len(attempts) == 2, attempts\n"
        "assert tiktoken.load.read_file_cached is runtime._read_bundled_file\n"
        "assert os.environ['TIKTOKEN_CACHE_DIR'].endswith('tiktoken_cache')\n"
    )
    assert result.returncode == 0, result.stderr
