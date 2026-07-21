# Tests/Internal_Prompts/test_import_hygiene.py
"""The Internal_Prompts package must stay off the config import chain.

P2 adds more prompt modules to this package; a careless top-level config
import would silently drag config.py into cold start. Runs in a subprocess
because in-process sys.modules is polluted by other tests' imports.
"""

import subprocess
import sys


def test_package_import_does_not_import_config():
    code = (
        "import sys; import tldw_chatbook.Internal_Prompts; "
        "sys.exit(1 if 'tldw_chatbook.config' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "tldw_chatbook.Internal_Prompts imported tldw_chatbook.config at "
        f"module import time. stderr: {result.stderr}"
    )
