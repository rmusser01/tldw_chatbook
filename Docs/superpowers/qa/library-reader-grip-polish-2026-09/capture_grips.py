"""Bootstrap the production-shaped Library Media grip capture utility."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
QA_ROOT = Path(tempfile.gettempdir()) / "tldw-chatbook-library-reader-grip-polish-qa"


def _configure_environment() -> None:
    """Prepare isolated configuration before importing the application."""
    qa_config = QA_ROOT / "config.toml"
    qa_data = QA_ROOT / "data"
    qa_data.mkdir(parents=True, exist_ok=True)
    qa_config.write_text(f'[paths]\ndata_dir = "{qa_data}"\n', encoding="utf-8")
    os.environ["TLDW_CONFIG_PATH"] = str(qa_config)
    os.environ["XDG_CONFIG_HOME"] = str(QA_ROOT / "xdg-config")
    os.environ["XDG_DATA_HOME"] = str(QA_ROOT / "xdg-data")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    homebrew_lib = Path("/opt/homebrew/lib")
    if homebrew_lib.is_dir():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(homebrew_lib))


def main() -> None:
    """Configure the process and run the capture implementation."""
    _configure_environment()
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(Path(__file__).with_name("_capture_grips_impl.py")), run_name="__main__")


if __name__ == "__main__":
    main()
