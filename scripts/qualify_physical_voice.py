#!/usr/bin/env python3
"""Run speculative-voice physical-device qualification."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Packaging.voice_physical_reports import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
