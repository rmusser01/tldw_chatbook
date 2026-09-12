#!/usr/bin/env python3
"""Run the speculative duplex voice qualification harness."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Packaging.qualify_speculative_voice import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
