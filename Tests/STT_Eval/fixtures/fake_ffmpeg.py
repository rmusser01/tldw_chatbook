#!/usr/bin/env python3
"""Tiny executable used to exercise the real subprocess boundary in tests."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


if sys.argv[1:] == ["-version"]:
    print("ffmpeg version fake-593 exact test build")
    print("configuration: deterministic test fixture")
    raise SystemExit(0)

if os.environ.get("FAKE_FFMPEG_FAIL") == "1":
    print("requested fake conversion failure", file=sys.stderr)
    raise SystemExit(23)

try:
    input_path = Path(sys.argv[sys.argv.index("-i") + 1])
    output_path = Path(sys.argv[-1])
except (ValueError, IndexError):
    print("invalid fake ffmpeg argument vector", file=sys.stderr)
    raise SystemExit(2)

if "-n" not in sys.argv or output_path.exists():
    print("fake ffmpeg requires a fresh -n output", file=sys.stderr)
    raise SystemExit(3)

shutil.copyfile(input_path, output_path)
