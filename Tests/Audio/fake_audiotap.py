"""Stand-in for the macOS `tldw-audiotap` helper and for `parec`.

Emits `--frames` frames of 640 bytes on stdout, then either holds until
stdin closes (`--hold`) or exits with `--exit-code`.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    sys.stderr.write("READY\n")
    sys.stderr.flush()
    frame = bytes([1, 0]) * 320
    out = sys.stdout.buffer
    for _ in range(args.frames):
        out.write(frame)
    out.flush()
    if args.hold:
        sys.stdin.buffer.read()  # block until the parent closes stdin
        return 0
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
