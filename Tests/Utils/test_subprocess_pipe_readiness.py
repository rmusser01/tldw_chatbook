"""Actual child-pipe bytes, timeout and EOF; no platform emulation."""

import os
import subprocess
import sys
import time

from Tests.pipe_readiness import pipe_readable


def test_child_response_readiness_does_not_consume_bytes():
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; sys.stdin.buffer.read(1); print('actual-response', flush=True)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as child:
        try:
            started = time.monotonic()
            assert not pipe_readable(child.stdout, 0.1)
            assert time.monotonic() - started < 1
            child.stdin.write(b"x")
            child.stdin.flush()
            assert pipe_readable(child.stdout, 10)
            assert child.stdout.readline() == b"actual-response" + os.linesep.encode()
            assert child.wait(timeout=10) == 0
            assert pipe_readable(child.stdout, 10)
            assert child.stdout.read() == b""
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
