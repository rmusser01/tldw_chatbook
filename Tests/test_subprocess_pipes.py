"""Behavioral evidence for bounded subprocess pipe observations."""

import subprocess
import sys
import time

import pytest

from Tests.Backup_Recovery.test_admission import line
from Tests.subprocess_pipes import pipe_ready, popen_with_captured_stderr


@pytest.fixture
def child_process(tmp_path):
    children = []

    def start(source, *, text=True):
        child = popen_with_captured_stderr(
            [sys.executable, "-u", "-c", source],
            tmp_path / f"child-{len(children)}.stderr",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=text,
        )
        children.append(child)
        return child

    yield start
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize("text", [False, True])
def test_line_reads_real_pipe_markers_in_order_without_read_ahead(child_process, text):
    child = child_process(
        "import os, sys; os.write(sys.stdout.fileno(), b'first\\nsecond\\n')",
        text=text,
    )

    assert line(child) == "first"
    assert line(child) == "second"
    assert child.wait(timeout=5) == 0


def test_ready_observation_leaves_response_available_for_the_reader(child_process):
    child = child_process("print('response', flush=True)")
    assert child.wait(timeout=5) == 0

    assert pipe_ready(child.stdout, 0)
    assert pipe_ready(child.stdout, 0)
    assert line(child) == "response"


def test_blocked_child_has_no_output_until_explicit_release(child_process):
    child = child_process(
        "import sys; print('attempting', flush=True); sys.stdin.readline(); "
        "print('entered', flush=True); sys.stdin.readline()"
    )
    assert line(child) == "attempting"

    assert not pipe_ready(child.stdout, 0.05)
    assert child.poll() is None
    child.stdin.write("release\n")
    child.stdin.flush()
    assert pipe_ready(child.stdout, 2)
    assert line(child) == "entered"
    assert not pipe_ready(child.stdout, 0.05)


def test_partial_line_uses_the_same_bounded_response_deadline(child_process):
    child = child_process(
        "import os, sys; print('ready', flush=True); "
        "os.write(sys.stdout.fileno(), b'partial'); sys.stdin.readline()"
    )
    assert line(child) == "ready"
    assert pipe_ready(child.stdout, 2)

    started = time.monotonic()
    with pytest.raises(AssertionError, match="child did not respond"):
        line(child, timeout=0.05)
    assert time.monotonic() - started < 1
    assert child.poll() is None


@pytest.mark.parametrize("partial", [b"", b"unfinished"])
def test_eof_cannot_be_mistaken_for_a_silent_child(child_process, partial):
    child = child_process(
        "import os, sys; print('ready', flush=True); sys.stdin.readline(); "
        f"os.write(sys.stdout.fileno(), {partial!r})"
    )
    assert line(child) == "ready"
    child.stdin.write("close\n")
    child.stdin.flush()
    assert child.wait(timeout=5) == 0

    assert pipe_ready(child.stdout, 2)
    with pytest.raises(AssertionError, match="child exited before response"):
        line(child, timeout=2)


def test_closed_pipe_errors_are_not_reported_as_a_silent_child(child_process):
    child = child_process("import sys; sys.stdin.readline()")
    child.stdout.close()

    with pytest.raises(ValueError):
        pipe_ready(child.stdout, 0.05)
    with pytest.raises(ValueError):
        line(child, timeout=0.05)


def test_child_readiness_observations_are_independent(child_process):
    script = (
        "import sys; print('attempting', flush=True); sys.stdin.readline(); "
        "print(sys.argv[0], flush=True); sys.stdin.readline()"
    )
    first, second = child_process(script), child_process(script)
    assert line(first) == line(second) == "attempting"

    second.stdin.write("release\n")
    second.stdin.flush()
    assert pipe_ready(second.stdout, 2)
    assert not pipe_ready(first.stdout, 0.05)
    assert line(second) == "-c"
    assert not pipe_ready(second.stdout, 0.05)
    first.stdin.write("release\n")
    first.stdin.flush()
    assert line(first) == "-c"


def test_child_marker_is_not_blocked_by_large_preserved_stderr(child_process):
    child = child_process(
        "import os, sys; os.write(sys.stderr.fileno(), b'trace\\n' * 20000); "
        "print('ready', flush=True); sys.stdin.readline()"
    )

    assert line(child, timeout=2) == "ready"
    assert child.stderr.read() == "trace\n" * 20000
    assert child.poll() is None
