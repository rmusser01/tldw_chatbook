"""Opt-in full-app tests whose selected profile owns one interpreter lifetime."""

import asyncio
import contextlib
import inspect
import os
import signal
import sys
import threading
import traceback

# The XML comes only from the selected pytest child in its private directory.
import xml.etree.ElementTree as ET  # nosec B405
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pytest_timeout import get_env_settings

_CHILD_NODE = "TLDW_TEST_PRIVATE_PROFILE_NODE"
_CHILD_STACKS = "TLDW_TEST_PRIVATE_PROFILE_STACKS"
_NO_CHILD_FAILURE = (
    "(the child's JUnit report records no failure or error; see the log tail)"
)
#: How much later the parent's own pytest-timeout fires, so that the child's
#: fires first (TASK-34789). They used to be equal and the parent's clock starts
#: first, so a hung child was only ever a bare parent timeout with the child's
#: log unread. The child keeps the whole budget the test had (a little more:
#: its start-up no longer counts against it), so this takes nothing from a
#: passing test; the slowest passing private-profile test in 189 CI logs took
#: 127.1 s of its 180 s. A hung test is reported up to this much later than the
#: lane's --timeout, by the child's own timeout naming the lane's value; the
#: lead covers the child's start-up (6.6 s for a trivial test in CI) and its
#: report. Should the parent's own timeout still fire, it too names the lane's
#: value.
_CHILD_LEAD_SECONDS = 30.0
#: What the parent keeps of its own limit to end a child that is still running
#: and report it, before the parent's pytest-timeout would fire instead.
_PARENT_RESERVE_SECONDS = 5.0
#: The child's test coroutine while it runs, for ``_note_stacks``.
_awaiting = None


def is_private_profile_child(request: pytest.FixtureRequest) -> bool:
    """Recognize only the exact opted-in case selected by its parent test."""
    return os.environ.get(_CHILD_NODE) == request.node.nodeid and getattr(
        request.function, "_private_profile_test", False
    )


def _child_failure_report(report: Path) -> str:
    """The child's own failure/error text from its JUnit XML, if it wrote one.

    The log tail alone is often teardown noise (DEBUG SQL lines), which left
    the failing assertion out of the parent's message entirely. A report with
    no failure/error element (exit 5: the child selected nothing) says so,
    rather than leaving a blank section before the tail.
    """
    try:
        cases = list(ET.parse(report).iter("testcase"))  # nosec B314
    except (OSError, ET.ParseError):
        return "(the child wrote no JUnit report)"
    quoted = "\n".join(
        f"--- child {element.tag}: {element.get('message', '')}\n{element.text or ''}"
        for case in cases
        for element in case
        if element.tag in ("failure", "error")
    )
    return quoted or _NO_CHILD_FAILURE


def _extend_own_alarm(armed: float) -> float:
    """Move the parent's own pytest-timeout ``_CHILD_LEAD_SECONDS`` later.

    pytest-timeout's signal method keeps its deadline only in ITIMER_REAL, and
    cancels the timer whatever its value when the test ends. Its thread method
    (Windows; a test off the main thread) arms no itimer and is left alone.

    Args:
        armed: Seconds left on ITIMER_REAL; 0 when it is not armed.

    Returns:
        The lead given, or 0.0 when the parent's limit did not move.
    """
    if armed <= 0 or not callable(signal.getsignal(signal.SIGALRM)):
        return 0.0
    signal.setitimer(signal.ITIMER_REAL, armed + _CHILD_LEAD_SECONDS)
    return _CHILD_LEAD_SECONDS


def _child_deadlines(budget: float, lead: float) -> tuple[float, float | None]:
    """The child's ``--timeout`` and how long the parent waits for the child.

    Args:
        budget: Seconds left of the test's own timeout; 0 for none.
        lead: How far the parent's own limit moved later; 0 for not at all.

    Returns:
        The child keeps the whole budget. Where the parent's limit moved, the
        parent waits until ``_PARENT_RESERVE_SECONDS`` before the moved limit;
        where it did not, it waits as it always did (``None``: no limit).
    """
    if budget <= 0:
        return 0.0, None
    if not lead:
        return budget, budget
    return budget, budget + lead - _PARENT_RESERVE_SECONDS


def _note_stacks(*_signal) -> None:
    """In the child: append every thread's stack and the test's chain of awaits.

    Runs when the child's own pytest-timeout fires and when the parent asks
    (SIGUSR1). pytest-timeout raises into the event loop, so its report cannot
    say where an async test was waiting, and its thread dump goes through
    pytest's capture, where a child that then cannot finish loses it.
    """
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        parts.append(f"--- thread {names.get(ident, ident)}\n")
        parts.extend(traceback.format_stack(frame))
    parts.append("--- the test, outermost await first\n")
    awaiting = _awaiting
    while awaiting is not None:
        frame = getattr(awaiting, "cr_frame", None)
        if frame is not None:
            parts.extend(traceback.format_stack(frame))
        awaiting = getattr(awaiting, "cr_await", None)
    with open(os.environ[_CHILD_STACKS], "a") as output:
        output.write("".join(parts))


async def _exited_within(process, seconds: float | None) -> bool:
    """Whether the child exits by itself within ``seconds`` (``None``: wait)."""
    try:
        await asyncio.wait_for(process.wait(), seconds)
    except TimeoutError:
        return False
    return True


def private_profile_test(function):
    """Run the original case under pytest after selecting a fresh profile."""

    @wraps(function)
    async def wrapped(*args, **kwargs):
        global _awaiting
        request = kwargs["request"]
        if is_private_profile_child(request):
            # pytest-timeout's signal method, where there is one (not Windows).
            own_timeout = (
                signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None
            )
            if os.environ.get(_CHILD_STACKS) and callable(own_timeout):

                def note_then_time_out(signum, frame):
                    __tracebackhide__ = True
                    with contextlib.suppress(Exception):
                        _note_stacks()
                    own_timeout(signum, frame)

                # pytest-timeout resets SIGALRM itself when the test ends.
                signal.signal(signal.SIGALRM, note_then_time_out)
                signal.signal(signal.SIGUSR1, _note_stacks)
            result = function(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result
            _awaiting = result
            try:
                return await result
            finally:
                _awaiting = None
        if _CHILD_NODE in os.environ:
            pytest.fail("private profile child node changed")
        work = request.getfixturevalue("tmp_path") / "private-profile-process"
        work.mkdir(mode=0o700)
        profile = work / "profile"
        profile.mkdir(mode=0o700)
        environment = dict(os.environ)
        environment.update(
            TLDW_TEST_PRIVATE_PROFILE_NODE=request.node.nodeid,
            TLDW_TEST_CONFIG_ROOT=str(profile),
            HOME=str(profile / "home"),
            USERPROFILE=str(profile / "home"),
            XDG_CONFIG_HOME=str(profile / "config"),
            XDG_DATA_HOME=str(profile / "data"),
            TLDW_CONFIG_PATH=str(profile / "config" / "config.toml"),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        )
        for name in (
            "TLDW_TEST_CONFIG_ROOT_OWNER",
            "PYTEST_XDIST_WORKER",
            "PYTEST_XDIST_WORKER_COUNT",
            "PYTEST_ADDOPTS",
        ):
            environment.pop(name, None)
        marker = request.node.get_closest_marker("timeout")
        timeout = marker.args[0] if marker and marker.args else None
        if timeout is None and marker:
            timeout = marker.kwargs.get("timeout")
        if timeout is None:
            timeout = get_env_settings(request.config).timeout
        timeout = float(timeout or 0)
        # The signal method arms ITIMER_REAL for this test: what is left of it
        # is the real budget, setup already having spent some.
        armed = (
            signal.getitimer(signal.ITIMER_REAL)[0]
            if hasattr(signal, "getitimer")
            else 0.0
        )
        child_timeout, wait = _child_deadlines(
            min(armed or timeout, timeout), _extend_own_alarm(armed)
        )
        report = work / "pytest.xml"
        log = work / "pytest.log"
        stacks = work / "stacks.txt"
        environment[_CHILD_STACKS] = str(stacks)
        coverage_plugin = request.config.pluginmanager.getplugin("_cov")
        coverage_controller = getattr(coverage_plugin, "cov_controller", None)
        coverage_args = []
        coverage_file = work / ".coverage"
        if coverage_controller is not None:
            # Keep each child's data private; the parent retains its report gate.
            environment["COVERAGE_FILE"] = str(coverage_file)
            for name in tuple(environment):
                if name.startswith("COV_CORE_"):
                    environment.pop(name)
            coverage_args = [
                "-p",
                "pytest_cov.plugin",
                "--cov-report=",
                "--cov-fail-under=0",
            ]
            sources = coverage_controller.cov_source
            coverage_args += (
                [f"--cov={source}" for source in sources] if sources else ["--cov"]
            )
            config_file = coverage_controller.cov.config.config_file
            if config_file:
                coverage_args += [f"--cov-config={config_file}"]
            if coverage_controller.cov.config.branch:
                coverage_args += ["--cov-branch"]
            if coverage_plugin.options.cov_context:
                coverage_args += [
                    f"--cov-context={coverage_plugin.options.cov_context}"
                ]
        with log.open("w") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                request.node.nodeid,
                "-p",
                "pytest_asyncio.plugin",
                "-p",
                "pytest_timeout",
                "-q",
                f"--timeout={child_timeout}",
                f"--basetemp={work / 'pytest'}",
                f"--junitxml={report}",
                *coverage_args,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                exited = await _exited_within(process, wait)
                if not exited and hasattr(signal, "SIGUSR1"):
                    # Past its own limit and still running (a teardown or an
                    # exit that cannot finish): ask where before ending it.
                    with contextlib.suppress(ProcessLookupError):
                        process.send_signal(signal.SIGUSR1)
                    await _exited_within(process, 1.0)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if coverage_controller is not None and coverage_file.is_file():
            # pytest-cov combines parallel files for serial and xdist reports.
            parent_file = Path(
                coverage_controller.topdir, coverage_controller.cov.config.data_file
            ).resolve()
            with NamedTemporaryFile(
                prefix=parent_file.name + ".", dir=parent_file.parent, delete=False
            ) as output:
                output.write(coverage_file.read_bytes())
        if process.returncode:
            asked = stacks.read_text() if stacks.is_file() else ""
            pytest.fail(
                f"{log}\n"
                + ("" if exited else f"(killed: still running after {wait:.0f}s)\n")
                + f"{_child_failure_report(report)}\n"
                + (f"--- child stacks at its deadline ---\n{asked}" if asked else "")
                + f"--- child log tail ---\n{log.read_text()[-16000:]}"
            )
        if coverage_controller is not None and not coverage_file.is_file():
            pytest.fail(f"private profile child did not produce coverage: {log}")
        # Parse our child's local pytest output, never externally supplied XML.
        cases = list(ET.parse(report).iter("testcase"))  # nosec B314
        if len(cases) != 1 or cases[0].get("name") != request.node.name:
            pytest.fail(
                f"private profile child did not execute its exact case: {report}"
            )
        if any(cases[0].find(kind) is not None for kind in ("failure", "error")):
            pytest.fail(
                f"private profile child did not pass its original case: {report}"
            )
        skipped = cases[0].find("skipped")
        if skipped is not None:
            pytest.skip(skipped.get("message", "original private-profile case skipped"))

    wrapped._private_profile_test = True
    return wrapped
