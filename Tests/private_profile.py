"""Opt-in full-app tests whose selected profile owns one interpreter lifetime."""

import asyncio
import inspect
import os
import sys

# The XML comes only from the selected pytest child in its private directory.
import xml.etree.ElementTree as ET  # nosec B405
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pytest_timeout import get_env_settings

_CHILD_NODE = "TLDW_TEST_PRIVATE_PROFILE_NODE"
_NO_CHILD_FAILURE = (
    "(the child's JUnit report records no failure or error; see the log tail)"
)


def _private_profile_checkpoint(phase, *, launcher_pid=None):
    """Opt-in bounded diagnostic metadata; never a readiness/cleanup proof."""
    prefix = os.environ.get("TLDW_TEST_PRIVATE_PROFILE_STAGE_PREFIX")
    if not prefix:
        return
    try:
        import hashlib
        import json
        import time

        role = "child" if _CHILD_NODE in os.environ else "parent"
        target = Path(prefix + ".private-" + role + ".json")
        source = Path(__file__)
        row = {
            "diagnostic_only": True,
            "terminal_source_native_coverage": False,
            "role": role,
            "phase": phase,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "launcher_pid": launcher_pid,
            "monotonic_ns": time.monotonic_ns(),
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        temporary = target.with_name(target.name + "." + str(os.getpid()) + ".tmp")
        temporary.write_text(json.dumps(row), encoding="utf-8")
        temporary.replace(target)
    except OSError:
        # A missing checkpoint is explicitly incomplete, never a body failure.
        return


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


def private_profile_test(function):
    """Run the original case under pytest after selecting a fresh profile."""

    @wraps(function)
    async def wrapped(*args, **kwargs):
        request = kwargs["request"]
        if is_private_profile_child(request):
            _private_profile_checkpoint("original_body_entered")
            try:
                result = function(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
            finally:
                _private_profile_checkpoint("original_body_exited")
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
            PYTHONIOENCODING="utf-8",
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
        report = work / "pytest.xml"
        log = work / "pytest.log"
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
        _private_profile_checkpoint("child_spawn_requested")
        with log.open("w", encoding="utf-8") as output:
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
                f"--timeout={timeout}",
                f"--basetemp={work / 'pytest'}",
                f"--junitxml={report}",
                *coverage_args,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            _private_profile_checkpoint("child_spawned", launcher_pid=process.pid)
            try:
                await asyncio.wait_for(process.wait(), timeout if timeout > 0 else None)
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
            pytest.fail(
                f"{log}\n{_child_failure_report(report)}\n"
                f"--- child log tail ---\n{log.read_text(encoding="utf-8")[-16000:]}"
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


_private_profile_checkpoint("module_loaded")
