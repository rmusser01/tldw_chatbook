"""Opt-in full-app tests whose selected profile owns one interpreter lifetime."""

import asyncio
import os
import sys

# The XML comes only from the selected pytest child in its private directory.
import xml.etree.ElementTree as ET  # nosec B405
from functools import wraps
from pathlib import Path

import pytest
from pytest_timeout import get_env_settings

_CHILD_NODE = "TLDW_TEST_PRIVATE_PROFILE_NODE"


def is_private_profile_child(request: pytest.FixtureRequest) -> bool:
    """Recognize only the exact opted-in case selected by its parent test."""
    return (
        os.environ.get(_CHILD_NODE) == request.node.nodeid
        and getattr(request.function, "_private_profile_test", False)
    )


def private_profile_test(function):
    """Run the original async case under pytest after selecting a fresh profile."""
    @wraps(function)
    async def wrapped(*args, **kwargs):
        request = kwargs["request"]
        if is_private_profile_child(request):
            return await function(*args, **kwargs)
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
            "TLDW_TEST_CONFIG_ROOT_OWNER", "PYTEST_XDIST_WORKER",
            "PYTEST_XDIST_WORKER_COUNT", "PYTEST_ADDOPTS",
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
        with log.open("w") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", request.node.nodeid,
                "-p", "pytest_asyncio.plugin", "-p", "pytest_timeout", "-q",
                f"--timeout={timeout}", f"--basetemp={work / 'pytest'}",
                f"--junitxml={report}",
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout if timeout > 0 else None)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if process.returncode:
            pytest.fail(f"{log}\n{log.read_text()[-16000:]}")
        # Parse our child's local pytest output, never externally supplied XML.
        cases = list(ET.parse(report).iter("testcase"))  # nosec B314
        if len(cases) != 1 or cases[0].get("name") != request.node.name:
            pytest.fail(f"private profile child did not execute its exact case: {report}")
        if any(cases[0].find(kind) is not None for kind in ("skipped", "failure", "error")):
            pytest.fail(f"private profile child did not pass its original case: {report}")

    wrapped._private_profile_test = True
    return wrapped
