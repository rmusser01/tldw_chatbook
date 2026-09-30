"""Real parent reports must retain execution from private-profile children."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.mark.parametrize("mode", ["serial", "xdist", "standalone", "disabled"])
def test_private_profile_child_coverage_reaches_parent_report(tmp_path, mode):
    repo = Path(__file__).resolve().parents[1]
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "child_only.py").write_text(
        "def value(flag):\n    if flag:\n        return 1\n    return 2\n"
    )
    with TemporaryDirectory(dir=repo / ".pytest_cache") as case_dir:
        case = Path(case_dir) / "test_child_probe.py"
        case.write_text(
            "import pytest\n"
            "from Tests.private_profile import private_profile_test\n"
            "@pytest.mark.asyncio\n"
            '@pytest.mark.parametrize("flag", [True, False])\n'
            "@private_profile_test\n"
            "async def test_child_only(request, flag):\n"
            "    from child_only import value\n"
            "    assert value(flag) == (1 if flag else 2)\n"
        )
        report = tmp_path / "coverage.xml"
        env = dict(os.environ)
        env.update(
            PYTHONPATH=os.pathsep.join((str(repo), str(probe))),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTEST_ADDOPTS="",
            COVERAGE_FILE=str(tmp_path / ".coverage"),
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(case),
            f"--rootdir={repo}",
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "pytest_timeout",
            "-q",
            f"--basetemp={tmp_path / 'parent'}",
        ]
        if mode != "standalone":
            command += [
                "-p",
                "pytest_cov.plugin",
                f"--cov={probe}",
                "--cov-branch",
                f"--cov-report=xml:{report}",
            ]
        if mode == "xdist":
            command += ["-p", "xdist.plugin", "-n", "2"]
        if mode == "disabled":
            command += ["--no-cov"]
        result = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode in {"standalone", "disabled"}:
        assert not report.exists()
        assert not list(tmp_path.rglob(".coverage*"))
        return
    tree = ET.parse(report)
    child = next(
        item for item in tree.iter("class") if item.get("filename") == "child_only.py"
    )
    lines = {int(line.get("number")): line for line in child.iter("line")}
    assert all(int(lines[number].get("hits")) > 0 for number in (1, 2, 3, 4))
    assert lines[2].get("condition-coverage") == "100% (2/2)"
