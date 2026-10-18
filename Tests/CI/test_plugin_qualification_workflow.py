"""Exercise the qualification entry point with the normal pytest network policy."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/plugin-qualification.yml"


def test_plugin_workflow_preserves_owned_loopback_and_asyncio(tmp_path):
    from tldw_chatbook.config import get_user_data_dir

    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["qualify"]["steps"]
    source = next(
        step["run"]
        for step in steps
        if step.get("name") == "Exercise explicit lifecycle and regression boundaries"
    )
    tree = ast.parse(source)
    argv = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
            and target.attr == "argv"
            for target in node.targets
        )
    )
    allowed = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "allowed"
            for target in node.targets
        )
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["QUALIFICATION_DATA_ROOT"] = str(get_user_data_dir().resolve())
    environment["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    assert environment["TLDW_TEST_CONFIG_ROOT"]
    prefix = "".join(source.splitlines(keepends=True)[: argv.lineno - 1])
    suffix = """
import asyncio
import pytest
from tldw_chatbook.config import get_user_data_dir

asyncio.run(asyncio.sleep(0))
status = pytest.main([
    "Tests/test_network_guard.py::test_loopback_only_mode_covers_connect_connect_ex_and_sendto",
    "-q", "--basetemp=" + TEMP_ROOT,
    "-o", "cache_dir=" + CACHE_ROOT,
])
assert status == 0, status
assert str(get_user_data_dir().resolve()) == os.environ["QUALIFICATION_DATA_ROOT"]
print("WORKFLOW_GUARD_CONTROLS_PASSED")
""".replace("TEMP_ROOT", repr(str(tmp_path / "nested-tmp"))).replace(
        "CACHE_ROOT", repr(str(tmp_path / "nested-cache"))
    )
    child = prefix + textwrap.indent(suffix, "    ")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", child],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WORKFLOW_GUARD_CONTROLS_PASSED" in result.stdout
