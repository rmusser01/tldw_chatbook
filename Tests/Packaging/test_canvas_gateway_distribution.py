"""Installed-wheel qualification for the native Canvas gateway."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from Tests.Packaging.test_installed_distribution import (
    _copy_build_inputs,
    _sanitized_build_env,
)

pytestmark = pytest.mark.integration

CANVAS_GATEWAY_PATHS = frozenset(
    {
        "tldw_chatbook/Canvas/capabilities.py",
        "tldw_chatbook/Canvas/gateway.py",
        "tldw_chatbook/Canvas/native_authority.py",
        "tldw_chatbook/Canvas/static/THIRD_PARTY_LICENSES.txt",
        "tldw_chatbook/Canvas/static/canvas_renderer.js",
        "tldw_chatbook/Canvas/static/canvas_shell.css",
        "tldw_chatbook/Canvas/static/canvas_shell.html",
        "tldw_chatbook/Canvas/static/canvas_shell.js",
        "tldw_chatbook/Canvas/static/canvas_runtime_worker.js",
        "tldw_chatbook/Canvas/static/quickjs-runtime.js",
        "tldw_chatbook/Canvas/static/runtime-manifest.json",
    }
)

_WHEEL_PROBE = r"""
import sys

wheel = sys.argv[1]
sys.path.insert(0, wheel)

from tldw_chatbook.Canvas.gateway import CanvasGateway
from tldw_chatbook.Canvas.runtime_assets import load_canvas_runtime_assets

gateway = CanvasGateway(authority=object())
assets = load_canvas_runtime_assets()
assert ".whl/" in sys.modules[CanvasGateway.__module__].__file__
assert gateway.started is False
assert assets.enabled
assert assets.renderer_javascript
assert assets.worker_javascript
print("canvas-gateway-wheel-ok")
"""


@pytest.fixture(scope="module")
def canvas_gateway_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source_root = tmp_path_factory.mktemp("canvas-gateway-distribution-source")
    _copy_build_inputs(source_root)
    dist_dir = source_root / "dist"
    command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=_sanitized_build_env(source_root / "build-state"),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_canvas_gateway_and_core_dependency_ship_in_wheel(
    canvas_gateway_wheel: Path,
) -> None:
    with zipfile.ZipFile(canvas_gateway_wheel) as archive:
        members = set(archive.namelist())
        metadata_names = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))

    assert CANVAS_GATEWAY_PATHS <= members
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist") or []
    ]
    aiohttp_requirements = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "aiohttp"
        and requirement.marker is None
    ]
    assert len(aiohttp_requirements) == 1
    requirement = aiohttp_requirements[0]
    assert str(requirement.specifier) == "<4,>=3.9"
    assert requirement.marker is None
    assert not requirement.extras
    assert requirement.url is None


def test_canvas_gateway_loads_packaged_runtime_from_wheel(
    canvas_gateway_wheel: Path,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _WHEEL_PROBE, str(canvas_gateway_wheel)],
        cwd=tmp_path,
        env=_sanitized_build_env(tmp_path / "probe-state"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "canvas-gateway-wheel-ok"
