"""Installed-artifact qualification for persistent Terminal support."""

from __future__ import annotations

from email.parser import Parser
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

from Tests.Packaging.test_installed_distribution import (
    _copy_build_inputs,
    _sanitized_build_env,
)


pytestmark = pytest.mark.integration

TERMINAL_RUNTIME_PATHS = frozenset(
    {
        "tldw_chatbook/Terminal/__init__.py",
        "tldw_chatbook/Terminal/backend.py",
        "tldw_chatbook/Terminal/contracts.py",
        "tldw_chatbook/Terminal/io_actors.py",
        "tldw_chatbook/Terminal/launch.py",
        "tldw_chatbook/Terminal/posix_backend.py",
        "tldw_chatbook/Terminal/posix_launcher.py",
        "tldw_chatbook/Terminal/protocol_gate.py",
        "tldw_chatbook/Terminal/screen_model.py",
        "tldw_chatbook/Terminal/session_manager.py",
        "tldw_chatbook/UI/Console_Modules/terminal.py",
        "tldw_chatbook/Widgets/Console/console_terminal_session_modal.py",
        "tldw_chatbook/Widgets/Console/console_terminal_workspace.py",
        "tldw_chatbook/css/components/_agentic_terminal.tcss",
    }
)


@pytest.fixture(scope="module")
def terminal_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source_root = tmp_path_factory.mktemp("terminal-distribution-source")
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
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_terminal_runtime_and_dependency_contract_ship_in_wheel(
    terminal_wheel: Path,
) -> None:
    with zipfile.ZipFile(terminal_wheel) as archive:
        members = set(archive.namelist())
        metadata_names = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))

    assert TERMINAL_RUNTIME_PATHS <= members
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist") or []
    ]
    pyte_requirements = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "pyte"
    ]
    assert len(pyte_requirements) == 1
    pyte_requirement = pyte_requirements[0]
    assert str(pyte_requirement.specifier) == "==0.8.2"
    assert pyte_requirement.marker is None
    assert not pyte_requirement.extras
    assert pyte_requirement.url is None
    assert all(
        canonicalize_name(requirement.name) != "pywinpty"
        for requirement in requirements
    )


def test_windows_terminal_backend_availability_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import app as app_module

    monkeypatch.setattr(app_module, "os", SimpleNamespace(name="nt"))

    with pytest.raises(OSError, match="^persistent Terminal backend unavailable$"):
        app_module._build_terminal_backend()
