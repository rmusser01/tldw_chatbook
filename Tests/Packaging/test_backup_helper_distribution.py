"""Distribution and offline runtime contracts for the Python age worker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path
from threading import Event

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = "tldw_chatbook/Backup_Recovery/age_worker.py"
FIXED_CIPHERTEXT = REPO_ROOT / "Tests/Backup_Recovery/fixtures/age_v1/official.age"
FIXED_PASSWORD = b"interop-synthetic-password"
FIXED_PLAINTEXT = bytes(range(256)) * 2048
RETIRED_MEMBERS = (
    "Packaging/backup_age/main.go",
    "tldw_chatbook/Backup_Recovery/helper_manifest.json",
    "tldw_chatbook/Backup_Recovery/_age/backup-age",
)


def _copy_build_source(destination: Path) -> None:
    """Copy only the inputs used for an isolated package build."""

    ignored = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", ".DS_Store", "build", "dist", "*.egg-info"
    )
    for name in ("tldw_chatbook", "Packaging"):
        shutil.copytree(REPO_ROOT / name, destination / name, ignore=ignored)
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "CHANGELOG.md",
        "requirements.txt",
    ):
        shutil.copy2(REPO_ROOT / name, destination / name)


def _build_environment() -> dict[str, str]:
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def _build_wheel(source: Path, output: Path) -> Path:
    # The current interpreter and complete argument list are fixed by the test.
    completed = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        env=_build_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _build_sdist(source: Path, output: Path) -> Path:
    # The current interpreter and complete argument list are fixed by the test.
    completed = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        env=_build_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(sdists) == 1
    return sdists[0]


@pytest.fixture(scope="module")
def built_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("python-backup-worker-artifacts")
    source = root / "source"
    source.mkdir()
    _copy_build_source(source)
    wheel = _build_wheel(source, root / "wheel")
    sdist = _build_sdist(source, root / "sdist")
    return wheel, sdist


def _uv() -> Path:
    name = "uv.exe" if os.name == "nt" else "uv"
    candidates = (
        Path(sysconfig.get_path("scripts")) / name,
        Path(sys.executable).with_name(name),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(
        "uv must be installed in the current interpreter scripts directory; "
        f"checked: {[str(candidate) for candidate in candidates]}"
    )


def test_uv_resolves_current_interpreter_scripts_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows setup-python installs uv under Scripts beside the interpreter root."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    uv = scripts / ("uv.exe" if os.name == "nt" else "uv")
    uv.touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(scripts))

    assert _uv() == uv


def _uv_environment(state: Path) -> dict[str, str]:
    for name in ("home", "config", "data", "cache", "state", "tmp"):
        (state / name).mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "HOME": str(state / "home"),
        "USERPROFILE": str(state / "home"),
        "XDG_CACHE_HOME": str(state / "cache"),
        "XDG_CONFIG_HOME": str(state / "config"),
        "XDG_DATA_HOME": str(state / "data"),
        "XDG_STATE_HOME": str(state / "state"),
        "TMPDIR": str(state / "tmp"),
        "TEMP": str(state / "tmp"),
        "TMP": str(state / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if "UV_CACHE_DIR" in os.environ:
        environment["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
    return environment


def _isolated_python(root: Path) -> tuple[Path, Path]:
    venv.EnvBuilder(symlinks=True, with_pip=False).create(root)
    scripts = root / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    # uv is resolved beside the selected test interpreter, never through PATH.
    completed = subprocess.run(  # nosec B603
        [
            str(_uv()),
            "pip",
            "install",
            "--python",
            str(python),
            "--offline",
            "setuptools>=77.0",
            "wheel",
            "pydantic>=2.4,<3",
            "loguru",
            "psutil",
            "pycryptodomex",
        ],
        env=_uv_environment(root.parent / "installer-state"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return python, scripts


def _isolated_runtime_environment(scripts: Path, state: Path) -> dict[str, str]:
    for name in ("home", "config", "data", "cache", "state", "tmp"):
        (state / name).mkdir(parents=True, exist_ok=True)
    environment = {
        "HOME": str(state / "home"),
        "PATH": str(scripts),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(state / "tmp"),
        "XDG_CACHE_HOME": str(state / "cache"),
        "XDG_CONFIG_HOME": str(state / "config"),
        "XDG_DATA_HOME": str(state / "data"),
        "XDG_STATE_HOME": str(state / "state"),
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _install_artifact(
    python: Path, artifact: Path, state: Path, *, editable: bool = False
) -> None:
    command = [
        str(_uv()),
        "pip",
        "install",
        "--python",
        str(python),
        "--offline",
        "--no-build-isolation",
        "--no-deps",
    ]
    if editable:
        command.append("--editable")
    command.append(str(artifact))
    # uv is resolved beside the selected test interpreter, never through PATH.
    completed = subprocess.run(  # nosec B603
        command,
        env=_uv_environment(state),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _runtime_probe(
    python: Path,
    scripts: Path,
    state: Path,
    *,
    expected_origin: Path,
) -> None:
    state.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXED_CIPHERTEXT, state / "fixed.age")
    script = r"""
import json
from pathlib import Path
import shutil
import socket
import sys
from threading import Event

class OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("network access attempted")

socket.socket = OfflineSocket
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("network access attempted")
)

from tldw_chatbook.Backup_Recovery import crypto

origin = Path(sys.argv[1]).resolve()
assert Path(crypto.__file__).resolve().is_relative_to(origin)
worker = Path(crypto.__file__).with_name("age_worker.py")
assert worker.is_file()
assert Path(crypto._package_resource_root()).resolve() == worker.parent.resolve()
assert shutil.which("go") is None
assert crypto.helper_capability() == (True, "available")

plain = Path("plain.bin")
encrypted = Path("encrypted.age")
restored = Path("restored.bin")
plain.write_bytes(b"packaged Python worker round trip" * 4096)
crypto.transform(plain, encrypted, password=b"packaging-round-trip", decrypt=False, cancel=Event())
crypto.transform(encrypted, restored, password=b"packaging-round-trip", decrypt=True, cancel=Event())
assert restored.read_bytes() == plain.read_bytes()

fixed = Path("fixed-restored.bin")
crypto.transform(Path("fixed.age"), fixed, password=b"interop-synthetic-password", decrypt=True, cancel=Event())
assert fixed.read_bytes() == bytes(range(256)) * 2048
print(json.dumps({"crypto": str(Path(crypto.__file__).resolve()), "worker": str(worker.resolve())}))
"""
    # The isolated interpreter is an absolute path owned by this test environment.
    completed = subprocess.run(  # nosec B603
        [str(python), "-c", script, str(expected_origin)],
        cwd=state,
        env=_isolated_runtime_environment(scripts, state),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""


def _sdist_members(sdist: Path) -> dict[str, bytes]:
    with tarfile.open(sdist, "r:gz") as archive:
        return {
            member.name.split("/", 1)[1]: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile() and "/" in member.name
        }


def test_worker_is_in_wheel_and_sdist(built_artifacts: tuple[Path, Path]) -> None:
    """Removing or transforming the worker bytes breaks package integrity."""

    wheel, sdist = built_artifacts
    source_worker = (REPO_ROOT / WORKER_PATH).read_bytes()
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert archive.read(WORKER_PATH) == source_worker
        wheel_metadata_name = next(
            name for name in names if name.endswith(".dist-info/WHEEL")
        )
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_metadata_name))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    assert wheel_metadata["Root-Is-Purelib"] == "true"
    assert any(
        requirement.lower().startswith("pycryptodomex")
        for requirement in metadata.get_all("Requires-Dist", [])
    )

    sdist_members = _sdist_members(sdist)
    assert sdist_members[WORKER_PATH] == source_worker
    for retired in RETIRED_MEMBERS:
        assert retired not in names
        assert retired not in sdist_members
    assert not any(name.startswith("Packaging/backup_age/") for name in names)
    assert not any(name.startswith("Packaging/backup_age/") for name in sdist_members)


def test_source_checkout_runs_python_worker_and_decrypts_fixed_fixture(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Backup_Recovery import crypto

    assert crypto.helper_capability() == (True, "available")
    source = tmp_path / "plain.bin"
    encrypted = tmp_path / "encrypted.age"
    restored = tmp_path / "restored.bin"
    source.write_bytes(b"source Python worker round trip" * 4096)
    crypto.transform(
        source, encrypted, password=b"source-round-trip", decrypt=False, cancel=Event()
    )
    crypto.transform(
        encrypted, restored, password=b"source-round-trip", decrypt=True, cancel=Event()
    )
    assert restored.read_bytes() == source.read_bytes()
    fixed = tmp_path / "fixed.bin"
    crypto.transform(
        FIXED_CIPHERTEXT, fixed, password=FIXED_PASSWORD, decrypt=True, cancel=Event()
    )
    assert fixed.read_bytes() == FIXED_PLAINTEXT


def test_editable_install_runs_offline_without_go_and_decrypts_fixed_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _copy_build_source(source)
    python, scripts = _isolated_python(tmp_path / "environment")
    _install_artifact(python, source, tmp_path / "installer-state", editable=True)
    _runtime_probe(python, scripts, tmp_path / "runtime", expected_origin=source)


def test_sdist_install_runs_offline_without_go_and_decrypts_fixed_fixture(
    built_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, sdist = built_artifacts
    python, scripts = _isolated_python(tmp_path / "environment")
    _install_artifact(python, sdist, tmp_path / "installer-state")
    _runtime_probe(
        python, scripts, tmp_path / "runtime", expected_origin=python.parent.parent
    )


def test_installed_wheel_runs_offline_without_go_and_decrypts_fixed_fixture(
    built_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    wheel, _ = built_artifacts
    python, scripts = _isolated_python(tmp_path / "environment")
    _install_artifact(python, wheel, tmp_path / "installer-state")
    _runtime_probe(
        python, scripts, tmp_path / "runtime", expected_origin=python.parent.parent
    )


def test_distribution_checker_accepts_python_worker_contract(
    built_artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    from Packaging.check_manifest import check_distribution

    wheel, sdist = built_artifacts
    dist = tmp_path / "dist"
    dist.mkdir()
    shutil.copy2(wheel, dist / wheel.name)
    shutil.copy2(sdist, dist / sdist.name)
    assert check_distribution(dist)


@pytest.mark.parametrize("retired_member", RETIRED_MEMBERS)
def test_distribution_checker_rejects_retired_go_members(
    built_artifacts: tuple[Path, Path], tmp_path: Path, retired_member: str
) -> None:
    from Packaging.check_manifest import check_distribution

    wheel, sdist = built_artifacts
    dist = tmp_path / retired_member.replace("/", "-")
    dist.mkdir()
    mutated_wheel = dist / wheel.name
    shutil.copy2(wheel, mutated_wheel)
    shutil.copy2(sdist, dist / sdist.name)
    with zipfile.ZipFile(mutated_wheel, "a") as archive:
        archive.writestr(retired_member, b"retired")
    assert check_distribution(dist) is False


def test_distribution_contract_requires_python_worker() -> None:
    """Dropping the worker from either application artifact breaks delivery."""
    from Packaging import check_manifest

    assert WORKER_PATH in check_manifest.REQUIRED_SDIST_PATHS
    assert WORKER_PATH in check_manifest.REQUIRED_WHEEL_PATHS
