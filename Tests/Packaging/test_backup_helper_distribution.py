"""Distribution and offline runtime contracts for the backup age helper."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tarfile
import venv
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    ("darwin", "arm64"),
    ("darwin", "amd64"),
    ("linux", "amd64"),
    ("linux", "arm64"),
    ("windows", "amd64"),
}


def _copy_build_source(destination: Path) -> None:
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


def _build_wheel(source: Path, output: Path, *, target: str | None) -> Path:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if target is None:
        environment.pop("TLDW_BACKUP_HELPER_TARGET", None)
    else:
        environment["TLDW_BACKUP_HELPER_TARGET"] = target
    completed = subprocess.run(
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
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _build_sdist(source: Path, output: Path) -> Path:
    completed = subprocess.run(
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(sdists) == 1
    return sdists[0]


@pytest.fixture(scope="module")
def built_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("backup-helper-wheels")
    source = root / "source"
    source.mkdir()
    _copy_build_source(source)
    pure = _build_wheel(source, root / "pure", target=None)
    goos, goarch = _native_tuple()
    native = _build_wheel(source, root / "native", target=f"{goos}/{goarch}")
    sdist = _build_sdist(source, root / "sdist")
    return pure, native, sdist


def _native_tuple() -> tuple[str, str]:
    return (
        {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}[platform.system()],
        {
            "aarch64": "arm64",
            "arm64": "arm64",
            "x86_64": "amd64",
            "AMD64": "amd64",
        }[platform.machine()],
    )


@pytest.fixture(scope="module")
def native_helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from Packaging.backup_age.build_helper import build_helper

    goos, goarch = _native_tuple()
    root = tmp_path_factory.mktemp("native-backup-helper")
    return build_helper(
        goos,
        goarch,
        root / ("backup-age.exe" if goos == "windows" else "backup-age"),
    )


def _write_delivery_manifest(root: Path, binary: Path, *, digest: str) -> None:
    goos, goarch = _native_tuple()
    (root / "helper_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "helpers": [
                    {
                        "age_version": "v1.3.2",
                        "arch": goarch,
                        "helper_version": "1",
                        "os": goos,
                        "protocol": 1,
                        "resource": (
                            "_age/backup-age.exe"
                            if goos == "windows"
                            else "_age/backup-age"
                        ),
                        "sha256": digest,
                        "status": "qualified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_checkout_without_helper_is_unavailable(monkeypatch, tmp_path):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: tmp_path)
    available, reason = crypto.helper_capability()
    assert available is False
    assert reason == "helper_unavailable"


def test_source_checkout_declares_every_target_unavailable() -> None:
    """An editable checkout cannot advertise unbuilt or merely cross-built helpers."""
    manifest = json.loads(
        (
            REPO_ROOT / "tldw_chatbook" / "Backup_Recovery" / "helper_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert {(entry["os"], entry["arch"]) for entry in manifest["helpers"]} == TARGETS
    assert {entry["status"] for entry in manifest["helpers"]} == {"unavailable"}
    assert {entry["reason"] for entry in manifest["helpers"]} == {"not_qualified"}


def test_native_wheel_is_platform_tagged_and_owns_one_helper(
    built_artifacts: tuple[Path, Path, Path],
) -> None:
    pure, native, _ = built_artifacts
    goos, _ = _native_tuple()
    assert pure.name.endswith("-py3-none-any.whl")
    assert not native.name.endswith("-py3-none-any.whl")

    binary_name = "backup-age.exe" if goos == "windows" else "backup-age"
    resource = f"tldw_chatbook/Backup_Recovery/_age/{binary_name}"
    with zipfile.ZipFile(pure) as archive:
        assert resource not in archive.namelist()
    with zipfile.ZipFile(native) as archive:
        names = archive.namelist()
        assert names.count(resource) == 1
        manifest = json.loads(
            archive.read("tldw_chatbook/Backup_Recovery/helper_manifest.json")
        )
        qualified = [
            entry for entry in manifest["helpers"] if entry["status"] == "qualified"
        ]
        assert len(qualified) == 1
        assert (
            qualified[0]["sha256"] == hashlib.sha256(archive.read(resource)).hexdigest()
        )
        resource_root = "tldw_chatbook/Backup_Recovery/_age"
        assert f"{resource_root}/LICENSE.age.txt" in names
        assert f"{resource_root}/THIRD_PARTY_NOTICES.txt" in names


def test_pure_wheel_excludes_a_stale_editable_helper(
    native_helper: Path, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _copy_build_source(source)
    resource_root = source / "tldw_chatbook/Backup_Recovery/_age"
    resource_root.mkdir()
    stale = resource_root / native_helper.name
    stale.write_bytes(native_helper.read_bytes())
    stale.chmod(0o755)

    wheel = _build_wheel(source, tmp_path / "dist", target=None)

    with zipfile.ZipFile(wheel) as archive:
        assert not any(
            name.startswith("tldw_chatbook/Backup_Recovery/_age/backup-age")
            for name in archive.namelist()
        )


def test_sdist_contains_the_complete_pinned_helper_build_inputs(
    built_artifacts: tuple[Path, Path, Path],
) -> None:
    _, _, sdist = built_artifacts
    with tarfile.open(sdist, "r:gz") as archive:
        names = {
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if member.isfile() and "/" in member.name
        }
    assert {
        "Packaging/backup_age/LICENSE.age.txt",
        "Packaging/backup_age/README.md",
        "Packaging/backup_age/THIRD_PARTY_NOTICES.txt",
        "Packaging/backup_age/build_helper.py",
        "Packaging/backup_age/go.mod",
        "Packaging/backup_age/go.sum",
        "Packaging/backup_age/main.go",
        "Packaging/backup_age/main_test.go",
        "Packaging/backup_age/qualification.json",
        "Packaging/backup_age/wheel_commands.py",
        "tldw_chatbook/Backup_Recovery/helper_manifest.json",
    } <= names
    assert not any(name.endswith(("/backup-age", "/backup-age.exe")) for name in names)


def test_distribution_checker_accepts_pure_and_native_helper_contracts(
    built_artifacts: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    from Packaging.check_manifest import check_distribution

    pure, native, sdist = built_artifacts
    for wheel in (pure, native):
        dist = tmp_path / wheel.stem
        dist.mkdir()
        shutil.copy2(sdist, dist / sdist.name)
        shutil.copy2(wheel, dist / wheel.name)
        assert check_distribution(dist)


def _helper_transform(binary: Path, mode: str, payload: bytes) -> bytes:
    password = b"packaged-upgrade-fixture"
    completed = subprocess.run(
        [str(binary), mode],
        input=struct.pack(">I", len(password)) + password + payload,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.stderr == b""
    return completed.stdout


def test_installed_native_wheel_runs_offline_without_go_and_interoperates(
    built_artifacts: tuple[Path, Path, Path],
    native_helper: Path,
    tmp_path: Path,
) -> None:
    _, native_wheel, _ = built_artifacts
    environment_root = tmp_path / "environment"
    venv.EnvBuilder(symlinks=True, with_pip=False).create(environment_root)
    scripts = environment_root / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    uv = Path(sys.executable).with_name("uv.exe" if os.name == "nt" else "uv")
    subprocess.run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--offline",
            "pydantic>=2.4,<3",
            "loguru",
            "psutil",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--offline",
            "--no-index",
            "--no-deps",
            str(native_wheel),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    state = tmp_path / "state"
    for name in ("home", "config", "data", "tmp"):
        (state / name).mkdir(parents=True)
    environment = {
        "HOME": str(state / "home"),
        "PATH": str(scripts),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(state / "tmp"),
        "XDG_CONFIG_HOME": str(state / "config"),
        "XDG_DATA_HOME": str(state / "data"),
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    assert platform.system() == "Darwin"
    sandbox = Path("/usr/bin/sandbox-exec")
    assert sandbox.is_file()
    probe = subprocess.run(
        [
            str(sandbox),
            "-p",
            "(version 1)(allow default)(deny network*)",
            str(python),
            "-c",
            """
import errno
import json
from pathlib import Path
import shutil
import socket
from threading import Event

network_probe = socket.socket()
network_probe.settimeout(1)
assert network_probe.connect_ex(("1.1.1.1", 53)) in {errno.EACCES, errno.EPERM}
network_probe.close()

class OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("network access attempted")

socket.socket = OfflineSocket
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("network access attempted")
)

from tldw_chatbook.Backup_Recovery import crypto

assert Path(crypto.__file__).resolve().is_relative_to(Path(__import__('sys').prefix))
assert shutil.which("go") is None
assert crypto.helper_capability() == (True, "available")
source = Path("plain.bin")
encrypted = Path("encrypted.age")
decrypted = Path("decrypted.bin")
source.write_bytes(b"installed wheel offline round trip" * 4096)
crypto.transform(source, encrypted, password=b"wheel-passphrase", decrypt=False, cancel=Event())
crypto.transform(encrypted, decrypted, password=b"wheel-passphrase", decrypt=True, cancel=Event())
assert decrypted.read_bytes() == source.read_bytes()

root = crypto._package_resource_root()
binary = root / ("backup-age.exe" if __import__('os').name == "nt" else "backup-age")
manifest_path = root.parent / "helper_manifest.json"
manifest_bytes = manifest_path.read_bytes()
binary_bytes = binary.read_bytes()
binary.write_bytes(binary_bytes + b"tampered")
assert crypto.helper_capability() == (False, "helper_integrity_mismatch")
binary.write_bytes(binary_bytes)
binary.chmod(0o755)
manifest = json.loads(manifest_bytes)
manifest["schema_version"] = 2
manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
assert crypto.helper_capability() == (False, "helper_unavailable")
manifest_path.write_bytes(manifest_bytes)
binary.unlink()
assert crypto.helper_capability() == (False, "helper_unavailable")
print(json.dumps({"encrypted": str(encrypted), "manifest": str(manifest_path)}))
""",
        ],
        cwd=state,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stderr == ""

    goos, _ = _native_tuple()
    binary_name = "backup-age.exe" if goos == "windows" else "backup-age"
    installed_manifest = Path(json.loads(probe.stdout)["manifest"])
    installed_helper = installed_manifest.parent / "_age" / binary_name
    # Restore after the destructive missing-resource probe before interoperability.
    installed_helper.write_bytes(native_helper.read_bytes())
    installed_helper.chmod(0o755)
    packaged_ciphertext = _helper_transform(installed_helper, "encrypt", b"packaged")
    assert (
        _helper_transform(native_helper, "decrypt", packaged_ciphertext) == b"packaged"
    )
    source_ciphertext = _helper_transform(native_helper, "encrypt", b"source-qualified")
    assert (
        _helper_transform(installed_helper, "decrypt", source_ciphertext)
        == b"source-qualified"
    )


def test_build_helper_rejects_an_unowned_target(tmp_path: Path) -> None:
    """A typo or new tuple cannot silently widen the release matrix."""
    from Packaging.backup_age.build_helper import build_helper

    destination = tmp_path / "helper"
    with pytest.raises(ValueError, match="unsupported backup helper target"):
        build_helper("freebsd", "amd64", destination)
    assert not destination.exists()


def test_build_helper_produces_the_pinned_native_protocol(tmp_path: Path) -> None:
    """Dropping the real build, target flags, or pinned protocol breaks this test."""
    from Packaging.backup_age.build_helper import build_helper

    goos, goarch = _native_tuple()
    executable = build_helper(
        goos,
        goarch,
        tmp_path / ("backup-age.exe" if goos == "windows" else "backup-age"),
    )

    completed = subprocess.run(
        [str(executable), "info"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert json.loads(completed.stdout) == {
        "age_version": "v1.3.2",
        "arch": goarch,
        "helper_version": "1",
        "os": goos,
        "protocol": 1,
    }
    assert completed.stderr == b""


def test_helper_capability_reports_a_packaged_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_helper: Path,
) -> None:
    """Replacing the installed resource cannot collapse into generic absence."""
    from tldw_chatbook.Backup_Recovery import crypto

    resource_root = tmp_path / "_age"
    resource_root.mkdir()
    installed = resource_root / native_helper.name
    installed.write_bytes(native_helper.read_bytes())
    installed.chmod(0o755)
    _write_delivery_manifest(tmp_path, installed, digest="0" * 64)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: resource_root)

    assert crypto.helper_capability() == (False, "helper_integrity_mismatch")


def test_helper_capability_accepts_the_installed_resource_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_helper: Path,
) -> None:
    from tldw_chatbook.Backup_Recovery import crypto

    resource_root = tmp_path / "_age"
    resource_root.mkdir()
    installed = resource_root / native_helper.name
    installed.write_bytes(native_helper.read_bytes())
    installed.chmod(0o755)
    _write_delivery_manifest(
        tmp_path,
        installed,
        digest=hashlib.sha256(installed.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: resource_root)

    assert crypto.helper_capability() == (True, "available")


@pytest.mark.parametrize("schema_version", [2, True], ids=["unsupported", "boolean"])
def test_helper_capability_rejects_a_malformed_delivery_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_helper: Path,
    schema_version: object,
) -> None:
    from tldw_chatbook.Backup_Recovery import crypto

    resource_root = tmp_path / "_age"
    resource_root.mkdir()
    installed = resource_root / native_helper.name
    installed.write_bytes(native_helper.read_bytes())
    installed.chmod(0o755)
    _write_delivery_manifest(
        tmp_path,
        installed,
        digest=hashlib.sha256(installed.read_bytes()).hexdigest(),
    )
    manifest_path = tmp_path / "helper_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: resource_root)

    assert crypto.helper_capability() == (False, "helper_unavailable")
