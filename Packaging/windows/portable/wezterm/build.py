"""Build ADR-255's modified WezTerm with private, checksum-pinned build tools."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from licenses import preserve_licenses

HERE = Path(__file__).resolve().parent
PACKAGES = ("wezterm-gui", "wezterm", "wezterm-mux-server")


def native_build_path(path: Path) -> Path:
    """Use an existing 8.3 spelling; never create aliases or change OS settings."""
    buffer = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    if not result or result >= len(buffer):
        raise ctypes.WinError()
    return Path(buffer.value)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(work: Path, item: dict[str, str]) -> Path:
    path = work / item["filename"]
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        urllib.request.urlretrieve(item["url"], path)
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return path


def relative_member(name: str) -> Path:
    windows = PureWindowsPath(name)
    posix = PurePosixPath(name)
    if windows.drive or windows.root or ".." in posix.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return Path(*posix.parts)


def extract_tar(archive: Path, destination: Path) -> None:
    """Drop the archive wrapper; materialize the source README link on Windows."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle:
            path = relative_member(member.name)
            if len(path.parts) < 2:
                continue
            target = destination.joinpath(*path.parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                # This is the only link in the verified upstream source tarball.
                if (
                    path.as_posix().endswith("/deps/harfbuzz/harfbuzz/README")
                    and member.linkname == "README.md"
                ):
                    continue
                raise ValueError(f"Unexpected archive link: {member.name}")
            if not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    readme = destination / "deps/harfbuzz/harfbuzz/README.md"
    if readme.exists():
        shutil.copyfile(readme, readme.with_name("README"))


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative_member(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unexpected archive link")
        bundle.extractall(destination)


def msvc_environment(vcvars: Path) -> dict[str, str]:
    """Capture vcvars only in this build process; never print the environment."""
    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        executable=os.environ.get("COMSPEC", "cmd.exe"),
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line and not line.startswith("=")
    )


def build(work: Path, vcvars: Path, jobs: int) -> None:
    if os.name != "nt" or not 1 <= jobs <= 4:
        raise ValueError("Build on Windows with one to four jobs")
    work.mkdir(parents=True, exist_ok=True)
    requested_work = work
    work = native_build_path(work)
    if len(str(work).encode("utf-8")) > 80:
        raise ValueError(
            "Native dependencies need a shorter --work path (80 bytes max)"
        )
    pins = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    archives = {name: download(work, item) for name, item in pins.items()}
    source = work / "src"
    if not source.exists():
        extract_tar(archives["source"], source)
    toolchain = work / "toolchain"
    for name in ("rustc", "cargo", "rust-std", "rustfmt"):
        component = work / name
        if not component.exists():
            extract_tar(archives[name], component)
        subdir = (component / "components").read_text(encoding="utf-8").strip()
        marker = toolchain / f".{name}-sha256"
        if not marker.exists() or marker.read_text() != pins[name]["sha256"]:
            shutil.copytree(component / subdir, toolchain, dirs_exist_ok=True)
            marker.write_text(pins[name]["sha256"])
    perl = work / "perl"
    if not perl.exists():
        extract_zip(archives["perl"], perl)
    env = msvc_environment(vcvars)
    for key in list(env):
        if key.upper().startswith(
            ("CARGO_", "RUST", "OPENSSL_", "X86_64_PC_WINDOWS_MSVC_OPENSSL_")
        ) or key.upper() in {
            "CC",
            "CXX",
            "AR",
            "LD",
            "CFLAGS",
            "CXXFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "CL",
            "_CL_",
            "PERL",
        }:
            del env[key]
    path_key = next(key for key in env if key.upper() == "PATH")
    env[path_key] = os.pathsep.join(
        [
            str(toolchain / "bin"),
            str(perl / "perl/bin"),
            str(Path(shutil.which("git")).parent),
            env[path_key],
        ]
    )
    env.update(
        {
            "CARGO_HOME": str(work / "cargo-home"),
            "CARGO_TARGET_DIR": str(source / "target"),
            "RUSTC": str(toolchain / "bin/rustc.exe"),
            "CARGO_BUILD_JOBS": str(jobs),
            "CARGO_INCREMENTAL": "0",
            "CARGO_TERM_COLOR": "never",
        }
    )
    temporary = work / "tmp"
    temporary.mkdir(exist_ok=True)
    env.update({key: str(temporary) for key in ("TEMP", "TMP", "TMPDIR")})
    env.update({"LANG": "C", "LC_ALL": "C"})
    patch = HERE / "portable-roots.patch"
    already_patched = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=source,
            env=env,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not already_patched:
        subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=source, env=env, check=True
        )
        subprocess.run(["git", "apply", str(patch)], cwd=source, env=env, check=True)
    cargo = str(toolchain / "bin/cargo.exe")
    command = [cargo, "build", "--locked", "--release", "-j", str(jobs)]
    for package in PACKAGES:
        command.extend(["-p", package])
    print("Building modified WezTerm; output in build.log", flush=True)
    with (work / "build.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    test_command = [
        sys.executable,
        "-B",
        str(HERE / "check_helpers.py"),
        "--source",
        str(source),
        "--rustc",
        str(toolchain / "bin/rustc.exe"),
        "--output",
        str(work / "helper-tests"),
    ]
    with (work / "test.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            test_command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    binaries = source / "target/release"
    versions = {}
    for executable, args in (
        (toolchain / "bin/rustc.exe", ["--version", "--verbose"]),
        (toolchain / "bin/cargo.exe", ["--version"]),
        (perl / "perl/bin/perl.exe", ["-e", "print $^V"]),
    ):
        versions[executable.name] = subprocess.check_output(
            [str(executable), *args], env=env, text=True
        ).strip()
    cmake = shutil.which("cmake", path=env[path_key])
    if cmake:
        versions["cmake.exe"] = subprocess.check_output(
            [cmake, "--version"], env=env, text=True
        ).strip()
    licenses = ["LICENSE.md", "NOTICE", "Cargo.lock", ".tag"]
    license_index = preserve_licenses(
        source, work / "cargo-home", work / "third-party-licenses"
    )
    provenance = {
        "status": "experimental-modified-wezterm",
        "requested_work": str(requested_work),
        "native_build_path": str(work),
        "built_utc": datetime.now(UTC).isoformat(),
        "inputs": pins,
        "toolchain": versions,
        "msvc": env.get("VCToolsVersion"),
        "windows_sdk": env.get("WindowsSDKVersion"),
        "nasm_available": shutil.which("nasm", path=env[path_key]) is not None,
        "patch_sha256": sha256(patch),
        "build_tooling_sha256": {
            name: sha256(HERE / name)
            for name in ("build.py", "check_helpers.py", "licenses.py", "inputs.json")
        },
        "cargo_lock_sha256": sha256(source / "Cargo.lock"),
        "third_party_license_index_sha256": sha256(license_index),
        "command": command,
        "test_command": test_command,
        "verification": {
            "build_exit_code": 0,
            "test_exit_code": 0,
            "build_log_sha256": sha256(work / "build.log"),
            "test_log_sha256": sha256(work / "test.log"),
        },
        "executables": {
            f"{name}.exe": sha256(binaries / f"{name}.exe") for name in PACKAGES
        },
        "licenses": {
            name: sha256(source / name) for name in licenses if (source / name).exists()
        },
    }
    (work / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built binaries: {binaries}\nProvenance: {work / 'provenance.json'}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("output/wezterm-build"))
    parser.add_argument(
        "--vcvars",
        type=Path,
        default=Path(
            "C:/Program Files/Microsoft Visual Studio/2022/Community/"
            "VC/Auxiliary/Build/vcvars64.bat"
        ),
    )
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    build(arguments.work.resolve(), arguments.vcvars.resolve(), arguments.jobs)
