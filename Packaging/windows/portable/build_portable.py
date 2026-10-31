"""Assemble the experimental Windows x64 payload; requires build-host pip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

SOURCE = Path(__file__).absolute().parent


def sha256(path: Path) -> str:
    """Hash a build input without leaving its handle open."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_archive(cache: Path, name: str, item: dict[str, str]) -> Path:
    """Download official pinned bytes, or validate an existing offline cache."""
    path = cache / f"{name}.zip"
    if not path.exists():
        print(f"Downloading {name} {item['version']}", flush=True)
        urllib.request.urlretrieve(item["url"], path)
    if sha256(path) != item["sha256"]:
        raise ValueError(
            f"{name} archive checksum mismatch; discard the cache and retry"
        )
    return path


def extract_archive(archive: Path, destination: Path) -> None:
    """Reject absolute, traversing and symlink members before extraction."""
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            windows = PureWindowsPath(member.filename)
            normalized = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                windows.drive
                or windows.root
                or normalized.is_absolute()
                or ".." in normalized.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member")
        bundle.extractall(destination)


def write_pristine_zip(destination: Path, archive_path: Path) -> None:
    """Exclude portable state even if a diagnostic has already been run."""
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in destination.rglob("*"):
            relative = path.relative_to(destination)
            if relative.parts[0].casefold() == "data":
                continue
            if path.is_file():
                archive.write(path, Path(destination.name) / relative)


def verified_wezterm_build(work: Path) -> dict:
    """Require matching patched binaries, pinned source and license provenance."""
    provenance = json.loads((work / "provenance.json").read_text(encoding="utf-8"))
    pins = json.loads((SOURCE / "wezterm" / "inputs.json").read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "experimental-modified-wezterm"
        or provenance.get("inputs", {}).get("source") != pins["source"]
        or provenance.get("patch_sha256")
        != sha256(SOURCE / "wezterm" / "portable-roots.patch")
    ):
        raise ValueError("WezTerm source or portable patch provenance mismatch")
    required = {"wezterm.exe", "wezterm-gui.exe", "wezterm-mux-server.exe"}
    if set(provenance.get("executables", {})) != required:
        raise ValueError("A matching GUI, CLI and mux-server build is required")
    for name in sorted(required):
        if (
            sha256(work / "src" / "target" / "release" / name)
            != provenance["executables"][name]
        ):
            raise ValueError(f"Modified WezTerm executable checksum mismatch: {name}")
    if sha256(work / "src" / "Cargo.lock") != provenance.get("cargo_lock_sha256"):
        raise ValueError("WezTerm Cargo.lock provenance mismatch")
    if sha256(work / "third-party-licenses" / "index.json") != provenance.get(
        "third_party_license_index_sha256"
    ):
        raise ValueError("WezTerm third-party license index provenance mismatch")
    notices = work / "third-party-licenses"
    index = json.loads((notices / "index.json").read_text(encoding="utf-8"))
    if set(index) != {"packages", "notices"} or not index["notices"]:
        raise ValueError("WezTerm license notice index is incomplete")
    for row in index["notices"]:
        name = row["file"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).is_absolute()
            or name in {".", ".."}
            or sha256(notices / name) != row["sha256"]
        ):
            raise ValueError("WezTerm license notice checksum mismatch")
    for name in ("LICENSE.md", "Cargo.lock", ".tag"):
        if sha256(work / "src" / name) != provenance.get("licenses", {}).get(name):
            raise ValueError("WezTerm upstream license/source metadata mismatch")
    return provenance


def install_wezterm_build(work: Path, terminal: Path) -> dict:
    """Use system ConPTY, install matching EXEs and preserve upstream notices."""
    provenance = verified_wezterm_build(work)
    # The pinned optional ConPTY host fast-fails from long executable paths.
    # Its absence selects WezTerm's existing kernel32 ConPTY implementation.
    for name in ("conpty.dll", "OpenConsole.exe"):
        (terminal / name).unlink(missing_ok=True)
    for name in provenance["executables"]:
        shutil.copy2(work / "src" / "target" / "release" / name, terminal / name)
    (terminal / "wezterm.pdb").unlink(missing_ok=True)
    shutil.copytree(work / "third-party-licenses", terminal / "third-party-licenses")
    shutil.copy2(work / "src" / "LICENSE.md", terminal / "LICENSE-modified-WezTerm.md")
    shutil.copy2(work / "provenance.json", terminal / "chatbook-build-provenance.json")
    shutil.copy2(
        SOURCE / "wezterm" / "portable-roots.patch",
        terminal / "CHATBOOK-PORTABLE.patch",
    )
    shutil.copy2(
        SOURCE / "wezterm" / "inputs.json", terminal / "CHATBOOK-WEZTERM-INPUTS.json"
    )
    (terminal / "CHATBOOK-MODIFICATIONS.txt").write_text(
        "Experimental modified WezTerm build for Chatbook.\n"
        "Windows-only process-local HOME, data and runtime path overrides.\n"
        "Uses Windows system ConPTY; optional conpty.dll/OpenConsole.exe omitted.\n"
        f"Source release: {provenance['inputs']['source']['version']}\n"
        f"Portable patch SHA-256: {provenance['patch_sha256']}\n",
        encoding="utf-8",
    )
    return provenance


def build(wheel: Path, destination: Path, cache: Path, wezterm_build: Path) -> Path:
    """Build a new payload and pristine ZIP without ever packaging user data."""
    if sys.platform != "win32":
        raise ValueError("Run this proof-of-concept builder on Windows x64")
    if sys.version_info[:2] != (3, 12) or sysconfig.get_platform() != "win-amd64":
        raise ValueError("Build with Python 3.12 x64 to match the embedded runtime ABI")
    archive_path = destination.with_suffix(".zip")
    if destination.exists() or archive_path.exists():
        raise ValueError("Output already exists; choose a new output directory")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("Provide a built Chatbook wheel")
    # Verify before creating output or downloading anything; never fall back to
    # the stock executable, which cannot honor the portable runtime selectors.
    verified_wezterm_build(wezterm_build)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((SOURCE / "inputs.json").read_text(encoding="utf-8"))
    archives = {
        name: verified_archive(cache, name, item) for name, item in inputs.items()
    }
    destination.mkdir(parents=True)
    runtime = destination / "runtime"
    extract_archive(archives["python"], runtime)
    extract_archive(archives["wezterm"], destination / "terminal")
    terminal = destination / "terminal"
    candidates = list(terminal.rglob("wezterm-gui.exe"))
    if len(candidates) != 1:
        raise ValueError("WezTerm archive does not contain one GUI executable")
    if candidates[0].parent != terminal:
        wrapper = candidates[0].parent
        for item in wrapper.iterdir():
            shutil.move(str(item), terminal / item.name)
        wrapper.rmdir()
    native_provenance = install_wezterm_build(wezterm_build, terminal)
    pth_files = list(runtime.glob("python*._pth"))
    if len(pth_files) != 1:
        raise ValueError("Unexpected embedded Python path configuration")
    pth_files[0].write_text(
        "python312.zip\n.\nLib/site-packages\n..\nimport site\n", encoding="utf-8"
    )
    site_packages = runtime / "Lib" / "site-packages"
    print("Installing Chatbook and its declared runtime dependencies", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--no-compile",
            "--target",
            str(site_packages),
            "--report",
            str(destination / "dependency-report.json"),
            str(wheel),
        ],
        check=True,
    )
    for name in (
        "portable.py",
        "app_entry.py",
        "Chatbook.cmd",
        "wezterm.lua",
        "README.md",
    ):
        shutil.copy2(SOURCE / name, destination / name)
    manifest = {
        "status": "experimental-spike",
        "architecture": "windows-x64",
        "inputs": inputs,
        "chatbook_wheel": {"name": wheel.name, "sha256": sha256(wheel)},
        "wezterm_build": native_provenance,
    }
    (destination / "portable-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_pristine_zip(destination, archive_path)
    print(f"Portable ZIP: {archive_path}", flush=True)
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("output/portable-cache"))
    parser.add_argument("--wezterm-build", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.wheel.absolute(),
        args.output.absolute(),
        args.cache.absolute(),
        args.wezterm_build.absolute(),
    )
