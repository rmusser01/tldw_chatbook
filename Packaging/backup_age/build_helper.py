"""Build entry point for the package-owned backup age helper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

TARGETS = frozenset(
    {
        ("darwin", "arm64"),
        ("darwin", "amd64"),
        ("linux", "amd64"),
        ("linux", "arm64"),
        ("windows", "amd64"),
    }
)
_MODULE_GRAPH = frozenset(
    {
        "tldw.chatbook/backup-age",
        "c2sp.org/CCTV/age v0.0.0-20260829155415-4448f2097b2d",
        "filippo.io/age v1.3.2",
        "filippo.io/edwards25519 v1.2.0",
        "filippo.io/hpke v0.4.0",
        "filippo.io/nistec v0.0.4",
        "github.com/rogpeppe/go-internal v1.16.0",
        "golang.org/x/crypto v0.55.0",
        "golang.org/x/net v0.57.0",
        "golang.org/x/sys v0.47.0",
        "golang.org/x/term v0.45.0",
        "golang.org/x/text v0.41.0",
        "golang.org/x/tools v0.49.0",
    }
)


def _go_environment(root: Path) -> dict[str, str]:
    """Return a private, offline Go build environment."""
    environment = {
        key: value
        for key in (
            "GOCACHE",
            "GOMODCACHE",
            "GOPATH",
            "PATH",
            "SYSTEMROOT",
            "TMPDIR",
        )
        if (value := os.environ.get(key))
    }
    environment.update(
        {
            "CGO_ENABLED": "0",
            "GOENV": "off",
            "GOPROXY": "off",
            "GOSUMDB": "off",
            "GOTOOLCHAIN": "local",
            "HOME": str(root / "home"),
        }
    )
    for name in ("GOCACHE", "GOMODCACHE", "GOPATH"):
        environment.setdefault(name, str(root / name.lower()))
    for name in ("HOME", "GOCACHE", "GOMODCACHE", "GOPATH"):
        Path(environment[name]).mkdir(parents=True, exist_ok=True)
    return environment


def _run(go: str, arguments: list[str], *, source: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        [go, *arguments],
        cwd=source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout


def build_helper(goos: str, goarch: str, destination: Path) -> Path:
    """Build one release-owned helper target into *destination*.

    The module graph is checked exactly and all Go dependency access is offline.
    Contributors must provision the pinned module cache before invoking this command.
    """
    if (goos, goarch) not in TARGETS:
        raise ValueError(f"unsupported backup helper target: {goos}/{goarch}")
    source = Path(__file__).resolve().parent
    go = shutil.which("go")
    if go is None:
        raise RuntimeError("Go is required to build the backup helper")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="backup-age-build-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        environment = _go_environment(temporary)
        environment.update({"GOOS": goos, "GOARCH": goarch})
        modules = frozenset(
            line.strip()
            for line in _run(
                go,
                ["list", "-mod=readonly", "-m", "all"],
                source=source,
                env=environment,
            ).splitlines()
            if line.strip()
        )
        if modules != _MODULE_GRAPH:
            raise RuntimeError("backup helper module graph differs from release pin")
        _run(go, ["mod", "verify"], source=source, env=environment)
        output = temporary / ("backup-age.exe" if goos == "windows" else "backup-age")
        _run(
            go,
            [
                "build",
                "-mod=readonly",
                "-trimpath",
                "-buildvcs=false",
                "-ldflags=-s -w -buildid=",
                "-o",
                str(output),
                ".",
            ],
            source=source,
            env=environment,
        )
        if not output.is_file():
            raise RuntimeError("Go did not produce the backup helper")
        output.chmod(0o755)
        os.replace(output, destination)
    return destination
