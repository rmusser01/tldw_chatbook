#!/usr/bin/env python3
"""Check whether a package version already exists on PyPI."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from tldw_chatbook.Utils.path_validation import validate_path


PYPI_JSON_BASE_URL = "https://pypi.org/pypi"
DEFAULT_PACKAGE_NAME = "tldw-chatbook"
DEFAULT_OUTPUT_NAME = "release_exists"


class HttpResponse(Protocol):
    def __enter__(self) -> "HttpResponse": ...

    def __exit__(self, *_args: object) -> None: ...

    def read(self, size: int) -> bytes: ...


class UrlOpen(Protocol):
    def __call__(self, url: str, timeout: int) -> HttpResponse: ...


def release_exists(
    package_name: str,
    version: str,
    *,
    base_url: str = PYPI_JSON_BASE_URL,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    """Return whether the package version already exists in the PyPI JSON API."""
    if not package_name:
        raise ValueError("package name cannot be empty")
    if not version:
        raise ValueError("version cannot be empty")

    package = urllib.parse.quote(package_name, safe="")
    quoted_version = urllib.parse.quote(version, safe="")
    url = f"{base_url.rstrip('/')}/{package}/{quoted_version}/json"

    try:
        with urlopen(url, timeout=30) as response:
            response.read(1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise

    return True


def resolve_github_output_path(
    output_path: str | os.PathLike[str] | None = None,
    *,
    runner_temp: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return a validated GitHub output file path, or ``None`` outside Actions."""
    raw_output_path = output_path or os.environ.get("GITHUB_OUTPUT")
    if not raw_output_path:
        return None

    raw_runner_temp = runner_temp or os.environ.get("RUNNER_TEMP")
    if not raw_runner_temp:
        raise ValueError("RUNNER_TEMP is required when GITHUB_OUTPUT is set")

    return validate_path(
        raw_output_path,
        raw_runner_temp,
        redact_paths=True,
        allow_hidden=True,
    )


def write_github_output(
    name: str,
    value: str,
    *,
    output_path: str | os.PathLike[str] | None = None,
    runner_temp: str | os.PathLike[str] | None = None,
) -> None:
    """Append a single GitHub Actions output assignment after path validation."""
    if not name or any(char in name for char in "=\r\n"):
        raise ValueError("GitHub output name is invalid")
    if any(char in value for char in "\r\n"):
        raise ValueError("GitHub output value cannot contain newlines")

    path = resolve_github_output_path(output_path, runner_temp=runner_temp)
    if path is None:
        return

    with path.open("a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def main(
    argv: list[str] | None = None,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> int:
    """Run the PyPI version-existence check for GitHub Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Project version to check on PyPI.")
    parser.add_argument(
        "--package",
        default=DEFAULT_PACKAGE_NAME,
        help=f"Normalized PyPI package name. Default: {DEFAULT_PACKAGE_NAME}",
    )
    parser.add_argument(
        "--base-url",
        default=PYPI_JSON_BASE_URL,
        help=f"PyPI JSON API base URL. Default: {PYPI_JSON_BASE_URL}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"GitHub Actions output name. Default: {DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args(argv)

    exists = (
        "true"
        if release_exists(
            args.package,
            args.version,
            base_url=args.base_url,
            urlopen=urlopen,
        )
        else "false"
    )
    write_github_output(args.output_name, exists)
    print(f"{args.output_name}={exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
