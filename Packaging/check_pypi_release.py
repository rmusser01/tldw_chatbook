#!/usr/bin/env python3
"""Check whether a package version already exists on PyPI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from packaging.version import InvalidVersion, Version
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from tldw_chatbook.Utils.path_validation import validate_path


PYPI_JSON_BASE_URL = "https://pypi.org/pypi"
PYPI_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_PACKAGE_NAME = "tldw-chatbook"
DEFAULT_OUTPUT_NAME = "publish_release"
FIXED_OUTPUT_NAMES = frozenset(("release_exists", "latest_version"))
PACKAGE_NAME_PATTERN = re.compile(
    r"^([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])$"
)
URL_ADAPTER = TypeAdapter(AnyUrl)


@dataclass(frozen=True)
class ReleaseDecision:
    """PyPI publication decision for a candidate package version."""

    release_exists: bool
    latest_version: str | None
    publish_release: bool


class HttpResponse(Protocol):
    def __enter__(self) -> "HttpResponse": ...

    def __exit__(self, *_args: object) -> None: ...

    def read(self, size: int = -1) -> bytes: ...


class UrlOpen(Protocol):
    def __call__(self, url: str, timeout: int) -> HttpResponse: ...


class PypiProjectResponse(BaseModel):
    """Validated subset of the PyPI project JSON response."""

    model_config = ConfigDict(extra="ignore")

    releases: dict[str, object] = Field(default_factory=dict)


class ReleaseCheckInput(BaseModel):
    """Validated command-line inputs for the PyPI release check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    package_name: str = Field(default=DEFAULT_PACKAGE_NAME, min_length=1)
    version: str = Field(min_length=1)
    base_url: str = Field(default=PYPI_JSON_BASE_URL, min_length=1)
    output_name: str = Field(default=DEFAULT_OUTPUT_NAME, min_length=1)

    @field_validator("package_name")
    @classmethod
    def validate_package_name(cls, value: str) -> str:
        return validate_package_name(value)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return validate_base_url(value)

    @field_validator("output_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        return validate_custom_output_name(value)


def validate_package_name(package_name: str) -> str:
    """Return a validated PyPI package name."""
    if not package_name:
        raise ValueError("package name cannot be empty")
    if not PACKAGE_NAME_PATTERN.fullmatch(package_name):
        raise ValueError("package name is invalid")
    return package_name


def validate_version(version: str) -> str:
    """Return a validated PEP 440 package version."""
    if not version:
        raise ValueError("version cannot be empty")
    try:
        Version(version)
    except InvalidVersion as exc:
        raise ValueError("version is invalid") from exc
    return version


def validate_base_url(base_url: str) -> str:
    """Return a validated HTTP(S) PyPI JSON API base URL."""
    if not base_url:
        raise ValueError("base URL cannot be empty")
    parsed = URL_ADAPTER.validate_python(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL must use http or https")
    return str(parsed).rstrip("/")


def validate_github_output_name(name: str) -> str:
    """Return a GitHub Actions output name that is safe to write."""
    if not name or any(char in name for char in "=\r\n"):
        raise ValueError("GitHub output name is invalid")
    return name


def validate_custom_output_name(name: str) -> str:
    """Return a custom output name that cannot overwrite fixed metadata."""
    validate_github_output_name(name)
    if name in FIXED_OUTPUT_NAMES:
        raise ValueError("custom output name cannot overwrite fixed release metadata")
    return name


def release_exists(
    package_name: str,
    version: str,
    *,
    base_url: str = PYPI_JSON_BASE_URL,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    """Return whether the package version already exists in the PyPI JSON API."""
    package_name = validate_package_name(package_name)
    version = validate_version(version)
    base_url = validate_base_url(base_url)

    package = urllib.parse.quote(package_name, safe="")
    quoted_version = urllib.parse.quote(version, safe="")
    url = f"{base_url}/{package}/{quoted_version}/json"

    try:
        with urlopen(url, timeout=PYPI_REQUEST_TIMEOUT_SECONDS) as response:
            response.read(1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise

    return True


def latest_release_version(
    package_name: str,
    *,
    base_url: str = PYPI_JSON_BASE_URL,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> str | None:
    """Return the latest valid package version from the PyPI JSON API.

    Args:
        package_name: PyPI project name to query.
        base_url: Base URL for the PyPI-compatible JSON API.
        urlopen: URL opener compatible with ``urllib.request.urlopen``.

    Returns:
        The newest valid release version string, or ``None`` when the project
        does not exist or has no valid release versions.

    Raises:
        ValueError: If ``package_name`` or ``base_url`` is invalid.
        ValidationError: If the PyPI JSON response has an unexpected shape.
        urllib.error.HTTPError: If the API returns a non-404 HTTP error.
    """
    package_name = validate_package_name(package_name)
    base_url = validate_base_url(base_url)

    package = urllib.parse.quote(package_name, safe="")
    url = f"{base_url}/{package}/json"

    try:
        with urlopen(url, timeout=PYPI_REQUEST_TIMEOUT_SECONDS) as response:
            payload = PypiProjectResponse.model_validate_json(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    versions: list[Version] = []
    for version_text in payload.releases:
        try:
            versions.append(Version(version_text))
        except InvalidVersion:
            continue

    if not versions:
        return None

    return str(max(versions))


def release_decision(
    package_name: str,
    version: str,
    *,
    base_url: str = PYPI_JSON_BASE_URL,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> ReleaseDecision:
    """Return whether a candidate package version should be published.

    Args:
        package_name: PyPI project name to query.
        version: Candidate package version.
        base_url: Base URL for the PyPI-compatible JSON API.
        urlopen: URL opener compatible with ``urllib.request.urlopen``.

    Returns:
        A release decision containing exact-version existence, the latest PyPI
        release, and whether publication should proceed.

    Raises:
        ValueError: If any input value is invalid.
        ValidationError: If the PyPI project JSON response has an unexpected
            shape.
        urllib.error.HTTPError: If the API returns a non-404 HTTP error.
    """
    package_name = validate_package_name(package_name)
    version = validate_version(version)
    base_url = validate_base_url(base_url)
    candidate_version = Version(version)
    exists = release_exists(package_name, version, base_url=base_url, urlopen=urlopen)
    latest_version = latest_release_version(
        package_name,
        base_url=base_url,
        urlopen=urlopen,
    )

    if exists:
        return ReleaseDecision(
            release_exists=True,
            latest_version=latest_version or version,
            publish_release=False,
        )

    if latest_version is None:
        return ReleaseDecision(
            release_exists=False,
            latest_version=None,
            publish_release=True,
        )

    return ReleaseDecision(
        release_exists=False,
        latest_version=latest_version,
        publish_release=candidate_version > Version(latest_version),
    )


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
    validate_github_output_name(name)
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

    try:
        check_input = ReleaseCheckInput(
            package_name=args.package,
            version=args.version,
            base_url=args.base_url,
            output_name=args.output_name,
        )
    except ValidationError as exc:
        parser.error(str(exc))

    decision = release_decision(
        check_input.package_name,
        check_input.version,
        base_url=check_input.base_url,
        urlopen=urlopen,
    )
    outputs = (
        ("release_exists", str(decision.release_exists).lower()),
        ("latest_version", decision.latest_version or ""),
        (check_input.output_name, str(decision.publish_release).lower()),
    )
    for name, value in outputs:
        write_github_output(name, value)
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
