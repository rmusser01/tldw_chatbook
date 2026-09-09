"""Distribution and release-policy tests for the native voice AEC companion."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import re
import stat
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
import zipfile

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = REPO_ROOT / "native" / "voice_aec"
PACKAGING_ROOT = REPO_ROOT / "Packaging"
APP_PYPROJECT = REPO_ROOT / "pyproject.toml"
AEC_PYPROJECT = NATIVE_ROOT / "pyproject.toml"
WHEEL_CHECKER_PATH = PACKAGING_ROOT / "check_voice_aec_wheel.py"
VERSION_CHECKER_PATH = PACKAGING_ROOT / "check_voice_aec_version_sync.py"
ATTESTATION_CHECKER_PATH = PACKAGING_ROOT / "verify_voice_aec_attestations.py"
WHEEL_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "voice-aec-wheels.yml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-voice-aec.yml"
RELEASE_GUIDE_PATH = REPO_ROOT / "Docs" / "Development" / "TTS" / "voice-aec-release.md"
FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "voice_aec_attestation"
ATTESTATION_STATEMENT_FIXTURE = FIXTURES_ROOT / "statement.json"
SIGSTORE_BUNDLE_FIXTURE = FIXTURES_ROOT / "statement.sigstore.json"
PYBIND11_LICENSE_PATH = NATIVE_ROOT / "PYBIND11_LICENSE.txt"

PACKAGE_NAME = "tldw-voice-aec"
APP_VERSION = "0.1.8.0"
UPSTREAM_COMMIT = "109e23c9cec3a44e67c08774874a409741b1e58a"
UPSTREAM_TREE = "71115034a67c1e7f98a4c2a61d80278db5a9b2ae"
REPOSITORY = "rmusser01/tldw_chatbook"
WORKFLOW_IDENTITY = (
    "https://github.com/rmusser01/tldw_chatbook/"
    ".github/workflows/voice-aec-wheels.yml@refs/heads/main"
)
SOURCE_SHA = "a" * 40
SOURCE_TREE_DIGEST = "b" * 64
WORKFLOW_REF = "refs/heads/main"
WORKFLOW_PATH = ".github/workflows/voice-aec-wheels.yml"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
EVENT_NAME = "push"
REPOSITORY_ID = "123456789"
REPOSITORY_OWNER_ID = "987654321"
RUNNER_ENVIRONMENT = "github-hosted"
RUN_ID = "1122334455"
RUN_ATTEMPT = "2"
INVOCATION_ID = f"{REPOSITORY_URL}/actions/runs/{RUN_ID}/attempts/{RUN_ATTEMPT}"
ATTESTATION_FIXTURE_DIGEST = "c" * 64
ATTESTATION_FIXTURE_FILENAME = (
    f"tldw_voice_aec-{APP_VERSION}-cp311-cp311-macosx_11_0_arm64.whl"
)
PYBIND11_VERSION = "3.1.0"
PYBIND11_LICENSE_SHA256 = (
    "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb"
)
PYBIND11_SOURCE_URL = "https://github.com/pybind/pybind11/tree/v3.1.0"
DIST_INFO_LICENSE_FILES = (
    "THIRD_PARTY_NOTICES.md",
    "PYBIND11_LICENSE.txt",
    "vendor/webrtc/LICENSE",
    "vendor/webrtc/OOURA_LICENSE",
    "vendor/webrtc/PATENTS",
)

ACTION_PINS = {
    "actions/checkout": (
        "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "v7.0.1",
    ),
    "actions/setup-python": (
        "5fda3b95a4ea91299a34e894583c3862153e4b97",
        "v7.0.0",
    ),
    "actions/upload-artifact": (
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "v7.0.1",
    ),
    "actions/download-artifact": (
        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "v8.0.1",
    ),
    "actions/attest-build-provenance": (
        "4d101475d8b20a2381f78447822ac1eab6504dd8",
        "v4.2.2",
    ),
    "pypa/cibuildwheel": (
        "1828c10ab37f080699c7b81cea34097c684a7074",
        "v4.2.0",
    ),
    "pypa/gh-action-pypi-publish": (
        "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
        "v1.14.2",
    ),
    "sigstore/gh-action-sigstore-python": (
        "790bc6befb9d733738f18d8f895854b453640ec9",
        "v3.5.0",
    ),
    "anchore/sbom-action": (
        "3ad7283483fc7af8ff2b4ea19663c2d5ca935e26",
        "v0.24.2",
    ),
}


def _load_script(path: Path):
    assert path.is_file(), f"required checker is missing: {path.relative_to(REPO_ROOT)}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_synthetic_wheel(
    directory: Path,
    *,
    version: str = APP_VERSION,
    python_tag: str = "cp311",
    platform_tag: str = "macosx_11_0_arm64",
    metadata_license_files: tuple[str, ...] = DIST_INFO_LICENSE_FILES,
    extra_members: dict[str, bytes] | None = None,
    omit_members: frozenset[str] = frozenset(),
    dist_info_root: str | None = None,
) -> Path:
    wheel = directory / (
        f"tldw_voice_aec-{version}-{python_tag}-{python_tag}-{platform_tag}.whl"
    )
    license_bytes = (NATIVE_ROOT / "vendor" / "webrtc" / "LICENSE").read_bytes()
    patents_bytes = (NATIVE_ROOT / "vendor" / "webrtc" / "PATENTS").read_bytes()
    notices_bytes = (NATIVE_ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()
    pybind11_license_bytes = PYBIND11_LICENSE_PATH.read_bytes()
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {PACKAGE_NAME}\n"
        f"Version: {version}\n"
        "License-Expression: BSD-3-Clause\n"
        + "".join(f"License-File: {name}\n" for name in metadata_license_files)
        + "\n"
    ).encode()
    upstream = (NATIVE_ROOT / "vendor" / "webrtc" / "UPSTREAM.json").read_bytes()
    patches = (NATIVE_ROOT / "vendor" / "webrtc" / "PATCHES.md").read_bytes()
    source_manifest = (NATIVE_ROOT / "vendor" / "webrtc" / "FILES.sha256").read_bytes()
    pristine_manifest = (
        NATIVE_ROOT / "vendor" / "webrtc" / "PRISTINE_FILES.sha256"
    ).read_bytes()
    dist_info = dist_info_root or f"tldw_voice_aec-{version}.dist-info"
    members = {
        "tldw_voice_aec/__init__.py": b"from ._native import AecProcessor\n",
        "tldw_voice_aec/_native.cpython-311-darwin.so": b"native-extension",
        "tldw_voice_aec/provenance/UPSTREAM.json": upstream,
        "tldw_voice_aec/provenance/PATCHES.md": patches,
        "tldw_voice_aec/provenance/FILES.sha256": source_manifest,
        "tldw_voice_aec/provenance/PRISTINE_FILES.sha256": pristine_manifest,
        "tldw_voice_aec/provenance/THIRD_PARTY_NOTICES.md": notices_bytes,
        "tldw_voice_aec/provenance/PYBIND11_LICENSE.txt": pybind11_license_bytes,
        "tldw_voice_aec/provenance/LICENSE": license_bytes,
        "tldw_voice_aec/provenance/PATENTS": patents_bytes,
        "tldw_voice_aec/provenance/OOURA_LICENSE": (
            NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE"
        ).read_bytes(),
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: synthetic-test\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {python_tag}-{python_tag}-{platform_tag}\n"
        ).encode(),
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md": notices_bytes,
        f"{dist_info}/licenses/PYBIND11_LICENSE.txt": pybind11_license_bytes,
        f"{dist_info}/licenses/vendor/webrtc/LICENSE": license_bytes,
        f"{dist_info}/licenses/vendor/webrtc/OOURA_LICENSE": (
            NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE"
        ).read_bytes(),
        f"{dist_info}/licenses/vendor/webrtc/PATENTS": patents_bytes,
        f"{dist_info}/RECORD": b"",
    }
    members.update(extra_members or {})
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(members.items()):
            if name not in omit_members:
                archive.writestr(name, contents)
    return wheel


def _write_synthetic_sdist(
    directory: Path,
    *,
    extra_members: dict[str, bytes] | None = None,
    omit_members: frozenset[str] = frozenset(),
) -> Path:
    sdist = directory / f"tldw_voice_aec-{APP_VERSION}.tar.gz"
    root = f"tldw_voice_aec-{APP_VERSION}"
    blocked_parts = {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "_skbuild",
        "build",
        "cmakefiles",
        "dist",
        "wheelhouse",
    }
    blocked_suffixes = (
        ".a",
        ".dll",
        ".dylib",
        ".lib",
        ".o",
        ".obj",
        ".pyc",
        ".pyd",
        ".pyo",
        ".so",
        ".tar.gz",
        ".whl",
        ".zip",
    )
    members = {}
    for path in NATIVE_ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(NATIVE_ROOT).as_posix()
        parts = tuple(part.lower() for part in Path(relative).parts)
        if any(
            part in blocked_parts
            or part.endswith(".egg-info")
            or part.startswith("cmake-build-")
            for part in parts
        ):
            continue
        if parts[-1].endswith(blocked_suffixes):
            continue
        members[relative] = path.read_bytes()
    members["PKG-INFO"] = (
        "Metadata-Version: 2.4\n"
        f"Name: {PACKAGE_NAME}\n"
        f"Version: {APP_VERSION}\n"
        "License-Expression: BSD-3-Clause\n"
        + "".join(f"License-File: {name}\n" for name in DIST_INFO_LICENSE_FILES)
        + "\n"
    ).encode()
    members.update(extra_members or {})
    with tarfile.open(sdist, "w:gz") as archive:
        for name, contents in sorted(members.items()):
            if name in omit_members:
                continue
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return sdist


def _attestation_from_checked_in_fixture(
    *, filename: str, digest: str, workflow_identity: str = WORKFLOW_IDENTITY
) -> tuple[bytes, bytes]:
    statement = json.loads(ATTESTATION_STATEMENT_FIXTURE.read_text(encoding="utf-8"))
    statement["subject"] = [{"name": f"dist/{filename}", "digest": {"sha256": digest}}]
    build_definition = statement["predicate"]["buildDefinition"]
    build_definition["buildType"] = "https://actions.github.io/buildtypes/workflow/v1"
    build_definition["externalParameters"] = {
        "workflow": {
            "ref": WORKFLOW_REF,
            "repository": REPOSITORY_URL,
            "path": WORKFLOW_PATH,
        }
    }
    build_definition["internalParameters"] = {
        "github": {
            "event_name": EVENT_NAME,
            "repository_id": REPOSITORY_ID,
            "repository_owner_id": REPOSITORY_OWNER_ID,
            "runner_environment": RUNNER_ENVIRONMENT,
        }
    }
    build_definition["resolvedDependencies"] = [
        {
            "uri": f"git+{REPOSITORY_URL}@{WORKFLOW_REF}",
            "digest": {"gitCommit": SOURCE_SHA},
        }
    ]
    statement["predicate"]["runDetails"] = {
        "builder": {"id": workflow_identity},
        "metadata": {"invocationId": INVOCATION_ID},
    }
    statement["predicate"]["tldw_source_tree_digest"] = SOURCE_TREE_DIGEST
    statement_bytes = json.dumps(
        statement, sort_keys=True, separators=(",", ":")
    ).encode()
    bundle = json.loads(SIGSTORE_BUNDLE_FIXTURE.read_text(encoding="utf-8"))
    bundle["messageSignature"]["messageDigest"]["digest"] = base64.b64encode(
        hashlib.sha256(statement_bytes).digest()
    ).decode()
    return statement_bytes, json.dumps(bundle, sort_keys=True).encode()


def _provenance_expectations() -> dict[str, str]:
    return {
        "repository": REPOSITORY,
        "workflow_identity": WORKFLOW_IDENTITY,
        "source_sha": SOURCE_SHA,
        "source_tree_digest": SOURCE_TREE_DIGEST,
        "workflow_ref": WORKFLOW_REF,
        "workflow_path": WORKFLOW_PATH,
        "event_name": EVENT_NAME,
        "repository_id": REPOSITORY_ID,
        "repository_owner_id": REPOSITORY_OWNER_ID,
        "runner_environment": RUNNER_ENVIRONMENT,
        "run_id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
    }


def _write_synthetic_release_bundle(directory: Path) -> tuple[Path, list[Path]]:
    bundle = directory / "bundle"
    dist = bundle / "dist"
    attestations = bundle / "attestations"
    dist.mkdir(parents=True)
    attestations.mkdir()
    wheel_targets = (
        ("macosx_11_0_x86_64", "macos-x86_64"),
        ("macosx_11_0_arm64", "macos-arm64"),
        ("win_amd64", "windows-x86_64"),
        ("manylinux_2_17_x86_64", "linux-x86_64"),
        ("manylinux_2_17_aarch64", "linux-aarch64"),
    )
    wheels = [
        _write_synthetic_wheel(
            dist,
            python_tag=python_tag,
            platform_tag=platform_tag,
        )
        for python_tag in ("cp311", "cp312", "cp313")
        for platform_tag, _ in wheel_targets
    ]
    sdist = dist / f"tldw_voice_aec-{APP_VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        source = NATIVE_ROOT / "pyproject.toml"
        archive.add(source, arcname=f"tldw_voice_aec-{APP_VERSION}/pyproject.toml")
    distributions = wheels + [sdist]
    digests = {path.name: _sha256_bytes(path.read_bytes()) for path in distributions}
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{digests[name]}  dist/{name}\n" for name in sorted(digests)),
        encoding="utf-8",
    )
    (bundle / "source-tree.sha256").write_text(
        f"{SOURCE_TREE_DIGEST}  native/voice_aec\n", encoding="utf-8"
    )
    for name in ("THIRD_PARTY_NOTICES.md", "PYBIND11_LICENSE.txt"):
        (bundle / name).write_bytes((NATIVE_ROOT / name).read_bytes())
    for name in ("LICENSE", "PATENTS"):
        (bundle / name).write_bytes(
            (NATIVE_ROOT / "vendor" / "webrtc" / name).read_bytes()
        )
    dependency_inventory = [
        {
            "name": "WebRTC AEC3",
            "version": UPSTREAM_COMMIT,
            "license": "BSD-3-Clause",
            "source_url": "https://webrtc.googlesource.com/src",
        },
        {
            "name": "Abseil",
            "version": "ac875ae5393d0516243cfd5d078cd4b098388f6b",
            "license": "Apache-2.0",
            "source_url": (
                "https://chromium.googlesource.com/chromium/src/third_party"
            ),
        },
        {
            "name": "Ooura FFT",
            "version": "NOASSERTION",
            "license": "LicenseRef-Ooura-FFT",
            "source_url": "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html",
        },
        {
            "name": "pybind11",
            "version": PYBIND11_VERSION,
            "license": "BSD-3-Clause",
            "source_url": PYBIND11_SOURCE_URL,
        },
    ]
    inventory_relationships = [
        {
            "distribution": distribution,
            "relationship": "DEPENDS_ON",
            "dependency": dependency["name"],
        }
        for distribution in sorted(digests)
        for dependency in dependency_inventory
    ]
    (bundle / "license-inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "distributions": sorted(digests),
                "licenses": [
                    "WebRTC BSD-3-Clause",
                    "WebRTC PATENTS",
                    "Abseil Apache-2.0",
                    "Ooura FFT LicenseRef-Ooura-FFT",
                    "pybind11 BSD-3-Clause",
                ],
                "dependencies": dependency_inventory,
                "relationships": inventory_relationships,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    distribution_packages = [
        {
            "name": name,
            "SPDXID": f"SPDXRef-Distribution-{index}",
            "checksums": [{"algorithm": "SHA256", "checksumValue": digests[name]}],
        }
        for index, name in enumerate(sorted(digests), start=1)
    ]
    dependency_packages = [
        {
            "name": "WebRTC AEC3",
            "SPDXID": "SPDXRef-WebRTC-AEC3",
            "versionInfo": UPSTREAM_COMMIT,
            "licenseConcluded": "BSD-3-Clause",
        },
        {
            "name": "Abseil",
            "SPDXID": "SPDXRef-Abseil",
            "versionInfo": "ac875ae5393d0516243cfd5d078cd4b098388f6b",
            "licenseConcluded": "Apache-2.0",
        },
        {
            "name": "Ooura FFT",
            "SPDXID": "SPDXRef-Ooura-FFT",
            "versionInfo": "NOASSERTION",
            "licenseConcluded": "LicenseRef-Ooura-FFT",
        },
        {
            "name": "pybind11",
            "SPDXID": "SPDXRef-pybind11",
            "versionInfo": PYBIND11_VERSION,
            "downloadLocation": PYBIND11_SOURCE_URL,
            "licenseConcluded": "BSD-3-Clause",
        },
    ]
    relationships = [
        {
            "spdxElementId": distribution["SPDXID"],
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": dependency["SPDXID"],
        }
        for distribution in distribution_packages
        for dependency in dependency_packages
    ]
    (bundle / "sbom.spdx.json").write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "dataLicense": "CC0-1.0",
                "name": "tldw-voice-aec release bundle",
                "hasExtractedLicensingInfos": [
                    {
                        "licenseId": "LicenseRef-Ooura-FFT",
                        "extractedText": (
                            NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE"
                        ).read_text(encoding="utf-8"),
                    }
                ],
                "packages": distribution_packages + dependency_packages,
                "relationships": relationships,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (bundle / "github-build-provenance.jsonl").write_text(
        '{"synthetic_github_attestation":true}\n', encoding="utf-8"
    )
    for name, digest in digests.items():
        statement, signature_bundle = _attestation_from_checked_in_fixture(
            filename=name, digest=digest
        )
        statement_path = attestations / f"{name}.provenance.json"
        statement_path.write_bytes(statement)
        (attestations / f"{name}.provenance.json.sigstore.json").write_bytes(
            signature_bundle
        )
    return bundle, distributions


def test_application_and_companion_versions_are_exactly_locked() -> None:
    checker = _load_script(VERSION_CHECKER_PATH)

    assert checker.check_version_sync(REPO_ROOT) == []
    assert checker.main([]) == 0


def test_public_version_tuple_matches_application_metadata() -> None:
    import tldw_chatbook

    assert ".".join(map(str, tldw_chatbook.VERSION_TUPLE)) == tldw_chatbook.__version__


def test_version_checker_rejects_public_tuple_drift(tmp_path: Path) -> None:
    checker = _load_script(VERSION_CHECKER_PATH)
    for relative in (
        "pyproject.toml",
        "native/voice_aec/pyproject.toml",
        "native/voice_aec/PYBIND11_LICENSE.txt",
        "tldw_chatbook/__init__.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    init = tmp_path / "tldw_chatbook/__init__.py"
    source = init.read_text(encoding="utf-8")
    source = re.sub(r"VERSION_TUPLE = .*", "VERSION_TUPLE = (9, 9, 9, 9)", source)
    init.write_text(source, encoding="utf-8")
    assert "VERSION_TUPLE" in "\n".join(checker.check_version_sync(tmp_path))


def test_pyprojects_pin_the_companion_distribution_exactly() -> None:
    app = tomllib.loads(APP_PYPROJECT.read_text(encoding="utf-8"))
    companion = tomllib.loads(AEC_PYPROJECT.read_text(encoding="utf-8"))

    # APP_VERSION remains the historical attestation fixture's version.
    current_version = "0.2.0"
    assert app["project"]["version"] == current_version
    assert companion["project"]["name"] == PACKAGE_NAME
    assert companion["project"]["version"] == current_version
    assert (
        f"{PACKAGE_NAME}=={current_version}"
        in app["project"]["optional-dependencies"]["speech_recording"]
    )


def test_pybind11_build_pin_and_reviewed_license_are_exact() -> None:
    companion = tomllib.loads(AEC_PYPROJECT.read_text(encoding="utf-8"))

    assert f"pybind11=={PYBIND11_VERSION}" in companion["build-system"]["requires"]
    assert not any(
        requirement.startswith("pybind11")
        and requirement != f"pybind11=={PYBIND11_VERSION}"
        for requirement in companion["build-system"]["requires"]
    )
    assert PYBIND11_LICENSE_PATH.is_file()
    assert _sha256_bytes(PYBIND11_LICENSE_PATH.read_bytes()) == PYBIND11_LICENSE_SHA256
    assert "PYBIND11_LICENSE.txt" in companion["project"]["license-files"]

    checker = _load_script(VERSION_CHECKER_PATH)
    assert checker.check_version_sync(REPO_ROOT) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("open-pin", "exact pybind11 pin"),
        ("altered-license", "does not match pybind11 v3.1.0"),
    ],
)
def test_version_checker_rejects_pybind11_contract_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    checker = _load_script(VERSION_CHECKER_PATH)
    native_root = tmp_path / "native" / "voice_aec"
    native_root.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(APP_PYPROJECT.read_bytes())
    companion = AEC_PYPROJECT.read_text(encoding="utf-8")
    if mutation == "open-pin":
        companion = companion.replace(
            f'"pybind11=={PYBIND11_VERSION}"', '"pybind11>=3,<4"'
        )
    (native_root / "pyproject.toml").write_text(companion, encoding="utf-8")
    license_bytes = PYBIND11_LICENSE_PATH.read_bytes()
    if mutation == "altered-license":
        license_bytes += b"altered"
    (native_root / "PYBIND11_LICENSE.txt").write_bytes(license_bytes)

    assert expected in "\n".join(checker.check_version_sync(tmp_path))


def test_wheel_checker_accepts_complete_platform_wheel(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(tmp_path)

    assert checker.check_wheel(wheel, expected_version=APP_VERSION) == []


def test_wheel_checker_accepts_empty_platform_directory_entries(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    dist_info = "tldw_voice_aec-0.2.0.dist-info"
    wheel = _write_synthetic_wheel(
        tmp_path,
        version="0.2.0",
        extra_members=dict.fromkeys(
            (
                f"{dist_info}/",
                f"{dist_info}/licenses/",
                f"{dist_info}/licenses/vendor/",
                f"{dist_info}/licenses/vendor/webrtc/",
                "tldw_voice_aec/",
                "tldw_voice_aec/provenance/",
            ),
            b"",
        ),
    )

    assert checker.check_wheel(wheel, expected_version="0.2.0") == []


@pytest.mark.parametrize(
    "name",
    [
        "../escape/",
        "/absolute/",
        "C:/absolute/",
        "C:relative/",
        "tldw_voice_aec\\bad/",
        "./tldw_voice_aec/",
        "tldw_voice_aec//provenance/",
        "tldw_voice_aec/../escape/",
        "./",
    ],
)
def test_wheel_checker_rejects_unsafe_directory_paths(
    tmp_path: Path, name: str
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(tmp_path, extra_members={name: b""})

    assert "unsafe archive members" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


@pytest.mark.parametrize(
    ("contents", "mode"),
    [(b"hidden payload", stat.S_IFDIR), (b"", stat.S_IFLNK), (b"", stat.S_IFREG)],
)
def test_wheel_checker_rejects_non_directory_payload_or_metadata(
    tmp_path: Path, contents: bytes, mode: int
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(tmp_path)
    directory = zipfile.ZipInfo("tldw_voice_aec/hidden/")
    directory.create_system = 3
    directory.external_attr = (mode | 0o755) << 16 | 0x10
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(directory, contents)

    assert "unsafe archive members" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


@pytest.mark.parametrize(
    "name", ["foreign-1.0.dist-info/", "tldw_voice_aec/nested.dist-info/"]
)
def test_wheel_checker_rejects_foreign_dist_info_directory(
    tmp_path: Path, name: str
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(tmp_path, extra_members={name: b""})

    assert "exact dist-info root" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


def test_wheel_checker_rejects_file_directory_collision(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(
        tmp_path, extra_members={"tldw_voice_aec/provenance/UPSTREAM.json/": b""}
    )

    assert "duplicate archive members" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


def test_wheel_checker_rejects_duplicate_directories(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(tmp_path, extra_members={"tldw_voice_aec/": b""})
    with zipfile.ZipFile(wheel, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("tldw_voice_aec/", b"")

    assert "duplicate archive members" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


@pytest.mark.parametrize(
    ("dist_info_root", "extra_members", "omit_member"),
    [
        ("tldw-voice-aec-0.1.8.0.dist-info", {}, None),
        ("nested/tldw_voice_aec-0.1.8.0.dist-info", {}, None),
        ("tldw_voice_aec-0.1.8.0.dist-info-marker", {}, None),
        (
            None,
            {"foreign-1.0.dist-info/METADATA": b"Name: foreign\nVersion: 1.0\n"},
            None,
        ),
        (
            None,
            {"foreign-1.0.DIST-INFO/METADATA": b"Name: foreign\nVersion: 1.0\n"},
            None,
        ),
        (None, {}, f"tldw_voice_aec-{APP_VERSION}.dist-info/RECORD"),
    ],
)
def test_wheel_checker_requires_one_exact_complete_dist_info_root(
    tmp_path: Path,
    dist_info_root: str | None,
    extra_members: dict[str, bytes],
    omit_member: str | None,
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    wheel = _write_synthetic_wheel(
        tmp_path,
        platform_tag="macosx_12_0_arm64",
        dist_info_root=dist_info_root,
        extra_members=extra_members,
        omit_members=frozenset({omit_member}) if omit_member else frozenset(),
    )

    assert "exact dist-info root" in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unexpected-library", "unexpected shared library"),
        ("missing-notices", "THIRD_PARTY_NOTICES.md"),
        ("missing-pybind11-license", "PYBIND11_LICENSE.txt"),
        ("missing-patch-series", "PATCHES.md"),
        ("missing-source-manifest", "FILES.sha256"),
        ("missing-pristine-manifest", "PRISTINE_FILES.sha256"),
        ("altered-pristine-manifest", "PRISTINE_FILES.sha256"),
        ("altered-patch-series", "PATCHES.md"),
        ("wrong-upstream", "pinned WebRTC provenance"),
    ],
)
def test_wheel_checker_fails_closed_on_distribution_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    extra: dict[str, bytes] = {}
    omitted: frozenset[str] = frozenset()
    if mutation == "unexpected-library":
        extra["tldw_voice_aec/.dylibs/libsurprise.dylib"] = b"unexpected"
    elif mutation == "missing-notices":
        omitted = frozenset({"tldw_voice_aec/provenance/THIRD_PARTY_NOTICES.md"})
    elif mutation == "missing-pybind11-license":
        omitted = frozenset({"tldw_voice_aec/provenance/PYBIND11_LICENSE.txt"})
    elif mutation == "missing-patch-series":
        omitted = frozenset({"tldw_voice_aec/provenance/PATCHES.md"})
    elif mutation == "missing-source-manifest":
        omitted = frozenset({"tldw_voice_aec/provenance/FILES.sha256"})
    elif mutation == "missing-pristine-manifest":
        omitted = frozenset({"tldw_voice_aec/provenance/PRISTINE_FILES.sha256"})
    elif mutation == "altered-pristine-manifest":
        extra["tldw_voice_aec/provenance/PRISTINE_FILES.sha256"] = b"tampered\n"
    elif mutation == "altered-patch-series":
        extra["tldw_voice_aec/provenance/PATCHES.md"] = b"undeclared patch\n"
    else:
        extra["tldw_voice_aec/provenance/UPSTREAM.json"] = json.dumps(
            {"commit": "0" * 40, "commit_tree": "0" * 40}
        ).encode()
    wheel = _write_synthetic_wheel(tmp_path, extra_members=extra, omit_members=omitted)

    assert expected in "\n".join(
        checker.check_wheel(wheel, expected_version=APP_VERSION)
    )


def test_wheel_checker_requires_complete_pep639_license_metadata(
    tmp_path: Path,
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    dist_info = f"tldw_voice_aec-{APP_VERSION}.dist-info"
    missing_patents_member = _write_synthetic_wheel(
        tmp_path,
        omit_members=frozenset({f"{dist_info}/licenses/vendor/webrtc/PATENTS"}),
    )

    assert "dist-info license file" in "\n".join(
        checker.check_wheel(missing_patents_member, expected_version=APP_VERSION)
    )

    missing_patents_header = _write_synthetic_wheel(
        tmp_path,
        platform_tag="macosx_12_0_arm64",
        metadata_license_files=tuple(
            name for name in DIST_INFO_LICENSE_FILES if not name.endswith("PATENTS")
        ),
    )
    assert "License-File headers" in "\n".join(
        checker.check_wheel(missing_patents_header, expected_version=APP_VERSION)
    )


def test_sdist_manifest_explicitly_excludes_generated_artifacts() -> None:
    companion = tomllib.loads(AEC_PYPROJECT.read_text(encoding="utf-8"))
    sdist = companion["tool"]["scikit-build"]["sdist"]

    assert set(sdist["exclude"]) >= {
        "dist/**",
        "build/**",
        "_skbuild/**",
        "wheelhouse/**",
        "cmake-build-*/**",
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
        "**/*.pyc",
        "**/*.pyo",
    }
    assert "vendor/webrtc/**" not in sdist.get("include", [])


def test_sdist_checker_accepts_closed_reviewed_source_archive(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(tmp_path)

    assert checker.check_sdist(sdist, expected_version=APP_VERSION) == []


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("dist/nested.whl", "generated/build artifact"),
        ("build/CMakeCache.txt", "generated/build artifact"),
        ("tests/__pycache__/test_binding.pyc", "generated/build artifact"),
    ],
)
def test_sdist_checker_rejects_generated_artifacts(
    tmp_path: Path, member: str, expected: str
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(
        tmp_path, extra_members={member: b"unqualified generated output"}
    )

    assert expected in "\n".join(
        checker.check_sdist(sdist, expected_version=APP_VERSION)
    )


@pytest.mark.parametrize(
    "member",
    [
        "vendor/webrtc/modules/audio_processing/aec3/unreviewed.cc",
        "tldw_voice_aec/injected.py",
    ],
)
def test_sdist_checker_rejects_unreviewed_source_files(
    tmp_path: Path, member: str
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(
        tmp_path, extra_members={member: b"unreviewed source"}
    )

    assert "unexpected source files" in "\n".join(
        checker.check_sdist(sdist, expected_version=APP_VERSION)
    )


def test_sdist_checker_validates_pep639_pkg_info(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(
        tmp_path,
        extra_members={
            "PKG-INFO": (
                "Metadata-Version: 2.1\n"
                f"Name: {PACKAGE_NAME}\n"
                f"Version: {APP_VERSION}\n\n"
            ).encode()
        },
    )

    errors = "\n".join(checker.check_sdist(sdist, expected_version=APP_VERSION))
    assert "Core Metadata 2.4" in errors
    assert "License-Expression" in errors
    assert "License-File headers" in errors


def test_sdist_checker_rejects_nonexact_pybind11_build_pin(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    pyproject = AEC_PYPROJECT.read_text(encoding="utf-8").replace(
        f'"pybind11=={PYBIND11_VERSION}"', '"pybind11>=3,<4"'
    )
    sdist = _write_synthetic_sdist(
        tmp_path, extra_members={"pyproject.toml": pyproject.encode()}
    )

    assert "pybind11 build pin" in "\n".join(
        checker.check_sdist(sdist, expected_version=APP_VERSION)
    )


def test_sdist_checker_requires_reviewed_source_manifest(tmp_path: Path) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(
        tmp_path, omit_members=frozenset({"vendor/webrtc/FILES.sha256"})
    )

    assert "missing required source" in "\n".join(
        checker.check_sdist(sdist, expected_version=APP_VERSION)
    )


def test_sdist_checker_requires_exact_pristine_source_manifest(
    tmp_path: Path,
) -> None:
    checker = _load_script(WHEEL_CHECKER_PATH)
    sdist = _write_synthetic_sdist(
        tmp_path,
        extra_members={"vendor/webrtc/PRISTINE_FILES.sha256": b"tampered\n"},
    )

    assert "differs from the checkout" in "\n".join(
        checker.check_sdist(sdist, expected_version=APP_VERSION)
    )


def test_plain_application_import_does_not_load_native_companion(
    tmp_path: Path,
) -> None:
    code = (
        "import sys\n"
        "import tldw_chatbook\n"
        "assert 'tldw_voice_aec' not in sys.modules\n"
        "assert not any(name.startswith('tldw_voice_aec.') for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_attestation_checker_accepts_checked_in_statement_and_bundle_fixture() -> None:
    checker = _load_script(ATTESTATION_CHECKER_PATH)

    assert ATTESTATION_STATEMENT_FIXTURE.is_file()
    assert SIGSTORE_BUNDLE_FIXTURE.is_file()
    assert (
        checker._verify_attestation(
            ATTESTATION_STATEMENT_FIXTURE,
            SIGSTORE_BUNDLE_FIXTURE,
            filename=ATTESTATION_FIXTURE_FILENAME,
            digest=ATTESTATION_FIXTURE_DIGEST,
            **_provenance_expectations(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("subject-digest", "subject does not match qualified bytes"),
        ("repository", "repository mismatch"),
        ("workflow-identity", "workflow identity mismatch"),
        ("source-sha", "source sha mismatch"),
        ("source-tree-digest", "source tree digest mismatch"),
        ("workflow-ref", "workflow ref mismatch"),
        ("workflow-path", "workflow path mismatch"),
        ("event-name", "event name mismatch"),
        ("repository-id", "repository id mismatch"),
        ("repository-owner-id", "repository owner id mismatch"),
        ("runner-environment", "runner environment mismatch"),
        ("run-id", "invocation id mismatch"),
        ("run-attempt", "invocation id mismatch"),
        ("build-type", "build type mismatch"),
        ("unexpected-external-parameter", "external parameters mismatch"),
        ("unexpected-internal-parameter", "internal parameters mismatch"),
        ("resolved-dependency-uri", "resolved dependencies mismatch"),
        ("statement-binding", "does not bind the provenance statement"),
    ],
)
def test_checked_in_attestation_fixture_fails_closed_on_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    checker = _load_script(ATTESTATION_CHECKER_PATH)
    statement = tmp_path / "statement.json"
    sigstore_bundle = tmp_path / "statement.sigstore.json"
    statement.write_bytes(ATTESTATION_STATEMENT_FIXTURE.read_bytes())
    sigstore_bundle.write_bytes(SIGSTORE_BUNDLE_FIXTURE.read_bytes())
    arguments = {
        "filename": ATTESTATION_FIXTURE_FILENAME,
        "digest": ATTESTATION_FIXTURE_DIGEST,
        **_provenance_expectations(),
    }
    if mutation == "subject-digest":
        arguments["digest"] = "d" * 64
    elif mutation == "repository":
        arguments["repository"] = "attacker/repository"
    elif mutation == "workflow-identity":
        arguments["workflow_identity"] = (
            "https://github.com/attacker/repository/workflow.yml@refs/heads/main"
        )
    elif mutation == "source-sha":
        arguments["source_sha"] = "d" * 40
    elif mutation == "source-tree-digest":
        arguments["source_tree_digest"] = "d" * 64
    elif mutation in {
        "workflow-ref",
        "workflow-path",
        "event-name",
        "repository-id",
        "repository-owner-id",
        "runner-environment",
        "run-id",
        "run-attempt",
    }:
        arguments[mutation.replace("-", "_")] = "wrong"
    elif mutation in {
        "build-type",
        "unexpected-external-parameter",
        "unexpected-internal-parameter",
        "resolved-dependency-uri",
    }:
        contents = json.loads(statement.read_text(encoding="utf-8"))
        build = contents["predicate"]["buildDefinition"]
        if mutation == "build-type":
            build["buildType"] = "https://example.invalid/build-type"
        elif mutation == "unexpected-external-parameter":
            build["externalParameters"]["untrusted"] = "injected"
        elif mutation == "unexpected-internal-parameter":
            build["internalParameters"]["github"]["untrusted"] = "injected"
        else:
            build["resolvedDependencies"][0]["uri"] = "git+https://example.invalid"
        statement.write_text(
            json.dumps(contents, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        sigstore = json.loads(sigstore_bundle.read_text(encoding="utf-8"))
        sigstore["messageSignature"]["messageDigest"]["digest"] = base64.b64encode(
            hashlib.sha256(statement.read_bytes()).digest()
        ).decode()
        sigstore_bundle.write_text(
            json.dumps(sigstore, sort_keys=True), encoding="utf-8"
        )
    else:
        statement.write_bytes(statement.read_bytes() + b"\n")

    assert expected in "\n".join(
        checker._verify_attestation(statement, sigstore_bundle, **arguments)
    )


def test_attestation_checker_accepts_complete_synthetic_bundle(tmp_path: Path) -> None:
    checker = _load_script(ATTESTATION_CHECKER_PATH)
    bundle, _ = _write_synthetic_release_bundle(tmp_path)

    assert (
        checker.verify_release_bundle(
            bundle,
            **_provenance_expectations(),
            expected_version=APP_VERSION,
        )
        == []
    )


def test_attestation_checker_structural_only_mode_is_closed_and_distinct(
    tmp_path: Path,
) -> None:
    checker = _load_script(ATTESTATION_CHECKER_PATH)
    bundle, distributions = _write_synthetic_release_bundle(tmp_path)
    for path in (bundle / "attestations").iterdir():
        path.unlink()
    (bundle / "attestations").rmdir()
    (bundle / "github-build-provenance.jsonl").unlink()

    assert (
        checker.verify_release_bundle(
            bundle,
            **_provenance_expectations(),
            expected_version=APP_VERSION,
            structural_only=True,
        )
        == []
    )
    signed_errors = checker.verify_release_bundle(
        bundle,
        **_provenance_expectations(),
        expected_version=APP_VERSION,
    )
    assert "missing bundle files" in "\n".join(signed_errors)

    (bundle / "attestations").mkdir()
    (bundle / "attestations" / f"{distributions[0].name}.provenance.json").write_text(
        "{}", encoding="utf-8"
    )
    assert "unexpected bundle files" in "\n".join(
        checker.verify_release_bundle(
            bundle,
            **_provenance_expectations(),
            expected_version=APP_VERSION,
            structural_only=True,
        )
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing-attestation", "missing attestation"),
        ("wrong-identity", "workflow identity"),
        ("extra-file", "unexpected bundle files"),
        ("missing-sbom-subject", "SBOM"),
        ("missing-sbom-relationships", "SBOM dependency relationships"),
        ("wrong-sbom-dependency-pin", "SBOM dependency metadata"),
        ("missing-pybind11-inventory", "license inventory dependency metadata"),
        ("wrong-pybind11-inventory-version", "license inventory dependency metadata"),
        ("wrong-pybind11-inventory-license", "license inventory dependency metadata"),
        ("wrong-pybind11-inventory-source", "license inventory dependency metadata"),
        ("missing-pybind11-license-notice", "license inventory notices"),
        ("missing-pybind11-relationship", "license inventory relationships"),
        ("wrong-pybind11-sbom-pin", "SBOM dependency metadata"),
        ("wrong-pybind11-sbom-license", "SBOM dependency metadata"),
        ("wrong-pybind11-sbom-source", "SBOM dependency metadata"),
        ("missing-pybind11-sbom-relationship", "SBOM dependency relationships"),
        ("missing-ooura-extracted-license", "Ooura extracted license"),
        ("altered-ooura-extracted-license", "Ooura extracted license"),
        ("extra-extracted-license", "Ooura extracted license"),
        ("wrong-sbom-distribution-hash", "SBOM distribution checksum"),
        ("missing-platform-wheel", "missing supported wheel targets"),
    ],
)
def test_attestation_checker_rejects_synthetic_bundle_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    checker = _load_script(ATTESTATION_CHECKER_PATH)
    bundle, distributions = _write_synthetic_release_bundle(tmp_path)
    wheel = distributions[0]
    statement = bundle / "attestations" / f"{wheel.name}.provenance.json"
    attestation = (
        bundle / "attestations" / (f"{wheel.name}.provenance.json.sigstore.json")
    )
    if mutation == "missing-attestation":
        attestation.unlink()
    elif mutation == "wrong-identity":
        digest = _sha256_bytes(wheel.read_bytes())
        wrong_statement, wrong_bundle = _attestation_from_checked_in_fixture(
            filename=wheel.name,
            digest=digest,
            workflow_identity="https://github.com/attacker/repo/workflow.yml@main",
        )
        statement.write_bytes(wrong_statement)
        attestation.write_bytes(wrong_bundle)
    elif mutation == "extra-file":
        (bundle / "unqualified.whl").write_bytes(b"extra")
    elif mutation == "missing-sbom-subject":
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        sbom["packages"] = []
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "missing-sbom-relationships":
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        sbom["relationships"] = []
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "wrong-sbom-dependency-pin":
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        next(package for package in sbom["packages"] if package["name"] == "Abseil")[
            "versionInfo"
        ] = "0" * 40
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "missing-pybind11-inventory":
        inventory_path = bundle / "license-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["dependencies"] = [
            dependency
            for dependency in inventory["dependencies"]
            if dependency["name"] != "pybind11"
        ]
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True), encoding="utf-8"
        )
    elif mutation.startswith("wrong-pybind11-inventory-"):
        inventory_path = bundle / "license-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        pybind11 = next(
            dependency
            for dependency in inventory["dependencies"]
            if dependency["name"] == "pybind11"
        )
        field = mutation.removeprefix("wrong-pybind11-inventory-")
        pybind11[field if field != "source" else "source_url"] = "wrong"
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "missing-pybind11-license-notice":
        inventory_path = bundle / "license-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["licenses"].remove("pybind11 BSD-3-Clause")
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "missing-pybind11-relationship":
        inventory_path = bundle / "license-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["relationships"] = [
            relationship
            for relationship in inventory["relationships"]
            if not (
                relationship["distribution"] == wheel.name
                and relationship["dependency"] == "pybind11"
            )
        ]
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True), encoding="utf-8"
        )
    elif mutation.startswith("wrong-pybind11-sbom-"):
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        pybind11 = next(
            package for package in sbom["packages"] if package["name"] == "pybind11"
        )
        field = mutation.removeprefix("wrong-pybind11-sbom-")
        key = {
            "pin": "versionInfo",
            "license": "licenseConcluded",
            "source": "downloadLocation",
        }[field]
        pybind11[key] = "wrong"
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    elif mutation == "missing-pybind11-sbom-relationship":
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        distribution_id = next(
            package["SPDXID"]
            for package in sbom["packages"]
            if package["name"] == wheel.name
        )
        sbom["relationships"] = [
            relationship
            for relationship in sbom["relationships"]
            if not (
                relationship["spdxElementId"] == distribution_id
                and relationship["relatedSpdxElement"] == "SPDXRef-pybind11"
            )
        ]
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    elif mutation in {
        "missing-ooura-extracted-license",
        "altered-ooura-extracted-license",
        "extra-extracted-license",
    }:
        sbom_path = bundle / "sbom.spdx.json"
        sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        if mutation == "missing-ooura-extracted-license":
            sbom["hasExtractedLicensingInfos"] = []
        elif mutation == "altered-ooura-extracted-license":
            sbom["hasExtractedLicensingInfos"][0]["extractedText"] += "altered"
        else:
            sbom["hasExtractedLicensingInfos"].append(
                {"licenseId": "LicenseRef-Injected", "extractedText": "injected"}
            )
        sbom_path.write_text(json.dumps(sbom, sort_keys=True), encoding="utf-8")
    elif mutation == "wrong-sbom-distribution-hash":
        sbom = json.loads((bundle / "sbom.spdx.json").read_text(encoding="utf-8"))
        next(package for package in sbom["packages"] if package["name"] == wheel.name)[
            "checksums"
        ][0]["checksumValue"] = "0" * 64
        (bundle / "sbom.spdx.json").write_text(
            json.dumps(sbom, sort_keys=True), encoding="utf-8"
        )
    else:
        wheel.unlink()

    assert expected in "\n".join(
        checker.verify_release_bundle(
            bundle,
            **_provenance_expectations(),
            expected_version=APP_VERSION,
        )
    )


def test_pull_request_workflow_covers_every_supported_wheel_and_never_publishes() -> (
    None
):
    assert WHEEL_WORKFLOW_PATH.is_file()
    workflow = WHEEL_WORKFLOW_PATH.read_text(encoding="utf-8")

    for python_version in ("cp311", "cp312", "cp313"):
        assert python_version in workflow
    for platform in (
        "macos-x86_64",
        "macos-arm64",
        "windows-x86_64",
        "linux-x86_64",
        "linux-aarch64",
    ):
        assert platform in workflow
    for required in (
        "cibuildwheel",
        "CIBW_TEST_COMMAND",
        "auditwheel",
        "delvewheel",
        "delocate",
        "check_voice_aec_wheel.py",
        "sbom.spdx.json",
        "SHA256SUMS",
        "source-tree.sha256",
        "sigstore",
        "verify: true",
        "verify-cert-identity: https://github.com/${{ github.workflow_ref }}",
        "verify-oidc-issuer: https://token.actions.githubusercontent.com",
        "attest-build-provenance",
        "THIRD_PARTY_NOTICES.md",
        "PATENTS",
        "PYBIND11_LICENSE.txt",
        '"name": "pybind11"',
        f'"versionInfo": "{PYBIND11_VERSION}"',
        PYBIND11_SOURCE_URL,
    ):
        assert required in workflow
    cibw_test_command = workflow.partition("CIBW_TEST_COMMAND:")[2].partition(
        "CIBW_REPAIR_WHEEL_COMMAND_LINUX:"
    )[0]
    test_paths = re.findall(r"\{project\}/[^\s]+\.py", cibw_test_command)
    assert test_paths == [
        "{project}/native/voice_aec/tests/test_binding.py",
        "{project}/Tests/Packaging/test_voice_aec_installed_wheel.py",
    ]
    for path in test_paths:
        assert (REPO_ROOT / path.removeprefix("{project}/")).is_file()
    assert (
        "test_delayed_echo_exposes_refined_fresh_per_instance_delay_evidence"
        in cibw_test_command
    )
    assert (
        "\n          python Packaging/check_voice_aec_wheel.py qualified/dist/*\n"
        in workflow
    )
    assert "gh-action-pypi-publish" not in workflow
    assert "twine upload" not in workflow


def test_pull_requests_are_structural_only_and_never_receive_oidc_permissions() -> None:
    workflow = WHEEL_WORKFLOW_PATH.read_text(encoding="utf-8")
    structural_job = workflow.split("  qualify:", 1)[1].split("  attest-qualified:", 1)[
        0
    ]
    signed_job = workflow.split("  attest-qualified:", 1)[1]

    assert "id-token: write" not in structural_job
    assert "attestations: write" not in structural_job
    assert "--structural-only" in structural_job
    assert "voice-aec-structural-only-${{ github.sha }}" in structural_job
    assert "github.event_name != 'pull_request'" in signed_job
    assert "github.ref == 'refs/heads/main'" in signed_job
    assert "id-token: write" in signed_job
    assert "attestations: write" in signed_job
    assert "sigstore/gh-action-sigstore-python" in signed_job
    assert "actions/attest-build-provenance" in signed_job
    assert "voice-aec-qualified-${{ github.sha }}" in signed_job


def test_wheel_workflow_verifies_pristine_vendor_tree_before_any_build_or_signing() -> (
    None
):
    workflow = WHEEL_WORKFLOW_PATH.read_text(encoding="utf-8")
    invocation = (
        "python native/voice_aec/tools/vendor_webrtc_aec.py --verify-vendor-tree"
    )
    build_job = workflow.split("  build-wheels:", 1)[1].split("  qualify:", 1)[0]
    qualify_job = workflow.split("  qualify:", 1)[1].split("  attest-qualified:", 1)[0]

    assert workflow.count(invocation) == 2
    assert build_job.index(invocation) < build_job.index("pypa/cibuildwheel")
    assert qualify_job.index(invocation) < qualify_job.index(
        "python -m build native/voice_aec --sdist"
    )
    assert workflow.index(invocation) < workflow.index(
        "actions/attest-build-provenance"
    )
    assert workflow.index(invocation) < workflow.index(
        "sigstore/gh-action-sigstore-python"
    )


def test_attestation_job_depends_exactly_on_qualification() -> None:
    workflow = yaml.safe_load(WHEEL_WORKFLOW_PATH.read_text(encoding="utf-8"))
    needs = workflow["jobs"]["attest-qualified"]["needs"]

    assert needs == "qualify" or needs == ["qualify"]


def test_wheel_workflow_emits_official_github_slsa_fields_and_ooura_license() -> None:
    workflow = WHEEL_WORKFLOW_PATH.read_text(encoding="utf-8")

    for required in (
        "https://actions.github.io/buildtypes/workflow/v1",
        '"externalParameters": {"workflow":',
        '"internalParameters": {"github":',
        '"resolvedDependencies"',
        '"gitCommit": source_sha',
        '"invocationId"',
        '"tldw_source_tree_digest"',
        '"hasExtractedLicensingInfos"',
        '"licenseId": "LicenseRef-Ooura-FFT"',
        "native/voice_aec/vendor/webrtc/OOURA_LICENSE",
    ):
        assert required in workflow


def test_release_workflow_downloads_and_publishes_only_qualified_bytes() -> None:
    assert RELEASE_WORKFLOW_PATH.is_file()
    workflow = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")

    for required in (
        "workflow_dispatch",
        "matrix_run_id",
        "voice-aec-v",
        "pypi-release",
        "download-artifact",
        "verify_voice_aec_attestations.py",
        "sigstore verify identity",
        '--cert-identity "$WORKFLOW_IDENTITY"',
        '--source-digest "$SOURCE_SHA"',
        "--source-ref refs/heads/main",
        "--deny-self-hosted-runners",
        "SHA256SUMS",
        "source-tree.sha256",
        "repository_id=",
        "repository_owner_id=",
        "run_attempt=",
        '--workflow-ref "refs/heads/main"',
        '--workflow-path ".github/workflows/voice-aec-wheels.yml"',
        '--runner-environment "github-hosted"',
        '--run-id "$RUN_ID"',
        '--run-attempt "$RUN_ATTEMPT"',
        "--require-hashes",
        "--hash=sha256:",
        "attestations: true",
    ):
        assert required in workflow
    assert "cibuildwheel" not in workflow
    assert "python -m build native/voice_aec" not in workflow
    assert "twine upload" not in workflow
    assert "if: startsWith(github.ref" not in workflow
    assert r"\n+" not in workflow


def test_release_archives_byte_identical_qualified_evidence_before_publishing() -> None:
    workflow = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")
    publish_job = workflow.split("  publish-exact-bytes:", 1)[1].split(
        "  verify-published-wheels:", 1
    )[0]

    assert "environment: pypi-release" in publish_job
    assert "contents: write" in publish_job
    assert "gh release view" in publish_job
    assert 'gh release create "$RELEASE_TAG" "$ASSET_NAME" --verify-tag' in publish_job
    assert "gh release upload" not in publish_job
    assert "--clobber" not in publish_job
    assert "existing release is missing qualified evidence" in publish_job
    assert "--json tagName,isDraft,isImmutable" in publish_job
    assert 'test "$ACTUAL_TAG" = "$RELEASE_TAG"' in publish_job
    assert 'test "$IS_DRAFT" = "false"' in publish_job
    assert 'test "$IS_IMMUTABLE" = "true"' in publish_job
    assert "gh release download" in publish_job
    assert "sha256sum" in publish_job
    assert "voice-aec-qualified-${SOURCE_SHA}.tar.gz" in publish_job
    assert "existing release evidence differs; refusing overwrite" in publish_job
    assert "gh-action-pypi-publish" in publish_job
    assert publish_job.index("ARCHIVE_SHA256=") < publish_job.index(
        'gh release create "$RELEASE_TAG" "$ASSET_NAME"'
    )
    assert publish_job.index("--json tagName,isDraft,isImmutable") < publish_job.index(
        "mkdir downloaded-asset"
    )
    assert publish_job.index("mkdir downloaded-asset") < publish_job.index(
        "gh-action-pypi-publish"
    )


@pytest.mark.parametrize("path", [WHEEL_WORKFLOW_PATH, RELEASE_WORKFLOW_PATH])
def test_voice_aec_workflows_pin_every_action_by_full_sha(path: Path) -> None:
    assert path.is_file()
    workflow = path.read_text(encoding="utf-8")
    for line in workflow.splitlines():
        if "uses:" not in line:
            continue
        reference = line.split("uses:", 1)[1].split("#", 1)[0].strip()
        assert "@" in reference
        action, revision = reference.rsplit("@", 1)
        assert len(revision) == 40 and all(
            char in "0123456789abcdef" for char in revision
        )
        assert action in ACTION_PINS
        expected_sha, expected_tag = ACTION_PINS[action]
        assert revision == expected_sha
        assert f"# {expected_tag}" in line


def test_release_guide_documents_fail_closed_order_and_rollback() -> None:
    assert RELEASE_GUIDE_PATH.is_file()
    guide = RELEASE_GUIDE_PATH.read_text(encoding="utf-8")

    for required in (
        "voice-aec-v<app-version>",
        "matrix workflow-run ID",
        "GitHub OIDC",
        "Sigstore",
        "Trusted Publishing",
        "pypi-release",
        "SBOM",
        "THIRD_PARTY_NOTICES.md",
        "PATENTS",
        "PYBIND11_LICENSE.txt",
        "pybind11 3.1.0",
        "speech_recording",
        "legacy",
        "never rebuild",
    ):
        assert required in guide
