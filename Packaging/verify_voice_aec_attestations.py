#!/usr/bin/env python3
"""Verify the closed content and provenance contract of a voice-AEC release bundle.

Cryptographic Sigstore and GitHub attestation verification is deliberately performed
by their official verifiers in the release workflow. This offline, standard-library
checker validates the already-verified statements and binds their subjects to the exact
qualified bytes, repository, workflow identity, source revision, and source-tree digest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native" / "voice_aec"
STRUCTURAL_ROOT_FILES = {
    "SHA256SUMS",
    "source-tree.sha256",
    "THIRD_PARTY_NOTICES.md",
    "PYBIND11_LICENSE.txt",
    "LICENSE",
    "PATENTS",
    "license-inventory.json",
    "sbom.spdx.json",
}
SIGNED_ROOT_FILES = {
    "github-build-provenance.jsonl",
}
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  dist/([^/]+)$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
WEBRTC_COMMIT = "109e23c9cec3a44e67c08774874a409741b1e58a"
ABSEIL_COMMIT = "ac875ae5393d0516243cfd5d078cd4b098388f6b"
PYBIND11_VERSION = "3.1.0"
PYBIND11_SOURCE_URL = "https://github.com/pybind/pybind11/tree/v3.1.0"
PYTHON_TAGS = {"cp311", "cp312", "cp313"}
PLATFORMS = {
    "macos-x86_64",
    "macos-arm64",
    "windows-x86_64",
    "linux-x86_64",
    "linux-aarch64",
}
EXPECTED_WHEEL_TARGETS = {
    (python_tag, platform) for python_tag in PYTHON_TAGS for platform in PLATFORMS
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_version(filename: str) -> str | None:
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) == 5 and parts[0] == "tldw_voice_aec":
            return parts[1]
    match = re.match(r"^tldw_voice_aec-([0-9][A-Za-z0-9_.!+-]*)\.tar\.gz$", filename)
    return match.group(1) if match else None


def _wheel_target(filename: str) -> tuple[str, str] | None:
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) != 5:
        return None
    _, _, python_tag, abi_tag, platform_tag = parts
    if python_tag not in PYTHON_TAGS or abi_tag != python_tag:
        return None
    if platform_tag.startswith("macosx_") and platform_tag.endswith("_x86_64"):
        platform = "macos-x86_64"
    elif platform_tag.startswith("macosx_") and platform_tag.endswith("_arm64"):
        platform = "macos-arm64"
    elif platform_tag == "win_amd64":
        platform = "windows-x86_64"
    elif "linux" in platform_tag and platform_tag.endswith("_x86_64"):
        platform = "linux-x86_64"
    elif "linux" in platform_tag and platform_tag.endswith("_aarch64"):
        platform = "linux-aarch64"
    else:
        return None
    return python_tag, platform


def _load_json(path: Path, label: str, errors: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid {label}: {error}")
        return None


def _signature_bundle(
    bundle_path: Path, statement_bytes: bytes, errors: list[str]
) -> None:
    bundle = _load_json(bundle_path, f"attestation {bundle_path.name}", errors)
    if not isinstance(bundle, dict):
        return
    signature = bundle.get("messageSignature")
    verification = bundle.get("verificationMaterial")
    if not isinstance(signature, dict) or not isinstance(verification, dict):
        errors.append(
            f"attestation {bundle_path.name} is not a complete Sigstore bundle"
        )
        return
    if not signature.get("signature"):
        errors.append(f"attestation {bundle_path.name} has no signature")
    if not (
        verification.get("certificate") or verification.get("x509CertificateChain")
    ):
        errors.append(f"attestation {bundle_path.name} has no signing certificate")
    if not verification.get("tlogEntries"):
        errors.append(f"attestation {bundle_path.name} has no transparency-log entry")
    message_digest = signature.get("messageDigest")
    expected_digest = base64.b64encode(
        hashlib.sha256(statement_bytes).digest()
    ).decode()
    if not isinstance(message_digest, dict) or (
        message_digest.get("algorithm") != "SHA2_256"
        or message_digest.get("digest") != expected_digest
    ):
        errors.append(
            f"attestation {bundle_path.name} does not bind the provenance statement"
        )


def _statement(statement_path: Path, errors: list[str]) -> dict[str, object] | None:
    try:
        statement = json.loads(statement_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid provenance statement {statement_path.name}: {error}")
        return None
    if not isinstance(statement, dict):
        errors.append(f"provenance statement {statement_path.name} must be an object")
        return None
    return statement


def _verify_attestation(
    statement_path: Path,
    bundle_path: Path,
    *,
    filename: str,
    digest: str,
    repository: str,
    workflow_identity: str,
    source_sha: str,
    source_tree_digest: str,
    workflow_ref: str,
    workflow_path: str,
    event_name: str,
    repository_id: str,
    repository_owner_id: str,
    runner_environment: str,
    run_id: str,
    run_attempt: str,
) -> list[str]:
    errors: list[str] = []
    try:
        statement_bytes = statement_path.read_bytes()
    except OSError as error:
        return [f"cannot read provenance statement {statement_path.name}: {error}"]
    _signature_bundle(bundle_path, statement_bytes, errors)
    statement = _statement(statement_path, errors)
    if statement is None:
        return errors
    if statement.get("_type") != "https://in-toto.io/Statement/v1":
        errors.append(
            f"provenance statement {statement_path.name} has the wrong in-toto type"
        )
    if statement.get("predicateType") != "https://slsa.dev/provenance/v1":
        errors.append(
            f"provenance statement {statement_path.name} is not SLSA provenance"
        )
    expected_subject = [{"name": f"dist/{filename}", "digest": {"sha256": digest}}]
    if statement.get("subject") != expected_subject:
        errors.append(
            f"provenance statement {statement_path.name} subject does not match qualified bytes"
        )

    predicate = statement.get("predicate")
    build_definition = (
        predicate.get("buildDefinition") if isinstance(predicate, dict) else None
    )
    external_parameters = (
        build_definition.get("externalParameters")
        if isinstance(build_definition, dict)
        else None
    )
    internal_parameters = (
        build_definition.get("internalParameters")
        if isinstance(build_definition, dict)
        else None
    )
    resolved_dependencies = (
        build_definition.get("resolvedDependencies")
        if isinstance(build_definition, dict)
        else None
    )
    run_details = predicate.get("runDetails") if isinstance(predicate, dict) else None
    builder = run_details.get("builder") if isinstance(run_details, dict) else None
    metadata = run_details.get("metadata") if isinstance(run_details, dict) else None
    repository_url = f"https://github.com/{repository}"
    expected_external_parameters = {
        "workflow": {
            "ref": workflow_ref,
            "repository": repository_url,
            "path": workflow_path,
        }
    }
    expected_internal_parameters = {
        "github": {
            "event_name": event_name,
            "repository_id": repository_id,
            "repository_owner_id": repository_owner_id,
            "runner_environment": runner_environment,
        }
    }
    expected_resolved_dependencies = [
        {
            "uri": f"git+{repository_url}@{workflow_ref}",
            "digest": {"gitCommit": source_sha},
        }
    ]
    expected_invocation_id = (
        f"{repository_url}/actions/runs/{run_id}/attempts/{run_attempt}"
    )
    if (
        not isinstance(build_definition, dict)
        or build_definition.get("buildType")
        != "https://actions.github.io/buildtypes/workflow/v1"
    ):
        errors.append(f"provenance statement {statement_path.name} build type mismatch")
    if not isinstance(external_parameters, dict):
        errors.append(
            f"provenance statement {statement_path.name} has no external parameters"
        )
    else:
        workflow = external_parameters.get("workflow")
        workflow = workflow if isinstance(workflow, dict) else {}
        for key, expected in expected_external_parameters["workflow"].items():
            if workflow.get(key) != expected:
                errors.append(
                    f"provenance statement {statement_path.name} workflow {key} mismatch"
                )
        if external_parameters != expected_external_parameters:
            errors.append(
                f"provenance statement {statement_path.name} external parameters mismatch"
            )
    if not isinstance(internal_parameters, dict):
        errors.append(
            f"provenance statement {statement_path.name} has no internal parameters"
        )
    else:
        github_parameters = internal_parameters.get("github")
        github_parameters = (
            github_parameters if isinstance(github_parameters, dict) else {}
        )
        for key, expected in expected_internal_parameters["github"].items():
            if github_parameters.get(key) != expected:
                label = key.replace("_", " ")
                errors.append(
                    f"provenance statement {statement_path.name} {label} mismatch"
                )
        if internal_parameters != expected_internal_parameters:
            errors.append(
                f"provenance statement {statement_path.name} internal parameters mismatch"
            )
    if resolved_dependencies != expected_resolved_dependencies:
        errors.append(
            f"provenance statement {statement_path.name} resolved dependencies mismatch"
        )
        dependencies = (
            resolved_dependencies if isinstance(resolved_dependencies, list) else []
        )
        resolved = dependencies[0] if len(dependencies) == 1 else {}
        digest_value = resolved.get("digest") if isinstance(resolved, dict) else None
        if not isinstance(digest_value, dict) or (
            digest_value.get("gitCommit") != source_sha
        ):
            errors.append(
                f"provenance statement {statement_path.name} source sha mismatch"
            )
    if not isinstance(predicate, dict) or (
        predicate.get("tldw_source_tree_digest") != source_tree_digest
    ):
        errors.append(
            f"provenance statement {statement_path.name} source tree digest mismatch"
        )
    if not isinstance(builder, dict) or builder.get("id") != workflow_identity:
        errors.append(
            f"provenance statement {statement_path.name} builder workflow identity mismatch"
        )
    if not isinstance(metadata, dict) or (
        metadata.get("invocationId") != expected_invocation_id
    ):
        errors.append(
            f"provenance statement {statement_path.name} invocation id mismatch"
        )
    return errors


def verify_release_bundle(
    bundle_root: Path,
    *,
    repository: str,
    workflow_identity: str,
    source_sha: str,
    source_tree_digest: str,
    expected_version: str,
    workflow_ref: str,
    workflow_path: str,
    event_name: str,
    repository_id: str,
    repository_owner_id: str,
    runner_environment: str,
    run_id: str,
    run_attempt: str,
    structural_only: bool = False,
) -> list[str]:
    """Return violations in one already-cryptographically-verified bundle."""
    errors: list[str] = []
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        return [f"release bundle is not a regular directory: {bundle_root}"]
    if not GIT_SHA.fullmatch(source_sha):
        errors.append("source SHA must be a lowercase 40-character Git object ID")
    if not SHA256.fullmatch(source_tree_digest):
        errors.append("source-tree digest must be a lowercase SHA-256")
    if not structural_only:
        if event_name not in {"push", "workflow_dispatch"}:
            errors.append("signed release evidence must come from a trusted event")
        if workflow_ref != "refs/heads/main":
            errors.append("signed release evidence must come from refs/heads/main")
        if runner_environment != "github-hosted":
            errors.append(
                "signed release evidence must come from a GitHub-hosted runner"
            )
        stable_ids = {
            "repository ID": repository_id,
            "repository owner ID": repository_owner_id,
            "run ID": run_id,
            "run attempt": run_attempt,
        }
        for label, value in stable_ids.items():
            if not value.isdigit() or int(value) < 1:
                errors.append(f"{label} must be a positive decimal integer")

    entries = list(bundle_root.rglob("*"))
    symlinks = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in entries
        if path.is_symlink()
    )
    if symlinks:
        errors.append(f"release bundle contains symlinks: {symlinks}")
    files = {
        path.relative_to(bundle_root).as_posix()
        for path in entries
        if path.is_file() and not path.is_symlink()
    }
    dist_names = sorted(
        PurePosixPath(name).name
        for name in files
        if PurePosixPath(name).parent.as_posix() == "dist"
    )
    if not dist_names:
        errors.append("release bundle contains no distributions")
    for filename in dist_names:
        version = _distribution_version(filename)
        if version != expected_version:
            errors.append(
                f"distribution {filename} does not encode expected version {expected_version}"
            )
    sdists = [name for name in dist_names if name.endswith(".tar.gz")]
    if len(sdists) != 1:
        errors.append(f"release bundle must contain exactly one sdist; found {sdists}")
    if not any(name.endswith(".whl") for name in dist_names):
        errors.append("release bundle contains no wheels")
    wheel_names = [name for name in dist_names if name.endswith(".whl")]
    wheel_targets = [_wheel_target(name) for name in wheel_names]
    missing_targets = sorted(EXPECTED_WHEEL_TARGETS - set(wheel_targets))
    if missing_targets:
        errors.append(f"missing supported wheel targets: {missing_targets}")
    if None in wheel_targets or len(wheel_targets) != len(set(wheel_targets)):
        errors.append(
            "wheel set contains an unsupported tag or duplicate Python/platform target"
        )

    expected_files = STRUCTURAL_ROOT_FILES | {f"dist/{name}" for name in dist_names}
    if not structural_only:
        expected_files |= SIGNED_ROOT_FILES
        expected_files |= {
            f"attestations/{name}.provenance.json" for name in dist_names
        }
        expected_files |= {
            f"attestations/{name}.provenance.json.sigstore.json" for name in dist_names
        }
    extras = sorted(files - expected_files)
    missing = sorted(expected_files - files)
    if extras:
        errors.append(f"unexpected bundle files: {extras}")
    if missing:
        errors.append(f"missing bundle files: {missing}")

    sums: dict[str, str] = {}
    sums_path = bundle_root / "SHA256SUMS"
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"cannot read SHA256SUMS: {error}")
        lines = []
    for line in lines:
        match = SHA256_LINE.fullmatch(line)
        if not match:
            errors.append(f"malformed SHA256SUMS line: {line!r}")
            continue
        digest, filename = match.groups()
        if filename in sums:
            errors.append(f"duplicate SHA256SUMS subject: {filename}")
        sums[filename] = digest
    if set(sums) != set(dist_names):
        errors.append(
            "SHA256SUMS distribution set mismatch: "
            f"expected {dist_names}, found {sorted(sums)}"
        )
    for filename, expected_digest in sums.items():
        distribution = bundle_root / "dist" / filename
        if distribution.is_file() and _sha256(distribution) != expected_digest:
            errors.append(f"SHA-256 mismatch for dist/{filename}")

    try:
        source_line = (
            (bundle_root / "source-tree.sha256").read_text(encoding="utf-8").strip()
        )
    except OSError as error:
        errors.append(f"cannot read source-tree.sha256: {error}")
    else:
        if source_line != f"{source_tree_digest}  native/voice_aec":
            errors.append("source-tree digest mismatch")

    canonical = {
        "THIRD_PARTY_NOTICES.md": NATIVE_ROOT / "THIRD_PARTY_NOTICES.md",
        "PYBIND11_LICENSE.txt": NATIVE_ROOT / "PYBIND11_LICENSE.txt",
        "LICENSE": NATIVE_ROOT / "vendor" / "webrtc" / "LICENSE",
        "PATENTS": NATIVE_ROOT / "vendor" / "webrtc" / "PATENTS",
    }
    for name, source in canonical.items():
        bundled = bundle_root / name
        if (
            source.is_file()
            and bundled.is_file()
            and bundled.read_bytes() != source.read_bytes()
        ):
            errors.append(f"bundled {name} differs from reviewed source")

    github_bundle = bundle_root / "github-build-provenance.jsonl"
    if github_bundle.is_file() and not github_bundle.read_bytes().strip():
        errors.append("GitHub build-provenance bundle is empty")

    inventory = _load_json(
        bundle_root / "license-inventory.json", "license inventory", errors
    )
    if not isinstance(inventory, dict) or inventory.get("distributions") != dist_names:
        errors.append("license inventory does not cover the exact distribution set")
    else:
        expected_inventory_licenses = {
            "WebRTC BSD-3-Clause",
            "WebRTC PATENTS",
            "Abseil Apache-2.0",
            "Ooura FFT LicenseRef-Ooura-FFT",
            "pybind11 BSD-3-Clause",
        }
        inventory_licenses = inventory.get("licenses")
        if (
            not isinstance(inventory_licenses, list)
            or len(inventory_licenses) != len(set(inventory_licenses))
            or set(inventory_licenses) != expected_inventory_licenses
        ):
            errors.append(
                "license inventory notices do not match the reviewed native licenses"
            )
        expected_inventory_dependencies = {
            "WebRTC AEC3": {
                "name": "WebRTC AEC3",
                "version": WEBRTC_COMMIT,
                "license": "BSD-3-Clause",
                "source_url": "https://webrtc.googlesource.com/src",
            },
            "Abseil": {
                "name": "Abseil",
                "version": ABSEIL_COMMIT,
                "license": "Apache-2.0",
                "source_url": (
                    "https://chromium.googlesource.com/chromium/src/third_party"
                ),
            },
            "Ooura FFT": {
                "name": "Ooura FFT",
                "version": "NOASSERTION",
                "license": "LicenseRef-Ooura-FFT",
                "source_url": "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html",
            },
            "pybind11": {
                "name": "pybind11",
                "version": PYBIND11_VERSION,
                "license": "BSD-3-Clause",
                "source_url": PYBIND11_SOURCE_URL,
            },
        }
        dependencies = inventory.get("dependencies")
        dependency_entries = (
            {
                item.get("name"): item
                for item in dependencies
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            if isinstance(dependencies, list)
            else {}
        )
        if (
            not isinstance(dependencies, list)
            or len(dependencies) != len(dependency_entries)
            or len(dependency_entries) != len(expected_inventory_dependencies)
            or dependency_entries != expected_inventory_dependencies
        ):
            errors.append(
                "license inventory dependency metadata does not match reviewed "
                "native pins"
            )
        inventory_relationships = inventory.get("relationships")
        relationship_keys = (
            {
                (
                    item.get("distribution"),
                    item.get("relationship"),
                    item.get("dependency"),
                )
                for item in inventory_relationships
                if isinstance(item, dict)
            }
            if isinstance(inventory_relationships, list)
            else set()
        )
        expected_inventory_relationships = {
            (distribution, "DEPENDS_ON", dependency)
            for distribution in dist_names
            for dependency in expected_inventory_dependencies
        }
        if (
            not isinstance(inventory_relationships, list)
            or len(inventory_relationships) != len(relationship_keys)
            or relationship_keys != expected_inventory_relationships
        ):
            errors.append(
                "license inventory relationships do not bind every distribution "
                "to every reviewed native dependency"
            )

    sbom = _load_json(bundle_root / "sbom.spdx.json", "SBOM", errors)
    expected_extracted_licenses = [
        {
            "licenseId": "LicenseRef-Ooura-FFT",
            "extractedText": (
                NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE"
            ).read_text(encoding="utf-8"),
        }
    ]
    if not isinstance(sbom, dict) or (
        sbom.get("hasExtractedLicensingInfos") != expected_extracted_licenses
    ):
        errors.append(
            "SBOM Ooura extracted license must exactly match the reviewed OOURA_LICENSE"
        )
    packages = sbom.get("packages") if isinstance(sbom, dict) else None
    sbom_names = (
        {item.get("name") for item in packages if isinstance(item, dict)}
        if isinstance(packages, list)
        else set()
    )
    required_sbom_names = set(dist_names) | {
        "WebRTC AEC3",
        "Abseil",
        "Ooura FFT",
        "pybind11",
    }
    if not required_sbom_names <= sbom_names:
        errors.append("SBOM does not cover the exact distribution set")
    else:
        package_entries = {
            name: [
                item
                for item in packages
                if isinstance(item, dict) and item.get("name") == name
            ]
            for name in required_sbom_names
        }
        if any(len(entries) != 1 for entries in package_entries.values()):
            errors.append("SBOM subjects must each have exactly one package entry")
        package_by_name = {
            name: entries[0]
            for name, entries in package_entries.items()
            if len(entries) == 1
        }
        for filename in dist_names:
            package = package_by_name.get(filename, {})
            if package.get("checksums") != [
                {"algorithm": "SHA256", "checksumValue": sums.get(filename)}
            ]:
                errors.append(
                    f"SBOM distribution checksum does not match dist/{filename}"
                )
        dependency_metadata = {
            "WebRTC AEC3": {
                "SPDXID": "SPDXRef-WebRTC-AEC3",
                "versionInfo": WEBRTC_COMMIT,
                "licenseConcluded": "BSD-3-Clause",
            },
            "Abseil": {
                "SPDXID": "SPDXRef-Abseil",
                "versionInfo": ABSEIL_COMMIT,
                "licenseConcluded": "Apache-2.0",
            },
            "Ooura FFT": {
                "SPDXID": "SPDXRef-Ooura-FFT",
                "versionInfo": "NOASSERTION",
                "licenseConcluded": "LicenseRef-Ooura-FFT",
            },
            "pybind11": {
                "SPDXID": "SPDXRef-pybind11",
                "versionInfo": PYBIND11_VERSION,
                "downloadLocation": PYBIND11_SOURCE_URL,
                "licenseConcluded": "BSD-3-Clause",
            },
        }
        if any(
            any(
                package_by_name.get(name, {}).get(key) != value
                for key, value in expected.items()
            )
            for name, expected in dependency_metadata.items()
        ):
            errors.append(
                "SBOM dependency metadata does not match reviewed native pins"
            )
        package_ids = {
            name: package.get("SPDXID") for name, package in package_by_name.items()
        }
        relationships = sbom.get("relationships") if isinstance(sbom, dict) else None
        relationship_keys = (
            {
                (
                    item.get("spdxElementId"),
                    item.get("relationshipType"),
                    item.get("relatedSpdxElement"),
                )
                for item in relationships
                if isinstance(item, dict)
            }
            if isinstance(relationships, list)
            else set()
        )
        dependency_names = ("WebRTC AEC3", "Abseil", "Ooura FFT", "pybind11")
        dependency_ids = {package_ids.get(name) for name in dependency_names}
        expected_relationships = {
            (package_ids.get(name), "DEPENDS_ON", dependency_id)
            for name in dist_names
            for dependency_id in dependency_ids
        }
        if (
            None in package_ids.values()
            or not expected_relationships <= relationship_keys
        ):
            errors.append(
                "SBOM dependency relationships do not bind every distribution to "
                "WebRTC AEC3, Abseil, Ooura FFT, and pybind11"
            )

    if not structural_only:
        for filename in dist_names:
            digest = sums.get(filename)
            statement = bundle_root / "attestations" / f"{filename}.provenance.json"
            attestation = (
                bundle_root
                / "attestations"
                / (f"{filename}.provenance.json.sigstore.json")
            )
            if not statement.is_file() or not attestation.is_file():
                errors.append(f"missing attestation for {filename}")
            elif digest:
                errors.extend(
                    _verify_attestation(
                        statement,
                        attestation,
                        filename=filename,
                        digest=digest,
                        repository=repository,
                        workflow_identity=workflow_identity,
                        source_sha=source_sha,
                        source_tree_digest=source_tree_digest,
                        workflow_ref=workflow_ref,
                        workflow_path=workflow_path,
                        event_name=event_name,
                        repository_id=repository_id,
                        repository_owner_id=repository_owner_id,
                        runner_environment=runner_environment,
                        run_id=run_id,
                        run_attempt=run_attempt,
                    )
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Verify one qualified release bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-identity", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--workflow-ref", required=True)
    parser.add_argument("--workflow-path", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--repository-owner-id", required=True)
    parser.add_argument("--runner-environment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="require the unsigned structural contract and reject signed-evidence files",
    )
    args = parser.parse_args(argv)
    errors = verify_release_bundle(
        args.bundle.resolve(),
        repository=args.repository,
        workflow_identity=args.workflow_identity,
        source_sha=args.source_sha,
        source_tree_digest=args.source_tree_digest,
        expected_version=args.expected_version,
        workflow_ref=args.workflow_ref,
        workflow_path=args.workflow_path,
        event_name=args.event_name,
        repository_id=args.repository_id,
        repository_owner_id=args.repository_owner_id,
        runner_environment=args.runner_environment,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        structural_only=args.structural_only,
    )
    if errors:
        for error in errors:
            print(f"voice AEC attestation error: {error}", file=sys.stderr)
        return 1
    if args.structural_only:
        print("voice AEC bundle is qualified for unsigned structural review only")
    else:
        print("voice AEC release bundle is structurally and semantically qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
