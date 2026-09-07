"""Explicit, verified installer for the curated Parakeet v2 INT8 bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen as _open_url


PARAKEET_V2_REPOSITORY = "istupakov/parakeet-tdt-0.6b-v2-onnx"
PARAKEET_V2_REVISION = "0bbb45a3365852604aef28b538a8f066f4ccaa85"
PARAKEET_V2_LICENSE = "CC-BY-4.0"
VERIFICATION_RECEIPT = ".tldw-verified.json"
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_FREE_SPACE_HEADROOM_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class BundleFile:
    """One immutable file in the curated bundle."""

    filename: str
    size_bytes: int
    sha256: str


PARAKEET_V2_FILES = (
    BundleFile(
        "config.json",
        97,
        "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
    ),
    BundleFile(
        "vocab.txt",
        9_384,
        "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
    ),
    BundleFile(
        "encoder-model.int8.onnx",
        652_184_014,
        "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155",
    ),
    BundleFile(
        "decoder_joint-model.int8.onnx",
        8_998_286,
        "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3",
    ),
)
PARAKEET_V2_TOTAL_BYTES = sum(file.size_bytes for file in PARAKEET_V2_FILES)


class ParakeetInstallError(RuntimeError):
    """Raised when the curated bundle cannot be installed safely."""


def parakeet_v2_install_dir() -> Path:
    """Return the immutable default installation directory."""
    from tldw_chatbook.Utils.paths import get_user_data_dir

    return (
        get_user_data_dir()
        / "models"
        / "stt"
        / f"parakeet-v2-int8-{PARAKEET_V2_REVISION[:12]}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parakeet_v2_bundle(directory: str | Path) -> bool:
    """Return whether ``directory`` is the complete curated bundle."""
    root = Path(directory)
    try:
        receipt_path = root / VERIFICATION_RECEIPT
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return False
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("repository") != PARAKEET_V2_REPOSITORY
            or receipt.get("revision") != PARAKEET_V2_REVISION
        ):
            return False
        for descriptor in PARAKEET_V2_FILES:
            path = root / descriptor.filename
            if path.is_symlink() or not path.is_file():
                return False
            if path.stat().st_size != descriptor.size_bytes:
                return False
            if _sha256(path) != descriptor.sha256:
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _source_url(filename: str) -> str:
    return (
        f"https://huggingface.co/{PARAKEET_V2_REPOSITORY}/resolve/"
        f"{PARAKEET_V2_REVISION}/{filename}"
    )


def install_verified_parakeet_v2(
    *,
    destination: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Download, verify, and atomically publish the curated v2 INT8 bundle."""
    target = Path(destination) if destination is not None else parakeet_v2_install_dir()
    if verify_parakeet_v2_bundle(target):
        return target
    if target.exists():
        raise ParakeetInstallError(
            f"Install destination exists but is not the verified bundle: {target}"
        )

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(file.size_bytes for file in PARAKEET_V2_FILES)
    if shutil.disk_usage(parent).free < total_bytes + _FREE_SPACE_HEADROOM_BYTES:
        raise ParakeetInstallError(
            f"Not enough free space to install {total_bytes:,} bytes at {parent}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=".parakeet-v2-install-", dir=str(parent))
    )
    downloaded = 0
    try:
        for descriptor in PARAKEET_V2_FILES:
            if Path(descriptor.filename).name != descriptor.filename:
                raise ParakeetInstallError("Invalid curated bundle filename")
            output_path = staging / descriptor.filename
            digest = hashlib.sha256()
            file_bytes = 0
            request = Request(
                _source_url(descriptor.filename),
                headers={"User-Agent": "tldw-chatbook-parakeet-installer/1"},
            )
            with _open_url(request, timeout=30) as response, output_path.open(
                "xb"
            ) as output:
                while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                    file_bytes += len(chunk)
                    if file_bytes > descriptor.size_bytes:
                        raise ParakeetInstallError(
                            f"{descriptor.filename} verification failed: "
                            "download exceeded the expected size"
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total_bytes, descriptor.filename)
            if (
                file_bytes != descriptor.size_bytes
                or digest.hexdigest() != descriptor.sha256
            ):
                raise ParakeetInstallError(
                    f"{descriptor.filename} verification failed: "
                    "size or SHA-256 did not match"
                )

        receipt = {
            "schema_version": 1,
            "repository": PARAKEET_V2_REPOSITORY,
            "revision": PARAKEET_V2_REVISION,
            "license": PARAKEET_V2_LICENSE,
            "files": [asdict(file) for file in PARAKEET_V2_FILES],
        }
        (staging / VERIFICATION_RECEIPT).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not verify_parakeet_v2_bundle(staging):
            raise ParakeetInstallError("Staged Parakeet v2 bundle verification failed")
        try:
            staging.rename(target)
        except FileExistsError:
            if not verify_parakeet_v2_bundle(target):
                raise ParakeetInstallError(
                    f"Install destination changed during installation: {target}"
                )
        return target
    except ParakeetInstallError:
        raise
    except Exception as exc:
        raise ParakeetInstallError(f"Parakeet v2 install failed: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
