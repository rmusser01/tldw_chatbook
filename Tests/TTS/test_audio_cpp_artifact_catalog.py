"""Boundaries for the pinned audio.cpp artifact-source manifest."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.request

import pytest


REPOSITORY = "audio-cpp/audio.cpp-gguf"
COMMIT = "597048d9a920592808d7d4e2acd7b9c4596a143a"
EXPECTED_MANIFEST_SHA256 = (
    "3692e9174f0cb132115a78e03357bbd10783e3b6c81b0a285c8c51cf193c5d4f"
)
TREE_URL = (
    f"https://huggingface.co/api/models/{REPOSITORY}/tree/{COMMIT}"
    "?recursive=true&expand=true"
)


def _manifest_refresh_docstring(function_name: str) -> str:
    script_path = (
        Path(__file__).parents[2] / "scripts" / "refresh_audio_cpp_artifact_manifest.py"
    )
    module = ast.parse(script_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return ast.get_docstring(functions[function_name]) or ""


def _valid_manifest() -> dict[str, Any]:
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "packages": [
            {
                "recipe_id": "audio-cpp-0.5.1.example",
                "recipe_revision": 1,
                "package_variant": "example_q8_0",
                "artifact_id": "audio-cpp-example-q8-0",
                "license_id": "Apache-2.0",
                "license_url": "https://example.invalid/LICENSE",
                "usage_notice": "Review the upstream model license before use.",
                "files": [
                    {
                        "source_path": "example/model.gguf",
                        "managed_path": "model.gguf",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checked_in_manifest_is_exact_pinned_reviewed_catalog_and_network_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime manifest loading touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    catalog = load_audio_cpp_artifact_source_manifest()

    assert catalog.repository == REPOSITORY
    assert catalog.commit == COMMIT
    assert len(catalog.packages) == 45
    assert "supertonic_3_q8_0" not in {
        package.package_variant for package in catalog.packages
    }
    manifest_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "TTS"
        / "audio_cpp_artifact_manifest.json"
    )
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST_SHA256
    raw = json.loads(manifest_bytes)
    assert raw["repository"] == catalog.repository
    assert raw["commit"] == catalog.commit
    assert len(raw["packages"]) == len(catalog.packages)
    assert raw["packages"] == sorted(
        raw["packages"],
        key=lambda package: (
            package["recipe_id"],
            package["recipe_revision"],
            package["package_variant"],
        ),
    )
    reviewed = {package["package_variant"]: package for package in raw["packages"]}
    assert reviewed["dramabox_q8_0"] == {
        "artifact_id": "audio-cpp-dramabox-q8-0",
        "files": [
            {
                "managed_path": "dramabox-q8_0.gguf",
                "sha256": (
                    "75e7e80fc748defb188cb902c34c62bc12539a7bba477215dccf59a7218a451e"
                ),
                "size_bytes": 18_942_803_808,
                "source_path": "DramaBox-GGUF/dramabox-q8_0.gguf",
            }
        ],
        "license_id": "LTX-2 Community License",
        "license_url": (f"https://huggingface.co/{REPOSITORY}/blob/{COMMIT}/README.md"),
        "package_variant": "dramabox_q8_0",
        "recipe_id": "audio-cpp-0.5.1.dramabox.dramabox_q8_0",
        "recipe_revision": 1,
        "usage_notice": (
            "Converted weights are provided as-is; review the original model license "
            "and validate the exact file, backend, and route before use."
        ),
    }
    assert reviewed["pocket_tts_english_q8_0"]["files"] == [
        {
            "managed_path": "pocket-tts-english-q8_0.gguf",
            "sha256": (
                "0315406421d515d9ffbde49ed998832ff2962562ef8abde440c85fa0a27d8b2a"
            ),
            "size_bytes": 127_856_704,
            "source_path": "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
        }
    ]
    assert reviewed["supertonic_3_orig"]["files"] == [
        {
            "managed_path": "supertonic-3-orig.gguf",
            "sha256": (
                "af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662"
            ),
            "size_bytes": 454_072_836,
            "source_path": "Supertonic-3-GGUF/supertonic-3-orig.gguf",
        }
    ]


def _reviewed_entries():
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
        load_audio_cpp_artifact_source_manifest,
    )

    manifest = load_audio_cpp_artifact_source_manifest()
    assert manifest.packages, "the reviewed pinned package list is still empty"
    return manifest, audio_cpp_curated_entries()


def test_audio_cpp_curated_entries_are_exact_recipe_joins() -> None:
    from urllib.parse import quote

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    manifest, entries = _reviewed_entries()
    packages = {package.key: package for package in manifest.packages}
    recipes = {
        recipe.model_library_artifact_ids[0]: recipe
        for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
        if recipe.model_library_artifact_ids
    }

    assert len(entries) == len(packages) == 45
    for descriptor, sources in entries:
        recipe = recipes[descriptor.reference.artifact_id]
        package = packages[
            (recipe.recipe_id, recipe.recipe_revision, recipe.package_variant)
        ]
        expected_urls = {
            file.managed_path: (
                f"https://huggingface.co/{manifest.repository}/resolve/"
                f"{manifest.commit}/{quote(file.source_path, safe='/')}"
            )
            for file in package.files
        }
        expected_paths = {signal.relative_path for signal in recipe.required_files}

        assert recipe.model_library_artifact_ids == (package.artifact_id,)
        assert descriptor.reference == ArtifactRef(
            package.artifact_id,
            manifest.commit,
            recipe.precision,
        )
        assert descriptor.role is ArtifactRole.ROOT
        assert descriptor.format is ArtifactFormat.GGUF
        assert descriptor.consumer == "audio_cpp"
        assert descriptor.model_family == recipe.family
        assert descriptor.model_id == recipe.default_public_model_id
        assert descriptor.precision == recipe.precision
        assert descriptor.runtime_name == "audio.cpp"
        assert descriptor.runtime_version_constraint == (
            f"{recipe.audio_cpp_release}@{recipe.audio_cpp_commit}"
        )
        exposed_platforms = (
            set()
            if descriptor.supported_os
            == descriptor.supported_architectures
            == ("unassigned",)
            else {
                (system, architecture)
                for system in descriptor.supported_os
                for architecture in descriptor.supported_architectures
            }
        )
        assert exposed_platforms <= {
            (item.system, item.architecture) for item in recipe.backend_evidence
        }
        assert descriptor.supported_os == ("unassigned",)
        assert descriptor.supported_architectures == ("unassigned",)
        assert tuple(
            (
                item.system,
                item.architecture,
                item.backend.value,
                item.state.value,
                item.evidence_reference,
            )
            for item in recipe.backend_evidence
        ) == tuple(
            (
                system,
                architecture,
                "cpu",
                "expected",
                "backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md"
                "#decision",
            )
            for system, architecture in (
                ("darwin", "arm64"),
                ("darwin", "x86_64"),
                ("linux", "aarch64"),
                ("linux", "x86_64"),
                ("windows", "x86_64"),
            )
        )
        assert descriptor.provenance == (
            ProvenanceClass.CHATBOOK_CURATED,
            ProvenanceClass.INTEGRITY_VERIFIED,
        )
        assert {file.path for file in descriptor.files} == expected_paths
        assert set(sources) == {file.path for file in descriptor.files}
        assert sources == expected_urls
        assert descriptor.source_url == next(iter(expected_urls.values()))
        assert descriptor.expected_installed_bytes == sum(
            file.size_bytes for file in descriptor.files
        )
        assert descriptor.license_id == package.license_id
        assert descriptor.license_id != "other"
        assert (
            descriptor.license_url
            == package.license_url
            == (
                f"https://huggingface.co/{manifest.repository}/blob/"
                f"{manifest.commit}/README.md"
            )
        )
        assert descriptor.usage_notice == package.usage_notice
        assert descriptor.dependencies == ()
        assert set(recipe.capabilities) <= {"tts", "clone", "design"}
        assert not any(
            token in descriptor.model_family
            for token in ("asr", "diar", "music", "separation")
        )


@pytest.mark.parametrize(
    "drift",
    ["unknown_recipe", "unknown_revision", "duplicate_artifact_owner"],
)
def test_audio_cpp_join_rejects_manifest_recipe_drift(drift: str) -> None:
    from dataclasses import replace

    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
        load_audio_cpp_artifact_source_manifest,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AUDIO_CPP_RECIPE_REGISTRY,
        AudioCppRecipeRegistry,
    )

    manifest = load_audio_cpp_artifact_source_manifest()
    assert manifest.packages, "the reviewed pinned package list is still empty"
    joined_recipe_id = manifest.packages[0].recipe_id
    recipes = list(AUDIO_CPP_RECIPE_REGISTRY.recipes)
    if drift == "unknown_recipe":
        recipes = [recipe for recipe in recipes if recipe.recipe_id != joined_recipe_id]
    elif drift == "unknown_revision":
        recipes = [
            replace(recipe, recipe_revision=recipe.recipe_revision + 1)
            if recipe.recipe_id == joined_recipe_id
            else recipe
            for recipe in recipes
        ]
    else:
        joined_artifact_id = manifest.packages[0].artifact_id
        local_only = next(
            recipe for recipe in recipes if not recipe.model_library_artifact_ids
        )
        recipes = [
            replace(recipe, model_library_artifact_ids=(joined_artifact_id,))
            if recipe.recipe_id == local_only.recipe_id
            else recipe
            for recipe in recipes
        ]

    with pytest.raises(ValueError, match="manifest recipe"):
        audio_cpp_curated_entries(AudioCppRecipeRegistry(tuple(recipes)))


def _join_with_optional_file(
    monkeypatch: pytest.MonkeyPatch,
    *,
    present_path: str | None,
    declared_optional_path: str = "optional.json",
    present_size: int = 1,
    optional_minimum_size: int = 1,
):
    from dataclasses import replace

    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog_module
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AudioCppArtifactSourceFile,
        audio_cpp_curated_entries,
        load_audio_cpp_artifact_source_manifest,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AUDIO_CPP_RECIPE_REGISTRY,
        AudioCppFileKind,
        AudioCppFileRole,
        AudioCppFileSignal,
        AudioCppRecipeRegistry,
    )

    manifest = load_audio_cpp_artifact_source_manifest()
    package = manifest.packages[0]
    recipe = next(
        item
        for item in AUDIO_CPP_RECIPE_REGISTRY.recipes
        if item.recipe_id == package.recipe_id
    )
    optional = AudioCppFileSignal(
        declared_optional_path,
        AudioCppFileKind.JSON,
        AudioCppFileRole.OTHER,
        minimum_size_bytes=optional_minimum_size,
    )
    files = package.files
    if present_path is not None:
        files += (
            AudioCppArtifactSourceFile(
                source_path=f"optional/{present_path}",
                managed_path=present_path,
                size_bytes=present_size,
                sha256="e" * 64,
            ),
        )
    test_manifest = replace(
        manifest,
        packages=(replace(package, files=files),),
    )
    test_registry = AudioCppRecipeRegistry(
        (replace(recipe, optional_files=(optional,)),)
    )
    monkeypatch.setattr(
        catalog_module,
        "load_audio_cpp_artifact_source_manifest",
        lambda: test_manifest,
    )
    return audio_cpp_curated_entries(test_registry)


def test_audio_cpp_join_allows_declared_optional_file_to_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _join_with_optional_file(monkeypatch, present_path=None)

    assert "optional.json" not in {file.path for file in entries[0][0].files}


def test_audio_cpp_join_includes_every_present_declared_optional_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _join_with_optional_file(monkeypatch, present_path="optional.json")
    descriptor, sources = entries[0]

    assert "optional.json" in {file.path for file in descriptor.files}
    assert set(sources) == {file.path for file in descriptor.files}


def test_audio_cpp_join_rejects_unknown_extra_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="file closure"):
        _join_with_optional_file(monkeypatch, present_path="unknown.json")


def test_audio_cpp_join_checks_present_optional_file_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="file closure"):
        _join_with_optional_file(
            monkeypatch,
            present_path="optional.json",
            present_size=1,
            optional_minimum_size=2,
        )


def test_audio_cpp_release_rows_have_exactly_one_cross_axis_outcome() -> None:
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AUDIO_CPP_RELEASE_ACCOUNTING,
        AUDIO_CPP_RECIPE_REGISTRY,
        AudioCppRecipeSupportState,
    )

    manifest, entries = _reviewed_entries()
    admitted_variants = {package.package_variant for package in manifest.packages}
    assert admitted_variants == {
        recipe.package_variant
        for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
        if recipe.model_library_artifact_ids
    }
    assert len(entries) == len(admitted_variants)

    outcomes: dict[str, str] = {}
    for row in AUDIO_CPP_RELEASE_ACCOUNTING:
        assert row.package_variant not in outcomes
        if row.state is AudioCppRecipeSupportState.EXPLICITLY_UNSUPPORTED:
            outcome = "explicitly_unsupported"
        elif row.package_variant in admitted_variants:
            outcome = "downloadable"
        else:
            outcome = "local_only"
        outcomes[row.package_variant] = outcome

    assert len(outcomes) == 67
    assert list(outcomes.values()).count("downloadable") == 45
    assert list(outcomes.values()).count("local_only") == 8
    assert list(outcomes.values()).count("explicitly_unsupported") == 14
    assert set(outcomes.values()) == {
        "downloadable",
        "local_only",
        "explicitly_unsupported",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "audio-cpp/not-the-official-repository"),
        ("commit", "main"),
        ("commit", COMMIT.upper()),
        ("commit", "0" * 40),
    ],
)
def test_manifest_rejects_any_non_pinned_repository_or_commit(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_rejects_duplicate_package_keys(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    duplicate = copy.deepcopy(payload["packages"][0])
    duplicate["artifact_id"] = "different-artifact-id"
    payload["packages"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate package key"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("path_field", ["source_path", "managed_path"])
def test_manifest_rejects_duplicate_paths_within_a_package(
    tmp_path: Path,
    path_field: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    duplicate = copy.deepcopy(payload["packages"][0]["files"][0])
    other_field = "managed_path" if path_field == "source_path" else "source_path"
    duplicate[other_field] = f"different/{duplicate[other_field]}"
    payload["packages"][0]["files"].append(duplicate)

    with pytest.raises(ValueError, match=f"duplicate {path_field}"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("path_field", ["source_path", "managed_path"])
@pytest.mark.parametrize(
    "value",
    [
        "../model.gguf",
        "models/../model.gguf",
        "/model.gguf",
        "C:/escape.gguf",
        "C:escape.gguf",
    ],
)
def test_manifest_rejects_path_traversal_or_absolute_paths(
    tmp_path: Path,
    path_field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["files"][0][path_field] = value

    with pytest.raises(ValueError, match=path_field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    "value",
    [" model.gguf", "model.gguf ", "model$.gguf", "model%2e.gguf", "model\n.gguf"],
)
@pytest.mark.parametrize("path_field", ["source_path", "managed_path"])
def test_manifest_paths_match_recipe_safe_relative_path_rules(
    tmp_path: Path,
    path_field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["files"][0][path_field] = value

    with pytest.raises(ValueError, match=path_field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("file", "size_bytes"),
        ("file", "sha256"),
        ("package", "license_id"),
        ("package", "license_url"),
        ("package", "usage_notice"),
    ],
)
def test_manifest_rejects_missing_integrity_or_reviewed_license_facts(
    tmp_path: Path,
    scope: str,
    field: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    owner = (
        payload["packages"][0]["files"][0]
        if scope == "file"
        else payload["packages"][0]
    )
    del owner[field]

    with pytest.raises(ValueError, match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_revision", True),
        ("recipe_revision", 1.0),
        ("files", ()),
    ],
)
def test_manifest_uses_exact_json_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0][field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_id", "x" * 257),
        ("usage_notice", "é" * 2049),
    ],
)
def test_manifest_rejects_oversized_text_facts(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0][field] = value

    with pytest.raises(ValueError, match=field):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_rejects_oversized_path_bytes(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["files"][0]["source_path"] = "é" * 513

    with pytest.raises(ValueError, match="source_path"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    "value", ["notice\nline", "notice\x00line", "notice\u200bline"]
)
def test_manifest_rejects_control_characters_in_text(
    tmp_path: Path,
    value: str,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["usage_notice"] = value

    with pytest.raises(ValueError, match="usage_notice"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    "value",
    [
        "https://",
        "https:///LICENSE",
        "http://example.invalid/LICENSE",
        " https://example.invalid/LICENSE",
        "https://example.invalid/LICENSE\n",
        "https://example.invalid/LI\x00CENSE",
        "https://example.invalid\\evil/LICENSE",
        "https://example.invalid/LICENSE?token=secret",
        "https://example.invalid/LICENSE#fragment",
        "https://user:password@example.invalid/LICENSE",
        "https://:443/LICENSE",
        "https://example.invalid:invalid/LICENSE",
        "https://exa_mple.invalid/LICENSE",
        "https://example.invalid/not|canonical",
    ],
)
def test_manifest_rejects_malformed_license_urls(tmp_path: Path, value: str) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    payload["packages"][0]["license_url"] = value

    with pytest.raises(ValueError, match="license_url"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_accepts_canonical_credential_free_https_license_url(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()

    manifest = load_audio_cpp_artifact_source_manifest(
        _write_manifest(tmp_path, payload)
    )

    assert manifest.packages[0].license_url == "https://example.invalid/LICENSE"


def test_manifest_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    path = tmp_path / "manifest.json"
    path.write_text(
        '{"repository":"audio-cpp/audio.cpp-gguf",'
        '"repository":"audio-cpp/audio.cpp-gguf",'
        f'"commit":"{COMMIT}","packages":[]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_audio_cpp_artifact_source_manifest(path)


def test_manifest_rejects_non_utf8_json(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    path = tmp_path / "manifest.json"
    path.write_bytes(json.dumps(_valid_manifest()).encode("utf-16"))

    with pytest.raises(ValueError, match="UTF-8 JSON"):
        load_audio_cpp_artifact_source_manifest(path)


def test_manifest_loader_reads_bounded_bytes_before_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog_module

    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * 33)
    monkeypatch.setattr(catalog_module, "_MAX_MANIFEST_BYTES", 32)

    with pytest.raises(ValueError, match="manifest exceeds"):
        catalog_module.load_audio_cpp_artifact_source_manifest(path)


def test_manifest_rejects_more_than_67_packages(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    template = payload["packages"][0]
    payload["packages"] = []
    for index in range(68):
        package = copy.deepcopy(template)
        package["recipe_id"] = f"recipe-{index}"
        package["artifact_id"] = f"artifact-{index}"
        package["package_variant"] = f"variant-{index}"
        payload["packages"].append(package)

    with pytest.raises(ValueError, match="packages exceeds"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_rejects_more_than_256_files_in_one_package(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    template = payload["packages"][0]["files"][0]
    payload["packages"][0]["files"] = [
        {
            **template,
            "source_path": f"source/{index}.gguf",
            "managed_path": f"managed/{index}.gguf",
        }
        for index in range(257)
    ]

    with pytest.raises(ValueError, match="files exceeds"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


def test_manifest_rejects_more_than_4096_total_files(tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        load_audio_cpp_artifact_source_manifest,
    )

    payload = _valid_manifest()
    package_template = payload["packages"][0]
    file_template = package_template["files"][0]
    payload["packages"] = []
    for package_index in range(17):
        package = copy.deepcopy(package_template)
        package["recipe_id"] = f"recipe-{package_index}"
        package["artifact_id"] = f"artifact-{package_index}"
        package["package_variant"] = f"variant-{package_index}"
        package["files"] = [
            {
                **file_template,
                "source_path": f"source/{package_index}/{file_index}.gguf",
                "managed_path": f"managed/{package_index}/{file_index}.gguf",
            }
            for file_index in range(256)
        ]
        payload["packages"].append(package)

    with pytest.raises(ValueError, match="total files exceeds"):
        load_audio_cpp_artifact_source_manifest(_write_manifest(tmp_path, payload))


class _Response(io.BytesIO):
    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content)), **(headers or {})}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_refresh_is_immutable_bounded_and_byte_deterministic(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    git_content = b'{"x":1}\n'
    payload = _valid_manifest()
    payload["packages"][0]["files"] = [
        {
            "source_path": "example/model.gguf",
            "managed_path": "model.gguf",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        {
            "source_path": "example/config.json",
            "managed_path": "config.json",
            "size_bytes": 1,
            "sha256": "2" * 64,
        },
    ]
    manifest_path = _write_manifest(tmp_path, payload)
    first_page = json.dumps(
        [
            {
                "type": "file",
                "path": "example/config.json",
                "oid": "b" * 40,
                "size": len(git_content),
            },
        ]
    ).encode()
    second_page = json.dumps(
        [
            {
                "type": "file",
                "path": "example/model.gguf",
                "oid": "c" * 40,
                "size": 10,
                "lfs": {"oid": "d" * 64, "size": 10, "pointerSize": 127},
            },
        ]
    ).encode()
    page_two_url = TREE_URL + "&cursor=page2"
    opened_urls: list[str] = []

    def recorded_urlopen(request: object) -> _Response:
        url = getattr(request, "full_url", str(request))
        opened_urls.append(url)
        if url == TREE_URL:
            return _Response(
                first_page,
                {"Link": f'<{page_two_url}>; rel="next"'},
            )
        if url == page_two_url:
            return _Response(second_page)
        if url.endswith("/resolve/" + COMMIT + "/example/config.json"):
            return _Response(git_content)
        raise AssertionError(f"unexpected URL: {url}")

    first = refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)
    second = refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)

    assert first == second
    assert hashlib.sha256(first).hexdigest() == (
        "635055396c87170fa84395215db956b1e23e210e74b0ad461e54c4501bf12aa2"
    )
    assert b'"sha256": "' + hashlib.sha256(git_content).hexdigest().encode() in first
    assert b'"sha256": "' + b"d" * 64 in first
    assert json.loads(first)["packages"][0]["license_id"] == "Apache-2.0"
    assert opened_urls
    assert all(COMMIT in url for url in opened_urls)
    assert all("/main/" not in url and not url.endswith("/main") for url in opened_urls)


def test_refresh_follows_validated_second_tree_page(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    page_two_url = TREE_URL + "&cursor=page2"
    responses = {
        TREE_URL: _Response(b"[]", {"Link": f'<{page_two_url}>; rel="next"'}),
        page_two_url: _Response(
            json.dumps(
                [
                    {
                        "type": "file",
                        "path": "example/model.gguf",
                        "oid": "a" * 40,
                        "size": 10,
                        "lfs": {"oid": "b" * 64, "size": 10},
                    }
                ]
            ).encode()
        ),
    }
    opened: list[str] = []

    def recorded_urlopen(request: object) -> _Response:
        url = getattr(request, "full_url", str(request))
        opened.append(url)
        return responses[url]

    output = refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)

    assert json.loads(output)["packages"][0]["files"][0]["sha256"] == "b" * 64
    assert opened == [TREE_URL, page_two_url]


@pytest.mark.parametrize(
    "next_url",
    [
        "https://evil.invalid/api/models/audio-cpp/audio.cpp-gguf/tree/"
        + COMMIT
        + "?cursor=x",
        "https://huggingface.co/api/models/other/repository/tree/"
        + COMMIT
        + "?cursor=x",
        "https://huggingface.co/api/models/audio-cpp/audio.cpp-gguf/tree/"
        + "0" * 40
        + "?cursor=x",
    ],
)
def test_refresh_rejects_unsafe_pagination_links(
    tmp_path: Path,
    next_url: str,
) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(b"[]", {"Link": f'<{next_url}>; rel="next"'})

    with pytest.raises(ValueError, match="pagination"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


@pytest.mark.parametrize("link", ["not-a-link", '<https://huggingface.co>; rel="next'])
def test_refresh_rejects_malformed_pagination_links(
    tmp_path: Path,
    link: str,
) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(b"[]", {"Link": link})

    with pytest.raises(ValueError, match="pagination"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_rejects_pagination_cycles(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(b"[]", {"Link": f'<{TREE_URL}>; rel="next"'})

    with pytest.raises(ValueError, match="pagination cycle"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_rejects_duplicate_paths_across_pages(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    page_two_url = TREE_URL + "&cursor=page2"
    entry = {
        "type": "file",
        "path": "example/model.gguf",
        "oid": "a" * 40,
        "size": 10,
        "lfs": {"oid": "b" * 64, "size": 10},
    }

    def recorded_urlopen(request: object) -> _Response:
        url = getattr(request, "full_url", str(request))
        headers = {"Link": f'<{page_two_url}>; rel="next"'} if url == TREE_URL else {}
        return _Response(json.dumps([entry]).encode(), headers)

    with pytest.raises(ValueError, match="duplicate path"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_bounds_pagination_pages_and_aggregate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.refresh_audio_cpp_artifact_manifest as refresh_module

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    monkeypatch.setattr(refresh_module, "_MAX_TREE_PAGES", 1)

    def page_limited_urlopen(_request: object) -> _Response:
        return _Response(b"[]", {"Link": f'<{TREE_URL}&cursor=two>; rel="next"'})

    with pytest.raises(ValueError, match="page limit"):
        refresh_module.refresh_manifest_bytes(
            manifest_path, COMMIT, urlopen=page_limited_urlopen
        )

    monkeypatch.setattr(refresh_module, "_MAX_TREE_PAGES", 32)
    monkeypatch.setattr(refresh_module, "_MAX_TREE_TOTAL_BYTES", 1)

    def byte_limited_urlopen(_request: object) -> _Response:
        return _Response(b"[]")

    with pytest.raises(ValueError, match="aggregate byte limit"):
        refresh_module.refresh_manifest_bytes(
            manifest_path, COMMIT, urlopen=byte_limited_urlopen
        )


@pytest.mark.parametrize("header", ["exceeds", "-1", " 5", "True", "+5"])
def test_read_bounded_normalizes_invalid_content_length(header: str) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import _read_bounded

    response = _Response(b"data", {"Content-Length": header})

    with pytest.raises(ValueError, match="payload has an invalid Content-Length"):
        _read_bounded(response, 8, "payload")


@pytest.mark.parametrize("commit", ["main", "A" * 40, "0" * 39, "0" * 41])
def test_refresh_refuses_non_exact_immutable_commit(commit: str) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import validate_commit

    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        validate_commit(commit)


def test_refresh_refuses_unknown_requested_file_shapes(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    tree = json.dumps(
        [{"type": "directory", "path": "example/model.gguf", "oid": "a" * 40}]
    ).encode()

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(tree)

    with pytest.raises(ValueError, match="unknown file shape"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_rejects_boolean_top_level_lfs_size(tmp_path: Path) -> None:
    from scripts.refresh_audio_cpp_artifact_manifest import refresh_manifest_bytes

    manifest_path = _write_manifest(tmp_path, _valid_manifest())
    tree = json.dumps(
        [
            {
                "type": "file",
                "path": "example/model.gguf",
                "oid": "a" * 40,
                "size": True,
                "lfs": {"oid": "b" * 64, "size": 1, "pointerSize": 127},
            }
        ]
    ).encode()

    def recorded_urlopen(_request: object) -> _Response:
        return _Response(tree)

    with pytest.raises(ValueError, match="unknown file shape"):
        refresh_manifest_bytes(manifest_path, COMMIT, urlopen=recorded_urlopen)


def test_refresh_command_runs_directly_without_network_for_empty_manifest(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[2]
    manifest_path = _write_manifest(
        tmp_path,
        {"repository": REPOSITORY, "commit": COMMIT, "packages": []},
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/refresh_audio_cpp_artifact_manifest.py",
            "--commit",
            COMMIT,
            "--manifest",
            str(manifest_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "packages": [],
    }


@pytest.mark.parametrize(
    ("function_name", "requires_raises"),
    [
        ("validate_commit", True),
        ("refresh_manifest_bytes", True),
        ("main", True),
    ],
)
def test_public_manifest_refresh_functions_use_google_style_docstrings(
    function_name: str,
    requires_raises: bool,
) -> None:
    docstring = _manifest_refresh_docstring(function_name)

    assert "Args:" in docstring
    assert "Returns:" in docstring
    if requires_raises:
        assert "Raises:" in docstring


def test_manifest_docstring_contract_does_not_mutate_import_state() -> None:
    before_path = tuple(sys.path)
    sentinel = object()
    before_module = sys.modules.get("audio_cpp_artifact_catalog", sentinel)

    _manifest_refresh_docstring("validate_commit")

    assert tuple(sys.path) == before_path
    assert sys.modules.get("audio_cpp_artifact_catalog", sentinel) is before_module


def test_refresh_command_writes_exact_bytes_to_explicit_output_without_site_packages(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[2]
    payload = {"repository": REPOSITORY, "commit": COMMIT, "packages": []}
    manifest_path = _write_manifest(tmp_path, payload)
    output_path = tmp_path / "nested" / "refreshed-manifest.json"
    output_path.parent.mkdir()
    expected_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/refresh_audio_cpp_artifact_manifest.py",
            "--commit",
            COMMIT,
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == b""
    assert output_path.read_bytes() == expected_bytes
