"""Sealed audio.cpp package recipes and release accounting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from uuid import UUID

import pytest


SUPERONIC_PACKAGES = {
    "supertonic_3_q8_0": ("supertonic-3-q8_0.gguf",),
    "supertonic_3_f16": ("supertonic-3-f16.gguf",),
    "supertonic_3_orig": ("supertonic-3-orig.gguf",),
    "supertonic_3_safetensors": (
        "config/tts.json",
        "config/unicode_indexer.json",
        "ggml/supertonic.safetensors",
    ),
}
POCKET_GGUF_PACKAGES = {
    f"pocket_tts_{language}_{precision}": (f"pocket-tts-{language}-{precision}.gguf",)
    for language in ("english", "german", "italian", "portuguese", "spanish")
    for precision in ("q8_0", "bf16")
}
POCKET_SAFETENSORS_FILES = (
    *(
        f"languages/english/embeddings/{voice}.safetensors"
        for voice in (
            "alba",
            "anna",
            "azelma",
            "bill_boerst",
            "caro_davy",
            "charles",
            "cosette",
            "eponine",
            "estelle",
            "eve",
            "fantine",
            "george",
            "giovanni",
            "jane",
            "javert",
            "jean",
            "juergen",
            "lola",
            "marius",
            "mary",
            "michael",
            "paul",
            "peter_yearsley",
            "rafael",
            "stuart_bell",
            "vera",
        )
    ),
    "languages/english/model.safetensors",
    "languages/english/tokenizer.model",
)
EXPECTED_INITIAL_PACKAGES = {
    **SUPERONIC_PACKAGES,
    **POCKET_GGUF_PACKAGES,
    "pocket_tts_english_safetensors": POCKET_SAFETENSORS_FILES,
}
EXPECTED_RELEASE_FAMILY_COUNTS = {
    "supertonic": 4,
    "pocket_tts": 11,
    "chatterbox": 3,
    "dramabox": 1,
    "miotts": 3,
    "vibevoice": 2,
    "moss_tts_nano": 2,
    "fish_audio": 2,
    "higgs_audio_tts": 2,
    "omnivoice": 4,
    "qwen3_tts": 9,
    "voxcpm2": 4,
    "confucius4_tts": 1,
    "vevo2": 3,
    "index_tts2": 4,
    "irodori_tts": 6,
    "moss_tts_local": 2,
    "glm_tts": 1,
    "inflect_v2": 1,
    "outetts": 1,
    "vietneu_tts": 1,
}


def _api():
    from tldw_chatbook.TTS.audio_cpp_recipes import (  # noqa: F401
        AUDIO_CPP_PINNED_COMMIT,
        AUDIO_CPP_PINNED_RELEASE,
        AUDIO_CPP_RECIPE_REGISTRY,
        AUDIO_CPP_RELEASE_ACCOUNTING,
        AudioCppBackendEvidenceState,
        AudioCppFileKind,
        AudioCppFileRole,
        AudioCppFileSignal,
        AudioCppMatchState,
        AudioCppPackageDescription,
        AudioCppPackageFileEvidence,
        AudioCppReferenceRequirement,
        AudioCppRecipeRegistry,
        AudioCppRecipeSupportState,
        AudioCppVoiceReferencePolicy,
    )

    return locals()


def _identity(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _description(recipe, *, missing=(), invalid=(), partial=False, permission=False):
    api = _api()
    evidence_type = api["AudioCppPackageFileEvidence"]
    description_type = api["AudioCppPackageDescription"]
    files = tuple(
        evidence_type(
            relative_path=signal.relative_path,
            size_bytes=128,
            identity=_identity(signal.relative_path),
            readable=True,
            metadata_valid=signal.relative_path not in invalid,
        )
        for signal in recipe.required_files
        if signal.relative_path not in missing
    )
    return description_type(
        canonical_root="/models/package",
        canonical_root_identity=_identity("root"),
        safe_name="package",
        files=files,
        partial=partial,
        permission_limited=permission,
    )


def test_registry_is_pinned_and_contains_every_initial_package_exactly_once() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]

    assert api["AUDIO_CPP_PINNED_RELEASE"] == "release-0.5.1"
    assert api["AUDIO_CPP_PINNED_COMMIT"] == (
        "238ab6a9e321c17de8e120559f57efeedaeb1345"
    )
    assert len(registry.recipes) == 15
    assert {recipe.package_variant for recipe in registry.recipes} == set(
        EXPECTED_INITIAL_PACKAGES
    )
    assert len({recipe.recipe_id for recipe in registry.recipes}) == 15


def test_registry_collection_cannot_be_replaced_after_construction() -> None:
    registry = _api()["AUDIO_CPP_RECIPE_REGISTRY"]

    with pytest.raises(AttributeError):
        registry.recipes = ()


def test_registry_rejects_conflicting_validation_contracts_for_one_path() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    recipe = registry.for_package("supertonic_3_orig")
    conflicting_signal = replace(
        recipe.required_files[0],
        kind=api["AudioCppFileKind"].JSON,
    )
    conflicting_recipe = replace(
        recipe,
        recipe_id="audio-cpp-0.5.1.supertonic.conflicting-path-contract",
        package_variant="conflicting_path_contract",
        default_public_model_id="conflicting-path-contract",
        required_files=(conflicting_signal,),
    )

    with pytest.raises(ValueError, match="conflicting file validation contracts"):
        api["AudioCppRecipeRegistry"]((recipe, conflicting_recipe))


@pytest.mark.parametrize("package_variant", sorted(EXPECTED_INITIAL_PACKAGES))
def test_recipe_has_exact_reviewed_layout_and_safe_projection(
    package_variant: str,
) -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    support_state = api["AudioCppRecipeSupportState"]
    backend_state = api["AudioCppBackendEvidenceState"]
    recipe = registry.for_package(package_variant)

    assert (
        tuple(signal.relative_path for signal in recipe.required_files)
        == (EXPECTED_INITIAL_PACKAGES[package_variant])
    )
    assert recipe.audio_cpp_release == "release-0.5.1"
    assert recipe.audio_cpp_commit == api["AUDIO_CPP_PINNED_COMMIT"]
    assert recipe.schema_version == 1
    expected_revision = 2 if package_variant in POCKET_GGUF_PACKAGES else 1
    assert recipe.recipe_revision == expected_revision
    assert recipe.support_state is support_state.APPROVED
    assert recipe.projection.family == recipe.family
    assert recipe.projection.task == "tts"
    assert recipe.projection.mode == "offline"
    assert recipe.source_links and all(
        api["AUDIO_CPP_PINNED_COMMIT"] in link for link in recipe.source_links
    )
    assert recipe.evidence_reference
    assert recipe.backend_evidence
    assert all(item.state is backend_state.EXPECTED for item in recipe.backend_evidence)
    if package_variant.endswith(("q8_0", "bf16", "f16", "orig")):
        assert (
            recipe.projection.model_relative_path
            == EXPECTED_INITIAL_PACKAGES[package_variant][0]
        )
    else:
        assert recipe.projection.model_relative_path is None


def test_tasks_and_pocket_language_options_follow_the_pinned_specs() -> None:
    registry = _api()["AUDIO_CPP_RECIPE_REGISTRY"]
    reference_requirement = _api()["AudioCppReferenceRequirement"]
    voice_reference_policy = _api()["AudioCppVoiceReferencePolicy"]

    for recipe in registry.recipes:
        if recipe.family == "supertonic":
            assert recipe.capabilities == ("tts",)
            assert recipe.projection.load_options == ()
            assert recipe.projection.session_options == ()
            assert recipe.reference_requirement is reference_requirement.NONE
            assert recipe.voice_reference_policy is voice_reference_policy.NATIVE_ONLY
            continue
        assert recipe.family == "pocket_tts"
        assert recipe.capabilities == ("tts", "clone")
        language = recipe.package_variant.split("_")[2]
        assert {(item.name, item.value) for item in recipe.projection.load_options} == {
            ("language", language)
        }
        assert {
            (item.name, item.value) for item in recipe.projection.session_options
        } == {("language", language)}
        if recipe.package_format.value == "gguf":
            assert recipe.recipe_revision == 2
            assert recipe.reference_requirement is reference_requirement.REQUIRED
            assert (
                recipe.voice_reference_policy
                is voice_reference_policy.REFERENCE_ONLY
            )
        else:
            assert recipe.reference_requirement is reference_requirement.OPTIONAL
            assert recipe.voice_reference_policy is voice_reference_policy.EITHER


@pytest.mark.parametrize(
    ("package_variant", "voice", "has_reference", "accepted"),
    (
        ("supertonic_3_orig", None, False, True),
        ("supertonic_3_orig", "narrator", False, True),
        ("supertonic_3_orig", None, True, False),
        ("pocket_tts_english_q8_0", None, True, True),
        ("pocket_tts_english_q8_0", "alba", True, False),
        ("pocket_tts_english_q8_0", None, False, False),
        ("pocket_tts_english_safetensors", None, False, True),
        ("pocket_tts_english_safetensors", "alba", False, True),
        ("pocket_tts_english_safetensors", None, True, True),
        ("pocket_tts_english_safetensors", "alba", True, False),
    ),
)
def test_recipe_policy_admits_only_declared_voice_reference_combinations(
    package_variant: str,
    voice: str | None,
    has_reference: bool,
    accepted: bool,
) -> None:
    recipe = _api()["AUDIO_CPP_RECIPE_REGISTRY"].for_package(package_variant)

    assert recipe.admits_voice_reference(
        has_voice=voice is not None,
        has_reference=has_reference,
    ) is accepted


def test_reserved_combined_policy_requires_and_emits_both_fields() -> None:
    api = _api()
    recipe = api["AUDIO_CPP_RECIPE_REGISTRY"].for_package(
        "pocket_tts_english_q8_0"
    )
    combined = replace(
        recipe,
        voice_reference_policy=(
            api["AudioCppVoiceReferencePolicy"].BOTH_REQUIRED_COMBINED
        ),
    )

    assert combined.admits_voice_reference(
        has_voice=True,
        has_reference=True,
    )
    assert not combined.admits_voice_reference(
        has_voice=False,
        has_reference=True,
    )
    assert not combined.admits_voice_reference(
        has_voice=True,
        has_reference=False,
    )


def test_recipe_policy_rejects_contradictory_clone_contracts() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    policies = api["AudioCppVoiceReferencePolicy"]
    requirements = api["AudioCppReferenceRequirement"]
    supertonic = registry.for_package("supertonic_3_orig")
    pocket_gguf = registry.for_package("pocket_tts_english_q8_0")

    with pytest.raises(ValueError, match="voice reference policy"):
        replace(supertonic, voice_reference_policy=policies.REFERENCE_ONLY)
    with pytest.raises(ValueError, match="voice reference policy"):
        replace(pocket_gguf, voice_reference_policy=policies.NATIVE_ONLY)
    with pytest.raises(ValueError, match="voice reference policy"):
        replace(
            pocket_gguf,
            reference_requirement=requirements.OPTIONAL,
        )
    with pytest.raises(TypeError, match="boolean"):
        pocket_gguf.admits_voice_reference(
            has_voice=1,  # type: ignore[arg-type]
            has_reference=True,
        )


def test_every_production_recipe_rejects_undeclared_both_fields() -> None:
    registry = _api()["AUDIO_CPP_RECIPE_REGISTRY"]

    assert all(
        not recipe.admits_voice_reference(has_voice=True, has_reference=True)
        for recipe in registry.recipes
    )


def test_recipe_records_are_frozen_and_reject_path_or_extension_attacks() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    file_signal = api["AudioCppFileSignal"]
    file_kind = api["AudioCppFileKind"]
    file_role = api["AudioCppFileRole"]
    recipe = registry.recipes[0]

    with pytest.raises(FrozenInstanceError):
        recipe.family = "changed"
    for path in (
        "/private/model.gguf",
        "../model.gguf",
        "models/../../model.gguf",
        "C:\\private\\model.gguf",
        "$HOME/model.gguf",
        "models\n/model.gguf",
    ):
        with pytest.raises(ValueError):
            file_signal(
                relative_path=path,
                kind=file_kind.GGUF,
                role=file_role.WEIGHT,
            )
    with pytest.raises(TypeError):
        replace(recipe, shell="curl example.test")
    with pytest.raises(ValueError):
        replace(recipe, required_files=list(recipe.required_files))
    with pytest.raises(ValueError):
        replace(recipe, backend_evidence=list(recipe.backend_evidence))
    with pytest.raises(ValueError):
        file_signal(
            relative_path="model.gguf",
            kind="gguf",  # type: ignore[arg-type]
            role=file_role.WEIGHT,
        )


@pytest.mark.parametrize("package_variant", sorted(EXPECTED_INITIAL_PACKAGES))
def test_every_reviewed_recipe_matches_only_complete_valid_evidence(
    package_variant: str,
) -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    match_state = api["AudioCppMatchState"]
    recipe = registry.for_package(package_variant)

    exact = registry.match(_description(recipe))
    missing = registry.match(
        _description(recipe, missing=(recipe.required_files[-1].relative_path,))
    )
    invalid = registry.match(
        _description(recipe, invalid=(recipe.required_files[0].relative_path,))
    )

    assert exact.state is match_state.EXACT
    assert len(exact.candidates) == 1
    assert exact.candidates[0].recipe.recipe_id == recipe.recipe_id
    # A multi-file layout remains recognizable when one companion disappears.
    # Removing the sole signal from a single-file package leaves no evidence from
    # which to infer a variant, so the truthful fail-closed state is Unknown.
    expected_missing_state = (
        match_state.INCOMPLETE
        if len(recipe.required_files) > 1
        else match_state.UNKNOWN
    )
    assert missing.state is expected_missing_state
    if expected_missing_state is match_state.INCOMPLETE:
        assert recipe.recipe_id in missing.recipe_ids
    assert invalid.state is match_state.INCOMPLETE
    assert recipe.recipe_id in invalid.recipe_ids


def test_near_match_partial_and_permission_evidence_fail_closed() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    description_type = api["AudioCppPackageDescription"]
    evidence_type = api["AudioCppPackageFileEvidence"]
    match_state = api["AudioCppMatchState"]
    recipe = registry.for_package("supertonic_3_orig")
    near = description_type(
        canonical_root="/models/package",
        canonical_root_identity=_identity("root"),
        safe_name="package",
        files=(
            evidence_type(
                relative_path="supertonic-4-orig.gguf",
                size_bytes=128,
                identity=_identity("near"),
                readable=True,
                metadata_valid=True,
            ),
        ),
    )

    assert registry.match(near).state is match_state.UNKNOWN
    assert registry.match(_description(recipe, partial=True)).state is (
        match_state.INCOMPLETE
    )
    assert registry.match(_description(recipe, permission=True)).state is (
        match_state.PERMISSION_LIMITED
    )


def test_conflicting_exact_recipes_remain_ambiguous_for_review() -> None:
    api = _api()
    production = api["AUDIO_CPP_RECIPE_REGISTRY"]
    registry_type = api["AudioCppRecipeRegistry"]
    match_state = api["AudioCppMatchState"]
    recipe = production.for_package("supertonic_3_orig")
    conflict = replace(
        recipe,
        recipe_id="audio-cpp-0.5.1.supertonic.supertonic_3_orig_alt",
        package_variant="supertonic_3_orig_alt",
        default_public_model_id="supertonic-3-orig-alt",
    )
    registry = registry_type((recipe, conflict))

    result = registry.match(_description(recipe))

    assert result.state is match_state.AMBIGUOUS
    assert result.recipe_ids == tuple(sorted((recipe.recipe_id, conflict.recipe_id)))
    assert len(result.candidates) == 2


def test_accepted_snapshot_must_equal_the_exact_recipe_revision_and_projection() -> (
    None
):
    registry = _api()["AUDIO_CPP_RECIPE_REGISTRY"]
    recipe = registry.for_package("supertonic_3_orig")
    candidate = registry.match(_description(recipe)).candidates[0]
    accepted = candidate.accept(public_model_id="narrator")
    second = candidate.accept(public_model_id="narrator")

    assert registry.validate_accepted(accepted) is recipe
    assert str(UUID(accepted.package_uuid)) == accepted.package_uuid
    assert accepted.package_uuid != second.package_uuid
    assert accepted.public_model_id == second.public_model_id
    with pytest.raises(ValueError, match="review"):
        registry.validate_accepted(accepted.model_copy(update={"recipe_revision": 2}))
    with pytest.raises(ValueError, match="review"):
        registry.validate_accepted(
            accepted.model_copy(
                update={
                    "projection": accepted.projection.model_copy(
                        update={"family": "pocket_tts"}
                    )
                }
            )
        )


def test_release_accounting_is_complete_and_truthful_for_all_21_families() -> None:
    api = _api()
    accounting = api["AUDIO_CPP_RELEASE_ACCOUNTING"]
    support_state = api["AudioCppRecipeSupportState"]
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    counts: dict[str, int] = {}
    for entry in accounting:
        counts[entry.family] = counts.get(entry.family, 0) + 1

    assert len(accounting) == 67
    assert counts == EXPECTED_RELEASE_FAMILY_COUNTS
    assert sum(entry.state is support_state.APPROVED for entry in accounting) == 15
    assert sum(entry.state is support_state.OPEN_GAP for entry in accounting) == 52
    assert not any(
        entry.state is support_state.EXPLICITLY_UNSUPPORTED for entry in accounting
    )
    assert {
        entry.package_variant
        for entry in accounting
        if entry.state is support_state.APPROVED
    } == {recipe.package_variant for recipe in registry.recipes}
    assert all(
        entry.recipe_id is not None
        if entry.state is support_state.APPROVED
        else entry.reason == "Recipe review is not complete."
        for entry in accounting
    )
    pinned_rows = "\n".join(
        sorted(f"{entry.family}:{entry.package_variant}" for entry in accounting)
    )
    assert sha256(pinned_rows.encode("utf-8")).hexdigest() == (
        "10d75a8ab499d15cbb49c73dc8a070d994c788dda66f5d82315747146c9d8480"
    )


def test_user_facing_verified_claims_exclude_expected_or_untested_tuples() -> None:
    api = _api()
    registry = api["AUDIO_CPP_RECIPE_REGISTRY"]
    registry_type = api["AudioCppRecipeRegistry"]
    backend_state = api["AudioCppBackendEvidenceState"]
    recipe = registry.recipes[0]

    assert registry.verified_support_claims() == ()
    verified_tuple = replace(
        recipe.backend_evidence[0],
        state=backend_state.VERIFIED,
        evidence_reference="qa/audio-cpp/release-0.5.1/darwin-arm64-cpu",
    )
    verified_recipe = replace(recipe, backend_evidence=(verified_tuple,))
    claims = registry_type((verified_recipe,)).verified_support_claims()

    assert len(claims) == 1
    assert claims[0].recipe_id == recipe.recipe_id
    assert claims[0].backend == verified_tuple.backend
    assert claims[0].evidence_reference == verified_tuple.evidence_reference
