"""Strict persisted schemas and stable identities for STT qualification."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Mapping, Union
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

MAX_BYTE_SIZE = 2**63 - 1
MAX_COUNT = 2**31 - 1
MAX_DURATION_SECONDS = 31 * 24 * 60 * 60
MAX_THREADS = 1024
MAX_BOOTSTRAP_ITERATIONS = 1_000_000
REQUIRED_RUNTIME_PACKAGES = frozenset({"onnx-asr", "onnxruntime", "faster-whisper"})

APPROVED_V3_LANGUAGE_ORDER = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
)
APPROVED_V3_LANGUAGES = frozenset(APPROVED_V3_LANGUAGE_ORDER)

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2048),
]
ArtifactFilename = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=255),
]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]*$",
    ),
]
LanguageCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        min_length=2,
        max_length=16,
        pattern=r"^[a-z]{2,3}(?:-[a-z0-9]+)*$",
    ),
]
Sha256 = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
PositiveByteSize = Annotated[int, Field(strict=True, gt=0, le=MAX_BYTE_SIZE)]
PositiveCount = Annotated[int, Field(strict=True, gt=0, le=MAX_COUNT)]
PositiveDuration = Annotated[
    float,
    Field(strict=True, gt=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
]
NonNegativeDuration = Annotated[
    float,
    Field(strict=True, ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
]
ThreadCount = Annotated[int, Field(strict=True, gt=0, le=MAX_THREADS)]
Ratio = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
SchemaVersion = Literal[1]


class MeasurementProfile(str, Enum):
    """Measurement modes with deliberately separate instrumentation."""

    QUALITY = "quality"
    THROUGHPUT = "throughput"
    MEMORY_REUSE = "memory_reuse"


class PrimaryMetric(str, Enum):
    """Primary error metric for a comparison population."""

    WER = "wer"
    CER = "cer"


class Precision(str, Enum):
    """Precisions in the approved qualification comparison."""

    INT8 = "int8"
    F32 = "f32"


class ModelFamily(str, Enum):
    """Closed model families approved by TASK-593."""

    PARAKEET_V2 = "parakeet_v2"
    PARAKEET_V3 = "parakeet_v3"
    FASTER_WHISPER_BASE = "faster_whisper_base"


class QualificationRole(str, Enum):
    """A model's single role in the qualification comparison."""

    CANDIDATE_INT8 = "candidate_int8"
    F32_REFERENCE = "f32_reference"
    COMPARISON_BASELINE = "comparison_baseline"


class LanguageScope(str, Enum):
    """Closed language coverage declared by a qualified model."""

    ENGLISH = "english"
    V3 = "v3"
    ALL_EVALUATED = "all_evaluated"


class VadMode(str, Enum):
    """How a qualified model supplies long-form VAD."""

    EXTERNAL = "external"
    RUNTIME_INTERNAL = "runtime_internal"


class StrictModel(BaseModel):
    """Base for immutable persisted records with a closed field set."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactFile(StrictModel):
    """One exact regular file belonging to a content-addressed artifact."""

    filename: ArtifactFilename
    size_bytes: PositiveByteSize
    sha256: Sha256

    @field_validator("filename")
    @classmethod
    def filename_is_one_contained_component(cls, value: str) -> str:
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        windows_basename = value.split(".", maxsplit=1)[0].upper()
        windows_reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{number}" for number in range(1, 10)),
            *(f"LPT{number}" for number in range(1, 10)),
        }
        if (
            value in {".", ".."}
            or value.endswith((".", " "))
            or windows_basename in windows_reserved
            or any(
                character.isspace() or unicodedata.category(character) in {"Cc", "Cf"}
                for character in value
            )
            or "/" in value
            or "\\" in value
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or posix.name != value
            or windows.name != value
        ):
            raise ValueError(
                "artifact filename must be one contained relative component"
            )
        return value


ArtifactFiles = Annotated[
    tuple[ArtifactFile, ...],
    Field(min_length=1, max_length=10_000),
]


def _reject_duplicate_artifact_filenames(
    files: tuple[ArtifactFile, ...],
) -> tuple[ArtifactFile, ...]:
    names = [artifact.filename for artifact in files]
    if len(names) != len(set(names)):
        raise ValueError("artifact filenames must be unique")
    return files


class VadSettings(StrictModel):
    """Pinned VAD settings that affect long-form qualification."""

    batch_size: PositiveCount


class VadVariant(StrictModel):
    """Pinned VAD artifact declaration and runtime provenance."""

    variant_id: Identifier
    precision: Precision
    runtime: NonEmptyStr
    repository: NonEmptyStr
    revision: NonEmptyStr
    license: NonEmptyStr
    files: ArtifactFiles
    settings: VadSettings

    _unique_files = field_validator("files")(_reject_duplicate_artifact_filenames)


class ModelCapabilities(StrictModel):
    """Capabilities required to interpret qualification evidence."""

    language_scope: LanguageScope
    supports_timestamps: StrictBool
    supports_long_form: StrictBool
    vad_mode: VadMode


class ModelVariant(StrictModel):
    """Pinned model artifact declaration and local execution identity."""

    variant_id: Identifier
    provider: Identifier
    model_id: Identifier
    family: ModelFamily
    qualification_role: QualificationRole
    precision: Precision
    runtime: NonEmptyStr
    repository: NonEmptyStr
    revision: NonEmptyStr
    license: NonEmptyStr
    files: ArtifactFiles
    vad_variant_id: Identifier | None
    capabilities: ModelCapabilities

    _unique_files = field_validator("files")(_reject_duplicate_artifact_filenames)


class ModelManifest(StrictModel):
    """Closed collection of model and VAD artifacts for one experiment."""

    schema_version: SchemaVersion
    models: Annotated[
        tuple[ModelVariant, ...],
        Field(min_length=1, max_length=1_000),
    ]
    vad_variants: Annotated[
        tuple[VadVariant, ...],
        Field(max_length=1_000),
    ]

    @model_validator(mode="after")
    def identities_and_vad_references_are_valid(self) -> "ModelManifest":
        model_ids = [model.variant_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model variant identities must be unique")

        vad_ids = [vad.variant_id for vad in self.vad_variants]
        if len(vad_ids) != len(set(vad_ids)):
            raise ValueError("VAD variant identities must be unique")
        known_vads = set(vad_ids)
        for model in self.models:
            if (
                model.vad_variant_id is not None
                and model.vad_variant_id not in known_vads
            ):
                raise ValueError(
                    f"model {model.variant_id!r} references unknown VAD variant "
                    f"{model.vad_variant_id!r}"
                )

        required_roles = {
            (
                ModelFamily.PARAKEET_V2,
                QualificationRole.CANDIDATE_INT8,
                Precision.INT8,
            ),
            (
                ModelFamily.PARAKEET_V2,
                QualificationRole.F32_REFERENCE,
                Precision.F32,
            ),
            (
                ModelFamily.PARAKEET_V3,
                QualificationRole.CANDIDATE_INT8,
                Precision.INT8,
            ),
            (
                ModelFamily.PARAKEET_V3,
                QualificationRole.F32_REFERENCE,
                Precision.F32,
            ),
            (
                ModelFamily.FASTER_WHISPER_BASE,
                QualificationRole.COMPARISON_BASELINE,
                Precision.INT8,
            ),
        }
        actual_roles = {
            (model.family, model.qualification_role, model.precision)
            for model in self.models
        }
        if actual_roles != required_roles or len(self.models) != len(required_roles):
            raise ValueError(
                "model manifest must contain exactly the approved qualification "
                "model roles"
            )

        for model in self.models:
            capabilities = model.capabilities
            if not (
                capabilities.supports_timestamps and capabilities.supports_long_form
            ):
                raise ValueError(
                    "qualification models must support timestamps and long-form audio"
                )

            if model.family in {
                ModelFamily.PARAKEET_V2,
                ModelFamily.PARAKEET_V3,
            }:
                expected_scope = (
                    LanguageScope.ENGLISH
                    if model.family is ModelFamily.PARAKEET_V2
                    else LanguageScope.V3
                )
                if (
                    model.provider != "onnx-asr"
                    or capabilities.language_scope is not expected_scope
                    or capabilities.vad_mode is not VadMode.EXTERNAL
                    or model.vad_variant_id is None
                ):
                    raise ValueError(
                        "Parakeet qualification models require their approved "
                        "language scope and external VAD"
                    )
            elif (
                model.provider != "faster-whisper"
                or capabilities.language_scope is not LanguageScope.ALL_EVALUATED
                or capabilities.vad_mode is not VadMode.RUNTIME_INTERNAL
                or model.vad_variant_id is not None
            ):
                raise ValueError(
                    "faster-whisper baseline requires all evaluated languages "
                    "and runtime-internal VAD"
                )
        return self


class CorpusSource(StrictModel):
    """Immutable provenance for one external corpus source."""

    source_id: Identifier
    repository: NonEmptyStr
    revision: NonEmptyStr
    source_url: NonEmptyStr
    license: NonEmptyStr
    artifact: ArtifactFile


class CorpusSample(StrictModel):
    """One pinned prepared sample and its evaluation identity."""

    sample_id: Identifier
    source_id: Identifier
    upstream_sample_id: NonEmptyStr
    prepared_file: ArtifactFile
    reference_text: Annotated[str, Field(max_length=1_000_000)]
    language: LanguageCode
    tags: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=256),
    ]
    cluster_id: Identifier
    duration_seconds: PositiveDuration

    @field_validator("tags")
    @classmethod
    def tags_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("sample tags must be unique")
        return value


class CorpusPopulation(StrictModel):
    """Fixed sample membership for one matrix population."""

    population_id: Identifier
    language: LanguageCode
    sample_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=MAX_COUNT),
    ]
    tags: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=256),
    ]

    @field_validator("sample_ids", "tags")
    @classmethod
    def members_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("population members and tags must be unique")
        return value


class CorpusManifest(StrictModel):
    """Immutable prepared corpus declaration."""

    schema_version: SchemaVersion
    sources: Annotated[
        tuple[CorpusSource, ...],
        Field(min_length=1, max_length=10_000),
    ]
    samples: Annotated[
        tuple[CorpusSample, ...],
        Field(min_length=1, max_length=MAX_COUNT),
    ]
    populations: Annotated[
        tuple[CorpusPopulation, ...],
        Field(min_length=1, max_length=10_000),
    ]
    derived_recipe_revisions: Annotated[
        tuple[NonEmptyStr, ...],
        Field(min_length=1, max_length=1_000),
    ]

    @model_validator(mode="after")
    def references_and_identities_are_valid(self) -> "CorpusManifest":
        source_ids = [source.source_id for source in self.sources]
        sample_ids = [sample.sample_id for sample in self.samples]
        population_ids = [population.population_id for population in self.populations]
        for label, identities in (
            ("source", source_ids),
            ("sample", sample_ids),
            ("population", population_ids),
        ):
            if len(identities) != len(set(identities)):
                raise ValueError(f"corpus {label} identities must be unique")

        known_sources = set(source_ids)
        samples_by_id = {sample.sample_id: sample for sample in self.samples}
        for sample in self.samples:
            if sample.source_id not in known_sources:
                raise ValueError(
                    f"sample {sample.sample_id!r} references unknown source "
                    f"{sample.source_id!r}"
                )

        for population in self.populations:
            for sample_id in population.sample_ids:
                if sample_id not in samples_by_id:
                    raise ValueError(
                        f"population {population.population_id!r} references "
                        f"unknown sample {sample_id!r}"
                    )
                if samples_by_id[sample_id].language != population.language:
                    raise ValueError(
                        f"population {population.population_id!r} language "
                        "does not match all member samples"
                    )
        return self


class AcquisitionMode(str, Enum):
    """How a preparation source is allowed to cross the trust boundary."""

    VERIFIED_DOWNLOAD = "verified_download"
    LOCAL_FILE = "local_file"


def _validate_archive_member_name(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = value.split("/")
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    unsafe_windows_component = any(
        part.endswith((".", " "))
        or ":" in part
        or part.split(".", maxsplit=1)[0].upper() in windows_reserved
        for part in parts
    )
    if (
        value != unicodedata.normalize("NFC", value)
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
        or unsafe_windows_component
        or any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)
    ):
        raise ValueError(
            "archive member name must be an unambiguous contained POSIX path"
        )
    return value


class ArchiveMember(StrictModel):
    """One exact regular file in a complete source-archive allowlist."""

    name: NonEmptyStr
    file_type: Literal["regular_file"]
    size_bytes: PositiveByteSize
    sha256: Sha256
    selected_for_preparation: StrictBool

    @field_validator("name")
    @classmethod
    def name_is_unambiguous_contained_posix_path(cls, value: str) -> str:
        return _validate_archive_member_name(value)


ArchiveMembers = Annotated[
    tuple[ArchiveMember, ...],
    Field(min_length=1, max_length=100_000),
]


class PreparationLimits(StrictModel):
    """Hard limits applied before and during corpus preparation."""

    max_member_count: PositiveCount
    max_file_bytes: PositiveByteSize
    max_uncompressed_bytes: PositiveByteSize
    staging_headroom_bytes: PositiveByteSize


class PreparationSource(StrictModel):
    """One immutable archive and its complete member allowlist."""

    source_id: Identifier
    repository: NonEmptyStr
    revision: NonEmptyStr
    source_url: NonEmptyStr
    license: NonEmptyStr
    acquisition_mode: AcquisitionMode
    archive: ArtifactFile
    members: ArchiveMembers

    @field_validator("members")
    @classmethod
    def members_are_unique_and_unambiguous(
        cls, value: tuple[ArchiveMember, ...]
    ) -> tuple[ArchiveMember, ...]:
        names = [member.name for member in value]
        comparison_names = [name.casefold() for name in names]
        if len(names) != len(set(names)) or len(comparison_names) != len(
            set(comparison_names)
        ):
            raise ValueError("archive member names must be unique and unambiguous")
        seen_prefixes: dict[tuple[str, ...], tuple[str, ...]] = {}
        for name in names:
            exact_prefix: list[str] = []
            comparison_prefix: list[str] = []
            for part in name.split("/"):
                exact_prefix.append(part)
                comparison_prefix.append(part.casefold())
                key = tuple(comparison_prefix)
                exact = tuple(exact_prefix)
                previous = seen_prefixes.setdefault(key, exact)
                if previous != exact:
                    raise ValueError(
                        "archive member names must be unique and unambiguous"
                    )
        return value

    @model_validator(mode="after")
    def download_url_is_fixed_and_credential_free(self) -> "PreparationSource":
        if self.acquisition_mode is not AcquisitionMode.VERIFIED_DOWNLOAD:
            return self
        parsed = urlsplit(self.source_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
        ):
            raise ValueError(
                "verified downloads require a credential-free HTTPS source URL"
            )
        return self


class NormalizedSampleRecipe(StrictModel):
    """Map one selected archive member to one normalized prepared sample."""

    recipe_type: Literal["normalize"]
    recipe_revision: NonEmptyStr
    sample_id: Identifier
    source_id: Identifier
    source_member: NonEmptyStr
    prepared_file: ArtifactFile

    @field_validator("source_member")
    @classmethod
    def source_member_is_contained(cls, value: str) -> str:
        return _validate_archive_member_name(value)


class SilenceRecipe(StrictModel):
    """Generate exact PCM silence for a declared duration."""

    recipe_type: Literal["silence"]
    recipe_revision: NonEmptyStr
    sample_id: Identifier
    duration_seconds: PositiveDuration
    prepared_file: ArtifactFile

    @field_validator("duration_seconds")
    @classmethod
    def duration_is_an_integral_pcm_frame(cls, value: float) -> float:
        if not (value * 16_000).is_integer():
            raise ValueError("duration must resolve to an integral 16 kHz frame")
        return value


class NoiseRecipe(StrictModel):
    """Apply fixed-seed bounded noise to a declared prepared source."""

    recipe_type: Literal["noise"]
    recipe_revision: NonEmptyStr
    sample_id: Identifier
    source_sample_id: Identifier
    seed: Seed
    noise_amplitude: Ratio
    source_gain: Ratio
    prepared_file: ArtifactFile


class ConcatenationRecipe(StrictModel):
    """Concatenate ordered prepared inputs with exact silence gaps."""

    recipe_type: Literal["concatenation"]
    recipe_revision: NonEmptyStr
    sample_id: Identifier
    source_sample_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=MAX_COUNT),
    ]
    silence_gaps_seconds: Annotated[
        tuple[NonNegativeDuration, ...],
        Field(max_length=MAX_COUNT),
    ]
    prepared_file: ArtifactFile

    @model_validator(mode="after")
    def gaps_match_inputs_and_resolve_to_frames(self) -> "ConcatenationRecipe":
        if len(self.silence_gaps_seconds) != len(self.source_sample_ids) - 1:
            raise ValueError(
                "concatenation requires exactly one silence gap between inputs"
            )
        if any(
            not (duration * 16_000).is_integer()
            for duration in self.silence_gaps_seconds
        ):
            raise ValueError(
                "concatenation gaps must resolve to integral 16 kHz frames"
            )
        return self


DerivedRecipe = Annotated[
    Union[SilenceRecipe, NoiseRecipe, ConcatenationRecipe],
    Field(discriminator="recipe_type"),
]


class PreparationManifest(StrictModel):
    """Closed source, normalization, derivation, and bound declaration."""

    schema_version: SchemaVersion
    sources: Annotated[
        tuple[PreparationSource, ...],
        Field(min_length=1, max_length=10_000),
    ]
    limits: PreparationLimits
    normalized_samples: Annotated[
        tuple[NormalizedSampleRecipe, ...],
        Field(max_length=MAX_COUNT),
    ]
    derived_recipes: Annotated[
        tuple[DerivedRecipe, ...],
        Field(max_length=MAX_COUNT),
    ]

    @model_validator(mode="after")
    def identities_and_recipe_references_are_closed(self) -> "PreparationManifest":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("preparation source identities must be unique")
        sources = {source.source_id: source for source in self.sources}

        known_samples: set[str] = set()
        output_names: set[str] = set()
        for recipe in self.normalized_samples:
            if recipe.sample_id in known_samples:
                raise ValueError("prepared sample identities must be unique")
            source = sources.get(recipe.source_id)
            if source is None:
                raise ValueError(
                    f"normalization references unknown source {recipe.source_id!r}"
                )
            matching_members = [
                member
                for member in source.members
                if member.name == recipe.source_member
            ]
            if (
                len(matching_members) != 1
                or not matching_members[0].selected_for_preparation
            ):
                raise ValueError(
                    "normalization source member must be declared and selected"
                )
            known_samples.add(recipe.sample_id)
            output_names.add(recipe.prepared_file.filename)

        for recipe in self.derived_recipes:
            if recipe.sample_id in known_samples:
                raise ValueError("prepared sample identities must be unique")
            if recipe.prepared_file.filename in output_names:
                raise ValueError("prepared output filenames must be unique")
            inputs: tuple[str, ...]
            if isinstance(recipe, NoiseRecipe):
                inputs = (recipe.source_sample_id,)
            elif isinstance(recipe, ConcatenationRecipe):
                inputs = recipe.source_sample_ids
            else:
                inputs = ()
            unknown_inputs = [
                sample_id for sample_id in inputs if sample_id not in known_samples
            ]
            if unknown_inputs:
                raise ValueError(
                    f"derived recipe references unknown input {unknown_inputs[0]!r}"
                )
            known_samples.add(recipe.sample_id)
            output_names.add(recipe.prepared_file.filename)

        normalized_output_names = [
            recipe.prepared_file.filename for recipe in self.normalized_samples
        ]
        if len(normalized_output_names) != len(set(normalized_output_names)):
            raise ValueError("prepared output filenames must be unique")
        return self


class SourceArchiveIdentity(StrictModel):
    """Verified source archive recorded in a completion receipt."""

    source_id: Identifier
    archive: ArtifactFile


class PreparationReceipt(StrictModel):
    """Terminal proof that an immutable prepared corpus is complete."""

    schema_version: SchemaVersion
    status: Literal["complete"]
    experiment_fingerprint: Sha256
    preparation_manifest_sha256: Sha256
    ffmpeg_executable: NonEmptyStr
    ffmpeg_version: NonEmptyStr
    source_archives: Annotated[
        tuple[SourceArchiveIdentity, ...],
        Field(min_length=1, max_length=10_000),
    ]
    prepared_files: Annotated[
        tuple[ArtifactFile, ...],
        Field(max_length=MAX_COUNT),
    ]


class MatrixRequirement(StrictModel):
    """Declared candidate/baseline population and its applicable profiles."""

    model_variant_id: Identifier
    baseline_variant_id: Identifier
    language: LanguageCode
    population_id: Identifier
    primary_metric: PrimaryMetric
    profiles: Annotated[
        tuple[MeasurementProfile, ...],
        Field(min_length=1, max_length=len(MeasurementProfile)),
    ]

    @model_validator(mode="after")
    def requirement_is_valid(self) -> "MatrixRequirement":
        if self.model_variant_id == self.baseline_variant_id:
            raise ValueError("candidate and baseline model variants must differ")
        if len(self.profiles) != len(set(self.profiles)):
            raise ValueError("measurement profiles must be unique")
        return self


class ComparisonMatrixCell(StrictModel):
    """One predeclared model/population/profile evidence obligation."""

    model_variant_id: Identifier
    baseline_variant_id: Identifier
    language: LanguageCode
    population_id: Identifier
    primary_metric: PrimaryMetric
    measurement_profile: MeasurementProfile
    min_sample_count: PositiveCount
    min_reference_units: PositiveCount
    min_audio_duration_seconds: PositiveDuration

    @model_validator(mode="after")
    def candidate_and_baseline_differ(self) -> "ComparisonMatrixCell":
        if self.model_variant_id == self.baseline_variant_id:
            raise ValueError("candidate and baseline model variants must differ")
        return self

    def identity_key(self) -> tuple[str, str, str, MeasurementProfile]:
        """Return the unique closed-matrix identity of this cell."""

        return (
            self.model_variant_id,
            self.baseline_variant_id,
            self.population_id,
            self.measurement_profile,
        )


class RuntimePackage(StrictModel):
    """Exact runtime package identity."""

    name: Identifier
    version: NonEmptyStr


class RuntimeSettings(StrictModel):
    """Non-variant environment settings shared by every experiment run."""

    python_version: NonEmptyStr
    packages: Annotated[
        tuple[RuntimePackage, ...],
        Field(min_length=1, max_length=1_000),
    ]
    operating_system: NonEmptyStr
    os_version: NonEmptyStr
    hardware: NonEmptyStr
    cpu: NonEmptyStr
    memory_bytes: PositiveByteSize
    execution_provider: NonEmptyStr
    intra_op_threads: ThreadCount
    inter_op_threads: ThreadCount

    @field_validator("packages")
    @classmethod
    def package_names_are_unique(
        cls, value: tuple[RuntimePackage, ...]
    ) -> tuple[RuntimePackage, ...]:
        names = [package.name for package in value]
        if len(names) != len(set(names)):
            raise ValueError("runtime package names must be unique")
        missing = REQUIRED_RUNTIME_PACKAGES - set(names)
        if missing:
            raise ValueError(
                "runtime packages must include exact onnx-asr, onnxruntime, "
                "and faster-whisper versions"
            )
        return value


class GateSettings(StrictModel):
    """All predeclared thresholds that affect qualification."""

    max_v2_baseline_wer_delta: Ratio
    max_english_slice_wer_delta: Ratio
    max_v3_primary_delta: Ratio
    max_v3_macro_delta: Ratio
    max_int8_f32_delta: Ratio
    min_inverse_real_time_factor: Annotated[
        float,
        Field(strict=True, gt=0, le=1_000_000, allow_inf_nan=False),
    ]
    max_peak_rss_bytes: PositiveByteSize
    max_memory_reuse_growth: Ratio


class BootstrapSettings(StrictModel):
    """Deterministic paired-bootstrap configuration."""

    seed: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
    iterations: Annotated[
        int,
        Field(strict=True, gt=0, le=MAX_BOOTSTRAP_ITERATIONS),
    ]


class EffectiveExecutionSettings(StrictModel):
    """Effective variant-specific settings included in a run identity."""

    execution_provider: NonEmptyStr
    device: NonEmptyStr
    intra_op_threads: ThreadCount
    inter_op_threads: ThreadCount
    vad_batch_size: PositiveCount | None


class RunIdentityInputs(StrictModel):
    """Variant-specific inputs used to derive a run fingerprint."""

    model_variant_id: Identifier
    measurement_profile: MeasurementProfile
    effective_settings: EffectiveExecutionSettings


class ExperimentManifest(StrictModel):
    """Resolved, closed experiment definition shared by every variant run."""

    schema_version: SchemaVersion
    v3_languages: tuple[LanguageCode, ...]
    corpus: CorpusManifest
    models: ModelManifest
    requirements: Annotated[
        tuple[MatrixRequirement, ...],
        Field(min_length=1, max_length=100_000),
    ]
    matrix: Annotated[
        tuple[ComparisonMatrixCell, ...],
        Field(min_length=1, max_length=1_000_000),
    ]
    runtime: RuntimeSettings
    gates: GateSettings
    bootstrap: BootstrapSettings
    normalizer_revision: NonEmptyStr
    metric_revision: NonEmptyStr
    harness_revision: NonEmptyStr

    @field_validator("v3_languages", mode="before")
    @classmethod
    def v3_language_set_is_exact(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("v3 language set must be an ordered collection")
        if not all(isinstance(language, str) for language in value):
            raise ValueError("v3 language set entries must be strings")
        if (
            len(value) != len(APPROVED_V3_LANGUAGES)
            or len(set(value)) != len(value)
            or set(value) != APPROVED_V3_LANGUAGES
        ):
            raise ValueError(
                "v3 language set must contain exactly the approved 24 languages"
            )
        return APPROVED_V3_LANGUAGE_ORDER

    @model_validator(mode="after")
    def closed_matrix_is_complete(self) -> "ExperimentManifest":
        model_ids = {model.variant_id for model in self.models.models}
        models_by_role = {
            (model.family, model.qualification_role): model
            for model in self.models.models
        }
        v2_candidate = models_by_role[
            (ModelFamily.PARAKEET_V2, QualificationRole.CANDIDATE_INT8)
        ].variant_id
        v2_f32 = models_by_role[
            (ModelFamily.PARAKEET_V2, QualificationRole.F32_REFERENCE)
        ].variant_id
        v3_candidate = models_by_role[
            (ModelFamily.PARAKEET_V3, QualificationRole.CANDIDATE_INT8)
        ].variant_id
        v3_f32 = models_by_role[
            (ModelFamily.PARAKEET_V3, QualificationRole.F32_REFERENCE)
        ].variant_id
        comparison_baseline = models_by_role[
            (
                ModelFamily.FASTER_WHISPER_BASE,
                QualificationRole.COMPARISON_BASELINE,
            )
        ].variant_id
        populations = {
            population.population_id: population
            for population in self.corpus.populations
        }
        required_requirement_keys: set[tuple[str, str, str]] = set()
        for declared_population in populations.values():
            if declared_population.language == "en":
                candidate = v2_candidate
                baselines = (v2_f32, comparison_baseline)
            elif declared_population.language in APPROVED_V3_LANGUAGES:
                candidate = v3_candidate
                baselines = (v3_f32, comparison_baseline)
            else:
                continue
            required_requirement_keys.update(
                (candidate, baseline, declared_population.population_id)
                for baseline in baselines
            )

        requirement_keys: set[tuple[str, str, str]] = set()
        v3_profiles_by_language: dict[str, set[MeasurementProfile]] = {
            language: set() for language in APPROVED_V3_LANGUAGES
        }
        all_profiles = set(MeasurementProfile)
        expected: dict[
            tuple[str, str, str, MeasurementProfile],
            tuple[str, PrimaryMetric],
        ] = {}
        for requirement in self.requirements:
            if requirement.model_variant_id not in model_ids:
                raise ValueError(
                    "closed comparison matrix references unknown model variant "
                    f"{requirement.model_variant_id!r}"
                )
            if requirement.baseline_variant_id not in model_ids:
                raise ValueError(
                    "closed comparison matrix references unknown baseline "
                    f"{requirement.baseline_variant_id!r}"
                )
            population = populations.get(requirement.population_id)
            if population is None:
                raise ValueError(
                    "closed comparison matrix references unknown population "
                    f"{requirement.population_id!r}"
                )
            requirement_key = (
                requirement.model_variant_id,
                requirement.baseline_variant_id,
                requirement.population_id,
            )
            if requirement_key in requirement_keys:
                raise ValueError("duplicate comparison matrix requirement")
            requirement_keys.add(requirement_key)
            if requirement.language != population.language:
                raise ValueError(
                    "closed comparison matrix requirement language must match "
                    "its population"
                )
            if set(requirement.profiles) != all_profiles:
                raise ValueError(
                    "closed comparison matrix must preserve exact v3 "
                    "language/profile coverage and required qualification pairings"
                )
            for profile in requirement.profiles:
                expected[(*requirement_key, profile)] = (
                    requirement.language,
                    requirement.primary_metric,
                )
                if requirement.language in v3_profiles_by_language:
                    v3_profiles_by_language[requirement.language].add(profile)

        if requirement_keys != required_requirement_keys:
            raise ValueError(
                "closed comparison matrix must preserve exact v3 "
                "language/profile coverage and required qualification pairings"
            )
        if any(
            profiles != all_profiles for profiles in v3_profiles_by_language.values()
        ):
            raise ValueError(
                "closed comparison matrix must preserve exact v3 "
                "language/profile coverage"
            )

        actual: dict[
            tuple[str, str, str, MeasurementProfile],
            ComparisonMatrixCell,
        ] = {}
        for cell in self.matrix:
            key = cell.identity_key()
            if key in actual:
                raise ValueError("duplicate comparison matrix cell")
            actual[key] = cell

        if set(actual) != set(expected):
            raise ValueError(
                "closed comparison matrix must contain every declared "
                "model/population/profile cell and no undeclared cells"
            )

        for key, cell in actual.items():
            expected_language, expected_metric = expected[key]
            if (
                cell.language != expected_language
                or cell.primary_metric != expected_metric
            ):
                raise ValueError(
                    "closed comparison matrix language and primary metric "
                    "must match the declared population requirement"
                )
        return self


def canonical_json(value: BaseModel | Mapping[str, object]) -> bytes:
    """Encode a model or mapping as deterministic UTF-8 JSON."""

    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def experiment_fingerprint(manifest: ExperimentManifest) -> str:
    """Return the stable identity of all non-variant experiment content."""

    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def run_fingerprint(
    experiment_identity: str,
    inputs: RunIdentityInputs,
) -> str:
    """Return a variant/profile identity within one resolved experiment."""

    if len(experiment_identity) != 64 or any(
        character not in "0123456789abcdef" for character in experiment_identity
    ):
        raise ValueError(
            "experiment fingerprint must be exactly 64 lowercase hex characters"
        )
    return hashlib.sha256(
        canonical_json(
            {
                "experiment_fingerprint": experiment_identity,
                "model_variant_id": inputs.model_variant_id,
                "measurement_profile": inputs.measurement_profile.value,
                "effective_settings": inputs.effective_settings.model_dump(mode="json"),
            }
        )
    ).hexdigest()


__all__ = [
    "APPROVED_V3_LANGUAGES",
    "AcquisitionMode",
    "ArchiveMember",
    "ArtifactFile",
    "BootstrapSettings",
    "ComparisonMatrixCell",
    "ConcatenationRecipe",
    "CorpusManifest",
    "CorpusPopulation",
    "CorpusSample",
    "CorpusSource",
    "DerivedRecipe",
    "EffectiveExecutionSettings",
    "ExperimentManifest",
    "GateSettings",
    "MatrixRequirement",
    "MeasurementProfile",
    "ModelCapabilities",
    "ModelFamily",
    "ModelManifest",
    "ModelVariant",
    "NoiseRecipe",
    "NormalizedSampleRecipe",
    "PreparationLimits",
    "PreparationManifest",
    "PreparationReceipt",
    "PreparationSource",
    "PrimaryMetric",
    "QualificationRole",
    "RunIdentityInputs",
    "RuntimeSettings",
    "SilenceRecipe",
    "SourceArchiveIdentity",
    "StrictModel",
    "VadVariant",
    "canonical_json",
    "experiment_fingerprint",
    "run_fingerprint",
]
