"""Persistent exact Parakeet source preferences and verified dispatch."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Literal

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    active_managed_parakeet_dir,
    parakeet_descriptor,
    parakeet_reference,
    parakeet_v2_managed_service,
    parakeet_vad_reference,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactError,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.config import get_cli_setting, save_settings_to_cli_config

from .contracts import ExecutionDevice
from .executor import (
    LocalSourceChangedError,
    ModelIdentity,
    validate_local_source_snapshot,
)
from .parakeet_dispatch import ParakeetDispatch, resolve_parakeet_dispatch
from .parakeet_external import ExternalParakeetVerifier, VerifiedExternalParakeet


class ParakeetSourceKey(str, Enum):
    """Stable configuration keys for every curated Parakeet root."""

    V2_INT8 = "v2_int8"
    V2_F32 = "v2_f32"
    V3_INT8 = "v3_int8"
    V3_F32 = "v3_f32"

    @property
    def model_id(self) -> str:
        return PARAKEET_V2_MODEL if self.value.startswith("v2_") else PARAKEET_V3_MODEL

    @property
    def precision(self) -> str:
        return "int8" if self.value.endswith("_int8") else "f32"

    @classmethod
    def from_values(cls, model_id: str, precision: str) -> ParakeetSourceKey:
        for key in cls:
            if (key.model_id, key.precision) == (model_id, precision):
                return key
        raise ValueError("unsupported Parakeet model and precision")


class ParakeetSourcePreference(str, Enum):
    """Authoritative source choice for one exact descriptor."""

    EXTERNAL = "external"
    MANAGED = "managed"


class ParakeetSourceErrorCode(str, Enum):
    """Stable path-free source service failure categories."""

    INVALID_SELECTION = "invalid_selection"
    MANAGED_UNAVAILABLE = "managed_source_unavailable"
    VAD_UNAVAILABLE = "managed_vad_unavailable"
    CONFIG_WRITE_FAILED = "config_write_failed"
    CONFIG_MISMATCH = "config_commit_mismatch"
    COPY_CONSENT_MISMATCH = "managed_copy_consent_mismatch"
    COPY_FAILED = "managed_copy_failed"
    COPY_INSUFFICIENT_SPACE = "managed_copy_insufficient_space"


class ParakeetSourceError(RuntimeError):
    """Stable path-free source resolution or commit failure."""

    def __init__(self, code: ParakeetSourceErrorCode) -> None:
        self.code = code
        super().__init__(f"Parakeet source failed: {code.value}")


@dataclass(frozen=True)
class ParakeetSourceRecord:
    """Persistent exact source record with a path-private representation."""

    model_id: str
    precision: str
    directory: Path | None = field(default=None, repr=False)
    preferred_source: ParakeetSourcePreference | None = None


@dataclass(frozen=True, repr=False)
class PreparedExternalSelection:
    """Verified external selection pending adoption or persistence."""

    key: ParakeetSourceKey
    verified: VerifiedExternalParakeet = field(repr=False)


@dataclass(frozen=True, repr=False)
class ExternalSourceConfigCommit:
    """Write-free prepared patch for one atomic configuration commit."""

    prepared: PreparedExternalSelection = field(repr=False)
    section_values: Mapping[str, Mapping[str, object]] = field(repr=False)


@dataclass(frozen=True)
class ManagedCopyConsent:
    """Consent bound to one exact managed-copy plan."""

    reference: ArtifactRef
    additional_bytes: int
    destination: Path


@dataclass(frozen=True)
class ManagedCopyPlan:
    """User-reviewable local-copy disk plan for one exact root."""

    reference: ArtifactRef
    additional_bytes: int
    destination: Path
    free_bytes: int
    already_installed: bool

    def grant(self) -> ManagedCopyConsent:
        """Grant this plan when its additional bytes fit on disk."""

        if self.additional_bytes > self.free_bytes:
            raise ParakeetSourceError(ParakeetSourceErrorCode.COPY_INSUFFICIENT_SPACE)
        return ManagedCopyConsent(
            reference=self.reference,
            additional_bytes=self.additional_bytes,
            destination=self.destination,
        )


_ReadSetting = Callable[[str, str, object], object]
_WriteSettings = Callable[[Mapping[str, Mapping[str, object]]], bool]
_DescriptorFor = Callable[[str, str], ArtifactDescriptor]
_ActiveManaged = Callable[[str, str], Path | None]
_DispatchResolver = Callable[..., ParakeetDispatch]


def _never_cancelled() -> bool:
    return False


class ParakeetSourceService:
    """Own exact source preferences and process-lifetime verification state."""

    def __init__(
        self,
        *,
        verifier: ExternalParakeetVerifier | None = None,
        read_setting: _ReadSetting = get_cli_setting,
        write_settings: _WriteSettings = save_settings_to_cli_config,
        descriptor_for: _DescriptorFor = parakeet_descriptor,
        active_managed: _ActiveManaged = active_managed_parakeet_dir,
        dispatch_resolver: _DispatchResolver = resolve_parakeet_dispatch,
        vad_ready: Callable[[], bool] | None = None,
        managed_service: ModelArtifactService | None = None,
    ) -> None:
        self.verifier = verifier if verifier is not None else ExternalParakeetVerifier()
        self._read_setting = read_setting
        self._write_settings = write_settings
        self._descriptor_for = descriptor_for
        self._active_managed = active_managed
        self._dispatch_resolver = dispatch_resolver
        self._managed_service = managed_service
        self._vad_ready = (
            vad_ready if vad_ready is not None else self._default_vad_ready
        )
        # ponytail: one lock is enough until source-preference write throughput matters.
        self._mutation_lock = threading.Lock()
        self._records = self._read_records()
        self._observed_scopes: set[str] = set()
        self._sync_configured_owners()

    def records(self) -> Mapping[ParakeetSourceKey, ParakeetSourceRecord]:
        """Return an immutable snapshot of valid exact source records."""

        return MappingProxyType(dict(self._records))

    def prepare_external(
        self,
        key: ParakeetSourceKey,
        directory: Path,
        *,
        owner: tuple[Literal["configured", "scope"], str] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] | None = None,
    ) -> PreparedExternalSelection:
        """Verify one external directory for the requested exact key."""

        self._require_key(key)
        descriptor = self._descriptor_for(key.model_id, key.precision)
        verified = self.verifier.verify(
            descriptor,
            Path(directory),
            owner=owner,
            cancelled=cancelled,
            progress=progress,
        )
        if verified.reference != descriptor.reference:
            raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)
        return PreparedExternalSelection(key=key, verified=verified)

    def retain_prepared(
        self,
        scope_id: str,
        prepared: PreparedExternalSelection,
    ) -> None:
        """Recheck and retain a prepared selection for one live scope."""

        if type(scope_id) is not str or not scope_id:
            raise ValueError("scope_id must be a non-empty string")
        self._validate_prepared(prepared)
        descriptor = self._descriptor_for(
            prepared.key.model_id,
            prepared.key.precision,
        )
        verified = self.verifier.verify(
            descriptor,
            prepared.verified.directory,
            owner=("scope", scope_id),
        )
        if verified != prepared.verified:
            raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)

    def prepare_config_commit(
        self,
        prepared: PreparedExternalSelection,
    ) -> ExternalSourceConfigCommit:
        """Recheck root and VAD readiness without writing configuration."""

        self._validate_prepared(prepared)
        self._require_vad_ready()
        records = dict(self._records)
        records[prepared.key] = ParakeetSourceRecord(
            model_id=prepared.key.model_id,
            precision=prepared.key.precision,
            directory=prepared.verified.directory,
            preferred_source=ParakeetSourcePreference.EXTERNAL,
        )
        values: Mapping[str, Mapping[str, object]] = {
            "transcription": {
                "parakeet_external_sources": self._serialize_records(records)
            }
        }
        return ExternalSourceConfigCommit(prepared=prepared, section_values=values)

    def accept_committed(self, commit: ExternalSourceConfigCommit) -> None:
        """Adopt a prepared patch only after its persisted record matches."""

        if type(commit) is not ExternalSourceConfigCommit:
            raise TypeError("commit must be an ExternalSourceConfigCommit")
        expected = self._parse_records(
            commit.section_values["transcription"]["parakeet_external_sources"]
        )
        persisted = self._read_records()
        if persisted.get(commit.prepared.key) != expected.get(commit.prepared.key):
            raise ParakeetSourceError(ParakeetSourceErrorCode.CONFIG_MISMATCH)
        self._records = persisted
        self._sync_configured_owners()

    def commit_external(
        self,
        prepared: PreparedExternalSelection,
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        """Persist one prepared external selection with exactly one write."""

        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        with self._mutation_lock:
            commit = self.prepare_config_commit(prepared)
            if cancelled():
                return
            self._write(commit.section_values)
            self.accept_committed(commit)

    def prefer_managed(
        self,
        key: ParakeetSourceKey,
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        """Prefer the exact managed root while remembering any directory."""

        self._require_key(key)
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        with self._mutation_lock:
            records = dict(self._records)
            prior = records.get(key)
            records[key] = ParakeetSourceRecord(
                model_id=key.model_id,
                precision=key.precision,
                directory=prior.directory if prior is not None else None,
                preferred_source=ParakeetSourcePreference.MANAGED,
            )
            self._persist_records(records, cancelled=cancelled)

    def on_root_activated(self, reference: ArtifactRef) -> None:
        """Prefer managed only when an exact curated Parakeet root activates."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        for key in ParakeetSourceKey:
            if self._descriptor_for(key.model_id, key.precision).reference == reference:
                self.prefer_managed(key)
                return

    def may_delete(self, reference: ArtifactRef) -> str | None:
        """Block configured external sources from losing their managed VAD."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        if reference != parakeet_vad_reference():
            return None
        if any(
            record.preferred_source is ParakeetSourcePreference.EXTERNAL
            for record in self._records.values()
        ):
            return (
                "Managed dependency is required by a configured external "
                "Parakeet source. Stop using the external source first."
            )
        return None

    def plan_managed_copy(
        self,
        verified: VerifiedExternalParakeet,
    ) -> ManagedCopyPlan:
        """Recheck one external root and report its managed-copy disk use."""

        _, descriptor = self._validated_external(verified)
        self._require_vad_ready()
        managed = self._managed_store_service()
        already_installed = any(
            item.descriptor == descriptor and item.error is None
            for item in managed.list_installed()
        )
        usage = managed.disk_usage()
        return ManagedCopyPlan(
            reference=descriptor.reference,
            additional_bytes=(
                0 if already_installed else descriptor.expected_installed_bytes
            ),
            destination=managed.artifact_path(descriptor.reference),
            free_bytes=usage.free_bytes,
            already_installed=already_installed,
        )

    def copy_into_managed(
        self,
        verified: VerifiedExternalParakeet,
        consent: ManagedCopyConsent,
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> ArtifactRef:
        """Install declared root files without activation or preference changes."""

        if type(consent) is not ManagedCopyConsent:
            raise TypeError("consent must be a ManagedCopyConsent")
        plan = self.plan_managed_copy(verified)
        if plan.already_installed:
            return plan.reference
        if consent != plan.grant():
            raise ParakeetSourceError(ParakeetSourceErrorCode.COPY_CONSENT_MISMATCH)
        _, descriptor = self._validated_external(verified)
        try:
            return self._managed_store_service().install(
                descriptor,
                verified.directory,
                declared_files_only=True,
                cancelled=cancelled,
            )
        except (ArtifactError, OSError):
            raise ParakeetSourceError(ParakeetSourceErrorCode.COPY_FAILED) from None

    def stop_using_external(
        self,
        key: ParakeetSourceKey,
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        """Forget the directory without erasing a managed preference."""

        self._require_key(key)
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        with self._mutation_lock:
            records = dict(self._records)
            prior = records.get(key)
            if prior is None:
                return
            if prior.preferred_source is ParakeetSourcePreference.MANAGED:
                records[key] = ParakeetSourceRecord(
                    model_id=key.model_id,
                    precision=key.precision,
                    preferred_source=ParakeetSourcePreference.MANAGED,
                )
            else:
                records.pop(key, None)
            self._persist_records(records, cancelled=cancelled)

    def resolve(
        self,
        key: ParakeetSourceKey,
        *,
        override: str | Path | None = None,
        scope_id: str | None = None,
    ) -> ParakeetDispatch:
        """Resolve the authoritative exact source without downloading."""

        self._require_key(key)
        if override is not None and str(override).strip():
            prepared = self.prepare_external(
                key,
                Path(override),
                owner=("scope", scope_id) if scope_id else None,
            )
            self._require_vad_ready()
            return self._external_dispatch(prepared)

        record = self._records.get(key)
        if record is not None and record.preferred_source is not None:
            if record.preferred_source is ParakeetSourcePreference.EXTERNAL:
                if record.directory is None:
                    raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)
                prepared = self.prepare_external(
                    key,
                    record.directory,
                    owner=("configured", key.value),
                )
                self._require_vad_ready()
                return self._external_dispatch(prepared)
            return self._managed_dispatch(key)

        if self._active_managed(key.model_id, key.precision) is not None:
            return self._managed_dispatch(key)

        legacy = self._read_setting(
            "transcription",
            "parakeet_onnx_model_dir",
            "",
        )
        if key is ParakeetSourceKey.V2_INT8 and type(legacy) is str and legacy.strip():
            migration_scope = "legacy-v2-int8-migration"
            prepared = self.prepare_external(
                key,
                Path(legacy),
                owner=("scope", migration_scope),
            )
            try:
                self.commit_external(prepared)
            finally:
                self.release_scope(migration_scope)
            return self._external_dispatch(prepared)

        return self._dispatch_resolver(
            model_id=key.model_id,
            precision=key.precision,
            model_dir=None,
        )

    def release_scope(self, scope_id: str) -> None:
        """Release verifier ownership for one explicit scope."""

        self._observed_scopes.discard(scope_id)
        self.verifier.release_scope(scope_id)

    def release_scopes_except(self, active_scope_ids: Collection[str]) -> None:
        """Release observed scopes that are no longer active."""

        active = {scope for scope in active_scope_ids if type(scope) is str and scope}
        for scope_id in self._observed_scopes - active:
            self.verifier.release_scope(scope_id)
        self._observed_scopes = active

    def close(self) -> None:
        self._observed_scopes.clear()
        self.verifier.close()

    def _managed_dispatch(
        self,
        key: ParakeetSourceKey,
    ) -> ParakeetDispatch:
        if self._active_managed(key.model_id, key.precision) is None:
            raise ParakeetSourceError(ParakeetSourceErrorCode.MANAGED_UNAVAILABLE)
        dispatch = self._dispatch_resolver(
            model_id=key.model_id,
            precision=key.precision,
            model_dir=None,
        )
        expected = parakeet_reference(key.model_id, key.precision)
        expected_tuple = (
            expected.artifact_id,
            expected.revision,
            expected.variant,
        )
        if dispatch.managed_artifact_ref != expected_tuple:
            raise ParakeetSourceError(ParakeetSourceErrorCode.MANAGED_UNAVAILABLE)
        return dispatch

    def _external_dispatch(
        self,
        prepared: PreparedExternalSelection,
    ) -> ParakeetDispatch:
        verified = prepared.verified
        key = prepared.key
        vad = parakeet_vad_reference()
        managed_store_root = (
            self._managed_service.artifacts_path.parent
            if self._managed_service is not None
            else managed_model_artifact_root().absolute()
        )
        return ParakeetDispatch(
            identity=ModelIdentity(
                provider_id="parakeet-onnx",
                model_id=key.model_id,
                root_revision=verified.reference.revision,
                closure_fingerprint=None,
                precision=key.precision,
                device=ExecutionDevice.CPU,
                local_snapshot_token=verified.snapshot.token,
            ),
            local_source=verified.snapshot,
            managed_store_root=managed_store_root,
            managed_artifact_ref=None,
            option_updates=MappingProxyType(
                {"transcription_model_dir": str(verified.directory)}
            ),
            managed_dependency_refs=((vad.artifact_id, vad.revision, vad.variant),),
        )

    def _validate_prepared(self, prepared: PreparedExternalSelection) -> None:
        if type(prepared) is not PreparedExternalSelection:
            raise TypeError("prepared must be a PreparedExternalSelection")
        descriptor = self._descriptor_for(
            prepared.key.model_id,
            prepared.key.precision,
        )
        if prepared.verified.reference != descriptor.reference:
            raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)
        try:
            validate_local_source_snapshot(prepared.verified.snapshot)
        except (LocalSourceChangedError, OSError):
            raise ParakeetSourceError(
                ParakeetSourceErrorCode.INVALID_SELECTION
            ) from None

    def _validated_external(
        self,
        verified: VerifiedExternalParakeet,
    ) -> tuple[ParakeetSourceKey, ArtifactDescriptor]:
        if type(verified) is not VerifiedExternalParakeet:
            raise TypeError("verified must be a VerifiedExternalParakeet")
        for key in ParakeetSourceKey:
            descriptor = self._descriptor_for(key.model_id, key.precision)
            if descriptor.reference == verified.reference:
                self._validate_prepared(
                    PreparedExternalSelection(key=key, verified=verified)
                )
                return key, descriptor
        raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)

    def _managed_store_service(self) -> ModelArtifactService:
        return (
            self._managed_service
            if self._managed_service is not None
            else parakeet_v2_managed_service()
        )

    def _default_vad_ready(self) -> bool:
        service = (
            self._managed_service
            if self._managed_service is not None
            else parakeet_v2_managed_service()
        )
        try:
            service._verify_installed(  # noqa: SLF001 - Task 3 replaces this with the public dependency lease.
                parakeet_vad_reference(),
                ArtifactRole.DEPENDENCY,
            )
        except (ArtifactError, OSError, TypeError, ValueError):
            return False
        return True

    def _require_vad_ready(self) -> None:
        try:
            ready = self._vad_ready()
        except Exception:
            ready = False
        if not ready:
            raise ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)

    def _read_records(self) -> dict[ParakeetSourceKey, ParakeetSourceRecord]:
        raw = self._read_setting(
            "transcription",
            "parakeet_external_sources",
            {},
        )
        return self._parse_records(raw)

    @staticmethod
    def _parse_records(raw: object) -> dict[ParakeetSourceKey, ParakeetSourceRecord]:
        if type(raw) is not dict:
            return {}
        records: dict[ParakeetSourceKey, ParakeetSourceRecord] = {}
        allowed = {"model_id", "precision", "directory", "preferred_source"}
        for key in ParakeetSourceKey:
            value = raw.get(key.value)
            if type(value) is not dict or not set(value).issubset(allowed):
                continue
            if (
                value.get("model_id") != key.model_id
                or value.get("precision") != key.precision
            ):
                continue
            raw_directory = value.get("directory")
            directory = None
            if raw_directory is not None:
                if type(raw_directory) is not str or not raw_directory.strip():
                    continue
                directory = Path(raw_directory)
                if not directory.is_absolute():
                    continue
            raw_preference = value.get("preferred_source")
            try:
                preference = (
                    None
                    if raw_preference is None
                    else ParakeetSourcePreference(raw_preference)
                )
            except (TypeError, ValueError):
                continue
            if preference is ParakeetSourcePreference.EXTERNAL and directory is None:
                continue
            if preference is None and directory is None:
                continue
            records[key] = ParakeetSourceRecord(
                model_id=key.model_id,
                precision=key.precision,
                directory=directory,
                preferred_source=preference,
            )
        return records

    @staticmethod
    def _serialize_records(
        records: Mapping[ParakeetSourceKey, ParakeetSourceRecord],
    ) -> dict[str, dict[str, str]]:
        serialized: dict[str, dict[str, str]] = {}
        for key in ParakeetSourceKey:
            record = records.get(key)
            if record is None:
                continue
            value = {
                "model_id": key.model_id,
                "precision": key.precision,
            }
            if record.directory is not None:
                value["directory"] = str(record.directory)
            if record.preferred_source is not None:
                value["preferred_source"] = record.preferred_source.value
            if len(value) > 2:
                serialized[key.value] = value
        return serialized

    def _persist_records(
        self,
        records: dict[ParakeetSourceKey, ParakeetSourceRecord],
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        values: Mapping[str, Mapping[str, object]] = {
            "transcription": {
                "parakeet_external_sources": self._serialize_records(records)
            }
        }
        if cancelled():
            return
        self._write(values)
        self._records = records
        self._sync_configured_owners()

    def _write(self, values: Mapping[str, Mapping[str, object]]) -> None:
        try:
            saved = self._write_settings(values)
        except Exception:
            raise ParakeetSourceError(
                ParakeetSourceErrorCode.CONFIG_WRITE_FAILED
            ) from None
        if not saved:
            raise ParakeetSourceError(ParakeetSourceErrorCode.CONFIG_WRITE_FAILED)

    def _sync_configured_owners(self) -> None:
        owners: dict[str, tuple[ArtifactRef, Path]] = {}
        for key, record in self._records.items():
            if (
                record.directory is not None
                and record.preferred_source is ParakeetSourcePreference.EXTERNAL
            ):
                descriptor = self._descriptor_for(key.model_id, key.precision)
                owners[key.value] = (descriptor.reference, record.directory)
        self.verifier.set_configured_owners(owners)

    @staticmethod
    def _require_key(key: ParakeetSourceKey) -> None:
        if type(key) is not ParakeetSourceKey:
            raise TypeError("key must be a ParakeetSourceKey")


__all__ = [
    "ExternalSourceConfigCommit",
    "ManagedCopyConsent",
    "ManagedCopyPlan",
    "ParakeetSourceError",
    "ParakeetSourceErrorCode",
    "ParakeetSourceKey",
    "ParakeetSourcePreference",
    "ParakeetSourceRecord",
    "ParakeetSourceService",
    "PreparedExternalSelection",
]
