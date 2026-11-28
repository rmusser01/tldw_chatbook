"""Freeze a Local Research export using the canonical owners and Copy policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy

from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from .copy_adapters import TextArtifactCopyOwner
from .copy_catalog import capture_copy_candidates, get_local_copy_policy
from .copy_manifest import ResearchCopyCandidate, copy_treatment
from .copy_models import CopyItemKind, CopyTreatment
from .copy_service import CopyOwnerActionError

_MAX_BUNDLE_CONTENT_BYTES = 32 * 1024 * 1024
_TREATMENT_LABELS = {
    CopyTreatment.CONTENT: "Export content",
    CopyTreatment.REFERENCE: "Reference only",
    CopyTreatment.METADATA_ONLY: "Metadata only",
    CopyTreatment.OMIT: "Omit (device-only)",
    CopyTreatment.BLOCKED: "Blocked (export unsupported)",
}


@dataclass(frozen=True, slots=True)
class _ExportEntry:
    candidate: ResearchCopyCandidate
    policy: WorkspaceTransferPolicy
    treatment: CopyTreatment
    document: bytes | None = field(default=None, repr=False)

    def manifest_line(self, file_name: str | None) -> str:
        item = self.candidate.source
        label = f"- {_TREATMENT_LABELS[self.treatment]} · {item.kind.value.replace('_', ' ')}"
        if self.treatment is CopyTreatment.OMIT:
            return label
        revision = f" · version {item.version}" if item.version is not None else ""
        path = f" · `{file_name}`" if file_name else ""
        return f"{label} `{item.owner_id}`{revision}{path}"


@dataclass(frozen=True, slots=True)
class PreparedWorkspaceExport:
    """One reviewed Local export with a frozen set of document bytes."""

    workspace: QualifiedWorkspaceRef
    entries: tuple[_ExportEntry, ...]
    content_bytes: int

    @property
    def content_count(self) -> int:
        return sum(entry.document is not None for entry in self.entries)

    async def confirm_current_policy(self, registry) -> None:
        """Require the same outbound policy immediately before writing."""
        for entry in self.entries:
            policy = await get_local_copy_policy(
                registry, entry.candidate, require_membership=entry.document is not None
            )
            if policy != entry.policy:
                raise ValueError(
                    "A workspace transfer policy changed. Review a new export."
                )

    def bundle_bytes(self) -> bytes:
        """Serialize the reviewed files and a readable manifest."""
        output = BytesIO()
        manifest = [
            "# Research workspace bundle",
            "",
            f"Local workspace `{self.workspace.workspace_id}`",
            "",
            "Device preferences, folders, annotations, and drafts are excluded.",
            "",
        ]
        files = []
        for index, entry in enumerate(self.entries, start=1):
            kind = entry.candidate.source.kind.value.replace("_", "-")
            file_name = (
                f"items/{index}-{kind}.md" if entry.document is not None else None
            )
            manifest.append(entry.manifest_line(file_name))
            if file_name is not None:
                files.append((file_name, entry.document))
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.md", "\n".join(manifest) + "\n")
            for file_name, data in files:
                archive.writestr(file_name, data)
        return output.getvalue()


async def prepare_workspace_export(
    ref, port, registry, *, chatbooks
) -> PreparedWorkspaceExport:
    """Read only content permitted by the captured Local owner and policy."""
    if ref.data_source is not WorkspaceDataSource.LOCAL:
        raise ValueError("Choose a Local workspace to export.")
    candidates = await capture_copy_candidates(port, ref)
    artifact_owner = (
        TextArtifactCopyOwner(registry, chatbooks, lambda _ref: None)
        if chatbooks is not None
        else None
    )
    content_bytes = 0
    entries = []
    for candidate in candidates:
        policy = await get_local_copy_policy(registry, candidate)
        treatment = copy_treatment(policy)
        document = None
        if treatment is CopyTreatment.CONTENT:
            source = candidate.source
            if source.kind is CopyItemKind.NOTE:
                note = await port.get_note(ref, source.owner_id)
                if not note or (note.ref, note.note_id, note.version) != (
                    ref,
                    source.owner_id,
                    source.version,
                ):
                    raise ValueError("A workspace note changed. Review a new export.")
                document = f"# {note.title}\n\n{note.content}\n".encode("utf-8")
            elif source.kind is CopyItemKind.ARTIFACT and artifact_owner is not None:
                try:
                    snapshot = await artifact_owner.inspect_source(source)
                except CopyOwnerActionError:
                    raise ValueError(
                        "A workspace artifact changed. Review a new export."
                    ) from None
                if snapshot.owner != source:
                    raise ValueError(
                        "A workspace artifact changed. Review a new export."
                    )
                body = snapshot.body
                document = f"# {body.title}\n\n{body.content}\n".encode("utf-8")
            else:
                treatment = CopyTreatment.BLOCKED
        if document is not None:
            content_bytes += len(document)
            if content_bytes > _MAX_BUNDLE_CONTENT_BYTES:
                raise ValueError(
                    "The selected workspace exceeds the 32 MiB export size limit."
                )
        entries.append(_ExportEntry(candidate, policy, treatment, document))
    return PreparedWorkspaceExport(ref, tuple(entries), content_bytes)
