"""Review and restore text entries from a Research Workspace export."""

import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
from typing import Literal
from uuid import UUID
from zipfile import BadZipFile, ZipFile

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Workspaces.registry_service import DuplicateWorkspace

from .contracts import QualifiedWorkspaceRef, require_capability
from .quick_notes import ResearchNoteSaveRequest


_MAX_BYTES = 32 * 1024 * 1024
_MAX_FILES = 1024
_MEMBER = re.compile(r"items/([1-9][0-9]*)-(note|artifact)\.md")
_MANIFEST_ENTRY = re.compile(
    r"^- Export content · (note|artifact) `[^`\n]+`(?: · version \d+)? · `(items/[^`\n]+)`$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class WorkspaceBundleItem:
    filename: str
    kind: Literal["note", "artifact"]
    title: str
    body: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedWorkspaceImport:
    fingerprint: str
    items: tuple[WorkspaceBundleItem, ...]


def read_workspace_bundle(path: Path) -> PreparedWorkspaceImport:
    """Read only the declared text files and never extract an archive to disk."""
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ValueError("Choose a Research Workspace .zip bundle.")
    if path.stat().st_size > _MAX_BYTES + 1024 * 1024:
        raise ValueError("Workspace bundle exceeds the import size limit.")
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            names = {item.filename for item in members}
            if len(names) != len(members) or len(members) > _MAX_FILES:
                raise ValueError("Workspace bundle files exceed the supported layout.")
            if (
                "manifest.md" not in names
                or sum(item.file_size for item in members) > _MAX_BYTES
            ):
                raise ValueError(
                    "Workspace bundle is missing its manifest or exceeds 32 MiB."
                )
            manifest = archive.read("manifest.md")
            text = manifest.decode("utf-8")
            if not text.startswith("# Research workspace bundle\n\nLocal workspace `"):
                raise ValueError(
                    "Choose a bundle exported from a Local Research Workspace."
                )
            declared = _MANIFEST_ENTRY.findall(text)
            if len({filename for _, filename in declared}) != len(declared) or {
                name for _, name in declared
            } != names - {"manifest.md"}:
                raise ValueError("Workspace bundle files do not match its manifest.")
            digest = sha256(manifest)
            items = []
            for kind, filename in declared:
                match = _MEMBER.fullmatch(filename)
                if not match or match[2] != kind:
                    raise ValueError(
                        "Workspace bundle files contain an unsupported owner."
                    )
                body_bytes = archive.read(filename)
                digest.update(sha256(body_bytes).digest())
                header, separator, body = body_bytes.decode("utf-8").partition("\n\n")
                title = header.removeprefix("# ").strip()
                if (
                    not separator
                    or not header.startswith("# ")
                    or not title
                    or len(title) > 1000
                    or any(ord(char) < 32 for char in title)
                ):
                    raise ValueError("A workspace bundle entry is missing its title.")
                items.append(
                    WorkspaceBundleItem(filename, kind, title, body.removesuffix("\n"))
                )
    except (BadZipFile, UnicodeError, RuntimeError, NotImplementedError) as exc:
        raise ValueError("Read a valid Research Workspace export bundle.") from exc
    return PreparedWorkspaceImport(digest.hexdigest(), tuple(items))


async def publish_workspace_bundle(
    prepared: PreparedWorkspaceImport, name: str, port, registry, chatbooks
) -> QualifiedWorkspaceRef:
    """Import canonical Notes and Chatbooks with an ID that survives a partial retry."""
    if not prepared.items:
        raise ValueError("This bundle contains no text entries to import.")
    if any(item.kind == "artifact" for item in prepared.items) and chatbooks is None:
        raise ValueError("Local Chatbook storage is required to import this bundle.")

    async def owner_call(function, *args, **kwargs):
        return await asyncio.to_thread(
            run_finite_local_worker, function, *args, **kwargs
        )

    workspace_id = "research-import-" + prepared.fingerprint[:32]
    ref = QualifiedWorkspaceRef("local", workspace_id)
    if any(item.kind == "note" for item in prepared.items):
        require_capability(await port.capabilities(ref), "save_note")
    workspace = await owner_call(registry.get_workspace, workspace_id)
    if workspace is None:
        try:
            workspace = await owner_call(
                registry.create_workspace, workspace_id=workspace_id, name=name
            )
        except DuplicateWorkspace:
            workspace = await owner_call(registry.get_workspace, workspace_id)
            if workspace is None:
                raise ValueError(
                    "Choose a different name for this workspace import."
                ) from None
    if workspace.archived:
        raise ValueError(
            "Restore the previous import from Archived workspaces to resume it."
        )
    for item in prepared.items:
        key = sha256(f"{prepared.fingerprint}/{item.filename}".encode()).hexdigest()
        try:
            if item.kind == "note":
                operation = UUID(hex=key[:32], version=4)
                result = await port.save_note(
                    ref,
                    ResearchNoteSaveRequest(
                        item.title,
                        item.body,
                        operation_id=f"research-note-{operation.hex}",
                    ),
                )
                if result.ref != ref:
                    raise ValueError(
                        "The Local Notes owner returned a different workspace"
                    )
            else:
                record = await chatbooks.create_chatbook(
                    name=item.title,
                    description="Imported Research Workspace artifact",
                    tags=["research", "artifact"],
                    categories=["Research", "Artifacts"],
                    metadata={
                        "artifact_source": "research_import",
                        "artifact_kind": "imported",
                        "content": item.body,
                    },
                    copy_operation_key=key,
                )
                await owner_call(
                    registry.link_membership,
                    workspace_id,
                    item_type="chatbook",
                    item_id=str(record["chatbook_id"]),
                    role="output",
                    title=item.title,
                )
        except Exception as exc:
            raise ValueError(
                "Import paused. Open the same bundle to resume saving entries."
            ) from exc
    return ref
