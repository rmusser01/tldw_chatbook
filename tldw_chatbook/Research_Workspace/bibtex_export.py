"""Build a metadata-only bibliography for exact workspace sources."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from urllib.parse import urlsplit

from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy

from .contracts import QualifiedWorkspaceRef, ResearchSourceSummary, WorkspaceDataSource
from .copy_catalog import _page_items, get_local_copy_policy
from .copy_manifest import ResearchCopyCandidate
from .copy_models import CopyItemKind, CopyOwnerRef


_BIBTEX_ESCAPE = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


def _safe_url(value: object) -> str:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return ""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme in {"https", "http"}
            and parsed.hostname
            and not (
                parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            )
        ):
            return value
    except ValueError:
        pass
    return ""


def render_bibtex(
    ref: QualifiedWorkspaceRef,
    sources: Iterable[tuple[ResearchSourceSummary, Mapping[str, object]]],
) -> str:
    """Render readable entries with stable keys across workspaces of this catalog."""
    seen = set()
    entries = []
    for source, metadata in sources:
        if source.ref != ref:
            raise ValueError("A bibliography source belongs to a different workspace")
        if source.catalog_item_id in seen:
            continue
        seen.add(source.catalog_item_id)
        identity = json.dumps(
            (
                ref.data_source,
                ref.server_profile_id,
                ref.principal_id,
                source.catalog_item_id,
            )
        )
        cite_key = "source-" + sha256(identity.encode("utf-8")).hexdigest()[:20]
        year_str = str(metadata.get("year") or metadata.get("published_at") or "")
        year = re.match(r"^([1-9][0-9]{3})(?:$|-)", year_str)
        author = metadata.get("author")
        fields = {"title": source.title}
        if isinstance(author, str) and author.strip():
            fields["author"] = author
        if year:
            fields["year"] = year[1]
        if url := _safe_url(metadata.get("url")):
            fields["url"] = url
        if "author" not in fields or "year" not in fields:
            fields["note"] = "Incomplete metadata"
        lines = [f"@misc{{{cite_key},"]
        for key, value in fields.items():
            escaped = " ".join(value.split()).translate(_BIBTEX_ESCAPE)
            lines.append(f"  {key} = {{{escaped}}},")
        entries.append("\n".join((*lines, "}")))
    return "\n\n".join(entries) + ("\n" if entries else "")


@dataclass(frozen=True, slots=True)
class PreparedWorkspaceBibtex:
    text: str
    policies: tuple[tuple[ResearchCopyCandidate, WorkspaceTransferPolicy], ...]

    async def confirm_current_policy(self, registry) -> None:
        for candidate, policy in self.policies:
            if (
                await get_local_copy_policy(
                    registry, candidate, require_membership=True
                )
                is not policy
            ):
                raise ValueError(
                    "A source transfer policy changed. Review a new bibliography."
                )


async def prepare_workspace_bibtex(ref, port, registry) -> PreparedWorkspaceBibtex:
    sources = await _page_items(
        lambda offset: port.list_sources(ref, limit=100, offset=offset)
    )
    entries, policies = [], []
    seen = set()
    for source in sources:
        if source.ref != ref:
            raise ValueError("A bibliography source belongs to a different workspace")
        if source.catalog_item_id in seen:
            continue
        seen.add(source.catalog_item_id)
        if ref.data_source is WorkspaceDataSource.LOCAL:
            if registry is None:
                raise ValueError("Local workspace transfer policy is unavailable.")
            owner = CopyOwnerRef(
                ref,
                CopyItemKind.MEDIA,
                source.catalog_item_id,
                source.catalog_item_version,
            )
            candidate = ResearchCopyCandidate(owner, membership_id=source.source_id)
            policy = await get_local_copy_policy(
                registry, candidate, require_membership=True
            )
            if policy is WorkspaceTransferPolicy.LOCAL_ONLY:
                continue
            policies.append((candidate, policy))
        metadata = await port.bibliography_metadata(source)
        entries.append((source, metadata))
    return PreparedWorkspaceBibtex(render_bibtex(ref, entries), tuple(policies))
