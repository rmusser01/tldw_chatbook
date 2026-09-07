"""Readable bundle/artifact rendering for the Research window (task-16483).

Pure functions over the two bundle shapes the window sees — the local
engine's ``{"run": ..., "artifacts": [...]}`` and the server's
name-to-content mapping — so run outputs are legible instead of raw JSON
dumps. Known artifact types render structurally; anything else falls back
to pretty-printed JSON.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

__all__ = [
    "default_artifact_for_bundle",
    "render_artifact",
    "render_bundle_summary",
]

_MAX_TEXT_CHARS = 8000


def _field(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _artifact_names(bundle: Any) -> list[str]:
    if isinstance(bundle, Mapping):
        artifacts = bundle.get("artifacts")
        if isinstance(artifacts, list):
            return [
                str(_field(artifact, "artifact_name") or "")
                for artifact in artifacts
                if _field(artifact, "artifact_name")
            ]
        # Server shape: the mapping's own keys ARE artifact names.
        return [str(key) for key in bundle.keys()]
    return []


def render_bundle_summary(bundle: Any) -> str:
    """One-screen bundle overview: run status plus the artifact inventory."""
    if bundle is None:
        return "No bundle loaded."
    lines: list[str] = []
    if isinstance(bundle, Mapping) and isinstance(bundle.get("run"), Mapping):
        run = bundle["run"]
        phase = f" ({run['phase']})" if run.get("phase") else ""
        lines.append(f"Run {run.get('id')} — {run.get('status')}{phase}")
        if run.get("query"):
            lines.append(f"Query: {run['query']}")
    names = _artifact_names(bundle)
    lines.append(f"Artifacts ({len(names)}):")
    lines.extend(f"  - {name}" for name in names)
    return "\n".join(lines) if lines else "Empty bundle."


def default_artifact_for_bundle(bundle: Any) -> str | None:
    """The artifact to open after a bundle load: the report when present,
    else the first artifact. Never the local shape's ``run`` record."""
    names = _artifact_names(bundle)
    if not names:
        return None
    for preferred in ("report_v1.md", "report.md"):
        if preferred in names:
            return preferred
    return names[0]


def _render_verification(content: Mapping) -> str:
    lines = [f"confidence: {content.get('confidence')}"]
    cv = content.get("citation_verification") or {}
    if cv:
        lines.append(
            f"citations: markers {cv.get('markers_resolved')}/{cv.get('markers_total')}"
            f" · quotes {cv.get('quotes_verified')}/{cv.get('quotes_checked')}"
            f" · uncited sentences {cv.get('uncited_sentences')}"
        )
    gate = content.get("gate")
    if isinstance(gate, Mapping) and gate.get("raw"):
        fallback = " (gate fallback)" if gate.get("fallback") else ""
        lines.append(f"gate: {gate.get('relevant')}/{gate.get('raw')}{fallback}")
    if content.get("relevant_count") is not None:
        lines.append(f"relevant results: {content.get('relevant_count')}")
    return "\n".join(lines)


def _render_claims(content: Mapping) -> str:
    claims = content.get("claims") or []
    lines = [f"claims: {content.get('claim_count', len(claims))}"]
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        text = str(claim.get("text") or "")
        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"  [{claim.get('status', '?')}] {claim.get('claim_id')}: {text}")
    return "\n".join(lines)


def _render_sources(content: Mapping) -> str:
    lines = []
    for item in content.get("evidence") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"[{item.get('id')}] {item.get('title') or 'Untitled'} — {item.get('url') or ''}"
        )
    return "\n".join(lines) if lines else "No sources."


def _render_budget(content: Mapping) -> str:
    return "\n".join(
        [
            f"searches {content.get('searches_used', 0)}",
            f"docs {content.get('docs_used', 0)}",
            f"tokens {content.get('tokens_settled', 0)}"
            + (" (estimated)" if content.get("tokens_estimated") else ""),
            f"runtime {content.get('runtime_elapsed_s', 0.0)}s",
        ]
    )


def _render_content(artifact_name: str, content: Any) -> str:
    if isinstance(content, str):
        text = content
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + "\n… [truncated]"
        return text
    if isinstance(content, Mapping):
        if artifact_name == "verification_summary.json":
            return _render_verification(content)
        if artifact_name == "claims.json":
            return _render_claims(content)
        if artifact_name == "sources.json":
            return _render_sources(content)
        if artifact_name == "budget_ledger.json":
            return _render_budget(content)
    return json.dumps(content, indent=2, sort_keys=True, default=str)


def render_artifact(artifact: Any) -> str:
    """Structured rendering for one loaded artifact."""
    if artifact is None:
        return "No artifact loaded."
    name = str(_field(artifact, "artifact_name", "") or "")
    header = (
        f"Artifact: {name}\n"
        f"Type: {_field(artifact, 'content_type', '')}\n"
        f"Version: {_field(artifact, 'artifact_version', 1)}\n"
        "Content:\n"
    )
    return header + _render_content(name, _field(artifact, "content"))
