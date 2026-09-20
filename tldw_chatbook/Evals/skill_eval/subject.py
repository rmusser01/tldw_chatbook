"""Build immutable SkillSubject snapshots. Structure reads only; never executes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

from .models import SkillSubject, digest_skill

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_REF_PATTERN = re.compile(r"(?:references|assets)/[A-Za-z0-9_./-]+")
_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py"}


class SubjectError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def parse_front_matter(content: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(content)
    if not match:
        return {}, content
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    body = content[match.end():]
    return meta, body


def _normalize_allowed_tools(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split()
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        return ()
    return tuple(p for p in (s.strip() for s in parts) if p)


def _bundle_paths_from_manifest(manifest) -> Tuple[str, ...]:
    if not manifest:
        return ()
    return tuple(
        str(entry["path"]) for entry in manifest
        if isinstance(entry, Mapping) and entry.get("path")
    )


def _referenced_files(body: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(_REF_PATTERN.findall(body)))


def _build(name, description, content, body, allowed_tools, referenced,
           bundle_paths, source_kind, source_path, trust_status,
           script_paths) -> SkillSubject:
    return SkillSubject(
        name=name, description=description or "", body=body,
        allowed_tools=allowed_tools, script_paths=script_paths,
        referenced_files=referenced, bundle_paths=bundle_paths,
        source_kind=source_kind, source_path=str(source_path),
        trust_status=trust_status or "unknown",
        digest=digest_skill(name, description or "", content),
        # Count the full SKILL.md document (front matter included), not just the
        # body: line_count is a provenance/size signal for the whole skill file.
        line_count=content.count("\n") + (1 if content and not content.endswith("\n") else 0),
    )


def subject_from_directory(path: str | Path) -> SkillSubject:
    root = Path(path)
    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        raise SubjectError(f"no SKILL.md under {root}")
    content = skill_md.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_front_matter(content)
    bundle = tuple(
        p.relative_to(root).as_posix() for p in sorted(root.rglob("*"))
        if p.is_file() and p != skill_md
    )
    return _build(
        name=str(meta.get("name") or root.name),
        description=str(meta.get("description") or ""),
        content=content, body=body,
        allowed_tools=_normalize_allowed_tools(meta.get("allowed_tools")),
        referenced=_referenced_files(body), bundle_paths=bundle,
        source_kind="directory", source_path=root,
        trust_status="unknown",
        script_paths=tuple(p for p in bundle if p.endswith(".py")),
    )


async def subject_from_store(service: Any, skill_name: str) -> SkillSubject:
    try:
        resp = await service.get_skill(skill_name)
    except Exception as exc:  # missing skill surfaces as many shapes; normalize
        raise SubjectError(f"skill {skill_name!r} not readable: {exc}") from exc
    content = str(resp.get("content") or "")
    meta, body = parse_front_matter(content)
    bundle = _bundle_paths_from_manifest(resp.get("bundle_files"))
    return _build(
        name=str(resp.get("name") or skill_name),
        description=str(resp.get("description") or meta.get("description") or ""),
        content=content, body=body,
        allowed_tools=_normalize_allowed_tools(meta.get("allowed_tools")),
        referenced=_referenced_files(body), bundle_paths=bundle,
        source_kind="store", source_path=str(resp.get("record_id") or skill_name),
        trust_status=str(resp.get("trust_status") or "unknown"),
        script_paths=tuple(
            str(e["path"]) for e in (resp.get("bundle_files") or [])
            if isinstance(e, Mapping) and str(e.get("path", "")).endswith(".py")
        ),
    )
