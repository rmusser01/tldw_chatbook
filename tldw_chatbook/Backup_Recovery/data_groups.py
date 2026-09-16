"""Installed whole-owner backup selection policy, without storage access.

Group membership is code authority. Inventory dependencies, shared-store labels,
and topology are already owner declarations; resolving them grants no filesystem
authority and does not replace inventory classification or archive validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import StorageItem


@dataclass(frozen=True)
class BackupGroup:
    """One installed user-facing selection of indivisible storage owners."""

    group_id: str
    label: str
    description: str


@dataclass(frozen=True)
class ResolvedGroups:
    """Deterministic requested scope, dependency closure, and support inventory.

    ``requested_groups=None`` preserves the Everything choice. Member and support
    IDs are disjoint and include selected unused/excluded inventory evidence;
    these IDs do not change an item's capture eligibility or completeness.
    """

    requested_groups: tuple[str, ...] | None
    effective_groups: tuple[str, ...]
    required_groups: tuple[str, ...]
    member_ids: tuple[str, ...]
    support_ids: tuple[str, ...]


BACKUP_GROUPS = (
    BackupGroup(
        "settings",
        "Settings",
        "Application configuration, preferences, themes, and retained settings history.",
    ),
    BackupGroup(
        "conversations",
        "Conversations, notes, and personas",
        "The shared conversation, note, persona, study, and quiz database with its related files and assets.",
    ),
    BackupGroup(
        "library",
        "Media library",
        "Ingested media, library collections, and ingestion job records.",
    ),
    BackupGroup(
        "prompts",
        "Prompts and grammars",
        "Saved prompts, templates, prompt history, and generation grammars.",
    ),
    BackupGroup(
        "workspaces",
        "Workspaces and agents",
        "Workspace records, change tracking, agent runs and history, and Kanban boards.",
    ),
    BackupGroup(
        "research",
        "Research and subscriptions",
        "Research projects and results, subscriptions, and briefing audio.",
    ),
    BackupGroup(
        "writing", "Writing", "Writing projects, manuscripts, and related records."
    ),
    BackupGroup(
        "evaluations",
        "Evaluations",
        "Evaluation runs and retained evaluation definitions.",
    ),
    BackupGroup(
        "automation",
        "Schedules, notifications, and sync",
        "Scheduled tasks, notification history, and durable event and sync state.",
    ),
    BackupGroup(
        "tools",
        "Tools and skills",
        "MCP definitions, context, retained permission history, execution history, and installed skills.",
    ),
    BackupGroup(
        "retrieval",
        "Search and retrieval",
        "RAG definitions, projections, indexing records, chunking templates, and custom tokenizers.",
    ),
    BackupGroup(
        "audio",
        "Audio and voices",
        "Audio history, voice definitions, voice profiles, and reference audio.",
    ),
    BackupGroup(
        "generation",
        "Generated media",
        "Generation styles and generated assets admitted by the temporary-media option.",
    ),
    BackupGroup(
        "chatbooks", "Saved chatbooks", "Saved chatbook archives and their registry."
    ),
    BackupGroup(
        "models",
        "Model artifacts",
        "Model artifacts admitted by the explicit model selection.",
    ),
    BackupGroup(
        "external_files",
        "External files",
        "Files in explicitly selected external folders.",
    ),
)

# Keep this exhaustive for shipped durable adapters. Runtime locks, caches,
# temporary scaffolds, device-only state, diagnostics, recovery control, and
# unknown census entries deliberately have no selectable group. The adapter
# coverage test catches a new durable owner needing an explicit policy choice.
_GROUP_OWNERS = {
    "settings": (
        "config",
        "config.history",
        "ui.state",
        "ui.emoji_recents",
        "ui.themes",
        "runtime.source_state",
        "tamagotchi.config",
    ),
    "conversations": (
        "db.chachanotes.primary",
        "study.local",
        "quiz.local",
        "notes.sync_bindings",
        "notes.file_notes",
        "notes.templates",
        "chat.attachments",
        "personas",
        "persona.assets",
        "persona.visual_identity",
        "persona.visual_identity_builtin",
        "chat.dictionaries",
        "chat.dictionary_history",
        "chat.rag_context",
        "feedback",
        "recovered.media",
    ),
    "library": ("db.media.primary", "db.library_collections", "db.library_ingest_jobs"),
    "prompts": (
        "db.prompts.primary",
        "chat.prompts",
        "chat.prompt_history",
        "chat.grammars",
    ),
    "workspaces": (
        "db.workspaces",
        "db.agent_runs",
        "agents.history",
        "workspaces.change_tracking",
        "kanban.local",
    ),
    "research": ("research.local", "db.subscriptions", "subscriptions.assets"),
    "writing": ("writing.local",),
    "evaluations": ("db.evals", "eval.definitions"),
    "automation": (
        "db.scheduled_tasks",
        "notifications.client",
        "runtime.event_state",
        "runtime.sync_state",
    ),
    "tools": (
        "mcp.local",
        "mcp.targets",
        "mcp.context",
        "mcp.permissions",
        "mcp.history",
        "skills",
    ),
    "retrieval": (
        "rag.definitions",
        "rag.projections",
        "db.rag_indexing",
        "chunking.templates",
        "tokenizers.custom",
    ),
    "audio": ("audio.history", "tts.voices", "tts.profile_store", "tts.references"),
    "generation": ("generation.assets", "generation.styles"),
    "chatbooks": ("chatbooks.registry", "chatbooks.archives"),
    "models": ("models.artifacts",),
    "external_files": ("external.files",),
}
_OWNER_GROUPS = {
    owner: group_id for group_id, owners in _GROUP_OWNERS.items() for owner in owners
}


def group_for_owner(owner_id: str) -> str | None:
    """Return the installed group ID, or None without inventing owner coverage.

    Unknown owners retain the inventory's existing unsupported-owner checks;
    returning None never qualifies an unknown payload for capture or restore.
    """
    return _OWNER_GROUPS.get(owner_id)


def _profile_id(item: StorageItem) -> str | None:
    """Identify ordinary profile-owned records without interpreting paths."""
    parts = item.logical_id.split(":", 3)
    if len(parts) >= 3 and parts[0] == "profile" and parts[2] == item.owner:
        return parts[1]
    return None


def resolve_inventory_groups(
    items: tuple[StorageItem, ...], group_ids: tuple[str, ...] | None
) -> ResolvedGroups:
    """Close named group selections over declared owners and dependencies.

    Args:
        items: Owner inventory for the already selected profiles. No paths are
            opened, and archive projections may omit path and topology metadata.
        group_ids: None selects Everything; a nonempty tuple names installed groups.

    Returns:
        Immutable scope with full required groups across every supplied profile.
        Configuration dependencies are same-profile whole-owner support unless
        Settings is selected. Other Settings owners are not implicit support.

    Raises:
        ValueError: Selection is empty, duplicated, malformed, or unknown; logical
            IDs collide; a required edge is absent; or an explicit selection would
            need an included owner without installed group policy.
    """
    if group_ids is not None and (
        type(group_ids) is not tuple
        or not group_ids
        or any(
            type(value) is not str or value not in _GROUP_OWNERS for value in group_ids
        )
        or len(set(group_ids)) != len(group_ids)
    ):
        raise ValueError("invalid_backup_groups")

    by_id = {}
    by_group: dict[str, set[str]] = {}
    shared: dict[str, set[str]] = {}
    config_by_profile: dict[str | None, set[str]] = {}
    for item in items:
        if item.logical_id in by_id:
            raise ValueError("duplicate_logical_id")
        by_id[item.logical_id] = item
        group = group_for_owner(item.owner)
        if group is not None:
            by_group.setdefault(group, set()).add(item.logical_id)
        if item.shared_group is not None:
            shared.setdefault(item.shared_group, set()).add(item.logical_id)
        if item.owner == "config":
            config_by_profile.setdefault(_profile_id(item), set()).add(item.logical_id)

    requested = None if group_ids is None else tuple(sorted(group_ids))
    initial = set(by_group) if requested is None else set(requested)
    effective = set(initial)
    members = {key for group in effective for key in by_group.get(group, ())}
    support: set[str] = set()
    pending = list(members)
    visited: set[str] = set()

    def require(key: str) -> None:
        target = by_id.get(key)
        if target is None:
            raise ValueError("dependency_unavailable")
        # Every admitted ID is already queued for closure. Repeated tree-parent
        # edges must not rescan the complete owner group or config cohort.
        if key in members or key in support:
            return
        if target.owner == "config" and "settings" not in effective:
            additions = config_by_profile[_profile_id(target)] - members - support
            support.update(additions)
        else:
            group = group_for_owner(target.owner)
            if group is None:
                if requested is not None and target.status in {
                    "included",
                    "included_directory",
                }:
                    raise ValueError("unsupported_group_owner")
                # Keep blocking dependency evidence visible to the existing
                # classifier, rather than silently making the subset complete.
                additions = {key} - members - support
                support.update(additions)
            else:
                effective.add(group)
                additions = by_group[group] - members
                members.update(additions)
                support.difference_update(additions)
        pending.extend(additions)

    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        item = by_id[key]
        if item.status == "unused":
            # An empty owner still needs its direct profile config to identify
            # its restore destination. Other historical dependencies stay inert.
            profile = _profile_id(item)
            config = f"profile:{profile}:config"
            if profile is not None and config in item.dependencies:
                if config not in by_id or by_id[config].owner != "config":
                    raise ValueError("dependency_unavailable")
                require(config)
            continue
        if item.status == "intentionally_excluded":
            continue
        for dependency in item.dependencies:
            require(dependency)
        if item.shared_group is not None:
            for related in shared[item.shared_group]:
                require(related)
        if item.metadata is not None:
            require(item.metadata.root_id)
            if item.metadata.parent_id is not None:
                require(item.metadata.parent_id)

    return ResolvedGroups(
        requested,
        tuple(sorted(effective)),
        tuple(sorted(effective - initial)),
        tuple(sorted(members)),
        tuple(sorted(support)),
    )
