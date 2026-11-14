"""Local archive budgets shared by capture and inspection (ADR-126)."""

from dataclasses import dataclass

# Recovery descriptors contain per-file identities and local target paths, so
# they are larger than bootstrap enrollment records or archive manifests.
RECOVERY_RECORD_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class ArchiveLimits:
    input_bytes: int = 2 * 1024**4
    decrypted_bytes: int = 2 * 1024**4
    expanded_bytes: int = 1024**4
    member_bytes: int = 256 * 1024**3
    members: int = 100_000
    manifest_bytes: int = 16 * 1024**2
    path_bytes: int = 1024

    reviewed_digest: str | None = None
    reviewed_expanded_bytes: int | None = None

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                value
                for name, value in vars(self).items()
                if not name.startswith("reviewed_")
            )
        ):
            raise ValueError("invalid_archive_limits")
        if (self.reviewed_digest is None) != (self.reviewed_expanded_bytes is None):
            raise ValueError("invalid_compression_review")
        if self.reviewed_digest is not None and (
            type(self.reviewed_digest) is not str
            or len(self.reviewed_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.reviewed_digest)
            or type(self.reviewed_expanded_bytes) is not int
            or self.reviewed_expanded_bytes < 0
        ):
            raise ValueError("invalid_compression_review")


def _read_limits(limits, byte_budget, *, kind="preview"):
    limits = ArchiveLimits() if limits is None else limits
    if type(limits) is not ArchiveLimits:
        raise TypeError(f"invalid_{kind}_limits")
    byte_budget = limits.expanded_bytes if byte_budget is None else byte_budget
    if type(byte_budget) is not int or not 0 < byte_budget <= limits.expanded_bytes:
        raise ValueError(f"invalid_{kind}_budget")
    return limits, byte_budget


class _TraceReadLimit(RuntimeError):
    """Body-free materialization refusal shared without importing Chat services."""


# Explicit installed owners; identifiers grant no path or maintenance authority.
_RAW_RECOVERY_LIMITS = {
    "research.paste_staging": 64 * 1024,
    "collections.archives": 1,
    "rag.definitions": 16 * 1024**2,
    "rag.projections": 256 * 1024**3,
    "recovered.media": 512 * 1024**2,
    "plugins": 256 * 1024**3,
    "generation.assets": 256 * 1024**3,
    "diagnostics.logs": 256 * 1024**3,
    "tokenizers.custom": 256 * 1024**3,
    "skills": 256 * 1024**3,
    "persona.assets": 256 * 1024**3,
    "persona.actor_packs": 256 * 1024**3,
    "persona.visual_identity": 256 * 1024**3,
    "persona.visual_identity_builtin": 256 * 1024**3,
    "tts.voices": 256 * 1024**3,
    "models.artifacts": 256 * 1024**3,
    "config": 16 * 1024**2,
    "config.history": 16 * 1024**2,
    "personas": 256 * 1024**3,
    "chat.dictionary_history": 256 * 1024**3,
    "chat.rag_context": 256 * 1024**3,
    "chat.grammars": 256 * 1024**3,
    "feedback": 256 * 1024**3,
    "audio.history": 256 * 1024**3,
    "chat.prompt_history": 256 * 1024**3,
    "ui.state": 256 * 1024**3,
    "ui.emoji_recents": 256 * 1024**3,
    "ui.themes": 256 * 1024**3,
    "chatbooks.registry": 256 * 1024**3,
    "chatbooks.archives": 256 * 1024**3,
    "chat.dictionaries": 256 * 1024**3,
    "chunking.templates": 256 * 1024**3,
    "notes.templates": 256 * 1024**3,
    "chat.prompts": 256 * 1024**3,
    "generation.styles": 256 * 1024**3,
    "external.files": 256 * 1024**3,
    "eval.definitions": 1024**4,
    "mcp.local": 16 * 1024**2,
    "mcp.targets": 16 * 1024**2,
    "mcp.context": 16 * 1024**2,
    "mcp.permissions": 16 * 1024**2,
    "mcp.history": 256 * 1024**3,
    "runtime.source_state": 16 * 1024**2,
    "tamagotchi.config": 16 * 1024**2,
    "workspaces.change_tracking": 256 * 1024**3,
    "agents.history": 256 * 1024**3,
    "subscriptions.assets": 256 * 1024**3,
}
