"""Local archive budgets shared by capture and inspection (ADR-126)."""

from dataclasses import dataclass


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
