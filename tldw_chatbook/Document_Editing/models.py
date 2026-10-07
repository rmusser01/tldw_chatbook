"""Immutable wire contracts for local structured-document workers."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RequestKey:
    authority_id: str
    document_id: str
    session_generation: int
    draft_revision: int
    language_revision: int
    request_id: int


@dataclass(frozen=True)
class DocumentRequest:
    key: RequestKey
    text: str
    language: str | None
    newline: Literal["\n", "\r\n"] = "\n"
    has_bom: bool = False
    is_excerpt: bool = False


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning"]
    message: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start <= self.end
        ):
            raise ValueError(
                "Diagnostic ranges must be ordered nonnegative scalar offsets"
            )


@dataclass(frozen=True)
class CheckResult:
    key: RequestKey
    state: Literal["checked", "unsupported", "resource_limited", "failed"]
    complete: bool
    diagnostics: tuple[Diagnostic, ...]
    truncated: bool
    format_allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.complete and self.state != "checked":
            raise ValueError("Only checked results can be complete")
        if self.format_allowed and (
            self.state != "checked"
            or not self.complete
            or self.truncated
            or any(
                d.severity == "error" or d.code == "duplicate_key"
                for d in self.diagnostics
            )
        ):
            raise ValueError("Formatting requires a complete qualified check")

    @classmethod
    def for_request(
        cls,
        request: DocumentRequest,
        state: Literal["checked", "unsupported", "resource_limited", "failed"],
        complete: bool,
        diagnostics: tuple[Diagnostic, ...],
        truncated: bool,
        format_allowed: bool,
    ) -> "CheckResult":
        """Build the result for an immutable source snapshot."""
        result = cls(
            request.key, state, complete, diagnostics, truncated, format_allowed
        )
        if request.is_excerpt and (
            result.complete or result.state == "checked" or result.format_allowed
        ):
            raise ValueError("Excerpts cannot report successful whole-document checks")
        if any(d.end > len(request.text) for d in result.diagnostics):
            raise ValueError("Diagnostic range exceeds source snapshot")
        return result


@dataclass(frozen=True)
class FormatResult:
    key: RequestKey
    state: Literal["formatted", "unchanged", "blocked", "resource_limited", "failed"]
    replacement: str | None
    check: CheckResult

    def __post_init__(self) -> None:
        if self.key != self.check.key:
            raise ValueError("Format and check identities must match")
        if (self.state == "formatted") != (self.replacement is not None):
            raise ValueError("Only formatted results carry a replacement")
        if self.state in ("formatted", "unchanged") and not self.check.format_allowed:
            raise ValueError(
                "Successful formatting requires an authoritative qualified check"
            )
