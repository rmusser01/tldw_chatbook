"""Pure local document checks and proposed edits; no persistence authority."""

from .models import CheckResult, Diagnostic, DocumentRequest, FormatResult
from .source_positions import MAX_DIAGNOSTICS, Budget, DocumentError


def _run(request: DocumentRequest, formatting: bool) -> tuple[CheckResult, str | None]:
    budget = Budget()
    try:
        if request.is_excerpt:
            raise DocumentError(
                "unsupported", "Open the whole document to check or format it."
            )
        if request.language not in ("json", "jsonl", "yaml"):
            raise DocumentError(
                "unsupported", "This language has no structured-document engine."
            )
        budget.source(request.text)
        if request.language == "jsonl" and request.has_bom:
            raise DocumentError("syntax", "JSONL does not allow a byte order mark.")
        if request.language == "yaml":
            from .yaml_document import analyze

            diagnostics, replacement = analyze(request.text, formatting, budget)
        else:
            from .json_document import analyze

            newline = "\r\n" if "\r\n" in request.text else request.newline
            diagnostics, replacement = analyze(
                request.text, request.language, newline, formatting, budget
            )
        if replacement is not None:
            budget.source(replacement)
        budget.check()
        truncated = len(diagnostics) > MAX_DIAGNOSTICS
        result = CheckResult.for_request(
            request,
            "checked",
            not truncated,
            tuple(diagnostics[:MAX_DIAGNOSTICS]),
            truncated,
            not truncated and not diagnostics,
        )
        return result, replacement
    except UnicodeEncodeError as error:
        issue = Diagnostic(
            "syntax",
            "error",
            "Source contains an invalid Unicode scalar.",
            error.start,
            error.end,
        )
        return (
            CheckResult.for_request(request, "checked", True, (issue,), False, False),
            None,
        )
    except DocumentError as error:
        state = (
            error.code
            if error.code in ("unsupported", "resource_limited", "failed")
            else "checked"
        )
        start = min(max(0, error.start), len(request.text))
        end = min(max(start, error.end), len(request.text))
        issue = Diagnostic(error.code, "error", str(error), start, end)
        return (
            CheckResult.for_request(
                request, state, state == "checked", (issue,), False, False
            ),
            None,
        )
    except (MemoryError, RecursionError):
        issue = Diagnostic(
            "resource_limited", "error", "Parser resource limit reached.", 0, 0
        )
        return (
            CheckResult.for_request(
                request, "resource_limited", False, (issue,), False, False
            ),
            None,
        )
    except Exception:  # noqa: BLE001 - source-safe parser boundary
        # Parser exceptions may contain the entire draft. Never log or return them.
        issue = Diagnostic(
            "failed",
            "error",
            "Document engine failed; keep the original draft and retry.",
            0,
            0,
        )
        return (
            CheckResult.for_request(request, "failed", False, (issue,), False, False),
            None,
        )


def check_document(request: DocumentRequest) -> CheckResult:
    """Check one immutable whole-document snapshot without changing its source."""
    return _run(request, False)[0]


def format_document(request: DocumentRequest) -> FormatResult:
    """Perform an authoritative check and propose a verified source edit."""
    check, replacement = _run(request, True)
    if not check.format_allowed:
        state = (
            check.state if check.state in ("resource_limited", "failed") else "blocked"
        )
        return FormatResult(request.key, state, None, check)
    if replacement == request.text:
        return FormatResult(request.key, "unchanged", None, check)
    return FormatResult(request.key, "formatted", replacement, check)
