"""Fixed pure-engine entry point and bounded JSON wire validation."""

import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path

if __name__ == "__main__":
    # Import the installed pure subpackage without the application's initializer.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Document_Editing.models import (
        CheckResult,
        Diagnostic,
        DocumentRequest,
        FormatResult,
        RequestKey,
    )
else:
    from .models import (
        CheckResult,
        Diagnostic,
        DocumentRequest,
        FormatResult,
        RequestKey,
    )

MAX_BYTES = 1024 * 1024
MAX_FRAME = 6 * MAX_BYTES + 65536
MESSAGES = {
    "blank_record": "Blank JSONL records are not allowed.",
    "syntax": "Invalid document syntax.",
    "duplicate_key": "Duplicate mapping key; formatting is blocked.",
    "unsupported": "This source representation is not qualified for checking or formatting.",
    "resource_limited": "Document processing reached its resource limit.",
    "failed": "Document engine failed; keep the original draft and retry.",
}


def refusal(request, state):
    return CheckResult.for_request(
        request,
        state,
        False,
        (Diagnostic(state, "error", MESSAGES[state], 0, 0),),
        False,
        False,
    )


def admit(request):
    if request.is_excerpt or request.language not in ("json", "jsonl", "yaml"):
        return refusal(request, "unsupported")
    try:
        if (
            len(request.text) > MAX_BYTES
            or len(request.text.encode("utf-8")) > MAX_BYTES
        ):
            return refusal(request, "resource_limited")
    except UnicodeEncodeError:
        return refusal(request, "unsupported")
    return None


def encode(value):
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME:
        raise ValueError("Frame exceeds limit")
    return struct.pack("!I", len(body)) + body


def decode_request(value):
    if (
        type(value) is not dict
        or set(value) != {"operation", "request"}
        or value["operation"] not in ("check", "format")
    ):
        raise ValueError("Invalid operation")
    fields = value["request"]
    if type(fields) is not dict or set(fields) != {
        "key",
        "text",
        "language",
        "newline",
        "has_bom",
        "is_excerpt",
    }:
        raise ValueError("Invalid request")
    key = fields["key"]
    names = {
        "authority_id",
        "document_id",
        "session_generation",
        "draft_revision",
        "language_revision",
        "request_id",
    }
    if type(key) is not dict or set(key) != names:
        raise ValueError("Invalid key")
    for name, item in key.items():
        if name in ("authority_id", "document_id"):
            if type(item) is not str or len(item) > 4096:
                raise ValueError("Invalid identity")
        elif type(item) is not int or not 0 <= item <= 2**53 - 1:
            raise ValueError("Invalid revision")
    if type(fields["text"]) is not str or (
        fields["language"] is not None
        and (type(fields["language"]) is not str or len(fields["language"]) > 100)
    ):
        raise ValueError("Invalid source")
    if fields["newline"] not in ("\n", "\r\n") or any(
        type(fields[x]) is not bool for x in ("has_bom", "is_excerpt")
    ):
        raise ValueError("Invalid representation")
    return value["operation"], DocumentRequest(**{**fields, "key": RequestKey(**key)})


def same_key(value, expected):
    fields = asdict(expected)
    return (
        type(value) is dict
        and value == fields
        and all(type(value[name]) is type(item) for name, item in fields.items())
    )


def decode_check(request, value):
    if (
        type(value) is not dict
        or set(value)
        != {"key", "state", "complete", "diagnostics", "truncated", "format_allowed"}
        or not same_key(value["key"], request.key)
    ):
        raise ValueError("Invalid result identity")
    if value["state"] not in (
        "checked",
        "unsupported",
        "resource_limited",
        "failed",
    ) or any(
        type(value[x]) is not bool for x in ("complete", "truncated", "format_allowed")
    ):
        raise ValueError("Invalid result state")
    if value["truncated"] and value["complete"]:
        raise ValueError("Truncated check cannot be complete")
    items = value["diagnostics"]
    if type(items) is not list or len(items) > 100:
        raise ValueError("Invalid diagnostics")
    diagnostics = []
    for item in items:
        if (
            type(item) is not dict
            or set(item) != {"code", "message", "severity", "start", "end"}
            or item["code"] not in MESSAGES
            or item["severity"] not in ("error", "warning")
            or type(item["message"]) is not str
            or len(item["message"]) > 512
        ):
            raise ValueError("Invalid diagnostic")
        # Never trust returned prose to be free of source, even from a broken child.
        diagnostics.append(Diagnostic(**{**item, "message": MESSAGES[item["code"]]}))
    return CheckResult.for_request(
        request,
        value["state"],
        value["complete"],
        tuple(diagnostics),
        value["truncated"],
        value["format_allowed"],
    )


def decode_result(request, operation, value):
    if operation == "check":
        return decode_check(request, value)
    if (
        type(value) is not dict
        or set(value) != {"key", "state", "replacement", "check"}
        or not same_key(value["key"], request.key)
        or value["state"]
        not in ("formatted", "unchanged", "blocked", "resource_limited", "failed")
    ):
        raise ValueError("Invalid format result")
    replacement = value["replacement"]
    if replacement is not None and (
        type(replacement) is not str or len(replacement.encode("utf-8")) > MAX_BYTES
    ):
        raise ValueError("Invalid replacement")
    check = decode_check(request, value["check"])
    return FormatResult(request.key, value["state"], replacement, check)


def main():
    try:
        size = struct.unpack("!I", sys.stdin.buffer.read(4))[0]
        if size > MAX_FRAME:
            return 2
        body = sys.stdin.buffer.read(size)
        if len(body) != size:
            return 2
        operation, request = decode_request(json.loads(body))
        from Document_Editing.engine import check_document, format_document

        result = (
            check_document(request)
            if operation == "check"
            else format_document(request)
        )
        sys.stdout.buffer.write(encode(asdict(result)))
        sys.stdout.buffer.flush()
        return 0
    except Exception:  # noqa: BLE001 - never expose source-bearing decoder errors
        # A parser/decoder exception may contain source; the parent reports failure.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
