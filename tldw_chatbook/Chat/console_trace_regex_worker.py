"""Disposable bounded subprocess for custom trace PII regex batches."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_custom_pii import CustomPIIRule
    from tldw_chatbook.Chat.console_trace_redaction import PIIFieldRedaction


CUSTOM_PII_WORKER_VERSION = 1
CUSTOM_PII_WORKER_INPUT_LIMIT = "custom_pii_input_limit"
CUSTOM_PII_WORKER_FIELD_LIMIT = "custom_pii_field_limit"
CUSTOM_PII_WORKER_RULE_LIMIT = "custom_pii_rule_limit"
CUSTOM_PII_WORKER_MATCH_LIMIT = "custom_pii_match_limit"
CUSTOM_PII_WORKER_OUTPUT_LIMIT = "custom_pii_output_limit"
CUSTOM_PII_WORKER_MEMORY_LIMIT = "custom_pii_memory_limit"
CUSTOM_PII_WORKER_TIMEOUT = "custom_pii_timeout"
CUSTOM_PII_WORKER_CRASH = "custom_pii_crash"
CUSTOM_PII_WORKER_MALFORMED_OUTPUT = "custom_pii_malformed_output"
CUSTOM_PII_WORKER_INVALID_BATCH = "custom_pii_invalid_batch"

_WORKER_REASONS = frozenset(
    {
        CUSTOM_PII_WORKER_INPUT_LIMIT,
        CUSTOM_PII_WORKER_FIELD_LIMIT,
        CUSTOM_PII_WORKER_RULE_LIMIT,
        CUSTOM_PII_WORKER_MATCH_LIMIT,
        CUSTOM_PII_WORKER_OUTPUT_LIMIT,
        CUSTOM_PII_WORKER_MEMORY_LIMIT,
        CUSTOM_PII_WORKER_INVALID_BATCH,
    }
)
_FLAG_BITS = {
    "ascii": re.ASCII,
    "dotall": re.DOTALL,
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
}
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z", re.ASCII)
_FIELD_PATH_PATTERN = re.compile(r"\$(?:/(?:\d+|@\d+(?:#key)?))*\Z", re.ASCII)
_APPLIED_RESPONSE_KEYS = frozenset({"version", "outcome", "matches", "enforced_limits"})
_OMITTED_RESPONSE_KEYS = frozenset({"version", "outcome", "reason", "enforced_limits"})


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate protocol keys instead of accepting last-value wins."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class CustomPIIWorkerLimits:
    """Parent and child bounds for one disposable regex batch."""

    deadline_ms: int = 500
    max_input_bytes: int = 1_048_576
    max_fields: int = 512
    max_field_codepoints: int = 1_000_000
    max_rules: int = 64
    max_matches: int = 10_000
    max_output_bytes: int = 1_048_576
    memory_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        bounds = (
            ("deadline_ms", self.deadline_ms, 10, 5_000),
            ("max_input_bytes", self.max_input_bytes, 32, 8_388_608),
            ("max_fields", self.max_fields, 1, 4_096),
            ("max_field_codepoints", self.max_field_codepoints, 1, 1_000_000),
            ("max_rules", self.max_rules, 1, 64),
            ("max_matches", self.max_matches, 1, 10_000),
            ("max_output_bytes", self.max_output_bytes, 64, 4_194_304),
            ("memory_bytes", self.memory_bytes, 33_554_432, 1_073_741_824),
        )
        for name, value, minimum, maximum in bounds:
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(name)


@dataclass(frozen=True, slots=True)
class CustomPIIBatchResult:
    """Content-free custom-rule ranges or one fail-closed omission."""

    available: bool
    field_redactions: tuple[PIIFieldRedaction, ...]
    omission_reason_code: str | None
    worker_terminated: bool = False
    enforced_limits: tuple[str, ...] = ()


def _unavailable(
    reason: str,
    *,
    terminated: bool = False,
) -> CustomPIIBatchResult:
    return CustomPIIBatchResult(False, (), reason, terminated)


def _rule_payload(rule: CustomPIIRule) -> dict[str, object]:
    return {
        "id": rule.rule_id,
        "category": rule.category,
        "pattern": rule.pattern,
        "flags": list(rule.flags),
    }


def run_custom_pii_batch(
    value: object,
    rules: Sequence[CustomPIIRule],
    *,
    limits: CustomPIIWorkerLimits | None = None,
    worker_path: Path | None = None,
) -> CustomPIIBatchResult:
    """Run one value and ruleset in one new bounded worker process.

    Args:
        value: JSON-like capture component to inspect.
        rules: Validated custom rules frozen for this capture.
        limits: Optional structural and process bounds.
        worker_path: Alternate worker executable used by process-boundary tests.

    Returns:
        Content-free field ranges, or a content-free omission reason. No raw
        field value, matched text, or pattern is copied into the result.
    """

    from tldw_chatbook.Chat.console_trace_custom_pii import CustomPIIRule
    from tldw_chatbook.Chat.console_trace_redaction import (
        PIIFieldRedaction,
        PIIRedactionSpan,
    )

    active_limits = limits or CustomPIIWorkerLimits()
    submitted_rules = tuple(rules)
    if any(not isinstance(rule, CustomPIIRule) for rule in submitted_rules):
        raise TypeError("rules")
    active_rules = tuple(rule for rule in submitted_rules if rule.enabled)
    if len(active_rules) > active_limits.max_rules:
        return _unavailable(CUSTOM_PII_WORKER_RULE_LIMIT)
    if not active_rules:
        return CustomPIIBatchResult(True, (), None)
    try:
        payload = json.dumps(
            {
                "version": CUSTOM_PII_WORKER_VERSION,
                "value": value,
                "rules": [_rule_payload(rule) for rule in active_rules],
                "limits": {
                    "max_fields": active_limits.max_fields,
                    "max_field_codepoints": active_limits.max_field_codepoints,
                    "max_rules": active_limits.max_rules,
                    "max_matches": active_limits.max_matches,
                },
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return _unavailable(CUSTOM_PII_WORKER_INVALID_BATCH)
    if len(payload) > active_limits.max_input_bytes:
        return _unavailable(CUSTOM_PII_WORKER_INPUT_LIMIT)

    executable = (
        Path(__file__).resolve() if worker_path is None else worker_path.resolve()
    )
    command = [
        sys.executable,
        "-I",
        os.fspath(executable),
        "--worker",
        "--memory-bytes",
        str(active_limits.memory_bytes),
        "--output-bytes",
        str(active_limits.max_output_bytes),
        "--cpu-seconds",
        str(max(1, math.ceil(active_limits.deadline_ms / 1_000))),
        "--input-bytes",
        str(active_limits.max_input_bytes),
    ]
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed interpreter and worker path
                command,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=os.name == "posix",
            )
        except OSError:
            return _unavailable(CUSTOM_PII_WORKER_CRASH)
        try:
            process.communicate(
                payload,
                timeout=active_limits.deadline_ms / 1_000,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return _unavailable(CUSTOM_PII_WORKER_TIMEOUT, terminated=True)
        if process.returncode != 0:
            return _unavailable(CUSTOM_PII_WORKER_CRASH)
        size = output.seek(0, os.SEEK_END)
        if size > active_limits.max_output_bytes:
            return _unavailable(CUSTOM_PII_WORKER_OUTPUT_LIMIT)
        output.seek(0)
        raw_output = output.read()

    try:
        decoded = json.loads(raw_output, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    if (
        not isinstance(decoded, Mapping)
        or type(decoded.get("version")) is not int
        or decoded.get("version") != CUSTOM_PII_WORKER_VERSION
    ):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    outcome = decoded.get("outcome")
    expected_keys = (
        _OMITTED_RESPONSE_KEYS
        if outcome == "omitted"
        else _APPLIED_RESPONSE_KEYS
        if outcome == "applied"
        else frozenset()
    )
    if set(decoded) != expected_keys:
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    enforced = decoded.get("enforced_limits", [])
    if (
        not isinstance(enforced, list)
        or any(item not in {"cpu", "memory", "output"} for item in enforced)
        or len(enforced) != len(set(enforced))
    ):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    if outcome == "omitted":
        reason = decoded.get("reason")
        return _unavailable(
            reason
            if type(reason) is str and reason in _WORKER_REASONS
            else CUSTOM_PII_WORKER_MALFORMED_OUTPUT
        )
    if outcome != "applied" or not isinstance(decoded.get("matches"), list):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    try:
        redactions = tuple(
            PIIFieldRedaction(
                field_path=item["field_path"],
                span=PIIRedactionSpan(
                    start_codepoint=item["start_codepoint"],
                    end_codepoint=item["end_codepoint"],
                    category=item["category"],
                    rule_id=item["rule_id"],
                    detector_version="custom-pii-v1",
                ),
            )
            for item in decoded["matches"]
            if isinstance(item, Mapping)
            and set(item)
            == {
                "field_path",
                "start_codepoint",
                "end_codepoint",
                "category",
                "rule_id",
            }
            and type(item["field_path"]) is str
            and len(item["field_path"]) <= 16_384
            and _FIELD_PATH_PATTERN.fullmatch(item["field_path"]) is not None
        )
    except (KeyError, TypeError, ValueError):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    if (
        len(redactions) != len(decoded["matches"])
        or len(redactions) > active_limits.max_matches
    ):
        return _unavailable(CUSTOM_PII_WORKER_MALFORMED_OUTPUT)
    return CustomPIIBatchResult(
        True,
        redactions,
        None,
        enforced_limits=tuple(sorted(enforced)),
    )


def _apply_resource_limits(
    *,
    memory_bytes: int,
    output_bytes: int,
    cpu_seconds: int,
) -> tuple[str, ...]:
    if os.name != "posix":
        return ()
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX Python normally provides it
        return ()
    enforced: list[str] = []
    memory_kinds = (
        (getattr(resource, "RLIMIT_RSS", None), getattr(resource, "RLIMIT_DATA", None))
        if sys.platform == "darwin"
        else (getattr(resource, "RLIMIT_AS", None),)
    )
    for kind in memory_kinds:
        if kind is None:
            continue
        try:
            _soft, hard = resource.getrlimit(kind)
            resource.setrlimit(kind, (memory_bytes, hard))
        except (OSError, ValueError):
            continue
        enforced.append("memory")
        break
    for name, kind, value in (
        ("output", getattr(resource, "RLIMIT_FSIZE", None), output_bytes),
        ("cpu", getattr(resource, "RLIMIT_CPU", None), cpu_seconds),
    ):
        if kind is None:
            continue
        try:
            _soft, hard = resource.getrlimit(kind)
            resource.setrlimit(kind, (value, hard))
        except (OSError, ValueError):
            continue
        enforced.append(name)
    return tuple(enforced)


def _omission(reason: str, enforced: Sequence[str] = ()) -> dict[str, object]:
    return {
        "version": CUSTOM_PII_WORKER_VERSION,
        "outcome": "omitted",
        "reason": reason,
        "enforced_limits": list(enforced),
    }


def _worker_match(request: object, enforced: Sequence[str]) -> dict[str, object]:
    if not isinstance(request, Mapping) or set(request) != {
        "version",
        "value",
        "rules",
        "limits",
    }:
        return _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)
    rules = request.get("rules")
    limits = request.get("limits")
    if (
        request.get("version") != CUSTOM_PII_WORKER_VERSION
        or not isinstance(rules, list)
        or not isinstance(limits, Mapping)
        or set(limits)
        != {"max_fields", "max_field_codepoints", "max_rules", "max_matches"}
    ):
        return _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)
    try:
        max_fields = limits["max_fields"]
        max_field_codepoints = limits["max_field_codepoints"]
        max_rules = limits["max_rules"]
        max_matches = limits["max_matches"]
    except KeyError:
        return _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)
    bounds = (
        (max_fields, 4_096),
        (max_field_codepoints, 1_000_000),
        (max_rules, 64),
        (max_matches, 10_000),
    )
    if any(
        type(value) is not int or not 1 <= value <= maximum for value, maximum in bounds
    ):
        return _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)
    if len(rules) > max_rules:
        return _omission(CUSTOM_PII_WORKER_RULE_LIMIT, enforced)

    compiled: list[tuple[str, str, re.Pattern[str]]] = []
    try:
        for rule in rules:
            if not isinstance(rule, Mapping) or set(rule) != {
                "id",
                "category",
                "pattern",
                "flags",
            }:
                raise ValueError
            flags = rule["flags"]
            rule_id = rule["id"]
            category = rule["category"]
            pattern = rule["pattern"]
            if (
                type(rule_id) is not str
                or not 0 < len(rule_id) <= 64
                or _TOKEN_PATTERN.fullmatch(rule_id) is None
                or type(category) is not str
                or not 0 < len(category) <= 64
                or _TOKEN_PATTERN.fullmatch(category) is None
                or type(pattern) is not str
                or not 0 < len(pattern) <= 2_048
                or not isinstance(flags, list)
                or len(set(flags)) != len(flags)
                or any(
                    type(flag) is not str or flag not in _FLAG_BITS for flag in flags
                )
            ):
                raise ValueError
            bits = 0
            for flag in flags:
                bits |= _FLAG_BITS[flag]
            compiled.append(
                (
                    rule_id,
                    category,
                    re.compile(pattern, bits),
                )
            )
    except (KeyError, TypeError, ValueError, re.error):
        return _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)

    matches: list[dict[str, object]] = []
    field_count = 0

    def inspect(text: str, path: str) -> None:
        nonlocal field_count
        field_count += 1
        if field_count > max_fields:
            raise OverflowError(CUSTOM_PII_WORKER_FIELD_LIMIT)
        if len(text) > max_field_codepoints:
            raise OverflowError(CUSTOM_PII_WORKER_INPUT_LIMIT)
        for rule_id, category, pattern in compiled:
            for match in pattern.finditer(text):
                if match.start() == match.end():
                    raise ValueError(CUSTOM_PII_WORKER_INVALID_BATCH)
                matches.append(
                    {
                        "field_path": path,
                        "start_codepoint": match.start(),
                        "end_codepoint": match.end(),
                        "category": category,
                        "rule_id": rule_id,
                    }
                )
                if len(matches) > max_matches:
                    raise OverflowError(CUSTOM_PII_WORKER_MATCH_LIMIT)

    def visit(item: object, path: str) -> None:
        if type(item) is str:
            inspect(item, path)
            return
        if isinstance(item, Mapping):
            for ordinal, (key, child) in enumerate(sorted(item.items())):
                if type(key) is not str:
                    raise ValueError(CUSTOM_PII_WORKER_INVALID_BATCH)
                inspect(key, f"{path}/@{ordinal}#key")
                visit(child, f"{path}/@{ordinal}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float and math.isfinite(item):
            return
        raise ValueError(CUSTOM_PII_WORKER_INVALID_BATCH)

    try:
        visit(request.get("value"), "$")
    except MemoryError:
        return _omission(CUSTOM_PII_WORKER_MEMORY_LIMIT, enforced)
    except (OverflowError, ValueError) as exc:
        reason = str(exc)
        return _omission(
            reason if reason in _WORKER_REASONS else CUSTOM_PII_WORKER_INVALID_BATCH,
            enforced,
        )
    return {
        "version": CUSTOM_PII_WORKER_VERSION,
        "outcome": "applied",
        "matches": matches,
        "enforced_limits": list(enforced),
    }


def _worker_main(args: argparse.Namespace) -> int:
    enforced = _apply_resource_limits(
        memory_bytes=args.memory_bytes,
        output_bytes=args.output_bytes,
        cpu_seconds=args.cpu_seconds,
    )
    try:
        payload = sys.stdin.buffer.read(args.input_bytes + 1)
        if len(payload) > args.input_bytes:
            response = _omission(CUSTOM_PII_WORKER_INPUT_LIMIT, enforced)
        else:
            response = _worker_match(json.loads(payload), enforced)
    except MemoryError:
        response = _omission(CUSTOM_PII_WORKER_MEMORY_LIMIT, enforced)
    except (UnicodeDecodeError, json.JSONDecodeError):
        response = _omission(CUSTOM_PII_WORKER_INVALID_BATCH, enforced)
    encoded = json.dumps(
        response,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(encoded) > args.output_bytes:
        encoded = json.dumps(
            _omission(CUSTOM_PII_WORKER_OUTPUT_LIMIT),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    try:
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
    except OSError:
        return 3
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--output-bytes", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--input-bytes", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    raise SystemExit(_worker_main(parsed) if parsed.worker else 2)
