"""Reject incomplete project-chat UAT evidence (stdlib-only, CLI and testable)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Deliberately independent of the parametrization: removing a variant must go red.
EXPECTED = {"fenced", "native", "compact-recovery", "wrong-family"}


def _require(condition: bool, message: str = "") -> None:
    """Keep evidence checks active under Python optimization."""
    if not condition:
        raise AssertionError(message)


def check_reports(junit: Path, artifacts: Path) -> None:
    """Require each exact pytest case and its completed, clean evidence receipt."""
    cases = [
        case
        for case in ET.parse(junit).iter("testcase")
        if case.get("classname") == "Tests.QA.test_project_chat_uat"
    ]
    expected_names = {f"test_project_chat_uat[{variant}]" for variant in EXPECTED}
    _require(
        Counter(case.get("name") for case in cases) == Counter(expected_names),
        "missing or duplicate UAT cases",
    )
    for case in cases:
        _require(
            not any(
                case.find(tag) is not None for tag in ("failure", "error", "skipped")
            ),
            f"nonpassing UAT case: {case.get('name')}",
        )
    receipts = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in artifacts.glob("*/receipt.json")
    ]
    _require(
        Counter(row.get("variant") for _, row in receipts) == Counter(EXPECTED),
        "missing or duplicate UAT receipts",
    )
    for path, row in receipts:
        _require(
            row.get("case") == "UAT-02"
            and row.get("status") == "passed"
            and row.get("phase") == "complete",
            f"unfinished UAT receipt: {path}",
        )
        _require(
            all(
                row.get(key)
                for key in (
                    "run_id",
                    "conversation_id",
                    "assistant_message_id",
                    "request_count",
                )
            ),
            f"missing run evidence: {path}",
        )
        _require(
            row.get("model_errors") == [] and row.get("blocked_destinations") == [],
            f"UAT protocol/network failure: {path}",
        )
        for name in ("viewport.txt", "viewport.svg"):
            _require(
                path.with_name(name).is_file()
                and path.with_name(name).stat().st_size > 0,
                f"missing viewport: {path}",
            )


def check_negative_control(junit: Path, artifacts: Path) -> None:
    """The deliberately damaged README must fail after a real file-tool result."""
    reason = "README read did not supply a verification code"
    cases = list(ET.parse(junit).iter("testcase"))
    _require(
        len(cases) == 1 and cases[0].get("name") == "test_project_chat_uat[fenced]"
    )
    _require(
        cases[0].find("error") is None and cases[0].find("skipped") is None,
        "negative control has a teardown error or skip",
    )
    failure = cases[0].find("failure")
    _require(
        failure is not None
        and reason in failure.get("message", "") + (failure.text or ""),
        "negative control failed for another reason",
    )
    receipts = list(artifacts.glob("*/receipt.json"))
    _require(len(receipts) == 1, "negative control receipt missing or duplicated")
    row = json.loads(receipts[0].read_text(encoding="utf-8"))
    _require(row.get("variant") == "fenced" and row.get("status") == "failed")
    _require(
        row.get("negative_control") == "missing-readme"
        and row.get("phase") == "await-answer"
    )
    _require(row.get("blocked_destinations") == [])
    errors = row.get("model_errors", [])
    _require(
        bool(errors) and all(error == "AssertionError: " + reason for error in errors),
        "negative control protocol failure was unexpected",
    )
    messages = [
        message
        for payload in row.get("requests", [])
        for message in payload.get("messages", [])
    ]
    _require(
        any(
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("Tool result for fs_read:")
            and "README deliberately lacks verification evidence." in message["content"]
            for message in messages
        ),
        "negative control never reached its file read",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.negative_control:
        check_negative_control(args.junit, args.artifacts)
        print(
            "Project-chat UAT: missing README evidence correctly rejected after fs_read."
        )
    else:
        check_reports(args.junit, args.artifacts)
        print("Project-chat UAT: all four cases passed with complete receipts.")


if __name__ == "__main__":
    main()
