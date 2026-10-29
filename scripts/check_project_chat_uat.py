"""Reject incomplete project-chat UAT evidence (stdlib-only, CLI and testable)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Deliberately independent of the parametrization: removing a variant must go red.
EXPECTED = {"fenced", "native", "compact-recovery", "wrong-family"}


def check_reports(junit: Path, artifacts: Path) -> None:
    """Require each exact pytest case and its completed, clean evidence receipt."""
    cases = [
        case
        for case in ET.parse(junit).iter("testcase")
        if case.get("classname") == "Tests.QA.test_project_chat_uat"
    ]
    expected_names = {f"test_project_chat_uat[{variant}]" for variant in EXPECTED}
    assert Counter(case.get("name") for case in cases) == Counter(
        expected_names
    ), "missing or duplicate UAT cases"
    for case in cases:
        assert not any(
            case.find(tag) is not None for tag in ("failure", "error", "skipped")
        ), f"nonpassing UAT case: {case.get('name')}"
    receipts = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in artifacts.glob("*/receipt.json")
    ]
    assert Counter(row.get("variant") for _, row in receipts) == Counter(
        EXPECTED
    ), "missing or duplicate UAT receipts"
    for path, row in receipts:
        assert (
            row.get("case") == "UAT-02"
            and row.get("status") == "passed"
            and row.get("phase") == "complete"
        ), f"unfinished UAT receipt: {path}"
        assert all(
            row.get(key)
            for key in (
                "run_id",
                "conversation_id",
                "assistant_message_id",
                "request_count",
            )
        ), f"missing run evidence: {path}"
        assert (
            row.get("model_errors") == [] and row.get("blocked_destinations") == []
        ), f"UAT protocol/network failure: {path}"
        for name in ("viewport.txt", "viewport.svg"):
            assert (
                path.with_name(name).is_file() and path.with_name(name).stat().st_size
            ), f"missing viewport: {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    check_reports(args.junit, args.artifacts)
    print("Project-chat UAT: all four cases passed with complete receipts.")


if __name__ == "__main__":
    main()
