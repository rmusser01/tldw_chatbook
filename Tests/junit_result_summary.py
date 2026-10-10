"""Publish JUnit outcome counts and parameter-free failure IDs, never details."""

import html
import os
import sys
from collections import Counter
from pathlib import Path

from Tests.junit_outcome_diff import ReportLoadError, load_outcomes


def escape_annotation(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> None:
    lines = ["## JUnit outcomes", ""]
    for argument in sys.argv[1:] if argv is None else argv:
        path = Path(argument)
        label = html.escape(path.name)
        try:
            outcomes = load_outcomes(path)
        except ReportLoadError:
            status = "missing" if not path.exists() else "unreadable or malformed"
            lines.append(f"- <code>{label}</code>: **incomplete** ({status} report)")
            message = f"{path.name}: incomplete {status} report"
            print("::error::" + escape_annotation(message))
            continue
        counts = Counter(outcomes.values())
        totals = ", ".join(
            f"{k}={counts[k]}" for k in ("pass", "fail", "error", "skip")
        )
        lines.append(f"- <code>{label}</code>: {totals}")
        failures = set()
        for key, value in outcomes.items():
            if value in ("fail", "error"):
                failures.add(key.split("[", 1)[0])
        for identifier in sorted(failures):
            lines.append(f"  - <code>{html.escape(identifier)}</code>")
            print("::error::" + escape_annotation(f"{path.name}: {identifier}"))
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
