"""Select reviewed dev checks; unknown or high-risk changes retain broad CI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess

# Representative subsystem checks; broader integration feedback moves to nightly.
# Shared inputs and authority/storage changes below never use this map.
SUBSYSTEM_TESTS = {
    "tldw_chatbook/Chat/": [
        "Tests/Chat/test_chat_functions.py::test_chat_api_handler_matrix_strips_or_preserves_ephemeral_marker",
        "Tests/Chat/test_chat_functions.py::test_handler_projection_matches_actual_dispatch_for_every_registered_endpoint",
    ],
    "tldw_chatbook/UI/": [
        "Tests/UI/test_screen_navigation.py::test_lazy_screen_registry_resolves_visible_shell_destinations",
        "Tests/UI/test_screen_navigation.py::test_main_navigation_route_ids_match_shell_destinations",
    ],
    "tldw_chatbook/Widgets/": ["Tests/Widgets/test_detach_safe_text_area.py"],
    "tldw_chatbook/Notes/": [
        "Tests/Notes/test_notes_library_unit.py",
        "Tests/Notes/test_notes_sync_models.py",
    ],
    "tldw_chatbook/css/": ["Tests/UI/test_design_token_governance.py"],
}
HIGH_RISK_NAMES = (
    "approval",
    "permission",
    "authority",
    "credential",
    "encrypt",
    "recovery",
    "migration",
    "storage",
    "workspace",
    "instruction",
    "agent_bridge",
    "settings",
    "sync",
    "replica",
    "reconcil",
    "executor",
    "journal",
    "file_notes",
)
BASE_TESTS = [
    "Tests/test_smoke.py",
    "Tests/MCP/test_approval_timeout_policy.py",
    "Tests/Chat/test_console_approval_scope_journeys.py",
]


def select_changes(changes: list[tuple[str, str]], root: Path) -> dict:
    """Return broad fallback or a nonempty, deterministic affected-test list."""
    targets = set(BASE_TESTS)
    known_tests = {
        target
        for tests in SUBSYSTEM_TESTS.values()
        for target in tests
        if "::" not in target
    }
    if not changes:
        return {
            "full": True,
            "targets": BASE_TESTS,
            "reason": "No trustworthy change range",
        }
    for status, path in changes:
        if status not in {"A", "M"}:
            return {
                "full": True,
                "targets": BASE_TESTS,
                "reason": f"Structural change: {path}",
            }
        subsystem = next((key for key in SUBSYSTEM_TESTS if path.startswith(key)), None)
        high_risk = any(name in path.lower() for name in HIGH_RISK_NAMES)
        if (
            subsystem
            and status == "M"
            and not high_risk
            and Path(path).suffix in {".py", ".tcss"}
        ):
            targets.update(SUBSYSTEM_TESTS[subsystem])
        elif path in known_tests and not high_risk:
            targets.add(path)
        elif path.startswith("backlog/tasks/") and path.endswith(".md"):
            # The unconditional preflight checks task IDs/files; runtime inputs
            # elsewhere under backlog/ and Docs/ remain broad.
            continue
        else:
            return {
                "full": True,
                "targets": BASE_TESTS,
                "reason": f"Unclassified or shared input: {path}",
            }
    for target in targets:
        if not (root / target.split("::", 1)[0]).is_file():
            raise ValueError(f"Selected test does not exist: {target}")
    return {
        "full": False,
        "targets": sorted(targets),
        "reason": "Reviewed source/test map",
    }


def changed_paths(base: str, head: str) -> list[tuple[str, str]]:
    """Include both sides of moves as deletion/addition; never hide a removal."""
    if not all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in (base, head)):
        raise ValueError("Expected immutable commit SHAs")
    # Checkout needs the tested merge and its base, not the repository's history.
    # Fetch only the immutable base if a shallow checkout did not include it.
    if subprocess.run(
        ["git", "cat-file", "-e", f"{base}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base],
            check=True,
            capture_output=True,
        )
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-status", "-z", base, head],
        check=True,
        capture_output=True,
    )
    fields = result.stdout.decode("utf-8").split("\0")
    if fields.pop() != "" or len(fields) % 2:
        raise ValueError("Malformed git diff")
    return list(zip(fields[::2], fields[1::2], strict=True))


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    kind = os.environ["GITHUB_EVENT_NAME"]
    enabled = os.environ.get("TIERED_ENABLED") == "true"
    pr = event.get("pull_request", {})
    # Force nightly/release/manual qualification, regardless of the opt-in.
    force_full = any(label.get("name") == "ci:full" for label in pr.get("labels", []))
    nightly = kind == "workflow_dispatch" and event.get("inputs", {}).get(
        "nightly"
    ) in (True, "true")
    light = (
        enabled
        and not force_full
        and (
            (kind == "pull_request" and pr.get("base", {}).get("ref") == "dev")
            or (kind == "push" and os.environ["GITHUB_REF"] == "refs/heads/dev")
        )
    )
    plan = {
        "full": True,
        "targets": BASE_TESTS,
        "reason": "Broad policy or release/manual event",
    }
    if light and kind == "push":
        # PR qualification already covers the diff; this checks the merged tree.
        # Activation requires protected dev with PR-only updates (ADR-360).
        plan = {"full": False, "targets": BASE_TESTS, "reason": "Merged dev smoke"}
    elif light:
        base = pr.get("base", {}).get("sha") if pr else event.get("before")
        head = os.environ["GITHUB_SHA"]
        plan = select_changes(changed_paths(base or "", head), Path.cwd())
    plan["light"] = light and not plan["full"]
    if nightly:
        # Qualify every smoke selection before opt-in and keep it qualified nightly.
        plan["light"] = True
        plan["targets"] = sorted(set(BASE_TESTS).union(*SUBSYSTEM_TESTS.values()))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in ("full", "light", "targets"):
            output.write(f"{key}={json.dumps(plan[key], separators=(',', ':'))}\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
