"""Select dev PR checks: sanity checks by default, broad CI for CI-shaping inputs.

Owner decision (TASK-42521, 2026-11-23): a dev PR runs sanity checks (preflight,
Tests/CI, the smoke set, its own changed test files and its subsystem's smoke
tests) and merges; the nightly full run on dev and the release run into main
are the broad gates. Only inputs that can break CI itself, or the schema, keep
the full run on the PR.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess

# Representative subsystem checks; broader integration feedback moves to nightly.
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
# What CI runs, how every suite starts, what gets installed, and the schema.
FULL_PREFIXES = (".github/", "scripts/ci/", "packages/")
FULL_FILES = {"pyproject.toml", "setup.py", "setup.cfg"}
# pytest configuration at any depth changes how every suite below it runs.
FULL_NAMES = {"conftest.py", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg"}
# Schema code: migration directories and the DB layers (TASK-42521 review I2).
SCHEMA_DIRS = {"DB", "db", "migrations"}
# Data read by the guard and budget suites (allowlists, snapshots).
GUARD_DATA_PREFIXES = ("Tests/Architecture/", "Tests/Performance/")
# Files no PR lane runs whole: too slow for the light job's cap, or carrying
# deselected known failures (scripts/library_shell_push_deselect.txt).
HEAVY_TEST_FILES = {
    "Tests/UI/test_library_shell.py",
    "Tests/Backup_Recovery/test_response_rules_restore.py",
}
# A changed test module imported by more test files than this is shared infrastructure.
IMPORTER_CAP = 8
# More files than this would not fit the light job's 30-minute cap serially.
LIGHT_FILE_CAP = 20
BASE_TESTS = [
    "Tests/test_smoke.py",
    "Tests/MCP/test_approval_timeout_policy.py",
    "Tests/Chat/test_console_approval_scope_journeys.py",
]


def _is_test_file(path: str) -> bool:
    name = Path(path).name
    return (
        path.startswith("Tests/") and name.startswith("test_") and name.endswith(".py")
    )


def _full_reason(status: str, path: str) -> str | None:
    """Why this change needs the broad run on the PR itself, or None."""
    parts = Path(path).parts
    if (
        path.startswith(FULL_PREFIXES)
        or path in FULL_FILES
        or parts[-1] in FULL_NAMES
        or re.fullmatch(r"requirements[^/]*\.txt", path)
    ):
        return f"CI, build or test configuration: {path}"
    if parts[0] == "tldw_chatbook" and SCHEMA_DIRS & set(parts[1:-1]):
        return f"Schema input: {path}"
    if path.startswith("Tests/") and path.endswith(".py") and not _is_test_file(path):
        # private_profile.py, real_profile_guard.py, shared rigs: suites import them.
        return f"Shared test infrastructure: {path}"
    if path.startswith(GUARD_DATA_PREFIXES) and not path.endswith(".py"):
        return f"Guard or budget data: {path}"
    if path in HEAVY_TEST_FILES:
        return f"Test file no PR lane runs whole: {path}"
    if status not in {"A", "M", "D"}:
        return f"Structural change: {path}"
    if status == "D" and path.startswith("tldw_chatbook/"):
        # A removed module can break any importer, and no sanity check imports them all.
        return f"Removed application file: {path}"
    return None


def _importers(path: str, root: Path, corpus: dict[str, str]) -> set[str]:
    """Test files whose import lines name this test module (an over-approximation)."""
    if not corpus:
        corpus.update(
            (str(f.relative_to(root)), f.read_text(errors="replace"))
            for f in (root / "Tests").rglob("*.py")
        )
    # ponytail: a text scan of import lines; misses `from . import (\n  test_x)`.
    pattern = re.compile(
        rf"^[ \t]*(?:from|import)[ \t][^\n]*\b{re.escape(Path(path).stem)}\b", re.M
    )
    return {f for f, text in corpus.items() if f != path and pattern.search(text)}


def select_changes(changes: list[tuple[str, str]], root: Path) -> dict:
    """Return the broad fallback or a nonempty, deterministic sanity-check list."""
    targets = set(BASE_TESTS)
    corpus: dict[str, str] = {}
    if not changes:
        return {
            "full": True,
            "targets": BASE_TESTS,
            "reason": "No trustworthy change range",
        }
    for status, path in changes:
        reason = _full_reason(status, path)
        if reason:
            return {"full": True, "targets": BASE_TESTS, "reason": reason}
        if _is_test_file(path):
            importers = _importers(path, root, corpus)
            if importers and (status == "D" or len(importers) > IMPORTER_CAP):
                return {
                    "full": True,
                    "targets": BASE_TESTS,
                    "reason": f"Shared test module ({len(importers)} importers): {path}",
                }
            for importer in importers:
                reason = _full_reason("M", importer)
                if reason:
                    return {"full": True, "targets": BASE_TESTS, "reason": reason}
            if status != "D":
                targets.add(path)
            targets.update(i for i in importers if _is_test_file(i))
            continue
        if status == "D":
            continue
        subsystem = next((key for key in SUBSYSTEM_TESTS if path.startswith(key)), None)
        if subsystem and Path(path).suffix in {".py", ".tcss"}:
            targets.update(SUBSYSTEM_TESTS[subsystem])
        # Docs, backlog, census files and other sources: preflight + smoke only.
    # The light job runs Tests/CI whole, and a file selected whole must not be
    # narrowed to its own smoke nodes (pytest collapses `f f::n` to `f::n`).
    whole = {t for t in targets if "::" not in t}
    targets = {
        t
        for t in targets
        if not t.startswith("Tests/CI/")
        and ("::" not in t or t.split("::", 1)[0] not in whole)
    }
    files = {t.split("::", 1)[0] for t in targets}
    if len(files) > LIGHT_FILE_CAP:
        return {
            "full": True,
            "targets": BASE_TESTS,
            "reason": f"{len(files)} test files exceed the light job's cap",
        }
    for target in targets:
        if not (root / target.split("::", 1)[0]).is_file():
            raise ValueError(f"Selected test does not exist: {target}")
    return {
        "full": False,
        "targets": sorted(targets),
        "reason": "Sanity checks; nightly and release runs are the broad gates",
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
        # Activation requires protected dev with PR-only updates (ADR-366).
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
