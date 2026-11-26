"""Select dev PR checks: sanity checks by default, broad CI for CI-shaping inputs.

Owner decision (TASK-42521, 2026-11-23): a dev PR runs sanity checks (preflight,
Tests/CI, the smoke set, its own changed test files and its subsystem's smoke
tests) and merges; the nightly full run on dev and the release run into main
are the broad gates. Only inputs that can break CI itself, or the schema, keep
the full run on the PR; so does anything a conftest.py loads, which starts
every suite below it. A widely imported test module or helper, or a selection
over the light job's cap, runs a capped, deterministic sample instead
(TASK-42626); the nightly and release runs cover the rest.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shlex
from pathlib import Path
import subprocess
import textwrap
import warnings

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
# At most this many runnable test files that import (or read) one changed
# module (or data file) run on the PR, breadth first; nightly and release run
# the rest.
IMPORTER_CAP = 8
# The PR's own selected files (its changed tests first, then the samples in
# turn): more would not fit the light job's 30-minute cap run one by one.
LIGHT_FILE_CAP = 20
# Application code that creates, alters or versions its own SQLite schema
# (uppercase DDL and PRAGMA, as this codebase writes them; case-sensitive to
# skip prose, and PRAGMA-only so a Python name `user_version` does not count).
SCHEMA_TEXT = re.compile(
    r"\b(?:ALTER TABLE|CREATE (?:TABLE|UNIQUE INDEX|INDEX|TRIGGER|VIEW|VIRTUAL TABLE))\b"
    r"|\bPRAGMA[ \t]+user_version\b"
)
# The PR lanes: the jobs the required check needs, and the UI lane's census.
LANE_WORKFLOW = ".github/workflows/derived-artifacts.yml"
UI_CENSUS = "scripts/ui_pr_gate_census.txt"
NOT_LANES = {"preflight", "dev-light", "release-qualification"}
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


@functools.lru_cache(maxsize=None)
def _imports(name: str, text: str) -> frozenset[str] | None:
    """Paths that loading the file `name` loads, as pytest's prepend import mode
    resolves its imports, or None when it does not parse.

    Covers the importer's own package `__init__.py` files, `Tests.*` absolute
    imports, relative imports, and bare imports of a module in the importer's
    own directory. Nested imports count, and so do the imports of a child
    script embedded as a string. Every package `__init__.py` on the way counts
    too. Paths no file has are kept, so a deleted module still counts.
    """
    tree = _parse(text)
    if tree is None:
        return None
    here = list(Path(name).parent.parts)
    paths: set[str] = set()

    def add(parts: list[str]) -> None:
        for end in range(1, len(parts) + 1):
            base = "/".join(parts[:end])
            paths.update((f"{base}.py", f"{base}/__init__.py"))

    add(here)
    trees = [tree]
    for node in (node for tree in trees for node in ast.walk(tree)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "import" in node.value:  # a child script the test runs
                script = textwrap.dedent(node.value)
                child = _parse(script)
                # A template with placeholders: its import lines, one by one.
                lines = [] if child else script.splitlines()
                imports = [line.strip() for line in lines]
                imports = [
                    line for line in imports if line.startswith(("import ", "from "))
                ]
                for sub in [child, *map(_parse, imports)]:
                    if sub is not None:
                        trees.append(sub)
            continue
        if isinstance(node, ast.Import):
            specs = [(0, alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            specs = [(node.level, node.module or "", [a.name for a in node.names])]
        else:
            continue
        for level, module, names in specs:
            dotted = module.split(".") if module else []
            if level:
                base = here[: len(here) - level + 1] + dotted
            elif dotted[:1] == ["Tests"]:
                base = dotted
            else:
                base = here + dotted  # a bare import of a module beside the importer
            add(base)
            for alias in names:
                add([*base, alias])  # `from pkg import mod` loads pkg/mod.py
    return frozenset(paths)


def _parse(text: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # invalid escapes in test strings
            return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _importers(
    path: str,
    corpus: dict[str, str],
    runnable=_is_test_file,
    first: list[str] | None = None,
) -> list[str]:
    """Files under Tests/ that load `path`, transitively, breadth first in corpus
    order: direct importers (or `first`, such as a data file's readers) before
    theirs. Unrunnable files are followed but not counted; the search stops
    once it has IMPORTER_CAP + 1 runnable ones. A file that does not parse
    counts as an importer, which errs toward more tests.
    """
    found: list[str] = []
    seen = {path}
    count = 0
    queue = [path]
    for current in queue:  # appended to while iterating: breadth first
        init = Path(current).name == "__init__.py"
        token = Path(current).parent.name if init else Path(current).stem
        package = f"{Path(current).parent}/"
        level = (
            first
            if current == path and first is not None
            else (
                name
                for name, text in corpus.items()
                if (token in text or (init and name.startswith(package)))
                and _loads(name, text, current)
            )
        )
        for name in level:
            if name in seen:
                continue
            seen.add(name)
            found.append(name)
            queue.append(name)
            count += bool(runnable(name))
            if count > IMPORTER_CAP:
                return found
    return found


def _loads(name: str, text: str, target: str) -> bool:
    imported = _imports(name, text)
    return imported is None or target in imported


def _conftest_loaded(corpus: dict[str, str]) -> set[str]:
    """Paths a conftest.py loads, transitively: they run at the start of every
    suite below that conftest, as it does. Missing paths count (see _imports)."""
    loaded: set[str] = set()
    queue = [name for name in corpus if Path(name).name == "conftest.py"]
    for name in queue:  # appended to while iterating
        for path in _imports(name, corpus[name]) or ():
            if path not in loaded:
                loaded.add(path)
                if path in corpus:
                    queue.append(path)
    return loaded


def _run_words(step: str) -> list[str]:
    """Shell words of a workflow step's `run:` script (never its name or env)."""
    match = re.search(r"(?m)^(?:[ ]{8})?run:[ \t]*(.*)$", step)
    if not match:
        return []
    lines = [match.group(1).lstrip("|>-")]
    for line in step[match.end() :].splitlines()[1:]:
        if line.strip() and not line.startswith(" " * 9):
            break  # the next key of this step
        lines.append(line)
    script = "\n".join(lines).replace("\\\n", " ")
    try:
        return shlex.split(script, comments=True)
    except ValueError:  # an unbalanced quote: fall back to plain words
        script = re.sub(r"(?m)(^|\s)#.*$", r"\1", script)
        return [re.sub(r"[\"']", "", word) for word in script.split()]


def _lanes(root: Path) -> tuple[dict[str, set[str]], set[str], set[str]]:
    """What the PR lanes run: target -> step ids, the ids they deselect, and the
    targets other jobs (push to dev, nightly) name."""
    path = root / LANE_WORKFLOW
    workflow = path.read_text(errors="replace") if path.is_file() else ""
    parts = re.split(r"(?m)^  ([A-Za-z0-9_-]+):[ \t]*\n", workflow)
    jobs = dict(zip(parts[1::2], parts[2::2], strict=True))
    needs = re.search(r"needs:\s*\[([^\]]*)\]", jobs.get("derived-artifacts", ""))
    lane_jobs = (
        {n.strip() for n in needs.group(1).split(",")} - NOT_LANES if needs else set()
    )
    steps: dict[str, set[str]] = {}
    deselected: set[str] = set()
    other: set[str] = set()
    for job, body in jobs.items():
        for number, step in enumerate(re.split(r"(?m)^      - ", body)):
            words = _run_words(step)
            for previous, word in zip(["", *words], words, strict=False):
                if job in lane_jobs and word.startswith("--deselect="):
                    deselected.add(word.split("=", 1)[1])
                elif job in lane_jobs and previous == "--deselect":
                    deselected.add(word)
                elif not word.startswith("Tests/") or previous == "--deselect":
                    continue
                elif job in lane_jobs:
                    steps.setdefault(word, set()).add(f"{job}#{number}")
                else:
                    other.add(word)
    census = root / UI_CENSUS
    if census.is_file():
        for line in census.read_text(errors="replace").splitlines():
            if line.strip().startswith("Tests/"):
                steps.setdefault(line.strip(), set()).add("ui-census")
    return steps, deselected, other


def _names(path: str, targets: set[str]) -> set[str]:
    return {t for t in targets if t == path or t.startswith(path + "::")}


def select_changes(changes: list[tuple[str, str]], root: Path) -> dict:
    """Return the broad fallback or a nonempty, deterministic sanity-check list."""
    if not changes:
        return {
            "full": True,
            "targets": BASE_TESTS,
            "reason": "No trustworthy change range",
        }

    def broad(reason: str) -> dict:
        return {"full": True, "targets": BASE_TESTS, "reason": reason}

    corpus: dict[str, str] = {}

    def tests_corpus() -> dict[str, str]:
        if not corpus:
            # Sorted, so every machine draws the same sample.
            corpus.update(
                sorted(
                    (str(f.relative_to(root)), f.read_text(errors="replace"))
                    for f in (root / "Tests").rglob("*.py")
                )
            )
        return corpus

    loaded: set[str] = set()

    def conftest_loaded() -> set[str]:
        if not loaded:
            loaded.update(_conftest_loaded(tests_corpus()))
        return loaded

    steps, deselected, other = _lanes(root)
    step_set = set(steps)
    lane_dirs = {
        t
        for t in steps
        if "::" not in t and not t.endswith(".py") and (root / t).is_dir()
    }

    def lane_nodes(path: str) -> set[str]:
        """The node ids the PR lanes run this file by; empty means run it whole."""
        named = _names(path, step_set)
        if path in named or any(
            path.startswith(d.rstrip("/") + "/") for d in lane_dirs
        ):
            return set()
        return named

    def lane_reason(path: str) -> str | None:
        """Why the light job cannot run this test file as the PR lanes do, or None."""
        if _names(path, deselected):
            return f"Test file a PR lane runs with deselected nodes: {path}"
        if not _names(path, step_set) and _names(path, other):
            return f"Test file only push or nightly jobs run: {path}"
        if len(set().union(*(steps[t] for t in lane_nodes(path)))) > 1:
            return f"Test file split across PR lane steps: {path}"
        return None

    @functools.cache
    def runnable(path: str) -> bool:
        """A related file the light job can run; Tests/CI runs whole anyway."""
        return (
            _is_test_file(path)
            and not path.startswith("Tests/CI/")
            and not _full_reason("M", path)
            and not lane_reason(path)
        )

    targets = set(BASE_TESTS)
    changed: list[str] = []  # the PR's own changed test files, kept first
    # Per changed module or data file: path, kind, sample, runnable found, skipped.
    samples: list[tuple[str, str, list[str], int, int]] = []
    for status, path in changes:
        reason = _full_reason(status, path)
        if reason:
            return broad(reason)
        if path.startswith("Tests/"):
            first = None
            if path.endswith(".py"):
                if path in conftest_loaded():
                    return broad(f"Loaded by conftest.py: {path}")
                if (
                    _is_test_file(path)
                    and status != "D"
                    and not path.startswith("Tests/CI/")
                ):
                    reason = lane_reason(path)
                    if reason:
                        return broad(reason)
                    changed.append(path)
            else:
                # Test data, changed or deleted: the files that name it read it, and
                # a helper among them passes it on to the files importing it.
                name = Path(path).name
                first = [f for f, text in tests_corpus().items() if name in text]
                loading = next(
                    (
                        f
                        for f in first
                        if f in conftest_loaded() or Path(f).name == "conftest.py"
                    ),
                    None,
                )
                if loading:
                    return broad(f"Read by {loading}, which conftest.py loads: {path}")
            # Sample the files that depend on it. One the light job cannot run is
            # skipped (and noted), but something must run.
            found = _importers(path, tests_corpus(), runnable, first)
            tests = [f for f in found if runnable(f)]
            covered = any(_is_test_file(f) and f.startswith("Tests/CI/") for f in found)
            if found and not tests and not covered:
                return broad(f"No test file the light job can run depends on: {path}")
            skipped = sum(
                _is_test_file(f) and not f.startswith("Tests/CI/") and not runnable(f)
                for f in found
            )
            kind = "importers" if first is None else "readers"
            samples.append((path, kind, tests[:IMPORTER_CAP], len(tests), skipped))
        elif status == "D":
            continue
        elif path.startswith("tldw_chatbook/") and path.endswith(".py"):
            source = root / path
            if source.is_file() and SCHEMA_TEXT.search(
                source.read_text(errors="replace")
            ):
                return broad(f"Schema code: {path}")
        subsystem = next((key for key in SUBSYSTEM_TESTS if path.startswith(key)), None)
        if subsystem and Path(path).suffix in {".py", ".tcss"}:
            targets.update(SUBSYSTEM_TESTS[subsystem])
        # Docs, backlog, census files and other sources: preflight + smoke only.
    changed.sort()
    own = changed[:LIGHT_FILE_CAP]
    for rank in range(IMPORTER_CAP):  # the slots left, shared out in turn
        for _, _, sample, _, _ in samples:
            if rank < len(sample) and sample[rank] not in own:
                if len(own) < LIGHT_FILE_CAP:
                    own.append(sample[rank])
    notes = []
    for path, kind, sample, total, skipped in samples:
        ran = sum(f in own for f in sample)
        if ran < len(sample) or total > IMPORTER_CAP or skipped:
            more = "+" if total > IMPORTER_CAP else ""
            note = f"sampled {ran} of {total}{more} {kind} of {path}"
            if skipped:
                note += f", skipping {skipped} the light job cannot run"
            notes.append(note)
    cut = changed[LIGHT_FILE_CAP:]
    if cut or any(f not in own for _, _, sample, _, _ in samples for f in sample):
        unrun = f", not running the PR's own {', '.join(cut)}" if cut else ""
        notes.insert(0, f"cut to {LIGHT_FILE_CAP} test files{unrun}")
    whole: set[str] = set()
    for path in own:
        nodes = lane_nodes(path)
        if nodes:
            targets |= nodes  # the lanes run this file only by node: never whole
        else:
            whole.add(path)  # run whole, as a lane does (or as no lane does yet)
    targets |= whole
    # A file selected whole must not be narrowed to its own smoke nodes
    # (pytest collapses `f f::n` to `f::n`).
    targets = {t for t in targets if "::" not in t or t.split("::", 1)[0] not in whole}
    for target in targets:
        if not (root / target.split("::", 1)[0]).is_file():
            raise ValueError(f"Selected test does not exist: {target}")
    sampled = "".join(f"{note}; " for note in notes)  # what ran, skipped or was cut
    return {
        "full": False,
        "targets": sorted(targets),
        "reason": f"Sanity checks; {sampled}nightly and release runs are the broad gates",
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
