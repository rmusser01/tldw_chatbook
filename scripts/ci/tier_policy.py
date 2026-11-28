"""Select dev PR checks: sanity checks by default, broad CI for CI-shaping inputs.

Owner decision (TASK-42521, 2026-11-23): a dev PR runs sanity checks (preflight,
Tests/CI, the smoke set, its own changed test files and its subsystem's smoke
tests) and merges; the nightly full run on dev and the release run into main
are the broad gates. Only inputs that can break CI itself, or the schema, keep
the full run on the PR; so does anything a conftest.py loads, which starts
every suite below it. A widely imported test module, helper or data file runs
a capped, deterministic sample of the dependents a PR lane runs instead, and a
selection over the light job's cap is cut (TASK-42626); the nightly and release
runs cover the rest.
"""

from __future__ import annotations

import ast
import collections
import functools
import itertools
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
# Migration directories: every file there is schema (TASK-42521 review I2). A
# DB-layer module elsewhere is schema only when its text says so (SCHEMA_TEXT).
SCHEMA_DIRS = {"migrations"}
# Data read by the guard and budget suites (allowlists, snapshots).
GUARD_DATA_PREFIXES = ("Tests/Architecture/", "Tests/Performance/")
# Files no PR lane runs whole: too slow for the light job's cap, or carrying
# deselected known failures (scripts/library_shell_push_deselect.txt).
HEAVY_TEST_FILES = {
    "Tests/UI/test_library_shell.py",
    "Tests/Backup_Recovery/test_response_rules_restore.py",
}
# A changed module (or data file) with more runnable dependents than this is
# sampled: only dependents a PR lane runs, breadth first, and at most this many
# sampled files per PR in all; nightly and release run the rest.
IMPORTER_CAP = 8
# The PR's own changed tests come first, followed by the capped samples.
# Keep the light job within its existing 15-minute bound.
LIGHT_FILE_CAP = 20
# Application code that creates, alters or versions its own SQLite schema
# (uppercase DDL and PRAGMA, as this codebase writes them; case-sensitive to
# skip prose, and PRAGMA-only so a Python name `user_version` does not count).
# ChaChaNotes stamps its version in a table, so a write there counts too.
SCHEMA_TEXT = re.compile(
    r"\b(?:ALTER TABLE|CREATE (?:TABLE|UNIQUE INDEX|INDEX|TRIGGER|VIEW|VIRTUAL TABLE))\b"
    r"|\bPRAGMA[ \t]+user_version\b"
    r"|\bUPDATE[ \t\n]+db_schema_version\b"
)
# Nodes whose first statement may be a docstring (not a child script).
DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
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
    """Paths under Tests/ that loading the file `name` loads, or None when it
    does not parse.

    Covers `Tests.*` absolute imports, relative imports, the importer's own
    package `__init__.py` files and every package `__init__.py` on the way.
    Nested imports count, and so do the imports of a child script a test
    embeds as a string (docstrings excluded). A bare `import mod` counts as
    the module beside the importer. pyproject runs pytest with
    `--import-mode=importlib`, which puts no test directory on sys.path, so
    this over-approximates (toward more tests); child scripts that add their
    own directory do import that way. Paths no file has are kept, so a
    deleted module still counts.
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
    docstrings = set()
    for node in (node for tree in trees for node in ast.walk(tree)):
        if isinstance(node, DOCUMENTED) and node.body:  # ast.walk: parents first
            first_statement = node.body[0]
            if isinstance(first_statement, ast.Expr):
                docstrings.add(id(first_statement.value))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "import" in node.value and id(node) not in docstrings:
                script = textwrap.dedent(node.value)
                child = _parse(script)
                # A template with placeholders: its import lines, one by one.
                lines = [] if child else script.splitlines()
                imports = [line.strip() for line in lines]
                imports = [
                    line for line in imports if line.startswith(("import ", "from "))
                ]
                for sub in [child, *map(_parse, imports)]:  # a child script
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


def _token_index(corpus: dict[str, str]) -> dict[str, list[str]]:
    """Each word in the Tests tree -> the files containing it, in corpus order."""
    index: dict[str, list[str]] = {}
    for name, text in corpus.items():
        for word in set(re.findall(r"\w+", text)):
            index.setdefault(word, []).append(name)
    return index


def _word(name: str) -> re.Pattern[str]:
    """`name` as a whole word: not inside a longer file, module or folder name
    (`guard.py` in `focus_guard.py`, `interop` in `interop-password`)."""
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")


def _component_before(text: str, start: int) -> str | None:
    """The quoted path component right before the quoted folder at `start`:
    `skills` in `"skills" / "review"`, or in a join's `"skills", "review"`. It
    looks on the folder's line, or on the line above (less a `#` comment) when
    the folder opens its line. A comma separates parts only in a join call."""
    line = text.rfind("\n", 0, start) + 1
    head = text[line:start]
    if not head.endswith(("'", '"')):
        return None
    head = head[:-1].rstrip()
    if not head and line:
        above = text.rfind("\n", 0, line - 1) + 1
        head = re.sub(r"#[^'\"\n]*$", "", text[above : line - 1]).rstrip()
    part = re.search(r"""["']([\w.-]+)["']\s*([/,])$""", head)
    if part is None or (
        part.group(2) == ","
        and not re.search(r"join(?:path)?\([^()]*$", head[: part.start()])
    ):
        return None
    return part.group(1)


def _readers(
    path: str,
    corpus: dict[str, str],
    index: dict[str, list[str]],
    common: bool = False,
) -> list[str]:
    """Files that name `path`, in corpus order: by its file name as a whole
    word, or for a `.py` by module name too (data a test reads; a helper a test
    runs by path or -m). A `.py` helper's precise readers (_names_path) win
    when it has any. Otherwise a `common` file name (one other tracked files
    under Tests/ share) needs its folder right before the name, or the folder
    as a quoted literal of its own with its parent folder also named. A
    Tests/CI file counts only if precise: its synthetic fixtures reuse names.
    """
    target = Path(path)
    names = [target.name]
    if path.endswith(".py"):
        names.append(path[:-3].replace("/", "."))
    pool: set[str] = set()
    for name in names:  # every word of a name is a word of a file naming it
        words = re.findall(r"\w+", name)
        pool.update(min((index.get(w, []) for w in words), key=len, default=[]))
    patterns = [_word(name) for name in names]
    readers = [
        f
        for f in sorted(pool)
        if f != path and any(pattern.search(corpus[f]) for pattern in patterns)
    ]
    precise = [f for f in readers if _names_path(f, corpus[f], path)]
    if path.endswith(".py") and precise:
        readers = precise
    elif common:
        # `age_v1/README.md` or `"age_v1" / "README.md"`; or a list of fixture
        # folders (`"executing-plans"`) in a file naming their parent folder too.
        folder, parent = target.parent.name, target.parent.parent.name
        # A longer path before the folder must go through the grandparent:
        # `skills/review/SKILL.md` is not `custom/review/SKILL.md`.
        through = rf"|(?<![\w-]){re.escape(parent)}/" if parent else ""
        beside = re.compile(
            rf"""(?:(?<![\w./-]){through}){re.escape(folder)}[\s'"/,)(]+"""
            rf"""{re.escape(target.name)}(?![\w-])"""
        )
        listed = re.compile(rf"""["']{re.escape(folder)}["']""")

        # A quoted component before the folder (`"skills" / "review"`) is a
        # longer path too, so it must be the grandparent.
        def named_beside(text: str) -> bool:
            for match in beside.finditer(text):
                if parent and text.startswith(f"{parent}/", match.start()):
                    return True  # already through the grandparent
                if _component_before(text, match.start()) in (None, parent):
                    return True
            return False

        readers = [
            f
            for f in readers
            if named_beside(corpus[f])
            or (
                listed.search(corpus[f])
                and parent not in ("", "Tests")
                and bool(_word(parent).search(corpus[f]))
            )
        ]
    return [f for f in readers if not f.startswith("Tests/CI/") or f in precise]


def _folder_readers(
    path: str,
    corpus: dict[str, str],
    index: dict[str, list[str]],
    folders: collections.Counter[str],
) -> list[str]:
    """Files that name the folder holding `path` as a quoted literal: a test
    that copies a fixture package by its folder (`interop_package("cursor-guard")`).
    Only a distinctive folder counts: one with a `-` or a digit in its name,
    below a top-level folder, that no other tracked folder shares. Measured on
    the tree, every wider form (domain words such as `rules` or `watchlists`,
    or ancestors further up) admitted hundreds of unrelated readers."""
    folder = Path(path).parent.name
    if len(Path(path).parts) < 4 or not re.search(r"[-\d]", folder):
        return []
    if folders[folder] > 1:
        return []
    literal = re.compile(rf"""["']{re.escape(folder)}["']""")
    words = re.findall(r"\w+", folder)
    pool = min((index.get(w, []) for w in words), key=len, default=[])
    return [
        f
        for f in pool
        if f != path and not f.startswith("Tests/CI/") and literal.search(corpus[f])
    ]


def _names_path(reader: str, text: str, path: str) -> bool:
    """Whether `reader` names `path` precisely: a quoted literal with a
    directory that the path ends with, its module name, or its file name from
    beside it (not just a common file name)."""
    target = Path(path)
    name = re.escape(target.name)
    for literal in re.findall(rf"""["']([^"'\n]*/{name})["']""", text):
        if f"/{path}".endswith("/" + re.sub(r"^(?:\.{1,2}/)+", "", literal)):
            return True
    if target.suffix == ".py" and _word(path[:-3].replace("/", ".")).search(text):
        return True
    return Path(reader).parent == target.parent and bool(
        _word(target.name).search(text)
    )


def _importers(
    path: str,
    corpus: dict[str, str],
    counts=_is_test_file,
    first: list[str] | None = None,
    index: dict[str, list[str]] | None = None,
    shared: collections.Counter[str] | None = None,
) -> list[str]:
    """Files under Tests/ that load `path`, transitively, breadth first in corpus
    order: direct importers (or `first`, such as a data file's readers) before
    theirs. Every file is followed, but the search stops once IMPORTER_CAP + 1
    of them satisfy `counts`. A file that does not parse counts as an
    importer, which errs toward more tests. `index` is _token_index(corpus);
    `shared` counts each file name among the tracked files (see select_changes),
    for the common-name rule of a helper met on the way.
    """
    index = _token_index(corpus) if index is None else index
    if shared is None:
        shared = collections.Counter(Path(name).name for name in corpus)
    found: list[str] = []
    seen = {path}
    count = 0
    queue = [path]
    for current in queue:  # appended to while iterating: breadth first
        init = Path(current).name == "__init__.py"
        token = Path(current).parent.name if init else Path(current).stem
        package = f"{Path(current).parent}/"
        # Only files with the module's name as a word can import it; a package's
        # own files load its __init__.py without naming it.
        named = index.get(token, [])
        if init:
            named = sorted({*named, *(n for n in corpus if n.startswith(package))})
        if current == path and first is not None:
            level = iter(first)
        else:
            level = (name for name in named if _loads(name, corpus[name], current))
            if current != path and not _is_test_file(current):
                # A helper met on the way that nothing imports: the files that
                # name it run it (a child process, a spec-loaded file).
                lead = next(level, None)
                level = (
                    itertools.chain([lead], level)
                    if lead
                    else iter(
                        _readers(current, corpus, index, shared[Path(current).name] > 1)
                    )
                )
        for name in level:
            if name in seen:
                continue
            seen.add(name)
            found.append(name)
            queue.append(name)
            count += bool(counts(name))
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

    @functools.cache
    def laned(path: str) -> bool:
        """A PR lane runs it: whole, by node or through a lane directory."""
        return bool(_names(path, step_set)) or any(
            path.startswith(d.rstrip("/") + "/") for d in lane_dirs
        )

    def sampleable(path: str) -> bool:
        return runnable(path) and laned(path)

    index: dict[str, list[str]] = {}

    def token_index() -> dict[str, list[str]]:
        if not index:
            index.update(_token_index(tests_corpus()))
        return index

    @functools.cache
    def tracked() -> tuple[str, ...]:
        """The tracked files under Tests/, once per plan. Outside a git checkout,
        when git lists no Tests files (a root inside another repository), or with
        no git binary: the files on disk."""
        try:
            listed = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--", "Tests"],
                capture_output=True,
                # A hook's GIT_DIR would point at another repository's index.
                env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
            )
            output = "" if listed.returncode else listed.stdout.decode(errors="replace")
        except OSError:  # no git binary
            output = ""
        paths = [n for n in output.split("\0") if n]
        if not paths:
            paths = [
                str(f.relative_to(root))
                for f in (root / "Tests").rglob("*")
                if "__pycache__" not in f.parts and f.is_file()
            ]
        return tuple(paths)

    deleted = collections.Counter(
        Path(p).name for s, p in changes if s == "D" and p.startswith("Tests/")
    )

    @functools.cache
    def shared() -> collections.Counter[str]:
        """How many files share each file name (LICENSE, SKILL.md): the tracked
        ones, plus those the PR deletes, which the tree no longer has."""
        return collections.Counter(Path(n).name for n in tracked()) + deleted

    @functools.cache
    def folders() -> collections.Counter[str]:
        """How many tracked folders under Tests/ share each folder name."""
        return collections.Counter(
            p.name for p in {p for n in tracked() for p in Path(n).parents} if p.name
        )

    def shorten(items: list[str], keep: int) -> str:
        """At most `keep` items, so a large PR's reason stays readable."""
        more = f", ... and {len(items) - keep} more" if len(items) > keep else ""
        return ", ".join(items[:keep]) + more

    # The input order must not matter; sorted by path, as git prints them.
    changes = sorted(changes, key=lambda change: (change[1], change[0]))
    # First the cheap reasons for the full run, so no search precedes them.
    for status, path in changes:
        reason = _full_reason(status, path)
        if reason:
            return broad(reason)
        if path.startswith("Tests/") and path.endswith(".py"):
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
        elif (
            status != "D" and path.startswith("tldw_chatbook/") and path.endswith(".py")
        ):
            source = root / path
            if source.is_file() and SCHEMA_TEXT.search(
                source.read_text(errors="replace")
            ):
                return broad(f"Schema code: {path}")

    targets = set(BASE_TESTS)
    changed: list[str] = []  # the PR's own changed test files, kept first
    samples: list[dict] = []  # per changed module or data file (see below)
    for status, path in changes:
        subsystem = next((key for key in SUBSYSTEM_TESTS if path.startswith(key)), None)
        if subsystem and status != "D" and Path(path).suffix in {".py", ".tcss"}:
            targets.update(SUBSYSTEM_TESTS[subsystem])
        if not path.startswith("Tests/"):
            continue  # docs, backlog, census files and other sources: smoke only
        if _is_test_file(path) and status != "D" and not path.startswith("Tests/CI/"):
            changed.append(path)
        first = None
        found = []
        if path.endswith(".py"):
            found = _importers(
                path, tests_corpus(), sampleable, None, token_index(), shared()
            )
        if not found and not _is_test_file(path):
            # Test data, or a helper nothing imports (a child process or a
            # spec-loaded file): the files that name it read or run it.
            common = shared()[Path(path).name] > 1
            first = _readers(path, tests_corpus(), token_index(), common)
            loading = next(
                (
                    f
                    for f in first
                    if (f in conftest_loaded() or Path(f).name == "conftest.py")
                    and _names_path(f, tests_corpus()[f], path)
                ),
                None,
            )
            if loading:
                return broad(f"Named at conftest time by {loading}: {path}")
            if not first:  # a fixture a test copies by its folder
                first = _folder_readers(path, tests_corpus(), token_index(), folders())
                loading = next(
                    (
                        f
                        for f in first
                        if f in conftest_loaded() or Path(f).name == "conftest.py"
                    ),
                    None,
                )
                if loading:
                    return broad(f"Named at conftest time by {loading}: {path}")
            if first:
                found = _importers(
                    path,
                    tests_corpus(),
                    sampleable,
                    first,
                    token_index(),
                    shared(),
                )
        # Run the files that depend on it. With more runnable ones than
        # IMPORTER_CAP, sample only those a PR lane runs (a subset of the full
        # run); unlaned ones have no known health on the PR's platform.
        tests = [f for f in found if _is_test_file(f)]
        in_ci = any(f.startswith("Tests/CI/") for f in tests)
        tests = [f for f in tests if not f.startswith("Tests/CI/")]
        run = [f for f in tests if runnable(f)]
        sampling = len(run) > IMPORTER_CAP
        sample = [f for f in run if laned(f)][:IMPORTER_CAP] if sampling else run
        samples.append(
            {
                "path": path,
                "kind": "importers" if first is None else "readers",
                "sample": sample,
                "sampling": sampling,
                "laned": sum(map(sampleable, run)),
                "unlaned": len(run) - sum(map(laned, run)) if sampling else 0,
                "skipped": len(tests) - len(run),
                "in_ci": in_ci,
                "tests": len(tests),
                "found": bool(found),
                "test": _is_test_file(path),
                # Laned dependents the light job cannot run: only the full run
                # reaches them, which matters if nothing laned runs here instead.
                "blocked": [
                    f
                    for f in tests
                    if not runnable(f) and (laned(f) or _names(f, other))
                ],
            }
        )
    changed.sort()
    own: list[str] = []
    # A change with dependents only the full run reaches gets its first laned
    # dependent first, ahead of the PR's own files, so a cut alone cannot send
    # the plan broad (see the check below); sampled ones count toward the 8.
    led = 0
    for e in samples:
        lead = next((f for f in e["sample"] if laned(f)), None)
        sampled = e["sampling"] and lead not in changed  # the PR's own file is not
        if e["blocked"] and lead and lead not in own and len(own) < LIGHT_FILE_CAP:
            if sampled and led >= IMPORTER_CAP:
                continue  # the 8 sampled files are spent: the check below goes broad
            own.append(lead)
            led += sampled
    own += [f for f in changed if f not in own][: LIGHT_FILE_CAP - len(own)]

    def share(entries: list[dict], limit: int) -> None:
        """Add the samples' files in turn: each one's first, then each second."""
        for rank in range(IMPORTER_CAP):
            for entry in entries:
                sample = entry["sample"]
                if rank < len(sample) and sample[rank] not in own and len(own) < limit:
                    own.append(sample[rank])

    share([e for e in samples if not e["sampling"]], LIGHT_FILE_CAP)
    budget = min(LIGHT_FILE_CAP, len(own) + IMPORTER_CAP - led)  # 8 sampled in all
    share([e for e in samples if e["sampling"]], budget)
    # After the cuts: blocked dependents and nothing laned that runs in their place.
    for e in samples:
        if e["blocked"] and not e["in_ci"]:
            if not any(laned(f) for f in e["sample"] if f in own):
                blocked = shorten(e["blocked"], 3)
                return broad(
                    f"Only the full run reaches the laned dependents of {e['path']}: "
                    + blocked
                )
    notes = []
    for e in samples:
        ran = sum(f in own for f in e["sample"])
        path, kind = e["path"], e["kind"]
        more = "+" if e["laned"] > IMPORTER_CAP else ""
        if not e["found"] and e["test"]:
            continue  # a test file nothing imports: it runs itself
        if not e["found"]:
            note = f"nothing imports or names {path}"
        elif not e["sample"] and e["in_ci"]:
            note = f"Tests/CI covers {path}"
        elif not e["tests"]:
            note = f"no test file depends on {path}"
        elif not e["sample"]:
            note = f"no PR lane runs a dependent of {path}"
            if e["tests"]:
                files = "file" if e["tests"] == 1 else "files"
                note += f"; only the release run covers its {e['tests']} test {files}"
        elif e["sampling"]:
            note = (
                f"sampled {ran} of {e['laned']}{more} {kind} of {path} a PR lane runs"
            )
            if e["unlaned"]:
                note += f", skipping {e['unlaned']}{more} no PR lane runs"
        elif ran < len(e["sample"]) or e["skipped"]:
            note = f"sampled {ran} of {len(e['sample'])} {kind} of {path}"
        else:
            continue
        if e["skipped"] and e["sample"]:
            note += f", skipping {e['skipped']} the light job cannot run"
        notes.append(note)
    cut = [f for f in changed if f not in own]
    dropped = {e["sampling"] for e in samples for f in e["sample"] if f not in own}
    if True in dropped and budget < LIGHT_FILE_CAP:
        notes.insert(0, f"{IMPORTER_CAP} sampled files in all")
    elif True in dropped:
        dropped.add(False)  # the 20-file cap cut the samples too
    if cut or False in dropped:
        unrun = f", not running the PR's own {shorten(cut, 5)}" if cut else ""
        notes.insert(0, f"cut to {LIGHT_FILE_CAP} test files{unrun}")
    if len(notes) > 10:
        notes = [*notes[:10], f"... and {len(notes) - 10} more notes"]
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
