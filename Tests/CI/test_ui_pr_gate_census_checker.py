"""Exercise the PR-gate census checker's FAILURE paths, not just its happy one.

The committed census passing proves the census is currently valid; it proves
nothing about the checker. A guard whose failure branches have never been
observed to fire is the exact shape this whole review is about -- a check that
cannot fail reads identically to a check that passes.

Each test below drives `main()` through one rejection with a temporary census
and asserts both the exit status and the diagnostic, so a future edit that
silently stops rejecting is caught.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ui_pr_gate_census.py"


def _checker(
    tmp_path: Path,
    lines: list[str],
    *,
    floor: int | None = None,
    root: Path | None = None,
):
    """Load a fresh checker module pointed at a throwaway census.

    Args:
        tmp_path: pytest's temporary directory.
        lines: Raw census lines to write.
        floor: Override MINIMUM_FILES; defaults to the module's own value.
        root: Override REPO_ROOT, so `is_file()` resolves against a fixture tree.

    Returns:
        The imported module, ready for `main()`.
    """
    spec = importlib.util.spec_from_file_location(f"cen_{tmp_path.name}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # the census must sit UNDER the repo root: the checker reports paths with
    # `relative_to(REPO_ROOT)`, so a census outside it raises ValueError.
    base = root if root is not None else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    census = base / "census.txt"
    census.write_text("\n".join(lines) + "\n", encoding="utf-8")
    module.CENSUS_PATH = census
    if root is not None:
        module.REPO_ROOT = root
    if floor is not None:
        module.MINIMUM_FILES = floor
    return module


def _tree(tmp_path: Path, *rel: str) -> Path:
    root = tmp_path / "repo"
    for r in rel:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    return root


def test_a_valid_census_passes(tmp_path, capsys):
    """The positive control: without it a checker that always fails looks correct."""
    root = _tree(tmp_path, "Tests/UI/test_a.py", "Tests/UI/test_b.py")
    m = _checker(
        tmp_path,
        ["# comment", "", "Tests/UI/test_a.py", "Tests/UI/test_b.py"],
        floor=2,
        root=root,
    )
    assert m.main() == 0
    assert "OK:" in capsys.readouterr().out


def test_comments_and_blank_lines_are_not_counted_as_entries(tmp_path):
    """A census padded with comments must not satisfy the floor."""
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(
        tmp_path, ["# one", "", "  ", "# two", "Tests/UI/test_a.py"], floor=2, root=root
    )
    assert m.main() == 1


def test_a_duplicate_entry_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(
        tmp_path, ["Tests/UI/test_a.py", "Tests/UI/test_a.py"], floor=1, root=root
    )
    assert m.main() == 1
    assert "duplicate entry" in capsys.readouterr().err


def test_a_path_outside_tests_ui_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/Chat/test_x.py")
    m = _checker(tmp_path, ["Tests/Chat/test_x.py"], floor=1, root=root)
    assert m.main() == 1
    assert "not a Tests/UI path" in capsys.readouterr().err


def test_a_censused_file_that_no_longer_exists_is_rejected(tmp_path, capsys):
    """The case the gate exists for: a rename silently shrinks what runs."""
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(
        tmp_path,
        ["Tests/UI/test_a.py", "Tests/UI/test_renamed_away.py"],
        floor=1,
        root=root,
    )
    assert m.main() == 1
    assert "listed file does not exist" in capsys.readouterr().err


def test_a_shrunk_census_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["Tests/UI/test_a.py"], floor=5, root=root)
    assert m.main() == 1
    assert "census has shrunk" in capsys.readouterr().err


def test_a_missing_census_file_is_rejected(tmp_path, capsys):
    m = _checker(tmp_path, ["Tests/UI/test_a.py"], floor=1, root=_tree(tmp_path))
    m.CENSUS_PATH = m.REPO_ROOT / "gone.txt"
    assert m.main() == 1
    assert "census file is missing" in capsys.readouterr().err


def test_the_committed_census_is_currently_valid():
    """Separately from the above: the real census in this repo passes today."""
    spec = importlib.util.spec_from_file_location("cen_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0


# ---- node id entries (TASK-33621.27) ---------------------------------------
# A P0 regression test whose file is too slow for the lane is gated on its own
# node id. Each rejection below is a way such an entry could silently stop
# gating its test, or take the whole shard down with it.

_NODE_SOURCE = (
    "import pytest\n\n"
    "@pytest.mark.parametrize('size', ['80x24'])\n"
    "async def test_stop_is_reachable(size):\n    pass\n\n"
    "class TestSaveMarkdown:\n    def test_escape_closes(self):\n        pass\n"
)


def _node_tree(tmp_path: Path) -> Path:
    root = _tree(tmp_path, "Tests/UI/test_other.py")
    (root / "Tests/UI/test_p0.py").write_text(_NODE_SOURCE, encoding="utf-8")
    return root


def test_node_id_entries_for_defined_tests_pass(tmp_path, capsys):
    """Positive control: function, parametrized and class-method node ids."""
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path,
        [
            # (Not also the bare test: a test beside one of its own
            # parametrizations overlaps -- review round 4.)
            "Tests/UI/test_p0.py::test_stop_is_reachable[80x24]",
            "Tests/UI/test_p0.py::TestSaveMarkdown::test_escape_closes",
            "Tests/UI/test_other.py",
        ],
        floor=3,
        root=root,
    )
    assert m.main() == 0, capsys.readouterr().err
    assert "OK:" in capsys.readouterr().out


def test_a_node_id_naming_a_renamed_test_is_rejected(tmp_path, capsys):
    """pytest would refuse the whole shard with 'not found'; fail here by name."""
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path,
        ["Tests/UI/test_p0.py::test_stop_was_renamed"],
        floor=1,
        root=root,
    )
    assert m.main() == 1
    assert "does not define" in capsys.readouterr().err


def test_a_node_id_naming_a_method_on_the_wrong_class_is_rejected(tmp_path, capsys):
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path,
        ["Tests/UI/test_p0.py::TestElsewhere::test_escape_closes"],
        floor=1,
        root=root,
    )
    assert m.main() == 1
    assert "does not define" in capsys.readouterr().err


def test_a_node_id_in_a_missing_file_is_rejected(tmp_path, capsys):
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path, ["Tests/UI/test_gone.py::test_stop_is_reachable"], floor=1, root=root
    )
    assert m.main() == 1
    assert "listed file does not exist" in capsys.readouterr().err


def test_a_node_id_beside_its_whole_file_is_rejected(tmp_path, capsys):
    """Overlapping pytest arguments collapse (ADR-103); keep exactly one."""
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path,
        ["Tests/UI/test_p0.py", "Tests/UI/test_p0.py::test_stop_is_reachable"],
        floor=2,
        root=root,
    )
    assert m.main() == 1
    assert "node id overlaps a whole-file entry" in capsys.readouterr().err


def test_an_entry_with_whitespace_is_rejected(tmp_path, capsys):
    """The lane and its shard pins read the census one token per entry."""
    root = _node_tree(tmp_path)
    m = _checker(
        tmp_path,
        ["Tests/UI/test_p0.py::test_stop_is_reachable[Not sent.]"],
        floor=1,
        root=root,
    )
    assert m.main() == 1
    assert "entry contains whitespace" in capsys.readouterr().err


# ---- parametrize ids and argument overlap (TASK-33621.27 review) -----------
# A renamed parametrize id used to pass the checker (it stripped `[...]`), and
# pytest then exits 4 and the whole shard runs nothing. The ids are now
# resolved statically; an id that cannot be resolved is refused by name.

_PARAM_SOURCE = (
    "import pytest\n\n"
    "SIZES = [(80, 24), (235, 52)]\n"
    "TRIGGERS = {'system-prompt': object(), 'image-only': object()}\n\n"
    "@pytest.mark.parametrize('route', ['enter', 'send-button'])\n"
    "@pytest.mark.parametrize('answer', ['escape', 'not-now'])\n"
    "async def test_stacked(route, answer):\n    pass\n\n"
    "@pytest.mark.parametrize('how', [1, True, None, 2.5])\n"
    "def test_scalars(how):\n    pass\n\n"
    "@pytest.mark.parametrize(('key', 'size'), [('enter', (80, 24))])\n"
    "def test_tuple_values(key, size):\n    pass\n\n"
    "@pytest.mark.parametrize('size', SIZES, ids=['narrow', 'wide'])\n"
    "def test_named_ids(size):\n    pass\n\n"
    "@pytest.mark.parametrize('trigger', list(TRIGGERS))\n"
    "def test_dict_keys(trigger):\n    pass\n\n"
    "@pytest.mark.parametrize('x', [pytest.param(1, id='one'), 2])\n"
    "def test_param_ids(x):\n    pass\n\n"
    "@pytest.mark.parametrize('size', SIZES, ids=lambda s: f'{s[0]}x{s[1]}')\n"
    "def test_callable_ids(size):\n    pass\n\n"
    "@pytest.mark.parametrize('action', _computed())\n"
    "def test_computed(action):\n    pass\n"
)


def _param_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_params.py"
    path.write_text(_PARAM_SOURCE, encoding="utf-8")
    return path


def _resolve(tmp_path: Path, node: str):
    spec = importlib.util.spec_from_file_location(f"cen_r_{tmp_path.name}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_node(_param_file(tmp_path), node)


@pytest.mark.parametrize(
    "node",
    [
        "test_stacked[escape-enter]",
        "test_stacked[not-now-send-button]",
        "test_scalars[1]",
        "test_scalars[True]",
        "test_scalars[None]",
        "test_scalars[2.5]",
        "test_tuple_values[enter-size0]",
        "test_named_ids[wide]",
        "test_dict_keys[image-only]",
        "test_param_ids[one]",
        "test_param_ids[2]",
        "test_callable_ids",
        "test_computed",
    ],
)
def test_resolvable_parametrize_ids_pass(tmp_path, node):
    """Positive control: the ids pytest would generate, closest decorator first."""
    assert _resolve(tmp_path, node) is None


@pytest.mark.parametrize(
    ("node", "reason"),
    [
        ("test_stacked[enter-escape]", "no parametrize id"),
        ("test_named_ids[narrow-ish]", "no parametrize id"),
        ("test_dict_keys[system-prompt-renamed]", "no parametrize id"),
        ("test_callable_ids[80x24]", "cannot be resolved statically"),
        ("test_computed[save-markdown]", "cannot be resolved statically"),
        ("test_missing[x]", "does not define"),
    ],
)
def test_unresolvable_or_renamed_parametrize_ids_are_refused(tmp_path, node, reason):
    """A renamed id fails here by name instead of emptying a whole shard."""
    problem = _resolve(tmp_path, node)
    assert problem is not None and reason in problem, problem


def test_a_renamed_parametrize_id_fails_the_census(tmp_path, capsys):
    root = _tree(tmp_path)
    target = root / "Tests/UI/test_params.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_PARAM_SOURCE, encoding="utf-8")
    m = _checker(
        tmp_path,
        ["Tests/UI/test_params.py::test_stacked[escape-shout]"],
        floor=1,
        root=root,
    )
    assert m.main() == 1
    assert "no parametrize id" in capsys.readouterr().err


def _overlaps(targets):
    spec = importlib.util.spec_from_file_location("cen_overlap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.overlapping_targets(targets)


def test_disjoint_pytest_targets_have_no_overlap():
    assert (
        _overlaps(
            [
                "Tests/CI",
                "Tests/UI/test_a.py",
                "Tests/UI/test_b.py::test_x",
                "Tests/UI/test_b.py::test_y[1]",
            ]
        )
        == []
    )


@pytest.mark.parametrize(
    "targets",
    [
        # The review's case: pytest then collected 1 of 21 tests, all green.
        ["Tests/UI/test_b.py", "Tests/UI/test_b.py::test_x"],
        ["Tests/UI/test_b.py::test_x", "Tests/UI/test_b.py"],
        ["Tests/CI", "Tests/CI/test_c.py"],
        ["Tests/CI", "Tests/CI/test_c.py::test_x"],
        ["Tests/UI/test_b.py::test_x", "Tests/UI/test_b.py::test_x"],
    ],
)
def test_overlapping_pytest_targets_are_named(targets):
    assert _overlaps(targets), targets


# ---- review round 4: ids pytest 8.4.2 actually generates ------------------
# Each expected id below was read from `pytest --collect-only` (8.4.2) on this
# exact source, so the checker is held to pytest's own output, not to a
# re-reading of its code.

_PYTEST_TRUTH_SOURCE = """import pytest

@pytest.mark.parametrize('x', [pytest.param(1, id='p'), 2], ids=['L1', 'L2'])
def test_precedence(x):
    pass

@pytest.mark.parametrize('b', [b'ab', b'\\xc3\\xa9', b'a\\nb'])
def test_bytes(b):
    pass

@pytest.mark.parametrize('c', [1j, 2+3j])
def test_complex(c):
    pass

@pytest.mark.parametrize('s', ['\\u00e9', 'tab\\there'])
def test_unicode_values(s):
    pass

@pytest.mark.parametrize('s', [1, 2], ids=['\\u00fc', 'plain'])
def test_unicode_list_ids(s):
    pass

@pytest.mark.parametrize('s', [pytest.param(1, id='\\u00f1')])
def test_unicode_param_id(s):
    pass

@pytest.mark.parametrize('d', ['a', 'a', 'x1', 'x1'])
def test_duplicates(d):
    pass

@pytest.fixture(params=['vanilla', 'mint'])
def flavour(request):
    return request.param

@pytest.mark.parametrize('n', [1])
def test_uses_param_fixture(n, flavour):
    pass
"""


def _resolve_truth(tmp_path: Path, node: str):
    path = tmp_path / "test_truth.py"
    path.write_text(_PYTEST_TRUTH_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"cen_t_{tmp_path.name}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_node(path, node)


@pytest.mark.parametrize(
    "node",
    [
        "test_precedence[p]",  # pytest.param(id=...) wins over ids=[...]
        "test_precedence[L2]",
        "test_bytes[ab]",
        "test_bytes[\\xc3\\xa9]",
        "test_bytes[a\\nb]",
        "test_complex[1j]",
        "test_complex[(2+3j)]",
        "test_unicode_values[\\xe9]",
        "test_unicode_values[tab\\there]",
        "test_unicode_list_ids[\\xfc]",
        "test_unicode_list_ids[plain]",
        "test_unicode_param_id[\\xf1]",
        "test_duplicates[a0]",
        "test_duplicates[a1]",
        "test_duplicates[x1_0]",
        "test_duplicates[x1_1]",
    ],
)
def test_ids_match_what_pytest_generates(tmp_path, node):
    assert _resolve_truth(tmp_path, node) is None


@pytest.mark.parametrize(
    ("node", "reason"),
    [
        ("test_precedence[L1]", "no parametrize id"),  # never generated
        ("test_complex[c0]", "no parametrize id"),
        ("test_unicode_values[é]", "no parametrize id"),  # pytest escapes it
        ("test_duplicates[a]", "no parametrize id"),
        ("test_uses_param_fixture[vanilla-1]", "parametrized fixture"),
        ("test_uses_param_fixture[1]", "parametrized fixture"),
    ],
)
def test_ids_pytest_never_generates_are_refused(tmp_path, node, reason):
    problem = _resolve_truth(tmp_path, node)
    assert problem is not None and reason in problem, problem


@pytest.mark.parametrize(
    "targets",
    [
        ["Tests/UI/test_b.py::test_x", "Tests/UI/test_b.py::test_x[1]"],
        ["Tests/UI/test_b.py::test_x[1]", "Tests/UI/test_b.py::test_x"],
        ["Tests/UI/test_b.py::TestC", "Tests/UI/test_b.py::TestC::test_y"],
    ],
)
def test_a_target_inside_another_listed_target_is_named(targets):
    """Review round 4: a test beside one of its own parametrizations, or a
    class beside one of its methods, collapses the same way a file does."""
    assert _overlaps(targets), targets


def test_prefix_without_a_node_boundary_is_not_an_overlap():
    assert (
        _overlaps(["Tests/UI/test_b.py::test_x", "Tests/UI/test_b.py::test_xy"]) == []
    )


def test_dead_defines_test_is_gone():
    spec = importlib.util.spec_from_file_location("cen_dead", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(module, "defines_test")


#: Census files listed by function (or, for a one-test file, by parametrize
#: case) instead of whole, so no UI Fast Lane shard takes one whole
#: (TASK-35834, TASK-36512). A test function or case added to one of them
#: runs in no lane until it is listed too, which the test below catches. It
#: counts top-level functions only, so it refuses a test class in a split file
#: rather than let the class's methods go unlisted.
SPLIT_CENSUS_FILES = (
    "Tests/UI/test_schedules_reminder_edit_identity.py",
    "Tests/UI/test_study_quiz_start_responsiveness.py",
    "Tests/UI/test_image_gen_demo_request_lifetime.py",
    "Tests/UI/test_mcp_import_validation_recovery.py",
    "Tests/UI/test_mcp_import_pending_file_drafts.py",
    "Tests/UI/test_chunking_lab_sample_import_drafts.py",
    "Tests/UI/test_dictation_button_delivery.py",
    "Tests/UI/test_dictation_view_retirement.py",
    "Tests/UI/test_settings_model_defaults_rows.py",
    "Tests/UI/test_settings_connect_rows.py",
    "Tests/UI/test_settings_connect_tab_budget.py",
    "Tests/UI/test_settings_default_model_picker.py",
    "Tests/UI/test_mcp_permissions_handoff.py",
)


def _case_count(function: ast.AST) -> int:
    """How many cases pytest makes of a test's literal parametrize lists."""
    count = 1
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "attr", "") == "parametrize"
        ):
            values = decorator.args[1]
            assert isinstance(
                values, (ast.List, ast.Tuple)
            ), f"{function.name}: list it by function, its cases are not a literal list"
            count *= len(values.elts)
    return count


@pytest.mark.parametrize("path", SPLIT_CENSUS_FILES)
def test_a_split_census_file_keeps_every_test_gated_once(path):
    root = SCRIPT.parents[1]
    spec = importlib.util.spec_from_file_location("cen_split", SCRIPT)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
    ]
    assert not classes, f"{path} defines test classes {classes}: rename it or write its tests as functions"
    functions = {  # pytest's default python_functions: the "test" prefix
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
        # pytest never collects a fixture, so listing one would refuse the shard
        and not any(checker._fixture_call(d) for d in node.decorator_list)
    }
    entries = [
        line.strip()
        for line in (root / "scripts/ui_pr_gate_census.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = [entry for entry in entries if entry.split("::", 1)[0] == path]
    assert path not in selected, f"{path} is listed whole again"
    twice = sorted({entry for entry in selected if selected.count(entry) > 1})
    assert not twice, f"listed twice: {twice}"
    listed = 0
    for name, function in functions.items():
        selector = f"{path}::{name}"
        cases = [entry for entry in selected if entry.startswith(selector + "[")]
        if selector in selected:
            listed += 1  # a function selector runs every parameter case
            assert not cases, f"{selector} is listed beside its own cases"
            continue
        # Listed by case: each id must be one pytest makes, and all of them listed.
        assert cases, f"{selector} is not in the census, so no lane runs it"
        for case in cases:
            assert (
                checker.resolve_node(root / path, case.split("::", 1)[1]) is None
            ), case
        assert len(set(cases)) == _case_count(function), f"{selector} misses a case"
        listed += len(set(cases))
    assert listed == len(selected), "a selector names no test this file defines"
    # Reading the actual matrix prevents a split that still deals every
    # selector back into one job, even though the census has no duplicates.
    import yaml

    jobs = yaml.safe_load(
        (root / ".github/workflows/derived-artifacts.yml").read_text(encoding="utf-8")
    )["jobs"]
    total = len(jobs["ui-fast-lane"]["strategy"]["matrix"]["shard"])
    assert (
        len({index % total for index, entry in enumerate(entries) if entry in selected})
        > 1
    )
