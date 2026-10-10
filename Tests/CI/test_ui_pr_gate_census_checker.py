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

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ui_pr_gate_census.py"


def _checker(tmp_path: Path, lines: list[str], *, floor: int | None = None,
             root: Path | None = None):
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
    m = _checker(tmp_path, ["# comment", "", "Tests/UI/test_a.py", "Tests/UI/test_b.py"],
                 floor=2, root=root)
    assert m.main() == 0
    assert "OK:" in capsys.readouterr().out


def test_comments_and_blank_lines_are_not_counted_as_entries(tmp_path):
    """A census padded with comments must not satisfy the floor."""
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["# one", "", "  ", "# two", "Tests/UI/test_a.py"],
                 floor=2, root=root)
    assert m.main() == 1


def test_a_duplicate_entry_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["Tests/UI/test_a.py", "Tests/UI/test_a.py"], floor=1, root=root)
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
    m = _checker(tmp_path, ["Tests/UI/test_a.py", "Tests/UI/test_renamed_away.py"],
                 floor=1, root=root)
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
            "Tests/UI/test_p0.py::test_stop_is_reachable",
            "Tests/UI/test_p0.py::test_stop_is_reachable[80x24]",
            "Tests/UI/test_p0.py::TestSaveMarkdown::test_escape_closes",
            "Tests/UI/test_other.py",
        ],
        floor=4,
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
        tmp_path, ["Tests/UI/test_params.py::test_stacked[escape-shout]"], floor=1, root=root
    )
    assert m.main() == 1
    assert "no parametrize id" in capsys.readouterr().err


def _overlaps(targets):
    spec = importlib.util.spec_from_file_location("cen_overlap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.overlapping_targets(targets)


def test_disjoint_pytest_targets_have_no_overlap():
    assert _overlaps(
        [
            "Tests/CI",
            "Tests/UI/test_a.py",
            "Tests/UI/test_b.py::test_x",
            "Tests/UI/test_b.py::test_y[1]",
        ]
    ) == []


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
