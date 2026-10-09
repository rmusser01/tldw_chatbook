"""B20: repository tree toggling uses a child index and direct checkbox handles.

Toggling a directory must cascade to every descendant through the
``_children_by_dir`` index with the node's stored ``Checkbox`` handle --
zero DOM queries in the toggle path -- while producing exactly the same
toggle outcomes (selection set, node.selected, checkbox values) as the
original per-descendant ``query_one`` walk.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Checkbox

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.Widgets.Coding_Widgets.repo_tree_widgets import (
    TreeNode,
    TreeView,
)


def _dir(name: str, path: str, children: list | None = None) -> dict:
    item = {"name": name, "path": path, "type": "tree"}
    if children is not None:
        item["children"] = children
    return item


def _file(name: str, path: str, size: int = 10) -> dict:
    return {"name": name, "path": path, "type": "blob", "size": size}


def _fixture_tree(n_deep_files: int = 200) -> list:
    """A directory with `n_deep_files` direct descendants (one nested level)."""
    top_level = [
        _file("README.md", "README.md"),
        _dir("src", "src"),
        _dir("docs", "docs"),
    ]
    tree = [
        _dir(
            "root",
            "root",
            [_file(f"m{i}.py", f"root/mod{i}.py") for i in range(n_deep_files)],
        ),
        *top_level,
    ]
    return tree


class _TreeHostApp(App[None]):
    def __init__(self, tree: TreeView) -> None:
        super().__init__()
        self._tree = tree

    def compose(self) -> ComposeResult:
        with Container():
            yield self._tree


async def _mounted_tree(pilot, tree: TreeView, deep_files: int = 200) -> None:
    await tree.load_tree(_fixture_tree(deep_files))
    await tree.expand_node(
        "root", [_file(f"m{i}.py", f"root/mod{i}.py") for i in range(deep_files)]
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_toggle_uses_child_index_and_zero_dom_queries(monkeypatch):
    tree = TreeView()
    app = _TreeHostApp(tree)
    async with app.run_test() as pilot:
        await _mounted_tree(pilot, tree, deep_files=200)

        query_calls: list[str] = []
        spies = []
        for cls in (TreeView, TreeNode):
            real_query = cls.query_one

            def spy_query(self, selector, *args, _real=real_query, _log=query_calls, **kwargs):
                _log.append(str(selector))
                return _real(self, selector, *args, **kwargs)

            monkeypatch.setattr(cls, "query_one", spy_query)
            spies.append((cls, real_query))

        tree.select_node("root", True)
        assert tree.selection == {"root", *(f"root/mod{i}.py" for i in range(200))}

        tree.select_node("root", False)
        assert tree.selection == set()

        monkeypatch.undo()

        assert query_calls == [], (
            f"toggle path must not query the DOM, got: {query_calls[:5]}"
        )


@pytest.mark.asyncio
async def test_toggle_cascades_to_all_descendants():
    """Golden toggle outcomes: selection set, node.selected, checkbox values."""
    tree = TreeView()
    app = _TreeHostApp(tree)
    async with app.run_test() as pilot:
        await _mounted_tree(pilot, tree, deep_files=200)

        # Select the directory: every descendant flips on.
        tree.select_node("root", True)
        await pilot.pause()

        assert tree.nodes["root"].selected is True
        for i in range(200):
            path = f"root/mod{i}.py"
            node = tree.nodes[path]
            assert node.selected is True, path
            checkbox = node.query_one(".tree-checkbox", Checkbox)
            assert checkbox.value is True, path
            assert checkbox.parent.has_class("tree-node-selected"), path
        root_checkbox = tree.nodes["root"].query_one(".tree-checkbox", Checkbox)
        assert root_checkbox.value is True

        # Unselected sibling files are untouched.
        assert tree.nodes["README.md"].selected is False
        assert tree.nodes["README.md"].query_one(
            ".tree-checkbox", Checkbox
        ).value is False

        # Deselect the directory: every descendant flips off.
        tree.select_node("root", False)
        await pilot.pause()

        assert "root" not in tree.selection
        for i in range(200):
            path = f"root/mod{i}.py"
            node = tree.nodes[path]
            assert node.selected is False, path
            checkbox = node.query_one(".tree-checkbox", Checkbox)
            assert checkbox.value is False, path
            assert not checkbox.parent.has_class("tree-node-selected"), path
        assert tree.get_selected_files() == []


@pytest.mark.asyncio
async def test_child_index_tracks_build_expand_and_collapse():
    tree = TreeView()
    app = _TreeHostApp(tree)
    async with app.run_test() as pilot:
        await _mounted_tree(pilot, tree, deep_files=5)

        assert set(tree._children_by_dir.get("root", ())) == {
            f"root/mod{i}.py" for i in range(5)
        }

        # Collapse removes the descendants; the index keeps only live paths.
        await tree.collapse_node("root")
        await pilot.pause()
        assert all(p not in tree.nodes for p in tree._children_by_dir.get("root", ()))

        # Re-expanding re-registers without duplicates.
        await tree.expand_node(
            "root", [_file(f"m{i}.py", f"root/mod{i}.py") for i in range(5)]
        )
        await pilot.pause()
        entries = tree._children_by_dir.get("root", [])
        assert len(entries) == len(set(entries)) == 5


@pytest.mark.asyncio
async def test_partial_child_selection_updates_parent_state():
    """Parent directory state follows child selection (index-served)."""
    tree = TreeView()
    app = _TreeHostApp(tree)
    async with app.run_test() as pilot:
        await _mounted_tree(pilot, tree, deep_files=3)

        tree.select_node("root/mod0.py", True)
        assert "root" not in tree.selection
        assert tree.nodes["root"].selected is False

        tree.select_node("root/mod1.py", True)
        tree.select_node("root/mod2.py", True)
        # All children selected -> parent marked selected.
        assert "root" in tree.selection
        assert tree.nodes["root"].selected is True


@pytest.mark.asyncio
async def test_deselecting_one_child_preserves_selected_siblings():
    tree = TreeView()
    app = _TreeHostApp(tree)
    async with app.run_test() as pilot:
        await _mounted_tree(pilot, tree, deep_files=2)

        for path in ("root/mod0.py", "root/mod1.py"):
            tree.nodes[path].query_one(".tree-checkbox", Checkbox).value = True
            await pilot.pause()
        assert tree.selection == {"root", "root/mod0.py", "root/mod1.py"}
        assert tree.nodes["root"].checkbox.parent.has_class("tree-node-selected")

        tree.nodes["root/mod0.py"].query_one(".tree-checkbox", Checkbox).value = False
        await pilot.pause()

        assert tree.selection == {"root/mod1.py"}
        assert tree.nodes["root/mod1.py"].checkbox.value is True
        assert tree.nodes["root"].checkbox.value is False
