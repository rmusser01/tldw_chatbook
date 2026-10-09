"""B18: SmartContentTree keystroke hygiene.

Covers three efficiency defects:

1. Search-as-you-type must debounce to one ``_apply_filters`` pass per
   settled query (house ``set_timer`` handle-cancel pattern, 0.3 s), not one
   full tree sweep per keystroke.
2. The searchable text must be computed once per node when the node is
   added; filtering reads the precomputed string instead of rebuilding the
   title/subtitle/metadata join on every keystroke.
3. The mount-time content load must run via ``run_worker`` (off the UI
   thread), not synchronously inside ``on_mount``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Input

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.UI.Widgets.SmartContentTree import (
    ContentNodeData,
    SmartContentTree,
)

SEARCH_DEBOUNCE_SECONDS = 0.3


class _TreeHostApp(App[None]):
    """Minimal host app mounting a single SmartContentTree."""

    def __init__(self, tree: SmartContentTree) -> None:
        super().__init__()
        self._tree = tree

    def compose(self) -> ComposeResult:
        with Container():
            yield self._tree


def _seed_data(n_notes: int = 8) -> dict:
    return {
        ContentType.NOTE: [
            ContentNodeData(
                type=ContentType.NOTE,
                id=f"note-{i}",
                title=f"Note {i}",
                subtitle=f"alpha{i}",
                metadata={"topic": f"topic-{i}"},
            )
            for i in range(n_notes)
        ],
    }


async def _wait_for_loaded(pilot, tree: SmartContentTree, timeout: float = 5.0) -> None:
    """Wait until the worker-driven load has populated the tree."""
    start = asyncio.get_event_loop().time()
    while not tree.all_nodes:
        await pilot.pause(0.05)
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError("SmartContentTree never loaded content")


@pytest.mark.asyncio
async def test_five_char_burst_debounces_to_one_filter_pass(monkeypatch):
    tree = SmartContentTree(load_content=_seed_data)
    app = _TreeHostApp(tree)

    filter_passes: list[int] = []
    real_apply = SmartContentTree._apply_filters

    def spy_apply(self):
        filter_passes.append(1)
        real_apply(self)

    monkeypatch.setattr(SmartContentTree, "_apply_filters", spy_apply)

    async with app.run_test() as pilot:
        await _wait_for_loaded(pilot, tree)
        filter_passes.clear()

        search = tree.query_one("#content-search", Input)
        search.focus()
        await pilot.pause()

        await pilot.press(*"abcde")
        # Nothing runs while the debounce window is open.
        assert filter_passes == []
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.2)

        assert len(filter_passes) == 1, (
            f"expected one debounced filter pass, got {len(filter_passes)}"
        )
        assert tree.search_query == "abcde"


@pytest.mark.asyncio
async def test_search_text_built_once_per_node_across_three_keystroke_sessions(
    monkeypatch,
):
    n_nodes = 8
    tree = SmartContentTree(load_content=lambda: _seed_data(n_nodes))
    app = _TreeHostApp(tree)

    real_build = SmartContentTree._build_search_text
    build_calls: list[str] = []

    def spy_build(item):
        build_calls.append(item.id)
        return real_build(item)

    monkeypatch.setattr(
        SmartContentTree, "_build_search_text", staticmethod(spy_build)
    )

    async with app.run_test() as pilot:
        await _wait_for_loaded(pilot, tree)

        # Built exactly once per node at load time.
        assert len(build_calls) == n_nodes, (
            f"expected {n_nodes} builds at load, got {len(build_calls)}"
        )

        search = tree.query_one("#content-search", Input)
        search.focus()
        await pilot.pause()

        # Three settled keystroke sessions must not rebuild any node's
        # searchable text.
        for ch in "xyz":
            await pilot.press(ch)
            await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.2)

        assert len(build_calls) == n_nodes, (
            "search text must be precomputed once per node, "
            f"but was rebuilt {len(build_calls) - n_nodes} times during filtering"
        )
        # Node payloads carry the precomputed string.
        assert all(
            node.data.search_text is not None for node in tree.all_nodes
        )


@pytest.mark.asyncio
async def test_mount_load_runs_off_the_ui_thread():
    threads: list[threading.Thread] = []

    def load_content():
        threads.append(threading.current_thread())
        return _seed_data(4)

    tree = SmartContentTree(load_content=load_content)
    app = _TreeHostApp(tree)

    async with app.run_test() as pilot:
        await _wait_for_loaded(pilot, tree)
        # The load completed.
        assert len(tree.all_nodes) == 4

    assert threads, "load callback never ran"
    assert threads[0] is not threading.main_thread(), (
        "load_content_callback ran synchronously on the UI thread in on_mount"
    )


@pytest.mark.asyncio
async def test_filter_semantics_preserved_with_precomputed_text():
    """Search matches title/subtitle/metadata exactly as before (query is
    lowercased and matched against the precomputed haystack)."""

    def load_content():
        return {
            ContentType.NOTE: [
                ContentNodeData(
                    type=ContentType.NOTE,
                    id="n1",
                    title="Quarterly report",
                    subtitle="finance",
                    metadata={"tag": "budget"},
                ),
                ContentNodeData(
                    type=ContentType.NOTE,
                    id="n2",
                    title="Shopping list",
                    subtitle=None,
                    metadata=None,
                ),
                ContentNodeData(
                    type=ContentType.NOTE,
                    id="n3",
                    title="Meeting notes",
                    subtitle="finance recap",
                    metadata=None,
                ),
            ],
        }

    tree = SmartContentTree(load_content=load_content)
    app = _TreeHostApp(tree)

    async with app.run_test() as pilot:
        await _wait_for_loaded(pilot, tree)

        # Title match
        tree.search_query = "quarterly"
        tree._apply_filters()
        assert tree.filtered_count == 1

        # Subtitle match (two items share "finance")
        tree.search_query = "finance"
        tree._apply_filters()
        assert tree.filtered_count == 2

        # Metadata match (lowercase metadata value)
        tree.search_query = "budget"
        tree._apply_filters()
        assert tree.filtered_count == 1

        # No match
        tree.search_query = "zzz-nope"
        tree._apply_filters()
        assert tree.filtered_count == 0

        # Category filter off hides the whole category regardless of search.
        tree.search_query = ""
        tree.category_filters[ContentType.NOTE] = False
        tree._apply_filters()
        assert tree.filtered_count == 0


@pytest.mark.asyncio
async def test_load_completion_reapplies_settled_search_and_preserves_selection():
    started = threading.Event()
    release = threading.Event()

    def load_content():
        started.set()
        assert release.wait(10), "test did not release the content loader"
        return {
            ContentType.NOTE: [
                ContentNodeData(type=ContentType.NOTE, id="alpha", title="Alpha"),
                ContentNodeData(type=ContentType.NOTE, id="beta", title="Beta"),
            ]
        }

    tree = SmartContentTree(load_content=load_content)
    tree.selected_content[ContentType.NOTE].add("beta")
    app = _TreeHostApp(tree)
    try:
        async with app.run_test() as pilot:
            while not started.is_set():
                await pilot.pause(0.01)
            tree.query_one("#content-search", Input).value = "alpha"
            await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.2)
            assert tree.search_query == "alpha"

            release.set()
            await _wait_for_loaded(pilot, tree)

            assert tree.filtered_count == 1
            assert [node.data.id for node in tree.all_nodes if node.display] == [
                "alpha"
            ]
            assert tree.get_selections() == {ContentType.NOTE: ["beta"]}
            tree.query_one("#select-all", Button).press()
            await pilot.pause()
            assert set(tree.get_selections()[ContentType.NOTE]) == {"alpha", "beta"}
    finally:
        release.set()
