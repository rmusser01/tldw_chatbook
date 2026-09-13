"""TASK-31260: the Library wiring clusters' hand-kept tuples stay accurate.

Each Library wiring test (export, collections, conversations reader+browse,
search+RAG) carries a frozen, hand-written snapshot of its cluster's method
names. Nothing re-verified those snapshots against the live ``LibraryScreen``
source: a future same-named method landing on the screen -- genuinely
subsystem-owned or coincidence -- was invisible to every wiring test, never
flagged as needing a cluster-membership decision (recipe §16 lesson 5).

This census re-derives, per cluster, the set of ``LibraryScreen`` methods
matching the cluster's naming pattern and asserts it equals what the wiring
tuples plus two explicit frozen lists account for:

* ``_CLUSTER_NO_DELEGATOR`` -- moved-but-unpruned names that legitimately
  have no screen delegator (recorded so their absence from the screen is a
  decision, not a surprise);
* ``_CLUSTER_STAYED`` -- pattern-matching screen methods outside the moved
  tuple (screen-owned by explicit choice or by another cluster's overlap).

Update the OWNING wiring test's tuple for real membership changes; a screen
method appearing or disappearing without that update fails here. The frozen
lists below change only when a name genuinely changes delegator-ness or
screen ownership -- record why in the cluster's comment.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

# Moved-but-unpruned names with no screen delegator at the census freeze
# (2026-09-11, origin/dev): the wiring tuple owns the name, the screen never
# forwarded it. `find_in_library_conversation` is an action-style entry
# point the controller exposes directly; the browse cluster's ten are
# state-field accessors and selection helpers the controllers call
# internally. A NEW name landing here means a moved method lost its last
# screen caller -- confirm that is intended before freezing it.
_NO_DELEGATOR: dict[str, frozenset[str]] = {
    "export": frozenset(),
    "collections": frozenset(),
    "conversations_reader": frozenset({"find_in_library_conversation"}),
    "conversations_browse": frozenset(
        {
            "_carry_selected_conversation_into_snapshot",
            "_conversation_message_count_label",
            "_conversation_record_id",
            "_conversation_records",
            "_conversation_updated_label",
            "_conversation_workspace_label",
            "_open_selected_conversation_handoff",
            "_selected_conversation_record",
            "open_selected_conversation_in_console",
            "use_selected_conversation_as_source",
        }
    ),
    "search_rag": frozenset({"_library_rail_search_placeholder"}),
}

# Pattern-matching screen methods OUTSIDE the moved tuple at the freeze:
# screen-owned stayers (e.g. the @work-thread export workers Textual's
# decorator pins to the screen) plus reader-cluster overlap the browse
# pattern ("library_conversation") also sweeps. A new name here is a new
# screen method the clusters did not absorb -- make that decision in the
# owning wiring test, not silently here.
_STAYED: dict[str, frozenset[str]] = {
    "export": frozenset(
        {
            "_apply_library_export_cancelled",
            "_apply_library_export_counts",
            "_apply_library_export_progress",
            "_apply_library_export_success",
            "_build_library_export_state",
            "_library_export_status_line",
            "_run_library_export_counts_worker",
            "_run_library_export_worker",
            "_start_library_export_counts_worker",
            "_start_library_export_worker",
            "_update_library_export_canvas_after_run",
            "handle_library_export_cancel",
        }
    ),
    "collections": frozenset(
        {
            "_library_collections_count_is_current",
            "_library_collections_rail_count",
            "_read_library_collections_count",
        }
    ),
    "conversations_reader": frozenset(),
    "conversations_browse": frozenset(
        {
            "_ensure_library_conversation_reader_selection",
            "_invalidate_library_conversation_reader_authority",
            "_library_conversation_block_sentence",
            "_library_conversation_handoff_ready",
            "_library_conversation_link_would_unblock",
            "_library_conversation_loaded_preview_selected",
            "_library_conversation_workspace_block",
            "_mirror_library_conversation_reader_preference",
            "_set_library_conversation_link_receipt",
            "_start_library_conversation_reader_selection",
            "_sync_library_conversation_reader",
            "_sync_library_conversation_reader_layout_from_shell",
            "action_library_conversation_open_console",
            # Reader-cluster-owned action entry point (recorded there as a
            # no-delegator moved name); browse's broader pattern sweeps it.
            "find_in_library_conversation",
            "handle_library_conversation_archive_action",
            "handle_library_conversation_row",
            "handle_library_conversation_scope",
            "handle_library_conversation_undo",
            "handle_library_conversation_view_archive",
            "handle_library_conversations_empty_clear_filter",
            "handle_library_conversations_empty_console",
            "handle_library_conversations_export_selected",
            "library_conversation_reader_messages_synced",
            "retry_library_conversation_reader",
            "show_library_conversation_reader_info",
            "show_library_conversation_reader_read",
        }
    ),
    "search_rag": frozenset(
        {
            "_execute_library_rag_answer",
            "_execute_library_rag_search",
            "_library_rag_panel_state",
            "_library_search_result_is_selected",
            "_load_library_search_history",
            "_mirror_library_rag_scope_recovery",
            "_patch_sibling_library_search_input",
            "_save_library_search_history",
            "handle_library_search_clear",
        }
    ),
}

# cluster -> (wiring test module, moved attr, pruned attr, name patterns)
_CLUSTERS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "export": (
        "test_library_export_wiring",
        "_EXPORT_CLUSTER_METHOD_NAMES",
        "_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED",
        ("library_export",),
    ),
    "collections": (
        "test_library_collections_wiring",
        "_COLLECTIONS_CLUSTER_METHOD_NAMES",
        "_COLLECTIONS_CLUSTER_SCREEN_DELEGATOR_PRUNED",
        ("library_collection",),
    ),
    "conversations_reader": (
        "test_library_conversations_wiring",
        "_READER_CLUSTER_METHOD_NAMES",
        "_READER_CLUSTER_SCREEN_DELEGATOR_PRUNED",
        ("conversation_reader",),
    ),
    "conversations_browse": (
        "test_library_conversations_wiring",
        "_BROWSE_CLUSTER_METHOD_NAMES",
        "_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED",
        ("library_conversation",),
    ),
    "search_rag": (
        "test_library_search_rag_wiring",
        "_RAG_SEARCH_CLUSTER_METHOD_NAMES",
        "_RAG_SEARCH_CLUSTER_SCREEN_DELEGATOR_PRUNED",
        ("library_rag", "library_search"),
    ),
}


def _screen_method_names() -> set[str]:
    source = (_PACKAGE_ROOT / "UI" / "Screens" / "library_screen.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LibraryScreen"
    )
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _wiring_sets(module_name: str, moved_attr: str, pruned_attr: str):
    spec = importlib.util.spec_from_file_location(
        module_name, _TESTS_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(getattr(module, moved_attr)), set(getattr(module, pruned_attr))


def test_every_cluster_named_screen_method_is_accounted_for() -> None:
    screen_methods = _screen_method_names()
    for cluster, (module_name, moved_attr, pruned_attr, patterns) in _CLUSTERS.items():
        moved, pruned = _wiring_sets(module_name, moved_attr, pruned_attr)
        census = {
            name
            for name in screen_methods
            if any(pattern in name for pattern in patterns)
        }
        expected = ((moved - pruned) - _NO_DELEGATOR[cluster]) | _STAYED[cluster]
        added = census - expected
        removed = expected - census
        assert not added and not removed, (
            f"{cluster} cluster membership drifted without a wiring-tuple "
            f"update ({module_name}).\n"
            f"  unaccounted screen additions: {sorted(added)}\n"
            f"  screen removals still claimed: {sorted(removed)}\n"
            "A same-named method appearing on or leaving LibraryScreen is a "
            "cluster-membership DECISION: record it in the owning wiring "
            "test's tuple (or its pruned/stayed sets and this census's "
            "frozen lists, with a reason), never silently."
        )
