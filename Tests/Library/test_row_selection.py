from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def test_toggle_add_remove():
    s = RowSelection("media")
    s.toggle("1"); s.toggle("2")
    assert s.is_selected("1") and s.count == 2
    s.toggle("1")
    assert not s.is_selected("1") and s.count == 1
    s.toggle("")  # empty id ignored
    assert s.count == 1


def test_select_all_and_clear():
    s = RowSelection("notes")
    s.select_all(["a", "b", "", "c"])   # empties skipped
    assert s.ids == frozenset({"a", "b", "c"})
    s.clear()
    assert s.count == 0


def test_reconcile_drops_absent_ids():
    s = RowSelection("conversations")
    s.select_all(["a", "b", "c"])
    s.reconcile(["b", "c", "d"])        # 'a' no longer rendered
    assert s.ids == frozenset({"b", "c"})


def test_export_scope_sorts_ids():
    s = RowSelection("media")
    s.select_all(["10", "2", "1"])
    assert s.export_scope() == ExportScope(kind="media", ids=("1", "10", "2"))
