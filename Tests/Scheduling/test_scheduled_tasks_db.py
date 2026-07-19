from tldw_chatbook.config import get_scheduled_tasks_db_path


def test_get_scheduled_tasks_db_path_returns_path():
    path = get_scheduled_tasks_db_path()
    assert path.name == "tldw_chatbook_scheduled_tasks.db"
