from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB


def test_migration_v0_to_v1(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 1
