"""A real native backup intent holds scheduler work until config is readable."""

import sys

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio
import threading
import time
from pathlib import Path

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import execution_scope
from tldw_chatbook.Backup_Recovery.admission import AdmissionCancelled
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

storage.admit_startup()

async def main():
    database = ScheduledTasksDB(Path.home() / 'scheduled.db')
    reminder = database.create_reminder_task(
        owner_id='local', title='Preserved through backup', schedule_kind='one_time',
        next_run_at='2026-01-01T00:00:00+00:00',
    )
    completed = []
    async def handler(row):
        completed.append(row['id'])
    scheduler = SchedulerLoop(database, {'reminder': handler})
    scheduler.queue.load()
    hold = next(iter(storage._holds.values()))
    cancel = threading.Event()
    errors = []
    def requesting():
        try:
            with hold.authority.maintenance(hold.names, 15, cancel=cancel):
                errors.append('maintenance entered while startup was held')
        except AdmissionCancelled:
            pass
        except BaseException as error:
            errors.append(repr(error))

    try:
        assert not await scheduler._emergency_stopped()
        # The scheduler admitted this tick before another process requested
        # maintenance; that request may reach a later config read before the
        # monitor has closed and drained local scheduler intake.
        with scheduler._maintenance_operation(), execution_scope(
            ('db.scheduled_tasks',), database.db_path,
        ) as allowed:
            assert allowed
            request = threading.Thread(target=requesting)
            request.start()
            try:
                deadline = time.monotonic() + 10
                while not hold.authority.pause_requested(hold.names):
                    assert time.monotonic() < deadline, 'native intent not observed'
                    await asyncio.sleep(.01)
                assert storage._pause is None
                assert await scheduler._emergency_stopped()
                await scheduler._dispatch_due(scheduler.clock())
                assert len(scheduler.queue) == 1 and not completed
            finally:
                cancel.set()
                await asyncio.to_thread(request.join, 10)
                assert not request.is_alive() and not errors, errors
        assert not hold.authority.pause_requested(hold.names)
        assert not await scheduler._emergency_stopped()
        await scheduler.tick()
        assert completed == [reminder]
        assert database.get_reminder_task(reminder)['last_status'] == 'completed'
        assert not scheduler.queue
    finally:
        database.close()

try:
    asyncio.run(main())
finally:
    storage._shutdown()
print('retired and reopened')
'''


def test_native_pause_intent_preserves_due_work_until_config_readmission(tmp_path):
    """No queue item is consumed while the real native intent refuses config."""
    _run(
        tmp_path, "scheduler", "intent", script=_SCRIPT,
        timeout=90 if sys.platform == "win32" else 45,
    )
