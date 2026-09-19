"""Unqualified Workflow stores cannot disappear from selected or full coverage."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import json
import os
import subprocess
import sys
from pathlib import Path

from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

route = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME']) / 'fixture'
data.mkdir(mode=0o700)
base = '[paths]\ndata_dir=' + json.dumps(str(data)) + '\n'
selector.write_text(base)
selector.chmod(0o600)
admission_authority(bootstrap.default_bootstrap_root())
prepare = """
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
baseline = preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),), options={'allow_partial': True})
_populate_required_dependencies(baseline)
for owner, cls in (
    ('db.evals', EvalsDB), ('db.library_collections', LibraryCollectionsDB),
    ('db.subscriptions', SubscriptionsDB), ('db.scheduled_tasks', ScheduledTasksDB),
):
    path = next(row.path for row in baseline.items if row.owner == owner and row.status == 'missing_required')
    store = cls(path, client_id='workflow-census-fixture')
    store.close()
assert not network_guard.blocked_attempts()
"""
result = subprocess.run([sys.executable, '-c', prepare], capture_output=True,
                        text=True, timeout=30)
assert result.returncode == 0, result.stderr[-4000:]
choices = ({}, {'data_groups': ('prompts',)})
for options in choices:
    baseline = preview_capture((selector,), options=options)
    assert baseline.complete, baseline.issues

outside = Path.home() / 'custom-workflow'
outside.mkdir(mode=0o700)
workflow = outside / 'authoring.db'
if route == 'default_existing':
    workflow = user_data_dir({'paths': {'data_dir': str(data)}}) / 'tldw_chatbook_workflows.db'
elif route.startswith('custom_'):
    selector.write_text(base + '[database]\nworkflows_db_path=' + json.dumps(str(workflow)) + '\n')
elif route == 'empty':
    selector.write_text(base + '[database]\nworkflows_db_path=""\n')

if route.endswith('_existing'):
    create = """
import sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
store = WorkflowsDB(Path(sys.argv[1]))
store.close()
assert not network_guard.blocked_attempts()
"""
    result = subprocess.run([sys.executable, '-c', create, str(workflow)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-4000:]
before = None if not workflow.exists() else (workflow.read_bytes(), workflow.stat().st_ino)
for options in choices:
    preview = preview_capture((selector,), options=options)
    if route in {'unset', 'empty'}:
        assert preview.complete, preview.issues
        assert not any(row.owner == 'workflows.local' for row in preview.items)
    else:
        assert not preview.complete
        rows = [row for row in preview.items if row.path == workflow]
        assert len(rows) == 1
        assert rows[0].owner == ('unknown' if route == 'default_existing' else 'workflows.local')
        assert rows[0].status == 'unsupported'
        assert 'unsupported' in preview.issues
    assert (None if not workflow.exists() else (workflow.read_bytes(), workflow.stat().st_ino)) == before
assert not network_guard.blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "route",
    ["custom_existing", "custom_absent", "default_existing", "unset", "empty"],
)
def test_workflow_inventory_refuses_unsupported_locations_without_opening_them(
    tmp_path, route
):
    _run(tmp_path, route, "census", script=_SCRIPT, timeout=90)
