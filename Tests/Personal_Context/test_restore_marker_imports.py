"""Unavailable-keyring status must not load the backup schema catalogs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest


def test_missing_keyring_status_reads_marker_without_backup_catalogs(tmp_path):
    script = dedent(
        r'''
        import sys
        from pathlib import Path

        import keyring
        from keyring.backends.null import Keyring as NullKeyring

        from Tests.network_guard import blocked_attempts, install
        install()
        keyring.get_keyring = lambda: NullKeyring()
        from tldw_chatbook.Personal_Context.bootstrap import bootstrap_personal_context_service

        path = Path(sys.argv[1]) / 'absent-personal-context.db'
        assert not path.exists()
        service = bootstrap_personal_context_service(db_path=path)
        status = service.status()
        assert status.locked and status.reason_code == 'profile_locked', status.reason_code
        assert not path.exists(), 'status created an absent profile database'
        assert not blocked_attempts()
        # Last: the original path fails only because marker lookup imports catalogs.
        unexpected = sorted(name for name in sys.modules if name in {
            'tldw_chatbook.Personal_Context.recovery',
            'tldw_chatbook.DB.recovery_operations',
            'tldw_chatbook.DB.recovery_workflow_schema',
        })
        assert not unexpected, unexpected
        print('locked status preserved without backup catalogs')
        '''
    )
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', '-c', script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert 'locked status preserved without backup catalogs' in result.stdout


def test_restore_marker_owner_refuses_writable_creation(tmp_path):
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    destination = tmp_path / "absent.db"
    with pytest.raises(ValueError, match="read-only"):
        connect_private_sqlite("personal_context.restore_marker", destination)
    assert not destination.exists()
