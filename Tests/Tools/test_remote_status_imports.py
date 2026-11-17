"""Local status projection must not initialize the SSH execution stack."""

from pathlib import Path
import subprocess
import sys
from textwrap import dedent


def test_remote_status_works_without_loading_ssh_execution():
    script = dedent(
        '''
        import sys
        from tldw_chatbook.Tools import remote_binding_status as status

        cache = status.RemoteBindingStatusCache()
        assert cache.status('fixture').state == status.BindingState.READY
        cache.record_transport_failure(
            'fixture', status.TransportFailureKind.UNREACHABLE, 'fixture', now=1.0
        )
        assert cache.status('fixture').state == status.BindingState.BLOCKED
        cache.record_transport_failure(
            'fixture', status.TransportFailureKind.OP_TIMEOUT, 'fixture', now=2.0
        )
        assert cache.status('fixture').state == status.BindingState.BLOCKED
        assert status.classify_probe_result(status.RemoteCallResult(
            admitted=False, response=b'{"outcome":"success"}', failure=None
        )) == status.PROBE_READY
        # Last: this is the cold-dependency regression, after actual behavior.
        unexpected = sorted(name for name in sys.modules if name in {
            'tldw_chatbook.Tools.remote_workspace_transport',
            'tldw_chatbook.Tools.remote_workspace_executor',
            'tldw_chatbook.Tools.remote_worker_bundle',
        })
        assert not unexpected, unexpected
        # Existing consumers still import exactly the same public types.
        from tldw_chatbook.Tools import remote_workspace_transport as legacy
        assert legacy.RemoteCallResult is status.RemoteCallResult
        assert legacy.TransportFailureKind is status.TransportFailureKind
        refusal = legacy.RemoteCallResult(False, None, legacy.TransportFailure(
            legacy.TransportFailureKind.UNREACHABLE, 255, 'fixture'
        ))
        assert status.classify_probe_result(refusal) == status.PROBE_BLOCKED
        print('status behavior and legacy exports preserved')
        '''
    )
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert 'status behavior and legacy exports preserved' in result.stdout
