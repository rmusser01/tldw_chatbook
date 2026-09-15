"""Live manager visibility and experiment intake use real private sources."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import gc, sys, threading, weakref
from pathlib import Path
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant, retained_issues
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ExperimentConfig, get_profile_manager, reset_profile_manager_cache
route = sys.argv[1]
manager = get_profile_manager()
app = SimpleNamespace(_rag_service=None)
def refused(call):
    try:
        call()
    except RecoveryRequired:
        return
    except TypeError as error:
        raise AssertionError('paused operation entered original serialization') from error
    raise AssertionError('operation did not refuse')
if route == 'census':
    alternate = get_profile_manager(Path.home()/'alternate')
    assert any(i.reason == 'alternate_source_unqualified' for i in retained_issues(app)), 'uncached manager invisible'
    reset_profile_manager_cache()
    newer = get_profile_manager()
    assert {manager, newer, alternate} <= set(ConfigProfileManager.live_instances())
    reference = weakref.ref(alternate)
    del alternate
    gc.collect()
    assert reference() is None, 'census retains owner'
    assert not retained_issues(app)
elif route == 'constructor':
    participant._maintenance_close_admission()
    destination = Path.home()/'not-created'
    try:
        refused(lambda: ConfigProfileManager(destination))
        assert not destination.exists()
    finally:
        participant._maintenance_resume()
    reopened = ConfigProfileManager(destination)
    assert destination.is_dir()
elif route == 'active':
    experiment = ExperimentConfig(save_results=False)
    manager._current_experiment = experiment
    manager._experiment_results[experiment.experiment_id] = []
    refused(participant._maintenance_close_admission)
    manager.record_experiment_result('hybrid_basic', 'private query', {'search_latency': 1})
    assert manager._experiment_results[experiment.experiment_id][0]['query'] == 'private query'
    assert manager.end_experiment()['total_queries'] == 1
    participant._maintenance_close_admission()
    participant._maintenance_resume()
elif route == 'transition':
    # No model service constructor: exact real methods on its initialized
    # experiment-only fields exercise the wrapper without opening engines.
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
    service = object.__new__(EnhancedRAGServiceV2)
    service.profile_manager = manager
    service._current_experiment = None
    entered, release = threading.Event(), threading.Event()
    original = manager.start_experiment
    def blocked(config):
        entered.set()
        assert release.wait(5)
        return original(config)
    manager.start_experiment = blocked
    errors = []
    def run():
        try:
            service.start_experiment(ExperimentConfig(save_results=False))
        except TypeError as error:
            errors.append(error)  # Existing Path JSON serialization failure.
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        refused(participant._maintenance_close_admission)
    finally:
        release.set()
        worker.join(5)
        participant._maintenance_resume()
    assert not worker.is_alive()
    assert len(errors) == 1
    refused(participant._maintenance_close_admission)
    service.end_experiment()
    participant._maintenance_close_admission()
    participant._maintenance_resume()
elif route == 'service_refusal':
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
    service = object.__new__(EnhancedRAGServiceV2)
    service.profile_manager = manager
    service._current_experiment = None
    participant._maintenance_close_admission()
    try:
        refused(lambda: service.start_experiment(ExperimentConfig()))
        assert service._current_experiment is None
        assert manager._current_experiment is None
    finally:
        participant._maintenance_resume()
elif route == 'nested_refusal':
    @participant.operation
    def accepted():
        participant._maintenance_close_admission()
        try:
            refused(lambda: manager.start_experiment(ExperimentConfig()))
            assert manager._current_experiment is None
        finally:
            participant._maintenance_resume()
    accepted()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "census",
        "constructor",
        "active",
        "transition",
        "service_refusal",
        "nested_refusal",
    ],
)
def test_manager_maintenance_boundaries(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)
