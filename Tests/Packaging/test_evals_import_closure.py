"""Evaluation package exports must not eagerly import unrelated runners."""

from __future__ import annotations

import pytest

from Tests.Packaging.test_chat_persistence_import_closure import _run_isolated_python


@pytest.mark.parametrize(
    "name, module_name",
    [
        ("EvaluationOrchestrator", "eval_orchestrator"),
        ("TaskLoader", "task_loader"),
        ("TaskLoadError", "task_loader"),
    ],
)
def test_evals_exports_are_lazy_and_preserve_canonical_identity(
    tmp_path, name, module_name
):
    result = _run_isolated_python(
        tmp_path,
        f"""
import importlib
import sys
import tldw_chatbook.Evals as package
assert package.__all__ == ['EvaluationOrchestrator', 'TaskLoader', 'TaskLoadError']
assert set(package.__all__) <= set(dir(package))
for module in ('eval_orchestrator', 'eval_runner', 'task_loader'):
    assert 'tldw_chatbook.Evals.' + module not in sys.modules
try:
    getattr(package, '_missing_export')
except AttributeError:
    pass
else:
    raise AssertionError('Unsupported export must raise AttributeError')
export = getattr(package, {name!r})
canonical = getattr(importlib.import_module('tldw_chatbook.Evals.' + {module_name!r}), {name!r})
assert export is canonical
assert getattr(package, {name!r}) is export
namespace = {{}}
exec('from tldw_chatbook.Evals import *', namespace)
for public_name in package.__all__:
    assert namespace[public_name] is getattr(package, public_name)
print('EVALS_EXPORTS_OK')
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "EVALS_EXPORTS_OK" in result.stdout


def test_evaluation_normalizers_do_not_load_the_server_client(tmp_path):
    result = _run_isolated_python(
        tmp_path,
        """
import sys
import tldw_chatbook.Evaluations_Interop as package
from tldw_chatbook.Evaluations_Interop.evaluation_normalizers import RESERVED_LOCAL_DATASET_SAMPLES_KEY
assert RESERVED_LOCAL_DATASET_SAMPLES_KEY == '__tldw_eval_samples__'
assert 'tldw_chatbook.Evaluations_Interop.server_evaluations_service' not in sys.modules
assert 'httpx' not in sys.modules
assert set(package.__all__) <= set(dir(package))
try:
    getattr(package, '_missing_export')
except AttributeError:
    pass
else:
    raise AssertionError('Unsupported export must raise AttributeError')
export = package.ServerEvaluationsService
from tldw_chatbook.Evaluations_Interop.server_evaluations_service import ServerEvaluationsService
assert export is ServerEvaluationsService
assert package.ServerEvaluationsService is export
namespace = {}
exec('from tldw_chatbook.Evaluations_Interop import *', namespace)
for public_name in package.__all__:
    assert namespace[public_name] is getattr(package, public_name)
print('EVALUATIONS_INTEROP_CLOSURE_OK')
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "EVALUATIONS_INTEROP_CLOSURE_OK" in result.stdout
