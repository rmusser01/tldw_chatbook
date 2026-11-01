"""Manual tool execution shares refusal types without loading automatic policy."""

import pytest

from Tests.private_profile import private_profile_test


_REFUSAL_IMPORT_ORDER_SCRIPT = r"""
import importlib
import json
import sys

prefix = "tldw_chatbook.Agents."
assert prefix + "agent_models" not in sys.modules
assert prefix + "automatic_work_budget" not in sys.modules
first_module = sys.argv[1]
assert first_module in {"agent_models", "automatic_work_budget"}
importlib.import_module(prefix + first_module)
models = importlib.import_module(prefix + "agent_models")
budget = importlib.import_module(prefix + "automatic_work_budget")
assert budget.AutomaticWorkRefused is models.AutomaticWorkRefused
cases = (
    ("cancelled", "cancelled"),
    ("tokens_budget", "automatic_limit"),
    ("review_required", "unknown_effect"),
    ("acceptance_required", "authority_changed"),
)
for reason, expected in cases:
    refusal = budget.AutomaticWorkRefused(reason)
    try:
        raise refusal
    except models.AutomaticWorkRefused as caught:
        assert caught is refusal
        assert caught.reason == reason
        assert str(caught) == reason
        assert caught.termination_reason.value == expected
assert "tldw_chatbook.app" not in sys.modules
assert prefix + "automatic_work_runtime" not in sys.modules
assert prefix + "agent_service" not in sys.modules
print("BUDGET_JSON:" + json.dumps({"first_module": first_module, "checked": len(cases)}))
"""


@pytest.mark.asyncio
@private_profile_test
async def test_cold_manual_tool_worker_does_not_load_automatic_budget(request):
    import sys
    import threading

    budget_module = "tldw_chatbook.Agents.automatic_work_budget"
    assert budget_module not in sys.modules

    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import _call_with_timeout
    from tldw_chatbook.Agents.automatic_work_runtime import (
        current_automatic_work,
        manual_work_scope,
    )

    caller = threading.current_thread()
    workers = []

    def run_tool():
        workers.append(threading.current_thread())
        assert current_automatic_work() is None
        return ToolResult(ok=True, content="manual worker completed")

    with manual_work_scope():
        assert current_automatic_work() is None
        result = _call_with_timeout(run_tool, 5.0, "manual-import-boundary")

    assert result.ok is True
    assert result.content == "manual worker completed"
    assert result.dispatch_state == "settled"
    assert len(workers) == 1
    assert workers[0] is not caller
    assert not workers[0].is_alive()
    assert current_automatic_work() is None
    assert budget_module not in sys.modules


@pytest.mark.asyncio
@pytest.mark.parametrize("first_module", ["agent_models", "automatic_work_budget"])
@private_profile_test
async def test_refusal_alias_import_orders_preserve_worker_classification(
    request, tmp_path, first_module
):
    import importlib

    from Tests.Performance.test_screen_preimport_payload_budget import _run_census

    # The private pytest fixture already admits agent_models. Check pure-model
    # import order in a fresh interpreter without replacing that admitted state.
    census = _run_census(tmp_path, _REFUSAL_IMPORT_ORDER_SCRIPT, first_module)
    assert census == {"first_module": first_module, "checked": 4}

    prefix = "tldw_chatbook.Agents."
    models = importlib.import_module(prefix + "agent_models")
    budget = importlib.import_module(prefix + "automatic_work_budget")
    runtime = importlib.import_module(prefix + "automatic_work_runtime")
    from tldw_chatbook.Agents.agent_service import _call_with_timeout

    assert budget.AutomaticWorkRefused is models.AutomaticWorkRefused
    assert runtime.AutomaticWorkRefused is models.AutomaticWorkRefused
    reason_groups = {
        models.RunTerminationReason.CANCELLED: ("cancelled",),
        models.RunTerminationReason.AUTOMATIC_LIMIT: (
            "generation_budget",
            "child_launch_budget",
            "model_call_budget",
            "tokens_budget",
            "output_tokens_budget",
            "wall_budget",
            "iteration_wall_budget",
        ),
        models.RunTerminationReason.UNKNOWN_EFFECT: (
            "history_unavailable",
            "review_required",
            "clock_unknown",
            "clock_reversed",
        ),
        models.RunTerminationReason.AUTHORITY_CHANGED: ("acceptance_required",),
    }
    for expected, reasons in reason_groups.items():
        for reason in reasons:
            refusal = budget.AutomaticWorkRefused(reason)
            with pytest.raises(models.AutomaticWorkRefused) as caught:
                raise refusal
            assert caught.value is refusal
            assert refusal.reason == reason
            assert str(refusal) == reason

            def refuse_tool():
                raise refusal

            result = _call_with_timeout(refuse_tool, 5.0, "refusal-alias")
            assert result.ok is False
            assert result.error == reason
            assert result.dispatch_state == "settled"
            assert result.termination_reason is expected
