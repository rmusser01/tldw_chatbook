"""Real evaluation ownership includes admission, callbacks, and native tails."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time, threading
from pathlib import Path
from Tests.Evals.test_eval_orchestrator import _seed_run_inputs, _ControlledEvalRunner, _orchestrator_result
import tldw_chatbook.Evals.eval_orchestrator as module
import tldw_chatbook.Evals.eval_runner as runners

async def main():
    case = sys.argv[1]
    if case.startswith("native"):
        entered, release = threading.Event(), threading.Event()
        def dispatch(**kwargs):
            entered.set()
            assert release.wait(4)
            return "native output"
        runners.chat_api_call = dispatch
        runner = object.__new__(runners.QuestionAnswerRunner)
        runner.provider_name, runner.model_id, runner.api_key = "mock", "model", None
        runner.request_timeout = .02 if case == "native_timeout" else 30
        task = asyncio.create_task(runner._call_llm("test"))
        assert await asyncio.to_thread(entered.wait, 1)
        try:
            if case == "native_cancel": task.cancel()
            await asyncio.sleep(.08)
            assert not task.done()
        finally:
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
        expected = TimeoutError if case == "native_timeout" else asyncio.CancelledError
        assert isinstance(result[0], expected)
    else:
        owner = module.EvaluationOrchestrator(Path.home()/"evals.db")
        task_id, model_id = _seed_run_inputs(owner)
        module.EvalRunner = lambda *args: _ControlledEvalRunner([_orchestrator_result("one")])
        entered, release = asyncio.Event(), asyncio.Event()
        name = "register_run" if case == "register" else "unregister_run"
        original = getattr(owner.concurrent_manager, name)
        async def boundary(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)
        setattr(owner.concurrent_manager, name, boundary)
        task = asyncio.create_task(owner.run_evaluation(task_id, model_id))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not owner._active_tasks
            owner._maintenance_close_admission()
            assert not await owner._maintenance_drain(time.monotonic())
            before = owner.db.list_runs()
            try: await owner.run_evaluation(task_id, model_id)
            except RuntimeError as error: assert str(error) == "evaluations_paused_for_maintenance"
            else: raise AssertionError("new evaluation was admitted")
            assert owner.db.list_runs() == before
        finally:
            release.set()
            run_id = await task
        assert await owner._maintenance_drain(time.monotonic()+1)
        assert owner.db.get_run(run_id)["status"] == "completed"
        assert len(owner.db.get_results_for_run(run_id)) == 1
        owner._maintenance_resume()
        second = await owner.run_evaluation(task_id, model_id)
        assert second != run_id
        owner.close()
    print("retired and reopened")
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "case", ["register", "unregister", "native_cancel", "native_timeout"]
)
def test_eval_maintenance(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)
