"""Private-process review/workflow admission integration, without external calls."""

import asyncio
import json
import os
import subprocess
import threading
import sys
from pathlib import Path

import pytest

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

os.environ.update(
    TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true",
    TLDW_AGENTS_AUTOWAKE_ENABLED="false",
    OPENAI_API_KEY="private-native-review-test",
)
network_guard.install()
real_profile_guard.install()


async def exercise():
    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from Tests.Chat.test_automatic_provider_budget import response
    from tldw_chatbook import config
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    mode = sys.argv[1]
    is_review = mode == "review-held"
    retained_failure = False
    scope_failed = threading.Event()
    failed_threads = []
    config.save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    config.save_setting_to_cli_config(
        "model_capabilities",
        "models",
        {
            "test-model": {
                "context_window": 32768,
                "max_input_tokens": 30000,
                "max_output_tokens": 2048,
            }
        },
    )
    from tldw_chatbook.model_capabilities import reload_capabilities

    reload_capabilities()

    class ReviewProvider(WorkflowProvider):
        def __init__(self):
            super().__init__()
            self.child_entered = threading.Event()
            self.child_returned = threading.Event()
            self.child_owner = None
            self.rig = None
            self.turns = {}

        def __call__(self, **kwargs):
            actor = current_run_actor()
            index = self.turns.get(actor.run_id, 0)
            self.turns[actor.run_id] = index + 1
            controller = self.rig.controller
            if actor.kind == "subagent":
                self.child_owner = current_execution_owner()
                assert self.child_owner is not None and self.child_owner.child
                if is_review:
                    review = next(iter(controller.code_review._reviews.values()))
                    binding = review.store._tab.runs[actor.run_id]
                    assert binding.role == "reviewer"
                self.child_entered.set()
                assert self.released.wait(25), "original review worker not released"
                self.child_returned.set()
                text = "Review complete."
            elif not is_review:
                text = (
                    fence(
                        "spawn_subagent", {"task": "Write a brief independent summary"}
                    )
                    if index == 0
                    else fence("wait_agents", {})
                    if index == 1
                    else "Delegated task complete."
                )
            elif index == 0:
                text = fence("load_tools", {"ids": ["code_review:code_review_prepare"]})
            elif index == 1:
                text = fence(
                    "code_review_prepare",
                    {
                        "target": {
                            "kind": "working_tree",
                            "binding_ref": self.rig.workflow.request.binding.binding_id,
                            "staged": False,
                            "unstaged": True,
                            "untracked_paths": ["source.py"],
                        }
                    },
                )
            elif index == 2:
                assert controller.code_review._reviews, kwargs
                review_id = next(iter(controller.code_review._reviews))
                text = fence(
                    "spawn_subagent", {"review_id": review_id, "review_slot_id": "R1"}
                )
            elif index == 3:
                text = fence("wait_agents", {})
            else:
                text = "Review turn complete."
            result = response()
            result["choices"][0]["message"]["content"] = text
            return result

    def fence(name, arguments):
        return (
            "```tool_call\n"
            + json.dumps({"name": name, "arguments": arguments})
            + "\n```"
        )

    provider = ReviewProvider()
    patch = pytest.MonkeyPatch()
    rig = await build_workflow_rig(
        Path(os.environ["XDG_DATA_HOME"]).parent,
        patch,
        provider,
        native_fleet_consumers=True,
        approve_missing_file_read=True,
    )
    provider.rig = rig
    root = Path(rig.workflow.request.binding.locator)
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "baseline",
        ],
        check=True,
        capture_output=True,
    )
    (root / "source.py").write_text("value = 1\n")
    rig.controller.app._loop = asyncio.get_running_loop()
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        fingerprint_canonical_locator,
    )

    locator = fingerprint_canonical_locator(str(root))
    rig.store.set_session_project_instruction_state(
        "workflow-test-chat",
        ProjectInstructionControlState(
            True,
            rig.workflow.request.binding.binding_id,
            locator,
        ),
    )
    # Exercise the original lazy initializer before native immutable capture.
    assert rig.controller.app.local_skills_service.trust_service is not None
    for name in ("fs_list", "fs_grep", "git_diff"):
        rig.controller.app.unified_mcp_service.approve_for_session(
            "local:__local__", name, profile_id="default"
        )
    capacity = rig.runtime.execution_capacity
    definition = rig.service.db.create_agent_definition(
        AgentDefinition(
            name="review-native",
            instructions="Review independently",
            provider="openai",
            model="test-model",
        )
    )
    config.save_setting_to_cli_config(
        "agents", "code_review", {"reviewers": [{"preset_id": definition}]}
    )
    footprint = WorktreeFootprint(
        writes=(WorktreeScope(root, True, capture_directory_chain(root)),)
    )
    pending = None

    async def answer_original_dispatches():
        while True:
            for decision in tuple(rig.runtime._project_dispatch_decisions.values()):
                assert decision.session_id == "workflow-test-chat"
                if not decision.completed.is_set():
                    assert rig.runtime.resolve_project_instruction_dispatch(
                        decision.decision_id, "proceed"
                    )
            await asyncio.sleep(0.01)

    answers = asyncio.create_task(answer_original_dispatches())
    try:
        pending = asyncio.create_task(
            rig.controller.submit_draft(
                "Review my changes" if is_review else "Delegate a brief summary",
                session_id="workflow-test-chat",
            )
        )
        async with asyncio.timeout(20):
            while not provider.child_entered.is_set():
                if pending.done():
                    assert pending.result().accepted
                    raise AssertionError(
                        (
                            "review child did not enter",
                            [
                                (m.role, m.content)
                                for m in rig.store.messages_for_session(
                                    "workflow-test-chat"
                                )
                            ],
                        )
                    )
                await asyncio.sleep(0.02)
        original = provider.child_owner
        assert original.is_admitted
        activity = rig.runtime.worktree_activity
        existing = tuple(activity._claims)
        original_tree = next(iter(rig.runtime._ordinary_turns))
        assert original_tree.review_possible is is_review
        assert ("code_review" in original_tree.other_maximum) is is_review
        if mode == "ordinary-scope-error":
            from tldw_chatbook.Chat.console_agent_sources import _retained_source_stack

            scope_code = _retained_source_stack.__wrapped__.__code__

            def interrupt_original_scope_return(frame, event, result):
                exit_frame = frame.f_back
                runner = exit_frame.f_back if exit_frame is not None else None
                if (
                    event == "return"
                    and frame.f_code is scope_code
                    and frame.f_locals.get("tree") is original_tree
                    and frame.f_locals.get("cleanup") == [True]
                    and runner is not None
                    and runner.f_code.co_name == "run_child_owned"
                    and runner.f_locals.get("child_owner") is original
                ):
                    failed_threads.append(threading.current_thread())
                    scope_failed.set()
                    raise RuntimeError("original child source-scope return interrupted")

            threading.setprofile_all_threads(interrupt_original_scope_return)
        assert existing
        with activity.prelaunch(footprint=footprint) as claim:
            assert claim is None
        assert not provider.child_returned.is_set()
        provider.released.set()
        result = await pending
        assert result.accepted and provider.child_returned.is_set()
        if mode == "ordinary-scope-error":
            assert await asyncio.to_thread(scope_failed.wait, 5)
            threading.setprofile_all_threads(None)
            assert len(failed_threads) == 1
            await asyncio.to_thread(failed_threads[0].join, 5)
            assert not failed_threads[0].is_alive()
            assert original.is_admitted and not original._root_finished
            member = original_tree.members[original]
            assert not member.host_returned and not member.durable
            assert original_tree.claim.id in activity._claims
            with activity.prelaunch(footprint=footprint) as claim:
                assert claim is None
            retained_failure = True
            raise RuntimeError("ORIGINAL_CHILD_SCOPE_RETURN_UNCERTAIN_CUSTODY_RETAINED")
        async with asyncio.timeout(10):
            while original.is_admitted or any(
                key in activity._claims for key in existing
            ):
                await asyncio.sleep(0.02)
        with activity.prelaunch(footprint=footprint) as claim:
            assert claim is not None
        print(
            "NATIVE_REVIEW_PHYSICAL_RETURN_RELEASED_ORIGINAL_WORKTREE_CLAIM", flush=True
        )
    finally:
        threading.setprofile_all_threads(None)
        provider.released.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        answers.cancel()
        await asyncio.gather(answers, return_exceptions=True)
        if retained_failure:
            # This negative leaves the same uncertain owner/claim intact.
            # Private process exit is not a successful retirement receipt.
            raise RuntimeError("ORIGINAL_CHILD_SCOPE_RETURN_UNCERTAIN_CUSTODY_RETAINED")
        await rig.close()
        assert not capacity.snapshot().executions
        patch.undo()
        print("retired and reopened", flush=True)


with user_fixture_default_owner():
    asyncio.run(exercise())
