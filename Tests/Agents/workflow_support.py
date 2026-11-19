"""Structural workflow fixtures; these grant no runtime authority."""

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents.goal_models import GoalBindingRef, GoalProviderRef
from tldw_chatbook.Agents.workflow_models import WorkflowRequest


def make_workflow_request(root):
    return WorkflowRequest(
        conversation_id="workflow-test-chat",
        objective="Correct the selected source",
        binding=GoalBindingRef(
            workspace_id="w", binding_id="b", locator=str(root.resolve()), access="rw"
        ),
        targets=(
            GoalProviderRef(
                provider="openai",
                model="test-model",
                config_ref="api_settings.openai",
                authority_ref="unauthenticated",
                endpoint_ref="provider-default",
            ),
        ),
        policy=AutomaticWorkLimits(generations=8, child_launches=12),
    )


def plan_data(root):
    request = make_workflow_request(root).model_dump(mode="json")
    return {
        "request": request,
        "root_identity": "device:1/inode:2",
        "roles": ["worker"],
        "tools": ["fs_read", "fs_write"],
        "deliverables": ["corrected source"],
        "phases": [
            {
                "id": "phase-1",
                "tasks": [
                    {
                        "id": "task-1",
                        "role": "worker",
                        "instructions": "Correct source",
                        "target": request["targets"][0],
                        "footprint": [
                            {"path": "source.py", "access": "write", "subtree": False}
                        ],
                        "allowed_tools": ["fs_read", "fs_write"],
                        "outputs": ["source.py"],
                    }
                ],
            }
        ],
    }
