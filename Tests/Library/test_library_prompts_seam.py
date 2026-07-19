"""``count_prompts`` seam contracts (Library Prompts Task 1).

Mirrors the Notes precedent (``NotesScopeService.count_notes`` /
``Notes_Library.count_notes``): an exact, non-deleted total that the
Library rail badge can render without needing a full paginated fetch just
to read a number.
"""

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.local_prompt_service import LocalPromptService
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY


@pytest.mark.asyncio
async def test_count_prompts_counts_non_deleted(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    svc = LocalPromptService(db)  # match the service's real constructor; adjust if it takes a provider callable
    assert await svc.count_prompts() == 0
    db.add_prompt(name="alpha", author="t", details="d", user_prompt="hello")
    db.add_prompt(name="beta", author="t", details="d", user_prompt="world")
    assert await svc.count_prompts() == 2
    db.soft_delete_prompt("alpha")
    assert await svc.count_prompts() == 1


@pytest.mark.asyncio
async def test_prompt_scope_service_count_prompts_passes_through_to_local_backend(tmp_path):
    """``PromptScopeService.count_prompts(mode="local")`` routes to the local
    backend's own ``count_prompts`` -- mirroring ``NotesScopeService.count_notes``'s
    local-scope passthrough -- without requiring a dedicated ``count``
    policy action (reuses the existing ``list`` action, same as notes)."""
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    local_service = ScopeLocalPromptService(db)
    service = PromptScopeService(local_service=local_service, server_service=None)

    assert await service.count_prompts(mode="local") == 0
    db.add_prompt(name="alpha", author="t", details="d", user_prompt="hello")
    db.add_prompt(name="beta", author="t", details="d", user_prompt="world")
    assert await service.count_prompts(mode="local") == 2
    db.soft_delete_prompt("alpha")
    assert await service.count_prompts(mode="local") == 1

    # Defaults to "local" per the interface contract.
    assert await service.count_prompts() == 1


@pytest.mark.asyncio
async def test_prompt_scope_service_count_prompts_rejects_server_backend(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    local_service = ScopeLocalPromptService(db)
    service = PromptScopeService(local_service=local_service, server_service=object())

    with pytest.raises(ValueError):
        await service.count_prompts(mode="server")


class _ActionRecordingPolicyEnforcer:
    """Records every ``action_id`` passed to ``require_allowed`` without ever
    denying -- lets a test enumerate exactly what a scope-service call emits.

    Mirrors ``Tests/Prompt_Management/test_prompt_scope_service.py``'s
    ``FakePolicyEnforcer`` (kept local here rather than imported across test
    modules, since this is the only behavior this file needs from it).
    """

    def __init__(self) -> None:
        self.actions: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.actions.append(action_id)


@pytest.mark.asyncio
async def test_prompt_scope_service_local_action_ids_from_library_ui_flows_are_registered(tmp_path):
    """Regression test for the Library Prompts Phase-1 gate defect: clicking
    a prompt row raised ``PolicyDeniedError: Unknown runtime-policy
    action_id: prompts.detail.local`` (from ``PromptScopeService.get_prompt``
    -> ``_enforce_policy``) because the runtime-policy registry
    (``tldw_chatbook/runtime_policy/registry.py``) had no ``DETAIL`` entry
    for the base ``prompts`` resource -- only ``prompts.preview.local``
    (a distinct, pre-existing action used by the unrelated prompt-chatbook
    "preview" flow, see ``Prompt_Management/prompt_chatbook_scope_service.py``)
    was registered.

    Exercises every ``PromptScopeService`` call the Library Prompts UI makes
    with ``mode="local"`` -- list/count/search (the rail snapshot seam),
    ``get_prompt`` (``_refresh_library_prompt_detail``), ``save_prompt``
    create + update (``_save_library_prompt``), and ``delete_prompt``
    (``_delete_library_prompt``) -- against a real ``PromptsDatabase``-backed
    local service, and asserts every action_id it emits is a known key of
    ``CAPABILITY_REGISTRY`` -- the same dict ``PolicyEngine.evaluate`` (and
    therefore ``ServicePolicyEnforcer.require_allowed``) looks up.
    """
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    local_service = ScopeLocalPromptService(db)
    policy = _ActionRecordingPolicyEnforcer()
    service = PromptScopeService(local_service=local_service, server_service=None, policy_enforcer=policy)

    created = await service.save_prompt(mode="local", name="Summarize", user_prompt="Summarize: {text}")
    prompt_id = created["local_id"]
    await service.list_prompts(mode="local")
    await service.count_prompts(mode="local")
    await service.search_prompts(mode="local", query="Summarize")
    await service.get_prompt(mode="local", prompt_identifier=prompt_id)
    await service.save_prompt(mode="local", prompt_identifier=prompt_id, name="Summarize v2")
    await service.delete_prompt(mode="local", prompt_identifier=prompt_id)

    emitted_action_ids = set(policy.actions)
    assert emitted_action_ids == {
        "prompts.create.local",
        "prompts.list.local",
        "prompts.detail.local",
        "prompts.update.local",
        "prompts.delete.local",
    }

    unregistered = sorted(emitted_action_ids - set(CAPABILITY_REGISTRY))
    assert not unregistered, (
        "PromptScopeService emitted local-mode action_ids the runtime-policy "
        f"registry does not recognize: {unregistered}"
    )
