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
