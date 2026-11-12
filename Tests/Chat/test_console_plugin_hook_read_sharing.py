"""Shared hook reads cannot erase retained plugin requirements after rebase."""

from types import SimpleNamespace

import pytest

from Tests.Chat import test_console_send_hook_read_sharing as sharing
from tldw_chatbook.Chat.console_hook_preparation import ConsoleHookAttemptRead

hook_file = sharing.hook_file
plugin_rig = sharing.rig

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize("capture", ["empty", "owned-row", "retained-requirement"])
async def test_shared_read_preserves_plugin_requirement_verification(
    plugin_rig, monkeypatch, capture
):
    rig = plugin_rig
    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    maximum = {"available_skills": []}
    if capture == "owned-row":
        maximum["available_skills"] = [{"plugin_owned": True}]
    elif capture == "retained-requirement":
        # Opaque proof remains for its real owner to authenticate. This test
        # substitutes that boundary's refusal, never issues plugin authority.
        maximum["_plugin_requirement_ceiling"] = object()
    configuration = SimpleNamespace(skill_context_maximum=maximum)

    async def hook_configuration(
        captured_maximum, *, scratch_manager, scratch_snapshot
    ):
        assert captured_maximum is maximum
        assert scratch_manager is runtime._scratch_spaces
        assert scratch_snapshot == runtime._scratch_spaces.snapshot(session_id)
        raise PermissionError("plugin_constraint_verification_failed")

    service = SimpleNamespace(hook_configuration=hook_configuration)
    skills = SimpleNamespace(local_service=SimpleNamespace(plugin_service=service))
    monkeypatch.setattr(rig.controller, "_skills_service", skills)
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_v2_configuration(shared.authority) is not None
    rig.reads.clear()
    if capture == "empty":
        assert (
            await runtime.prepare_hooks_v2(
                session_id, configuration=configuration, _hook_read=shared
            )
            is None
        )
        assert not rig.reads
    else:
        with pytest.raises(
            PermissionError, match="^plugin_constraint_verification_failed$"
        ):
            await runtime.prepare_hooks_v2(
                session_id, configuration=configuration, _hook_read=shared
            )
        assert len(rig.reads) == 1
    assert runtime.get_hooks_v2(session_id) is None
