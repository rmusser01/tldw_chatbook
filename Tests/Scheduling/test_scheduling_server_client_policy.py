"""task-2722: the scheduling server client must keep policy refusals typed."""

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientPolicyError,
    ServerClientValidationError,
)


def test_policy_error_is_a_validation_error_subtype():
    # Existing callers catch ServerClientValidationError; the refined type must
    # not slip past them.
    assert issubclass(ServerClientPolicyError, ServerClientValidationError)


@pytest.mark.asyncio
async def test_client_translates_policy_denial_to_policy_error():
    class DenyingService:
        async def list_reminders(self, **kwargs):
            raise PolicyDeniedError(
                action_id="notifications.reminders.list.server",
                reason_code="server_mode_required",
                user_message=(
                    "notifications.reminders.list.server requires server mode."
                ),
                effective_source="local",
                authority_owner="server",
            )

    client = SchedulingServerClient(DenyingService())
    with pytest.raises(ServerClientPolicyError):
        await client.list_reminders()
