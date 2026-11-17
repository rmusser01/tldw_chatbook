"""The selected backend's original synchronous pre-mount permission entry."""

from __future__ import annotations

from .console_execution_process import ConsoleExecutionClient, ConsoleProcessProfile


class ConsoleProcessPermissionService:
    """Forward pre-mount setup to the sole backend permission owner.

    Mounted Settings actions retain their asynchronous PermissionAction path.
    This object contains neither a permission store nor a permission cache.
    """

    def __init__(self, client: ConsoleExecutionClient, profile: ConsoleProcessProfile):
        if type(client) is not ConsoleExecutionClient or type(profile) is not ConsoleProcessProfile:
            raise TypeError("Original Console process owner and profile are required")
        self._client, self._profile = client, profile

    def set_global_default(
        self,
        state: str,
        *,
        profile_id: str = "default",
        expected_profile_digest: str | None = None,
        expected_revision: int | None = None,
    ) -> None:
        """Return only after the original backend mutation physically retires."""
        self._client.set_global_default_before_mount(
            self._profile,
            state=state,
            profile_id=profile_id,
            expected_profile_digest=expected_profile_digest,
            expected_revision=expected_revision,
        )
