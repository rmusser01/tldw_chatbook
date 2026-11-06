"""Atomic first-conversation policy and project-context persistence."""

from . import chat_persistence_service as host


def persist_console_conversation_with_policy(
    self,
    *,
    conversation_id: str,
    policy_candidate: host.ConsoleLibraryPolicyCandidate,
    conversation_kwargs: host.Mapping[str, object],
    project_context_json: str | None = None,
) -> host.ConsoleLibraryPolicySnapshot:
    """Commit a first conversation row and its Library policy together."""
    self.validate_workspace_target(**conversation_kwargs)
    with self.db.transaction(immediate=True):
        created_id = self.create_conversation(
            conversation_id=conversation_id,
            **dict(conversation_kwargs),
        )
        if created_id != conversation_id:
            raise RuntimeError("Persistence returned an unexpected conversation id.")
        result = self.console_library_policy_repository.insert(
            conversation_id,
            policy_candidate,
        )
        if result.status is not host.ConsoleLibraryPolicyWriteStatus.COMMITTED:
            raise RuntimeError(
                "Console Library policy could not be committed with conversation."
            )
        if project_context_json is not None:
            self.db.set_conversation_console_project_context(
                conversation_id, project_context_json
            )
    return result.snapshot
