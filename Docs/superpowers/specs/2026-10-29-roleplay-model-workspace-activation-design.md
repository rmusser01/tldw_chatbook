# Preserve roleplay chat activation through model selection

The user approved fixing the investigated root cause and merging a reviewed PR
against `dev`. Starting a global character/persona chat from a named workspace
must move the workspace registry to that conversation's scope before returning
control to the UI. Applying or cancelling Switch model must preserve its chat,
messages and assistant identity.

## Approach

Reuse `ConsoleWorkspaceController._set_active_workspace_for_console_session` at
the affected roleplay activation sites. The session controller already consumes
that seam; the character controller receives the same named constructor callback.
Keep pristine-tab reuse and conversation scope unchanged. Keep resume-time
reconciliation, which must still honor workspace changes made in other screens.

Suppressing resume reconciliation would hide stale authority and break intentional
cross-screen workspace changes. Moving every store activation into the UI layer
would broaden this bug fix beyond the affected entry points.

## Verification

Regressions cover picker-created character chats, character/persona handoffs,
pristine-tab reuse, and model-picker Apply/Cancel from a named workspace. Assert
the originating session, messages, assistant identity, applied provider/model,
workspace registry and session count. Use test provider listings and no model API.

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md;
backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: restore existing activation ownership without changing scope or policy.
