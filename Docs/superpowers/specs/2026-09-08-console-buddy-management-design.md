# Console Buddy & Persona management

Status: Approved by the user for implementation on 2026-09-08.

## Outcome

A user can keep one visible Buddy beside any application destination, independently
of a Persona. Menu → Buddy in the Console and a settings control on the Buddy open
one Buddy & Persona Management modal. A Buddy explicitly follows one conversation
or one workspace. Switching screens, hiding the Buddy, or closing its modal does
not stop accepted work. Chatbook ships first; the same behavior is then ported to
tldw_server. Multiple visible Buddies are v2.

## Independent artwork

The Buddy has a real local visual owner and a versioned visual binding. Existing
immutable visual packs, renderers, validation, content attribution, and private
storage remain authoritative. Selecting artwork does not manufacture a Persona or
change prompts, models, tools, memory, or permissions. Existing Persona-backed
Buddy selection is migrated by copying its visual binding/artwork into the Buddy
owner, preserving attribution and geometry. Removing or editing the source Persona
does not remove or modify this independent Buddy. Imported packs remain separate
from the application distribution. A user can preview states before selection.

## Management modal

Use native Textual controls and the application's existing modal/theme conventions.
The scrollable body has Buddy, Follow, Persona, and Notifications & voice sections.
A fixed footer provides Apply and Cancel. Nothing mutates on Cancel. The modal
supports keyboard selection, visible labels, errors beside the affected control,
small terminals, and returning focus to its opener.

Buddy controls enable/disable, select/import artwork, preview expressions, choose
Dynamic or Static animation and size. Static and global reduce-motion suppress
animation without hiding an expression. Follow names an explicit conversation or
workspace; switching Console selection never retargets it. Persona controls manage
the selected conversation's assistant assignment, including None, through the
existing Persona service. Persona changes apply to future turns and never mutate
a running request. Workspace default Persona is a separate, clearly labelled
setting for future new conversations.

## Conversation interaction

Clicking a conversation Buddy opens a quick interaction modal for its exact bound
conversation, displaying the name, transcript, current activity, questions and
approvals. It permits typed replies and supported voice chat without navigating
the underlying destination. Replies carry the explicit conversation ID; ambient
Console selection is never an implicit target. Preserve drafts per target and on
close, offer Open in Console, and restore the underlying screen and focus. Deleted
or inaccessible targets are unavailable, never replaced by a nearby conversation.
Existing tool and approval checks apply unchanged.

## Workspace interaction

A workspace Buddy displays an activity inbox divided into Needs you, Running, and
Results. Track running conversations, unresolved decisions/questions, and completed
conversations with unseen results; do not fill the inbox with all historical chats.
Selecting a row opens that conversation's transcript and typed reply inside the
modal. No workspace-mode voice input is offered, including while viewing a row.
Opening the inbox does not acknowledge every result or answer a question.

Optional spoken output uses one application-owned serial queue. Prefix each spoken
response with its conversation name. Questions take priority; consolidate repeated
progress. Provide pause, skip, and mute. Speech does not mark a decision answered.
Notifications do not steal focus. Muting/hiding changes presentation, not execution.
Use existing TTS configuration and approval/voice authority; no new provider choice
or dependency is introduced. Background mode never leaves a microphone recording.

## Workspace Persona defaults

Reuse WorkspaceAssistantDefaults (ADR-079) and existing Settings controls. Every
new conversation creation surface resolves the chosen workspace's default once.
An explicit Persona or explicit None overrides inheritance. Existing, copied, and
moved conversations keep their assignments. Changing or clearing the default affects
future new conversations only. An unavailable default is visibly reported rather
than replaced with another Persona. Buddy selection remains independent.

## Navigation and runtime

Latest dev c0a42150 reuses and suspends the Console route (TASK-31520). Normal tab
navigation does not call the cancelling teardown path. Verify this with real mounted
navigation and active work before changing lifecycle code. Reuse the existing
ConsoleRuntime/controller/store as execution owners; do not add another job system.
Buddy UI is a projection and command surface, not a new execution authority.

Runs retain their accepted conversation, workspace, model, instructions, attachments,
and tool authority. Queues and parked approvals survive navigation under the current
approval policy. Explicit Stop, session deletion/close, and actual app shutdown keep
their intended cancellation behavior. A modal must neither reset the queue nor alter
a running turn's admission context. New async work must not depend on a modal's DOM.

## Boundaries and delivery

No restart-resumption guarantee is added. Ephemeral conversations are not silently
made durable. Unsupported remote operations display a reason instead of falling back
to local identity. The server port follows completed Chatbook validation, reusing
portable pack formats and behavioral contracts while retaining server-owned auth,
conversation identity, run lifetime and storage. v2 multiple Buddies will reuse explicit
bindings and the shared speech queue without changing v1 single-Buddy behavior.

## Verification

Targeted tests exercise real navigation with streaming and parked approvals, shutdown,
independent visual ownership and legacy migration, modal Apply/Cancel and focus,
workspace default precedence at creation seams, exact-target replies while another
conversation is selected, inbox acknowledgement, speech ordering and voice restrictions.
Use isolated test databases/profiles and native Textual pilots at normal and compact
sizes. Do not run the full repository suite without explicit user authorization.

## Architecture

[ADR-139](../../../backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md)
records independent ownership and scope. Existing workspace defaults follow
[ADR-079](../../../backlog/decisions/079-workspace-assistant-defaults.md).
