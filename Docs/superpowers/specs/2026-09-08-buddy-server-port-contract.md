# Buddy interaction contract for the later tldw_server port

Status: Chatbook behavior contract; server implementation is not part of this change.
Source: [approved design](2026-09-08-console-buddy-management-design.md) and
[ADR-139](../../../backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md).

The server port should offer the same user choices: independently owned Buddy
artwork; explicit conversation/workspace follow; one visible Buddy in v1; optional
Persona assignment and future-conversation workspace defaults; Dynamic/Static;
in-place conversation and workspace interaction; opt-in named serial speech.

## Authority and storage

- Use server-owned authenticated principal, workspace and conversation identities.
  Never accept Chatbook's live session UUID or a browser's currently selected tab
  as authority. Recheck access and membership before reading, replying, answering
  decisions, applying Persona changes or acknowledging results.
- Store Buddy profiles separately from Persona prompts. Reuse validated native
  pack manifests, immutable asset hashes and preserved artwork attribution; do not
  import profile paths, local tool permission grants or local receipt identifiers.
- Store workspace default Persona with an explicit None/optout representation.
  Resolve once at creation. Explicit request assignment overrides inheritance;
  restore, move and copy preserve identity.
- Update a conversation's Persona, prompt and settings atomically with an expected
  version. Reject busy or stale owners. Fence subsequent run and speech snapshots.

## Runtime and UI

- Server jobs own accepted work independently of route, component, websocket or
  browser-tab lifetime. Closing an interaction surface never invokes Stop. Preserve
  the server's explicit cancellation and restart policies; do not infer resumability.
- Reuse existing approval/question execution authorities and exact round IDs.
  Viewing or speaking a prompt is never approval. Timers use answerable time where
  the server supports finite interactive deadlines.
- Workspace inbox entries derive from live activity and existing unseen outcomes.
  Acknowledge exact selected result identities, not a whole workspace or conversation.
  Load saved conversations explicitly without changing the underlying destination.
- Conversation dictation must stop on interaction close; workspace mode has no
  microphone input. Reuse the server's configured STT/TTS destination disclosures
  and permissions. One application playback queue names each conversation, prioritizes
  questions, and supports pause/resume, skip and mute without affecting jobs.
- Preserve drafts and keyboard focus; new activity cannot navigate or focus a page.
  Test ordinary and compact viewports, reduced motion and stale/deleted targets.

## Port acceptance evidence

Exercise two simultaneous conversations while viewing another destination: stream
and queue completion; question and tool approval; exact-target text reply; stale
membership/access denial; saved-result opening and individual acknowledgement;
Persona update rollback; workspace creation precedence; and named speech queue
ordering/cancellation. Include navigation/socket-disconnection tests against the
real server job owner, and distinguish deterministic gateway tests from real
provider/audio verification. Cross-user data isolation must be tested server-side.
