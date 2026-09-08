# Buddies in Console and across screens

Open **Console → Menu → Buddy** to manage Buddy artwork, its follow target and
the target's Persona. The floating Buddy's **⚙** control opens the same form;
focus the Buddy and press **m** for keyboard access.

Choose installed artwork or enter a native `.tldw-persona-vpack` / `.zip` path.
Preview an expression, choose **Dynamic** or **Static**, set its size, and Apply.
Global Reduce motion also suppresses animation. Cancel discards form changes.
Pixel Migu is available in a fresh profile. Imported artwork and its creator,
licence and notices remain in the local Buddy library. Choosing artwork does not
create or assign a Persona. Existing Persona-backed Buddy selections are copied
to an independent owner; editing the source Persona does not change that copy.

**Follow** chooses one specific conversation or workspace. Selecting a different
Console tab or navigating to Watchlists does not change that target. A normal
conversation's first save preserves its Buddy link for later launches. Temporary
conversation links last only for the current app run. A missing, deleted or
inaccessible target is shown as unavailable and is never replaced automatically.
This version displays one Buddy; multiple simultaneous Buddies are planned later.

Click the Buddy, or focus it and press **Enter**, to interact. Drag its body to
move it; the lower-right grip resizes it. Closing or hiding it does not stop a run.

## Conversation Buddy

The interaction window shows the pinned conversation's transcript, current work,
questions and supported approval cards. Type a reply and Send without changing
the underlying screen. Each conversation keeps its Buddy draft when the window
closes; its Console composer draft remains separate. Existing permissions still
apply. If Console has staged inputs the Buddy cannot display, review those in
Console before sending through the Buddy.

**Dictate** uses existing Console speech recognition. Stop recording, review the
transcribed draft, then Send. Closing the interaction window discards an active
recording. This is push-to-talk dictation; it does not enable background listening.
**Open Console** is the explicit way to switch to the full conversation interface,
including review operations that require it.

See [conversation interaction details](console/buddy-conversation.md) for quick
reply restrictions, decision review and dictation setup.

## Workspace Buddy

The workspace inbox separates **Needs you**, **Running**, and unread **Results**.
It tracks active work and unseen outcomes rather than listing every historical
conversation. Open a row to review or send a directed text reply. Workspace mode
does not offer microphone input, including inside a conversation opened from it.

Opening or refreshing the inbox does not clear results. **Mark seen** applies to
the selected result only. Questions still require an answer, and approvals still
require their existing confirmation. Updates never take keyboard focus.

## Personas and speech

The management form can keep the current Persona, choose another, or choose
**None** for the followed conversation. Changes affect future turns and are
unavailable while a run, queue or decision is active. A Persona change is saved
atomically with its prompt and settings. Existing memory-write permission requires
review before assigning a different Persona.

For a workspace, the Persona control is labelled as a default for **future new
conversations**. The same default is available in workspace creation/details.
Explicit Persona/None choices override inheritance; existing, moved and copied
conversations keep their assignment. Clearing a workspace default keeps it cleared.

Optional **Speak responses with conversation names** uses the existing TTS
configuration. Each spoken update starts with its conversation name, and one queue
prevents overlapping Buddy responses. Questions take priority over waiting results.
Use **Pause / Resume**, **Skip**, or **Mute** in the interaction window; Resume
restarts the interrupted update. If the configured destination needs consent,
**Confirm speech** opens the existing destination confirmation. It never opens
automatically or steals focus. Speech waits while Buddy dictation records.

Speech does not acknowledge a result or answer a question. Hiding/disabling the
Buddy stops its spoken presentation without cancelling conversation work. Closing
only its interaction window leaves configured speech available. Saved results
without a live conversation session remain available to review; their bodies are
not automatically loaded solely for speech.

Normal navigation keeps accepted Console runs, queued follow-ups and pending
decisions alive. A finite decision timer advances only while the decision is
actually answerable. Explicit Stop, closing a session, and app shutdown retain
their cancellation behavior. This feature does not resume unfinished provider
runs after an app restart. Chatbook currently supports local Buddy interaction;
the server UX will be ported separately.
