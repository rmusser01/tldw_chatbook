# Buddy named speech implementation plan

Task: TASK-32084 (speech AC1–3; root owns integrated journeys and documentation).
Spec: `Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md`.

ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: app-owned speech presentation reuses existing TTS destination consent,
request snapshots and exact playback ownership without changing agent execution.

1. Add a bounded serial queue in `Persona_Buddy/speech.py`. Immutable named items
   carry exact validity probes; pending questions precede responses. Never enqueue
   streaming increments. Repeated receipt/message/question keys coalesce. Pause
   interrupts the current utterance and resumes it from its beginning; skip drops
   only the current item; mute clears speech while leaving all run state alone.
2. Extend existing store-issued TTS snapshots with an optional explicit owner ID;
   omitted owner retains the existing active-session requirement. Reuse every
   content, variant, authorship and durable-version guard. Test stale owner and
   default-visible behavior separately.
3. Add a guarded named-utterance entry to the existing TTS handler. Use configured
   character/default/global resolution, exact consented destination admission,
   and request-owned playback lifecycle. Validate after asynchronous resolution
   and at provider admission; wait for playback completion and stop only this owner.
4. `UI/Navigation/buddy_speech.py` owns enabled-only periodic projection across
   navigation. Capture live bound completed messages and current questions, use
   workspace receipt IDs without acknowledging them, reject profile/rebind/content
   changes and never enable a microphone. Unknown/unloaded result bodies remain
   in the inbox for explicit review rather than silently loading another session.
5. Reuse the sanitized TTS consent modal from an explicit speech-controls action.
   Destination consent is scoped to the current binding/profile and never inferred
   from enabling a checkbox. No background notification opens a modal.
6. Add a small reusable controls widget; root inserts it and awaits coordinator
   `aclose()` during actual app shutdown. Tests use deterministic fake playback,
   real store snapshots and isolated native modal controls; no live paid TTS/full suite.
