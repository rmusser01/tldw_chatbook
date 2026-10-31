# Auto transcription defaults and preview implementation plan

> Implement inline using the TDD workflow; targeted verification only.

**Goal:** Fresh installs use base for Auto, and users see the effective model before Start without changing existing settings.
**Architecture:** Change the fresh config template, reuse the dependency-free batch routing seam for model resolution, and append a plain-text preview to the existing commit summary. Capture the configured default with the existing ingest-options cache rather than reading config on each render.
**Tech stack:** Python, Textual, pytest; no new dependencies or styles.
**Spec:** User request in this task and TASK-32755.4 acceptance criteria.

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: Routine bug fix within the existing config and routing boundaries, with no migration or policy change for existing users.

## Review focus

- Existing distil/custom model settings must survive loading byte-for-byte.
- Auto ignores an inactive explicit model; explicit Faster Whisper honors its selected model.
- Local model paths must not appear in the summary or be interpreted as markup.
- Server and non-audio imports must not advertise a local model.
- Provider/model changes must refresh the mounted summary beside Start.

## One implementation task

- [x] Add failing real-config default/preservation tests and preview/routing parity cases to Tests/Library/test_transcription_readiness.py.
- [x] Add a mounted summary update regression to Tests/UI/test_library_ingest_readiness.py.
- [x] Change only the fresh transcription.default_model template to base; share model resolution across readiness, queue option construction and preview.
- [x] Capture the configured default in UI/Screens/library_screen.py's existing options payload and pass it to Library/library_ingest_state.py without extra config reads during render.
- [x] Run the two targeted files, related routing/state/cache checks, lint/format and UI governance. Record any unchanged baseline failures.
- [x] Update import documentation and TASK-32755.4 notes; self-review the diff and independently review the final change.

## Full-review follow-up (TASK-32755.5)

External config edits can be detected while Library remains mounted. Bind local
Auto readiness and the execution snapshot to the already displayed model, so
neither an external edit nor time spent queued silently changes it. Persist the
original preference snapshot separately; the form and saved Auto choice must
survive. No render-time config I/O or new ADR is needed (ADR-025 still applies).
The targeted regression traverses actual enqueue, preference persistence and
queue option resolution, including saved/custom defaults and server bypass.
