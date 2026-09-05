---
id: TASK-31551
title: >-
  Meeting transcription phase 1: live You/Others transcript with native system
  audio and Library handoff
status: In Progress
assignee: []
created_date: '2026-09-05 00:42'
updated_date: '2026-09-05 04:43'
labels:
  - audio
  - meetings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Record Zoom or in-person meetings from the TUI with a live labelled transcript, persist crash-safe audio, and hand the recording to Library ingest with diarization. Spec: Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Meetings screen records mic plus system audio on macOS and Linux and shows a live transcript labelled You/Others
- [x] #2 Stopping a meeting produces mixed.wav plus transcript.jsonl and meeting.json in the meetings folder and queues a Library audio ingest with diarization
- [x] #3 A meeting survives tab switches and app quit without losing recorded audio (headers patched, recovery offered on next visit)
- [x] #4 Console dictation and hands-free refuse to start while a meeting is active
- [x] #5 All new logic is covered by hardware-free tests and the suite is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-04-meeting-transcription.md tasks 1-12 in order
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Approach.** Tasks 0-12 implemented the feature end to end (tap probing,
capture/mixing, dictation-driven session/sinks, the app-owned
`meeting_session_owner`, the Console/hands-free refusal guard, the Meetings
screen, and the macOS Swift helper + packaging). Task 13 closed the program:
full suite, live TUI verification under tmux with a scratch profile, the user
guide, and this close-out. AC #1 (call-mode You/Others transcript with real
system audio) is **intentionally left unchecked** — this host has no macOS
System Audio Recording grant, and nobody may run `tccutil` or change system
permissions, so call mode could not be exercised. Everything else was
live-verified.

**Two real regressions were found and fixed during live verification** (both
were invisible to the full pre-existing test suite for well-understood
reasons, documented as new entries in `backlog/docs/lessons-testing-evidence.md`
and `backlog/docs/lessons-live-verification.md`):

1. **The Meetings screen rendered completely empty in the real app.**
   `#meetings-workbench` was never added to the `height: 1fr` ID-scoped CSS
   override that nine sibling screens (Artifacts, Personas, Watchlists,
   Workflows, MCP, ACP, Skills, Settings, Evals) all require for the shared
   `ds-panel destination-workbench` class combo — without it, `.ds-panel`'s
   `height: auto; min-height: 3` collapsed the whole workbench and both its
   panes to zero visible rows. Task 11's mounted pilot tests never caught
   this because their harness does not load the real CSS bundle. Fixed by
   adding `#meetings-workbench` to the existing list in
   `tldw_chatbook/css/components/_agentic_terminal.tcss` and rebuilding the
   generated bundle. Commit: `fix: Meetings workbench renders empty (missing
   height:1fr override)`.
2. **Recovering any crashed meeting crashed the whole app.**
   `recover_folder()` spread `meeting.json`'s payload (which always includes
   a `"folder"` key, from the real writer) into
   `update_meeting_json(folder, **payload)`, whose first parameter is also
   `folder` — a 100% reproducible
   `TypeError: update_meeting_json() got multiple values for argument
   'folder'` on every real recovery. The recover worker is
   `@work(thread=True)` with Textual's default `exit_on_error=True`, so this
   took the entire app down, not just the screen. The two existing
   `recover_folder` tests hand-write a `meeting.json` fixture that omits the
   colliding `"folder"` key, so neither ever exercised the real shape. Found
   by `kill -9`-ing the app mid-meeting and pressing Recover after relaunch;
   reproduced deterministically in isolation, fixed by dropping the
   redundant `"folder"` key from the spread, added a regression test with a
   realistic fixture. Commit: `fix: recover_folder() crashes the app on
   every real recovery`.

**Live verification (scratch profile `TLDW_CONFIG_PATH=/tmp/meetings-verify/
config.toml`, `users_name = "verify_meetings"`, tmux 200x50):**

- Navigated Home → Meetings via **F11** (command palette "Tab Navigation:
  Switch to Meetings" also works; **Ctrl+2**-style digit chords cannot be
  sent through tmux — a known, already-documented limitation, not new).
- Rail on mount: `System audio: Native (macOS tap)` /
  `Transcriber: auto (finalises per segment)` /
  `Speaker labels after the meeting: off (torch, torchaudio, speechbrain,
  sklearn missing)` / `Recording other people may require their consent.`
  Mic picker enumerated "System default" and "MacBook Pro Microphone";
  system-source picker offered "Native (auto)" plus the same device list.
- **Start** (SGR click on the real Button once rendered) → Start disabled,
  Pause/Stop enabled, timer advancing, mic/system level meters present.
  `System audio:` stayed **`Native (macOS tap)` for the entire session with
  no "System source lost" transition**, even though the helper almost
  certainly failed (no permission grant on this host) and
  `meeting.json`'s `system_source` field also recorded `"Native (macOS
  tap)"` post-hoc — spec §7 calls for a "System source lost" indicator here.
  Per the controller's explicit instruction this was **not fixed** in this
  task; flagged for the whole-branch review.
- Spoke test audio via macOS `say` (no live human speaker or physical
  audio-loopback available to this automated session) — **mic level stayed
  at 0% for the whole session** (no speaker→mic acoustic path in this
  environment, and/or no live microphone permission for this headless
  process), so **no transcript rows were produced and the You/Others
  labelling itself could not be observed even in room mode**. This is an
  honest gap distinct from the already-expected call-mode limitation: the
  Start/Stop/timer/level/footer/file/ingest-job machinery is fully verified,
  but no live speech content was ever captured in this session.
- **Stop** → footer: `Saved 0 segments, 00:02:51. 3 failed segment(s).
  Folder: /Users/…/verify_meetings/meetings/2026-09-04_2121. Library ingest
  queued: ingest-job-1.` (0 segments / 3 failed matches the silent-room
  input above, not an error state.)
- `ls` on that folder: `meeting.json mixed.wav others.wav transcript.jsonl
  you.wav` — exactly the required file set.
  `python -c "import wave; wave.open('mixed.wav').getnframes()"` → `2747840`
  (> 0; 16 kHz mono, 171.74 s, matching the session length).
- **Open in Library** → switched to Library's Import view with "This queue:
  1 queued" / "● queued · mixed.wav" visible. The durable
  `tldw_chatbook_library_ingest_jobs.db` row for `ingest-job-1` carries the
  correct `source_path`, `title`, `detected_type: "audio"`, and
  `ingest_options: {"diarization": true}` — the meeting → Library handoff
  contract (AC #2) is proven structurally. The job itself never reached
  `"done"`: it stayed `"queued"` while observed live, and by the end of the
  session (after several subsequent app restarts for the crash/recovery
  test below) its final DB state was `"failed"` with
  `error: "Interrupted by app restart"` — a sensible, honest failure
  classification from the Library ingest subsystem given how many times
  this session restarted the app out from under it, not a Meetings-side
  defect. It never progressed far enough to confirm the actual
  transcription pass, most likely because this venv's optional
  audio-transcription dependencies (`faster-whisper`, `audio_processing`,
  etc.) all report unavailable at startup; not investigated further as it
  is a Library-ingest-subsystem property, not part of the Meetings feature
  under test.
- **Quit mid-meeting** (`Ctrl+Q`) exited immediately with **no confirmation
  dialog** — contrary to the plan's expectation, but the app's own shutdown
  hook (`meeting_session_owner.shutdown()`) gracefully finalized the
  meeting anyway: `meeting.json` showed `"stop_reason": "shutdown"`,
  `"recovered": false`, correct WAV byte counts, no data loss. Confirms
  half of AC #3 (quit does not lose recorded audio) without exercising the
  crash/recovery half, so a genuine crash was also tested:
- **Crash + recovery** (`kill -9` the app process mid-meeting) left
  `meeting.json` with `"ended_at": null` and `mixed.wav`'s header reporting
  0 frames despite ~837 KB of real audio bytes on disk — the exact
  crash-safety scenario spec §7 describes. Relaunching showed
  `Unfinished meeting found: 2026-09-04_2135` with **Recover** enabled;
  pressing it (after the fix above) produced footer
  `Recovered 2026-09-04_2135: Library ingest queued: ingest-job-2.` and
  `meeting.json` updated to `"recovered": true, "stop_reason": "crash",
  "ended_at": "…"` with the WAV header correctly patched. AC #3 fully
  verified.
- **Console guard (AC #4) was NOT live-clicked.** The scratch profile has no
  provider configured (deliberately, to avoid provider setup work out of
  scope for this task), so Console's composer stays locked behind a "Set up
  provider" card and the `#console-dictation` mic button is unreachable.
  AC #4 is checked on the strength of the existing, already-green,
  hardware-free `Tests/UI/test_console_meeting_guard.py` (5/5 passing,
  reconfirmed in the Step 1 full-suite run below) plus a direct code read
  of `dictation.py`/`hands_free.py`'s `is_active` guard and their exact
  toast copy (`"Meeting in progress: stop it in Meetings before using
  Console dictation."` / "...before using hands-free.") — not a
  fabricated live click.

**Full suite (Step 1):** `.venv/bin/python -m pytest -q -p no:cacheprovider
--timeout=600 Tests --ignore=Tests/Packaging/test_profile_core_packaging.py
--ignore=Tests/TTS/test_chatterbox_validation.py
--ignore=Tests/Web_Scraping/Confluence` (the five ignored modules fail to
COLLECT in this venv — missing `playwright`, `setuptools`, `torch`).
See task-13-report.md for the full failure-triage table; all failures
outside the known pre-existing lists were reproduced on a throwaway detached
`origin/dev` worktree before being called pre-existing. No failure was found
in a file this branch touched other than the two fixed above.

**Deviations from the plan/spec, all deliberate:**
- Swift helper source lives at `tldw_chatbook/Audio/audiotap/main.swift`
  (inside the package), not `Packaging/macos/audiotap/main.swift` as an
  earlier spec draft said — the spec's §3.6 heading has been corrected to
  match.
- Privacy auto-clear is set directly on the built dictation service's dict,
  not via `update_privacy_settings()`, because that method persists to
  `config.toml` as a side effect the meeting session must not trigger.
- The Meetings hotkey is **F11** (not documented in the original plan),
  chosen because `SHELL_DESTINATION_SHORTCUTS` requires every destination
  to have one.
- Test-only symbol names were corrected to the real production names
  instead of adding aliases (`COMPREHENSIVE_CONFIG_RAW`,
  `TAB_HELP_TEXT`/`NAVIGATION_TABS`) — see Task 11's report for detail.
- The macOS helper's runtime behavior (spawns, exits 2 for permission,
  restarts once) is unverified end-to-end on this host; see the
  `TLDW_RUN_AUDIOTAP_HELPER_TEST=1` follow-up instructions in
  `Docs/User_Guide/meetings.md`.

**Files changed in task 13:** `tldw_chatbook/css/components/_agentic_terminal.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt bundle),
`tldw_chatbook/Audio/meeting_owner.py`, `Tests/Audio/test_meeting_owner.py`,
`Docs/User_Guide/meetings.md` (new), `Docs/User_Guide/index.md`,
`Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md`,
`backlog/docs/lessons-testing-evidence.md`,
`backlog/docs/lessons-live-verification.md`, this task file, and four new
follow-up task files (31586-31589).

**Status stays "In Progress"**, not Done, because AC #1 (the You/Others
call-mode transcript) remains unchecked and unverifiable on this host without
a system permission grant nobody is authorized to make in this task.
