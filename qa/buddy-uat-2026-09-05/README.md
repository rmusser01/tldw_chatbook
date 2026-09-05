# Migu Buddy UAT fixes on current dev

PR base: `68f9d865fad623db6ec02e19632090c1140b3c89`.
Work branch: `codex/migu-buddy-uat-fixes`. Task: TASK-31585.

## Scope

Current dev already contains the Buddy implementation and realtime/fleet
extractions (PRs #1927, #1939, #1970 and #2094). This patch carries only the
remaining UAT defects forward; it preserves the newer Buddy widget, controller
boundaries, durable turn admission and diagnostic inventory format.

- Apply final native mouse-release coordinates for move and resize. Existing
  coordinate-based hit testing already handles XTerm events with no widget.
- Permit profile-owned Persona Visual imports, drafts and active-version edits.
  Pin directory device/inode while retaining file metadata, digest, descriptor,
  containment and symlink checks. Publication may change ancestor metadata.
- Route readback through trusted Manual Speak snapshots and playback ownership,
  so terminal playback clears the speaking presentation.
- Terminalize cancelled/unavailable project-instruction recovery. Refused
  transient echoes are excluded from provider history; already durably accepted
  turns retain recovery ownership under current dev's existing contract.
- Restore refused drafts after the acceptance callback consumed them, only when
  the composer still owns the same visible session. Preserve newer edits.
- Prune nested Python environments from diagnostic scanning, preserving
  serialized POSIX ordering and application modules merely named `venv`.
- Remove exception capture from lazy LLM mount and Library lifecycle failures.
  Reviewed inventory changes are exactly two diagnostic digests: 14 LLM-window
  calls and 104 Library-screen calls remain. No owners or sinks were added.

ADR required: no new ADR. Existing ADR-074 (Persona Visual/Buddy), ADR-037
(trusted speech), ADR-069 (project instructions), and ADR-029 (private diagnostic
boundary) govern these repairs. No new schema, dependency or provider transport.

## Verification

Tests import the actual PR tree using `PYTHONPATH` and `python -m pytest`.
The existing development Python environment supplies dependencies only.
No full repository suite was run.

- Native Buddy widget + publication + importer: **171 passed**.
- Independent review: **15 focused regressions passed**, covering readback,
  setup recovery, draft ownership and nested environments.
- Final focused setup, readback, draft, voice-guard and diagnostic regressions:
  **23 passed**; the final diagnostic registry assertion also passed separately.
- Broader Console/speech selection: **140 passed, 6 failed**. All six failures
  reproduce on the pristine base with the same environment.
- Broader diagnostic selection: **66 passed, 2 failed, 1 skipped**. Both failures
  reproduce on the pristine base; the skip needs unavailable pinned Git objects.
- Diagnostic inventory: 573 owners, 1,338 TASK-492 calls, 7,599 TASK-494
  calls, 10 sink files. Statement-level review confirms exactly two changed
  constant-message signatures; no call counts changed. Final rebuild reports
  no drift.
- New files pass Ruff; changed ranges were checked for formatting. Fatal-rule
  checks retain a pre-existing `F821` (`Iterable`) in `library_screen.py:36855`,
  reproduced on the pristine base. Existing broad lint debt is not represented
  as clean.

Born-red tests demonstrated final release position loss, profile-owned
publication rejection, stale readback ownership, recovery stuck in validating,
consumed draft loss, cross-session draft restoration, dependency inventory
contamination and exception capture before their respective fixes.
Independent review found and corrected decorator placement and visible-composer
ownership during the port.

### Base failures retained for follow-up

- `test_console_enter_snapshots_draft_before_late_keystrokes`
- `test_console_double_enter_sends_once_and_loses_nothing`
- `test_console_fresh_profile_first_send_resolves_real_session_not_sentinel`
- `test_console_enter_no_op_press_restores_draft_and_unblocks_next_send`
- `test_token_omission_notice_keeps_content_free_source_metadata`
- `test_project_instruction_disable_terminalizes_and_allows_retry`
- `test_reviewed_diagnostic_changes_are_metadata_only`
- `test_task_15743_exception_types_survive_loguru_forwarding`

The last two concern stale moved diagnostic labels and an existing activity
receipt metadata assertion. The draft PR does not waive these checks or claim
complete repository governance sign-off.

## Earlier live evidence and limits

`prior-live-evidence.json` contains minimized synthetic-fixture results from the
older working checkout. It is historical live evidence, not a claim that those
hardware runs were repeated on this PR tree. Real Terminal move/resize, a real
DeepSeek reply, a five-second microphone/local-transcription check, and local
Kokoro playback were exercised there. Kokoro drained 128,000 PCM bytes with
Migu idle → speaking → idle; microphone capture stopped after 5.02 seconds,
recognized the known phrase, released the device, and returned to idle.
Normal configuration remained unchanged. Microphone content was neither saved
nor sent to a provider. Temporary speech dependencies/model were not added to
the project or enabled in the user's normal profile.

Before final sign-off: resolve or classify the base failures and CI, then run
live OpenAI realtime UAT with a credential configured in the application's
settings. That provider was not configured during this UAT. Local speech success
does not validate the OpenAI realtime transport.
