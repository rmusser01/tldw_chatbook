---
id: TASK-39040
title: 'Windows audio fallback runs a database path through cmd.exe'
status: Done
assignee: []
created_date: '2026-11-05 18:05'
updated_date: '2026-11-05 18:05'
labels:
  - security
  - bug
dependencies: []
references:
  - tldw_chatbook/TTS/audio_player.py
  - tldw_chatbook/TTS/windows_audio_launch.py
  - tldw_chatbook/UI/Screens/artifacts_screen.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a Windows host without the legacy Windows Media Player, audio playback hands the clip's path to cmd.exe. Python quotes an argument for cmd.exe only when it contains a space, so a space-free file name containing `&` reaches cmd.exe as a command separator. The Artifacts screen plays a `briefings` audio path taken from the database, checked only for existence and a few patterns, so a tampered or corrupted row is a command injection. This exists on dev today (found by the TASK-38227 audit, which this blocks); this PR fixes it. Owner decision 2026-11-05, "Fix sink, startfile (Rec.)": play through `os.startfile` with no cmd.exe, and confine the Artifacts row to the briefing audio folder like the Library and Watchlists players. Accepted cost: pause and stop cannot reach a clip handed to the default app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On Windows without the legacy player, a clip opens through `os.startfile` with its path unchanged; no audio playback code builds a cmd.exe command, a path containing `&`, `|` or `^` never reaches a shell, and a file that is not audio is refused
- [x] #2 Artifacts-screen playback refuses an audio path that resolves outside the briefing audio folder, symlinks included, with a literal toast that never shows the path
- [x] #3 When the `os.startfile` fallback is used, the app says playback moved to the default audio app and never reports a pause or stop it did not perform
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both defects existed on dev `29592ecc9b`; this PR fixes them.

**Sink.** `SimpleAudioPlayer.play()`'s Windows branch now calls the new `TTS/windows_audio_launch.py`, imported inside that branch only (never resident on macOS or Linux; it also keeps the pre-import census at 426,508 of 426,512 LOC, down from dev's 426,512). With wmplayer.exe present the list argv is unchanged. Without it, `hand_off()` calls `os.startfile(path)` (ShellExecute, no command line is parsed) and refuses any extension outside the player's audio set (`_ALL_FILE_FORMATS`), because ShellExecute runs whatever an extension is registered to run: on dev, `payload.bat` would have run through `cmd /c start` too.

**Artifacts.** The Play handler re-checks the cached row with `audio_file_path_is_safe` (the helper the Watchlists player and the row's own listing in `daily_reports_view` use) before any other check. A path outside `briefing_audio_dir()`, or an in-folder link that resolves outside it, is refused with a literal `markup=False` warning toast that names no path. Note: on dev the Artifacts screen is not routed in the live app (ADR-172 aliases `artifacts` to Library), so this sink was reachable there only through code that mounts the screen directly; the cmd.exe sink itself was reachable from every live playback path.

**Ruling (stop/pause truthfulness on the fallback).** Smallest truthful option: say so, once per hand-off, and never claim control the app lacks.
- The player records a handed-off clip as current and PLAYING with `handed_off=True` and no process. It cannot see the clip end, and keeping it current means no surface frees the file while the default app is opening it (the Console deletes its cached clip, and the Speech playground deletes its PCM copy, as soon as playback ends).
- `pause()` and `resume()` return False (Windows has no pause branch; `_supports_pause=True` keeps pause off the stop-and-restart path, which would report a pause). `stop()` releases the slot but returns False, so the Speech playground no longer says "Playback stopped" (on dev it did: stopping the cmd.exe wrapper left the default app playing).
- `hand_off()` shows the literal toast "Playing in your default audio app. Pause or stop it there." (`markup=False`) in the running app, found through Textual's `active_app` context variable (private API, as `console_session_surface.py` already uses; with no active app it logs instead). `AsyncAudioPlayer.play` now runs the call in a copy of the caller's context so the toast also crosses its executor thread.
- Chosen over reporting FINISHED or IDLE at once: those would let the Console delete the clip before the default app opened it, the Speech playground would say "Playback complete", and IDLE would make the Console report "Failed".
- Residual, accepted with the cost: the in-app transport cannot tell when the default app finishes, so the Console header keeps "Playing" until the next clip, a stop, or its 300 s bound, and hands-free waits its text-length bound. Not verified on a Windows host.

**Other changes.** `stop()` resets state with `AudioPlayerInfo()` instead of eight field assignments (same defaults, and the new flag cannot outlive a stop). The Windows `_find_player` comment no longer claims COM pause support. User Guide: one Troubleshooting entry in `console/voice-and-hands-free.md`. Diagnostic inventory re-pinned after reading `--statements`: three new log lines in `windows_audio_launch.py` name no path or user text; `audio_player.py` has no changed statement (its two legacy debug rows gained a `self._current.state` path expression because `play()` now builds `_current` from `file_path`).

**Tests.** `Tests/TTS/test_audio_player.py` (+18 cases: startfile path unchanged and no cmd.exe argv; `&`, `|`, `^`, `%VAR%` names untouched; non-audio refused; wmplayer argv unchanged; the Ruling at the player; the toast from the loop thread, `asyncio.to_thread` and `AsyncAudioPlayer` in a real Textual app; the Speech playground's copy; a static guard that no TTS, Audio or Media_Playback list literal names cmd.exe) and `Tests/UI/test_artifacts_screen_reports.py` (+2: outside row refused, escaping symlink refused; the in-folder case was already pinned). Seventeen of the new cases and both Artifacts cases fail on dev; all pass here. Twelve named mutations each turn at least one new case red; restores were assert-once swaps from a backup, verified with `cmp`.

Files: `tldw_chatbook/TTS/audio_player.py`, `tldw_chatbook/TTS/windows_audio_launch.py` (new), `tldw_chatbook/UI/Screens/artifacts_screen.py`, `Tests/TTS/test_audio_player.py`, `Tests/UI/test_artifacts_screen_reports.py`, `Docs/User_Guide/console/voice-and-hands-free.md`, `Docs/security/production-diagnostic-inventory.json`.
<!-- SECTION:NOTES:END -->

## Renumbering provenance — 2026-11-05

Filed first as TASK-38970 after a sweep of every remote ref and worktree found 38950 as the maximum. Before this branch was pushed, `origin/fu/r8-filing` landed its own TASK-38970 (that branch mints 389xx ids up to 38994), so this task moved to TASK-39040, with every reference in its slice. It was never pushed under the old id.
