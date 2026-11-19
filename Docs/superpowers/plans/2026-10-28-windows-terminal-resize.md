# Windows Terminal resize implementation plan

**Goal:** Deliver native resize events to every Chatbook screen after a Windows Terminal resize/maximize.

**Scope:** TASK-37173. User requested implementation and a native before/after test. Keep Textual 8.x and existing screen CSS; use the existing `TextualAppGuards` compatibility boundary. Work in the current checkout, preserving unrelated untracked files.

ADR required: no
ADR path: N/A
Reason: Routine upstream compatibility bug fix; no new runtime/dependency contract. Preserve ADR-097 boot budgets.

- [x] Capture unmodified Windows Terminal behavior with independent OS-size and Textual-event observations. Windows changed 120x36 -> 280x71 -> 120x36; no corresponding Textual resize events.
- [x] Run `Tests/App/test_windows_terminal_resize.py` red: selected native driver fails to enable resize reporting.
- [x] Add a native Windows driver adapter through `TextualAppGuards.get_driver_class`; after each upstream start, retain current input flags and enable `ENABLE_WINDOW_INPUT`. Keep upstream mode restoration and explicit custom drivers.
- [x] Run targeted driver/guard tests and Ruff. Check startup, restart, cleanup, custom constructor/environment drivers, and non-Windows selection.
- [x] Run the production Logs screen in Windows Terminal with the selected driver; record OS/app/screen/content/footer dimensions across maximize, restore, and maximize again.
- [x] Review the diff, update architecture/task notes with evidence and remaining limits, and mark Done only after native verification.

**Native verification:** The desktop automation inventory does not expose Windows Terminal windows. The user performs resize gestures; the diagnostic records measurements directly from the native process. This is native event delivery evidence, not a Pilot resize substitute.

**Execution notes:** Native paired test confirmed stock 98x42 versus fixed 135x68 for the same 135x68 viewport. The initial QA TERM=dumb override was removed from the launcher. The existing guard wiring test needed bootstrap_profile after its failure was reproduced with the selector disabled. Final targeted tests: 16 passed, 1 Unix-only skip. Independent review: no actionable findings. See qa/windows-terminal-resize-2026-10-28/README.md.
