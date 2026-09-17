# Lessons: clipboard delivery

## A successful copy call is not evidence of a usable clipboard

**Incident (2026-09-17, TASK-32754):** A user on latest main reported an empty
clipboard after an install-command success toast. The Library called Textual's
OSC 52 route, which has no delivery acknowledgement. Existing UI tests only
recorded the method call. The feature dialog also used native pyperclip without
readback; some subprocess backends can silently fail.

Confirm native delivery by readback and test the mounted button separately from
the native boundary. Label OSC 52 delivery as unconfirmed and keep the literal
command selectable. Qt's in-process readback is insufficient for a short-lived
helper: its X11 clipboard needs an event loop and ownership disappears on exit
([Qt documentation](https://doc.qt.io/qt-6/qclipboard.html#notes-for-x11-users)).

**Review reproduction:** Timing out an await on a worker thread left a blocked
clipboard write alive; it could overwrite the next command after its timeout.
Own and terminate the native subprocess tree before returning a failed copy.
The regression releases a real descendant after timeout and verifies that it
cannot perform the stale write. Native desktop verification must preserve and
restore every existing clipboard format, not only its text representation.
