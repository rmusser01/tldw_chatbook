# TASK-37173 native Windows Terminal resize evidence

Environment: Windows Terminal Preview, native Python 3.12.10, Textual 8.2.8.
The user operated the window resize gestures because the desktop automation
inventory did not expose Terminal windows. The probe reads OS sizes independently
of Textual and uses the production LogsScreen, full app CSS, and TextualAppGuards
with synthetic records. It does not start Chatbook background services.

## Controlled side-by-side result

Both panes ran `native_probe.py` in the same Terminal window with separate private
test profiles. `--before` explicitly selects stock WindowsDriver; the fixed pane
uses the normal TextualAppGuards driver selection. Both render streams were TTYs,
and both processes reported a Windows Terminal session.

| Measurement | Stock driver | Fixed driver |
| --- | --- | --- |
| Native input mode | 512 / 0x200 | 520 / 0x208 |
| Initial OS and layout | 98x42 | 98x42 |
| Final OS viewport | 135x68 | 135x68 |
| Final app and Logs layout | **98x42 (stale)** | **135x68** |
| Footer region (x, y, width, height) | (0, 41, 98, 1) | (0, 67, 135, 1) |

The user confirmed: **“Yes, only the right pane resizes correctly.”**

Canonical native traces: `paired-before.jsonl`, `paired-after.jsonl`.
SVGs are Textual compositor exports from the native run, not desktop screenshots.
Intermediate drag snapshots can precede the final layout pass; the final fixed
snapshot at 65.625 seconds records all widgets settled at 135x68.

The earlier `after-logs.jsonl` trace also captured expansion to 252x56 and shrink
to 174x22 then 174x34, with the footer settling on each last row. Its initial 80x25
layout was a QA-launcher artifact: inherited `TERM=dumb`. The paired probe removes
that automation-only override before importing Rich; both initial sizes then match.
The original `before-minimal.jsonl` shows OS resize 120x36 -> 280x71 -> 120x36 with
no corresponding Textual resize events.

## Automated checks

- Born-red regression: `Resize events must be enabled`, actual input mode 512.
- Final targeted run: **16 passed, 1 skipped** (Unix-only case on Windows), recorded
  in `targeted-tests.txt`. Existing Pydantic/pytest-asyncio deprecation warnings remain.
- Ruff check and format checks passed for all changed Python files.
- Existing app wiring test initially failed `raw_source_selection_changed`, also
  reproduced with the new driver selector disabled. Its real-app import now uses
  the repository's `bootstrap_profile` marker, keeping the admitted config source.
- Independent review found no actionable issues; reviewer independently verified
  the installed suspend/close/resume lifecycle. Their separate test process was
  blocked by storage admission; the successful run above is the execution evidence.

## Reproduce

From the repository in Windows Terminal, run the probe with the repository venv:

```powershell
.venv/Scripts/python.exe qa/windows-terminal-resize-2026-10-28/native_probe.py --label fresh-after --profile C:/path/to/private/test-profile
```

For stock behavior add `--before` and use a different label/profile. The label must
be fresh because evidence files are opened exclusively. Resize, maximize, restore,
then Ctrl+Q to exit. The diagnostic also exits automatically after 15 minutes.
Use an owned private profile parent; do not weaken existing Temp ACLs.
