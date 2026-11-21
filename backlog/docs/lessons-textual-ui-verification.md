# Lessons: Textual UI verification

Read alongside [the general testing evidence lessons](lessons-testing-evidence.md).

## Scratch UI diagnostics need the existing fixture plugin and an actual input-delivery witness

**Post-PR3207 book audit, 2026-10-22.** Two scratch audiobook probes outside
`Tests/` failed private-profile admission with `unsafe_parent: missing_parent`.
The decorator alone did not load the bootstrap fixtures. A scratch-local
`conftest.py` with `pytest_plugins = ["Tests.conftest"]` admitted the existing
native private profile without weakening any storage or ACL check. Those first
runs are setup failures, not product evidence.

The same audit then treated `Pilot.press()` task completion as key delivery
while a parent widget callback was deliberately held. Both controls failed that
oracle, although their child TextAreas already contained the typed `x`.
Pilot was still waiting for pump settlement. The qualified probe instead
witnessed the actual live child text before releasing the callback: the normal
import control passed and the later-edit retention assertion failed, with real
read/Generate content recorded. When holding a pump, prove the specific child
input was delivered rather than requiring the whole Pilot operation to settle.

## Replacing selected Input text in queued-message probes

**TASK-36361, 2026-10-22.** Three queued-input Settings probes used
`Input.insert_text_at_cursor` as if it replaced selected seed text. Their setup
assertions failed because the old text remained. Calling the actual control's
`Input.replace(later, 0, len(field.value))` supplied the intended replacement
while its `Changed` message remained queued. The three corrected baseline
cases then failed at the intended raw-draft retention assertions with unchanged
production source. The first three setup failures were excluded and both runs
were retained separately. Check the resulting live control text before treating
a failed race probe as product evidence.

## A flow that passes on the factory app can stall on the real app

**TASK-38970, 2026-11-10.** The User Guide driver re-ran the mounted Watchlists
UAT's flow on a real `TldwCli` instead of `Tests/UI/app_factory._build_test_app`.
It stalled three ways before completing. The real context meter called
`cached_context_window` on the scripted gateway, which the factory app never
did. Plain chat's light tool surface offered no Watchlists tools, so
`load_tools` failed and an approved call came back "Tool not permitted"; the
fix was `[tools] full_agent_tools_enabled = true`. An approval loop that marked
a round done when it pressed **Allow once** stalled because the first press
landed while the button was still disabled, and `Button.press()` on a disabled
button does nothing. Before reusing a factory-app flow on the real app, list
what the factory configures, and press a control only once `disabled` is
False.

## At 80x24 the voice chip's real budget is 28 cells, not WARMUP_MESSAGE_MAX_CELLS (TASK-33621.21, 2026-11-21)

**Incident.** The still-preparing note was first written as "Still preparing… keep
waiting or Cancel" ("◐ " + 39 = 41 cells). The chip itself painted whole at
80x24, and only a compositor-cell check of the mic button caught that Cancel was
gone: `_painted_cells` returned `''` for it. A throwaway probe of the real
ChatScreen measured why. While preparing, the chip is exempt from the width cap
and the draft is already at 0 cells, so every extra chip cell pushes the action
row off the right edge: a 28-cell chip ("◐ Still preparing… or Cancel") keeps
all 11 cells of Cancel, 29 clips its padding, 31 clips the label, 41 removes it.
`WARMUP_MESSAGE_MAX_CELLS = 40` (written for an older 42-cell chip) would allow
copy that hides Cancel, and the busy copy's full-width presentation offers 41
cells at 80 columns. At 100 columns the 41-cell copy fits.

**What to do.** For any preparing-chip copy, assert the mic button's painted
cells at 80x24 (`_painted_cells(screen, mic)`), not just the chip's own render
line. Keep chip copy within 28 cells including the glyph. "◐ Downloading speech
model… 1.2 GB" (36 cells) is already over it; only a test-only download probe
reaches it today, so check it again when TASK-33621.110 adds real progress.
