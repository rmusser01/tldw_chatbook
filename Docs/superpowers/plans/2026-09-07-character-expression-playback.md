# Character expression playback implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Execute inline in the existing isolated worktree.

**Goal:** Let Console characters animate available expressions, with a saved Dynamic/Static preference and reliable static fallback.

**Architecture:** The existing character controller continues selecting immutable Visual Identity assets. A disposable avatar widget decodes and presents animation without remounting the rail per frame. Settings use the canonical staged Appearance model and its existing live-refresh signal.

**Tech Stack:** Python 3.11+, existing Pillow, Textual 8.x and Rich mosaic renderer; no new dependencies or storage schema.

**Spec:** [Reviewed playback design](../specs/2026-09-07-character-expression-playback-design.md)

**Backlog:** TASK-32023

ADR required: yes
ADR path: [ADR-144](../../../backlog/decisions/144-character-expression-playback.md)
Reason: persistent motion preference and disposable avatar lifecycle.

## Global constraints

- `appearance.character_expression_mode`: `dynamic` or `static`, default `dynamic`.
- Global animation-off and Reduce motion override Dynamic without changing the saved choice.
- Automatic reactions disabled keep neutral/static unless an explicit manual reaction is selected.
- Static uses encoded frame zero and never rewrites asset bytes.
- Account at most 64 MiB of prepared RGBA buffers across active and in-flight work; one preparation at a time. Preflight codec canvases before loading.
- Maximum 30 paints/second; elapsed-time coalescing, finite loops stop on the final frame, hidden time is excluded.
- Off-thread decode/render, no database calls or rail remount on frame ticks, stale-result rejection on identity/geometry/motion changes.
- Preserve native image limits and optional Pillow behavior. Use targeted tests and disposable profiles only.

## Files and interfaces

- Create `Chat/character_expression_playback.py`: config policy, bounded image preparation and pure timeline selection. `expression_motion_enabled(config: Mapping[str, Any], *, react: bool, manual: bool) -> bool`; `prepare_expression(data: bytes, size: tuple[int, int], *, animate: bool) -> PreparedExpression`; `PreparedExpression.frame_at(elapsed_ms: float) -> tuple[int, bool]`; `close() -> None` releases owned buffers.
- Create `Widgets/Console/character_expression_avatar.py`: mounted owner of preparation, current renderable, visibility clock and one timer. `CharacterExpressionAvatar(data: bytes, *, box: tuple[int, int], animate: bool, is_current: Callable[[], bool], monochrome: bool, mode: str, id: str)`.
- Modify `UI/Console_Modules/character.py`: carry validated animated bytes/dimensions, include motion/renderer in request identity, preserve manual precedence, avoid caching decoded animated frames in the transcript cache.
- Modify `UI/Console_Modules/left_rail.py`: fail soft when a background geometry remount raises.
- Modify `UI/Screens/chat_screen.py`: build animated avatar for animated assets, use existing cell fitting and static fallback, expose current frame to the portrait viewer.
- Modify `UI/Screens/settings_appearance_defaults.py`, `settings_screen.py`, and `config.py`: default/validation/persistence, selector/help, live refresh.
- Test new pure module in `Tests/Chat/test_character_expression_playback.py`; mounted widget in `Tests/UI/test_character_expression_avatar.py`; extend existing settings and character-controller suites.

## Task 1: Settings and motion policy

- [x] Add failing model round-trip/invalid-value tests and a policy matrix, including manual reactions with automatic reactions off.

```python
values = load_appearance_defaults({"appearance": {"character_expression_mode": "static"}})
assert values.character_expression_mode == "static"
sections = build_appearance_save_sections({}, values)
assert load_appearance_defaults(sections).character_expression_mode == "static"
assert expression_motion_enabled({}, react=False, manual=True)
assert not expression_motion_enabled({"appearance": {"reduce_motion": True}}, react=True, manual=True)
```

- [x] Run the new tests and observe missing-field/module failures.
- [x] Add the dataclass field, normalized loading and strict save validation; add F9 selector with Dynamic/Static help. Stage changes using `_stage_appearance_value`, refresh suppression help using draft values, and use existing save/revert methods.

```python
self._stage_appearance_value("character_expression_mode", str(event.value))
self._mark_appearance_settings_staged()
```

- [x] Run `pytest Tests/UI/test_settings_appearance_defaults.py Tests/Chat/test_character_expression_playback.py -q` and the mounted Appearance save/revert tests.
- [x] Review and commit only the settings/policy change and its evidence.

## Task 2: Bounded preparation and encoded timeline

- [x] Add real GIF/WebP/PNG fixtures with unequal delays, finite/infinite loops, transparency/disposal, corrupt timing and budget rejection. Assert actual decoded pixels and elapsed-time indices.

```python
prepared = prepare_expression(two_frame_gif, (32, 32), animate=True)
assert prepared.frames[0].getpixel((0, 0)) != prepared.frames[1].getpixel((0, 0))
assert prepared.frame_at(0) == (0, False)
assert prepared.frame_at(100) == (1, False)
prepared.close()
```

- [x] Run those tests red. Implement sequential composited decoding under one preparation lock, with a shared reservation counter for retained buffers plus temporary canvases; release reservations on exceptions and close. Header preflight enforces native byte/dimension/frame/decoded-pixel limits before loading.
- [x] Normalize GIF repeat counts and WebP/APNG play counts; reject nonpositive durations for animation and fall back to validated frame zero. Skip APNG's separate default image for animation while retaining it for Static. Single-frame images are valid static assets.
- [x] Implement timeline selection using cumulative positive durations and elapsed time; preserve elapsed time while paused and stop finite playback on the final frame. Never allocate all full-size frames or retain all Rich renderables.
- [x] Run `pytest Tests/Chat/test_character_expression_playback.py -q`, including reservation release and simultaneous resize/preparation tests; record a subprocess codec memory probe separately from the buffer accounting assertion.
- [x] Review and commit the decoder and its tests.

## Task 3: Mounted avatar and Console integration

- [x] Add mounted tests that compare rendered frames, prove Static frame zero, hidden-time pause, unchanged request continuity, stale worker rejection, and unmount cleanup. Add controller tests for a same-asset settings change and manual reaction precedence.

```python
async with app.run_test() as pilot:
    await pilot.pause(0.05)
    first = avatar.render()
    await pilot.pause(0.15)
    assert avatar.render() != first
```

- [x] Run tests red. Implement the widget with one asynchronous preparation/render job, generation checks after every await, and content-only `Static.update(..., layout=False)` for pixel frames. Graphics updates use the existing image widget property. Stop timers on unmount; hidden widgets retain their current frame and reset their clock on resume.
- [x] Wire the controller request to effective motion and renderer mode; only immutable bytes/dimensions enter the spec cache. Build a new widget on expression/mode/geometry change, never on a frame change. A failed/over-budget preparation shows a concise static fallback reason in its tooltip.
- [x] Run targeted suites for the new avatar plus existing Console character avatar/controller, off-loop geometry, copy budget and Settings Appearance flows. Capture two mounted Console frame screenshots/SVGs demonstrating different pixels and a frozen Static result.
- [x] Run formatter/linter on changed code, self-review ownership and cancellation paths, record exact checks in TASK-32023, and commit. Mark Done only after all acceptance criteria have evidence. Conversion and Petdex are separate tasks and receive no runtime changes here.


## Execution notes

Implemented inline on `codex/buddy-import-design`. The three logical steps are
committed together so the selector, decoder and Console path are reviewed as one
usable feature. No dependency, database schema, portable pack or server contract
changed. The widget accepts optional validated neutral bytes for safe fallback.

Source inspection during validation showed that `is_on_screen` alone includes
widgets covered by the Console setup overlay. Playback now checks the topmost
widget at the avatar center, and evidence enters a conversation with a synthetic
user message before capture. The existing background geometry worker also needed
a fail-soft mount boundary; the injected mount-failure test exposed that omission.

See [verification](../reviews/2026-09-07-character-expression-playback-verification.md)
for exact evidence and baseline limitations. No full-suite claim is made.
