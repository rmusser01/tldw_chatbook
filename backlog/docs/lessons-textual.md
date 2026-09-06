# Lessons: Textual framework traps

Working knowledge about the Textual version this repo pins (8.x, currently 8.2.8) —
API gotchas that cost a debug cycle when first met. Not decisions (see
`backlog/decisions/`) — these are traps that have actually cost time here, kept so the
next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore gets ignored. If you add one, bring the incident.

---

## `run_worker` does not forward positional arguments to the callable

**TASK-16314 review round, 2026-08-14 (commit `14373a7ac`).** The trajectory screen's
live poller needed to run `self._live_rebuild_worker(revision)` in an exclusive worker
carrying the revision the rebuild was built for, so a slow older rebuild could be
identified and dropped when a newer revision had since been observed. The natural call
— `run_worker(self._live_rebuild_worker, revision, thread=True, exclusive=True)` —
cannot work: Textual 8.2.8's signature is
`run_worker(work, name='', group='default', description='', exit_on_error=True,
start=True, exclusive=False, thread=False)` with **no `*args` passthrough**. The extra
positional binds to `name` (verified by introspection against the installed 8.2.8),
and the worker invokes the callable with zero arguments — for a method with a required
parameter that surfaces as a confusing `TypeError` inside the worker; for a callable
with optional parameters it would run silently on defaults, which is worse. The pilot
caught it; the fix is a closure:

```python
self.run_worker(
    lambda: self._live_rebuild_worker(revision),
    thread=True,
    group="trajectory-live",
    exclusive=True,
)
```

**What to do.** When a worker callable needs arguments, close over them (a `lambda`
or `functools.partial`) — never pass them positionally after the callable to
`run_worker`; they are constructor parameters of the worker, not arguments of your
function, and nothing warns you. Same discipline for `timer`/`set_interval` callbacks
that need captured state. If the arguments are cheap to recompute inside the worker,
prefer passing a zero-argument method and re-reading state there — it also avoids
holding stale references across `exclusive=True` cancellations.

---

## A mutable `reactive([])` default is one shared object — and reassigning an empty-equal value does not un-share it

**TASK-15771, 2026-08-15.** `reactive([])` / `reactive({})` with a literal default
installs the *same* list/dict object on every instance of the widget class:
`Reactive._initialize_reactive` does `default_or_callable() if callable(...) else
default_or_callable`, and a literal is not callable. Any in-place mutation
(`.append`, `[k] =`, `.insert`, `.clear`, ...) then leaks across every instance and
every future instance, including screen remounts. The package-wide AST sweep found
**41** such declarations — in two rounds: the first sweep reported 27, and the task
review proved it structurally blind to the subscripted-generic spelling
`reactive[list[dict[str, Any]]]([])`, which parses as `Call(func=Subscript(...))`
and slipped past a `Name`/`Attribute`-only function-name match (14 more sites, all
in `UI/Watchlists_Modules/`; runtime-identical bug). An AST detector for a call
must unwrap `ast.Subscript`, or the generically-annotated spelling of the exact
same call is invisible to it. Born-red two-instance tests demonstrated the leak on
`CharacterVoiceWidget.characters`/`voice_assignments`,
`ChapterEditorWidget.chapters`, and `CollectionsTagWindow.selected_keywords`
(`Tests/Widgets/test_reactive_default_aliasing.py`).

The half of the trap that reverses classifications: **"reassigns before use" is not
a defense when the reassigned value is empty-equal.** `Reactive._set` in 8.2.8 only
runs `setattr(obj, self.internal_name, value)` inside
`if always or self._always_update or current_value != value:` — so
`self.chapters = chapters if chapters else []` in `ChapterEditorWidget.__init__`
compared `[] != []`, stored nothing, and the instance kept aliasing the class-shared
default it then `.insert()`ed into. The site read as safe in review; the born-red
test proved it was not. Only a mutation-sites trace plus this mechanism check
classifies correctly.

**What to do.** Always declare mutable reactive defaults as callables —
`reactive(list)` / `reactive(dict)` / `reactive(set)`, or
`reactive(lambda: [seed])` for non-empty defaults.
`Tests/Architecture/test_reactive_mutable_default_inventory.py` pins the package
at zero for the forms it detects — mutable literals, comprehensions, module-level
shared mutables, and `list()/dict()/set()` call results, in both the bare and the
subscripted-generic call spellings. It does NOT see shared mutable *instance*
defaults (`reactive(SomeClass())`; 5 known occurrences at review time) — a green
run is not clearance for that form. If the guard fails, fix the default — never a
"reassign before use" workaround.

---

## Square brackets in widget text are console markup — an unescaped `[Word]` is deleted from the UI without a warning

**TASK-19734, 2026-08-21.** The import wizard's mislabelled tag checkbox was
relabelled to name what it actually does — prefix imported item names with
`[Imported]` — written as the plain string
`'Prefix imported item names with "[Imported]"'`. Textual renders label and
`Static` text as console markup, so `[Imported]` parsed as an (unknown) style tag
and was dropped: the shipped control read `Prefix imported item names with ""`.
The behaviour is silent — no exception, no log line, no styling change — and it
only surfaced because a test asserted the substring was in the rendered label:

```python
>>> str(Checkbox('… with "[Imported]"').label)
'… with ""'
>>> str(Checkbox(r'… with "\[Imported]"').label)   # escaped
'… with "[Imported]"'
```

Escape with a raw string and a leading backslash (`r'\[Imported]'`) anywhere a
literal bracket must reach the user, and **assert on the rendered label**
(`str(widget.label)` / `str(static.renderable)`), never on the source constant —
a test that checks the constant passes while the UI shows nothing. This bites
hardest on exactly the copy that most needs to be literal: names, prefixes,
placeholders, file globs, `[Imported]`-style markers.

---

## `Static.update()` lays out the view by default — a one-row repaint is not free

**TASK-21117, 2026-08-23.** The Console Inspector's outer scroll hint is a
pinned one-row `Static` whose copy blanks at the bottom of the rail. Splitting
the pure-scroll path off the geometry reconcile removed every whole-rail
`refresh(layout=True)` from a wheel gesture (8 → 0 over 8 frames), but a probe
counting `Screen._refresh_layout` calls showed the gesture still cost 11 screen
layout passes where it should have cost 9. The residue was the copy repaint
itself: `Static.update(content)` takes `layout: bool = True` and calls
`self.refresh(layout=layout)`, so painting two characters into a slot whose
height is pinned by inline styles still scheduled a view layout — twice per
gesture, once entering and once leaving the bottom.

```python
def update(self, content: VisualType = "", *, layout: bool = True) -> None:
    ...
    self.refresh(layout=layout)          # layout=True unless you say otherwise
```

Pass `layout=False` when the slot's size cannot change (height pinned at
compose time, width container-driven), and keep a test that holds the pinned
geometry to account (`assert hint.region == hint_region` across the gesture) so
the assumption cannot rot into stale geometry. The same applies to the
`Static.content` setter, which is `refresh(layout=True)` with no opt-out — and
note that `content` is also the cheapest way to read back what was last painted,
which is how that repaint skips a no-op write without a shadow copy that every
other writer would have to remember to invalidate.

**Recurred twice more, so it is now a guarded census — TASK-21595, 2026-08-25.**
After TASK-21692 (the composer blink: 396 `Widget.arrange` per 6 ticks on an
*idle* composer) and TASK-21134 item 7 (media-viewer match-nav), a package-wide
AST census of every repeating-clock root found two more: `SplashScreen.
_update_animation`, repainting a full-viewport `Static` at the shipped cards'
0.01–0.1 s `animation_speed` — **10–100 whole-screen reflows per second during
startup** — and `PersonaBuddyWidget._paint_frame` at the pet's own frame rate.
Both measured 20 → 0 `_refresh_layout` / `reflow` / `arrange` per 20 ticks.
`Tests/Architecture/test_timer_path_static_update_inventory.py` now rebuilds
that census on every run and fails any clock-reachable `.update(` that neither
passes `layout=` nor carries a stated exemption. Two things that census taught:

1. **Grepping `set_interval` is not a census.** One of the two clocks is a
   `set_timer` callback that re-arms *itself* — an interval spelled as a chain
   of one-shots. Enumerate roots structurally (including the self-rearming
   one-shot), then walk the call graph; the repaint is usually four to six hops
   from the timer, in a different module.
2. **`layout=False` is only sound where the box cannot be content-sized, and the
   pin may not be in the stylesheet.** The Persona Buddy frame's real pin is
   `frame.styles.width/height = "100%"` assigned *inline* by `_apply_geometry`,
   which beats every sheet — so mutating its CSS to `auto` (even rebuilding the
   generated bundle) left the geometry test green, a surviving mutant that was a
   finding about the test. Prove the claim with an A/B against the layout engine
   instead of reading CSS: paint the content with `layout=True` and record the
   geometry, *scrub* with a deliberately different shape (without the scrub the
   second arm inherits the first arm's geometry and passes vacuously), then
   paint the same content with `layout=False` and compare — carrying sibling
   regions and painted per-row cell widths, not just `outer_size`.

---

## `set_timer(0.0)` never fires — silently

**TASK-21110, 2026-08-23.** The splash/initial-screen overlap is armed with
`self.set_timer(SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS, ...)`. A/B-ing that
delay, the "0.0 s" arm looked like the best of both worlds: the boot-time win with
none of the splash-animation stutter the 0.2 s arm showed. It was measuring nothing.
Textual 8's `Timer._run` computes `count = int((now - start) / _interval + 1)`, so a
zero interval raises `ZeroDivisionError` **inside the timer's own asyncio task**. Nobody
retrieves that task's exception, so there is no traceback in normal operation, no log
line, and no callback — the arm's `import_on_loop` was still 430 ms, i.e. the pre-import
had never happened at all. Reproduced standalone in nine lines:

```python
class A(App):
    def on_mount(self):
        self.set_timer(0.0, lambda: fired.append("zero"))   # never runs
        self.set_timer(0.2, lambda: fired.append("two"))    # runs
# -> fired == ["two"], plus an un-retrieved ZeroDivisionError task
```

**What to do.** Treat 0 as an invalid `set_timer` delay. If "as soon as possible" is
what you mean, use `call_after_refresh` (or `call_later`); if a constant feeds the
delay, branch on `> 0` rather than trusting whoever edits it next, and keep a test that
the zero case still schedules. And when a perf arm comes back looking free, check that
the thing you were measuring actually ran before you believe it.

---

## Monkeypatching an `@on`-decorated handler on the class does not patch it

**TASK-21110, 2026-08-23.** An instrumentation probe wrapped
`TldwCli.on_splash_screen_closed` to timestamp splash close, and recorded the splash
closing at 6.09 s on a boot where it demonstrably closed at 3.53 s. Textual's
`_MessagePumpMeta` snapshots `@on`-decorated handlers as **raw function objects** into
`cls._decorated_handlers` at class-creation time, so a later class-attribute assignment
is invisible to that dispatch — the original still ran. Worse, the naming-convention
fallback in `_get_dispatch_methods` skips a method only when it carries `_textual_on`,
which the replacement did not, so the wrapper was *also* dispatched, a second time,
much later. One handler, two invocations, and a timestamp from the wrong one.

**What to do.** Do not patch a method that is both `@on`-decorated and named
`on_<message>`; instrument the thing it calls, or the message's own sender (here,
`SplashScreen.close`). Calling such a handler directly from a test is fine — it is only
class-level replacement plus framework dispatch that splits in two.

---

## `display = False` does not stop a widget's timers — and a paused Timer you don't hold is garbage-collected mid-pause

**TASK-23022, 2026-08-27.** Six progress widgets mounted `display: none`
(`ModelInstallProgress`'s indeterminate bar on four Lab views + Library, the
Personas CCP overlay's `LoadingIndicator`, plus a seventh found in the audit on
the Console inspector rail) burned **960 of 1018 timer fires / 15 s changing
zero pixels — 88% of the Lab screen's idle CPU**. Textual 8.2.8 gates only the
*repaint* on `is_on_screen` (`dom.py`'s `automatic_refresh`, itself a
`find_widget` raise/catch per fire); the timers themselves —
`Bar.watch_percentage`'s 15 Hz `auto_refresh` when `percentage is None`,
`ProgressBar.on_mount`'s unconditional `set_interval(1, self.update)` (armed
even with `show_eta=False`), `LoadingIndicator._on_mount`'s 16 Hz — run
forever regardless of `display`. Three mechanism facts that shaped the fix
(`Widgets/pausable_progress.py`; guarded by
`Tests/Architecture/test_progress_widget_clock_guard.py`):

1. **You cannot suppress a base class's `on_mount` by overriding it.**
   `_get_dispatch_methods` walks the MRO and dispatches every class's own
   naming-convention handler — subclass and base BOTH run. To govern a clock a
   base arms, intercept `set_interval` itself (every arm flows through it) or
   `event.prevent_default()` away the whole chain.
2. **`Show`/`Hide` events track the LAYOUT map, not the viewport.** The
   compositor arranges with `visible_only=False`, so `display`-hidden subtrees
   leave the map (Hide fires, scrolled-out widgets don't), and a widget
   mounted hidden receives *neither* event — the initial state must be
   "paused until first Show".
3. **A paused `Timer` whose reference was discarded is destroyed by cycle GC
   mid-pause.** Running timers are rooted by the event loop (sleep handle /
   Event waiter); a *paused* one blocked on its own `Event.wait()` exists only
   in the task↔timer reference cycle. With weak tracking the paused ETA timer
   vanished — "Task was destroyed but it is pending!" on stderr, and the clock
   would silently never resume on Show. Hold paused timers **strongly**.
   (`Timer._skip` defaults True, so resume fast-forwards without a fire
   burst, and `Timer.stop()` works from the paused state, so unmount/quit
   teardown is unaffected — both verified by lifecycle tests and a live
   Ctrl+Q walk.)

Fires are the evidence currency here: idle CPU % is load-sensitive, but
fires / 15 s reproduced the review's numbers exactly (1017 vs 1018) across
every interleaved run.

---

## `super().on_unmount()` under MRO dispatch runs the base body TWICE — the teardown-side twin of the `on_mount` trap (TASK-31418, 2026-09-05)

Same mechanism as fact #1 in the `display = False` lesson above, on the
teardown side. `MessagePump._get_dispatch_methods` walks `self.__class__.__mro__`
and calls EVERY distinct implementation of a lifecycle handler for one event.
So a subclass that both overrides `on_unmount` AND calls `super().on_unmount()`
runs the base body twice — once from its explicit call, once from Textual's
own walk.

Probed on the installed Textual 8.2.8 with a two-level `Screen` subclass whose
base and child each append their name: one Unmount event yielded
`['child', 'base', 'base']` — the base fired twice. The identical double-fire
reproduced for `on_mount` and `on_screen_resume`; all three MRO-dispatched
lifecycle handlers are affected.

Harmless while `BaseAppScreen.on_unmount` only logs, but the next
non-idempotent teardown added to a base handler (a close, a release, a
decrement, a dispatch) becomes a double-teardown bug in every subclass still
carrying `super().on_unmount()`, and the symptom surfaces far from the line
that caused it — which is why this was fixed while the base body was still
idempotent rather than after a real corruption.

**Convention (this repo): a subclass handler for a lifecycle event Textual
dispatches by MRO does NOT call `super().on_*()`** — the dispatcher already
runs the base. Every `BaseAppScreen` / `SafeModalDismissMixin` subclass follows
this for `on_mount` / `on_unmount` / `on_screen_resume`, and each site carries a
`# No super().on_*(): the dispatcher already invokes <Base>.on_* separately`
comment naming the base whose handler would otherwise double-fire.

The one exception is a `super().on_*()` whose base target is NOT itself a
dispatched handler — a plain method reachable ONLY through the explicit
`super()` call (e.g. `change_review_screen.py`'s `super().on_mount()`, whose
docstring marks it "mandatory, not politeness"). There, removing the call
DROPS the teardown, so it stays. **Classify each site before touching it:**
redundant (base is a dispatched handler → remove the super call) vs
load-bearing (base is reachable only via super → keep it).

Guarded by `Tests/UI/test_on_unmount_mro_convention.py`: a runtime count test
pins the base `on_unmount` firing exactly once under the no-super convention,
and an AST scan fails if any screen/modal `on_unmount` re-introduces a
`super().on_unmount()` call.

---

## A cached widget reference cannot be validated by `is_mounted` — it lags detachment, and `_pruning` marks the corpse first

**TASK-23025, 2026-08-28.** To get Library resize frames and focus changes off
the per-frame DOM walks (71.6 queries/frame), invariant chrome references were
cached and validated with `cached.is_mounted` before use. The existing test
`test_compact_notes_list_keeps_its_scroll_offset_across_a_sync` caught the
hole: a targeted canvas sync REPLACES `#library-notes-list`, and the scroll
restore, resolving its owner through the cache, scrolled the doomed old list
("notes list scroll fell 12 -> 0"). Mechanism, from Textual 8.2.8 source:
`App._prune` marks the whole pruned subtree `_pruning = True` and posts
`Prune()` *synchronously*, but the actual detach (`_unregister` →
`_detach`, which nulls `_parent`) and the `_is_mounted = False` flip happen
later when the message is processed — so there is a window where the corpse
still answers `is_mounted` as True while `query_one` (which walks the live
tree) already resolves the replacement. Validation that matches what a query
would return is: `not widget._pruning and widget.is_mounted` **and** the
`_parent` chain walks back to the caching screen (a handful of attribute
hops — still ~zero cost next to a DOM walk). With that check the cache is
bit-for-bit equivalent to querying; the mutation arm (validation weakened
back to `is_mounted`-only) re-fails the same pre-existing scroll test.
Related earlier incident: task-2200 ("`is_mounted` ≠ in-the-DOM").

---

## Related

- `lessons-testing-evidence.md` — includes the Pilot-harness traps (detached widget
  references after recompose, bare-`App` harnesses that never load the app stylesheet)
- `lessons-live-verification.md` — why a green suite can still miss live-only defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps

## `refresh(recompose=True)` can orphan `app.focused` and soft-lock ALL keyboard input (TASK-22281, 2026-08-25)

**What happened.** UAT finding F-1: on a cold full-track walk of the first-run
wizard, entering the Speech step killed every key — ctrl+n/ctrl+b, Escape, Tab,
even the app-level ctrl+p palette — while rendering stayed alive. 2/2
reproducible on fresh profiles, 0/2 warm. Diagnosis: `show_step()`'s focus fix
focused a child of the incoming step an instant after the step's first
`on_show` scheduled a `refresh(recompose=True)`; the recompose detached that
child, and **Textual 8.2.8 leaves `app.focused` pointing at the detached
widget**. Key events then dispatch into a dead message pump, so binding
resolution never runs at any level. The wizard soft-locked until the process
was killed. Warm entries skipped the lazy load's recompose (`_loaded` gate),
which is why Resume "fixed" it.

**The rule.** If a widget subtree can recompose while one of its children may
hold focus, focus must be re-anchored after every recompose — a one-shot fix at
the focusing site is re-orphaned by the next recompose (the load-completion
callback recomposed again moments later). The fix that held: `SetupStep`
overrides `refresh()` to `call_after_refresh(self._heal_orphaned_focus)`
whenever `recompose=True`; the heal no-ops if focus is alive or the step is
hidden, else refocuses same-id-in-new-tree → preferred_focus → first focusable
→ nav bar. Regression test: walk the cold path with Pilot and assert BOTH
`app.focused.is_attached` AND that a real `pilot.press("ctrl+b")` still
navigates — the mechanism assertion alone is necessary but not sufficient.

**Diagnostic trap discovered en route.** `logger.info(...)` from wizard code
never reaches the persistent app log — the sink records only the structured
`diagnostics.*` events — and loguru's default stderr sink is swallowed by the
TUI. Instrument with a plain append-to-file probe gated by an env var (or a
diagnostics event); a probe you cannot see is indistinguishable from a probe
that never fired.

## Textual's focus order is VISUAL (y, x) order, not DOM order (TASK-21142, 2026-08-25)

**What happened.** To make Tab reach Next before the abandon button, the
wizard footer's DOM was reordered (Next composed first) with dock CSS keeping
the visual convention. The focus chain measurably did not change: Screen's
`focus_chain` sorts siblings by `_focus_sort_key` = `(y - margin_top,
x - margin_left)` from each widget's virtual region. DOM order only breaks
ties. The fix that worked was changing the VISUAL order (Windows-wizard
footer: progress left, right-aligned Back/Next/Exit) so the sort itself
produces the desired traversal.

**The rule.** To change Tab order in Textual, change where widgets sit on
screen (or intercept keys); moving them in compose() while CSS restores the
old geometry changes nothing.

**Sibling trap from the same task (TASK-21148).** A widget hidden only via a
`.hidden` class is display-none ONLY where the app stylesheet is loaded;
bare-App test harnesses have no such rule, so the "hidden" widget keeps its
docked row and shifts every geometry below it. Anything meant to be
invisible in all hosts must also set `widget.display = False` (or carry the
rule in DEFAULT_CSS).

---

## `can_focus` + `display` is not the focus chain — ask `screen.focus_chain`

**TASK-23194, 2026-08-29.** A UX audit of the Console Context rail reported three
zero-size focusable widgets in the Agent and Workspace sections — including a text
`Input` — and concluded that Tab could land on a control painting nothing, filing it
as an accessibility defect. It was wrong, and the wrongness came from the query, not
the rail: the audit enumerated widgets with `getattr(w, "can_focus", False) and
w.display`.

A widget's own `.display` reports **its own** `styles.display`, not whether it is
actually reachable. All three offenders sat under a hidden ancestor
(`console-workspace-context-action-row` had `display=False`; the steering bar was
`display: none`), so each child still answered `display=True` while being unreachable
and painting at region `(0, 0, 0×0)`.

Textual already accounts for this. `Screen.focus_chain` walks the DOM tracking
ancestor visibility, and probing it gave `in_focus_chain=False` for all three. There
was no defect and no fix — the finding was withdrawn and the test rewritten to pin the
invariant that IS user-facing: nothing **in `screen.focus_chain`** may have a zero-size
region.

```python
# Wrong: answers "is this widget itself displayed", not "can Tab reach it".
[w for w in rail.query("*") if w.can_focus and w.display]

# Right: Textual's own reachability answer.
rail_nodes = set(rail.query("*").nodes) | {rail}
[w for w in screen.focus_chain if w in rail_nodes]
```

## `Widget.size` excludes borders and padding; `outer_size` includes them

**TASK-23193, 2026-08-29.** Measuring the Context rail's vertical budget, section
headers reported `size.height == 1` while the rendered rail plainly showed two rows per
header. The audit reconciled that by inventing a mechanism — a uniform "2 blank rows +
1 separator" gutter between sections — and recommended collapsing a gutter that does
not exist.

`size` is the **content** region. `.console-rail-section-header` carries
`border-top: solid`, which consumes a row outside the content box, so a header is 1
content row + 1 border row. The rest of the apparent slack was inside section bodies,
not between them. The row totals in the audit were right; the explanation was not, and
a fix aimed at the invented gutter would have missed.

When reconciling a measured height against what a capture shows, compare `outer_size`
(or `region.height`) — and remember a `border-*` rule silently costs a row per edge in
a rail where rows are the scarce resource.

## A widget reference taken before a recompose is stale, and `Button.press()` no-ops on it silently

**TASK-23193, 2026-08-29.** After a default-layout change, two
`test_console_new_workspace` tests failed with "Workspace create modal did not open".
The handler looked broken; it was not. The helper did:

```python
button = console.query_one("#console-new-workspace", Button)   # captured early
rail.activate_section("workspace")                              # tray recomposes here
...                                                             # awaits
button.press()                                                  # presses a corpse
```

`ConsoleWorkspaceContextTray` re-mounts its children when its section opens, so the
captured `button` was detached: `display=False`, `region=(0, 0, 0×0)`. Textual's
`Button.press()` opens with `if self.disabled or not self.display: return self` — it
posts nothing and raises nothing, so the failure surfaces one layer away as "the
handler never ran". Two debugging rounds went into the handler and the message routing
before a probe printed `button.display`.

Re-query after **every** await that could trigger a recompose, and take the reference
the caller will act on *after* the last one — scrolling counts, because
`scroll_to_widget` can itself provoke another reconciliation pass.

## Textual focuses the clicked widget BEFORE the press bubbles — a click-outside dismissal must not restore the popup's opener

**TASK-25709, 2026-08-30.** Wiring click-outside dismissal for the Context rail's
conversation action menu, the obvious implementation reused the menu's existing
`Dismissed` handler, which returns focus to the opener asterisk. In Textual 8.2.8
`Screen._forward_event` calls `set_focus(focusable_widget)` on every `MouseDown` and
only then dispatches the event into the widget tree (`textual/screen.py:1607-1610`) —
so by the time a screen-level `on_mouse_down` dismissal runs, focus has ALREADY moved
to the clicked widget. Posting an opener-restore from there yanks focus back to the
rail after the user clicked into the composer. The same applies to an Escape issued
while focus sits outside the menu.

Thread the restore through the dismissal cause: the popup's own in-menu Escape
restores the opener (focus was inside the popup); outside-click and stranded-Escape
paths skip it. The pinned test is
`test_click_outside_closes_the_menu_without_dispatching`, which asserts focus is NOT
the opener after the click.

## `run_worker(exclusive=True)` CANCELS the group — it never queues behind it

**schedules-redesign PR-3 Qodo round, 2026-09-03.** The Automations pane's in-place
row editors dispatch one `save_definition` worker per commit, and `save_definition`
merges the payload onto the row it reads at entry — a read-merge-write. A first review
found that grouping every commit under one key made a second row's edit cancel the
first mid-flight (fixed by keying the group per FIELD); the Qodo round then found the
mirror bug: per-field groups let two fields of the SAME definition run concurrently, so
the slower one wrote back a snapshot taken before the faster one landed. The pinned
test (`test_two_fields_of_one_definition_both_land_without_a_lost_update`) fails on the
per-field version with `KeyError: 'model'` — the first edit is simply gone.

The obvious fix — key the group per definition and keep `exclusive=True` — does NOT
serialize them. `WorkerManager._new_worker` cancels the group's running workers and
then starts the new one; there is no queue. That version fails the same test from the
other side, with `WorkerCancelled: Worker was cancelled, and did not complete.`
raised out of `pilot.app.workers.wait_for_complete()`.

`exclusive=True` means "only the newest matters" (a live filter, a repaint, a search).
When every dispatch must actually land, that is the wrong primitive: hold an
`asyncio.Lock` keyed by whatever must serialize, and leave the worker non-exclusive.
The group name is then only a label for observability.

---

## A `Select` posts `Changed` the moment it MOUNTS with a value preselected — a close-on-first-Changed handler self-destructs

**Schedules redesign PR-3, final review M8/F8 (2026-09-03), re-probed on Textual 8.2.8.**
The detail panes' in-place row editors mount a `Select` with the row's current value
preselected. `Select.value` is `var(NULL, init=False)`, so `_on_mount`'s
`_init_selected_option` assignment is a **real** change from `NULL` — and `_watch_value`
turns it into a posted `Changed`. The handler therefore fires with the current value
before the user has touched anything.

The consequence bit twice. A handler written as "on Changed, commit and close the editor"
closes the dropdown the instant it opens, so the control is unusable; a review ruling that
prescribed exactly that (make a same-owner pick call `end_edit`) had to be adjudicated
**not implementable** for this reason. The docstring at `task_detail.py:1271` records the
probe.

The mirror fact matters as much: a genuine re-pick of the **same** option posts *nothing*,
because `Select._update_selection` assigns only `if value != self.value`. So the
"unchanged value" branch is reachable ONLY from the synthetic mount event.

**What to do.** A `Changed` handler on a preselected `Select` cannot distinguish the mount
echo from a real commit by the event alone — compare against the **stored** value and
no-op when they match (`task_detail.py:1200`, `definition_detail.py:1391`). Never close,
persist, or navigate on the first `Changed` after `begin_edit`/mount.

---

## `DataTable.clear()` posts a `RowHighlighted` for row 0 before the rows come back

**Schedules redesign PR-4, final review F12.** Every table re-render clobbered the
selection: `clear()` posts a `RowHighlighted` for row 0 *before* the new rows are added, so
the handler re-rendered row 0's detail (overwriting `_selected_row_id` on the way) and only
then did the cursor restore win it back. `move_cursor` back to the restored row then posts
a **second** echo — for a row the render had already fed the detail pane directly.

Both are echoes of the render's own work, not user intent, and both are *stale by the time
they are processed*: the table's live cursor has already moved on.

**What to do.** Guard a `RowHighlighted` handler with two checks (`schedules_workbench.py:1598`):

```python
if event.cursor_row != event.data_table.cursor_row:
    return          # the event is stale — the live cursor moved on
if self._visible_rows[event.cursor_row].row_id == self._selected_row_id:
    return          # already rendered; a refresh's direct feed did it
```

The first is the general rule for any `DataTable` message: **the index in the event is a
snapshot, the table is the authority.** The second is the unchanged-selection discipline
that makes a re-feed idempotent.

---

## Never carry a row INDEX across an `await` — capture the row's IDENTITY

**Three separate occurrences across the schedules handoff and redesign programmes.** Same
class each time: an index that was correct when it was read, resolved against a list that
had changed by the time it was used.

1. **The worst one (PR-4 fix wave F2).** The narrow-width pushed detail pane was fed by
   index. A background refresh that dropped the open row fell through `_render_table`'s
   `target_index = 0` and re-fed the overlay with a **different row's data while its header
   still named the original** — a full-screen pane whose Delete button targeted the wrong
   reminder. Fixed by pinning the pushed pane to `_pushed_row_id` (`UnifiedRow.row_id`),
   feeding it only for that identity, and auto-popping with a notice when the row leaves
   the queue (keyed off what EXISTS, never off what the current filter SHOWS — a filter
   narrowing must not close an open pane).
2. **Audit-view highlight race (task-18940 slice 4).** The run-history pane loads a
   definition's server audit trail on highlight. Two quick highlights raced; the guard is
   that a newer highlight wins, keyed on the definition id the load was started for.
3. **The `RowHighlighted` echoes above** — the same bug in message form.

**What to do.** The moment a handler contains an `await`, treat every index it holds as
expired. Capture the row's stable id before the await and re-resolve after it; when a
worker's result comes back, check the id it was started for against the current selection
before painting anything. `exclusive=True` is not a substitute — see the `run_worker`
entry above for why cancellation is a different primitive from serialization.

---

## A geometry or `.display` test without `CSS_PATH = BUNDLED_STYLESHEET` measures nothing

**Schedules redesign PR-1 task 3 and PR-4 task 6.** Width-driven behaviour in this app
lives in app-tier rules (`css/features/_scheduling.tcss`, reached through the bundle).
`ConsolidatedCSSApp` loads the per-screen sheets but **not** the app bundle, so in a bare
harness every `.compact` rule is simply absent — a test asserting that a pane hides below
84 columns passes or fails for reasons unrelated to the rule it claims to cover.

`Tests/UI/test_schedules_responsive_floor.py` says it outright: without the app tier
"every `.compact` rule is absent and the geometry claims measure nothing."

**What to do.** Any test asserting on `region`, `.display`, or a width breakpoint sets
`CSS_PATH = BUNDLED_STYLESHEET` (see `Tests/UI/consolidated_css.py`). A test that
deliberately runs *without* it — forcing `.display` directly to isolate a non-CSS claim —
must say so in its docstring, as `test_schedules_keyboard_map.py:115` does. And remember
the sibling trap from the paint-over hunt: widget-tier CSS (`BUNDLED_CSS`/`DEFAULT_CSS`)
loses to app-tier rules regardless of specificity.

---

## Target CSS by CLASS on the subject — never an ancestor-scoped bare type

**TASK-25810's ratchet, enforced at `Tests/Performance/test_textual_css_fastpath.py`.**
Textual indexes each rule under its **rightmost** selector only. So `#panel Button` is a
candidate for **every** `Button` in the app — all ~110 of them — and each pays a full
selector evaluation before the ancestor filter rejects it. Measured 2026-08-30: rules of
this shape were **93% of all per-node candidate work** on a 502-node Console.

`MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` is a ratchet under ADR-097's discipline: pinned at
274 (measured 264 + 10 slack), and **never raised**. On a breach the fix is to re-key the
new rule — give its subject a class carried only by the intended widgets,
`#panel Button` -> `Button.panel-action` — not to widen the budget. When re-keying work
lands, the constant is LOWERED so the freed headroom is banked.

The same discipline governs the boot-parsed CSS byte budget (`MAX_BOOT_PARSED_CSS_BYTES`),
whose comment records the reason both are ratchets rather than limits: *"the CSS byte
budget's history is three cycles of silent regrowth."*

**What to do.** Write `Widget.purpose-class`, not `#container Widget`. Check both ratchets
before opening a PR that adds CSS, and attribute any growth to the segment that caused it —
the failure message names segments and sources precisely so that attribution is not
guesswork.
