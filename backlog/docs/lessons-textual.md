# Lessons: Textual framework traps

Working knowledge about the Textual version this repo pins (8.x, currently 8.2.8) —
API gotchas that cost a debug cycle when first met. Not decisions (see
`backlog/decisions/`) — these are traps that have actually cost time here, kept so the
next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore gets ignored. If you add one, bring the incident.

---

## Covered screens retain old geometry, and a fixed number of Pilot pauses does not prove quiescence

**TASK-32562 backup UAT, 2026-09-14–15.** Recomposition below stacked backup
screens left the retained Console tray's old positive region in place while its
new children had no layout. A native reproduction on `63756` counted 3,839
height-fit callbacks in 0.25 seconds: requesting another refresh could not lay
out a screen hidden beneath an opaque screen. The reviewed correction waits
once for the owning screen's layout signal and releases the subscription on
delivery or unmount. The analogous bounded-section correction in `2031d` keeps
ordinary scheduling for current/background-visible screens; an initial broader
wait broke an existing visible resize test and was narrowed before publication.

The Windows returned-layout test then passed its geometry, scroll, focus, hint
and allocator assertions but still had a pending callback after three
`Pilot.pause()` calls. The last pause can itself enqueue reconciliation. The
`dc96b` test correction observes completion with a finite two-second wait,
retaining both the earlier covered-screen callback stability checks and the
final callback-count stability assertion. It does not permit an infinite loop.
The exact Windows repeat passed all eight bounded-layout cases, both guidance
cases and the installed plaintext backup/restore/Open journey (59 native and
16 selected product passes overall). See the
[artifact and source audit](../../Docs/Development/backup-uat-remediation-evidence-20260913/windows-dc96b-restore-independent-review-20260915.md)
and [remediation record](../../Docs/Development/backup-uat-remediation-2026-09-13.md).

Use actual screen visibility/layout events to resume deferred geometry work,
and test quiescence as a bounded condition with a separate callback-stability
check. A positive old region or a fixed count of Pilot turns establishes neither.

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

## `call_after_refresh` is two message hops, not a layout pass — a refresh-counted wait can never see a widget's first `Resize` (TASK-34000.25, 2026-10-09)

`MessagePump.call_after_refresh` posts `InvokeLater` to itself and then `call_later`s
the callback on the app: it runs after the pending messages, with no real time and no
compositor reflow in between. A chain that re-arms itself through it therefore burns
its whole budget inside one frame. Incident: the Media Reader's reading-position
restore (task-31968's `settle(8)`, one `scroll_to(immediate=True)` per hop, re-armed
while the offset clamped short) was moved onto the rail-return path. The Raw view
builds its wrap index on its FIRST `Resize` — delivered by the compositor after a
layout, i.e. a frame later — so all eight re-applies ran on the same unindexed view at
`max_scroll_y == 0`, and raising the budget to 48 changed nothing (the probe recorded 48
of 48 at 0, then 287 rows one `pilot.pause()` later). Routing the hop through
`call_later` was no better: same reason. The fix was a layout SIGNAL, not a count: the
Raw view posts `Indexed` after `_build_index_now`, Textual's `Markdown` posts
`TableOfContentsUpdated` after its last block batch mounts, and
`LibraryMediaContentBody.run_when_laid_out` hands the continuation to whichever applies.
Rule: `call_after_refresh` orders you after *messages already queued*; it says nothing
about layout. To wait for geometry, wait for the event that produces it (`Resize`, a
widget's own "built" message), and keep a refresh-count only as the bound on a body that
is already laid out and merely growing.

## A pane whose content just fits flips its scrollbar on focus — reserve the gutter

**TASK-33007.6, 2026-10-04.** Folding the rarely used Providers & Models controls
made the card fit the Settings detail pane at 211x44 (32 rows of content in a 32-row
pane). Focusing the Default model picker opens two more rows. That brings in the
vertical scrollbar, and leaving the picker takes it away. Each flip changed the
content width by one column (122 → 121 → 122) and re-laid out the whole card.

`test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left[focus-moves-away]`
went from 0/12 to 6/12 failures. Its 0.05 s blur timer now fired about 225 ms after
the blur instead of about 85 ms, which is later than the test's `pause(0.2)`.

A CSS bisect pinned it down. Restoring either the card's frame or the Advanced
disclosures' old 3-row frames brought the delay back to about 82 ms. Both only make
the content overflow all the time, so the scrollbar never flips. Removing the new
draft-status refresh hook did nothing (8/10 still failed). `scrollbar-gutter: stable`
on `#settings-detail-pane-body` fixed it: the delay returned to 82–91 ms, and the
test failed 0 times in 12 runs.

A pane whose content sits within a row of its height will meet this sooner or later,
because any focus-driven growth crosses the fold. Reserve the gutter on the scroll
container. Pin the width with a test: measure the card's width at rest and with the
growing control focused. That test failed (121 vs 122) when the gutter was removed.

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

**It recurred: TASK-34000.1, found by the whole-branch review on 2026-10-04.** The
Library note autosave got a maximum wait, and its delay was computed as
`max(0.0, min(debounce, burst_start + max_wait - now))`. No constant was zero, so nothing
looked like this entry; the zero appeared only once a burst outlived its max wait (a quit
prompt held open, a rail switch away and back, or a key landing between the deadline and
the callback). The timer callback was also the only thing that ended a burst, so every
later keystroke armed another dead timer: autosave stopped for good under a header that
still said "changes save automatically", and stopping the app raised
`ZeroDivisionError`. The unit test asserted `delays[9] == 0.0`, with the comment "saves
at once", against a fake `set_timer` that recorded the delay and never ran a timer, and
it passed the task's own reviews.

**What to do, in addition.** A *computed* delay is the same trap as a zero constant:
clamp it to a small positive floor where it is computed (`AUTOSAVE_MIN_DELAY_SECONDS`,
0.05 s), not at the call. A fake `set_timer` cannot tell a delay that fires from one that
never will, so at least one test must arm the worst-case delay on a real message pump
(`App().run_test()`, a dozen lines) and assert that the callback ran and that leaving
`run_test()` did not raise.

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

**Is there ever a safe `super().on_*()`? Not to a normally-defined base
handler.** A base `on_mount`/`on_unmount` defined in its own class `__dict__` is
separately MRO-dispatched, so an explicit `super()` call ALWAYS runs it a second
time — there is no "shadowing." The legitimate way to keep a base body that runs
once *and* is invoked explicitly is the `BaseWizard.on_mount` pattern: end the
base handler by calling a PLAIN method (`_post_mount_hook()`), and let subclasses
override *that* plain method instead of `on_mount` — a non-dispatched method runs
exactly once. So the convention reduces to: a subclass never calls `super().on_*()`
for a dispatched handler; a base that needs an explicit call exposes a plain,
non-`on_*` method for it (as `BaseWizard` does).

**The mount side carried the same latent bug (TASK-31822, 2026-09-06,
converted).** All 19 live `super().on_mount()` calls found repo-wide —
including the two Console modals and `change_review_screen.py`'s
`ChangeGitCommitModal`/`ChangeGitPushModal`, whose docstring misdescribed the
mechanism as "ordinary attribute lookup … so defining one here SHADOWS the
mixin's" (it does not — Textual walks the MRO and dispatches both, so that
`super().on_mount()` double-fired `SafeModalDismissMixin.on_mount`) —
classified as redundant: every one resolved to a base (`SafeModalDismissMixin`
in 18 sites, `LibraryAdaptiveReaderShell` in 1) whose `on_mount` is defined in
its own class `__dict__` and therefore already separately MRO-dispatched.
Zero sites needed the plain-method escape hatch; `BaseWizard._post_mount_hook`
remains the only genuine run-once-and-callable case in the repo. One site
(`PersonalContextReviewModal.on_mount`) had no body besides the `super()`
call, so the override was deleted outright rather than left as a dead
pass-through. The two misleading `change_review_screen.py` docstrings were
corrected to describe the MRO walk instead of "shadowing."

One extra check the mount side needed that the unmount side did not:
Textual's `_get_dispatch_methods` walks `self.__class__.__mro__` — most
derived class first — so a subclass's own `on_mount` is dispatched *before*
its base's separately-dispatched `on_mount`. That means removing a leading
`super().on_mount()` does not just drop a redundant call, it also **reorders**
the base's body to run strictly after the whole subclass method returns
(rather than inline, before the subclass's later statements). This is safe
only if nothing later in the subclass method reads state the base's `on_mount`
sets. Audited every site for reads of `SafeModalDismissMixin`'s
`_safe_cancel_pending` / `_safe_opener_focus_ref` / `_safe_opener_focus_id` /
`_safe_mount_generation` / `_safe_backdrop_event_in_attempt` (none found) and
the `LibraryAdaptiveReaderShell` site (base does layout sync + a deferred
`post_message`, subclass queries a static DOM id — no overlap either) before
converting.

Guarded by `Tests/UI/test_on_unmount_mro_convention.py` (unmount) and its
sibling `Tests/UI/test_on_mount_mro_convention.py` (mount, TASK-31822): each
pairs a runtime count test pinning the base handler firing exactly once under
the no-super convention with an AST scan that fails if any screen/modal/widget
re-introduces a `super().on_*()` call to a dispatched handler.

---

## `is_mounted` is still False inside `on_mount` — a guard on it makes the mount-time apply a silent no-op (TASK-34000.25, 2026-10-09)

Textual sets `_is_mounted = True` in the `finally` AFTER `_dispatch_message(events.Mount())`
returns (`message_pump.py`, `_pre_process`), so any method called from an `on_mount` handler
that early-returns on `not self.is_mounted` stores its input and applies nothing. Incident:
`LibraryNotesCanvas.apply_session_state` guards on `is_mounted`; `on_mount` →
`_apply_post_compose_state` called it to set the editor's `display` flags, and the call never
passed the guard — a freshly composed editor kept every surface as composed, with the "This
note changed elsewhere — Overwrite / Reload" callout visible under "Saved". Every other open
path re-synced a moment later and masked it; the rail return to a retained, untouched note did
not, which is the review's "false conflict render" (S-02), reproduced live at 160x45 and in a
Pilot probe that logged `is_mounted=False` at both calls. Widening the guard to `is_attached`
was NOT the fix: the dead apply had been dead since the canvas was written, and two pinned
behaviours (the emptied-blank GC on Back, the compact Preview scroll memory) went red once it
ran. The fix composes the display-gated surfaces from the state in `_compose_editor` itself.
Rule: inside `on_mount`, `is_mounted` is False; a compose must not rely on a later apply to
hide what it composed, and before "repairing" a dead mount-time call, run the pinned suite —
code has grown around its absence.

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

## Removing menu borders cannot make too many actions fit

**TASK-32033, Copy selection.** Adding the eighth action to the Console
selection menu made its seven-row transcript fixture fail containment:
removing the border and hint still left eight action rows, and clamping the
offset put the last action over the composer. The menu now caps its height
to the owner and scrolls after the existing compacting pass. The regression
checks normal and ANSI modes, including keyboard wrap from the first action
to the last and back, with both focused labels visible inside the owner.

Qodo's follow-up found that the height cap and hidden hint persisted after
the owner grew. Four resize regressions reproduced this for terminal and
transcript changes in both color modes. A capped menu may not resize when
its owner grows, so the fix uses the screen layout signal to detect changed
owner bounds, clears the temporary constraints, and measures again.

**What to do.** When adding an action to a floating menu, exercise an owner
shorter than the action count. Compact styling needs a scrolling fallback
when the actions alone exceed the available rows. Exercise shrink/grow
cycles too; the child's own resize event cannot reliably detect available
space changing around a capped child.

---

## A screen-owned worker must check the active category before updating shared chrome

**TASK-32189 review, 2026-09-09.** The Web Search controller kept an explicit
search test alive while the user opened Overview. Its completion correctly
discarded stale provider evidence, but the generic draft-status callback still
replaced Overview's State banner with Web Search's banner. The callback updated
shared status widgets before checking the active category. A mounted test with
a paused probe reproduced the wrong banner after navigation.

**What to do.** Keep category-specific draft ownership independent of panel
remounts, but gate shared status/inspector updates on the active category. A
background completion may refresh its own rail marker. The regression is
`test_probe_completion_does_not_repaint_another_category` in
`Tests/UI/test_settings_web_search.py`.

## Raw editor async work needs the live document and a lifetime beyond the screen

**TASK-32190 review, 2026-09-09.** The first raw-draft controller passed its
model race tests but lost a real typed character when a status refresh ran
between TextArea's document update and delivery of `TextArea.Changed`. A second
mounted probe removed Settings during a save: cancelling the screen worker did
not stop its file-writing thread, but did skip runtime publication to the app
and leave a copied draft falsely conflicted. Synchronous initial reads also
acquired the config writer's lock during composition.

**What to do.** Read files in workers, capture the editor document before async
revision checks, and avoid assigning stale model text during status-only paints.
Give persistence completion an app-owned worker and retain the live session in
the existing memory-only navigation store. Rebind view callbacks on restoration
and detach only callbacks still owned by the outgoing view. The regressions are
`test_status_refresh_cannot_erase_a_queued_keystroke`,
`test_save_finishes_after_settings_destination_is_recreated`, and
`test_initial_config_read_does_not_block_ui_thread` in
`Tests/UI/test_settings_raw_draft.py`.

**TASK-32191 recurrence, 2026-09-09.** The guided Web Search `Input` controls had
the same pending-event loss, and copied navigation state lost the selected
backend and save result. Retaining the live session fixed those failures, but
review found two event-order traps: a new panel mounted before the old panel's
unmount and took callback ownership, so ownership-guarded invalidation was
skipped; and Clear emitted a refresh that captured old pending text and undid
the clear. Invalidate evidence at the new view boundary as well as teardown,
and capture input before applying explicit Clear/Revert intent. A real-write
fault-injection test also exposed false failure wording after a successful
write followed by reload failure: preserve the committed baseline and report
the disk-write and runtime-refresh outcomes separately. These regressions live
in `Tests/UI/test_settings_web_search_lifecycle.py`.

**PR #2562 Qodo review, 2026-09-10.** The guided editor's committed-write
handling did not cover raw replacement. Six real-file regressions reproduced
stale baselines or false failure wording when snapshot reads, runtime
publication, or the Settings refresh callback failed after replacement, with
and without newer edits. Return the committed snapshot independently of refresh
success. If the snapshot itself is unavailable, report the successful write
and require reload before another save; retain newer edits and tell users to
copy them before restarting. The regression is
`test_committed_raw_save_survives_refresh_failure` in
`Tests/UI/test_settings_raw_draft.py`.
Independent re-review then caught a second use of the missing snapshot: a
remount treated it as a first load and silently adopted an external edit.
Two mounted regressions now keep that state blocked across navigation and
validation until an explicit successful Revert establishes the new baseline.

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

## `A:focus-within B` restyles B only if A carries a `:focus-within` rule of its own (TASK-33007.3, 2026-10-03)

The Settings Default model picker opens across its row while it holds focus:
`#settings-model-row:focus-within .settings-source-word { display: none; }` and a
matching width rule on the picker. The picker's own `:focus-within` rules
applied, but the live 211x44 capture showed the row's Source word and help still
painted, and the picker squeezed to a third of the row (it now shared `1fr` with
two siblings that should have gone).

`Screen._update_focus_styles` does not re-match every selector. It walks the
focused widget's ancestors from the screen down and restyles the subtree of the
FIRST ancestor whose `_has_focus_within` is set, and `Stylesheet.apply` sets that
flag on a node only when a rule whose selector ENDS on that node uses
`:focus-within`. A descendant selector `A:focus-within B` ends on `B`, so it flags
`B`, never `A`. Here the picker was the outermost flagged node, so only its
subtree was restyled; its siblings in the row never heard about the focus change.

**The rule:** when a `:focus-within` on a container must restyle that
container's other children, give the container a `:focus-within` rule of its
own (any property; `#settings-model-row:focus-within { height: auto; }` here),
and pin the effect with an assertion on a sibling's `display`, not only on the
focused widget's subtree. Related trap from the same task: `Button.press()`
returns without posting `Pressed` while the button is `display: none`, so a test
that presses a focus-revealed button must focus its owner first.

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

**The `DescendantFocus` MESSAGE lands the other way round (task-32100, 2026-09-10).**
`set_focus` runs before the press, but the event it posts is a queued message, so a
`DescendantFocus` handler runs AFTER the `Button.Pressed` handler for the same click.
A handler that reads a focus change as "the user took control" therefore revokes work
the press handler started ~20 ms earlier. Library Notes: clicking a note row started a
folder-tree locator, and the click's own focus event then superseded it — abandoned on
every open, traced as `supersede -> 3 / locator start gen=3 / supersede -> 4 / locator
end -> False`. Fence background work on the focus intent only when that work will
actually take focus; work that just repaints has no stake in it. Only the real gesture
reproduces this — calling the same coroutine directly from a test posts no focus event
and passes.

**A list that cannot take focus loses a held click if it closes on blur (TASK-33007.2
review round 1, 2026-10-03).** The Settings Provider combobox kept its list at
`can_focus = False`, so the control stayed the one Tab stop, and closed the list 50 ms after
the control blurred. A press on a row focuses the nearest focusable ANCESTOR instead, here
the scrolling detail pane (`get_focusable_widget_at` walks `ancestors_with_self`). The
control blurs, and the timer closes the list. `OptionList` chooses only on `Click`, which
App builds at `MouseUp`, and only when the same widget is still under the pointer
(`app.py` `on_event`). Measured in tmux with SGR mouse sequences: a 0 s hold chose Ollama,
a 0.1 s hold chose a **wrong** row (Arcee AI), and 0.3 s or 1.0 s holds were lost.
`pilot.click` could never show this: it forwards `MouseDown`, `MouseUp` and `Click`
straight to the screen, so no focus change can come between them. To reproduce it,
post the events through App the way the driver does
(`app.post_message(events.MouseDown(None, x, y, 0, 0, 1, False, False, False, x, y))`,
wait, then `MouseUp`). Then poll for the outcome: the Click, `OptionSelected` and
`Select.Changed` chain is posted after `pause()`'s idle wait starts, so a fixed
`pause(0.2)` flaked under `-n 8`. The fix: when the timer finds that focus went to a
container of the open list and the pointer is over the list, re-focus the control
instead of closing. Guard the lookup with `QueryError`, because the timer can fire
during teardown.

## `Screen`'s Tab binding is not `priority`, so a burst types past it — and `pilot.press` can never show you

**task-32106 / PR #2571 review round 1, 2026-09-10.** A live report said typing
title–Tab–body in one burst appended the body to the title. I could not reproduce it
and closed the criterion, having disproved a mechanism nobody proposed (Textual's
parser has no burst-to-`Paste` heuristic — true, and irrelevant). The real mechanism
is message dispatch: `Screen.BINDINGS`' `Binding("tab", "app.focus_next")` is **not**
`priority=True`, so `Key(tab)` is *posted* to the focused `Input` and bubbles one
message-queue hop per ancestor, while the App keeps dequeuing the following keys and
forwarding each to `self.focused` — still the field the user meant to leave. Stock
Textual 8, nothing from this repo:

```
pilot  elapsed=1268.9ms  title='My first note'      body='hello'
burst  elapsed=   0.2ms  title='My first notehello' body=''
```

**Two lessons, and the second is the expensive one.** (i) A field that must hand focus
over mid-sentence needs its own `priority=True` Tab binding — namespaced (`screen.` /
`app.`), because a bare action resolves against the `Input`, which has no
`action_focus_next`, and the binding then silently never fires. It belongs on every
field of the form, `TextArea` included (a peer hit the same symptom on the note body),
and on a `TextArea` it is correct only while `tab_behavior == "focus"` — assert that
rather than trusting the default. Put it on the FIELDS, not on the screen: a
priority Tab at screen level preempts every `on_key` Tab trap the screen owns (here,
the note delete prompt's). (ii) `pilot.press` is
the OPPOSITE of a burst: `App._press_keys` awaits `wait_for_idle(0)` twice plus the
animator between every key, so the loop fully drains between keystrokes (~200 ms each
here). Any defect whose trigger is "faster than the event loop" is invisible to it, and
a test written with it is green by construction. Post the keys yourself with no awaits
— `ev = events.Key(k, char); ev.set_sender(app); app._driver.send_message(ev)` — then
one `pause()`. A task that says "1 s gaps behave" is telling you the drained-loop
harness cannot see it.

## A terminal's Alt+letter carries the letter, so a focused field types it — `pilot.press` sends none

**TASK-33006.4, 2026-10-02.** Chat settings bound `Binding("alt+m", "change_model")` on
the modal and a pilot test pressed `alt+m` with Temperature focused: green. Live in tmux
(`send-keys M-m`), Alt+M typed an "m" into Temperature and nothing opened. Textual's
xterm parser names the key `alt+m` but passes the letter through as `character="m"`, so
the focused `Input` handles it as text before a non-priority screen binding is reached;
`pilot.press("alt+m")` builds the `Key` with no character, so the test cannot see it (the
composer met the same trap as TASK-1800, `console_composer_bar._is_modified_chord`). A
screen-level Alt chord that must work from a text field needs `priority=True`, and its
test should post the event the driver posts: `app.post_message(events.Key("alt+m",
"m"))`, which failed before the fix and passes after (`test_alt_m_opens_pick_mode_from_a_focused_field`).

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

**`Input` does the same (TASK-33007.2, 2026-10-03).** Settings' one-row Provider control is
an `Input` composed with the chosen provider's name, whose `Changed` handler filters the
provider list. Its mount echo arrived unfocused with the value "Anthropic" and filtered the
resting list down to one row, so the help line read "1 found" before anyone typed. Ignoring
that echo then left the help blank, because the resting copy had only ever been written by
the echo's refresh. Treat a `Changed` whose value equals the committed display text as no
query (`handle_provider_search_changed`), and compose the resting copy directly.

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

**The reverse trap: a new widget sized only by an app-tier class breaks bare harnesses.**
TASK-33003.5 added an Esc hint to the Chat settings footer as
`Static(..., classes="... w-auto")`. `.w-auto` lives in the app bundle
(`css/utilities/_helpers.tcss`). Most modal tests (`ModalHarness`, `_SettingsCloseHarness`)
mount without the bundle, so there the Static fell back to full width. It pushed the whole
button row past the right edge (`Cancel` at x=211 on a 211-column screen), and
`pilot.click("#console-settings-cancel")` raised `OutOfBounds` in tests that never mention
the hint. The fix was a `Label`: its own `DEFAULT_CSS` is `width: auto`, which every harness
loads. **What to do:** give a widget added to a shared row geometry that holds without the
bundle: the widget type's own `DEFAULT_CSS`, or the owning class's `DEFAULT_CSS`. Then
probe `region` once under a bare harness as well as under `TldwCli.CSS_PATH`.

## A focus walker's geometry guard must PASS OVER an off-screen target, never rule it out

**TASK-34000.8, 2026-10-09.** F6 landed on the Library note editor's Save while
its region sat past the right edge of a 120-column terminal (the header was one
strip shaped from the shell's breakpoint, not the pane's width). The obvious
guard -- in `Widgets/workbench_focus.py`, skip a preferred target whose region is
empty or outside `screen.region` -- fixed that and broke
`test_narrow_f6_reveals_reader_and_returns_through_items_grip` at 50x25: the narrow
Artifacts stage keeps its reader COLLAPSED until F6 focuses `#library-artifacts-body`,
and that focus is what reveals it. An unseen target is sometimes the whole point of
the walk. The rule that holds both: prefer the first preferred target that is on
screen; only when none is, fall back to the first focusable one (focusing it may
reveal its pane). "Empty region" is never disqualifying on its own -- a collapsed
pane's child has one too. And "outside the screen" is not disqualifying either
(review I-1 of the same task): a control a scrollable pane has merely SCROLLED out
of view (`region.y < 0`, `allow_vertical_scroll` True on the pane) must stay the
landing, because `focus()` scrolls it in -- F6 into a scrolled Settings form landed
mid-form before that was pinned. The pass-over applies only to a control that no
scrolling ancestor could reveal: clipped by a non-scrollable ancestor on an axis it
cannot scroll (`_scrolling_could_reveal` in `Widgets/workbench_focus.py`). Measure
visibility against the compositor's clip (`screen.find_widget(w).clip`), not
`screen.region`: a control past its own pane's edge is inside the screen and still
unseeable. And "a scrollable ancestor clips it" is not the end of the walk (PR #3055
review, Important 2): the first version returned True at the first scrollable
clipping ancestor, so a control scrolled out of a pane that was ITSELF laid out past
a non-scrollable parent's edge counted as revealable and F6 landed on it
(`test_workbench_focus_passes_over_a_scrolled_out_control_whose_pane_is_itself_clipped`).
Scrolling brings the control into that pane's content region at best, so continue the
walk with that region in hand: every clipping ancestor has to be scrollable on the
overflowed axis.

Two measurements from the same task worth keeping: (1) a content-sized compact
`Button` is `len(label) + 4` cells on the wide stage -- `padding: 0 1` plus
Textual's `line-pad: 1` on both sides -- and horizontal sibling margins COLLAPSE
to the larger one (a `margin-left: 4` after a `margin-right: 1` costs 4, not 5),
so derive a row's one-line minimum from the labels and measure the chrome once;
(2) to keep a sibling from moving when a control comes and goes, hide the control
with `visible = False` (cells reserved, dropped from `focus_chain` and
`get_widget_at` in 8.2.8), not `display = False` -- it replaces a hand-measured
`min-width` that was only ever right at one size.

---

## `allow_vertical_scroll` is False whenever the content fits -- a scroll-owner test needs overflow

**TASK-34000.7, 2026-10-08.** The fix gave the wide `#library-notes-list` `overflow-y: auto`,
and the gated test pinned it with `lst.allow_vertical_scroll is True` at 120x36 and 160x45
(green). The extended sibling asserted the same at 200x50, 235x52, 100x50 and 119x40 and all
four failed on the FIXED tree with `allow_vertical_scroll` False, `overflow_y=auto`. Textual's
property is `is_scrollable and show_vertical_scrollbar`, and the scrollbar is shown only when
`virtual_size` exceeds the container -- the 22-row first page simply fit those panes, so there
was nothing to scroll. The assertion was measuring the fixture, not the rule.

**What to do.** Assert the rule's intent (`styles.overflow_y == "auto"`) separately from the
geometry, and make the content overflow before asserting `allow_vertical_scroll`, `scroll_y` or
a reveal -- here by pressing the real "More notes" pager twice (which also proved the pager
reachable). A `max_scroll_y > 0` sanity assertion first turns "fits" into a readable failure
instead of a false RED. And re-resolve the list after any reload: the pager press replaces
`#library-notes-list` (see the recompose entry above), so the pre-press handle has no children.

## A reveal-on-open is not a reveal-on-resize -- Textual never re-scrolls the focused widget

**TASK-34000.13, 2026-10-08.** The Library Notes delete prompt got its fix in two halves: an
app-tier `height: auto` (it had been a `1fr` child squeezed to the one leftover row inside Info's
`VerticalScroll`) and a `call_after_refresh` reveal that scrolls the whole prompt into Info and
focuses Cancel in place. The gated arms were green. The extended arm that opened the prompt at
160x45 and resized to 120x36 was RED on the FIXED tree: `max_scroll_y=3`, `scroll_y=0`, both
buttons below Info's fold, Tab still trapped inside the prompt -- the review's blind-Enter shape
again, one resize later. Textual re-lays out on resize but does not scroll a focused widget back
into view; `focus()`'s `scroll_visible` happens once, at focus time.

**What to do.** A "scroll X into view when it appears" fix needs a second owner for "keep X in view
while it is open". The cheapest durable one here was the destination widget's own `on_resize`
(`LibraryNotesCanvas._keep_delete_prompt_in_view`: if the prompt is displayed,
`call_after_refresh(prompt.scroll_visible, immediate=True, force=True)`, no focus change -- the
user may be on Delete by then), which also keeps the screen's already-breached size ratchet
untouched. And put the resize-while-open arm in the extended sibling of any reveal fix: it is the
one arm the open-time test cannot stand in for.

## A one-edge `margin-bottom` rule replaces the whole margin, not just its edge

**TASK-33003.1, Chat settings disclosures, 2026-09-28.** A collapsed Chat
settings section measured `margin (0, 0, 1, 0)` even though its own class,
`.console-settings-modal-section { margin: 1 0 0 0; }`, asks for a top
margin. The winner was the leaked `Collapsible.-collapsed { margin-bottom: 1; }`
from the retired evals sheet: at (0,1,1) it outranks the (0,1,0) class, and
Textual stores `margin-bottom` as a full `margin` spacing whose other edges
are 0. It does not merge per edge the way browser CSS does. A toy app
confirmed it: `.sec { margin: 1 0 0 0 }` plus `Static.x { margin-bottom: 2 }`
gives `(0, 0, 2, 0)`. Expanding the section switched the section's spacing
from below to above, because only then did the class rule win.

**What to do.** Read a one-edge `margin-*`/`padding-*` declaration as
"margin: 0 ... <edge> ...". A higher-specificity rule that means to adjust
one edge silently zeroes the other three that a lower rule set. Restate every
edge you need in the winning rule, and measure `styles.margin` rather than
reading the sheets. The same effect makes a `margin-bottom: 0` that sits next
to `margin-left: 1` in the same rule redundant (`#remote-variant-sort`).

## Widening `#card > .row` to `#card .row` also catches nested groups' rows

**TASK-33007.6 fix round 1, 2026-10-04.** Task 6 moved two compact-workbench
rows into Advanced disclosures, and it changed
`#settings-providers-models-card > .settings-input-row` to a descendant selector
so the rows still stacked at <=100 columns. That selector also matched Catalog
refresh's per-provider rows and Custom endpoints' edit rows. Neither had matched
before, and both have compact rules of their own. Each provider row gained the
rule's `margin-bottom`, and the open catalog group grew from 71 to 96 rows at
100x40. Full-screen captures cannot show this, because the rule applies only at
<=100 columns. The fix names the containers that received the moved rows. When a
selector follows rows into a new wrapper, list every row it matches before and
after, by layout, height and `styles.margin`, at the width where it applies.

## An empty Static still takes its row: hide it, don't just clear it

**TASK-33003.8, Chat settings choice rows, 2026-09-30.** Each provider-choice
row (Reasoning effort, Reasoning summary, Verbosity, Thinking) ends with a
recovery-copy `Static` that is empty unless a restored value is obsolete. The
modal only ever called `update("")` on it, so it stayed displayed. With no
width it took the whole row, the `1fr` Select beside it resolved to 0
columns, and all that painted was the Static's thick error edge, `█`, plus
its margin row. Reasoning and thinking levels could not be chosen in Chat
settings, and origin/dev 89dd84943a shows the same row (with the old label
"Reasoning"). The tests stayed green because they set and read
`Select.value` and never looked at painted text.

**What to do.** An optional line must set `display = bool(copy)` wherever its
copy changes, including at compose. A note that shares a row with a control
should be a `Label` (its own `DEFAULT_CSS` is `width: auto`), not a `Static`.
Prove a row paints with `screen._compositor.render_strips()` text at the
control's region, after real key presses, not by reading `.value`
(`test_console_settings_choice_rows_paint_their_select`). Live-driver trap:
a mouse click opens a **blank** Select's list with **nothing** highlighted, so
the first Down lands on the blank prompt and a second Down reaches the first
choice. Opening it with Enter highlights the blank prompt row, so a single
Down is enough, as the pilot tests do. A Select that already holds a value
highlights that value whichever way it opens. (Textual 8.2.8: a click only
toggles `expanded`, whose watcher calls `overlay.select(None)` for a blank
value; only `action_show_overlay`, bound to Enter/Down/Space/Up, then calls
`action_first()`. Reproduced in a pilot probe on the modal's own Select
construction; see the task-33003.8 Implementation Notes and the live captures
in qa/model-config-p3-2026-09-28/task-8/.)

## A Collapsible's background paints only its title row: the body is `Contents`

**TASK-33003.3 follow-up, Chat settings Advanced generation, 2026-09-30.**
TASK-33003.1 set `ConsoleSettingsModal Collapsible { background:
$ds-surface-panel }` to drop the global $surface band. The expanded body still
painted $surface, because the app-wide `Collapsible > Contents { background:
$surface }` (components/_widgets.tcss) styles the `Contents` child, and the
child's own background covers the parent's. $surface is also the Chat settings
field fill, so TASK-33003.3's 12-column number fields read as full-row fields
behind a one-column edge (painted field and body both (30,30,30),
1.00:1). The width test stayed green: it measured `region.width`, which was 12.
The shipped captures showed the defect, and nobody measured their colours.

**What to do.** To restyle a disclosure's body, style `<scope> Collapsible >
Contents` as well as the Collapsible. Prove that a sized field reads as sized
by painted colour, not by region width: compare the compositor cell just past
the field's right end with the field's own fill
(`test_console_settings_disclosure_fields_read_as_sized`, red on three themes
before the fix).

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

## A UI-thread owner method must never take the owner's stop lock (TASK-31826)

`MeetingSessionOwner._stop_lock` is held across `session.stop()`, which blocks on
`call_from_thread` (the ingest submit runs on the UI thread). A controller ruling had the
learning-offer release path take `_stop_lock` "so nothing closes a worker under a batch pass";
the re-reviewer's probe deadlocked the app: a "Not now" press on the UI thread waited for the
lock while the stop worker waited for the UI thread. The phase-1 final review found the same
shape (C2, the session RLock across the ingest submit). Rule: any owner method a screen may call
on the UI thread takes only a short pointer-swap lock and runs `close()` outside every lock; the
structural guarantee ("the retained slot is written only after `session.stop()` returned") is
what prevents the mid-batch close, not a lock.

## A class flip restyles the flipped node's ENTIRE subtree — `update=False` is the escape for query-only markers (phase C task 2.5, 2026-09-08)

`node.add_class` / `remove_class` / `set_class` / `toggle_class` call
`DOMNode.update_node_styles()` by default, which is
`App.update_styles(node)` -> `stylesheet.update_nodes(node.walk_children(with_self=True))`
— **one `Stylesheet.apply` for every descendant**, not one for the node. The
same is true of the `disabled` reactive, because `:disabled` is a pseudo-class.
So a marker class set on a container is priced by the size of that container's
subtree, and setting two markers on the same node pays it twice.

**The incident.** Library phase C put a route marker
(`.library-media-route` / `.library-notes-route`) on the browse shell, with
`apply_route` as its single writer — a good design that kept ~35 route probes
honest. Instrumenting `Stylesheet.apply` by trigger
(`Helper_Scripts/library_restyle_attribution_probe.py`) showed those two
`set_class` calls were **238 of the 423 apply calls on a rail switch, 43 ms of
its 86 ms of restyle** — the largest single originator, bigger than every
widget mount on the switch put together. Neither class appears in any
stylesheet rule, so every one of those applies recomputed the same styles.

**The fix, and the guard that makes it honest.** `set_class(..., update=False)`
is Textual's own opt-out. It is only correct while no rule depends on the
class, so pin that rather than assume it — scan the PARSED stylesheet
(`app.stylesheet.rules`, each `RuleSet.selectors`), not the `.tcss` sources,
because widget `DEFAULT_CSS` is part of the same stylesheet and a grep of the
css directory misses it. `Tests/UI/test_library_phase_c_switch_storm.py::
test_route_marker_classes_have_no_stylesheet_rules` is the worked example; it
fails the moment a rule starts depending on a marker and names the seam to
restore. Visit the routes that mount the relevant widgets BEFORE scanning — a
widget's `DEFAULT_CSS` only joins the stylesheet once that class has been
mounted.

Corollary worth knowing: `Stylesheet.rules_map` is keyed by each rule's
RIGHTMOST selector only, so "is this class in `rules_map`?" does NOT answer
"can this class affect anything" — a rule like `.marker Button {}` is filed
under `Button`. Scan the whole selector text.

## Focus on a container, and a screen's first lazy CSS load, restyle whole subtrees too (TASK-33628.5.1, 2026-10-05)

The entry above, "A class flip restyles the flipped node's ENTIRE subtree", has two
twins that cost far more in a long Console chat.
`Widget.watch_has_focus` calls `update_node_styles()`, so a focusable scroll container
restyles every descendant each time it gains or loses focus. The Console's focus cue
was a class on the transcript region, the ancestor of every row, so each focus change
restyled every row twice. With 3,000 rows mounted, one change restyled 72,165 nodes and
held the loop for 29-37 s. Even when the restyle is scoped, a `scrollbar-color` change
inside it refreshes every descendant, because colours inherit: 7.2 s live with a
scrolled-back window. The other twin is `App._load_screen_css`: the first push of a
screen whose `CSS_PATH` is not loaded yet reads the sheet and then runs
`stylesheet.update(app)`, which restyles every node of every screen. The first Delete
receipt cost 9.3 s that way with 3,000 rows mounted.

**What to do.** Count restyles by spying on `Stylesheet.apply` around the action, and
count refreshes by spying on `Widget.refresh`; wall-clock hides which node paid. Scope a
focus restyle to the nodes its rules can actually match:
- override `watch_has_focus` to call `stylesheet.update_nodes((self,))`;
- set an ancestor's cue class with `update=False`, then update that ancestor and the
  one child its rule paints.
To skip the colour refresh, `ConsoleTranscript.watch_has_focus` shadows its own
`styles.base.refresh` with a `children=False` wrapper for the length of the update. That
leans on Textual 8 internals, so recheck it on any Textual upgrade
(`test_a_focus_change_restyles_no_transcript_row` fails if rows start refreshing again).
Read a lazy sheet before the push with `stylesheet.read()` + `reparse()`. Each shortcut
is exact only while no rule reaches another node, so pin that with a fidelity test.
`Tests/UI/test_console_long_chat_bounds.py` has both: one walks every loaded rule, the
other restyles the whole subtree and compares every computed style.

## A batch of new widgets is laid out two or three times; pace by settled layout, not by size (TASK-33628.5.1, 2026-10-09)

**Incident.** After the Console stopped mounting every later row, Undo of a
3,000-message Delete still blocked the loop for 196-316 ms (harness, 160x48),
and the same at 60 messages. It mounted one load-shaped window, 64 rows and
about 520 widgets, in one batch. Counting arrange-cache misses per
`Compositor.reflow` (wrap `textual.widget.arrange`) showed three passes of
400-466 misses each. The first was the batch's own layout. The second was the
relayout each new widget's `virtual_size` asks for: it is a `layout=True`
reactive that `_size_updated` sets inside the pass. The third came after the
transcript's scrollbar reappeared and changed its width. A relayout with
nothing new cost 6-7 ms at 1,074 widgets. Mounting a screenful per batch
helped only once each batch waited for the passes of the one before it.
Gated on "the last row has a size", one pass still re-arranged two batches
(158 ms, 299 misses), because the relayout requests were still in flight.

**What to do.** To spread a large mount, start each batch only when nothing
is pending: no `_layout_required` on the container's nodes or its screen,
seen at two checks a poll apart. A widget clears its flag only as it posts
the request to the screen, and by the next poll the screen holds it.
`Tests/UI/test_console_undo_restore_pacing.py` records the flags at each
mount; dropping the check fails it. Two more traps:
- `call_after_refresh` queues on `app.screen`, the top screen. While a modal
  is up, a transcript's callback waits on the modal's idle, not on the
  screen being laid out.
- The app's own `event_loop_stall` records start at 250 ms, so they cannot
  show a block moving from 240 to 120 ms. A thread that pings the loop with
  `call_soon_threadsafe` every 5 ms measures what input would wait, and it
  holds no frames.

## A node's `@on` handlers run BEFORE its `on_<message>` method — `event.stop()` cannot un-run either (phase C task 3, 2026-09-09)

Phase C moved 16 canvas-origin `@on` rows from `LibraryScreen` onto
`LibraryMediaCanvas`. That canvas already carried a residency gate from task 2:

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    if not self.display:          # parked off-route: refuse
        event.stop()
        event.prevent_default()
```

The gate looked like it would cover the migrated rows too. It does not, and
the reason is dispatch ORDER inside a single node. `MessagePump._get_dispatch_
methods` walks the MRO and, per class, yields that class's `_decorated_handlers`
FIRST and the naming-convention method (`on_button_pressed`) SECOND. So the
`@on`-decorated handler runs before the gate. And `event.stop()` only sets
`_stop_propagation`, which `_on_message` reads AFTER the whole dispatch loop —
it stops BUBBLING, it cannot cancel another handler on the same node.
`prevent_default()` is no better here: `_no_default_action` is only checked at
the top of each `for cls in MRO` iteration, so it skips PARENT classes, never
the rest of the current one.

Verified with a 20-line spike before designing around it (a `Vertical` with one
`@on(Button.Pressed, "#b")` and one `on_button_pressed`; the order list came
back `['decorated', 'gate']`), then reproduced on the landed code: with the
migration in place and the refusal removed, pressing Sort on the HIDDEN
resident Media canvas opened its chooser while the user was reading Notes.

**The rule:** a same-node guard implemented as `on_<message>` protects only
handlers on ANCESTOR nodes. The moment a region widget starts catching its own
messages with `@on`, the guard has to move inside those handlers — one shared
seam they all call first, not a separate method that merely runs later.

**And the trap inside the trap:** the existing pin for that gate
(`test_hidden_resident_media_canvas_does_not_process_row_presses`) stayed GREEN
through the whole hazard, because the row it presses (`.library-media-row`) was
NOT one of the migrated handlers. A guard's pin only covers the handlers that
route through the guard; migrating a handler out from under one silently
narrows what the pin proves without changing the pin's result.

## `@on` handlers must live on the ChatScreen — `Console_Modules/message.py` is a controller, not a mixin (TASK-32312, 2026-09-10)

**TASK-32312.** Wiring a `ConsoleThinkingEditRequested` event handler for the
thinking-block edit feature, I added `@on(ConsoleThinkingEditRequested)` to a method
in `UI/Console_Modules/message.py` — and the handler silently never ran (no modal, no
toast, test timeout). That module's methods are not screen methods:
`ChatScreen` constructs exactly one `ConsoleMessageController` in `__init__` (kept at
`self._message`) and delegates specific methods into it. A plain controller object is
not a Textual message pump, so `@on` tags on it are inert, and the controller has no
`query_one` — `self.query_one(...)` inside it raises `AttributeError`, which a broad
`except Exception` then swallowed into a misleading `None` return. The second trap:
my first fix probed `hasattr(console, 'on_console_thinking_edit_requested')` via an
**instance-attribute spy**, which stayed empty — Textual dispatch walks
`type(self).__mro__`, so instance-attr replacement is invisible to delivery and
"handler not called" conclusions from it are unreliable.

**What to do.** Put `@on(...)` handlers on `ChatScreen` itself
(`UI/Screens/chat_screen.py`) and delegate one line into
`self._message.<controller_method>(event)`. Inside the controller, reach the DOM and
screen through `self._screen` (`self._screen.query_one(...)`), never `self.` — the
module's own header says "never a back-door through `self.screen`". When probing
message delivery in tests, replace the handler on a subclass or count side effects
(posted modals, store writes), not via instance attributes.

## A toast belongs to the screen that is current when it FIRES, and it docks over that screen's bottom chrome (task-32266, 2026-09-11)

`App.notify()` hands the notification to `self.screen`, and Textual's
`ToastRack` is `dock: bottom; align: right bottom; layer: _toastrack`. So a
toast raised from an ASYNC completion does not land on the surface that asked
for it — it lands on whatever is current a second or two later, on top of that
surface's docked footer.

The incident: the first-run wizard's Voice step saves TTS settings by posting
`STTSSettingsSaveEvent` to the app while the user presses Next. The shared
handler announced the publication with "Settings saved successfully!". By the
time the write settled the wizard had advanced one or two steps, so the toast
painted over the Protect step's buttons, and — walking at normal speed — over
the Summary's docked exit actions, "Write your first note" included. Nobody
who read `FirstRunSetupWizard.py` would find it: the emitter is
`Event_Handlers/STTS_Events/stts_events.py`, three modules away, and the
wizard's own `notify()` calls are all unrelated. Three reviewers filed it as
"the wizard's completion toast" — the wizard never raised one.

**The rule:** a component that posts a work request to an app-level handler and
then AWAITS the result renders its own outcome; the handler's toast is
duplication that will land somewhere else. Give such requests an explicit
opt-out (`notify_outcome=False` here) rather than repositioning the rack —
scoped CSS only relocates a message that should not exist on that screen.

**Reproduction note:** the default toast timeout is 5 s, so a scripted walk
with 2.5 s between steps can easily capture the toast on one step and miss it
on the next. Reproducing the Summary case needed the step advances tightened to
~1.3 s. A single clean capture is not evidence the toast cannot reach a later
screen.

## A widget's BUNDLED_CSS cannot override an app-tier rule, and build_css.py will not put it in the screen sheet for you (task-32250, 2026-09-11)

**What happened.** The Import once review collapses a run of interchangeable
rows into one `Collapsible`, and the pager budgets a page by RENDERED rows, so
that disclosure has to be one line. `tldw_cli_modular.tcss` styles every
`Collapsible` app-wide (`min-height: 3`, a round border, a 3-row
`CollapsibleTitle`, a bottom margin) — five lines of chrome for one summary.
A `LibraryNoteImportCanvas .note-import-run { ... }` rule in the canvas's own
`BUNDLED_CSS` changed nothing, despite far higher specificity: widget-tier CSS
loses to app-tier CSS in Textual regardless of the selector.

The obvious next move — "Library rules go in the screen-owned sheet" — does
not work by hand either: `screen_agentic_library.tcss` is GENERATED, and
`check_bundle_sync.py` fails the moment you edit it. Running `build_css.py`
after adding the rule to a WIDGET's `BUNDLED_CSS` routes it to
`widget_defaults_self.tcss`, i.e. straight back to the tier that already lost.
Only a rule in the SCREEN's own `BUNDLED_CSS` reaches the screen sheet.

**What to do.** To beat an app-wide type rule from inside a widget, either move
the rule into the owning screen's `BUNDLED_CSS` and regenerate, or set the
properties as inline styles on the instance (`widget.styles.min_height = 1`),
which is the one tier above app CSS. Verify by rendering, not by reading the
selector: the first attempt here looked correct and did nothing.

---

## User-visible hotkeys live in four places, not one — sweep all of them

**TASK-32458, 2026-09-10 (nav renumbering).** Rebinding the shell-destination
F-key tail (F7–F11 → F2/F3/F4/F5/F7) and re-seating Artifacts touched the
shortcut map in `shell_destinations.py`, the label scheme in
`UI/Navigation/main_navigation.py`, and the strip order — the easy part. A
review pass initially claimed "no code UI copy teaches these keys" off a
quoted-string grep, and that was wrong: the Console settings modal teaches
"F9 Settings > Console behavior" (`Widgets/Console/console_settings_modal.py`,
asserted by `test_console_context_controls.py`), other modules ship
"...or in F9 Settings, before sending." style copy in multi-line constants the
quote-anchored regex missed, and `Docs/` carries the key in 20+ files. The
missed modal copy would have shipped a dead key in a user-facing message; it
surfaced only because a second, looser sweep (`grep -rn "F9"` minus
hex-color/review-round noise) ran during implementation.

**What to do.** A key that users are taught (nav hotkeys, F1-help entries,
footer hints) exists in up to four layers: (1) the binding, (2) label/copy
strings in Python — including multi-line constants, so grep for the bare
token, not `"quoted"` patterns, (3) tests that assert on those strings,
(4) `Docs/`. Sweep all four with one loose grep for the token and filter
noise by eye; a quote-anchored regex is not evidence of absence.

---

## The MRO walk also runs the BASE's private `_on_*` handler — and it runs LAST, so an inline effect in a subclass gets undone (task-32251, 2026-09-11)

The `super().on_mount()` lesson above is about a base body running *twice*.
The same dispatcher has a second, sharper consequence for the private
`_on_<event>` handlers Textual's own widgets use: the base implementation
runs for the event **whether or not you call `super()`**, and because
`_get_dispatch_methods` walks `self.__class__.__mro__` most-derived-first,
it runs **after** yours. If the base handler's job is to SET something your
override wants to set differently, you lose.

The incident: `PathInput` (the picker path field) needed the click that
focuses it to select the pre-filled directory, so typing an absolute path
replaces it instead of appending — the defect was a field holding
`/Users/me/Users/me/.cache/...` after a click and a type. The obvious
override was

```python
async def _on_mouse_down(self, event):
    await super()._on_mouse_down(event)
    self.action_select_all()          # <- never survives
```

and it failed the first test run with the typed text spliced in at the click
offset. `Input._on_mouse_down` sets `self.selection = Selection.cursor(...)`;
the dispatcher called it again after the override returned and collapsed the
selection to the click point. Calling `super()` was not merely redundant here,
it was misleading — deleting it changed nothing.

**The fix that works: defer past the dispatch.** `self.call_next(
self.action_select_all)` runs after every MRO handler for that message has
been invoked. `call_after_refresh` works too where a layout pass is wanted.

**Corollaries.**
- A subclass `_on_focus` / `_on_key` that only records state is safe (nothing
  to undo) and still must not call `super()`.
- `self.has_focus` inside `_on_mouse_down` is always `True` and tells you
  nothing: `Screen._forward_event` focuses a clicked widget BEFORE forwarding
  the `MouseDown` to it. So "was this the click that focused me?" has to be
  reconstructed from event ORDER — `set_focus` only posts `Focus` when focus
  actually moves, so the focusing click always arrives as `Focus` then
  `MouseDown`.
- **Necessary is not sufficient, and the first cut shipped the difference.**
  A flag armed on `Focus` and consumed by the next `MouseDown` also fires for
  a Tab focus followed much later by a deliberate click-to-place-the-caret —
  the review reproduced it: Tab in, click at offset 3, whole value selected,
  next keystroke wipes it. What separates them is the pointer: a deliberate
  click needs a `MouseMove` across the widget AFTER the focus, and a genuine
  click-to-focus cannot have one, because there the move PRECEDES the focus
  (Textual forwards `MouseMove` without touching focus; only `MouseDown`
  focuses). So disarm on `_on_mouse_move` as well as on `_on_key`.
- `Pilot.click` posts `[MouseDown, MouseUp, Click]` and **no** `MouseMove`,
  while `Pilot.mouse_down`/`mouse_up` each post one. A test that only clicks
  is therefore not exercising the terminal's own event shape — `pilot.hover`
  first, or the pointer-derived half of your logic is untested.
- `Input.select_on_focus` defaults to `True` already; the reason click-to-
  focus behaved differently from Tab-to-focus is entirely this ordering.

## A recomposed child has no width yet — measure the container that SURVIVES the recompose (task-32554, 2026-09-14)

The Import once confirmation had to render a selected folder's path whole when
the pane could hold it and middle-elide only when it could not, so the canvas
composed the line at a compact floor and widened it afterwards from the
mounted width, scheduled with `call_after_refresh` from `on_mount`,
`_after_recompose` and `on_resize`.

A widget-level pin at 190 columns went green. The live app at 235x52 still
showed `/Users/…/w4-imp…vault` — the 48-character floor — and only snapped to
the full path when the terminal was resized.

- **`refresh(recompose=True)` remounts the CHILDREN.** The `Static` the fit
  measured was a brand-new widget on every recompose, and inside the
  `call_after_refresh` that the recompose itself scheduled it was still
  unmeasured: `content_size.width == 0`. The guard for "not laid out yet"
  then returned, and nothing re-armed until a `Resize` arrived — which is why
  a manual terminal resize "fixed" it and no code path did.
- **The canvas that OWNS the child keeps its width across the child's
  recompose.** Measuring `self.content_size.width` (less the scrolling body's
  own chrome) and falling back to it whenever the child reads 0 makes the
  first pass correct.
- **A bare single-widget host cannot show this.** There the widget mounts
  once, is laid out once, and the child is measured by the time the callback
  runs. The defect only exists where the widget is remounted by a parent's
  sync — so the pin has to run on the real screen route, not only on a host
  app. Both pins are kept: the host for the copy rule, the screen for the
  measurement.

## Library Notes replays a captured focus after every canvas sync — fix the ROLE, not the timing (task-32540, 2026-09-14)

After "Select folder", Import once's confirmation pane left focus on the
stepper's back button, several Tab stops past the three actions the pane had
just offered. The obvious fix — focus the pane from the picker's dismiss
callback — did not hold, and neither did doing it from `call_after_refresh`,
nor doing it *before* the snapshot landed.

- **Every canvas-scoped Notes sync captures a portable focus IDENTITY before
  recomposing and replays it after** (`_capture_library_notes_focus_identity`
  → `_restore_library_notes_after_targeted_sync`). Whatever you set
  imperatively is overwritten by that replay, which runs last. Tracing it
  needs a `set_focus` wrapper that prints a stack — the symptom alone
  ("focus is on the back button") names neither the writer nor the ordering.
- **The replay resolves a semantic ROLE, and an unresolvable role falls back
  per phase.** Here the SELECT-phase fallback named `#note-import-add-source`
  — a button the FOLDER branch of that phase never composes — so the chain
  fell through to the back button. Registering the pane's own scroll owner as
  a role (both halves: widget id → role, role → selector) and inserting it in
  that fallback chain fixed it at the seam, with no timing to lose.
- **Corollary:** a focus fallback that names one branch's control is a latent
  defect for every other branch of the same phase. Prefer a target the phase
  always composes.

## Keep control choices outside the saved ID domain (TASK-32776, 2026-09-18)

The real Persona service accepts `none` and `auto` as IDs. The workspace Select
used those same strings for its None and automatic-create controls: no-edit Apply
cleared a saved `none` Persona, and Create replaced a saved `auto` Persona with a
new identity. Plain Enum control values fixed both without rejecting legitimate
stored IDs. Preserve those values through form snapshots and recomposition;
calling `str(Select.value)` erases that distinction and breaks control dispatch.
Real-service regressions
and four native cells verified exact saved IDs, labels and memory confirmation.

## Narrowing a Select's options orphans the value already saved (TASK-33002 final review C1, 2026-09-27)

TASK-33002.1 removed "minimal" from Settings' Reasoning effort Select for
llama.cpp, because the request drops it. Compose and sync used the narrowed list,
so a profile that had already saved "minimal" was mapped to `Select.NULL` and
showed "Inherit default". Two failures followed, and both passed the task review
and its tests:
- **Revert** still mapped against the full list and assigned `select.value =
  "minimal"` to a Select that lacked the option. Textual raises
  `InvalidSelectValueError`. The `try` caught only `QueryError`, so the error
  escaped a button handler and the app exited.
- **An unrelated Save** rebuilt the profile from the widget. NULL became "",
  and the saved value was deleted. Save had touched only Temperature.

Both reproduced only with a value saved *before* the list shrank, and no test
seeded one.

**What to do.** When a Select's option list narrows by provider, model or family:
- Route every writer (compose, sync on identity change, Revert) through ONE
  options helper.
- Have that helper keep a saved, still-legal value as a labelled option, as
  `_model_profile_enum_options` in settings_screen.py does with "minimal (not
  supported here)". Silently mapping it to NULL is the bug.
- Test the three paths that change the option list under a saved value: open
  with it saved, switch identity and then Revert, and Save an unrelated field.
  Assert the value that is written.

## `is_mounted` never goes False, and `push_screen_wait` needs a worker (Qodo review of PR #2799, 2026-09-23)

Two Textual facts that turned three "crash guard" fixes into no-ops. Both were
found only by mounting a real app; both were invisible to the `__new__`-plus-fake
unit tests that shipped with the fixes.

- **`widget.is_mounted` is sticky True.** In Textual 8.2.8 `_is_mounted` is
  assigned `False` once in `MessagePump.__init__` and `True` in exactly one
  other place (`message_pump.py:612`); nothing ever clears it. A removed widget
  and a popped screen both still report `is_mounted is True`. Measured:
  after `await widget.remove()`, `is_mounted=True`, `is_attached=False`,
  `is_running=False`. **`is_attached` is the liveness predicate** — it walks
  `_parent` to the DOM root. `SchedulesWorkbench._run_sync`'s `finally:` guard
  was entered on the very teardown it was written to skip, raised `NoMatches`,
  and was swallowed by the `except` under it. (There are ~40 more
  `is_mounted`-as-liveness reads in `FirstRunSetupWizard.py` alone; they were
  left alone as out of that PR's scope, but they are the same class.)
- **Removing a node cancels that node's workers** (`Widget._on_unmount` →
  `WorkerManager.cancel_node`), and the cancellation lands ON the await — so a
  worker's post-await DOM code never runs on the dismiss path at all. The
  corollary matters for testing: you *cannot* reproduce "dismiss the wizard
  mid-await and watch the app exit" by removing the widget. A screen's OWN
  worker is different — cancellation propagates through `finally:`, so a
  `finally:` body still runs against a detached DOM. That is the case worth a
  test.
- **`App.push_screen_wait` raises `NoActiveWorker` outside a Textual worker.**
  `VoiceCloningWindow._delete_profile` is reached from `on_button_pressed` (the
  message pump) and from `action_delete_profile` (a bare `asyncio.create_task`)
  — neither is a worker. The fix that replaced a compose-time `AttributeError`
  with `push_screen_wait` therefore still showed no dialog and deleted nothing;
  a unit test whose fake app defined `push_screen_wait` asserted the dialog's
  `message` and hid it. Use the `push_screen(dialog, callback)` form unless you
  are already inside `@work`.

**The method lesson:** a fake that lets you *set* the state under test
(`step._is_mounted = False`, `app.push_screen_wait = ...`) is asserting about a
state production may never reach. For a guard whose whole subject is the Textual
lifecycle, mount it. Mutation-check the new test too: two of the four mounted
journeys added here stayed green when `exit_on_error=False` was deleted, because
the cancellation above makes that path unreachable — the AST pin is what actually
holds that kwarg.

## A keyring read measured on macOS is not what the UI loop pays on Linux (TASK-32921/32922/32924, 2026-09-23)

A Fedora user reported random multi-second UI lag that nobody could reproduce
on the macOS dev machines. The cause that best fit: keyring reads on the UI loop,
repeated per item. Every send and every Chat visit computed a trust status for
**each installed skill**, and each one re-read a keyring entry (`load_marker`).
Server mode re-read one on **every API call**, and the Image/Video Gen panels
did up to 7 per `compose()`. The costs recorded in the code comments were macOS
numbers (11.3–18.2 ms, Keychain via ctypes). On Linux each read is a
SecretService round trip over D-Bus to gnome-keyring, and a **locked**
keyring can block on an unlock prompt. N skills meant N prompts per send.

- Treat any `keyring.get_password` reachable from a handler, timer, `compose()`
  or send path as a potential multi-second call, whatever the macOS timing says.
- Cache in the store's **read method**, the one place every caller routes
  through. Have the store's own writes clear the cache (after the write, with a
  generation bump so an in-flight read cannot re-cache the old value). Cache
  failures only briefly (2 s here), so one pass raises once but a user's Retry
  still reads fresh; `skill_trust_service.py:240-258` explains why failures
  must never be latched.
- A cached rollback marker is safe **only** because verification is an exact
  match (generation plus digest): staleness can fail closed but never accept a
  rolled-back manifest. Re-check that property before caching any other
  security anchor.

## Read the stall record's frames before theorising about lag (TASK-32920, 2026-09-23)

The first diagnosis of that Fedora report ranked six causes by reading code.
Measurement demoted two of them: the quadratic thinking-delta capture costs
0.2 ms per delta, and the per-tick store walk costs 3.7 ms at 1,000 messages.
It also missed the keyring reads entirely. This machine's own log held 42
`event_loop_stall` records, up to 9.6 s, and none could be attributed:
`active_timers` lists what was *scheduled*, not what was *running*. The
2-minute `footer-db-size-periodic` timer appeared in every record without
being the cause. Stall records now carry the loop's sampled stack: `leaf_*` is
the innermost frame, often a library; `site_*`/`caller_*` are the two deepest
tldw_chatbook frames. **Ask for those lines first.** One sample marks where
the loop was when it crossed the threshold. That pinpoints a single long call
(keyring, locked SQLite write). For a long run of short calls it is only a
representative point: the first live capture named
`Backup_Recovery.native_files.pinned_directory` during boot, which is cheap
syscalls inside a longer synchronous stretch.


## A raising `exit_on_error=False` worker surfaces as a missing widget, not an error

**TASK-33081, settings category swap, 2026-09-23.** An off-loop config
refresh added to a pane-swap path passed `reload=True` POSITIONALLY through
`asyncio.to_thread` to the keyword-only `get_image_generation_config` -- a
`TypeError` raised inside an `exit_on_error=False` worker. The worker's
`finally` still cleared the swap-pending flag, so the swap looked settled,
the panes simply never recomposed, and 16 settings tests failed with bare
`NoMatches` while captured logs showed no traceback at all. Surfacing it
required calling the worker's coroutine DIRECTLY outside the worker and
reading the raise.

**What to do.** When a worker-driven rebuild silently produces nothing,
invoke the worker's coroutine directly in a repro before reading anything
into the compose path. Treat `asyncio.to_thread(fn, <scalar>)` as a smell
for keyword-only APIs, and give test doubles the real keyword-only
signatures. Any awaited off-loop work inserted into a swap/compose chain
also needs helpers to WAIT for the swap's settle flag rather than assume a
single pause covers it.


## `Screen.dismiss()` pops the TOP screen, and the popped screen's waiter is never resolved (TASK-33622.10, 2026-09-30)

**Incident.** Making Ctrl+Q a priority binding let the quit flow push its
prompt over any open modal. A real-`TldwCli` Pilot test that then had the
COVERED modal call a bare `self.dismiss()` from its own timer -- what an
async poll or a worker callback does -- left the quit worker waiting forever:
`_quit_in_progress` stayed `True` and every later Ctrl+Q was a no-op for the
session. The same red reproduced for the Console quit prompt, a dirty form's
discard prompt and Settings' theme-leave prompt.

**Why (Textual 8.2.8).** `Screen.dismiss()` resolves ITS OWN result callback
and then calls `app.pop_screen()` unconditionally, which pops whatever is on
top -- not the caller. `App.pop_screen` calls the popped screen's
`_pop_result_callback()`, which discards the waiter without resolving it, so
`push_screen_wait` on that prompt never returns. The caller is left on top as
a zombie whose callback already fired: measured with a two-modal probe, its
SECOND `dismiss()` raises `asyncio.InvalidStateError` (the waiter's future is
already done) -- inside a timer that is an app-level exception.

**Review follow-up: the zombie is the real crash.** Two reviewer probes drove
the REAL `VideoPlayerScreen` under the quit prompt: both of its own closes ran
a bare `self.dismiss(None)` -- `_notify_and_dismiss` (reached from activation,
pump and seek failures) and the stream time box in `_refresh_status` (a
0.25 s interval). Each popped the prompt; the quit flow correctly ended as
Stay; then the player's next close -- the user's `q`, or with no user action
at all the time box's next tick -- raised `InvalidStateError` and the app
exited with code 1, skipping the approved quit cleanup. Treating the vanish
as Stay had fixed the hang and left the crash.

**What to do.** Async self-closing (timers, polls, worker completions) must
dismiss only when `self.app.screen is self` (ADR-031). That is now enforced at
the shared primitive: `SafeModalDismissMixin.dismiss` refuses (logs at debug,
returns a completed awaitable, delivers nothing) while the modal is covered or
already popped, which covers the switcher, the video player and the other
mixin modals without per-site guards. Before trusting a refusal in a primitive
121 modal classes inherit, it was measured: a temporarily instrumented refusal
branch, run over every test file naming a mixin class (and again with the
bootstrap profile forced for the files the per-test sandbox fails closed on),
recorded refusals in only the four tests written to cause them. A periodic
caller needs one more line so it does not act on a refused close every tick:
the time box returns early while covered and closes on its first tick back on
top. Plain `ModalScreen`s are not covered by that (about two in five of the
app's modal classes); for those, `await_quit_prompt` finishes the zombie's
interrupted close once its prompt vanished -- a top screen whose newest
`ResultCallback.future` is resolved (not cancelled) yet still stacked can only
be one whose pop went astray. Every prompt the quit flow owns goes through
`await_quit_prompt`, and
`Tests/Architecture/test_quit_flow_prompt_choke_point.py` fails on a
`push_screen_wait` / `wait_for_dismiss=True` in any `confirm_quit` /
`prepare_for_quit` path. Do not "fix" the hang by cancelling the orphaned
future: a late `dismiss` would then raise inside the prompt. Still open: a
plain modal that self-closes under some OTHER covering screen -- one that
background code pushed, since Ctrl+Q is the only priority app binding, so the
only key that opens a screen over a modal -- is repaired by nothing. A static
scan for plain modals that dismiss from a timer or worker found one:
`LibraryCharacterRepairDialog._apply_owned`, which dismisses when its repair
worker finishes (the quit prompt over it is handled; nothing else is).

## A name shaped like markup exits the whole app, and one escaper fits only one parser (TASK-34400, 2026-10-04)

**Incident.** A Roleplay character named `[/]` (an imported card can carry it) made the
app exit as soon as the Inspector showed it: `Static.update(f"Selected: {name}")` on a
markup-on `Static` raised `MarkupError` while the screen was drawn. TASK-32533's
keep-alive (`app_lifecycle._handle_exception`) only covers exceptions inside a widget's
message handler; render, compositor and tooltip-timer errors still go to Textual's
default, which exits. A sweep of the Roleplay code then found about 30 more sinks of
the same kind, and `[@click=app.quit]x` turned names into live click actions.

**What parses a `str` as markup on Textual 8.2.8** (all measured): `Static`/`Label`
(construct and `.update`), `Button` labels, `Select` option prompts (the current label
and the dropdown), `OptionList`/`Option` and `SelectionList` prompts, `RadioButton` and
`Checkbox` labels, `DataTable` string cells (at render), `border_title`, tooltips (the
tooltip is a `Static`), `Collapsible` titles, and `notify()` unless `markup=False`.
Literal forms: `markup=False` on an owned `Static`, `textual.content.Content(text)` or
`rich.text.Text(text)` for prompts, labels, cells and tooltips.

**The escaper trap.** `Utils.input_validation.escape_markup` is correct for Textual's
parser (`Content.from_markup`), including backslashes. It is **wrong** for Rich's
`Text.from_markup`: Rich reads `\\[` as an escaped backslash plus a live tag, so
`a\[/]b` (or an ordinary LaTeX reply `\[x^2\]`) crashed the preview transcript, which
escaped with `escape_markup` and then parsed with Rich. `textual.markup.escape` is
wrong on Textual 8.2.8 for `[/` and `[TODO] y`. Prefer building literal `Text` or
`Content` over escaping; escape only into a shared markup-on widget you do not own.

**The test trap.** A markup sink crashes only when it is drawn. A hidden widget, a
clipped dropdown label or an unopened dropdown never parses, so a test that only
"sets the value" passes on the broken code. Paint the surface (open the dropdown,
render the tooltip text through `Static.update` the way the tooltip timer does), or
assert the literal type (`Text`/`Content`) when the surface cannot be painted in the
test, and prove the test red on the unfixed code. Tests:
`Tests/UI/test_roleplay_hostile_names.py`, `Tests/UI/test_roleplay_hostile_text_surfaces.py`.

## A coroutine handed to `app.call_later` is awaited on the app pump: Enter froze every key, a click did not (TASK-33622.16, 2026-10-03)

**Incident.** Found live during TASK-33622.15: after **Enter** sent
`/generate-video` to MiniMax, the "Generate video?" confirm ignored Escape, F1
and Ctrl+Q, and so did the storage choice after it, until the paid generation
resolved; a click on **Send** did not freeze. An await-chain probe of each
pump's task in a mounted test on dev (`01a2020981`) showed the APP pump parked in
`MessagePump.on_callback` → `_send_console_message_from_visible_action` →
`_dispatch_console_command` → `_console_command_generate_video` →
`Worker.wait()` (the confirm's `push_screen_wait` worker). On the Send route
the app pump was idle and the Console's own pump was parked in
`on_button_pressed`, so the composer's Stop button was dead for the whole paid
run. TASK-33621.28 had fixed this exact freeze for the send's hook review
only. The same send also awaited every slash command inline, so the freeze came
back through `/generate-video`.

**Why (Textual 8.2.8).** Enter schedules the send with
`app.call_later(coroutine_function)`, and `MessagePump.on_callback` AWAITS a
coroutine callback on the pump it was posted to. That is the app pump, which
dispatches every key. A `Button.Pressed` handler is awaited on the screen's
pump instead. So "click works, Enter freezes" means the send awaited
something user-paced.

**What to do.** A send path reached from a key must not await anything with a
user-paced lifetime (a modal, a remote job). Hand it to a worker, as
`UI/Console_Modules/command_handoff.py` does for every slash command. When you
fix "pump parked by awaiting X", list everything else that path awaits. Second
trap, found live on the fix build: Ctrl+Q under the confirm quit the app, and
shutdown cancelled the confirm's waiting worker. `Worker.wait()` then raised
`WorkerCancelled` through the command's own worker, and that showed up as an
`unhandled_exception` on the way out. A worker that waits on a modal must
treat `WorkerCancelled` as "over", not "broken". Test it with keys delivered
the way the driver does (`_key`), and use a bounded `_pump_runs` poll on BOTH
pumps (`Tests/UI/test_console_video_send_freeze.py`). Do not use Pilot here:
its idle wait never returns while a pump is parked.

**Third trap: unfreezing a flow makes its "nobody can act now" code
reachable (checkpoint review of the same fix).** Once the hand-off kept the
Console live during a generation, the user could type, switch chats or press
Stop -- and `/generate-video`'s failure path, written when none of that was
possible, ran `composer.clear_draft()` then pasted the saved command back. A
review repro on the fix build (Enter the command → Generate → type "what about
a sailboat?" → Stop) wiped the typed text with no undo, and after a chat switch
it wiped the OTHER chat's draft, because every Console chat shares one
composer. The pasted-back command was also unusable: a draft holding a paste is
never parsed as a command, so Enter sent it to the model as chat. Five mounted
tests went red on exactly that. `/generate-image` had the same code. The fix
(`UI/Console_Modules/command_draft.py`): take only the revision the send
captured (`commit_captured_draft`), and put it back only into the same draft
scope while that is still empty. A chat switch, a load or another send always
advances the composer's draft generation; typing does not. Restore with
`restore_stashed_draft`, which brings back the original segments, so the draft
is not marked as a paste. **When a fix lets the user act during something that
used to block them, grep that flow for every save/clear/restore of shared UI
state and ask what happens if they typed in between.**

**Fourth trap: the same holds for "the active chat", in every handler, not
just the one you fixed (PR #3006 review).** Handing every slash command to a
worker made all of them run with the Console live, and most handlers resolve
"the active session" or clear "the composer" after an await -- written when
the send parked the pump, so nothing could change in between. A mounted repro:
`/system Terse`, prompt search parked, switch to a new chat, release -- the
prompt applied to the NEW chat and wiped its typed draft. An audit of all 17
handlers found four more that answered after an await (`/doctor`, `/skills`,
`/fewer-permission-prompts`, `/stream-video`), and live, `/stream-video`
against a loopback server holding the request posted its error into the chat
switched to. The fix binds the command to its origin in the worker
(`command_handoff.COMMAND_ORIGIN`): refuse at start and after an await when
that chat no longer shows, and post awaited answers with an explicit
`session_id`. **Making a call asynchronous changes the contract of everything
it calls: list each handler's awaits and what it re-reads after them.**

---

## A handler that awaits the removal of its own ancestor never returns -- and `asyncio.wait_for` cannot get you out (TASK-34000.4, 2026-10-04)

**Incident.** Library ▸ Media ▸ "Export…" froze the whole app (review finding
L-01): no repaint, no key read, Ctrl+Q dead, 0% CPU, no traceback anywhere.
`LibraryMediaCanvas` owned the `@on` row for the button and was the one
`async` forwarder of its sixteen, so the press ran on the CANVAS's pump:
canvas handler → controller → `_open_library_export_canvas` →
`_apply_library_open_item_surface` → `await self.recompose()` on the screen.
An await-chain probe of every pump's task on the reproduced freeze showed two
pumps parked on the same removal, not one: the canvas at `recompose:1713`,
and the APP pump in `_flush_next_callbacks → AwaitRemove.__call__`, because
`App._prune` always hands the removal to `app.call_next` as well. The app
pump reads every key, which is why Ctrl+Q died; `Screen.recompose` also holds
`App.batch_update` across the removal, which is why nothing painted.

**Why (Textual 8.2.8).** `remove_children` / `recompose` return an
`AwaitRemove` over the pump tasks of the removed ROOTS, leaving out only the
current task. Each removed widget, as its loop exits, gathers its CHILDREN's
pump tasks (`Widget._message_loop_exit`), with no timeout. So code running on
the pump of a widget BELOW a removed root waits for the root, the root waits
for its children, and the chain ends at the pump doing the waiting.

**Which awaits hang -- measured, one process per shape, against a plain
`Screen`.** Never landed: a handler on a descendant awaiting the screen's
recompose; the same coroutine via the descendant's own `call_later`; a
widget inside an outgoing child awaiting `host.remove_children`. Completed:
the removed ROOT awaiting its own removal; `screen.call_next` /
`screen.call_later` / `screen.call_after_refresh`; the descendant's own
`call_after_refresh` (Textual runs it on the screen's task);
`refresh(recompose=True)`; a screen-owned worker. A worker owned by a removed
widget is a third outcome: it is cancelled when its owner unmounts, between
the removal and the mount, and leaves the surface empty.

**What to do.** A region widget forwards an event and returns; it never
awaits the screen or the controller it forwards to. The controller hands a
surface swap to the screen's pump (`self.call_next(...)`, already bound to
the screen) or to a screen-owned worker. `BaseAppScreen.recompose` and
`BaseAppScreen.children_safe_to_await_removing` now refuse the hanging shape
with `SurfaceSwapSelfAwaitError` before anything is torn down
(`UI/Navigation/surface_swap_guard.py`), and the Library wires the second at
every awaited canvas-host child removal (projection, repair loop, snapshot
reconcile, browse route swap); they do NOT cover the worker shape. A seam
that already has a safe fallback may take it for a refusal -- the Library's
open-surface seam logs the refusal at ERROR by widget name and schedules the
whole-screen refresh on the screen's own pump, so the user still gets the
surface; letting the error reach the handler instead had Textual close the
offending canvas's pump, which blanked the Items pane behind the keep-alive
toast. If you write such a check yourself, compare TASKS (`node._task is
asyncio.current_task()`), not `textual._context.active_message_pump`: that
variable names the pump that SCHEDULED the code, and it read "the canvas" on
two of the shapes that complete.

**The test trap, which cost a ten-minute hung run.** A red freeze test hangs
the suite in three places unless you plan for each: Pilot's idle wait stops
on the parked pump; `run_test`'s exit waits for the pump tasks; and
`asyncio.wait_for` cannot cancel its way out -- `Task.cancel()` cancels the
awaited future, a `gather` cancels its children, each child is a pump task
awaiting the next `gather`, and around a wait CYCLE that recursion ends in
`RecursionError` inside the timeout callback while the await stays parked.
`Tests/UI/pump_probe.py` has the working recipe: poll on wall clock, read the
pump tasks' `cr_await` chains to say who is parked and where (the failure
message then names the handler), cut the cycle at one edge on a red run
(`unpark`), and keep `@pytest.mark.timeout` as the outer bound -- SIGALRM is
the only one that holds whatever the loop is doing.

## `self.log` raises NoActiveAppError once the app has exited — shutdown paths must use the module logger

**PR #3016 review round, 2026-10-06.** Surfacing an ignored retention result
in `LibraryFileNotesWorkspace.shutdown()` (Qodo finding 10) added
`self.log.warning(...)` on the result-failure path. Sixteen workspace tests
that end with `await workspace.shutdown()` AFTER `run_test()` exits (e.g.
`test_folder_files_save_copy_keeps_editor_editable`) failed with
`textual._context.NoActiveAppError` from `message_pump.py`: the DOM logger
resolves the `active_app` ContextVar, which no longer exists once the app
unmounted. The pre-existing exception branch had the same latent hazard but
only fired on thrown errors; a result-level failure (`replica-error` from
`replica=None`) is routine, so the trap fired on every such test. Confirmed
by file-swap A/B (HEAD pass, fixed fail, bisected to the workspace file).
Fixed by logging through the module-level loguru `logger`, which needs no
app context. **What to do:** any code that can run during or after app
teardown (shutdown, workers outliving the app, `call_from_thread` stragglers)
must not use `self.log` / DOM logging — use the module logger, and treat
"16 tests suddenly fail with NoActiveAppError" as this exact signature.

---

## `width: auto` on a card container renders blank, and screenshot words
are non-breaking (issue #365, 2026-10-07)

Bisected a blank `ModalScreen` card property by property: every declaration
was innocent except `width: auto` on the card `Container` — children
measured 0×0 and the whole dialog exported as empty space, while the skip
and auto-dismiss tests PASSED vacuously (`app.screen is not dialog` was
true because nothing rendered). The repo's working `ConfirmationDialog`
sizes its card with a FIXED `width` (`width: 60; height: auto`) — auto
width on these containers is the trap, not the norm. Separately,
`App.export_screenshot()` emits styled words as separate SVG `<text>`
spans joined by `&#160;` (non-breaking space), so multi-word copy
assertions like `"Session summary" in svg` fail on fully-rendered text.
**Card modals: fixed width. Screenshot asserts: normalize `&#160;`/`\xa0`
to spaces first, and pair any screen-popped assertion with a render
assertion so it cannot pass vacuously.**
