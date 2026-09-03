# Library screen decomposition — the recipe

Source documents (read these first; this doc assembles their operative
rules into one checklist-shaped reference for whoever runs the next
subsystem series):

- Plan: `Docs/superpowers/plans/2026-09-01-library-decomposition-foundation.md`
- Spec: `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md`
- Parent doctrine: `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
  (approved rev 3) and `DESIGN.md` §7 — the One Rule, the One Home Rule
  (`UI/Library_Modules/`), the Six Migration Rules, and the dependency-naming
  canon this doc cites by example below.

Everything here is Library-specific application of that doctrine, not a
restatement of it.

## 1. The per-subsystem PR series

Each of the eleven subsystems below (see §8) ships as a small series of
pure-move PRs, always in this order:

1. **State PR** — a `Library<Subsystem>State` dataclass in
   `UI/Library_Modules/library_<subsystem>_state.py` holding every field the
   subsystem exclusively owns, moved verbatim (identical defaults; computed
   defaults become constructor arguments so `__init__` evaluation order is
   preserved). The screen keeps every original attribute name alive as a
   generated getter/setter `@property` shim, between sentinel comments,
   pointing at `self._<subsystem>_state.<field>`.
2. **Controller PR(s)** — one controller per PR (Six Migration Rules, rule
   1; never batch two subsystems, never batch two controllers of the same
   subsystem into one PR). Each controller owns a cluster of the subsystem's
   moved methods under their original names; the screen keeps one-line
   delegators for every externally-referenced name (`@on` handlers,
   `action_*`, anything a test reaches directly).
3. **Cleanup PR** — the one PR type allowed to edit tests. Deletes the
   sentinel-wrapped shim block, retargets remaining screen-side references
   to the state object directly, deletes delegators nothing external still
   reaches (prove deadness with `grep -rn "<delegator>" Tests/ tldw_chatbook/`
   per delegator before deleting it), retargets test attribute paths and
   patch targets with **assertions kept byte-for-byte**, and lowers the
   `_BUDGETS` ratchet row (§6) to the post-cleanup measurement.

### The byte-for-byte canon

Moved method bodies are **never edited** — not even to retarget a call or an
attribute. Every name a moved body references that is not the controller's
own state is rebound in the constructor, under the *same name* the body
already used, so the body reads identically before and after the move. Two
binding kinds only (a third kind — "reach through `screen` because the
target has no controller of its own yet" — is retired, not available to
this plan):

1. **Framework services** (`run_worker`, `post_message`, `set_timer`,
   `set_interval`, `is_mounted`, `call_after_refresh`, …) are live-read from
   the screen via `@property` on every access — never snapshotted. A value
   captured once at construction goes stale the instant a test replaces the
   attribute on the screen instance afterward.
2. **Everything else** the body depends on that is not its own state is a
   **named constructor dependency** — a controller's dependencies are its
   signature, discoverable by reading the constructor, not by reading every
   `@property` on the class. Each is a callable the *caller* (the screen)
   constructs to close over the screen's own attribute lookup at *call*
   time, not at construction time — this is exactly why a monkeypatched
   name keeps working after the move: the lambda re-reads `self.<name>`
   (or the successor controller) on every invocation.

The canonical worked example is `ConsoleDictationController.__init__`
(`tldw_chatbook/UI/Console_Modules/dictation.py:659`), whose own docstring
states this rule in the words above. Read it before writing the first
Library controller constructor — every Library controller's constructor
should be recognizable as the same shape.

## 2. Field ownership — the authoritative script, and the ≥2-subsystems rule

**Do not hand-list which fields belong to a subsystem.** Compute exclusive
ownership mechanically, per subsystem, with this script (verbatim from the
plan's Task 6 Step 1 — swap the `conv_fields`/`OTHER_SUBSYSTEM` prefixes for
the subsystem in hand):

```python
.venv/bin/python - <<'PY'
import ast
from collections import defaultdict
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
def attrs(m, store_only=False):
    for node in ast.walk(m):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if not store_only or isinstance(node.ctx, ast.Store): yield node.attr
init = next(m for m in methods if m.name == "__init__")
fields = set(attrs(init, store_only=True))
conv_fields = {f for f in fields if f.startswith(("_library_conversation", "_conversation"))}
OTHER_SUBSYSTEM = ("_library_notes","_library_media","_library_prompt","_library_skill","_library_ingest","_library_export","_library_collections","_library_rag")
for f in sorted(conv_fields):
    users = [m.name for m in methods if m.name != "__init__" and f in set(attrs(m))]
    non_conv = [u for u in users if "conversation" not in u]
    tagged = [f"{u} ({next((p.lstrip('_') for p in OTHER_SUBSYSTEM if p in u), 'shell/plumbing')})" for u in non_conv]
    print(f"{f}: non-conversation users={tagged or 'NONE'}")
PY
```

Classification of the output:

- `NONE` → moves into the subsystem's state object.
- Non-subsystem users that are **shell/plumbing methods** (rail switch,
  snapshot apply, shell-state build) → still moves; shims keep them working
  and that subsystem's cleanup PR retargets them.
- Non-subsystem users belonging to **another subsystem** (name matches
  another subsystem's prefix) → **stays on the screen as shared shell
  state**, accessed via a named dependency callable, never forced into a
  subsystem state object. Record each such decision in this doc's
  per-subsystem table entry when that subsystem's series lands.

**The ≥2-subsystems rule:** a field referenced by two or more subsystems is
shared shell state by definition and is never moved by an extraction PR.
Known examples from the spec: `_library_selected_row_id` (226 refs),
`_library_lifecycle` (83), `_library_snapshot_state_generation` (35),
`_pending_library_source_open` (29).

## 3. Monkeypatch-name routing — do not move a patched name until cleanup

**Any name a test monkeypatches on `LibraryScreen` keeps its whole call
graph routed through the screen** until that subsystem's cleanup PR
retargets the tests. This prevents the "monkeypatch bypass breaks tests
inside a pure move" failure: a moved internal call silently starts
resolving free names through its *new* module's globals instead of the
screen's, so a test's `monkeypatch.setattr(LibraryScreen, "<name>", ...)`
stops reaching the call it used to intercept.

**The known list is four names, not three** (PR 0a's execution corrected
the plan's draft "trio" framing once the tests were actually read):
`_list_local_source_snapshot`, `_refresh_local_source_snapshot`,
`_apply_local_source_snapshot`, `_refresh_library_note_detail`. These stay
screen-routed for the reason the spec gives — the trio (plus the note-detail
refresh) is shared shell infrastructure feeding notes+media+conversations
counts, has dozens of internal call sites, and is monkeypatched directly
across `Tests/UI/test_library_entry_compose_once.py`,
`Tests/UI/test_library_prompts_canvas.py`, and `Tests/UI/test_library_shell.py`.

**A second, distinct failure mode surfaced by PR 0a: module-globals
coupling.** `_INGEST_OPTIONS_CACHE_ATTR`, `_read_library_ingest_options_from_config`,
and `_library_ingest_options_for` are module-level `FunctionDef`s that read
each other via Python's ordinary free-variable resolution — which binds to
the *defining* module's `__globals__`, fixed at definition time, not to
wherever the function is later re-exported. `_library_ingest_options_for`
internally calls `_read_library_ingest_options_from_config`; several tests
(`Tests/UI/test_library_ingest_options_cache.py`,
`Tests/UI/test_library_screen.py::test_load_ingest_options_from_config`, the
three `test_task_33*_options_round_trip_persisted_config` tests) patch
`get_cli_setting` / `_read_library_ingest_options_from_config` **on the
`library_screen` module object**. While both functions live in
`library_screen.py` the patch reaches the internal call; move only
`_library_ingest_options_for` (or move both to a *different* module and
leave a bare re-export) and the internal call keeps resolving the free name
through its own module's globals — the patch is silently bypassed and 5
tests fail deterministically. The fix is not a shim: both functions had to
stay together, in `library_screen.py`, permanently (see
`tldw_chatbook/UI/Screens/library_screen.py:755` and
`tldw_chatbook/UI/Library_Modules/screen_helpers.py`'s module docstring for
the full trace).

**The general rule this generalizes to:** before moving any function (not
just a method — this bit module-level `FunctionDef`s, which the per-method
"@on stays, everything else can move" mental model does not cover), `grep
-rn "<name>" Tests/` for direct monkeypatching of it or of any name it
reads from module globals. A hit on either means: keep the whole
call-graph on its current module until that surface's cleanup PR proves the
patches have been retargeted.

**Three more bypass shapes, found by the conversations exemplar (§11 has
the full incident write-ups): unbound fake-`self` calls, an instance-
attribute monkeypatch on a real object, and — the one that isn't a test at
all — a shared helper resolving a subsystem's attribute name at RUNTIME
via `getattr(screen, f"...")`/a dict-of-name-strings. None of these are
"monkeypatching a name" in the literal sense this section's `grep`
recommends, so that grep alone will not find them. §11's per-shape fix
recipe (leave the method real, retarget the fixture to match, or route the
dynamic lookup through `operator.attrgetter`) is the accommodation for
each.**

## 4. The transform whitelist

An extraction PR may contain **only**:

- Verbatim block moves (method/function bodies byte-for-byte, per §1's
  canon).
- Import-path changes.
- Constructor/property bindings — the two binding kinds in §1, plus
  generated controller-local `@property` accessors for the subsystem's own
  state fields (same generator shape as the state-PR shim generator, §1
  step 1, applied to the controller instead of the screen).
- Screen-side delegator one-liners for externally-referenced names
  (`@on`/`action_*`/anything a test reaches directly) — the `@on(...)`
  decorator line stays on the screen, copied verbatim.

Nothing else. No renames, no logic edits, no cleanups, no "while I'm here"
fixes — those land as separate, attributable changes. Receiver
normalization (direct `self._state.` access, dropping screen-routed
same-subsystem hops) is explicitly deferred to that subsystem's cleanup PR;
it never happens in a move PR.

This whitelist supersedes an earlier draft idea from this plan's own
drafting — a bespoke "receiver-rewrite transform whitelist" that would have
permitted rewriting internal call sites during the move itself. It was
dropped in favor of the stricter byte-for-byte body discipline the
dictation extraction (§1) demonstrated: bodies unedited, names rebound only
in the constructor. See the spec's "Relationship to the parent doctrine —
deltas, declared" section for the full list of drafting ideas that lost to
doctrine precedent.

## 5. Rollback, not fix-forward

A landed extraction implicated in a regression is **reverted**, not fixed
forward. Pure moves revert cleanly, and single-candidate attribution — "this
one commit changed nothing but where code lives, so it is the only
candidate if behaviour changed" — is the property the pure-move policy
exists to buy. Fixing forward reintroduces exactly the ambiguity the policy
was designed to remove.

## 6. Measure after final rebase, lower budgets in the landing PR

This plan's Global Constraints state it directly: **this file churns ~14
commits/day.** Never trust a line-count or method-count number carried over
from an earlier point in planning or review — every task locates its
material and re-measures with the provided scripts at execution time.
**Rebase onto latest `origin/dev` immediately before each PR's final
measurement**; budgets (`_BUDGETS` in
`Tests/Architecture/test_screen_size_ratchet.py`) are measured *after* that
rebase and lowered *in the landing PR itself*, never in a follow-up. Console
wave 3 landed red twice from stale-base numbers before this rule was
adopted — see that file's own module docstring for the incident this
mechanism exists to prevent.

## 7. Sweep evidence — xdist + paired baseline, not the literal CI command

The full regression net for a Library change is `Tests/UI -k "library"`
(4,200+ tests). **The literal single-process command,
`.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q`, is
CI's job, not this recipe's per-task evidence requirement** — a
single-process attempt of the full sweep runs for roughly an hour or more
(observed: ~59 minutes before hitting a session's own time budget, still
genuinely executing the whole way per stack-sample profiling, not hung).
Waiting on that per task makes the recipe unusable for the fast, frequent
PRs this plan calls for.

**The accepted per-task evidence is an xdist sweep paired with a
pristine-baseline comparison**, run at execution time:

```bash
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal
```

(the exact invocation used for PR 0a's Task 1 evidence; `-n 8 --dist
worksteal` is already available in the worktree venv). Procedure:

1. Run that command on your branch; capture the pass/fail counts and the
   set of failing test names.
2. Run the **identical** command against the pristine merge base (e.g.
   `git stash -u` back to a clean tree at the base commit, or check out
   `origin/dev` in a scratch worktree) — same `-n`/`--dist` config, so any
   parallelization-only flakiness is present in both runs equally.
3. Diff the two failure-name sets. `pytest-xdist` itself introduces
   real, non-deterministic ordering/parallelization flakiness in this
   suite (CSS-geometry and terminal-size-sensitive Pilot tests are not
   uniformly safe under heavy parallel load) — expect a handful of tests to
   flip in *both* directions between any two xdist runs, baseline included.
   **Only failures unique to your branch's run (absent from the baseline
   run) count against the task.** A failure present in both runs, or only
   in the baseline run, is pre-existing or run-to-run noise, not evidence
   of a regression.
4. For each failure unique to your branch, re-run it directly,
   single-process, in isolation and combined with the other unique
   failures, before concluding it is a real regression rather than
   xdist-specific ordering/shared-state flakiness that happened to land on
   your branch's run and not the baseline's.

This is the same technique and the same command PR 0a's Task 1 used to
surface and confirm a real, deterministic regression (7 tests, 100%
reproducible in every combination) against a backdrop of ~330+ ordinary
xdist-noise failures neither run's raw pass/fail count could have
distinguished on its own. Report both counts and the diffed unique-failure
list as the task's evidence; a CI run of the literal single-process command
remains the authoritative confirmation once time permits, but is not
required per-task.

### Documented pre-existing failures (do not re-derive these)

Tests confirmed, by at least one Library-decomposition task, to fail
identically on a pristine baseline (`git stash -u` to the pre-task tree)
and therefore not attributable to any extraction/cleanup PR. Check this
list before spending time re-proving one of these is pre-existing; add to
it (with the task that found it) rather than letting the next series
rediscover the same red from scratch.

- `Tests/UI/test_library_content_hub.py::test_library_conversations_empty_state_is_honest_and_blocks_actions`
- `Tests/UI/test_library_shell.py::test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-conversations-#library-conversations-reader-shell]`
- `Tests/UI/test_library_shell.py::test_library_conversations_reentry_preserves_applied_page_and_query`
- `Tests/UI/test_library_shell.py::test_library_conversations_reentry_does_not_load_when_dirty_editor_vetoes`
  (all four found by Task 7, reconfirmed by Tasks 8 and 9)
- `Tests/UI/test_library_selection_updates.py::test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`
  (found by Task 9; confirmed identical on both HEAD and a pristine
  baseline via `git stash -u` + rerun. **Not selectable by the
  `-k "conversation and library"` filter** — its name contains neither
  "conversation" nor "conversations" — and invisible to the full xdist
  sweep's paired-baseline *diff* specifically because it fails on both
  sides equally, so it's absorbed into the shared ~330+-failure backdrop
  rather than surfaced as a unique-to-branch or unique-to-baseline name.
  Only a direct per-file run surfaces it; run each retargeted test file
  individually, not just the aggregate sweeps, and don't summarize that
  check as "all green" without checking every file's own result.)
- `Tests/Architecture/test_screen_size_ratchet.py`'s two `chat_screen.py`-scoped
  rows (`test_screen_does_not_grow_past_its_budget[chat_screen.py]`,
  `test_task_22507_4_does_not_worsen_chat_screen_base`) — concurrent,
  unrelated `chat_screen.py` growth from other work on `dev`; reconfirmed
  pre-existing by every task in this series via the same `git stash -u`
  method.
- Wave-2 Task 2 (export state PR) found 14 more, all reconfirmed identical
  (same 14, no more/fewer) on a `git stash -u` baseline of the pre-task
  tree via a direct node-id rerun (not the xdist sweep):
  `Tests/UI/test_library_export_cancel.py::
  test_cancel_apply_current_run_sets_cancelled_status` (an unbound-fake
  `SimpleNamespace` call into `_apply_library_export_cancelled` hits
  `AttributeError: 'types.SimpleNamespace' object has no attribute
  '_sync_library_emergency_guard_presentation'` — that call was added to
  the method by other work after this fake's flat attribute list was last
  updated); `Tests/UI/test_library_honesty_accessibility.py::
  test_escape_works_on_export_and_staging_canvases`,
  `test_rail_entry_to_export_after_media_origin_does_not_claim_media`;
  `Tests/UI/test_library_per_click_recompose_t21116.py::
  test_export_open_from_media_is_canvas_scoped`; six
  `Tests/UI/test_library_shell.py` tests (the three
  `test_library_shell_note_export_pushes_file_save_dialog` parametrizations,
  `test_library_shell_note_write_export_file_writes_expected_content`,
  `test_library_shell_note_write_export_file_rejects_invalid_path`,
  `test_library_shell_notes_selected_export_opens_exact_selection_scope`)
  and four `test_library_note_keyboard_capability_matrix` parametrizations
  (`multi_select_export`, `export_markdown_text` × two terminal sizes) —
  these all time out waiting for `#library-notes-row-0` (or a sibling
  selector) to mount within its 30s budget, consistent with machine-load-
  sensitive DOM-mount flakiness rather than a logic bug, but confirmed
  100% reproducible (same set, both runs) rather than randomly flipping.
- Wave-2 Task 3 (export controller PR) found 1 more, confirmed identical
  via `git stash -u` (same root cause as the `test_cancel_apply_current_
  run_sets_cancelled_status` row above -- both trip on the SAME stale-fake
  `_sync_library_emergency_guard_presentation` gap, in different completion
  handlers): `Tests/Library/test_library_export_roundtrip.py::
  test_library_export_success_records_a_durable_receipt_with_the_real_path`
  -- its unbound-`SimpleNamespace` fake for `_apply_library_export_success`
  never gained a `_sync_library_emergency_guard_presentation` entry, and
  the method's own `run_id`-staleness guard does not return early before
  reaching that call for this test's inputs (`run_id=1` matches the fake's
  `_library_export_run_id=1`), so it raises `AttributeError` regardless of
  whether the method is a delegator or its original body -- this task's
  own move neither causes nor fixes it. Also newly found by this task,
  affecting `Tests/Library/` specifically (a directory this recipe's own
  `-k "export and library"` per-task check had not previously needed to
  cover, since Task 2 found its 14 entirely within `Tests/UI/`): the
  "unbound fake-self" bypass shape reaches at a much larger scale here
  than the conversations exemplar's own 5-method precedent -- 9 Export
  methods, across 6 test files, 4 of them in `Tests/Library/` rather than
  `Tests/UI/`. See `library_export_controller.py`'s module docstring for
  the full per-name accounting; future subsystems' controller-PR sweeps
  should widen their `-k` search beyond `Tests/UI` before trusting a clean
  result, and should expect this shape to scale with how much a
  subsystem's tests favor unbound-`SimpleNamespace`/`Mock` unit-style
  calls over full-harness ones.
- Wave-2 Task 4 (export cleanup PR) found 1 more, confirmed identical on a
  `git stash -u` pristine baseline of the pre-task tree (same command,
  isolated to the 3 files that carried it):
  `Tests/UI/test_library_choice_strips.py::
  test_media_type_strip_works_in_both_layouts` -- times out waiting for
  the narrow-layout `library-notes-compact` CSS class within its 15s
  budget; reproduces identically on both trees, consistent with the
  machine-load-sensitive DOM-mount-timing flakiness this file's sibling
  rows already describe, not a logic regression. Task 4's own field-
  retarget pass touches none of this test's assertions or the media-type
  choice-strip code path.
- Wave-2 Task 5 (collections state PR) found 3 more, confirmed identical
  on a `git stash -u` pristine baseline of the pre-task tree via a direct
  node-id rerun of `-k "collection and library"` (Tests/UI + Tests/Library,
  361 passed/3 failed on both branch and baseline):
  `Tests/UI/test_library_shell.py::
  test_library_starter_deep_link_opens_hidden_collection_or_note_route`
  (`WorkerFailed: AttributeError("'types.SimpleNamespace' object has no
  attribute 'active_authority'")` -- an unbound-fake-scope-service test
  fixture missing a newly-added attribute the real service now carries);
  `Tests/UI/test_library_shell.py::
  test_library_landing_continue_receipt_accepts_only_authoritative_source_scopes[browse-collections-expected_scope4]`
  (`state["library_continue_receipt"]` is `None` instead of the expected
  receipt dict); `Tests/Library/test_library_collections_service.py::
  test_get_library_collection_supported_types_round_trip_public_ids`
  (`member["source_ref"]` is a real ref string instead of the expected
  `None`) -- none touch a field or line this task's move edited; this
  task's own diff is a pure field relocation plus a 2-line `_BUDGETS`
  update.
- Wave-2 Task 6 (collections controller PR) reconfirmed the same 3
  Task-5 failures identical via the narrow `-k "collection and library"`
  check (Tests/UI + Tests/Library), plus a 4th, flip-flopping name not
  attributable to either tree: `Tests/UI/test_library_prompt_
  collections.py::test_library_screen_membership_load_retry_and_apply_
  retry_are_distinct` failed on the PRISTINE baseline run but not the
  branch run of that same narrow check (360p/4f baseline vs 361p/3f
  branch) -- a Prompts "Prompt Collections" test (the unrelated feature
  this task's own 3 excluded methods belong to), confirmed pure noise by
  a second, independent reproduction in the full xdist sweep's own
  branch-unique set. **New forward note: running the branch and baseline
  full xdist sweeps CONCURRENTLY (two 8-worker invocations sharing one
  machine) measurably amplifies flakiness beyond this recipe's historical
  ~330-340 backdrop** -- this task's own concurrent run measured 349
  failed/3890 passed (branch) vs 344 failed/3895 passed (baseline), both
  above the documented range, with 12 branch-unique and 7 baseline-unique
  names (vs the export/collections-state series' own 2-and-9-ish norms).
  11 of the 12 branch-unique names passed cleanly on a single-process
  combined re-run; the 1 that reproduced,
  `Tests/UI/test_library_media_reader_traversal_t22207.py::
  test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`,
  passed in TRUE isolation and reproduced identically on the PRISTINE
  baseline under the SAME combined-invocation conditions (a shared-state/
  ordering sensitivity to which OTHER tests ran earlier in the process,
  not to which code version is loaded) -- confirmed pre-existing, not a
  regression, and not Collections-related (Media reader cluster; this
  task's diff touches zero Media-reader code).
  **Correction (fix round 1, post-review): one of the 11 that passed
  cleanly, `Tests/UI/test_library_adaptive_reader_closeout.py::
  test_closeout_single_app_route_cycle`, is NOT unrelated to Collections
  -- its own `DESTINATION_CONTRACT` includes a `"collections"` entry and
  the test cycles every destination, which for "collections" traverses
  the shared reader-shell dispatcher calling two of this task's own moved
  methods (`_sync_library_collections_reader_layout_from_shell`,
  `_mirror_library_collections_reader_preference`).** It was re-verified
  with the strongest available method for a single test: temporarily
  `git checkout bca923b4c -- tldw_chatbook` inside the task's own
  worktree (confirmed via `git diff --stat` that only `library_screen.py`
  changed, i.e. a clean pre-controller-move state), the test run in true
  isolation there, then `git checkout HEAD -- tldw_chatbook` restored
  (confirmed `git status` clean after) -- **passes identically at HEAD
  and at base**, closing the question directly rather than by the flawed
  "the test's name doesn't mention Collections" inference the task's own
  first-draft report used. Any future subsystem's sweep-triage should
  check a failing/flaky test's OWN fixture/contract content (not just its
  name or file) before asserting it is unrelated to the subsystem being
  moved -- a destination-cycling test like this one will touch every
  subsystem's own dispatchers by design. **Future tasks should
  prefer running the branch and baseline full sweeps SEQUENTIALLY, not
  concurrently, when machine time allows** -- the concurrent shortcut
  used here to save wall-clock time cost real investigation effort this
  task's own report accounts for in full.

## 8. Subsystem order (spec, "Order of work")

Sequenced cold-to-hot so the conversations exemplar never fights rebases,
and hot subsystems migrate in short, fast series once the recipe above is
rehearsed. Churn = commits touching `library_screen.py` in the trailing 30
days whose subjects name the subsystem (measured 2026-09-01):

| Order | Subsystem | Churn | Notes |
|---|---|---|---|
| 1 | **conversations** (exemplar) — **complete** (Tasks 6–9) | 10 | 68 methods, 19 fields (2026-09-01 estimate); see §11 for the series' actual, as-landed numbers |
| 2 | export | 3 | recipe rehearsal |
| 2 | collections | 6 | recipe rehearsal |
| 2 | search | 6 | recipe rehearsal |
| 3 | skills | 15 | |
| 3 | RAG / onboarding plumbing | 16 | |
| 3 | ingest | 23 | |
| 4 | prompts | 41 | |
| 4 | media | 55 | |
| 4 | notes | 72 | most scarred; its sync controller (`canvas_sync.py`) already lives in `UI/Library_Modules/` from PR 0a |
| 5 | final shell pass | — | residual focus/lifecycle plumbing, delegator table tidy, `compose_content` reduced to the region-yielding skeleton |

Roughly 35–50 small PRs total; every intermediate state ships (no feature
freeze, per the plan's Global Constraints — never two subsystems'
extraction PRs in flight at once).

Phase C (region ownership — moving canvas-origin `@on` handlers and state
into the already-existing canvas widgets) is a separate, later, explicitly
behaviour-changing series per subsystem, gated on that subsystem's phase-A
series being fully landed including cleanup, dense mounted coverage, and a
concrete motivating change. **First motivated candidates: media and
notes**, motivated by the measured 139–380 ms rail-mode-switch main-thread
freeze (§9's probe is that fix's before/after acceptance evidence). See the
spec's "Phase C — region ownership" section; out of scope for this recipe's
pure-move PRs.

## 9. Probe usage — before/after evidence

`Helper_Scripts/library_click_probe.py` boots the real `LibraryScreen`
headless, clicks through the rail modes, and reports per-click settle time,
max main-thread gap, recompose/full-update counts, and widget mount/removal
counts. Headless numbers exclude terminal-write cost (paint-to-terminal
bytes are not produced without a real terminal) — this is a main-thread
compute-and-DOM instrument, not end-to-end latency; that framing is honest
noise-tolerance, not a limitation to work around.

```bash
.venv/bin/python Helper_Scripts/library_click_probe.py
```

Run it and keep the report table before starting a controller-move PR
(Task 7/8-shaped work, not the state or cleanup PRs, since only a
controller move touches code that runs during a click) and again after, as
the PR's "a pure move must not move these numbers outside noise" evidence.
A pure move changing *where* code lives must not change the click-latency
numbers; a checkpoint that drifts is a signal the move was not pure and is
grounds to stop and investigate before merging, not to update the recipe.

## 10. `.git-blame-ignore-revs` — one-time setup and the per-PR rule

Every pure-move commit's hash is appended to `.git-blame-ignore-revs`, in
the **same PR** that makes the move, so `git blame` keeps resolving lines to
the author who actually wrote the logic rather than to whichever PR most
recently relocated the file it lives in.

**One-time, per clone**, so `git blame` actually consults the file:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

This is a local git config setting (not committed, not inherited from the
repo) — every clone/worktree that wants blame-through-moves needs to run it
once. `git blame` and `git blame --ignore-revs-file` both work without it,
but plain `git blame`/most editor blame integrations only honor the
ignore-file automatically once this config is set.

## 11. The conversations exemplar, as landed — actual numbers and lessons

The exemplar series (Tasks 6–9: state PR, two controller PRs, cleanup PR)
is complete. This section replaces the plan's 2026-09-01 estimates with
what actually landed, and records what the rehearsal taught the recipe
above — read this before running the next subsystem (export, churn 3, per
§8).

### Methods/fields moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 6 | State | 28 fields → `LibraryConversationsState` (0 methods; a programmatic property-shim block, not per-field getters, to fit the line ratchet — see the shim's own sentinel-comment history for why) | 45134 → 45134 lines (net zero; shim added what `__init__` lost), 1300 methods (unchanged) |
| 7 | Controller 1 | 21 methods → `LibraryConversationReaderController` (5 `@on` handlers + 16 plain) | 45134 → 44715 lines, 1300 methods (pure move: 21 bodies out, 21 one-line delegators in) |
| 8 | Controller 2 | 40 methods → `LibraryConversationsController` (10 `@on` handlers + 30 plain), 7 more excluded (see below) | 44715 → 44060 → 44084 lines (a review-fix round added +24 lines of documentation, no logic change), 1300 methods (unchanged: pure move) |
| 9 | Cleanup | Shim block deleted; every remaining screen-side field reference retargeted to `self._conversations_state.<field>` (90+12 occurrences via one mechanical AST-driven pass, plus 4 methods excluded from Task 8 that had to be retargeted and then had their TEST fakes retargeted to match, per §3's monkeypatch-routing doctrine); 18 of the 61 screen delegators deleted (repo-wide census: zero references anywhere outside their own one-line body); 11 of the ledger's 12 dead imports removed + 1 dead controller import removed (the ledger's 12th, `LIBRARY_CONVERSATION_READER_MAX_CHARS`, turned out to be pinned by PR 0a's re-export contract — see the lesson below — and was restored) | 44084 → **43974 lines, 1282 methods** (18 fewer `FunctionDef`s — exactly the 18 pruned delegators) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`45134/1300 → 44715/1300 → 44060/1300 → 44084/1300 → 43974/1282` (final).

**61 screen delegators, not 68 methods**: the 68-method 2026-09-01 estimate
included the 7 methods Task 8 found could never move (shell-owned, or
reached by a test through a bypass shape a pure move can't survive — see
below); 21 (reader) + 40 (browse) = 61 delegators actually landed, of
which 15 are `@on` handlers, 6 are cross-controller wiring-lambda targets,
22 more have a genuine external reference (another screen method, a test,
or a production caller), and **18 had none** and were deleted in the
cleanup PR.

### Lessons

**Lower the ratchet in the SAME PR that moves the code, never deferred.**
Task 7's own execution first followed the plan's (wrong) instruction to
defer lowering to the cleanup task, producing a real gate slip
(`test_budget_is_not_left_slack_after_a_wave` red between Task 7 and Task
9) before a mid-series correction reversed it. Every move task in this
series ended up lowering its own pin in its own scope; Task 9's lowering
is the *last* one, not the only one.

**The `startswith` enumeration trap.** A discovery script that filters
"already-handled" names with `name.startswith(("_library_conversation",
"_conversation"))` silently swallows any OTHER name that happens to share
that prefix but isn't actually a cluster method or a state field —
Task 7's `_conversation_records`/`_conversation_record_id` (general
browse-cluster helpers) were missed this way and only caught by
re-deriving the bind list without the shortcut. Any enumeration script for
a future subsystem should cross-check its "internal, already covered"
filter against the actual state-field list and the actual cluster list,
not a prefix guess.

**Three, now four, distinct test-bypass shapes a pure move can silently
break** — the recipe's §3 documented the first (class-level
`monkeypatch.setattr`); this series found three more, each requiring a
different accommodation:

1. **Unbound fake-`self` calls** (Task 8, exclusions #2–6): a test builds a
   bare `SimpleNamespace`/hand-built fake with only the flat attribute
   names the ORIGINAL method body needed, then calls
   `LibraryScreen.<method>(fake, event)` unbound. A moved body would reach
   for a `_conversations_controller` the fake doesn't have. Fixed by
   leaving the method real and full-bodied on the screen (not moved) — and
   in Task 9, when the field-retargeting pass touched these same methods'
   *field* references (a cleanup-PR-legal edit the move PR couldn't make),
   the fakes needed a matching retarget: flat kwargs became a nested
   `_conversations_state=SimpleNamespace(...)` constructor argument. This
   is squarely inside the cleanup PR's "retarget test attribute pokes"
   mandate — not a new exception, just the mandate reaching a fixture
   builder instead of a bare attribute assignment.
2. **Instance-attribute monkeypatch** (Task 8, exclusion #7): a test does
   `screen.<method> = lambda: payload` on one REAL, fully-constructed
   instance, expecting an internal sibling call to observe the patch. Once
   both methods live on the controller, the sibling's `self.<method>()`
   resolves against the CONTROLLER instance, which never saw the patch
   applied to the SCREEN instance. Only the full paired-baseline xdist
   sweep (§7) caught this — narrower suites never touched the failing
   test file.
3. **Dynamic `getattr`/dict-string dispatch** (Task 9, new this task): a
   shared, multi-subsystem helper builds an attribute NAME as a runtime
   string (an f-string like `f"_library_{kind}_row_selection"`, or a
   `{destination: "_library_<x>_reader_preferences"}` lookup dict) and
   resolves it with plain `getattr`/`setattr`. Neither the byte-for-byte
   body diff nor an AST literal-attribute-reference retarget script can
   find this shape — the attribute name never appears as a literal
   `self.<name>` expression anywhere. Two independent instances surfaced
   in this one cleanup task: `canvas_sync.py`'s `_apply_library_row_toggle`
   (whose `AttributeError` was silently swallowed into a
   `logger.debug` + `screen.refresh(recompose=True)` fallback — a full
   recompose masquerading as normal operation, caught only by a stale
   captured-widget-reference check, not an exception) and
   `library_screen.py`'s own `_replace_library_reader_preference`/
   `_persist_library_reader_preference` (a 7-destination dict of
   attribute-name strings). Both were fixed the same way:
   `operator.attrgetter("_conversations_state.<field>")(screen)` for
   reads (it resolves a dotted path and a flat name identically, so it's
   a transparent passthrough for the other, not-yet-extracted,
   destinations) and a small `_assign_...attribute(owner, path, value)`
   helper for writes. **Any future subsystem's cleanup PR should grep for
   `getattr(screen,` / `getattr(self,` (and their `setattr` siblings) with
   an f-string or dict-literal argument before deleting that subsystem's
   shim** — a plain "does this literal name still resolve" check is not
   enough.

**"Dead within this file" is not the same question as "dead."** Task 7's
report listed `LIBRARY_CONVERSATION_READER_MAX_CHARS` among nine names it
called dead imports — true in the narrow sense that nothing in
`library_screen.py`'s own logic reads it — but PR 0a's own
`test_screen_still_re_exports_every_moved_name` (§10's sibling contract
test, `Tests/Architecture/test_library_support_layer_surface.py`) pins
`library_screen.py` to keep re-exporting every name Task 1 moved to
`Library_Modules/`, specifically so other modules can keep importing
support names FROM the screen rather than needing to know they moved.
Deleting that one import broke the contract test; it had to be restored.
**Before deleting an import a prior task's report calls "dead," check
whether the name is a member of any `_SURFACE`-shaped re-export contract
first** — a single-occurrence `grep` (the import line and nothing else)
proves the name is unused HERE, not that nothing depends on it being
importable FROM here.

**The static-method delegator pattern.** A moved cluster method that was a
bare `@staticmethod`/`@classmethod` on the screen (no `self` in its own
signature) can't dispatch through `self._controller.<name>(...)` the way
an instance method does — Task 7's first pass on
`_conversation_reader_record_version` dropped the decorator to gain a
`self` to reach the controller through (harmless there — no external
caller used it as a static method — but a latent risk). Task 8 established
the corrected shape instead: keep the decorator, and forward straight to
the CONTROLLER CLASS object (`return LibraryConversationsController.<name>(...)`),
which needs no instance at all. Task 9's strengthened wiring-test regex
(`test_screen_delegates_*_handlers`) accepts either forwarding spelling
(`self._controller.<name>(` or `ControllerClass.<name>(`) for exactly this
reason.

**The `_safe_text` class-binding pattern.** A moved `@classmethod` body
that calls `cls._safe_text(...)` needs `_safe_text` to exist on the
CONTROLLER class, but `_safe_text` is a general, non-Conversations-owned
`@staticmethod` that stays on the screen. Task 8's fix: one module-level
class-attribute assignment, `LibraryConversationsController._safe_text =
staticmethod(LibraryScreen._safe_text)`, executed from `library_screen.py`
(not the controller module, to avoid a circular import) after both
classes are defined. The gotcha a review caught: a plain class-attribute
assignment always REPLACES whatever was previously on the class under
that name — an earlier draft that also defined a `_safe_text` `@property`
on the controller (backed by an injected constructor parameter) had that
property silently and permanently overwritten the instant this module
loaded, making the property, its parameter, and its backing attribute
dead code with a misleading docstring. The corrected version carries only
the one class-level binding, documented in-place with this exact
incident.

## 12. The export series, as landed — the recipe's first non-exemplar rehearsal

Export (churn 3, §8) is the first subsystem to run this recipe after the
conversations exemplar wrote it. Its series (wave-2 tasks 2–4: state PR,
controller PR, cleanup PR) is complete. This section records what actually
landed and what the rehearsal changed or reconfirmed about the recipe
above — read this before running the next subsystem (collections/search,
also churn 6, per §8).

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 2 | State | 13 fields → `LibraryExportState` (0 methods; a programmatic property-shim block, one prefix — no plural-name split needed, unlike Conversations) | 43965 → 43930 lines (net -35: the shim block added less than `__init__` lost, unlike the conversations exemplar's net-zero), 1282 methods (unchanged) |
| 3 | Controller | 22 methods → `LibraryExportController` (of 51 "export"-named candidates: 18 belong to other subsystems, 2 excluded for a NEW bypass shape — "framework-decorator self-type assertion", `@work`'s `isinstance(self, DOMNode)` runtime assertion — and 9 excluded as unbound-fake-self/silent-Mock-auto-attribution, found only by running the battery) | 43930 → 43432 lines, 1282 methods (unchanged: pure move, 22 `FunctionDef`s out, 22 delegators in) |
| 4 | Cleanup | Shim block (13 properties) deleted; every remaining screen-side field reference retargeted to `self._export_state.<field>` (42 literal `self._library_export_<field>` occurrences via one mechanical regex pass, AST-verified against the same census the conversations exemplar used, plus 1 dynamic-dispatch site — see below); 1 of the 22 screen delegators deleted (repo-wide census: zero references anywhere outside its own one-line body); 5 named dead imports removed, each verified single-occurrence and checked against `_SURFACE` first | 43432 → **43413 lines, 1281 methods** (1 fewer `FunctionDef` — exactly the 1 pruned delegator) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`43965/1282 → 43930/1282 → 43432/1282 → 43413/1281` (final).

**22 screen delegators, not 51 methods**: the 51-method naive census
(matching name substring `"export"`) is the same trap §2's `startswith`
lesson warns about, applied to methods instead of fields — Task 3's own
report reads every one of the 51 bodies rather than trusting the name
match, and finds only 22 genuinely Export-owned AND movable. Of those 22,
**21 have a genuine external reference** (mostly the screen-resident
round-2/round-3-excluded siblings calling back into the delegator, plus 5
`@on` handlers and one `action_*`) and **1 had none**
(`_library_export_is_server_mode`, reached only by the controller's own
internal `self.<name>()` calls) and was deleted in the cleanup PR. This
21-of-22 keep ratio is sharply higher than the conversations exemplar's
43-of-61 (18 pruned) — not a sign of a shallower census, but a direct
consequence of Export's own round-2/round-3 exclusions: 11 of the 51
naive candidates stayed screen-resident specifically because a test or a
framework decorator reaches them unbound, and every one of those 11
still calls its sibling delegators internally (`self.<delegator_name>()`),
which is exactly the shape that keeps a delegator alive. Conversations
had no round-2/round-3-shaped exclusion class this large, so its moved
cluster called itself controller-to-controller far more often, orphaning
far more of the screen-side one-liners.

### Lessons

**A fourth bypass shape, new to this series: "framework-decorator
self-type assertion".** Textual's `@work(...)` decorator wraps a method in
a closure that asserts `isinstance(self, DOMNode)` at CALL time (read from
`textual/_work_decorator.py` via `inspect.getsource`, not assumed). A
plain controller object is not a `DOMNode`, so a moved `@work`-decorated
method would raise `AssertionError` on every call, not just under test —
this is not a test-bypass shape at all, but a permanent runtime contract
the framework itself enforces. Two methods
(`_run_library_export_counts_worker`, `_run_library_export_worker`) stay
on `LibraryScreen`, UNMOVED, decorator and body byte-for-byte untouched.
Add this to recipe §3's catalogue alongside the class-level monkeypatch
and the conversations exemplar's three: **before moving any `@work`- (or
similarly self-type-asserting-decorator-)wrapped method, read the
decorator's own source**, not just its name, to confirm it does not bind
`self` to a concrete type the target controller cannot satisfy.

**The "unbound fake-self" bypass shape scales with a subsystem's test
style, sometimes far past the exemplar's own precedent — and can require
widening the sweep root.** Conversations' Task 8 found 5 methods reached
this way, in one test file, entirely within `Tests/UI/`. Export's Task 3
found 9, across SIX test files, FOUR of them in `Tests/Library/` — a
directory the recipe's own canonical `-k "library"`/`-k "export and
library"` sweep commands do not cover by default (they scope to
`Tests/UI`). None of the 9 were caught by static analysis (free-name
resolution, byte-for-byte diff, or a read of the moved bodies); every one
surfaced only by running the verification battery and reading a
traceback. **Any future subsystem's controller-PR battery should
deliberately widen its `-k` search beyond `Tests/UI` before trusting a
clean result** — this recipe's §7 sweep-command documentation has been
updated with this forward note since Task 3 landed; Task 4's cleanup
confirmed the same 2 files needed touching again for the field-level
retarget (`Tests/Library/test_library_export_execution.py`,
`Tests/Library/test_library_export_roundtrip.py`).

**The dynamic-dispatch bypass shape (recipe §11 lesson 3, generalized) can
share ONE helper across two unrelated dispatch mechanisms.** Export's
cleanup found exactly one dynamic-dispatch site touching its own state:
`_library_open_choice_strip`/`_close_open_library_choice_strip` (a
FOUR-subsystem-shared, not Export-exclusive, converged Escape-handler)
returns/consumes a visibility-attribute NAME as a plain string, resolved
with `setattr(self, visibility_attr, False)`. Media/Prompts/Skills still
keep flat screen attributes for their own visibility fields; only
Export's moved to `self._export_state.quality_choices_visible` (Task 2).
Rather than write a SECOND `_assign_...attribute(owner, path, value)`
helper duplicating the shape, this task reused the conversations
exemplar's own `_assign_library_reader_preferences_attribute` (already a
generic dotted-vs-flat passthrough, `owner`/`attribute`/`value` in shape,
with no reader-preferences-specific logic in its body) for the SECOND,
unrelated call site, extending its docstring to document both callers
rather than asserting the reuse silently. **Before writing a new
`_assign_...attribute` helper for a newly-found dynamic-dispatch site,
check whether an existing one is already fully generic in behavior** — the
shape (write through a possibly-dotted path, flat-name passthrough
otherwise) is likely to recur exactly, and one documented helper serving
two call sites beats two near-identical ones.

**The screen-shim wiring test's own retirement, confirmed a second time.**
The state PR's `test_state_object_fields_match_the_shim_surface` (pinning
that every `LibraryExportState` field has a matching generated property
shim on `LibraryScreen`) was deleted wholesale in the cleanup PR, exactly
as the conversations exemplar's Task 9 first established for
`LibraryConversationsState`'s equivalent test — there is nothing left on
the screen for that assertion to check once the shim block is gone. The
controller-side `test_export_controller_exposes_every_state_field`
(already added in Task 3, unchanged by Task 4) covers the equivalent job
from the surviving side. Any future subsystem's state-PR wiring test
should expect its own screen-shim-surface assertion to be a cleanup-PR
deletion, not a retarget, from the day it's written.

**Sweep evidence came back cleaner than the exemplar's own, not worse.**
The full xdist paired-baseline sweep (recipe §7) found **zero**
branch-unique failures this time (329 shared with the pristine baseline,
2 baseline-unique — noise in the opposite direction) — cleaner than the
conversations exemplar's own 5+4 split. This is not evidence the sweep
step is skippable for a "small" cleanup PR: the field-retarget pass
touched 13 test files and one dynamic-dispatch site shared across four
subsystems, any one of which could plausibly have broken something the
narrower `-k` checks don't cover. The sweep is the check that would have
caught it if it had.

## 13. The collections series, task 1 (state PR) — as landed

Collections (churn 6, §8) is the second subsystem to run this recipe.
Wave-2 task 5 (state PR, collections series 1/3) is complete; the
controller and cleanup tasks have not yet run. This section records what
landed and what this task's own execution added to or reconfirmed about
the recipe above.

### Cluster enumeration

`ast` walk of `LibraryScreen` for method names containing "collection"
(case-insensitive): **67 methods** (2026-09-02 snapshot, matches the
wave-2 plan's estimate exactly). Of those, **3 are Prompts-owned**
(`handle_library_prompts_collection`, `_apply_library_prompt_collection`,
`_sync_library_prompt_collection_label` -- the unrelated "Prompt
Collections" feature, confirmed by reading each body: they use
`_library_prompt_collections_controller`/`_library_prompt_browse_controller`
and have nothing to do with the Library Collections/captures subsystem),
excluded per the export series' own documented substring-match trap. Of
the remaining 64, 42 carry a distinct `@on` decorator; 41 once the one
Prompts false-positive (`handle_library_prompts_collection`) is dropped.

### Field ownership (recipe §2 script, `_library_collections` prefix)

**27 `__init__`-scoped fields** (not ~28 as the wave-2 plan estimated --
re-derived, not trusted). Unlike Export's `origin_row_id`, no field was
missed by the `__init__`-scoped census: a full class-level `AnnAssign`
scan found zero collections-owned class-level-only attributes. Of the 27:

- **26 are exclusively-owned state** -- MOVE. All non-collection
  `__init__`-scoped consumers found by the script are shell/plumbing
  (`on_unmount`, `save_state`/`restore_state`, `apply_navigation_context`,
  `_build_library_shell_input`, `_select_library_rail_row_after_source_
  admission`, `_persist_library_reader_preference`,
  `request_library_reader_layout_refresh`) or a generic multi-subsystem
  dispatcher whose name happens to contain another subsystem's prefix as
  a substring without being owned by it
  (`_toggle_library_media_reader_pane`, which branches on
  `_library_selected_row_id` across Collections/Conversations/Notes/
  Media -- the same "tagging heuristic is name-based, not body-based"
  caveat the recipe's own script output already carries). No field is
  shared with a SECOND subsystem's OWN methods, so the ≥2-subsystems rule
  never triggers and none is BLOCKED.
- **1 is wiring, not state**: `_library_collections_capture_controller`
  holds a live `LibraryCollectionsCaptureController` instance (the
  `_conversation_reader_controller` precedent the task brief named in
  advance). It stays a plain `LibraryScreen` attribute, constructed at
  its original position, untouched by this move.

**The saved-searches census** (the brief's flagged contested boundary,
"collections vs the search cluster"): `_library_collections_saved_
searches` / `_library_collections_saved_searches_total` are referenced in
exactly 5 places, ALL inside `library_screen.py`, ALL inside
collection-cluster methods --
`_library_collections_capture_presentation` (reads both, for the
render-only presentation object), `_load_library_collections_capture_
entry` (writes both, from `scope.list_saved_searches(1)`), and
`select_library_collection_capture_scope` (reads `saved_searches` to
resolve a pressed saved-search scope row). A repo-wide grep confirms zero
references anywhere outside `library_screen.py`. The `SavedCaptureSearch`
type and `list_saved_searches` method live in `Library/collections_
capture_models.py` and `Library/collections_capture_service.py` --
Collections' own capture-scope service layer, not a generic "search"
module. The "search cluster" the brief warned about is a DIFFERENT,
unrelated feature living in the same file: the Library-wide search rail
and its history (`handle_library_search_changed`, `_load_library_search_
history`, `_persist_library_search_history`, `clear_library_search_
history`, `rerun_library_search_from_history`, …) -- the top search bar's
submit/history mechanism, conceptually adjacent (both are "search") but
architecturally unconnected: none of those methods reference either
saved-searches field, and Collections' saved searches are per-scope
filter presets persisted through the capture repository, not search-bar
history. **Verdict: MOVE, uncontested** -- the census resolves the
brief's flagged ambiguity outright; no BLOCKED condition arose.

**Total: 26 fields moved, 1 field (wiring) stayed, 0 fields BLOCKED.**

### Characterization spot-check

41 `@on`-bound Collections selectors checked with a per-id `grep -rn`
across `Tests/` followed by a manual read of the surrounding lines for an
actual `.press()`/`.click()`/direct-value-assignment/`Input.Submitted`
interaction. **A same-line-only grep undercounts coverage**: one selector
(`retry_library_collection_quick_capture`, `#library-collections-capture-
retry-confirm`) looked unpressed under a same-line check but is genuinely
covered once the press on the line AFTER its `query_one` call is read --
the same multi-line query-then-press shape the export series' own report
already flagged. 24 of the 41 are genuinely covered (23 direct + this
one). The remaining **17 are genuine gaps**, pinned into `Tests/UI/
test_library_collections_characterization.py` across 5 grouped test
functions (mirroring this codebase's own walkthrough-style tests, which
routinely exercise several related handlers inside one continuous Pilot
session). No live bugs found -- all 17 are coverage gaps, not behavior
bugs. See that file's own module docstring for the full per-selector
accounting.

**A new bypass-adjacent gotcha, not from a test but from writing new
characterization tests against this subsystem's OWN async recompose
shape**: pressing a button immediately after a PRIOR async transition's
`_wait_for_condition` resolves true can race that transition's own
trailing `_refresh_library_collections_capture_reader()` call --
`_run_library_collections_capture_transition` schedules its SECOND
recompose only after `await task` completes, and a condition watching
`controller.state` can observe the state mutation before that second
recompose has actually rebuilt the DOM. A widget fetched via
`_wait_for_selector` immediately after such a condition resolves can be
briefly stale, and pressing a toggle in that window can appear to silently
no-op (the flag DOES flip, but the query for the newly-revealed content
made in the same tick sees the pre-toggle tree). Symptom: `NoMatches` on
a selector that "should" exist, or a 30s `_wait_for_selector` timeout on
one that should appear instantly. Fix: a few extra `await pilot.pause()`
calls after the setup condition, before touching DOM elements that depend
on it -- not a code bug, a test-timing gap. Cost real debugging time in
this task (two of five characterization tests needed this fix before they
passed reliably); worth checking for in any future subsystem's
characterization tests that select a row/item and then immediately act on
its detail pane in the same test.

### `LibraryCollectionsState` + shims

`tldw_chatbook/UI/Library_Modules/library_collections_state.py`: a
`@dataclass` with all 26 fields, verbatim defaults, matching the export
precedent's single-prefix shape (`_library_collections_` for every
field -- Collections' own subsystem name is already plural, so unlike
Conversations no field needed a DIFFERENT prefix variant; no
`COLLECTIONS_PLURAL_STATE_FIELDS` constant exists here). Three fields
(`reader_preferences`, `reader_persistence_locks`, `reader_layout`) are
entangled with other subsystems' shared init code exactly like the
conversations exemplar's own trio and keep their original `__init__`
assignment lines untouched.

**A new deviation from the "construct at the position of the first
removed field" default, required by entanglement ordering**:
`reader_preferences` is entangled with a tuple-unpack shared with Media/
Conversations/Notes/File Notes/Prompts/Skills that executes chronologically
BEFORE Collections' other (non-entangled) fields are assigned in the
original `__init__` -- unlike the conversations exemplar and the export
series, where every entangled field's original line happened to sit AFTER
the position their state object was constructed at. Constructing
`self._collections_state` at the "first removed field" position (which
sits after the shared tuple-unpack) would raise `AttributeError` the
first time that unpack's `self._library_collections_reader_preferences`
target tried to route through the not-yet-installed property into a
not-yet-existing object. Fixed by constructing `self._collections_state =
LibraryCollectionsState()` at the SAME early point
`self._conversations_state` is constructed instead -- before the shared
tuple-unpack, not at the first non-entangled field's position. This is
still a pure, behaviorally-transparent move: every one of the 23
non-entangled fields' defaults is a static literal (no field needed a
constructor argument, unlike Export's computed `form` default), so
constructing the dataclass earlier than usual has no observable
side effect. **Any future subsystem whose entangled reader-preferences
field is NOT the chronologically-last thing assigned before its own
non-entangled block should check this ordering explicitly** before
assuming "construct at the first removed field" is safe -- it is only
safe when no entangled field's original line precedes that position.

Shim: mirrors Export's single-prefix generated-property-loop shape
exactly (`for _cos_field in dataclasses.fields(LibraryCollectionsState):
setattr(LibraryScreen, "_library_collections_" + _cos_field.name,
property(...))`), sentinel-wrapped, installed at module end.

### Size ratchet

**The stale-pin gap this task's own brief flagged (43413 pin vs 43412
true measurement) is now closed.** `git show HEAD:...|wc -l` on the
pre-task tree measured 43412, one below the recorded `_BUDGETS` pin of
43413 -- a 1-line slack that had gone unnoticed since the export cleanup
PR. This task's own net change (import +1, field-block removal -23,
early-construction line +2, shim block +20) landed at **43410 lines,
1281 methods** (measured fresh, post-edit) -- both below the stale pin
AND below the true pre-task baseline, so `_BUDGETS` is lowered to
`43410` in this same commit, closing the 1-line gap rather than carrying
it forward. Pin trajectory:
`43413 (stale) / 43412 (true) -> 43410` (methods unchanged at 1281 -- a
pure field move, zero method bodies touched).

### Wiring test -- TDD evidence

`Tests/Architecture/test_library_collections_wiring.py` was written and
run FIRST, before `library_collections_state.py`'s shim installation
existed on `LibraryScreen`:

```
FAILED Tests/Architecture/test_library_collections_wiring.py::test_state_object_fields_match_the_shim_surface
```

RED confirmed (an assertion failure -- the module already existed by the
time the test was written in this task's own execution order, but no
shim property existed on `LibraryScreen` yet). After the screen edit
landed:

```
Tests/Architecture/test_library_collections_wiring.py .            [100%]
1 passed
```

GREEN. Scope matches the export series' own Task 2 precedent exactly
(state-object fields <-> shim surface only; a controller PR in this
series will add the full-cluster/same-name-delegator-forwarding shape
later).

### Sweep evidence — zero real branch-unique failures

The full xdist paired-baseline sweep (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`) found 333 failed/3906 passed on
this task's branch versus 340 failed/3894 passed on a `git stash -u`
pristine baseline of the pre-task tree. Diffing the failure-name sets: 2
unique to branch, 9 unique to baseline only, 331 shared. Both
branch-unique failures
(`test_library_prompt_history_no_change_keeps_selection_and_retry_available`,
`test_library_starter_production_geometry_and_focus_order[size1]`) were
confirmed pure xdist ordering/shared-state noise by a direct single-process
rerun (individually and combined) — both pass cleanly every time, and
neither touches Collections or shares a fixture with this task's diff.
**Zero real regressions** — matching the export series' own Task 2 sweep
result more closely than its Task 3/4 sweeps (which each surfaced genuine
new bypass-shape exclusions), consistent with this task's pure
field-relocation shape.

## 14. The collections series, task 2 (controller PR) — as landed

Wave-2 task 6 (collections controller PR, collections series 2/3) is
complete. Full derivation, cluster table, and byte-for-byte/free-name
verification method are in `task-6-report.md`; this section records the
headline numbers and the one new finding for the next subsystem.

**64 of the 67 "collection"-named candidates move, uncontested** (3 are
Prompts-owned, reconfirmed from task 5's own field-level exclusion). This
is the first controller PR in this recipe's rehearsal with **zero method-
level exclusions** — no `@work` hazard, no class-level or instance-
attribute monkeypatch, no unbound-fake-self/silent-Mock call, found for
ANY of the 64 by a script-driven sweep of the whole `Tests/` tree (not a
sample). Contrast export (29 of 51 excluded) and see §12/§13 for why:
Collections' cluster is small, cohesive, and entirely mediated through one
pre-existing headless engine (`LibraryCollectionsCaptureController`) that
tests exercise through full-harness Pilot sessions rather than unbound
`SimpleNamespace` unit calls.

**The brief's own "browse-controller-delegation exclusion" instruction
named a file that does not exist.** `library_collections_browse_
controller.py` is not a real module; the only pre-existing Collections
controller is `library_collections_capture_controller.py` (a headless
orchestration engine, not a screen-delegation target). A guard test,
`Tests/UI/test_product_maturity_phase39_library_collections.py::
test_collections_route_has_no_generic_container_controller_or_panel`,
asserts the STRING `"LibraryCollectionsBrowseController"` (a retired
container-based design) never appears in `library_screen.py` — this is a
naming-retirement guard, not a live controller. The new controller is
named `LibraryCollectionsController` (matching the state-PR's
`LibraryCollectionsState` and the export series' own naming), which does
not collide with the guard. **Any future subsystem's controller-PR brief
should verify a named "existing controller to check delegation against"
actually exists (`find`/`grep` for the literal filename) before trusting
the brief's own naming** — this one was a stale/confused reference, caught
only by reading the actual guard test instead of assuming the file was
just not yet explored.

Pin trajectory (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`):
`43410/1281 -> 42486/1281` (methods unchanged — pure move, 64
`FunctionDef`s out, 64 one-line delegators in).

**Byte-for-byte verification by TOOLING, not manual diff-reading**: this
task extracted each of the 64 method's exact source segments with an
`ast`-driven script (using the ORIGINAL file's own line offsets, never
hand-retyped), assembled the new controller module from that extracted
text plus a hand-written header/footer, then re-verified with a SECOND
script that re-parsed both the pre-move screen file and the finished
controller module and asserted byte-for-byte identity per method. For a
move this size (64 methods, ~1300 lines), this generalizes past the
"read every body once" discipline earlier tasks used: **write the
extraction and the verification as scripts, not as a sequence of manual
Read/Edit operations**, for any future subsystem whose cluster exceeds
~40-50 methods.

**Correction (fix round 1, post-review) -- a dynamic-dispatch census gap,
new sub-shape**: this task's own module docstring and report originally
claimed a clean dynamic-dispatch sweep (no `getattr`/`setattr` call using
an f-string or dict-literal argument touches a Collections name, beyond
the one pre-existing reader-preferences hit §12 already documents). That
census script only matched a dict-literal/f-string passed DIRECTLY as the
call's own argument. It missed a two-step shape that DOES exist inside
this task's own moved cluster: `retain_library_collection_quick_capture_
input` builds a small DOM-id-keyed dict, `.get()`s a NAME into a local
variable, then calls `setattr(self, attribute, event.value)` with that
variable on a later line. **Safe by construction** (the three dispatched
names are among the 26 fields with a full `property(get, set)` shim, not
read-only), but the census's own claim of completeness was wrong until
corrected in review. **Any future subsystem's dynamic-dispatch census
should also grep for a `dict.get(...)` result assigned to a variable that
later flows into `setattr`/`getattr` within the same function** -- not
just a literal argument passed directly to the call -- to avoid
re-missing this shape.

**Sweep evidence — concurrent branch/baseline xdist runs amplify
flakiness measurably; see §7's updated documented-failures entry for the
full numbers and per-test disposition.** Zero real regressions confirmed,
but the concurrent-run shortcut (both full sweeps launched at once to
save wall-clock time) cost real investigation effort re-running 12
branch-unique names individually/combined and, for the one that
reproduced, against the pristine baseline too. **Future tasks should run
the two full sweeps sequentially when time allows.**
