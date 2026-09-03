# Task 2 report — Export series 1/3: characterization spot-check + LibraryExportState

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `477704580` (Task 1,
census anti-slack guard). Two commits landed:

- `a0c8a6410` — `test(library): characterization pins for the export extraction series`
- `f4e8acecf` — `refactor(library): export state object + shims (export series 1/3)`

## 1. Cluster enumeration

`ast` walk of `LibraryScreen` for method names containing `"export"`
(case-insensitive): **51 methods** — matches the wave-2 plan's snapshot
exactly (re-derived at execution time, not trusted from the plan).

Of those 51, 20 are bound to a distinct `@on(Button.Pressed|Input.Changed,
"#<selector>")` decorator (some methods carry two decorators for a
markdown/context-menu sibling pair). The characterization pass checked
every one of the 20 selectors with `grep -rn "#<id>" Tests/` followed by a
manual read of the surrounding lines for an actual `.press()`/`.click()`/
`insert_text_at_cursor()` call — not merely an id reference (the brief's
"not just id mentions" instruction; conversations Task 5's own
`.disabled`-assertion trap repeated itself twice here).

## 2. Characterization decision table

| Selector / handler | DOM-driven? | Disposition |
|---|---|---|
| `#library-export-submit` / `handle_library_export_submit` | Yes (`.click()`/`.press()` in `test_library_export_receipt.py`, `test_library_shell.py`) | Covered, no pin |
| `#library-export-cancel` / `handle_library_export_cancel` | **No** — only `query_one(...).display` check + 3 unbound `LibraryScreen.handle_library_export_cancel(fake, None)` calls | **PIN** |
| `#library-note-export-md`, `#library-note-context-export-md` / `handle_library_export_markdown` | Yes (`test_library_shell_note_export_pushes_file_save_dialog`, parametrized `.press()`) | Covered |
| `#library-note-context-export-txt` / `handle_library_note_export_text` | Yes (same parametrized test) | Covered |
| `#library-note-export-txt` / `handle_library_note_export_text` | No direct press | **SKIP** — same handler as the row above via a second `@on` decorator on one method body (no `event.button.id` branch); sibling id already exercises it |
| `#library-conversations-export-selected` / `handle_library_conversations_export_selected` | **No** — id referenced in 3 files, always for `.disabled` assertions | **PIN** |
| `#library-media-export-selected` / `handle_library_media_export_selected` | **No** — same pattern across 4 files | **PIN** |
| `#library-prompt-export`, `#library-prompts-export`, `#library-prompts-export-selected` | Yes (`test_library_prompts_canvas.py`, multiple `.press()`) | Covered |
| `#library-media-export` / `handle_library_media_export` | Yes (5 files press it) | Covered |
| `#library-conversations-export`, `#library-notes-export`, `#library-notes-export-selected` (bulk, not "-selected" for conversations was already covered above; these three are the bulk/other-selected siblings) | Yes (`test_library_shell.py` presses each) | Covered |
| `#library-export-name` / `handle_library_export_name_changed` | Yes (`name_input.insert_text_at_cursor(...)`) | Covered |
| `#library-export-description` / `handle_library_export_description_changed` | **No** — zero references anywhere in `Tests/` | **PIN** |
| `#library-export-quality`, `.library-export-quality-choice` | Yes (`test_library_choice_strips.py`, real presses) | Covered |
| `#library-export-destination` / `handle_library_export_choose_destination` | Yes (multiple `.press()` + `FileSave` dialog assertions) | Covered |
| `#library-rag-open-import-export` / `open_import_export_from_library_rag` | Yes (`test_library_shell.py:7155`) | Covered |
| `#library-collections-legacy-recovery-export` / `choose_library_collection_legacy_recovery_export` | **No** — zero references anywhere; its write-path (`_export_library_collection_legacy_recovery`) is pinned elsewhere but only via a direct method call that bypasses this handler and its `FileSave` push | **PIN** |

**5 pins written** into `Tests/UI/test_library_export_characterization.py`
(commit `a0c8a6410`), all passing against current code pre-move (inverted
TDD confirmed — see §6). No live bugs found; all 5 gaps are coverage
gaps, not behavior bugs. The remaining 31 non-`@on` private export
helpers are reached transitively by the same well-covered submit/counts/
destination/note/prompt pipeline (mirrors the conversations exemplar's
identical blanket finding for its own 34 helpers) and were not
individually re-pinned.

## 3. Ownership table (recipe §2 script, `_library_export` prefix)

Script output (12 `__init__`-scoped fields, all `NONE` or shell/plumbing-
only non-export users — no field shared with another subsystem):

| Field | Non-export `__init__` users | Verdict |
|---|---|---|
| `scope` | NONE | MOVE |
| `counts` | `on_mount` (shell/plumbing) | MOVE |
| `counts_request_id` | NONE | MOVE |
| `form` | NONE | MOVE (computed default → constructor arg) |
| `quality_choices_visible` | `_library_open_choice_strip` (shell/plumbing) | MOVE |
| `running` | `_library_emergency_return_eligibility` (shell/plumbing) | MOVE |
| `error` | NONE | MOVE |
| `status` | NONE | MOVE |
| `run_id` | NONE | MOVE |
| `cancel_event` | NONE | MOVE |
| `last_path` | `save_state`, `restore_state` (shell/plumbing) | MOVE |
| `last_at` | `save_state`, `restore_state` (shell/plumbing) | MOVE |

**The `_library_export_origin_row_id` verdict** (brief's flagged boundary
case): this field is **not an `__init__` assignment at all** — it was a
plain class-level annotated default (`_library_export_origin_row_id: str
= ""`, line 988, docstring: "Class-level default for the same
restored-session reason as the other class-level route defaults"). The
`__init__`-scoped ownership script structurally cannot see it (a second,
new variant of the recipe's documented `startswith`-enumeration trap: this
one isn't a prefix-filter miss, it's a "field never assigned in `__init__`"
miss). Found by a manual `grep -n "_library_export_origin_row_id"
tldw_chatbook/UI/Screens/library_screen.py` across the whole file.

Consumer census (5 total references):
- `_open_library_export_canvas` (write) — export-owned.
- `action_library_export_back` (read + clear) — export-owned.
- `_select_library_rail_row_after_source_admission` (write, clears to
  `""` on a plain rail switch) — **shell/plumbing** (this is the exact
  "rail-switch shell code" the brief named).
- `_library_route_shortcuts_for_current_state` (read, for the Export
  canvas's footer "back to X" label) — **shell/plumbing**: a general
  footer/F1 shortcut projector that branches on `_library_selected_row_id`
  across Ingest/Export/Study/Collections, not owned by any one subsystem.

Both non-export consumers are shell/plumbing-only (no other subsystem's
prefix appears in either method's name) — **verdict: MOVE**, per the
recipe's explicit rule ("Non-subsystem users that are shell/plumbing
methods... still moves; shims keep them working"). The class-level
attribute was deleted; `LibraryExportState.origin_row_id: str = ""`
supplies the identical default through the generated property shim.

**Total: 13 fields moved, 0 fields stayed.** No ≥2-subsystem sharing was
found for any export field, so no BLOCKED case arose.

One incidental finding recorded for the *future* cleanup PR's benefit (not
a Task 2 blocker): `_close_open_library_choice_strip` resolves
`quality_choices_visible`'s screen-facing name via a runtime
`setattr(self, visibility_attr, False)` where `visibility_attr` can be the
literal string `"_library_export_quality_choices_visible"` (recipe §3
lesson 3's "dynamic getattr/dict-string dispatch" bypass shape). This is
harmless for Task 2 — `setattr(instance, name, value)` on a real
`LibraryScreen` instance dispatches through the installed property
descriptor identically to `instance.name = value` — but the export
series' eventual cleanup PR should grep for this pattern before deleting
the shim block, per the recipe's own standing instruction.

## 4. `LibraryExportState` + shims

`tldw_chatbook/UI/Library_Modules/library_export_state.py`: a
`@dataclass` with all 13 fields, verbatim defaults. `form`'s dataclass
default is a momentary placeholder (`field(default_factory=dict)`);
`__init__` passes the real value (`self._default_library_export_form()`)
as a constructor argument at the exact position the original assignment
line occupied, per the recipe's "computed defaults become constructor
arguments" rule — the only field in this series needing that treatment
(unlike conversations' three *entangled*-default fields, export's `form`
default is self-dependent but NOT shared with another subsystem, so it
folds cleanly into the one constructor call rather than needing its
original `__init__` line left untouched).

All 13 fields use the single `_library_export_` prefix — the ownership
analysis found no field needing a plural variant (conversations needed
`_library_conversations_` for `row_selection`/`select_mode`; export has no
analogous case), so `CONVERSATIONS_PLURAL_STATE_FIELDS`'s sibling constant
was deliberately not created for export.

Shim: a module-level `for _es_field in dataclasses.fields(LibraryExportState):
setattr(LibraryScreen, "_library_export_" + _es_field.name, property(...))`
loop at the end of `library_screen.py`, sentinel-wrapped
(`--- BEGIN/END generated export-state shims ---`), installed exactly like
the conversations exemplar's own state-PR shim block. `_n=` default-arg
closures bind both getter and setter per field.

## 5. Wiring test — TDD evidence

`Tests/Architecture/test_library_export_wiring.py` was written and run
FIRST, before `library_export_state.py` existed:

```
ERROR Tests/Architecture/test_library_export_wiring.py
E   ModuleNotFoundError: No module named 'tldw_chatbook.UI.Library_Modules.library_export_state'
1 error in 0.37s
```

RED confirmed (collection error, not merely an assertion failure — the
state object genuinely did not exist yet). After the state module + shim
landed:

```
Tests/Architecture/test_library_export_wiring.py .            [100%]
1 passed, 94 warnings in 0.92s
```

GREEN. Scope matches the brief's deliverable 5 exactly ("state-object
fields ↔ shim surface") — this task is the state PR only; the wave-2
plan's "full-cluster loop, same-name forwarding regex" language describes
the file's eventual shape across all 3 export tasks, not Task 2's own
scope (mirroring conversations Task 6's own wiring-test commit, which
shipped the identical single-assertion shape before its controller PRs
added delegation-regex checks).

## 6. Verification battery

**Wiring test RED → GREEN**: see §5.

**Characterization file all-PASS** (inverted TDD, run against current
code before any state-PR edit, then reconfirmed after):
```
5 passed, 94 warnings in 6.50s
```

**Size ratchet — ceiling AND slack green**, lowered in this same commit
per recipe §6 (measured post-move, not carried over):
`_BUDGETS["library_screen.py"]`: `43965/1282` → `43930/1282` (net
**-35 lines**, 0 methods delta — line-neutral-to-slightly-negative, as
expected: the shim loop is smaller than the multi-paragraph field
comments it replaced). Both `test_screen_does_not_grow_past_its_budget`
and `test_budget_is_not_left_slack_after_a_wave` pass for the
`library_screen.py` row.

**Recompose ratchet (with its slack guard) + support-layer surface**: all
green — this move touches zero `refresh(recompose=True)` call sites (pure
field relocation), so the census pin (63) and its new anti-slack guard
(task-27019, landed by Task 1) are unaffected.

**`-k "export and library"` suite with stash-baseline comparison**: 14
failures on this task's branch. A direct rerun of the exact same 14 node
ids against a `git stash -u` baseline (state-PR diff removed, prior
characterization commit kept — confirmed via `git diff` that the
characterization commit touched only its own new test file) reproduced
**the identical 14, no more, no fewer**. All 14 confirmed pre-existing and
appended to the recipe's documented list (recipe §7) with root-cause notes
(one unbound-fake `SimpleNamespace` missing a newly-added method call;
thirteen `#library-notes-row-0`-mount timeouts consistent with
machine-load-sensitive DOM flakiness, 100% reproducible both sides).

**Full xdist paired-baseline sweep** (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`):
- Branch: 332 failed, 3902 passed (1247.67s)
- Baseline (`git stash -u`): 334 failed, 3900 passed (1271.84s)
- Diff: **0 failures unique to branch**, 2 failures unique to baseline
  only (better on branch, absorbed noise, not attributable to this task),
  332 shared (pre-existing backdrop).

**Preflight** (`./scripts/preflight.sh`): all six checks green (CSS
bundle, profile-owned-path census, diagnostic inventory, backlog task
ids, chachanotes table allowlist, index plan pins) — run twice, before
and after the sweep/stash cycle.

## 7. Files changed

- `Tests/UI/test_library_export_characterization.py` (new, commit
  `a0c8a6410`) — 5 characterization pins.
- `tldw_chatbook/UI/Library_Modules/library_export_state.py` (new, commit
  `f4e8acecf`) — `LibraryExportState` dataclass.
- `Tests/Architecture/test_library_export_wiring.py` (new, commit
  `f4e8acecf`) — state-field/shim wiring test.
- `tldw_chatbook/UI/Screens/library_screen.py` (modified, commit
  `f4e8acecf`) — import added, class-level `origin_row_id` attribute
  removed, 12-field `__init__` block replaced with one
  `LibraryExportState(...)` constructor call, generated shim block
  appended at module end.
- `Tests/Architecture/test_screen_size_ratchet.py` (modified, commit
  `f4e8acecf`) — `_BUDGETS` row lowered to `43930/1282`.
- `backlog/docs/library-decomposition-recipe.md` (modified, commit
  `f4e8acecf`) — 14 new pre-existing failures appended to the §7
  documented list.

No `.git-blame-ignore-revs` entries added (brief's explicit instruction:
state PRs are not body moves).

## 8. Self-review

- Ownership script re-derived at execution time (12 fields), not trusted
  from the plan's ~12 estimate; cross-checked with a broader whole-class
  attribute scan that surfaced the 13th (`origin_row_id`) and ruled out a
  14th false positive (`_library_export_is_server_mode`, a method, not a
  field, caught by checking Store-vs-Load/Call context).
- Every `git rev-parse`-worthy hash in this report was read from actual
  `git log`/`git show`/`git commit` output, never typed from memory.
- Byte-for-byte canon respected for method bodies: no method body touched,
  no receiver rewritten, no "while I'm here" cleanup.
  **Correction (fix round 1): this bullet originally also claimed the
  carried-forward `__init__` field COMMENTS were verbatim — that was
  false.** Review caught three concrete misses: the 8-line "Export canvas
  state (F4 Task 2)" header (dropped from `scope`/`counts`/`form`
  entirely, with the `__init__` call-site comment then falsely asserting
  the detail lived in the state module), `run_id`'s comment silently
  renamed `_library_export_running`/`_library_export_error`/
  `_library_export_status` to the new field's own bare names `running`/
  `error`/`status` (repeated once more in `status`'s own comment, not
  separately named by the reviewer but the same defect), and
  `cancel_event`'s comment dropped its trailing "Nothing sets it yet in
  this task -- the Cancel button and navigate-away wiring land in Task 5"
  sentence. Root cause: I paraphrased/retyped the field comments while
  writing the new dataclass instead of copy-pasting the base text and
  only touching code, and then asserted "verbatim" in this section
  without diffing against `git show 477704580:...` to check. Fixed in
  commit `264314c5f` (§9) — every comment block is now confirmed,
  by an automated normalized-substring check against the base file, to
  contain the original wording unchanged, including the
  `counts_request_id` comment's original double space after "below.".
  The lesson: "I carried this verbatim" is a claim that needs the same
  evidence discipline as any other test/verification claim in this
  recipe — diff against the base commit, don't trust memory of having
  typed it carefully.
- Both ratchets (size, recompose) verified green with THIS commit's own
  measurement, not deferred — recipe §6's explicit lesson from the
  conversations exemplar's Task 7 near-miss.
- Sweep evidence follows recipe §7's procedure exactly: xdist run,
  paired `git stash -u` baseline at identical `-n`/`--dist` config, diff
  restricted to branch-unique failures, then a targeted node-id rerun (not
  the full sweep) to individually confirm the smaller `-k` subset's 14
  failures before trusting them as pre-existing.
- No BLOCKED conditions encountered: every export field resolved
  unambiguously to MOVE under the recipe's existing rules, including the
  boundary case the brief flagged in advance.

## 9. Fix round 1 — verbatim-comment review finding

Commit `264314c5f`, `docs(library): restore verbatim comments in the
export state move (fix round 1)`.

**What happened**: during the state-PR move, three field comments in
`library_export_state.py` were retyped/paraphrased rather than
copy-pasted from base `477704580`, and §8's self-review asserted
byte-for-byte comment carry without actually diffing against the base
commit to check. Review caught it:

1. The 8-line "Export canvas state (F4 Task 2)" header comment (base
   `library_screen.py:3232-3238`) — explaining `_library_export_counts`'
   None-until-landed semantics and why `_library_export_form` is a plain
   dict — was dropped entirely. `scope` and `counts` carried no comment
   in the new module, and the `__init__` call-site comment falsely
   claimed the per-field detail "lives" in the state module when it
   didn't yet.
2. `run_id`'s comment silently renamed `_library_export_running`/
   `_library_export_error`/`_library_export_status` to the new field's
   own bare names (`running`/`error`/`status`) — and `status`'s own
   comment made the identical substitution once more, not separately
   named by the reviewer but the same defect, caught in my own re-check.
3. `cancel_event`'s comment dropped its trailing sentence: "Nothing sets
   it yet in this task -- the Cancel button and navigate-away wiring
   land in Task 5."

**Fix**: restored all three verbatim, including
`counts_request_id`'s original double space after "below." (a fourth,
reviewer-named detail). Also fixed `origin_row_id`'s comment on the same
pass (not named by the reviewer, same defect on re-inspection): its
original 4th sentence ("Class-level default for the same
restored-session reason as the other class-level route defaults.") had
been replaced rather than kept-and-supplemented; restored verbatim with
the new explanation added alongside it, clearly marked as new prose.

**Verification**: every restored block was checked with an automated
normalized-substring comparison against `git show 477704580:
tldw_chatbook/UI/Screens/library_screen.py` (whitespace/comment-marker
collapsed, but the `below.  Scope` double space explicitly preserved and
separately confirmed present in the actual file via a direct read) before
committing — not eyeballed. `Tests/Architecture/
test_library_export_wiring.py` and `Tests/UI/
test_library_export_characterization.py` re-run green (6 passed,
comment-only change; the size ratchet is untouched since
`library_screen.py` itself was not part of this fix — only
`library_export_state.py` changed, 32 insertions / 15 deletions, all
comment text).

**Correction to this report's own §8**: the original self-review's
"Byte-for-byte canon respected... The only non-mechanical edit is the
docstring/constructor-call phrasing" bullet was inaccurate on the comment
axis; §8 above has been amended in place rather than left standing next
to a silent contradiction.
