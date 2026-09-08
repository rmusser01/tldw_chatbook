# Wave-8 Task 4 — Wave close AND PROGRAM CLOSE

Branch `refactor/library-decomp-wave8-notes`, worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
task parent `4ab86c34d` (task 3's close), wave start `889e12b86`.

**The final task of the eight-wave Library decomposition program.**

| Commit | What |
|---|---|
| `66e572868` | `fix(library): media Select all/Clear silently full-screen recomposed` — the wave's ONE deliberate behaviour change, its own commit, FIRST. **No blame-ignore entry** (not a move) |
| `96409c16e` | `docs(library): wave-close stale-prose sweep — 11 class-3 sites, all line-neutral` |
| `bad16ab8f` | `chore(backlog): file the two wave-8 findings that have no owner` — TASK-32040, TASK-32041, plus a wave-8 block on TASK-31249 |
| `9381fdad3` | `docs(library): program close — trajectory, lessons, and the phase-C handoff` |

Nothing pushed. `progress.md` was not touched. No `.git-blame-ignore-revs`
row was added by this task: neither commit is a pure move, which is §10's own
criterion.

---

## 1. The media defect — FIXED, not filed (coordinator ruling)

### 1.1 Reproduced before touched

Three measurements at the task parent, none inherited:

```
hasattr(LibraryMediaController, "_media_state")        -> False
hasattr(LibraryNotesController, "_media_state")        -> True
hasattr(LibraryMediaController, "_library_media_row_selection") -> True   (the shim loop)
```

The third line is the one that makes the defect asymmetric and is why the
mutation table below has an off-diagonal: on a controller the FLAT name
still resolves (its own permanent shim loop), so only the NEW dotted
spelling breaks there.

Live line numbers, each measured on the tree it is quoted against (§24):

| Fact | Line | Tree |
|---|---|---|
| `screen._media_state.selected_media_id = media_state.selected_id` | `canvas_sync.py:448` | this tree |
| the swallowing `except Exception` in `_sync_library_canvas` | `canvas_sync.py:725` | this tree |
| `_sync_library_canvas(self, "media")` in `handle_library_media_select_all` | `library_media_controller.py:2692` | pre-fix (`4ab86c34d`) |
| … in `handle_library_media_select_clear` | `library_media_controller.py:2701` | pre-fix |
| the same two calls after the fix | `:2731` / `:2740` | this tree |
| `self._media_state_accessor = media_state_accessor` | `library_media_controller.py:691` | both |
| the notes analogue this fix mirrors | `library_notes_controller.py:957–977` | this tree |

**Two of these corrected a hand-off citation.** Task 3's report and recipe
§23 both cited `canvas_sync.py:437` and `:710`; those were correct at task
3's PARENT and eleven/fifteen lines stale on the tree the hand-off pointed
me at, because task 3's own cleanup added lines above them. The brief's
`library_notes_controller.py:956-976` is `957–977` measured by
`ast` (`decorator_list` 957, `def` 958, `end_lineno` 977). Both are in §24's
ledger.

### 1.2 RED first

The `[controller]` leg was added and run BEFORE the property existed:

```
FAILED …::test_media_row_toggle_resolves_the_dotted_state_path[controller]
E  AttributeError: '_Controller' object has no attribute '_media_state'
   tldw_chatbook/UI/Library_Modules/canvas_sync.py:240
E  AssertionError: fallback recompose must not fire
1 failed, 1 passed, 8 deselected
```

The `AttributeError` is the defect; the `AssertionError` is the `refresh`
double turning the SILENT degradation into a red test — the same mechanism
the conversations and notes siblings use.

### 1.3 The fix, and the mutation table

One `@property _media_state` on `LibraryMediaController` returning
`self._media_state_accessor()`, in the shape its own sibling
`library_notes_controller.py` already carries. **No moved body touched** —
all 140 movers unchanged, and none of them spells `self._media_state`; they
read the flat properties from the generated shim loop, which is what keeps
them byte-for-byte.

The guard is parametrized over the receiver exactly like this wave's notes
guard: the `[controller]` leg subclasses the REAL `LibraryMediaController`,
overriding only `query_one` / `refresh` (framework-service properties a
double cannot assign over) and `_library_media_analyze_reason`, and leaves
`_media_state` INHERITED so the accessor under test is the production one.

| Mutation | `[screen]` | `[controller]` |
|---|---|---|
| remove the `_media_state` accessor property | passes | **FAILS** |
| revert `canvas_sync.py`'s media branch to the computed flat name | **FAILS** | passes |
| neither | passes | passes |

Both directions were run, both reded exactly one leg, and `canvas_sync.py`
was restored byte-for-byte afterwards (`git diff` on it is empty).

### 1.4 No regression, proven by pairing rather than argued

`Tests/UI/test_library_multiselect_media.py` +
`Tests/UI/test_library_media_trash.py` measure **5 failed / 190 passed**
WITH the change and **5 failed / 190 passed** WITHOUT it (the three tracked
files checked out to HEAD, run, restored) — the SAME five names both times.
`library_media_controller.py` re-pinned 4630 → 4669 in the same commit (§6).

### 1.5 What this fix could NOT have

The natural end-to-end coverage — a real-row Pilot test pressing
`#library-media-row-0` and reading the real bulk-action labels — is
`test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_
marker_label_both_directions`, which is **TASK-31880's standing red**. Wave
7 had to guard the fifth spelling with a hand-built screen double for the
same reason; wave 8 now had to guard its receiver dimension the same way.
That is two consecutive waves paying the same bill, and it is why TASK-31880
is named as the coverage gate in the phase-C handoff rather than left in the
flake list.

---

## 2. Stale-doc sweep — 11 class-3 sites, only 3 of them known

Recipe §3's standing prose census, tokenize-based (COMMENT and STRING tokens
only), over the **100 deleted field names AND the 26 pruned method names**,
across `tldw_chatbook/`, all of `Tests/`, `Docs/`, `backlog/`, `scripts/`,
`Helper_Scripts/`, with frozen historical roots excluded.

**231 raw occurrences before / 117 after. The ±3-line screen-attribution
filter gives 21 before / 20 after, and every remaining candidate was read.**

The three the hand-off named were confirmed at their live lines and fixed
(`library_prompts_state.py:297`, `screen_constants.py:167`,
`library_unavailable_navigation.py:716`). **Eight more were found by this
sweep**, and two of them are shapes the ±3-line filter structurally cannot
see:

- `library_notes_state.py:542` and `:552` — the state module cross-references
  its OWN sibling fields by their pre-move flat spellings, ten lines from
  where those fields are defined under their new bare names. No "screen"
  token anywhere near either.
- `library_notes_canvas.py:190` — the widget the subsystem paints through.

Plus `library_screen.py:16673` (a MEDIA method's comment naming
`_library_note_preview`), `:18183` (naming `_library_note_title_user_edited`
as "the provenance guard below", three lines above
`self._notes_state.title_user_edited`), `test_library_shell.py:27451`
and `:27554` (two names), and
`test_library_notes_characterization.py:155`.

**Every fix is line-neutral** — `git diff --numstat` is `n n` for all eight
files, and `library_screen.py` still measures 32230/1258.

**One shape was deliberately LEFT, and it is the interesting one.**
`library_notes_controller.py:3353`/`:3474` spell
`_library_note_session_blank_id` and `_library_note_pending_blank_gc_id` in
comments that sit INSIDE MOVED BODIES. Those bodies are frozen by the
byte-for-byte canon and the flat names still resolve there through the
controller's own shim loop — so the identical phrase is a DEFECT in
`library_notes_state.py` and CORRECT in `library_notes_controller.py`. A
census that fixed both would have broken §1's canon.

Class 1/2 left as designed: `test_library_media_trash.py:4146` ("where it
used to be"), `test_library_notes_folder_navigator.py:259` ("was moved up"),
`library_screen.py`'s successor comment, and `library_notes_state.py`'s
cutover-guard narrative with its own commit citation.

**Read and ruled OUT of scope, recorded rather than skipped silently:**
`library_notes_state.py:535`/`:553`/`:594` name `_mark_library_note_dirty`
and `_ensure_library_notes_sync_config_loaded`, neither of which exists in
the tree in any spelling. Neither is in the 100 deleted fields or the 26
pruned delegators — they are stale from an earlier notes-adaptive refactor,
not from this program, and inventing a replacement name would be a guess.

The four prior-extracted notes wiring modules
(`library_note_import_controller.py`, `library_notes_sync_controller.py`,
`library_notes_work_session.py`, `note_session_port.py`) were swept for
present/future-tense staleness: **`note_session_port.py:155`'s reference to
`_patch_library_note_list_from_session` is LIVE** (the method is still
screen-resident at `library_screen.py:17978`, a task-2 exclusion), and the
other three carry no flat-name or future-tense references at all.

---

## 3. Recipe — the program's institutional memory

One commit. Every number in it was derived at this tree or from `git show`,
never carried over from a report — and three of the brief's own figures did
not survive that (below).

### 3.1 §2 — the read/write refinement (task 1's TASK-4 MUST, undone until now)

§2's flat "a field referenced by two or more subsystems … is never moved"
now carries the refinement four waves have actually applied: **an
other-subsystem-named WRITER blocks a move; an other-subsystem-named READ is
a route guard and does not.** With the notes series' own split recorded by
NAME (12 cross-tagged → 2 BLOCKED / 10 MOVE, the two blocked being the only
ones with a media-named writer) and media's four-member shell-family rule
beside it. Read literally, the old sentence would have blocked media's own
`view` and the prompts series' `_library_prompts_mutation_in_flight`, both
of which moved and are load-bearing precedents today.

### 3.2 §3 — two new spellings' worth of catalogue

- **The fifth spelling's RECEIVER dimension**, with the resolution table
  that makes the mechanism legible (flat resolves on a controller, dotted
  does not — the inverse of the natural reading), the notes dual-receiver
  guard, the media defect as the worked example, and the one-line `ast`
  census that answers it. Plus the rule to report **31 call sites across 26
  distinct movers** as two numbers, because the METHOD count is the
  load-bearing one.
- **The SIXTH spelling** — every `ast.Attribute` whose `value` is
  `Name("LibraryScreen")`, anywhere, whatever encloses it. Written with the
  explicit warning NOT to anchor the census on `partial`, since the
  enclosing expression is exactly what varies. 27 targets at the parent, 22
  notes candidates; zero for the movers and 196 for the exclusions at the
  landed tree.
- **The two receiver-census HOLES** (per-name instead of per-binding-SITE;
  the `getattr` receiver a dotted census scores zero), which between them hid
  four of the notes cleanup's six seeds, plus "fixing the first instance of
  a shape does not close the shape".
- **The thread hand-off spelling set** — all ten spellings named, with the
  1,478-site count and the independent 1,419-site re-derivation, and the
  point that a one-spelling zero is not an answer.
- **The prose-sweep subsection** gains this close's own 11-of-which-3-known
  result, the two shapes the ±3-line filter cannot see, and the one shape
  that must be LEFT (a moved body's comment, frozen by the canon).

### 3.3 §7 — the ladder's fourth rung, and three instrument notes

- **A bidirectional unique split is NOT a disposition.** Quoted with the
  wave-8 numbers (52/760 vs 47/765, 6 branch-unique + 1 parent-unique,
  concentrated in two files neither of them the extracted subsystem — and
  three of the seven were a real 3/3-vs-0/3 regression). The mechanism is
  stated plainly: a bidirectional split is what a real regression looks like
  when it lands inside a noisy batch, which is the normal case.
- **The backdrop-absorption note**: a paired sweep is structurally blind to
  "already documented", so §7's list has to be consulted BY NAME,
  deliberately.
- **The by-name rule's third incident**, named as such:
  `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure` has now
  been reported as new three times, and the second happened *after* the
  entry written because of the first.
- **The `grep -r` skips dot-directories note** — this program's evidence
  lives under `.superpowers/`, so a plain recursive grep silently misses
  every prior task's record. Use `command grep` or an explicit path.
- **Batch composition**: the 30-second DOM-mount timeouts make wall time a
  function of FAILURE count, with the measured 75-minutes-to-22% vs 3m14s
  pair.
- **Three new documented reds** (task 1's hand-off), each with its rates and
  its disposition, including the cutover guard's full vacuity story.

### 3.4 §23 — the wave-8 close

Pin trajectory re-derived per COMMIT across all 20 commits in
`889e12b86..HEAD`, which produced two facts no report carried:

- **The controller was born at 5232**, not the 5216 the brief said nor the
  5254 §23's own table said. 5214 is post-fix-round-1, 5216 post-docstring-
  correction, 5254 post-fix-round-2. This is the SECOND consecutive wave
  whose "born at" figure was a post-fix-round value (wave 7: born 4474,
  reported 4461).
- **The screen pin went DOWN and back UP inside task 2** (32286 → 32325),
  the same shape wave 7 had, and for the same kind of reason: a
  battery-found exclusion putting a body back.

Plus the media-defect section rewritten as FIXED with the mutation table
and correct line citations, and the prose-sweep result.

### 3.5 §24 (new) — evidence discipline, the citation-accuracy ledger

**A `file:line` is a number; a range is two numbers; nothing in the battery
reads one.** Three rules with their incidents: derive a citation from the
tree you ship against (task 3's `:437`/`:710` hand-off, correct at its
parent and stale where it pointed me); re-measure a range rather than
adjusting it (task 2's `:17651-17655` → `:17652-17656`, and this brief's
`:956-976` → `957–977`); a citation used as a REASON is load-bearing (task
2's withdrawn Form-E claim on `test_screen_navigation.py:3225`).

**The third strike is the one that earns the section**: it happened INSIDE
the review round convened to fix count and claim accuracy — the claim that
`library_conversation_reader_controller.py` declares `_library_notes_source`
is wrong; its only such property duplicates media's focus-intent one. An
accuracy round is not self-verifying.

### 3.6 §25 (new) — the program close

The eight-wave trajectory table, and the arithmetic that closes exactly:

```
45,134 − 17,544 (the waves' own PRs) + 4,640 (dev, between waves) = 32,230
 1,300 −    160                      +   118                     =  1,258
```

**Dev put 4,640 lines and 118 methods back between the waves — a 26.4%
give-back against the program's own extraction**, positive in six of the
seven inter-wave gaps. That is the strongest argument for the ratchet in
the document, and it is visible only from a whole-program table.

The landed architecture: **9 state dataclasses (2,698 lines) and 10
controllers (28,329 lines measured, not pinned), 368 fields and 795 method
bodies moved byte-for-byte, 154 delegators pruned.** Membership derived from
each file's `git log --diff-filter=A` creation commit rather than from its
name — `library_collections_capture_controller.py` and
`library_media_browse_controller.py` read like program output and are not.

**The brief's "eight state objects + eight controllers" is wrong and is
corrected in the recipe**: nine subsystems across eight waves (wave 2 landed
two), nine state objects, ten controllers (the exemplar split into reader +
browse). §1's stale "eleven subsystems" is corrected the same way.

**The six duplicate definitions, measured rather than asserted:**

| | `FunctionDef`s | distinct names | duplicated |
|---|---|---|---|
| before the move (`915e76b71`) | 1290 | **1277** | 13 |
| after (`9e13f0207`) | 1284 | **1277** | 7 |

The distinct count is identical, which is the real pure-move invariant and
the one §25 now tells the next series to assert. Seven duplicated names
survive on the screen today, so the shape is not notes-specific.

Then the three reusable catalogues (six census spellings; the bypass-shape
catalogue with "exclusion rate measures TEST DEBT, not entanglement"; the
four-rung ladder) and the four-part phase-C handoff: the probe baseline, the
mount-storm acceptance target (**114 mounts for a notes switch, 177–179 for
media, and 38/85 even on a re-click of the SAME row**, against steady-state
node counts of 114/119 — load-independent, so usable as an acceptance
target), TASK-31880 as the coverage gate, and `canvas_sync.py` as the first
surface.

---

## 4. Filings

Commit `bad16ab8f`.

**The id sweep, run IMMEDIATELY before filing.** Local max: **32013**. Max
across all **647** local + remote refs (`git for-each-ref` × `git ls-tree`
over `backlog/tasks` and `backlog/drafts`): **32033**. The CLI probe —
create a throwaway task, read the id it assigns, delete it — offered
**32014**, a number already held on an unmerged branch and *below* the swept
max. That is precisely the false-low answer
`backlog/docs/lessons-backlog-hygiene.md` invented the probe habit to
expose, and it fired on the first try. Leapfrogged past 32033; the probe
file was deleted, not left behind.

**TASK-32040 — the permanently-vacuous cutover guard** (task 1's TASK-4
MUST, task 3's re-verification). The filing carries the whole evidence
chain: RED at `889e12b86`; flipped GREEN by the state move for a reason the
guard did not intend; made permanent by the shim deletion (`attr` is
`auto_sync_timer`, which does not start with the guard's prefix, so the
census can never see it again in any file at any revision); and **not
naively retargetable** — a method-name prefix repoint false-positives on
`_library_notes_sync_controller`, the WIRING binding accessor, and a
field-following census goes RED again on a vestigial-but-real field. Four
ACs, and the filing OWNS the product decision (delete / retarget / retire)
rather than pre-empting it. This is the one wave-8 finding deliberately not
fixed in flight.

**TASK-32041 — the dispatcher receiver contract has no guard.** Generalised
from the defect this task fixed, and measured rather than asserted: of the
10 controllers this program created, **8 declare no `_<subsystem>_state`
accessor property**, and five of those forward a bare `self` into
`_sync_library_canvas`. Today's exposure is closed by inspection — only the
media (`:448`) and notes (`:470`) legs read a dotted state path, and both
controllers now declare the accessor — so the ACs ask for a guard that
re-derives the receiver census on every run, not for a fix. The `search`
leg's own comment ("the controller has no `_rag_search_state` attribute at
all", `canvas_sync.py:517–526`) is quoted as the model: two authors met the
same hazard and only one left something that would catch the next person.

**TASK-31249 gains a wave-8 block rather than a third id**, matching how
waves 4, 5 and 7 appended to the same census: the new
`test_library_file_notes_workspace.py` paint settle race with BOTH rates
(6/10 branch, 3/10 isolated parent), the 66/63 recompose ratchet
re-confirmed unmoved for a third wave, pointers to 32040/32041, and the note
that wave 8 is the second wave to pay TASK-31880's bill.

**Checked first, not double-filed:** TASK-31815 (modal-inventory blocker),
TASK-31880 (the real-row coverage gate), TASK-31881 (media route round-trip),
TASK-31976, TASK-32013 (media exclusion debt), TASK-31249. TASK-31249 already
owns the recompose ratchet, both dev-side ratchet rows,
`test_on_screen_suspend_stops_every_timer_in_isolation`, the modal-dismissal
inventory and `test_screen_navigation.py`'s 32-name regression — none of
which needed re-filing.

---

## 5. Battery

All runs `-p no:randomly`, `.venv/bin/python -m pytest`, branch venv Python
**3.14.2**. Isolated baseline worktree at the wave start `889e12b86` with
its own `uv venv --python 3.14`: **interpreter parity proven** (3.14.2
both), **package parity proven** (`uv pip list` name lists `diff`ed to
identical, 106 rows), and the isolated venv verified to resolve its OWN tree
(`tldw_chatbook.__file__` under the scratch path).

| Check | Result |
|---|---|
| 10 guard suites (9 wiring + support layer) + BOTH size ratchets + recompose ratchet + preimport closure + ui-ready census + `test_library_selection_updates.py`, ONE process | **145 passed / 6 failed**, 40.25s |
| the 6 | EXACTLY the documented standing reds — 2 × `chat_screen.py`, `library_media_browse_controller.py` 371-pin, `library_conversations_controller.py` slack mirror, the 66/63 recompose ratchet, `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure` |
| `library_screen.py` row | **GREEN at 32230 / 1258** |
| `library_notes_controller.py` row | **GREEN at 5276** |
| `library_media_controller.py` row | **GREEN at 4669** (re-pinned by the behaviour fix) |
| screen reuse + notes characterization + modal dismissal + inspection admission | **219 passed / 2 failed**, 146.59s |
| the 2 | TASK-31249's, re-confirmed by their exact failure TEXT rather than by name: `AttributeError: 'LibraryScreen' object has no attribute '_unavailable_navigation'` at `library_screen.py:7867`, and `unresolved modal constructor … SkillImportChoiceModal(snapshot.candidates)` at `test_library_modal_dismissal.py:856` |
| **BOTH dual-receiver guards** | notes `[screen]`+`[controller]` and media `[screen]`+`[controller]` all pass; the media pair mutation-verified both ways (§1.3) |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** — CSS bundles, profile-owned paths, diagnostic inventory (592 owners, unchanged), **"No duplicate task IDs across 3438 task files"** (which is what certifies this close's own two filings), chachanotes allowlist, index plan pins |
| fresh `_measure()` | screen **(32230, 1258)** vs pin `(LibraryScreen, 32230, 1258)`; notes controller **5276** vs 5276; media controller **4669** vs 4669 — exact matches, zero drift, nothing to lower |

**Documented reds only.** Every failure was checked against recipe §7 BY
NAME before being called anything — the meta-lesson this wave earned twice
and the program three times.

---

## 6. Paired baseline

§7's own net, whole-wave span, identical command on both trees, run
SEQUENTIALLY (branch first, then the isolated worktree at `889e12b86`):
`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`.

| | failed | passed | wall |
|---|---|---|---|
| branch | **397** | 4381 | 1460.60s (24:20) |
| isolated parent `889e12b86` | **397** | 4375 | 1529.22s (25:29) |

**393 shared, 4 branch-unique, 4 parent-unique.** That is a bidirectional
split — and by the rung this very wave added to §7, **that is not a
disposition**. All eight went through the third level: each node ALONE,
trees INTERLEAVED inside one load window, n=3, raised to **n=10** for the
three timing-shaped ones.

| Node | branch | parent | verdict |
|---|---|---|---|
| `test_screen_navigation.py::test_search_route_round_trips_to_the_library_rag_row` | **3/3** | **3/3** | deterministic on BOTH — a member of §7's documented 32-name `test_screen_navigation.py` regression |
| `test_screen_navigation.py::test_search_all_palette_command_lands_on_library_with_honest_toast` *(parent-unique)* | **3/3** | **3/3** | same file, same standing regression |
| `test_audio_cpp_model_library_handoff.py::…reveals_slow_load_once_and_keeps_error_retry` | 3/10 | **7/10** | flaky on both; PARENT worse |
| `test_library_media_reader_traversal_t22207.py::test_focus_traversal_builds_zero_bodies_for_pass_through_rows` | 4/10 | **6/10** | flaky on both; PARENT worse |
| `test_library_media_reader_traversal_t22207.py::test_loading_banner_paints_in_place_without_body_rebuild` *(parent-unique)* | 3/10 | **4/10** | flaky on both; PARENT worse |
| `test_library_media_reader_flow.py::test_filter_uses_authoritative_search_and_restores_page_three_anchor` | 0/3 | 0/3 | xdist noise |
| `test_library_multiselect_conversations.py::…toggles_it_in_select_mode` *(parent-unique)* | 0/3 | 0/3 | xdist noise |
| `test_library_prompts_canvas.py::…undo_refreshes_applied_page_and_preserves_basket` *(parent-unique)* | 0/3 | 0/3 | xdist noise |

**Zero real regressions**, and two things worth keeping.

**The bidirectional split was manufactured out of a SHARED backdrop.** The
two `test_screen_navigation.py` names are 3/3 red on both trees and belong
to the same documented 32-name regression; xdist happened to scatter one
into each run. That is the backdrop-absorption mechanism §7 now warns
about, seen from the other side: absorption can hide a documented red, and
partial absorption can invent a split.

**Every timing-shaped name is worse at the PARENT.** 3/10 vs 7/10, 4/10 vs
6/10, 3/10 vs 4/10. That is the outcome an interleaved third level exists to
make visible and that a back-to-back batch reliably gets backwards — and
two of the three are media-named, i.e. exactly the names this task's own
behaviour fix could plausibly have broken. They were measured, not argued.

229 of the branch's 397 failures are `test_library_shell.py`, the
~226–230-red whole-file backdrop TASK-31249 records as an environment fact
for this machine. Load: 3.03 at start, 4.98 at the hand-over, 5.82 at end.

---

## 7. Probe — the last pre-phase-C baseline

`Helper_Scripts/library_click_probe.py` is byte-identical across the wave
(`git diff 889e12b86 HEAD --` on it is empty). **118 probe invocations
across FIVE trees**, and the reason it took five is the finding.

### 7.1 What the standing procedure said, and why it was wrong

§9's order-swapped pair (round 1 branch-first, round 2 baseline-first) came
out with the branch slower on **15 of 16** wall-clock measurements. An
interleaved n=6 pair narrowed that to exactly ONE row —
`media (switch-in)`, +21 ms median, **branch slower in 6/6** — with every
load-independent column identical. That is a directional, reproducible
signal, and §9 says a drift is grounds to stop and investigate.

Each cheaper explanation was eliminated by measurement, in this order:

| Hypothesis | Test | Result |
|---|---|---|
| first-run warm-up | order-swapped pair (§9's own rule) | still directional both rounds |
| the WAVE's moves | third tree at the wave tip `4ab86c34d`, interleaved n=16 | tip **−2** vs parent — non-monotonic in commit order, so not the wave |
| THIS task's commits | fourth tree at `66e572868` (behaviour fix only), interleaved n=10 | fix-only **−6** vs parent; the whole delta attaches to `66e572868..HEAD`, whose entire diff is **twelve comment lines and two backlog files** |
| position within the round | rotate the order to `parent,tip,fix,branch` | branch last: still **+23** |
| the checkout LOCATION | fifth tree: `git worktree add --detach <scratch> HEAD` (`diff -r` vs the working tree **empty**) | **this was the whole effect** |

| same source (HEAD), two locations, n=10 interleaved | `media (switch-in)` median | vs parent |
|---|---|---|
| working worktree, `~/Documents/GitHub/...` | 500 ms [482–702] | **+23** |
| scratch worktree, `/private/tmp/...` | 470 ms [428–490] | **−6** |
| wave-start parent, scratch worktree | 476 ms [471–505] | — |

Identical bytes. A 30 ms median swing and a visibly dirtier distribution
(max 702 vs 490) purely from where the tree sits. **§3 puts the isolated
baseline at a scratch path and the branch is the working worktree, so every
probe pair this program has run for eight waves compared two different
filesystems.** The order swap protects against warm-up; nothing protected
against this, and from the numbers alone the two are indistinguishable —
wave-6's "branch slower on all sixteen measurements" is equally consistent
with it. Recipe §9 now requires probing the branch from a scratch worktree
too.

### 7.2 The valid pair — same filesystem, n=10 interleaved, medians

| interaction | settle br / base | max gap br / base | recompose | full-update | mounts br / base | nodes |
|---|---|---|---|---|---|---|
| media (switch-in) | **470** / 476 | 134 / 124 | 0 / 0 | 2 / 2 | 177–179 / 177 | 119 / 119 |
| media (re-click same) | **303** / 308 | 46 / 46 | 0 / 0 | 2 / 2 | 85–89 / 85–89 | 119 / 119 |
| media (re-click same, 2nd) | **302** / 304 | 46 / 47 | 0 / 0 | 2 / 2 | 85–89 / 85–89 | 119 / 119 |
| notes (switch) | **350** / 355 | 128 / 132 | 0 / 0 | 2 / 2 | 114 / 114 | 114 / 114 |
| notes (re-click same) | 258 / 258 | 42 / 43 | 0 / 0 | 1 / 1 | 38 / 38 | 114 / 114 |
| media (switch-back) | **384** / 388 | 89 / 91 | 0 / 0 | 1 / 1 | 175–179 / 175–179 | 119 / 119 |
| notes (switch, 2nd) | **368** / 374 | 157 / 160 | 0 / 0 | 2 / 2 | 114 / 114 | 114 / 114 |
| media (switch-back, 2nd) | **384** / 390 | 84 / 84 | 0 / 0 | 1 / 1 | 175–179 / 175–179 | 119 / 119 |

**The branch is equal or FASTER on all eight rows.** The load-independent
columns are the verdict and they are exact: `recompose` is **0 on every row
of all 118 runs across all five trees**; `full-update` follows the identical
pattern on every tree; `nodes` is identical row-for-row (119 media / 114
notes); `mounts` value sets are identical, wobbling 2–4 within each tree as
prior closes recorded. Band: settle 243–494 ms, max gap 37–179 ms. Load
averages 3.51–7.41, recorded per round.

**One honest note.** The `full-update` pattern is `2/2/2/2/1/1/2/1`, where
waves 2–7 recorded `2/2/2/1/1/1/1/1` — the two notes-switch rows moved from
1 to 2. That change is present at `889e12b86` as well as on the branch, so
it landed in dev's churn between the wave-7 close and the wave-8 start, not
in this wave. Recorded rather than smoothed over, because the next close
will otherwise re-derive it as new.

### 7.3 This is the phase-C baseline

Recorded prominently in recipe §25's handoff, with the mount-storm reading:
a single rail-mode switch mounts **114 widgets (notes) / 177–179 (media)**
against steady-state node counts of 114 / 119, and a re-click of the SAME
already-selected row still mounts **38 / 85–89**. Those mounts are the
freeze phase C exists to remove, they are load-independent, and they give
phase C a falsifiable success condition: **the re-click rows go to ~0
mounts with `recompose` still 0.**

---

## 8. Deviations, errata and process notes

**Three of the brief's own numbers did not survive verification, and all
three are corrected in the shipped artifacts rather than worked around.**
This is §24's rule applied to the instruction set itself:

1. **"controller born 5216 → 5276."** Measured per commit, the notes
   controller was born at **5232** (`9e13f0207`); 5214 is post-fix-round-1,
   5216 post-docstring-correction, 5254 post-fix-round-2, 5276 post-cleanup.
   The recipe's wave-8 trajectory table states the whole chain and names the
   error, exactly as the wave-7 close did for its own controller.
2. **"eight state objects + eight controllers as the landed
   architecture."** Measured: **nine** state dataclasses and **ten**
   controllers across **nine** subsystems in eight waves — wave 2 landed
   export and collections together, and the conversations exemplar is the
   only subsystem that split into two controllers. Membership was derived
   from `git log --diff-filter=A` per file, because two dev-owned
   controllers (`library_collections_capture_controller.py`,
   `library_media_browse_controller.py`) read exactly like program output.
3. **`library_notes_controller.py:956-976`** for the notes accessor
   property: it spans **957–977** by `ast` (`decorator_list` 957, `def` 958,
   `end_lineno` 977). Filed as the fourth entry in §24's ledger.

**Two hand-off citations were stale where they pointed.** Task 3's
`canvas_sync.py:437` and `:710` were correct at task 3's PARENT and are
`:448` and `:725` on the tree the hand-off pointed me at, because that
cleanup added eleven lines above them. Recipe §23 carried the stale pair;
it now carries the correct ones with the tree each was measured on, and a
parenthetical saying what happened.

**"Eight of ten controllers lack the accessor" is a measured claim.** It
comes from an `ast` walk of each controller class body for `@property`
methods matching `_[a-z_]+_state`, cross-referenced with a regex for
`self._<x>_state_accessor =` assignments and with the bare-`self`
dispatcher call sites and their `kind` literals — not from reading the two
files this task touched.

**The recipe's own §23 heading said the notes cleanup was "the second
largest of the program"; the wave-8 net (−3,547) is second to prompts'
−3,819.** Stated with the full ordering rather than as a superlative,
because a superlative is the shape that goes stale silently.

**Deliberately NOT done:**

- **No `.git-blame-ignore-revs` row.** Neither of this task's code-touching
  commits is a pure move: `66e572868` is a behaviour fix and `96409c16e` is
  a comment sweep. §10's criterion is "pure-move commit", and adding rows
  for these would launder a behaviour change out of `git blame`.
- **No `Docs/User_Guide/` page touched.** CLAUDE.md asks for a User-Guide
  update on PRs that change a screen's UI. The behaviour fix REMOVES an
  unintended whole-screen recompose — the user sees the same pixels arrive
  without a full repaint — and renders nothing differently. Considered and
  declined, consistent with all eight prior closes.
- **`progress.md` untouched**, per the brief.
- **Nothing pushed.**
- **`library_notes_controller.py:3353`/`:3474` left carrying flat names in
  comments** — inside moved bodies, frozen by the byte-for-byte canon (§2 of
  this report).
- **`_mark_library_note_dirty` / `_ensure_library_notes_sync_config_loaded`
  left stale** in `library_notes_state.py` — pre-existing from an earlier
  refactor, not this program's, and no replacement name exists to point at.

**The probe investigation is the largest deviation from the plan's shape,
and it was the right one.** §9 prescribes an order-swapped pair; that pair
returned a directional result, and the recipe's own words are that a drift
is "grounds to stop and investigate before merging, not to update the
recipe." Investigating cost 118 probe runs and five trees, and it found a
methodology defect that has silently affected every probe pair this program
ran. The alternative — reporting "branch slower on 15 of 16 measurements,
probably warm-up" — would have been the fourth time in this program that a
confident cheaper explanation was wrong.

**Process notes.** Staging was from explicit path lists, never `git add -A`
(wave-7's erratum), and `git status` was checked for untracked files before
each commit. The isolated baseline worktree was created once with proven
interpreter and package parity, reused for the whole close, and removed at
task close. Every number quoted below comes from a run that reached its own
summary line; nothing is extrapolated from a progress percentage.
