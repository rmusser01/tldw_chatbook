# Fresh-lens review of the approved spec "Roleplay on the Library frame (sub-project B)"

- **Spec reviewed:** `Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md` (3,247 lines; written against `origin/dev @ 84247cb843`).
- **Checked against:** `origin/dev` at `eba4305d83` and `6958e8dfa9`, the open PRs #2862, #2949, #2952, #2953 and #2957, the review evidence in this folder, and the live captures.
- **Five lenses:** drift (what changed on dev since the spec), simplicity, Library consistency, UX risk, and delivery/safety.
- **Process:** every issue was independently re-verified. Refuted or not-worth-it items are listed under "Dropped". Appendix B's resolved issues were not re-raised. No owner ruling (Q1-Q12) or approved section is re-litigated on preference.

---

## Headline

**The design holds. No finding needs an approved design decision reversed.** The spec is not ready to plan from as written, though. Nine defects would either fail CI or lose user work if implemented literally:

1. **B0 cannot merge as written.** The spec reads the CSS budget from a stale snapshot. Real boot CSS headroom on dev is **50 bytes**, not about 24 KB, so B0's planned +1.1 KB fails the ADR-097 ratchet in CI (MF-01).
2. **The new Roleplay stylesheet is loaded the way a repo test forbids** (`PersonasScreen.CSS_PATH`). It must go through the app's route map (MF-02).
3. **The test infrastructure is described inaccurately.** The styled test harness is described three conflicting ways and arrives one slice late (MF-03). The live harness's restart check cannot work, and its scripts may disappear from scratch before B1 (MF-04).
4. **Four interaction holes can lose work or strand the user.**
   - An editor can appear without a work session, bringing back the "Test chat starved to 46 columns" blocker (MF-05).
   - Several Escape and Cancel paths send focus into a pane that is closed (MF-06).
   - "Save and continue" cannot save an editor that is hidden behind another view, and B9's new draft types have no save or discard path at all (MF-07).
   - The entry editor can write one entry's text onto another (MF-09).
5. **Imported names can crash the app or run app actions.** A name like `[/]` or `[@click=app.quit]Ada[/]` on any markup-enabled surface (header, title row, border, button) does this. This was reproduced in the project's Textual 8.2.8 (MF-08).

The other 24 must-fix items are cheap text corrections: stale numbers, missing table rows, wrong arithmetic, and wording that claims Library behaviour the Library does not have.

**Recommended:** eleven improvements, ranked. The most valuable are:

- show the per-kind empty state from B5b, not B10, so the work pane is not blank for four slices (RC-1);
- split B2, B5b and B9 so that the safety fixes ship before the visual work (RC-2);
- tie the leave and return rules to explicit hooks rather than to the screen being rebuilt each visit, which PERF-22 will change (RC-4).

**Owner decisions:** three small ones, in plain language below:

- what the `<` `>` keys do next to the Edit chip (OD-1);
- whether to unify bulk selection across Library and Roleplay now or later (OD-2);
- how a second grip click behaves during an edit (OD-3).

**Effort:** about a day of spec editing, plus committing the live harness (half a day), before B0's plan starts.

---

## Must fix before planning

Severity reflects verified impact. "Edit" gives the exact change. Section and line numbers refer to the spec at commit c21d5cb04c.

| ID | Sev | What is wrong | Where | Sources |
|---|---|---|---|---|
| MF-01 | high | Boot CSS headroom is 50 B, not ≈24 KB, so B0's +1.1 KB fails CI | §5.10 row; B0 gates; §2.12 item 2; §2.1 | DR-01, DR-03 |
| MF-02 | high | Lazy sheet loaded from `PersonasScreen.CSS_PATH`, which a repo test forbids | §2.12 item 3; B1 scope; §5.7.4 | DR-02, DL-01 |
| MF-03 | high | Styled test harness mis-described and arriving one slice late | §5.7.1; B2 tests; B1 acceptance | DL-02, DR-02 |
| MF-04 | high | Live harness not portable; the B7 restart check cannot work | §5.7.5; K14 | DL-03 |
| MF-05 | high | Authoring views can show with no work session (the 46-column Test chat returns) | §2.3; §4.1; §4.11; B7 | SI-01 |
| MF-06 | high | Escape, Cancel and Done send focus into closed panes or a hidden drill-in list | §4.5 rungs 3 and 5; §4.11; §5.7.2; K21 | UX-01, LC-03, UX-03 |
| MF-07 | high | "Save and continue" cannot save hidden editors; B9's domains have no save or discard path | §3.12; R41; B3; B9 | SF-01 |
| MF-08 | high | Untrusted names can crash the app (MarkupError) or run app actions (`@click`) | R33; §1.3; §1.5.1; §3.2; K11 | SF-03 |
| MF-09 | high | The entry form follows the cursor; Update writes to the cursor row | §3.9.2-§3.9.5; rung 4; B9 | SF-04 |
| MF-10 | medium | Deleting another row loads over a dirty item; the "complete" trigger table misses six triggers | §4.11 Delete; §3.12 | UX-05, SF-02 |
| MF-11 | medium | List windows can raise DuplicateID (which can exit the app), skip rows, and cannot restore past row 200 | §1.5.6; §1.5.4; B2 | SF-05, SI-04 |
| MF-12 | medium | The one Import picker drops .webp cards, Markdown cards and Buddy packs; its sniff has no bounds | §1.4.6; B2 | SF-07 |
| MF-13 | medium | No failure states: "Loading Ada…" can hang; conflict copy says "Reselect" | §3.2; §4.11 | UX-08 |
| MF-14 | medium | B1 cannot meet its PS line gate; the ratchet is already red | B1 gates; §5.7.4 | DL-04 |
| MF-15 | medium | B2 deletes a file listed in the PR-gate census, so preflight fails; B's new tests never gate PRs | §5.4; B2; §5.7.4 | DL-05 |
| MF-16 | medium | Latency gates are wall-clock through Pilot, and no 5,000-row dataset exists | B2 acceptance; B6; §5.10 | DL-06 |
| MF-17 | medium | The quit path is changing under B (#2949, TASK-33622.14), with no dependency rows | §3.12; §5.5; §5.6; G9 | DR-06, SF-09 |
| MF-18 | low | ADR-210 (accepted after the base) changes the 80x24 arithmetic and Console's header | §1.1; §2.10; §5.6; §5.7.2; G11/G12 | DR-04, DR-05 |
| MF-19 | low | Stale ratchet numbers | §2.12 item 1; §5.10; Q10; K4 | DR-08 |
| MF-20 | low | Line citations drift; no rule for re-anchoring; duplicated facts have already diverged | header; G1; §5.11; §5.13; K5 | DR-09, DR-07, SI-02, DL-15 |
| MF-21 | low | Evidence links and the A prerequisites exist only in open PR #2957 | §5.13 | DR-10 |
| MF-22 | low | The entries split promises an editor ≥55 columns where only 50 exist | §3.4; §3.9.2 | SI-10 |
| MF-23 | low | R6 promises a `‹ Roleplay` stage that R38 defers; B0's `StageReturnBar` has no user in B | R6; R17; §2.1; B0 | SI-11, LC-10 |
| MF-24 | low | "House consistency" claim is false; Library divergences unrecorded | §4.6; §4.13 | LC-01, LC-13, LC-03 |
| MF-25 | low | G1 and G9a state rules the Library does not meet, or state them ambiguously | §5.9 G1, G9a | LC-05, LC-04 |
| MF-26 | low | Mockup grips do not match the shared renderer, which trims the noun at 24 rows | MK1, MK2, MK9a; §1.1 | LC-11 |
| MF-27 | low | §2.3 relies on an item "prev/next" that does not exist | §2.3 line 667 | UX-04 |
| MF-28 | low | Landing row counts contradict MK8; empty-profile state undefined | §3.10.2; MK7/MK8 | UX-09 |
| MF-29 | low | B6 aggregates: incomplete invalidation, regex validity is not a COUNT, index gate incomplete | §1.4.2; B6 | SF-10, SI-05, DL-08 |
| MF-30 | low | Persistence edge cases: `relevance` sort persisted; per-kind sort values; lost debounced writes | §2.11 | SF-11 |
| MF-31 | low | The ADR-011 capture list misses B11's new 3-second timer | §2.13; §5.4 item 5 | SF-13 |
| MF-32 | low | B7's tests cover the +TRY breakpoints that are B8's scope | B7 tests; B8 | SI-06 |
| MF-33 | low | §5.3 does not register B6's interim narrowing or require B6 and B7 to ship together | §5.3 | DL-10 |

### MF-01: Boot CSS has 50 bytes of headroom, so B0 as written fails CI (high)

**Problem.**
- The spec's "≈584 KB" figure comes from `boot_css_bytes.json`'s `total` (583,433), which is a stale snapshot.
- The test's own census (`_boot_parsed_css_census()`, `Tests/Performance/test_boot_css_byte_budget.py:174`) measures 608,088 B at 84247cb843 (2 B headroom) and 608,040 B on dev at eba4305d83 and 6958e8dfa9 (50 B headroom). The ceiling is 608,090 B.
- B0 moves `css/features/_library.tcss:109-176` (1,173 B without comments, currently in a lazy sheet) into a new boot component. That fails `test_boot_parsed_css_bytes_stay_within_budget`.
- Commit abe5bb548b on dev records the same failure on PR #2941: "dev had 2 B of headroom … the constant is not raised".
- B1 frees only about 216 B, and B5b is the first large saving.

**Edit.**
1. **§5.10 "Boot parsed CSS" row.**
   - "Today" cell: "608,040 / 608,090 B on origin/dev @ eba4305d83 (50 B headroom; 608,088 B = 2 B at the design base). Measure with `_boot_parsed_css_census()`. The `total` in `boot_css_bytes.json` (583,433) is a stale snapshot, not headroom."
   - "Expected effect" cell: "Every slice is net ≤0 boot bytes in its own PR. B0 adds nothing to boot. B1's never-composed AT deletions free ≈216 B. B5b's −1,898 B is the first large saving."
2. **§2.12 item 2.** Replace it with:
   > **Neutral shell rules stay out of boot (ADR-097: defer first).**
   > - `AdaptivePaneShell`/`AdaptivePaneGrip` carry a destination class.
   > - The Library's block stays in its lazy sheet, re-keyed to the Library destination class.
   > - Roleplay's equivalent goes in `_roleplay.tcss` (B1).
   > - A parity test pins the two rule sets equal after prefix normalisation.
   > - With destination-prefixed selectors there is no cross-split overlap, so the boot placement reason (`build_css.py:438, 894-906` at base) no longer applies.
   > - Fallback: ship the boot component together with an equal or larger boot deletion in the same PR (the order then becomes B1 → B0). An owner-signed ADR-097 exception is the last resort.
3. **B0 gate (line 2355).** "Boot CSS net ≤0 in this PR, measured with `_boot_parsed_css_census()`."
4. **§2.1, line 626.** Delete "Side benefit: Library grips stop depending on visit order … (§2.12)."
5. **§2.12 item 3.** Append:
   > - B's modals and choosers (delete modal, New chooser, Import picker, tag picker, Buddy chooser) carry no rule-bearing `BUNDLED_CSS`. Their rules live in `_roleplay.tcss`, which is loaded app-wide before any of them can open.
   > - Any harness-fallback minimum in a widget or modal `BUNDLED_CSS` costs boot bytes, so it must be zero or offset in the same PR.
   > - Centre a modal through a `roleplay-`-prefixed class it adds to itself. A type-only selector has no owner token, so it stays in boot (abe5bb548b).
   > - A modal later opened from outside Roleplay uses `build_css.MODAL_OWNED_SPLIT_SHEETS` (on dev since TASK-33628.2).
6. **§5.4.** Add a DoD line: "Boot CSS net ≤0 per PR (ADR-097 order: defer, shed, owner exception)."

Line 77 and line 2305 ("B0 ∥ B1") stay as they are under the defer option.

### MF-02: Load the Roleplay sheet through the app's route map, not `CSS_PATH` (high)

**Problem.**
- `Tests/UI/test_css_build_integrity.py:1113-1155` (base and dev) requires every non-agentic split sheet to be in `TldwCli._SCREEN_OWNED_ROUTE_CSS`, with the message "Screens must NOT take it onto CSS_PATH instead". A companion test, `:1160-1185`, bans `screen_feature_*` sheets on a screen's `CSS_PATH`.
- The reason (`app.py:2496-2514`): Textual loads `CSS_PATH` under every app, including the unstyled test harnesses, and this flipped three geometry tests on 2026-09-04.
- About 456 Roleplay test instantiations run on that unstyled tier.
- None of these integrity tests is in the PR-gate census, so a violation would show only in nightly runs.

**Edit.**
1. **§2.12 item 3 (line 885) and B1 scope (line 2369).** Replace "loaded from `PersonasScreen.CSS_PATH`" with:
   > registered as `TAB_PERSONAS: ("screen_feature_roleplay.tcss",)` in `TldwCli._SCREEN_OWNED_ROUTE_CSS` (app.py:2509 at base, :2477 on dev), and loaded by `_ensure_screen_owned_css` (`app_navigation.py:52`) on the first Ctrl+4. Never on `PersonasScreen.CSS_PATH`.
2. **B1 scope.** Add three changes:
   - add `PersonasScreen` to the screens checked by `test_screens_do_not_take_owned_sheets_onto_css_path`;
   - add `"screen_feature_roleplay.tcss"` to `_SPLIT_SHEET_OWNERS` (`Tests/UI/test_consolidated_css_harness.py:285`), listing only harnesses that really load the sheet. Do **not** list `PersonasScreen`: that would exempt the bundle-only `StyledPersonasTestApp`, which is exactly the harness the scan should catch;
   - add an `app.py` row to the §5.11 collision table (app.py is hot: #2862, #2949).
3. **§2.13 "Lazy CSS" row (line 908).** "parsed once on the first Ctrl+4 via the route map".
4. **§5.7.4.** For every slice that touches CSS, add `Tests/UI/test_css_build_integrity.py`, `test_consolidated_css_harness.py`, `test_css_staleness_manifest.py` and `test_component_pattern_governance.py`.

### MF-03: Name two harness tiers, and create them in B1 (high)

**Problem.**
- §5.7.1 says `StyledRoleplayTestApp` "loads the consolidated bundle (`Tests/UI/consolidated_css.py:122`)". Line 122 is `ConsolidatedCSSApp`, which loads **no** app bundle.
- B2 says the harness is built on `_build_test_app`, a real `TldwCli`, and that it "consolidates" the two mock-delegating harnesses. That means rewriting the fixtures of about 456 tests, which K5 does not count.
- `TldwCli.CSS_PATH` does not include the `screen_feature_*` sheets (`app.py:829-839`).
- B1 already asserts painted rows ("exactly 1 row at 80x24", "first list item on row 19"), but the harness arrives only in B2.

**Edit.**
1. **§5.7.1 "Mounted Pilot, real CSS" row, and B2's harness bullet.** Replace with:
   > Two tiers in `Tests/UI/roleplay_frame_harness.py`, created in **B1** (test-only; B0 stays Library-only):
   > 1. `RoleplayMockApp`: today's delegating `PersonasTestApp`, moved verbatim and re-exported from its old locations. Existing tests are untouched. Its styled variant `StyledRoleplayMockApp` uses `CSS_PATH = list(APP_STYLESHEETS)` (`consolidated_css.py:64`: the bundle plus every split sheet).
   > 2. `roleplay_full_app()`, modelled on `Tests/UI/full_app_destination_context.py`: a real `TldwCli` from `_build_test_app`, with a temporary ChaChaNotes seeded through the app's own APIs. It arrives through the real route, so `_ensure_screen_owned_css` runs (`:106-112`). Use it for journeys, volume tests and save→reload persistence.
   >
   > Unstyled-tier tests never assert geometry.
2. **B2 tests (lines 2424-2426).** Delete "It consolidates …". An existing test moves only when a slice re-pins it.
3. **B1.** Add a self-test: deleting one `_roleplay.tcss` header rule turns a header geometry assertion red under both styled tiers. Move the size-matrix fixture and `assert_painted_inside` into B1.

### MF-04: Commit a portable live harness now (high)

**Problem.** As of now:
- `launch.sh` and `seedrun.sh` hard-code absolute paths, and `seedrun.sh` ignores `APP_WT`.
- `seed.py`, `gapcheck/scale_seed.py`, `persona_more.py` and `lore_fix.py` refuse to run outside the scratch `ROOT`.
- `mkprofile.sh` rewrites `config.toml` (`} > "$CFG"`) on **every** launch, including `REUSE=1`. Recipe line 2831 ("restart: preferred state restored (B7)") therefore wipes the `[roleplay.reader]` section it is meant to check.
- PR #2957 carries only `HARNESS.md`.
- The scratch directory is under `/private/tmp`, which macOS purges.
- Putting `runs/` (391 MB) or the profile databases under `Docs/` would not be git-ignored.

**Edit.** Replace the K14 bullet (line 2820) and the K14 row (line 2998) with:

> **Now, not "before B1".** Commit the harness *scripts* under `harness/`, in PR #2957 or a small docs follow-up.
> - Parameterise them by:
>   - `HARNESS_STATE` (default `<main checkout>/.worktrees/.rp-harness-state/`, which is git-ignored and outlives slice worktrees);
>   - `APP_WT` (honoured by both `launch.sh` and `seedrun.sh`);
>   - `PY`.
> - Every `ROOT`/"REFUSING" guard keys on `HARNESS_STATE`.
> - Add `make_profiles.sh`: empty skeleton → first boot → `preseed` snapshot → `seed.py` → `golden` → volume copy. `seed.py` generates its own card fixtures.
> - `launch.sh` skips `mkprofile.sh` when `REUSE=1` and `config.toml` exists; it then rewrites only the path keys.
> - Add `launch.sh --self-test`: launch → write a sentinel `[roleplay.reader]` key → quit → relaunch with REUSE → the key is still present.
> - Profiles, runs and captures never go under `Docs/`.
>
> K14 mitigation: "done before planning starts".

### MF-05: Show an authoring view only during a work session (high)

**Problem.**
- §2.3 starts a session only on a listed set of intents.
- B7's acceptance (line 2616) says `< >` never starts one, yet `< >` and ←→ (line 1830) step onto Edit from Card.
- Restore and list loads open "the remembered view" (line 2161), and a return starts INACTIVE (line 2144).
- The Card row (line 1021) adds a third view-memory rule, with no stated precedence.
- Result: Edit or Test chat can appear in BROWSE geometry (content 46 at 120x36). That is Appendix B's blocker ("Test chat starved to 46"), and it contradicts line 683 ("56-column minimum is met everywhere").
- J4 (line 2807) walks straight into it.
- The `‹ Kind` crumb from Edit has the same hole.
- ItemMemory is placed in B5b's scope (line 2521) but in B6 in the module map (line 2965).

**Edit.**
1. **§4.1.** Add:
   > **I11. An authoring view is shown only during a work session.**
   > - Authoring views: Edit, Look, Tool policy, Options, the entry editor, Test chat, Try it.
   > - While INACTIVE, only an authoring intent shows one, and it starts the session first.
   > - Every end event (the crumb, a kind switch, deleting the loaded item, Start) leaves a read view showing. Any draft stays guarded, signalled by the title-row word `unsaved`.

   Reword I8 to: "**Within a session**, view switches are lossless and never move panes (R9)."
2. **§4.11 and §3.5.1.** Add:
   > Restores and list loads while INACTIVE never land in an authoring view. They open the last *read* view used for that kind, else the kind default. This is §3.5.1's Card rule, generalised to all four kinds.

   Line 2161: "in the remembered view" → "in the remembered **read** view".
3. **View memory.** Keep one view memory: `ModeMemory.view`, read views only. Drop `ItemMemory.view` and `ItemMemory.focus_id`; ItemMemory keeps `entry_anchor` and `conversation_id`. Place ItemMemory in one slice, B5b, and make the B5b scope and the module map agree.
4. **`< >` and ←→ while INACTIVE.** Follow OD-1. With the recommended option, B7's line 2616 stays as written; add "while no session is active, `< >` and ←→ step over authoring views".

### MF-06: No transition may focus a widget in a closed pane or hidden list (high)

**Problem.**
- At 120x36, EDIT closes the list (`– | – | 110`).
- Rung 5 (line 1808) and the Cancel/Discard/Done row (line 2164) send focus to "the list row if the edit began there", which is in a closed pane.
- Closed shell panes are also disabled (`library_adaptive_reader_shell.py:389-392`), so the `focus()` call silently fails. Textual then falls back (`screen.py:1020-1060`) to a "visible" sibling chosen by `focusable`, which ignores `display` (`widget.py:2368-2375`). Focus can land on a hidden button or on `None`.
- Rung 3 (line 1806) sends an entry-editor field to "the entries list at the anchored row". In drill-in that list is hidden but *not* disabled, so ENTRIES keys (`space`, `del`, `n`, Enter) act on a list the user cannot see.
- This is the J2 path at the design centre.

**Edit.**
1. **I9.** Append:
   > A named target whose pane, stage or drill-in container is not displayed, or is disabled, falls back in this order: the invoker → the view strip → the work pane's F6 target.
2. **Rung 5 focus cell and §4.11 Cancel/Discard/Done row.** "the invoker if its pane is displayed (the list row in BROWSE, or at ≥142 where EDIT keeps the list), else the view strip".
3. **Rung 3, entry editor.**
   > - Split: the entries list at the anchored row.
   > - Drill-in: the drill-in heading's focusable `‹ Entries` control (WORK_BTN context, no letters armed, footer `esc back to entries`). Rung 4 closes the drill-in from there.
   >
   > T5 gains a drill-in variant of its after-esc chip.
4. **§5.7.2 item 9.** Replace with:
   > After every transition, `app.focused` is not None, it and every ancestor are displayed and not disabled, and it has a non-empty region. Run at 120x36 for:
   > - `e` from the list → Esc, Esc;
   > - `e` → Ctrl+S → Done;
   > - Esc from a drill-in entry field.
5. **K21 (line 3005).** Add:
   > The mitigation depends on this focus fallback. Harness journey at 120x36: `e` from the list → Ctrl+S → Done → focus on the strip, footer shows `esc to list`, and one Esc returns to the list.

### MF-07: The guard needs per-domain save and discard executors (high)

**Problem.** At 84247cb843 (unchanged on dev):
- `_save_aggregate_roleplay_drafts` saves the form by calling `action_personas_save()` (PS:16066).
- That action returns unless `_edit_mode` is create or edit (PS:16301), and presses Save only if the view is displayed (PS:16303-16312).
- B makes view switches lossless (I8) and makes Ctrl+S focus-scoped (R41, §4.6). A guarded switch fired from Card, Chats, the list or the rail therefore saves nothing. The domain stays dirty, and the recovery dialog (Retry or Stay) loops.
- The discard branch (PS:16166-16178) resets only the character and persona editors.
- B9 adds `entry_form_dirty` and `lore_settings_dirty` to the **predicate only**. "Discard and continue" therefore leaves `RoleplayDraftSnapshot.is_clean` False (`character_conversation_navigation.py:213-230`), and the user cannot leave.

**Edit.** Add to §3.12:

> **Executors.**
> - The aggregate has one `save_domain(d)` and one `discard_domain(d)` per domain: character form, persona form, character visuals, persona visuals, attachments, entry form, lore/dictionary Options.
> - They do not depend on focus or on which view is showing. Each calls the editor's own save or discard path, with the same validation as its commit button, and never calls `roleplay_save` or Ctrl+S.
> - `_run_guarded`, `confirm_navigation`, the inline entry guard and the quit prompt (MF-17) all use them.
> - A failed save (validation or `ConflictError`) never continues. It opens `RoleplayDraftRecoveryDialog` (`character_conversation_navigation.py:285-310`, Retry · Stay). Stay opens the view that holds the domain, with focus on the first invalid field.
> - Discarding every domain leaves `is_clean` True.

Supporting edits:
- **B3 re-pin (line 2469).** "The guard-path caller (`_save_aggregate_roleplay_drafts`) moves to the executors, not to `roleplay_save`."
- **B5b tests, extended in B9 and B11.** For each domain: make it dirty, switch to Card, Test chat or Chats, then switch kind. Then check:
  - "Save and continue" saves the record and the switch happens;
  - "Discard" leaves the domain clean and the switch happens;
  - an invalid field blocks the switch and shows the recovery dialog.
- **B9 acceptance.** Cover "Save and continue" and "Discard and continue", not just the trigger, for entry and Options drafts made from a non-editor view.

### MF-08: Untrusted text renders literally on every surface (high)

**Problem.**
- Reproduced in the venv (Textual 8.2.8):
  - `Static`, `Label`, `Button` labels and `border_title` raise `MarkupError` on `x [/]`;
  - a markup-on Static showing `[@click=app.quit]Ada[/]` renders as "Ada", and clicking it runs the action. With `markup=False`, nothing runs.
- Today `#personas-selected-name` and the `DestinationHeader` subtitle and status (`workbench_widgets.py:203-241`) are markup-on.
- B puts item names in the title row and header subtitle, the server label in the header status, tag text in border titles, and conversation titles (often written by an LLM) on the landing.
- R33 and K11 cover only OptionList prompts and toasts. Appendix B resolved only "toast markup".

**Edit.** Replace R33 with:

> **Untrusted text is literal everywhere.**
> - Applies to all text the app did not author: item, book, entry, tag, persona and voice names; conversation titles; server labels; search and filter text; file names; exception text.
> - Rendering rules:
>   - `Static` and `Label` use `markup=False`;
>   - OptionList prompts, border titles, `Button` labels, tooltips and footer chips receive `Content(text)` or `textual.markup.escape(text)`, never a raw str;
>   - toasts use `notify(..., markup=False)`.
> - Roleplay passes escaped text to `DestinationHeader` at its own call sites. A literal mode for the shared header is a follow-up (FU-2).

Supporting edits:
- **B1.** Add a hostile-name test that each later slice extends.
  - Fixture: a card, lore book, dictionary, entry key, tag, persona, conversation title and server label set to `[/]`, `[b]x` and `[@click=app.record('x')]N[/]`.
  - Assertions: every Roleplay surface shows the literal text; no `MarkupError` is raised; no rendered segment carries `@click` meta; clicking each name runs nothing.
- **K11.** Extend to "every Static, border, button, tooltip and chip surface".

### MF-09: The entry form owns its entry, not the cursor (high)

**Problem.**
- In both entry widgets, `RowHighlighted` refills the form, including on programmatic cursor moves during reloads (`personas_lore_detail.py:567-581`; `personas_dictionary_detail.py:738-748`).
- Update targets `selected_entry_id`, which is read from `table.cursor_row` (`personas_lore_detail.py:405-414, 597-610`).
- Once `entry_form_dirty` exists, either of two things happens:
  - arrows, filtering, `space`, `del`, Move or A's re-select silently overwrite the draft; or
  - if B honours I6, Ctrl+S writes #200's text onto the #150 under the cursor.
- The spec (§3.9.2 "the editor always shows the anchored entry", §3.9.4 re-select by id) does not say which, and rung 4 does not say what happens to a dirty form.

**Edit.** Add to §3.9.5 and B9:

> - The entry form owns `form_entry_id` and its version. They are set only by an explicit open (Enter, click, prev/next), after the inline guard; `+ New entry` sets None.
> - `RowHighlighted` never fills the form.
> - Update, Ctrl+S and the editor's Delete… target `form_entry_id`, never the cursor.
> - Reloads, filter changes and A's re-select move only the cursor and the anchor. They refill the form only when it is clean and `form_entry_id` has changed.
> - If `form_entry_id` was deleted elsewhere, the form says "This entry was deleted · Keep as new / Discard".
> - Rung 4 with a dirty form opens the inline guard (Update / Discard / Stay).

**B9 acceptance.** With a dirty form, the form text is byte-identical after each of these: arrows, 5 filter keystrokes, `space` and `del` on another row, Move, and a background reload. Arrowing to another row and pressing Ctrl+S updates the form's entry and leaves the cursor row unchanged in the DB.

Coordinate with TASK-33782 (RP-014 anchor) before B9 (FU-4).

### MF-10: Split Delete into loaded and not-loaded cases; complete the trigger table (medium)

**Problem.**
- §4.11's Delete row (line 2167) always loads "the neighbour at the same index" and ends the session.
- B's `del` acts on the cursor row (§4.6), which I6 makes distinct from the loaded item. Deleting another row would therefore replace a dirty loaded item without the ADR-046 dialog.
- §3.12, which calls itself "the complete trigger table", has no rows for Delete, Duplicate, same-kind New, Import… or dictionary Revert…. All of these are guarded today (PS:7483-7517, 14747-14763), except Revert, which reloads Options with no draft guard (PS:6376-6440). B5b's parametrised test mirrors that table, so it would not catch these guards being dropped.

**Edit.**
1. **§4.11 Delete row.** Split it into two rows:
   - **The deleted set includes the loaded item:** the neighbour at the same index loads, never row 0; an empty kind shows its empty state; the session ends; focus goes to the list at the neighbour.
   - **Only other rows are deleted:** the loaded item, its view, its draft and the session are untouched; the cursor moves to the deleted row's neighbour; focus stays on the list.
2. **§3.12.** Add these rows:
   - Delete of the loaded item: guard first, then the delete modal (today's order). The modal names it "(open · unsaved changes are discarded)".
   - Delete of other rows: no veto.
   - Entry delete of the entry held in a dirty form: inline guard. Another entry: no veto. The copy says "permanently", because `world_book_manager.py:644-658` is a hard DELETE.
   - Duplicate, New… in the same kind, Import… (before the picker), and Revert… (guard, then the revert confirm): Yes.
3. **Wording.**
   - The `o` row's "Guard: none" → "the leave guard (`confirm_navigation`)".
   - The runtime-source row → "not reachable while Roleplay is active (it is changed in Settings, behind the leave guard)".
4. **B5b test.** Add about 8 parametrised cases, plus a Pilot test: dirty Edit + `del` on another row keeps the draft, the editor width and the session.

### MF-11: Make list windows safe against changes between fetches (medium)

**Problem.**
- Windows append with `add_options` (line 456). Textual 8.2.8 appends options one at a time and raises `DuplicateID` partway through a window (`_option_list.py:394-404`). Workers default to `exit_on_error=True`.
- An insert ahead of the loaded end (import, or a card saved from Console) shifts the next OFFSET window, which produces a duplicate and possibly an app exit.
- A delete ahead skips a row, which is RP-080's class of bug.
- There is no load-through rule. At 348 characters, a selection past row 200 cannot be restored, which regresses report.md:908 ("page 5 and the selection came back"), and End stops at row 200.

**Edit.** Add to §1.5.6:

> - `add_options` receives only ids not already present. A duplicate is logged, never raised.
> - Every window carries (kind, query, sort, tag, runtime source, list generation) and is dropped unless all of them still match.
> - Any Roleplay write, or a detected external change, reloads from the anchor (`set_options`, then restore the cursor by id).
> - **Load-through:** a move to an id or index past the loaded end fetches windows up to the target in the window worker, then moves the cursor. This covers End/Ctrl+End, restore of cursor, anchor or selection, import selection, deep link, and the delete neighbour. `N of M` uses the COUNT total.

Also:
- **§1.5.4.** Add "cross-kind counts run only when the current kind's result is empty".
- **B2 acceptance.**
  - At 348 characters, a selection past row 200 restores.
  - After an insert ahead, a delete ahead and a scroll to the end, every id appears exactly once and no exception escapes.
  - A late window from a superseded query is dropped.

Keyset paging is optional.

### MF-12: The one Import picker keeps every format and bounds its sniff (medium)

**Problem.**
- Today's character picker accepts `.webp` cards, `.md`/`.markdown` cards and `.tldw-persona-vpack` Buddy packs (PS:569-600, 13736-13776; test_personas_workbench.py:13325). §1.4.6 lists none of them.
- `.md` is ambiguous (card or dictionary), but "Import as" covers only JSON.
- The sniff has no bounds. The existing handlers run `validate_path_simple`, a 10 MB cap, a bounded PNG decode and archive limits (PS:13816-13830; `Actor_Packs/importer.py:57-59`).
- Auto-select after import runs outside `_run_guarded` (PS:13955-13958), and B extends it across kinds.

**Edit.** Rewrite the Import bullet:

> - The picker accepts every format the four pickers accept today:
>   - cards (`.png .webp .json .md .markdown`);
>   - Buddy packs (`.tldw-persona-vpack`);
>   - actor packs (`.tldw-actor-pack`);
>   - lore books (`.json`);
>   - chat dictionaries (`.json .md`).
> - The sniff is bounded and advisory:
>   - it runs off-thread, after `validate_path_simple` and the 10 MB cap;
>   - it reads at most 64 KB: signatures and chunk headers, never a pixel decode;
>   - for JSON it reads top-level keys from a bounded parse, and RecursionError or UnicodeDecodeError means "unknown";
>   - it never opens an archive: actor and Buddy packs go by extension to their own review flows.
> - An ambiguous JSON **or Markdown** file asks "Import as: …".
> - The result is auto-selected only if the aggregate predicate is clean and no newer intent has arrived. Otherwise a literal-text toast says `Imported <name> · in <Kind>`.

**B2 tests.**
- Every format routes correctly.
- An 11 MB file is refused before it is read.
- A deeply nested 9 MB JSON gives "not a Roleplay file" without a crash.
- The sniffer never calls `zipfile.ZipFile` or `Image.load`.
- A draft started during a slow import is never replaced.

### MF-13: Define load and save failure states (medium)

**Problem.**
- §3.2's state words have no failure value.
- "Loading Ada…" with actions disabled (§4.11) has no failure or deleted-id path.
- PS has 20 `except ConflictError` sites, and their copy says "Reselect and try again" (PS:5669-5672, 6739-6743). Under B's guard, that becomes a dialog whose Save fails again.

**Edit.**
- **§3.2.** Add the state word `couldn't open · Retry`.
- **§4.11.** Add a row: "Load fails or the id is gone → title row `Couldn't open <name> · Retry`; actions are disabled with that reason; the list cursor stays; a vanished id leaves the list and memory."
- **ConflictError on save.** Use the spec's own "Changed elsewhere · Reload" (§4.11, line 2154) in place of "Reselect and try again". A failed save inside the guard goes to the MF-07 recovery path.
- **Tests.** Pilot tests with an injected `ConflictError` and with a deleted id.

### MF-14: B1's PS line gate is unreachable as scoped (medium)

**Problem.**
- `personas_screen.py` has 16,533 lines (base and dev) against a 16,436 budget (`test_module_size_ratchet.py:97`), so B1 needs net −97.
- B1's ledger moves about 80 lines, and keeps `_purpose_line_text` until B6.
- §5.7.4 makes the ratchet a gate for B0 too, but B0 never touches PS.
- No PR CI job runs this ratchet, which is how it went red unnoticed.

**Edit.**
- **B1 gate (line 2387).** "PS ≤16,436, measured. Precondition: #2862 (−117) has landed, or a zero-behaviour ratchet-repair PR lands before B1 (FU-3)." Do not move the footer-hint builder in B1, because B3 replaces it.
- **§5.7.4.** "Until the PS row is green, judge `test_module_size_ratchet.py` as 'does not grow versus the paired base arm'. B0 is exempt."

### MF-15: Keep the PR-gate census in step (medium)

**Problem.**
- `scripts/ui_pr_gate_census.txt:83-90` lists eight Roleplay files, including `test_personas_library_pane_paging.py`.
- B2 deletes that file, and `check_ui_pr_gate_census.py` (preflight.sh:126 and the required job) fails on a missing path.
- None of B's new test files would gate PRs. PS had 73 commits from other work since 2026-08-15.

**Edit.**
- **§5.4 DoD.** Add:
  > Each slice adds its new fast test files to `scripts/ui_pr_gate_census.txt` and raises `MINIMUM_FILES`, keeping the UI Fast Lane within 20 minutes. A slice that deletes or renames a censused file replaces the entry in the same commit (B2: paging → `test_roleplay_item_list.py`).
- **Targeted test sets.** Add `test_personas_preview_restore.py` to B4, `test_personas_dictionary_validation.py` to B9, and `test_personas_character_generation_ui.py` to B11. Add `test_personas_workbench_foundation.py` and `test_personas_library_rail_focus_outline.py` to §5.7.4.

### MF-16: Measure latency in a way CI can trust (medium)

**Problem.**
- B2's "keystroke → painted rows ≤250 ms at 348 and 5,000" names no measurement method, and no 5,000-row dataset exists (the volume profile is 348).
- lessons-testing-evidence.md:9369-9385 forbids absolute latency taken through `pilot.press()+pause()`. :13200-13225 records a wall-clock assertion that failed on 4 of 6 runs.

**Edit.** Rewrite B2's end-to-end bullet in two parts:

> **(a) CI.** Deterministic counts on `roleplay_full_app()`, with a pytest fixture seeding 348 and 5,000 characters. Per settled keystroke:
> - exactly one window query (≤200 rows);
> - one `set_options`;
> - 0 widget mounts and 0 recomposes;
> - prompts built equal to window rows.
>
> Stall detection uses only the existing discriminating heartbeat form (`test_personas_deferred_center_views.py:318-358`).
>
> **(b) Evidence, not a pytest threshold.** In-app `perf_counter` from the key event to the first idle after paint. Report the median of at least 10 ABBA-interleaved live-harness runs at 348 and 5,000, recorded in the §5.10 ledger.

Apply the same split to B6's counts worker and to the PERF-22 arrival row.

### MF-17: Record the quit-path work that is in flight (medium)

**Problem.**
- PR #2949 (open) makes Ctrl+Q a priority binding that runs under modals. It routes every prompt through `await_quit_prompt` and adds a second ADR-031 refinement, which overlaps G9.
- TASK-33622.14 (p1, data loss: PS has no `confirm_quit`) exists only in open PR #2952. It needs a helper module, which would breach the 557/557 pre-import census.
- The spec mentions it only in parentheses (line 1411), and §5.5 and §5.6 have no rows for either.

**Edit.**
- **§3.12, new row "Quit (Ctrl+Q)":**
  > Guarded once TASK-33622.14 lands. Its hook calls `roleplay_has_unsaved_work()` and the MF-07 executors, so B9's domains are covered with no second list. Prompts go through `await_quit_prompt` (#2949).
- **§5.5.** Add TASK-33622.14 as **S** for B9. Whichever lands second extends the quit prompt to B9's domains. B9's Ctrl+Q Pilot test applies once 33622.14 is on dev.
- **§5.6.** Add two rows:
  - PR #2949: G9 rebases onto its ADR-031 text. A B modal that guards its own close answers `confirm_quit` via `confirm_quit_discarding_edits`.
  - TASK-33622.14: if it lands after B1, its hook imports the predicate from `PM/roleplay_frame_state.py` (census +0).

### MF-18: Accommodate ADR-210 (low)

**Problem.**
- ADR-210 was accepted on dev an hour after the base (merge 922440b93e). It ships a 1-row nav below 35 rows for every destination (step 8, TASK-33627).
- It gives Console a separate one-row header with no title.
- It edits DESIGN.md:107 and the header contract, the same text that G11 and G12 edit.
- The spec never mentions it.
- The design centre (36+ rows) is unaffected. The 80x24 and 60x24 numbers, mockups and absolute-row asserts all shift by 2 rows.

**Edit.**
- **Line 223.** "rows 1-N nav bar (shell; N = 3 today, 1 below 35 rows once ADR-210 step 8 / TASK-33627 ships)". R = H − (N + 4).
- **§2.10.** Footnote the 80x24 column: "with ADR-210's compact nav: first list item row 5, R 19, list rows 18".
- **§5.7.2 item 1, and B1's "row 19".** "relative to the measured bottom of the shell nav and the header, never absolute screen rows".
- **§5.6.** Add a row for ADR-210 step 8: "whichever lands second re-captures MK9a/MK9b and B7's 80x24/60x24 screenshots; no design change".
- **Line 273 and G12.** Remove "Console" from "used by Console, Lab and Roleplay". G12 → "an inline one-row DestinationHeader variant for Lab and Roleplay; Console's one-row header is ADR-210's separate exception. B1 writes both as sibling entries in one 'Destination header variants' list at DESIGN.md:328-330; whichever of B1 and ADR-210 step 8 lands second rebases onto the other."
- **G11.** "Add Roleplay as a sibling of ADR-210's Console variant at :107."
- **"Rules confirmed compliant" table.** Add ADR-210. B's ADR-011, 015, 031 and 120 amendments append to the existing "Amended by" lines.

### MF-19: Refresh the ratchet numbers (low)

Measured on base and dev:

- **Line 873.** "273/274 (one slack)".
- **Line 876.** "B2: 5 rules in `PersonasLibraryPane`":
  - pagebar `Button`;
  - toolbar+filterbar `Button`;
  - the stacked-controls pair;
  - `#personas-library-rows ListItem Static`;
  - the recovery-row Static.
- **Line 880.** "Up to 11 rules (273 → ≈262)".
- **§5.10.**
  - Broad-selector row: "273 / 274".
  - Pre-import LOC row: "411,958 on dev (snapshot 411,886)".
  - Largest-route row: add "115,452 if #2862 lands first (Library 105,452)".
- **Q10 and K4.** "the 59 LOC is measured at #2862's merge-base; plan for zero".

### MF-20: Add a re-anchoring rule instead of re-numbering (low)

**Problem.**
- Anchors drift faster than a refresh can keep up: `action_show_workbench_help` was at 1689 (base), then 1686, then 1681 on current dev.
- `build_css.py`, `app.py` and four ADRs have shifted.
- The two copies of the gains table have already diverged: the summary has a "Read-only card width" row, and the copy in §0.1 does not.
- PR #2953 reindents AT:634-826 (whitespace only), and its head measures 607,171 B boot CSS.

**Edit.**
- **Header "Base" bullet.** Add: "Line numbers are pinned to 84247cb843. Each plan re-anchors against current origin/dev by symbol (`build_css.SCREEN_OWNED_SPLITS`, `TldwCli.BINDINGS` f6, `_show_generic_screen_help`, `action_personas_save`). AT rules are cited by selector and tests by name, never by line."
- **G1 numbering note.** "Number assigned at merge after the all-remotes sweep (lessons-backlog-hygiene). 198 and 210 are on dev; 199, 200 and 204 are claimed on other branches."
- **§5.11.** Add an `AT` row for PR #2953: "reindent only; rebase and re-measure boot bytes". Label `AT:1354, 2536` as "#2862's hunk anchors at its merge-base ef831d9f38".
- **§5.13.**
  - Add: "One home per fact; other mentions cite the R, Q, I or G id."
  - Add a short normative index (id → owning section) at the top.
  - Keep one gains table and cite it from the other place.
- **K5.** "New tests go in per-slice files; plans cite test names, not lines."

### MF-21: Merge PR #2957 before or with the spec PR (low)

The spec's evidence links and TASK-33781..33793 exist only on #2957's branch.

**Edit.** Add to §5.13: "Merge #2957 before, or together with, this spec's PR. B0's plan starts only once both are on dev."

### MF-22: Fix the entries split arithmetic (low)

**Problem.** At `w` = 100, content is 96, so a 46-column list leaves an editor of 50, not ≥55. The split resolves to exactly `w` = 100 at 142 and 175-195, and +TRY-E at 200.

**Edit.** Lines 1006 and 1260: "list = clamp(round(0.45·(w−4)), 44, 60); editor = the remainder (≥50)". The 160x45 figure (editor 58) is unaffected.

### MF-23: Reword R6 and move `StageReturnBar` to the shed (low)

**Problem.**
- R6 says "`‹ Roleplay` exists only below 64 columns", but R38 and §2.8 defer that stage to a follow-up. In B it exists nowhere.
- B0's promotion of `StageReturnBar`, and its edit of `library_emergency_return.py`, therefore have no Roleplay user in B. Their only payoff is the post-#2862 −1 module shed.

**Edit.**
- **R6.** "`‹ Roleplay` arrives with the single-stage follow-up (R38). In B, below 64 columns, the `‹ <Kind>` crumb and Esc rungs 7 and 10 are the way back."
- Remove `StageReturnBar` and the `library_emergency_return.py` edit from R17, §2.1 (lines 612-613), B0 (line 2337) and the module map. Introduce `StageReturnBar` at the post-#2862 shed, where `library_emergency_return.py` folds into `adaptive_pane_shell.py`.
- The module ledger is unchanged (Q10's +3/−1 holds). B0 touches one Library module fewer.
- Update the "`LibraryEmergencyReturn` reuse → B0" entry in Appendix B to match.

### MF-24: Record the real key divergences from the Library (low)

**Problem.** §4.6 "House consistency" (lines 1859-1862) names only Library `t` as a divergence. Several Library keys mean something else:

| Key | Library meaning | Source |
|---|---|---|
| `s` / `space` | Select mode / select row | LS:703-704 |
| `[ ]` | next/previous item | LS:1156-1157 |
| `l` | read later | LS:1183 |
| `c` | Use in Console / resume conversation | LS:1184, 1190-1195 |
| `ctrl+f` | Reader Find | LS:1202 |
| `m` | toggle reviewed | LS:1207 |
| `e` | export selected | LS:1039-1042 |

In addition:
- The Library advertises `ctrl+n`, not `n`, for New (LS:1257-1262), and `u` exists only on Search/RAG (LS:1244).
- DESIGN.md:125 says a key must mean the same thing across destinations.
- Library Media loads a row once the cursor settles on it, with no Enter needed (LS:10710-10720). This divergence from I6 is not recorded either.
- The Library editor Escape takes one press back to the list (LS:1075-1076, declared before the blur binding at LS:1128). This divergence is not recorded either.

**Edit.**
- Replace the bullet with a table headed "Same key, different meaning in the Library". Cover m, s, space, l, c, `[ ]`, e and t, and mark each one:
  - pre-existing: ADR-152 kind keys, F-040 marks;
  - new in B: `e`, `t`.

  Note that `ctrl+f` means find on both screens, scoped differently.
- Add these rows to §4.13:
  - "Library Media loads after the cursor settles; Roleplay loads on Enter (loads are guarded and fenced)";
  - "Library editors: one Escape = ‹ Back to list. Roleplay: field, then cancel, then list, because Roleplay editors hold guarded drafts and the session closes the list".

See OD-2 for convergence.

### MF-25: Scope G1 and clarify G9a (low)

**Problem.**
- G1 promises "one collapse grammar" and "one round border per pane" in a shared ADR. The Library's ordinary routes keep a text "Collapse" button, a 3-cell handle with unpersisted state, and three solid border layers (`library_rail.py:538-576`; `_library.tcss:49-84`; capture `library-landing-120x36.txt`).
- In G9a, "only while that Save is offered and enabled" can be read as gating the binding. That conflicts with §4.2, where `check_action("roleplay_save")` returns True everywhere.

**Edit.**
- **G1.** "One collapse grammar and one round border per pane **on adaptive-shell routes**. The Library's ordinary routes and its outer frame are recorded exceptions until FU-1."
- **G9a.** Reword so that "only while" governs what the key **runs** (the visible, enabled Save path), not whether it is bound. State either:
  - the focus-scoped rule as the house rule, with the Library skill save (LS:1047, LS:25243-25251, not focus-scoped) grandfathered; or
  - the two tiers, explicitly.

  Library alignment goes to FU-1. Q5's Roleplay behaviour is unchanged.

### MF-26: Regenerate the mockup grips from the renderer (low)

**Problem.**
- The renderer draws one arrow at mid-height on items grips (two only on the rail grip) and trims the noun to the rows above the first arrow (`library_adaptive_reader_shell.py:187-205`).
- MK1 draws two arrows on the items grip.
- MK9a (80x24) shows the full "Characters", but the real grip reads "Character" there.

**Edit.**
- Regenerate the grips in MK1, MK2 and MK9a from the renderer (R40).
- §1.1: "below about 25 rows the grip noun is trimmed to the rows above its arrow".

### MF-27: Remove the dangling item "prev/next" (low)

Item-level prev/next does not exist. `[ ]` switches kinds and `< >` switches views.

**Edit.** Line 667: delete "or prev/next", or say "entry prev/next never changes the item". Optionally document the manual-reopen batch-edit loop in F1: shift+F6 → ↓ → `e`.

### MF-28: Make the landing's caps and empty state explicit (low)

**Problem.**
- §3.10.2 promises Continue = 5 rows and Recently edited = 4. MK8 (120x36) shows 3 + 3 with no hint line and `▾ more`, which is internally inconsistent.
- A fresh profile (3 built-in characters, 0 chats) is undefined.
- "Pick one on the left, or:" is wrong when the current kind is empty.

**Edit.** Add to §3.10.2:
- Continue shows 5 rows (3 when R < 38). Recently edited shows 4 (3 when R < 38). The hint line drops first. Needs attention is never the section cut.
- An empty Continue becomes one line: "Your character chats will appear here. Pick a character, then Chat now." An empty Recently edited is hidden.
- With an empty current kind, the text reads "Import or create one:".

Add an `empty`-profile harness capture.

### MF-29: Fix the B6 aggregate data rules (low)

**Problem.**
- Invalidation hooks (line 2571) miss dictionary Revert, import, delete, Duplicate and Options save. Warnings can therefore stay stale within a visit.
- "Invalid-regex counts" cannot be a COUNT: `entries_json` is one JSON blob per dictionary, and validity is computed in Python (`ChaChaNotes_DB.py:1677`; `personas_dictionary_validation.py:33`).
- B6's index gate cites only the plan pin. CLAUDE.md gotcha 1 also requires a migration, a version bump and an `EXPECTED_CHACHANOTES_INDEXES` entry.

**Edit.**
- **Line 312.** "Totals, chat counts and entry/off counts use COUNT/GROUP BY. Dictionary validity comes from one bulk `SELECT id, entries_json … WHERE deleted = 0`, parsed in the counts worker within a ledger-recorded budget."
- **B6.** "One `invalidate_scent(kind, id)` is called from every Roleplay write path: save, import, duplicate, delete, toggle, entry operations, Options save, Revert."
- **B6 acceptance.** Reverting a dictionary clears its warning without leaving Roleplay.
- **B6 gate.** "Plan first on the existing indexes (`idx_conv_char`, `idx_world_book_entries_book`). Any new index follows gotcha 1 in full."

### MF-30: Close the persistence edge cases (low)

**Problem.**
- `relevance` is set on the first search keystroke and can be picked with `s` (PS:2521-2537, 4059-4063). The "explicit sort change" rule would then persist a search-only order.
- Sort values differ by kind.
- The debounced write is a screen worker, so leaving inside the debounce window loses it.

**Edit.** Add to §2.11:
- `sort` persists per kind, from that kind's own values; `relevance` is never persisted.
- A pending debounced write is flushed on leave and on quit.
- While an environment override is active for a key, the toggle works for the session and is not written. This may be documented in the guide only.

Add a unit test for each.

### MF-31: Add B11 to the ADR-011 capture list, or drop the timer (low)

**Problem.** §3.11's "Saved 14:02 for 3 s" (B11) is a new timer, but §2.13 and §5.4 item 5 omit B11.

**Edit.** Clear the state on the next key or focus change instead, as the L6 chip does, so no timer is added. Otherwise, add B11 to both lists.

### MF-32: Move the +TRY test points to B8 (low)

**Problem.** B7's profile grid (line 2623) includes 222, 223, 233, 234 and 238. These are the +TRY and +TRY-E thresholds, and those postures are B8's scope.

**Edit.** B7 keeps 196, 200 and 220 for BROWSE, EDIT and ENTRIES. Move 222-238 and their reducer rows to B8.

### MF-33: Register B6's interim narrowing and keep B6/B7 in one release (low)

**Problem.**
- Between B6 and B7 the work pane returns to today's widths (≥58 / ≥78 / ≥108 versus B5b's ≥86 / ≥115 / ≥160).
- B5a's Chats split at 160, which appears after B5b, disappears until B7.
- The spec accepts this trade (line 2309, K10). It does not say that a release must not be cut between B6 and B7.

**Edit.** In §5.3, add a note: "B6 and B7 land in one release window (no release cut between them). The Chats split at 160 is absent from B6 until B7."

---

## Recommended improvements (ranked)

| Rank | ID | Title | Cost | Sources |
|---|---|---|---|---|
| 1 | RC-1 | Show the kind empty state from B5b, not B10 | small; about 11 re-pins move from B10 to B5b | UX-06 |
| 2 | RC-2 | Split B2, B5b and B9 where the halves are independent; decide D14 at plan time | about 3 more PRs, 2 with no visual change | DL-08, DL-07 |
| 3 | RC-3 | Queue rules: A tasks count against the one-PS rule; check B5a's gating tasks are alive; B8 reviewed in parallel | text only | DL-09, SI-06 |
| 4 | RC-4 | Leave and return rules that hold if Roleplay becomes a reusable screen | a leave hook plus one test axis | DL-11, SF-08 |
| 5 | RC-5 | Tests must prove they discriminate; compare suites against a paired base run | about 2× runtime on touched suites | DL-12 |
| 6 | RC-6 | Performance acceptance with 5,000 rows loaded (resize, session start and end, End) | one fixture plus measurements | SF-06 |
| 7 | RC-7 | Journey tests exist from the first slice they touch | small scaffolding | DL-13 |
| 8 | RC-8 | Test chat uses the You row's name | small change in the preview controller | UX-12 |
| 9 | RC-9 | Keep the User Guide truthful (retired Send to Console draft; key-table parity test) | one parser test plus guide prose | UX-11, DL-14 |
| 10 | RC-10 | Small copy fixes: position label, "Kind:", Card disclosure at 120, Hide × tooltip | strings and re-pins | LC-08, UX-07, UX-10, UX-13 |
| 11 | RC-11 | Rail details: no key-hint flicker while counts load; mirror the Library's `rail_state` key | spec text | LC-07, LC-12 |

### RC-1: Show the kind empty state from B5b (UX-06)

**Why.**
- Today, Personas, Lore and Dictionaries with nothing selected still show guidance. It comes from the Inspector ("Selected: none … Pick a character or persona") and the Try-it hint (capture `roleplay-dictionaries-nothing-selected-160x45.txt`).
- B5b deletes the Inspector, and B4 moves Try it behind a loaded item, but `PersonasKindEmptyState` arrives only in B10.
- For four slices those kinds therefore show a bordered, wordless work pane. This contradicts §2.6 "Never empty", and B6 makes no-selection restores more common.
- Characters are unaffected, because they keep their existing empty Static.

**Edit.**
- Move the static part of `PersonasKindEmptyState` into B5b: the gloss sentence plus `[ New … ]` `[ Import… ]`, for both "has items, nothing loaded" and "no items". It already lives in `PW/personas_work_pane.py`, which B5b creates. Keep the "3 most recently edited" list, which needs B10's worker, in B10.
- Add a §5.3 row: "B5b: kind empty states replace the Inspector's no-selection text; the work header hides its title row and action bar while nothing is loaded."
- B5b acceptance: each kind with nothing loaded shows its empty-state copy at 120x36 and 160x45.

### RC-2: Split the riskiest slices where the halves do not depend on each other (DL-08, DL-07)

**Why.**
- B2 bundles a widget migration (91 + 46 + 345 refs) with three new flows. If the D14 census forces fallback B at merge, the New chooser, Import… and empty states are thrown away along with the migration.
- B5b bundles the safety work (guard, MF-07 executors, generation fence) with the largest layout change.
- B9 bundles the draft-domain fix that Appendix B rated a blocker with the entries layout.
- Big PS PRs wait longer in the one-at-a-time queue and rebase more often.

**Edit.**
- **B2.**
  - B2a: RoleplayItemList, DB windows, prompts, cursor by id, list keys, ListView re-pins.
  - B2b: New chooser, one Import…, search-aware empty states. These survive a D14-B fallback.
- **B5b.**
  - B5b-1: `_run_guarded` through the aggregate, the executors, the dialog, the generation fence and ItemMemory. No layout change, so no screenshots.
  - B5b-2: work header, strip, More actions, Info, Inspector deletion and its 101-ref re-pin.
- **B9.**
  - B9a: the draft domains, their executors and the ADR-046 amendment. This still needs B9's A prerequisites and B3's projection.
  - B9b: the entries layout.
- **Appendix A, §1.5.6 and the B2 Fallback.** "The D14 dry-run census is step 1 of B2's plan, before any production code; the plan is written for the chosen design." B3 and B10 already put their censuses first.
- Leave B6, B7 and the other slices unsplit unless planning shows a PR is unreviewable.

### RC-3: Make the queue rules explicit (DL-09, SI-06)

**Why.**
- All eleven A tasks are still To Do, and several edit PS. The spec does not say whether they count against "one PS-touching slice in review at a time" (line 2307).
- B5a waits for "the panel API agreed with TASK-22988/TASK-31243". Both tasks say In Progress, but 22988 was last updated 2026-08-27, and 31243's branch is already merged into dev.

**Edit.** Add to §5.1:
- "A tasks count against the one-PS rule."
- "Before B5a, confirm TASK-22988 and TASK-31243 are active. If they are stale, fold their open ACs into B5a instead of waiting."
- "B8 touches no PS. It may be in review in parallel with B9-B11 once B7 has merged."

A separate review lane for pure modules is optional. Use it only if a stall actually appears.

### RC-4: Leave and return rules that do not depend on rebuilding (DL-11, SF-08)

**Why.**
- Owner decision D4 in TASK-33281 (PERF-22) makes Roleplay a reusable screen. A reused instance is suspended rather than rebuilt (`app_navigation.py:852-896`).
- Several rules hold only because the screen is rebuilt on each visit:
  - "leaving clears marks" (R16, bulk-delete safety);
  - "a return starts INACTIVE" (§4.11);
  - "counts load at arrival" (§1.4.2).
- `app_navigation.py:577-586` also tells reusable screens to make `confirm_navigation` return True. Following that would leave drafts in a suspended screen.

**Edit.** Add a "Visit lifecycle" paragraph to §4.11:

> - **On leave** (unmount or suspend): clear marks, end the work session, close More actions and transient guards, and cancel the window and aggregate workers.
> - **On arrival or resume:** refresh counts and invalidated aggregates; re-fetch the loaded record by id and version (reload if clean, "Changed elsewhere · Reload" if dirty); apply §4.3.
> - Roleplay keeps `confirm_navigation` under reuse and does not adopt TASK-31520's "return True".

Test the hook through suspend and resume now. Add the warm parametrisation when PERF-22 lands. Do not flip `reusable=True` in tests first.

### RC-5: Tests that prove they discriminate (DL-12)

**Why.**
- For new widgets, "fails on the pre-slice code" (§5.4 item 2) fails on ImportError or NoMatches, which proves nothing about the assertion.
- "Pass unchanged" (B0) cannot be judged against suites that are already partly red on dev.

**Edit.** Rewrite §5.4 item 2:

> (a) Every new geometry or containment assertion is shown to discriminate by a named mutation that leaves the code importable: drop a rule, force `height: auto`, or un-hide a slot. Record the mutation in the task notes.
>
> (b) The §5.7.4 suites the slice touches run on both arms (pre-slice base and slice head) and are judged by the diff of the failure sets.

B0 acceptance: "no new failures versus the paired base arm".

### RC-6: Performance with many rows already loaded (SF-06)

**Why.**
- Prompts are rebuilt on every width change, and Textual clears its caches on every resize.
- With 5,000 options loaded, `set_options` takes 417 ms (spec line 470), over the 250 ms gate. Session start and end at 175-195 change the list width (42 ↔ 32).
- B2 measures only keystroke → first window.

**Edit.**
- Add to B2: with 5,000 rows loaded, a 120↔160 resize, a session start/end, and End each settle within budget. Measure CI counts plus recorded evidence, as in MF-16.
- Meet the budget first with width-independent or debounced prompt rebuilds, and with marks and aggregates applied in one batch. Adopt a resident cap only if that fails.

### RC-7: Journey tests from the first slice they touch (DL-13)

**Why.** J2 spans B4, B5b, B7 and B11, so without scaffolding it tends to be written last.

**Edit.** Add to §5.7.3:

> Each journey is created in `Tests/UI/test_roleplay_journeys.py` by the first slice it touches. It has two tests:
> - a passing prefix test covering the delivered steps;
> - a full-journey test marked `xfail(strict=True, raises=NotYetDelivered)`, where an undelivered step raises that sentinel explicitly.
>
> A plain xfail would hide regressions in steps already delivered.

### RC-8: The test chat uses your name (UX-12)

**Why.**
- B6 adds "You: Alex · your name in chats".
- The preview still hard-codes "User" (`personas_preview_controller.py:98, 182`; `personas_preview_pane.py:40`).
- Two adjacent surfaces would disagree about what the AI calls the user. Only "speaker labels" are assigned to the content batch (§6 row 045).

**Edit.** In B6, the character preview's `{{user}}` substitution (prompt and greeting cycler) uses the effective name the You row resolves. Add a test.

### RC-9: Keep the guide truthful (UX-11, DL-14)

**Why.**
- `roleplay-chat-dictionaries.md:151, 187-188, 210` and `characters-and-personas.md:109, 232-242, 307` document Send to Console draft and Ctrl+Enter as working. B3 and B5b retire both, but B12 cleans up the prose only later.
- Nothing checks the "generated" key table against F1.
- `check_guide_claim_strings.py` exits 0 by default.

**Edit.**
- B3's and B5b's guide deltas: "Send to Console draft (Ctrl+Enter) is switched off: the staged card never reached the model. Use Chat now (`o`)." Remove the old prose.
- B3 adds `Tests/UI/test_roleplay_guide_keys.py`, which compares the table between `<!-- roleplay-keys:begin/end -->` markers with the projection's F1 groups.
- B12: call the claim-strings script a read-list, or pass `--fail-on-miss`.

### RC-10: Small copy fixes (LC-08, UX-07, UX-10, UX-13)

These are all string or mockup changes.

1. **List border counts.**
   - Bottom border: `#4 of 28`, the entries convention, so it cannot be confused with the top border's `3 of 28` match count. It is blank when the search hides the loaded row.
   - Top border unchanged.
2. **Kind label.** In B3, relabel `Modes:` (PS:1563-1568) to `Kind:` so the screen matches the footer's "kind" for B3-B5b. Re-pin `test_destination_visual_parity_correction.py:1793`.
3. **Card at 120 columns.** At content width 46, the Card's World value reads `1 copy · not in chats`, instead of `1 lore book (a copy)`, which drops the disclosure. Add the Card at-a-glance row to §3.13's table.
4. **Hide ×.** Tooltip "Hide for characters (remembered)", and an F1 note "Keep beside › / Hide × (remembered per kind)". No toast: grips also persist silently.

### RC-11: Rail details (LC-07, LC-12)

**Why.**
- At 24 cells, `Lore books (…)  l` fits, but `Lore books (3 · 2 on)` does not. The key hint therefore appears and then vanishes on every arrival.
- The rail-section config diverges from the Library's `[library.rail_state] sections = {…_open}`.

**Edit.**
- **§1.4.3.** "While a kind's count is loading, its key hint is not painted." Change "ported from `LibraryRail._row_label`" to "modelled on". Do not delegate or golden-test across the two functions, because their fit steps differ by design.
- **§2.11.** Use `[roleplay.rail_state] sections = { cast_open, world_info_open, create_import_open }`. Keep `nav_open`, mapped by Roleplay's own adapter. The shared normaliser is not changed, so B0's byte-identical guarantee holds.

---

## Decisions for the owner

These three are genuine product calls. Each has a recommendation, and none blocks planning if you accept the recommendation.

### OD-1: When you press `>` on the Card and the next tab is Edit, what should happen? (SI-01, MF-05)

**Background.** When you start editing (press `e` or click Edit), the side panes close so the editor gets the width. The spec also says the arrow keys `<` `>` move between a character's tabs (Card, Edit, World, Chats…) and "never" close panes. Those two rules collide when the next tab is Edit.

**Options.**
- **A (recommended).** While you are not editing, `<` `>` skip over the editing tabs (Edit, Look, Test chat…). You get to them with `e`, `t` or a click.
  - Arrow-browsing never moves the panes, which is what the approved spec already promises.
  - Edit is still one key away.
- **B.** Moving onto an editing tab with `<` `>` starts editing, exactly as clicking Edit does. The panes close at 120 columns.
  - The tab row feels continuous.
  - Arrow-browsing can unexpectedly rearrange the screen, and the panes stay closed until you leave the edit session.

**Why A.** It keeps two approved promises intact ("arrow keys never move panes" and B7's "`< >` never start a session") and adds no new way for the layout to jump.

### OD-2: Should Roleplay and the Library use the same keys for selecting several items? (LC-01, MF-24)

**Background.** In Roleplay you mark items with `m` (or `space`) and a marks bar appears. In the Library you press `s` to enter a Select mode, then `space` to select. A few other letters also mean different things on the two tabs (`c`, `l`, `[ ]`). Most of these differences existed before this design.

**Options.**
- **A (recommended).** Keep both for now, and list every difference honestly in the spec (MF-24). File a separate "house list grammar" task to pick one model later (FU-5).
- **B.** Unify them inside this programme. This reopens keys you have already approved (F-040, ADR-152) and changes Library tests mid-programme.

**Why A.** The keys work on different tabs and in different contexts, and Roleplay's versions are guarded or undoable, so real harm is low. Unifying is worth doing, but as its own decision with its own tests.

### OD-3: While editing, what happens if you click the second closed pane open? (UX-02)

**Background.** At 120 columns, editing closes both side panes. If you click the list's grip, the list reopens. If you then click the Nav grip, the Nav opens and the list closes again, because only one side pane fits next to the editor. The panes swap rather than both coming back. The Library handles this differently: reopening Nav by hand ends its edit session.

**Options.**
- **A (recommended).** Keep the swap and write it down as a deliberate difference from the Library. Make the Escape step that "goes to the Nav" focus the Nav grip (one extra key, no layout jump) instead of closing the list you are in.
- **B.** Do what the Library does: a hand-opened second pane ends the session and keeps it ended until you leave the item. Both panes come back, but the editor shrinks to 50 columns at 120.

**Why A.** B can leave you editing in a 50-column pane, which is the starvation problem the session design was built to prevent. A keeps the editor wide and fixes the surprising Escape.

---

## Follow-ups to file separately (not spec changes)

- **FU-1: Library adoption of the Roleplay-frame contracts.** One umbrella task with a checklist (LC-06, LC-02, LC-04, LC-05, LC-03).
  - Items:
    - a Start-style rail row that returns to the landing;
    - one delete modal that names its targets;
    - one Escape owner as a pure projection;
    - dynamic F6 targets;
    - a header subtitle;
    - 1-2 line list rows;
    - an XS list stage;
    - grouped F1;
    - per-destination open-item memory;
    - conversation hand-off renamed "Continue in Console" with `o` as an alias of `c`;
    - Ctrl+S focus-scoped on the skill editor and added to the Prompts editor;
    - adaptive grips and one round border on ordinary routes.
  - Mark these as owner calls:
    - the Save/Discard/Stay dialog for Prompts and Skills (Notes autosave);
    - inline delete confirmations;
    - the Escape-chain rewrite (one-press Back today).
  - Point §7.1's H4 note at this task.
- **FU-2: Literal rendering mode for the shared `DestinationHeader`.** Subtitle and status for Console, Lab and Roleplay, with their header pins re-run (SF-03).
- **FU-3: Ratchet hygiene.**
  - A zero-behaviour PS ratchet-repair PR, if #2862 is not landing soon.
  - Add `test_module_size_ratchet.py` and the CSS integrity tests (`test_css_build_integrity.py`, `test_consolidated_css_harness.py`) to a required PR CI job. Today none of them gates PRs, which is how the PS row went red unnoticed (DL-04, DL-01).
- **FU-4: Coordination notes.** These go to the owners of in-flight work.
  - TASK-33627 (ADR-210 step 8): shared "Destination header variants" subsection in DESIGN.md (MF-18).
  - TASK-33622.14 and PR #2949: the quit hook uses the aggregate and its executors (MF-17).
  - TASK-33782 (RP-014): `form_entry_id` versus the cursor anchor (MF-09).
  - TASK-22988 and TASK-31243: status check (RC-3).
- **FU-5: House list grammar.** Only if you choose OD-2 A: one bulk-selection model and one Console hand-off key across Library and Roleplay, as an ADR-152 amendment.

---

## Dropped

### Issues dropped entirely (7)

- **SI-03** (pin footer properties, not lines). The exact-line pins are house convention, and they catch retain-prefix truncation at 120. The property tests the issue proposes already exist (I4, F1 ⊇ footer).
- **SI-07** (bind Ctrl+S earlier). The phasing is owner ruling Q5's recorded rationale, and the interim is truthful (L6 chip). Binding Options before B9 moves its button contradicts §3.1.
- **SI-08** (copy Library's TAB_REGION locally). Roleplay would be the third copy of the same contract, and the hook is opt-in and guarded. A local copy grows PS, whose ratchet is red.
- **SI-09** (defer or trim the landing's Needs attention). It is part of the landing the owner approved (D11-B), and no evidence of harm was given. It is the only list-level route to the broken rule and to the D6 disclosure.
- **LC-09** (Import… folds first in the rail). Create & import folds only below about 22 terminal rows, under the 24-row floor, and `i` and the landing remain. This would re-litigate R5.
- **LC-14** (match the Library's footer wording). At 120 columns it pushes E1 to 122 cells and loses its Escape hint, and it re-pins six or more lines, for wording parity only.
- **SF-12** (bulk delete without optimistic locking). Caches hold the unfiltered records with versions, and windows apply to characters only. No path in the spec reaches the unversioned delete.

### Parts of verified issues not adopted

- **SI-02:** splitting or freezing the 340 KB spec into core and per-slice files. Replaced by the light index and one-home rule in MF-20.
- **SI-04:** a 1,000-row first window, and dropping the 5,000-row gate. The window stays 200 and the gate stays.
- **SI-05:** a separate B6b slice for the aggregates.
- **SI-06:** reordering B8 after B11, and a "core B" stop line. 200+ columns is part of your design centre.
- **UX-02:** "→ INACTIVE" on a manual reopen. It would let the next authoring intent re-close the panes. It is offered only as the sticky-cancel option in OD-3.
- **UX-03:** option (b), where Done or Cancel ends the session. It re-litigates R9. The proposed chip copy was also wrong for the focus context.
- **UX-04:** new `alt+↑/↓` item keys and title-row buttons. Alt+arrow delivery is unreliable across terminals, and the title row is already tight.
- **UX-09:** moving Needs attention above Continue. It reorders the approved landing without evidence.
- **UX-10:** moving the D6 row out of Needs attention, or adding "Got it". It changes approved landing content, with no new evidence.
- **UX-11:** a screen-wide Ctrl+Enter binding kept only to explain the retirement. Terminals deliver it as Enter or Ctrl+J.
- **UX-13:** a toast on Hide ×. Grips persist silently, and the collapse grammar stays one rule.
- **SF-05:** mandatory keyset paging. The invariants in MF-11 suffice.
- **SF-06:** a bidirectional resident cap. Adopt it only if RC-6's measurement fails.
- **SF-07:** a single immutable read shared by the sniff and the handler. Nice to have.
- **SF-09:** TASK-33622.14 as a hard prerequisite, and absorbing it into B5b. It is a pre-existing bug, now an S dependency.
- **SF-10:** a version-keyed regex cache and chat counts limited to loaded rows. Measured cost is 24-121 ms off-thread.
- **LC-03(c):** changing the Library's one-press Escape. Moved to FU-1 as an owner call.
- **LC-07:** delegating Library rail fitting, or a golden test across both functions. Their fit steps differ by design.
- **LC-12:** teaching the shared normaliser `nav_open`. It would break B0's byte-identical guarantee.
- **LC-13:** relaxing I6 to load on cursor settle. There is no harm evidence; raise it only if J1 live verification shows slow scanning.
- **DL-04:** moving the footer-hint builder in B1. B3 replaces it, so it would be handled twice.
- **DL-08:** the full 5→13 PR split. Only B2, B5b and B9 are split (RC-2).
- **DL-10:** options (a) and (c), which change the approved order or add an interim collapse.
- **DL-11:** flipping `reusable=True` in tests before PERF-22's suspend/resume audit.
- **DL-15:** mass-moving re-pinned tests out of `test_personas_workbench.py`. A move is itself a conflict risk.
- **DR-03:** making `MODAL_OWNED_SPLIT_SHEETS` a requirement. No B modal opens outside Roleplay, so it is mentioned only as a contingency.
- **DR-09:** rewriting individual anchors to today's dev, and the proposed "free ADR numbers" list. Anchors drift faster than a refresh, and 199, 200 and 204 are claimed on other branches.

---

## Strengths to keep

- **The code-fact base is still current.** No commit since 84247cb843 touches `personas_screen.py`, `Persona_Widgets/`, `Persona_Modules/`, `library_screen.py`, `Library_Modules/`, `Widgets/Library/`, `adaptive_reader_state.py`, `base_app_screen.py` or `test_personas_workbench.py`. Every load-bearing citation re-checked on dev still holds.
- **Work sessions (R9, §2.3).** Geometry follows an explicit session, never a view, focus or Save. This resolves both earlier blockers. MF-05 closes its one hole without loosening it.
- **One aggregate unsaved predicate (R24)** drives every signal, with one guard dialog that keeps "Stay". It is the right hook for the quit path and for the executors in MF-07.
- **One pure keyboard projection (I4)** produces `check_action`, the footer, F1 and the Escape label: advertised == armed. The Escape ladder is one binding plus a pure function, simpler than the Library's 15 order-dependent bindings.
- **Compose once, toggle display (ADR-115)** with mount and remove counters on every transition. This is what keeps drafts and widget identity safe across views, sessions, the Try column and drill-in.
- **"Copy the contracts, not the code" (§2.1).** The resolver is promoted byte-identical with same-object aliases, and Roleplay never imports `Widgets/Library/`.
- **Library defects are deliberately not copied:** the ADR-046 three-choice dialog instead of the veto toast, one delete modal, one framing layer, rows of 1-2 lines, and the Start row in place of the landing-only-once defect.
- **Budget discipline:** Q10's measure → defer/shed → owner exception, ratchets lowered in the same PR, snapshots refreshed only through the script, and "never hand-merge generated output". MF-01 extends this to boot CSS rather than replacing it.
- **Scheduling isolation:** B5c is the only slice gated on TASK-33621.2, so a Console slip blocks nothing else. Only one PS slice is in review at a time.
- **The interim register (§5.3)** names the slice that removes each interim, which makes every intermediate state testable. RC-1 and MF-33 add the two missing rows.
- **Honest data states:** `(…)` and `couldn't load` instead of a false `0`, search-aware "no match" with `Also in:`, readiness never dropped while the primary is disabled, marks stored by id with a modal that names its targets, and a new book or dictionary created as an unsaved draft rather than "Untitled".
- **Existing safety machinery is reused, not replaced:** `ConfirmationDialog` (markup-off, Cancel first) as the one delete modal, the import path checks, size caps and archive bounds, and today's guard-then-confirm order for a dirty delete.
