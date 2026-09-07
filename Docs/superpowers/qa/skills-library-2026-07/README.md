# Library ▸ Skills — Phase-1 gate QA evidence (2026-07-14)

## Scope

- **Branch:** `claude/skills-spec`, HEAD `ab421f9e` (`fix(skills): style the import row —
  source+bundle+pin`), on top of Tasks 1–5 (`0ba5c140`..`ab421f9e`): list canvas, SKILL.md
  editor + trust panel, existing-skill rename refusal, import (trust-pending) + `skills` route
  retirement behind the Library alias.
- **Recipe:** textual-serve, playwright bundled Chromium, viewport 2050×1240, `dsf=1`,
  `?fontsize=12` (font-size query param — the recipe's smaller font is not cosmetic: at the
  default font the Library Skills editor's Trust panel and Save/Delete row sit below the
  viewport fold and this canvas family does not support mouse-wheel scroll — see caveats),
  `https://**` route-abort, `body.-first-byte` gate. Scripts reused verbatim: scratchpad
  `serve_qa.py` / `cap.py`. Port 9092.
- **Profile:** fresh `qa-home-skills` (scratchpad), adapted from the `setup_home_agent.py`/
  `setup_home_sys.py` precedent — splash off, `[console.onboarding] first_send_completed = true`,
  provider config untouched (irrelevant to a Skills-only gate). `PYTHON_KEYRING_BACKEND=
  keyring.backends.fail.Keyring` set for both the seeding script and the served app process so
  the local skill trust marker store falls back to a plain file under the profile instead of the
  developer's real macOS Keychain — ephemeral QA trust state never touches the real Keychain, and
  both processes make the identical secure/insecure keyring decision (confirmed:
  `FileSkillTrustGenerationMarkerStore`, `reduced_rollback_protection=True`).
- **Corpus:** real `obra/superpowers` 6.1.1 skills copied from
  `~/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/` — the same
  upstream skillset Task 5's fixtures/compat findings are drawn from, not synthetic data.
  Headless seeding script: `seed_home_skills.py` (scratchpad), reusing the exact production
  `LocalSkillsService`/`SkillTrustService` construction app.py itself performs (including
  `build_skill_trust_marker_store_with_fallback`), so this script's on-disk state is
  byte-consistent with what the served app reads.

## Seeding sequence (why both populations exist)

1. `SkillTrustService.bootstrap_trust(passphrase)` runs **first, before any skill exists on
   disk** — creates trust manifest generation 1 with an **empty** `skills` dict. This matters:
   `trust_uninitialized` (what every skill would show if bootstrap ran *after* import) is **not**
   in `library_skills_canvas._TRUST_REVIEWABLE_STATUSES`, so the Library editor's "Review
   changes" button would stay permanently disabled for a never-bootstrapped skill. Bootstrapping
   an empty store first means every subsequently-imported skill lands as `quarantined_added`
   ("needs review", **reviewable**) instead.
2. 12 real skill directories imported via `LocalSkillsService.import_skill` (directory-name
   derivation — Task 5's real import path): `brainstorming`, `dispatching-parallel-agents`,
   `executing-plans`, `finishing-a-development-branch`, `receiving-code-review`,
   `requesting-code-review`, `subagent-driven-development`, `systematic-debugging`,
   `test-driven-development`, `using-superpowers`, `verification-before-completion`,
   `writing-skills`. All land `quarantined_added` against the generation-1 manifest.
3. 3 of the 12 approved headlessly via the real trust service's own
   `unlock_with_passphrase` → `capture_review` → `trust_reviewed_snapshot` sequence:
   `test-driven-development`, `using-superpowers`, `verification-before-completion`.
4. 2 more real skills (`using-git-worktrees`, `writing-plans`) deliberately left **off disk**
   for the live Import-flow capture (5).
5. During live capture, a 4th skill (`brainstorming`) was approved **live**, through the actual
   served app's passphrase-modal UI (not the headless script) — see capture 4.

## Captures (`Docs/superpowers/qa/skills-library-2026-07/`)

1. **`rail-skills-count-2026-07-14.png`** — Library Browse rail: `Skills (12)` (pre-import
   count = 12 imported = 3 trusted + 9 needs-review; the count formula is `len(available_skills)
   + len(blocked_skills)` from `LocalSkillsService.get_context()`).
2. **`list-both-populations-2026-07-14.png`** — the list canvas after the live approve
   (capture 4) with BOTH populations visibly distinct: `✓ brainstorming` (just flipped live),
   `✓ test-driven-development`, `✓ using-superpowers`, `✓ verification-before-completion`
   (trusted, bold) vs. `⚠ dispatching-parallel-agents`, `⚠ executing-plans`, `⚠
   finishing-a-development-branch`, `⚠ receiving-code-review`, `⚠ requesting-code-review`, `⚠
   subagent-driven-development`, `⚠ systematic-debugging`, `⚠ writing-skills` (needs-review) —
   real superpowers names throughout, plus this same screenshot IS the "row flips to ✓" evidence
   for capture 4 (brainstorming was `⚠` before the approve, `✓` after, same list).
3. **`detail-editor-trusted-skill-2026-07-14.png`** — the SKILL.md editor open on
   `test-driven-development` (a headlessly-approved TRUSTED real skill): read-only Name Input
   with the "Rename isn't supported…" hint, Description/Argument hint/Allowed tools fields, the
   user-invocable/disable-model-invocation/context toggle buttons, Model override with its "Not
   applied in v1." hint, the real Body content (`# Test-Driven Development (TDD)` …), Supporting
   files (`testing-anti-patterns.md (8251 bytes)`), the save-marks-needs-review warning (visible
   because the skill is currently trusted — saving would re-quarantine it), Trust panel reading
   `Trust: trusted` with Unlock/Review/Approve all correctly disabled, and the Save/Delete row.
4. **Trust panel on a BLOCKED skill, driven end-to-end through the real passphrase modal**
   (4 frames, `brainstorming`):
   - `trust-panel-blocked-before-2026-07-14.png` — post-unlock state:
     `Trust: new untrusted file (SKILL.md, spec-document-reviewer-prompt.md,
     visual-companion.md)` — the real changed-files list for this skill's two real supporting
     files plus SKILL.md itself.
   - `trust-panel-passphrase-modal-2026-07-14.png` — the real `SkillTrustPassphraseModal`
     ("Unlock Local Skill Trust", masked `Input(password=True)`, Cancel/Submit).
   - `trust-panel-review-captured-2026-07-14.png` — after "Review changes": the changed-files
     line populates and Approve becomes enabled.
   - `trust-panel-after-approve-2026-07-14.png` — after Approve (passphrase re-entered through
     the same real modal, per `_approve_library_skill_trust`'s own re-prompt): `Trust: trusted`,
     all three trust actions correctly disabled. `list-both-populations-2026-07-14.png` (capture
     2) is the "row flips to ✓" confirmation in the list.
5. **Import flow** (2 real skills imported live, one Import press each):
   - `import-outcome-1-2026-07-14.png` — path typed
     (`.../superpowers/6.1.1/skills/using-git-worktrees`), outcome line **"1 imported ·
     re-review it in the trust panel"**, count 12→13, new `using-git-worktrees` row (⚠) present.
   - `import-outcome-2-2026-07-14.png` — same row, second skill (`writing-plans`) imported
     immediately after; count 13→14.
6. **`skills-route-alias-2026-07-14.png`** — command palette entry **"Tab Navigation: Switch to
   Skills (Library)"** (typed "skills", Enter) lands on Library ▸ Skills directly — rail
   highlighted, `Skills (14)` (post-import count), toast **"Switched to Skills"** bottom-right.

## Service-read evidence (not pixels alone)

Read directly from the profile's on-disk skills index/trust manifest after the full session
(`.../qa-home-skills/.local/share/tldw_cli/default_user/skills/`):

```
index (tldw_chatbook_skills.json): count 14, all real names, version=1 each
  (using-git-worktrees / writing-plans present — confirms both live imports landed)

trust manifest (trust/skill_trust_manifest.json):
  generation: 5
  trusted skills in manifest: ['brainstorming', 'test-driven-development',
                               'using-superpowers', 'verification-before-completion']
  audit events: ['trust_bootstrap', 'trust_approved', 'trust_approved',
                 'trust_approved', 'trust_approved']
```

4 trusted + 10 needs-review = 14, matching every rail count and list capture above exactly.
The 5 audit events are the 1 bootstrap + 4 approvals (3 headless + 1 live), confirming the trust
manifest's own history matches the seeding narrative, not just the final snapshot.

## Compat findings from Task 5 (summarized, full detail in `task-5-report.md`)

Real superpowers `SKILL.md` files use only `name`/`description` frontmatter (no
`argument_hint`/`allowed-tools`/`model`/`context`/etc. anywhere in the real 14-skill set) —
tldw_chatbook's schema supports materially more than any real skill exercises. Filename-derived
naming (`import_skill_file` deriving from a literal `SKILL.md` filename) would have collided
every real skill onto the name `"skill"` — the real import path avoids this by using the
directory name instead, confirmed live again here (all 12 headless imports + 2 live imports
landed under their correct real names). Nested reference subfolders (`using-superpowers/
references/`, `brainstorming/scripts/`, `subagent-driven-development/scripts/`,
`writing-skills/examples/`) are silently skipped as supporting files (flat-only model) —
reproduced live in this session's seeding output (see `seed_home_skills.py` console log:
"skipped nested dirs" printed for exactly those 4 real skills, non-fatal).

## Caveats (honest, not glossed over)

- **FIXED (commit `9370f381`) — The Library Skills editor's Trust panel is unreachable at the
  recipe's default font size.** At 2050×1240 without `?fontsize=12`, the editor's lower content
  (warnings, Trust panel, Save/Delete) renders below the viewport and this canvas (a plain
  `Vertical`, not a `VerticalScroll`) does not respond to mouse-wheel scroll at all — confirmed by
  testing wheel scroll on the (also overflowing) list view with a 2000px delta with no effect.
  `?fontsize=12` is the same real textual-serve feature the brief names in its recipe; it is not
  optional for this screen, it is required. This matches a previously-noted constraint
  (plain-Vertical canvas clipping) rather than a new one. **Fix:** `LibrarySkillsListCanvas` now
  subclasses `VerticalScroll` (the same house pattern already used by
  `LibraryExportCanvas`/`LibraryIngestCanvas`), giving mouse-wheel scroll, the default keyboard
  scroll bindings, and automatic focus-jump-into-view for free via Textual's own
  `ScrollableContainer`. Re-captured at DEFAULT font (no `?fontsize=12`) — see "Fix wave
  (2026-07-14)" below.
- **FIXED (commit `afe122d7`) — A brand-new install cannot bootstrap trust from the Library UI at
  all.** The Library skill editor's "Unlock" button only ever calls `unlock_with_passphrase`
  (never `bootstrap_trust`, confirmed by its own docstring: "never bootstraps from this editor"),
  and is only enabled when `trust_status == "trust_locked"` — never the true first-run
  `trust_uninitialized` state. The standalone `SkillsScreen`'s "Bootstrap trust" button is the
  only in-app bootstrap entry point, but that screen is no longer reachable via any live route
  (Task 5 retired the `skills` route behind the Library alias). This QA session worked around the
  gap with a headless `bootstrap_trust` call (the same pattern the product's own tests use); it
  was not a capture artifact — it was a genuine Phase-1 product gap, flagged here rather than
  silently fixed at the time (out of that gate task's scope). **Fix:** the Trust panel now renders
  a dedicated first-run "Set up skill trust" state whenever `trust_status ==
  "trust_uninitialized"` — an explanation line plus a single action driving a new
  `SkillTrustBootstrapModal` (twice-entry passphrase confirmation; refuses to dismiss on a
  mismatch) that calls the real `SkillTrustService.bootstrap_trust(passphrase)` directly. See "Fix
  wave (2026-07-14)" below for the live end-to-end capture.
- **`Ctrl+P` does not open the command palette in this browser-driven harness** — Chromium
  intercepts it as its native print shortcut before it reaches the page. Worked around by
  clicking the footer's "Palette Menu" label directly (`^p Palette Menu`), which does dispatch
  the real action. Real keyboard use in an actual terminal is unaffected; this is a
  browser-harness-only wrinkle worth remembering for future Ctrl+P-dependent captures.
- Two mistargeted early clicks (before coordinates were recalibrated for `?fontsize=12`'s
  smaller cell grid) landed inside the Body `TextArea` instead of the Unlock button; verified via
  screenshot that no text was actually inserted and Save was never pressed, so no skill content
  was mutated. Not included in the final capture set.

## Fix wave (2026-07-14) — re-capture of both fixed states

User directive: fix both gate findings above before Phase 2, then re-capture the fixed states at
DEFAULT font (no `?fontsize=12`) to prove the fold problem is genuinely gone, not merely worked
around. Recipe reused verbatim: scratchpad `serve_qa.py`/`cap.py`, viewport 2050×1240, port 9092,
`https://**` route-abort, `body.-first-byte` gate, `1.02`-corrected click/scroll coordinates. CSS
prebuilt (`$PY tldw_chatbook/css/build_css.py`) before capture. Stale servers killed between runs
(one port, one profile at a time).

### Profile A — `qa-home-skills-2` (bootstrapped, for the FIX 1 scroll proof)

Seeded (scratchpad `seed_home_skills2.py`, a trimmed copy of the original `seed_home_skills.py`
with the same real production construction path and the same real obra/superpowers corpus):
`bootstrap_trust` first (empty baseline), then 4 real skills imported
(`brainstorming`/`test-driven-development`/`using-superpowers`/`writing-skills`), then 2 approved
(`test-driven-development`/`using-superpowers`).

- **`fix1-editor-below-fold-default-font-2026-07-14.png`** — `test-driven-development`'s editor
  opened at DEFAULT font, unscrolled: the field stack runs off the bottom of the viewport
  ("Supporting files" sits right at the fold) — the same overflow the original gate found — but
  a scrollbar is now visibly present on the canvas's right edge (the structural proof the
  container is a real `VerticalScroll`, not the old clipping `Vertical`).
- **`fix1-trust-panel-scrolled-into-view-default-font-2026-07-14.png`** — the SAME editor, SAME
  DEFAULT font, after mouse-wheel scroll (repeated wheel deltas over the canvas's own scrollbar
  gutter, not over the Body `TextArea`, which has its own independent inner scroll and would
  otherwise absorb the wheel events): the full Trust panel (`Trust: locked`,
  Unlock/Review changes/Approve) AND the Save/Delete row are now both visible in one screenshot —
  reachable without `?fontsize=12` for the first time.

### Profile B — `qa-home-skills-2-fresh` (never bootstrapped, for the FIX 2 live bootstrap proof)

Seeded (scratchpad `seed_home_skills2_fresh.py`): one real skill (`brainstorming`) imported;
`SkillTrustService.bootstrap_trust` deliberately **never called** — the true first-run shape.
`PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring` set for the served app subprocess this
time too (a first pass without it left the served app's real trust construction resolving to the
actual macOS Keychain, which silently failed partway through `bootstrap_trust` — a snapshot file
was written but no manifest; re-ran clean after exporting the same env var the original gate used
for both processes).

- **`fix2-trust-setup-state-uninitialized-2026-07-14.png`** — `brainstorming`'s editor, scrolled
  to the Trust panel: `Trust: not initialized`, the exact `_TRUST_SETUP_EXPLANATION_COPY` line,
  and a single **"Set up skill trust"** action — no Unlock/Review changes/Approve row at all (the
  first-run setup state).
- **`fix2-trust-setup-passphrase-modal-2026-07-14.png`** — pressing that action opens the real
  `SkillTrustBootstrapModal` ("Set Up Local Skill Trust", "New trust passphrase" (focused) +
  "Confirm trust passphrase", Cancel/Submit).
- **`fix2-trust-setup-after-bootstrap-recompose-2026-07-14.png`** — immediately after submitting
  matching passphrases (via the confirm field's `Input.Submitted`, i.e. Enter): the editor
  recomposed back to the TOP (Name/Description/…), proving a real state change occurred (a
  no-op/cancelled submit leaves the scroll position and setup state untouched, which is exactly
  what happened on the first mis-configured attempt).
- **`fix2-trust-panel-after-live-bootstrap-2026-07-14.png`** — a fresh reload of the same skill
  (new `cap.py` session = new app process, in-memory keys cleared, on-disk manifest persists):
  `Trust: locked` with the NORMAL Unlock/Review changes/Approve row now rendered — the
  "Set up skill trust" state never reappears once the store is genuinely bootstrapped on disk.

Service-read confirmation: `.../qa-home-skills-2-fresh/.../skills/trust/skill_trust_manifest.json`
exists after the live bootstrap (it did not before); `.../trust/snapshots/brainstorming-1.json`
present, generation 1.

## Verification

- **Sweep:** `Tests/Skills Tests/Library Tests/UI/test_library_skills_canvas.py
  Tests/UI/test_library_shell.py Tests/UI/test_screen_navigation.py
  Tests/UI/test_destination_shells.py` = **1087 passed, 2 failed, 1 skipped** (775s).
  The 2 failures — `test_library_shell_note_conflict_during_preview_reads_live_text` and
  `test_library_shell_export_registry_failure_warns_it_wont_appear_in_artifacts` — are **not**
  in the brief's exempt list (first-time-replay orientation test, 3 MCP
  `#unified-mcp-panel` failures), so each was independently confirmed pre-existing and
  unrelated to Skills, not silently waved through:
  - Both pass in isolation (`pytest <both tests> -q` → 2 passed).
  - `Tests/UI/test_library_shell.py` run standalone (its own 257 tests) → **257 passed, 0
    failed** — the failures only appear when this module runs combined with the other suites in
    the same pytest process (order/global-state-dependent flake), never on their own.
  - Each test's introducing commit (`git log -S`) is `30c8527e`/`8d597882` — save-conflict and
    export-registry-failure fixes, both predating and unrelated to any Skills commit
    (`0ba5c140`..`ab421f9e`). The Skills-era diff to this test file
    (`db8a64ec`/`0ba5c140`) is purely additive (+71 lines, 0 deletions) — it could not have
    altered either existing test's behavior.
  - Net: **no NEW failures** introduced by Skills Tasks 1–6; gate is green.
- Every population/count claim above cross-checked against the real on-disk index + trust
  manifest (see "Service-read evidence"), not screenshots alone.
- No PR opened, no Phase 2 work started — this is the Phase-1 gate checkpoint only.
