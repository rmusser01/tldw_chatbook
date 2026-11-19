# Tier-2 code review — tldw_chatbook

You are reviewing the **~0.8M lines of `tldw_chatbook` that have never been reviewed**.
Read `CLAUDE.md` and `AGENTS.md` first; they override anything here.

## 0. This is a continuation, not a new review

**Prerequisite.** `qa/core-code-review-2026-09-17/PROMPT.md` was never committed — it
exists only in the main checkout, while everything else it produced (`report.md`,
`slices/`, `candidates/`) *is* on `dev`. So after creating your worktree, copy both
review directories across before starting:

```bash
git worktree add ../tldw-review-t2 origin/dev
cp <main-checkout>/qa/core-code-review-2026-09-17/PROMPT.md \
   ../tldw-review-t2/qa/core-code-review-2026-09-17/PROMPT.md
cp -r <main-checkout>/qa/tier2-code-review-2026-09-21 ../tldw-review-t2/qa/
```

Read `qa/core-code-review-2026-09-17/PROMPT.md` in full before doing anything. Its
**§0 (ground rules)**, **§2 (what you are looking for — D1 issues / D2 efficiency /
D3 poor patterns / D4 duplication-vs-shared-helper, and the "known-deliberate, do NOT
flag" list)**, **§4 (rules of evidence)** and **§5 (output contract, severities, report
shape)** apply here **verbatim and unchanged**. This file states only what is different.
Where the two disagree, this file wins; where this file is silent, that one governs.

Restated because they are the ones that get broken: read-only, fresh worktree of
`origin/dev`, never commit, `.venv/bin/python` only (system `python3` is 3.9), no
`timeout(1)` on this machine, **do not run the app** (it regenerates
`css/tldw_cli_modular.tcss` in whatever worktree it boots from), targeted pytest only,
never verify with `-k`, writes confined to `qa/tier2-code-review-2026-09-21/` and your
scratchpad.

## 1. Scope

### In scope for D1/D2/D3 — the 24 slices below

Tier 1 was reviewed on 2026-09-17 and is **out of scope for D1/D2/D3** — but see §2,
D4 still covers it. Line counts are `wc -l` measured 2026-09-21 at dev `3722a85748`
(clean tree, 1,795,026 package lines); treat them as budget, not gospel.

| # | Slice | Paths | Lines |
|---|---|---|---:|
| S01 | Notes A | `Notes/` — sync engine, conflict resolution, file services | 64k (split) |
| S02 | Notes B | `Notes/` — templates, importers, remainder | ↑ |
| S03 | TTS A | `TTS/` — backends | 61k (split) |
| S04 | TTS B | `TTS/` — remainder | ↑ |
| S05 | Library | `Library/` | 44k |
| S06 | API client | `tldw_api/` | 39k |
| S07 | Speech in | `Audio/`, `STT/` | 39k |
| S08 | Scheduling | `Subscriptions/`, `Scheduling/` | 41k |
| S09 | Characters | `Character_Chat/`, `Prompt_Management/`, `Internal_Prompts/` | 35k |
| S10 | Chunking | `Chunking/`, `Embeddings/`, `RAG_Admin/` | 27k |
| S11 | Evals+ingest | `Evals/`, `Local_Ingestion/` | 44k |
| S12 | Media | `Media/`, `Media_Creation/`, `Media_Playback/`, `Image_Generation/`, `Video_Generation/` | 26k |
| S13 | Canvas | `Canvas/`, `Workspaces/`, `Research_Workspace/` | 35k |
| S14 | Web | `Personal_Context/`, `Web_Scraping/`, `WebClipper/`, `Web_Server/` | 27k |
| S15 | Models | `Model_Artifacts/`, `LLM_Management/`, `LLM_Provider_Catalog/`, `Local_Inference/`, `Models/` | 21k |
| S16 | Persona | `Persona_Visual/`, `Persona_Buddy/`, `Actor_Packs/`, `Petdex/`, `Widgets/Tamagotchi/` | 26k |
| S17 | Small pkgs | `Chatbooks/ Sharing/ Outputs/ Notifications/ Metrics/ Stats/ state/ Terminal/ Tool_Packs/ Home/ Coding/ Config_Files/` plus the **Python** under `css/` — `css/Themes/themes.py` (2.0k), `css/build_css.py` (1.2k), `css/widget_css.py` (0.95k), `css/check_bundle_sync.py`, `css/tie_aware_stylesheet.py`, `css/Themes/theme_tester.py` | 35k |
| S18 | Screens A | `UI/Screens/` — `watchlists_collections_screen.py` (14.3k), `llm_screen.py` (5.2k), `change_review_screen.py` (5.0k), `evals_screen.py`, `trajectory_screen.py`, `research_workspace_screen.py`, `settings_speech_tts.py`, `model_*_view.py` | 40k |
| S19 | Screens B | `UI/Screens/` — every remaining `*.py` | 35k |
| S20 | UI root | `UI/*.py` (top level only) | 31k |
| S21 | UI modules A | `UI/Wizards/`, `UI/Speech/`, `UI/Watchlists_Modules/` | 45k |
| S22 | UI modules B | `UI/{Evals,LLM_Management,CCP_Modules,Research_Workspace_Modules,Widgets,Study_Modules,Workbench,Chunking_Lab_Modules,Lab_Modules,Views,Writing_Modules,Research_Modules,Subscription_Modules}/` | 22k |
| S23 | Widgets rest | `Widgets/{Media,Prompts,TTS,NewIngest,ModelArtifacts,Coding_Widgets,Writing,Note_Widgets,Home,Study,Evals}/`, plus the top-level `Widgets/*.py` added **since** the 2026-09-17 review (`status_line.py`, `backup_group_selector.py`, `select_values.py`, `pattern_gallery.py`) — the rest of `Widgets/*.py` is Tier 1 | 15k |
| S24 | Interop | the 31 `*_Interop/` packages — **one cluster finding, not per-file** (see §2) | 66k |
| S25 | **Backup/restore** | `Backup_Recovery/` — 80 files. **Read §1.1 before starting this one.** | 43k |
| S26 | Workflows | `Workflows/`, `UI/Workflows_Modules/` | 7k |

### 1.1 S25 is the priority slice

`Backup_Recovery/` landed on `dev` on **2026-09-21** (commit `b5251e9a6e`, TASK-32628) —
after the 2026-09-17 review, so it has never been reviewed at all. Three things make it
the first slice to staff, not the last:

- **It is already the root of the previous review's largest efficiency finding.** v1
  profiled `get_cli_setting` at ~11 ms warm and traced it to
  `Backup_Recovery/config_participants.py` → `storage_admission.acquire_storage` →
  **24,100 `posix.open` calls for 50 config reads (482 syscalls per read)**. v1 could
  only cite that as a cause from outside; this run can read it. `TASK-32804.1` owns the
  fix and is **In Progress** — coordinate with it, do not duplicate it.
- **It is a trust boundary.** `crypto.py`, `credentials.py`, `rollback_credentials.py`,
  `credential_policies.py`, `storage_admission.py`, `admission_runtime.py`. Apply v1
  §2's D1 security list here in full, and v1 §4's evidence rules without exception — a
  confident reading of a crypto path is not a finding.
- **43k lines written in one push** is where copy-paste lands. Expect it to be a large
  contributor to your D4 "New" bucket.

**Excluded, always:** `tldw_chatbook/.venv/`, `Third_Party/`, `Tests/` (used as evidence,
never reviewed), `Utils/Splash_Screens/`, `__pycache__/`, and all non-Python assets
(`css/**/*.tcss`, `assets/`). Note that `css/` is **not** excluded as a package — it
holds 4.6k lines of real Python (the theme system and the CSS bundle generator that
`scripts/preflight.sh` checks); it is reviewed under S17.

### Coverage is a check, not a judgement call

The 2026-09-17 run stopped at 21 of its 29 slices and declared itself complete. **Three
of its four P0s were in the eight it skipped.** Nothing blocked it; it decided it had
enough. So:

- `progress.md` carries one row per slice S01–S26 **from the start**, each `pending`.
- A slice reaches a terminal state only with its files-read-in-full / sampled /
  mechanical-only counts filled in. `sampled` is a legitimate terminal state;
  **absent is not.**
- `slice_paths.txt` **already exists in this directory**, verified at dev `3722a85748`
  to leave zero files unclaimed (`UNCLAIMED: 0`). Use it as written; regenerate it only if paths have moved, in
  which case it must claim every path in the table above **plus the Tier-1 paths**, so
  the whole package is accounted for rather than merely unmatched:

```
tldw_chatbook/app.py
tldw_chatbook/config.py
tldw_chatbook/Constants.py
tldw_chatbook/model_capabilities.py
tldw_chatbook/Logging_Config.py
tldw_chatbook/emergency_stop.py
tldw_chatbook/cli.py
tldw_chatbook/__init__.py
tldw_chatbook/__main__.py
tldw_chatbook/chunking_engine_version.py
tldw_chatbook/runtime_policy
tldw_chatbook/Chat
tldw_chatbook/Agents
tldw_chatbook/Tools
tldw_chatbook/MCP
tldw_chatbook/DB
tldw_chatbook/LLM_Calls
tldw_chatbook/Event_Handlers
tldw_chatbook/Utils
tldw_chatbook/RAG_Search
tldw_chatbook/UI/Screens/chat_screen.py
tldw_chatbook/UI/Screens/library_screen.py
tldw_chatbook/UI/Screens/settings_screen.py
tldw_chatbook/UI/Screens/personas_screen.py
tldw_chatbook/UI/Console_Modules
tldw_chatbook/UI/Library_Modules
tldw_chatbook/UI/MCP_Modules
tldw_chatbook/UI/Persona_Modules
tldw_chatbook/UI/Navigation
tldw_chatbook/Widgets/Console
tldw_chatbook/Widgets/Library
tldw_chatbook/Widgets/Persona_Widgets
tldw_chatbook/Widgets/Settings_Widgets
tldw_chatbook/Widgets/Chat_Widgets
```

  (`Widgets/*.py` top level is Tier-1 too — claim it explicitly, it is not a directory
  prefix.)

- Before writing the verdict line, run the completeness check and paste its output
  into `report.md`:

```bash
.venv/bin/python - <<'PY'
import pathlib
claimed = [l.strip() for l in pathlib.Path(
    "qa/tier2-code-review-2026-09-21/slice_paths.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")]
EXCL = (".venv", "Third_Party", "__pycache__", "Splash_Screens")
miss = [p for p in pathlib.Path("tldw_chatbook").rglob("*.py")
        if not any(x in p.parts for x in EXCL)
        and not any(str(p) == c or str(p).startswith(c.rstrip("/") + "/") for c in claimed)]
print("UNCLAIMED:", len(miss))
for m in miss[:40]:
    print("  ", m)
PY
```

A non-zero count is a scope bug to fix before the review ends, not a footnote.

## 2. D4 runs repo-wide — including Tier 1

D1/D2/D3 stop at the slice table. **D4 does not.** A cluster like `_maybe_await`
(66 definitions today) straddles both tiers; judging it from one slice produces a
recommendation that is wrong for the copies you could not see. So every D4 census,
cluster read and canonical-home decision covers **all ~1.7M lines** of
`tldw_chatbook/`, minus the always-excluded paths.

### Reuse the existing tooling — do not rewrite it

The 2026-09-17 scripts are committed on `dev` and work. Run them; do not re-author them:

```bash
mkdir -p qa/tier2-code-review-2026-09-21/candidates
C=qa/core-code-review-2026-09-17/candidates
.venv/bin/python -W ignore $C/dup_census.py tldw_chatbook \
    qa/tier2-code-review-2026-09-21/candidates
.venv/bin/python $C/helper_adoption.py . \
    qa/tier2-code-review-2026-09-21/candidates/helper_adoption.tsv
```

`$C/pattern_greps.py` hard-codes the Tier-1 path list in its `T1` variable (line 5).
Copy it into your own `candidates/` and replace that list with the slice-table paths —
that is the only edit it needs.

### Report deltas, not absolutes — and compare like with like

**The committed `$C/dup_*.tsv` files are TIER-1-SCOPED** (519 / 92 / 247 rows). Your run
is repo-wide. Diffing a repo-wide census against those files manufactures growth that is
not there — this trap was walked into while writing this prompt. The repo-wide baseline
is the one quoted in v1 §3 Phase 1a: **1,823 same-name / 244 verbatim / 665 shape**
groups at dev `d8fb4053f9`.

Measured 2026-09-21 at dev `3722a85748`, repo-wide, clean tree: **1,943 / 264 / 709**.
Against v1's like-for-like 1,823 / 244 / 665, that is **+120 same-name, +20 verbatim
clone groups, +44 shape groups in four days.** Meanwhile the known clusters are flat to
the file: `_maybe_await` 60 byte-identical, `_enforce_policy` 43, `_cancel` 24.

Read those two facts together, because they say different things: **nothing has removed
any of the known duplication, and new clone groups are appearing faster than old ones
are being consolidated.** `TASK-32808` is where the removal work lives and it has not
run. Note also that by-name counts sit above byte-identical ones (`_maybe_await` 65 by
name vs 60 identical; `_enforce_policy` 48 vs 43) — that gap is drift between copies,
not growth, and **the two counts are not interchangeable.**

Report three buckets **separately**:

- **Flat** — clusters unchanged since 2026-09-17. Expect the large ones to be here. A
  cluster that survived a complete review-and-fix cycle untouched is the argument for a
  guard (below), not for re-filing it.
- **New** — clusters that exist only in Tier-2 code and were never seen. This is the
  bucket this run exists to produce.
- **Closed** — rows a merged stream PR actually removed. Verify by census, not by
  reading the PR.

Where a v1 seed number and your census disagree, say which basis each used
(byte-identical body / same name / raw occurrence) before calling it a change.

### Every D4 recommendation carries its guard

A canonical home is half a recommendation. The known clusters sat through an entire
review-file-fix cycle at exactly the same size while twenty new clone groups appeared
beside them. Nothing removed them and nothing stops copy N+1 from landing. So each D4
cluster names both:

1. **the canonical home** — an existing `Utils/` module by preference (see v1 §2's
   helper list and your fresh `helper_adoption.tsv`); a new module only when none fits,
   with the reason it does not; **and**
2. **the guard that keeps it consolidated**, in this repo's own established shape: an
   authoring-time script wired into `scripts/preflight.sh` (precedent:
   `scripts/check_persistent_diagnostic_inventory.py`,
   `check_profile_owned_path_inventory.py`, `check_index_plan_pins.py`,
   `check_schema_table_allowlist.py` — all run from that one file and from the required
   `Derived artifacts` CI job), or a ratchet test (precedent:
   `Tests/.../test_module_size_ratchet.py`).

   If a cluster genuinely cannot be guarded mechanically, say so and say why. Do not
   invent a guard for a cluster too small to deserve one — under ~5 copies, "adopt it
   and move on" is the right answer, and a guard nobody needs is the same
   over-engineering the review is supposed to find.

### `*_Interop/` (S24)

31 packages, 66k lines, one template: `__init__` + `server_<x>_service.py` +
`<x>_scope_service.py`, with `_normalize_mode` / `_require_client` / `_enforce_policy`
copied with only the enum name changed. Produce **one** cluster finding with the
template, the current copy count, and the outward/downward LOC walk. Do not read the
packages individually. `TASK-32808.6` ("Give the scope-service scaffold one home")
already owns this and is **To Do** — cite it, report what has changed since it was
filed, and flag that it needs an ADR before implementation. Do not re-derive it.

## 3. Already handled — derive it, do not assume it

97 tasks exist from the 2026-09-17 review (`TASK-32800`–`32811` plus dotted children).
Their statuses move; this prompt does not. In Phase 0, build the already-handled list
**from the backlog at run time**:

```bash
for p in 32800 32801 32802 32803 32804 32805 32806 32807 32808 32809 32810 32811; do
  backlog task list --parent $p --plain
done
grep -rl 'core-review' backlog/tasks
```

Also grep `backlog/tasks/` titles for
`dedup|duplicate|consolidat|god module|split|Utils|helper` to catch owners from outside
that review.

As of 2026-09-21, **`TASK-32808` (adopt the shared helpers) is 6 To Do / 4 In Progress
and `TASK-32807` (delete dead code) is 4 To Do** — most of the D4 and deletion work is
filed and not yet done. A finding that restates one of those is noise. A finding that
shows one of them is now *wrong or insufficient* — the cluster grew, the canonical home
moved, a merged PR changed the shape — is exactly what this run is for.

Read `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md`,
`lessons-textual.md` and `lessons-console-wiring.md` before Phase 2. They are short and
they encode mistakes that have already cost time in this repo.

## 4. Method

v1 §3's five phases, unchanged, with these substitutions:

- **Phase 0** additionally writes `slice_paths.txt`, the already-handled list, and the
  census deltas against the 2026-09-17 TSVs.
- **Phase 1** runs the three existing scripts repo-wide (§2), not a re-authored census.
- **Phase 2** dispatches one read-only subagent per slice S01–S26, ≤6 concurrent. Each
  gets: v1 §0/§2/§4/§5, this file's §1 and §3, the candidate rows whose paths fall in
  its slice, and the already-handled list. It returns files read in full vs sampled
  (honestly — it feeds the coverage table), findings in v1 §5 format, and every
  candidate row marked `confirmed / retired (why) / unverified (check)`. **A slice
  reporting zero findings must state what it read.** That is a coverage statement, not
  a clean bill.
- **Phase 3** is yours, not a subagent's: the repo-wide D4 pass per §2, each cluster
  with its canonical home *and* its guard.
- **Phase 4** verifies every P0, every P1, and every S-sized "helper exists, ignored"
  claim by reproduction. Anything unreproduced stays a finding at
  `Confidence: inferred`, gets a row in "Left UNVERIFIED" with the literal command that
  would settle it, and may **not** be P0/P1.
- **Phase 5** writes the deliverables, runs the completeness check, then stops.

## 5. Output

Into `qa/tier2-code-review-2026-09-21/`: `report.md`, `gap-candidates.md`,
`candidates/*.tsv`, `progress.md` — shapes exactly per v1 §5, plus two sections in
`report.md` that v1 did not have:

- **Census delta** — the grew / new / closed table from §2.
- **Coverage** — the S01–S26 table with each slice's terminal state and counts, followed
  by the completeness-check output.

`gap-candidates.md` groups ✅ into proposed batch tasks by canonical home, one task per
PR, and lists already-owned findings with their task ids at the top.

**Do not file the tasks. Do not fix anything. Do not commit.**

## 6. Stopping

Stop when Phase 5 is written and the completeness check prints `UNCLAIMED: 0`. Finish
with the verdict line and the path to `report.md`. Prefer depth on `Backup_Recovery/`
(S25, see §1.1), `Notes/`, `Library/`, `tldw_api/`, `Evals/`, `Local_Ingestion/` and
`UI/Screens/` over breadth elsewhere — but
"prefer depth" never authorises leaving a slice row non-terminal.
