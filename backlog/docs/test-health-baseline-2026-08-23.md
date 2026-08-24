# Test-suite health baseline — 2026-08-23

Measured at `origin/dev` **`b2b1e2e0d3c6d144388aba67cbd4bee123f35665`**; rebased onto
`73a43c71f` for review. Two repairs were needed before it could be measured at all:
TASK-20972 (fixed on dev by `1796ff16a`, not by this branch -- see §4) and TASK-19570.1.

Environment: macOS 24.6.0, `.venv` Python 3.12.11, pytest 8.4.2, pytest-xdist 3.8.0,
textual 8.2.8 (the exact pin). Every optional group installed, so nothing is skipped
for want of a dependency.

> **Why this document exists.** `core-tests` has never once finished in CI, so roughly
> 1,778 test files — everything outside `Tests/UI` — have had no verdict of any kind.
> This is the first measurement of that half.

---

## 1. Headline

| Half | Passed | Failed | Errors | Skipped | Source |
|---|---|---|---|---|---|
| **Core** (`Tests` minus `Tests/UI`) | **42,435** | **144** | 3 | 229 | measured here, 102 chunks, 30.8 min |
| **UI** (`Tests/UI`) | 14,424 | 428 | 28 | 5 | CI run 32647831275, 12 shards |
| **Total** | ~56,859 | ~572 | 31 | 234 | |

Repo-wide collection: **58,023 tests collected, exit 0, zero collection errors** at the
measured commit; **58,066** re-verified at the rebase target `73a43c71f`. It exited
non-zero until TASK-20972's collection error was fixed.

**The suite is far healthier than its reputation.** The core half fails at 0.34%. The
problem is not that the suite is bad; it is that nothing has been reading it.

---

## 2. Why CI reports nothing

- `core-tests` terminates at **exactly 2h00m** on every run — 15:14:56→17:15:22,
  15:02:45→17:03:16, 12:22:35→14:23:03, 11:25:48→13:26:22. That is
  `timeout-minutes: 120` expiring, which GitHub surfaces as `cancelled`, not `failure`.
  It reads like supersession and is not.
- **It is slow, not hung** — corrected from this document's first draft, which guessed a
  hang because 30.8 minutes locally seemed too far from >120 in CI. The job's own log
  settles it (run 32647831275, ubuntu leg): pytest starts emitting at `15:44:55` and is
  **still printing progress steadily, with no stall,** when the cap kills it at `17:39`,
  having reached **60%**. Steady progress at the kill is what distinguishes the two, and
  they have different fixes. Extrapolated, the suite needs **~190 minutes** on a 4-vCPU
  runner — public repos get 4, so `-n auto` is 4 workers against 8 faster cores locally,
  which accounts for the gap without a hang. Sharding is therefore the right fix, and
  TASK-21411 applies it. The guess was cheap to make and would have been expensive to act
  on: it pointed at hunting a hang that does not exist.
- `test-summary` reported **`success` while all 12 UI shards were red**: it declares
  `needs:` and `if: always()` but never inspects the needed jobs' conclusions.
- Zero successful `test.yml` runs in the last 400. 173 cancelled; of the 12 that
  completed, 12 failed.
- Branch protection on `dev` requires only `Derived artifacts reproduce from their sources`.
- `nightly-deep` has never fired (TASK-19600), so `--run-slow`, Windows and py3.11/3.13
  are unverified.

---

## 3. The reds are clustered, not scattered

Six causes account for well over half of everything red.

| # | Cause | UI | Core | Status |
|---|---|---|---|---|
| **A** | Library rail composes the **compact starter rail**: the shared test factory never wrote the persisted lifecycle, so every test app looked like a brand-new profile | 143 | — | **Fixed**, TASK-21280 |
| **B** | `ChatScreen` built by `__new__`, bypassing `__init__`; production's `_console_chat_store` setter has since grown a dependency on `_fleet` | 52 | 65 | open |
| **C** | `types.SimpleNamespace` app doubles missing attributes production now calls (`run_worker`, `_library_note_import_execution_active`, …) | 20 | 5 | open |
| **D** | `Docs` MCP inventory drifted from the code it documents | — | 14 | open |
| **E** | Prompt-browse settle race | 19 | — | open |
| **F** | Displaced `@parametrize` block aborting collection repo-wide | 24 | — | **Fixed on dev**, `1796ff16a` |

**A, B and C are one family**: the harness presents production with something it no
longer recognises — a fresh profile, a half-built screen, a stub grown stale. That, not
tautological assertions, is this suite's dominant defect class. It is precisely the trap
`lessons-testing-evidence.md` records as *"a fake written to match your call site
validates the mistake."*

Top core files by failure count:

| Count | File |
|---|---|
| 40 | `Tests/Chat/test_console_generation_actions.py` |
| 19 | `Tests/Chat/test_console_h3_image_edit.py` |
| 14 | `Tests/MCP/test_mcp_documentation_contract.py` |
| 6 | `Tests/Chat/test_chat_mocked_apis.py` |
| 4 | `Tests/Performance/test_app_startup_performance.py` |
| 4 | `Tests/LLM_Management/test_mlx_lm.py` |
| 4 | `Tests/Chat/test_console_video_actions.py` |

Red chunks (21 of 102): Agents 4, Architecture 2, Character_Chat 1, **Chat 86**, Chunking 3,
Evals 1, Image_Generation 3, integration 3, Internal_Prompts 1, LLM_Calls 3,
LLM_Management 4, **MCP 14**, Media_DB 1, Notes 2, Performance 4, ProductionApp 3, QA 1,
RAG_Search 1, TTS 1, Widgets 5, ROOT 1.

Three tests attempt **real outbound HTTPS to public addresses** and are correctly refused
by the egress guard — they were never stubbed, and the guard is doing its job.

---

## 4. A correction worth keeping

The TASK-20972 collection error was two `@pytest.mark.parametrize` decorators declaring
five parameters above `test_wide_files_task_return_restores_database_browse_receipt`,
whose signature takes none.

This branch originally **deleted** them, on the evidence that the test was introduced
without them and that its body references none of the five. Both facts are true and the
conclusion drawn from them -- "debris, never wired up, no coverage lost" -- was wrong.
`1796ff16a` on dev shows what actually happened: the block had **slid up by one test**.
It belongs to `test_path_transition_authority_names_file_operation_and_settles`, which
takes all five parameters by name. The correct repair moves the decorators down; deleting
them would have silently dropped four real parametrized cases while leaving collection
green and every assertion passing.

**The lesson generalises:** when a decorator block does not fit the function beneath it,
check the function *below* it before concluding it is debris. A displaced decorator and an
abandoned one are indistinguishable from the signature mismatch alone, and only one of the
two resolutions preserves coverage. "The body does not reference these names" is evidence
of displacement, not of worthlessness.

## 5. How this was measured, and how to reproduce it

Two things silently corrupt any run here and are worth stating plainly.

1. **The venv's editable install points at a different branch.**
   `__editable___tldw_chatbook_0_1_8_0_finder.py` maps `tldw_chatbook` to
   `.worktrees/task-2512-mcp-unified`. Always invoke as `.venv/bin/python -m pytest`
   **from inside the worktree under test**, so cwd wins on `sys.path`. Verify, do not
   assume: `Tests/test_import_identity`-style assertion on `tldw_chatbook.__file__`.
2. **The OS keyring** was reachable from every mounted-app test until TASK-19570.1.

Method: chunk by top-level directory, `-n 8 --dist loadscope --max-worker-restart=3
--timeout=300 --continue-on-collection-errors`, one JSON report per chunk, a `.done`
marker per chunk so a killed run resumes at chunk granularity. Hour-scale single
invocations are not completable on this machine — concurrent sessions kill them, and
190 worktrees share this checkout.

## 6. What this baseline does not cover

- **`Tests/UI` was not re-measured locally**; its numbers come from CI run 32647831275,
  whose head is `ac1aa2da5` on `fix/inventory-drift-21100` — near dev, but not dev.
- **Order dependence is untested.** Chunking by directory is itself an isolation
  boundary, so a failure that only appears in a whole-suite ordering would not show here.
  `nightly-deep`'s serial run is the intended canary and has never executed.
- **The fd-leak signal was not re-checked.** TASK-19520 measured open fds growing by 956
  against a limit of 200. A failure appearing only late in a long run should be
  re-verified in isolation before it is believed.
