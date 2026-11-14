# Tier-2 code review — tldw_chatbook — 2026-09-21

**Verdict:** The 890k lines that had never been reviewed are, on the whole, carefully written — the hardest
surfaces (archive parsing, the notes importer, the TTS profile store, model downloads) are *better* defended than
the shared helpers they decline to use — but the review found **3 P0s, 57 P1s, and one structural problem worth more
than any of them: of the repo's wired, green, CI-required guards, three are each wrong in a different way, and the
duplication census has not moved in the 4 days since ten fix-stream PRs merged. The biggest risk is not a defect in
this list; it is that the mechanisms the repo trusts to tell it these defects are gone currently report success
without proving it.**

---

## Scope and provenance

| | |
|---|---|
| Worktree | `/Users/macbook-dev/Documents/GitHub/tldw-review-t2` (fresh, read-only, never committed) |
| SHA | `origin/dev` = `3722a857480b94b30fd4755f3f8e3002bd163ec3` |
| Tree clean | yes — `git status --porcelain` shows only the two untracked `qa/` review directories |
| Python | 3.12.11 (`.venv/bin/python`; system `python3` is 3.9 and was not used) |
| Textual | 8.2.8 |
| `scripts/preflight.sh` | **all 9 derived-artifact checks passed** |
| Lines reviewed | **890,356** across 26 slices (the prompt's table said ~0.8M across 24; see "Scope bug" below) |

### ruff fatal-only baseline (`--select E9,F63,F7,F82`), per Tier-2 package

**2 fatals in 890k lines**, both in one file:

```
tldw_chatbook/Audio/meeting_owner.py:657:41: F821 Undefined name `MeetingCapture`
tldw_chatbook/Audio/meeting_owner.py:738:38: F821 Undefined name `MeetingCapture`
```

Every other Tier-2 package — all 52, plus the 31 `*_Interop` packages — reports **0**. Recorded as baseline, not
fixed. Both are annotation-only (`from __future__ import annotations` at line 15, so the module imports and runs);
`typing.get_type_hints()` on those functions does raise `NameError`, but a repo-wide check found the only production
`get_type_hints` is `MCP/gateway_runtime.py:515`, applied to MCP tool handlers, which never touches these
signatures. **Demoted to P3.** Full evidence in `phase4-verification.md`.

### One pre-existing red, recorded as baseline

`Tests/Architecture/test_module_size_ratchet.py` is **5 failed, 9 passed** at this SHA — not caused by this review:

```
$ .venv/bin/python -m pytest Tests/Architecture/test_module_size_ratchet.py -q
FAILED …[Chat/console_chat_controller.py]      29529 > 29367
FAILED …[UI/MCP_Modules/mcp_workbench.py]       6774 > 6760
FAILED …[UI/Screens/personas_screen.py]        16449 > 16436
FAILED …[Widgets/Console/console_transcript.py] 8399 > 8353
FAILED test_budget_is_not_left_slack[app.py]   slack 152 > 50
5 failed, 9 passed in 0.94s
```

**TASK-32809.1 ("re-pin the red size ratchets", In Progress) is still outstanding.** This matters for the
recommendation in "Duplication clusters": the five new rows this review proposes should land **in the same commit**
as that re-pin, or the ratchet stays red and the new rows are invisible.

### Scope bug found in Phase 0 and fixed

`slice_paths.txt` as delivered — which the prompt states was "verified 2026-09-21 to leave zero files unclaimed" —
left **102 files / 50,631 lines unclaimed**:

| Unclaimed | Files | Lines | Status |
|---|---:|---:|---|
| `tldw_chatbook/Backup_Recovery/` | 80 | 42,955 | **live** — imported by `app.py`, `config.py`, `cli.py` |
| `tldw_chatbook/Workflows/` | 10 | 4,383 | **live** — imported by `app.py`, `UI/Screens/workflows_screen.py` |
| `tldw_chatbook/UI/Workflows_Modules/` | 8 | 2,467 | live |
| 4 stray `tldw_chatbook/Widgets/*.py` | 4 | 427 | live — one of them, `Widgets/status_line.py`, is a **shared helper with 17 importers** |

Neither package is dead code and neither is new enough to excuse: `Backup_Recovery/` landed under `TASK-32628` and
is, by function, the highest-stakes data surface in the repo (backup, restore, staging, rollback, credential
capture, publication, admission control). It had had **zero review coverage in any tier, ever.**

Added as slices **S25** and **S26**; `slice_paths.txt` amended with a comment recording why. The completeness check
now prints `UNCLAIMED: 0` (output at the end of "Coverage"). **S25 immediately returned a measured P1** — an
O(N²) fsync loop in restore publication — which is the concrete argument that the omission mattered.

This is the same failure mode the prompt warns about from the 2026-09-17 run ("three of its four P0s were in the
eight slices it skipped"), one level up: not a slice skipped during execution, but two packages absent from the
scope table before execution began.

### "Already handled" — derived from the backlog at run time

Built from `backlog task list --parent <id> --plain` for `TASK-32800`–`32811` plus a title grep across all 4,305
task files. **20 open consolidation-shaped tasks** at this SHA. The ones that matter most below, because several
findings in this report show them **insufficient as scoped** rather than restating them:

| Task | Status | This review's relationship to it |
|---|---|---|
| `TASK-32803.1/.5` one timestamp helper + adopt | **Done** | **Five independent slices found writers its guard structurally cannot see** (S01, S05, S06, S08, S11) |
| `TASK-32805.5` reconcile strict-JSON families | **Done** | S06 found a **fourth, unenumerated** family (`tldw_api/`), absent from ADR-175's list |
| `TASK-32806.8` image decompression bomb | In Progress | S11 (`Local_Ingestion/`, zero guards) and S12 (the shared image converter) are outside its named scope |
| `TASK-32807.1`-`.6` delete dead code | 3 In Progress / 3 To Do | Six slices found dead clusters in packages **none of the six sub-tasks names** |
| `TASK-32808.4` boolean coercion | **Done** | S12 found a re-roll in the sibling module the sweep missed, **with drift** |
| `TASK-32808.5` adopt the atomic-write helper | **Done** | Four slices found missed sites — **and the helper itself is the weakest of the three implementations in the repo** (see the repo-wide finding) |
| `TASK-32808.6` scope-service scaffold one home | To Do | S24 found its **load-bearing premise is stale** and its AC list under-scopes the work |
| `TASK-32809.2` size budgets for god modules | In Progress | Four slices found modules larger than rows already on the list; the list is hand-picked with no glob |
| `TASK-32804.12` remaining sync-work-on-the-loop | To Do | S12 found its highest-value member (117 of 138 call sites in one module, one of them blocking HTTP) |

Full list in `gap-candidates.md`.

---

## Census delta

**The prompt's quoted repo-wide baseline does not reproduce, and the correct baseline is recorded here.**

The prompt states the repo-wide baseline is "1,823 same-name / 244 verbatim / 665 shape groups at dev
`d8fb4053f9`" and that a 2026-09-21 measurement at `ea2d7b22a8` gave "1,823 / 244 / 664 — flat". Running the
committed `dup_census.py` unmodified against a checkout of `d8fb4053f9` gives **1,931 / 267 / 709**. I could not
reproduce the quoted figures with any exclusion set, so I re-measured the baseline myself and used that:

| | same-name (≥3 files) | verbatim clone groups | shape clone groups |
|---|---:|---:|---:|
| Baseline, re-measured at `d8fb4053f9` | 1,931 | 267 | 709 |
| This run, `3722a857` | **1,943** | **264** | **709** |
| Delta | +12 | **−3** | **0** |

So the prompt's *conclusion* — flat — is right, and its *numbers* are not. Ten stream PRs merged and the
duplication mass did not move.

The committed `qa/core-code-review-2026-09-17/candidates/*.tsv` are **Tier-1-scoped** (519 / 92 / 247 rows, paths
written without the `tldw_chatbook/` prefix). Diffing a repo-wide census against them manufactures growth that is
not there; this run's TSVs are repo-wide and were diffed only against the repo-wide baseline above.

### Flat

Every large cluster is unchanged, to the copy:

| Cluster | defs | distinct bodies | distinct shapes | LOC |
|---|---:|---:|---:|---:|
| `_maybe_await` | 65 | 4 | 3 | 253 |
| `_enforce_policy` | 51 | 6 | 6 | 234 |
| `_require_client` | 47 | — | **1** | 326 |
| `_normalize_mode` | 46 | 46 | 5 | 440 |
| `_dump` | 39 | 3 verbatim groups | — | 278 |
| `_identity` | 31 | — | — | 245 |

Also flat: `Utils/Utils.py::ensure_directory_exists` **0 importers** against 51 files doing raw
`mkdir(parents=True, exist_ok=True)`; `truncate_content` **0 importers** against 24 files; `form_components` 2
importers.

**These clusters survived a complete review-file-fix cycle at exactly the same size.** None of them has a guard.

### New

| Cluster | Where | Note |
|---|---|---|
| `_coerce_int` ×4, byte-identical | `LLM_Calls/{mistral,openrouter,groq,deepseek}.py` | **Landed 2026-09-19, two days after the baseline, in a *consolidation* PR** (`TASK-32852`) — and in the same four modules `TASK-32808.4` had just cleaned of `_coerce_bool`. `Utils` has `coerce_bool_flag` and **no integer counterpart.** |
| `mistral.py` provider template ×4 each | `_mistral_turn_response`, `_log_usage_metrics`, `_log_error_metrics`, `validate_finish` | the provider template cloned again by the same PR |
| `Workflows/` session lifecycle ×2 | `subscribe`, `discard_setup`, `begin_close`, `abort_quit` | fresh duplication inside **the package the scope table missed** |
| `_reveal_focused_control` ×2 | `UI/MCP_Modules/mcp_audit_mode.py:539`, `Widgets/Library/library_search_rag_panel.py:92` | |
| `build_conflict_comparison` ×2 | `Notes/notes_sync_conflicts.py:254`, `Notes/file_notes_conflict_compare.py:169` | same name, same four bound constants, same elision marker string, **two different truncation semantics**. Missed by the shape hash. |
| root-overlap ×3 | `Notes/` | two semantics (lexical vs `samefile`); **the lexical one guards sync-root admission and the legacy module does not trust it** |

The `_coerce_int` row is the single most useful data point in this review. Full evidence in
`phase4-verification.md`.

### Closed — verified by census diff, not by reading the PRs

- `_datetime_to_iso` ×5 files (`runtime_policy/source_state.py` et al.) — TASK-32803
- `_coerce_bool` ×3 (`Image_Generation/config.py` et al.) — TASK-32808.4
- The `LLM_Calls/moonshot.py` validator family — **8 shape groups** (`_positive_integer`, `_positive_number`,
  `_nonnegative_number`, `_nonnegative_integer`, `_normalize_call_batch`, `_normalize_stop`,
  `_normalize_response_format`, `_json_shape_is_bounded`) — stream #2738
- `Utils/Utils.py::extract_text_recursive` / `extract_text_from_segments`
- `Utils/ui_helpers.py`, `Utils/pagination.py`, `Widgets/base_components.py` — **deleted** (TASK-32807.6;
  `base_components.py` in `5f3adeca33`). All three were 0-importer helpers in the 2026-09-17 seed table.

---

## Duplication clusters (D4) — repo-wide, per §2

Every cluster below carries **both** a canonical home **and** a guard decision. The measurements are repo-wide
(all ~1.7M lines minus the always-excluded paths), not slice-scoped. Working notes:
`candidates/phase3-clusters.md`.

### The clusters that are real

| Cluster | Copies | Helper exists? | Drift | Outward / downward LOC | Canonical home | Size | Rec |
|---|---:|---|---|---|---|---|---|
| **scope-service scaffold** (`_maybe_await` 65, `_enforce_policy` 51, `_require_client` 47, `_normalize_mode` 46, `_identity` 31, `_dump` 39) | 47 services, 31 in `*_Interop` | no | **none in the bodies; the drift is in the `mode=None` default (25 SERVER / 19 LOCAL) and in what each call site passes to the gate** | **1,776 out / ~0 in** (stdlib only, +1 edge if it wraps `run_finite_local_worker`) | `runtime_policy/` | L | ✅ — owned by **TASK-32808.6**, whose premise is stale |
| **`atomic_file_ops` is the weakest of three implementations** | 17 importers | **yes, and it is the problem** | helper fsyncs the file, **never the parent dir**, plain `os.fsync` not Darwin `F_FULLFSYNC`; `Personal_Context/key_protector.py` and `Utils/private_paths` are both **stronger** | — | harden `Utils/atomic_file_ops.py` | M | ✅ **new** — inverts TASK-32808.5 |
| **`Utils/timestamps.py` non-adopters the ADR-173 guard cannot see** | 6 slices found writers | yes (28 importers) | `strftime("…%f")[:-3]+"Z"` and `datetime.now(tz).isoformat()` both pass the guard | — | `Utils/timestamps.py` + widen the guard | M | ✅ **new** — shows TASK-32803.5 insufficient |
| **`_coerce_int`** | 4, byte-identical | **no** (`coerce_bool_flag` has no integer counterpart) | none | ~30 / 0 | `Utils/Utils.py` | S | ✅ **new, landed after the baseline** |
| **`set_status_line` non-adopters** | ~6 of ~23 `_set_status` | yes (17 importers) | **crash shape**: `is_mounted` guard ≠ `missing_ok=True` | — | `Widgets/status_line.py` | S | ➖ owned by **task-32861** |
| **`run_finite_local_worker` non-adopters** | 3 adopters, 5 re-rolls, 2 opposite fail-directions | **yes, 9 importers** | `_is_memory_backed` threads when unsure; `_call_off_loop` refuses to | — | `Backup_Recovery/participants.py` | M | ✅ **new** — the helper TASK-32808.6 must wrap, named nowhere in it |
| **filename sanitizers** | 7 | partial | **measured**: the ComfyUI copy is the **strictest**; `Utils/text.py` passes `..`, NUL bytes and leading dashes | — | new `Utils/filename_safety.py` with **two** entry points | M | ✅ — **TASK-32808.2's direction must be inverted** |
| **one-shot LLM call trio** (`_error_text`, `_effective_max_tokens`, `_invoke_chat`) | 3 | no | `_error_text` overshoots `ERROR_CHAR_CAP` by 6 in 2 of 3; the DeepSeek reasoning-budget fix landed three times by hand | — | `Chat/one_shot_call.py` | M | ✅ **new** |
| **`maintenance_drain`** | **20+**, not the census's 5 | no | docstring + error string only | — | — | M | ➖ owned by **TASK-32808.11**; arity is larger than filed |
| **image magic-byte sniffing** | 11 modules | **yes, in the wrong package** | none found | — | move to `Utils/` | S | ✅ — *the placement inside `Image_Generation/adapters/` is why nobody adopted it* |
| **`_as_dict`** | 8 | no | **opposite contract on the fallthrough**: `return dict(value)` vs `raise TypeError` | — | `runtime_policy/` (a .6 member) | S | ✅ |
| **freeze family** | 5 | no | **none — semantically identical**; the census put it in `dup_shape`, it belongs in `dup_verbatim` modulo the name | — | `Utils/` | S | ✅ |

### The clusters that are NOT real — retired with evidence

A consolidation PR acting on the census rows alone would break each of these.

| Census row | Verdict |
|---|---|
| `_perform_safe_cancel` 45 defs | **28 distinct shapes.** A template-method override against `request_safe_cancel`/`dismiss_safe_once`/`run_cancel_effect_once`. **Consolidating would be a regression.** The 2026-09-17 seed table read it by *name*. |
| `_initialize_schema` 16 defs / **2,120 LOC — the largest mass in the census** | `@abstractmethod` on `DB/base_db.py:811`. Sixteen different schemas. |
| `_get_connection` 9 subclass overrides | **Retired, but my first statement of the reason was wrong and S17 caught it.** I wrote "all call `super()._get_connection()`". **Seven do; two do not** — `Notifications/event_state_repository.py:207-221` and `DB/Library_Ingest_Jobs_DB.py` call `connect_private_sqlite(...)` **directly**, both with `check_same_thread=False`, which `BaseDB` cannot pass. The event-state one documents this at length (TASK-21131: held connections are closed from another thread, and sqlite3's default guard would refuse that). **The conclusion is unchanged — all nine route through `connect_private_sqlite` and the PRAGMA divergences carry per-store task references, one saying *"Do NOT copy this pattern into a store that holds connections"* — but anyone auditing by grepping for `super()._get_connection()` gets a false negative on those two.** |
| `relocate`/`capture`/`validate` across 11 `*/recovery.py` | **Two slices reached opposite readings.** S25 (from `Backup_Recovery/`) ruled it Protocol conformance — `models.py:131-137` declares `OwnerAdapter(Protocol)` with no base class, so every owner must supply its own. S08 (from `Scheduling/`) called them copy-pasted bodies differing only in a string literal. **Both are right about different halves:** it *is* Protocol conformance, *and* a `OwnerAdapterBase` supplying the two default bodies would delete ~40 lines across 12 packages at zero behavioural risk. **Rec: ➖ low-value, do it only if `Backup_Recovery/models.py` is being touched anyway.** |
| `_coerce_bool` 11 defs | **7 distinct bodies with real divergence.** TASK-32808.4 (Done) deliberately scoped itself to 3 verbatim copies and **documented why the rest differ** — and they do: `coerce_bool_flag` stringifies ints, so `5` → `default` where the rail-state copies return `True`. **Adopting it would be a behaviour change.** Two internal 3-copy groups remain, each collapsible *locally*. |
| `Utils/Splash.py` 0 importers | Census artifact — its one importer is inside `Utils/Splash_Screens/`, which the census excludes. |

### The guard argument

**The repo already owns the guard technology, and it worked.** `scripts/check_timestamp_writers.py` +
`scripts/timestamp_writer_census.tsv` is a shrink-only ratchet keyed on `module<TAB>symbol<TAB>kind<TAB>count`,
wired into `preflight.sh` and the required `Derived artifacts` CI job. TASK-32803 used it to take a cluster the
2026-09-17 review measured at ≥7 drifted formats down to zero — and to *hold* it there. Five siblings exist in the
same file.

Every cluster in the **Flat** bucket has no guard. Every cluster in the **New** bucket landed into a codebase with
no guard covering it. `_coerce_int` is the proof: a PR whose stated purpose was consolidation shipped four fresh
byte-identical copies **forty-eight hours after** the review that catalogued the pattern, into the same four modules
that PR's predecessor had just cleaned. Nothing could object.

**But a guard is not free. Three of the repo's wired, green, CI-required guards are each wrong in a different
way, and this review found all three:**

| Guard | Failure mode | Found by |
|---|---|---|
| `check_timestamp_writers.py` | **Too narrow** — matches two idioms, not the contract. Reports `OK` with an **empty census** while six slices independently found writers it cannot see. | S01, S05, S06, S08, S09, S11, S13 |
| `check_textual_worker_contract.py` (W002) | **Wrong predicate** — counts a `query_one` in a `finally:` body as *guarded*, hiding **59 sites**, and `finally` is the variant that runs during cancellation. | S19 |
| `check_canvas_mermaid_assets.py` | **Circular** — 2 of its 6 "reproduced" outputs are copied out of the directory the check later compares them against. | S13 |

A green guard is a claim. These three claim more than they prove — which is the argument for **reviewing guards as
code**, and for one rule: *a guard must assert the contract, not the two idioms that were wrong last time.* An empty
ratchet read as "closed" is worse than no ratchet, because it retires the question.

**Recommended guards, in priority order — note that the top three are repairs to existing guards, not new ones:**

| Guard | Asserts | Why it earns its place |
|---|---|---|
| **Fix `check_textual_worker_contract.py`'s W002 section test** | only a `Try.body` with non-empty `handlers` counts as guarded | a one-line fix to an *existing* guard that is currently lying; re-pin the 59 as baseline |
| **Widen `check_timestamp_writers.py`** with a third kind for `.now(<any>).isoformat()` and the `strftime(…%f)[:-3]` shape | the ADR-173 *contract*, not two idioms | same — an existing guard that reports green while the thing it guards drifts |
| **Fix `check_canvas_mermaid_assets.py`** | either genuinely derive the two `_v2` files from pinned inputs, or move them to a separately-declared *vendored, not derived* inventory with its own digests | the check currently compares two of its six outputs against the directory it copied them from |
| **`scripts/check_scope_service_contract.py`** | every public method on a `*ScopeService` transitively reaches `_enforce_policy`, minus a shrink-only exemption TSV | **found 24 ungated public methods, 3 of them real gaps, in one afternoon.** It is the one thing in that cluster a reviewer cannot see and a count cannot answer |
| **A ~20-line AST check for shadowed method definitions** | no class defines the same method twice | `ruff --select F811` **flags none of the 11 instances** — an intervening method between the two defs suppresses it (reproduced in an isolated 22-line file) |

**Explicitly NOT recommended:** a guard against a 48th `_maybe_await`. The consolidation removes the reason anyone
writes one, and `git grep -c "def _maybe_await"` in review catches the rest. A guard nobody needs is the same
over-engineering this review exists to find.

---

## Dead or under-adopted shared helpers

| Helper | Importers | Hand-rolled equivalents | Rec |
|---|---:|---|---|
| `Utils/Utils.py::ensure_directory_exists` | **0** | 86 raw `mkdir(parents=True, exist_ok=True)` in 51 files | **delete** — 0 adoption across two reviews |
| `Utils/Utils.py::truncate_content` | **0** | 24 files inline `[:n] + "..."` | adopt or delete (**TASK-32808.3**, In Progress) |
| `Widgets/form_components.py` | 2 | — | keep; 2 importers across a TUI of hundreds of forms is a standing question, not a defect |
| `Utils/secure_temp_files.py` | 7 | **65** files raw `tempfile.*` | **most "bypasses" are correct** — several slices found the local shapes *stricter* (`O_EXCL\|O_NOFOLLOW`, mode-at-open, fd identity pinning). Re-scope before sweeping |
| `Utils/atomic_file_ops.py` | 17 | 28 files `os.replace` without it | **harden the helper first** — see the D4 table |
| `Utils/optional_deps.py` | 40 | 61 files `try/except ImportError` without it | keep; most are documented degrade-correctly guards |
| `Utils/path_validation.py` | 111 | 4 inline checks | **3 of the 4 are stronger than the helper.** `safe_join_part`/`validate_filename` does **not** reject `:` (Windows drive-relative) and `validate_path` defaults `allow_hidden=False`. **Adopting it at `STT/executor_worker.py:503` would be a functional regression.** Settle both before any sweep |
| `Utils/log_sanitizer.py` | 19 | 0 in `Personal_Context/`, `Web_Scraping/`, `WebClipper/`, `Web_Server/` | census row, not per-site findings |
| `Utils/ui_helpers.py`, `Utils/pagination.py`, `Widgets/base_components.py` | — | — | **DELETED** ✅ since 2026-09-17 |

---

## Verified-fine (do not "fix")

Beyond the prompt's seed list, this review **retired** the following with evidence. Each would have been a plausible
finding from the census alone:

- **`Backup_Recovery/archive_reader.py`** independently re-parses the ZIP central directory before handing the
  stream to `zipfile`, bounds decompression per-chunk against the declared size, verifies CRC, rejects
  symlink/dir/non-deflate members, and gates high-compression archives behind an explicit
  `CompressionReviewRequired`. **Path traversal, symlink escape and decompression bombs are all genuinely closed.**
- **`Notes/note_import_discovery.py`** is descriptor-relative throughout, identity-checks the leaf **before open,
  after open, and after read**, then re-walks lexically a second time. The best-hardened filesystem code in the repo.
- **`Model_Artifacts/fetch.py`** enforces `max_bytes` mid-stream, per-hop egress, cross-origin credential stripping
  at both header and client-`auth` level, `follow_redirects=False`, HTTPS-downgrade rejection, and SHA-256
  verification with bounded refetch.
- **`TTS/voice_bundle_codec.py`**: 40 MiB archive cap, 33 MiB uncompressed, expansion ratio 100, per-member limits,
  member-name allowlist, `allowZip64=False`, **no path extraction at all**.
- **`TTS/profile_repository.py`**: fd-relative `O_NOFOLLOW|O_DIRECTORY` opens, file **and parent** fsync at every
  publication point, a 12-stage fault-injection checkpoint enum, `_OpaqueAuthority` refusing pickle/copy, a
  `published`/`uncertain` split after `os.replace`. **None of the three data-loss shapes the brief asked about.**
- **`Backup_Recovery/credentials.py` secret containment holds.** Every `except` raises `from None` with a fixed
  opaque code; the whole 43k-line package has **9 logging call sites**, all constant format strings, the only
  dynamic value being `type(error).__name__`.
- **`Subscriptions/` XXE**: `pytest Tests/Subscriptions/test_watchlist_opml_entity_expansion.py -q` → **14 passed**;
  fully hardened, no register entry in that slice.
- **`Backup_Recovery/age_worker.py` integrity pin is current** — `shasum -a 256` matches `crypto.py:29` exactly.
- **Shipped placeholder TTS credentials are unreachable** — `DEFAULT_APP_TTS_CONFIG` is merged under a different key
  than the backends read. (A *user-written* `[app_tts]` section is honoured — that is the separate P1.)
- **`persona_visual_participants` vs `visual_identity_participants` locking** — both lock, at different
  granularities. **A cross-slice conflict resolved in favour of the retiring slice.**
- `Local_Ingestion/__init__.py`'s PEP 562 lazy `__init__`; `UI/Logs_Window.py:459 _compile_pattern`;
  `Subscriptions`' scheduler `thread=True` — all as the prompt's seed list says.

**Method note worth keeping.** Ten `*_Interop` packages export through a PEP-562 `__getattr__`/`import_module`
table. A naive import-graph walk reports **7 dead packages and 662 unreachable modules repo-wide**; the
lazy-aware walk (resolving string constants that name a module) reports **0 dead packages**. *The naive answer is
the trap.* Any future reachability sweep must resolve string-literal imports.

---

## Retired / contested

Findings raised during this review and then retired, per rule 6:

| Raised | Retired because |
|---|---|
| `_maybe_await` hides sync-sqlite-on-the-loop on the persona path (the seed table's open question) | **Not on the path I traced first.** `list_characters` hits sqlite but **has no callers**; `list_persona_profiles` is reached from the UI but filters an **in-memory list**. My module-level regex ranking (`.execute(` anywhere in the module) scored it "105 sync DB methods" — a weak signal that was wrong. **S09 and S12 then found real instances elsewhere** (`character_persona_scope_service.py:289 get_character`, and 117 of 138 sites in `Media/media_reading_scope_service.py`, one of them a **blocking HTTP fetch** with no timeout). |
| `Evaluations_Interop` shadowed `_enforce_policy` is a live P0 `TypeError` | The `TypeError` reproduces, but **the containing method has no caller in its own class**. Latent, P3. |
| `ChatScreen.on_button_pressed` shadow loses functionality | Real double definition (201 dead lines), but an AST diff found **zero string constants present only in the dead body** — duplicated-then-extended, not lost. Still a trap for the next editor. |
| S25: persona-visual locking has drifted | S15/S16 retired it; **both lock**, per-source vs per-candidate. The *duplication* half survives. |
| S21: the endpoint probe has 5 callers | **4.** An import line was counted alongside its call. S19's count was right; S21's restore-vector contribution is the valuable half. |
| The 31 `*_Interop` packages contain the repo's largest dead block (66k lines) | **All 31 are reachable** from the composition root. Residue is **753 lines (1.1%)**. |
| `Notes/` template rendering is a jinja2 sandbox risk | `grep -rn "jinja2" tldw_chatbook/Notes/` → **zero**. The notes template system is a 51-line JSON store with no rendering at all. **The real finding there is durability.** |
| `LLM_Management/` supervises inference servers and has `subprocess` timeout gaps | `grep -rn "subprocess.run"` over that package → **zero matches.** Supervision lives in `Event_Handlers/`. |
| `library_ingest_state.py:2707` bypasses `path_validation` (2026-09-17 seed entry) | `os.path.commonpath` there derives a **display label** for a batch header. No containment check, no filesystem access. `path_validation` has no "common ancestor" helper and would be the wrong tool. |
| `STT/executor_worker.py:503` bypasses `path_validation` (2026-09-17 seed entry) | The inline block is **stricter** — it rejects `PureWindowsPath` drives/roots/backslashes, which `validate_filename` does not. **Adoption would be a regression.** |

---

## Method appendix

**Scripts.** Reused the 2026-09-17 tooling unmodified where possible, per §2:
`candidates/dup_census.py` (run repo-wide), `candidates/helper_adoption.py` (patched: `Widgets/base_components.py`
no longer exists), `candidates/pattern_greps.py` (patched: `T1` replaced with the 207 Tier-2 paths; tolerate
missing paths), `candidates/variants.py` (patched: worktree path). Outputs in `candidates/`.

**Counts.** `dup_by_name.tsv` 1,943 rows · `dup_verbatim.tsv` 264 · `dup_shape.tsv` 709 (2,526 files scanned).
`patterns/` 28 TSVs over 1,442 Tier-2 files; largest categories: `function_body_import` 1,957 rows /364 files,
`run_worker_coroutine` 454/47, `except_exception_return` 409/188, `legacy_markers` 919/229, `try_import_guard`
365/144, `except_exception_pass` 335/118, `fetchall_no_limit` 259/51, `raw_1024x1024` 208/111.

**Subagents.** 24 read-only agents over 26 slices, ≤6 concurrent. Each received the ground rules, the four
dimensions, the rules of evidence, the run-time "already handled" list, its own file list with line counts, and the
candidate rows falling inside its slice. Per-slice reports in `slices/`.

**Verification.** `phase4-verification.md` records every P0/P1 reproduction, every demotion, and the three
cross-slice conflicts resolved.

---

## Two corrections to this report's own Phase 3, found late

### 1. `Chunking/engine/**` is vendored and must be excluded from every D4 recommendation

```
$ head -6 tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml
# Vendored from tldw_server — NEVER hand-edit files listed here (spec §5.2).
# Re-sync: python Helper_Scripts/sync_chunking_engine.py
[upstream] repo = ".../tldw_server.git"  commit = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

$ # manifest vs tree
vendored listed: 38 | .py present: 39 | not vendored: ['__init__.py'] | vendored lines: 14,398
$ ls Helper_Scripts/sync_chunking_engine.py   -> exists
```

**14,398 of the 890,356 lines reviewed are not this repo's to change.** Any census row whose members all sit inside
`Chunking/engine/**` is **not actionable here** (`chunk_generator` ×4, `_coerce_bool_option` ×2, `tokenizer` ×2), and
a row that *mixes* vendored and local members can only move the local copy. A consolidation PR touching them would
be silently reverted by the sync script.

This also means the repo-wide census numbers in "Census delta" include ~14.4k vendored lines. The **flat** reading
is unaffected (the same lines were in both measurements), but any future "duplication per 1k lines" metric should
subtract them.

**A live consequence:** `Chunking/engine/security_logger.py` carries a **local hand-edit** — the
`Utils/timestamps.utc_now_iso` adoption from `c255ec3936` (task-32803.1) — in a file the manifest says never to edit
and the sync script will overwrite. `Tests/Architecture/test_vendor_pin_consistency.py` pins the upstream *commit
sha*, not per-file content, so nothing records that the fix must be re-applied. **The ADR-173 fix lives in a file
that reverts on the next re-sync.**

### 2. The fail-direction question blocking TASK-32808.6's ADR now has an answer

S24 found that nine scope services thread local work off the loop in **five mutually incompatible shapes with two
opposite fail-directions**, and that AC#2 ("the threading rule the fixed service already documents") is therefore
unimplementable as written. S10 settled it:

**Standardize on "do not thread unless positively confirmed safe." `RAG_Admin`'s direction is correct.**

The two predicates compute the *same thing* for the real local services —
`local_rag_admin_service.py:309-329 diagnostics_are_thread_safe()` is literally
`not bool(getattr(self.media_db, "is_memory_db", False))`, the same expression as `_is_memory_backed`. They diverge
only on an **unrecognized** synchronous local service:

- **RAG_Admin:** attribute absent → **inline**. Failure mode: a UI freeze.
- **Chat/Media:** `media_reading_scope_service.py:196` and `chat_conversation_scope_service.py:195` both end in a
  bare `return await asyncio.to_thread(fn, ...)` for shapes they could not identify. Failure mode:
  `sqlite3.ProgrammingError` from a `check_same_thread` connection — **or a silently *empty* `:memory:` database
  returning a wrong answer with no error at all.**

**The hazards are not symmetric. Wrongly inlining costs latency; wrongly threading costs correctness.**
`local_rag_admin_service.py:317-322` states exactly this in prose — *"it opens a different, empty database and the
census would silently report zero legacy items instead of raising"* — and then the two siblings choose the opposite
default.

**Blast radius, so the ADR does not overstate it: this divergence has zero production impact today.** Both
`_is_memory_backed` sites gate on `mode == local` *and* `not iscoroutinefunction` before reaching the fallback, and
both `RAG_Admin` server methods are `async def`, so the `iscoroutinefunction` branch fires first. It is a
latent-default problem — which is precisely the kind of thing an ADR should settle rather than a P1 should fix.

---

## Executive summary — the ten that matter

1. **[P0][D1]** Console builds **no** Library tool provider on the default configuration — a factory passes a
   constructor kwarg deleted 20 days earlier, and the `TypeError` is swallowed into a WARNING.
   `Chat/console_runtime.py:675` · **S05** · **S**
2. **[P0][D1]** A Schedules sync while the view is on "This device" mirrors server-owned reminders *and* automation
   definitions under `owner_id="local"`, arming them for local execution — the double-execution ADR-077 exists to
   prevent. `Scheduling/services/sync_engine.py:265` · **S08** · **M**
3. **[P0][D1]** `SimpleAudioPlayer.stop()` SIGKILLs the whole app's process group, because its children are spawned
   without `start_new_session`. `TTS/audio_player.py:348,556` · **S03** · **S**
4. **[P1][D1]** A default install fails every PDF ingest with `AttributeError: 'NoneType' object has no attribute
   'FileDataError'` instead of "install tldw_chatbook[pdf]". `Local_Ingestion/PDF_Processing_Lib.py:614` · **S11** · **S**
5. **[P1][D1]** 117 of 138 local-mode call sites in `Media/media_reading_scope_service.py` bypass the threading seam
   built for them — one of them a **blocking HTTP fetch with no timeout** on the event loop. **S12** · **M**
6. **[P1][D1]** A re-chunk that yields zero chunks hard-DELETEs every stored chunk row and reports `"rechunked"`.
   `Library/library_rechunk_service.py:301` · **S05** · **S**
7. **[P1][D1]** One unreadable read of the Chatterbox voice-profile file plus one later edit erases the user's
   entire voice library; the Higgs sibling takes a backup first. `TTS/backends/chatterbox_voice_manager.py:48` · **S03** · **S**
8. **[P1][D1]** The endpoint probe's chat branch reaches the wire with the user's `base_url` **and** API key and no
   egress check — while the TTS branch of the same function checks. A restored profile archive can set the host.
   `UI/Screens/settings_endpoint_probe.py:607` · **S19 + S21** · **M**
9. **[P1][D2]** Restore publication is **O(N²) in `F_FULLFSYNC` calls** — **8.28 s measured with a 12-record
   journal**; projects to hours at the admitted artifact ceiling. `Backup_Recovery/journal.py:1626` · **S25** · **M**
10. **[P1][D1]** The recorder's VAD gate discards the tail of every capture chunk that is not a multiple of 20 ms —
    **6.67% of audio measured at `buffer_duration_ms = 150`** — on the one path where the audio *is* the product,
    and the correct version is in the same package. `Audio/recording_service.py:511` · **S07** · **S**

### And the finding that is about the review process rather than the code

**`slice_paths.txt` left 102 files / 50,631 lines unclaimed**, including the entire `Backup_Recovery/` package.
That package had never been reviewed in any tier, and it returned a measured P1 within an afternoon.

---

## Findings

**283 findings: 3 P0 · 57 P1 · 115 P2 · 108 P3.** Full text, evidence, recommended correction, size, ADR check,
confidence, pinning test and "already covered" for **every** finding is in the per-slice reports under `slices/`,
in the §5 format. Reproduced below: all three P0s in full, then the complete P1 index.

### P0 [D1] — Console builds **no** Library tool provider at all on the default configuration
- Where: `Chat/console_runtime.py:675` vs `Library/local_library_tool_service.py:448-460`; swallowed at
  `Chat/console_chat_controller.py:27191-27198` and `:20767-20785`.
- Evidence: `inspect.signature(LocalLibraryToolService.__init__)` has no `collections_service` and no `**kwargs`;
  constructing it as the factory does raises `TypeError`. The parameter was removed in `5dd1077df6` (2026-09-01);
  the sibling factory `UI/Console_Modules/library_activity.py` **was** updated. The broken factory is the one that
  runs — `console_runtime.py:3647` uses `kwargs.update(...)`, not `setdefault`, so it **overwrites** the screen's
  maintained factory. `direct_library_tools` defaults to `True`, and the `LibraryRagToolProvider` fallback sits in
  the *other* branch, so nothing is returned at all. Full chain in `phase4-verification.md`.
- Why it matters: on stock config **every Console agent run silently loses all 18 `library_*` tools**, with no
  fallback. The only trace is one WARNING line.
- Recommended correction: delete the argument; better, delete `_library_provider_for_app` and let the screen's
  factory survive — it also threads `capture_kwargs(turn_context)`, which the runtime copy drops.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: **none, and that is the cause** — both integration tests inject a *fake* `library_provider_factory`,
  so no test ever constructs the real one.
- Already covered: none.

### P0 [D1] — A Schedules sync on "This device" arms server-owned reminders and automations for local execution
- Where: `Scheduling/services/sync_engine.py:265-273` (`target_owner = owner_id if … else self.owner_id`), stored by
  `_apply_pulled_reminders` / `upsert_automation_definitions_from_server`; trigger
  `UI/Screens/scheduling/schedules_workbench.py:5403` passes `service.owner_id` — **the UI view toggle** — straight
  through.
- Evidence: reproduced against a real `ScheduledTasksDB` — a mirrored server reminder shows
  `PriorityQueue.pop_due` → `ARMED LOCALLY`, and a server automation definition shows `ARMABLE LOCALLY`. Every
  ADR-077 guard keys on the `server:` prefix (`scheduler/queue.py:37`), and **the pull is the one writer that never
  puts it on.**
- Why it matters: ADR-077's **rejected** alternative is named as *"Both sides execute, dedupe at delivery … for
  agent work this is double execution with nondeterministic ordering, and dedupe after the fact cannot un-run side
  effects."* Duplicate notifications, and for `recurring_question` definitions **a second unattended LLM run per
  occurrence at the user's expense.**
- Recommended correction: `SchedulingService._active_server_owner_id()` already computes the right value **and its
  docstring already states the rule**; thread it into `sync_now`/`pull` as the storage owner.
- Size: M · ADR: no (ADR-077 decides it) · Confidence: **verified**
- Pinning test: none — every pull test constructs `SyncEngine(..., owner_id="server:1")`; the one `"local"` case
  passes `server_client=None` and never pulls.
- Already covered: none.

### P0 [D1] — `SimpleAudioPlayer.stop()` SIGKILLs the whole tldw_chatbook process group
- Where: `TTS/audio_player.py:556-559` (`os.killpg(os.getpgid(...), SIGKILL)`); children spawned at `:348`, `:353`,
  `:360` with **no `start_new_session`**.
- Evidence: executed — a `Popen` child with no `start_new_session` inherits the parent's process group
  (`os.getpgid(child) == os.getpgrp()` → **True**), so `killpg` targets the app itself and everything else in the
  terminal's foreground group. **The repo ships the correct pairing six times elsewhere, each with a comment; TTS is
  the only site that calls `killpg` without it.**
- Why it matters: the user presses Stop on TTS playback and **the whole TUI is SIGKILLed** — no shutdown, no
  `close_tts_resources()`, no DB quiesce.
- Size: S · ADR: no · Confidence: verified (pgid identity proven; the `TimeoutExpired` precondition — a player
  wedged in uninterruptible I/O >0.5 s after SIGKILL — is inferred, not reproduced)
- Already covered: **no.** TASK-32806.5 names only `server_lifecycle.py`, and the defect is worse here: that task's
  site *leaks* a child, this one *kills the parent*.

### P1 — complete index (full text in the named slice report)

| Sev | Dim | Slice | Finding |
|---|---|---|---|
| P1 | D1/D4 | `S01-notes-a` | The File Notes editor-save path writes without `fsync`, so a crash between `os.replace` and writeback loses both the old and new note bytes |
| P1 | D1 | `S01-notes-a` | The reconciler's `stale_observation` safety gate is unreachable: the only production caller derives both generations from the identical expression |
| P1 | D1 | `S02-notes-b` | 11 of 14 `NotesScopeService` async methods run synchronous SQLite on the event loop; 3 offload |
| P1 | D2 | `S02-notes-b` | every note save reads the entire `keywords` table into Python and scans it in O(n·m) |
| P1 | D1 | `S02-notes-b` | `delete_workspace_note` takes a required `version` and silently discards it; its two siblings honour theirs |
| P1 | D1 | `S02-notes-b` | workspace source and artifact deletion are gated by the *update* policy action, not *delete* |
| P1 | D1 | `S02-notes-b` | Obsidian `[[wikilinks]]` are rewritten only on the Create path; "Update existing" stores them unresolved |
| P1 | D1 | `S03-S04-tts` | A transient read error on the Chatterbox voice-profile file permanently destroys every Chatterbox voice profile; there is no backup |
| P1 | D1 | `S03-S04-tts` | The Kokoro ONNX model is downloaded, its SHA-256 is computed, and the digest is explicitly thrown away; the artifact is loaded by onnxruntime unverified, with no size cap and no durability barrier |
| P1 | D1+D2 | `S03-S04-tts` | Audiobook generation runs the entire decode/concat/normalize/ffmpeg pipeline on the event loop, and its `ffmpeg` call has no `timeout=` and inherits the TUI's stdin |
| P1 | D1 | `S03-S04-tts` | The two remote TTS backends read credentials env-first (contradicting the settled ADR-012 amendment) and gate them on truthiness, so `<API_KEY_HERE>` and whitespace-padded keys reach the provider |
| P1 | D1 | `S03-S04-tts` | Higgs voice-profile backups are named with naive local time, so a DST fall-back overwrites a backup, and `restore_from_backup` then restores the *older* file |
| P1 | D1 | `S03-S04-tts` | audio.cpp child-process diagnostics are Rich-escaped into a `markup=False` RichLog — and TASK-32802.1's sweep would make it *worse* here, not better |
| P1 | D1 | `S05-library` | A re-chunk that yields zero chunks hard-DELETEs every stored chunk row for the item, inserts nothing, and reports `status: "rechunked"` |
| P1 | D1 | `S05-library` | A cancelled capture extraction never settles: the service's cleanup reason is not in the repository's allowlist, the refusal is swallowed, and the row stays `processing` with a live 300 s lease and no Retry |
| P1 | D1 | `S06-tldw-api` | every non-2xx response on an SSE endpoint escapes the package's exception family as a raw `httpx.ResponseNotRead` |
| P1 | D1 | `S06-tldw-api` | `tldw_api` is a fourth, unenumerated strict-JSON family: the tldw_server wire boundary accepts `NaN`, silently last-wins duplicate keys, and lets a deep body escape as a bare `RecursionError` |
| P1 | D1 | `S07-speech-in` | The recorder's VAD gate silently discards the tail of every capture chunk that is not an exact multiple of the 20 ms VAD frame; a user-set `dictation.buffer_duration_ms` of 150 loses 10 ms of speech out of every 150 ms |
| P1 | D1/D4 | `S07-speech-in` | Voiceprint writes (biometric-derived) are atomic but not durable: `Audio/voiceprint.py` re-rolls the atomic-write helper without its `fsync`, after TASK-32808.5 closed |
| P1 | D1 | `S08-scheduling` | Watchlists `sitemap` sources fetch sitemap-**discovered** URLs with `trusted_origins` seeded from those same discovered URLs, so a `<loc>` naming a private/loopback address is fetched |
| P1 | D1 | `S09-characters` | An imported character card whose `character_book` survives conversion permanently kills world-info for that character, and the failure is logged at DEBUG only |
| P1 | D1/D4 | `S09-characters` | Chat-dictionary regex keys bypass the ReDoS validator that world-info entries use; an imported card can wedge a send thread indefinitely |
| P1 | D1 | `S10-S26-chunking-workflows` | The OpenAI-compatible embedding backend makes raw `requests` calls to a config-supplied URL with no egress policy |
| P1 | D1 | `S11-evals-ingest` | A default install crashes with `AttributeError: 'NoneType' object has no attribute 'FileDataError'` on every PDF ingest instead of saying "install tldw_chatbook[pdf]" |
| P1 | D1 | `S11-evals-ingest` | The image-ingest path has neither a byte cap nor a decompression-bomb guard, while eight other modules in the repo have both |
| P1 | D1 | `S11-evals-ingest` | The default MOBI path reads the whole file into memory and then iterates it byte-by-byte in Python |
| P1 | D1 | `S11-evals-ingest` | Plaintext and HTML ingest read the entire user-picked file into memory with no ceiling |
| P1 | D1 | `S11-evals-ingest` | A successfully committed media row is reported to the user as an ingest failure when the post-commit chunking-config UPDATE fails |
| P1 | D1 | `S12-media` | Every image adapter's format-conversion path decodes backend-supplied image bytes with no pixel cap; an 87 KB PNG expands to 90.25 M pixels and converts successfully |
| P1 | D1 | `S12-media` | 117 of 138 local-mode scope-service call sites run synchronous sqlite (and in one case a blocking HTTP scrape) inline on the event loop; only 21 use the threading seam built for exactly this |
| P1 | D1 | `S12-media` | `_clone_git_repository` is the only `subprocess.run` in the slice with no `timeout=`; a hostile or slow git host wedges the ingest worker forever |
| P1 | D1 | `S12-media` | The yt-dlp stream branch egress-validates one hop, then hands the URL to ffmpeg, which follows redirects itself — the exact bypass the module's own redirect walk exists to prevent |
| P1 | D1 | `S12-media` | Reading-list export silently loses body content on any per-item DB error, and silently truncates or empties on filters it cannot honour |
| P1 | D1 | `S13-canvas` | `ShadowRepo._locked()` spins forever, at ~28,000 iterations/second with a `logger.warning` each, when the cross-process lock dir exists but cannot be `rmdir`'d |
| P1 | D1 | `S13-canvas` | `set_folder_binding_access` performs the same binding-metadata read-modify-write as the exclusion editors but does not take `_BINDING_EXCLUSION_EDIT_LOCK`, so one silently reverts the other |
| P1 | D1 | `S14-web` | Every production Confluence API call bypasses the module's own egress guard; the pinning test exercises the one call shape production never uses |
| P1 | D1 | `S14-web` | `Article_Scraper/__init__.py` is empty, so `Subscriptions`' generic scraper silently disables article extraction on every clip |
| P1 | D1 | `S15-S16-models-persona` | `[model_catalog] use_models_dev` is documented to users as a working feature but nothing in production ever fetches the catalog, so the lookup layer is permanently empty |
| P1 | D1 | `S17-small-pkgs-css` | An expired/invalid server token leaves the notification observer in a permanent silent retry loop; no path in `Notifications/` classifies a 401 or offers re-auth |
| P1 | D1 | `S17-small-pkgs-css` | `Metrics/metrics_logger.timeit` times coroutine *creation*, not execution: 8 async call sites record ~0 s and `status="success"` for every failure |
| P1 | D2 | `S17-small-pkgs-css` | The server-notification observer executes 15.2 SQLite statements and 2 `BEGIN IMMEDIATE` write transactions **per event** synchronously on the Textual event loop; half of those write transactions are empty |
| P1 | D1 | `S18-screens-a` | `evals_screen.py` still uses `rich.markup.escape`, which does not escape `[TODO]`-shaped brackets, so a bench named `[TODO] Q3 plan` loses that token from the Run button, its tooltip and the delete-confirmation dialog |
| P1 | D1 | `S18-screens-a` | `change_review_screen.py` puts raw git stderr into `notify()` with markup ON: a repo path like `src/[id]/page.tsx` loses that segment, and a message containing `[/` crashes the app in the toast renderer |
| P1 | D1 | `S18-screens-a` | Three watchlists write workers have no error handling at all and are dispatched with `exit_on_error` at its app-exiting default, in a file where every sibling write worker guards and toasts |
| P1 | D1 | `S19-screens-b` | `StatsScreen` crashes when a chat topic contains a non-ASCII word |
| P1 | D1 | `S19-screens-b` | The Settings/Console endpoint probe's chat path reaches the wire with the user's `base_url` **and** the resolved API key, bypassing `Utils/egress.py`; the TTS path in the same function does not |
| P1 | D1 | `S19-screens-b` | `SchedulesWorkbench._run_sync`'s `finally` dereferences the DOM after an awaited network sync, in a worker with `exit_on_error=True` |
| P1 | D1 | `S20-ui-root` | `LogsWindow.append_record` mutates the Textual DOM from whatever thread called `logger.*`, with no marshalling |
| P1 | D1 | `S20-ui-root` | Voice Cloning ▸ Delete profile raises `AttributeError` before the confirm dialog can render |
| P1 | D1 | `S21-ui-modules-a` | The first-run wizard's "Test" button sends the user's API key to a host the config file can supply, with no `Utils/egress.py` check |
| P1 | D1 | `S21-ui-modules-a` | Two first-run workers exit the app on an unguarded post-await `query_one`, because they run at `exit_on_error`'s app-exiting default |
| P1 | D1 | `S21-ui-modules-a` | Voice-blend import/export reads and writes a user-chosen path with no `path_validation`, no size cap, no shape validation, and a non-atomic write — while the sibling export in the same package does all three |
| P1 | D1 | `S22-S23-ui-modules-b-widgets` | Every flashcard row in Study renders with its queue-state badge deleted; the code writes it, Textual eats it |
| P1 | D1 | `S22-S23-ui-modules-b-widgets` | User-typed and wire-sourced text reaches markup-parsing sinks on 11 files across both slices; `[/…]` raises `MarkupError` inside `_compositor.reflow`, which nothing catches |
| P1 | D4 | `S22-S23-ui-modules-b-widgets` | `Widgets/Home/home_rail.py` has a non-escaping same-named twin of the Library helper that *does* escape; TASK-32802.1's finding text says the opposite |
| P1 | D3 | `S22-S23-ui-modules-b-widgets` | `Widgets/NewIngest/` (5 modules, 1,240 lines) has zero production importers and is unreachable from `app.py` or the screen registry |
| P1 | D2 | `S25-backup-recovery` | The restore publication loop re-`fsync`s the entire journal history once per move record, making publication O(N²) in `F_FULLFSYNC` calls |

---

## Coverage

**Every slice reached a terminal state.** `sampled` is a legitimate terminal state; absent is not.
Per-slice reports — findings, evidence, candidate triage, D4 notes, UNVERIFIED — are in `slices/`.

| Slice | Area | Files | Lines | State | Read in full | Sampled | Mechanical only | Findings |
|---|---|---:|---:|---|---:|---:|---:|---|
| S01 | Notes A - sync engine, conflict resolution, file services | 20 | 40126 | done | 19 | 1 | 0 | 12 |
| S02 | Notes B - templates, importers, remainder | 21 | 26458 | done | 6 | 12 | 3 | 13 |
| S03 | TTS A - backends | 29 | 26521 | done | 3 | 17 | 9 | 14 (S03+S04 joint) |
| S04 | TTS B - remainder | 62 | 39395 | done | 5 | 18 | 39 | 14 (S03+S04 joint) |
| S05 | Library | 57 | 45150 | done | 16 | 16 | 25 | 14 |
| S06 | API client | 61 | 38737 | done | 14 | 21 | 26 | 14 |
| S07 | Speech in | 55 | 39378 | done | 3 | 26 | 26 | 11 |
| S08 | Scheduling | 87 | 41236 | done | 13 | 25 | 49 | 8 |
| S09 | Characters | 66 | 35065 | done | 5 | 22 | 39 | 10 |
| S10 | Chunking | 70 | 26734 | done | 1 | 11 | 58 | 13 (S10+S26 joint) |
| S11 | Evals+ingest | 74 | 44016 | done | 9 | 18 | 47 | 17 |
| S12 | Media | 52 | 26474 | done | 11 | 17 | 24 | 13 |
| S13 | Canvas | 53 | 35510 | done | 6 | 28 | 19 | 11 |
| S14 | Web | 50 | 27083 | done | 9 | 16 | 25 | 12 |
| S15 | Models | 39 | 20758 | done | 3 | 10 | 26 | 14 (S15+S16 joint) |
| S16 | Persona | 50 | 23660 | done | 0 | 21 | 29 | 14 (S15+S16 joint) |
| S17 | Small pkgs + css python | 80 | 38316 | done | 8 | 24 | 48 | 15 |
| S18 | Screens A | 11 | 39343 | done | 1 | 8 | 2 | 11 |
| S19 | Screens B | 68 | 40757 | done | 35 | 8 | 25 | 15 |
| S20 | UI root (UI/*.py top level) | 33 | 31100 | done | 9 | 13 | 11 | 8 |
| S21 | UI modules A | 55 | 45614 | done | 5 | 23 | 27 | 12 |
| S22 | UI modules B | 77 | 29039 | done | 2 | 22 | 53 | 13 (S22+S23 joint) |
| S23 | Widgets rest | 39 | 10709 | done | 4 | 6 | 29 | 13 (S22+S23 joint) |
| S24 | Interop cluster | 178 | 68546 | done | 2 | 9 | 167 | 2 |
| S25 | Backup_Recovery (added by this run) | 80 | 42955 | done | 10 | 28 | 42 | 9 |
| S26 | Workflows + strays (added by this run) | 22 | 7676 | done | 18 | 3 | 1 | 13 (S10+S26 joint) |

**Totals: 26 slices · 890,356 lines · 217 files read in full · 423 sampled · 849 mechanical only.**

Honest reading of those numbers: **mechanical-only is the majority (849 of 1,489 files)**, and every slice states
in its own report what that means for its conclusions. Where a slice reported few findings over many files, it says
what it read — that is a coverage statement, not a clean bill. The largest single mechanical-only block is
`Chunking/engine/**` (38 files / 14,398 lines), which is **vendored and not this repo's to change**.

### Completeness check

```
$ .venv/bin/python qa/tier2-code-review-2026-09-21/check_coverage.py
UNCLAIMED: 0
```

`slice_paths.txt` as delivered printed `UNCLAIMED: 102`. It was amended in Phase 0 with `Backup_Recovery/`,
`Workflows/`, `UI/Workflows_Modules/` and four stray `Widgets/*.py`, and a comment recording why. The check above
is against the amended file; every `.py` under `tldw_chatbook/` minus the always-excluded paths is now claimed by
exactly one slice. The script is checked in next to this report — it is the prompt's logic with v1 PROMPT.md
§52's exclusion set (`.venv`, `Third_Party`, `Tests`, `__pycache__`) applied, and it is runnable as printed
from the worktree root.

---

## Legacy reachability

Deletion candidates found across the 26 slices. Importer counts are **production only** (AST-based, relative levels
resolved, function-body imports included). Reachability was resolved by **running** `resolve_screen_route()` where a
route was involved, not by reading the registry. **Nothing here is filed.**

| Symbol / module | Lines | Marker | Prod importers | Reachable from entry? | Pinning tests | Row |
|---|---:|---|---:|---|---|---|
| `Evals/` legacy run stack (`eval_runner`, `specialized_runners`, `dataset_validator`, `dataset_loader`, `base_runner`, `ui_integration`, `_run_admitted_evaluation`) | ~9,000 | — | — | **No** — `handle_start_evaluation` does not exist; `ABTestOrchestrator` is constructed nowhere | 3 test files keep it green | **ruling needed** — carries a `subprocess` code-exec sandbox; ADR `031-…` |
| `Widgets/Tamagotchi/` widget half | 2,181 | — | 0 | **No** — no screen, no route, no `compose()` | CSS + timer-inventory rows | **product decision** — the storage half is wired into backup/recovery |
| `MediaWindow_v2.py` + `media_screen.py` | 2,725 | — | 2 (both inside the cluster) / 0 | **No** — all 52 route targets were run through `resolve_screen_route()`; none resolves to `MediaScreen` | 12 files incl. a 1,510-line parity test | **delete (L)** — done. NB `media_screen.py` also appears in the *keep* row below; that row is wrong, see its note |
| `Web_Scraping/Confluence/` + dead `Article_Extractor_Lib` functions | ~3,150 | — | 0 | No | the XXE register | **delete** — removes 2 register entries |
| `Widgets/NewIngest/` | 1,240 | `"Legacy … compatibility exports"` | 0 | **No** | 6 test files **are** the keep-alive | **delete** |
| `CodeRepoCopyPasteWindow.py` (+ `repo_tree_widgets.py` 726) | 1,946 | — | 0 | No | 399-line test-only lifeline | **delete** |
| `Local_Ingestion/API_Endpoint_Sample.py` | 906 | — | 0 | No | none | **delete** |
| `Audio/dictation_service.py` | 652 | — | 0 | No — via a 32807.1 module | 1 (patches the widget) | **delete with 32807.1** |
| `Models/evaluation_state.py` | 616 | — | **0 anywhere incl. tests** | No | none | **delete** (+ its diagnostic-inventory row) |
| `SiteConfigSettings.py` | 599 | — | 0 | No | 225-line test-only lifeline | **delete** |
| `Chatbooks_Window.py` | 483 | — | 0 | No | none | **delete** |
| `UI/Screens/schedules_screen.py` | 516 | `DEPRECATED` | 0 | **No** — route → `SchedulesWorkbench` | none import the class | **delete** — its stated reason for existing is false |
| `TTS/utils/` | 512 | — | 0 | No | 2 architecture rows | **delete** — and its downloader can never succeed |
| `Prompt_Management/Prompt_Engineering.py` | 590 | — | 0 | **Cannot import** (`tldw_Server_API` absent) | none | **delete — but extract its metaprompt for task-474 first** |
| `Audio/dictation_metrics.py` | 380 | — | 0 | No | none | **delete** — orphaned *by* TASK-32810 |
| `Local_Inference/mlx_lm_inference_local.py` | ~250 | — | 0 | No | **30 tests asserting the dead code's behaviour** | **delete** — a fixer will patch it by mistake |
| `UI/CCP_Modules/ccp_message_manager.py` | 317 | — | 0 (a re-export) | No | 1 | **delete** |
| `Audio/console_dictation.py` | 252 | — | 0 | No | 1 | **delete** — the module that replaced it says so in a comment |
| `Widgets/Note_Widgets/note_creation_modal.py` | 265 | — | 1 (itself deferred-dead) | No | — | **delete** — transitively dead, named nowhere |
| `UI/Widgets/config_search_widget.py` | 228 | — | 1 (`Tools_Settings_Window`) | No | 2 CSS assertions | **delete with 32807.4** — else .4 orphans it |
| `UI/Workbench/route_inventory.py` | 118 | — | 0 (lazy export only) | No | 1 | **delete** |
| ~~6 `*_Interop` modules~~ | 753 | — | n/a | **YES — reachable** | `Tests/Sync_Interop/` | **RETRACTED 2026-09-22 — do NOT delete.** This row contradicted this report's own evidence in two places: §Retired/contested says *"All 31 are reachable from the composition root; residue is 753 lines (1.1%)"*, and `slices/S24-interop-cluster.md:106-108` says the naive walk reports 7 dead packages but *"the lazy-aware walk reports **0 dead packages**. The naive answer is the trap here."* The 753 lines are **within-package residue, not deletable modules**. |
| ~~`Media_Creation/swarmui_client.py` + `image_generation_service.py`~~ | 868 | — | **2 live / 1 live** | **YES** | 1 | **RETRACTED 2026-09-22 — do NOT delete.** Traced live: `chat_screen.py` → `UI/Console_Modules/image.py` → `console_generate_image.py:436` imports `ImageGenerationService`, which imports `SwarmUIClient` at module scope (`image_generation_service.py:11`). Also re-exported by `Media_Creation/__init__.py:5`. The "0 live" figure was wrong. |
| `Coding/code_mapper.py` | 358 | — | 0 | **Cannot import** (`diskcache` undeclared) | none | **delete** |
| `Chatbooks_Window_Improved.py`, `ChatbookExportManagementWindow.py`, `ChatbookTemplatesWindow.py` | — | — | ≥1 live | **Yes** | — | **keep** |
| `skills_screen.py`, ~~`media_screen.py`~~, `tools_settings_screen.py` | — | mixed | ≥1 | No | — | **keep** — the registry documents each reason; `tools_settings` is TASK-32807.4's. **`media_screen.py` should not be in this row**: it appears in the delete row above, and its stated keep reason (the registry's save_state/restore_state unit tests) is a test-only lifeline whose four tests were already red at baseline. Deleted. |
| `library_conversations_screen.py`, `image_gen_demo_screen.py` | 167 | `"THROWAWAY"` | 1 each | partial | 1 | **unknown** — needs an owner's call |

---

## Left UNVERIFIED

Consolidated from all 26 slices. Full per-slice tables are in `slices/`.

**The environmental constraint that produced most of these:** the ADR-126 recovery gate makes `Tests/UI/*` and
several other suites error at fixture setup in a clean worktree (`RecoveryRequired` from
`Backup_Recovery/raw_participants.py` / `app.py:1116 APP_CONFIG = load_settings()`). Combined with the
do-not-run-the-app rule, that is why the claims below are traced rather than executed. Three slices hit it
independently.

| Claim | Why not verified | Check to run |
|---|---|---|
| The **P0** Library-tool loss produces no tool calls in a live Console run | app must not be run; the static chain is fully verified | launch Console on default config, send a message that should call a Library tool, then `rg 'library_provider_factory failed' ~/.local/share/tldw_cli/logs/*.log` |
| The **P0** double-fire is observable as two user-visible notifications | needs a real tldw_server; the client side is proven | create a reminder server-side, toggle Schedules to "This device", press `s`, wait past `next_run_at`, compare the inbox against the server's run ledger |
| The **P0** `killpg` actually fires (a player surviving SIGKILL >0.5 s) | cannot induce uninterruptible sleep read-only; **the pgid identity is proven** | monkeypatch `process.wait` to raise `TimeoutExpired` twice, patch `os.killpg`, assert the recorded pgid `!= os.getpgrp()` |
| Whether a **non-restore** path can set `api_settings.<p>.base_url` — **this is what decides P1 vs P0 on the egress finding** | chatbook import was checked and cleared; sync/profile-import paths were not | `rg -n "base_url" tldw_chatbook/Chatbooks/ tldw_chatbook/Backup_Recovery/ tldw_chatbook/Sync_Interop/` |
| The O(N²) restore fsync at a real replace-mode `len(prepared.artifacts)` | a *fresh* restore collapses to ~1 publication unit; a *replace over an existing profile* does not, and only the latter was projected | `pytest Tests/Backup_Recovery/test_f9_replacement_workflow.py -q -p countflush -s`, read `maxjournal`, N = maxjournal/6 |
| That ffmpeg follows the redirect the yt-dlp branch hands it | needs a live redirecting host | `ffmpeg -loglevel debug -i <302 to a private IP> -f null -` and read the debug log for the second request |
| The VAD audio drop is audible / degrades WER | no microphone; the byte accounting **is** measured | set `buffer_duration_ms = 150`, dictate a fixed sentence 10× with and without the `_gate_carry` patch, diff WER |
| Every "reproduced in a live app" UI crash (Stats non-ASCII topic, Voice-Cloning delete, the two first-run workers, the `finally` DOM deref, the Study `[/]` deck name) | the ADR-126 gate above | each slice names a specific Pilot/`app.run_test()` test that would be **born red** |
| Whether the 849 mechanical-only files hold further D1s | budget; each was swept for the pattern set, **which is a negative scan, not a read** | the per-slice tables name the specific files worth reading first, in priority order |
