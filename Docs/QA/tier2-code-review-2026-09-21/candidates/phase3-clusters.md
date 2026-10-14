# Phase 3 working notes — repo-wide D4 (lead, not a subagent)

Measured at `origin/dev` `3722a857480b94b30fd4755f3f8e3002bd163ec3`, repo-wide
(`tldw_chatbook/` minus `.venv`, `Third_Party`, `__pycache__`, `Splash_Screens`).
Scripts: `candidates/variants.py`, `candidates/dup_by_name|verbatim|shape.tsv`,
`candidates/helper_adoption.tsv`.

## Cluster measurements (by BODY, not by name)

| Cluster | defs | distinct bodies | distinct shapes | LOC | Reading |
|---|---:|---:|---:|---:|---|
| `_maybe_await` | 65 | 4 | 3 | 253 | **True clone.** 59 byte-identical; the other 4 are a one-line ternary of the same thing. Zero behavioural drift. |
| `_enforce_policy` | 51 | 6 | 6 | 234 | **True clone.** 43 + 4 (inverted `if`) = 47 semantically identical; 4 add an action-id composition step. Security gate. |
| `_require_client` | 47 | — | **1** | 326 | **True clone.** One shape, error message only. |
| `_normalize_mode` | 46 | 46 | 5 | 440 | **Template.** 41 are the same enum-coercion body with the enum name substituted; 5 are genuinely different (2 string-set, 1 `or "local"` null handling, 2 unrelated namesakes). |
| `_identity` | 31 | — | — | 245 | template |
| `_dump` | 39 | 3 verbatim groups (11/4/3) | — | 278 | template |
| `_initialize_schema` | 16 | 16 | — | 2120 | **RETIRED — `@abstractmethod` on `DB/base_db.py:811`.** Largest LOC mass in the census and not duplication at all. |
| `_perform_safe_cancel` | 45 | — | **28** | — | **RETIRED — template-method override** against `request_safe_cancel`/`dismiss_safe_once`/`run_cancel_effect_once`. Consolidating would be a regression. |
| `_get_connection` (BaseDB subclasses) | 9 overrides | — | — | — | **RETIRED —** all call `super()._get_connection()`; PRAGMA divergence is documented per-store with task refs. |
| `_set_status` | 23 | — | 15 | — | **Helper exists, 74% adopted.** `Widgets/status_line.py::set_status_line` (17 importers). Owner: task-32861 (To Do). |
| `_coerce_bool` | 11 | **7** | 7 | — | **RETIRED as a cluster.** TASK-32808.4 (Done) deliberately scoped to the 3 verbatim wide-vocabulary copies and documented why the rest differ — and they do: rail_state returns `value != 0` for any int (so `5`→True) where `coerce_bool_flag` stringifies and falls to `default`. Two internal 3-copy verbatim groups remain (Character_Chat ×3, Chat/Home/Library rail_state ×3) — each collapsible locally without touching vocabulary. P3, owner task-32808.9. |
| `_coerce_int` | 4 | **1** | 1 | — | **NEW, see below.** |
| `_clean_text` | 7 | 5 | 5 | — | Low value: 2–3 line helpers with three different return contracts (`str` / `str \| None` / raises). Two 2-copy verbatim pairs only. P3, owner task-32808.9. |
| `_reject_json_constant` | 7 | 3 | 2 | — | Each is a `parse_constant=` hook raising its own local exception type — deliberately local. One helper with a raise-hook parameter is the shape. Owner: task-32855 (To Do). |

## Helper adoption, repo-wide (files that bypass the helper)

| Helper | Importers | Bypassers | Seed (2026-09-17) | Delta |
|---|---:|---:|---|---|
| `Utils/atomic_file_ops.py` | **17** | 28 files `os.replace` without it | 9 importers / 31 hand-rollers | **+8 adopted** (TASK-32808.5 Done) |
| `Utils/secure_temp_files.py` | 7 | **65** files raw `tempfile.*` | 7 / 46 | **grew by ~19 files** |
| `Utils/Utils.py::ensure_directory_exists` | **0** | 51 files raw `mkdir(parents=True, exist_ok=True)` (86 occurrences) | 0 / 73 files (106 occ.) | shrank, still 0 adoption |
| `Utils/Utils.py::truncate_content` | **0** | 24 files inline `[:n] + "..."` | 0 / 26 | flat. Owner: task-32808.3 (In Progress) |
| `Utils/optional_deps.py` | 40 | 61 files `try/except ImportError` without it | 43 / 102 | shrank |
| `Utils/Utils.py::format_size_bytes` | **2** | ~6 hand-rolled formatters | seed listed 10 + 318 raw `1024*1024` | **+2 adopted** (task-32808.1 In Progress); raw `1024*1024` now 324 occ. in 170 files |
| `Utils/input_validation.py::strict_json_loads` | **2** (`Chat/thinking_blocks`, `Chat/provider_continuation`) | 2 `_strict_json_loads` in `LLM_Calls/` + 7 `_reject_json_constant` | — | TASK-32805.5 (Done) closed the `Chat/` pair; `LLM_Calls/` and 5 Tier-2 copies remain. Owner: task-32855 |
| `Widgets/status_line.py::set_status_line` | 17 | ~6 `_set_status` hand-rollers | new helper (task-32861) | 74% adopted |
| `Widgets/form_components.py` | **2** | — | 2 | flat |
| `Utils/timestamps.py` | 28 | `Notes/` = 0 | — | TASK-32803.5 Done; guard is blind to the aware-non-canonical shape (see S01-P3) |
| `Utils/ui_helpers.py` | — | — | 0 importers (dead) | **DELETED** ✅ |
| `Utils/pagination.py` | — | — | 0 importers (dead) | **DELETED** ✅ |
| `Widgets/base_components.py` | — | — | seed helper | **DELETED** ✅ (`5f3adeca33`) |

## The three buckets (§2 of the prompt)

**FLAT** — unchanged since 2026-09-17, survived a complete review-file-fix cycle:
`_maybe_await` 65, `_enforce_policy` 51, `_require_client` 47, `_normalize_mode` 46,
`_identity` 31, `_dump` 39, `ensure_directory_exists` 0 importers, `truncate_content`
0 importers, `form_components` 2 importers. Every one of these has **no guard**.

**NEW** — exists only in code the 2026-09-17 run never saw, or landed after it:
- `_coerce_int` × 4 (`LLM_Calls/{mistral,openrouter,groq,deepseek}.py`), byte-identical,
  landed 2026-09-19 under TASK-32852 — a *consolidation* PR — two days after TASK-32808.4
  removed `_coerce_bool` from the same four modules and shipped `coerce_bool_flag` with
  **no integer counterpart**. See `phase4-verification.md` "Repo-wide D4-1".
- `LLM_Calls/mistral.py` shape group ×4 each: `_mistral_turn_response`, `_log_usage_metrics`,
  `_log_error_metrics`, `validate_finish` — the provider template cloned again.
- `Workflows/session.py` + `Workflows/authoring.py`: `subscribe`, `discard_setup`,
  `begin_close`, `abort_quit` — fresh duplication inside the package the scope table missed.
- `_reveal_focused_control` ×2 (`UI/MCP_Modules/mcp_audit_mode.py:539`,
  `Widgets/Library/library_search_rag_panel.py:92`).
- `Notes/notes_sync_conflicts.build_conflict_comparison` ↔
  `Notes/file_notes_conflict_compare.build_conflict_comparison` — same name, same four
  bound constants, same elision marker string, **two different truncation semantics**
  (truncate-mid-line vs drop-the-line). Missed by the shape hash. From S01.
- Three root-overlap implementations in `Notes/` with two semantics (lexical vs `samefile`),
  the lexical one guarding sync-root admission. From S01.

**CLOSED** — removed by a merged stream PR, verified by census diff, not by reading the PR:
- `_datetime_to_iso` × 5 files (`runtime_policy/source_state.py` et al.) — TASK-32803.
- `_coerce_bool` × 3 (`Image_Generation/config.py` et al.) — TASK-32808.4.
- The `LLM_Calls/moonshot.py` validator family: `_positive_integer`, `_positive_number`,
  `_nonnegative_number`, `_nonnegative_integer`, `_normalize_call_batch`, `_normalize_stop`,
  `_normalize_response_format`, `_json_shape_is_bounded` — 8 shape groups — stream #2738.
- `Utils/Utils.py::extract_text_recursive` / `extract_text_from_segments` pair.
- `Utils/ui_helpers.py`, `Utils/pagination.py`, `Widgets/base_components.py` — deleted.
- `transaction` × 3 (`DB/AgentRuns_DB.py` et al.) — partially; a 3-file group reappeared
  rooted at `DB/Library_Collections_DB.py:651`.

## Outward / downward LOC walk — the scope-service scaffold (S24, owner task-32808.6)

**Outward (what a consolidation deletes):** `_maybe_await` 253 + `_enforce_policy` 234 +
`_require_client` 326 + `_normalize_mode` 440 (the 41 template copies ≈ 390) +
`_identity` 245 + `_dump` 278 = **~1,700 lines** across 47 scope services and their
server-service siblings, in 31 `*_Interop` packages plus 16 in-tree packages.

**Downward (what the shared base pulls in):** `inspect` (stdlib), `typing`, and a
`TLDWAPIClient` reference that is `TYPE_CHECKING`-only in 47 of 47 copies today — so a
base class in e.g. `runtime_policy/scope_service_base.py` imports **no runtime module
outside stdlib**. It does not import a screen, a DB, or `tldw_api` at runtime. The
downward cost is ~60 lines. This is the cleanest ratio in the census: ~1,700 out, ~60 in.

**Why it has not happened:** it needs an ADR (a cross-package interface: 47 services
change base class), which is why `TASK-32808.6` has sat at To Do. Nothing has changed
since it was filed except that the copy counts are identical, which is the argument for
doing it, not against.

## The guard gap — the central Phase 3 finding

The repo already owns the guard technology and it demonstrably works:

```
$ ./scripts/preflight.sh | grep -A1 'timestamp writers'
=== timestamp writers ===
timestamp writers: 0 datetime.utcnow() site(s), 0 naive datetime.now().isoformat()
occurrence(s) (0 pinned).
```
`scripts/check_timestamp_writers.py` + `scripts/timestamp_writer_census.tsv` is a
shrink-only ratchet keyed on `module<TAB>symbol<TAB>kind<TAB>count`, wired into
`preflight.sh` and the required `Derived artifacts` CI job. TASK-32803 used it to take a
cluster the 2026-09-17 review measured at ≥7 drifted formats down to zero, and to *hold* it
there. Siblings: `check_index_plan_pins.py`, `check_schema_table_allowlist.py`,
`check_profile_owned_path_inventory.py`, `check_persistent_diagnostic_inventory.py`,
`check_textual_worker_contract.py` — six guards, all in the same file, all the same shape.

Every cluster in the FLAT bucket has **no** guard. Every cluster in the NEW bucket landed
into a codebase with no guard covering it. The `_coerce_int` case is the proof: a PR whose
stated purpose was consolidation shipped four fresh byte-identical copies, forty-eight hours
after the review that catalogued the pattern, and nothing objected.

**Caveat recorded against my own recommendation:** a guard is not free and not always right.
Under ~5 copies, "adopt it and move on" is the answer and a guard is the same
over-engineering this review is supposed to find. And a guard can be blind: S01-P3 shows
`check_timestamp_writers.py` reading green with an empty census while `Notes/` writes a
non-canonical aware shape ADR-173 names as one to eliminate — because the guard checks two
idioms, not the contract. A guard must assert the *contract*, not the two idioms that were
wrong last time.
