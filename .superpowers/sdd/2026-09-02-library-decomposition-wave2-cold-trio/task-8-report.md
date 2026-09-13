# Task 8 (search series, entanglement gate) — BLOCKED at the gate

**Verdict: BLOCKED.** The search cluster's cross-call entanglement with the RAG
cluster is well over the wave plan's 1/3 threshold under every reasonable
reading of the count. No move (state object, controller, wiring test, or any
other recipe step) was performed; `library_screen.py` and every other tracked
file are untouched. This report is the census the wave plan's Tasks 8–9
section requires before any BLOCKED call: "report BLOCKED with the census — do
nothing else."

## 1. Cluster enumeration (mechanical)

`ast`-walked `LibraryScreen` for method names containing `"search"`
(case-insensitive), 2026-09-02 snapshot (`library_screen.py`, 42411 lines /
1267 methods, matching the current `_BUDGETS` pin exactly):

**23 raw matches** — matches the wave plan's Tasks 8–9 "~23" estimate exactly.

```
_apply_library_rag_search_outcome
_execute_library_rag_search
_flush_library_prompts_search
_focus_library_media_content_search_input
_focus_library_search_input
_library_rail_search_placeholder
_load_library_search_history
_patch_sibling_library_search_input
_persist_library_search_history
_queue_library_prompts_search
_record_library_search_history
_refresh_search_rag_panel_state_widgets
_save_library_search_history
_stop_library_prompts_search_debounce
clear_library_search_history
handle_library_media_content_search_next
handle_library_media_content_search_prev
handle_library_media_content_search_submitted
handle_library_media_trash_search_changed
handle_library_media_trash_search_submitted
handle_library_search_changed
handle_library_search_submitted
rerun_library_search_from_history
```

### Exclusions verified by reading each body ("check each")

**3 Prompts-owned** (the Prompt-list filter debounce; each operates on
`_library_prompt_browse_controller` / `_library_prompts_*` fields, unrelated
to the top search bar):
`_stop_library_prompts_search_debounce`, `_queue_library_prompts_search`,
`_flush_library_prompts_search`.

**6 Media-owned** (in-viewer content search and Media-Trash browse search;
each operates on `_library_media_content_*` / `_library_media_trash_*` fields
and controllers, unrelated to the top search bar):
`_focus_library_media_content_search_input`,
`handle_library_media_content_search_next`,
`handle_library_media_content_search_prev`,
`handle_library_media_content_search_submitted`,
`handle_library_media_trash_search_changed`,
`handle_library_media_trash_search_submitted`.

**14 remaining — the actual Search-cluster candidate set** (the top
Library-wide search bar, its history mechanism, and the RAG-search-submit
path it drives):

```
_apply_library_rag_search_outcome
_execute_library_rag_search
_focus_library_search_input
_library_rail_search_placeholder
_load_library_search_history
_patch_sibling_library_search_input
_persist_library_search_history
_record_library_search_history
_refresh_search_rag_panel_state_widgets
_save_library_search_history
clear_library_search_history
handle_library_search_changed
handle_library_search_submitted
rerun_library_search_from_history
```

## 2. The RAG-named reference set

Same mechanical `ast` walk for method names containing `"rag"`
(case-insensitive): **39 methods**, all following the established
`_library_rag_*` / `*_library_rag_*` naming convention this recipe's own §2
script already tags with the `OTHER_SUBSYSTEM` prefix `"_library_rag"`:

```
_apply_library_rag_answer, _apply_library_rag_scope_recovery_block,
_apply_library_rag_search_outcome, _execute_library_rag_answer,
_execute_library_rag_search, _focused_library_rag_result_card_index,
_library_rag_answer_chat_kwargs, _library_rag_panel_state,
_library_rag_scope_summary, _mirror_library_rag_scope_recovery,
_open_library_rag_result_by_index, _refresh_library_rag_answer_widgets,
_refresh_library_rag_history_widget, _refresh_library_rag_query_status_widgets,
_refresh_library_rag_results_widgets, _refresh_search_rag_panel_state_widgets,
_reset_library_rag_answer_state, _reset_library_rag_in_flight_status,
_reset_library_rag_retrieval_state, _reveal_library_rag_results,
_select_library_rag_result_by_index, _stage_library_rag_result_in_console,
_start_library_rag_answer, _start_library_rag_query,
_sync_library_rag_scope_toggle_and_run_gate_widgets,
_use_library_rag_result_in_console, action_library_rag_result_card_open,
action_library_rag_result_card_select, action_library_rag_use_in_console,
cycle_library_rag_mode, open_import_export_from_library_rag,
open_library_rag_result, run_library_rag_query, select_library_rag_result,
submit_library_rag_query, sync_library_rag_history_collapsed,
toggle_library_rag_scope_source, update_library_rag_query,
use_selected_library_rag_result_in_console
```

Two of the 14 search-cluster candidates (`_apply_library_rag_search_outcome`,
`_execute_library_rag_search`) and one more by a near-identical pattern
(`_refresh_search_rag_panel_state_widgets`, "search_rag" rather than
"rag_search" but the same panel-state family as
`_refresh_library_rag_query_status_widgets` etc.) are themselves members of
this RAG-named set — i.e. the mechanical "search"-substring census and the
"rag"-substring census already overlap by construction, before any call-graph
analysis is done.

## 3. Cross-call census (AST, both directions)

For each of the 14 candidates: `self.<name>(...)` calls found inside its own
body (**callees**), and every other `LibraryScreen` method whose body calls
`self.<candidate>(...)` (**callers**) — both filtered to names in the 39-strong
RAG-named set above. `self_rag_named` flags a candidate whose own name is
already in that set.

| # | Search-cluster method | self_rag_named | RAG callees | RAG callers | Entangled? |
|---|---|---|---|---|---|
| 1 | `_apply_library_rag_search_outcome` | **yes** | `_library_rag_panel_state`, `_refresh_search_rag_panel_state_widgets`, `_start_library_rag_answer` | `_execute_library_rag_search` | **YES** |
| 2 | `_execute_library_rag_search` | **yes** | `_apply_library_rag_search_outcome` | `_start_library_rag_query` | **YES** |
| 3 | `_focus_library_search_input` | no | NONE | NONE | no |
| 4 | `_library_rail_search_placeholder` | no | NONE | NONE | no |
| 5 | `_load_library_search_history` | no | NONE | NONE | no |
| 6 | `_patch_sibling_library_search_input` | no | NONE | `update_library_rag_query` | **YES** |
| 7 | `_persist_library_search_history` | no | NONE | NONE | no |
| 8 | `_record_library_search_history` | no | NONE | `_start_library_rag_query` | **YES** |
| 9 | `_refresh_search_rag_panel_state_widgets` | **yes** | `_apply_library_rag_scope_recovery_block`, `_library_rag_panel_state`, `_library_rag_scope_summary`, `_refresh_library_rag_answer_widgets`, `_refresh_library_rag_history_widget`, `_refresh_library_rag_query_status_widgets`, `_refresh_library_rag_results_widgets` | `_apply_library_rag_answer`, `_apply_library_rag_search_outcome`, `_select_library_rag_result_by_index`, `_start_library_rag_query`, `update_library_rag_query` | **YES** |
| 10 | `_save_library_search_history` | no | NONE | NONE | no |
| 11 | `clear_library_search_history` | no | `_library_rag_panel_state`, `_refresh_library_rag_history_widget` | NONE | **YES** |
| 12 | `handle_library_search_changed` | no | NONE | NONE | no |
| 13 | `handle_library_search_submitted` | no | `_start_library_rag_query` | NONE | **YES** |
| 14 | `rerun_library_search_from_history` | no | `_start_library_rag_query` | NONE | **YES** |

**Ratio (full candidate set): 8 / 14 = 0.571 (57.1%).**

### Sensitivity check — most conservative possible reading

Even if the 3 methods whose own name is already RAG-tagged
(`_apply_library_rag_search_outcome`, `_execute_library_rag_search`,
`_refresh_search_rag_panel_state_widgets`) are stripped entirely out of the
Search cluster first (treated as outright RAG-owned, not merely
"entangled"), the remaining 11 candidates still show:

| Method | RAG callees | RAG callers | Entangled? |
|---|---|---|---|
| `_focus_library_search_input` | NONE | NONE | no |
| `_library_rail_search_placeholder` | NONE | NONE | no |
| `_load_library_search_history` | NONE | NONE | no |
| `_patch_sibling_library_search_input` | NONE | `update_library_rag_query` | **YES** |
| `_persist_library_search_history` | NONE | NONE | no |
| `_record_library_search_history` | NONE | `_start_library_rag_query` | **YES** |
| `_save_library_search_history` | NONE | NONE | no |
| `clear_library_search_history` | `_library_rag_panel_state`, `_refresh_library_rag_history_widget` | NONE | **YES** |
| `handle_library_search_changed` | NONE | NONE | no |
| `handle_library_search_submitted` | `_start_library_rag_query` | NONE | **YES** |
| `rerun_library_search_from_history` | `_start_library_rag_query` | NONE | **YES** |

**Ratio (conservative reading): 5 / 11 = 0.4545 (45.5%).**

Both readings clear the >1/3 (33.3%) threshold by a wide margin — the
conservative floor (45.5%) is itself 12 points over the gate.

## 4. Field-level corroboration (not the gate metric, but consistent with it)

Recipe §2 script (`_library_search` field prefix), `__init__`-scoped fields:

- `_library_search_history` — 4 users total; one is RAG-named
  (`_library_rag_panel_state`, which reads the history to render the RAG
  panel's own history widget), the other 3 are the search-cluster's own
  (`_record_library_search_history`, `clear_library_search_history`,
  `rerun_library_search_from_history`). Not exclusively Search-owned by the
  strict ≥2-subsystems rule (recipe §2) — the same field the wave plan
  flagged as "possibly exclusive" is directly read by a RAG method.
- `_library_rag_searched_query` — 7 users; 4 are RAG-named
  (`_library_rag_panel_state`, `_reset_library_rag_retrieval_state`,
  `_start_library_rag_query`) plus the search-cluster's own
  `_apply_library_rag_search_outcome`, confirming the wave plan's "likely
  RAG-owned" prediction.

Neither of the two candidate fields is cleanly exclusive to a
Search-only cluster; both cross into RAG's own field usage. This mirrors the
method-level finding above rather than contradicting it.

## 5. Why the entanglement is structural, not coincidental

Reading the bodies (not just the call graph) confirms the shape: the top
Library search bar's submit handler (`handle_library_search_submitted`) and
its "rerun from history" action (`rerun_library_search_from_history`) both
call `_start_library_rag_query` directly — i.e. the search bar's own submit
path *is* the RAG query entry point, not a sibling of it. `_execute_library_
rag_search` and `_apply_library_rag_search_outcome` form a two-step
call/callback pair that IS the search execution — there is no separate
"search executes, then RAG optionally reacts" boundary; RAG execution *is*
what "search" submits to. `_refresh_search_rag_panel_state_widgets` is the
single fan-out point that refreshes every RAG panel sub-widget after a
search-history or RAG-panel-state change, called by both search-history
mutators and RAG answer/query methods alike. This is architecturally
consistent with the collections series' own finding (recipe §13) that a
name-based "search" tag can mask a DIFFERENT feature entirely (Collections'
saved-searches turned out to be uncontested and unconnected) — but here the
opposite holds: the top search bar and RAG are the same feature wearing two
names, not two adjacent features that merely share the word "search".

## 6. Disposition

Per the wave plan's Tasks 8–9 section: **"If the analysis shows search is too
entangled with RAG to extract alone (>1/3 of its methods calling or being
called by rag-prefixed methods), STOP and report: the right answer may be a
combined search+rag wave-3 series, and that is a controller ruling, not an
implementer improvisation."**

Ratio measured: **57.1% (8/14) under the direct reading, 45.5% (5/11) under
the most conservative possible reading** — both far past the 33.3% gate.

**No move was attempted.** No state object, no controller, no wiring test, no
characterization spot-check, no `_BUDGETS` change. `git status` is clean;
`library_screen.py` is byte-identical to `HEAD` (`91feba4a7`/`1e466ffac` tip
at task start). This report is the deliverable; the controller decides
whether Task 9 becomes a combined search+RAG wave-3 series or a different
split.
