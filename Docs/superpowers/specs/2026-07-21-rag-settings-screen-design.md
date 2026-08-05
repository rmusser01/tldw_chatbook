# SP3 — RAG Settings Screen — Design

**Date:** 2026-07-21 · **Status:** Draft (owner review pending)
**Part of:** `2026-07-21-rag-settings-profiles-overview.md` (sub-project 3 of 3)
**Goal:** a dedicated RAG settings screen with a profile manager and a full-`ProfileConfig` editor, absorbing and replacing the narrow "Library RAG" category. Comparatively mechanical — the correctness work lives in SP1/SP2 — but with three real seams.

## 1. Category

Repurpose `SettingsCategoryId.LIBRARY_RAG` into a full "RAG" category (keep the enum value `"library-rag"` to avoid churn, relabel the display + the contract-registry blurbs at `settings_screen.py:683`). The old `SettingsLibraryRagDefaults` dataclass and its `build/load/validate` trio are replaced. The contract-registry "Affected config / Recovery / Boundary" text is rewritten (it currently says "AppRAGSearchConfig.rag.search and …retriever defaults" and "Library owns indexing" — both change).

## 2. Save target changes (main seam)

The narrow `build_library_rag_save_sections` writes `AppRAGSearchConfig.rag.search/.retriever` — the deprecated value keys SP2 removes. SP3 must not write those. The settings framework **already dispatches save per category** (`_save_library_rag_sections`/`_save_appearance_sections`/… at `:9853–9876`, and the build dispatch at `:9460–9509`), so the RAG category gets a **custom save** that:

1. Writes the **active profile file** via SP2's manager (the profile is the persisted unit).
2. Writes only the single pointer key `[rag.service].profile` through the existing `SettingsConfigAdapter` (a one-key section write it already supports).
3. Triggers SP2's project + service reset.

So the screen's "Save" is a profile write, not a config-section write.

## 3. Two regions

**Region 1 — Profile manager.** A list grouped **Builtins (read-only, ~13)** vs **Your profiles**, active one marked. Actions:
- *Set active* → runs SP2's switch (project + service reset + off-thread reload) with progress feedback ("switching profile / rebuilding service…"); surfaces the SP1 consequence when the fingerprint changes: "this profile uses a different embedding/chunking → new index, currently empty · Backfill." Set-active is **separate from the form's Save** and is async — it must not block the settings screen.
- *Clone* → new user-profile file seeded from the selected profile (the only way to "edit" a builtin).
- *Rename* / *Delete* (user profiles only; delete offers to drop the profile's fingerprinted index via SP1's `delete_index`; deleting the active profile falls back to a builtin).
- *New from builtin*.

**Region 2 — Config editor.** The full `ProfileConfig` in collapsible groups (search · embeddings · chunking · vector store · retriever · reranking · indexing), editing the **active** profile.
- If the active profile is a builtin, the form is **read-only** with a prominent "Clone to edit" affordance.
- **Index-determining fields** (embedding model + every chunk-output-affecting field + `distance_metric`, per SP1's fingerprint) carry a visual marker; changing one shows the "re-points to a new (empty) index; backfill needed" warning — the SP1↔SP3 seam.
- **Two triggers, one consequence.** An index re-point (new fingerprint → empty collection) happens on *both* set-active to a differently-fingerprinted profile (Region 1) *and* editing+saving an index-determining field on the active profile (here). Both fire the same warning + Backfill affordance and both go through SP2's service reset — the implementation must not handle only set-active.
- **Reranking group** presents `enable_reranking` together with `reranking_config`, even though they live in different parts of the object tree (`enable_reranking` is `rag_config.search.enable_reranking` at `config.py:267`; `reranking_config` is a top-level `ProfileConfig` field). SP3 must set the field that `create_rag_service` actually threads to `self.enable_reranking` (also a service kwarg with a fallback, `enhanced_rag_service_v2.py:50/152`) — verified at plan time, or the toggle is inert.

## 4. Draft model (the trickiest detail)

The settings framework keys drafts by *category* (`_settings_drafts[category]`), assuming one stable object per category. SP3's editor's object is the *active profile*, which changes via set-active within the same category. Unsaved edits to profile A + set-active to B would otherwise leak A's edits onto B or lose them silently. SP3:
- keys the editor draft by **`(category, profile-id)`**, and
- on set-active with a dirty editor, **prompts save/discard** and clears the draft on switch.

Fallback (plan-time verification #4): if the framework's category-keyed draft map cannot take a composite key without a deeper refactor, keep per-profile editor drafts in the screen's own state (outside the framework map) and hand the framework only the active profile's draft — the switch-prompt behavior is unchanged.

## 5. Validation

- Actually call `RAGConfig.validate()` — memory flags it as a **zero-caller** method today, so wiring it here closes a latent gap rather than inventing new validation.
- `validate()` covers only `rag_config`; SP3 additionally validates the `reranking_config`/`processing_config` sections (ranges, enums, non-empty model name) so the whole `ProfileConfig` is checked.
- Reuse the existing `validate_*`/draft/dirty discipline for field-level feedback.

## 6. Layout / rendering

The form is materially larger than any existing settings sub-screen (`RAGConfig` ≈ 5 sub-configs × ~5–10 fields + reranking/processing ≈ 40 fields vs. the ~10 typical). To avoid the known traps:
- Collapsible groups inside a **scroll container** (the "plain Vertical clips mounted rows" geometry lesson).
- **Value-aware dirty-marking** and building the form in grouped chunks rather than one giant `recompose` (the recompose-remount Changed-echo race lesson).

## 7. Index status readout (light; full management deferred)

For the active profile only, show its fingerprinted collection state — built / empty / needs-backfill — with the provenance SP1 stamps ("built with model X / chunk 400·100 / N vectors") and a **Backfill** action that communicates re-embed cost. Full multi-index management (list/delete every on-disk collection with sizes) reuses SP1's `list_indexes`/`delete_index` but is a **follow-up**, keeping SP3 focused.

## 8. Testing

- Category renders; save writes the profile file + pointer (not `AppRAGSearchConfig.rag.*`); custom save dispatch wired.
- Profile manager: set-active runs the switch and surfaces index consequence; clone/rename/delete; builtin actions refused; delete-active fallback.
- Editor: builtin → read-only + clone affordance; index-field change → warning; `enable_reranking` toggle actually reaches the service (integration, not mocked accessor — the config-caps lesson); full-`ProfileConfig` validation incl. rerank/processing.
- Draft: `(category, profile-id)` keying; dirty + set-active → prompt; draft cleared on switch.
- Layout: large form scrolls without clipping; grouped mount, no Changed-echo dirty race.
- **QA:** textual-serve + playwright captures of the two-region layout, set-active flow, builtin read-only state, index-empty warning — for owner screen approval before merge (established gate); geometry checked at the standard capture resolution.

## 9. Plan-time verifications

1. That toggling `rag_config.search.enable_reranking` in the active profile actually changes `self.enable_reranking` on the rebuilt service (not overridden by the kwarg fallback).
2. That the per-category custom-save hook can persist outside `SettingsConfigAdapter.save_sections` cleanly (write the profile file, then a one-key pointer section).
3. The full field inventory of `ProfileConfig`/`RAGConfig` to lay out the form groups, and which fields are index-determining (shared with SP1's fingerprint list).
4. That the draft framework tolerates a composite `(category, profile-id)` draft key without deeper refactor.
