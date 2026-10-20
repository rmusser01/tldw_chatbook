---
id: TASK-35011
title: Choosing a discovered model as the default adds it to the provider's saved model list when Settings saves
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-14 17:10
updated_date: 2026-10-20 04:29
labels:
- settings
- providers
- model-discovery
- model-config
- ux-review-2026-10-14
dependencies:
- TASK-33007
- TASK-35010
- TASK-33641
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After PR #3057 (Phase 7), choosing a model that Discover models listed only stages it as the default for new chats. TASK-33007.3 AC#4 deliberately keeps it off the provider's saved list. So the saved list, Switch model and the next Settings visit do not reflect a choice the user believes they saved, which is the core of the 'discovery does not save' complaint.

The owner ruled on 2026-10-14 ('Pick = use as default'): a discovered model the user chooses becomes the default, as a draft that P7's quit guard protects, and joins the provider's saved list. Save selected becomes a secondary 'Add to model list' action for adding several at once. The ruling supersedes TASK-33007.3 AC#4 and needs an ADR-002 amendment. Because the draft can outlive the visit (leaving Settings keeps it, and the quit prompt's Keep editing returns to it), the choice has to survive leaving Settings too.

The same controls carry smaller verified problems:
- The button shares its save verb with s and has no 'applies immediately' label (ADR-033; A2).
- Tab reaches it before the list it acts on (B6).
- The Inspector has no guidance for Discover, the list, the button or Clear (B19).
- The capability line promises that saving resolves capabilities, which it never does (C-M2).
- An add of already-saved models does not say how to make one the default (C4 residual).

Plan: `backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md`. Findings closed: A2, B6, B19, C-M2, C4 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Choosing a model from the current discovery listing in Default model and pressing s saves it as the default for new chats and adds it to that provider's saved model list in the same save, and the receipt names the model added
- [ ] #2 Until s is pressed, choosing that model writes nothing to the config file, and Revert, Discard, or quitting with Discard leave the file byte-identical
- [ ] #3 Leaving Settings and coming back before s, including through the quit prompt's Keep editing, keeps the pending addition, and s still adds the model
- [ ] #4 Before s, the Saved model list title says that one model joins the list on save
- [ ] #5 A model staged any other way (one already in the saved list, a catalog model the server did not list, or a typed Custom ID) is saved only as the default and leaves the saved list unchanged
- [ ] #6 The addition is made to the saved list as the config file holds it at that moment, so a model added or removed outside Settings since the app started is respected
- [ ] #7 If the saved list changes on disk while s is saving, nothing is written, the draft is kept, and the message says to press s again
- [ ] #8 If the provider no longer has a single saved model list when s is pressed, s still saves the default and the receipt says why the model was not added
- [ ] #9 Save selected reads 'Add to model list' and is marked as applying immediately, and its receipt names the models added, or says they were already listed and that a model becomes the default by choosing it in Default model
- [ ] #10 When Add to model list fills an empty Default model, its receipt says that s still has to save that default
- [ ] #11 From Discover models, Tab reaches the discovered list before Add to model list and Clear
- [ ] #12 With Discover models, the discovered list, Add to model list or Clear focused, the Inspector names the control, says what it does and says whether it writes anything
- [ ] #13 The capability line no longer ties capabilities to saving, on the Settings discovery card and in Console's Chat settings readiness detail
- [ ] #14 ADR-002 (backlog/decisions/002-openai-compatible-model-discovery.md) records the 2026-10-14 'Pick = use as default' ruling and states that it supersedes TASK-33007.3 AC#4 and the '[providers] list did not change' clause of TASK-33007.3 AC#10
- [ ] #15 The Settings User Guide says that choosing a discovered model in Default model and pressing s saves it as the default and adds it to the provider's saved list, and describes Add to model list (applies immediately, adds several at once), the Discover models, list, Add to model list, Clear order, the 'joins on save' title and the new receipts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follows the plan's TASK-35011 mechanisms 1-9 (backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md), built on T1 (TASK-35010).

1. New `UI/Settings_Modules/saved_model_list.py`: the Saved model list block, `saved_models_summary` and `saved_models_not_listed` move in from the card (verbatim, then changed), with `_activate_saved_model_if_field_empty`; it holds the pending-addition helpers, the `s` preparation and its receipt suffix and conflict copy, Add to model list's receipts and retry steps, and the Inspector rows. The card imports it at load; settings_screen only inside functions.
2. A pick records a pending addition (provider key, model id) only when the id is in the current discovery listing and not saved; any other pick (catalog, Custom ID) clears it. It counts only while the Providers & Models draft stages exactly that provider and model, so Revert, Discard and a successful save end it with the draft. It is cleared at the provider, endpoint, credential, API key and deep-link provider changes and in Revert, never by the reset's other callers (resume, Discover, Clear). `save_state`/`restore_state` carry it (a new view-state key in `SETTINGS_SNAPSHOT_KEYS`, shape-checked).
3. `s`: while a pending addition matches, read the list under the config lock (`get_atomic_config_snapshot`), resolve the key (`resolve_provider_list_key`), and add `providers.<key>` = that list + the model (`append_models_to_provider_list`, which drops sample ids) to the same mutation, guarded by a pure `locked_snapshot_precondition` comparing the locked list with the one read. On success: refresh the lists in place, the picker and the discovery widgets; the toast (markup=False) names the model added. On a conflict: nothing written, draft kept, "press s again". A key that no longer resolves saves the default and says why the model was not added.
4. Validator: `_validate_combined_provider_settings_mutation` admits one `providers` key only when the model is set, the key's `provider_config_key` is the mutation's provider and the value is a list of safe strings holding the model.
5. Block: Discover models on its own row, then the list, then Add to model list (renamed, reworded tooltip, instant-apply hint) and Clear (B6); the title says "1 joins on save" while a pending addition matches (A2).
6. Receipts (C4 residual, AC#9-10): Add names the models added, or says they were already listed and how a model becomes the default; filling an empty Default model says s still saves it; retry steps name Add to model list.
7. Inspector (B19): content rows for Discover, the list, Add and Clear; the Save row is the instant-apply copy for Add and "saves nothing" for the other three.
8. Capability line (C-M2): Settings' duplicate constant goes; the shared `MODEL_CAPABILITY_UNKNOWN_WARNING` is reworded once.
9. ADR-002 amendment and settings.md. Size rows never rise: settings_screen.py nets <= 0, the card's row lowers, `saved_model_list.py` gets a birth row with no PR lane (NO-2 (b), PR #3036 touches derived-artifacts.yml).

Tests first in Tests/UI/test_settings_pick_joins_saved_list.py (the plan's cases 1-12, private profile, census tail), each shown red before its fix; rewrite only the pinned tests the plan names (plus T1's receipts); covering run at head vs a base worktree by failure name.
<!-- SECTION:PLAN:END -->
