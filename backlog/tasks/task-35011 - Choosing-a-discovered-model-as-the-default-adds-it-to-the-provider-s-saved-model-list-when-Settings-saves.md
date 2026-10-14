---
id: TASK-35011
title: Choosing a discovered model as the default adds it to the provider's saved model list when Settings saves
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- providers
- model-discovery
- model-config
- ux-review-2026-10-14
dependencies:
- TASK-33007
- TASK-35010
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
- [ ] #13 The capability line no longer ties capabilities to saving, on every surface that shows it
<!-- AC:END -->
