---
id: TASK-35010
title: Saved model lists keep what the config file holds, and shipped sample model ids never pose as saved models or defaults
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- providers
- model-discovery
- config
- data-integrity
- ux-review-2026-10-14
dependencies:
- TASK-33007
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two verified problems make saved model lists untrustworthy in Settings ▸ Providers & Models.

C3 (data loss): Save selected appends to the startup snapshot of [providers] that the app keeps in memory, then rewrites that provider's whole list. A model added to the list after the app started, through Advanced Config or by another program, is silently removed. A model the user deleted comes back. Switch model and the Settings provider-switch fallback keep listing removed models until restart, because nothing refreshes the in-memory lists after an Advanced Config save. The config cache can also miss an edit made in place (TASK-34943, TASK-34700), so the fix has to read the file itself at write time instead of trusting any cached copy.

A-pm-design-M2: the shipped template seeds the vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 lists with sample ids no server serves (vllm-model-z, tabby-model, aphrodite-engine, custom-model-alpha and others). Switching Settings to one of these providers pre-fills Model with a sample id, so readiness names a model that does not exist and the documented 'Save selected fills an empty Model' rule (TASK-369) never fires. Switch model lists the samples as ready rows. Discovery-first cloud providers already ship empty lists. Existing profiles already hold the samples, so the fix has to cover files written by older versions without rewriting them at startup. It must also leave the shipped-default detection alone (the TASK-33005.6 ruling that keeps untouched local servers under NOT RUNNING) and change nothing about what is sent to a provider. Recognising the old samples by a frozen set of their values is a deliberate, scoped departure from TASK-34100.2 AC#9's marker rule: the set covers only ids that older versions already wrote, which carry no marker, and TASK-34100.2's marker rule still governs the template.

Plan: `backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md`. Findings closed: C3, A-pm-design-M2 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Save selected keeps every model the config file holds for that provider at the moment it writes: a model added after the app started, through Advanced Config, by another program replacing the file, or by an in-place edit of the same file, is kept, and a model removed that way does not come back
- [ ] #2 Save selected's receipt counts only the models it actually added to the file, and says the selected models are already saved when the file already lists all of them
- [ ] #3 The startup catalog refresh with save to config keeps such external changes the same way and still adds newly listed models
- [ ] #4 After Save selected, an Advanced Config save or a Config reload, Switch model (Alt+M), the Default model picker and the Saved model list rows and counts show the file's saved lists without a restart
- [ ] #5 A new profile's saved model lists for vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 are empty
- [ ] #6 On a profile whose config file still holds the old sample lists, Switch model, the Default model picker, the Saved model list counts and the first-run wizard's model list show none of the sample ids, and real ids saved beside them still show
- [ ] #7 Switching Settings to one of those providers on such a profile leaves Default model not set, also when the provider's settings table still holds the shipped sample model, and Save selected then fills it as an unsaved draft
- [ ] #8 The next Settings write to such a provider's list removes the sample ids from the file and keeps the real ids, and the app does not rewrite the config file at startup for this
- [ ] #9 A provider whose settings table is still exactly the shipped one is still treated as never set up, so an untouched local default that refuses stays under NOT RUNNING in Switch model
- [ ] #10 Dreams stories, briefings and Console sends use the same model for every provider as before
- [ ] #11 The Settings User Guide says the vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 saved lists start empty and fill through Discover models, that old sample ids are ignored and dropped the next time Settings saves that list, and that Save selected adds to the list as the config file holds it
<!-- AC:END -->
