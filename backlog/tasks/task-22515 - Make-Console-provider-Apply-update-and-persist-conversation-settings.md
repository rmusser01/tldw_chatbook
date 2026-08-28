---
id: TASK-22515
title: Make Console provider Apply update and persist conversation settings
status: To Do
assignee: []
created_date: '2026-08-28 05:52'
updated_date: '2026-08-28 07:35'
labels: []
dependencies: []
references:
  - ADR-095
documentation:
  - >-
    Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Console Provider/Model popover and full Console Settings apply provider-generation choices and compaction to the exact conversation immediately, preserve them across restart through their existing owners, and give mouse and keyboard users the same reliable Apply behavior. Add explicit exact-model and new-chat default actions that remain truthful across live, disk, and runtime-publication failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mouse and keyboard `Apply to this chat` both close the popover and update the exact originating conversation once.
- [ ] #2 Provider, model, effective generation values, streaming, and compaction apply immediately to all work whose execution context is resolved after Apply while already-captured work remains unchanged.
- [ ] #3 Quick Provider and full Console Settings use one exact-origin Apply orchestration and the same durable conversation-generation contract while compaction keeps its existing context-policy owner.
- [ ] #4 Reopening a persisted conversation after restart restores its applied generation settings and compaction without storing credentials or endpoints.
- [ ] #5 Changing either provider or model rebases untouched fields from the target model's existing default chain, drops stale endpoint and unsupported fields, preserves visibly marked deliberate edits, and restores keyed A → B → A drafts.
- [ ] #6 `Full settings…` transfers the complete quick draft, compaction draft, field provenance, and exact origin into the full Model view without applying or discarding edits.
- [ ] #7 The quick footer exposes `Apply to this chat`, `Defaults…`, `Full settings…`, and `Cancel`; the Defaults substate replaces that footer with `Save as model default`, `Make default for new chats`, and `Back` while stating that compaction stays with this chat.
- [ ] #8 `Save as model default` applies live, closes, and field-masked-patches the exact literal provider/model profile only; quick saves temperature and streaming while full Settings saves all supported exposed fields, and blank removes the exact profile override.
- [ ] #9 `Make default for new chats` performs the same model-profile save plus an atomic global provider/model update that affects every eligible blank new chat immediately after runtime publication and across reboot.
- [ ] #10 Ctrl+T, temporary, workspace-created, and initial pristine blank chats use the saved global provider/model and exact model profile; existing/open conversations and deliberate Duplicate, Branch, Continue, or explicit handoff settings remain unchanged.
- [ ] #11 Only full Settings `Make default for new chats` may persist an explicitly dirty, checked endpoint; its preview contains a sanitized host and conservative Local/LAN/Remote-or-unknown classification, with no DNS lookup, credentials, or URL details.
- [ ] #12 Compaction remains conversation-only through its existing context-policy owner and is never copied into model profiles or global defaults.
- [ ] #13 Unsaved conversations stage both conversation-durable components until first persistence, while temporary conversations remain non-durable unless promoted.
- [ ] #14 Invalid input, duplicate activation, and dismissed deferred callbacks cannot create a false-success close, double commit, or teardown error.
- [ ] #15 Conversation generation/context-policy failures remain visibly identified per session and revision with Retry, while quick-surface context-policy failure may be labeled compaction.
- [ ] #16 Default mutation failure before file replacement is app-global and offers `Retry default save` / `Discard retry`; successful file replacement with runtime-publication failure offers cache-only `Refresh running app` / `Dismiss` and never repeats the disk mutation.
- [ ] #17 Locked exact-field config mutation preserves sibling model profiles, unexposed fields, literal punctuated model IDs, and unrelated concurrent edits; a newer explicit default action supersedes stale retry intent.
- [ ] #18 Targeted tests cover interaction and 60×24/72×24 layout, exact-origin execution, provider/model draft rebasing, quick-to-full transfer, persistence/resume, eligible new-chat paths, inheritance, endpoint safety, config concurrency, staged persistence/promotion, partial failures, retry/discard/dismiss, and restart behavior.
<!-- AC:END -->
