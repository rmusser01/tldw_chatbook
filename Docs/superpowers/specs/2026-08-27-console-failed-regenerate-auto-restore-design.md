# Console Failed-Regenerate Auto-Restore — Design

**Date:** 2026-08-27
**Status:** Approved in conversation; pending document review
**Task:** `TASK-571`

## Problem

Regenerating an assistant message creates a new sibling and moves the active leaf to
that sibling. If the provider fails or returns no content, the sibling becomes
`failed`. Provider-context projection skips failed messages, while the prior good
answer is now off the active path, so the next send loses that answer until the user
manually swipes back.

The failure is not data loss: the original answer and the failed sibling both remain
in the conversation tree. It is an active-path recovery problem.

## Decision

After a regenerate attempt settles, automatically restore the original assistant
message as the active leaf when the newly created sibling has status `failed`.

- Successful regenerations keep the new sibling active.
- User-stopped regenerations keep their partial `stopped` sibling active.
- Failed and contentless regenerations restore the original answer.
- The failed sibling remains stored, traceable, and reachable through sibling
  navigation for inspection or retry.
- Existing failure run state and toast behavior remains unchanged.

This is implemented as a postcondition in `ConsoleChatController.regenerate_message`,
the one call site that knows both the original anchor and the replacement sibling.
The generic streaming engine and `ConsoleChatStore` terminal semantics remain
unchanged.

## Data flow

1. Validate the regenerate request without mutating the tree.
2. Create a new assistant sibling and stream into it normally.
3. Let generic streaming settle the sibling and publish the existing failure state.
4. Read the settled sibling. If its status is `failed`, call the existing
   `set_active_leaf(session_id, anchor_id)` API.
5. Record the regenerate trace against the actual replacement sibling and its failed
   status, regardless of which leaf is active afterward.

For durable conversations, `set_active_leaf` persists the restored pointer through
the existing local-only active-leaf setter. No message content, sibling ownership,
or source branch is mutated.

If the session or replacement disappears while the stream is settling, the existing
session-closed behavior wins; the controller does not attempt to recreate state.

## User experience

The existing provider-failure toast/run state still explains that regeneration
failed. The transcript returns to the prior complete response instead of leaving the
user on an empty or failed branch. Branch controls still expose the failed sibling if
the user wants to inspect or retry it.

No new modal, action, keybinding, or persistent preference is introduced.

## Testing

- Update the empty-stream branching regression to prove the original answer becomes
  the active leaf and is present in the next provider payload.
- Update the transport-failure regression to prove the same restoration while the
  failed sibling remains present and retryable.
- Keep success and stopped-stream regressions proving those sibling branches remain
  active.
- Verify the focused branching/controller suites and run changed-file static checks.
- Exercise the mounted/live Console flow with a controlled post-validation provider
  failure and confirm the original answer is visibly restored in one automatic step.

## Alternatives rejected

### Leave the failed branch active and add a swipe-back hint

This preserves the current active-path footgun and depends on the user noticing and
following recovery copy before sending again.

### Add a store-level recovery abstraction

Only regenerate knows the semantic relationship between the anchor and replacement.
A new store API or recovery map would duplicate information already held by the
controller.

### Thread regeneration metadata through generic streaming

This would couple every stream terminal path to one branching feature and expand the
change surface without improving the outcome.

## ADR check

ADR required: no

ADR path: N/A

Reason: this is a bounded correction to failure recovery using the existing branching
and active-leaf contracts. It adds no schema, storage policy, service boundary,
security rule, or long-lived UI structure.
