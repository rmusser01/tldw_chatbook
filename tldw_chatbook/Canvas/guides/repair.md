# Canvas repair

Work on the user's accepted artifact using received tool diagnostics or an
explicit repair request. A browser failure never automatically submits a chat
turn. Do not claim to see browser diagnostics that have not been supplied, and
do not invent a status-query tool.

## Read the current parent before editing

1. Use `canvas_list` if necessary to identify the intended Canvas.
2. Use `canvas_read` to inspect its current complete selected source, revision ID,
   and exact runtime profile. Respect explicit historical selection and branch
   semantics; never assume a remembered version is still selected.
3. Apply the requested correction to that source. Load only missing relevant
   guidance (`basics`, `controls`, or `mermaid`), reusing text already in context.
4. Use `canvas_update` with the complete replacement and the read revision's
   `expected_parent_revision_id`.

On a revision conflict, reread the current selection and adapt the requested edit
to the newer source. Do not overwrite unseen work or retry the same stale parent.
Repeated conflicts require reporting the conflict and letting the user decide,
not unbounded retries.

## Report the state supported by evidence

- **Staged:** source was accepted for this turn. Atomic turn settlement still
  determines whether it persists; do not call it saved yet.
- **Saved:** the revision persisted. This alone does not prove preview success.
- **Preview pending:** browser preparation has not yet reported success.
- **Preview ready:** received evidence confirms that exact revision rendered.
- **Preview failed:** source can remain saved despite a parser, layout, script,
  or quota failure. Inspect that revision's diagnostics and source. An old image
  or explicit View previous does not prove the new revision rendered.
- **Source-only:** the exact runtime profile cannot execute. Preserve readable
  source and history; do not present an unavailable preview as working.

## One bounded correction, then reassess

Use concrete diagnostics to narrow the fix: remove an unsupported API, correct an
explicit diagram declaration, shorten labels, or split an over-budget diagram.
Keep the accepted purpose and current profile. No cache, retry loop, fetched
library, external runnable HTML, setting change, or alternate runtime is an
automatic workaround for a refusal.

After one failed repair attempt, stop, explain the remaining limitation using
available evidence, and let the user choose whether to spend more on repair.
Do not generate speculative versions in a loop. Ordinary bounded corrections
are covered by the accepted request; unrelated artifacts and substantial
unrequested redesigns need a new offer.

Guides are documentation, not profile-admission authority. Unknown, retired,
revoked, missing, or integrity-failed profiles stay source-only without
substitution. Follow current exact-profile guidance and runtime checks. Adapting
source to a current profile requires the existing explicit **new Canvas** flow;
retain the original source, profile, and history instead of silently migrating.
Packaged runtime updates require restarting the native host or served parent and
children; refreshing a browser does not reload profile policy.

If Canvas tools are unavailable, explain briefly and continue with useful chat
content. If only a guide is unavailable, use already available authoritative
runtime guidance when sufficient; otherwise report the limitation rather than
inventing APIs.
