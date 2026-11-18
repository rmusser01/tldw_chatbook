# ADR-364: Keep native finalized Style ownership

Renumbered from ADR-230 under the oldest-date rule.

Status: Rejected candidate; retain the native ownership contract
Date: 2026-11-11
Related task: TASK-31966

## Context and control

The owner authorized comparing the remaining three stability candidates. A
previous disposable control cleared two native Styles.node links after successful
widget teardown, with exact-type and CPython reference-count guards. It removed
cycles but did not clear the latency matrix. No production implementation exists.

A fresh Textual 8.2.8 / CPython 3.12.11 native control mounts a Static, removes it,
borrows its styles only after teardown, copies its inline style, remounts the
same widget and changes its width to 13. Native ownership survives and width
becomes 13. The candidate passes the exclusive-reference check but loses all
three owner associations; after remount its width remains 80. The check proves
exclusivity only at one instant, not absence of future borrowers or reuse.

## Decision

Reject clearing finalized Style owner links as a production latency remedy.
Keep native borrowing, copies, remount refresh, teardown errors and GC policy.
Do not replace owners with weak proxies, add a global descriptor interception,
or retire virtual component styles under this candidate's approval.

Restoring links only on attach could address remount refresh but would still
change late borrowing/copy semantics before attach. A weak proxy changes owner
identity and lifetime. Either needs a separately proven compatibility design,
not an extension of the failed reference-count control.

The two independent app-owned candidates avoid repeated render metadata reads
and discarded transcript policy reads. Their correctness tests and fresh latency
comparison determine what is retained; neither is presumed to satisfy 50ms.
All 50ms loop / 100ms busy-paint limits and parent qualification gaps remain.

## Later restoration control

A disposable missing-attribute restoration variant preserves exact Style objects
and passes late borrowing, inline copy and native remount width 13. It still fails
an ordinary shallow Widget copy: before any later original-owner lookup, the
copy's shared base Style points to the copy rather than the original Widget.
Native copying keeps the original owner. The first observer inadvertently read
the original styles before checking ownership and repaired the candidate state;
the corrected observer checks ownership first and exposes the mismatch.

Do not adopt the restoration variant or add copy/descriptor interception to
paper over this failure. Both controls and the failed observer remain local.

A further disposable control preserves all Style owners but clears native
rendering caches only after successful Widget exit. Clearing style lines still
breaks remount paint: native retains the text, the candidate paints a blank row.
BoxModel-only clearing passes that small control; layout-only has no populated
entries. These observations do not establish a safe general cache-retirement
policy. A stricter owner-exclusivity probe sees 13/14 Widget references and
retires nothing; do not relax its count to treat unknown references as internal.

Plan: [continuation](../../Docs/superpowers/plans/2026-10-21-task31966-owner-free-style-colors.md).
