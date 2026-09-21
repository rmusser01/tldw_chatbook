# PR2720 merged closeout

PR2720 merged as `7a758b8196bd3083ae9f878551d01dde7f4ec1ad` at
2026-09-21 15:13:15 UTC. Owner approval covers the Audit selection gallery;
Qodo's nullable docs and isolated coverage findings are addressed, and its
intentional deferred-import finding is dismissed. All three threads are resolved.
The final head is `46d5dadbc7fd702811f2efe2b24bcf0a6a6d0ebf`.

CI run 35614819365 passed 1,152 main cases and 123 admission cases (one expected
failure), with successful artifact, performance, CSS and backlog guards. Its
checkout `dd109e8bdb` combines that head with PR2723 security dev `e02604a459`.
Local checks of that candidate passed 46 cases and reproduced two upstream size
ratchet failures. Native run005 passed four cells and twenty captures.

## Actual merge differs from the CI candidate

PR2739 performance merged at 15:13:07 UTC, eight seconds before PR2720. Thus the
actual parent is `037a43dd32`, and the actual tree `6a28a52fa4374f4e63ba2dc81962e046eff5eaaf`
is **not** the CI candidate tree. Both are recorded separately in `merge.json`;
no claim is made that current-head CI tested the later performance changes.

The actual merge was qualified separately before closing TASK-32834:
**382 targeted checks passed**, covering all Workbench cases and Audit identity/
selection. Five architecture size checks failed identically on the incoming
parent: app, Console controller, Personas, transcript and MCP Workbench. Audit
leaves the first four unchanged and shrinks Workbench by seven lines. Exact
measurements are retained in `actual-merge/upstream-ratchets.json`; these are
upstream size-governance debt under existing TASK-32809, not an Audit regression.
No budget was raised and no failure is represented as passing.

All nine artifact guards pass on the actual merge. Independent review of the
incoming MCP/config interactions found no actionable issue: captured profile
identity, store CAS/locking, inspector ownership and exact tool keys survive the
off-loop writes; config defaults are preserved. Source styles and Audit identity
logic did not change during the incoming performance merge.

Fresh native run006 also passes all four theme/size cells with normal app and
fixture exits, released lock, ten healthy databases, zero chats/messages, unchanged
defaults/sentinels, zero network/tool calls and verified source/module provenance.
All twenty terminal captures match the approved content after timestamp
normalization. Eighteen SVGs also match after generated-ID normalization. The two
remaining filtered captures differ only in the input caret (one compact rectangle
is zero-width) and match the already
visually inspected Qodo replay. Capture hashes and compact differences are retained.

The initial sandbox-only Mermaid check could not fetch its pinned public input;
the explicit network-enabled retry and actual-merge preflight pass. Initial
failures and their provenance remain in the receipt. No full suite was run.

TASK-32834 is complete within its Audit selection scope. Saved PR2721 compact
filters and PR2722 inspector guidance remain separate visual-review slices.
