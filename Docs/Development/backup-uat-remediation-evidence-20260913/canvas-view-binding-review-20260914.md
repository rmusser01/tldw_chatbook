# Canvas binding caller idempotence: independent review

**Approved within the reviewed two-product-file scope. No actionable new finding.** Base d6406d4dc86720f9916e5416bf3e8de8cc7e647a; exact candidate/test/probe hashes recorded separately. No repo edits or full app boot by reviewer.

The runtime records ownership metadata on the actual installed binding after the original live policy gate, inside its existing Canvas lock. Direct binder calls retain their policy read, callback installation, listener/rebind behavior and return contract. The new query performs only state/identity comparisons: exact MethodType self/function comparison or callable identity, never arbitrary callable equality. It rejects detached/mismatched view, generation/controller mismatch, absent binding/controller, closed maintenance, disable latch and disposal. UI calls avoid redundant installation only for its currently attached view; an already-superseded view does not invoke the binder. No enabled decision is cached, no operation is widened, and effect/watcher/owner guards remain unchanged.

Independent checks:

- New repository lifecycle module: **9 passed in12.29s**, including native initial-pause retry, live pause effect refusal, direct binder compatibility, same-view stability, callback/controller replacement, same-view reclaim, clearing, successor and late outgoing lookup. Log `/private/tmp/uat-canvas-binding-independent-tests.log`.
- Real native import proof: **same-view PASS**. Original capture, real initial apply and compilation, actual ChatScreen store accessor, then original capture commit succeeds. Binding/generation/capture remain unchanged. **Successor PASS**: actual new attached view installs new callbacks; late old lookup does not replace them; old capture differs only in generation and is refused; fresh public native import succeeds. This reverses the independently proved baseline same-view refusal without weakening real generation invalidation.
- Private predicate probe: **PASS** across all five callback slots using callable objects whose equality raises; identity hits and changed-object misses execute no equality callbacks. Actual close/drain/resume, detach/reclaim, latch and disposal produce the required query outcomes. No native guard/result replacement.
- `git diff --check` passed.

Concurrency limit: attachment fields still belong to the existing attention lock, while the query uses the Canvas lock. This is not a newly atomic view claim. A true query result is used only to avoid an unnecessary UI binding call, never as effect permission. The existing race where a different thread changes the attachment after the caller's check but before a false-result bind is not claimed solved here; direct binder behavior is unchanged. No deterministic new ownership failure was observed or inferred from that theoretical interval. The tested successor handoff and late old lookup are sequential real lifecycle transitions, not proof of every concurrent interleaving.

Fixture limits: the import probes use the actual screen accessor on a narrow attached view holder, a live store-derived scope, real runtime/controller/compiler/import owner checks, a supported ephemeral Canvas session and real native-bound config. They are not mounted Textual or durable SQLite Canvas publication acceptance. The existing subprocess fixture's test environment is retained. No Windows startup performance improvement or60s pass is claimed; native platform and full UI verification remain the parent's integration work.

Receipts: `/private/tmp/uat-canvas-view-binding-independent-hashes.json`.
Reusable private probes: `/private/tmp/uat-canvas-import-candidate-probe.py`, `/private/tmp/uat-canvas-binding-query-predicate-probe.py`.

## Final frozen scope refresh

**Approval retained. Final repository native module independently11PASS15.48s**, log `/private/tmp/uat-canvas-binding-independent-final11.log`. Runtime product hash is unchanged from the prior approval. Screen callback construction changed only from keyword `dict(...)` to the equivalent string-key literal. The added two permanent import cases retain real capture, initial apply, compiler and commit; same-view commit succeeds, actual successor refuses the old capture and fresh import succeeds, with one actual Canvas at the end. Original nine lifecycle cases remain.

Runner review confirms exactly three added literal entries for the new module: existing full `_PRODUCT_TESTS`, existing support diagnostic tuple and existing native-close selection. No mode, deadline, assertions or existing node is removed. Parent reports native-close36/support-diagnostic348 collection; independent review verified the entries and the actual11-case module rather than repeating broad collection.

Parent additionally reports44 focused passes40.84s and all three old direct-binding test bodies passing in fresh native fixture `/private/tmp/uat-canvas-binding-native-compat-wjw0szgg`. Those are parent-run evidence, separate from the reviewer's exact baseline/current ordinary-fixture refusal comparison. The latter conclusively attributes the three ordinary failures to unchanged config selection refusal; no guards were relaxed. Parent static comparison reports no new Ruff/Bandit findings; reviewer `git diff --check` is clean.

Final four-file hashes and independent11 log hash: `/private/tmp/uat-canvas-view-binding-final-independent-hashes.json`. Installed mounted integration and future Windows results remain separate acceptance evidence; no startup performance claim follows from these11 tests.
