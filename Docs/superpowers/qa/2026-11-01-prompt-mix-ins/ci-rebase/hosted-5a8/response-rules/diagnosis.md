# UI26 current-head diagnosis — cause before repair

Head diagnosed: `5a8f23eb9dfabeda25d46ed168d8ac675f483217`. Original hosted evidence is `/private/tmp/mixins-5a8-ui26.log`. Production source remained read-only throughout diagnosis. The new permanent regression was added only after the cause and narrow correction were accepted; the original four end-to-end cases remain unchanged.

## Confirmed cause

`ConsoleSessionController._sync_console_session_draft` reads `store.session_input_snapshot(visible_session_id)` while reconciling a switch snapshot. That visible owner can already have been closed. Its neighboring old-session draft write tolerates precisely this absence, but the newly consulted old snapshot does not. The resulting KeyError escapes a screen-owned sync worker and reaches Textual's fatal exception handler. Current-active snapshots must still fail if their required owner disappears; this is only an optional old resident draft witness.

The complete pre-repair method AST is identical at dev88af, b110, and5a8 (`draft-sync-baseline-proof.json`). This fault is inherited, not introduced by the reviewed extraction or Mix-ins type. It is nevertheless a concrete lifecycle defect under the user's all-valid-issues requirement.

The hosted primary assertion is the `/rules` modal wait at test line146, after successful learning/repair and reopening. The hosted log contains only two Send-handler entries, so the final click never reached command parsing. Its final exception stack names the retired old visible session, while a new blank session is active. Hosted output alone does not timestamp when that first app exception occurred.

A b110 isolated source-overlay run reproduced the exact timeout/state and same KeyError with unmodified test body and production callable identities. Metadata timing showed: original close returned at37.3518 with old composer binding; resume captured that retired owner at37.3520; original background draft sync consulted it at37.6308; App._handle_exception received WorkerFailed(KeyError old owner) at37.6397; `/rules` click began only at59.4627. Therefore the KeyError precedes and prevents command delivery in this reproduced schedule, rather than being solely teardown masking. The first-events summary and full metadata event stream are retained.

**Qualification:** this whole-interpreter profiling run took105.45s and substantially changed scheduling. The original30s action deadline/assertions were retained. It proves causal ordering in that schedule; it does not establish the uninstrumented failure rate, reproduce hosted load, or independently prove the hosted exception timestamp. Repeated later DuplicateIds reports in this run followed the first fatal KeyError; they are not promoted to a separate finding.

A second, narrow **no-profiler** controlled native probe confirmed the actual lifecycle defect in7.70s. It held only the original resume tree read after the real switch-snapshot capture, then invoked the original close owner. The old session was removed; the close's original draft reconciliation raised KeyError for that retired snapshot owner; releasing the original loader allowed the original resume to succeed. No session/source/authority field was fabricated. This diagnostic replaced only the selected test callable in memory while retaining its normal UI collection/bootstrap/fixture ownership; it is not a pass claim for the original test body.

## Smallest approved repair

Catch KeyError only around that optional **old visible** snapshot lookup and skip only its resident-revision override when absent. Continue new-owner draft loading, undo/suffix reconciliation, exact binding, and revision publication. Keep current-active reads and all surviving-old revision checks unchanged. Do not add sleeps, increase deadlines, bypass source admission, return early, or suppress whole-sync exceptions.

The session module is exactly6990 lines at its unchanged6990 ceiling. Independent review approved moving only `_consume_visible_agent_handoff`'s unchanged executable body into the already resident `composer_run_controls` owner beside its existing authored/received draft consumption. The original SessionController method/signature/doc stays as a lazy delegation facade. Its live self/store/composer operations, incarnation/revision/generation/edit-serial/segments checks and order remain exact. No new module, eager dependency closure, or raised size pin is needed.

The permanent Event-barrier regression drives original resume, actual snapshot capture, original close, original resume publication, correct new composer/transcript, and actual `/rules` Send/modal with no provider/helper calls. Adjacent existing saved-projection, handoff and received-input controls retain surviving-session revision/suffix and current input authority coverage. Root owns task/governance/census records and final independent review.

## Preserved observations

- `current-exact-normal.log`: original exact node passed12.47s; it may have overlapped the already-started UI31 diagnostic, so this is explicitly qualified.
- `current-observed.log`: original exact body passed11.75s under private callable wrappers. Those wrappers can alter stock-method identity admission, so this is path observation only, not stock-authority or performance qualification. The saved event stream was reduced to metadata; no full config payload is retained.
- `b110-profile.log`: original exact body failed105.45s as described above. The11 changed production Python blobs were supplied via a private import loader, retaining normal module paths/resources and bootstrap. Repository/index/HEAD were not altered. See `b110-overlay-manifest.json` and private blob tree.
- `current-lifecycle.log` / `.observation.json`: bounded controlled native diagnostic passed7.70s, confirming retired-owner KeyError. Outer diagnostic90s process bound was not reached; original action deadline unchanged.

No startup/performance test or full suite was run. Diagnosis native processes have exited; permanent regression RED/GREEN and final frozen verification are recorded in report.md.
