# Independent send-readiness fixture review

Verdict: no remaining actionable Critical, Important, or Minor findings in the settled test-only patch against `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

Scope: `Tests/UI/test_console_send_ready_attachment.py`, connected production control-bar/config-sync/replay bodies, and the implementer's retained targeted results. No product edits, test edits, or test executions were performed by this reviewer.

## Behavior and evidence

The original Workbench/composer test called `_sync_console_control_bar()` without inspecting its Boolean result. Production documents and implements False when checked config cannot be acquired immediately and a fresh coalesced replay owns the unfinished render. The existing replay timer is 0.2 seconds; a generic pilot pause does not itself guarantee a completed projection. The new `_sync_controls` helper waits for an actual successful projection with a monotonic four-second bound and yields to Textual between unsuccessful attempts. It replaces only the two affected unchecked calls. Both button-state comparisons, all staged payload cases, and the existing real image-only send/provider-payload oracle remain intact.

The added native control holds the actual `_settings_rebuild_lock()` on another thread. It waits the real checked-display lifetime rather than overriding a guard, clock, proof, or configuration authority. After expiry, while the lock is still held, it stages the blocked image and checks explicit projection refusal. It also proves the bounded helper refuses to claim completion under that condition and that both buttons still show their distinct old enabled state. The lock releases in finally, the holder is joined and verified stopped, and subsequent waiting only pumps the native pilot and observes both buttons becoming disabled. There is no direct projection call after release. Finally, the visible send action refuses and the attachment remains staged.

Connected source confirms `_sync_console_control_bar` calls `run_console_presentation_sync`; config entry attempts native locks without blocking and defers on contention. The existing coalesced callback re-enters the real projection after its timer. This fixture neither supplies a fake checked proof nor changes permissions, attachment custody, or send authority.

## Review corrections incorporated

Two proof gaps in the initial new control were raised and corrected before closeout: (1) directly calling the completion helper after lock release could drive the projection itself rather than demonstrate native replay; (2) staging before the one-second wait could let a still-fresh proof paint the final state before release. The settled test observes convergence without direct refresh and stages only after expiry, asserting the old state before release. These are resolved test-evidence issues, not outstanding product defects.

## Retained checks inspected

Evidence directory: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-f095-ui3-anqs9n0b`.

- `final-module.log` / `.json` / `.xml`: five tests passed in 16.69 seconds. This ran before the final strengthening of the added control; the four original readiness/custody/image-send oracle bodies and completion helper were unchanged afterward.
- `final-replay-control.log` / `.json` / `.xml`: final strengthened control passed in 6.43 seconds on current production.
- `baseline-final-control.log` / `.json` / `.xml`: the identical final control passed in 7.11 seconds on untouched `ba1` production with only the test copied. This supports attribution to the existing deferred projection contract; it does not claim the original hosted timing failure was deterministically reproduced locally.

These are implementer-produced execution artifacts read by this reviewer, not reruns. No full-suite, hosted-CI, or broader performance success is claimed.

## Byte qualification

Reviewed working-tree SHA-256 for `Tests/UI/test_console_send_ready_attachment.py`: `95d9e09eace05edb39a13dcfebcb54090bfa75319438399b6bde47cb15acc781`.
