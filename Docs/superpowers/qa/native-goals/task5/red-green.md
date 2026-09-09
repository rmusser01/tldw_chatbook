# Focused tests-first evidence

The implementation report retains temporary red logs and exact gate selections.
These are the observed failures and subsequent covered behavior, not a claim
that a scripted provider is a real-model test.

| Red observation | Final coverage |
|---|---|
| Missing setup/status module | Mounted controls/setup6 + review/navigation10 |
| Missing subscribe/reference factory/launch lifetime owner | Real notification detach, trusted owner, cancelled-view/setup-drain tests |
| Canonical save did not accept agents section | Actual saved-file atomic batch, Revert and fleet independence |
| Actual palette had no Goal runs command | Real discover/activate test |
| Fenced provider refused at native_goal_required | Real fenced-controller trusted CLI regression + plain-provider refusal |
| Outgoing handoff lacked final/intermediate distinction | Captured real outgoing request regression; both actual endpoint traces retained |
| Service lacked exact checkpoint/run lookup | Actual Changes callback pins saved native run and goal conversation |
| Setup read only TOML despite enabled environment | Actual owner-derived setup stages immutable request without dispatch |
| Backoff test made2calls after Pause/Resume rather than1 | Clear wake before awaited read; controlled stale-Ready/Pause interleaving |
| Accept stayed visible after selecting older checkpoint | Two real checkpoints; selection clears captured decision; old Review refuses stale_result_review |

The first broad final run had186passes,1live skip and2failures: the lost-wake bug
above, and an occupancy assertion against a fixture whose change tracker was
intentionally disabled. The fixture now uses real ShadowRepoService and
ChangeTurnTracker. A later Stop assertion expected clean Stopped while actively
interrupting a provider; the service correctly retained recovery_required and
unknown spend, which the final mounted test asserts. These harness corrections
are distinguished from production fixes.
