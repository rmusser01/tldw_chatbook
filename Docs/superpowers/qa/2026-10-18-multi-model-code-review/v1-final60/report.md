# Single qualification after CI repair

**Failed: AC24 remains open and PR3286 remains draft.** The one approved run used
26/60generation requests and2/3fresh children. It exited2 at
`live_report_incomplete_R1`; the unused allowance does not permit another
invocation. [Spending](spending.json); [independent review](outcome-review.md).

## Candidate and CI

Source:`b35bb489eaf62d8b10c6ab9779e3634411b02663`. All normal PR checks passed
before launch:55success,3expected skips, none pending/failed.
[Required CI](https://github.com/rmusser01/tldw_chatbook/actions/runs/40829740686).
The narrow shutdown guard passed native Windows:31focused cases and all7full-app
GGUF cases. Observed inventory[False] closing fell from142.37s/136.45CPU seconds
to17.22s/0.31CPU seconds. Windows eval and the repaired notes-sync shard passed.
The shutdown change had60targeted local passes and a clear scoped review.

## First failure and completion blocker

Luna initially submitted `findings` with an empty records array. The host refused
`invalid_record_count`; no-findings reports instead use coverage. Luna then
submitted empty coverage, received completion-repair feedback for the rejected
ID, and supplied a placeholder “No actionable finding” to repair that ID.
It finally requested completion with `empty_findings:true` despite the accepted
placeholder. Completion was acknowledged at revision3, but terminal sealing
correctly refused the contradiction. Accepted decisions and sealing were not
modified to force success.

Three narrow offline controls confirm the host gap: contradictory completion
is acknowledged but unsealable, while consistent nonempty and genuinely empty
reports seal. This establishes delayed cross-record validation/feedback; it
does not prove that the initial sparse feedback forced the model's placeholder
or that every no-findings recovery path fails. No new production repair cycle
or live retry was started.

R2/DeepSeek sealed and both original reports were presented. Its actual defect
observation is useful, but does not replace verifier evidence. The primary then
ended with an application-reported provider error; physical HTTP was200, so
its deeper cause remains unverified. V1/Haiku never launched, expansion and
unified output are unavailable, and native[2,1] completion was not demonstrated.

## Integrity and retained evidence

Actual new generations:9Qwen,10Luna,7DeepSeek,0Haiku;5metadata GETs separately.
Cumulative actual usage:412generation requests/28children. Native two-child
limits remained active. Source/team/config stayed unchanged and shutdown
released all recorded owners. No usage_unknown pause was observed.
Historical Haiku round-trip and scripted fault/duplicate/batching controls
retain their original scope; they do not qualify this failed live journey.

[Verified retention manifest](retention.json) preserves the raw packet, receipts,
review, Windows success log and offline reproducer. No evidence was deleted.
The task stays In Progress; no ready-for-review or merge action was performed.
