# Console response rules

Use `/omfg <what was wrong>` after a completed text answer to teach the Console a response check. For example:

```text
/omfg You claimed tests passed without saying which tests ran.
```

The Console drafts a structured rule and tests it against the recorded answer plus three synthetic controls. A successful local save activates the rule for the current Chat. The agent then attempts to repair its answer or finish missing work, retaining completed actions and the original answer.

This is an example test, not a guarantee of future model behavior. An unavailable helper, missing evidence, ambiguous applicability, malformed judge output or failed calibration is reported honestly. It does not count as passing.

## Managing rules

Open `/rules` or **Chat settings → Response rules**. Use `/rules workspace` for the Chat's named Workspace, or `/rules global` for the current local profile. The canonical **F9 Settings → Hooks → Response rules · this profile** opens the same manager.

The manager shows local pins, inherited rules and inactive drafts. Review the title, applicability, detector, correction guidance and recorded examples. Literal checks support required text, forbidden text and headings, with explicit letter-case matching. Semantic applicability and semantic checks require a judge.

**Test** creates an inactive revision. **Save** activates the exact tested definition only if its source and binding are still current. Changed criteria require testing; unchanged predicates and examples can reuse their recorded validation after explicit guidance review. If the original answer is unavailable, select a completed replacement example and Test again. Failed or cancelled testing retains the editor text and the existing pin.

**Disable here** stops a rule in the selected scope. **Exclude here** masks the logical rule, including future inherited revisions. **Delete pin** removes the local override, allowing a broader binding to apply again. **Promote** previews the exact stored revision before applying it to a Workspace or the current profile; private examples and source text stay in their original scope.

Nearest scope wins: Chat, then Workspace, then current-profile global. Global rules remain local to this profile and device. Promotion is explicit.

## Checking and repair

Checks run after successful text generation and required lifecycle settlement. They use the completed answer, its original task and already available execution evidence. They do not retrieve new documents, read files or grant tool permissions.

Statuses distinguish **Checking rules**, **Passed active rules**, **Rule violation**, **Couldn't verify**, **No applicable rules** and **Correction limit reached**. Mixed results disclose unavailable checks. Activate a rule status to open its manager and Disable action. Current primary work and approval requests take precedence over an earlier result.

A confirmed violation can request at most two native repairs, within the shared maximum of three automatic native/hook follow-ups and 120 seconds. The remaining primary run budget can reduce these limits. Hooks keep their own authority and may veto continuation. Waiting user input takes priority; unavailable checks do not request repair.

Each helper is bounded to 30 seconds and 4,096 output tokens. The application admits at most four physical rule helpers, with at most one for a live Chat. Cancelled work retains its capacity until the actual worker finishes. Learning allows at most three candidate attempts within 120 seconds. Stop remains available after text generation while checks are running.

Helper usage appears in the existing cost totals with its own purpose and pricing. Unknown or partial usage stays explicit. These helper totals currently last for the application session.

## Local history and recovery

Private rules, fixtures, validations and assessments use local sidecar storage. They are excluded from ordinary conversation metadata, sync and export. Temporary Chats keep the same records in memory until an explicit Save adopts them atomically; failed Save retains memory state for retry. Forks do not copy Chat-local rules.

Reopening a Chat does not replay learning, checks or repairs. Imported restore data remains inactive and inert. Exact local rollback is a separate recovery operation. Removing the original answer can make an editor test require a replacement example.

If activation succeeds but initial repair fails or is stopped, the rule remains active and the notice distinguishes that result. The original answer and completed work remain available.

Architecture: [ADR-219](../decisions/219-console-learned-response-rules.md). Qualification: [implementation evidence](../../Docs/superpowers/reviews/2026-10-03-console-response-rules.md).
