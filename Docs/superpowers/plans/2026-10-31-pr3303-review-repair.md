# PR3303 loss-dialog privacy implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, then a fresh-context superpowers:requesting-code-review of the final branch.

**Goal:** Preserve the new complete close/quit loss warning without copying private prompt-derived tab titles into the dialog.

**Architecture:** Keep the existing lazy shared inventory, controller revision fence and shutdown owners. Identify each affected session as `Tab N` in the store's creation order, which is also the tab-strip order. Both close and quit use that label; missing sessions fall back to `Tab`. No title provenance, persistence, or new state owner.

**Tech Stack:** Python 3.12, Textual 8, pytest.

**Spec:** TASK-33628.1 AC2 and TASK-33628.17; ADR-098 content-free lifecycle disclosure and ADR-033 state ownership.

ADR required: no new ADR.
ADR path: `backlog/decisions/098-visible-bounded-console-prompt-queue.md` (clarification), `033-application-session-state-ownership.md`.
Reason: enforce the existing privacy boundary; no new storage or service contract.

## Constraints and ruling

- Preserve current dev's recovered-image admission/drain, worker retirement, config shutdown, and lifecycle revalidation.
- Targeted local checks and independent review only; hosted CI and size limits are waived by the user. Drafts and PR3267/3297 stay excluded. No release work.
- Ruling: use positional labels for every title kind, including explicitly renamed tabs. Renamed text can itself contain private prompt text, and the current model has no trustworthy title provenance. This trades personalized loss-dialog names for guaranteed content freedom. Titles remain visible in the tab strip. Explain the positional convention in dialog copy and documentation.
- Leave the separately tracked notice-count, duplicate logical-input count and decrease-only reprompt issues in TASK-33628.16/.18/.19; retain the revision/identity fence.

## Repair and verification

- [x] Record ADR clarification and mark affected task criteria pending before edits.
- [x] Strengthen the real Temporary send fixture with a short private prompt; remove the Affected-tabs privacy exemption. Add close-title, renamed-tab and Home-screen quit assertions. Show the causal failure on the rebased implementation.
- [x] Replace title-derived labels in `UI/Console_Modules/loss_inventory.py` and `session.py` with positional labels. Test all title kinds, loss-free gaps in tab numbering, missing-session fallback, and close/quit agreement.
- [x] Update the user guide, census count (preserving every dev entry) and exact changed-module measurements. Remove the incoming duplicate composer budget.
- [x] Run the changed UI tests plus reached app-quit, modal/vanished-prompt, close-under-question, browser fence, workflow lifecycle, recovered-image and worker-retirement controls. Diagnose any failures against exact current dev; do not weaken guards. See the integration QA for baseline-matched workflow/pending-round failures and unexercised later paths.
- [ ] Record frozen receipts and qualifications, complete task notes, obtain independent final review, then refresh live draft/head/threads/auto state and merge with the authorized narrow CI bypass.
