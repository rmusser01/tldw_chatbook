# PR3300 Terminal ownership review repairs

Execute inline; obtain an independent Superpowers review of the exact final head before publication.

Goal: finish the existing Terminal focus/return contract without changing its runtime authority or Windows refusal.

ADR required: no new ADR.
ADR path: backlog/decisions/099-persistent-terminal-session-runtime-boundary.md, backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md.
Reason: implement the existing bounded user-only Terminal input and conversation-owned reading-state contract. No new backend, storage, provider projection, or authority.

Initial source: af1addda74c8dfa4b8b524cddf13814e4acd7d67. Current preparation rebases onto PR3298 merge db7e04cf484b4c0b803366b00f2bbae1b5550042. Preserve PR3299 cue/focus integration when it lands. Draft PRs, PR3267/3297, release and main are excluded. Hosted CI and size limits are waived; local targeted checks and independent review remain required.

1. Confirm current-dev preservation, task scope and existing ADRs. Keep both rail-state test families and exact measured size metadata.
2. Add causal regressions before production changes: held transcript sync retains pending reading state; an owner switch before the restore callback never receives the previous conversation offset; focused capturing Terminal paste reaches the bounded manager path, with released/modal/refused controls; voice Escape remains with a capturing Terminal and keeps its existing route elsewhere.
3. Keep return reading state until the existing successful transcript publication path can restore it. Recheck session and mounted center at the actual after-refresh callback; retain it through deferral and discard it only after restoration or owner change. No retry timer or parallel sync owner.
4. Route Paste from the actual focused capturing viewport through the existing atomic paste API and selected screen bracketed-paste mode. Never forward it to the model or bypass runtime admission/bounds. Gate only the conflicting priority voice Escape while that viewport owns input.
5. Correct released-input cue copy and new fixed visual padding through existing design tokens; rebuild generated CSS. Preserve size rather than compressing logic to fit a budget.
6. Run only affected Terminal mounted controls, rail-state units and changed CSS/pin contracts with private fixtures and frozen sources. Do not weaken Windows refusal, networking, security or timeout guards. Qualify any unavailable POSIX real-PTY evidence explicitly; author historical live claims are not new local results.
7. Update existing task notes/criteria and bounded follow-up33622.23 after its defect is actually resolved; record exact failures/passes/limitations, commit, obtain independent review, and merge in oldest-ready order.

Initial review: C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3300-initial-review.md (stored in the PR3298 preparation parent). Important findings are deferred-return restoration, missing Paste routing and priority voice Escape ownership. Minor released-mode cue and token padding are included only as bounded fixes. No broader terminal/parser/backend redesign is authorized by this plan.


Verified adjustment: Textual 8 `_styles_builder._process_integer` explicitly rejects zero for `line-pad`, including token-expanded zero. Native receipt r-103ed829 reached 5 fast passes but all four mounted cases stopped at stylesheet parse. Retain the existing bounded Python `line_pad` workaround (already used by composer/rail controls); only ordinary padding moves to its token. This rejects that portion of the minor source-review suggestion based on executable parser evidence, without changing label geometry.


Independent integration review follow-up: bind captured/restored state to the transcript region's published owner, keep it pending through deferred selection reveal until the actual completion callback, resolve paste target and mode from one current manager view snapshot, and surface only InputOfferResult.safe_message for refused paste. Five causal failures are retained in r-a22cf718 (4) and r-3186d3b4 (1). Register the new return/input regression module in the existing admission-sensitive Console P0 lane; full historical Terminal integration enrollment remains separate TASK-33622.22. No hosted workflow is dispatched.
