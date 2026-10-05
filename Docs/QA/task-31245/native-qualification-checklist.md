# Character navigation: native qualification checklist

Status: **HOLD — partial macOS evidence retained; qualification incomplete.**

Earlier clean `81c0d22ff9538f75135c09eb0c43a3f31b7a0ccc` observed Active MRU
switching and cold/warm History reuse at 244×73. Latest clean
`18acbb41836a3262e0cdc94b3f7346cd1f89db9b` retries retain current/selected
Active marking, pointer access to Character chats and normal Ctrl+Q; background
typing/resize controls did not reliably commit. Required viewports, Character
workflows, Windows and participants remain unqualified. At latest app return,
28 database descriptors remained open. See
`fixture-rebuild-2026-10-04.md` for exact outcomes and raw receipt paths; do not
promote this partial walkthrough into a complete native or resource pass.

This closes the external evidence gaps in TASK-31243/31244/31245. Pilot,
PTY dependency probes, old screenshots and service timings do not substitute
for this walkthrough. Run only after the controller records a clean frozen
source commit, a prepared disposable synthetic profile, its fixture manifest
and the exact launch command. The earlier ignored launch packet is unavailable
in the current checkout; do not launch against a real profile to replace it.

## Record before starting

- Full source commit and clean working-tree receipt; dataset/manifest hashes.
- OS, terminal and shell versions, Python and Textual versions.
- Terminal cells, font, font size, zoom and input/key-remapping settings.
- Operator and date; identify each screenshot/recording by case and source.
- Confirm a dedicated window, synthetic conversations, no real-profile access,
  no provider requests, and no changes to the operator's existing sessions.

Keep macOS and Windows Terminal results separate. Do not mark an unavailable
host, blocked input, interrupted process, or unobserved quit as passed.

## Native keyboard and pointer workflow

Repeat at 52×20 and 120×50; additionally inspect 72×35 and 80×24 during resize.
Use actual native keyboard/pointer input, not injected Pilot events.

1. Open Console with two distinct chats. Record current identity and tab count.
   Ctrl+K opens Active; blank Enter selects the most recently used **other** tab.
   Reopen it and confirm current-tab marking differs from the highlighted target.
2. Type an Active query, use arrows and Enter. Only the highlighted exact chat
   activates. Zero matches widen only through the visible explicit action.
3. Cycle Active → History → Character chats with Shift+F3. Current mode, selected row
   and Enter destination must be visible. Active/History share their per-visit
   query; Character chats keeps its own query and never silently widens.
4. In History, resume one saved chat cold, then resume it again warm. Repeat
   through Character chats. Record the exact conversation ID, transcript marker,
   active tab and tab count every time: no duplicate opens or title-only match.
5. Use Character Keyword search. Each unselected result has only its two-line
   summary; additional detail appears only for the selected result. Exact Enter
   reaches Console with the expected transcript and composer focus.
6. Exercise precommit Escape and a controlled slow activation from the prepared
   fixture. Preserve the prior chat and query when cancelled. After commit,
   Escape must not roll back or retarget the committed open. Record which
   boundary was actually observed; do not infer it from an elapsed delay.
7. In Context, find Character immediately below Conversations. Check at most
   four headers, only current/most-recent expanded on first use, at most five
   recent chats per ordinary group, date order and exact `View all N in Roleplay`.
   Check explicit disclosure preference after return/reopen.
8. Search Context globally, activate an exact result, clear/Escape, and verify
   browse disclosure, focus and scroll return. Use Continue search to transfer
   the accepted query into Character chats. No Meaning control is expected yet.
9. Follow `View all N in Roleplay`, find an older conversation, open its saved
   preview and return. Exercise Stay, Discard and Save with a synthetic draft;
   unsuccessful save must not discard it or navigate.
10. Select an Unavailable-character fixture. Recovery opens only the exact
    requested Library inspection. Refusal/cancellation retains the switcher
    query/highlight; returning from accepted recovery restores the source anchor.
11. Rename a bound active chat through tab/F2, then an inactive chat through the
    rail menu. At confirmation, saved row, open tab and switcher titles agree;
    the inactive rename does not switch tabs. Selecting it later shows the new
    transcript header. A fixture-controlled refusal leaves the old titles.
12. At compact sizes, verify readable target/detail, cell-aware truncation,
    reachable Cancel, keyboard focus order, paging and actual pointer targets.
    Resize back wide without persisting responsive disclosure or stealing focus.
13. Quit normally through the implemented app control. Record clean process
    exit and resource receipt; a controller signal is not native quit evidence.

For every case record: observed result, exact identity where relevant, evidence
file, and **Pass / Fail / Blocked / Not run**. Keep failures and attempts; a
successful rerun does not erase the first result.

## Three first-time participants

Use three people unfamiliar with this navigation, each with a fresh disposable
profile/disclosure state. Provide the same synthetic task, not the steps above:

> You have several agents and conversations open. Find and resume the older
> conversation with the named character and its unique transcript marker.

Start timing after the scenario is read. Record elapsed time, wrong destinations,
duplicate tabs, hesitation, assistance and the actual resumed identity. Do not
coach during the two-minute task; any intervention is recorded as assistance.
At least two of three must resume the exact chat within two minutes unaided.

Afterward ask each participant to explain the difference between a character
card, a saved conversation and an open Console tab. Then ask them to recover the
Unavailable-character fixture and record their path and comprehension. Use
participant codes, not personal information, in committed receipts.

## Separate performance gate

The native workflow above is not a latency benchmark. Retain standalone Keyword
retrieval provenance for a freshly generated 10,000-chat / 250,000-selected-
message corpus and 30-query manifest. Then run the existing production-owner
latency harness alone, on the frozen head, at 52×20 and 120×50. Preserve raw
preparation/activation and busy-paint timings, observer limits and automatic GC
records. Limits remain 50 ms maximum event-loop interval and 100 ms busy paint.
ADR-198's boot freeze is an accepted partial improvement, not a waiver or a
passing matrix. Windows, participant and resource outcomes remain independent.
