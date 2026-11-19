---
id: TASK-41527
title: "User Guide accuracy sweep: Library Tab claim and remaining dated changelog prose"
status: Done
assignee: ['@claude']
updated_date: 2026-11-19 16:10
created_date: 2026-11-17 05:27
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/library.md
- Docs/User_Guide
dependencies:
- TASK-41101
- TASK-41147
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs sweep (TASK-41101, round 15). (1) `library.md` says "Tab stays on this screen"; the old stamp said Tab walks the top tab bar, so check the claim from a cold landing and correct the page or the screen. (2) Dated changelog prose remains in `notes.md`, `file-notes.md`, `agent-runs-and-tools.md`, `settings` and a `collections.md` blockquote; it belongs in the task notes. The small body corrections are in TASK-41147.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Tab claim is checked live from a cold landing and the page matches
- [x] #2 The listed pages have no dated changelog prose left, found by a grep recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Drive a cold landing headlessly (real app, run_test through run.sh): boot straight onto Library, and enter Library from Console; record where focus starts and where each Tab lands. Correct the page to match, or report if the screen should change (this stream is docs-only).
2. Grep the listed pages (notes.md, file-notes.md, agent-runs-and-tools.md, settings.md and settings/*.md, collections.md) for dated changelog prose: "(Was …", "superseded", "Fixed as task", "as of TASK-", "Until <date>", "Before this", "used to", "previously", "Measured on <date>", "corrected <date>", ISO dates.
3. Remove each changelog clause, or restate it in the present tense where it carries a current fact; leave hits that are product copy, example dates or links, and say which.
4. Re-run the grep, Tests/Docs, preflight, task-file check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Tab claim was wrong for a cold landing, so the page now says what happens. The dated changelog prose on the listed pages is gone, each clause dropped or restated in the present tense.

**AC#1, the Tab claim.** "Live" here means the real app driven headlessly; this round's rules forbid launching the TUI itself. The check used the real app (`_build_unsplashed_test_app`, the `TldwCli` boot path), headless at 235x52, through run.sh, in a throwaway probe (deleted; copy at scratchpad `fu/r17dc/probe_r17dc.py`). Two cold landings were tried:
- boot straight onto Library (`configured_default="library"`);
- boot to Console, then Ctrl+3.

Both landed the same way. Focus starts on the nav bar's `nav-home` ("⌃1 Home"). Tabs 1-14 walk `nav-console` … `nav-kanban`. Tab 15 enters the content at `library-rail-collapse`, then `library-ingest-top-button` (Import…), `library-search-input`, and on through the rail. From `nav-home`, Shift+Tab lands on `library-hub-action-search` (the last control in the Library content), and `/` and F6 each focus `library-search-input`.

So the old stamp was right: Tab from a cold landing walks the top bar. The page's "Tab stays on this screen" holds only once focus is inside the content. The code agrees. `LibraryScreen` rebinds Tab (`UI/Screens/library_screen.py:976`) and confines it to `_LIBRARY_TAB_REGION` (`:8598`) only while focus is inside the content or nothing is focused; from the chrome it walks the whole chain (`:8766`). First-paint focus is left to Textual's `AUTO_FOCUS` (`UI/Navigation/base_app_screen.py:311`), which picks the first focusable widget, the nav bar's first button.

Changing the screen is a code change, outside this docs-only stream, so I corrected the page: `library.md`'s Tab / Shift+Tab row now describes both states and the ways into the content. The in-content half is pinned by `test_library_crit8_keyboard.py::test_tab_from_a_library_canvas_never_reaches_the_nav_bar`. That test is plain and dies locally on the `RecoveryRequired` trap; under a temporary `@pytest.mark.bootstrap_profile` marker (reverted) it passed. Whether a cold landing should put focus in the content instead is a product question; I list it as a proposed follow-up.

**AC#2, dated changelog prose.** Pages: `library/notes.md`, `library/file-notes.md`, `console/agent-runs-and-tools.md`, `settings.md` and `settings/*.md`, `library/collections.md`. The grep, run from `Docs/User_Guide`:

```
P='\((W|w)as |(W|w)as "|superseded|[Ff]ixed (as|in|by) (task|TASK)|corrected 20|[Uu]ntil 20[0-9]{2}|rewritten (by|on|in) |[Aa]s first shipped|[Bb]efore this|as of (task|TASK)-|[Mm]easured on 20|[Pp]reviously|used to|20[0-9]{2}-[01][0-9]-[0-3][0-9]'
grep -n -E "$P" library/notes.md library/file-notes.md console/agent-runs-and-tools.md settings.md settings/*.md library/collections.md
```

At the base it matched 50 lines (notes 19, file-notes 8, agent-runs 14, settings.md 4, settings/rag.md 2, collections 3; personal-context-profile 0). The 33 changelog lines among them were removed, or restated where they carried a current fact:
- **notes.md**: four "(Was … superseded by task-N)" parentheticals (32612, 32604, 32271/32611 twice); "(Was "Sync folder (name unavailable before cutover)" … task-32451)"; "; was "N applied · durable receipt recorded""; three "Before this …" sentences (receipts, the CRLF postcondition entry, deleting a synced note); and the New-note width paragraph's "no longer inflates … it once kept" and "before this, pressing **New** …". The CRLF paragraph keeps its current facts in the present tense ("An ordinary edit … completes"; an entry an older build left open reads "⚠ Needs attention · Next: Resolve recovery", with `NOTES_TREE_SYNC_ATTENTION_STATUS` at `Library/library_notes_tree_state.py:34` and the "Resolve recovery" label at `UI/Library_Modules/library_notes_sync_controller.py:279`).
- **file-notes.md**: "(Was "Back to navigator" …)"; the Session Git "(Was headed "Prepare session for commit" … rewritten by task-15122 on 2026-08-11.)"; "(was "Trust Session Git repository?")"; the Chunking Lab "(Was … superseded by task-32064 …)".
- **agent-runs-and-tools.md**: the "(This paragraph previously promised … corrected 2026-08-14 … task-16300 …)" aside; "(Measured on 2026-08-17 …)"; "(Fixed as task-17500, 2026-08-17.) As first shipped …" (the bold present-tense lead stays); "The list used to live in the left rail's Agent section …"; "exactly as before this behavior existed".
- **settings.md**: "(they used to stop with "API Key required")"; "as of TASK-21591".
- **settings/rag.md**: "used to be rejected; the engine now clamps" became "is not rejected: the engine clamps".
- **collections.md**: the "This page was rewritten. Until 2026-09-08 … (task-32057 AC#1, 2026-09-11) …" blockquote became one present-tense paragraph. It names Collections and Quick Capture (`Widgets/Library/library_collections_capture_reader.py:512`), says the older generic tables are read-only, and notes there is no "Add to collection" (no match in `tldw_chatbook/`). "superseded generic Collections tables" now reads "older".

After, the same grep matches 17 lines, none of them changelog:
- example dates inside UI copy (notes.md 492, 743, 882, 960);
- QA/plan link paths (agent-runs 21, 1840, 3085; settings/rag.md 384);
- present-tense "previously linked/trusted/saved/stored" (file-notes 214, 516; agent-runs 151, 1889, 1977; settings 1726);
- "cancelled or superseded child" as a state name (agent-runs 1538);
- "most-used" (settings 994);
- notes.md 491, "a note written before this version picks up its `[[Title]]` links the next time it is saved", which is guidance for old notes, not history.

Undated "no longer …" contrasts (for example settings.md's Theme lines, agent-runs "Sub-agents … no longer run one at a time") state present behaviour and carry no date or task, so they are out of this criterion and were left.

Checks: no test or script pins any removed sentence (grep of `Tests/` and `scripts/`; the only hits are unrelated code comments). `Tests/Docs`: 44 passed.

**Fix round 1 (review r17dc-rev1, finding 3).** The grep above missed task-stamped history in `settings/rag.md`:
- the Reranker provider paragraph's "since TASK-17065 the reranker dispatches…";
- step 5's "since TASK-17065 those calls really go out. Before that fix reranking silently no-opped…".

Both are now present tense: the reranker dispatches through the same table, so the name you pick is the provider that gets called and billed. Step 5 keeps the live warning: "A profile with the toggle already ticked spends on its next search without you touching anything; untick **Enable reranking** to opt out."

The recorded grep is widened with `[Bb]efore that` and `since (task|TASK)-`:

```
P='\((W|w)as |(W|w)as "|superseded|[Ff]ixed (as|in|by) (task|TASK)|corrected 20|[Uu]ntil 20[0-9]{2}|rewritten (by|on|in) |[Aa]s first shipped|[Bb]efore this|[Bb]efore that|since (task|TASK)-|as of (task|TASK)-|[Mm]easured on 20|[Pp]reviously|used to|20[0-9]{2}-[01][0-9]-[0-3][0-9]'
```

Re-run across the same pages, it matches 20 lines, none of them changelog. They are the 17 listed above, with their line numbers shifted by the rag.md edit, plus three present-tense "before that":
- agent-runs 1499, "review before that chain can wake again";
- settings.md 1447, "the theme that was active before that change", the Revert target;
- settings/personal-context-profile.md 118, "pressed again before that, it asks".

A separate check for `[Ss]ince (task|TASK)-`, `[Ss]ince 20…`, `Before that fix` and `After (task|TASK)-` finds nothing. No test or script pins the two removed rag.md sentences; the hits for "silently no-opped", "since TASK-17065" and "really is the" are unrelated code comments.

Files: `Docs/User_Guide/library.md`, `library/notes.md`, `library/file-notes.md`, `library/collections.md`, `console/agent-runs-and-tools.md`, `settings.md`, `settings/rag.md`.
<!-- SECTION:NOTES:END -->
