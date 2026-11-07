---
id: TASK-38226
title: 'Character card Markdown parser is quadratic on adversarial input'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 18:43'
updated_date: '2026-11-05 18:00'
labels:
  - character-chat
  - performance
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
load_character_card_from_string_content's Markdown branch (its frontmatter and fenced-JSON regexes) takes quadratic time on adversarial input, so a 10 MB crafted `.md` card would hold the card import's worker for a long time. It exists on dev today. Roleplay frame B2b's sniff never runs that parser, but a file the sniff routes to cards still reaches it; B2b does not fix the parser. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The parser reads a 10 MB adversarial Markdown file in linear time, pinned by a test with a time-independent bound (operation count or input-size scaling)
- [x] #2 Every Markdown card that imports on dev still imports the same
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-measure both regexes through the real parser on dev and pick crafted inputs that take seconds there.
2. Replace each regex with a linear scan in a new, lazily imported `Character_Chat/card_markdown_scan.py` that returns exactly the regex's `group(1)`.
3. Keep dev's regexes in the test as the oracle: Hypothesis, a seeded sweep, a corpus of normal cards and Roleplay's import fixtures, comparing both the extracted text and the whole parser's output.
4. Pin linearity: a fixed bound on the few-KB inputs (red on dev) and a 2.5 MB to 10 MB input-scaling ratio.
5. Update the B2b sniff's comments, which named the parser as quadratic.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Exists on dev (29592ecc9b); fixed here.** Through `load_character_card_from_string_content` on dev, measured in this task's tests:
- `"---\n"` plus 2,000 newlines (2 KB) took 2.7 s, and 6.0 s in a standalone run. That is cubic: 0.2 s at 500 newlines and 0.6 s at 1,000.
- `"---\nx"` plus 48,000 newlines took 2.8 s.
- `"```{"` repeated 12,000 times took 2.6 s (quadratic).

With the scan the same inputs take 0.016, 0.47 and 0.018 ms. The worst crafted 10 MB families take about 0.15 s of parser CPU time.

**Approach.** `Character_Chat/card_markdown_scan.py` holds `frontmatter_body()` and `fenced_json_object()`. Each returns exactly what dev's regex `group(1)` returned, for any string. Each makes a fixed number of linear scans:
- `str.find`, `rfind` and `startswith`;
- one anchored `\s*`;
- three possessive searches whose start positions never re-read each other's input.

The derivation from the regex's backtracking order is in each docstring. The parser imports the module inside the function, so the screen pre-import's module set is unchanged. Everything after extraction (the YAML load, the JSON validation, the logging) is unchanged.

**Trade-off.** A Python loop over `str.find` candidates was prototyped first. It is linear, and its operations can be counted. But a crafted 10 MB card in which every character is a near-miss candidate took 12.6 s (``"`" * 10M``). The C-level searches take 0.34 s. A Python operation counter cannot see inside the regex engine, which is where the bug was. So AC#1 is pinned two ways:
- by input-size scaling of the parser's own `thread_time` (2.5 to 10 MB, the least of three runs, ratio under 10; linear measures 3.5-4.5, over six crafted families);
- by a fixed 0.5 s bound on the three inputs above. That bound is the red test on dev; the scaling test cannot be, since dev never finishes those sizes.

**AC#2.** Dev's two regexes are kept verbatim in the test as the oracle. The comparisons are:
- Hypothesis on structured frontmatter and fence shapes with noise, including Unicode whitespace, CRLF and `ſ` (which `re.IGNORECASE` matches to `s`);
- a seeded 40,000-string sweep;
- a 302-entry corpus of normal cards. It covers V1, V2, V3 and Unicode cards; compact and indented JSON; ```` ```json ````, bare, ```` ```JSON ```` and inline fences; prose; an earlier code fence; two cards; a BOM; YAML variants; YAML errors that fall back to a fence; CRLF copies; and the suite's own Markdown card texts;
- Roleplay's import fixtures.

For the corpus and the fixtures, the whole parser's output is compared, with the scan and with dev's regexes patched in. A spy proves the patched regexes ran. A cross-arm run of the corpus, the fixtures and a 20,000-string sweep through the real parser on dev and on this branch gave identical output.

**Live check.** The real Roleplay Import picker was driven, mounted at 160x45 with real stores, through `roleplay_import_app`'s harness, timing from the Import click to the toast:
- The three crafted `.md` files were refused in 0.24-0.52 s on this branch and in 1.61-2.27 s on dev, with the same "Import failed: the file did not contain a valid character card." toast.
- `card_frontmatter.md` and `card_json_block.markdown` imported Mara Quill and Odo Fenn on both arms.

**Sniff.** The B2b sniff's docstring, its `_JSON_FENCE` comment and its test docstring said the parser was quadratic. They now say it was, until this task made it linear. The sniff still never calls it: the sniff reads only 64 KiB, which can cut a card short, and the parser logs an error for every non-card while repeating work the importer does anyway.

**Not fixed, exists on dev.** YAML alias expansion. `yaml.safe_load` keeps anchors shared, but `json.dumps` expands them, so a 217-byte frontmatter becomes 5.8 MB of JSON, and each extra alias level multiplies that by 10. It is not a regex issue and is outside these ACs, so it is reported for a follow-up task.

**Files.**
- `tldw_chatbook/Character_Chat/card_markdown_scan.py` (new)
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- `tldw_chatbook/UI/Persona_Modules/roleplay_import_sniff.py` (comments)
- `Tests/Character_Chat/test_character_import_markdown.py`
- `Tests/UI/test_roleplay_import_sniff.py` (docstring)
- `Tests/Architecture/test_module_size_ratchet.py`: the Character_Chat_Lib row is lowered from 4,775 to 4,773.
- `backlog/docs/lessons-testing-evidence.md`
<!-- SECTION:NOTES:END -->
