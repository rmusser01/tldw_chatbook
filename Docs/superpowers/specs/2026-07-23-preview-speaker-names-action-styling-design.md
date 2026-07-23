# TASK-437 — Preview conversation: real speaker names + styled RP action text

- **Date:** 2026-07-23
- **Task:** TASK-437 (RP/character-card UX review). Preview labels speakers "character:"/"you:" and renders `*action*` asterisks literally.
- **Branch base:** origin/dev (tip `8a46af45e`).

## Problem

The Roleplay preview transcript labels every reply `character:` and every user line `you:` (hardcoded f-strings), and single-asterisk RP action spans (`*smiles*`) render as literal asterisks. Both make the surface feel out-of-genre. Two small, independent changes:

- **AC#1:** label preview messages with the character's actual name (and the persona/user name once available).
- **AC#2:** render single-asterisk `*action*`/emphasis spans as styled (italic) rather than literal asterisks.

## Key mechanics (verified)

- **Labels are hardcoded, display-only.** All five label sites are inline f-strings in `Widgets/Persona_Widgets/personas_preview_pane.py` — `append_user` (`f"you: {text}"`, :171-173), `append_reply` (`f"character: {text}"`, :175-177), `begin_reply` (`"character:"`, :179-187), `append_reply_chunk` (`f"character: {self._partial_text}"`, :200-208), `_render_seed_lines` (greeting, `f"character: {self._greeting}"`, :263-264). No constant, no name lookup.
- **The prefix is NOT a parsed contract.** `transcript_text()` has exactly two consumers: `personas_preview_controller.py:251` (a non-empty boolean guard — label-agnostic) and `:337` `open_in_console()`, which stages it as a plain-text handoff **body** (`body=transcript[:LIMIT]`), never parsed into roles. The LLM request is built from `self.history` (role dicts), not the labelled transcript. So changing labels cannot break any parser/provider path — the handoff body simply reads with real names, which is better.
- **The character name IS knowable but not threaded.** The controller receives `character_name` at `reset_for_character` (:80) and derives it at `handle_character_loaded` (:245, `record.get("name") or state.selected_entity_name`), but discards it; the pane has no screen/state reference. The persisted source is `state.selected_entity_name` (`personas_state.py:51`).
- **No persona/user-name concept exists yet.** Persona selection blanks the preview (`_select_profile` → `preview.reset("")`); the user side is hardcoded `"User"`/`"you"`. Introducing an active persona/user name is explicitly **TASK-442**. So AC#1's "once available" ⇒ set the character name now; the user label stays the default `"you"` until TASK-442.
- **Every line is `Static(line, markup=False)`** (`_append_line` :278-284, and the streaming/seed variants), so `*action*` renders literally. `markup=False` was a deliberate guard against `MarkupError` on unmatched Rich brackets in transcript text — pinned by `test_markup_like_transcript_content_renders_without_raising` (`test_personas_preview.py:382-396`).
- **`Static.renderable` is a repo compatibility shim** (`tldw_chatbook/__init__.py:30`) — `property(lambda self: self.content, lambda self, v: self.update(v))`. It returns the **original content object** passed to `Static`. Tests read `str(line.renderable)` (`_line_texts`, `test_personas_preview.py:41-42`). Therefore, to keep those reads returning **plain** text (and to preserve the literal-bracket regression), the styled content must be a Rich **`Text` object**, not a markup string.
- **Textual internal-name hazard:** `self._render_markup` is a Textual `Widget` instance attribute (the `markup=` flag). A pane method named `_render_markup` would shadow it → `self._render_markup(line)` becomes a `TypeError` (calling a bool). The helper must use a different name (`_styled_line`).

## Design

### AC#1 — real speaker names

Add two label fields + a setter to `PersonasPreviewPane`:
```python
# in __init__
self._character_label = "character"
self._user_label = "you"

def set_speakers(self, *, character: str | None = None, user: str | None = None) -> None:
    """Set transcript speaker labels; empty/None keeps the current label."""
    if character:
        self._character_label = character
    if user:
        self._user_label = user
```
Change the five label sites to interpolate `self._character_label` / `self._user_label` instead of the literals `"character"` / `"you"`. The `role_class` (`personas-preview-line-you` / `-character`) is **unchanged** — styling/test selectors stay role-based, decoupled from the display name.

Wire `set_speakers(character=<name>)` at the two character-seeding entry points, **before** the seed/render, so the greeting and turns carry the name (verified callers: `reset_for_character` and `restore_conversation` are called only from `_select_character`; `handle_character_loaded` is the async load completion):
- **Screen `_select_character`** (`personas_screen.py:1779`, has `entity_name`) — `self.query_one(PersonasPreviewPane).set_speakers(character=entity_name)` **before** it dispatches to `preview.restore_conversation(...)` (:1805) or `preview.reset_for_character(...)` (:1819). One site covers both the normal-select and TASK-434 restore paths — **no `restore_conversation` signature change**.
- **Controller `handle_character_loaded`** (name derived at :245) — `pane.set_speakers(character=name)` before its `pane.seed_greeting(...)`, so the async load refines the label if the loaded record's name is fuller/different than the selection name.

Empty/unknown name ⇒ label stays `"character"` (today's behavior). User label stays `"you"` (TASK-442 owns the user/persona name).

### AC#2 — styled `*action*`, surgically (Rich `Text`, not the Markdown widget)

Add a helper that escapes Rich markup, italicizes single-asterisk spans, and returns a Rich `Text`:
```python
# module import: from rich.text import Text ; from rich.markup import escape ; import re
_ACTION_SPAN = re.compile(r"\*([^*\n]+)\*")

def _styled_line(self, line: str) -> Text:
    """Render a transcript line: escape Rich markup, italicize *action* spans."""
    return Text.from_markup(self._ACTION_SPAN.sub(r"[i]\1[/i]", escape(line)))
```
`_append_line` (and the streaming/seed variants) keep `self._lines` **plain** (so `transcript_text()` stays raw — correct for copy/handoff/restore) but mount/update the `Static` with the styled `Text`:
```python
def _append_line(self, line: str, role_class: str) -> None:
    self._lines.append(line)                       # plain text, unchanged
    self.query_one("#personas-preview-transcript", VerticalScroll).mount(
        Static(self._styled_line(line),
               classes=f"personas-preview-line {role_class}", markup=False)
    )
```
The streaming path (`begin_reply` mounts, `append_reply_chunk` `.update(...)`) and `_render_seed_lines` use `self._styled_line(...)` the same way.

Why this is safe and correct (verified in a mounted probe):
- `escape(line)` turns every `[` into a literal, so unmatched Rich brackets never raise — the regression guard holds. `Text.from_markup` then applies only our own well-formed `[i]…[/i]` pairs (the regex always balances).
- Passing a `Text` object (not a markup string) means the `renderable` shim returns the `Text`, so `str(line.renderable)` is **plain**: `"Bob: *waves* at [bold]x"` → `"Bob: waves at [bold]x"` (asterisks consumed, italic span applied, `[bold]` literal); `"character: [/oops]"` → `"character: [/oops]"` (literal, no backslash — regression preserved); `"you: 5 * 3"` → unchanged (no closing `*`).
- Markdown widget was rejected: full CommonMark over-interprets chat text (`"# hi"`→heading, `"1. x"`→list), which is wrong for a transcript.

**Known limitation (accepted, mirrors markdown):** spaced math like `5 * 3 * 2` italicizes the middle span. RP actions (`*waves*`) are the target; flanking-rule refinement is out of scope.

## Testing (`Tests/UI/test_personas_preview.py`)

- **AC#1 default:** with no `set_speakers`, greeting/reply lines still label `"character:"`, user lines `"you:"` (unchanged fallback).
- **AC#1 named:** call `pane.set_speakers(character="Sherlock Holmes")`, seed + append → `_line_texts` and `transcript_text()` show `"Sherlock Holmes: …"` for replies/greeting and `"you: …"` for user lines. A controller-level test that `reset_for_character`/`handle_character_loaded` call `set_speakers` with the record name (the mock-pane test in `test_personas_preview_restore.py` can assert the call).
- **AC#2 helper (pure):** `pane._styled_line("*waves*")` is a `Text` whose `str()` is `"waves"` and which carries an italic span; `_styled_line("[/oops]")` str is `"[/oops]"`.
- **AC#2 rendered:** append `"*smiles*"` → the mounted line's `str(renderable)` contains `"smiles"` and **no literal `*`**; assert the italic span exists (`any("italic" in str(s.style) for s in line.renderable.spans)`).
- **Regression preserved:** `test_markup_like_transcript_content_renders_without_raising` still passes (content seeded with no character name → label stays `"character"`, `[/oops]`/`[bold]unclosed` render literally, no raise). Update only if its expected label string is affected (it is not — no name is set).
- **Update existing label assertions:** `test_seed_append_reset_roundtrip` (:84-111), `test_transcript_lines_carry_role_classes` (:173-183), the streaming test (:271-298), and the test-reply tests (:186-215) currently assert literal `"character: …"`/`"you: …"`; where they don't set a name, they stay `"character:"`/`"you:"` (no change); add name-based variants for AC#1.

## Risks / mitigations

- **Label out of sync with the seeded character:** `set_speakers` is called at each seeding point before the render, so greeting + turns always carry the current character's name.
- **MarkupError:** eliminated — `escape()` first, then only balanced `[i]` pairs; `Text.from_markup` on that never raises.
- **Test-accessor semantics:** the `renderable` shim returns `content`; passing a `Text` keeps `str(renderable)` plain (verified). `transcript_text()` reads `self._lines` (plain), independent of rendering.
- **Streaming flicker:** a partial `*wav` shows the asterisk until its closing `*` arrives, then italicizes — acceptable.

## Non-goals

- The active persona / user-name substitution — the user label stays `"you"` (**TASK-442**).
- Full markdown (bold, lists, headings) — only single-asterisk italic per AC#2.
- Streaming behavior changes (**TASK-439**); this task only renders whatever the existing stream path produces.
- Per-speaker color/weight styling of the transcript lines (no CSS for `.personas-preview-line-*` exists today; out of scope).
