# Codex-Style `$`-Mention Skill Invocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Console user skill invocation to the Codex convention — `$skill-name` becomes THE invocation (leading form keeps `{{args}}`; embedded mentions splice argless at position), and `/skill-name` invocation is hard-removed.

**Architecture:** A pure mention scanner (code-span-aware, exact case-sensitive matching) lives in `console_skill_resolver`; `_apply_skill_substitution` in the controller gains a leading-`$` branch (today's semantics re-sigiled) plus an embedded splice pass with a new non-aborting notes channel; the composer's `/`-fallback claiming and the `/skills` run form are deleted.

**Tech Stack:** Python 3.11+, pytest. No new dependencies.

**Design doc:** `Docs/superpowers/specs/2026-07-23-skills-dollar-mention-invocation-design.md` (read it if a requirement here seems ambiguous — it governs).

## Global Constraints

- **Sigil rules (verbatim from spec):** leading `$skill-name args…` = arg-bearing whole-message form (args → `{{args}}`, inline-replace / fork-takeover, untrusted → refuse); embedded `$skill-name` anywhere else = **argless** splice at the mention's position preserving ALL surrounding prose; multiple embedded mentions each expand.
- **Embedded matching is exact + case-SENSITIVE** against canonical (lowercase) skill names: `$PATH`/`$HOME`/`$5` stay literal. Leading form keeps `resolve_skill_command`'s case-insensitive/unique-prefix behavior.
- **Code spans skipped:** mentions inside ``` fenced blocks or inline `backtick` spans stay literal.
- **No recursion:** single pass over the user's original text; a `$mention` inside a spliced body is inert.
- **Fork cannot be embedded:** embedded mention whose `execute_skill` result has `execution_mode != "inline"` stays literal (no note). Embedded untrusted (SkillTrustBlockedError) stays literal + a system note. A leading resolved mention does NOT get its args embedded-scanned.
- **Hard remove `/skill-name` at BOTH layers:** the composer fallback resolver (chat_screen registration + `make_skill_fallback_resolver` itself) and the controller's leading-`/` branch. `/skills` bare list stays; its `<name> [args]` run form is removed.
- **Payload ordering preserved:** skills → dictionaries → world-info at all 4 send sites; the deliberate skip-skills site (search "may execute skills with side effects") stays skill-free.
- **Ephemeral only:** splicing happens on the provider payload; transcripts keep raw text.
- **Tests:** run via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q`; trust/UI suites need the `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` prefix. `Tests/Skills/test_skills_library_flow.py::test_skill_editor_canvas_scrolls_trust_panel_into_view` is a known flaky (cross-test pollution) — re-run in isolation, don't chase.
- **Commit hygiene:** `git add` ONLY the files each task names — NEVER `git add -A` (the tracked scratch file `.superpowers/sdd/progress.md` must not be swept in).

## File Structure

| File | Change |
|------|--------|
| `tldw_chatbook/Chat/console_skill_resolver.py` | Add `MENTION_SIGIL`, `SkillMention`, `find_embedded_mentions`, `_code_span_mask`, `SKILL_MENTION_SKIPPED_NOTE`; `format_skills_list` rows `$name`; DELETE `make_skill_fallback_resolver` (Task 4) |
| `tldw_chatbook/Chat/console_chat_controller.py` | `_apply_skill_substitution` (def at `:1727`): leading branch re-sigiled to `$`, embedded splice pass, return becomes 3-tuple with notes; 4 call sites updated |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Remove fallback registration (`:1865-1867`) + `make_skill_fallback_resolver` import (`:100`); `/skills` run form → hint; pick-submit composes `$name`; blocked-hint copy → `$name` |
| `Tests/Chat/test_console_skill_resolver.py`, `Tests/Chat/test_console_skill_substitution.py`, `Tests/UI/test_console_skill_commands.py`, `Tests/Chat/test_console_command_grammar.py` | New scanner tests; flip `/`-pinning tests to `$`; delete fallback-resolver tests |

---

### Task 1: Pure mention scanner + `$name` list rows (resolver)

**Files:**
- Modify: `tldw_chatbook/Chat/console_skill_resolver.py` (constants near `:28`; `format_skills_list` near `:155`)
- Test: `Tests/Chat/test_console_skill_resolver.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MENTION_SIGIL = "$"`; `SkillMention(start: int, end: int, name: str)` frozen dataclass (`start` = index of the `$`, `end` = one past the last token char); `find_embedded_mentions(text: str, names: frozenset[str]) -> tuple[SkillMention, ...]`; `SKILL_MENTION_SKIPPED_NOTE` (str, `.format(name=...)`). Task 3 consumes all four.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_skill_resolver.py`)

```python
from tldw_chatbook.Chat.console_skill_resolver import (
    SkillMention,
    find_embedded_mentions,
)

_NAMES = frozenset({"code-review", "style-guide", "path"})


def test_embedded_mention_found_mid_prose():
    text = "please $style-guide this draft"
    mentions = find_embedded_mentions(text, _NAMES)
    assert mentions == (SkillMention(start=7, end=19, name="style-guide"),)
    assert text[7:19] == "$style-guide"


def test_embedded_mention_trailing_punctuation_stays_prose():
    mentions = find_embedded_mentions("run $style-guide.", _NAMES)
    assert mentions[0].name == "style-guide"
    assert mentions[0].end == 16  # the "." is not part of the token


def test_case_sensitive_exact_match_only():
    # $PATH stays literal even though a skill named "path" exists.
    assert find_embedded_mentions("echo $PATH", _NAMES) == ()
    assert find_embedded_mentions("echo $path", _NAMES)[0].name == "path"
    # prefix / unknown / numeric stay literal
    assert find_embedded_mentions("$style", _NAMES) == ()
    assert find_embedded_mentions("$5 and $100", _NAMES) == ()


def test_multiple_mentions_all_found_in_order():
    text = "$code-review then $style-guide"
    names = [m.name for m in find_embedded_mentions(text, _NAMES)]
    assert names == ["code-review", "style-guide"]


def test_code_spans_are_skipped():
    fenced = "look:\n```sh\necho $path\n```\nand $path here"
    mentions = find_embedded_mentions(fenced, _NAMES)
    assert len(mentions) == 1
    assert fenced[mentions[0].start :].startswith("$path here"[:5])
    inline = "use `$path` literally but $path expands"
    inline_mentions = find_embedded_mentions(inline, _NAMES)
    assert len(inline_mentions) == 1
    assert inline_mentions[0].start == inline.rindex("$path")


def test_skills_list_rows_use_dollar_sigil():
    from tldw_chatbook.Chat.console_skill_resolver import (
        SkillCommandCandidate,
        format_skills_list,
    )
    listing = format_skills_list(
        (SkillCommandCandidate(name="code-review", description="d"),)
    )
    assert "$code-review" in listing
    assert "/code-review" not in listing
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_skill_resolver.py -q`
Expected: FAIL — `ImportError: cannot import name 'SkillMention'`.

- [ ] **Step 3: Implement** (in `console_skill_resolver.py`)

Add near the existing constants (`SKILL_ARGS_MAX` is at `:28`):

```python
MENTION_SIGIL = "$"
"""Leading character of a Codex-style skill mention (``$skill-name``)."""

_MENTION_TOKEN = re.compile(r"[A-Za-z0-9-]+")

SKILL_MENTION_SKIPPED_NOTE = (
    'Skipped "${name}" — this skill needs review before it can run. '
    "Open /skills to review it."
)


@dataclass(frozen=True)
class SkillMention:
    """One embedded ``$skill-name`` mention found in a draft.

    Args:
        start: Index of the ``$`` sigil in the scanned text.
        end: Index one past the last token character.
        name: The matched canonical (lowercase) skill name.
    """

    start: int
    end: int
    name: str


def _code_span_mask(text: str) -> list[bool]:
    """Return a per-character mask, True inside markdown code spans.

    Fenced blocks: a line whose stripped form starts with ``````` toggles
    fence state; fence lines and everything inside are masked. Inline spans:
    paired backticks within a non-fence line are masked inclusively; an
    unpaired backtick masks nothing.
    """
    mask = [False] * len(text)
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            for i in range(pos, pos + len(line)):
                mask[i] = True
        elif in_fence:
            for i in range(pos, pos + len(line)):
                mask[i] = True
        else:
            i = 0
            while i < len(line):
                if line[i] == "`":
                    close = line.find("`", i + 1)
                    if close == -1:
                        break
                    for j in range(i, close + 1):
                        mask[pos + j] = True
                    i = close + 1
                else:
                    i += 1
        pos += len(line)
    return mask


def find_embedded_mentions(
    text: str, names: frozenset[str]
) -> tuple[SkillMention, ...]:
    """Find embedded ``$skill-name`` mentions eligible for splicing.

    Exact, case-SENSITIVE matching against ``names`` (canonical lowercase
    skill names): ``$PATH`` stays literal even when a skill named ``path``
    exists. Mentions inside markdown code spans are skipped. Single pass —
    callers must never re-scan spliced output (no recursion).

    Args:
        text: The draft text to scan (the user's original message).
        names: Canonical skill names eligible for expansion.

    Returns:
        Non-overlapping mentions in document order.
    """
    mask = _code_span_mask(text)
    mentions: list[SkillMention] = []
    index = 0
    while index < len(text):
        if text[index] == MENTION_SIGIL and not mask[index]:
            match = _MENTION_TOKEN.match(text, index + 1)
            if match is not None and match.group(0) in names:
                mentions.append(
                    SkillMention(start=index, end=match.end(), name=match.group(0))
                )
                index = match.end()
                continue
        index += 1
    return tuple(mentions)
```

(`re` and `dataclass` are already imported in this module; add them if not.)

In `format_skills_list` (near `:155`), change both row-building lines from `f"/{candidate.name} — {candidate.description}"` / `f"/{candidate.name}"` to `f"${candidate.name} — {candidate.description}"` / `f"${candidate.name}"`. Update any pre-existing `format_skills_list` test that asserts `/name` rows to `$name` (that is an intended flip — document it in your report).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_skill_resolver.py -q`
Expected: PASS (all, including flipped list-row assertions).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_skill_resolver.py Tests/Chat/test_console_skill_resolver.py
git commit -m "feat(skills): pure \$-mention scanner with code-span masking + \$name list rows"
```

---

### Task 2: Controller leading-`$` form (re-sigil the command branch)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — `_apply_skill_substitution` (def `:1727`; the leading-`/` check is at `:1780-1786`)
- Test: `Tests/Chat/test_console_skill_substitution.py` (flip)

**Interfaces:**
- Consumes: `MENTION_SIGIL` (Task 1).
- Produces: leading `$skill-name args…` behaves exactly as today's leading `/skill-name args…` (resolve via `resolve_skill_command` — case-insensitive/prefix — args capped via `cap_skill_args` → `{{args}}`; inline replaces final message; fork drops history except a leading system message; `SkillTrustBlockedError` → `(original_messages, SKILL_UNTRUSTED_REFUSE)`). Return signature UNCHANGED in this task (2-tuple; Task 3 widens it).

- [ ] **Step 1: Flip the pinning tests**

`Tests/Chat/test_console_skill_substitution.py` has 11 tests pinning the `/` sigil (messages like `"/code-review look at this"`). Change every invocation string from `/name…` to `$name…` (e.g. `"$code-review look at this"`). Do NOT weaken any assertion — only the sigil in message fixtures changes. Add one new test:

```python
@pytest.mark.asyncio
async def test_leading_slash_no_longer_invokes(monkeypatch):
    # Hard removal: a leading /code-review message passes through untouched.
    controller, skills = _make_controller()  # use the file's existing helper pattern
    messages = [{"role": "user", "content": "/code-review look at this"}]
    out, refuse = await controller._apply_skill_substitution(messages)
    assert out == messages
    assert refuse is None
    assert skills.executions == []
```

(Adapt the construction line to the file's existing controller-building helper — read the file first; every existing test constructs the controller the same way.)

- [ ] **Step 2: Run to verify failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_skill_substitution.py -q`
Expected: FAIL — the flipped `$` tests find no substitution (code still checks `COMMAND_PREFIX`), and the new `/`-passthrough test fails (old code still expands `/`).

- [ ] **Step 3: Implement**

In `_apply_skill_substitution`, import `MENTION_SIGIL` from `console_skill_resolver` (extend the existing import at the top of the file) and change the leading check (`:1780-1786`):

```python
        content = provider_messages[final_index].get("content")
        if not isinstance(content, str) or not content.startswith(MENTION_SIGIL):
            return provider_messages, None

        word, rest = _split_skill_command_word(content)
        name = word[len(MENTION_SIGIL) :]
        if not name:
            return provider_messages, None
```

Everything below (`resolve_skill_command` → `cap_skill_args` → `execute_skill` → inline/fork/refuse) stays byte-identical. Update the docstring's `COMMAND_PREFIX` reference to `MENTION_SIGIL`. If `COMMAND_PREFIX` is now unused in this file, remove its import.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_skill_substitution.py Tests/Chat/test_console_chat_controller.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_skill_substitution.py
git commit -m "feat(skills): leading \$skill-name form replaces /skill-name in payload substitution"
```

---

### Task 3: Controller embedded splice + non-aborting notes channel

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — `_apply_skill_substitution` + its FOUR call sites (grep `_apply_skill_substitution(` — currently `:465`, `:1143`, `:1201`, `:1265` region; re-grep, lines shift)
- Test: `Tests/Chat/test_console_skill_substitution.py` (extend)

**Interfaces:**
- Consumes: `find_embedded_mentions`, `SkillMention`, `SKILL_MENTION_SKIPPED_NOTE` (Task 1).
- Produces: `_apply_skill_substitution` returns `tuple[list[dict[str, Any]], str | None, tuple[str, ...]]` — `(messages, refuse, notes)`. `refuse` aborts the send (unchanged semantics); `notes` never abort — each is appended by the caller as a system row using the same mechanism the caller already uses for `refuse` copy, then the send proceeds.

- [ ] **Step 1: Write the failing tests** (append; adapt construction to the file's helper)

```python
@pytest.mark.asyncio
async def test_embedded_mention_splices_preserving_prose():
    controller, skills = _make_controller()  # existing helper pattern
    messages = [{"role": "user", "content": "summarize, $code-review it, then list"}]
    out, refuse, notes = await controller._apply_skill_substitution(messages)
    assert refuse is None and notes == ()
    content = out[0]["content"]
    assert content.startswith("summarize, RENDERED[")
    assert content.endswith(" it, then list")
    assert "$code-review" not in content
    # embedded is ARGLESS
    assert skills.executions == [("code-review", "")]


@pytest.mark.asyncio
async def test_embedded_fork_mention_left_literal():
    controller, skills = _make_controller(mode="fork")
    messages = [{"role": "user", "content": "please $code-review this"}]
    out, refuse, notes = await controller._apply_skill_substitution(messages)
    assert out == messages          # untouched
    assert refuse is None and notes == ()


@pytest.mark.asyncio
async def test_embedded_untrusted_left_literal_with_note():
    controller, skills = _make_controller(raise_trust=True)
    messages = [{"role": "user", "content": "please $code-review this"}]
    out, refuse, notes = await controller._apply_skill_substitution(messages)
    assert out == messages          # prose never lost
    assert refuse is None           # embedded never aborts
    assert len(notes) == 1 and "code-review" in notes[0]


@pytest.mark.asyncio
async def test_leading_resolved_mention_does_not_scan_args():
    controller, skills = _make_controller()
    messages = [{"role": "user", "content": "$code-review also $code-review"}]
    out, refuse, notes = await controller._apply_skill_substitution(messages)
    # Leading form: ONE execution with the rest as args; no embedded pass.
    assert skills.executions == [("code-review", "also $code-review")]


@pytest.mark.asyncio
async def test_leading_unresolved_falls_through_to_embedded():
    controller, skills = _make_controller()
    messages = [{"role": "user", "content": "$notaskill but $code-review works"}]
    out, refuse, notes = await controller._apply_skill_substitution(messages)
    content = out[0]["content"]
    assert content.startswith("$notaskill but RENDERED[")
    assert skills.executions == [("code-review", "")]
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_skill_substitution.py -q`
Expected: FAIL — `ValueError: not enough values to unpack` (2-tuple return) and no embedded expansion.

- [ ] **Step 3: Implement**

Restructure `_apply_skill_substitution` (keep the guards through `final_index`/`content` as in Task 2), then:

```python
        candidates_context = await self._skills_service.get_context(mode="local")
        candidates = self._skill_candidates_from_context(candidates_context)

        # --- Leading form: message starts with a resolvable $skill-name.
        if content.startswith(MENTION_SIGIL):
            word, rest = _split_skill_command_word(content)
            name = word[len(MENTION_SIGIL) :]
            if name:
                resolution = resolve_skill_command(name, rest, candidates)
                if resolution.kind == "resolved":
                    args = cap_skill_args(rest)
                    try:
                        result = await self._skills_service.execute_skill(
                            resolution.name, mode="local", args=args
                        )
                    except SkillTrustBlockedError as exc:
                        refuse = SKILL_UNTRUSTED_REFUSE.format(
                            name=resolution.name, reason=exc.reason_code
                        )
                        return provider_messages, refuse, ()
                    # ... existing rendered/fork/inline blocks, each return
                    # gaining a trailing empty notes tuple: `, ()`.

        # --- Embedded pass: leading absent or unresolved.
        names = frozenset(candidate.name for candidate in candidates)
        mentions = find_embedded_mentions(content, names)
        if not mentions:
            return provider_messages, None, ()

        rendered_by_name: dict[str, str | None] = {}
        notes: list[str] = []
        for mention in mentions:
            if mention.name in rendered_by_name:
                continue
            try:
                result = await self._skills_service.execute_skill(
                    mention.name, mode="local", args=""
                )
            except SkillTrustBlockedError:
                rendered_by_name[mention.name] = None
                notes.append(SKILL_MENTION_SKIPPED_NOTE.format(name=mention.name))
                continue
            execution_mode = (
                result.get("execution_mode") if isinstance(result, Mapping) else None
            )
            rendered = (
                result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
            )
            # Fork (or anything non-inline) cannot splice: leave literal, no note.
            rendered_by_name[mention.name] = (
                rendered if execution_mode == "inline" else None
            )

        new_content = content
        for mention in reversed(mentions):
            body = rendered_by_name.get(mention.name)
            if body is None:
                continue
            new_content = (
                new_content[: mention.start] + body + new_content[mention.end :]
            )
        if new_content == content:
            return provider_messages, None, tuple(notes)
        new_messages = list(provider_messages)
        new_messages[final_index] = {
            "role": ConsoleMessageRole.USER.value,
            "content": new_content,
        }
        return new_messages, None, tuple(notes)
```

Update the docstring (leading + embedded semantics, 3-tuple contract). Then update **all four call sites** (grep `= await self._apply_skill_substitution(`): the unpack becomes `provider_messages, refuse, skill_notes = ...`; immediately after each site's existing `refuse` handling, append each note in `skill_notes` as a system row using the SAME mechanism that site uses for `refuse` copy — but do NOT abort for notes (the send continues). The skip-skills site (comment "may execute skills with side effects") is untouched. Import `find_embedded_mentions` and `SKILL_MENTION_SKIPPED_NOTE` alongside the existing resolver imports.

- [ ] **Step 4: Run to verify pass + regression**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/ -q`
Expected: PASS (all Chat suites — the 4 call-site updates must not break run-turn tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_skill_substitution.py
git commit -m "feat(skills): embedded \$-mention splicing with non-aborting skipped-skill notes"
```

---

### Task 4: Hard-remove `/` invocation (composer layer) + `/skills` run form

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (fallback registration `:1865-1867`, import `:100`, `_console_command_skills` handler + pick-submit path — grep `_submit` near the skills handler; blocked-hint copy); `tldw_chatbook/Chat/console_skill_resolver.py` (DELETE `make_skill_fallback_resolver`)
- Test: `Tests/UI/test_console_skill_commands.py`, `Tests/Chat/test_console_skill_resolver.py`, `Tests/Chat/test_console_command_grammar.py` (flip/delete)

**Interfaces:**
- Consumes: nothing new.
- Produces: no fallback claiming — a `/former-skill` draft parses `KIND_UNKNOWN` (grammar hint → armed second-Enter literal send, existing behavior); `/skills` bare → list (unchanged); `/skills <name>…` → a transcript hint row with EXACTLY this copy: `Run skills by typing $<name> — /skills only lists them.` (substituting the typed name); the skill-picker submit path composes `$name [args]` instead of `/name [args]`.

- [ ] **Step 1: Write/flip the failing tests**

In `Tests/UI/test_console_skill_commands.py`: flip every submit fixture from `/name…` to `$name…` where the test exercises invocation-through-submit; change the `/skills <name>` run-form tests to assert the new hint copy (`"Run skills by typing $"` appears; no skill executed); assert the pick-submit path produces a draft/message starting with `$`. In `Tests/Chat/test_console_skill_resolver.py`: DELETE the `make_skill_fallback_resolver` tests (the factory is being deleted). In `Tests/Chat/test_console_command_grammar.py`: any test registering the skill fallback flips to asserting `/former-skill` → `KIND_UNKNOWN`.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_skill_commands.py Tests/Chat/test_console_skill_resolver.py Tests/Chat/test_console_command_grammar.py -q`
Expected: FAIL — old `/` behavior still present.

- [ ] **Step 3: Implement**

1. Delete the registration block at `chat_screen.py:1865-1867` (`self._console_command_registry.register_fallback_resolver(make_skill_fallback_resolver(...))`) and the `make_skill_fallback_resolver` import (`:100`). The `_console_skill_candidates` snapshot itself STAYS (the blocked-hint path and `/skills` list still use it).
2. Delete `make_skill_fallback_resolver` from `console_skill_resolver.py` (the whole factory; `KIND_FALLBACK` import goes with it if now unused there).
3. In the `/skills` handler (`_console_command_skills`, dispatch map near `:10879`): keep the bare-args list branch; replace the `<name> [args]` run branch with a transcript hint row: `f"Run skills by typing ${name} — /skills only lists them."` (reuse the same transcript-row mechanism the list response uses).
4. The skill-picker submit path (the method that submits a raw `/name [args]` command as the user turn — grep `"/"` composition near the skills handler): compose `f"${name}"` / `f"${name} {args}"` instead.
5. Blocked-hint copy (`_console_skill_blocked_match_response` region): if its copy teaches `/skills <name>` as the run form, update to the `$name` convention. The hint itself STAYS (it lives in the `KIND_UNKNOWN` branch, independent of the fallback resolver).

- [ ] **Step 4: Run to verify pass**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_skill_commands.py Tests/Chat/test_console_skill_resolver.py Tests/Chat/test_console_command_grammar.py Tests/UI/test_console_command_composer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_skill_resolver.py Tests/UI/test_console_skill_commands.py Tests/Chat/test_console_skill_resolver.py Tests/Chat/test_console_command_grammar.py
git commit -m "feat(skills)!: hard-remove /skill-name invocation; /skills runs point to \$name"
```

---

### Task 5: Copy sweep + full regression

**Files:**
- Modify: any file the sweep finds still teaching `/skill-name` invocation (user-visible copy/docstrings only — NOT historical spec docs)
- Test: full affected suites

- [ ] **Step 1: Sweep for stale invocation copy**

Run: `grep -rn '"/skills <name>\|/skill-name\|typed.*"/"\|COMMAND_PREFIX' tldw_chatbook/Chat/ tldw_chatbook/UI/Screens/chat_screen.py | grep -vi "test"` and review each hit: user-visible copy or docstrings that teach the removed `/name` run convention flip to `$name`; grammar/registry internals that legitimately still use `/` for registered commands (`/skills`, `/prompt`, …) stay.

- [ ] **Step 2: Full regression**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/ Tests/UI/test_console_skill_commands.py Tests/UI/test_console_command_composer.py Tests/Skills/ -q`
Expected: 0 failed (modulo the known flaky, re-run in isolation).

- [ ] **Step 3: Commit** (only if the sweep changed files)

```bash
git add <exact files the sweep changed>
git commit -m "chore(skills): sweep remaining /skill-name invocation copy to \$name"
```

---

## Notes for the executor

- The spec governs on any ambiguity; its §2 safety rules (case-sensitive embedded match, code-span skip, no recursion, fork-embedded literal) are load-bearing.
- Line numbers in this plan were verified on branch `worktree-skills-dollar-mention` at `184d260f3` — re-grep before editing; they shift.
- `console_agent_bridge` (`_BridgeSkillRunner`) is UNTOUCHED — model-invoked skills are a different mechanism.
- The stored-transcript invariant: raw user text (including `$mentions`) is persisted; only the ephemeral payload renders.
