# Image Generation — Console Card with Variants (P2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/generate-image` in the Console chat produces a persistent assistant message rendered as a bordered "Image Generation" card (image + Style/Source/Seed/Prompt/Negative) with variant `◀ n/N ▶` browsing, Keep (canonical), and Regenerate-appends — backed by a v24→v25 sidecar table and narrow, crash-safe persistence operations.

**Architecture:** Variant bytes ride the existing attachments contract (position 0 = messages-row scalar mirror, ≥1 = `message_attachments`); per-variant metadata lives in a new `message_generation_metadata` sidecar keyed `(message_id, position)`; selected/kept variant IS position 0. Keep = targeted position swap; regenerate = single-INSERT append — **never** the full-list `update_message_content(attachments=...)` rewrite (data-loss footgun: in-memory bytes may be `None` and the contract overwrites). The card is a new transcript row kind mirroring the image-row pattern.

**Tech Stack:** Python ≥3.11, Textual, SQLite (ChaChaNotes), the Phase-1 `Image_Generation` engine (`worker.build_request`/`run_generation`, `listing`), rich-pixels/textual-image, loguru.

**Design spec (read first):** `Docs/superpowers/specs/2026-07-23-image-gen-console-card-variants-design.md`

## Global Constraints

- **Worktree:** ALL work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-chat-card`, branch `claude/image-gen-chat-card-phase2`. Subagents start in the MAIN checkout — `cd` into the worktree as the first action; never touch the main checkout's `tldw_chatbook/`.
- **Test command** (worktree has NO local `.venv`; use the MAIN checkout venv with cwd=worktree so worktree source wins):
  `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-chat-card && python -m pytest <paths> -v`
- **Git hygiene:** the repo tracks a stale `.superpowers/sdd/progress.md` — `git add` ONLY your own files by explicit path; never `git add -A`/`.`/`-am`.
- **Test hygiene:** `ruff check` your new/modified files before committing (no F401/E401/E702); no unused fixtures.
- **Schema:** current `_CURRENT_SCHEMA_VERSION = 24` (`tldw_chatbook/DB/ChaChaNotes_DB.py:160`). Task 1 claims **v25** — RE-VERIFY at execution time that no other ref has claimed v25 (`git grep "_CURRENT_SCHEMA_VERSION = 25\|v24_to_v25" $(git for-each-ref --format='%(refname)') -- tldw_chatbook/DB/` from the main checkout must be empty). If claimed, renumber to the next free version and adjust every literal in Task 1.
- **Store invariant:** every in-memory attachments mutation flows through `ConsoleChatStore._set_message_attachments` (`Chat/console_chat_store.py:684` — re-bases positions from 0, mirrors `attachments[0]` into the scalar image fields).
- **NEVER use `update_message_content(attachments=[...])` for keep/append** — narrow ops only (spec §5.2; the full rewrite NULLs stored bytes when in-memory `MessageAttachment.data` is `None`).
- `run_generation` is BLOCKING — only ever call it on a thread worker (`group="imagegen-console"`, `exclusive=False` + in-flight guards; `exclusive=True` would cancel a running generation).
- `loguru` is the app logger. Parameterized SQL only. Type hints + Google docstrings on public APIs.
- Existing suites that must stay green: `Tests/ChaChaNotesDB/`, `Tests/Chat/`, `Tests/Image_Generation/`.

## File structure

```
tldw_chatbook/DB/ChaChaNotes_DB.py                     # MODIFY: v25 migration + sidecar CRUD + narrow ops
tldw_chatbook/DB/migrations/chachanotes_v24_to_v25_message_generation_metadata.sql  # NEW: DDL mirror
tldw_chatbook/Chat/chat_persistence_service.py          # MODIFY: create_message generation_metadata kwarg + 2 wrappers
tldw_chatbook/Image_Generation/config.py                # MODIFY: default_batch + max_variants_per_message
tldw_chatbook/config.py                                 # MODIFY: template keys
tldw_chatbook/Chat/console_generate_image.py            # NEW: pure parse helper + prompt-excerpt helper
tldw_chatbook/Chat/console_chat_models.py               # MODIFY: GenerationVariantMeta + message field
tldw_chatbook/Chat/console_chat_store.py                # MODIFY: create/append/keep/hydrate generation APIs
tldw_chatbook/Chat/console_command_grammar.py           # MODIFY: register /generate-image
tldw_chatbook/Chat/console_message_actions.py           # MODIFY: generation-variant gating + keep action
tldw_chatbook/Widgets/Console/console_generation_card.py # NEW: the card widget
tldw_chatbook/Widgets/Console/console_transcript.py     # MODIFY: generation-card row kind + suppression
tldw_chatbook/UI/Screens/chat_screen.py                 # MODIFY: dispatch entries + handler + card specs + actions
Tests/ChaChaNotesDB/test_message_generation_metadata.py # NEW
Tests/Chat/test_console_generate_image.py               # NEW (parse + flow)
Tests/Chat/test_console_generation_store.py             # NEW (store APIs)
Tests/Chat/test_console_generation_card.py              # NEW (renderer/actions)
```

---

### Task 1: Schema v24→v25 — `message_generation_metadata` sidecar + CRUD

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (`:160` version, `~:2489` SQL consts region, `~:3861` runners region, `:4020-4041` steps dict)
- Create: `tldw_chatbook/DB/migrations/chachanotes_v24_to_v25_message_generation_metadata.sql`
- Test: `Tests/ChaChaNotesDB/test_message_generation_metadata.py`

**Interfaces:**
- Produces (on `CharactersRAGDB`):
  - `set_message_generation_metadata(self, message_id: str, rows: list[dict]) -> None` — authoritative full rewrite for one message (DELETE then INSERT, one transaction; mirrors `set_message_attachments`, `ChaChaNotes_DB.py:7220`). Each row dict: `position:int(>=0), prompt:str, negative_prompt:str, backend:str, model:str|None, seed:int|None, style:str|None, params_json:str`.
  - `get_generation_metadata_for_messages(self, message_ids: Sequence[str]) -> dict[str, list[dict]]` — batch fetch, rows ordered by position (mirrors the batch-attachments query shape near `ChaChaNotes_DB.py:7260`).

- [ ] **Step 0: Re-verify v25 is free** (Global Constraints command). Record the result in your report.
- [ ] **Step 1: Write the failing tests**

```python
# Tests/ChaChaNotesDB/test_message_generation_metadata.py
import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(tmp_path / "t.db", "test-client")
    yield d
    d.close_connection()

def _mk_message(db):
    conv_id = db.add_conversation({"title": "t"})
    return conv_id, db.add_message({
        "conversation_id": conv_id, "sender": "assistant",
        "content": "[image] a red dragon",
        "image_data": b"png0", "image_mime_type": "image/png",
    })

def _row(pos, seed=1):
    return {"position": pos, "prompt": "a red dragon", "negative_prompt": "blurry",
            "backend": "swarmui", "model": None, "seed": seed, "style": None,
            "params_json": "{}"}

def test_migration_reaches_v25(db):
    with db.transaction() as cur:
        v = cur.execute(
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            ("rag_char_chat_schema",)).fetchone()[0]
    assert v == 25

def test_set_and_batch_get_roundtrip(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0, seed=7), _row(1, seed=8)])
    got = db.get_generation_metadata_for_messages([mid])
    assert [r["seed"] for r in got[mid]] == [7, 8]
    assert got[mid][0]["backend"] == "swarmui"

def test_set_is_authoritative_rewrite(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0), _row(1)])
    db.set_message_generation_metadata(mid, [_row(0, seed=99)])
    got = db.get_generation_metadata_for_messages([mid])
    assert len(got[mid]) == 1 and got[mid][0]["seed"] == 99

def test_cascade_delete_with_message(db):
    conv_id, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0)])
    with db.transaction() as cur:  # hard-delete to exercise ON DELETE CASCADE
        cur.execute("DELETE FROM messages WHERE id=?", (mid,))
    assert db.get_generation_metadata_for_messages([mid]) == {}
```

> If `add_conversation`/`add_message` signatures differ (read them: `ChaChaNotes_DB.py:7075` region), align the fixture calls minimally — assertions stay.

- [ ] **Step 2: Run → FAIL** (`no such table` / missing methods / version 24).
  Run: `python -m pytest Tests/ChaChaNotesDB/test_message_generation_metadata.py -v`
- [ ] **Step 3: Implement the migration**

Add the SQL const (next to `_MIGRATE_V23_TO_V24_SQL`, `~:2489`):

```python
    # tldw_chatbook/DB/migrations/chachanotes_v24_to_v25_message_generation_metadata.sql
    _MIGRATE_V24_TO_V25_SQL = """
CREATE TABLE IF NOT EXISTS message_generation_metadata(
  message_id      TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position        INTEGER NOT NULL CHECK (position >= 0),
  prompt          TEXT    NOT NULL,
  negative_prompt TEXT    NOT NULL DEFAULT '',
  backend         TEXT    NOT NULL,
  model           TEXT,
  seed            INTEGER,
  style           TEXT,
  params_json     TEXT    NOT NULL DEFAULT '{}',
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, position)
);
CREATE INDEX IF NOT EXISTS idx_msg_gen_meta_message ON message_generation_metadata(message_id);
UPDATE db_schema_version
   SET version = 25
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 24;
"""
```

Runner `_migrate_from_v24_to_v25(self, conn)` — copy the `_migrate_from_v23_to_v24` shape verbatim (`:3861-3886`: try/executescript/verify `_get_db_version(conn)==25` else `SchemaError`, same logging; no PRAGMA column-guard needed — `CREATE TABLE IF NOT EXISTS` is the guard). Register `24: self._migrate_from_v24_to_v25` in `migration_steps` (`:4020-4041`), bump `_CURRENT_SCHEMA_VERSION = 25` (`:160`) with a comment `# Adds message_generation_metadata sidecar (image-gen P2a). NOT sync-integrated (deliberate, v19/v24 precedent); prompts NOT in FTS.`. **Also add the sidecar DDL to the base-schema CREATE block next to `message_attachments`** (`~:2404` region pattern — follow how v19's table appears for fresh DBs; read how `message_attachments` is present in the fresh-create path and mirror it). Write the DDL mirror file.

- [ ] **Step 4: Implement the CRUD** (next to `set_message_attachments`, `:7220`): `set_message_generation_metadata` = validate positions unique/≥0, one `transaction()`: `DELETE FROM message_generation_metadata WHERE message_id=?` then executemany INSERT. `get_generation_metadata_for_messages` = chunked `IN (...)` SELECT ordered by `message_id, position`, return dict of row-dicts (omit `created_at`).
- [ ] **Step 5: Run → PASS**, then regression: `python -m pytest Tests/ChaChaNotesDB/ -q` (all green, incl. fresh-create + migration property tests).
- [ ] **Step 6: Commit** `feat(db): v24->v25 message_generation_metadata sidecar + CRUD (image-gen P2a)` (stage only the 3 files).

---

### Task 2: Narrow persistence ops — append variant + keep swap + atomic create

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (new ops near `set_message_attachments`)
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (wrappers + `create_message` kwarg)
- Test: `Tests/ChaChaNotesDB/test_message_generation_metadata.py` (extend) + `Tests/Chat/test_chat_persistence_service.py` (extend)

**Interfaces:**
- `CharactersRAGDB.append_message_attachment_with_metadata(self, message_id: str, *, data: bytes, mime_type: str, display_name: str = "", generation_metadata: dict | None = None) -> int` — computes next position (`1 + max(table positions)` or `1`; requires the message row to exist with a position-0 image, else `ValueError`), INSERTs the attachment row and (if given) the sidecar row **in one transaction**, bumps the message row's `version`/`last_modified` following the existing message-update idiom (read how `set_message_attachments` / message updates handle version — mirror it). Returns the new position.
- `CharactersRAGDB.swap_message_attachment_with_scalar(self, message_id: str, position: int) -> None` — one transaction: read scalar (`messages.image_data/image_mime_type`) + table row at `position`; write table-row bytes into the scalar and scalar bytes into the table row; swap the two sidecar rows' `position` keys (0 ↔ `position`, via a temp position to dodge the PK); bump message `version`/`last_modified`. `ValueError` if `position < 1` or row missing. **Only the two affected variants' bytes are read/written.**
- `ChatPersistenceService.create_message(..., generation_metadata: Optional[Sequence[Mapping]] = None)` — when supplied, writes sidecar rows **inside the same transaction** as the row insert + attachments write (extend the existing transaction body, `chat_persistence_service.py:439`; sidecar write failure rolls back everything).
- `ChatPersistenceService.append_message_attachment(message_id, *, data, mime_type, display_name="", generation_metadata=None) -> int` and `ChatPersistenceService.keep_message_attachment(message_id, position) -> None` — thin wrappers.

- [ ] **Step 1: Write the failing tests** (extend `Tests/ChaChaNotesDB/test_message_generation_metadata.py`)

```python
def test_append_attachment_with_metadata_single_insert(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0)])
    pos = db.append_message_attachment_with_metadata(
        mid, data=b"png1", mime_type="image/png", generation_metadata=_row(1, seed=42))
    assert pos == 1
    got = db.get_generation_metadata_for_messages([mid])
    assert [r["position"] for r in got[mid]] == [0, 1]
    with db.transaction() as cur:
        row = cur.execute(
            "SELECT data FROM message_attachments WHERE message_id=? AND position=1",
            (mid,)).fetchone()
    assert row[0] == b"png1"

def test_swap_makes_kept_variant_position_zero(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0, seed=7)])
    db.append_message_attachment_with_metadata(
        mid, data=b"png1", mime_type="image/png", generation_metadata=_row(1, seed=42))
    db.swap_message_attachment_with_scalar(mid, 1)
    msg = db.get_message_by_id(mid)          # read helper: align name to real API
    assert msg["image_data"] == b"png1"       # kept variant now canonical
    with db.transaction() as cur:
        row = cur.execute(
            "SELECT data FROM message_attachments WHERE message_id=? AND position=1",
            (mid,)).fetchone()
    assert row[0] == b"png0"                  # old canonical demoted, bit-identical
    got = db.get_generation_metadata_for_messages([mid])
    by_pos = {r["position"]: r["seed"] for r in got[mid]}
    assert by_pos == {0: 42, 1: 7}            # sidecar re-keyed with the swap

def test_swap_rejects_bad_position(db):
    _, mid = _mk_message(db)
    with pytest.raises(ValueError):
        db.swap_message_attachment_with_scalar(mid, 0)
    with pytest.raises(ValueError):
        db.swap_message_attachment_with_scalar(mid, 3)   # no such row
```

And in `Tests/Chat/test_chat_persistence_service.py` (follow that file's existing fixture style — it uses a real in-memory DB):

```python
def test_create_message_with_generation_metadata_atomic(persistence, db):
    conv_id = db.add_conversation({"title": "t"})
    mid = persistence.create_message(
        conversation_id=conv_id, sender="assistant", content="[image] x",
        attachments=[{"position": 0, "data": b"a", "mime_type": "image/png"},
                     {"position": 1, "data": b"b", "mime_type": "image/png"}],
        generation_metadata=[
            {"position": 0, "prompt": "x", "negative_prompt": "", "backend": "swarmui",
             "model": None, "seed": 1, "style": None, "params_json": "{}"},
            {"position": 1, "prompt": "x", "negative_prompt": "", "backend": "swarmui",
             "model": None, "seed": 2, "style": None, "params_json": "{}"},
        ])
    got = db.get_generation_metadata_for_messages([mid])
    assert [r["seed"] for r in got[mid]] == [1, 2]
```

> Align fixture names/read-helper (`get_message_by_id`) to the real APIs in those test files — read them first. Assertions stay.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** per the Interfaces block. Read the existing message-update version-bump idiom before writing the swap (grep `version = version + 1` / `last_modified` updates in `ChaChaNotes_DB.py`); mirror it exactly. The sidecar 0↔k swap uses a three-step temp position (e.g. `-1` is blocked by the CHECK — use a large sentinel like `1000000`) inside the same transaction.
- [ ] **Step 4: Run → PASS**, then `python -m pytest Tests/ChaChaNotesDB/ Tests/Chat/test_chat_persistence_service.py -q` all green.
- [ ] **Step 5: Commit** `feat(db): narrow attachment ops (append+swap) + atomic generation-metadata create (P2a)`.

---

### Task 3: Config — `default_batch` + `max_variants_per_message`

**Files:**
- Modify: `tldw_chatbook/Image_Generation/config.py`, `tldw_chatbook/config.py`
- Test: `Tests/Image_Generation/test_config_loader.py` (extend)

**Interfaces:**
- `ImageGenerationConfig.default_batch: int` (default **1**, clamped ≥1) and `.max_variants_per_message: int` (default **8**, clamped ≥1) — new `DEFAULT_IMAGE_BATCH = 1`, `DEFAULT_MAX_VARIANTS_PER_MESSAGE = 8` constants; keys added to `_GLOBAL_KEYS`; dataclass fields; builder reads with `_coerce_int` + clamp (mirror how `inline_max_bytes`/poll intervals clamp).

- [ ] **Step 1: Failing tests** (extend the existing file, reuse its `_reset_cache` fixture + `_read_image_generation_toml`/`_keyring_get` patch style):

```python
def test_batch_and_variant_cap_defaults(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.default_batch == 1 and cfg.max_variants_per_message == 8

def test_batch_and_variant_cap_from_toml_clamped(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"default_batch": 3, "max_variants_per_message": 0}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.default_batch == 3 and cfg.max_variants_per_message == 1  # clamped >=1
```

- [ ] **Step 2: RED. Step 3:** implement + add both keys to the `[image_generation]` template block in `tldw_chatbook/config.py` (next to `max_prompt_length`). **Step 4: GREEN** + `python -m pytest Tests/Image_Generation/ -q` all green. **Step 5: Commit** `feat(imagegen): default_batch + max_variants_per_message config (P2a)`.

---

### Task 4: Pure helpers — `parse_generate_image_args` + content marker

**Files:**
- Create: `tldw_chatbook/Chat/console_generate_image.py`
- Test: `Tests/Chat/test_console_generate_image.py`

**Interfaces:**
- `parse_generate_image_args(args: str) -> GenerateImageArgs` where `GenerateImageArgs` is a frozen dataclass `(backend: str | None, prompt: str)`. Leading `:backend` token (`:swarmui a dragon` → `("swarmui", "a dragon")`); no colon → `(None, args.strip())`; `:backend` alone or empty prompt → `prompt=""` (caller refuses).
- `generation_content_marker(prompt: str, limit: int = 80) -> str` → `"[image] <prompt trimmed to limit>"` (ellipsis when trimmed).

- [ ] **Step 1: Failing tests**

```python
# Tests/Chat/test_console_generate_image.py
import pytest
from tldw_chatbook.Chat.console_generate_image import (
    parse_generate_image_args, generation_content_marker,
)

@pytest.mark.parametrize("args,backend,prompt", [
    ("a red dragon", None, "a red dragon"),
    (":swarmui a red dragon", "swarmui", "a red dragon"),
    (":openrouter   spaced  prompt ", "openrouter", "spaced  prompt"),
    (":swarmui", "swarmui", ""),
    ("", None, ""),
    ("   ", None, ""),
    (": lonely colon", None, ": lonely colon"),  # bare ':' is not a backend token
])
def test_parse_table(args, backend, prompt):
    parsed = parse_generate_image_args(args)
    assert (parsed.backend, parsed.prompt) == (backend, prompt)

def test_content_marker_trims():
    assert generation_content_marker("a red dragon") == "[image] a red dragon"
    long = "x" * 200
    marker = generation_content_marker(long)
    assert marker.startswith("[image] ") and len(marker) <= 8 + 80 + 1 and marker.endswith("…")
```

- [ ] **Step 2: RED. Step 3:** implement (mirror `console_prefill.py`'s module style — pure, no Textual imports). **Step 4: GREEN. Step 5: Commit** `feat(console): /generate-image arg parsing + content marker helpers (P2a)`.

---

### Task 5: Store — generation model, atomic create, append, keep, hydration

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (near `MessageAttachment`, `:188`)
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_generation_store.py`

**Interfaces:**
- `console_chat_models.GenerationVariantMeta` — frozen dataclass: `prompt: str, negative_prompt: str, backend: str, model: str | None, seed: int | None, style: str | None, params: dict[str, Any]` (+ `to_row(position) -> dict` / `from_row(dict) -> GenerationVariantMeta` converters handling `params_json`).
- `ConsoleChatMessage.generation_metadata: tuple[GenerationVariantMeta, ...] = ()` — empty tuple ⇔ not a generation message. Index i ↔ attachment position i.
- Store methods (all keep the in-memory tuple ordering aligned with attachment positions via `_set_message_attachments`):
  - `append_generation_message(self, session_id, *, content: str, variants: Sequence[tuple[bytes, str, GenerationVariantMeta]], persist: bool = False) -> ConsoleChatMessage` — builds attachments 0..N-1 + metadata tuple; persists via `create_message(attachments=..., generation_metadata=...)` (atomic; bytes are fresh — the authoritative list is safe HERE and only here).
  - `append_generation_variant(self, session_id, message_id, *, data: bytes, mime_type: str, meta: GenerationVariantMeta, persist: bool = True) -> int` — probes `persistence.append_message_attachment` (getattr-optional, like `get_attachments_for_messages`, `console_chat_store.py:139`), appends to the in-memory attachments tuple through `_set_message_attachments` and to `generation_metadata`. Returns new position.
  - `keep_generation_variant(self, session_id, message_id, *, position: int, persist: bool = True) -> None` — reorders the in-memory attachments tuple (kept → index 0 by swap) + `generation_metadata` in lockstep; persists via probed `persistence.keep_message_attachment`. **If the in-memory bytes for either affected variant are `None`, the in-memory tuple still swaps (bytes stay `None`)** — persistence reads bytes from the DB, so nothing is lost (the spec's footgun scenario).
  - `hydrate_generation_metadata(self, session_id, rows_by_message: Mapping[str, Sequence[dict]]) -> None` — populates messages' `generation_metadata` from DB rows at conversation load (caller batch-fetches via probed `persistence.get_generation_metadata_for_messages`... wrapper added on the persistence service delegating to the Task-1 DB API).

- [ ] **Step 1: Failing tests**

```python
# Tests/Chat/test_console_generation_store.py
# Follow Tests/Chat/test_console_chat_store.py's existing fixture/fake-persistence style — read it first.
import pytest
from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta

def _meta(seed):
    return GenerationVariantMeta(prompt="p", negative_prompt="", backend="swarmui",
                                 model=None, seed=seed, style=None, params={})

def test_append_generation_message_sets_metadata_and_mirror(store_with_session):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="[image] p",
        variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))])
    assert msg.image_data == b"a"                      # position-0 mirror
    assert [m.seed for m in msg.generation_metadata] == [1, 2]
    assert [a.position for a in msg.attachments] == [0, 1]

def test_keep_swaps_in_memory_and_calls_persistence(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="c", variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
        persist=True)
    store.keep_generation_variant(sid, msg.id, position=1)
    assert msg.image_data == b"b" and [m.seed for m in msg.generation_metadata] == [2, 1]
    assert fake_persistence.kept == [(msg.id, 1)]      # persistence op invoked

def test_keep_with_byteless_memory_does_not_null(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="c", variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
        persist=True)
    # simulate a rehydrated-without-bytes message (bytes|None contract)
    from dataclasses import replace
    store._set_message_attachments(msg, tuple(replace(a, data=None) for a in msg.attachments))
    store.keep_generation_variant(sid, msg.id, position=1)
    assert fake_persistence.kept == [(msg.id, 1)]      # narrow op used; NO full-list rewrite call
    assert not getattr(fake_persistence, "full_rewrites", [])

def test_append_variant_respects_positions(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="c", variants=[(b"a", "image/png", _meta(1))], persist=True)
    pos = store.append_generation_variant(sid, msg.id, data=b"b", mime_type="image/png",
                                          meta=_meta(2))
    assert pos == 1 and msg.attachments[1].data == b"b"
    assert [m.seed for m in msg.generation_metadata] == [1, 2]
```

The fake persistence gains `append_message_attachment` (records + returns next position), `keep_message_attachment` (records into `.kept`), and asserts-by-absence for full rewrites (record any `update_message_content(attachments=...)` call into `.full_rewrites`).

- [ ] **Step 2: RED. Step 3:** implement per Interfaces (read the real fake-persistence class in the existing test file and extend it there or in a shared helper — match local convention). **Step 4: GREEN** + `python -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_generation_store.py -q` green. **Step 5: Commit** `feat(console): store generation-variant model + atomic create/append/keep/hydrate (P2a)`.

---

### Task 6: Command wiring + generation flow (chat_screen)

**Files:**
- Modify: `tldw_chatbook/Chat/console_command_grammar.py` (constants + `default_console_registry()`, `:166`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` `:11172`, `dispatch_map` `:11205`, new handler near `_console_command_prefill` `:11476`)
- Test: `Tests/Chat/test_console_generate_image.py` (extend with flow tests)

**Interfaces:**
- Grammar: `GENERATE_IMAGE_COMMAND_NAME = "generate-image"`, `GENERATE_IMAGE_COMMAND_HANDLER_ID = "generate-image"`, registered `ConsoleCommand` with help text `"Generate an image: /generate-image [:backend] <prompt>"`.
- `async def _console_command_generate_image(self, parse: CommandParse) -> None` on ChatScreen:
  1. `parse_generate_image_args(parse.args)`; empty prompt → status line with usage, return.
  2. Resolve backend: explicit or `cfg.default_backend`; refuse (status line) when unresolvable or not `is_configured` (via `listing.list_image_models_for_catalog()`).
  3. In-flight guard: `self._imagegen_inflight_sessions: set[str]` — refuse a second command for the session while one runs.
  4. Clear the composer draft (mirror how `/prefill` consumes the draft).
  5. `self.run_worker(self._run_generation_batch(...), group="imagegen-console", exclusive=False)`... **read how the screen runs thread work** — `run_generation` is sync/blocking, so the batch loop runs via `asyncio.to_thread` inside an async worker OR a `@work(thread=True)` method (mirror `_prep_console_images` `:3135` / the demo screen). Batch loop: for i in range(default_batch): `build_request` (variant requests after the first force `seed=-1` when an explicit seed was configured — identical-image guard) → `run_generation` → collect `(bytes, mime, GenerationVariantMeta)`; per-variant failures collected.
  6. Back on the UI loop: k ≥ 1 successes → `store.append_generation_message(..., content=generation_content_marker(prompt), variants=successes, persist=True)`, refresh transcript + card specs, status `"k/N generated"` when partial; k = 0 → status/system line with the error, **no message**.
- `GenerationVariantMeta.seed` records the *requested* seed (engine results don't return the backend's actual seed — display "random" for `-1`/None in the card; noted spec drift, acceptable for P2a).

- [ ] **Step 1: Failing tests** — pure-logic level (the screen method's core is factored so it's testable without a full app):

```python
# extend Tests/Chat/test_console_generate_image.py
# Factor the batch loop into a pure-ish helper in console_generate_image.py so it's testable:
#   run_generation_batch(*, backend, prompt, negative_prompt, seed, count,
#                        generate=worker.run_generation, build=worker.build_request)
#       -> BatchResult(successes: list[tuple[bytes, str, GenerationVariantMeta]],
#                      errors: list[str])
from tldw_chatbook.Chat.console_generate_image import run_generation_batch

class _Res:
    def __init__(self, b): self.content = b; self.content_type = "image/png"; self.bytes_len = len(b)

def test_batch_all_succeed():
    calls = []
    def gen(req): calls.append(req); return _Res(b"img")
    out = run_generation_batch(backend="swarmui", prompt="p", negative_prompt=None,
                               seed=None, count=2, generate=gen)
    assert len(out.successes) == 2 and out.errors == []

def test_batch_partial_failure_keeps_successes():
    n = {"i": 0}
    def gen(req):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("boom")
        return _Res(b"img")
    out = run_generation_batch(backend="swarmui", prompt="p", negative_prompt=None,
                               seed=None, count=3, generate=gen)
    assert len(out.successes) == 2 and len(out.errors) == 1

def test_batch_explicit_seed_only_first_variant():
    seeds = []
    def gen(req): seeds.append(req.seed); return _Res(b"img")
    run_generation_batch(backend="swarmui", prompt="p", negative_prompt=None,
                         seed=1234, count=3, generate=gen)
    assert seeds == [1234, -1, -1]      # identical-image guard
```

- [ ] **Step 2: RED. Step 3:** implement `run_generation_batch` in `console_generate_image.py` (blocking; the screen calls it via thread), then the grammar registration + 3 dispatch edits + the screen handler per Interfaces (read `_console_command_prefill` for draft-consume/status-line idioms; read the demo screen for the thread pattern). **Step 4: GREEN** + `python -m pytest Tests/Chat/test_console_generate_image.py -q`; also `python -c "import tldw_chatbook.app"` clean. **Step 5: Commit** `feat(console): /generate-image command + generation batch flow (P2a)`.

---

### Task 7: Card renderer — transcript row + widget + spec build + image-row suppression

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_generation_card.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`_TranscriptRow` `:205`, spec setter near `set_image_specs` `:547`, `_transcript_rows` `:804`, `_reconcile_rows` signature entries `:887`, `_build_row_widget` `:950`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_build_generation_card_specs` near `_build_console_image_specs` `:3115`; exclusion in `_build_console_image_specs`/`_recent_console_image_messages` `:3100`; wire setter near `:10578`)
- Test: `Tests/Chat/test_console_generation_card.py`

**Interfaces:**
- `ConsoleGenerationCardSpec` (frozen, in `console_generation_card.py`): `message_id: str, browsed_index: int, variant_count: int, meta: GenerationVariantMeta, mode: str, pixels, pil` (render fields mirror `ConsoleImageRowSpec`, `console_image_view.py:263` — the browsed variant's decoded image rides the existing `ConsoleImageRenderCache`, cache key `f"{message_id}:{browsed_index}"`).
- `ConsoleGenerationCard(Widget)` — bordered panel titled "Image Generation": browsed image (graphics/pixels per mode) + details block `Style` (meta.style or "Custom") / `Source` (backend) / `Seed` (value or "random") / `Prompt` / `Negative`, and a `n/N` indicator. Browsing/keep controls arrive via the action row (Task 8) — the card renders state only.
- Transcript: `set_generation_card_specs(self, specs: Mapping[str, ConsoleGenerationCardSpec]) -> None`; row kind `"generation-card"` emitted INSTEAD of the `"image"` row for messages present in the card-spec map; reconcile signature includes `(browsed_index, variant_count, mode, meta-hash)`.
- Screen: `_build_generation_card_specs()` builds specs from store messages with non-empty `generation_metadata` + screen-held browse state `self._generation_browse: dict[str, int]` (ephemeral); `_build_console_image_specs` skips those message ids (no double render, no 16-window slot burn).

- [ ] **Step 1: Failing tests** — pure/row-level (transcript rows are testable without a mounted app, per the module's own design):

```python
# Tests/Chat/test_console_generation_card.py
# Follow the row-building test style used for image rows (grep existing
# console_transcript tests for _transcript_rows / set_image_specs usage — mirror it).
def test_generation_card_row_replaces_image_row(...):
    # message with generation metadata + card spec registered:
    #  -> rows contain kind=="generation-card" for that message and NO kind=="image" row
def test_card_signature_changes_on_browse(...):
    # same message, browsed_index 0 vs 1 -> different reconcile signatures
def test_image_specs_exclude_card_messages(...):
    # _recent_console_image_messages / _build_console_image_specs skip card message ids
```

Write these as real tests against the actual helpers (the sketch above names the behavior; the implementer writes the concrete arrange/act/assert after reading the existing transcript tests — assertions as named are mandatory).

- [ ] **Step 2: RED. Step 3:** implement per Interfaces (card widget renders with Rich `Panel`/`Table` inside a mounted widget — the image-row widget `_image_row_widget` `:982` is the template for graphics/pixels handling). **Step 4: GREEN** + transcript/console suites: `python -m pytest Tests/Chat/ -q -k "transcript or generation"` green. **Step 5: Commit** `feat(console): generation card row + widget + image-row suppression (P2a)`.

---

### Task 8: Actions — `< >` generation gating, Keep, regenerate-appends

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py` (`available_actions` `:101`, gating `:107`, dispatch checks `:231-290`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (action dispatch: keep/browse/regenerate branches; regenerate in-flight guard `self._imagegen_inflight_message_ids: set[str]`; cap check `max_variants_per_message`)
- Test: `Tests/Chat/test_console_generation_card.py` (extend) + existing action-service tests stay green

**Interfaces:**
- `available_actions(..., generation_variant_count: int = 0, generation_browsed_index: int = 0)` — when `generation_variant_count > 0`: `< >` visible iff `generation_variant_count > 1` (boundary-enabled by browsed index; **takes precedence over sibling gating** for these two ids), new `("keep", "keep")` action visible iff `generation_browsed_index != 0`; regenerate stays visible (assistant-gated as today).
- Screen dispatch branches (generation messages only): `variant-previous/next` → mutate `self._generation_browse[message_id]` + rebuild card specs (ephemeral — no store/DB write); `keep` → `store.keep_generation_variant(...)` + reset browse to 0 + rebuild; `regenerate` → refuse when in-flight or at cap (status line), else thread-generate ONE variant (reuse `run_generation_batch(count=1)`, seed guard as Task 6) → `store.append_generation_variant(...)` → browse to the new index. Non-generation messages keep today's behavior exactly.

- [ ] **Step 1: Failing tests** — action-service level (pure): `< >` hidden at count 1, visible at 2 with correct boundary enables; `keep` only when browsed ≠ 0; sibling-based gating unchanged when `generation_variant_count == 0` (regression guard). Screen-level: keep resets browse and calls store; regenerate refused at cap / while in-flight (mock store + fake generate). Follow the existing `console_message_actions` test file's style.
- [ ] **Step 2: RED. Step 3: implement. Step 4: GREEN** + `python -m pytest Tests/Chat/ -q` (all Console suites green — the action-service signature change must not break existing callers/tests; default kwargs keep old call sites valid). **Step 5: Commit** `feat(console): generation-variant actions — browse/keep/regenerate-append (P2a)`.

---

### Task 9: Round-trip, regression sweep, live verify

**Files:**
- Test: `Tests/Chat/test_console_generation_store.py` (extend: reload round-trip)
- No production files expected (fixes only if the sweep finds regressions)

- [ ] **Step 1: Reload round-trip test** — create a generation message with 2 variants against REAL persistence (in-memory ChaChaNotes, as `test_chat_persistence_service.py` does), `keep` position 1, then simulate reload: fresh store, load conversation messages + `get_generation_metadata_for_messages` + `hydrate_generation_metadata`; assert position-0/canonical is the kept variant, metadata order matches, `generation_metadata` non-empty ⇒ card-eligible. RED→GREEN if any wiring is missing.
- [ ] **Step 2: Full regression sweep:**
  `python -m pytest Tests/ChaChaNotesDB/ Tests/Chat/ Tests/Image_Generation/ -q` — all green (attribute pre-existing failures by running the same suites on the base commit if anything is red; only new failures block).
- [ ] **Step 3: Lint:** `ruff check` on every file this plan touched → clean.
- [ ] **Step 4: App boots:** `python -c "import tldw_chatbook.app"` clean.
- [ ] **Step 5: Live TUI verify** (controller-level, tmux recipe from the `verify` skill): launch the app with a scratch `TLDW_CONFIG_PATH` profile, Console tab → `/generate-image :swarmui a red dragon` against a local SwarmUI if available (else assert the graceful not-configured status line + `/generate-image` usage line for empty prompt) → confirm the card renders, `< >`/keep/regenerate behave, and the conversation reopens with the card intact. Record observations.
- [ ] **Step 6: Commit** any fixes `fix(console): P2a round-trip/regression fixes` + final ledger note.

---

## Self-review notes (already applied)

- Spec §5.2's footgun is enforced twice: the store test asserting no full-rewrite call, and Task 2's swap reading bytes from the DB.
- Type consistency: `GenerationVariantMeta` (Task 5) is the one metadata type used by Tasks 6-9; DB rows (Tasks 1-2) use the dict shape with `params_json`, converted at the store boundary (`to_row`/`from_row`).
- Task 7/8's test steps name mandatory assertions rather than full code because they must be written against the real transcript/action-test harness styles — implementers read the existing sibling test files first (named in each task). All other tasks carry complete test code.
- Out of scope (do NOT build): TTS, Style-preset picker, prompt-from-context, sync/FTS for the sidecar, persistent action rows, `Media_Creation` cleanup.
