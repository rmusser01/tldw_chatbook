# TASK-208 Active Ingest Source Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent accidental duplicate active Library imports while preserving an explicit, one-shot second-press override across Local, Server, single-file, folder, keyboard, and external-model submission paths.

**Architecture:** Add a Textual-free canonical source key and active-job query to the existing ingest registry, then make `LibraryIngestQueueMixin.submit_library_ingest_job` the authoritative outer admission boundary. The Library screen previews the same registry state and extends its existing two-press consent grammar with a stable request fingerprint; the app repeats the check immediately before the first local append or remote call, and folder members route through a private already-admitted seam.

**Tech Stack:** Python 3.11+, Textual 8.x, stdlib `dataclasses`/`os.path`/`urllib.parse`, pytest, existing `LibraryIngestJobRegistry`, `LibraryIngestQueueMixin`, and Library canvas state/update seams.

## Global Constraints

- Active admission includes only `QUEUED`, `PARSING`, and `WRITING`; `DONE`, `FAILED`, `SKIPPED`, and `CANCELLED` never block.
- Identity is `(backend origin, canonical source)`; Local and Server are separate scopes.
- Canonicalization is lexical only: no symlink resolution, stat, content read, hash, SQLite query, or network access.
- HTTP(S) normalization lowercases scheme/host, removes default ports and fragments, treats an empty path as `/`, and preserves path bytes and query ordering.
- Folder admission is all-or-nothing and runs once before any member is appended or sent.
- The override is reason-specific, one-shot, request-local, and never persisted or exposed as a setting. It carries an opaque deterministic candidate-set digest/count and consented active IDs; the app accepts it only for an exact current candidate identity whose active matches are all covered.
- Expected refusal data contains only a bounded count, `(job_id, state)` references, and the opaque candidate digest/count required for late-refusal re-arming; its string/repr must not reveal source, title, keywords, options, or progress metadata.
- Exact active-source copy is `Import active. Start again to queue a duplicate.`, `2 active files. Start again to queue all.`, or `Import active; 2 may fail. Start again to queue.` as applicable.
- The fixed gate remains `markup=False`, one row high, and fully visible at `72x18`; color or glyphs are not required to understand the instruction.
- The existing `0.3` second dead zone remains authoritative for double-click/key-repeat rejection.
- No schema migration, dependency, persistent preference, content-hash fallback, redirect-aware identity, or historical uniqueness policy is in scope.
- ADR required: yes.
- ADR path: `backlog/decisions/065-active-ingest-source-admission-and-override.md` (accepted before implementation).
- Reason: ADR-065 owns the durable source identity, active-state scope, folder atomicity, app-boundary authority, privacy-safe refusal, and one-shot override policy.

## File Structure

- Modify `tldw_chatbook/Library/library_ingest_jobs.py`: immutable source key, lexical normalizer, active-state definition, privacy-safe duplicate-consent scope/refusal references, and copy-returning registry query.
- Modify `tldw_chatbook/app.py`: one authoritative outer admission check, exact-scope one-shot override argument, and private already-admitted child routing.
- Modify `tldw_chatbook/Library/library_ingest_state.py`: pure exact-copy projection for duplicate-only and combined consent.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: stable consent snapshot/fingerprint, preview query, first/second press behavior, late authoritative-refusal handling, and external-scope release.
- Modify `Tests/Library/test_library_ingest_jobs.py`: path/URL normalization, state/origin partitioning, copy isolation, and refusal privacy contracts.
- Modify `Tests/App/test_submit_library_ingest_job.py`: Local/Server authority, terminal allowance, mutation-sensitive atomic folder routing, scoped override consumption, and side-effect fencing.
- Modify `Tests/UI/test_library_ingest_inline_consent.py`: consent union, candidate/blast-radius fingerprint invalidation, keyboard/dead-zone behavior, late refusal, real screen-to-app scope forwarding, and external-resource ownership.
- Modify `Tests/UI/test_library_ingest_canvas.py`: exact one-row painted-compositor evidence at `72x18`.
- Modify `Docs/User_Guide/library/import-and-export.md`: active-import first-press refusal and deliberate second-press behavior.
- Modify `backlog/tasks/task-208 - Optional-source_path-dedup-for-ingest-submissions-idempotency.md`: plan link, checked acceptance criteria, implementation notes, verification evidence, and Done status after every gate passes.

## Final review refinement

The final whole-branch review found that the original Boolean override could
authorize a submit-time folder expansion or active-match set that the screen had
not previewed. The fix wave retains ADR-065 and replaces that Boolean with the
privacy-safe exact duplicate-consent scope above. TDD must cover added/removed or
changed folder members between presses, a newly active match absent from stale
preflight, and an identical warning list whose tooling affected count changes.
The scope also records whether its bounded active-ID references are complete and
must fail closed when they are not. Any Boolean-override snippets in the original
task steps below describe the superseded RED/first implementation, not the final
public coordinator interface.
At least one regression must drive the real screen state machine into the real
app coordinator rather than asserting either boundary in isolation.

---

### Task 1: Pure active-source identity and registry query

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Modify: `Tests/Library/test_library_ingest_jobs.py`

**Interfaces:**
- Consumes: `IngestJobState`, `LibraryIngestJob`, registry insertion order, and the existing copy-on-read contract.
- Produces: `ACTIVE_INGEST_STATES`, `ActiveIngestSourceKey`, `ActiveIngestJobRef`, `ActiveIngestSubmissionRefused`, `normalize_active_ingest_source(source, *, origin)`, and `LibraryIngestJobRegistry.find_active_source_matches(sources, *, origin)`.

- [ ] **Step 0: Recheck for competing TASK-208 work before implementation**

Run the repository's required in-flight and moved-dev checks before writing production code:

```powershell
gh pr list --state all --search "208" --json number,title,state,headRefName,url
git fetch origin
git log --oneline origin/dev -S "TASK-208" -- backlog/tasks
$task208MergeBase = git merge-base origin/dev HEAD
git log --oneline "$task208MergeBase..origin/dev" -- tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py
```

If another implementation exists, reconcile it against the approved spec before editing rather than producing a second complete implementation. If only upstream adjacent changes exist, rebase first and rerun the baseline focused tests.

- [ ] **Step 1: Write failing filesystem-source normalization tests**

Add tests that patch `os.path.normcase` to prove both Windows-like folding and case-sensitive behavior without touching the filesystem:

```python
def test_active_source_key_normalizes_relative_dot_segments_and_case(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        library_ingest_jobs.os.path,
        "normcase",
        lambda value: value.replace("/", "\\").lower(),
    )

    relative = normalize_active_ingest_source(
        ".\\Folder\\..\\Folder\\NOTE.txt", origin="local"
    )
    absolute = normalize_active_ingest_source(
        str(tmp_path / "folder" / "note.TXT"), origin="local"
    )

    assert relative == absolute
    assert relative.origin == "local"


def test_active_source_key_preserves_case_when_platform_normcase_does(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(library_ingest_jobs.os.path, "normcase", lambda value: value)

    upper = normalize_active_ingest_source("Note.txt", origin="local")
    lower = normalize_active_ingest_source("note.txt", origin="local")

    assert upper != lower
```

- [ ] **Step 2: Write failing conservative URL normalization tests**

```python
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("HTTPS://Example.COM", "https://example.com/"),
        ("http://example.com:80/a#one", "http://EXAMPLE.com/a#two"),
        ("https://example.com:443/a?q=1", "https://example.com/a?q=1"),
    ],
)
def test_active_source_key_normalizes_only_safe_url_equivalences(left, right):
    assert normalize_active_ingest_source(
        left, origin="server"
    ) == normalize_active_ingest_source(right, origin="server")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://example.com/A", "https://example.com/a"),
        ("https://example.com/a?x=1&y=2", "https://example.com/a?y=2&x=1"),
        ("https://example.com/a%2Fb", "https://example.com/a/b"),
        ("https://example.com:444/a", "https://example.com/a"),
    ],
)
def test_active_source_key_preserves_meaningful_url_distinctions(left, right):
    assert normalize_active_ingest_source(
        left, origin="server"
    ) != normalize_active_ingest_source(right, origin="server")
```

- [ ] **Step 3: Run the normalization tests and verify RED**

Run:

```powershell
python -m pytest Tests/Library/test_library_ingest_jobs.py -k "active_source_key" -q --basetemp=.pytest-tmp-task208-key-red
```

Expected: collection/import failure because `normalize_active_ingest_source` and `ActiveIngestSourceKey` do not exist.

- [ ] **Step 4: Implement the immutable key and lexical normalizer**

Add the following public shape beside `IngestJobState`; keep URL reconstruction in a small private helper and reject non-Local/Server origins:

```python
ACTIVE_INGEST_STATES: frozenset[IngestJobState] = frozenset(
    {
        IngestJobState.QUEUED,
        IngestJobState.PARSING,
        IngestJobState.WRITING,
    }
)


@dataclass(frozen=True, slots=True)
class ActiveIngestSourceKey:
    origin: str
    canonical_source: str


def normalize_active_ingest_source(
    source: str,
    *,
    origin: str,
) -> ActiveIngestSourceKey:
    normalized_origin = str(origin).strip().lower()
    if normalized_origin not in {"local", "server"}:
        raise ValueError("origin must be 'local' or 'server'")
    value = str(source).strip()
    if not value:
        raise ValueError("source must not be blank")

    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"http", "https"}:
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if not host:
            raise ValueError("http(s) source requires a host")
        rendered_host = f"[{host}]" if ":" in host else host
        raw_userinfo = (
            f"{parsed.netloc.rsplit('@', 1)[0]}@"
            if "@" in parsed.netloc
            else ""
        )
        port = parsed.port
        if port is not None and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            rendered_host = f"{rendered_host}:{port}"
        rendered_netloc = f"{raw_userinfo}{rendered_host}"
        canonical = urlunsplit(
            (scheme, rendered_netloc, parsed.path or "/", parsed.query, "")
        )
    else:
        expanded = os.path.expanduser(value)
        canonical = os.path.normcase(
            os.path.abspath(os.path.normpath(expanded))
        )
    return ActiveIngestSourceKey(normalized_origin, canonical)
```

Import `os` and `urlsplit`/`urlunsplit` from the standard library. The `raw_userinfo` branch preserves credentials byte-for-byte if the existing shared validator admits them while still normalizing only the host and default port.

- [ ] **Step 5: Run the normalization tests and verify GREEN**

Run:

```powershell
python -m pytest Tests/Library/test_library_ingest_jobs.py -k "active_source_key" -q --basetemp=.pytest-tmp-task208-key-green
```

Expected: all new normalization cases pass without creating or reading any test files beyond `tmp_path` directory setup.

- [ ] **Step 6: Write failing registry-state, origin, ordering, and copy-isolation tests**

```python
def test_find_active_source_matches_filters_state_origin_and_visibility(tmp_path):
    registry = LibraryIngestJobRegistry()
    source = str(tmp_path / "a.txt")
    queued = registry.submit(source_path=source, origin="local")
    parsing = registry.submit(source_path=source, origin="local")
    registry.mark_parsing(parsing.job_id)
    writing = registry.submit(source_path=source, origin="local")
    registry.mark_parsing(writing.job_id)
    registry.mark_writing(writing.job_id)
    terminal = registry.submit(source_path=source, origin="local")
    registry.mark_parsing(terminal.job_id)
    registry.mark_writing(terminal.job_id)
    registry.mark_done(terminal.job_id, media_id=1)
    registry.submit(source_path=source, origin="server")

    matches = registry.find_active_source_matches([source], origin="local")

    assert [job.job_id for job in matches] == [
        queued.job_id,
        parsing.job_id,
        writing.job_id,
    ]
    matches[0].source_path = "mutated"
    assert registry.get_job(queued.job_id).source_path == source


def test_find_active_source_matches_deduplicates_candidate_keys(tmp_path):
    registry = LibraryIngestJobRegistry()
    source = str(tmp_path / "a.txt")
    job = registry.submit(source_path=source)

    matches = registry.find_active_source_matches(
        [source, str(tmp_path / "." / "a.txt")], origin="local"
    )

    assert [item.job_id for item in matches] == [job.job_id]
```

- [ ] **Step 7: Write failing privacy-safe refusal tests**

```python
def test_active_ingest_refusal_exposes_only_bounded_safe_refs():
    refs = tuple(
        ActiveIngestJobRef(f"ingest-job-{index}", IngestJobState.QUEUED)
        for index in range(ACTIVE_INGEST_REF_LIMIT + 2)
    )
    refusal = ActiveIngestSubmissionRefused(refs)

    assert len(refusal.matches) == ACTIVE_INGEST_REF_LIMIT
    assert "ingest-job-1" not in str(refusal)
    assert "source_path" not in repr(refusal)
    assert set(vars(refusal)) == {"matches", "match_count"}
```

- [ ] **Step 8: Run the query/refusal tests and verify RED**

Run:

```powershell
python -m pytest Tests/Library/test_library_ingest_jobs.py -k "find_active_source_matches or active_ingest_refusal" -q --basetemp=.pytest-tmp-task208-query-red
```

Expected: failures because the registry query and refusal contract are absent.

- [ ] **Step 9: Implement copy-returning registry matching and safe refusal types**

Use a bounded immutable reference rather than storing job copies on the exception:

```python
ACTIVE_INGEST_REF_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class ActiveIngestJobRef:
    job_id: str
    state: IngestJobState


class ActiveIngestSubmissionRefused(RuntimeError):
    def __init__(self, matches: Iterable[ActiveIngestJobRef]) -> None:
        materialized = tuple(matches)
        self.match_count = len(materialized)
        self.matches = materialized[:ACTIVE_INGEST_REF_LIMIT]
        super().__init__(f"Active ingest admission refused ({self.match_count} matches).")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(match_count={self.match_count}, "
            f"states={tuple(ref.state.value for ref in self.matches)!r})"
        )
```

Add the registry read in internal insertion order, excluding hidden jobs and returning `_copy_job` copies:

```python
def find_active_source_matches(
    self,
    sources: Iterable[str],
    *,
    origin: str,
) -> tuple[LibraryIngestJob, ...]:
    keys: set[ActiveIngestSourceKey] = set()
    for source in sources:
        try:
            keys.add(normalize_active_ingest_source(source, origin=origin))
        except (TypeError, ValueError, OSError):
            continue
    if not keys:
        return ()
    return tuple(
        _copy_job(job)
        for job in self._jobs
        if not (job.superseded or job.dismissed)
        and job.state in ACTIVE_INGEST_STATES
        and normalize_active_ingest_source(
            job.source_path, origin=job.origin
        ) in keys
    )
```

Guard a malformed stored job inside the loop instead of allowing it to abort the whole query; use a private `_active_source_key_or_none` helper so both candidates and stored jobs follow the same failure policy.

- [ ] **Step 10: Run the complete pure registry module**

Run:

```powershell
python -m pytest Tests/Library/test_library_ingest_jobs.py -q --basetemp=.pytest-tmp-task208-registry
```

Expected: the full module passes, including existing transition, restore, persistence, copy-isolation, and listener contracts.

- [ ] **Step 11: Run scoped static checks and commit Task 1**

Run:

```powershell
python -m ruff check tldw_chatbook/Library/library_ingest_jobs.py Tests/Library/test_library_ingest_jobs.py
python -m py_compile tldw_chatbook/Library/library_ingest_jobs.py Tests/Library/test_library_ingest_jobs.py
git diff --check
```

Then stage only the two Task 1 files and commit:

```powershell
git add -- tldw_chatbook/Library/library_ingest_jobs.py Tests/Library/test_library_ingest_jobs.py
git commit -m "feat(library): define active ingest source identity"
```

---

### Task 2: Authoritative app-boundary admission and atomic folder routing

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`

**Interfaces:**
- Consumes: `LibraryIngestJobRegistry.find_active_source_matches`, `ActiveIngestJobRef`, `ActiveIngestSubmissionRefused`, `_expand_library_ingest_source`, `_resolve_ingest_backend`, `_submit_server_ingest_job`, `_submit_web_clip_job`, and the local parse-pool submission path.
- Produces: `submit_library_ingest_job(..., allow_active_duplicate: bool = False)`, `_submit_library_ingest_job_admitted(..., backend: str, batch_id: str | None)`, and `_submit_local_library_ingest_job(...)`.

- [ ] **Step 1: Write failing Local/Server and terminal-state admission tests**

```python
def test_submit_refuses_active_local_duplicate_before_second_append(tmp_path):
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    first = app.submit_library_ingest_job(source_path=str(source))
    before_ids = [job.job_id for job in app.library_ingest_jobs.jobs()]

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(source_path=str(source))

    assert [job.job_id for job in app.library_ingest_jobs.jobs()] == before_ids
    assert caught.value.matches == (
        ActiveIngestJobRef(first.job_id, first.state),
    )


def test_terminal_local_job_does_not_block_reingestion(tmp_path):
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    first = app.submit_library_ingest_job(source_path=str(source))
    app.library_ingest_jobs.mark_parsing(first.job_id)
    app.library_ingest_jobs.mark_writing(first.job_id)
    app.library_ingest_jobs.mark_done(first.job_id, media_id=1)

    second = app.submit_library_ingest_job(source_path=str(source))

    assert second.job_id != first.job_id


def test_local_active_job_does_not_block_server_submission(monkeypatch, tmp_path):
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    app.submit_library_ingest_job(source_path=str(source))
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "server")
    remote = MagicMock(return_value=_make_job(origin="server"))
    monkeypatch.setattr(app, "_submit_server_ingest_job", remote)

    app.submit_library_ingest_job(source_path=str(source))

    remote.assert_called_once()


def test_submit_refuses_active_server_duplicate_before_remote_call(
    monkeypatch, tmp_path
):
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "server")
    active = app.library_ingest_jobs.submit(
        source_path=str(source), origin="server"
    )
    remote = MagicMock()
    monkeypatch.setattr(app, "_submit_server_ingest_job", remote)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(source_path=str(source))

    assert caught.value.matches == (
        ActiveIngestJobRef(active.job_id, IngestJobState.QUEUED),
    )
    remote.assert_not_called()
```

- [ ] **Step 2: Write failing all-or-nothing folder tests**

Use a real expanded folder and spy on the future private seam:

```python
def test_folder_refusal_occurs_before_any_admitted_child(monkeypatch, tmp_path):
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    first = folder / "first.txt"
    matching = folder / "matching.txt"
    first.write_text("first")
    matching.write_text("matching")
    app.submit_library_ingest_job(source_path=str(matching))
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)

    with pytest.raises(ActiveIngestSubmissionRefused):
        app.submit_library_ingest_job(source_path=str(folder))

    admitted.assert_not_called()


def test_confirmed_folder_routes_every_member_once_without_reentry(
    monkeypatch, tmp_path
):
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
    app.library_ingest_jobs.submit(source_path=str(paths[1]))
    original = app._submit_library_ingest_job_admitted
    admitted_calls = []

    def record(**kwargs):
        admitted_calls.append((kwargs["source_path"], kwargs["batch_id"]))
        return original(**kwargs)

    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", record)

    app.submit_library_ingest_job(
        source_path=str(folder), allow_active_duplicate=True
    )

    assert [source for source, _batch_id in admitted_calls] == [
        str(path) for path in paths
    ]
    batch_ids = {batch_id for _source, batch_id in admitted_calls}
    assert len(batch_ids) == 1
    assert None not in batch_ids
```

- [ ] **Step 3: Write failing refusal privacy and side-effect-fence tests**

Assert the refusal happens before `_send_server_ingest_job`, `_top_up_ingest_parse_pool`, and a second job ID allocation, and that `str`/`repr` contain none of the source, title, author, keywords, prompt options, or progress text:

```python
def test_direct_refusal_is_privacy_safe_and_starts_no_work(monkeypatch, tmp_path):
    app = _minimal_app(media_db="present")
    source = tmp_path / "private-name.txt"
    source.write_text("secret")
    app.submit_library_ingest_job(source_path=str(source), title="Private title")
    top_up = MagicMock()
    monkeypatch.setattr(app, "_top_up_ingest_parse_pool", top_up)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(
            source_path=str(source),
            title="Private title",
            keywords=("private-keyword",),
            ingest_options={"generic": {"custom_prompt": "private-prompt"}},
        )

    rendered = f"{caught.value!s} {caught.value!r}"
    for secret in (
        str(source),
        "Private title",
        "private-keyword",
        "private-prompt",
    ):
        assert secret not in rendered
    top_up.assert_not_called()
```

- [ ] **Step 4: Run app admission tests and verify RED**

Run:

```powershell
python -m pytest Tests/App/test_submit_library_ingest_job.py -k "active or folder_refusal or confirmed_folder or privacy_safe" -q --basetemp=.pytest-tmp-task208-app-red
```

Expected: duplicate submissions are still accepted, `allow_active_duplicate` is unknown, and the private child seam does not exist.

- [ ] **Step 5: Split outer admission from admitted child routing**

Extend the public signature with a default-off keyword and capture expansion/backend exactly once:

```python
def submit_library_ingest_job(
    self,
    *,
    source_path: str,
    ingest_options: dict[str, Any] | None = None,
    title: str = "",
    author: str = "",
    keywords: tuple[str, ...] = (),
    perform_analysis: bool = False,
    chunk_enabled: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_id: str | None = None,
    allow_active_duplicate: bool = False,
) -> LibraryIngestJob:
    backend = self._resolve_ingest_backend()
    expanded = self._expand_library_ingest_source(source_path)
    sources = tuple(expanded) if expanded is not None else (source_path,)
    matches = self.library_ingest_jobs.find_active_source_matches(
        sources, origin=backend
    )
    if matches and not allow_active_duplicate:
        raise ActiveIngestSubmissionRefused(
            ActiveIngestJobRef(job.job_id, job.state) for job in matches
        )
```

Preserve the existing empty-folder failure path for non-screen callers after the guard. For a non-empty folder, mint one batch ID and call the private method directly for every member; do not recursively call `submit_library_ingest_job`:

```python
def _submit_library_ingest_job_admitted(
    self,
    *,
    source_path: str,
    ingest_options: dict[str, Any],
    title: str,
    author: str,
    keywords: tuple[str, ...],
    perform_analysis: bool,
    chunk_enabled: bool,
    chunk_size: int,
    batch_id: str | None,
    backend: str,
) -> LibraryIngestJob:
    if backend == "server":
        submit_remote = (
            self._submit_web_clip_job
            if is_web_clip_source(source_path)
            else self._submit_server_ingest_job
        )
        return submit_remote(
            source_path=source_path,
            ingest_options=ingest_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
        )
    return self._submit_local_library_ingest_job(
        source_path=source_path,
        ingest_options=ingest_options,
        title=title,
        author=author,
        keywords=keywords,
        perform_analysis=perform_analysis,
        chunk_enabled=chunk_enabled,
        chunk_size=chunk_size,
        batch_id=batch_id,
    )
```

Extract the current Local classification/registry append/media-DB/top-up tail verbatim into `_submit_local_library_ingest_job`. Keep the existing Parakeet batch-scope `try/finally` around the entire admitted folder loop, and keep `title=""` for every folder child.

- [ ] **Step 6: Run the focused app admission tests and verify GREEN**

Run:

```powershell
python -m pytest Tests/App/test_submit_library_ingest_job.py -k "active or folder_refusal or confirmed_folder or privacy_safe" -q --basetemp=.pytest-tmp-task208-app-green
```

Expected: active duplicates refuse before side effects; a reason-specific override admits the captured batch exactly once.

- [ ] **Step 7: Run the complete app submission and runner regressions**

Run the synchronous app module first, then the queue-runner module with a repository-local temp root:

```powershell
python -m pytest Tests/App/test_submit_library_ingest_job.py -q --basetemp=.pytest-tmp-task208-app
python -m pytest Tests/Library/test_library_ingest_runner.py -q --basetemp=.pytest-tmp-task208-runner
```

Expected: all product assertions pass. If Windows Proactor setup collides with the repository network guard, record the exact setup errors and rerun only the established affected coroutine nodes with their existing `allow_network` opt-in; do not classify a harness-only setup failure as a product pass.

- [ ] **Step 8: Run scoped static checks and commit Task 2**

Run:

```powershell
python -m ruff check --ignore E402 tldw_chatbook/app.py Tests/App/test_submit_library_ingest_job.py
python -m py_compile tldw_chatbook/app.py Tests/App/test_submit_library_ingest_job.py
git diff --check
```

Do not bulk-format `app.py`; isolate any inherited whole-file lint debt from changed-line findings. Then commit only the Task 2 files:

```powershell
git add -- tldw_chatbook/app.py Tests/App/test_submit_library_ingest_job.py
git commit -m "feat(library): guard active ingest admission"
```

---

### Task 3: Stable two-press screen consent and external-resource handling

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_ingest_inline_consent.py`

**Interfaces:**
- Consumes: the Task 1 registry query/normalizer/refusal, the Task 2 `allow_active_duplicate` keyword, current preflight candidate paths, `_build_ingest_options_snapshot`, `_update_library_ingest_gate`, and `_enqueue_library_ingest_snapshot`.
- Produces: `active_ingest_start_confirm_line`, `_LibraryIngestStartConsent`, `_current_library_ingest_start_consent(submitted_source: str, gate_state: LibraryIngestCanvasState | None = None) -> _LibraryIngestStartConsent`, and expected-refusal handling that preserves the form and releases an untransferred external scope.

- [ ] **Step 1: Write failing pure exact-copy tests**

Add to `Tests/UI/test_library_ingest_inline_consent.py`:

```python
@pytest.mark.parametrize(
    ("active_files", "is_folder", "tooling_files", "expected"),
    [
        (1, False, 0, "Import active. Start again to queue a duplicate."),
        (2, True, 0, "2 active files. Start again to queue all."),
        (1, False, 2, "Import active; 2 may fail. Start again to queue."),
    ],
)
def test_active_ingest_confirm_copy_is_exact(
    active_files, is_folder, tooling_files, expected
):
    assert active_ingest_start_confirm_line(
        active_source_count=active_files,
        is_folder=is_folder,
        tooling_affected_count=tooling_files,
    ) == expected
    assert len(expected) <= 48
```

- [ ] **Step 2: Write failing consent fingerprint and reason-specific override tests**

Extend `_minimal_library_screen` with a real `LibraryIngestJobRegistry`, a deterministic `_build_ingest_options_snapshot`, and the new consent attribute. Add cases that prove:

```python
def _stage_plain_file(screen: LibraryScreen, tmp_path) -> str:
    source = tmp_path / "file.txt"
    source.write_text("body")
    screen._library_ingest_form.path = str(source)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"generic": [str(source)]},
        total_files=1,
    )
    return str(source)


def test_active_job_lifecycle_transition_preserves_armed_consent(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    job = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    armed = screen._library_ingest_start_consent
    screen.app_instance.library_ingest_jobs.mark_parsing(job.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(job.job_id)

    assert screen._current_library_ingest_start_consent(source).fingerprint == (
        armed.fingerprint
    )


def test_tooling_only_consent_cannot_override_late_duplicate(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    duplicate = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen._library_ingest_start_consent.active_job_ids == (
        duplicate.job_id,
    )


def test_active_duplicate_second_press_passes_one_shot_override(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert kwargs["allow_active_duplicate"] is True
```

Add separate pins for source/options/backend/warning/membership changes disarming, identical warning refresh preserving, blur/focus preserving, Escape disarming, and Enter sharing the same method. Explicitly move the active job through `QUEUED -> PARSING -> WRITING` between presses.

Use a table-driven fingerprint test and explicit folder/combined assertions:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        lambda screen: setattr(screen._library_ingest_form, "title", "changed"),
        lambda screen: setattr(screen._library_ingest_form, "author", "changed"),
        lambda screen: setattr(screen._library_ingest_form, "keywords", "changed"),
        lambda screen: screen._library_ingest_form.type_options.setdefault(
            "generic", {}
        ).update({"custom_prompt": "changed"}),
        lambda screen: setattr(
            screen.app_instance, "_resolve_ingest_backend", lambda: "server"
        ),
    ],
)
def test_request_mutation_changes_active_consent_fingerprint(
    mutation, tmp_path
):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)
    before = screen._current_library_ingest_start_consent(source).fingerprint

    mutation(screen)

    after = screen._current_library_ingest_start_consent(source).fingerprint
    assert after != before


def test_folder_preview_counts_distinct_active_files(tmp_path):
    screen = _minimal_library_screen()
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
        screen.app_instance.library_ingest_jobs.submit(source_path=str(path))
    screen._library_ingest_form.path = str(folder)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"generic": [str(path) for path in paths]},
        total_files=2,
    )

    screen._submit_library_ingest_form()

    assert screen._library_ingest_start_consent.active_source_count == 2
    consent = screen._library_ingest_start_consent
    assert active_ingest_start_confirm_line(
        active_source_count=consent.active_source_count,
        is_folder=consent.is_folder,
        tooling_affected_count=consent.tooling_affected_count,
    ) == (
        "2 active files. Start again to queue all."
    )


def test_combined_tooling_and_active_warning_takes_two_presses(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_pdf(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    consent = screen._library_ingest_start_consent
    assert active_ingest_start_confirm_line(
        active_source_count=consent.active_source_count,
        is_folder=consent.is_folder,
        tooling_affected_count=consent.tooling_affected_count,
    ) == (
        "Import active; 1 may fail. Start again to queue."
    )
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert screen.app_instance.submit_library_ingest_job.call_count == 1
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "allow_active_duplicate"
        ]
        is True
    )
```

- [ ] **Step 3: Run the copy/fingerprint tests and verify RED**

Run:

```powershell
python -m pytest Tests/UI/test_library_ingest_inline_consent.py -k "active_ingest or lifecycle_transition or tooling_only or one_shot_override" -q --basetemp=.pytest-tmp-task208-screen-red
```

Expected: exact-copy helper, consent snapshot, and override keyword forwarding are absent.

- [ ] **Step 4: Implement the pure exact-copy projection**

In `library_ingest_state.py`, add:

```python
def active_ingest_start_confirm_line(
    *,
    active_source_count: int,
    is_folder: bool,
    tooling_affected_count: int,
) -> str:
    if active_source_count and tooling_affected_count:
        return (
            f"Import active; {tooling_affected_count} may fail. "
            "Start again to queue."
        )
    if active_source_count and is_folder:
        noun = "file" if active_source_count == 1 else "files"
        return (
            f"{active_source_count} active {noun}. "
            "Start again to queue all."
        )
    if active_source_count:
        return "Import active. Start again to queue a duplicate."
    return ""
```

Add `start_confirm_line: str = ""` to `build_library_ingest_state`. A non-empty override is sufficient to make `start_confirm_active` true even without tooling warnings; when armed, use the override first and otherwise retain `forecast_consent_line(forecast)` for tooling-only behavior.

- [ ] **Step 5: Implement the immutable screen consent snapshot and fingerprint**

Define a private frozen carrier near the other screen-local dataclasses:

```python
@dataclass(frozen=True, slots=True)
class _LibraryIngestStartConsent:
    fingerprint: str
    active_job_ids: tuple[str, ...]
    active_source_count: int
    tooling_affected_count: int
    is_folder: bool

    @property
    def owed(self) -> bool:
        return bool(self.active_job_ids or self.tooling_affected_count)

    @property
    def allows_active_duplicate(self) -> bool:
        return bool(self.active_job_ids)
```

Build the fingerprint from JSON-safe values with stable ordering:

```python
fingerprint_payload = {
    "source": submitted_source,
    "backend": backend,
    "title": self._safe_text(form.title, max_length=300),
    "author": self._safe_text(form.author, max_length=200),
    "keywords": parse_keywords(form.keywords),
    "options": self._build_ingest_options_snapshot(),
    "warnings": form.preflight.warnings if form.preflight else [],
    "active_job_ids": tuple(job.job_id for job in matches),
}
fingerprint = json.dumps(
    fingerprint_payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    default=str,
)
```

For preview candidates, flatten the already-captured `preflight.type_groups` paths. Treat the source as a folder snapshot when candidates exist and its normalized key is not among them; never call `Path.is_dir`, `_expand_library_ingest_source`, or `analyze_path` from this preview. Query the registry with the flattened candidates for a folder and with the submitted source otherwise. Count distinct normalized matched sources, while keeping stable matching job IDs in registry order.

- [ ] **Step 6: Replace the Boolean/warning carrier with equality-based consent**

Initialize `_library_ingest_start_consent: _LibraryIngestStartConsent | None = None`. Remove the screen-owned `_library_ingest_start_confirm_armed` Boolean and `_library_ingest_start_confirm_warnings` list, update their tests/call sites to inspect the consent object, and derive the render-state `start_confirm_armed` Boolean from `consent is not None` so two state carriers cannot drift.

Update `_submit_library_ingest_form` to follow this exact decision order:

```python
pending = self._current_library_ingest_start_consent(submitted_source, gate_state)
armed = self._library_ingest_start_consent
if pending.owed:
    if armed is None or armed.fingerprint != pending.fingerprint:
        self._library_ingest_start_consent = pending
        self._library_ingest_start_confirm_armed_at = time.monotonic()
        self._update_library_ingest_gate(self._build_library_ingest_state())
        return
    if (
        time.monotonic() - self._library_ingest_start_confirm_armed_at
        < self._START_CONFIRM_DEAD_ZONE_SECONDS
    ):
        return
    allow_active_duplicate = armed.allows_active_duplicate
    self._disarm_library_ingest_start_confirm()
else:
    allow_active_duplicate = False
    if armed is not None:
        self._disarm_library_ingest_start_confirm()
self._do_submit_ingest(
    submitted_source,
    allow_active_duplicate=allow_active_duplicate,
)
```

Pass `allow_active_duplicate` into the captured `submit_kwargs`. Update `_build_library_ingest_state` to supply `start_confirm_armed=consent is not None` and the active/combined copy from `active_ingest_start_confirm_line`.

When the registry listener fires, recompute the current consent before repainting: preserve it when only state tokens change for the same job IDs; disarm when active membership changes. Continue disarming on source/form/backend changes, preflight invalidation, rail reset/exit, and Escape. Preserve consent on identical preflight results and focus movement. Correct `_disarm_library_ingest_start_confirm`'s docstring so it no longer claims path-field blur disarms consent.

- [ ] **Step 7: Write failing late-authoritative-refusal and external-scope tests**

```python
def test_late_active_refusal_preserves_form_without_generic_error(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.submit_library_ingest_job.side_effect = (
        ActiveIngestSubmissionRefused(
            (ActiveIngestJobRef("ingest-job-7", IngestJobState.QUEUED),)
        )
    )

    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {}}
    )

    assert screen._library_ingest_form.path == source
    assert screen._library_ingest_start_consent is not None
    screen.app_instance.notify.assert_not_called()


def test_late_active_refusal_releases_untransferred_external_scope(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    screen.app_instance.submit_library_ingest_job.side_effect = (
        ActiveIngestSubmissionRefused(
            (ActiveIngestJobRef("ingest-job-7", IngestJobState.PARSING),)
        )
    )

    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {"audio_video": {}}},
        generation=3,
        scope_id="scope-3",
    )

    service.release_scope.assert_called_once_with("scope-3")
    assert screen._library_ingest_form.path == source
```

- [ ] **Step 8: Catch expected refusal separately and re-arm current consent**

In `_enqueue_library_ingest_snapshot`, add an `except ActiveIngestSubmissionRefused` branch before the generic exception. Compare pre/post job ID sets, release `scope_id` when no job was created, clear external busy/progress ownership, preserve the form and last-submission state, recompute the current consent, and update only the gate line. Do not log at error/warning level, call `notify`, save option defaults, clear fields, scroll the queue, or create a failed receipt.

If the current preview cannot reconstruct matching sources, build a safe generic duplicate consent from the refusal's stable job IDs and use `Import active. Start again to queue a duplicate.`; do not infer a folder-file count from job count, and do not attach the exception or job objects to screen state.

- [ ] **Step 8a: Prove preview blocks external preparation before ownership transfer**

Add a direct regression in the same module:

```python
def _stage_external_parakeet_audio(screen, tmp_path) -> str:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"RIFF")
    form = screen._library_ingest_form
    form.path = str(source)
    form.preflight = _preflight(
        type_groups={"audio_video": [str(source)]},
        total_files=1,
    )
    form.type_options["audio_video"] = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model_dir": str(tmp_path / "model"),
    }
    screen._prepare_library_external_submission = MagicMock()
    return str(source)


def test_active_preview_blocks_external_preparation_before_retain(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_external_parakeet_audio(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()

    screen._prepare_library_external_submission.assert_not_called()
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    service.retain_prepared.assert_not_called()
    assert screen._library_ingest_form.path == source
```

- [ ] **Step 9: Run the complete inline-consent module**

Run:

```powershell
python -m pytest Tests/UI/test_library_ingest_inline_consent.py -q --basetemp=.pytest-tmp-task208-inline
```

Expected: existing tooling-only two-press behavior remains green, and new active/combined behavior passes for button, Enter, dead zone, focus, Escape, lifecycle transitions, late refusal, and scope release.

- [ ] **Step 10: Run focused Library state and shell regressions**

Run:

```powershell
python -m pytest Tests/Library/test_library_ingest_state.py -q --basetemp=.pytest-tmp-task208-state
python -m pytest Tests/UI/test_library_shell.py -k "ingest and (start or consent or preflight or registry)" -q --basetemp=.pytest-tmp-task208-shell
```

Expected: state projection and focused mounted-shell assertions pass. Separate any established Windows Proactor/network-guard setup failure from product results and rerun only affected coroutine nodes with their documented opt-in.

- [ ] **Step 11: Run scoped static checks and commit Task 3**

Run:

```powershell
python -m ruff check --ignore E721 tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_ingest_inline_consent.py
python -m py_compile tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_ingest_inline_consent.py
git diff --check
```

Isolate inherited `library_screen.py` whole-file findings rather than bulk-reformatting the file. Then commit only the Task 3 files:

```powershell
git add -- tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_ingest_inline_consent.py
git commit -m "feat(library): confirm active duplicate imports inline"
```

---

### Task 4: Painted geometry, documentation, and task closeout

**Files:**
- Modify: `Tests/UI/test_library_ingest_canvas.py`
- Modify: `Docs/User_Guide/library/import-and-export.md`
- Modify: `backlog/tasks/task-208 - Optional-source_path-dedup-for-ingest-submissions-idempotency.md`

**Interfaces:**
- Consumes: exact copy from Task 3, the existing fixed-height `#library-ingest-start-quiet-line`, `_CanvasHost`, Backlog task markers, and ADR-065.
- Produces: painted compositor evidence, truthful user documentation, complete implementation notes, and verified TASK-208 closeout.

- [ ] **Step 1: Write the failing `72x18` painted-compositor test**

Mount the real canvas for every active-source sentence and inspect the actual compositor strips, not only `Static.renderable`:

```python
@pytest.mark.parametrize(
    "copy",
    [
        "Import active. Start again to queue a duplicate.",
        "2 active files. Start again to queue all.",
        "Import active; 2 may fail. Start again to queue.",
    ],
)
@pytest.mark.asyncio
async def test_active_ingest_confirm_copy_fits_fixed_gate_at_72x18(copy):
    form = LibraryIngestFormState(path="C:/docs/a.txt")
    state = build_library_ingest_state(
        (),
        form=form,
        start_confirm_armed=True,
        start_confirm_line=copy,
    )
    app = _CanvasHost(state)

    async with app.run_test(size=(72, 18)) as pilot:
        await pilot.pause()
        quiet = app.query_one("#library-ingest-start-quiet-line", Static)
        start = app.query_one("#library-ingest-start", Button)
        strips = app.screen._compositor.render_strips()
        painted = "".join(
            strip.text
            for strip in strips[quiet.region.y : quiet.region.bottom]
        )

        assert quiet.region.height == 1
        assert copy in painted
        assert "…" not in painted
        assert quiet.region.bottom <= start.region.y
```

Add a mounted in-place stability test that captures actual widget identities and interaction state before arming:

```python
@pytest.mark.asyncio
async def test_active_confirm_update_preserves_start_input_focus_cursor_and_scroll(
    tmp_path,
):
    app = _build_test_app()
    _seed_conversations(app, ())
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        path_input = screen.query_one("#library-ingest-path", Input)
        start = screen.query_one("#library-ingest-start", Button)
        canvas = screen.query_one(LibraryIngestCanvas)
        path_input.value = str(tmp_path / "active.txt")
        path_input.cursor_position = 3
        path_input.focus()
        canvas.scroll_to(y=2, animate=False, force=True, immediate=True)
        start_region = start.region
        scroll_y = canvas.scroll_y

        screen._library_ingest_start_consent = _LibraryIngestStartConsent(
            fingerprint="active-test",
            active_job_ids=("ingest-job-1",),
            active_source_count=1,
            tooling_affected_count=0,
            is_folder=False,
        )
        screen._update_library_ingest_gate(screen._build_library_ingest_state())
        await pilot.pause()

        assert screen.query_one("#library-ingest-path", Input) is path_input
        assert screen.query_one("#library-ingest-start", Button) is start
        assert screen.focused is path_input
        assert path_input.cursor_position == 3
        assert canvas.scroll_y == scroll_y
        assert start.region == start_region
```

Import the private carrier into the test module only for this direct state-projection pin; do not add a production testing seam.

- [ ] **Step 2: Run the compositor test and verify RED if any copy clips**

Run:

```powershell
python -m pytest Tests/UI/test_library_ingest_canvas.py -k "active_ingest_confirm_copy" -q --basetemp=.pytest-tmp-task208-geometry-red
```

Expected before final UI wiring: failure because `start_confirm_line` is absent or the exact copy is not painted. After Task 3, this may already pass; retain the test as the required independent rendered evidence.

- [ ] **Step 3: Make the smallest geometry correction if the painted test is red**

Keep `start_quiet_line.styles.height = 1`, `markup=False`, and the existing commit-bar layout. Prefer shortening only non-binding incidental copy or correcting width allocation; do not add wrapping, animation, a modal, or move the Start button. Re-run the exact compositor test after each change.

- [ ] **Step 4: Update the Library user guide**

Add a concise paragraph beside the existing Start-import consent documentation:

```markdown
If the same source is already queued, parsing, or being written for the chosen
Local or Server destination, the first Start press queues nothing and the line
beside Start says the import is active. Press Start again after the brief
double-press guard to deliberately queue one duplicate. Local and Server are
separate, finished jobs do not block another import, and a folder is admitted
as one batch: the first press queues none of its files, while the confirmed
second press queues the whole unchanged selection.
```

Explain that a combined tooling warning and active-source warning still takes two presses total, not three, and that editing the request or leaving the canvas cancels pending consent.

- [ ] **Step 5: Run the full focused verification matrix**

Run each module independently with a repository-local temp root so one harness boundary does not erase evidence from the others:

```powershell
python -m pytest Tests/Library/test_library_ingest_jobs.py -q --basetemp=.pytest-tmp-task208-final-jobs
python -m pytest Tests/App/test_submit_library_ingest_job.py -q --basetemp=.pytest-tmp-task208-final-app
python -m pytest Tests/Library/test_library_ingest_state.py -q --basetemp=.pytest-tmp-task208-final-state
python -m pytest Tests/UI/test_library_ingest_inline_consent.py -q --basetemp=.pytest-tmp-task208-final-inline
python -m pytest Tests/UI/test_library_ingest_canvas.py -q --basetemp=.pytest-tmp-task208-final-canvas
python -m pytest Tests/integration/test_library_ingest_flow.py -q --basetemp=.pytest-tmp-task208-final-flow
```

Then run the directly affected runner and focused mounted-shell selection. Stop and split a command that exceeds five minutes without useful progress rather than treating a hang as evidence:

```powershell
python -m pytest Tests/Library/test_library_ingest_runner.py -q --basetemp=.pytest-tmp-task208-final-runner
python -m pytest Tests/UI/test_library_shell.py -k "ingest and (start or consent or preflight or registry)" -q --basetemp=.pytest-tmp-task208-final-shell
```

- [ ] **Step 6: Run final static, privacy, and diff gates**

Run:

```powershell
python -m ruff check --ignore E402,E721 tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/app.py tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_ingest_jobs.py Tests/App/test_submit_library_ingest_job.py Tests/UI/test_library_ingest_inline_consent.py Tests/UI/test_library_ingest_canvas.py
python -m py_compile tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_ingest_jobs.py Tests/App/test_submit_library_ingest_job.py Tests/UI/test_library_ingest_inline_consent.py Tests/UI/test_library_ingest_canvas.py
rg -n "source_path|title|author|keywords|custom_prompt|system_prompt|ingest_options|progress" Tests/Library/test_library_ingest_jobs.py Tests/App/test_submit_library_ingest_job.py Tests/UI/test_library_ingest_inline_consent.py
git diff --check origin/dev...HEAD
git status --short
```

The `rg` output is an audit list: inspect each hit in refusal tests/implementation and confirm sensitive fields appear only as negative assertions or ordinary submission data, never in exception payload/string/repr or expected-refusal diagnostics.

- [ ] **Step 7: Self-review the implementation against every specification clause**

Read `Docs/superpowers/specs/2026-08-13-task-208-active-ingest-source-admission-design.md` and explicitly verify:

- every active and terminal state;
- Local/Server partitioning;
- path and URL normalization boundaries;
- no filesystem/network/DB work in preview/query;
- folder all-or-nothing admission and non-recursive child routing;
- refusal privacy and pre-side-effect ordering;
- stable job IDs rather than lifecycle states in consent;
- reason-specific override and tooling-only late-duplicate behavior;
- blur/focus preservation and corrected disarm docstring;
- external scope release with no generic failure receipt;
- exact one-row copy and `72x18` painted geometry;
- no new preference, schema, dependency, or historical uniqueness behavior.

Fix any mismatch before updating the task.

- [ ] **Step 8: Complete Backlog task hygiene**

In the task file:

1. Check all nine acceptance criteria only after their corresponding evidence is green.
2. Add `## Implementation Notes` summarizing the pure key/query, authoritative outer guard, private folder child seam, consent fingerprint, refusal privacy, resource release, exact copy, documentation, and test evidence.
3. Record the ADR disposition as ADR-065 implemented; ADR-014 remains unchanged.
4. Record any Windows harness-only setup limitation separately from product results.
5. Add a lessons entry only if implementation produced a new repeatable incident; do not invent one for checklist completeness.
6. Set status to Done with `backlog task edit 208 -s Done`, then verify with `backlog task 208 --plain` and `git diff` that the CLI preserved the detailed notes and criteria.

- [ ] **Step 9: Commit documentation and closeout evidence**

Stage only the geometry test, user guide, and task file:

```powershell
git add -- Tests/UI/test_library_ingest_canvas.py Docs/User_Guide/library/import-and-export.md "backlog/tasks/task-208 - Optional-source_path-dedup-for-ingest-submissions-idempotency.md"
git commit -m "docs(library): explain active ingest admission"
```

- [ ] **Step 10: Verify the branch is clean and ready for review**

Run:

```powershell
git status --short
git log --oneline --decorate -6
git diff --stat origin/dev...HEAD
```

Expected: clean worktree; commits are reviewable by pure contract, app authority, screen consent, and documentation/closeout boundaries.
