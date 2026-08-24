# MCP Profile-Driven RAG Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCP `search_rag` follow the active RAG profile by default while preserving its boolean API, media-only boundary, response shape, and truthful score interpretation.

**Architecture:** Keep MCP on its existing media adapter, resolve `plain|semantic|hybrid` through one lightweight active-profile helper, and borrow the process-wide enhanced RAG runtime only for semantic/hybrid requests. Fix the already-emitted `$in` media filter and reranker-construction disclosure at their shared engine owners; reuse Library's score-kind helper in the inspector.

**Tech Stack:** Python 3.11+, asyncio, Textual 8.x, SQLite/FTS5, existing simplified RAG runtime, pytest, Ruff, Backlog.md CLI

**Spec:** `Docs/superpowers/specs/2026-08-23-task-3500-mcp-profile-driven-rag-search-design.md`

**Backlog task:** `backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/084-mcp-profile-driven-rag-search-contract.md`

**Reason:** TASK-3500 changes the lasting public meaning of `use_semantic=True` and makes MCP a media-confined consumer of the process-wide profile-driven RAG runtime. ADR-084 records that service contract, runtime ownership, and score/degradation provenance.

---

## File map

- Modify `tldw_chatbook/RAG_Search/simplified/active_config.py`: add the lightweight shared search-mode normalizer/resolver and use it in full config resolution.
- Modify `tldw_chatbook/Library/library_local_rag_search_service.py`: delegate unknown-mode normalization to the shared helper.
- Modify `tldw_chatbook/RAG_Search/simplified/rag_service.py`: support the existing single-key `$in` metadata filter at all semantic/keyword post-filter sites.
- Modify `tldw_chatbook/RAG_Search/simplified/search_service.py`: remove eager construction, add profile routing, lazy shared-runtime acquisition, media confinement, and one result formatter.
- Modify `tldw_chatbook/MCP/tools.py` and `tldw_chatbook/MCP/server.py`: preserve the API while routing true/omitted requests through the profile-aware adapter and documenting the compatibility rule.
- Modify `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py`: centralize reranker setup, clear stale state, and disclose safe construction failure on returned base results.
- Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`: carry score kind and preserved vector score into the shared weak-match evaluator.
- Modify `tldw_chatbook/Library/library_rag_state.py`: correct the stale duck-typing docstring after the MCP shim gains provenance fields.
- Modify `Docs/Design/MCP.md` and `Docs/User_Guide/mcp.md`: document true/omitted profile routing and false keyword override.
- Add `Tests/RAG/simplified/test_metadata_filter_matching.py`.
- Modify `Tests/RAG/test_active_config_resolution.py`, `Tests/Library/test_library_rag_mode_resolution.py`, `Tests/RAG/simplified/test_search_service.py`, `Tests/RAG_Search/test_reranker_construction.py`, `Tests/MCP/test_rag_search_tool.py`, `Tests/MCP/test_builtin_tool_imports.py`, `Tests/MCP/test_mcp_documentation_contract.py`, and `Tests/UI/test_mcp_inspector.py`.
- Modify the TASK-3500 task file only during closeout, after local evidence is recorded.

No new production module, dependency, public field, response key, configuration key, schema migration, or ADR is needed.

### Task 0: Freeze the approved plan before production work

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-23-task-3500-mcp-profile-driven-rag-search-design.md`
- Add: `Docs/superpowers/plans/2026-08-23-task-3500-mcp-profile-driven-rag-search.md`
- Modify: `backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md`

- [x] **Step 1: Record the approved spec/base**

Set the spec status to approved and its reviewed base to `a84e6ba09`.

- [x] **Step 2: Add the mandatory Backlog implementation plan**

Use the Backlog CLI to add a concise `## Implementation Plan` linking this
detailed plan and recording:

```text
ADR required: yes
ADR path: backlog/decisions/084-mcp-profile-driven-rag-search-contract.md
Reason: TASK-3500 changes the lasting MCP request/runtime contract.
```

Then inspect the rendered task:

```bash
backlog task 3500 --plain
```

Expected: status remains `In Progress`; all six ACs remain unchecked; the
external plan link and ADR disposition are present before code changes.

- [x] **Step 3: Commit the approved planning artifacts**

```bash
git add Docs/superpowers/specs/2026-08-23-task-3500-mcp-profile-driven-rag-search-design.md Docs/superpowers/plans/2026-08-23-task-3500-mcp-profile-driven-rag-search.md "backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md"
git commit -m "docs(rag): plan TASK-3500 implementation"
```

- [x] **Step 4: Confirm a clean pre-implementation branch**

```bash
git status --short --branch
```

Expected: no uncommitted files; the branch contains only approved design/ADR
and plan commits before Task 1 starts.

### Task 1: Establish one lightweight active search-mode rule

**Files:**
- Modify: `Tests/RAG/test_active_config_resolution.py:204-307`
- Modify: `Tests/Library/test_library_rag_mode_resolution.py:235-255`
- Modify: `tldw_chatbook/RAG_Search/simplified/active_config.py:31-195,263-265`
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py:95-116,1129-1154`

- [x] **Step 1: Pin and implement the normalizer as one RED/GREEN slice**

First add `test_normalize_rag_search_mode_accepts_only_known_exact_values`,
parameterized over `plain`, `semantic`, `hybrid`, unknown strings, uppercase,
`None`, and a non-string.

Run only that node:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_active_config_resolution.py::test_normalize_rag_search_mode_accepts_only_known_exact_values -q
```

Expected RED: collection/import fails because `normalize_rag_search_mode` does
not exist. Add only `_RAG_SEARCH_MODES` and `normalize_rag_search_mode()` from
Step 3 below, rerun the exact node, and expect PASS.

- [x] **Step 2: Pin active-profile/env resolution as its own RED/GREEN slice**

Add a parameterized coupling test covering stored `plain`, `semantic`, `hybrid`, an unknown stored value, a valid `RAG_SEARCH_MODE` override, and an unknown override. Pin both the narrow resolver and full config resolver to the same expected result:

```python
@pytest.mark.parametrize(
    ("stored_mode", "env_mode", "expected"),
    [
        ("plain", None, "plain"),
        ("semantic", None, "semantic"),
        ("hybrid", None, "hybrid"),
        ("future-mode", None, "semantic"),
        ("plain", "hybrid", "hybrid"),
        ("plain", "future-mode", "semantic"),
    ],
)
def test_search_mode_only_resolution_agrees_with_full_config(
    active, monkeypatch, stored_mode, env_mode, expected
):
    ...
    assert resolve_active_rag_search_mode() == expected
    assert resolve_active_rag_config().search.default_search_mode == expected
```

Add a fresh-interpreter check mirroring `test_top_k_only_resolution_does_not_import_torch`, calling `resolve_active_rag_search_mode()` and expecting `NO_TORCH`.

Run only the coupling node first:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_active_config_resolution.py::test_search_mode_only_resolution_agrees_with_full_config -q
```

Expected RED: import/attribute failure for missing
`resolve_active_rag_search_mode`, after the normalizer node is already green.
Add only the resolver and `_apply_env_overrides()` delegation below; rerun and
expect all parameter cases PASS. Then run the fresh-process node separately and
expect `NO_TORCH`.

- [x] **Step 3: Use these minimal shared mode implementations**

In `active_config.py`, add:

```python
_RAG_SEARCH_MODES = frozenset({"plain", "semantic", "hybrid"})


def normalize_rag_search_mode(value: object) -> str:
    """Return a supported search mode, defaulting unknown values to semantic."""
    return value if isinstance(value, str) and value in _RAG_SEARCH_MODES else "semantic"


def resolve_active_rag_search_mode() -> str:
    """Resolve only the active profile's search mode without building RAG runtime state."""
    profile = _resolved_active_profile()
    base = (
        profile.rag_config.search.default_search_mode
        if profile
        else RAGConfig().search.default_search_mode
    )
    return normalize_rag_search_mode(os.getenv("RAG_SEARCH_MODE") or base)
```

Change `_apply_env_overrides()` to call the same normalizer:

```python
config.search.default_search_mode = normalize_rag_search_mode(
    os.getenv("RAG_SEARCH_MODE") or config.search.default_search_mode
)
```

Do not strip or lowercase values; exact known strings retain Library's current behavior and everything else safely becomes `semantic`.

- [x] **Step 4: Pin and implement Library delegation as one RED/GREEN slice**

In the Library mode test, monkeypatch the module's imported normalizer and
assert `_resolve_profile_search_mode()` delegates to it, so a future local
tuple cannot silently return.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_rag_mode_resolution.py::test_resolve_profile_search_mode_delegates_normalization -q
```

Expected RED: the monkeypatched normalizer is never called and the old local
tuple returns its own answer. Then import `normalize_rag_search_mode`, remove
`_PROFILE_SEARCH_MODES`, and end `_resolve_profile_search_mode()` with:

```python
return normalize_rag_search_mode(mode)
```

Update its return docstring to say `plain`, `semantic`, or `hybrid` instead of naming the deleted tuple. Rerun the exact node and expect PASS.

- [x] **Step 5: Run the aggregate focused mode tests**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_active_config_resolution.py Tests/Library/test_library_rag_mode_resolution.py -q
```

Expected: all tests PASS, including both fresh-process no-`torch` checks.

- [x] **Step 6: Commit the shared rule**

```bash
git add Tests/RAG/test_active_config_resolution.py Tests/Library/test_library_rag_mode_resolution.py tldw_chatbook/RAG_Search/simplified/active_config.py tldw_chatbook/Library/library_local_rag_search_service.py
git commit -m "feat(rag): share active profile search mode"
```

### Task 2: Make the existing media-type `$in` filter real

**Files:**
- Add: `Tests/RAG/simplified/test_metadata_filter_matching.py`
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py:1494-1556,3085-3230`

- [x] **Step 1: Pin and implement the value matcher as one RED/GREEN slice**

Cover exact scalar equality, exact mapping equality without `$in`, valid single/multi-value `$in`, and malformed `$in` (string, mapping, extra operator key, non-collection) failing closed. Then exercise all three production sites:

Run the matcher node before adding the helper:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_metadata_filter_matching.py::test_metadata_filter_value_matching -q
```

Expected RED: import fails because `_metadata_filter_value_matches` does not
exist. Add only this helper, rerun the node, and expect PASS:

```python
def _metadata_filter_value_matches(actual: Any, expected: Any) -> bool:
    """Match exact values plus the existing single-key ``$in`` form."""
    if not isinstance(expected, abc.Mapping) or "$in" not in expected:
        return actual == expected
    if set(expected) != {"$in"}:
        return False
    allowed = expected["$in"]
    if (
        not isinstance(allowed, abc.Collection)
        or isinstance(allowed, (str, bytes, bytearray, abc.Mapping))
    ):
        return False
    try:
        return actual in allowed
    except (TypeError, ValueError):
        return False
```

- [x] **Step 2: Pin each production call site after the helper is green**

Add and run these nodes one at a time, before wiring that site:

1. `_semantic_search(..., filter_metadata={"media_type": {"$in": ["pdf", "video"]}})` keeps only matching real `SearchResult` rows from a stub vector store.
2. `_process_keyword_results_basic(...)` keeps the matching media row.
3. `_create_keyword_result_with_citations(...)` keeps the matching row and rejects a non-member.

Use `RAGService.__new__` plus narrow embedding/vector seams; do not initialize a model or replace the result processors with test-only copies.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_metadata_filter_matching.py::test_semantic_search_uses_membership_filter -q
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_metadata_filter_matching.py::test_keyword_basic_uses_membership_filter -q
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_metadata_filter_matching.py::test_keyword_citations_use_membership_filter -q
```

Expected RED for each node in turn: the matching row is excluded because that
site still compares the literal `$in` mapping for equality. Replace only that
site's `item_meta.get(k) == v` / `r.metadata.get(k) == v` expression with the
helper, rerun its exact node to PASS, then move to the next site. Preserve the
semantic site's current `metadata.get` behavior and the keyword sites' current
`if k in item_meta` behavior.
This sequencing is the mutation evidence for all three call-site wires.

- [x] **Step 3: Document the exact supported filter contract**

Update the public/filter parameter documentation at every owner that currently
says equality-only: `RAGService.search`, `_semantic_search_scoped`,
`_semantic_search`, `_process_keyword_results_basic`, and
`_create_keyword_result_with_citations`. Each must say exact equality plus the
single-key `$in` membership form; do not advertise a general operator language.

- [x] **Step 4: Run focused engine coverage**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_metadata_filter_matching.py Tests/RAG_Search/test_hybrid_allowlist_pushdown.py Tests/RAG_Search/test_keyword_leg_pushdown.py -q
```

Expected: all tests PASS; existing allowlist and keyword pushdown behavior remains unchanged.

- [x] **Step 5: Commit the root fix**

```bash
git add Tests/RAG/simplified/test_metadata_filter_matching.py tldw_chatbook/RAG_Search/simplified/rag_service.py
git commit -m "fix(rag): honor media type membership filters"
```

### Task 3: Route MCP through the active profile and shared runtime

**Files:**
- Modify: `Tests/RAG/simplified/test_search_service.py:31-355`
- Modify: `Tests/MCP/test_rag_search_tool.py:21-105,123-199`
- Modify: `tldw_chatbook/RAG_Search/simplified/search_service.py:8-109`
- Modify: `tldw_chatbook/MCP/tools.py:273-315`

- [x] **Step 1: Make construction lazy in an isolated RED/GREEN slice**

Use the real `SimplifiedRAGSearchService(media_db)` constructor. Monkeypatch
the obsolete factory with `raising=False` to fail if called (so the same test
still works once the import is removed), then assert construction succeeds,
stores the real database, and leaves `rag_service is None`.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_constructor_does_not_build_enhanced_runtime -q
```

Expected RED: the fail-fast obsolete factory is called from `__init__`. Remove
`load_settings`/`create_rag_service`, add only the constructor below, rerun the
exact node, and expect PASS.

In `search_service.py`, import `asyncio`, `get_shared_rag_service`, and `resolve_active_rag_search_mode`. Constructor becomes:

```python
def __init__(self, media_db: MediaDatabase):
    self.media_db = media_db
    self.rag_service = None
```

- [x] **Step 2: Add plain profile routing as its own RED/GREEN slice**

Add `test_profile_plain_routes_to_keyword_without_resolving_shared_runtime`,
using the real MediaDatabase keyword path and a fail-fast shared resolver.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_profile_plain_routes_to_keyword_without_resolving_shared_runtime -q
```

Expected RED: `profile_search` is missing. Add only `profile_search()` with the
plain branch and a temporary call to the not-yet-implemented enhanced helper
for other normalized modes. Rerun the exact node and expect PASS.

- [x] **Step 3: Add semantic/hybrid confinement one parameterized RED/GREEN slice**

Add `test_profile_enhanced_routes_are_media_confined`, parameterized for
`semantic`/`hybrid` and single/multi-value media filters. Assert exact
`search_type`, `top_k`, `filter_metadata`, and
`metadata_allowlist={"source_type": ("media",)}`.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_profile_enhanced_routes_are_media_confined -q
```

Expected RED: `_enhanced_search` is missing. Add the helper and formatter below,
then rerun the exact node and expect every parameter case PASS.

For this confinement slice, use the injected runtime seam only; leave
production shared-runtime acquisition deliberately unimplemented so Step 4
can prove its own lifecycle/fallback behaviors:

```python
async def profile_search(self, query, limit=10, media_types=None):
    mode = resolve_active_rag_search_mode()
    if mode == "plain":
        return await self.keyword_search(query, limit, media_types)
    return await self._enhanced_search(query, limit, media_types, search_type=mode)

async def _enhanced_search(self, query, limit, media_types, *, search_type):
    service = self.rag_service
    if service is None:
        raise RuntimeError("shared runtime acquisition is not implemented")
    filter_metadata = (
        {"media_type": {"$in": media_types}} if media_types else None
    )
    results = await service.search(
        query=query,
        top_k=limit,
        search_type=search_type,
        filter_metadata=filter_metadata,
        metadata_allowlist={"source_type": ("media",)},
    )
    return self._format_enhanced_results(results)
```

Keep `semantic_search()` as an explicit semantic wrapper around `_enhanced_search`. Extract the existing mapping loop unchanged into `_format_enhanced_results()`. Do not catch `service.search()` exceptions, cache the production shared service, or mutate its config.

- [x] **Step 4: Pin lifecycle, fallback, error, and metadata behavior individually**

After the injected enhanced route exists, add/run these exact nodes in order:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_each_enhanced_request_resolves_current_shared_service -q
```

Expected RED: the deliberate “shared runtime acquisition is not implemented”
error is raised. Replace only the `service = self.rag_service` acquisition with
an explicit injection-or-shared lookup (do not use truthiness):

```python
service = (
    self.rag_service
    if self.rag_service is not None
    else await asyncio.to_thread(get_shared_rag_service)
)
```

Rerun the lifecycle node and expect PASS with two getter calls/two different
services. Then add the unavailable-`None` node:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_unavailable_shared_runtime_falls_back_to_unscored_keyword_search -q
```

Expected RED: `None.search` raises. Add only this fallback, rerun, and expect
PASS:

```python
if service is None:
    return await self.keyword_search(query, limit, media_types)
```

Then add the acquisition exception node:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_shared_runtime_acquisition_exception_falls_back_to_keyword_search -q
```

Expected RED: the resolver exception propagates. Wrap only the acquisition in
`try/except`, log the unavailable runtime, set `service = None`, and reuse the
just-green `None` fallback. Rerun and expect PASS.

Finally add the search-exception and metadata nodes as GREEN compatibility pins
against the implementation already reached in Step 3:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_enhanced_search_exception_propagates -q
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py::test_enhanced_formatter_preserves_complete_metadata -q
```

Expected: both PASS. They guard the two intentional negative constraints: do
not broaden the acquisition catch around `service.search()`, and do not drop
nested fusion/reranking provenance in formatting.

- [x] **Step 5: Preserve the public MCP switch in its own RED/GREEN slice**

Update the MCP tool stub to expose `profile_search()`. Add/rename exact tests
`test_perform_rag_search_default_uses_profile_search` and
`test_perform_rag_search_false_forces_keyword_search`, preserving the exact
seven-key successful response assertion.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_rag_search_tool.py::test_perform_rag_search_default_uses_profile_search Tests/MCP/test_rag_search_tool.py::test_perform_rag_search_false_forces_keyword_search -q
```

Expected RED: the default test records a `semantic_search` call instead of a
`profile_search` call, while the false test already passes and protects the
compatibility override. Change only `MCPTools.perform_rag_search`:

In `MCPTools.perform_rag_search`, route only the branch:

```python
if use_semantic:
    results = await self.rag_service.profile_search(
        query=query, limit=limit, media_types=media_types
    )
else:
    results = await self.rag_service.keyword_search(
        query=query, limit=limit, media_types=media_types
    )
```

Keep the existing formatter and `[{'error': ...}]` exception boundary. Update the Args text: false forces media keyword search; true/omitted follows the active profile.

- [x] **Step 6: Run aggregate adapter, tool, and shared-runtime tests**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/simplified/test_search_service.py Tests/MCP/test_rag_search_tool.py -q
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_ingestion_indexing.py -q -k SharedRagService
```

Expected: all selected tests PASS. The plain-mode test proves no model/runtime construction; the real MediaDatabase keyword tests prove the fallback/override path.

- [x] **Step 7: Commit the MCP routing change**

```bash
git add Tests/RAG/simplified/test_search_service.py Tests/MCP/test_rag_search_tool.py tldw_chatbook/RAG_Search/simplified/search_service.py tldw_chatbook/MCP/tools.py
git commit -m "feat(mcp): follow active RAG profile"
```

### Task 4: Make reranker-construction degradation truthful and reset-safe

**Files:**
- Modify: `Tests/RAG_Search/test_reranker_construction.py:113-241,290-390`
- Modify: `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py:151-170,310-362,436-466`

- [x] **Step 1: Pin construction state as one RED/GREEN slice**

Add `test_reranker_construction_failure_records_safe_unavailability`. Patch the
factory to raise `RuntimeError("secret-token-value")`; assert the service stays
usable, `reranker is None`, the reason contains `RuntimeError`, and neither the
reason nor captured construction warning contains the fake secret.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_reranker_construction.py::test_reranker_construction_failure_records_safe_unavailability -q
```

Expected RED: `_reranker_unavailable_reason` is missing and the warning exposes
the exception message. Add `_configure_reranker()` and call it from `__init__`,
then rerun the exact node to PASS.

- [x] **Step 2: Centralize reranker setup with this implementation**

Add one owner method and call it from both `__init__` and `switch_profile`:

```python
def _configure_reranker(self) -> None:
    self.reranker = None
    self._reranker_unavailable_reason = None
    if not (self.enable_reranking and self.reranking_config):
        return
    try:
        self.reranker = create_reranker_from_config(self.reranking_config)
        logger.info(f"Initialized {self.reranking_config.strategy} reranker")
    except Exception as exc:
        exception_name = type(exc).__name__
        self._reranker_unavailable_reason = (
            f"reranker construction failed ({exception_name})"
        )
        logger.warning(
            f"Failed to construct {self.reranking_config.strategy} reranker "
            f"({exception_name}); continuing without reranking"
        )
```

Do not log or expose `str(exc)` on this construction path.

- [x] **Step 3: Pin one-result disclosure as one RED/GREEN slice**

Add `test_construction_failure_tags_one_base_result_without_mutating_original`.
Patch only `EnhancedRAGService.search` to return one deterministic real
`SearchResult`; exercise real `EnhancedRAGServiceV2.search()`.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_reranker_construction.py::test_construction_failure_tags_one_base_result_without_mutating_original -q
```

Expected RED: the returned result has no `reranking_skipped` disclosure. Add
the branch below, rerun, and expect the returned copy to be tagged while the
original result metadata remains unchanged.

After the existing reranker execution branch, add:

After the existing reranker execution branch:

```python
if (
    should_rerank
    and self.reranker is None
    and self._reranker_unavailable_reason
    and results
):
    results = _tag_first_result(
        results, "reranking_skipped", self._reranker_unavailable_reason
    )
```

This deliberately includes a single result, per ADR-084. Do not create a tag when reranking is disabled or no base result exists.

- [x] **Step 4: Pin profile-switch cleanup as one RED/GREEN slice**

Add `test_switch_profile_clears_stale_reranker_and_unavailability_reason`,
using this exact state sequence:

1. Start from the construction-failure service produced in Step 1 and assert
   `reranker is None` plus a non-empty safe failure reason.
2. Switch to a valid reranking profile whose factory now succeeds; assert a
   reranker installs and the prior failure reason becomes `None`.
3. Switch to a profile with no reranking config; assert both the reranker and
   reason are `None`.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_reranker_construction.py::test_switch_profile_clears_stale_reranker_and_unavailability_reason -q
```

Expected RED: the valid profile can retain the prior construction-failure
reason, and the disabled profile can retain the previous reranker because the
duplicated switch block does not reset state first. Replace that block with
`self._configure_reranker()`, rerun, and expect every transition above to PASS.

- [x] **Step 5: Run all shared reranking degradation coverage**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_reranker_construction.py Tests/RAG_Search/test_reranker_degraded_paths.py -q
```

Expected: all tests PASS; existing runtime failure/degraded and cache-poisoning behavior remains green.

- [x] **Step 6: Commit shared-runtime honesty**

```bash
git add Tests/RAG_Search/test_reranker_construction.py tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py
git commit -m "fix(rag): disclose unavailable reranker setup"
```

### Task 5: Teach the MCP inspector the shared score vocabulary

**Files:**
- Modify: `Tests/UI/test_mcp_inspector.py:1811-1875`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:30-34,502-554`
- Modify: `tldw_chatbook/Library/library_rag_state.py:1692-1722`

- [x] **Step 1: Pin one fused-row defect before changing the shim**

Add `test_extract_scored_rows_reads_nested_hybrid_vector_score`, using fused
top-level score `0.016` and nested vector score `0.8`. Assert the extracted
shim says `hybrid_fusion`/`0.8` and the summary emits no weak notice.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_mcp_inspector.py::TestSummarizeToolResultAllWeakNotice::test_extract_scored_rows_reads_nested_hybrid_vector_score -q
```

Expected RED: `_ScoredRow` has no `score_kind`/`vector_score`, and the fused
`0.016` is misread as weak vector similarity. Implement the shim/extraction
change in Step 2, rerun this node, and expect PASS.

- [x] **Step 2: Extend the duck-typed shim with shared provenance**

Import `library_rag_result_score_kind` from `Library.library_rag_score_kinds`. Change the shim to:

```python
__slots__ = ("score", "score_kind", "vector_score")

def __init__(self, score: object, score_kind: str, vector_score: float | None) -> None:
    self.score = (
        score
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else None
    )
    self.score_kind = score_kind
    self.vector_score = vector_score
```

In `_extract_scored_rows`, explicitly pass nested metadata first:

```python
score_kind, vector_score = library_rag_result_score_kind(
    row.get("metadata"), row
)
scored_rows.append(_ScoredRow(row.get("score"), score_kind, vector_score))
```

Do not teach `library_rag_result_score_kind()` to recurse or duplicate score rules in MCP.

- [x] **Step 3: Add the remaining provenance matrix after the seam is reachable**

Add direct `_extract_scored_rows` assertions plus summary assertions for:

| Row | Expected weak notice |
| --- | --- |
| vector score `0.1` | yes |
| hybrid fused score `0.016`, nested vector score `0.8` | no |
| hybrid fused score `0.016`, nested vector score `0.1` | yes |
| FTS-only hybrid, nested vector score `None` | no |
| reranker score (including an out-of-cosine value) | no |
| keyword score `None` | no |
| `reranking_skipped` plus unchanged vector score `0.1` | yes |

Use the real metadata shapes written by fusion/reranking. Assert the extracted
shim's `score_kind` and `vector_score`, not only the rendered sentence.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_mcp_inspector.py::TestSummarizeToolResultAllWeakNotice -q
```

Expected: PASS. Passing only the outer row would make the fused and reranker
cases fail, so the matrix is the mutation guard for nested metadata ordering.

- [x] **Step 4: Correct the stale shared-helper docstring**

Update `library_rag_all_matches_weak()`'s Args note so it describes the MCP shim as carrying `.score`, `.score_kind`, and `.vector_score` rather than claiming it has only a score slot.

- [x] **Step 5: Run inspector and Library score-kind coverage**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_mcp_inspector.py Tests/Library/test_library_rag_state.py -q
```

Expected: all tests PASS; non-RAG row shapes and empty/error summaries remain unchanged.

- [x] **Step 6: Commit truthful score interpretation**

```bash
git add Tests/UI/test_mcp_inspector.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py tldw_chatbook/Library/library_rag_state.py
git commit -m "fix(mcp): interpret RAG score provenance"
```

### Task 6: Publish and pin the compatibility rule

**Files:**
- Modify: `Tests/MCP/test_builtin_tool_imports.py:185-225`
- Modify: `Tests/MCP/test_mcp_documentation_contract.py`
- Modify: `tldw_chatbook/MCP/server.py:660-672`
- Modify: `Docs/Design/MCP.md:198-207`
- Modify: `Docs/User_Guide/mcp.md:43-62,261-299`

- [x] **Step 1: Write failing public-copy contract tests**

In the built-in manifest test, assert `search_rag` retains the same four properties, `use_semantic.default is True`, and the AST-derived tool description names the active RAG profile.

In the documentation contract, normalize both MCP documents and require all three facts: false forces media keyword search; true or omission follows the active profile; the recognized modes are `plain`, `semantic`, and `hybrid`. Add a User Guide assertion that weak-match disclosure uses actual similarity provenance: a hybrid row may use its preserved vector leg, while FTS-only hybrid, reranker, and unscored keyword rows are not cosine-banded.

- [x] **Step 2: Run the documentation contracts and record RED evidence**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py -q
```

Expected: FAIL only on the new profile-routing copy assertions; inventory/schema-shape assertions remain green.

- [x] **Step 3: Update tool and user-facing documentation**

Change the standalone registration's first docstring line to:

```python
"""Search media using the active RAG profile unless keyword search is forced."""
```

In both documents, state:

```text
`use_semantic` remains a boolean compatibility switch: `false` forces media
keyword search; `true` or omission follows the active RAG profile's `plain`,
`semantic`, or `hybrid` search mode.
```

Keep the standalone inventory count/order untouched. In the user guide, place this under “Standalone behavior and controls,” outside the exact-inventory block.

Also replace the existing inspector paragraph that says the notice fires only
for “scored (semantic) rows.” Say instead that it considers only rows carrying
an actual vector similarity: ordinary semantic rows use their score, hybrid
rows use the preserved vector leg when present, and FTS-only hybrid, reranker,
and unscored keyword rows do not trigger a cosine-similarity claim.

- [x] **Step 4: Run MCP documentation and tool contracts**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py Tests/MCP/test_rag_search_tool.py -q
```

Expected: all tests PASS with the public field names/default and seven-key result rows unchanged.

- [x] **Step 5: Commit public documentation**

```bash
git add Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py tldw_chatbook/MCP/server.py Docs/Design/MCP.md Docs/User_Guide/mcp.md
git commit -m "docs(mcp): explain profile-driven RAG search"
```

### Task 7: Focused verification and TASK-3500 closeout

**Files:**
- Verify all files above
- Modify: `backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md`

- [x] **Step 1: Run the focused behavioral suite**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_active_config_resolution.py Tests/Library/test_library_rag_mode_resolution.py Tests/RAG/simplified/test_metadata_filter_matching.py Tests/RAG/simplified/test_search_service.py Tests/RAG_Search/test_reranker_construction.py Tests/RAG_Search/test_reranker_degraded_paths.py Tests/MCP/test_rag_search_tool.py Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py Tests/UI/test_mcp_inspector.py Tests/Library/test_library_rag_state.py Tests/MCP/test_local_control_service.py -q --tb=short
```

Expected: all tests PASS. Record exact pass count, duration, warnings, and any sandbox-only pytest temp cleanup messages separately.
`Tests/Library/test_library_rag_state.py` owns the helper-hardening coverage;
the score-kind filename originally proposed for this command does not exist.

- [x] **Step 2: Freshly rerun the load-bearing engine/lifecycle guards**

Run these separately so an ingestion-only `-k` filter cannot deselect either
engine module:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_hybrid_allowlist_pushdown.py -q --tb=short
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG_Search/test_keyword_leg_pushdown.py -q --tb=short
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_ingestion_indexing.py -q --tb=short -k SharedRagService
```

Expected: all three commands PASS, freshly proving all-leg media confinement,
keyword-leg filtering, and shared singleton reset/generation ownership.

- [x] **Step 3: Run static and patch hygiene checks on changed Python files**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/RAG_Search/simplified/active_config.py tldw_chatbook/Library/library_local_rag_search_service.py tldw_chatbook/RAG_Search/simplified/rag_service.py tldw_chatbook/RAG_Search/simplified/search_service.py tldw_chatbook/MCP/tools.py tldw_chatbook/MCP/server.py tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py tldw_chatbook/Library/library_rag_state.py Tests/RAG/test_active_config_resolution.py Tests/Library/test_library_rag_mode_resolution.py Tests/RAG/simplified/test_metadata_filter_matching.py Tests/RAG/simplified/test_search_service.py Tests/RAG_Search/test_reranker_construction.py Tests/MCP/test_rag_search_tool.py Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py Tests/UI/test_mcp_inspector.py
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/RAG_Search/simplified/active_config.py tldw_chatbook/Library/library_local_rag_search_service.py tldw_chatbook/RAG_Search/simplified/rag_service.py tldw_chatbook/RAG_Search/simplified/search_service.py tldw_chatbook/MCP/tools.py tldw_chatbook/MCP/server.py tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py tldw_chatbook/Library/library_rag_state.py Tests/RAG/test_active_config_resolution.py Tests/Library/test_library_rag_mode_resolution.py Tests/RAG/simplified/test_metadata_filter_matching.py Tests/RAG/simplified/test_search_service.py Tests/RAG_Search/test_reranker_construction.py Tests/MCP/test_rag_search_tool.py Tests/MCP/test_builtin_tool_imports.py Tests/MCP/test_mcp_documentation_contract.py Tests/UI/test_mcp_inspector.py
git diff --check a84e6ba09...HEAD
```

Expected: all commands exit `0`; Ruff and `git diff --check` report no findings. Repository CI and the full suite are intentionally excluded per user direction.

**Closeout deviation (2026-08-24).** The prescribed whole-file Ruff gates were
run over the dynamic `git diff --name-only a84e6ba09...HEAD -- '*.py'` set
(21 HEAD files; 20 have an `a84e6ba09` counterpart and one is new). They do
not exit zero because the checkout retains baseline debt: the changed-file
whole-file check reports seven `E702` findings in
`Tests/RAG/test_active_config_resolution.py` and seven `E402` findings in
`library_local_rag_search_service.py`; the pinned-base snapshots report those
categories plus one `E402` and two `F401` findings. Whole-file format likewise
still reports baseline candidates. The sole new Python file,
`test_metadata_filter_matching.py`, was formatted and passed its focused test
and `ruff format --check`. A TASK-3500 `E402` on the shared mode import was
repaired by relocating that import into the existing top import block
(`fix(rag): keep shared mode import ordered`). After the style closeout, a
line-differential audit intersects each changed Python file's task-added lines
from `git diff --unified=0 a84e6ba09` with the current-source lines changed by
`ruff format --diff`; it reports zero remaining TASK-3500-owned formatter
overlaps. This differential evidence is the acceptance gate for static hygiene,
and `git diff --check a84e6ba09...HEAD` remains required to exit zero.

**Review-loop correction (2026-08-24).** A closeout review reopened Task 7 to
make the behavioral command reproducible, correct the vacuous-selector exit
evidence, and format only TASK-3500-owned lines. The completed plan records the
additional focused style shard and the post-fix line-differential audit; no
semantic production change was introduced by this review loop.

- [x] **Step 4: Review the final diff against scope and ADR-084**

```bash
git diff --stat a84e6ba09...HEAD
git diff a84e6ba09...HEAD -- tldw_chatbook/RAG_Search/simplified/search_service.py tldw_chatbook/MCP/tools.py tldw_chatbook/RAG_Search/simplified/rag_service.py tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py
```

Confirm: no Library multi-source routing, no MCP-local service cache, no eager enhanced construction, no non-media allowlist, no fabricated keyword/vector score, no raw construction exception disclosure, and no public schema/key changes.

- [x] **Step 5: Record TDD discrimination evidence**

In the task notes, preserve the actual RED results collected in Tasks 1-6. Together they prove the focused guards detect: missing/changed mode arms, removed media confinement, equality-only `$in` regression at semantic and keyword sites, stale shared-service reuse, missing reranker construction disclosure, and blind vector-score defaults. Do not replace these with generic “tests added” prose.

No live model/provider UAT is required: the change does not alter a provider contract, and the focused tests exercise the runtime boundary with deterministic injected shared services plus the real MediaDatabase. Do not download embedding/reranker models or call a cloud provider merely for this task.

- [x] **Step 6: Complete TASK-3500 only with actual local evidence**

Use Backlog CLI to add concise implementation notes naming the approach, ADR-084, exact commands/counts, warnings, changed files, and the explicit user-directed CI exclusion. State whether a new lesson was needed; do not invent one. Replace each `<actual ...>` token below with observed evidence before running it:

```bash
backlog task edit 3500 --notes "Implemented profile-driven MCP media retrieval through the shared runtime; added exact and membership metadata filtering; made reranker construction degradation safe and reset-clean; and taught the inspector shared score provenance. ADR: backlog/decisions/084-mcp-profile-driven-rag-search-contract.md. Verification: focused suite <actual pass count/time>; hybrid allowlist <actual pass count/time>; keyword pushdown <actual pass count/time>; shared lifecycle <actual pass count/time>; Ruff and diff checks <actual results>; warnings <actual warnings or none>. Repository CI/full suite excluded by explicit user direction. Lessons: <actual disposition>." --plain
```

Then check all six ACs, inspect the rendered task, and only then set `Done`:

```bash
backlog task edit 3500 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --plain
backlog task 3500 --plain
backlog task edit 3500 -s Done --plain
```

- [x] **Step 7: Commit task completion metadata**

```bash
git add "backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md"
git commit -m "docs: complete TASK-3500"
```

- [x] **Step 8: Confirm final branch state**

```bash
git status --short --branch
backlog task 3500 --plain
git log --oneline --decorate -12
```

Expected: clean worktree; TASK-3500 is `Done` with six checked ACs, ADR/implementation notes, and exact local evidence; commits are based on pinned base `a84e6ba09`.
