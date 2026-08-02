# TASK-596.1 Remote Model Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users explicitly discover and download pinned Hugging Face GGUF files through the managed model store without activating or presenting arbitrary models as runtime-compatible.

**Architecture:** Add one narrow `httpx` adapter that parses bounded Hugging Face metadata into immutable values and a one-item managed catalog. A lazy Textual Remote view owns search, resolution, and the existing preflight/provision lifecycle. Small additive changes let shared acquisition install without activation and let shared inventory/consent widgets represent an unassigned model honestly.

**Tech Stack:** Python 3.11+, `httpx`, Textual, existing `ModelArtifactService` / `ArtifactAcquisitionService`, pytest, `httpx.MockTransport`.

**Design spec:** `Docs/superpowers/specs/2026-08-01-task-596-1-remote-model-discovery-design.md`

**ADR required:** no

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already owns remote artifact provenance, managed acquisition, activation, and runtime boundaries. This task adds a Hugging Face adapter and an install-without-activation option inside that boundary.

**Baseline:** The affected pre-change suite passes: 70 tests across stream fetch, provision/install, shared model widgets/state, Installed, and Lab model-screen adoption. Loopback fixture tests require permission to bind local ports in the Codex sandbox.

**Execution precondition:** Before Task 1, fetch and rebase this documentation
branch onto the latest `origin/dev`, confirm TASK-596.1/spec/plan paths still have
no collision, and rerun the same 70-test baseline. Stop for a real product
conflict; resolve only mechanical documentation conflicts without changing the
approved behavior. Then use the Backlog CLI to add this plan and the approved
design spec to TASK-596.1's Documentation field and add the implementation-plan
summary, including the ADR decision above, before changing production code.

---

## File map

**Create**

- `tldw_chatbook/Model_Artifacts/remote_huggingface.py` — bounded Hugging Face metadata adapter, GGUF grouping, descriptor/catalog/source-map construction.
- `tldw_chatbook/UI/Screens/model_remote_view.py` — idle Remote UI, generation-fenced workers, managed install lifecycle.
- `Tests/Model_Artifacts/test_remote_huggingface.py` — adapter, parsing, grouping, identity, descriptor, and source-map tests.
- `Tests/UI/test_model_remote_view.py` — lazy UI, worker fencing, consent, provision policy, and retry tests.

**Modify**

- `tldw_chatbook/Model_Artifacts/fetch.py` — reject HTTPS-to-HTTP redirect downgrade in the existing per-hop loop.
- `tldw_chatbook/Model_Artifacts/acquisition.py` — add keyword-only `activate: bool = True` to `provision()`.
- `tldw_chatbook/UI/Screens/model_browser_state.py` — carry `activation_allowed` and unassigned-consumer copy.
- `tldw_chatbook/UI/Screens/model_installed_view.py` — pass activation policy while retaining Delete.
- `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py` — optionally omit only Activate.
- `tldw_chatbook/Widgets/ModelArtifacts/plan_panel.py` — separate license/reference copy and neutral integrity footer.
- `tldw_chatbook/Widgets/ModelArtifacts/install_modal.py` — optional required acknowledgment.
- `tldw_chatbook/UI/LLM_Management_Window.py` — mount/map Remote and preserve no-I/O activation.
- `tldw_chatbook/UI/Screens/llm_screen.py` — add the Remote rail row.
- `Tests/Model_Artifacts/test_stream_fetch.py` — downgrade regression.
- `Tests/Model_Artifacts/test_provision_install.py` — install-without-activation regression.
- `Tests/UI/test_model_browser_state.py` — unassigned inventory state.
- `Tests/UI/test_model_installed_view.py` — Delete-without-Activate rendering.
- `Tests/UI/test_model_artifact_widgets.py` — plan copy and acknowledgment gate.
- `Tests/UI/test_llm_screen_lab_adoption.py` — Remote rail/mount integration.
- `backlog/tasks/task-596.1 - Add-bounded-Hugging-Face-GGUF-discovery-to-the-managed-model-browser.md` — plan link, completed ACs, and implementation notes.

Do not export the adapter from `Model_Artifacts/__init__.py`, add a provider registry, add a dependency, or touch `Widgets/HuggingFace/`.

---

### Task 1: Add the bounded Hugging Face metadata adapter

**Files:**

- Create: `tldw_chatbook/Model_Artifacts/remote_huggingface.py`
- Create: `Tests/Model_Artifacts/test_remote_huggingface.py`

- [ ] **Step 1: Write failing tests for fixed-origin request behavior**

Use `httpx.MockTransport` to assert:

- free-text search trims the submitted query and sends it as the `search` query
  parameter in one explicit GET to `https://huggingface.co/api/models` with a
  50-result limit;
- search passes `follow_redirects=False` and attaches a supplied bearer token only to the fixed Hugging Face origin;
- 401/403, 404, 429, timeout, malformed JSON, and bodies over 2 MiB become one `RemoteDiscoveryError(code, retryable)` without raw upstream content.

Representative test shape:

```python
@pytest.mark.asyncio
async def test_search_is_explicit_bounded_and_fixed_origin() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
    )
    assert await adapter.search("  whisper  ", token="secret") == ()
    assert requests[0].url.host == "huggingface.co"
    assert requests[0].url.params["search"] == "whisper"
    assert requests[0].url.params["limit"] == "50"
    assert requests[0].headers["authorization"] == "Bearer secret"
```

- [ ] **Step 2: Run the focused test and verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_remote_huggingface.py -q
```

Expected: FAIL because `remote_huggingface` does not exist.

- [ ] **Step 3: Implement only the adapter values, one error type, and bounded JSON reader**

Keep the public surface small:

```python
@dataclass(frozen=True)
class RemoteModelSummary:
    repository: str
    private: bool
    gated: Literal["none", "auto", "manual"]
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None


class RemoteDiscoveryError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details


class HuggingFaceRemoteAdapter:
    async def search(
        self, query: str, *, token: str | None = None
    ) -> tuple[RemoteModelSummary, ...]: ...


def is_exact_repository(value: str) -> bool: ...
```

The private reader streams decoded bytes and stops above 2 MiB before
`json.loads`. Validate the spec's search/repository/access/text bounds.
Normalize gated values `False`, `"auto"`, and `"manual"`; omit malformed
optional popularity/update fields. Keep the same private bounded reader for
Task 2's repository-resolution request.

- [ ] **Step 4: Add search parsing and exact-ID tests**

Cover private boolean, all gated states, 96-character repository cap, optional
popularity/update omission, the 50-result cap, and exact `owner/repository`
classification. Commit, license, file-count, and candidate tests belong to Task
2 so Task 1 remains independently buildable.

- [ ] **Step 5: Run adapter tests and static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_remote_huggingface.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/remote_huggingface.py Tests/Model_Artifacts/test_remote_huggingface.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/remote_huggingface.py Tests/Model_Artifacts/test_remote_huggingface.py
git commit -m "feat: add bounded Hugging Face model metadata adapter"
```

---

### Task 2: Resolve repositories and GGUF candidates into one managed catalog

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/remote_huggingface.py`
- Modify: `Tests/Model_Artifacts/test_remote_huggingface.py`

- [ ] **Step 1: Write failing repository-resolution and grouping tests**

Assert repository resolution sends exactly
`GET https://huggingface.co/api/models/{owner}/{repository}?blobs=true`, with no
redirect following and a supplied token only on that same origin. Assert the
request query contains `blobs=true`; this is required for LFS size/digest
metadata. Cover exact
`[0-9a-f]{40}` commit validation, the 2,048-entry refusal,
`cardData.license`/all `NOASSERTION` shapes, one LFS-backed single `.gguf`, one
complete three-shard set, malformed/oversized sets, missing LFS metadata,
directory-aware grouping, deterministic path ordering, and first-100 truncation
metadata. Rejected shard members must never reappear as singles. Assert that an
incomplete group produces a bounded warning containing its candidate label and
missing five-digit indexes; if no eligible candidate remains, assert the same
warning is carried in `RemoteDiscoveryError.details`.

- [ ] **Step 2: Run grouping tests and verify failure**

Run the new grouping nodes in `Tests/Model_Artifacts/test_remote_huggingface.py`.
Expected: FAIL because grouping/catalog construction is absent.

- [ ] **Step 3: Implement repository resolution and immutable candidate values**

```python
_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$"
)

@dataclass(frozen=True)
class RemoteGGUFFile:
    upstream_path: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True)
class RemoteGGUFCandidate:
    label: str
    files: tuple[RemoteGGUFFile, ...]
    total_bytes: int

@dataclass(frozen=True)
class ResolvedRemoteModel:
    repository: str
    commit: str
    license_id: str
    review_url: str
    candidates: tuple[RemoteGGUFCandidate, ...]
    total_candidate_count: int
    warnings: tuple[str, ...]
```

Add `HuggingFaceRemoteAdapter.resolve(repository, token=None)` using Task 1's
bounded reader and the exact model-info request above. Retain at most 20
display-safe incomplete-shard warnings; bound candidate labels and list only
missing five-digit indexes. No GGUF header parsing or quantization inference
belongs here.

- [ ] **Step 4: Write failing descriptor/catalog tests**

Assert full canonical artifact-ID SHA-256, commit revision,
`not-declared` variant/precision, unassigned fields, only
`LOCAL_INTEGRITY_RECORDED`, portable managed names, exact sizes/digests,
first-payload `source_url`, fixed-origin pinned source-map URLs, bounded label,
license/review URL, no dependencies, and the compatibility warning.

- [ ] **Step 5: Implement the minimal one-item catalog wrapper**

```python
@dataclass(frozen=True)
class ResolvedRemoteCatalog:
    artifact: ArtifactDescriptor
    sources: Mapping[ArtifactRef, Mapping[str, str]]

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        if ref != self.artifact.reference:
            raise KeyError(ref)
        return self.artifact
```

Expose one pure `build_remote_catalog(resolved, candidate)` function. Do not add
a registry, cache, persistence layer, or model inspection.

- [ ] **Step 6: Run tests and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_remote_huggingface.py -q
git add tldw_chatbook/Model_Artifacts/remote_huggingface.py Tests/Model_Artifacts/test_remote_huggingface.py
git commit -m "feat: map remote GGUF candidates to managed artifacts"
```

---

### Task 3: Add install-only acquisition and HTTPS downgrade protection

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py:1035-1241`
- Modify: `tldw_chatbook/Model_Artifacts/fetch.py:148-190`
- Modify: `Tests/Model_Artifacts/test_provision_install.py`
- Modify: `Tests/Model_Artifacts/test_stream_fetch.py`

- [ ] **Step 1: Write failing install-without-activation tests**

Provision an already installed root with `activate=False`. Assert `core.activate`
is not called, no activate progress is emitted, the root is returned, and no
active selector exists. Keep the existing default-activation test unchanged.

- [ ] **Step 2: Run focused tests and verify failure**

Expected: FAIL because `provision()` does not accept `activate`.

- [ ] **Step 3: Implement the additive keyword**

Add `activate: bool = True` after `progress` and change only the tail:

```python
if not activate:
    return root
activated = await self._run_core_call(
    "activate", root, functools.partial(self._core.activate, root)
)
self._emit_indeterminate_progress(progress_state, "activate", root)
return activated
```

Update docstring, return, progress, and error wording. Do not add another method.

- [ ] **Step 4: Write and run a failing HTTPS downgrade test**

Use `httpx.MockTransport` and a patched egress check. Return
`302 Location: http://storage.example/model.gguf` from an initial HTTPS URL.
Assert `FetchTransportError` occurs before a second request.

- [ ] **Step 5: Add one guard to the existing redirect loop**

```python
if origin.scheme == "https" and current.scheme != "https":
    raise FetchTransportError("HTTPS redirect downgrade")
```

Do not add a new redirect helper or Remote fetcher.

- [ ] **Step 6: Run affected tests and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_stream_fetch.py Tests/Model_Artifacts/test_provision_install.py -q
git add tldw_chatbook/Model_Artifacts/acquisition.py tldw_chatbook/Model_Artifacts/fetch.py Tests/Model_Artifacts/test_provision_install.py Tests/Model_Artifacts/test_stream_fetch.py
git commit -m "feat: support managed install without activation"
```

Expected: PASS. In Codex, permit loopback binding for existing fixture tests.

---

### Task 4: Make shared controls honest for unassigned installs

**Files:**

- Modify: `tldw_chatbook/UI/Screens/model_browser_state.py:59-220`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py:245-285`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py:48-106`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/plan_panel.py:39-76`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/install_modal.py:49-97`
- Modify: `Tests/UI/test_model_browser_state.py`
- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `Tests/UI/test_model_artifact_widgets.py`

- [ ] **Step 1: Write failing state/widget tests**

Assert a non-broken unassigned consumer yields `activation_allowed=False` and
compatibility-not-verified copy even after `ready=True`; broken state still says
Needs repair; other consumers keep current copy; Delete remains with no
Activate; `set_pending()` handles absent Activate; plan copy separates License
and Source review page and uses the neutral footer.

- [ ] **Step 2: Run tests and verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py -q
```

Expected: FAIL on missing state/control inputs and old copy.

- [ ] **Step 3: Implement the single consumer authority**

Add `activation_allowed: bool` to `InventoryRow` and branch before active/ready:

```python
activation_allowed = descriptor.consumer != "unassigned"
if is_broken:
    action_hint = "Needs repair — Repair"
elif not activation_allowed:
    action_hint = "Downloaded · runtime compatibility not verified"
elif item.active:
    ...
```

Do not inspect other sentinel fields in UI code.

- [ ] **Step 4: Add the optional activation flag without losing Delete**

Add `allow_activation: bool = True`. Yield/query Activate only when true; always
yield Delete. Preserve defaults and intent messages.

- [ ] **Step 5: Add optional unknown-license acknowledgment**

Use Textual's existing `Checkbox`:

```python
def __init__(..., required_acknowledgment: str | None = None) -> None:
    self.required_acknowledgment = required_acknowledgment
    self._acknowledged = required_acknowledgment is None

@on(Checkbox.Changed)
def _acknowledgment_changed(self, event: Checkbox.Changed) -> None:
    self._acknowledged = event.value
    self.query_one(f"#{self.confirm_id}", Button).disabled = self.ungrantable
```

Include `not self._acknowledged` in `ungrantable`. Existing callers pass nothing.

- [ ] **Step 6: Run tests and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py -q
git add tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py tldw_chatbook/Widgets/ModelArtifacts/plan_panel.py tldw_chatbook/Widgets/ModelArtifacts/install_modal.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py
git commit -m "feat: represent unassigned managed models honestly"
```

---

### Task 5: Add the lazy Remote view and managed install wiring

**Files:**

- Create: `tldw_chatbook/UI/Screens/model_remote_view.py`
- Create: `Tests/UI/test_model_remote_view.py`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py:258-269, 907-940, 1106-1137`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py:30-51`
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py:60-79, 271-300`

- [ ] **Step 1: Write failing idle/search fencing tests**

Inject adapter/service/resolver factories. Assert compose/mount calls none;
pressing Search is required; exact `owner/repository` resolves directly; other
text searches; later generations discard older search/resolve callbacks; and
auth/rate-limit/timeout/oversize/no-GGUF errors have sanitized retry copy.
Use the adapter's pure `is_exact_repository()` helper rather than duplicating its
identifier grammar in the view. When a result includes shard warnings, assert
the view renders the bounded candidate name and missing indexes.

- [ ] **Step 2: Run new UI tests and verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_remote_view.py -q
```

Expected: FAIL because `model_remote_view` does not exist.

- [ ] **Step 3: Implement the smallest stateful Remote view**

Follow `CuratedView`'s worker pattern and retain only:

```python
self._search_generation = 0
self._resolve_generation = 0
self._results: tuple[RemoteModelSummary, ...] = ()
self._resolved: ResolvedRemoteModel | None = None
self._selected_catalog: ResolvedRemoteCatalog | None = None
self._pending_report = None
self._operation_reference: ArtifactRef | None = None
```

Use one Input, one Search button, result/candidate buttons, existing progress,
and plain Statics. No sort, filter, pagination, cache, model card, or details
screen.

- [ ] **Step 4: Write failing managed-install tests**

Assert candidate selection calls preflight with the exact catalog/source map;
uses `EnvConfigCredentialResolver`; opens the shared modal with acknowledgment
only for `NOASSERTION`; provisions the same values with `activate=False`; posts
existing progress/status messages; shows the compatibility warning; and triggers
the existing Installed refresh message. Assert Search and candidate controls are
disabled from preflight start through consent/provision completion (or failure),
so another search cannot replace the catalog used by the pending operation.

- [ ] **Step 5: Implement by adapting Curated's flow**

The only provision difference is:

```python
await acquisition.provision(
    report.root,
    report.grant(),
    catalog,
    sources=catalog.sources,
    progress=make_progress_callback(self.post_message),
    activate=False,
)
```

Create the credential resolver inside worker paths. Pass a resolved token to
metadata calls but never store it on the view. Derive control disabled state
from the existing pending preflight/provision state; do not add another workflow
object or state machine.

- [ ] **Step 6: Wire rail/mount without eager work**

Add `remote` before `download-models` in mapping and rail order. Compose
`RemoteView(id="remote-models-view")`. `_start_view_work("remote", ...)` returns
without search/resolve. Update rail/no-network tests and add a Remote press test
that still observes zero calls before Search.

- [ ] **Step 7: Run UI integration and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py -q
git add tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/UI/Screens/llm_screen.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "feat: add managed Remote GGUF discovery view"
```

---

### Task 6: Verify the slice and close TASK-596.1

**Files:**

- Modify only for findings: files already listed above
- Modify through Backlog CLI: `backlog/tasks/task-596.1 - Add-bounded-Hugging-Face-GGUF-discovery-to-the-managed-model-browser.md`

- [ ] **Step 1: Add one real managed-flow integration test if Task 5 mocks below acquisition**

Use `httpx.MockTransport`, a patched no-network egress check, a temporary managed
root, tiny GGUF bytes, and the real SHA-256. Resolve metadata, build catalog,
preflight, grant, and provision with `activate=False`. Assert bytes/manifest,
local-integrity provenance, unassigned consumer, and no active selector. Never
download a real model.

- [ ] **Step 2: Run the affected regression gate**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py -q
```

Expected: PASS. In Codex, permit loopback binding for fixture servers.

- [ ] **Step 3: Run static/import-boundary checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/remote_huggingface.py tldw_chatbook/Model_Artifacts/fetch.py tldw_chatbook/Model_Artifacts/acquisition.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/Widgets/ModelArtifacts tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/UI/Screens/llm_screen.py Tests/Model_Artifacts/test_remote_huggingface.py Tests/UI/test_model_remote_view.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/Model_Artifacts/remote_huggingface.py tldw_chatbook/Model_Artifacts/fetch.py tldw_chatbook/Model_Artifacts/acquisition.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_credentials_and_boundaries.py -q
git diff --check
```

Expected: PASS. Do not broaden cleanup to unrelated baseline findings.

- [ ] **Step 4: Perform live macOS evidence without a real model download**

With deterministic mocked/fixture metadata and tiny payload, verify Remote opens
idle, Search is explicit, candidate renders, unknown license gates Install,
install completes, Installed says compatibility is unverified, Activate is
absent, and Delete remains. Record Windows/Linux as preserved CI gates, not
local evidence.

- [ ] **Step 5: Request final code review**

Use `@superpowers:requesting-code-review`. Address Critical/Important issues,
rerun the gate, and document advisory deferrals instead of adding machinery.

- [ ] **Step 6: Complete the Backlog task through the CLI**

After verification and review:

```bash
backlog task edit 596.1 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6
backlog task edit 596.1 --notes "Implemented bounded Hugging Face GGUF discovery through the shared managed acquisition flow; remote artifacts remain Local integrity recorded, unassigned, and inactive. Added explicit search, pinned LFS metadata resolution, shard grouping, credential-safe downloads, unknown-license consent, focused tests, and platform-neutral UI wiring. ADR-025 remains authoritative."
backlog task edit 596.1 -s Done
```

- [ ] **Step 7: Commit closeout and prepare the PR**

```bash
git add 'backlog/tasks/task-596.1 - Add-bounded-Hugging-Face-GGUF-discovery-to-the-managed-model-browser.md'
git commit -m "docs: close remote GGUF discovery task"
git status --short
```

Expected: clean worktree. Push `codex/task-596-remote-model-discovery` and open a
PR against `dev` only after the user approves execution and final review is clean.
