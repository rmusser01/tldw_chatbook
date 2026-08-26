# ComfyUI MiniMax H3 Video Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship sanitized MiniMax H3 base and Spectrum ComfyUI workflows, make the generic ComfyUI adapter apply their request contract strictly, and verify a real MP4 result without ever placing an original export or original prompt in the repository.

**Architecture:** Keep `ComfyUIVideoAdapter` as one workflow-driven HTTP adapter behind `VideoGenRequest` and `VideoGenResult`. Packaged H3 graphs advertise mutable or fixed controls through exact node titles; the adapter prepares a deep copy, proves every supported request value was applied or validated, then uses ComfyUI's standard HTTP API. Source exports are transformed only in an isolated temporary directory, and only renamed prompt-sanitized results enter the package.

**Tech Stack:** Python 3.11+, pytest, stdlib `json`/`secrets`/`dataclasses`, httpx through existing egress-aware helpers, ComfyUI API-format JSON, Backlog.md, jq for the one-time isolated asset transformation.

## Global Constraints

- The user-supplied exports are external, read-only evidence. Never modify them or place a raw copy under the repository.
- Do not copy any original prompt text into packaged workflows, tests, fixtures, documentation, comments, logs, commit messages, or review notes.
- The only filler prompt committed is: “An atmospheric cinematic shot of a red sailboat crossing a calm lake at sunrise. Gentle wind ripples the water and nearby reeds while the camera slowly tracks from left to right. Natural ambient sound with distant birds and soft water. No text, logos, or watermarks.”
- Do not commit original export filenames, absolute source paths, raw exports, backups, patches, or generated videos.
- Transform raw copies only inside a newly created, validated temporary directory outside the workspace; remove it after source hashes and sanitized outputs are verified.
- Assign and validate every source/temp variable in the same foreground shell session that uses it; never assume shell variables persist across tool calls.
- Stage imported assets and all later changes by exact path. Never use `git add -A` or `git add .`.
- Ship only `minimax_h3_t2v.json` and `minimax_h3_t2v_spectrum.json`; remove packaged Wan2.2 and SVD assets.
- The base H3 workflow is the default. Spectrum is opt-in and fails clearly when `SpectrumApplyMiniMaxH3` is unavailable.
- H3 defaults are 864×480, five seconds, seed 0, native 24 FPS, and MP4 output.
- H3 rejects incompatible FPS, output format, ratio, or unsupported programmatic controls before queueing.
- Preserve model filenames, class types, sampler topology, the H3 `17k + 5` frame expression, audio path, and every edge not explicitly changed by the approved allowlist.
- The static H3 image-edit export is out of scope and must use a separate `Image_Generation` task. It must not be imported here.
- Add no provider registry abstraction. Future FAL and other APIs remain separate adapters behind the existing request/result contract.
- ADR required: no new ADR. Existing ADR: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`.
- Do not mark `TASK-3401.6` Done until real ComfyUI evidence verifies classes, history output, MP4 container, dimensions, duration, FPS, and audio.

---

## File Structure

- Modify `backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md`: H3-only requirements, plan, notes, and evidence.
- Create `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json`: sanitized default graph.
- Create `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json`: sanitized Spectrum graph.
- Delete `tldw_chatbook/Video_Generation/workflows/wan22_t2v.json` and `svd_xt_i2v.json`.
- Create `Tests/Video_Generation/test_comfyui_workflow_assets.py`: prompt hygiene, topology, defaults, and inventory.
- Modify `tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py`: strict controls, effective metadata, format checks, and output parsing.
- Modify `Tests/Video_Generation/test_comfyui_adapter.py`: adapter TDD and retained transport behavior.
- Modify `tldw_chatbook/Video_Generation/config.py` and `Tests/Video_Generation/test_config_loader.py`: default H3 workflow.
- Modify `tldw_chatbook/Chat/console_generate_video.py`, `Tests/Chat/test_console_generate_video.py`, and `tldw_chatbook/UI/Screens/chat_screen.py`: style-negative compatibility.
- Create `Docs/superpowers/qa/2026-08-09-comfyui-minimax-h3-uat.md`: prompt-free real-server evidence.

---

### Task 1: Align the Backlog Contract and Prove a Clean Baseline

**Files:**
- Modify: `backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md`
- Reference: `Docs/superpowers/specs/2026-08-09-comfyui-minimax-h3-video-workflows-design.md`
- Reference: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

**Interfaces:**
- Consumes: approved H3-only design and current task record.
- Produces: acceptance criteria authorizing exactly this plan; provenance baseline with no original export tracked.

- [ ] **Step 1: Re-read the task, ADR, and lessons**

```bash
backlog task 3401.6 --plain
sed -n '1,240p' backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md
sed -n '1,240p' backlog/docs/lessons-testing-evidence.md
sed -n '1,220p' backlog/docs/lessons-live-verification.md
sed -n '1,220p' backlog/docs/lessons-backlog-hygiene.md
```

Expected: task is In Progress; ADR-044 owns the boundary; no new ADR is needed.

- [ ] **Step 2: Replace stale task requirements before production edits**

Use `apply_patch` so AC markers survive. Add separate checkboxes for:

```markdown
- [ ] Base and Spectrum MiniMax H3 API workflows ship as renamed sanitized copies; original exports and prompts are absent from the repository.
- [ ] Base H3 is the ComfyUI default; Spectrum is opt-in and names `SpectrumApplyMiniMaxH3` when unavailable.
- [ ] Prompt, seed, width, height, duration, numeric ratio, native 24 FPS, and MP4 format are applied or rejected before submission.
- [ ] Submit, polling, cancellation, output enumeration, download, image upload, and trusted-origin behavior remain covered.
- [ ] Real ComfyUI evidence confirms required classes, output descriptor, MP4, 864×480 defaults, five-second duration, 24 FPS, and audio.
```

Link the approved spec and this plan. Record:

```text
ADR required: no new ADR
ADR path: backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md
Reason: direct implementation of ADR-044's existing ComfyUI boundary.
```

- [ ] **Step 3: Verify refs contain no original exports**

Use filename checks only; never embed or print a source prompt:

Resolve the two external basenames from the active attachments into shell-local
variables, then run without writing those names into the plan or repository:

```bash
base_name=$(basename "$base_source")
spectrum_name=$(basename "$spectrum_source")
git ls-files | rg -F "$base_name" && exit 1 || true
git ls-files | rg -F "$spectrum_name" && exit 1 || true
git log --all --oneline -- ":(glob)**/$base_name" ":(glob)**/$spectrum_name"
```

Expected: no tracked path or commit returned.

- [ ] **Step 4: Record source hashes without recording source identity**

Resolve the two API exports from active task attachments. Do not echo paths.
This step proves they are readable and hashable; Task 2 recomputes the hashes in
the single foreground import session that consumes them:

```bash
test -f "$base_source" && test -f "$spectrum_source"
base_hash_before=$(shasum -a 256 "$base_source" | cut -d' ' -f1)
spectrum_hash_before=$(shasum -a 256 "$spectrum_source" | cut -d' ' -f1)
test -n "$base_hash_before" && test -n "$spectrum_hash_before"
```

Expected: both hashes are non-empty; no content is printed or persisted.

- [ ] **Step 5: Commit only the task-contract correction**

```bash
git add -- 'backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md'
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: align ComfyUI task with H3 workflows"
```

Expected staged list: exactly the task file.

---

### Task 2: Import Only Sanitized H3 Workflow Copies

**Files:**
- Create: `Tests/Video_Generation/test_comfyui_workflow_assets.py`
- Create: `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json`
- Create: `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json`
- Delete: `tldw_chatbook/Video_Generation/workflows/wan22_t2v.json`
- Delete: `tldw_chatbook/Video_Generation/workflows/svd_xt_i2v.json`

**Interfaces:**
- Consumes: two external API exports read-only; real `/object_info` proof that `SaveVideo` accepts `mp4`.
- Produces: two sanitized graphs with exact controls for the adapter.

- [ ] **Step 1: Confirm SaveVideo supports MP4**

With the user's server running, fetch `/object_info` to a validated temporary directory outside the repo. Inspect only `SaveVideo`; do not print workflow data:

```bash
schema_root=$(mktemp -d /private/tmp/tldw-h3-schema.XXXXXX)
test -n "$schema_root" && test -d "$schema_root" && test ! -L "$schema_root"
case "$schema_root" in /private/tmp/tldw-h3-schema.*) ;; *) exit 1 ;; esac
curl -fsS http://127.0.0.1:8188/object_info -o "$schema_root/object_info.json"
jq -e '.SaveVideo' "$schema_root/object_info.json" >/dev/null
jq -e '[.SaveVideo.input.required.format | .. | strings | ascii_downcase] | index("mp4") != null' "$schema_root/object_info.json" >/dev/null
jq -e '.MiniMaxH3ImageToVideo' "$schema_root/object_info.json" >/dev/null
rm -rf -- "$schema_root"
```

Expected: all checks exit 0. Inspect the H3 width/height schema without printing
prompt-related fields and confirm the source workflow's 32-pixel alignment
contract. Otherwise stop and revise the design.

- [ ] **Step 2: Write failing asset tests containing only the safe prompt**

Create `Tests/Video_Generation/test_comfyui_workflow_assets.py`:

```python
import json
from pathlib import Path

WORKFLOW_DIR = Path(__file__).parents[2] / "tldw_chatbook" / "Video_Generation" / "workflows"
SAFE_FILLER = (
    "An atmospheric cinematic shot of a red sailboat crossing a calm lake at sunrise. "
    "Gentle wind ripples the water and nearby reeds while the camera slowly tracks from "
    "left to right. Natural ambient sound with distant birds and soft water. No text, "
    "logos, or watermarks."
)


def _load(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def _prompts(graph: dict) -> list[str]:
    return [node["inputs"]["prompt"] for node in graph.values()
            if node.get("class_type") == "MiniMaxH3ImageToVideo"]


def test_h3_assets_are_sanitized_api_graphs():
    for name in ("minimax_h3_t2v.json", "minimax_h3_t2v_spectrum.json"):
        graph = _load(name)
        assert graph and all(node.get("class_type") for node in graph.values())
        assert _prompts(graph) == [SAFE_FILLER]
        assert graph["105:104"]["inputs"]["width"] == 864
        assert graph["105:104"]["inputs"]["height"] == 480
        assert graph["105:104"]["inputs"]["length"] == ["105:107", 1]
        assert graph["105:104"]["_meta"]["title"] == "Prompt Width Height"
        assert graph["105:15"]["inputs"]["noise_seed"] == 0
        assert graph["105:15"]["_meta"]["title"] == "Seed"
        assert graph["105:111"]["inputs"]["value"] == 5
        assert graph["105:111"]["_meta"]["title"] == "Duration"
        assert graph["105:91"]["inputs"]["fps"] == 24
        assert graph["105:91"]["_meta"]["title"] == "Native FPS"
        assert graph["92"]["inputs"]["format"] == "mp4"
        assert "115" not in graph


def test_spectrum_is_opt_in_and_preserves_model_routes():
    base = _load("minimax_h3_t2v.json")
    spectrum = _load("minimax_h3_t2v_spectrum.json")
    assert "SpectrumApplyMiniMaxH3" not in {node["class_type"] for node in base.values()}
    assert spectrum["105:120"]["class_type"] == "SpectrumApplyMiniMaxH3"
    assert spectrum["105:9"]["inputs"]["model"] == ["105:120", 0]
    assert spectrum["105:16"]["inputs"]["model"] == ["105:120", 0]


def test_obsolete_assets_are_not_shipped():
    assert not (WORKFLOW_DIR / "wan22_t2v.json").exists()
    assert not (WORKFLOW_DIR / "svd_xt_i2v.json").exists()
```

- [ ] **Step 3: Run the asset tests and verify RED**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_workflow_assets.py -q
```

Expected: FAIL because H3 destinations are absent and obsolete assets exist.

- [ ] **Step 4: Create validated temporary raw copies outside the repo**

Run Steps 4 through 8 in one foreground shell session so `base_source`,
`spectrum_source`, hashes, and `import_root` cannot go stale between tool calls.
Assign both exact source paths at the start of that session from the active
attachments; do not save or echo those assignments.

```bash
import_root=$(mktemp -d /private/tmp/tldw-h3-import.XXXXXX)
test -n "$import_root" && test -d "$import_root" && test ! -L "$import_root"
case "$import_root" in /private/tmp/tldw-h3-import.*) ;; *) exit 1 ;; esac
cleanup_import() {
  if test -n "${import_root:-}" && test -d "$import_root" && test ! -L "$import_root"; then
    case "$import_root" in
      /private/tmp/tldw-h3-import.*) find "$import_root" -maxdepth 1 -type f -print && rm -rf -- "$import_root" ;;
    esac
  fi
}
trap cleanup_import EXIT
cp -- "$base_source" "$import_root/base.raw.json"
cp -- "$spectrum_source" "$import_root/spectrum.raw.json"
base_hash_before=$(shasum -a 256 "$base_source" | cut -d' ' -f1)
spectrum_hash_before=$(shasum -a 256 "$spectrum_source" | cut -d' ' -f1)
test -n "$base_hash_before" && test -n "$spectrum_hash_before"
jq -e 'type == "object" and length > 0' "$import_root/base.raw.json" >/dev/null
jq -e 'type == "object" and length > 0' "$import_root/spectrum.raw.json" >/dev/null
```

Expected: two API objects outside the workspace; no content printed.

- [ ] **Step 5: Produce sanitized temporary graphs without reading old prompts into output**

Run the same jq transform for each raw temporary copy:

```bash
safe_prompt='An atmospheric cinematic shot of a red sailboat crossing a calm lake at sunrise. Gentle wind ripples the water and nearby reeds while the camera slowly tracks from left to right. Natural ambient sound with distant birds and soft water. No text, logos, or watermarks.'
jq --arg safe_prompt "$safe_prompt" '
  .["105:104"].inputs.prompt = $safe_prompt
  | .["105:104"].inputs.width = 864
  | .["105:104"].inputs.height = 480
  | .["105:104"]._meta.title = "Prompt Width Height"
  | .["105:15"].inputs.noise_seed = 0
  | .["105:15"]._meta.title = "Seed"
  | .["105:111"].inputs.value = 5
  | .["105:111"]._meta.title = "Duration"
  | .["105:91"].inputs.fps = 24
  | .["105:91"]._meta.title = "Native FPS"
  | .["92"].inputs.format = "mp4"
  | del(.["115"])
' "$import_root/base.raw.json" > "$import_root/minimax_h3_t2v.json"
jq --arg safe_prompt "$safe_prompt" '
  .["105:104"].inputs.prompt = $safe_prompt
  | .["105:104"].inputs.width = 864
  | .["105:104"].inputs.height = 480
  | .["105:104"]._meta.title = "Prompt Width Height"
  | .["105:15"].inputs.noise_seed = 0
  | .["105:15"]._meta.title = "Seed"
  | .["105:111"].inputs.value = 5
  | .["105:111"]._meta.title = "Duration"
  | .["105:91"].inputs.fps = 24
  | .["105:91"]._meta.title = "Native FPS"
  | .["92"].inputs.format = "mp4"
  | del(.["115"])
' "$import_root/spectrum.raw.json" > "$import_root/minimax_h3_t2v_spectrum.json"
```

Expected: each sanitized graph has exactly one generation prompt equal to `safe_prompt`.

- [ ] **Step 6: Prove topology drift is allowlisted**

Canonicalize raw and sanitized graphs after deleting every allowed-to-change field, then compare:

```bash
canonical_filter='del(.["115"], .["105:104"].inputs.prompt, .["105:104"].inputs.width, .["105:104"].inputs.height, .["105:104"]._meta.title, .["105:15"].inputs.noise_seed, .["105:15"]._meta.title, .["105:111"].inputs.value, .["105:111"]._meta.title, .["105:91"].inputs.fps, .["105:91"]._meta.title, .["92"].inputs.format)'
jq -S "$canonical_filter" "$import_root/base.raw.json" > "$import_root/base.raw.canonical.json"
jq -S "$canonical_filter" "$import_root/minimax_h3_t2v.json" > "$import_root/base.safe.canonical.json"
jq -S "$canonical_filter" "$import_root/spectrum.raw.json" > "$import_root/spectrum.raw.canonical.json"
jq -S "$canonical_filter" "$import_root/minimax_h3_t2v_spectrum.json" > "$import_root/spectrum.safe.canonical.json"
cmp "$import_root/base.raw.canonical.json" "$import_root/base.safe.canonical.json"
cmp "$import_root/spectrum.raw.canonical.json" "$import_root/spectrum.safe.canonical.json"
```

Expected: both `cmp` calls exit 0.

- [ ] **Step 7: Re-hash originals, import sanitized files, and remove obsolete assets**

```bash
base_hash_after=$(shasum -a 256 "$base_source" | cut -d' ' -f1)
spectrum_hash_after=$(shasum -a 256 "$spectrum_source" | cut -d' ' -f1)
test "$base_hash_before" = "$base_hash_after"
test "$spectrum_hash_before" = "$spectrum_hash_after"
cp -- "$import_root/minimax_h3_t2v.json" tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json
cp -- "$import_root/minimax_h3_t2v_spectrum.json" tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json
```

Delete the two obsolete tracked files with `apply_patch`. Do not use a recursive shell deletion in the repository.

- [ ] **Step 8: Remove only the validated temporary directory**

```bash
cleanup_import
trap - EXIT
```

Expected: the exact temporary import directory is removed; sanitized package files remain.

- [ ] **Step 9: Run asset tests and verify GREEN**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_workflow_assets.py -q
```

Expected: PASS.

- [ ] **Step 10: Stage only sanitized changes and commit**

```bash
git add -- Tests/Video_Generation/test_comfyui_workflow_assets.py tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json tldw_chatbook/Video_Generation/workflows/wan22_t2v.json tldw_chatbook/Video_Generation/workflows/svd_xt_i2v.json
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: ship sanitized MiniMax H3 workflows"
```

Expected staged paths: one asset test, two renamed H3 assets, and two obsolete deletions. No original filename, path, or prompt is present.

---

### Task 3: Make Parameterization Strict and Return Effective Metadata

**Files:**
- Modify: `Tests/Video_Generation/test_comfyui_adapter.py`
- Modify: `tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py`

**Interfaces:**
- Consumes: titles `Prompt Width Height`, `Seed`, `Duration`, `Native FPS`; `VideoGenRequest`.
- Produces: `_PreparedWorkflow(graph, duration_seconds, fps, width, height, resolved_seed)` and strict `_parameterize_workflow(...) -> _PreparedWorkflow`.

- [ ] **Step 1: Replace Wan/SVD fixtures with a safe minimal H3 fixture**

Use only synthetic safe text:

```python
def _h3_workflow():
    return {
        "gen": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"prompt": "safe placeholder", "width": 864, "height": 480, "length": ["expr", 1]},
            "_meta": {"title": "Prompt Width Height"},
        },
        "seed": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 0},
            "_meta": {"title": "Seed"},
        },
        "duration": {
            "class_type": "PrimitiveFloat",
            "inputs": {"value": 5},
            "_meta": {"title": "Duration"},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {"fps": 24, "images": ["frames", 0], "audio": ["audio", 0]},
            "_meta": {"title": "Native FPS"},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {"format": "mp4", "codec": "auto", "video": ["video", 0]},
            "_meta": {"title": "Save Video"},
        },
    }
```

- [ ] **Step 2: Write failing strict-control tests**

Add:

```python
def test_h3_preparation_applies_request_and_reports_effective_values(adapter):
    prepared = adapter._parameterize_workflow(
        _h3_workflow(),
        _request(seed=41, width=1280, height=704, duration_seconds=6, fps=24, ratio="16:9"),
        None,
    )
    assert prepared.graph["gen"]["inputs"]["prompt"] == "a lighthouse in a storm"
    assert prepared.graph["gen"]["inputs"]["width"] == 1280
    assert prepared.graph["gen"]["inputs"]["height"] == 704
    assert prepared.graph["seed"]["inputs"]["noise_seed"] == 41
    assert prepared.graph["duration"]["inputs"]["value"] == 6
    assert (prepared.width, prepared.height, prepared.duration_seconds, prepared.fps) == (1280, 704, 6.0, 24.0)
    assert prepared.resolved_seed == 41


@pytest.mark.parametrize("fps", [12, 23, 25, 30])
def test_h3_native_fps_rejects_non_24(adapter, fps):
    with pytest.raises(VideoGenerationError, match="native FPS.*24"):
        adapter._parameterize_workflow(_h3_workflow(), _request(fps=fps), None)


def test_requested_value_without_eligible_control_fails(adapter):
    graph = _h3_workflow()
    graph["gen"]["inputs"]["width"] = ["linked", 0]
    with pytest.raises(VideoGenerationError, match="width.*Prompt Width Height"):
        adapter._parameterize_workflow(graph, _request(width=1280), None)


def test_h3_rejects_incompatible_ratio_and_format(adapter):
    with pytest.raises(VideoGenerationError, match="ratio"):
        adapter._parameterize_workflow(_h3_workflow(), _request(ratio="1:1"), None)
    with pytest.raises(VideoGenerationError, match="MP4"):
        adapter._parameterize_workflow(_h3_workflow(), _request(video_format="webm"), None)
```

Also cover omitted defaults, `seed=-1`, seed below `-1`, `adaptive`, the exact three-percent ratio boundary, missing prompt/duration/native-FPS controls, misspelled titles, and graph immutability.
Add a case rejecting a supplied width or height that is not divisible by 32.

- [ ] **Step 3: Run strict tests and verify RED**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_adapter.py -k 'preparation or native_fps or eligible_control or incompatible_ratio or defaults or seed' -q
```

Expected: FAIL because `_PreparedWorkflow` and strict validation do not exist.

- [ ] **Step 4: Add the prepared-workflow value object and titles**

Import `dataclass` and `secrets`, then add:

```python
@dataclass(frozen=True)
class _PreparedWorkflow:
    graph: dict[str, Any]
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    resolved_seed: int | None


_TITLE_CONTROLS.update({
    "promptwidthheight": frozenset({"prompt", "width", "height"}),
    "duration": frozenset({"duration"}),
    "nativefps": frozenset({"native_fps"}),
})
```

Keep existing generic title conventions for user workflows.

- [ ] **Step 5: Make input mutation observable**

```python
@staticmethod
def _set_input(inputs: dict[str, Any], fields: tuple[str, ...], value: Any) -> str | None:
    for field in fields:
        if field in inputs and not isinstance(inputs[field], list):
            inputs[field] = value
            return field
    return None
```

Add a helper that raises `VideoGenerationError` naming the request field and expected title whenever required injection returns `None`.

- [ ] **Step 6: Implement exact H3 defaults and validation**

Start seed resolution with:

```python
requested_seed = request.seed
if requested_seed is not None and requested_seed < -1:
    raise VideoGenerationError("ComfyUI seed must be -1 or a non-negative integer")
resolved_seed = secrets.randbelow(2**63) if requested_seed == -1 else requested_seed
```

Then implement these exact rules:

- recognize the H3 fixed-control contract by `class_type == "MiniMaxH3ImageToVideo"`, never by source path or model filename;
- always inject `request.prompt` through a `prompt` control;
- retain graph defaults for omitted seed, dimensions, duration, and FPS, and extract their effective values;
- inject supplied seed, dimensions, and duration only into direct eligible fields;
- require supplied H3 width and height to be positive multiples of 32;
- validate `Native FPS` is 24 and reject any different explicit FPS;
- parse numeric `W:H` and require relative aspect error `<= 0.03`; reject `adaptive`;
- require `SaveVideo.inputs.format` and the request format to be `mp4` for the H3 graph;
- reject a non-empty negative prompt or non-`None` model/sampler/steps/cfg-scale when no matching documented control exists;
- return `_PreparedWorkflow` without mutating the input graph.

Do not compute `duration * fps` for H3. The preserved expression owns the `17k + 5` frame count.

- [ ] **Step 7: Thread effective facts into `VideoGenResult`**

Update the flow:

```python
prepared = self._parameterize_workflow(workflow, request, image_name)
prompt_id = self._queue_prompt(base_url, prepared.graph)
descriptor = self._poll_for_output(base_url, prompt_id, cancel_event, prepared.graph)
return self._download_output(base_url, descriptor, prepared)
```

Build the result with:

```python
return VideoGenResult(
    content=content,
    content_type=content_type,
    bytes_len=len(content),
    duration_seconds=prepared.duration_seconds,
    fps=prepared.fps,
    width=prepared.width,
    height=prepared.height,
    resolved_seed=prepared.resolved_seed,
)
```

Leave `resolved_model=None`; do not infer model identity from arbitrary graph internals.

- [ ] **Step 8: Run the adapter suite and verify GREEN**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_adapter.py -q
```

Expected: PASS.

- [ ] **Step 9: Mutation-check the silent-injection guard**

Temporarily make `_set_input` return a field without assigning. Run:

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_adapter.py -k 'applies_request or eligible_control' -q
```

Expected: FAIL. Reverse that exact temporary patch and rerun; expected PASS.

- [ ] **Step 10: Commit strict controls**

```bash
git add -- Tests/Video_Generation/test_comfyui_adapter.py tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py
git diff --cached --check
git commit -m "feat: enforce ComfyUI workflow controls"
```

---

### Task 4: Set the H3 Default and Keep Console Styles Compatible

**Files:**
- Modify: `tldw_chatbook/Video_Generation/config.py`
- Modify: `Tests/Video_Generation/test_config_loader.py`
- Modify: `tldw_chatbook/Chat/console_generate_video.py`
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

**Interfaces:**
- Consumes: `comfyui_default_workflow`, H3 workflow filenames, style-derived negative text.
- Produces: `DEFAULT_COMFYUI_WORKFLOW`; `should_forward_video_style_negative_prompt(backend, workflow_name) -> bool`.

- [ ] **Step 1: Write failing config-default tests**

```python
def test_comfyui_default_workflow_is_base_h3(monkeypatch):
    monkeypatch.setattr(config, "_read_video_generation_toml", lambda: {})
    cfg = config.get_video_generation_config(reload=True)
    assert cfg.comfyui_default_workflow == "minimax_h3_t2v.json"


def test_comfyui_explicit_spectrum_workflow_wins(monkeypatch):
    monkeypatch.setattr(
        config,
        "_read_video_generation_toml",
        lambda: {"comfyui": {"default_workflow": "minimax_h3_t2v_spectrum.json"}},
    )
    cfg = config.get_video_generation_config(reload=True)
    assert cfg.comfyui_default_workflow == "minimax_h3_t2v_spectrum.json"
```

- [ ] **Step 2: Write failing style-forwarding tests**

```python
@pytest.mark.parametrize("workflow", ["minimax_h3_t2v.json", "minimax_h3_t2v_spectrum.json"])
def test_h3_style_negative_prompt_is_not_forwarded(workflow):
    assert not should_forward_video_style_negative_prompt("comfyui", workflow)


def test_sd_cpp_style_negative_prompt_is_forwarded():
    assert should_forward_video_style_negative_prompt("stable_diffusion_cpp", None)
```

The helper applies only to style-generated negative text. A programmatic negative prompt still reaches the adapter and is rejected when unsupported.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_config_loader.py Tests/Chat/test_console_generate_video.py -k 'default_workflow or spectrum_workflow or style_negative' -q
```

Expected: FAIL because the constant and helper do not exist.

- [ ] **Step 4: Implement the default constant**

In `config.py`:

```python
DEFAULT_COMFYUI_WORKFLOW = "minimax_h3_t2v.json"
```

Set `comfyui_default_workflow` to the nested configured value or this constant. Import the same constant in `ComfyUIVideoAdapter`; remove its Wan literal fallback.

- [ ] **Step 5: Implement and use style-negative filtering**

In `console_generate_video.py`:

```python
_H3_COMFYUI_WORKFLOWS = frozenset({
    "minimax_h3_t2v.json",
    "minimax_h3_t2v_spectrum.json",
})


def should_forward_video_style_negative_prompt(backend: str, workflow_name: str | None) -> bool:
    return not (
        backend.strip().lower() == "comfyui"
        and str(workflow_name or "").strip() in _H3_COMFYUI_WORKFLOWS
    )
```

In `chat_screen.py`, after backend/config resolution and before dispatch:

```python
if negative_text and not should_forward_video_style_negative_prompt(
    backend, cfg.comfyui_default_workflow
):
    negative_text = None
```

Do not branch on model names or prompt content.

- [ ] **Step 6: Run config, helper, and template tests**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_config_loader.py Tests/Video_Generation/test_video_templates.py Tests/Chat/test_console_generate_video.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit default and style compatibility**

```bash
git add -- tldw_chatbook/Video_Generation/config.py Tests/Video_Generation/test_config_loader.py tldw_chatbook/Chat/console_generate_video.py Tests/Chat/test_console_generate_video.py tldw_chatbook/UI/Screens/chat_screen.py
git diff --cached --check
git commit -m "feat: default ComfyUI video to MiniMax H3"
```

---

### Task 5: Parse SaveVideo Output Safely and Run Live UAT

**Files:**
- Modify: `Tests/Video_Generation/test_comfyui_adapter.py`
- Modify: `tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py`
- Create: `Docs/superpowers/qa/2026-08-09-comfyui-minimax-h3-uat.md`

**Interfaces:**
- Consumes: prepared graph, ComfyUI history, configured trusted origin.
- Produces: output selection restricted to supported output nodes; prompt-free real-server evidence.

- [ ] **Step 1: Write failing output-node tests**

The adapter must not select an unrelated preview when the graph has `SaveVideo`:

```python
def test_output_selection_uses_save_video_node_not_preview(adapter):
    graph = _h3_workflow()
    history = {
        "job": {
            "outputs": {
                "preview": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]},
                "save": {"videos": [{"filename": "clip.mp4", "subfolder": "video", "type": "output"}]},
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }
    assert adapter._find_output_descriptor(history, "job", graph)["filename"] == "clip.mp4"
```

Add cases for an arbitrary list-valued collection under `SaveVideo`, malformed descriptors, non-MP4 output, and terminal success without a valid output-node descriptor.

- [ ] **Step 2: Run output tests and verify RED**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_adapter.py -k 'output_selection or save_video' -q
```

Expected: FAIL because the parser does not accept the graph and scans fixed collection names globally.

- [ ] **Step 3: Restrict descriptor discovery to output node ids**

Add:

```python
_SUPPORTED_OUTPUT_CLASSES = frozenset({"SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP"})


@staticmethod
def _output_node_ids(graph: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(node_id)
        for node_id, node in graph.items()
        if node.get("class_type") in _SUPPORTED_OUTPUT_CLASSES
    )
```

Change `_find_output_descriptor(history, prompt_id, graph)` to inspect only those ids. Within each selected output object, iterate list-valued collections and accept only a dict with a non-empty `filename`, string-compatible `subfolder`/`type`, and supported media suffix. Preserve terminal-failure-before-output behavior.

- [ ] **Step 4: Run adapter tests and verify GREEN**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_adapter.py -q
```

Expected: PASS.

- [ ] **Step 5: Run one base H3 generation against real ComfyUI**

Use the neutral filler or a new harmless prompt, never an original prompt. Queue one short local generation with packaged defaults. Do not copy generated media into the repo.

Verify all of:

```text
required classes present
prompt accepted
history reaches success
selected output belongs to node 92
downloaded content type is video/mp4
container probe reports MP4
width 864 and height 480
24 FPS
duration matches the five-second H3 frame-grid result within container tolerance
audio stream present
```

Probe only the ephemeral output:

```bash
ffprobe -v error -show_entries format=format_name,duration -show_entries stream=codec_type,width,height,r_frame_rate -of json "$generated_video"
```

Expected: MP4-family format, one 864×480 video stream at 24 FPS, and an audio stream.

- [ ] **Step 6: Verify Spectrum initialization**

Select `minimax_h3_t2v_spectrum.json`. If the node is installed, run one short generation and verify the same contract. If absent, verify the error names `SpectrumApplyMiniMaxH3` and no `/prompt` call occurs.

- [ ] **Step 7: Record prompt-free UAT evidence**

Create `Docs/superpowers/qa/2026-08-09-comfyui-minimax-h3-uat.md` containing only:

```markdown
# ComfyUI MiniMax H3 UAT

- Date and origin class (`localhost` or configured trusted host; no credentials)
- Required class names and availability
- Packaged workflow filename
- Prompt id reduced to a non-sensitive suffix
- History output node id, collection key, and descriptor field names
- HTTP content type and byte length
- ffprobe format, duration, dimensions, frame rate, and stream types
- Spectrum success or exact missing-class outcome
- Confirmation that no media, source path, raw export, or prompt text was committed
```

Do not include any request prompt, source prompt, external path, or generated bytes.

- [ ] **Step 8: Commit parser and UAT evidence**

```bash
git add -- Tests/Video_Generation/test_comfyui_adapter.py tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py Docs/superpowers/qa/2026-08-09-comfyui-minimax-h3-uat.md
git diff --cached --check
git diff --cached --name-only
git commit -m "test: verify MiniMax H3 against ComfyUI"
```

---

### Task 6: Full Verification, Follow-ups, and Closeout

**Files:**
- Modify: `backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md`
- Create through Backlog CLI: H3 static image-edit follow-up.
- Create through Backlog CLI: MIME-driven video-extension follow-up.

**Interfaces:**
- Consumes: all implementation commits and live evidence.
- Produces: verified Done task and two separately scoped follow-up records.

- [ ] **Step 1: Run focused suites**

```bash
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_workflow_assets.py Tests/Video_Generation/test_comfyui_adapter.py Tests/Video_Generation/test_config_loader.py Tests/Video_Generation/test_video_templates.py -q
.venv/bin/python -m pytest Tests/Chat/test_console_generate_video.py Tests/Chat/test_console_video_actions.py Tests/Chat/test_console_video_message.py -q
```

Expected: PASS.

- [ ] **Step 2: Run reachable regression suites**

```bash
.venv/bin/python -m pytest Tests/Video_Generation -q
.venv/bin/python -m pytest Tests/RuntimePolicy -q
.venv/bin/python -m pytest --collect-only -q
```

Expected: PASS. For instability, compare failure sets from the identical command on the baseline; never compare counts from different commands.

- [ ] **Step 3: Run static checks**

```bash
.venv/bin/python -m ruff check tldw_chatbook/Video_Generation tldw_chatbook/Chat/console_generate_video.py Tests/Video_Generation Tests/Chat/test_console_generate_video.py
git diff --check
```

Expected: PASS.

- [ ] **Step 4: Run final provenance gates without source prompt literals**

Resolve the two external basenames into shell-local variables, then run:

```bash
base_name=$(basename "$base_source")
spectrum_name=$(basename "$spectrum_source")
git ls-files | rg -F "$base_name" && exit 1 || true
git ls-files | rg -F "$spectrum_name" && exit 1 || true
.venv/bin/python -m pytest Tests/Video_Generation/test_comfyui_workflow_assets.py -q
git status --short -- tldw_chatbook/Video_Generation/workflows Tests/Video_Generation/test_comfyui_workflow_assets.py
```

Expected: no original filename tracked; safe-prompt equality tests pass; only intended sanitized paths appear.

- [ ] **Step 5: Self-review the complete diff**

```bash
git diff origin/dev...HEAD --stat
git diff origin/dev...HEAD -- tldw_chatbook/Video_Generation Tests/Video_Generation tldw_chatbook/Chat/console_generate_video.py Tests/Chat/test_console_generate_video.py 'backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md'
```

Check for silent injection, linked-input mutation, path escape, output-node confusion, prompt leakage, model-file changes, and unrelated edits.

- [ ] **Step 6: File the two approved follow-ups after a collision sweep**

Sweep all remote refs/worktrees per `lessons-backlog-hygiene.md`, then create tasks named:

```text
H3 static image edit through Image_Generation
MIME-driven generated-video file extensions
```

The image task states that nodes 154 and 166 are removed from a sanitized copy and node 165 is canonical. Neither task includes an original prompt, export, or source path. Re-read each with `backlog task <id> --plain`.

- [ ] **Step 7: Complete notes and acceptance criteria**

Implementation Notes cover:

```text
sanitized base and Spectrum assets
strict controls and effective metadata
default workflow and style compatibility
real object_info/history/container/audio evidence
ADR-044 reuse and no-new-ADR decision
tests and static checks
original exports unchanged and absent from Git
follow-up task ids
```

Check an AC only when evidence exists. If live UAT is unavailable, leave the task In Progress.

- [ ] **Step 8: Mark Done only after every Definition-of-Done item passes**

```bash
backlog task edit 3401.6 -s Done
git add -- 'backlog/tasks/task-3401.6 - ComfyUI-video-adapter-with-shipped-workflow-assets.md'
git diff --cached --check
git commit -m "docs: complete ComfyUI H3 adapter task"
```

Expected: all ACs checked, notes present, status Done, and no unrelated file staged.

---

## Final-review remediation addendum (2026-08-09)

This addendum supersedes Task 4's filename-list style helper. Console
compatibility is determined by loading the selected confined graph off the UI
loop and checking for `MiniMaxH3ImageToVideo`; classification itself makes no
network request. The positive style suffix remains intact, only a style-derived
negative is suppressed for H3, and explicit programmatic negatives continue to
the adapter for rejection.

The final-review round also:

1. builds wheel and sdist in validated temporary directories, checks that each
   contains exactly the two approved H3 JSON assets and no Wan/SVD asset, then
   proves both graphs load from a fresh wheel install;
2. rejects H3 reference images before object-info or upload side effects while
   preserving the generic input-image path;
3. keeps polling through pending preview/unrelated outputs, distinguishes
   terminal failure from terminal success without media, and budgets HTTP calls
   and waits against one bounded deadline;
4. validates effective width, height, duration, FPS, and seed before queueing;
5. pins the exact Base/Spectrum model, sampler, audio, and frame-grid topology,
   plus the live-observed node `92` / `images` history shape; and
6. runs only the user-authorized touched-file tests and targeted static,
   distribution, diff, and provenance gates. Existing live UAT is reused; no
   live generation, RuntimePolicy, full collection, or full repository suite is
   rerun.

ADR required: no new ADR

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: remediation preserves ADR-044's established ComfyUI provider,
trusted-origin, request/result, and ephemeral-storage boundaries.
