# ADR-080: Keep model memory scenarios local, explicit, and runtime-neutral

Status: Accepted
Date: 2026-08-22
Related Task: TASK-20938

## Decision

Chatbook will expose a provider-neutral machine-memory observation and GGUF
memory-scenario contract for model discovery. The contract is advisory: it
compares one model's heuristic allowance at exactly 32,768 or 65,536 tokens
with installed system RAM, but it does not claim that the model supports either
context, runtime compatibility, successful inference, GPU offload, or
acceptable speed.

Machine observations remain process-memory only. They are never persisted,
synchronized, or included in ordinary or debug logs. The probe collects only
bounded platform, architecture, total/available physical-memory, memory-kind,
and accelerator-memory facts. System-memory and accelerator evidence have
independent states so missing GPU evidence cannot weaken a valid RAM capacity
observation. It never collects hostnames, serial numbers,
GPU UUIDs, driver inventories, process lists, or network data.

The supported observation platforms are macOS, Linux, and Windows. Physical
memory comes from the already-required `psutil` dependency and is observed off
the Textual event loop. Apple Silicon is classified as unified memory: its
physical-memory value is shown once and is never added again as VRAM. On Linux
and Windows, dedicated accelerator memory is optional evidence. A bounded,
fixed-argument `nvidia-smi` probe may run only from explicit trusted system
locations, with no shell, a short timeout, a strict output cap, and sanitized
parsing.
Linux may additionally read the kernel-documented AMD DRM VRAM counter from
resolved system-owned sysfs paths. Intel DRM observation is deferred until a
primary kernel contract is identified. Multiple accelerators remain separate and
are never summed because usable combinations depend on runtime configuration.
Missing accelerator evidence does not prevent a system-memory estimate.
Accelerator values are labeled observed/reported evidence and never usable
runtime capacity; OS reservation, vGPU, MIG, and runtime policy may differ.

The v1 estimation policy uses the exact candidate byte total and integer-only
allowances rounded upward to MiB:

- runtime allowance = `max(1 GiB, 10% of GGUF bytes)`;
- 32K context allowance = `max(4 GiB, 25% of GGUF bytes)`;
- 64K context allowance = twice the 32K allowance;
- machine reserve = `max(2 GiB, 20% of total physical memory)`;
- RAM working budget = total physical memory minus that reserve.

For each scenario, an estimated load within the RAM working budget is
**within budget**, an estimate above that budget but within total physical
memory **crosses the reserve**, and an estimate above total physical memory
**exceeds installed RAM**. Missing or invalid physical-memory evidence yields
**unknown**. Current available memory is displayed as volatile evidence and
produces a separate close-other-workloads warning when below the scenario
estimate, but it does not change the stable capacity classification.

The Remote Models UI will lead with the 65,536-token result and show the
32,768-token fallback. Examples include `64K scenario within RAM budget`,
`32K within budget · 64K crosses reserve`, and `32K exceeds installed RAM`.
Every expanded row shows both estimated loads and the RAM working budget that
produced its label. Adjacent copy states that these are memory scenarios, not
model-context or runtime checks, and that observed VRAM does not affect the
rating. Ratings never disable candidate selection, consent, or installation.

The recomposition-stable LLMScreen owns the accepted process-session snapshot,
observation time, probe generation, and Worker. A separate
format/provider-neutral module owns immutable observation values, bounded
platform probes, and pure estimation. RemoteView requests rechecks and renders
immutable presentation state. Observation begins lazily after a repository
resolves; only the current generation may publish. A failed refresh retains the
previous valid RAM snapshot with fixed observed-at copy. A replacement
RemoteView is hydrated after deferred mounting. Candidate estimate statics
update in place so a finishing probe cannot destroy keyboard focus.

At a measured RemoteView content width below 72 cells, the repository workflow
uses one-pane drill-down with a text-labeled Back action and collapsed estimate
details. At 72 cells or wider, results and detail remain side by side with
expanded estimate details. Exact filenames remain untruncated in the scrolling
detail pane at every width.

## Context

Deterministic filename, size, and shard guidance helps users understand GGUF
variants but cannot answer the next practical question: whether a download is
plausible on their machine. A raw comparison between file size and current free
RAM would be unstable and misleading. Context cache, runtime buffers, the
operating system, unified memory, discrete VRAM, and multi-GPU policy all affect
actual use.

The Models screen currently has exact candidate bytes but not downloaded GGUF
metadata, a selected runtime, GPU-layer settings, or a proven accelerator
backend. Fetching provider-specific file headers would add network and adapter
contracts, while importing native ML runtimes to inspect accelerators can
initialize hardware or abort headless processes. The estimate therefore needs
an explicit heuristic policy whose inputs, classification boundary, and
limitations are visible.

ADR-025 remains authoritative for artifact truth, structural GGUF admission,
managed storage, provenance, and the rule that generic GGUF validity is not
runtime compatibility. This ADR adds only local capability observation and an
advisory comparison projection.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Compare GGUF bytes directly with currently available RAM | Omits runtime/context memory and makes labels fluctuate with unrelated machine load. |
| Add RAM and VRAM into one capacity number | Double-counts Apple unified memory and assumes an unselected runtime can combine discrete devices. |
| Query llama.cpp, llamafile, CUDA, Metal, or ML frameworks | Couples browsing to runtime installation and can initialize native hardware before the user selects a runtime. |
| Fetch and parse remote GGUF headers | Adds provider-specific range requests, new remote trust/bounds, and network work that deterministic browsing does not otherwise require. |
| Use model-size-only context labels without showing assumptions | Produces false precision; the chosen heuristic remains useful only when its policy and inputs are inspectable. |
| Persist a machine profile | Creates stale device-local state and a privacy/synchronization contract for facts cheap enough to observe per process. |
| Sum multiple GPU memories | Whether memory can be combined depends on tensor-split and runtime policy, which is outside download selection. |
| Require users to enter RAM and VRAM | Adds avoidable setup burden and allows stale or malformed values to masquerade as detected evidence. |

## Consequences

- A new provider-neutral model capability module becomes the single owner of
  snapshot types, bounded observation, and pure GGUF memory estimates;
  LLMScreen owns the accepted process-session value and probe lifecycle.
- Valid Linux and Windows RAM observations remain usable even when independent
  accelerator evidence is partial or not observed.
- CPU architecture is evidence context only. Core count, throughput, and model
  speed are not estimated.
- Apple Silicon receives one unified-memory value rather than misleading RAM
  plus VRAM totals.
- 65,536 tokens is the primary memory scenario and 32,768 is the fallback.
  Neither label claims that the model supports that context. Smaller context
  configurations may use less memory, but v1 does not add more presets.
- The allowances are transparent policy values, not architecture-derived KV
  cache measurements. Their outcomes describe only whether the policy total is
  within the chosen RAM boundary.
- Exact context memory remains unknown until architecture metadata and runtime
  settings are available. A future runtime-specific estimator must add a new
  evidence tier rather than silently changing the meaning of these labels.
- Accelerator observations are informational in v1 and do not change the
  system-memory rating.
- Probe failures and unsupported platforms leave deterministic variant guidance
  and installation fully usable.
- Focused tests must cover integer boundaries, all observation states, injected
  macOS/Linux/Windows sources, output/time/path bounds, privacy, stale refreshes,
  and production 80-column compositor behavior. Automated tests do not invoke
  real accelerator tools.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md)
- [TASK-20938](../tasks/task-20938%20-%20Add-hardware-aware-GGUF-machine-fit-estimates.md)
- [ADR-025: Shared artifacts and runtime routing](025-shared-stt-artifacts-and-runtime-routing.md)
