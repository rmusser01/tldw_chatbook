"""The one-click sample bench: gating, target resolution, and execution.

Per the design spec's empty-state table
(``Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md``,
"Empty states and first run"): a fresh install has zero benches, zero
datasets, zero runs, and possibly zero configured providers. The sample
bench is what makes this screen's value legible before a user authors
anything -- it creates the loaded-nouns snippet set (the spec's own worked
example: two minimal pairs differing by one loaded word, grouped
``neutral``/``loaded`` so the group-mean aggregate has something to
aggregate), wires it to a real target, and runs it.

**Textual-free by design** (stdlib + the word bench engine only), mirroring
``evals_state.py``'s own reasoning: this module is unit-testable without
mounting anything, and the UI layer (``library_rail.py``'s button,
``evals_screen.py``'s worker) only ever calls its two public entry points,
``provider_is_configured`` and ``create_and_run_sample_bench``.

**Why the target resolution is narrow, not general.** ``capture_client.
WordBenchCaptureClient`` expects a bare OpenAI-compatible ROOT url and
appends ``/v1/completions``/``/v1/chat/completions`` itself -- exactly how
``llama_cpp``'s own config comment documents its ``api_url``
(``"llama.cpp server root; the .../v1/chat/completions path is appended
automatically"``). Every OTHER provider's ``[api_settings.<provider>]``
template bakes its OWN path convention into ``api_url`` differently (e.g.
``oobabooga``'s already ends in ``/v1/chat/completions``; ``koboldcpp``'s is
a non-OpenAI-shaped ``/api/v1/generate``) -- reusing any of those verbatim as
``WordBenchCaptureClient``'s ``base_url`` would silently double up or
misroute the request. ``llama_cpp`` is also the ONLY provider the design
spec calls fixture-verified ("a provider without a captured fixture is not
supported") and the one Task 4's live verification actually runs against.
Resolving a target for any OTHER provider correctly would need a per-
provider URL-shape adapter this module does not build -- see ``Do not
fabricate`` in the PR 3b Task 2 brief: inventing that mapping is exactly the
kind of fabrication this module must not do. A pre-existing ``eval_models``
row of ANY provider is still reused as-is when one exists (see
``resolve_sample_target``) -- that is real, already-configured data, not an
invented mapping.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from loguru import logger

from ...Chat.provider_readiness import is_valid_provider_api_key
from ...DB.Evals_DB import EvalsDB
from ...Evals.word_bench.capture_client import WordBenchCaptureClient
from ...Evals.word_bench.models import BenchConfig, Snippet, Target
from ...Evals.word_bench.runner import CancelToken, CaptureClientLike, ProgressFn, WordBenchRunner
from ...Evals.word_bench.storage import save_bench
from .evals_state import EvalsViewModel
from .snippet_editor import import_snippets_into_dataset

#: The spec's own worked example (design doc "Empty states and first run" +
#: its Δ-baseline mockup): two minimal pairs differing by exactly one loaded
#: word, grouped so the group-mean aggregate (``analysis.group_means``) has
#: two real groups to aggregate rather than rendering no group rows at all.
SAMPLE_SNIPPETS: tuple[tuple[str, str], ...] = (
    ("The protestors were", "neutral"),
    ("The rioters were", "loaded"),
    ("The government said", "neutral"),
    ("The regime said", "loaded"),
)

#: Base names for the created rows. A short id suffix is appended at
#: creation time (see ``_unique_name``) -- both ``eval_tasks.name`` and
#: ``eval_datasets.name`` are UNIQUE with no ``deleted_at`` exemption
#: (``Evals_DB.py``'s schema), so a bare literal here would raise a
#: sqlite3 UNIQUE-constraint error on a second click after the first
#: sample bench (or its dataset) was deleted -- the exact "no benches"
#: condition that makes the button reappear in the first place.
SAMPLE_BENCH_NAME = "loaded-nouns (sample)"
SAMPLE_DATASET_NAME = "loaded-nouns (sample)"
SAMPLE_TARGET_NAME = "Sample target (llama.cpp)"

#: The word bench's own preflight canary already answers "is this endpoint
#: reachable and sane" honestly, post-creation -- this K is a modest default
#: for a 4-row demo grid, not a claim about the target's capability.
SAMPLE_TOP_K = 20


def _unique_name(base: str) -> str:
    return f"{base} {uuid.uuid4().hex[:8]}"


def _llama_cpp_settings(app_config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(app_config, Mapping):
        return {}
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return {}
    section = api_settings.get("llama_cpp")
    return section if isinstance(section, Mapping) else {}


def configured_llama_cpp_url(app_config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The user's configured llama.cpp root URL, or ``None`` if unset.

    Reads ``api_settings.llama_cpp.api_url`` directly -- config only, never a
    network call, so this is safe to call from a passive render path (the
    library rail composes on every selection change).
    """
    url = _llama_cpp_settings(app_config).get("api_url")
    return url.strip() if isinstance(url, str) and url.strip() else None


def _configured_llama_cpp_model_id(app_config: Optional[Mapping[str, Any]]) -> str:
    model = _llama_cpp_settings(app_config).get("model")
    return model.strip() if isinstance(model, str) else ""


def _configured_llama_cpp_api_key(app_config: Optional[Mapping[str, Any]]) -> Optional[str]:
    settings = _llama_cpp_settings(app_config)
    configured = settings.get("api_key")
    if isinstance(configured, str) and is_valid_provider_api_key(configured):
        return configured.strip()
    env_var = settings.get("api_key_env_var")
    env_name = env_var.strip() if isinstance(env_var, str) and env_var.strip() else "LLAMA_CPP_API_KEY"
    env_value = os.environ.get(env_name)
    if env_value and is_valid_provider_api_key(env_value):
        return env_value.strip()
    return None


def resolve_sample_target(
    view_model: EvalsViewModel, app_config: Optional[Mapping[str, Any]]
) -> Optional[dict[str, Any]]:
    """The ``eval_models`` row the sample bench will target, or ``None``.

    Reuses an existing ``llama_cpp`` row if one is already registered (real,
    already-configured data -- no invention needed); otherwise mints a new
    one from the configured ``llama_cpp`` endpoint, but only when that
    endpoint is actually declared in config (``configured_llama_cpp_url``
    returns non-``None``). Returns ``None`` when neither is available --
    callers must not create a bench pointing at an invented target (see the
    module docstring).
    """
    db = view_model.db
    if db is None:
        return None
    existing = db.list_models(provider="llama_cpp")
    if existing:
        return existing[0]

    url = configured_llama_cpp_url(app_config)
    if url is None:
        return None
    # eval_models.model_id is NOT NULL and Evals_DB.create_model rejects an
    # empty string outright -- but llama.cpp's own config convention is
    # `model = ""` ("often not needed if server serves one model"; the
    # request payload's `model` field is ignored by a single-model server
    # either way, per capture_client.py's own payload). "default" here is a
    # placeholder a real server discards, never a claim about a specific
    # model that exists.
    model_id = _configured_llama_cpp_model_id(app_config) or "default"
    new_id = db.create_model(
        name=_unique_name(SAMPLE_TARGET_NAME), provider="llama_cpp", model_id=model_id
    )
    return db.get_model(new_id)


def provider_is_configured(
    view_model: EvalsViewModel, app_config: Optional[Mapping[str, Any]]
) -> bool:
    """Whether the sample bench (and, per requirement 1, the normal rail)
    has a real target to work with.

    Deliberately the SAME check ``resolve_sample_target`` itself would
    succeed or fail on -- a single source of truth, so this function can
    never say "configured" while the button it gates would fail to find a
    target, or vice versa.
    """
    return resolve_sample_target(view_model, app_config) is not None


@dataclass(frozen=True)
class SampleBenchResult:
    """What the one-click sample bench produced."""

    task_id: str
    dataset_id: str
    target_id: str
    run_group_id: str


def _default_client_factory(
    app_config: Optional[Mapping[str, Any]],
) -> Callable[[Target], CaptureClientLike]:
    url = configured_llama_cpp_url(app_config) or "http://localhost:8080"
    api_key = _configured_llama_cpp_api_key(app_config)

    def _factory(_target: Target) -> CaptureClientLike:
        return WordBenchCaptureClient(base_url=url, api_key=api_key)

    return _factory


def _mark_orphaned_runs_cancelled(db: EvalsDB, task_id: str) -> None:
    """Best-effort cleanup after a HARD cancellation interrupts
    ``runner.run`` mid-``await`` (e.g. Textual's ``exclusive=True`` worker
    mechanism cancelling an in-flight worker at an arbitrary suspension
    point) -- as opposed to ``WordBenchRunner``'s own COOPERATIVE
    ``cancel_token`` path, which already marks its rows ``"cancelled"``
    cleanly whenever it is the one to observe the cancellation (checked
    once per snippet/target iteration). A hard cancellation can land
    between that check and the row being marked, or inside the network
    call itself, leaving a run row stuck at ``"running"`` forever -- a
    permanent ghost in the rail's Runs list that never completes or fails.

    Scoped to THIS ``task_id`` only, and only rows still ``"running"`` --
    a run that already finished (``"completed"``) or was already marked
    cancelled by the cooperative path is left untouched.
    """
    try:
        runs = db.list_runs(task_id=task_id, limit=10_000)
    except Exception:
        logger.opt(exception=True).warning(
            f"Could not list runs for task {task_id!r} during cancellation cleanup."
        )
        return
    for run in runs:
        if run.get("status") != "running":
            continue
        try:
            db.update_run_status(run["id"], "cancelled")
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not mark orphaned run {run.get('id')!r} cancelled."
            )


async def create_and_run_sample_bench(
    view_model: EvalsViewModel,
    app_config: Optional[Mapping[str, Any]],
    *,
    client_factory: Optional[Callable[[Target], CaptureClientLike]] = None,
    progress: Optional[ProgressFn] = None,
    cancel_token: Optional[CancelToken] = None,
) -> SampleBenchResult:
    """Creates the sample dataset, bench, and (if needed) target, then runs
    it -- the full one-click flow.

    Args:
        view_model: The screen's read side; ``view_model.db`` must be a real
            ``EvalsDB`` (callers check this via ``provider_is_configured``
            first, but this function re-checks and raises rather than
            silently no-op-ing if called directly against a wiring-failed
            service).
        app_config: The app's loaded settings (``TldwCli.app_config``),
            read only for ``api_settings.llama_cpp``.
        client_factory: Overrides the real HTTP client -- tests inject a
            fake here (mirroring ``Tests/Evals/word_bench/test_runner.py``'s
            own ``FakeClient`` convention) so this function never makes a
            real network call under test. ``None`` (the default, production
            path) builds a real ``WordBenchCaptureClient`` against the
            configured llama.cpp endpoint.
        progress: Forwarded verbatim to ``WordBenchRunner.run`` -- this is
            the ONLY execution path in the app today (nothing else calls
            the runner in production), so a caller driving a visible
            "N/M" running state has nowhere else to get it from.
        cancel_token: Forwarded verbatim to ``WordBenchRunner.run`` -- lets
            a caller request COOPERATIVE cancellation (checked once per
            cell; the runner itself then marks its rows ``"cancelled"`` and
            returns normally). A caller relying on a HARD cancellation
            instead (e.g. an exclusive Textual worker being superseded)
            should not expect this token to help -- see
            ``_mark_orphaned_runs_cancelled``, used below for that case.

    Returns:
        The created bench/dataset/target ids and the resulting run group id
        -- the run has already completed (or failed target-by-target; see
        ``WordBenchRunner.run``, which persists ``CellError`` rows rather
        than raising on an unreachable target) by the time this returns.

    Raises:
        RuntimeError: If the evaluation service is unavailable, or no
            target could be resolved (callers should have already checked
            ``provider_is_configured`` and hidden the offer in that case;
            this is a defensive re-check, not the primary gate).
        asyncio.CancelledError: If this coroutine itself is hard-cancelled
            (e.g. by Textual's ``exclusive=True`` worker mechanism) while
            ``runner.run`` is in flight. Re-raised after marking any
            already-created run rows for this bench ``"cancelled"`` (see
            ``_mark_orphaned_runs_cancelled``) -- never swallowed, since a
            caller (and Textual's own worker bookkeeping) needs to see the
            real cancellation.
    """
    db = view_model.db
    if db is None:
        raise RuntimeError("The evaluation service is unavailable.")

    target_row = resolve_sample_target(view_model, app_config)
    if target_row is None:
        raise RuntimeError(
            "No configured target is available for the sample bench."
        )

    dataset_id = db.create_dataset(
        name=_unique_name(SAMPLE_DATASET_NAME),
        format="custom",
        source_path=f"inline:{SAMPLE_DATASET_NAME}",
    )
    snippet_dicts = [
        {"id": str(uuid.uuid4()), "text": text, "group": group, "note": None}
        for text, group in SAMPLE_SNIPPETS
    ]
    import_snippets_into_dataset(db, dataset_id, snippet_dicts)

    target = Target(
        id=target_row["id"],
        name=target_row["name"],
        provider=target_row["provider"],
        model_id=target_row["model_id"],
    )
    config = BenchConfig(
        name=_unique_name(SAMPLE_BENCH_NAME),
        prompt_mode="raw",
        top_k=SAMPLE_TOP_K,
        dataset_id=dataset_id,
        target_ids=(target.id,),
    )
    task_id = save_bench(db, config)

    snippets = [
        Snippet(id=s["id"], text=s["text"], group=s["group"]) for s in snippet_dicts
    ]
    factory = client_factory or _default_client_factory(app_config)
    runner = WordBenchRunner(db, factory)
    try:
        outcome = await runner.run(
            config, [target], snippets, task_id,
            progress=progress, cancel_token=cancel_token,
        )
    except asyncio.CancelledError:
        _mark_orphaned_runs_cancelled(db, task_id)
        raise

    return SampleBenchResult(
        task_id=task_id,
        dataset_id=dataset_id,
        target_id=target.id,
        run_group_id=outcome.group_id,
    )
