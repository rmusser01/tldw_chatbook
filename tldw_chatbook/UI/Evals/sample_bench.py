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
from urllib.parse import urlparse, urlunparse

from loguru import logger

from ...Chat.provider_readiness import is_valid_provider_api_key
from ...DB.Evals_DB import EvalsDB
from ...Evals.word_bench.capture_client import WordBenchCaptureClient
from ...Evals.word_bench.models import BenchConfig, Snippet, Target
from ...Evals.word_bench.runner import CancelToken, CaptureClientLike, ProgressFn, WordBenchRunner
from ...Evals.word_bench.storage import _unique_name, load_bench, save_bench
from .evals_state import EvalsViewModel
from .snippet_editor import dataset_snippets, import_snippets_into_dataset

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
#: creation time (see ``storage._unique_name``, imported above -- the
#: engine layer owns it now, task-1482) -- both ``eval_tasks.name`` and
#: ``eval_datasets.name`` are UNIQUE with no ``deleted_at`` exemption
#: (``Evals_DB.py``'s schema), so a bare literal here would raise a
#: sqlite3 UNIQUE-constraint error on a second click after the first
#: sample bench (or its dataset) was deleted -- the exact "no benches"
#: condition that makes the button reappear in the first place.
SAMPLE_BENCH_NAME = "loaded-nouns (sample)"
SAMPLE_DATASET_NAME = "loaded-nouns (sample)"
SAMPLE_TARGET_NAME = "Sample target (llama.cpp)"

#: task-1482 Task 6: the base name for a row created via bench_editor.py's
#: "Create target" button (an authored bench's own target, never part of
#: the one-click sample flow above) -- passed as `resolve_sample_target`'s
#: `name` override so that flow's row does not read as though it came from
#: SAMPLE_TARGET_NAME's unrelated demo.
BENCH_EDITOR_TARGET_NAME = "llama.cpp target"

#: The word bench's own preflight canary already answers "is this endpoint
#: reachable and sane" honestly, post-creation -- this K is a modest default
#: for a 4-row demo grid, not a claim about the target's capability.
SAMPLE_TOP_K = 20


def _llama_cpp_settings(app_config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(app_config, Mapping):
        return {}
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return {}
    section = api_settings.get("llama_cpp")
    return section if isinstance(section, Mapping) else {}


#: Endpoint paths a real ``api_url`` is commonly configured with, stripped
#: back to the bare ROOT ``WordBenchCaptureClient`` needs (it appends
#: ``/v1/completions`` / ``/v1/chat/completions`` itself -- see
#: ``_normalize_llama_cpp_root``). Ordered longest-first so the first match
#: is the most specific one. ``/completion`` (singular) is llama.cpp's OWN
#: native completion endpoint and is what this machine's real config
#: carries; without stripping it the client would request
#: ``.../completion/v1/completions`` -> 404 -> a grid of ``CellError``
#: cells blaming the server for what is really a URL shape.
_LLAMA_CPP_ENDPOINT_SUFFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
    "/completions",
    "/completion",
    "/v1",
)


def _normalize_llama_cpp_root(url: str) -> str:
    """Reduce a configured llama.cpp ``api_url`` to the bare root.

    ``WordBenchCaptureClient`` is documented to take a ROOT and append its
    own path, so a configured value that already carries an endpoint path
    would double up. Only the recognised suffixes in
    ``_LLAMA_CPP_ENDPOINT_SUFFIXES`` are removed, and only from a value
    that parses as an absolute ``scheme://host`` URL -- anything else is
    returned as-is (minus a trailing slash) rather than guessed at.
    Query strings and fragments are dropped: a ROOT has neither, and
    carrying one through would corrupt the appended path.
    """
    stripped = url.strip()
    parsed = urlparse(stripped)
    if not parsed.scheme or not parsed.netloc:
        return stripped.rstrip("/") or stripped
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    for suffix in _LLAMA_CPP_ENDPOINT_SUFFIXES:
        if lowered.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", ""))


def configured_llama_cpp_url(app_config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The user's configured llama.cpp ROOT URL, or ``None`` if unset.

    Reads ``api_settings.llama_cpp.api_url`` directly -- config only, never a
    network call, so this is safe to call from a passive render path (the
    library rail composes on every selection change).

    The configured value is normalised to a bare root (see
    ``_normalize_llama_cpp_root``): the config template documents a root,
    but real in-use configs carry llama.cpp's own ``/completion`` endpoint
    (or an OpenAI-shaped ``/v1``), and handing either to
    ``WordBenchCaptureClient`` verbatim produces a doubled path that fails
    every cell of the one-click sample bench.

    Args:
        app_config: The app's config mapping, or ``None``. Read via
            ``api_settings.llama_cpp.api_url``; any other shape (missing
            section, non-string/blank value) is treated as unset.

    Returns:
        The normalised root URL, or ``None`` if no ``llama_cpp`` endpoint
        is configured.
    """
    url = _llama_cpp_settings(app_config).get("api_url")
    if not isinstance(url, str) or not url.strip():
        return None
    normalized = _normalize_llama_cpp_root(url)
    return normalized or None


def _configured_llama_cpp_model_id(app_config: Optional[Mapping[str, Any]]) -> str:
    model = _llama_cpp_settings(app_config).get("model")
    return model.strip() if isinstance(model, str) else ""


def _configured_llama_cpp_api_key(app_config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The llama.cpp API key to send, env-first per this project's
    documented precedence (env vars -> config.toml -> defaults; see
    CLAUDE.md). ``LLAMA_CPP_API_KEY`` is the same env var name
    ``config.py``'s own ``llama_cpp`` template documents as
    ``api_key_env_var``, so a value set there always wins over whatever is
    committed to ``config.toml``, matching how every other provider in
    this app resolves credentials.
    """
    settings = _llama_cpp_settings(app_config)
    env_var = settings.get("api_key_env_var")
    env_name = env_var.strip() if isinstance(env_var, str) and env_var.strip() else "LLAMA_CPP_API_KEY"
    env_value = os.environ.get(env_name)
    if env_value and is_valid_provider_api_key(env_value):
        return env_value.strip()
    configured = settings.get("api_key")
    if isinstance(configured, str) and is_valid_provider_api_key(configured):
        return configured.strip()
    return None


def _existing_sample_target(db: EvalsDB) -> Optional[dict[str, Any]]:
    """The already-registered ``llama_cpp`` ``eval_models`` row, if any.

    A pure read, shared by the gate and the resolver so the two can never
    disagree about what "already configured" means.
    """
    existing = db.list_models(provider="llama_cpp")
    return existing[0] if existing else None


def resolve_sample_target(
    view_model: EvalsViewModel,
    app_config: Optional[Mapping[str, Any]],
    *,
    create: bool = False,
    name: str = SAMPLE_TARGET_NAME,
) -> Optional[dict[str, Any]]:
    """The ``eval_models`` row the sample bench will target, or ``None``.

    Reuses an existing ``llama_cpp`` row if one is already registered (real,
    already-configured data -- no invention needed); otherwise mints a new
    one from the configured ``llama_cpp`` endpoint, but ONLY when
    ``create=True`` AND that endpoint is actually declared in config
    (``configured_llama_cpp_url`` returns non-``None``). Returns ``None``
    when neither is available -- callers must not create a bench pointing
    at an invented target (see the module docstring).

    Args:
        create: Whether this call may WRITE a new ``eval_models`` row.
            Defaults to ``False`` -- a predicate that mutates the database
            is the defect this parameter exists to prevent: the library
            rail's gate runs inside ``compose()``, so with creation on by
            default merely OPENING the Evals screen persisted an invented
            target row on every fresh install (``config.py`` ships an
            ``api_url`` default, so the condition is near-universal).
            Only ``create_and_run_sample_bench`` -- the click path that
            genuinely needs a row to point a run at -- passes ``True``.
            Ask ``provider_is_configured`` whether a row WOULD be
            resolvable; it answers without writing.
        name: The base name for a newly CREATED row (irrelevant when an
            existing row is reused instead -- see above). Defaults to
            ``SAMPLE_TARGET_NAME``, the one-click sample bench's own
            wording; ``bench_editor.py``'s Task 6 "Create target" button
            (via ``evals_screen.py``, this module's other caller of the
            ``create=True`` path) passes ``BENCH_EDITOR_TARGET_NAME``
            instead, so a target created from an authored bench does not
            read as though it came from the unrelated one-click flow.
            Always passed through ``storage._unique_name`` before the
            write, exactly like the default.

    Returns:
        The resolved ``eval_models`` row (a real, DB-backed dict), or
        ``None`` if there is nothing to reuse and either ``create`` is
        ``False`` or no ``llama_cpp`` endpoint is configured to mint one
        from.
    """
    db = view_model.db
    if db is None:
        return None
    existing = _existing_sample_target(db)
    if existing is not None:
        return existing

    url = configured_llama_cpp_url(app_config)
    if url is None or not create:
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
        name=_unique_name(name), provider="llama_cpp", model_id=model_id
    )
    return db.get_model(new_id)


def provider_is_configured(
    view_model: EvalsViewModel, app_config: Optional[Mapping[str, Any]]
) -> bool:
    """Whether the sample bench (and, per requirement 1, the normal rail)
    has a real target to work with.

    **A question, never a mutation.** This is called from
    ``LibraryRail._benches_section_body`` inside ``compose()``: rendering
    the screen must perform no writes. It therefore asks the same two
    conditions ``resolve_sample_target(..., create=True)`` succeeds on --
    an existing ``llama_cpp`` row, or a configured endpoint it could mint
    one from -- WITHOUT minting anything. The equivalence is pinned by
    ``test_provider_is_configured_matches_resolve_sample_target_exactly``
    (and its read-only companion) in
    ``Tests/UI/test_evals_empty_states.py``, so this can never say
    "configured" while the button it gates would fail to find a target, or
    vice versa.
    """
    db = view_model.db
    if db is None:
        return False
    if _existing_sample_target(db) is not None:
        return True
    return configured_llama_cpp_url(app_config) is not None


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

    # create=True: this IS the click path, the one place a real target row
    # may be persisted (see resolve_sample_target's own `create` docs --
    # the rail's render-time gate must never reach this).
    target_row = resolve_sample_target(view_model, app_config, create=True)
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


@dataclass(frozen=True)
class RunBenchResult:
    """What running an existing bench produced."""

    task_id: str
    run_group_id: str


def _resolve_targets(db: EvalsDB, config: BenchConfig) -> list[Target]:
    """The bench's target columns, resolved from their ``eval_models`` rows.

    Uses the same lookup ``bench_editor.py``'s target table renders from
    (``db.get_model``) -- the row a deleted ``eval_models`` row leaves
    dangling there as ``"(deleted target <id>) — unresolvable"``. This
    function raises instead of rendering a placeholder: a run cannot
    proceed with a target it cannot resolve to real provider/model_id
    values.

    Raises:
        RuntimeError: Naming the first ``target_id`` that no longer
            resolves to a live, non-deleted ``eval_models`` row.
    """
    targets: list[Target] = []
    for target_id in config.target_ids:
        model = db.get_model(target_id)
        if model is None:
            raise RuntimeError(
                f"Target {target_id!r} could not be resolved — its "
                "eval_models row is missing or was deleted."
            )
        targets.append(
            Target(
                id=model["id"],
                name=model["name"],
                provider=model["provider"],
                model_id=model["model_id"],
            )
        )
    return targets


def _load_snippets(db: EvalsDB, dataset_id: str) -> list[Snippet]:
    """The bench's dataset snippets, via the same inline-storage reader
    ``snippet_editor.py``'s read path uses (``dataset_snippets``) rather
    than a second query against ``eval_datasets.metadata``.

    Raises:
        RuntimeError: If the dataset no longer exists, or exists but has
            no snippets to run against -- an empty grid is never a valid
            run.
    """
    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        raise RuntimeError(f"Dataset {dataset_id!r} was not found.")
    raw_snippets = dataset_snippets(dataset)
    if not raw_snippets:
        name = dataset.get("name") or dataset_id
        raise RuntimeError(f"Dataset {name!r} has no snippets to run.")
    return [
        Snippet(
            id=str(snippet["id"]),
            text=str(snippet.get("text") or ""),
            group=snippet.get("group"),
            note=snippet.get("note"),
        )
        for snippet in raw_snippets
    ]


async def run_existing_bench(
    view_model: EvalsViewModel,
    app_config: Optional[Mapping[str, Any]],
    task_id: str,
    *,
    client_factory: Optional[Callable[[Target], CaptureClientLike]] = None,
    progress: Optional[ProgressFn] = None,
    cancel_token: Optional[CancelToken] = None,
) -> RunBenchResult:
    """Runs an already-saved bench -- the engine call behind the Run Bench
    button.

    Sibling of ``create_and_run_sample_bench``: that function creates a
    dataset, bench, and (if needed) target before running them; this one
    runs a bench that already exists (its dataset, target(s), and
    ``eval_tasks`` row were all created earlier, e.g. by a bench author
    using ``bench_editor.py``), resolving everything it needs from the
    database rather than building it fresh. Shares
    ``_default_client_factory`` and ``_mark_orphaned_runs_cancelled`` with
    that function, so both entry points behave identically in production
    and under a hard cancellation.

    Args:
        view_model: The screen's read side; ``view_model.db`` must be a
            real ``EvalsDB`` (callers should already know this from the
            screen being usable at all, but this function re-checks and
            raises rather than silently no-op-ing if called directly
            against a wiring-failed service).
        app_config: The app's loaded settings (``TldwCli.app_config``),
            read only for ``api_settings.llama_cpp`` when
            ``client_factory`` is not supplied.
        task_id: The bench's ``eval_tasks`` row id (``storage.save_bench``'s
            return value).
        client_factory: Overrides the real HTTP client -- tests inject a
            fake here, mirroring ``create_and_run_sample_bench``'s own
            parameter (and ``Tests/Evals/word_bench/test_runner.py``'s
            ``FakeClient`` convention) so this function never makes a real
            network call under test. ``None`` (the default, production
            path) builds a real ``WordBenchCaptureClient`` against the
            configured llama.cpp endpoint.
        progress: Forwarded verbatim to ``WordBenchRunner.run`` -- lets a
            caller drive a visible "N/M" running state.
        cancel_token: Forwarded verbatim to ``WordBenchRunner.run`` -- lets
            a caller request COOPERATIVE cancellation (checked once per
            cell; the runner itself then marks its rows ``"cancelled"`` and
            returns normally). A caller relying on a HARD cancellation
            instead (e.g. an exclusive Textual worker being superseded)
            should not expect this token to help -- see
            ``_mark_orphaned_runs_cancelled``, used below for that case,
            same as ``create_and_run_sample_bench``.

    Returns:
        The bench's task id and the resulting run group id -- the run has
        already completed (or failed target-by-target; see
        ``WordBenchRunner.run``, which persists ``CellError`` rows rather
        than raising on an unreachable target) by the time this returns.
        Each call creates a NEW run group: there is no cross-run cache, so
        re-running a bench after a failed attempt leaves the failed run
        group's rows untouched and produces a second, independent one.

    Raises:
        RuntimeError: If the evaluation service is unavailable, ``task_id``
            does not name an existing, readable bench, the bench has no
            targets configured (task-1482: a draft bench created via the
            rail's "+ New bench" starts with ``target_ids=()``), any of
            its targets no longer resolve to a live ``eval_models`` row,
            or its dataset is missing or has no snippets.
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

    try:
        config = load_bench(db, task_id)
    except Exception as exc:
        # load_bench raises TypeError for a task_id with no matching row
        # (get_task returns None) and can raise KeyError for a task_id
        # that exists but isn't a word bench (its config_data lacks
        # prompt_mode/top_k) -- both are "this isn't a runnable bench",
        # collapsed here into one RuntimeError naming the id, mirroring
        # bench_editor.py's own broad `except Exception` around this same
        # call.
        raise RuntimeError(f"Bench {task_id!r} could not be read: {exc}") from exc

    if not config.target_ids:
        # task-1482 fix round 1: a draft bench created via the rail's
        # "+ New bench" (no targets wired on until the bench editor's
        # Task 6) must never reach `runner.run` with an empty target list
        # -- `create_run_group` loops over `targets`, so zero targets
        # silently produces a run group with ZERO `eval_runs` rows, which
        # then reads back as "this run could not be found" the moment
        # anything tries to select it. `_primary_action_state` already
        # blocks the button for this exact case, but this is the engine
        # seam itself: belt-and-suspenders for any other caller (a future
        # CLI/API entry point, a test driving this function directly)
        # that does not go through the UI gate.
        raise RuntimeError(f"Bench {config.name!r} has no targets to run.")

    targets = _resolve_targets(db, config)
    snippets = _load_snippets(db, config.dataset_id)

    factory = client_factory or _default_client_factory(app_config)
    runner = WordBenchRunner(db, factory)
    try:
        outcome = await runner.run(
            config, targets, snippets, task_id,
            progress=progress, cancel_token=cancel_token,
        )
    except asyncio.CancelledError:
        _mark_orphaned_runs_cancelled(db, task_id)
        raise

    return RunBenchResult(task_id=task_id, run_group_id=outcome.group_id)
