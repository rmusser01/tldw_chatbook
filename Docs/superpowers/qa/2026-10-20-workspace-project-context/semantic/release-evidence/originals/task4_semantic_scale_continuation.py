"""Prepared continuation: reuse nested Ready; build wide only after parent ruling."""

from __future__ import annotations

import os
import site

site.addsitedir(os.environ["TLDW_TEST_SHARED_SITE"])

import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

SDD = Path(__file__).resolve().parent
REPO = SDD.parents[2]
RUN_NAME = os.environ["TLDW_SCALE_RUN_NAME"]
if re.fullmatch(r"task4-semantic-scale-continuation-[a-z0-9-]+", RUN_NAME) is None:
    raise ValueError("invalid_scale_run_name")

# Resolve top-level packages before project/test imports or factory/model work.
origins = {}
for package in ("tldw_chatbook", "Tests"):
    spec = importlib.util.find_spec(package)
    actual = str(Path(spec.origin).resolve()) if spec and spec.origin else None
    expected = str((REPO / package / "__init__.py").resolve())
    origins[package] = {"actual": actual, "expected": expected}
(SDD / f"{RUN_NAME}-origins.json").write_text(
    json.dumps(origins, indent=2), encoding="utf-8"
)
if any(item["actual"] != item["expected"] for item in origins.values()):
    raise ValueError("foreign_project_or_tests_module_origin")

import httpx
import psutil
import requests

from Tests import conftest as _test_bootstrap  # noqa: F401 - private profile
from Tests.Agents.test_project_index_tool import invoke, provider
from Tests.UI.app_factory import _build_test_app, drain_created_dirs, drain_created_runtimes
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path
from tldw_chatbook.Embeddings.Embeddings_Lib import _HuggingFaceEmbedder
from tldw_chatbook.RAG_Search.projection_scope import ProjectionExecutionScope
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
from tldw_chatbook.Workspaces.project_index.runtime import (
    close_project_index_service,
    get_project_index_service,
    project_index_key,
)
from tldw_chatbook.Workspaces.project_index.semantic import (
    _resolved_spec,
    _settings_for_project_model,
    _target,
)
from tldw_chatbook.Workspaces.project_index.service import binding_bytes


BASE = SDD.parents[1] / "k5-scale"
PROFILE = BASE / "stable-profile"
MODEL = os.environ["TLDW_TEST_OFFLINE_EMBEDDING_MODEL"]
MAX_BUILD_SECONDS = 3600
MAX_QUERY_SECONDS = 300
MAX_NATIVE_JOIN_SECONDS = 300
MAX_HOLD_SECONDS = 90
MAX_STATE_READ_SECONDS = 30
EXPECTED_NESTED_READY_ID = "b5bc9b0b2ceb4e0c92d41b631662f5aa"
EXPECTED_NESTED_SEMANTIC_REVISION = 3
EXPECTED_NESTED_MANIFEST_REVISION = 329
EVENTS = SDD / f"{RUN_NAME}-events.jsonl"

_lock = threading.Lock()
_sample_lock = threading.Lock()
_context = {"shape": None, "phase": "preflight"}
_metrics = {}
_network_attempts = []
_hold_armed = threading.Event()
_encoded = threading.Event()
_release = threading.Event()
_original_config = config.load_cli_config_and_ensure_existence
_original_init = _HuggingFaceEmbedder.__init__
_original_embed = _HuggingFaceEmbedder.embed
_original_wrapper_init = EmbeddingsServiceWrapper.__init__
_original_wrapper_close = EmbeddingsServiceWrapper.close
_original_carrier_close = ProjectionExecutionScope.close


def event(name, **fields):
    with _lock:
        record = {
            "time_monotonic": time.monotonic(),
            "shape": _context["shape"],
            "phase": _context["phase"],
            "event": name,
            **fields,
        }
        with EVENTS.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def error_location(error):
    frame = error.__traceback__
    while frame is not None and frame.tb_next is not None:
        frame = frame.tb_next
    if frame is None:
        return None
    return f"{Path(frame.tb_frame.f_code.co_filename).name}:{frame.tb_lineno}:{frame.tb_frame.f_code.co_name}"


def phase(shape, value):
    with _lock:
        _context.update(shape=shape, phase=value)


def measured(shape):
    return _metrics.setdefault(
        shape,
        {
            "hf_embed": [],
            "hf_constructor": [],
            "wrapper_constructor": [],
            "wrapper_close": [],
            "carrier_close": [],
        },
    )


def configured(*args, **kwargs):
    data = dict(_original_config(*args, **kwargs))
    embeddings = dict(data.get("embedding_config", {}))
    models = dict(embeddings.get("models", {}))
    models["minilm-local"] = {
        "provider": "huggingface",
        "model_name_or_path": MODEL,
        "device": "cpu",
    }
    embeddings["models"] = models
    data["embedding_config"] = embeddings
    return data


def observed_hf_init(self, cfg):
    started = time.monotonic()
    try:
        _original_init(self, cfg)
    except BaseException as error:
        event("hf_constructor_error", error_class=type(error).__name__, location=error_location(error))
        raise
    shape = _context["shape"]
    measured(shape)["hf_constructor"].append(
        {"seconds": time.monotonic() - started, "device": str(self._device), "dtype": str(self._dtype)}
    )
    event("hf_constructor_return", device=str(self._device), dtype=str(self._dtype))


def observed_wrapper_init(self, *args, **kwargs):
    started = time.monotonic()
    _original_wrapper_init(self, *args, **kwargs)
    measured(_context["shape"])["wrapper_constructor"].append(time.monotonic() - started)


def observed_embed(self, texts, *, as_list=False):
    shape = _context["shape"]
    current_phase = _context["phase"]
    started = time.monotonic()
    event("hf_embed_enter", texts=len(texts))
    try:
        result = _original_embed(self, texts, as_list=as_list)
    except BaseException as error:
        event("hf_embed_error", error_class=type(error).__name__, location=error_location(error))
        raise
    completed = time.monotonic()
    dimension = len(result[0]) if len(result) else 0
    measured(shape)["hf_embed"].append(
        {
            "phase": current_phase,
            "texts": len(texts),
            "vectors": len(result),
            "dimension": dimension,
            "seconds": completed - started,
            "actual_encode_completed": completed,
        }
    )
    event("hf_encode_completed", texts=len(texts), vectors=len(result), dimension=dimension)
    if current_phase == "cancel" and _hold_armed.is_set() and len(texts) > 1:
        _encoded.set()
        event("hf_held_after_actual_encode_before_return")
        if not _release.wait(MAX_HOLD_SECONDS):
            event("hf_hold_expired")
            raise TimeoutError("held_actual_producer_release_expired")
        event("hf_embed_released")
    return result


def observed_wrapper_close(self):
    _original_wrapper_close(self)
    measured(_context["shape"])["wrapper_close"].append(time.monotonic())
    event("wrapper_close_return")


def observed_carrier_close(self):
    _original_carrier_close(self)
    measured(_context["shape"])["carrier_close"].append(time.monotonic())
    event("carrier_close_return")


def blocked_http(*_args, **_kwargs):
    _network_attempts.append(time.monotonic())
    raise AssertionError("offline_embedding_http_attempted")


def corpus_witness(root):
    """Stream all retained file bytes, with stable relative-path ordering."""
    digest = sha256()
    count = total = 0
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        if any((Path(directory) / name).is_symlink() for name in directories):
            raise ValueError("unexpected_corpus_directory_link")
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if not path.is_file() or path.is_symlink():
                raise ValueError("unexpected_corpus_member")
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            content = sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    content.update(block)
            digest.update(
                f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0{content.hexdigest()}\n".encode()
            )
            count += 1
            total += stat.st_size
    return {"files": count, "bytes": total, "path_stat_content_sha256": digest.hexdigest()}


def preflight_shape(app, shape):
    launch = json.loads((BASE / f"{shape}-launch.json").read_text(encoding="utf-8"))
    root = Path(launch["corpus"])
    if root != BASE / shape or not root.is_dir():
        raise ValueError("unexpected_retained_corpus_root")
    if str(config.get_user_data_dir()) != str(PROFILE / "home/.local/share/tldw_cli/default_user"):
        raise ValueError("unexpected_retained_profile")
    registry = app.workspace_registry_service
    binding = registry.get_runtime_binding(launch["key"]["binding_id"])
    if binding is None or binding.locator != str(root):
        raise ValueError("retained_registry_binding_changed")
    service = get_project_index_service(app)
    key = project_index_key(app, launch["key"]["workspace_id"], binding.binding_id, root)
    if asdict(key) != launch["key"]:
        raise ValueError("retained_key_changed")
    keyword = service.status(key)
    if keyword.state != "Ready" or keyword.indexed != 20000:
        raise ValueError("retained_keyword_not_ready")
    if service._factory.data_root != config.get_user_data_dir():
        raise ValueError("retained_service_data_root_changed")
    semantic = service.semantic_state(key)
    policy = service.policy(keyword.scope)
    if shape == "wide" and (
        semantic.state != "Off" or semantic.enabled or semantic.semantic_revision != 0
    ):
        raise ValueError("wide_first_semantic_build_preflight_changed")
    event(
        "retained_key_preflight",
        shape=shape,
        keyword_state=keyword.state,
        keyword_indexed=keyword.indexed,
        keyword_epoch=keyword.scope.epoch,
        semantic_state=semantic.state,
        semantic_revision=semantic.semantic_revision,
    )
    return service, key, root, keyword, semantic, policy


def retained_ready_generation(service, key, semantic, artifact_fingerprint):
    """Read the exact persisted generation under the genuine service fence."""
    with service._store(key, cleanup_only=True) as store:
        with service._read_fenced(store, key) as authority:
            if authority is None:
                raise ValueError("nested_ready_authority_unavailable")
            generation = store.semantic_ready_generation(
                semantic.scope, semantic, artifact_fingerprint
            )
            if generation is None:
                raise ValueError("nested_ready_model_or_scope_incompatible")
            with store.connection() as conn:
                count, first, last = conn.execute(
                    "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM semantic_members "
                    "WHERE generation_id=?",
                    (generation.generation_id,),
                ).fetchone()
    return generation, {"count": count, "first_ordinal": first, "last_ordinal": last}


def generation_receipt_row(generation):
    """Flatten the projected scope to the prior raw SQLite row's exact shape."""
    row = asdict(generation)
    row.pop("scope")
    row.update(
        namespace=generation.scope.key.namespace,
        epoch=generation.scope.epoch,
        processing_revision=generation.scope.processing_revision,
    )
    return row


def keyword_without_shared_cache_bytes(status):
    return replace(status, cache_bytes=0)


def checked_keyword_snapshot(service, key, baseline, policy, stage):
    observed = service.status(key)
    observed_policy = service.policy(baseline.scope)
    if (
        keyword_without_shared_cache_bytes(observed)
        != keyword_without_shared_cache_bytes(baseline)
        or observed_policy != policy
    ):
        raise AssertionError(f"keyword_or_policy_changed_{stage}")
    return {"status_all_fields": asdict(observed), "policy": asdict(observed_policy)}


async def bounded_thread(call, seconds, name, owners):
    task = asyncio.create_task(asyncio.to_thread(call), name=name)
    owners.append(task)
    done, _ = await asyncio.wait({task}, timeout=seconds)
    if not done:
        event("bounded_owner_pending", owner=name, seconds=seconds)
        raise TimeoutError(f"{name}_bound")
    return task.result()


def sampler(process, cache, current_phase, stop, peaks, errors, baseline):
    while not stop.is_set():
        try:
            parent = process.memory_info().rss
            children = 0
            for child in process.children(recursive=True):
                try:
                    children += child.memory_info().rss
                except psutil.Error:
                    pass
            size = binding_bytes(cache)
            with _sample_lock:
                sample = peaks.setdefault(
                    current_phase[0],
                    {"parent_rss": baseline, "combined_rss": baseline, "whole_binding_bytes": 0},
                )
                sample["parent_rss"] = max(sample["parent_rss"], parent)
                sample["combined_rss"] = max(sample["combined_rss"], parent + children)
                sample["whole_binding_bytes"] = max(sample["whole_binding_bytes"], size)
            event(
                "sample",
                sampled_parent_rss=parent,
                sampled_combined_rss=parent + children,
                whole_binding_bytes=size,
            )
        except BaseException as error:
            errors.append(type(error).__name__)
            event("sampler_error", error_class=type(error).__name__)
            return
        stop.wait(5)


async def qualify_shape(snapshot):
    service, key, root, keyword_before, initial_state, policy_before = snapshot
    shape = key.workspace_id.removeprefix("qualification-")
    phase(shape, "source_before")
    report = {
        "shape": shape,
        "outcome": "adverse_incomplete",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "initial_semantic_state": initial_state.state,
        "initial_semantic_revision": initial_state.semantic_revision,
        "keyword_manifest_revision": keyword_before.manifest_revision,
        "keyword_status_before_all_fields": asdict(keyword_before),
        "keyword_policy_before": asdict(policy_before),
        "old_final_01_full_prebuild_keyword_status_available": False,
        "qualification_kind": (
            "persisted_nested_ready_query_cancel_only" if shape == "nested"
            else "first_wide_build_query_cancel"
        ),
        "bounds_seconds": {
            "build": MAX_BUILD_SECONDS,
            "query": MAX_QUERY_SECONDS,
            "native_join": MAX_NATIVE_JOIN_SECONDS,
            "actual_encode_hold": MAX_HOLD_SECONDS,
        },
    }
    owners = []
    build_run = cancel_run = None
    sample_stop = threading.Event()
    sample_thread = None
    peaks = {}
    sample_errors = []
    current_phase = ["preflight"]
    process = psutil.Process()
    cache = index_store_path(service._factory.data_root, key).parent.parent
    report_path = SDD / f"{RUN_NAME}-{shape}-report.json"

    def save_report(stage):
        report["last_completed_stage"] = stage
        report["current_phase"] = _context["phase"]
        with _sample_lock:
            report["phase_peaks_sampled_5s_absolute"] = {
                name: dict(values) for name, values in peaks.items()
            }
        report["sampler_errors"] = list(sample_errors)
        report["owners"] = {
            "build_native_retired": build_run.native_retired if build_run is not None else None,
            "build_thread_alive": build_run.thread.is_alive() if build_run is not None else None,
            "cancel_native_retired": cancel_run.native_retired if cancel_run is not None else None,
            "cancel_thread_alive": cancel_run.thread.is_alive() if cancel_run is not None else None,
            "pending_to_thread_tasks": [task.get_name() for task in owners if not task.done()],
            "sampler_thread_alive": sample_thread.is_alive() if sample_thread is not None else None,
        }
        pending_path = report_path.with_suffix(".json.tmp")
        pending_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending_path.replace(report_path)

    try:
        save_report("source_before_started")
        report["source_before"] = await bounded_thread(
            lambda: corpus_witness(root), 900, "corpus_before", owners
        )
        save_report("source_before_completed")
        if report["source_before"]["files"] != 20000:
            raise AssertionError("retained_corpus_file_count_changed")
        if shape == "nested":
            old = json.loads(
                (SDD / "task4-semantic-scale-final-01-nested-report.json").read_text(
                    encoding="utf-8"
                )
            )
            if report["source_before"] != old["source_after"]:
                raise AssertionError("nested_corpus_changed_since_final_01")
            report["nested_source_matches_final_01"] = True
        report["rss_baseline_before_model_bytes"] = process.memory_info().rss
        report["whole_binding_baseline_bytes"] = await bounded_thread(
            lambda: binding_bytes(cache), 120, "binding_baseline", owners
        )
        sample_thread = threading.Thread(
            target=sampler,
            args=(
                process, cache, current_phase, sample_stop, peaks, sample_errors,
                report["rss_baseline_before_model_bytes"],
            ),
            name=f"task4-scale-{shape}-sampler",
        )
        sample_thread.start()

        phase(shape, "consent")
        current_phase[0] = "consent"
        state = await bounded_thread(lambda: service.semantic_state(key), 30, "consent_read", owners)
        candidate = _settings_for_project_model("minilm-local")
        spec, destination = _resolved_spec("minilm-local")
        target = _target(spec, destination)
        approved = replace(
            candidate,
            approved_destination_fingerprint=candidate.destination_fingerprint,
        )
        report["selected_model"] = {
            "configured_alias": "minilm-local",
            "selected_cached_model": MODEL,
            "resolved_model_snapshot": target.model_name,
            "artifact_fingerprint": target.artifact_fingerprint,
            "config_fingerprint": approved.config_fingerprint,
            "destination_fingerprint": approved.destination_fingerprint,
            "provider_id": approved.provider_id,
        }
        save_report("selected_model_resolved")
        if shape == "nested":
            if report["selected_model"] != old["selected_model"]:
                raise AssertionError("nested_selected_model_changed_since_final_01")
            if (
                state != initial_state
                or state.state != "Ready"
                or not state.enabled
                or state.semantic_revision != EXPECTED_NESTED_SEMANTIC_REVISION
                or state.ready_generation_id != EXPECTED_NESTED_READY_ID
                or state.settings != approved
                or keyword_before.manifest_revision != EXPECTED_NESTED_MANIFEST_REVISION
            ):
                raise AssertionError("nested_persisted_ready_consent_changed")
            prior_receipt = json.loads(
                (SDD / "task4-nested-ready-read-01-child.json").read_text(encoding="utf-8")
            )
            if (
                prior_receipt["ready_generation"]["generation_id"] != EXPECTED_NESTED_READY_ID
                or prior_receipt["ready_generation"]["semantic_revision"]
                != EXPECTED_NESTED_SEMANTIC_REVISION
                or prior_receipt["ready_generation"]["manifest_revision"]
                != EXPECTED_NESTED_MANIFEST_REVISION
            ):
                raise AssertionError("nested_readonly_receipt_identity_changed")
            generation, members = await bounded_thread(
                lambda: retained_ready_generation(
                    service, key, state, target.artifact_fingerprint
                ),
                120,
                "nested_ready_generation_read",
                owners,
            )
            if (
                generation.generation_id != EXPECTED_NESTED_READY_ID
                or generation.semantic_revision != EXPECTED_NESTED_SEMANTIC_REVISION
                or generation.manifest_revision != EXPECTED_NESTED_MANIFEST_REVISION
                or generation.chunk_count != 20000
                or generation.dimension != 384
                or generation.checkpoint_chunk_id != 20000
                or generation.claim_token is not None
                or generation.artifact_fingerprint != target.artifact_fingerprint
                or members != {"count": 20000, "first_ordinal": 1, "last_ordinal": 20000}
                or state.coverage != 20000
                or generation_receipt_row(generation) != prior_receipt["ready_generation"]
            ):
                raise AssertionError("nested_persisted_ready_generation_changed")
            report["reused_nested_generation"] = asdict(generation)
            report["reused_nested_members"] = members
            report["semantic_build_seconds"] = None
            report["nested_build_timing_replaced"] = False
            report["ready_generation_id"] = generation.generation_id
            report["keyword_after_ready_validation"] = await bounded_thread(
                lambda: checked_keyword_snapshot(
                    service, key, keyword_before, policy_before, "nested_ready"
                ),
                30,
                "nested_keyword_ready_validation",
                owners,
            )
            event("nested_persisted_ready_reused", generation_id=generation.generation_id)
        else:
            if state.state != "Off" or state.enabled:
                prior = state
                state = await bounded_thread(
                    lambda: service.disable_semantic(
                        key,
                        expected_epoch=prior.scope.epoch,
                        expected_semantic_revision=prior.semantic_revision,
                    ),
                    MAX_NATIVE_JOIN_SECONDS,
                    "revoke_retained_semantic",
                    owners,
                )
                report["prebuild_revoke"] = {
                    "from_state": prior.state,
                    "from_revision": prior.semantic_revision,
                    "returned_state": state.state,
                    "returned_revision": state.semantic_revision,
                }
                event("retained_semantic_revoked", **report["prebuild_revoke"])
            if state.state != "Off" or state.enabled:
                raise AssertionError("prebuild_semantic_not_off")
            semantic = await bounded_thread(
                lambda: service.enable_semantic(
                    key,
                    approved,
                    expected_epoch=keyword_before.scope.epoch,
                    expected_semantic_revision=state.semantic_revision,
                ),
                120,
                "enable_semantic",
                owners,
            )
            report["enabled_semantic_revision"] = semantic.semantic_revision
            phase(shape, "build")
            current_phase[0] = "build"
            started = time.monotonic()
            job_id = await bounded_thread(
                lambda: service.refresh_semantic(
                    key,
                    expected_epoch=keyword_before.scope.epoch,
                    expected_semantic_revision=semantic.semantic_revision,
                ),
                120,
                "refresh_semantic",
                owners,
            )
            build_run = service._semantic_runs[job_id]
            event("build_started", job_revision=semantic.semantic_revision)
            save_report("build_started")
            while build_run.thread.is_alive():
                elapsed = time.monotonic() - started
                if elapsed >= MAX_BUILD_SECONDS:
                    build_run.stop.set()
                    await bounded_thread(
                        lambda: build_run.thread.join(MAX_NATIVE_JOIN_SECONDS),
                        MAX_NATIVE_JOIN_SECONDS + 10,
                        "timed_out_build_join",
                        owners,
                    )
                    report["build_timeout_retirement"] = {
                        "thread_alive": build_run.thread.is_alive(),
                        "native_retired": build_run.native_retired,
                    }
                    raise TimeoutError("semantic_build_exceeded_3600_seconds")
                await asyncio.sleep(min(2, MAX_BUILD_SECONDS - elapsed))
            report["semantic_build_seconds"] = time.monotonic() - started
            if not build_run.native_retired or build_run.error is not None:
                raise AssertionError("semantic_build_retirement_unproven")
            ready = await bounded_thread(lambda: service.semantic_state(key), 30, "ready_read", owners)
            if ready.state != "Ready" or not ready.ready_generation_id:
                raise AssertionError("semantic_ready_unproved")
            report["keyword_after_build"] = await bounded_thread(
                lambda: checked_keyword_snapshot(
                    service, key, keyword_before, policy_before, "wide_build"
                ),
                30,
                "wide_keyword_after_build",
                owners,
            )
            report["ready_generation_id"] = ready.ready_generation_id
            report["build_hf_embeds"] = len(
                [x for x in measured(shape)["hf_embed"] if x["phase"] == "build"]
            )
        save_report("ready_verified")

        phase(shape, "query")
        current_phase[0] = "query"
        rig = SimpleNamespace(service=service, root=root, key=key)
        query = "qualificationdeepmarker"
        before_embed = len(measured(shape)["hf_embed"])
        before_ctor = len(measured(shape)["wrapper_constructor"])
        query_started = time.monotonic()
        result = await bounded_thread(
            lambda: invoke(provider(rig), {"query": query}),
            MAX_QUERY_SECONDS,
            "selected_semantic_query",
            owners,
        )
        query_seconds = time.monotonic() - query_started
        query_embeds = measured(shape)["hf_embed"][before_embed:]
        query_ctors = measured(shape)["wrapper_constructor"][before_ctor:]
        payload = json.loads(result.content) if result.ok else {}
        if (
            not result.ok
            or payload.get("semantic_status") != "ready"
            or "f19999.txt" not in result.content
            or not query_embeds
            or any(x["phase"] != "query" or x["dimension"] != 384 for x in query_embeds)
            or not query_ctors
        ):
            raise AssertionError("selected_query_semantic_proof_failed")
        report["queries"] = [{
            "selected_marker": query,
            "whole_tool_seconds": query_seconds,
            "wrapper_constructor_seconds": query_ctors,
            "actual_hf_encode_seconds": [x["seconds"] for x in query_embeds],
            "actual_hf_encode_dimensions": [x["dimension"] for x in query_embeds],
            "semantic_status": payload["semantic_status"],
            "selected_path_found": True,
        }]
        report["keyword_after_query"] = await bounded_thread(
            lambda: checked_keyword_snapshot(
                service, key, keyword_before, policy_before, "query"
            ),
            30,
            "keyword_after_query",
            owners,
        )
        save_report("selected_query_verified")

        phase(shape, "cancel")
        current_phase[0] = "cancel"
        _encoded.clear()
        _release.clear()
        _hold_armed.set()
        current = await bounded_thread(lambda: service.semantic_state(key), 30, "cancel_state", owners)
        close_before = len(measured(shape)["wrapper_close"])
        carrier_before = len(measured(shape)["carrier_close"])
        cancel_id = await bounded_thread(
            lambda: service.refresh_semantic(
                key,
                expected_epoch=current.scope.epoch,
                expected_semantic_revision=current.semantic_revision,
            ),
            120,
            "cancel_refresh",
            owners,
        )
        cancel_run = service._semantic_runs[cancel_id]
        save_report("cancel_build_started")
        if not await bounded_thread(lambda: _encoded.wait(120), 125, "actual_encode_wait", owners):
            raise AssertionError("cancel_actual_hf_encode_not_reached")
        if not cancel_run.thread.is_alive():
            raise AssertionError("cancel_worker_not_alive_at_hold")
        cancel_started = time.monotonic()
        disable = asyncio.create_task(
            asyncio.to_thread(
                service.disable_semantic,
                key,
                expected_epoch=current.scope.epoch,
                expected_semantic_revision=current.semantic_revision,
            ),
            name=f"{shape}_disable",
        )
        owners.append(disable)
        event("disable_submitted_after_actual_encode")
        # A single bounded read proves durable Off while the producer is held.
        await asyncio.sleep(10)
        observed = await bounded_thread(
            lambda: service.semantic_state(key),
            MAX_STATE_READ_SECONDS,
            "durable_off_read",
            owners,
        )
        report["cancel_hold"] = {
            "point": "after_actual_hf_encode_before_vectors_return_to_service",
            "observed_state_while_held": observed.state,
            "worker_alive_while_held": cancel_run.thread.is_alive(),
            "disable_done_while_held": disable.done(),
        }
        save_report("durable_off_while_actual_producer_held")
        if observed.state != "Off" or not cancel_run.thread.is_alive() or disable.done():
            raise AssertionError("durable_revocation_while_held_unproved")
        _release.set()
        _hold_armed.clear()
        done, _ = await asyncio.wait({disable}, timeout=MAX_NATIVE_JOIN_SECONDS)
        if not done:
            raise TimeoutError("disable_retirement_bound")
        returned = disable.result()
        await bounded_thread(
            lambda: cancel_run.thread.join(MAX_NATIVE_JOIN_SECONDS),
            MAX_NATIVE_JOIN_SECONDS + 10,
            "cancel_worker_join",
            owners,
        )
        report["cancel"] = {
            "returned_state": returned.state,
            "disable_to_retirement_seconds": time.monotonic() - cancel_started,
            "thread_alive_after_join": cancel_run.thread.is_alive(),
            "native_retired": cancel_run.native_retired,
            "wrapper_close_delta": len(measured(shape)["wrapper_close"]) - close_before,
            "carrier_close_delta": len(measured(shape)["carrier_close"]) - carrier_before,
        }
        if (
            returned.state != "Off"
            or cancel_run.thread.is_alive()
            or not cancel_run.native_retired
            or report["cancel"]["wrapper_close_delta"] < 1
            or report["cancel"]["carrier_close_delta"] < 1
        ):
            raise AssertionError("cancel_physical_retirement_unproved")
        if (await bounded_thread(lambda: service.semantic_state(key), 30, "off_after_cancel", owners)).state != "Off":
            raise AssertionError("stale_semantic_ready_after_cancel")
        report["keyword_after_cancel"] = await bounded_thread(
            lambda: checked_keyword_snapshot(
                service, key, keyword_before, policy_before, "cancel"
            ),
            30,
            "keyword_after_cancel",
            owners,
        )
        if _network_attempts:
            raise AssertionError("offline_http_attempted")
        report["functional_phases_complete"] = True
        save_report("cancellation_and_keyword_verified")
    except BaseException as error:
        report["error_class"] = type(error).__name__
        report["error_location"] = error_location(error)
        event("shape_adverse", error_class=type(error).__name__)
        save_report("error_before_cleanup")
        raise
    finally:
        phase(shape, "cleanup")
        current_phase[0] = "cleanup"
        _release.set()
        _hold_armed.clear()
        save_report("cleanup_started")
        for run in (cancel_run, build_run):
            if run is not None and run.thread is not None and run.thread.is_alive():
                run.stop.set()
                try:
                    await bounded_thread(
                        lambda worker=run: worker.thread.join(MAX_NATIVE_JOIN_SECONDS),
                        MAX_NATIVE_JOIN_SECONDS + 10,
                        "cleanup_native_worker_join",
                        owners,
                    )
                except BaseException as error:
                    report["cleanup_join_error_class"] = type(error).__name__
                    save_report("cleanup_join_adverse")
        sample_stop.set()
        if sample_thread is not None:
            try:
                await bounded_thread(lambda: sample_thread.join(30), 40, "sampler_join", owners)
            except BaseException as error:
                report["sampler_join_error_class"] = type(error).__name__
                save_report("sampler_join_adverse")
        report["sampler_retired"] = sample_thread is None or not sample_thread.is_alive()
        report["sampler_errors"] = sample_errors
        save_report("cleanup_owners_joined")
        pending = [task for task in owners if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=MAX_NATIVE_JOIN_SECONDS)
        save_report("owner_drain_before_final_witness")
        if any(not task.done() for task in owners):
            report["final_witness_error_class"] = "PendingOwner"
            save_report("final_witness_skipped_pending_owner")
        else:
            try:
                report["source_after"] = await bounded_thread(
                    lambda: corpus_witness(root), 900, "corpus_after", owners
                )
                report["source_unchanged"] = report.get("source_before") == report["source_after"]
                report["whole_binding_bytes_final"] = await bounded_thread(
                    lambda: binding_bytes(cache), 120, "binding_final", owners
                )
                report["rss_final_bytes"] = process.memory_info().rss
                save_report("final_witnesses_completed")
            except BaseException as error:
                report["final_witness_error_class"] = type(error).__name__
                save_report("final_witness_adverse")
        pending = [task for task in owners if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=MAX_NATIVE_JOIN_SECONDS)
        report["network_attempts"] = len(_network_attempts)
        save_report("owner_drain_completed")
        if (
            not report.get("functional_phases_complete")
            or not report["sampler_retired"]
            or sample_errors
            or not report.get("source_unchanged", False)
            or "final_witness_error_class" in report
            or "cleanup_join_error_class" in report
            or "sampler_join_error_class" in report
            or "whole_binding_bytes_final" not in report
            or "rss_final_bytes" not in report
            or report["owners"]["pending_to_thread_tasks"]
            or report["owners"]["build_thread_alive"]
            or report["owners"]["cancel_thread_alive"]
        ):
            report["outcome"] = "adverse_cleanup_or_source_unproved"
        else:
            report["outcome"] = (
                "qualified_continuation" if shape == "nested" else "qualified_shape"
            )
        save_report("final_report_saved")
        event("shape_report_saved", outcome=report["outcome"])
    expected_outcome = "qualified_continuation" if shape == "nested" else "qualified_shape"
    if report["outcome"] != expected_outcome:
        raise AssertionError("shape_cleanup_or_source_unproved")


async def main():
    preflight = json.loads((PROFILE / "preflight-2.json").read_text(encoding="utf-8"))
    data = config.get_user_data_dir()
    if str(data) != preflight["data"] or list(verified_data_identity()) != preflight["verified_identity"]:
        raise ValueError("retained_profile_identity_changed")
    app = _build_test_app(user_data_dir=data)
    service_closed = False
    preflight_owners = []
    try:
        # Validate BOTH retained bindings before any cache mutation.
        snapshots = [
            await bounded_thread(
                lambda selected=shape: preflight_shape(app, selected),
                120,
                f"preflight_{shape}",
                preflight_owners,
            )
            for shape in ("nested", "wide")
        ]
        for snapshot in snapshots:
            await qualify_shape(snapshot)
        if _network_attempts:
            raise AssertionError("offline_http_attempted")
    finally:
        phase(None, "app_cleanup")
        _release.set()
        pending_preflight = [task for task in preflight_owners if not task.done()]
        if pending_preflight:
            await asyncio.wait(pending_preflight, timeout=MAX_NATIVE_JOIN_SECONDS)
        event(
            "preflight_owner_retirement",
            pending=[task.get_name() for task in preflight_owners if not task.done()],
        )
        close_task = asyncio.create_task(close_project_index_service(app), name="close_project_index")
        try:
            done, _ = await asyncio.wait({close_task}, timeout=MAX_NATIVE_JOIN_SECONDS)
            if done:
                close_task.result()
                service_closed = True
                drain_task = asyncio.create_task(drain_created_runtimes(), name="drain_app_runtimes")
                drained, _ = await asyncio.wait({drain_task}, timeout=MAX_NATIVE_JOIN_SECONDS)
                if not drained:
                    raise TimeoutError("app_runtime_drain_bound")
                drain_task.result()
                drain_created_dirs()
            else:
                raise TimeoutError("app_service_close_bound")
        finally:
            event("app_cleanup", service_closed=service_closed, close_task_done=close_task.done())
    reports = [
        json.loads((SDD / f"{RUN_NAME}-{shape}-report.json").read_text(encoding="utf-8"))
        for shape in ("nested", "wide")
    ]
    if not service_closed or [x["outcome"] for x in reports] != [
        "qualified_continuation", "qualified_shape"
    ]:
        raise AssertionError("scale_qualification_incomplete")
    (SDD / f"{RUN_NAME}-complete.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    with (
        patch.object(config, "load_cli_config_and_ensure_existence", configured),
        patch.object(_HuggingFaceEmbedder, "__init__", observed_hf_init),
        patch.object(_HuggingFaceEmbedder, "embed", observed_embed),
        patch.object(EmbeddingsServiceWrapper, "__init__", observed_wrapper_init),
        patch.object(EmbeddingsServiceWrapper, "close", observed_wrapper_close),
        patch.object(ProjectionExecutionScope, "close", observed_carrier_close),
        patch.object(requests.sessions.Session, "request", blocked_http),
        patch.object(httpx.Client, "send", blocked_http),
        patch.object(httpx.AsyncClient, "send", blocked_http),
    ):
        try:
            asyncio.run(main())
        except BaseException as error:
            adverse = {
                "outcome": "adverse_incomplete",
                "error_class": type(error).__name__,
                "error_location": error_location(error),
                "network_attempts": len(_network_attempts),
                "shape_reports_present": [
                    shape for shape in ("nested", "wide")
                    if (SDD / f"{RUN_NAME}-{shape}-report.json").exists()
                ],
            }
            (SDD / f"{RUN_NAME}-adverse.json").write_text(
                json.dumps(adverse, indent=2), encoding="utf-8"
            )
            event("process_adverse", **adverse)
            raise
