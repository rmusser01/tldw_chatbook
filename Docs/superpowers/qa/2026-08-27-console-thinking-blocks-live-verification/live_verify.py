"""Isolated mounted/live verification for Console model-thinking disclosures.

The script deliberately drives production controllers, persistence, request
preparation, and Textual widgets with deterministic adapter-edge fixtures. It writes
fresh painted SVG frames and a JSON observation ledger beneath ``QA_EVIDENCE_ROOT``.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree


EVIDENCE_ROOT = Path(os.environ["QA_EVIDENCE_ROOT"]).resolve()
PROFILE_ROOT = EVIDENCE_ROOT / "profile"
CAPTURE_ROOT = EVIDENCE_ROOT / "captures"
SCRATCH_ROOT = EVIDENCE_ROOT / "scratch"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


for variable in (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "TLDW_CONFIG_PATH",
    "TMPDIR",
):
    value = os.environ.get(variable)
    if not value or not _is_within(Path(value), EVIDENCE_ROOT):
        raise RuntimeError(f"{variable} is not isolated beneath QA_EVIDENCE_ROOT")

for directory in (
    PROFILE_ROOT / "home",
    PROFILE_ROOT / "xdg-config",
    PROFILE_ROOT / "data",
    PROFILE_ROOT / "cache",
    PROFILE_ROOT / "tmp",
):
    directory.mkdir(parents=True, exist_ok=True)

config_path = Path(os.environ["TLDW_CONFIG_PATH"])
if not config_path.exists():
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            (
                "[paths]",
                f"data_dir = {json.dumps(str(PROFILE_ROOT / 'data'))}",
                "",
                "[first_run]",
                "setup_completed = true",
                "",
                "[console]",
                "show_model_thinking = true",
                'thinking_history_policy_default = "auto"',
                "",
                "[model_catalog]",
                "enabled = false",
                "",
            )
        ),
        encoding="utf-8",
    )

CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

# Imports happen only after the environment isolation contract is validated.
from textual.widgets import Select, Static  # noqa: E402

from Tests.UI.app_factory import (  # noqa: E402
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_console_context_controls import (  # noqa: E402
    _ContextHarness,
    _settings,
    _state,
)
from Tests.UI.test_console_thinking_disclosures import (  # noqa: E402
    StyledThinkingTranscriptHarness,
)
from Tests.integration import test_console_thinking_end_to_end as joined  # noqa: E402
from tldw_chatbook.Chat.console_chat_controller import (  # noqa: E402
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_models import (  # noqa: E402
    PROPRIETARY_THINKING_NOTICE,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore  # noqa: E402
from tldw_chatbook.Chat.console_session_settings import (  # noqa: E402
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Chat.chat_persistence_service import (  # noqa: E402
    ChatPersistenceService,
)
from tldw_chatbook.Chat.console_turn_grouping import (  # noqa: E402
    project_thinking_activities,
)
from tldw_chatbook.Chat.console_provider_gateway import (  # noqa: E402
    ProviderProprietaryThinkingEvidence,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB  # noqa: E402
from tldw_chatbook.UI.Screens.settings_context_memory import (  # noqa: E402
    load_show_model_thinking,
)
from tldw_chatbook.Widgets.Console.console_assistant_turn import (  # noqa: E402
    ConsoleActivityDisclosure,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (  # noqa: E402
    ConsoleSettingsModal,
)
from tldw_chatbook.Widgets.Console.console_transcript import (  # noqa: E402
    ConsoleTranscript,
)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_frame(
    app, name: str, *, required_markers: tuple[str, ...]
) -> dict[str, object]:
    svg = app.export_screenshot(title=name, simplify=True)
    svg = "\n".join(line.rstrip() for line in svg.splitlines())
    root = ElementTree.fromstring(svg)
    painted = " ".join(
        html.unescape(fragment).strip()
        for fragment in root.itertext()
        if fragment.strip()
    )
    painted = " ".join(painted.split())
    for marker in required_markers:
        if " ".join(marker.split()) not in painted:
            raise AssertionError(f"painted frame {name!r} omitted marker {marker!r}")
    path = CAPTURE_ROOT / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    return {
        "path": str(path.relative_to(EVIDENCE_ROOT)),
        "sha256": hashlib.sha256(svg.encode("utf-8")).hexdigest(),
        "markers": list(required_markers),
    }


def _disclosure(
    transcript: ConsoleTranscript,
    assistant,
) -> ConsoleActivityDisclosure:
    activity = project_thinking_activities(assistant=assistant)[0]
    return transcript.query_one(
        f"#console-activity-disclosure-{activity.activity_id}",
        ConsoleActivityDisclosure,
    )


async def _capture_live_lifecycle(observations: dict[str, object]) -> None:
    db = CharactersRAGDB(SCRATCH_ROOT / "lifecycle.sqlite", "live-lifecycle")
    gateway = joined._PausedThinkingGateway()
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Live thinking verification")
    store.active_session_id = session.id
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    try:
        send = asyncio.create_task(controller.submit_draft("Live question"))
        await asyncio.wait_for(gateway.evidence_seen.wait(), timeout=3)
        live_messages = store.messages_for_session(session.id)
        live = live_messages[-1]
        assert live.role is ConsoleMessageRole.ASSISTANT
        assert live.content == ""
        assert live.thinking is not None

        app = StyledThinkingTranscriptHarness()
        async with app.run_test(size=(100, 30)) as pilot:
            transcript = app.query_one(ConsoleTranscript)
            transcript.set_messages(live_messages, session_id=session.id)
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            disclosure = _disclosure(transcript, live)
            activity_id = disclosure.activity_message_id
            assert disclosure.expanded
            assert activity_id in transcript._pending_thinking_auto_collapse
            live_frame = _save_frame(
                app,
                "01-live-displayable-expanded",
                required_markers=("Thinking", joined.DISPLAYABLE_THINKING),
            )

            gateway.release_answer.set()
            result = await asyncio.wait_for(send, timeout=3)
            assert result.accepted
            completed_messages = store.messages_for_session(session.id)
            completed = completed_messages[-1]
            transcript.set_messages(completed_messages, session_id=session.id)
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            collapsed = _disclosure(transcript, completed)
            assert collapsed is disclosure
            assert not collapsed.expanded
            assert activity_id not in transcript._pending_thinking_auto_collapse
            collapsed_frame = _save_frame(
                app,
                "02-answer-boundary-collapsed",
                required_markers=("Thinking", joined.VISIBLE_ANSWER),
            )

            assert load_show_model_thinking({}) is True
            assert transcript.set_model_thinking_visible(False)
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            assert not list(transcript.query(ConsoleActivityDisclosure))
            hidden_frame = _save_frame(
                app,
                "03-presentation-setting-off",
                required_markers=(joined.VISIBLE_ANSWER,),
            )

            assert transcript.set_model_thinking_visible(True)
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            restored_visibility = _disclosure(transcript, completed)
            assert not restored_visibility.expanded

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None
        durable = db.get_message_by_id(completed.persisted_message_id)
        assert durable["thinking_blocks_json"] is not None

        resumed_store, resumed_session = joined._resume_store(db, conversation_id)
        restored = resumed_store.messages_for_session(resumed_session.id)[-1]
        restarted_app = StyledThinkingTranscriptHarness()
        async with restarted_app.run_test(size=(100, 30)) as pilot:
            transcript = restarted_app.query_one(ConsoleTranscript)
            transcript.set_messages(
                resumed_store.messages_for_session(resumed_session.id),
                session_id=resumed_session.id,
            )
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            historical = _disclosure(transcript, restored)
            assert not historical.expanded
            assert historical.detail_available
            assert not historical.detail_stack.children
            restart_frame = _save_frame(
                restarted_app,
                "04-restart-history-collapsed",
                required_markers=("Thinking", joined.VISIBLE_ANSWER),
            )

        observations["live_lifecycle"] = {
            "actual_event_only": True,
            "expanded_live": True,
            "collapsed_at_answer": True,
            "one_time_transition_consumed": True,
            "default_visibility": True,
            "off_hides_immediately": True,
            "on_restores_collapsed": True,
            "restart_collapsed_lazy": True,
            "durable_envelope_present": True,
            "frames": [live_frame, collapsed_frame, hidden_frame, restart_frame],
        }
    finally:
        gateway.release_answer.set()
        db.close_connection()
        drain_active_service_patches()
        drain_created_dirs()


async def _capture_proprietary_and_no_evidence(
    observations: dict[str, object],
) -> None:
    proprietary_db = CharactersRAGDB(
        SCRATCH_ROOT / "proprietary.sqlite", "live-proprietary"
    )
    proprietary_gateway = joined._ScriptedEvidenceGateway(
        "proprietary",
        ProviderProprietaryThinkingEvidence(
            provider="vllm",
            model="joined-reasoner",
            protocol="chat_completions",
            source_format="reasoning_content",
        ),
        joined.VISIBLE_ANSWER,
    )
    proprietary_store = ConsoleChatStore(
        persistence=ChatPersistenceService(proprietary_db)
    )
    proprietary_session = proprietary_store.create_session(title="Proprietary")
    proprietary_store.active_session_id = proprietary_session.id
    proprietary_controller = ConsoleChatController(
        store=proprietary_store,
        provider_gateway=proprietary_gateway,
    )

    no_evidence_db = CharactersRAGDB(
        SCRATCH_ROOT / "no-evidence.sqlite", "live-no-evidence"
    )
    no_evidence_gateway = joined._ScriptedEvidenceGateway(
        "displayable", joined.VISIBLE_ANSWER
    )
    no_evidence_store = ConsoleChatStore(
        persistence=ChatPersistenceService(no_evidence_db)
    )
    no_evidence_session = no_evidence_store.create_session(title="No evidence")
    no_evidence_store.active_session_id = no_evidence_session.id
    no_evidence_controller = ConsoleChatController(
        store=no_evidence_store,
        provider_gateway=no_evidence_gateway,
    )

    try:
        proprietary_result = await proprietary_controller.submit_draft("Question")
        assert proprietary_result.accepted
        proprietary = proprietary_store.messages_for_session(proprietary_session.id)[-1]
        raw = proprietary_db.get_message_by_id(proprietary.persisted_message_id)[
            "thinking_blocks_json"
        ]
        assert PROPRIETARY_THINKING_NOTICE not in raw
        assert "text" not in raw

        app = StyledThinkingTranscriptHarness()
        async with app.run_test(size=(100, 30)) as pilot:
            transcript = app.query_one(ConsoleTranscript)
            transcript.set_messages(
                proprietary_store.messages_for_session(proprietary_session.id),
                session_id=proprietary_session.id,
            )
            await transcript.refresh_messages()
            disclosure = _disclosure(transcript, proprietary)
            disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
            await pilot.pause(0.1)
            assert disclosure.expanded
            assert disclosure.status == "unavailable"
            assert (
                transcript.thinking_detail_text(disclosure.activity_message_id)
                == PROPRIETARY_THINKING_NOTICE
            )
            proprietary_frame = _save_frame(
                app,
                "05-proprietary-expanded",
                required_markers=("Thinking", PROPRIETARY_THINKING_NOTICE),
            )

        no_evidence_result = await no_evidence_controller.submit_draft("Question")
        assert no_evidence_result.accepted
        no_evidence = no_evidence_store.messages_for_session(no_evidence_session.id)[-1]
        assert no_evidence.thinking is None
        assert project_thinking_activities(assistant=no_evidence) == ()

        no_evidence_app = StyledThinkingTranscriptHarness()
        async with no_evidence_app.run_test(size=(100, 24)) as pilot:
            transcript = no_evidence_app.query_one(ConsoleTranscript)
            transcript.set_messages(
                no_evidence_store.messages_for_session(no_evidence_session.id),
                session_id=no_evidence_session.id,
            )
            await transcript.refresh_messages()
            await pilot.pause(0.1)
            assert not list(transcript.query(ConsoleActivityDisclosure))
            no_evidence_frame = _save_frame(
                no_evidence_app,
                "06-capable-no-evidence-no-row",
                required_markers=(joined.VISIBLE_ANSWER,),
            )

        observations["evidence_honesty"] = {
            "proprietary_exact_notice": PROPRIETARY_THINKING_NOTICE,
            "proprietary_raw_text_absent": True,
            "application_notice_not_durable": True,
            "capable_no_event_has_no_row": True,
            "frames": [proprietary_frame, no_evidence_frame],
        }
    finally:
        proprietary_db.close_connection()
        no_evidence_db.close_connection()


async def _capture_policy_states(observations: dict[str, object]) -> None:
    policies: dict[str, object] = {}
    for policy in ("auto", "include", "exclude"):
        state = _state(thinking_policy=policy)
        policies[policy] = {
            "saved": state.thinking_history.saved_policy,
            "effective": state.thinking_history.effective_label,
            "editable": state.thinking_history.effective_label != "Required",
        }

    required_state = _state(
        thinking_policy="exclude",
        effective_thinking_policy="required",
    )
    policies["required"] = {
        "saved": required_state.thinking_history.saved_policy,
        "effective": required_state.thinking_history.effective_label,
        "editable": required_state.thinking_history.effective_label != "Required",
    }

    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=required_state,
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        await pilot.pause(0.1)
        select = app.screen.query_one(
            "#console-context-thinking-history-policy", Select
        )
        effective_widget = app.screen.query_one(
            "#console-context-thinking-history-effective", Static
        )
        effective = str(effective_widget.renderable)
        assert select.value == "exclude"
        assert select.disabled
        assert "Effective: Required" in effective
        assert "provider continuation" in effective
        select.scroll_visible(top=True, immediate=True)
        await pilot.pause(0.1)
        required_frame = _save_frame(
            app,
            "07-thinking-history-required",
            required_markers=("Thinking history replay", "Effective: Required"),
        )
        await pilot.click("#console-settings-cancel")

    observations["policy_presentation"] = {
        "states": policies,
        "required_copy_mentions_provider_continuation": True,
        "frame": required_frame,
    }


async def _run_functional_controls(observations: dict[str, object]) -> None:
    replay_results: list[dict[str, object]] = []
    for saved, continuation_required, effective, count in (
        ("auto", False, "auto", 1),
        ("include", False, "include", 1),
        ("exclude", False, "exclude", 0),
        ("include", True, "required", 1),
    ):
        await joined.test_durable_owner_replay_is_counted_and_dispatched_exactly_once(
            saved,
            continuation_required,
            effective,
            count,
        )
        replay_results.append(
            {
                "saved": saved,
                "continuation_required": continuation_required,
                "effective": effective,
                "expected_replay_groups": count,
            }
        )

    backend_root = SCRATCH_ROOT / "backend-refusal"
    backend_root.mkdir(exist_ok=True)
    for disposition in ("displayable", "proprietary"):
        await joined.test_unsupported_persistent_backend_refuses_before_provider_and_recovers(
            backend_root,
            disposition,
        )

    plain_root = SCRATCH_ROOT / "plain-model"
    plain_root.mkdir(exist_ok=True)
    await joined.test_plain_local_model_uses_real_resolver_and_dispatches_on_v0_backend(
        plain_root
    )

    resumed_root = SCRATCH_ROOT / "resumed-refusal"
    resumed_root.mkdir(exist_ok=True)
    for action in ("retry", "bypass"):
        await joined.test_resumed_preparation_thinking_refusal_preserves_exact_owner(
            resumed_root,
            action,
        )

    observations["functional_controls"] = {
        "replay": replay_results,
        "unsupported_displayable_refused_pre_provider": True,
        "unsupported_proprietary_refused_pre_provider": True,
        "plain_same_backend_model_dispatches_on_v0": True,
        "resumed_retry_owner_recoverable": True,
        "resumed_bypass_owner_recoverable": True,
    }


async def main() -> None:
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    before = _fingerprint(config_path)
    observations: dict[str, object] = {
        "profile": {
            "root": str(PROFILE_ROOT.relative_to(EVIDENCE_ROOT)),
            "config": str(config_path.relative_to(EVIDENCE_ROOT)),
            "config_sha256_before": before,
            "isolated_environment": True,
        }
    }

    await _capture_live_lifecycle(observations)
    await _capture_proprietary_and_no_evidence(observations)
    await _capture_policy_states(observations)
    await _run_functional_controls(observations)

    after = _fingerprint(config_path)
    observations["profile"]["config_sha256_after"] = after  # type: ignore[index]
    observations["profile"]["config_unchanged"] = before == after  # type: ignore[index]
    if before != after:
        raise AssertionError("isolated profile config changed during verification")

    output = EVIDENCE_ROOT / "observations.json"
    output.write_text(
        json.dumps(observations, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "observations": str(output)}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
