"""Unit tests for the pure first-run wizard Speech transcription state module.

TASK-1301: language/precision enumeration must come from the canonical STT
policy/catalog (tldw_chatbook.STT.routing), never a hand-rolled list, and
"selectable" must be gated by what a curated model-artifact descriptor can
actually download today -- see SpeechSetupStep in FirstRunSetupWizard.py for
the live wiring against tldw_chatbook.Model_Artifacts.curated_registry.
"""

from tldw_chatbook.STT.routing import RoutingPolicy
from tldw_chatbook.UI.Wizards.first_run_speech_step_state import (
    SpeechLanguageOption,
    SpeechPrecisionOption,
    SpeechPrefill,
    TRANSCRIPTION_SECTION,
    build_speech_transcription_commit,
    read_speech_prefill,
    recommended_speech_selection,
    resolve_speech_selection,
    routing_policy,
    should_persist_speech_config,
    speech_language_options,
    speech_precision_options,
    speech_prefill_status,
)


class TestRoutingPolicy:
    def test_shared_policy_is_a_real_routing_policy(self):
        assert isinstance(routing_policy(), RoutingPolicy)

    def test_english_is_owned_by_v2_not_the_v3_validated_set(self):
        """RoutingPolicy.__post_init__ rejects 'en'/'auto' in
        validated_v3_languages -- English must never appear there."""
        assert "en" not in routing_policy().validated_v3_languages


class TestRecommendedSelection:
    def test_recommended_is_parakeet_v2_english_int8(self):
        provider_id, model_id, language = recommended_speech_selection()
        policy = routing_policy()
        assert provider_id == policy.parakeet_provider_id == "parakeet-onnx"
        assert model_id == policy.parakeet_v2_model_id == "nemo-parakeet-tdt-0.6b-v2"
        assert language == "en"


class TestSpeechLanguageOptions:
    def test_english_selectable_when_its_model_is_curated(self):
        policy = routing_policy()
        options = speech_language_options(
            curated_model_ids=frozenset({policy.parakeet_v2_model_id})
        )
        assert all(isinstance(o, SpeechLanguageOption) for o in options)
        english = next(o for o in options if o.code == "en")
        assert english.selectable is True
        assert english.model_id == policy.parakeet_v2_model_id
        assert english.display_name == "English"

    def test_v3_languages_present_but_not_selectable_without_a_curated_v3_descriptor(self):
        policy = routing_policy()
        options = speech_language_options(
            curated_model_ids=frozenset({policy.parakeet_v2_model_id})
        )
        others = [o for o in options if o.code != "en"]
        assert others, "the STT catalog declares more than just English for Parakeet"
        assert all(o.selectable is False for o in others)
        assert all(o.model_id == policy.parakeet_v3_model_id for o in others)

    def test_english_becomes_unselectable_when_nothing_is_curated(self):
        options = speech_language_options(curated_model_ids=frozenset())
        english = next(o for o in options if o.code == "en")
        assert english.selectable is False

    def test_options_are_sourced_from_the_stt_routing_policy_not_hand_rolled(self):
        options = speech_language_options(curated_model_ids=frozenset())
        codes = frozenset(o.code for o in options if o.code != "en")
        assert codes == routing_policy().validated_v3_languages

    def test_no_duplicate_codes(self):
        options = speech_language_options(curated_model_ids=frozenset())
        codes = [o.code for o in options]
        assert len(codes) == len(set(codes))


class TestSpeechPrecisionOptions:
    def test_int8_selectable_when_curated(self):
        options = speech_precision_options(curated_precisions=frozenset({"int8"}))
        assert all(isinstance(o, SpeechPrecisionOption) for o in options)
        int8 = next(o for o in options if o.value == "int8")
        assert int8.selectable is True

    def test_f32_present_but_not_selectable_without_a_curated_f32_descriptor(self):
        options = speech_precision_options(curated_precisions=frozenset({"int8"}))
        f32 = next(o for o in options if o.value == "f32")
        assert f32.selectable is False

    def test_no_curated_precisions_makes_int8_unselectable_too(self):
        options = speech_precision_options(curated_precisions=frozenset())
        int8 = next(o for o in options if o.value == "int8")
        assert int8.selectable is False


class TestBuildSpeechTranscriptionCommit:
    def test_commit_shape_matches_the_transcription_section(self):
        commit = build_speech_transcription_commit(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v2",
            language="en",
        )
        assert commit == {
            TRANSCRIPTION_SECTION: {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }


class TestResolveSpeechSelection:
    """PR #1184 review (finding 2): commit() must persist what the user
    actually PRESSED, not a hardcoded constant. Today only English/INT8 is
    selectable, so this always equals recommended_speech_selection() -- but
    the moment a second curated descriptor makes another combination
    selectable, resolve_speech_selection() is the one place that has to
    notice and follow the pressed radios instead."""

    def test_nothing_selected_falls_back_to_recommended(self):
        """No live radio pressed yet (e.g. the step never mounted, or
        commit() runs before on_show()) -- skip-safe fallback."""
        assert (
            resolve_speech_selection(
                selected_language="",
                selected_precision="",
                curated_model_ids=frozenset(),
                curated_precisions=frozenset(),
            )
            == recommended_speech_selection()
        )

    def test_todays_only_selectable_combo_is_byte_identical_to_recommended(self):
        """The BYTE-IDENTICAL pin: pressing the one combination that is
        selectable today must resolve to exactly what commit() persists
        today -- no behavior change yet, divergence-proofing for later."""
        policy = routing_policy()
        result = resolve_speech_selection(
            selected_language="en",
            selected_precision="int8",
            curated_model_ids=frozenset({policy.parakeet_v2_model_id}),
            curated_precisions=frozenset({"int8"}),
        )
        assert result == recommended_speech_selection()
        assert result == (policy.parakeet_provider_id, policy.parakeet_v2_model_id, "en")

    def test_hypothetical_second_selectable_language_is_honored(self):
        """Divergence-proofing: once a second curated descriptor makes a v3
        language selectable, the commit payload must follow the PRESSED
        radio, not silently keep persisting the v2/English default.

        Mutation check: an implementation that hardcodes
        ``return recommended_speech_selection()`` regardless of the
        selection makes this assertion fail (it would still return the
        English/v2 tuple)."""
        policy = routing_policy()
        v3_language = sorted(policy.validated_v3_languages)[0]
        result = resolve_speech_selection(
            selected_language=v3_language,
            selected_precision="int8",
            curated_model_ids=frozenset(
                {policy.parakeet_v2_model_id, policy.parakeet_v3_model_id}
            ),
            curated_precisions=frozenset({"int8"}),
        )
        assert result == (policy.parakeet_provider_id, policy.parakeet_v3_model_id, v3_language)
        assert result != recommended_speech_selection()

    def test_unselectable_language_falls_back_even_when_curated_elsewhere(self):
        """A pressed language whose model is not (yet) curated must never be
        persisted, even if some other model id happens to be curated."""
        policy = routing_policy()
        v3_language = sorted(policy.validated_v3_languages)[0]
        result = resolve_speech_selection(
            selected_language=v3_language,
            selected_precision="int8",
            curated_model_ids=frozenset({policy.parakeet_v2_model_id}),  # no v3
            curated_precisions=frozenset({"int8"}),
        )
        assert result == recommended_speech_selection()

    def test_unselectable_precision_falls_back_even_when_language_is_selectable(self):
        """Both radios must resolve to a currently-selectable option --
        precision alone can veto a selectable language too (defensive: the
        UI never lets a disabled radio get pressed, but the pure resolver
        does not trust that)."""
        policy = routing_policy()
        result = resolve_speech_selection(
            selected_language="en",
            selected_precision="f32",
            curated_model_ids=frozenset({policy.parakeet_v2_model_id}),
            curated_precisions=frozenset({"int8"}),  # f32 not curated
        )
        assert result == recommended_speech_selection()

    def test_unknown_language_code_falls_back(self):
        """A stale/garbage id (should never happen via the real RadioSet)
        must not raise -- fall back to the recommended selection."""
        policy = routing_policy()
        result = resolve_speech_selection(
            selected_language="zz",
            selected_precision="int8",
            curated_model_ids=frozenset({policy.parakeet_v2_model_id}),
            curated_precisions=frozenset({"int8"}),
        )
        assert result == recommended_speech_selection()


class TestReadSpeechPrefill:
    def test_empty_config_returns_blank_prefill(self):
        assert read_speech_prefill({}) == SpeechPrefill()

    def test_missing_transcription_section_returns_blank_prefill(self):
        assert read_speech_prefill({"chat_defaults": {}}) == SpeechPrefill()

    def test_reads_the_persisted_transcription_section(self):
        cfg = {
            TRANSCRIPTION_SECTION: {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }
        prefill = read_speech_prefill(cfg)
        assert prefill.provider_id == "parakeet-onnx"
        assert prefill.model_id == "nemo-parakeet-tdt-0.6b-v2"
        assert prefill.language == "en"

    def test_shipped_template_defaults_never_read_as_parakeet_onnx(self):
        """The shipped [transcription] template defaults to faster-whisper (or
        a platform MLX provider) with model "distil-large-v3" -- never
        "parakeet-onnx". A fresh, wizard-untouched install must not read back
        as configured by THIS step (see build_summary_rows' speech row,
        which keys "configured" off provider_id == the Parakeet ONNX
        provider id specifically, not off model_id being merely non-empty)."""
        cfg = {
            TRANSCRIPTION_SECTION: {
                "default_provider": "faster-whisper",
                "default_model": "distil-large-v3",
                "default_language": "en",
            }
        }
        prefill = read_speech_prefill(cfg)
        assert prefill.provider_id != routing_policy().parakeet_provider_id


class TestSpeechPrefillStatus:
    """AC#5's "re-run prefills" clause: the step must show what is already
    persisted, not just silently overwrite it (Important 3)."""

    def test_nothing_persisted_is_blank(self):
        assert speech_prefill_status(SpeechPrefill()) == ""

    def test_already_parakeet_onnx_reports_already_the_default(self):
        prefill = SpeechPrefill(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v2",
            language="en",
        )
        text = speech_prefill_status(prefill)
        assert "Already" in text
        assert "nemo-parakeet-tdt-0.6b-v2" in text
        assert "en" in text

    def test_runtime_absent_states_the_config_without_directing_to_a_button(self):
        """Final-review residual of NEW-2: with the onnx-asr runtime absent
        the "use as default" affordance is never composed, so the sentence
        must not send the user to a control that is not on screen -- and
        must not promise that installing here can switch anything.

        Args:
            self: Test instance.

        Returns:
            None.
        """
        prefill = SpeechPrefill(
            provider_id="remote-whisper", model_id="x", language="auto"
        )
        text = speech_prefill_status(
            prefill, installed_active=True, runtime_installed=False
        )
        assert text == "Currently configured: remote-whisper."
        assert "below" not in text and "installing" not in text

    def test_different_provider_reports_current_default_and_consequence(self):
        prefill = SpeechPrefill(
            provider_id="remote-whisper", model_id="whisper-1", language="auto"
        )
        text = speech_prefill_status(prefill)
        assert "remote-whisper" in text
        assert "Parakeet v2" in text

    def test_shipped_template_default_also_reports_as_a_current_default(self):
        """The template's own default (faster-whisper/distil-large-v3) is a
        REAL persisted value from the user's point of view even though it
        was never explicitly chosen -- it must be shown, not hidden, since
        installing/activating here would replace it (Important 3)."""
        prefill = SpeechPrefill(
            provider_id="faster-whisper", model_id="distil-large-v3", language="en"
        )
        text = speech_prefill_status(prefill)
        assert "faster-whisper" in text

    def test_not_installed_yet_mentions_installing_or_activating(self):
        """Default state (nothing installed/active, nothing acted this run):
        the original "installing or activating here" copy is still accurate
        -- both are genuinely still possible paths."""
        prefill = SpeechPrefill(
            provider_id="remote-whisper", model_id="whisper-1", language="auto"
        )
        text = speech_prefill_status(prefill)
        assert "installing or activating" in text

    def test_installed_and_active_does_not_claim_installing_or_activating(self):
        """Review NEW-2: when the artifact is already installed AND active,
        neither "installing" nor "activating" is a real action (Activate is
        disabled) -- the old sentence promised an outcome no control could
        deliver. installed_active=True must drop that false claim."""
        prefill = SpeechPrefill(
            provider_id="remote-whisper", model_id="whisper-1", language="auto"
        )
        text = speech_prefill_status(prefill, installed_active=True)
        assert "installing or activating" not in text
        assert "remote-whisper" in text
        assert "Parakeet v2" in text

    def test_acted_this_run_reports_pending_switch(self):
        """After the user opts in this run (installed, activated, or used
        the new "use as default" affordance), the sentence must describe
        what WILL happen, not repeat a now-stale offer."""
        prefill = SpeechPrefill(
            provider_id="remote-whisper", model_id="whisper-1", language="auto"
        )
        text = speech_prefill_status(
            prefill, installed_active=True, acted_this_run=True
        )
        assert "will become your default" in text.lower()
        assert "remote-whisper" in text

    def test_already_matching_ignores_the_new_flags(self):
        """The "already your default" early return is unaffected by
        installed_active/acted_this_run -- there's nothing to switch."""
        prefill = SpeechPrefill(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v2",
            language="en",
        )
        text = speech_prefill_status(
            prefill, installed_active=True, acted_this_run=True
        )
        assert "Already" in text


class TestShouldPersistSpeechConfig:
    """Important 3's no-clobber gate, isolated as a pure decision."""

    def test_inactive_never_persists_regardless_of_action(self):
        assert should_persist_speech_config(active=False, acted_this_run=True) is False
        assert should_persist_speech_config(active=False, acted_this_run=False) is False

    def test_active_but_no_action_this_run_does_not_persist(self):
        """The byte-identical pin: a re-run that just Nexts through an
        already-active artifact (e.g. installed earlier via Library) must
        not touch existing [transcription] config."""
        assert should_persist_speech_config(active=True, acted_this_run=False) is False

    def test_active_and_acted_this_run_persists(self):
        assert should_persist_speech_config(active=True, acted_this_run=True) is True
