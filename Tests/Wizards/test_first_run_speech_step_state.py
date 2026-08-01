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
