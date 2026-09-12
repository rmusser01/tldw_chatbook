---
target: Speech & TTS settings sub-screen (settings speech tts)
total_score: 22
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-09-11T23-45-55Z
slug: tldw-chatbook-ui-screens-settings-speech-tts-py
---
# Critique — Speech & TTS settings sub-screen (settings_speech_tts.py + speech_tts_settings_panel.py)

⚠️ DEGRADED: partial — Assessment A (design review) ran as an isolated sub-agent; Assessment B's sub-agent hit a usage limit mid-run, so the deterministic detector and the live tmux walkthrough were completed inline in the parent context.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Superb instrumentation, illegible vocabulary: "Observed … configuration revision 3 | catalog revision none" (panel ~2796-2818) — revision arithmetic is not user status |
| 2 | Match System / Real World | 1 | "Model policy: Exact / First available", "Voice policy: Exact / Server default" (panel 2473-2541) name the data model; "Restore Non-secret Defaults" leaks threat-model vocabulary into a button |
| 3 | User Control and Freedom | 4 | Best-in-class 3-way leave guard (Cancel/Discard/Save) on category switch, screen exit, provider switch; draft survives recomposition |
| 4 | Consistency and Standards | 2 | Adjacent Selects "Default TTS Provider" vs "Configure Provider" share identical 7 options, different semantics; Model/Voice value is a Select for audio.cpp but free-text for six providers |
| 5 | Error Prevention | 3 | Validation focuses offending field + expands its Collapsible; plaintext-HTTP consent explicit; free-text voice IDs unchecked for existence |
| 6 | Recognition Rather Than Recall | 1 | Voice/model IDs for 6/7 providers typed from memory (live: "Voice value | alloy" free text); no voice picker in Settings; Kokoro file locations docs-only |
| 7 | Flexibility and Efficiency | 2 | s/r accelerators + search deep-link ("speech"→1 match→Enter, verified live) good; but every verification is a forced Settings→Save→Speech Lab round-trip, and the handoff button disables itself when dirty |
| 8 | Aesthetic and Minimalist Design | 1 | Live at 235x52: banner+status rows precede first control; live at 100x30: governance banner fills the ENTIRE first viewport, zero editable controls visible; 7-line provenance paragraphs per package |
| 9 | Error Recovery | 3 | Failure copy names the next action; diagnosis still requires saved-vs-applied revision reasoning |
| 10 | Help and Documentation | 2 | Inline copy dominated by scope-police; nothing answers "where do Kokoro model files come from?"; F1 help + docs exist (live footer honest) |
| **Total** | | **22/40** | **Acceptable — significant improvements needed** |

## Design Specificity Verdict

Hyper-specific to this product — saturated with Chatbook-only constructs (Speech Lab, Studio preferences, Guided setup, Fresh/Stale/Unverified/Missing choice states, credential provenance rungs, Owner IDs). The specificity is almost entirely about internal architecture (ownership, revisions, provenance) rather than the user's task ("make my chat read aloud"). It reads as an ADR-039 ownership contract rendered as a form: honest to the architect, bureaucratic to the operator.

Deterministic scan: detector inapplicable to Python TUI sources (no markup) — clean by vacuity, exit 0 both target and directory.

Live evidence: scope banner, "Model policy/Voice policy" rows, and the raw id leak ("audio.cpp, audio_cpp" in the Scope Inspector AND the bottom status line, confirmed at both 235x52 and 100x30) all render exactly as the source shows.

## What's Working

1. Provenance without secrets — credentials never displayed; env-var shadowing disclosed; the "inspectable, legible authority" principle executed properly.
2. The leave-guard is airtight — every ownership boundary funnels through one confirm_leave() with a 3-way modal; recoverability is real.
3. Honest responsive behavior — at 100x30 text wraps, inspector truncates with an explicit "▼ more — scroll" affordance, footer hints advertise only implemented actions (s/r/F1/F6/Ctrl+P/Ctrl+Q).

## Priority Issues

1. [P1] Local-provider first-run dead end (Kokoro) — form assumes artifacts that don't exist; the dependency fact ("Local Kokoro: Unavailable — tldw_chatbook[local_tts]") is computed but hidden in the collapsed Scope inspector. Fix: surface dependency rows inside the Kokoro/Chatterbox/Higgs forms with install extra + model-download pointer; add path-existence check on Browse.
2. [P1] Free-text "Exact model ID"/"Exact voice ID" for six providers — recall over recognition; wrong IDs surface only at speak time. Fix: render known IDs (LEGACY_DEFAULT_MODELS/VOICES already imported) as Select options with "custom…" escape hatch + "browse voices in Speech Lab" bridge.
3. [P1] Governance copy crowds out task copy — live-confirmed: at 100x30 the banner consumes the whole first viewport; three "Open Speech Lab" buttons on one screen. Fix: one-line banner + tooltip, one Lab action per screen, move duplicated status Statics into the inspector.
4. [P2] State banner contradicts leave guard for this category — generic banner says "switching categories keeps this draft" but Speech & TTS resolves the draft on leave (task-2708 already tracks this).
5. [P2] Flat 15-row provider forms (Chatterbox/Higgs) + one-shot "Restore Non-secret Defaults" with no preview — apply the audio.cpp sub-section/collapsible pattern; preview what Restore resets.

## Persona Red Flags

Jordan (first-timer): "Model policy"/"Voice policy" schema vocabulary; Kokoro "Path to model file" with zero download guidance; first Configure-Provider change can trigger the unsaved-changes modal because Default-Provider auto-fill dirtied the draft (verified in source: handle_default_provider_changed auto-fills → has_unsaved_changes → leave modal); Save payoff line is "Runtime reconfiguration: Kokoro: unchanged" — system talk at the moment they wanted "you're done".

Alex (power user): cannot test from Settings at all by design, and the one fast path (audio.cpp handoff) disables itself when dirty ("Save Settings before opening Speech Lab"); voice IDs discovered in Speech Lab must be retyped into Settings free-text; inspector speaks in pending generations ("Saved generation 3 is pending; generation 2 remains active…") requiring revision math; ElevenLabs output format is a flat 9-option codec list.

## Minor Observations

- Label casing inconsistent in one card: "Default TTS Provider" vs "Model policy" vs "ONNX model file".
- Raw provider id leaks: "audio.cpp, audio_cpp" (inspector + status line, live-confirmed), "Owner ID: app_tts.kokoro", "audio_cpp parameters".
- Three "Open Speech Lab" buttons per screen; two simultaneously visible for audio.cpp.
- Terminology drift: "Lab > Speech > Voice Profiles" vs "Speech Lab" vs "Studio preferences" — three names, two places.
- Speed range only in placeholder ("0.25 - 4.0"); audio.cpp users see it permanently disabled with the reason in a separate line rather than on the field.
- Realtime (microphone INPUT engine) card lives inside output-speech settings between Provider setup and Inspector.
- VAD disabled-not-hidden with rationale copy is genuinely good — copy that pattern elsewhere.
- TTS_MODULE_GUIDE.md still says "S/TT/S tab" — doc drift vs "Speech Lab".

## Questions to Consider

1. If Settings is contractually forbidden from testing, why does first-time provider setup live here rather than in Speech Lab with Settings as link-out?
2. Who is "Model policy: Exact / First available" for? Is the policy concept earning its two rows, or is the persistence schema showing through?
3. The panel spends ~10x the vertical space explaining what it is not allowed to do than helping produce sound — is the honesty budget spent on the architect's anxieties instead of the user's question?
4. Seven providers in registry order — should the picker lead with the cloud-vs-local split every user actually decides first?
