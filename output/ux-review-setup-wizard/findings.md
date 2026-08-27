# UX/UAT Review — First-Run Setup Wizard

**Method**: Live UAT against the real TUI (tmux, isolated HOME + `TLDW_CONFIG_PATH`), driven as four personas:

- **P1 — First-time, non-technical**: accepts defaults, uses obvious keys (Enter, arrows, Tab), reads everything, easily scared by jargon/errors.
- **P2 — First-time, technical**: reads fast, wants to know *what* will be written where, tries keyboard shortcuts, may pick Full setup.
- **P3 — Power user, non-technical** (returning user on a new machine): wants Quick path done in <2 min, has a key in a password manager.
- **P4 — Power user, technical**: local models (Ollama/llama.cpp), env vars already set, wants Full track, RAG/tools, expects to script/skip.

**Environment**: 140x40 terminal (also spot-checked at other sizes), commit `3ed7d9420` + working tree, dark theme default.
Captures in `captures/` (numbered, `.ansi` = raw).

Severity: **S1** blocker / **S2** major friction / **S3** polish / **IDEA** improvement opportunity.

---

## Findings

### W — Welcome step

- **W-1 (S3, visual)**: An empty full-width maroon/red-tinted strip renders below the track radio box (row 23 in `01-welcome-quick-140x40.ansi`, bg `rgb(53,34,39)`). It's a reserved error/message area whose error background paints even when there is no message. To P1 this reads as "something is already wrong". Should be invisible (transparent bg / `display:none`) until it has content.
- **W-2 (IDEA)**: Track options carry no effort estimate. "Quick setup (~2 min)" / "Full setup (~10 min, 11 steps)" is a standard wizard affordance that helps P1 pick confidently and lets P3 confirm the fast path.
- **W-3 (IDEA)**: No one-line value proposition for the app itself. P1 arriving from an install instruction sees "Welcome to tldw chatbook" but never learns what the app is. One sentence ("Chat with LLMs, take notes, ingest media — all in your terminal") would anchor the rest of the wizard's jargon.
- **W-4 (S3, copy)**: "Quick setup — provider, model, voice & summary" enumerates internal step names as the *description*. "provider" and "voice" are jargon for P1; better to describe outcomes: "Connect an AI service and start chatting".

### N — Navigation & keyboard model (cross-step)

- **N-1 (S2, keyboard)**: **Enter does not advance the wizard.** On Welcome, with the recommended track selected and the radio focused, Enter does nothing (capture 02). The universal "form + Enter = continue" reflex of P1/P3 is unmet, and there is zero feedback. Ctrl+N is only discoverable from the tiny status-bar hint. Enter on a step's non-input widget should trigger Next (inputs excepted where Enter means submit-value).
- **N-2 (S2, keyboard)**: **The abandon action is the first Tab stop after step content.** From the Welcome radio, Tab lands on "Skip setup" (capture 03→04); on Protect, Tab from the content lands on "Exit setup" (capture 15). A user doing web-form "Tab, Enter" repeatedly triggers the exit flow. Reorder footer focus (Next first, or step-content → Next → Back → Exit) and/or give each step initial focus on its primary control.
- **N-3 (S3, consistency)**: Footer button is "Skip setup" on step 1 but "Exit setup" on steps 2+. If the semantic difference (never-show-again vs. resume-later) is intentional, it's invisible to users; the dialogs' copy does differ, but the button label change reads as drift, not meaning.
- **N-4 (S3, IA/copy)**: Both skip/exit dialogs point to "Settings ▸ Diagnostics" to rerun/resume setup. "Diagnostics" is where users look when something is broken, not for onboarding. Consider a more findable home ("Settings ▸ General ▸ Run setup again") or an alias in both places.
- **N-5 (praise, keep)**: Both guard dialogs default focus to "Keep going" (safe action), style the destructive choice as danger, and the Exit dialog states exactly what is already saved. This is the right pattern — keep it.
- **N-6 (S2, progress)**: **Step total changes mid-flight** — "Step 2 of 5" becomes "Step 3 of 6" after entering an API key (Protect step joins the track; captures 05→10). Moving goalposts undermine the progress indicator's core promise. Either always show Protect (marked "skipped — no keys" when keyless) or annotate the insertion ("+1 step: protect your new key").
- **N-7 (S2, trust)**: **Tracker checkmarks conflate "visited" with "succeeded".** Provider and Model show ✓ even when authentication demonstrably failed (capture 11). Users read ✓ as "this part is OK". A failed/attention state (e.g. `!` in amber) exists in classic wizard vocabularies and this tracker already has state styling.
- **N-8 (S2, keyboard)**: **Radio highlight ≠ selection.** On Welcome, pressing Down moves the RadioSet highlight to "Full setup" but ● stays on Quick; pressing Next then silently proceeds on the *Quick* track (captures 35–37 — this is exactly how my first Full-track attempt failed). Textual's default semantics, but for a 2-option track chooser, selection-should-follow-highlight (the WAI-ARIA radio-group pattern), or Next should honor the highlighted option.
- **N-9 (S2, keyboard)**: **After Ctrl+B (Back), focus does not return to the step's primary control.** Arrow keys and Space then do nothing, with no visible focus indicator to explain why (captures 31–34). Each `show_step` should restore focus via the step's `preferred_focus()`.

### P — Provider step

- **P-1 (S2, layout)**: At 140x40 the step immediately overflows: the step title "Connect a provider" + explanatory subtitle scroll out of view, the provider list is clipped mid-frame, and **two nested scroll regions** (list scrollbar + step-body scrollbar) coexist (captures 06/07). The one sentence that explains what to do disappears first. The detail panel has ~3 rows of dead vertical space (collapsible padding) that could pay for the title.
- **P-2 (S3, IA — praise + gap)**: Grouping (Popular/Cloud/Local/Other, Popular = OpenAI/Anthropic/Ollama/llama.cpp) is good IA. But only ~5 rows are visible in a 7-row viewport with no count ("12 providers") or type-ahead filter; P4 hunting for OpenRouter/Groq must arrow blindly through a clipped list.
- **P-3 (S2, guidance gap)**: No "where do I get a key?" affordance. For P1, the OpenAI panel says only "API key required. Set OPENAI_API_KEY or add api_key under [api_settings.openai]." A first-time user without a key has no pointer (e.g. "Get one at platform.openai.com → API keys"). The env-var/TOML phrasing is written for P4 but is the *primary* helper text shown to everyone; invert it: "Paste your key above — or set OPENAI_API_KEY and we'll pick it up."
- **P-4 (S3, copy)**: After typing a key, the helper becomes "A replacement API key is ready for this provider." (capture 09). On a fresh install nothing is being replaced; "ready" is also not a state the user asked about. Say what happens next: "Key staged — it will be checked when you continue."
- **P-5 (S2, error flow)**: With a bad key, **Next leaves the Provider step silently**; the failure surfaces one step later (see M-1). The step that owns the fix never shows the error.
- **P-6 (S2, silent failure)**: For local providers, **Detect and Test give zero feedback when the server isn't running** — no "scanning…", no "nothing found at localhost:11434", no error; the screen is byte-identical before and after (captures 39–42, Ollama, nothing listening). The user cannot tell the buttons work at all. Every probe needs an in-progress state and a result state, success or failure.
- **P-7 (S3, copy vs behavior)**: The subtitle promises "Local servers just need to be running — **we'll look for them**", but nothing looks automatically; detection is a manual button. Either auto-probe the selected local provider on selection (and say "Looking for Ollama…"), or drop the promise.
- **P-8 (S3)**: "Detect" vs "Test" — two adjacent verbs with no explanation of the difference (scan candidate ports vs probe the URL as given?). Label by outcome: "Find my server" / "Test this address".

### M — Model step

- **M-1 (S2, error UX)**: Auth failure is reported inside the Model step's option list as a pseudo radio row: "○ Connection failed (authentication). Retry or enter a model ID below." (capture 10). (a) It's rendered as a selectable option; (b) "Retry" cannot succeed — the key is wrong and the fix lives one step back, which the message never says ("← Back to fix your API key" is the honest affordance); (c) an error strip styled as an error would be scannable — this looks like a list item.
- **M-2 (S1, silent failure path)**: With failed auth and the **pre-filled fallback model**, pressing Next simply advances — Model gets a ✓ and the wizard proceeds to completion with a provider+model pair that has never worked (captures 10→11). P1 will finish setup "green" and hit their first error later, in the chat window, with no link back to setup. The quick path's whole job is a working first chat; this is the one place the wizard should push back ("This key failed authentication — continue anyway?").
- **M-3 (S3)**: The fallback input arrives pre-filled with a model ID; combined with M-2 it invites Next-through. Pre-fill is fine only once the connection verified. (For Ollama the input is properly empty with a `model-id` placeholder — the pre-fill is provider-specific.)
- **M-4 (S3, copy)**: Failure copy is category-generic where it could be provider-aware: "Connection failed (connection error)" for unreachable Ollama (capture 43) is both redundant and unactionable — "Ollama isn't running. Start it (`ollama serve`), then Retry." tells every persona exactly what to do. Note the Retry affordance is *correct* here (unlike the auth case, M-1) — the two failure kinds deserve different guidance.

### V — Voice step

- **V-1 (S2, fold)**: At 40 rows, the step shows raw plumbing (Service/Endpoint/Auth/Model) and hides its human parts — sample text, **"Test and Hear"**, status, "Use as default" — below the fold with only a thin scrollbar hint (captures 11→13). The primary action of the step is invisible.
- **V-2 (S2, persona fit)**: For P1 the screen opens with an endpoint URL (`http://127.0.0.1:8765/v1/audio/speech`) and model IDs, no one-line explanation of what voice is *for*, whether PocketTTS is bundled or needs installing, and no visible "skip this" (the Welcome promise "every step can be skipped" has no on-screen affordance — the mapping "Next = skip" is never stated). Suggested shape: lead with "Hear replies aloud (optional). PocketTTS runs locally — no account needed." + [Test voice] and tuck Endpoint/Model/Format/Speed into an Advanced collapsible.
- **V-3 (S3)**: Why is Voice on the *quick* track at all? Quick's promise is fastest-to-first-chat; Voice is the most config-heavy screen on that track. Consider demoting it to the full track (data: how many quick-track users configure voice vs. Next through it?).
- **V-4 (S3, copy)**: "Needs test. You can save this configuration while offline." is terse-robotic and buries the reassurance. "Not tested yet — that's fine, you can save now and test later."

### K — Protect-keys step & password dialog

- **K-1 (S2, safety gap)**: Neither the step nor the dialog warns what happens if the master password is forgotten (are keys recoverable? is the config lost?). This is the single riskiest commitment in the wizard and the consequence is unstated (captures 14/16).
- **K-2 (S2, keyboard)**: **Escape does not close the password dialog** (capture 18→19) — no visible way to back out by keyboard; Enter inside the fields doesn't submit either. Cancel/Submit still work but see K-3.
- **K-3 (S2, layering bug)**: The "Password must be at least 8 characters" error panel renders **on top of the dialog's Cancel/Submit buttons and never auto-dismisses** (captures 17–20). After one failed submit, a keyboard user must Tab onto invisible buttons; a mouse user has nothing to click. The buttons stayed functional while hidden (capture 21) — pure z-order/lifetime defect.
- **K-4 (S3)**: The 8-character minimum is disclosed only after a failed submit; the live strength meter appears only once typing starts. State requirements up front in the dialog body.
- **K-5 (S3, copy)**: "Setup Master Password" — "Setup" used as a verb (should be "Set up", as the wizard itself does elsewhere) and Title Case where every other heading is sentence case.
- **K-6 (S3)**: No show-password toggle in either field.
- **K-7 (S3, layout)**: The Protect step body is one small button and ~15 blank rows; the step underuses the space it fought the other steps for.

### T — Toasts / feedback (cross-step)

- **T-1 (S3)**: A global "Settings saved successfully!" toast fires mid-wizard (after Voice commit) and overlaps the wizard footer/Next button (capture 14). Inside a wizard, per-step commits should be silent or in-context; a generic app toast raises "wait, saved *what*?" and obscures navigation at the exact moment of use.

### F — Full track (P2/P4 pass)

- **F-1 (S1, soft-lock — release blocker)**: **The Full track dead-ends at the Speech step on a cold first walk.** On two independent fresh profiles (one with Ollama selected, one with no provider at all), reaching "Speech transcription (optional)" (step 6 of 10) for the first time via Next — with the `onnx-asr` extra absent, the state of every default install — kills ALL keyboard input: Ctrl+N, Ctrl+B, Escape, Tab, and even the app-global Ctrl+P command palette do nothing (captures 45–63; Tab produces a byte-identical frame). Render loop alive (resize reflows), app idle, no log entries, no on-screen error, Next still *looks* enabled. The user must kill the terminal. **Repro precision**: 2/2 on fresh-profile first entry via Next; 0/2 when the same step is entered warm — via the crash-recovery "Resume" (which lands directly on Speech and works, captures 64–69) or via Back-then-Next afterwards (captures 70–72). So it's a cold-first-entry race (likely something the step mounts/kicks off on first entry swallowing focus/events), and the hidden workaround is: kill the app, relaunch, choose Resume. Needs a bug task + a regression test walking the full track cold with optional deps absent.
- **F-2 (S2, layout)**: On the full track the progress tracker **drops every step title** — 10 anonymous numbered boxes (capture 38). The quick track shows names; the full track, where orientation matters most, shows none. The code's own intent (task-2154.9 FR-02: "name the steps the tracker will show") is defeated at exactly the width it matters. Titles could sit under the boxes at ~12 cols/step.
- **F-3 (S3, render bug)**: Step 10's box renders its number as "1" — double digits truncated to the first digit (capture 38). "1" where "10" belongs actively misleads.
- **F-4 (S3)**: Provider step advances with *nothing selected* and no acknowledgement ("skipping provider — you can connect one later"). Consistent with skippability, but silent; a one-line confirmation would prevent "wait, did it take my choice?" doubt.

### R — RAG / Speech steps (full track)

- **R-1 (praise + S3)**: RAG's missing-deps state is honest and calm ("Skipping for now is fine"), but says "install … with your package manager" instead of showing the actual command (`pip install "tldw_chatbook[embeddings_rag]"`). P2/P4 want to copy-paste; P1 needs the exact string even more. Also "RAG" is never glossed — one clause ("lets the assistant search your own documents") would cover P1.
- **R-2 (S2, IA)**: The Speech step is the wizard's jargon peak: "Parakeet v2 (English, INT8)", "onnx-asr", "transcribe.cpp GGUF" — with three parallel affordances ("Use model from disk…", "Review and install…", "Use an existing transcribe.cpp GGUF…") and no visual hierarchy between the recommended path and the escape hatches. The status line "No existing local transcribe.cpp GGUF configured." floats *between* buttons (captures 45/51).
- **R-3 (S3, copy)**: "Skip and set this up later from Lab ▸ Models" appears twice in the first two paragraphs of the same screen.
- **R-4 (S3)**: A ~25-language radio list (its own scroll region) plus a Precision radio (INT8/F32 — jargon) make the *optional* step one of the longest in the wizard. A select/dropdown for language and "Precision" under Advanced would halve it.

### G — Tools / Notes / Appearance steps (full track, reached warm)

- **G-1 (praise, keep)**: The Tools step is the wizard's best persona balance: switches default OFF, plain-language names ("Find files", not `glob_files`), one-line descriptions, ⚠ on data-mutating tools, and the reassurance that risky tools still raise per-call approval cards (capture 73). This is the template the Speech step should follow.
- **G-2 (S3)**: Each tool switch consumes a 3-row box + spacer — 8 tools ≈ 32 rows, forcing scroll for a screen that is conceptually a checklist. Compact single-row switches would show all tools at once.
- **G-3 (S3, semantic misuse)**: The Notes step's reassurance line "Nothing is activated during first-run setup." is composed into the `.setup-step-error` slot and renders in **bold red-on-maroon error styling** (capture 74 ANSI) — calm information dressed as a failure. It also means any real commit error on this step would *replace* the explanation. Give it a neutral class.
- **G-4 (S3)**: The Notes step is a screen that exists to say "you can't do this here" (pointer to Library → Notes → Add from files…). Fine as orientation, but consider folding it into the Summary line it already duplicates, or adding a real action ("open Library after setup" checkbox).
- **G-5 (praise + S3)**: Appearance: curated theme shortlist with "(current)" marker and "Show all themes…" expansion is exactly right. Gaps: splash-card names render as raw snake_case ids (`ascii_aquarium`, `ant_colony`), the card list truncates at 10 with no "show all cards" (themes got one), and cards — unlike themes — have no preview, so choosing one is guesswork (capture 75).
- **G-6 (praise, keep)**: The crash-recovery dialog ("Continue setup? … Resume / Start over / Later", with the honest "Credentials are not retained in setup recovery" caveat, capture 65) is clear, correctly scoped, and Resume lands on the right step with prior steps' ✓ intact.
- **G-7 (S3, first impressions)**: Before the TUI mounts, the terminal shows a wall of raw log output — DEBUG lines including the string "CRITICAL DEBUG:", WARNINGs about missing optional modules (capture 64) — for a second or more on every cold start. A first-time user's literal first contact with the app is internal debug spew with alarming words. Startup should be log-quiet on stdout/stderr by default.

### S — Summary step

- **S-1 (S1, trust — the review's headline finding)**: The summary renders **"✓ Provider" and "✓ Default model — gpt-5.6-terra" for a key that failed authentication minutes earlier** (capture 22). Root cause confirmed in code: `build_first_run_summary_actions(provider_configured=…)` in `first_run_setup_state.py:525` keys off *saved*, not *working* — the model-discovery step's auth failure is never threaded into the summary, so the `review_provider` primary action that exists for exactly this situation is unreachable when it's most needed. Fix: propagate the discovery outcome; render "! Provider — key failed an authentication check" and make **Review provider** the primary button.
- **S-2 (praise, keep)**: The rest of the summary is excellent: per-item states with *reasons* ("RAG — optional — embeddings deps not installed"), explicit "✗ Key encryption — API keys are stored as plain text", the config file path, and a re-run pointer. This is the transparency the rest of the flow should match.
- **S-3 (S3)**: "✓ Provider" doesn't name the provider. "✓ Provider — OpenAI" costs nothing and confirms the one fact the user actually chose.
- **S-4 (S3, visual)**: The config path hard-wraps mid-character across two lines (capture 22). Middle-truncate (`…/uat-profile/config.toml`) or let it wrap at path separators.
- **S-5 (S3)**: Exit buttons "Start chatting / Explore Home / Review settings" are outcome-oriented — good — but inconsistently cased ("Explore Home" vs sentence case elsewhere).
- **S-6 (praise, keep)**: The *unsaved*-provider case is handled beautifully: "✗ Provider — no credentials or saved endpoint", primary action flips to "Review provider setup", and that button really returns to the Provider step (captures 76–77). Even the Key-encryption glyph is state-sensitive: "✗ … stored as plain text" when keys exist, neutral "– off" when there are none. This is precisely the machinery S-1 asks to extend to the saved-but-failing case.

### H — Post-wizard handoff (Console-owned, but it's the wizard's landing)

- **H-1 (S2, interruption stack)**: "Start chatting" lands in Console and immediately opens a **"Check model lists online?"** consent modal (capture 23) — a fourth decision before the promised first chat, and logically odd since the wizard itself already contacted the provider (discovery + auth probe) with the user's key. Fold this consent into the wizard (checkbox on Provider or Summary), so setup ends interruption-free.
- **H-2 (S2, jargon dialog)**: The user's *first message* triggered "Project instructions need a folder … Stale folders cannot be selected. / No eligible folders / `no_eligible_binding` / [Disable] [Cancel]" (capture 24). Raw internal error code on screen; P1 has no idea what project instructions are or why "Hello" needs a folder. On a fresh profile this feature should not intercept the first send at all.
- **H-3 (S2, dead end — observed with the bad key)**: After cancelling that dialog, the send sat ≥30s at "Run: Validating provider." / composer "Send blocked — **finish provider setup to continue**" / an empty Assistant block, with no error, no retry/cancel affordance, and no link to the setup it demands (captures 25–27). Note the copy directly contradicts the wizard's just-shown "✓ Provider". (Caveat: bad key + sandboxed network in this UAT; the hang may resolve differently live — but the blocking copy and the missing affordance are real regardless.)

### E — First-run entry conditions

- **E-1 (S3, discoverability)**: With `OPENAI_API_KEY` exported (the standard dev pattern — P2's most likely state), a fresh install boots **straight to Console with no first-run acknowledgement at all** (capture 80): no "using your OPENAI_API_KEY" notice, no mention that a setup wizard exists (voice, tools, theme, key encryption all undiscovered). Correct not to nag — but a one-time dismissible line ("Found OPENAI_API_KEY — you're ready. Run setup any time: Settings ▸ Diagnostics") would serve P2 at zero cost. There's also an irony: the wizard's own helper text *recommends* env vars, and following that advice guarantees you never see the wizard on the next machine.

### Z — Terminal-size resilience

- **Z-1 (S2)**: At **80x24 — the stock macOS Terminal.app size a first-timer is most likely to use** — the Welcome step opens with its own title and subtitle scrolled out of view (the user sees two truncated radio rows with zero context, "(recomm…"), and the Provider step shows the heading plus exactly *one* provider row, with the API-key panel entirely below the fold (captures 81–82). Everything technically works by scrolling, but nothing says so.
- **Z-2 (IDEA)**: The app knows its minimum comfortable size; when the terminal is below it, say so ("tldw chatbook works best at 100×30 or larger — enlarge the window if you can") the way many TUIs do, at least during first-run.
- **Z-3 (S3)**: Even at 140 cols, steps scroll while showing 3–7 rows of decorative dead space (collapsible padding, triple blank rows in the Authentication panel, 15 blank rows on Protect). A vertical-density pass on the wizard stylesheet would fix P-1/V-1/Z-1 together.


---

## Persona verdicts

- **P1 — first-time, non-technical (Quick track)**: Can finish the wizard in ~2 minutes and the copy mostly holds their hand — but the flow's *happy path lies to them* when their key is wrong (S-1/M-2), Enter doesn't do what they expect (N-1), Tab+Enter threatens to exit (N-2), the Voice step reads like a router config page (V-2), and their reward for finishing is three consecutive interruptions before the first reply (H-1/H-2/H-3). **Verdict: completes setup, likely fails at first chat with no path back.**
- **P2 — first-time, technical**: Env-var users never meet the wizard (E-1). Wizard users who choose Full **hit the Speech soft-lock and have to kill the app** (F-1); if they stumble into Resume, the rest of the track is actually good (Tools step is exemplary). Helper text speaks their language (env vars, TOML sections, exact pip extras — R-1 aside).
- **P3 — returning, non-technical**: Quick track is genuinely quick; recovery/resume is excellent (G-6). Their risks are the keyboard traps (N-1/N-2/N-8/N-9) and the stock 80x24 terminal (Z-1).
- **P4 — power user, technical (local models)**: Local-provider flow undermines its own promise — no auto-detection despite "we'll look for them", and Detect/Test fail silently (P-6/P-7). Model-step errors aren't provider-aware (M-4). Full track blocked by F-1.

## Top 10 by priority

1. **F-1** Full track soft-locks at Speech on cold first entry — app must be killed (S1, bug).
2. **S-1 + M-2 + N-7** One system: thread probe outcomes into step state, tracker glyphs, and Summary so a failed key is never shown as ✓ (S1, trust).
3. **H-1/H-2/H-3** The post-wizard landing must not stack interruptions or dead-end the first send; "Send blocked — finish provider setup" needs a button that opens provider setup (S2).
4. **N-1/N-2** Enter advances; abandon action last in focus order; per-step initial focus (S2, keyboard).
5. **K-2/K-3** Password dialog: Escape cancels; error toast must not cover Cancel/Submit indefinitely (S2).
6. **P-6/P-7** Local-provider probes: auto-run on selection, visible progress + result (S2).
7. **N-8/N-9** Track radio: selection follows highlight; Back restores focus (S2).
8. **P-1/V-1/Z-1/Z-3** Vertical-density pass + keep step titles pinned; 80x24 "enlarge terminal" hint (S2, layout).
9. **F-2/F-3** Full-track tracker: keep titles, fix the two-digit box (S2/S3).
10. **P-3/M-1/M-4** Error and helper copy: where-to-get-a-key link, "← Back to fix your key" on auth failure, provider-aware connection errors (S2/S3).

## Coverage notes

- Live-driven: Welcome, Provider (cloud + local + none), Model (auth-fail, connection-fail), Voice, RAG (deps absent), Speech (deps absent), Tools, Notes, Appearance, Summary (both success and review-provider variants), Protect + password dialog, skip/exit dialogs, crash-recovery Resume, post-wizard Console handoff, env-var entry condition, 140x40 + 140x50/55 + 80x24 sizes.
- Not covered live: a *valid* API key end-to-end (no real credential was spent), Protect-step encryption actually applied (cancelled at dialog), voice "Test and Hear" audio, RAG/Speech with optional deps installed, "Start over" recovery path, mouse-only interaction.
- Environment caveat: run under a sandboxed harness; network to api.openai.com worked (real 401s), localhost probes legitimately refused. The F-1 soft-lock and H-3 hang deserve one confirmation on a normal machine.

---

## Remediation status (2026-08-25, branch `fix/setup-wizard-uat`)

All findings addressed across 11 backlog tasks (TASK-22281…21149), one commit per task, each live-verified against the real TUI and green on the wizard suites (~880 tests):

| Commit | Task | Findings closed |
|---|---|---|
| `fix(wizard): heal keyboard focus orphaned by step recompose` | 21139 | **F-1** (root cause: recompose detaches the focused widget; Textual keeps dispatching into the dead node) |
| `fix(wizard): pin step errors to always-visible chrome` | 21140 | W-1, G-3, hidden `show_step_error` surface, phantom "Skip this step" copy |
| `fix(wizard): master-password dialog UX` | 21141 | K-1…K-6 (K-3 root cause: app-wide `.error-message` rule inflating the dialog error over its buttons) |
| `feat(wizard): keyboard model` | 21142 | N-1, N-2, N-8; N-9 proved to be F-1 collateral, already cured |
| `feat(wizard): probe outcomes drive tracker, model gate, and summary` | 21143 | **S-1**, M-1, M-2, M-4, N-7, P-5 (live-verified against a real 401) |
| `fix(wizard): local-provider probe feedback visible and adjacent` | 21144 | P-6, P-7 (observed silence = F-1 collateral + below-fold status), P-8 |
| `fix(console): first-chat handoff` | 21145 | H-2 (no folder modal on fresh-profile first send), H-3 (30s validation bound + clickable "Send blocked" reason via `app.run_setup_wizard`) |
| `feat(wizard): model-list consent lives on the Summary step` | 21146 | H-1 (+ SetupCheckbox structural glyphs — the color-only checked state UAT misread) |
| `feat(startup): quiet cold-start terminal + one-time env-key notice` | 21147 | G-7 (0 noise lines; `TLDW_VERBOSE_STARTUP=1` restores), E-1 |
| `feat(wizard): layout and density pass` | 21148 | P-1, V-1, V-2, Z-1, Z-2, Z-3, F-2, F-3, G-2*, N-6, S-4 (*switch rows kept; density recovered elsewhere) |
| `feat(wizard): copy pass` | 21149 | W-2, W-3, P-3, P-4, V-4, R-1, R-3, S-3, G-5; N-3/N-4/S-5/W-4 dispositions documented |

Deferred by design: V-3 (Voice stays on the quick track, now one-line-purpose-first — usage data should decide), N-4's Settings-IA move, the pre-existing `dev` failures in Tests/Chat (~68) and the order-dependent `test_rerun_over_settings_review_settings_returns_to_settings` flake — all noted in task records.
