"""``LibraryIngestState`` -- the Ingest subsystem's own fields.

State PR of the Ingest extraction series (wave-5 task 1,
``.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest``; recipe:
``backlog/docs/library-decomposition-recipe.md``; export series --
``library_export_state.py`` -- is the worked example this mirrors most
closely, since Ingest also needed no plural-prefix split and no entangled
reader-preferences trio). Every field here was moved verbatim out of
``LibraryScreen.__init__`` in ``tldw_chatbook/UI/Screens/library_screen.py``
-- same default, same type. ``library_screen.py`` keeps every original
``_library_ingest_<field>`` attribute name alive as a generated getter/
setter ``@property`` shim pointing at ``self._ingest_state.<field>`` (a
sentinel-wrapped block right after the ``LibraryScreen`` class body); a
later controller PR in this series moves the subsystem's methods here too.

Every field uses the SAME ``_library_ingest_`` prefix -- the recipe §2
ownership script (run against the ``_library_ingest`` prefix family) found
20 ``__init__``-scoped fields, all exclusively Ingest-owned (every
non-ingest consumer is a shell/plumbing method -- e.g.
``_sync_library_ingest_rail_for_width``/``_set_library_rail_collapsed`` for
``auto_collapsed_rail``, ``_apply_parakeet_v2_install_result``/``_on_
preflight_retry`` for ``form``, ``on_mount`` for ``last_done_count``,
``_apply_library_external_preparation``/``_library_emergency_return_
eligibility`` for ``start_confirm_armed_at``/``start_consent``), so no
field needed a plural variant and none is BLOCKED by the >=2-subsystems
rule. No field holds a live controller/coordinator instance (unlike the
``_conversation_reader_controller``/``_library_collections_capture_
controller`` "capture-controller" precedent) -- every field here is plain
data, so there is no wiring-field exclusion either.

**The ingest-options trio is explicitly OUT OF SCOPE for this module and
was NOT moved.** ``_INGEST_OPTIONS_CACHE_ATTR``, ``_read_library_ingest_
options_from_config`` and ``_library_ingest_options_for`` (module-level
``FunctionDef``s at ``library_screen.py:605-692``) are permanently
screen-routed -- several tests monkeypatch ``get_cli_setting``/
``_read_library_ingest_options_from_config`` on the ``library_screen``
module object, expecting the patch to reach ``_library_ingest_options_
for``'s internal free-name call, which only resolves correctly while both
functions share that module's own ``__globals__`` (recipe §3's oldest
documented module-globals-coupling hazard). None of the trio reads or
writes any ``self._library_ingest_*`` instance field -- confirmed by
reading both function bodies -- so this state move touches neither the
trio nor anything it depends on.

Two DIFFERENT, adjacent field clusters that share initialization code with
Ingest but are NOT Ingest fields, and therefore NOT part of this dataclass,
were confirmed by reading every consumer rather than assumed by proximity:

- ``_library_external_submit_generation``/``_scope_id``/``_worker``/
  ``_backend``/``_consent``/``_busy``/``_status`` -- a SEPARATE "external
  source" onboarding feature (VAD-consent preparation for an external
  root), interleaved in the same ``__init__`` region. Two ingest-named
  methods (``_do_submit_ingest``, ``_enqueue_library_ingest_snapshot``)
  read several of these, but every field's PRIMARY/majority consumer set
  is external-preparation-owned (``_apply_library_external_preparation``,
  ``_apply_library_external_vad_progress``, ``_confirm_library_external_
  vad``, ``_invalidate_library_external_submission``, ``_set_library_
  external_status``, ``compose_content``) -- these stay screen-resident
  shared shell state, reached by a future controller PR via a named
  dependency, exactly the shape the recipe's ownership script calls
  "non-subsystem users belonging to another subsystem."
- ``_transcribe_cpp_configured`` -- read by two ingest-named methods
  (``_build_library_ingest_state``, ``_load_library_ingest_options_from_
  config``) but ALSO by ``_apply_transcribe_cpp_gguf_result`` (an
  unrelated model-install completion handler); stays screen-resident.

All 20 fields' original ``__init__`` lines are simple, uncomputed
literals or no-argument factory calls (``threading.Lock()``,
``LibraryIngestFormState()``, ``[]``, ``set()``) -- unlike the
conversations/collections/skills exemplars' entangled reader-preferences
trio, or Export's own computed ``form`` default
(``self._default_library_export_form()``), NOTHING here depends on
runtime/config data or another subsystem's shared init call. So
``self._ingest_state = LibraryIngestState()`` is constructed, with no
constructor arguments, at the position of the first removed field
(``auto_collapsed_rail``'s original line), and every one of the 20
original assignment lines is deleted outright -- there is no
"entangled/computed placeholder" complication at all, matching the export
series' own simplest fields (``running``, ``error``, ...) but, unusually,
for the ENTIRE cluster rather than a fraction of it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from textual.timer import Timer
from textual.worker import Worker

from ...Library.library_ingest_jobs import LibraryIngestJob
from ...Library.library_ingest_state import (
    LibraryIngestFormState,
    LibraryIngestLastSubmission,
)
from .screen_support_types import _LibraryIngestStartConsent


@dataclass
class LibraryIngestState:
    """Every field the Ingest subsystem exclusively owns."""

    auto_collapsed_rail: bool = False

    # A backend choice remains pending while its config write runs in a
    # thread. The canvas keeps rendering the persisted app resolver until
    # completion; this target sequences rapid clicks. An older completion
    # can neither remain the final persisted value nor repaint a newer
    # choice.
    backend_target: str | None = None
    backend_generation: int = 0
    backend_save_lock: threading.Lock = field(default_factory=threading.Lock)

    # Ingest canvas form echo -- a single bundled mutable dataclass
    # (rather than a scatter of scalar fields like the sync panel
    # above) since every field here is reset together on rail
    # re-entry (see ``_reset_library_ingest_transient_state``); the
    # job queue itself is registry-owned, not screen state.
    form: LibraryIngestFormState = field(default_factory=LibraryIngestFormState)

    # Dedupe counter for the "poke the source snapshot on transitions
    # into done" rule (Task 5's registry listener): only re-fetch when
    # the registry's done-job count has grown since the last time this
    # screen checked. Seeded from the live registry in ``on_mount``
    # (not here) so a re-mounted, cached screen instance never treats
    # jobs that finished in a previous mount as a fresh transition.
    last_done_count: int = 0

    # Pre-flight analysis worker for the ingest path field. Cancelled
    # and replaced on every new trigger so rapid edits never stack.
    preflight_worker: Worker | None = None

    # Monotonic stamp for pre-flight validity. Worker cancellation is
    # cooperative, so a cancelled worker can still deliver its result;
    # `_apply_library_ingest_preflight_result` drops any result whose
    # generation is no longer current (task-2011).
    preflight_generation: int = 0

    # (task-2015) While-typing validation: each path edit restarts this
    # timer; its fire runs the pre-flight so feedback no longer waits
    # for blur.
    path_debounce_timer: Timer | None = None

    # (task-2015) Batch-settle toast bookkeeping: active-job count at the
    # last registry tick, and the (done, failed) counts captured when the
    # queue went from idle to active -- the settle toast reports deltas
    # against that baseline.
    last_active_count: int = 0
    batch_baseline: tuple[int, int, int, int] = (0, 0, 0, 0)

    # (task-2015) Two-press "Clear finished": first press arms, second
    # clears; any registry mutation disarms.
    clear_finished_armed: bool = False
    clear_finished_armed_at: float = 0.0

    # Two-press inline Start consent for tooling risk and active-source
    # duplicates. The immutable request fingerprint is the sole armed
    # carrier, so lifecycle repaint tokens cannot steal consent and a
    # changed source/options/backend/membership cannot inherit it.
    start_consent: _LibraryIngestStartConsent | None = None
    start_confirm_armed_at: float = 0.0

    # (task-3313) Session-scoped snapshot of the last submitted batch,
    # captured at submit time before the form auto-clears; feeds the
    # "Retry this batch" affordance. Deliberately NOT persisted (the
    # jobs DB has sources but not staged options) and deliberately NOT
    # cleared by rail re-entry -- it is submission history, not form
    # state.
    last_submission: LibraryIngestLastSubmission | None = None

    # (xhigh review + live-verify round) Two-press consent for the
    # DESTRUCTIVE half of "Retry this batch". Re-staging replaces
    # path/title/author/keywords/options wholesale with no undo, and
    # the ``r`` accelerator can fire it from any non-text focus -- so
    # when the re-stage would discard work the user entered since the
    # submit, the first press only arms (the affordance's own label
    # becomes the confirm) and the second replaces the form. A form
    # that holds nothing the re-stage would discard skips consent
    # entirely: friction with nothing at stake is just friction.
    retry_confirm_armed: bool = False
    retry_confirm_armed_at: float = 0.0

    # (task-2130) Durable session ledger: terminal jobs snapshotted at
    # Clear-finished time so Recent imports (incl. failure records)
    # survives the registry removal.
    recent_ledger: list[LibraryIngestJob] = field(default_factory=list)

    # (task-2043) Failed rows whose inline error details are expanded.
    expanded_details: set[str] = field(default_factory=set)
