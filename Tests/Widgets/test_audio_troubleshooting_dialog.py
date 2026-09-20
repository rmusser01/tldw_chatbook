# TASK-32830: ``AudioTroubleshootingDialog.on_mount`` used to call
# ``self.run_worker(self._initialize_audio())`` -- but ``_initialize_audio``
# is ``@work``-decorated, so calling it already returns a started ``Worker``;
# re-passing that Worker to ``run_worker`` always raised
# ``WorkerError('Unsupported attempt to run an async worker')`` on every
# mount, production included (documented in .superpowers/wave2-report.md).
# These tests pin the mount-and-scan contract.

from textual.app import App
from textual.widgets import Label, Select

from tldw_chatbook.Widgets.audio_troubleshooting_dialog import (
    AudioTroubleshootingDialog,
)


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _StubDictationService:
    """Boundary stub for the audio backend seam.

    The real ``LazyLiveDictationService`` constructor loads the CLI config,
    which trips this machine's pre-existing
    ``RecoveryRequired: raw_source_selection_changed`` condition when config
    first loads mid-test-session (see .superpowers/wave2-report.md and the
    wide-tier builder notes in Tests/UI/test_modal_wide_tier.py for the same
    environment seam). The device scan flow under test only needs the
    service's synchronous device listing.
    """

    def __init__(self, *_args, **_kwargs):
        self.audio_device = None

    def get_audio_devices(self):
        return [
            {
                "name": "Stub Microphone",
                "id": 1,
                "channels": 1,
                "sample_rate": 16000,
                "is_default": True,
            }
        ]

    def set_audio_device(self, device_id):
        self.audio_device = device_id


async def test_audio_troubleshooting_dialog_mounts_and_runs_scan_worker(
    monkeypatch,
):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.audio_troubleshooting_dialog"
        ".LazyLiveDictationService",
        _StubDictationService,
    )

    dialog = AudioTroubleshootingDialog()
    app = _ScreenHost(dialog)

    async with app.run_test() as pilot:
        # on_mount must start the initialization worker without wrapping the
        # @work-decorated method's Worker in run_worker (WorkerError crashes
        # the app, which run_test re-raises).
        await pilot.pause()
        assert app.screen is dialog

        # The device-scan worker itself must actually run to completion: the
        # lazy dictation service is constructed, the enumerated devices land
        # in the (now enabled) device Select, and the status line leaves its
        # initial "Checking audio system..." state behind.
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(dialog.dictation_service, _StubDictationService)

        device_select = dialog.query_one("#device-select", Select)
        assert device_select.disabled is False
        assert device_select.value == 1

        status_text = str(dialog.query_one("#status-text", Label).renderable)
        assert status_text == "✅ Audio system ready"
        assert not dialog.is_testing


async def test_device_scan_is_submitted_as_a_thread_worker(monkeypatch):
    """Qodo PR-2751 finding 1: _get_devices_safe must run as a thread worker.

    The enumeration is a blocking, synchronous sounddevice query; submitting
    it as an async worker raises WorkerError('Request to run a non-async
    function as an async worker') and the scan never completes. This pins
    the submission flags rather than just the symptom.
    """
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.audio_troubleshooting_dialog"
        ".LazyLiveDictationService",
        _StubDictationService,
    )

    dialog = AudioTroubleshootingDialog()
    submissions = []
    real_run_worker = dialog.run_worker

    def _recording_run_worker(work, **kwargs):
        submissions.append((work, kwargs))
        return real_run_worker(work, **kwargs)

    # Instance-level wrapper: both the @work wrapper (for the async
    # _initialize_audio worker) and the scan submission inside it resolve
    # ``self.run_worker`` at call time, so every submission is captured and
    # still delegated to the real machinery.
    dialog.run_worker = _recording_run_worker

    app = _ScreenHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    scan_submissions = [
        (work, kwargs)
        for work, kwargs in submissions
        if getattr(work, "__name__", "") == "_get_devices_safe"
    ]
    assert len(scan_submissions) == 1, submissions
    _work, kwargs = scan_submissions[0]
    assert kwargs.get("thread") is True
