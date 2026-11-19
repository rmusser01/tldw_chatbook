"""Initializer publication stays with its selected source and loop owner."""

import asyncio
import threading

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source_before", "owner_before", "source_held", "owner_held"])
async def test_trust_initializer_refuses_changed_source_without_publication(change):
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    owner = ServiceWiringMixin()
    owner._local_skill_trust_service = None
    owner._local_skill_trust_service_build_lock = asyncio.Lock()
    entered, release = threading.Event(), threading.Event()
    source_current = change != "source_before"
    owner_current = change != "owner_before"
    candidate = object()
    calls = []
    reads = set()
    loop_thread = threading.current_thread()

    def build():
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(3), "test-owned builder release timed out"
        return candidate

    def current_owner(expected=None):
        assert threading.current_thread() is loop_thread
        return owner_current and owner._local_skill_trust_service is expected

    owner._build_local_skill_trust_service = build
    task = asyncio.create_task(owner.ensure_local_skill_trust_service(
        _source_current=lambda: source_current,
        _owner_current=current_owner,
        _read_observers=(reads,),
    ))
    try:
        if change.endswith("held"):
            deadline = asyncio.get_running_loop().time() + 3
            while not entered.is_set() and not task.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.001)
            if task.done():
                await task
            assert len(reads) == 1
            read = next(iter(reads))
            assert read.creator is owner and read.task is task and not read.retired.done()
            if change == "source_held":
                source_current = False
            else:
                owner_current = False
        release.set()
        with pytest.raises(RuntimeError, match="console_skill_trust_(source|owner)_changed"):
            await task
        assert owner._local_skill_trust_service is None
        assert not owner._local_skill_trust_service_build_lock.locked()
        assert not reads
        assert len(calls) == int(change.endswith("held"))
        assert all(thread is not loop_thread for thread in calls)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


# --- TASK-40719: the active publication and the published slot decide, not the caller ---


class _WrappedGetter:
    """A delegating wrapper around the stock getter, as a profiler or subclass adds."""

    @property
    def local_skill_trust_service(self):
        from tldw_chatbook.app_service_wiring import ServiceWiringMixin

        return ServiceWiringMixin.local_skill_trust_service.fget(self)


class _WrappedInitializer:
    async def ensure_local_skill_trust_service(self, **kwargs):
        from tldw_chatbook.app_service_wiring import ServiceWiringMixin

        return await ServiceWiringMixin.ensure_local_skill_trust_service(self, **kwargs)


def _trust_owner(wrapper=None, built=None):
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    bases = (wrapper, ServiceWiringMixin) if wrapper else (ServiceWiringMixin,)
    owner = type("Owner", bases, {})()
    owner._local_skill_trust_service = None
    owner._local_skill_trust_service_build_lock = asyncio.Lock()
    owner._build_local_skill_trust_service = lambda: built
    return owner


def _publication(app):
    """The runtime-owned registration the native preparation opens for ``app``."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Skills_Interop.skill_trust_models import (
        _SkillTrustPublisher,
        _skill_publisher_identity,
        _skill_trust_publisher,
    )

    @contextmanager
    def scope():
        thread_id, task = _skill_publisher_identity()
        preparation = SimpleNamespace(trust_source=SimpleNamespace(app=app))
        marker = _SkillTrustPublisher(preparation, thread_id, task, None)
        token = _skill_trust_publisher.set(marker)
        try:
            yield marker
        finally:
            marker.active = False
            _skill_trust_publisher.reset(token)

    return scope()


async def _first_use(owner, entry):
    if entry == "getter":
        return owner.local_skill_trust_service
    return await owner.ensure_local_skill_trust_service()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["getter", "initializer"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["stock", "wrapped"])
async def test_first_use_build_is_recorded_however_the_trust_service_is_reached(entry, wrapped):
    wrapper = (_WrappedGetter if entry == "getter" else _WrappedInitializer) if wrapped else None
    built = object()
    owner = _trust_owner(wrapper, built)

    with _publication(owner) as marker:
        assert await _first_use(owner, entry) is built

    assert marker.candidate is built


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["getter", "initializer"])
async def test_a_setter_installed_service_is_never_recorded_as_built(entry):
    owner = _trust_owner(built=object())
    installed = object()

    with _publication(owner) as marker:
        owner.local_skill_trust_service = installed
        assert await _first_use(owner, entry) is installed

    assert marker.candidate is None


def test_a_build_on_another_thread_is_not_recorded_for_this_publication():
    import contextvars

    built = object()
    owner = _trust_owner(built=built)
    with _publication(owner) as marker:
        worker = threading.Thread(
            target=contextvars.copy_context().run,
            args=(lambda: owner.local_skill_trust_service,),
        )
        worker.start()
        worker.join(5)
        assert owner._local_skill_trust_service is built

    assert marker.candidate is None


def test_a_build_for_another_app_or_after_the_publication_is_not_recorded():
    other = _trust_owner(built=object())
    owner = _trust_owner(built=object())
    with _publication(owner) as marker:
        assert other.local_skill_trust_service is not None
    assert owner.local_skill_trust_service is not None

    assert marker.candidate is None


def test_only_the_published_slot_value_is_recorded():
    from tldw_chatbook.app_skill_wiring import record_native_skill_trust_candidate

    owner = _trust_owner(built=object())
    published = owner.local_skill_trust_service
    with _publication(owner) as marker:
        record_native_skill_trust_candidate(owner, object())
        record_native_skill_trust_candidate(owner, None)

    assert published is not None and marker.candidate is None
