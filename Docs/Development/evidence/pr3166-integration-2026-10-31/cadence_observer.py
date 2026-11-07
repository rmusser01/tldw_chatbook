"""Record only cadence input indices and publication call sites."""

import traceback


def pytest_runtest_setup(item):
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    warm = spend.ConsoleReadinessConfigProjection.warm

    async def observed_warm(self):
        result = await warm(self)
        print(
            "WARM",
            result,
            "age",
            spend.time.monotonic() - self.at,
            "pending",
            self.pending,
            "key",
            spend._same_readiness_key(self._key(), self.key),
            "has_value",
            self.value is not None,
            flush=True,
        )
        return result

    spend.ConsoleReadinessConfigProjection.warm = observed_warm
    original = config_sync._projection_inputs
    publish = config_sync.publish_console_presentation

    def inputs(screen):
        objects, values = original(screen)
        prior = config_sync._cadence(screen)
        if values != prior.values or prior.requested != prior.served:
            changed = [
                i for i, (a, b) in enumerate(zip(values, prior.values or ())) if a != b
            ]
            print(
                "CADENCE",
                changed,
                "requests",
                prior.requested,
                prior.served,
                "publication",
                prior.published,
                flush=True,
            )
        return objects, values

    def published(*args, **kwargs):
        print(
            "PUBLICATION",
            [frame.name for frame in traceback.extract_stack(limit=5)],
            flush=True,
        )
        return publish(*args, **kwargs)

    config_sync._projection_inputs = inputs
    config_sync.publish_console_presentation = published
    item._restore_cadence = lambda: (
        setattr(spend.ConsoleReadinessConfigProjection, "warm", warm),
        setattr(config_sync, "_projection_inputs", original),
        setattr(config_sync, "publish_console_presentation", publish),
    )


def pytest_runtest_teardown(item):
    item._restore_cadence()
