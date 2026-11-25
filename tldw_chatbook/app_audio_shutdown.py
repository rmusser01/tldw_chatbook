"""Close app audio intake and observe the retained output custodian."""


def begin_audio_shutdown(app):
    """Close future audio admission synchronously, including an unused lazy host."""
    app._audio_output_closed = True
    listening = vars(app).get("article_listening_service")
    if listening is not None:
        listening.begin_close()
    owner = vars(app).get("_audio_output")
    observation = owner.begin_shutdown() if owner is not None else None
    for release in tuple(vars(app).get("_retired_handoff_files", ())):
        try:
            release()
        except Exception as error:
            from loguru import logger

            logger.warning(
                "Retired audio file cleanup failed type={}", type(error).__name__
            )
    return observation


async def observe_audio_shutdown(app, budget_seconds: float) -> bool:
    """Observe checked local retirement within the caller's shutdown budget."""
    import asyncio
    from loguru import logger

    observation = begin_audio_shutdown(app)
    if observation is None:
        return True
    owner = app._audio_output
    try:
        confirmed = await asyncio.wait_for(observation, budget_seconds)
    except TimeoutError:
        confirmed = False
    if owner.handoff_epoch is not None:
        logger.info("Audio was handed to an external player")
    if not confirmed:
        logger.warning("Audio retirement unconfirmed for owner {}", owner.owner)
    return confirmed
