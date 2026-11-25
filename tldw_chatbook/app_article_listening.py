"""App composition for the local Watchlists listening transport."""


def build_article_listening_service(app, db):
    """Compose a profile-bound transport without opening speech or touching audio files."""
    from .Subscriptions.article_listening_service import ArticleListeningService
    from .Subscriptions.article_listening_store import (
        ArticleListeningStore,
        ListeningConflict,
    )
    from tldw_chatbook.config import (
        get_cli_config_path,
        get_cli_setting,
        get_user_data_dir,
    )
    from .Subscriptions.article_listening_models import (
        ListeningAuthority,
        listening_profile_key,
    )

    selector, root = get_cli_config_path(), get_user_data_dir()
    profile = listening_profile_key(selector)

    def authorize():
        state, generation = app.runtime_policy.snapshot()
        if (
            state.active_source != "local"
            or get_cli_config_path() != selector
            or app.subscriptions_db is not db
        ):
            raise ListeningConflict("local_listening_authority_required")
        from .runtime_policy.types import PolicyDeniedError

        try:
            app.service_policy_enforcer.require_allowed(
                action_id="watchlists.items.detail.local", runtime_state_override=state
            )
        except PolicyDeniedError:
            raise ListeningConflict("local_listening_authority_required") from None
        return ListeningAuthority(profile, "local", generation)

    async def freeze():
        from .Subscriptions.article_listening_audio import freeze_listening_voice

        default = get_cli_setting("app_tts", "default_profile_id", None)
        default = str(default).strip() if default else None
        profiles = await app._ensure_tts_profile_service() if default else None
        return await freeze_listening_voice(
            tts_service=app.tts_service,
            default_profile_id=default,
            profile_service=profiles,
        )

    async def acquire(ticket, run, current):
        from tldw_chatbook.UI.audio_output_admission import acquire_audio_output

        return await acquire_audio_output(
            app, ticket, "Watchlists", run, current=current
        )

    async def speech(voice):
        profiles = (
            await app._ensure_tts_profile_service() if voice.dependencies else None
        )
        return app.tts_service, profiles

    return ArticleListeningService(
        lambda: ArticleListeningStore(db, current_authority=authorize),
        root,
        freeze_voice=freeze,
        authorize=authorize,
        output=lambda: app.audio_output,
        acquire_output=acquire,
        speech_services=speech,
    )
