"""Revision-checked article listening snapshots and source-owned storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.subscriptions_schema import listening_brief_items
from tldw_chatbook.TTS.controlled_wav_profile import WavBookmark

from .article_listening_lease import ListeningBusy, ListeningLease, hold_listening
from .article_listening_models import (
    AudioMetadata,
    AudioReservation,
    EntryCandidate,
    FrozenVoice,
    ListeningAuthority,
    ListeningCandidate,
    ListeningCheckpoint,
    ListeningConflict,
    ListeningEntry,
    ListeningSession,
    VoiceDependency,
    MAX_TEXT_BYTES,
    MAX_SESSION_BYTES,
    listening_profile_key,
    _KEY,
    _encoded,
    _digest,
    _hash,
    _int,
    _id,
)

from .item_persist import (
    CONTENT_FORMAT_MARKDOWN,
    CONTENT_FORMAT_TEXT,
    CONTENT_KIND_ARTICLE,
)

__all__ = [
    "listening_profile_key",
    "ArticleListeningStore",
    "AudioMetadata",
    "AudioReservation",
    "ListeningLease",
    "ListeningBusy",
    "EntryCandidate",
    "FrozenVoice",
    "ListeningAuthority",
    "ListeningCandidate",
    "ListeningCheckpoint",
    "ListeningConflict",
    "ListeningEntry",
    "ListeningSession",
    "VoiceDependency",
]


class ArticleListeningStore:
    """Store snapshots and issue source- and revision-bound audio reservations."""

    def __init__(
        self,
        db: SubscriptionsDB,
        *,
        current_authority: Callable[[], ListeningAuthority],
    ):
        self.db = db
        self.database = db.listening_binding
        self._current = current_authority
        self._profile = self._authority().profile
        self._issued_audio = WeakValueDictionary()

    def _authority(self, expected=None):
        current = self._current()
        if (
            type(current) is not ListeningAuthority
            or current.profile != getattr(self, "_profile", current.profile)
            or expected is not None
            and expected != current
        ):
            raise ListeningConflict("listening_authority_changed")
        return current

    def mutation_lease(self, *, related_paths: tuple[Path, ...] = ()):
        return hold_listening(
            self.database,
            profile=self._authority().profile,
            related_paths=related_paths,
            memory=self.db.is_memory_db,
        )

    def _state(self, c):
        profile = self._authority().profile
        row = c.execute(
            "SELECT * FROM listening_session WHERE profile=?", (profile,)
        ).fetchone()
        if row is None:
            return {
                "id": None,
                "profile": profile,
                "revision": 0,
                "incarnation": "",
                "transport": "empty",
            }
        return row

    def _checked(self, c, revision, authority, *, new_intent=False):
        self._authority(authority)
        state = self._state(c)
        # Clear retains a witness. It presents None only to the replacement CAS.
        if state["transport"] == "empty":
            if revision is not None:
                raise ListeningConflict("listening_revision_changed")
        elif (
            revision is None
            or _int(revision) != state["revision"]
            or (
                state["profile"] != authority.profile
                or not new_intent
                and state["policy_epoch"] != authority.policy_epoch
            )
        ):
            raise ListeningConflict("listening_revision_changed")
        return state

    def load_session(self) -> ListeningSession | None:
        with self.db.artifact_read_snapshot() as c:
            return self._load(c)

    def _load(self, c):
        s = self._state(c)
        if s["transport"] == "empty":
            return None
        entries = []
        foreground = None
        for r in c.execute(
            "SELECT e.*,a.status FROM listening_entries e LEFT JOIN article_audio a "
            "ON a.generation=e.artifact WHERE session_id=? ORDER BY slot",
            (s["id"],),
        ):
            voice = (
                FrozenVoice.from_json(r["voice"]) if r["voice"] is not None else None
            )
            point = (
                WavBookmark(r["bookmark_hash"], r["bookmark_rate"], r["bookmark_frame"])
                if r["bookmark_hash"]
                else None
            )
            availability = (
                "unavailable" if r["item_id"] is None else r["status"] or "pending"
            )
            entry = ListeningEntry(
                r["entry_id"],
                r["item_id"],
                r["source_id"],
                r["title"],
                r["text"],
                r["converter"],
                voice,
                r["artifact"],
                point,
                availability,
                r["unavailable_reason"],
            )
            if r["slot"] == -1:
                foreground = entry
            else:
                entries.append(entry)
        return ListeningSession(
            s["revision"],
            s["incarnation"],
            ListeningAuthority(s["profile"], "local", s["policy_epoch"]),
            FrozenVoice.from_json(s["default_voice"]),
            s["playback_generation"],
            s["brief_id"],
            tuple(entries),
            foreground,
            s["cursor"],
            bool(s["resume_queue"]),
            "completed" if s["transport"] == "completed" else "paused",
        )

    def _snapshot(self, c, item_id, voice, converter, prepare_text, origin):
        if item_id is None:
            return EntryCandidate.unavailable("missing_item", origin=origin)
        row = c.execute(
            "SELECT subscription_id,length(CAST(title AS BLOB)),length(CAST(content AS BLOB)),"
            "CASE WHEN content_kind IS NULL AND content_format IS NULL THEN NULL "
            "WHEN content_kind=? AND content_format IN (?,?) THEN content_format ELSE '' END "
            "FROM subscription_items WHERE id=?",
            (
                CONTENT_KIND_ARTICLE,
                CONTENT_FORMAT_TEXT,
                CONTENT_FORMAT_MARKDOWN,
                _int(item_id, 1),
            ),
        ).fetchone()
        if not row:
            return EntryCandidate.unavailable("missing_item", origin=origin)
        fmt = row[3]
        if fmt == "":
            return EntryCandidate.unavailable("unsupported_content", origin=origin)
        kind = CONTENT_KIND_ARTICLE if fmt is not None else None
        if (row[1] or 0) > 2048 or (row[2] or 0) > MAX_TEXT_BYTES:
            raise ValueError("listening_article_limit")
        title, raw = c.execute(
            "SELECT title,content FROM subscription_items WHERE id=?", (item_id,)
        ).fetchone()
        if raw is None or isinstance(raw, str) and not raw.strip():
            return EntryCandidate.unavailable("empty_body", origin=origin)
        if type(raw) is not str:
            return EntryCandidate.unavailable("unsupported_content", origin=origin)
        text = prepare_text(raw, kind, fmt)
        if type(text) is not str:
            raise ValueError("invalid_listening_text")
        return (
            EntryCandidate(
                item_id,
                row[0],
                title or "",
                text,
                converter,
                voice,
                origin,
            )
            if text.strip()
            else EntryCandidate.unavailable("empty_body", origin=origin)
        )

    def prepare_item(
        self,
        item_id: int,
        *,
        authority: ListeningAuthority,
        voice: FrozenVoice,
        converter: str,
        prepare_text: Callable[[str, str | None, str | None], str],
    ) -> ListeningCandidate:
        self._authority(authority)
        _int(item_id, 1)
        with self.db.artifact_read_snapshot() as c:
            s = self._state(c)
            image = self.db.listening_image
            origin = (authority, self.database, image)
            return ListeningCandidate(
                authority,
                self.database,
                image,
                s["revision"],
                s["incarnation"],
                voice,
                None,
                (),
                self._snapshot(c, item_id, voice, converter, prepare_text, origin),
            )

    def prepare_brief(
        self,
        brief_id: int,
        *,
        authority: ListeningAuthority,
        voice: FrozenVoice,
        converter: str,
        prepare_text: Callable[[str, str | None, str | None], str],
    ) -> ListeningCandidate:
        self._authority(authority)
        _int(brief_id, 1)
        with self.db.artifact_read_snapshot() as c:
            s = self._state(c)
            image = self.db.listening_image
            origin = (authority, self.database, image)
            queue = []
            total = 0
            for item in listening_brief_items(c, brief_id):
                entry = self._snapshot(c, item, voice, converter, prepare_text, origin)
                total += len((entry.text or "").encode())
                if total > MAX_SESSION_BYTES:
                    raise ValueError("listening_brief_limit")
                queue.append(entry)
            if not queue:
                raise ListeningConflict("no_playable_articles")
            return ListeningCandidate(
                authority,
                self.database,
                image,
                s["revision"],
                s["incarnation"],
                voice,
                brief_id,
                tuple(queue),
                None,
            )

    def _insert(self, c, session_id, slot, e, origin):
        if e.origin != origin:
            raise ListeningConflict("listening_entry_authority_changed")
        key = uuid4().hex
        if e.item_id is None:
            c.execute(
                "INSERT INTO listening_entries(session_id,slot,entry_id,unavailable_reason) VALUES (?,?,?,?)",
                (session_id, slot, key, e.unavailable_reason),
            )
            return
        source = c.execute(
            "SELECT subscription_id FROM subscription_items WHERE id=?", (e.item_id,)
        ).fetchone()
        if not source or source[0] != e.source_id:
            raise ListeningConflict("listening_source_changed")
        voice = asdict(e.voice)
        serialized = _encoded(voice)
        c.execute(
            "INSERT INTO listening_entries(session_id,slot,entry_id,item_id,source_id,title,text,speech_hash,"
            "converter,voice,voice_hash,dependency_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                slot,
                key,
                e.item_id,
                e.source_id,
                e.title,
                e.text,
                _hash(e.text),
                e.converter,
                serialized,
                _hash(serialized),
                _hash(_encoded(voice["dependencies"])),
            ),
        )

    def replace_session(
        self,
        candidate: ListeningCandidate,
        *,
        expected_revision: int | None,
        lease: ListeningLease | None = None,
    ) -> ListeningSession:
        a = self._authority(candidate.authority)
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            s = self._checked(c, expected_revision, a, new_intent=True)
            image = self.db.listening_image
            if (
                candidate.database != self.database
                or candidate.image != image
                or (
                    s["revision"],
                    s["incarnation"],
                )
                != (candidate.observed_revision, candidate.observed_incarnation)
            ):
                raise ListeningConflict("stale_listening_candidate")
            if candidate.brief_id is not None:
                authorized = listening_brief_items(c, candidate.brief_id)
                if len(authorized) != len(candidate.queue) or any(
                    e.item_id is not None and e.item_id != item
                    for e, item in zip(candidate.queue, authorized, strict=True)
                ):
                    raise ListeningConflict("brief_membership_changed")
            entries = list(enumerate(candidate.queue))
            if candidate.foreground:
                entries.append((-1, candidate.foreground))
            if not any(e.item_id is not None for _, e in entries):
                raise ListeningConflict("no_playable_articles")
            session_id = s["id"]
            if session_id is None:
                session_id = c.execute(
                    "INSERT INTO listening_session(profile,policy_epoch,revision,incarnation,transport) "
                    "VALUES (?,?,0,'','empty')",
                    (a.profile, a.policy_epoch),
                ).lastrowid
            c.execute("DELETE FROM listening_entries WHERE session_id=?", (session_id,))
            for slot, entry in entries:
                self._insert(c, session_id, slot, entry, (a, self.database, image))
            c.execute(
                "UPDATE listening_session SET revision=revision+1,incarnation=?,profile=?,policy_epoch=?,"
                "default_voice=?,brief_id=?,cursor=?,resume_queue=0,transport='paused',playback_generation=playback_generation+1,"
                "created_at=unixepoch(),updated_at=unixepoch() WHERE id=?",
                (
                    uuid4().hex,
                    a.profile,
                    a.policy_epoch,
                    _encoded(asdict(candidate.voice)),
                    candidate.brief_id,
                    0 if candidate.queue else None,
                    session_id,
                ),
            )
            return self._load(c)

    def clear(
        self, *, expected_revision: int, lease: ListeningLease | None = None
    ) -> None:
        a = self._authority()
        _int(expected_revision)
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            s = self._checked(c, expected_revision, a, new_intent=True)
            c.execute("DELETE FROM listening_entries WHERE session_id=?", (s["id"],))
            c.execute(
                "UPDATE listening_session SET revision=revision+1,incarnation=?,default_voice=NULL,"
                "brief_id=NULL,cursor=NULL,resume_queue=0,transport='empty',playback_generation=playback_generation+1,"
                "updated_at=unixepoch() WHERE id=?",
                (uuid4().hex, s["id"]),
            )

    def checkpoint(
        self,
        update: ListeningCheckpoint,
        *,
        expected_revision: int,
        lease: ListeningLease | None = None,
    ) -> ListeningSession:
        a = self._authority(update.authority)
        _int(expected_revision)
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            s = self._checked(c, expected_revision, a)
            if (s["incarnation"], s["playback_generation"]) != (
                update.incarnation,
                update.playback_generation,
            ):
                raise ListeningConflict("listening_generation_changed")
            for entry_id, point in update.bookmarks:
                changed = c.execute(
                    "UPDATE listening_entries SET bookmark_frame=? WHERE entry_id=? AND session_id=? "
                    "AND bookmark_hash=? AND bookmark_rate=? AND artifact IN ("
                    "SELECT generation FROM article_audio WHERE status='ready' AND content_frames>=?)",
                    (
                        point.frame,
                        entry_id,
                        s["id"],
                        point.sha256,
                        point.sample_rate,
                        point.frame,
                    ),
                ).rowcount
                if changed != 1:
                    raise ListeningConflict("invalid_listening_bookmark")
            if update.foreground != "keep":
                c.execute(
                    "DELETE FROM listening_entries WHERE slot=-1 AND session_id=?",
                    (s["id"],),
                )
                if isinstance(update.foreground, EntryCandidate):
                    self._insert(
                        c,
                        s["id"],
                        -1,
                        update.foreground,
                        (a, self.database, self.db.listening_image),
                    )
            slots = {
                row[0]
                for row in c.execute(
                    "SELECT slot FROM listening_entries WHERE session_id=?", (s["id"],)
                )
            }
            if (
                not slots
                or (update.cursor is None) != (s["brief_id"] is None)
                or (
                    update.cursor is not None
                    and update.cursor not in slots
                    or update.resume_queue
                    and (update.cursor is None or -1 not in slots)
                )
            ):
                raise ListeningConflict("invalid_listening_cursor")
            c.execute(
                "UPDATE listening_session SET revision=revision+1,cursor=?,resume_queue=?,transport=?,"
                "updated_at=unixepoch(),playback_generation=playback_generation+1 WHERE id=?",
                (update.cursor, int(update.resume_queue), update.transport, s["id"]),
            )
            return self._load(c)

    def _entry(self, c, entry_id):
        row = c.execute(
            "SELECT e.* FROM listening_entries e JOIN listening_session s ON s.id=e.session_id WHERE e.entry_id=? AND s.profile=?",
            (_id(entry_id), self._authority().profile),
        ).fetchone()
        if not row or row["item_id"] is None:
            raise ListeningConflict("listening_entry_unavailable")
        return row

    @staticmethod
    def _identity(entry, authority):
        return (
            entry["item_id"],
            authority.profile,
            entry["speech_hash"],
            entry["converter"],
            entry["voice_hash"],
            entry["dependency_hash"],
        )

    def find_reusable(self, entry_id: str) -> tuple[int, ...]:
        """Return matching DB generations; asset readers must verify the retained bytes."""
        a = self._authority()
        with self.db.artifact_read_snapshot() as c:
            s = self._state(c)
            self._checked(c, s["revision"], a)
            entry = self._entry(c, entry_id)
            return tuple(
                r[0]
                for r in c.execute(
                    "SELECT generation FROM article_audio WHERE (item_id,profile,speech_hash,converter,voice_hash,"
                    "dependency_hash)=(?,?,?,?,?,?) AND status='ready' ORDER BY last_used DESC,generation DESC",
                    self._identity(entry, a),
                )
            )

    def reserve_audio(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        regenerate: bool = False,
        lease: ListeningLease | None = None,
    ) -> AudioReservation:
        a = self._authority()
        _int(expected_revision)
        if type(regenerate) is not bool:
            raise ValueError("invalid_regeneration_intent")
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            s = self._checked(c, expected_revision, a)
            e = self._entry(c, entry_id)
            if e["artifact"] is not None and not regenerate:
                raise ListeningConflict("explicit_regeneration_required")
            c.execute(
                "UPDATE article_audio SET item_id=NULL,speech_hash=NULL,converter=NULL,"
                "voice_hash=NULL,dependency_hash=NULL,status='pending_delete',last_used=NULL,created_at=NULL,"
                "target_entry=NULL,target_revision=NULL,regenerate=NULL WHERE target_entry=? AND status='pending'",
                (entry_id,),
            )
            key = uuid4().hex + ".wav"
            generation = c.execute(
                "INSERT INTO article_audio(item_id,profile,speech_hash,converter,voice_hash,dependency_hash,"
                "file_key,status,target_entry,target_revision,regenerate,created_at,last_used) "
                "VALUES (?,?,?,?,?,?,?,'pending',?,?,?,unixepoch(),unixepoch())",
                (
                    *self._identity(e, a),
                    key,
                    entry_id,
                    expected_revision,
                    int(regenerate),
                ),
            ).lastrowid
            claim = AudioReservation(
                generation,
                key,
                entry_id,
                e["item_id"],
                expected_revision,
                s["incarnation"],
                a,
                self.database,
                self.db.listening_image,
            )
            self._issued_audio[generation] = claim
            return claim

    @staticmethod
    def _attach(c, entry, a, generation, regenerate):
        row = c.execute(
            "SELECT * FROM article_audio WHERE generation=? AND status='ready' AND "
            "(item_id,profile,speech_hash,converter,voice_hash,dependency_hash)=(?,?,?,?,?,?)",
            (_int(generation, 1), *ArticleListeningStore._identity(entry, a)),
        ).fetchone()
        if row is None:
            raise ListeningConflict("article_audio_identity_changed")
        if (
            entry["artifact"] is not None
            and not regenerate
            and entry["artifact"] != generation
        ):
            raise ListeningConflict("explicit_regeneration_required")
        frame = entry["bookmark_frame"] if entry["artifact"] == generation else 0
        c.execute(
            "UPDATE listening_entries SET artifact=?,bookmark_hash=?,bookmark_rate=?,bookmark_frame=? "
            "WHERE entry_id=?",
            (generation, row["sha256"], row["sample_rate"], frame, entry["entry_id"]),
        )
        c.execute(
            "UPDATE article_audio SET last_used=unixepoch() WHERE generation=?",
            (generation,),
        )
        c.execute(
            "UPDATE listening_session SET revision=revision+1,updated_at=unixepoch() WHERE id=?",
            (entry["session_id"],),
        )

    def attach_audio(
        self,
        entry_id: str,
        generation: int,
        *,
        expected_revision: int,
        lease: ListeningLease | None = None,
    ) -> ListeningSession:
        a = self._authority()
        _int(expected_revision)
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            self._checked(c, expected_revision, a)
            self._attach(c, self._entry(c, entry_id), a, generation, False)
            return self._load(c)

    def _reservation(self, c, claim):
        self._check_issued(claim)
        if claim.database != self.database:
            raise ListeningConflict("article_audio_reservation_changed")
        a = self._authority(claim.authority)
        s = self._checked(c, claim.revision, a)
        entry = self._entry(c, claim.entry_id)
        row = c.execute(
            "SELECT regenerate FROM article_audio WHERE generation=? AND file_key=? AND "
            "target_revision=? AND target_entry=? AND status='pending' AND "
            "(item_id,profile,speech_hash,converter,voice_hash,dependency_hash)=(?,?,?,?,?,?)",
            (
                claim.generation,
                claim.file_key,
                claim.revision,
                claim.entry_id,
                *self._identity(entry, a),
            ),
        ).fetchone()
        if (
            row is None
            or s["incarnation"] != claim.incarnation
            or entry["item_id"] != claim.item_id
        ):
            raise ListeningConflict("article_audio_reservation_changed")
        return entry, bool(row[0])

    def _check_issued(self, claim):
        if (
            type(claim) is not AudioReservation
            or self._issued_audio.get(claim.generation) is not claim
            or self.db.listening_image != claim.image
        ):
            raise ListeningConflict("article_audio_reservation_changed")

    def check_reservation(
        self, claim: AudioReservation, *, lease: ListeningLease
    ) -> None:
        """Preflight a finalizer under its retained custody before physical publication."""
        with self.db.listening_transaction(
            lease=lease, profile=self._authority().profile
        ) as c:
            self._reservation(c, claim)

    def cancel_audio(
        self, claim: AudioReservation, *, lease: ListeningLease | None = None
    ) -> None:
        """Revoke only this captured pending generation and retain its safe cleanup key."""
        a = self._authority(claim.authority)
        if claim.database != self.database:
            raise ListeningConflict("article_audio_reservation_changed")
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            self._check_issued(claim)
            changed = c.execute(
                "UPDATE article_audio SET item_id=NULL,speech_hash=NULL,converter=NULL,"
                "voice_hash=NULL,dependency_hash=NULL,status='pending_delete',last_used=NULL,created_at=NULL,"
                "target_entry=NULL,target_revision=NULL,regenerate=NULL "
                "WHERE generation=? AND status='pending' AND profile=? AND file_key=? AND item_id=? "
                "AND target_entry=? AND target_revision=?",
                (
                    claim.generation,
                    a.profile,
                    claim.file_key,
                    claim.item_id,
                    claim.entry_id,
                    claim.revision,
                ),
            ).rowcount
            if changed != 1:
                raise ListeningConflict("article_audio_reservation_changed")

    def seal_audio_custody(
        self, claim: AudioReservation, witness: str, *, lease: ListeningLease
    ) -> None:
        """Journal a verified local stage under the exact pending reservation."""
        with self.db.listening_transaction(
            lease=lease, profile=self._authority().profile
        ) as c:
            self._reservation(c, claim)
            changed = c.execute(
                "UPDATE article_audio SET custody_witness=? WHERE generation=? "
                "AND status='pending' AND custody_witness IS NULL",
                (_digest(witness), claim.generation),
            ).rowcount
            if changed != 1:
                raise ListeningConflict("article_audio_reservation_changed")

    def complete_audio(
        self,
        reservation: AudioReservation,
        metadata: AudioMetadata,
        *,
        lease: ListeningLease | None = None,
        custody_witness: str | None = None,
    ) -> ListeningSession:
        """Commit metadata for the asset owner's validated, published immutable generation."""
        a = self._authority(reservation.authority)
        if type(metadata) is not AudioMetadata or reservation.database != self.database:
            raise ListeningConflict("article_audio_reservation_changed")
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            entry, regenerate = self._reservation(c, reservation)
            changed = c.execute(
                "UPDATE article_audio SET status='ready',sha256=?,sample_rate=?,content_frames=?,"
                "file_frames=?,byte_size=?,last_used=unixepoch(),target_entry=NULL,target_revision=NULL,"
                "regenerate=NULL WHERE generation=? AND status='pending' AND custody_witness IS ?",
                (
                    metadata.sha256,
                    metadata.sample_rate,
                    metadata.content_frames,
                    metadata.file_frames,
                    metadata.byte_size,
                    reservation.generation,
                    custody_witness,
                ),
            ).rowcount
            if changed != 1:
                raise ListeningConflict("article_audio_reservation_changed")
            self._attach(c, entry, a, reservation.generation, regenerate)
            return self._load(c)

    def mark_audio_missing(
        self, generation: int, *, lease: ListeningLease | None = None
    ) -> None:
        a = self._authority()
        with self.db.listening_transaction(lease=lease, profile=a.profile) as c:
            self._authority(a)
            changed = c.execute(
                "UPDATE article_audio SET status='missing' WHERE generation=? AND profile=? AND status='ready'",
                (_int(generation, 1), a.profile),
            ).rowcount
            if changed:
                c.execute(
                    "UPDATE listening_session SET revision=revision+1,updated_at=unixepoch() WHERE "
                    "id IN (SELECT session_id FROM listening_entries WHERE artifact=?)",
                    (generation,),
                )

    def pending_deletions(self) -> tuple[tuple[int, str], ...]:
        """Expose only committed safe cleanup keys; actual unlink requires a held lease."""
        if self.db.conn.in_transaction:
            raise ListeningConflict("cleanup_requires_committed_state")
        with self.db.artifact_read_snapshot() as c:
            rows = tuple(
                (r[0], r[1])
                for r in c.execute(
                    "SELECT generation,file_key FROM article_audio "
                    "WHERE status='pending_delete' AND profile=? ORDER BY generation",
                    (self._authority().profile,),
                )
            )
            if any(type(key) is not str or not _KEY.fullmatch(key) for _, key in rows):
                raise ValueError("unsafe_article_audio_key")
            return rows

    def audio_inventory(self, generation: int | None = None) -> tuple:
        """Return this selector's storage receipts and reference protection."""
        profile = self._authority().profile
        query = (
            "SELECT generation,file_key,item_id,status,sha256,sample_rate,content_frames,file_frames,byte_size,"
            "last_used,CASE WHEN typeof(custody_witness)='text' AND length(CAST(custody_witness AS BLOB))=64 "
            "THEN custody_witness WHEN custody_witness IS NOT NULL THEN '' END AS custody_witness, "
            "EXISTS(SELECT 1 FROM listening_entries e WHERE e.artifact=a.generation) "
            "AS referenced FROM article_audio a WHERE profile=?"
        )
        params = (profile,)
        if generation is not None:
            query += " AND generation=?"
            params += (_int(generation, 1),)
        with self.db.artifact_read_snapshot() as c:
            return tuple(c.execute(query + " ORDER BY generation", params))

    def relinquish_audio(
        self, claim: AudioReservation, *, lease: ListeningLease
    ) -> None:
        """Drop only this issued generation's cleanup rights on custody loss."""
        profile = self._authority().profile
        with self.db.listening_transaction(lease=lease, profile=profile) as c:
            self._check_issued(claim)
            if claim.authority.profile != profile or claim.database != self.database:
                raise ListeningConflict("article_audio_reservation_changed")
            # Cancellation or source deletion may already have scrubbed the
            # target fields. The issued generation/key/image still name exactly
            # its tombstone; it must never authorize deleting a substituted file.
            c.execute(
                "DELETE FROM article_audio WHERE generation=? AND profile=? AND file_key=? "
                "AND status IN ('pending','pending_delete') AND NOT EXISTS "
                "(SELECT 1 FROM listening_entries WHERE artifact=article_audio.generation)",
                (claim.generation, profile, claim.file_key),
            )

    def retire_audio(self, generations: list[int], *, lease: ListeningLease) -> None:
        """Commit invalidation only for still-unreferenced, current-profile rows."""
        profile = self._authority().profile
        with self.db.listening_transaction(lease=lease, profile=profile) as c:
            c.executemany(
                "UPDATE article_audio SET status='pending_delete',item_id=NULL,speech_hash=NULL,converter=NULL,"
                "voice_hash=NULL,dependency_hash=NULL,sha256=NULL,sample_rate=NULL,content_frames=NULL,"
                "file_frames=NULL,byte_size=NULL,last_used=NULL,created_at=NULL,target_entry=NULL,"
                "target_revision=NULL,regenerate=NULL WHERE profile=? AND generation=? AND status<>'pending_delete' "
                "AND NOT EXISTS(SELECT 1 FROM listening_entries e WHERE e.artifact=article_audio.generation)",
                ((profile, _int(g, 1)) for g in generations),
            )

    def finish_audio_deletion(
        self, generation: int, key: str, witness: str | None, *, lease: ListeningLease
    ) -> None:
        """Retire the exact tombstone after its private names are gone."""
        profile = self._authority().profile
        with self.db.listening_transaction(lease=lease, profile=profile) as c:
            c.execute(
                "DELETE FROM article_audio WHERE profile=? AND generation=? AND file_key=? "
                "AND status='pending_delete' AND custody_witness IS ?",
                (profile, _int(generation, 1), key, witness),
            )
