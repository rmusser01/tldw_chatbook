"""Incremental speaker clustering over voice embeddings (pure numpy, no torch)."""
from __future__ import annotations

import numpy as np

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe on zero vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
    """
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class OnlineClusterer:
    """Assign each embedding to the nearest centroid within `threshold` cosine
    distance, else start a new cluster up to `max_speakers`; past the cap, fold
    into a non-pinned cluster. A pinned cluster centroid is never corrupted.

    Args:
        threshold: Cosine distance threshold (1 - similarity) for matching.
        max_speakers: Maximum number of clusters to maintain.
    """
    def __init__(self, threshold: float = 0.25, max_speakers: int = 8) -> None:
        self._threshold = threshold
        self._max = max_speakers
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._n = 0

    def assign(self, embedding: np.ndarray) -> str:
        """Assign embedding to a cluster ID.

        Args:
            embedding: Voice embedding vector.

        Returns:
            Cluster ID string (e.g., "S1", "S2").
        """
        emb = np.asarray(embedding, dtype=np.float32)
        best_id, best_sim = None, -1.0
        for cid, cen in self._centroids.items():
            sim = _cos(emb, cen)
            if sim > best_sim:
                best_id, best_sim = cid, sim

        # Check if near_enough to join best cluster
        near_enough = best_id is not None and (1.0 - best_sim) <= self._threshold
        if near_enough:
            # Join best cluster (even if pinned, similar voices legitimately join)
            cid = best_id
            n = self._counts[cid] + 1
            self._centroids[cid] = (self._centroids[cid] * self._counts[cid] + emb) / n
            self._counts[cid] = n
            return cid

        # Not near enough: create new cluster if under cap
        if len(self._centroids) < self._max:
            self._n += 1
            cid = f"S{self._n}"
            self._centroids[cid] = emb.copy()
            self._counts[cid] = 1
            return cid

        # Cap reached: fold into nearest NON-PINNED cluster
        best_id_nonpinned, best_sim_nonpinned = None, -1.0
        for cid, cen in self._centroids.items():
            if cid not in self._pinned:
                sim = _cos(emb, cen)
                if sim > best_sim_nonpinned:
                    best_id_nonpinned, best_sim_nonpinned = cid, sim

        if best_id_nonpinned is not None:
            # Found a non-pinned cluster to fold into
            cid = best_id_nonpinned
            n = self._counts[cid] + 1
            self._centroids[cid] = (self._centroids[cid] * self._counts[cid] + emb) / n
            self._counts[cid] = n
            return cid

        # All clusters are pinned: return nearest but do NOT update centroid
        return best_id

    def pin(self, cluster_id: str) -> None:
        """Mark a cluster as pinned (centroid never corrupted by folds).

        Args:
            cluster_id: Cluster ID to pin (e.g., "S1").
        """
        self._pinned.add(cluster_id)

    def centroids(self) -> dict[str, np.ndarray]:
        """Return a copy of current cluster centroids.

        Returns:
            Dict mapping cluster ID to centroid vector.
        """
        return {k: v.copy() for k, v in self._centroids.items()}

def reconcile(live_centroids: dict[str, np.ndarray],
              final: list[tuple[str, np.ndarray]]) -> dict[str, str]:
    """Match final cluster ids to live cluster ids by cosine, one-to-one.

    Best-first greedy matching: the closest (final, live) pair anywhere wins,
    then both are taken out of the running. The pairing MUST be injective --
    a live cluster is one person, so two distinct final clusters can never
    both be it. Mapping each final id independently to its nearest live id
    (Qodo Q11) quietly re-collapsed everything the authoritative batch pass
    had just separated: with one live cluster and two real speakers, both
    final clusters mapped back to "S1". The caller mints a fresh live-style
    id for whatever stays unmatched.

    Args:
        live_centroids: Dict mapping live cluster ID to centroid vector.
        final: List of (cluster_id, centroid) tuples from a batch pass.

    Returns:
        Dict mapping final cluster ID to its matched live cluster ID. Final
        clusters left over (more final clusters than live ones, or no live
        centroids at all) are absent.
    """
    pairs = [
        (
            -_cos(np.asarray(fcen, np.float32), np.asarray(lcen, np.float32)),
            fid,
            lid,
        )
        for fid, fcen in final
        for lid, lcen in live_centroids.items()
    ]
    pairs.sort()  # most similar first; ids break ties deterministically
    out: dict[str, str] = {}
    taken: set[str] = set()
    for _neg_sim, fid, lid in pairs:
        if fid in out or lid in taken:
            continue
        out[fid] = lid
        taken.add(lid)
    return out


def merged_speaker_names(
    transitions: list[tuple[str | None, str]],
    names: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Spec §4: when the authoritative Stop pass folds two *differently
    user-named* live clusters into one, keep BOTH names on the surviving
    cluster ("Alice / Bob") and flag it for the user to resolve -- never
    silently drop one.

    Pure Python (no numpy): it works off the per-segment id transitions the
    session's Stop overlay already produces, because speaker *names* never
    cross the diarizer pipe (privacy, spec §3.4) and so can only be reconciled
    where the name map lives -- the app process, not the worker that holds the
    centroids `reconcile` compares.

    Args:
        transitions: one ``(old_id, new_id)`` per meeting segment the Stop
            pass covered -- the near-live cluster id it carried and the
            reconciled id it took. ``old_id`` is None for a segment never
            labelled live.
        names: the meeting's ``cluster_id -> user name`` map. Only
            user-assigned names are present; generic "Speaker N" ids are
            absent.

    Returns:
        ``(merged_names, flagged)``:
          - ``merged_names``: ``{survivor_id: "Alice / Bob"}`` to merge into
            the name map, one entry per collision.
          - ``flagged``: survivor ids, in first-seen order.
    """
    targets: dict[str, set[str]] = {}
    order: list[str] = []
    seen_new: set[str] = set()
    for old, new in transitions:
        if new not in seen_new:
            seen_new.add(new)
            order.append(new)
        if old:
            targets.setdefault(old, set()).add(new)
    merged_names: dict[str, str] = {}
    flagged: list[str] = []
    for survivor in order:
        # Named live clusters folded ENTIRELY into `survivor` (kept none of
        # their own segments -- a cluster that still owns some of its segments
        # is not a merge and its name is not at risk).
        absorbed = [
            old for old, news in targets.items()
            if old != survivor and news == {survivor} and old in names
        ]
        if not absorbed:
            continue
        parts: list[str] = []
        for candidate in ([survivor] if survivor in names else []) + absorbed:
            label = names[candidate]
            if label not in parts:      # duplicate names are not a collision
                parts.append(label)
        if len(parts) >= 2:
            merged_names[survivor] = " / ".join(parts)
            flagged.append(survivor)
    return merged_names, flagged
