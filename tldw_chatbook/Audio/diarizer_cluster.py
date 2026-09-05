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
    """Map each final cluster id to the nearest live cluster id by cosine.

    Args:
        live_centroids: Dict mapping live cluster ID to centroid vector.
        final: List of (cluster_id, centroid) tuples from a batch pass.

    Returns:
        Dict mapping final cluster ID to the nearest live cluster ID.
        Skips final clusters with no live centroids to match.
    """
    out: dict[str, str] = {}
    for fid, fcen in final:
        best_id, best_sim = None, -1.0
        for lid, lcen in live_centroids.items():
            sim = _cos(np.asarray(fcen, np.float32), np.asarray(lcen, np.float32))
            if sim > best_sim:
                best_id, best_sim = lid, sim
        if best_id is not None:
            out[fid] = best_id
    return out
