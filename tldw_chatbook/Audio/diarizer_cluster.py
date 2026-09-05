"""Incremental speaker clustering over voice embeddings (pure numpy, no torch)."""
from __future__ import annotations
import numpy as np

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class OnlineClusterer:
    """Assign each embedding to the nearest centroid within `threshold` cosine
    distance, else start a new cluster up to `max_speakers`; past the cap, fold
    into the nearest. A pinned cluster is never dropped."""
    def __init__(self, threshold: float = 0.25, max_speakers: int = 8) -> None:
        self._threshold = threshold
        self._max = max_speakers
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._n = 0

    def assign(self, embedding: np.ndarray) -> str:
        emb = np.asarray(embedding, dtype=np.float32)
        best_id, best_sim = None, -1.0
        for cid, cen in self._centroids.items():
            sim = _cos(emb, cen)
            if sim > best_sim:
                best_id, best_sim = cid, sim
        near_enough = best_id is not None and (1.0 - best_sim) <= self._threshold
        if not near_enough and len(self._centroids) < self._max:
            self._n += 1
            cid = f"S{self._n}"
            self._centroids[cid] = emb.copy()
            self._counts[cid] = 1
            return cid
        cid = best_id  # nearest existing (also the cap-fold path)
        n = self._counts[cid] + 1
        self._centroids[cid] = (self._centroids[cid] * self._counts[cid] + emb) / n
        self._counts[cid] = n
        return cid

    def pin(self, cluster_id: str) -> None:
        self._pinned.add(cluster_id)

    def centroids(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._centroids.items()}

def reconcile(live_centroids: dict[str, np.ndarray],
              final: list[tuple[str, np.ndarray]]) -> dict[str, str]:
    """Map each final cluster id to the nearest live cluster id by cosine."""
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
