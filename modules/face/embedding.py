from __future__ import annotations

import numpy as np


class IdentityEmbedder:
    """
    Extracts 512-dim ArcFace L2-normalised embeddings via InsightFace.
    Used for post-generation identity verification.
    """

    def __init__(self, aligner: "FaceAligner"):
        self.aligner = aligner

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        faces = self.aligner.app.get(image_bgr)
        if not faces:
            raise ValueError("No face found for embedding extraction.")
        face = max(faces, key=lambda f: f.det_score)
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raise ValueError("Face embedding is unavailable for the detected face.")
        return embedding.astype(np.float32)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))
