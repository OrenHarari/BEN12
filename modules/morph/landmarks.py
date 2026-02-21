from __future__ import annotations

import cv2
import numpy as np


# Mapping from InsightFace 106-point indices to a 68-point compatible subset.
# Selected to cover jaw, eyebrows, nose, eyes, and mouth regions.
_IF106_TO_68 = [
    # Jaw contour — 17 points (indices 0-32, every 2nd)
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    # Right eyebrow — 5 points
    33, 34, 35, 36, 37,
    # Left eyebrow — 5 points
    42, 43, 44, 45, 46,
    # Nose bridge — 4 points
    51, 52, 53, 54,
    # Nose bottom — 5 points
    57, 58, 59, 60, 61,
    # Right eye — 6 points
    66, 67, 68, 69, 70, 71,
    # Left eye — 6 points
    75, 76, 77, 78, 79, 80,
    # Outer mouth — 12 points
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    # Inner mouth — 8 points
    96, 97, 98, 99, 100, 101, 102, 103,
]


class LandmarkExtractor:
    """
    Extract 68-point landmarks from an aligned face image
    using InsightFace's 106-point detector.
    """

    def __init__(self, aligner: "FaceAligner"):
        self.aligner = aligner

    def extract_landmarks(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Returns:
            pts: [68, 2] float32 (x, y) pixel coordinates
        """
        faces = self.aligner.app.get(image_bgr)
        if not faces:
            raise ValueError("No face found for landmark extraction.")
        face = max(faces, key=lambda f: f.det_score)
        pts_106 = face.landmark_2d_106  # [106, 2]
        pts_68 = pts_106[_IF106_TO_68]  # [68, 2]
        return pts_68.astype(np.float32)


class DelaunayMorpher:
    """
    Compute Delaunay triangulation on averaged landmark positions
    and produce triangle index triples for morphing.
    """

    # 8 border anchors as normalised (x, y) fractions of image size
    _BORDER = [
        (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
        (0.0, 0.5),              (1.0, 0.5),
        (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
    ]

    def compute_triangulation(
        self,
        pts_src: np.ndarray,    # [N, 2]
        pts_dst: np.ndarray,    # [N, 2]
        image_size: tuple[int, int],  # (H, W)
    ) -> list[tuple[int, int, int]]:
        """
        Returns a list of (i, j, k) index triples into the combined
        point array [pts_mid | border_pts].
        """
        H, W = image_size
        border = np.array(
            [[int(x * (W - 1)), int(y * (H - 1))] for x, y in self._BORDER],
            dtype=np.float32,
        )
        pts_mid = ((pts_src + pts_dst) / 2.0).astype(np.float32)
        pts_all = np.vstack([pts_mid, border])  # [N+8, 2]

        rect = (0, 0, W, H)
        subdiv = cv2.Subdiv2D(rect)
        for pt in pts_all:
            x, y = float(pt[0]), float(pt[1])
            # Clamp to rect interior to avoid cv2 exception
            x = max(0.5, min(W - 1.5, x))
            y = max(0.5, min(H - 1.5, y))
            subdiv.insert((x, y))

        tri_list = subdiv.getTriangleList()  # [M, 6] float32
        n = len(pts_all)

        def find_idx(px: float, py: float) -> int | None:
            for i in range(n):
                if abs(pts_all[i, 0] - px) < 1.5 and abs(pts_all[i, 1] - py) < 1.5:
                    return i
            return None

        triangles: list[tuple[int, int, int]] = []
        for tri in tri_list:
            idxs = []
            ok = True
            for ci in range(3):
                idx = find_idx(tri[ci * 2], tri[ci * 2 + 1])
                if idx is None:
                    ok = False
                    break
                idxs.append(idx)
            if ok and len(idxs) == 3:
                triangles.append(tuple(idxs))  # type: ignore[arg-type]

        return triangles
