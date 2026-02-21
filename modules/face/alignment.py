from __future__ import annotations

import cv2
import numpy as np


class FaceAligner:
    """
    Detect faces with InsightFace RetinaFace, align to canonical 512x512 crop
    using a 5-point affine transform (ArcFace template).
    """

    # ArcFace canonical 5-point template (normalised to [0,1])
    _TEMPLATE_NORM = np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    )

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        import insightface
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def align_face(
        self,
        image_bgr: np.ndarray,
        target_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Returns:
            aligned_bgr: np.ndarray  [target_size, target_size, 3]
            transform_matrix: np.ndarray [2, 3]
            metadata: dict (bbox, kps, landmark_2d_106, det_score)
        """
        faces = self.app.get(image_bgr)
        if not faces:
            raise ValueError("No face detected in the image.")

        face = max(faces, key=lambda f: f.det_score)
        kps = face.kps.astype(np.float32)  # [5, 2]

        dst = self._TEMPLATE_NORM * target_size
        M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.RANSAC)

        aligned = cv2.warpAffine(
            image_bgr,
            M,
            (target_size, target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        meta = {
            "bbox": face.bbox,
            "kps": kps,
            "landmark_2d_106": face.landmark_2d_106,
            "det_score": float(face.det_score),
        }
        return aligned, M, meta
