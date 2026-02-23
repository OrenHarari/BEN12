from __future__ import annotations

import cv2
import numpy as np


class TriangleWarper:
    """
    Landmark-based face morphing using Delaunay triangle affine warps.

    For each frame at interpolation factor alpha in [0, 1]:
      1. Compute midpoint landmark positions: pts_mid = lerp(pts_src, pts_dst, alpha)
      2. For each triangle:
         - Warp img_src toward pts_mid  -> contribution_src
         - Warp img_dst toward pts_mid  -> contribution_dst
      3. Cross-dissolve: result = (1-alpha)*contribution_src + alpha*contribution_dst
    """

    @staticmethod
    def _build_border_pts(pts_src: np.ndarray, pts_dst: np.ndarray, H: int, W: int):
        border_norm = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5),              (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]
        border_src = np.array([[x * (W - 1), y * (H - 1)] for x, y in border_norm], dtype=np.float32)
        border_dst = border_src.copy()
        pts_all_src = np.vstack([pts_src, border_src])
        pts_all_dst = np.vstack([pts_dst, border_dst])
        return pts_all_src, pts_all_dst

    @staticmethod
    def _warp_tri_into(
        src_img: np.ndarray,
        dst_buf: np.ndarray,
        src_tri: np.ndarray,   # [3, 2] float32
        dst_tri: np.ndarray,   # [3, 2] float32
    ) -> None:
        """Affine-warp one triangle from src_img into dst_buf in-place."""
        # Ensure proper data types and shape
        src_tri = np.asarray(src_tri, dtype=np.float32).reshape(3, 2)
        dst_tri = np.asarray(dst_tri, dtype=np.float32).reshape(3, 2)
        
        r_dst = cv2.boundingRect(dst_tri.astype(np.int32))
        r_src = cv2.boundingRect(src_tri.astype(np.int32))

        dst_tri_local = dst_tri - np.array([r_dst[0], r_dst[1]], dtype=np.float32)
        src_tri_local = src_tri - np.array([r_src[0], r_src[1]], dtype=np.float32)

        # Create triangle mask in dst local space
        mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, dst_tri_local.astype(np.int32), 1.0)

        # Clip src patch
        src_patch = src_img[
            max(0, r_src[1]):r_src[1] + r_src[3],
            max(0, r_src[0]):r_src[0] + r_src[2],
        ]
        if src_patch.size == 0:
            return

        # getAffineTransform requires exactly (3,2) float32 arrays
        M = cv2.getAffineTransform(src_tri_local.astype(np.float32), dst_tri_local.astype(np.float32))
        warped = cv2.warpAffine(
            src_patch,
            M,
            (r_dst[2], r_dst[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Write into dst_buf using mask
        y1 = max(0, r_dst[1])
        y2 = min(dst_buf.shape[0], r_dst[1] + r_dst[3])
        x1 = max(0, r_dst[0])
        x2 = min(dst_buf.shape[1], r_dst[0] + r_dst[2])

        mask_crop = mask[: y2 - y1, : x2 - x1, np.newaxis]
        warped_crop = warped[: y2 - y1, : x2 - x1]
        dst_buf[y1:y2, x1:x2] = (
            dst_buf[y1:y2, x1:x2] * (1.0 - mask_crop) + warped_crop * mask_crop
        )

    @classmethod
    def morph_frame(
        cls,
        img_src: np.ndarray,          # uint8 BGR [H, W, 3]
        img_dst: np.ndarray,          # uint8 BGR [H, W, 3]
        pts_src: np.ndarray,          # [N, 2] float32
        pts_dst: np.ndarray,          # [N, 2] float32
        triangles: list[tuple[int, int, int]],
        alpha: float,                 # 0.0 = src, 1.0 = dst
    ) -> np.ndarray:
        """
        Produce a morphed frame at blend ratio alpha.
        """
        H, W = img_src.shape[:2]
        pts_all_src, pts_all_dst = cls._build_border_pts(pts_src, pts_dst, H, W)
        pts_all_mid = (1.0 - alpha) * pts_all_src + alpha * pts_all_dst

        img_src_f = img_src.astype(np.float32)
        img_dst_f = img_dst.astype(np.float32)

        warped_src = np.zeros_like(img_src_f)
        warped_dst = np.zeros_like(img_dst_f)

        for i, j, k in triangles:
            tri_src = pts_all_src[[i, j, k]]
            tri_dst = pts_all_dst[[i, j, k]]
            tri_mid = pts_all_mid[[i, j, k]]

            cls._warp_tri_into(img_src_f, warped_src, tri_src, tri_mid)
            cls._warp_tri_into(img_dst_f, warped_dst, tri_dst, tri_mid)

        morphed = (1.0 - alpha) * warped_src + alpha * warped_dst
        return np.clip(morphed, 0, 255).astype(np.uint8)
