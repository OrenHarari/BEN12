from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IFBlock(nn.Module):
    """
    Single coarse-to-fine block of IFNet (RIFE v4.6).
    Estimates optical flow residual + soft blend mask at one scale.
    """

    def __init__(self, in_planes: int, c: int = 90):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv_prelu(in_planes, c, 3, 2, 1),
            _conv_prelu(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            *[_conv_prelu(2 * c, 2 * c) for _ in range(6)]
        )
        self.lastconv = nn.ConvTranspose2d(2 * c, 5, 4, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow_scaled = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) * (1.0 / scale)
            x = torch.cat((x, flow_scaled), dim=1)

        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        out = self.lastconv(feat)
        out = F.interpolate(out, scale_factor=scale * 2, mode="bilinear", align_corners=False)

        flow_out = out[:, :4] * scale * 2
        mask_out = out[:, 4:5]
        return flow_out, mask_out


class IFNet(nn.Module):
    """
    RIFE IFNet v4.6 — Intermediate Flow Estimation Network.

    Given two frames I0 and I1 (concatenated along channel dim),
    produces an intermediate frame at timestep t ∈ [0, 1].

    Input:  x [B, 6, H, W]   (I0 RGB + I1 RGB, values in [0, 1])
    Output: [B, 3, H, W]     intermediate frame in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)   # first block: no prior flow
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[float] | None = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = [8.0, 4.0, 2.0, 1.0]

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None

        t_tensor = torch.zeros_like(img0[:, :1]).fill_(timestep)

        for idx, block in enumerate([self.block0, self.block1, self.block2]):
            scale = scale_list[idx]
            if flow is None:
                inp = torch.cat([img0, img1, t_tensor], dim=1)  # [B, 7, H, W]
                flow, mask = block(inp, None, scale)
            else:
                w0 = _backward_warp(img0, flow[:, :2])
                w1 = _backward_warp(img1, flow[:, 2:4])
                inp = torch.cat([img0, img1, w0, w1, mask, t_tensor], dim=1)  # [B, 11, H, W] → subset
                # Re-slice to expected 7+4 channels: use img0, img1, mask, t as context
                inp = torch.cat([img0, img1, t_tensor, mask], dim=1)  # [B, 7, H, W]
                d_flow, d_mask = block(inp, flow, scale)
                flow = flow + d_flow
                mask = mask + d_mask

        w0 = _backward_warp(img0, flow[:, :2])
        w1 = _backward_warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1.0 - m)


def _conv_prelu(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p), nn.PReLU(out_c))


def _backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp img using 2-channel optical flow."""
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
    new_grid = grid + flow
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    return F.grid_sample(img, new_grid, align_corners=True, mode="bilinear", padding_mode="border")
