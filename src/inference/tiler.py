"""
=============================================================================
BharatSRM-Net v4: Overlapping Tile Inference Engine with 2D Hanning Window
=============================================================================
Section 7.2 Specification:
  - Tiles whole scenes into overlapping 256x256 LR patches with stride margins.
  - Reconstructs at 4x scale (1024x1024 HR patches).
  - Blends overlapping predictions with 2D Hanning window to eliminate boundary seam artifacts.
=============================================================================
"""


import numpy as np
import torch


def create_2d_hanning_window(height: int, width: int) -> np.ndarray:
    """Constructs a 2D Hanning window of shape (height, width) for smooth spatial blending."""
    h_win = np.hanning(height)
    w_win = np.hanning(width)
    window = np.outer(h_win, w_win)
    # Ensure non-zero edges for numeric stability
    window = np.clip(window, 1e-4, 1.0)
    return window.astype(np.float32)


class TiledInferenceEngine:
    """Overlapping patch tiler and seamless 2D Hanning reconstructor."""

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 64,
        scale_factor: int = 4,
        device: str = "cpu",
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.scale_factor = scale_factor
        self.hr_tile_size = tile_size * scale_factor
        self.device = torch.device(device)

        # Precompute 2D Hanning window for HR tiles: (1, 1, 1024, 1024)
        self.hanning_hr = torch.from_numpy(
            create_2d_hanning_window(self.hr_tile_size, self.hr_tile_size)
        ).unsqueeze(0).unsqueeze(0).to(self.device)

    def predict_large_scene(
        self,
        model: torch.nn.Module,
        scene_lr: torch.Tensor,
        scene_mask: torch.Tensor | None = None,
        scene_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Runs seamless super-resolution inference over large satellite scenes.

        Args:
            model: BharatSRMNetV4 model
            scene_lr: (1, 10, H, W) Full scene Sentinel-2 tensor
            scene_mask: (1, 1, H, W) Full scene validity mask (Optional)
            scene_dem: (1, 2, H, W) Full scene CartoDEM tensor (Optional)

        Returns:
            Dict containing:
                - 'sr_image': (1, 4, 4H, 4W) Blended HR image
                - 'variance': (1, 4, 4H, 4W) Blended uncertainty variance map
        """
        model.eval()
        _, _, H, W = scene_lr.shape
        out_H, out_W = H * self.scale_factor, W * self.scale_factor

        # Accumulators
        hr_image_accum = torch.zeros(1, 4, out_H, out_W, device=self.device)
        hr_var_accum = torch.zeros(1, 4, out_H, out_W, device=self.device)
        weight_accum = torch.zeros(1, 1, out_H, out_W, device=self.device)

        if scene_mask is None:
            scene_mask = torch.ones(1, 1, H, W, device=self.device)
        if scene_dem is None:
            scene_dem = torch.zeros(1, 2, H, W, device=self.device)

        y_starts = list(range(0, max(1, H - self.tile_size + 1), self.stride))
        if y_starts[-1] + self.tile_size < H:
            y_starts.append(H - self.tile_size)

        x_starts = list(range(0, max(1, W - self.tile_size + 1), self.stride))
        if x_starts[-1] + self.tile_size < W:
            x_starts.append(W - self.tile_size)

        with torch.no_grad():
            for ys in y_starts:
                for xs in x_starts:
                    ye = min(ys + self.tile_size, H)
                    xe = min(xs + self.tile_size, W)

                    # Extract patch
                    patch_lr = scene_lr[:, :, ys:ye, xs:xe].to(self.device)
                    patch_mask = scene_mask[:, :, ys:ye, xs:xe].to(self.device)
                    patch_dem = scene_dem[:, :, ys:ye, xs:xe].to(self.device)

                    # Model forward
                    out = model(patch_lr, patch_mask, patch_dem)
                    sr_patch = out["sr_image"]  # (1, 4, 4*th, 4*tw)
                    var_patch = out["variance"]  # (1, 4, 4*th, 4*tw)

                    # HR spatial coordinates
                    out_ys, out_ye = ys * self.scale_factor, ye * self.scale_factor
                    out_xs, out_xe = xs * self.scale_factor, xe * self.scale_factor

                    # Blend with Hanning window
                    cur_h = out_ye - out_ys
                    cur_w = out_xe - out_xs
                    if cur_h == self.hr_tile_size and cur_w == self.hr_tile_size:
                        win = self.hanning_hr
                    else:
                        win = torch.from_numpy(
                            create_2d_hanning_window(cur_h, cur_w)
                        ).unsqueeze(0).unsqueeze(0).to(self.device)

                    hr_image_accum[:, :, out_ys:out_ye, out_xs:out_xe] += sr_patch * win
                    hr_var_accum[:, :, out_ys:out_ye, out_xs:out_xe] += var_patch * win
                    weight_accum[:, :, out_ys:out_ye, out_xs:out_xe] += win

        # Normalize by accumulated weights
        weight_safe = torch.clamp(weight_accum, min=1e-5)
        final_sr = hr_image_accum / weight_safe
        final_var = hr_var_accum / weight_safe

        return {
            "sr_image": torch.clamp(final_sr, 0.0, 1.0),
            "variance": final_var,
            "std": torch.sqrt(final_var),
        }
