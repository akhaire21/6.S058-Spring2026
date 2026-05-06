"""
Optical flow caching using RAFT-small.

Per sequence, compute dense flow F_t for t=2..N where F_t maps I_{t-1} -> I_t,
and cache as a single .npz file with array shape (N-1, H, W, 2) downsampled
to a tractable resolution.

We downsample to 240x135 (1/8 of 1920x1080) to keep memory reasonable.
At test time we sample flow at full-res track positions by bilinear interpolation
and rescale to the original image's pixel units.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from PIL import Image


_RAFT_MODEL = None
_RAFT_TRANSFORMS = None


def _get_raft():
    global _RAFT_MODEL, _RAFT_TRANSFORMS
    if _RAFT_MODEL is None:
        weights = Raft_Small_Weights.DEFAULT
        _RAFT_MODEL = raft_small(weights=weights, progress=False).eval().cuda()
        _RAFT_TRANSFORMS = weights.transforms()
    return _RAFT_MODEL, _RAFT_TRANSFORMS


def _load_frame_tensor(path: Path):
    """Load a frame as a [3,H,W] float tensor in [0,255]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t


def _resize_to_multiple_of_8(t: torch.Tensor, target_h: int = 256, target_w: int = 448):
    """RAFT requires H,W divisible by 8. Resize to fixed shape."""
    return TF.resize(t.unsqueeze(0), [target_h, target_w], antialias=True).squeeze(0)


@torch.no_grad()
def cache_optical_flow(sequence, cache_root: str | Path, batch_size: int = 4) -> Path:
    """Compute and cache dense optical flow for a sequence.
    
    Output: <cache_root>/<dataset>/<split>/<seq>.npz with arrays:
        flow:    (N-1, FH, FW, 2)  float16  flow at downsampled resolution
        scale_x: float              ratio orig_w / FW (to scale flow back up)
        scale_y: float              ratio orig_h / FH
    """
    cache_root = Path(cache_root)
    out = cache_root / sequence.dataset / sequence.split / f"{sequence.info.name}.npz"
    if out.exists():
        print(f"  [flow] skip (exists): {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    
    model, transforms = _get_raft()
    H, W = sequence.info.img_height, sequence.info.img_width
    
    # Downsample target: 256x448 (RAFT-friendly, ~1/4 of 1080p in each dim)
    FH, FW = 256, 448
    
    flows = []
    prev = None
    for f_idx, img_path in sequence.iter_frames():
        cur = _resize_to_multiple_of_8(_load_frame_tensor(img_path), FH, FW).cuda()
        if prev is not None:
            # transforms expects [B,3,H,W]
            p, c = transforms(prev.unsqueeze(0), cur.unsqueeze(0))
            flow = model(p, c)[-1]  # final iter, shape [1,2,H,W]
            flows.append(flow.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16))
        prev = cur
    
    flow_arr = np.stack(flows, axis=0) if flows else np.empty((0, FH, FW, 2), dtype=np.float16)
    np.savez_compressed(
        out,
        flow=flow_arr,
        scale_x=W / FW,
        scale_y=H / FH,
        orig_h=H,
        orig_w=W,
    )
    print(f"  [flow] wrote {out.name}: {flow_arr.shape}")
    return out


def load_cached_flow(npz_path: Path):
    """Returns dict with flow, scale_x, scale_y, orig_h, orig_w."""
    z = np.load(npz_path)
    return {
        "flow": z["flow"],          # (N-1, FH, FW, 2) at flow-grid scale (NOT pixel units)
        "scale_x": float(z["scale_x"]),
        "scale_y": float(z["scale_y"]),
        "orig_h": int(z["orig_h"]),
        "orig_w": int(z["orig_w"]),
    }


def sample_flow_at(flow_data: dict, frame_idx: int, cx: float, cy: float, w: float, h: float):
    """Sample mean flow inside a bbox at frame_idx (1-indexed). Returns (dx, dy) in
    ORIGINAL pixel units. frame_idx is the 'destination' frame; flow array index
    is frame_idx - 2 (mapping I_{frame_idx-1} -> I_{frame_idx})."""
    if frame_idx < 2:
        return 0.0, 0.0
    flow_idx = frame_idx - 2
    flow_arr = flow_data["flow"]
    if flow_idx >= flow_arr.shape[0]:
        return 0.0, 0.0
    
    sx, sy = flow_data["scale_x"], flow_data["scale_y"]
    fh, fw = flow_arr.shape[1], flow_arr.shape[2]
    
    # Convert pixel bbox to flow-grid bbox
    fx0 = max(0, int((cx - w/2) / sx))
    fx1 = min(fw, int((cx + w/2) / sx) + 1)
    fy0 = max(0, int((cy - h/2) / sy))
    fy1 = min(fh, int((cy + h/2) / sy) + 1)
    if fx1 <= fx0 or fy1 <= fy0:
        return 0.0, 0.0
    
    patch = flow_arr[flow_idx, fy0:fy1, fx0:fx1]
    if patch.size == 0:
        return 0.0, 0.0
    
    # Flow values from RAFT are in flow-grid pixel units (i.e. distance moved
    # in the downsampled image). Scale up to original pixel units.
    mean_dx = float(patch[..., 0].mean()) * sx
    mean_dy = float(patch[..., 1].mean()) * sy
    return mean_dx, mean_dy
