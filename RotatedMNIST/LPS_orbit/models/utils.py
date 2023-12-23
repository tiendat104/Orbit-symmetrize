import torch
from kornia.geometry.transform import affine
from kornia.core import Tensor
from kornia.utils.misc import eye_like
from typing import Optional, Tuple, Union

def index_to_perm(perms):
    # idx: [B, N]
    device = perms.device
    B, N = perms.shape
    hs = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) # [B, N, N]
    hs = hs.gather(1, perms[...,None].expand(-1,-1,N)) # [B, N, N]
    return hs # [B, N, N]

def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center

def get_rotation_matrix2d(center: Tensor, rot2d_matrix: Tensor, scale: Tensor) -> Tensor:
    # rot2d_matrix: [b, 2, 2]
    shift_m = eye_like(3, center) # [b, 3, 3]
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center) # [b, 3, 3]
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center) # [b, 3, 3]
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center) # [b, 3, 3]
    rotat_m[:, :2, :2] = rot2d_matrix

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    
    return affine_m[:, :2, :]  # [b, 2, 3]

def _compute_rotation_matrix(rot2d_matrix: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Compute a pure affine rotation matrix."""
    # rot2d_matrix: [b, 2, 2]
    scale: torch.Tensor = torch.ones_like(center)
    matrix: torch.Tensor = get_rotation_matrix2d(center, rot2d_matrix, scale) # [b, 2, 3]
    return matrix

def rotate(
    tensor: torch.Tensor,
    rot2d_matrix: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> torch.Tensor:
    b, c, h, w = tensor.shape
    assert rot2d_matrix.shape == (b, 2, 2)
    
    if center is None:
        center = _compute_tensor_center(tensor)
    center = center.expand(tensor.shape[0], -1) # [b, 2]
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(rot2d_matrix, center) # [b, 2, 3]
    
    return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)

def rotate_kornia_API(
    tensor: torch.Tensor,
    rot2d_matrix: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> torch.Tensor:
    b, c, h, w = tensor.shape
    assert rot2d_matrix.shape == (b, 2, 2) 
    rot2d_matrix = rot2d_matrix.transpose(-1, -2) # [b, 2, 2]
    rotated_images = rotate(tensor, rot2d_matrix, center, mode, padding_mode, align_corners)
    return rotated_images  # [b, c, h, w]
    