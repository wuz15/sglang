from typing import List, Optional, Tuple

import torch


def esimd_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, len: int
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_add(a, b, c, len)

def esimd_kernel_uni(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor, t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor,
    i0: int, i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int, i8: int, i9: int, 
    f0: float, f1: float, f2: float, f3: float, f4: float,
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_kernel_uni(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, f0, f1, f2, f3, f4)


def esimd_mul_lgrf(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, flag: int, len: int
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_mul_lgrf(a, b, c, flag, len)