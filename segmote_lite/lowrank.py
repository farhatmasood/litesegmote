"""Truncated-SVD factorisation of trained layers.

A trained weight ``W`` of shape (out, in) is replaced by the product ``B A`` with
``A`` of shape (rank, in) and ``B`` of shape (out, rank), both initialised from the
truncated singular value decomposition of ``W``. Parameter count falls from
``out * in`` to ``rank * (out + in)``, which is a saving whenever
``rank < out * in / (out + in)``.

The same idea applies to a transposed convolution by unfolding its kernel into a
matrix over the channel axes.
"""
from typing import Tuple

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Two-factor replacement for ``nn.Linear``.

    The forward map is ``up(down(x))`` so the effective weight is
    ``up.weight @ down.weight``. The bias is carried by the second factor.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def effective_weight(self) -> torch.Tensor:
        return self.up.weight.detach() @ self.down.weight.detach()


def factorize_linear(linear: nn.Linear, rank: int) -> LowRankLinear:
    """Approximate ``linear`` by a rank-``rank`` factorisation of its weight.

    With ``W = U S Vh`` the truncation keeps the leading ``rank`` triplets and splits
    the singular values evenly between the two factors, which keeps their scales
    comparable and leaves the composed map equal to the rank-``rank`` approximation
    of ``W``.
    """
    out_f, in_f = linear.weight.shape
    r = int(min(rank, out_f, in_f))
    W = linear.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    root = torch.sqrt(S[:r])
    down_w = root.unsqueeze(1) * Vh[:r]
    up_w = U[:, :r] * root.unsqueeze(0)

    module = LowRankLinear(in_f, out_f, r, bias=linear.bias is not None)
    with torch.no_grad():
        module.down.weight.copy_(down_w.to(linear.weight.dtype))
        module.up.weight.copy_(up_w.to(linear.weight.dtype))
        if linear.bias is not None:
            module.up.bias.copy_(linear.bias.detach())
    return module.to(linear.weight.device)


class LowRankConvTranspose2d(nn.Module):
    """Channel-factorised transposed convolution.

    Spatial upsampling runs on ``rank`` channels, then a pointwise convolution
    expands to the output width. Parameters fall from ``in * out * kh * kw`` to
    ``rank * (in * kh * kw + out)``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride, rank: int, bias: bool = True):
        super().__init__()
        self.up_spatial = nn.ConvTranspose2d(in_ch, rank, kernel_size, stride, bias=False)
        self.mix = nn.Conv2d(rank, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.up_spatial(x))


def factorize_conv_transpose2d(conv: nn.ConvTranspose2d, rank: int) -> LowRankConvTranspose2d:
    """Approximate ``conv`` by unfolding its kernel to a matrix and truncating.

    The weight of shape (in, out, kh, kw) is reshaped to (in * kh * kw, out) before
    decomposition, so the retained subspace is shared across kernel positions.
    """
    in_ch, out_ch, kh, kw = conv.weight.shape
    r = int(min(rank, in_ch * kh * kw, out_ch))
    W = conv.weight.detach().float()
    M = W.permute(0, 2, 3, 1).reshape(in_ch * kh * kw, out_ch)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    root = torch.sqrt(S[:r])
    A = U[:, :r] * root.unsqueeze(0)
    B = Vh[:r] * root.unsqueeze(1)

    module = LowRankConvTranspose2d(
        in_ch, out_ch, (kh, kw), conv.stride, r, bias=conv.bias is not None
    )
    with torch.no_grad():
        up_w = A.reshape(in_ch, kh, kw, r).permute(0, 3, 1, 2).contiguous()
        module.up_spatial.weight.copy_(up_w.to(conv.weight.dtype))
        module.mix.weight.copy_(B.t().reshape(out_ch, r, 1, 1).to(conv.weight.dtype))
        if conv.bias is not None:
            module.mix.bias.copy_(conv.bias.detach())
    return module.to(conv.weight.device)


def relative_error(original: nn.Module, factorised: nn.Module) -> float:
    """Frobenius relative error between a layer weight and its factorised form."""
    if isinstance(factorised, LowRankLinear):
        W = original.weight.detach().float()
        approx = factorised.effective_weight().float()
    else:
        in_ch, out_ch, kh, kw = original.weight.shape
        W = original.weight.detach().float().permute(0, 2, 3, 1).reshape(in_ch * kh * kw, out_ch)
        up = factorised.up_spatial.weight.detach().float()
        r = up.shape[1]
        A = up.permute(0, 2, 3, 1).reshape(in_ch * kh * kw, r)
        B = factorised.mix.weight.detach().float().reshape(out_ch, r).t()
        approx = A @ B
    return (torch.norm(W - approx) / (torch.norm(W) + 1e-12)).item()


def spectrum(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return singular values and the cumulative fraction of squared spectral energy."""
    W = weight.detach().float()
    if W.dim() == 4:
        in_ch, out_ch, kh, kw = W.shape
        W = W.permute(0, 2, 3, 1).reshape(in_ch * kh * kw, out_ch)
    S = torch.linalg.svdvals(W)
    energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    return S, energy
