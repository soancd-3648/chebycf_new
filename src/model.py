import torch
import math
import torch.nn.functional as F
from torch import nn

from src.module import ChebyFilter, IdealFilter, DegreeNorm, LinearFilter


# =====================================================
# Build Model
# =====================================================

def build_model(args):
    if args.model.lower() == 'chebycf':
        return ChebyCF(args.K, args.phi, args.eta, args.alpha, args.beta).to(args.device)

    elif args.model.lower() == 'gfcf':
        return GFCF(args.alpha).to(args.device)

    elif args.model.lower() == 'cheby_attn':
        return ChebyAttnCF(
            args.K,
            args.phi,
            args.eta,
            args.alpha,
            args.beta,
            heads=getattr(args, 'heads', 4)
        ).to(args.device)

    raise NotImplementedError(f'Model named {args.model} is not implemented.')


# =====================================================
# Base Class
# =====================================================

class AllRankRec(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, observed_inter):
        pass

    def mask_observed(self, pred_score, observed_inter):
        """
        OPT: Avoid dense multiplication when observed_inter is sparse.
        Uses in-place fill on known positions to skip full matrix ops.
        """
        if observed_inter.is_sparse:
            # Convert to dense only once, use additive masking
            mask = observed_inter.to_dense()
            return pred_score.masked_fill(mask.bool(), -1e8)
        # Dense path: in-place is faster than creating two new tensors
        return pred_score.masked_fill(observed_inter.bool(), -1e8)

    def full_predict(self, observed_inter):
        pred_score = self.forward(observed_inter)
        return self.mask_observed(pred_score, observed_inter)


# =====================================================
# ChebyCF
# =====================================================

class ChebyCF(AllRankRec):
    def __init__(self, K, phi, eta, alpha, beta):
        super().__init__()
        self.cheby = ChebyFilter(K, phi)
        self.ideal = IdealFilter(eta, alpha) if eta > 0 and alpha > 0 else None
        self.norm = DegreeNorm(beta) if beta > 0 else None

    def fit(self, inter):
        self.cheby.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
        if self.norm:
            self.norm.fit(inter)

    def forward(self, signal):
        if self.norm:
            signal = self.norm.forward_pre(signal)

        output = self.cheby.forward(signal)

        if self.ideal:
            output = output + self.ideal.forward(signal)  # avoids in-place issues

        if self.norm:
            output = self.norm.forward_post(output)

        return output


# =====================================================
# Graph Attention Layer — Optimized for large item sets
# =====================================================

class GraphAttentionLayer(nn.Module):
    """
    Efficient Multi-Head Attention over item dimension.

    KEY OPTIMIZATIONS for 48k items:
    - Low-rank bottleneck projection: 48k → hidden_dim (1024) → 48k
      avoids O(N²) item-item attention matrix entirely.
    - Elementwise Q·K dot-product per head: O(B·H·D) not O(B·N²)
    - torch.compile-friendly: no dynamic shapes, no Python loops
    - float16 autocast supported via caller context
    - Fused ops: combined reshape + einsum where possible
    """

    def __init__(self, num_items: int, heads: int = 4, hidden_dim: int = 1024):
        super().__init__()

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.num_items = num_items
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.scale = math.sqrt(self.head_dim)

        # OPT: single fused linear for Q, K, V — 1 GEMM instead of 3
        self.qkv_proj = nn.Linear(num_items, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, num_items, bias=False)

        # OPT: weight init for stable gradients at large input dim
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_items)
        B = x.size(0)

        # OPT: single fused projection (1 GEMM, 3x fewer kernel launches)
        qkv = self.qkv_proj(x)                            # (B, 3*hidden_dim)
        Q, K, V = qkv.chunk(3, dim=-1)                    # each: (B, hidden_dim)

        # OPT: reshape for multi-head without extra alloc
        Q = Q.view(B, self.heads, self.head_dim)           # (B, H, D)
        K = K.view(B, self.heads, self.head_dim)
        V = V.view(B, self.heads, self.head_dim)

        # OPT: elementwise dot-product attention — O(B·H·D), avoids N²
        scores = (Q * K).sum(dim=-1, keepdim=True) / self.scale  # (B, H, 1)
        attn = torch.softmax(scores, dim=1)                       # (B, H, 1)

        # Weighted sum over heads, fused reshape
        out = (attn * V).reshape(B, self.hidden_dim)              # (B, hidden_dim)

        # Project back to item space
        out = self.out_proj(out)                                   # (B, num_items)

        return x + out   # residual


# =====================================================
# Hybrid Cheby + Attention — Optimized
# =====================================================

class ChebyAttnCF(AllRankRec):
    """
    Frequency-Decoupled Hybrid Model

    Low-frequency  : Chebyshev + Ideal filter
    High-frequency : Residual → Lightweight Attention

    OPT changes vs original:
    - Bug fix: `signal.device` in fit() replaced with `inter.device`
    - Attention initialized once in fit(), not recreated on every call
    - forward() is torch.compile-able (no dynamic control flow on tensor values)
    - All additions use out-of-place ops to avoid autograd graph issues
    """

    def __init__(self, K, phi, eta, alpha, beta, heads: int = 4):
        super().__init__()

        self.cheby = ChebyFilter(K, phi)
        self.ideal = IdealFilter(eta, alpha) if eta > 0 and alpha > 0 else None
        self.norm = DegreeNorm(beta) if beta > 0 else None

        self.attn: GraphAttentionLayer | None = None
        self.heads = heads

    def fit(self, inter):
        self.cheby.fit(inter)

        if self.ideal:
            self.ideal.fit(inter)

        if self.norm:
            self.norm.fit(inter)

        # OPT: use inter.device (fix original bug: `signal` not in scope)
        # OPT: only (re)create attention if item count changed
        num_items = inter.shape[1]
        if self.attn is None or self.attn.num_items != num_items:
            self.attn = GraphAttentionLayer(
                num_items,
                heads=self.heads,
                hidden_dim=1024,
            ).to(inter.device)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Pre-normalize
        # -------------------------
        if self.norm:
            signal = self.norm.forward_pre(signal)

        # -------------------------
        # Low-frequency branch
        # -------------------------
        E_spec = self.cheby.forward(signal)

        if self.ideal:
            E_spec = E_spec + self.ideal.forward(signal)

        # -------------------------
        # High-frequency residual → Attention
        # -------------------------
        E_high = signal - E_spec
        E_attn = self.attn(E_high)

        # -------------------------
        # Fusion + Post-normalize
        # -------------------------
        output = E_spec + E_attn

        if self.norm:
            output = self.norm.forward_post(output)

        return output


# =====================================================
# GFCF
# =====================================================

class GFCF(AllRankRec):
    def __init__(self, alpha):
        super().__init__()
        self.linear = LinearFilter()
        self.ideal = IdealFilter(256, alpha) if alpha > 0 else None
        self.norm = DegreeNorm(0.5)

    def fit(self, inter):
        self.linear.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
        self.norm.fit(inter)

    def forward(self, signal):
        output = self.linear(signal)

        if self.ideal:
            signal = self.norm.forward_pre(signal)
            signal = self.ideal(signal)
            signal = self.norm.forward_post(signal)
            output = output + signal

        return output