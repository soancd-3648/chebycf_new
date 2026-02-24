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
        return pred_score * (1 - observed_inter) - 1e8 * observed_inter

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
            output += self.ideal.forward(signal)

        if self.norm:
            output = self.norm.forward_post(output)

        return output


# =====================================================
# Graph Attention Layer (Dense version)
# =====================================================

class GraphAttentionLayer(nn.Module):
    """
    Efficient Multi-Head Attention over item dimension
    Reduced hidden dimension for speed & memory efficiency.
    """

    def __init__(self, num_items, heads=4, hidden_dim=1024):
        super().__init__()

        self.num_items = num_items
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        # Project to lower dimension
        self.q_proj = nn.Linear(num_items, hidden_dim, bias=False)
        self.k_proj = nn.Linear(num_items, hidden_dim, bias=False)
        self.v_proj = nn.Linear(num_items, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, num_items, bias=False)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        # x: (batch_size, num_items)

        B = x.size(0)

        # Project
        Q = self.q_proj(x)  # (B, hidden_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads
        Q = Q.view(B, self.heads, self.head_dim)
        K = K.view(B, self.heads, self.head_dim)
        V = V.view(B, self.heads, self.head_dim)

        # Attention score (scaled dot-product)
        scores = (Q * K).sum(dim=-1) / self.scale   # (B, heads)
        attn = torch.softmax(scores, dim=-1)        # (B, heads)

        # Apply attention
        out = (attn.unsqueeze(-1) * V).reshape(B, self.hidden_dim)

        # Project back to item space
        out = self.out_proj(out)

        return x + out


# =====================================================
# Hybrid Cheby + Attention
# =====================================================

class ChebyAttnCF(AllRankRec):
    """
    Frequency-Decoupled Hybrid Model

    Low-frequency  : Chebyshev + Ideal
    High-frequency : Residual → Attention
    """

    def __init__(self, K, phi, eta, alpha, beta, heads=4):
        super().__init__()

        self.cheby = ChebyFilter(K, phi)
        self.ideal = IdealFilter(eta, alpha) if eta > 0 and alpha > 0 else None
        self.norm = DegreeNorm(beta) if beta > 0 else None

        self.attn = None
        self.heads = heads

    def fit(self, inter):
        self.cheby.fit(inter)

        if self.ideal:
            self.ideal.fit(inter)

        if self.norm:
            self.norm.fit(inter)

        # Initialize attention after knowing item size
        num_items = inter.shape[1]
        self.attn = GraphAttentionLayer(
        num_items,
        heads=self.heads,
        hidden_dim=1024   # có thể thử 512 nếu vẫn chậm
        ).to(signal.device)

    def forward(self, signal):

        original_signal = signal

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
            E_spec += self.ideal.forward(signal)

        # -------------------------
        # High-frequency residual
        # -------------------------
        E_high = signal - E_spec

        # -------------------------
        # Attention on high-frequency
        # -------------------------
        E_attn = self.attn(E_high)

        # -------------------------
        # Fusion
        # -------------------------
        output = E_spec + E_attn

        # -------------------------
        # Post-normalize
        # -------------------------
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
            output += signal

        return output