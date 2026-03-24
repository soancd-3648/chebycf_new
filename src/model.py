import torch
import math
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
    Multi-head attention across item dimension.
    Applied on high-frequency residual.
    """

    def __init__(self, num_items, heads=4):
        super().__init__()

        self.num_items = num_items
        self.heads = heads

        self.q_proj = nn.Linear(num_items, num_items)
        self.k_proj = nn.Linear(num_items, num_items)
        self.v_proj = nn.Linear(num_items, num_items)
        self.out_proj = nn.Linear(num_items, num_items)

        self.layernorm = nn.LayerNorm(num_items)

    def forward(self, x):
        # x: (batch_size, num_items)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Compute dense attention over the item dimension.
        # Q/K: (batch, num_items) → scores: (batch, num_items, num_items)
        scores = torch.matmul(Q.unsqueeze(2), K.unsqueeze(1)) / math.sqrt(self.num_items)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V.unsqueeze(-1)).squeeze(-1)

        out = self.out_proj(out)

        return self.layernorm(x + out)


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
        self.attn = GraphAttentionLayer(num_items, self.heads)

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