import math

from einops import einsum, rearrange, reduce
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        I: in_features
        O: out_features
        """
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.w_OI = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                0, std ** 2, -3 * std, 3 * std
            )
        )

    def forward(self, x_BsI: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        I: in_features
        """
        x_BsO = einsum(x_BsI, self.w_OI, "... I, O I -> ... O")
        return x_BsO


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        E: vocab_size / number of embeddings
        H: hidden_dim
        """
        super().__init__()
        self.embeddings_EH = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                0, 1, -3, 3
            )
        )

    def forward(self, x_Bs: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        E: vocab_size / number of embeddings
        H: hidden_dim
        """
        x_BsH = self.embeddings_EH[x_Bs]
        return x_BsH


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
        H: hidden_dim / d_model
        """
        super().__init__()
        self.eps = eps
        self.g_H = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x_BsH: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        H: hidden_dim / d_model
        """
        in_dtype = x_BsH.dtype
        x_BsH = x_BsH.to(torch.float32) # upcast for stability
        rms_Bs = torch.sqrt(reduce(x_BsH ** 2, "... H -> ...", 'mean') + self.eps)
        result_BsH = (x_BsH / rms_Bs[..., None]) * self.g_H
        return result_BsH.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x_BsH: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        H: hidden_dim
        F: feed forward dim
        """
        w1x_BsF = self.linear1(x_BsH)
        w3x_BsF = self.linear3(x_BsH)
        silu_BsF = w1x_BsF * torch.sigmoid(w1x_BsF)
        out_BsH = self.linear2(silu_BsF * w3x_BsF)
        return out_BsH


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device = None
    ):
        """
        L: max_seq_len
        K: d_k
        """
        super().__init__()
        positions_L = torch.arange(max_seq_len, device=device)
        k_idxs_K = torch.arange(1, d_k // 2 + 1, device=device) # formula is sensitive to range of k (0-idx or 1-idx)
        thetas_LK = positions_L[..., None] / torch.pow(theta, (2 * k_idxs_K - 2) / d_k)
        self.register_buffer("cos_LK", torch.cos(thetas_LK), persistent=False)
        self.register_buffer("sin_LK", torch.sin(thetas_LK), persistent=False)

    def forward(
        self,
        x_BsLK: torch.Tensor,
        token_positions_BsL: torch.Tensor
    ) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        L: seq_len
        K: d_k
        """
        x0_BsLK = x_BsLK[..., ::2]
        x1_BsLK = x_BsLK[..., 1::2]
        cos_BsLK = self.cos_LK[token_positions_BsL]
        sin_BsLK = self.sin_LK[token_positions_BsL]
        out0_BsLK = cos_BsLK * x0_BsLK - sin_BsLK * x1_BsLK
        out1_BsLK = sin_BsLK * x0_BsLK + cos_BsLK * x1_BsLK
        out_BsLK = rearrange([out0_BsLK, out1_BsLK], "two ... L K -> ... L (K two)")
        return out_BsLK


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    # logsumexp trick
    c = x.max(i, keepdim=True)[0]
    x_small = x - c

    exp_x = torch.exp(x_small)
    sum_exp_x = exp_x.sum(i, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    q_BsLqK: torch.Tensor,
    k_BsLkK: torch.Tensor,
    v_BsLkV: torch.Tensor,
    attention_mask_LqLk: torch.Tensor | None = None
):
    """
    Bs: 0 or more batch dimensions
    Lq: seq_len (queries)
    Lk: seq_len (keys)
    K: d_k, dimension for key/query
    V: d_v, dimension for values
    """
    d_k = q_BsLqK.shape[-1]
    qk_BsLqLk = einsum(q_BsLqK, k_BsLkK, "... Lq K, ... Lk K -> ... Lq Lk")
    if attention_mask_LqLk is not None:
        qk_BsLqLk[~attention_mask_LqLk] = -torch.inf
    attention_weight_BsLqLk = softmax(qk_BsLqLk / math.sqrt(d_k), -1)
    out_BsLqV = einsum(attention_weight_BsLqLk, v_BsLkV, "... Lq Lk, ... Lk V -> ... Lq V")
    return out_BsLqV


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x_BsLH: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        L: seq_len
        H: hidden_dim
        D: head dim
        """
        q_BsLH
        pass
