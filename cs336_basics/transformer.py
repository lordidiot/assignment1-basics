from contextlib import nullcontext
import math
import os
from typing import Any, Callable, Optional
import typing

from einops import einsum, rearrange, reduce
import numpy as np
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
        # https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.masked_fill_.html
        qk_BsLqLk = qk_BsLqLk.masked_fill(~attention_mask_LqLk, -torch.inf)
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
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x_BsLH: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        L: seq_len
        H: hidden_dim
        D: heads
        E: head feature dim
        """
        seq_len = x_BsLH.shape[-2]
        q_BsLH = self.q_proj(x_BsLH)
        k_BsLH = self.k_proj(x_BsLH)
        v_BsLH = self.v_proj(x_BsLH)
        q_BsDLE = rearrange(q_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)
        k_BsDLE = rearrange(k_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)
        v_BsDLE = rearrange(v_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)
        attention_mask_LL = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        mixed_BsDLE = scaled_dot_product_attention(q_BsDLE, k_BsDLE, v_BsDLE, attention_mask_LL)
        mixed_BsLH = rearrange(mixed_BsDLE, "... D L E -> ... L (D E)")
        out_BsLH = self.o_proj(mixed_BsLH)
        return out_BsLH


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.register_buffer(
            "attention_mask_LL",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)),
            persistent=False
        )

    def forward(self, x_BsLH: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        L: seq_len
        H: hidden_dim
        D: heads
        E: head feature dim
        """
        seq_len = x_BsLH.shape[-2]
        q_BsLH = self.q_proj(x_BsLH)
        k_BsLH = self.k_proj(x_BsLH)
        v_BsLH = self.v_proj(x_BsLH)

        q_BsDLE = rearrange(q_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)
        k_BsDLE = rearrange(k_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)
        v_BsDLE = rearrange(v_BsLH, "... L (D E) -> ... D L E", D=self.num_heads)

        # rope (only q and k, NOT v)
        q_BsDLE = self.rope(q_BsDLE, token_positions)
        k_BsDLE = self.rope(k_BsDLE, token_positions)

        attention_mask_LL = self.attention_mask_LL[:seq_len, :seq_len]
        mixed_BsDLE = scaled_dot_product_attention(q_BsDLE, k_BsDLE, v_BsDLE, attention_mask_LL)
        mixed_BsLH = rearrange(mixed_BsDLE, "... D L E -> ... L (D E)")
        out_BsLH = self.o_proj(mixed_BsLH)
        return out_BsLH


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.device = device
        self.attention = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x_BsLH: torch.Tensor) -> torch.Tensor:
        """
        Bs: 0 or more batch dimensions
        L: seq_len
        H: hidden_dim
        """
        seq_len = x_BsLH.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        x_BsLH = x_BsLH + self.attention(self.rmsnorm1(x_BsLH), token_positions)
        x_BsLH = x_BsLH + self.ff(self.rmsnorm2(x_BsLH))
        return x_BsLH


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x_BsL: torch.Tensor) -> torch.Tensor:
        x_BsLH = self.token_embeddings(x_BsL)
        x_BsLH = self.layers(x_BsLH)
        x_BsLH = self.rmsnorm(x_BsLH)
        logits_BsLV = self.lm_head(x_BsLH)
        return logits_BsLV


def has_nan(x: torch.Tensor) -> bool:
    return torch.any(x != x)


def cross_entropy_loss(logits_BsV: torch.Tensor, targets_Bs: torch.Tensor) -> torch.Tensor:
    """
    Bs: 0 or more batch dimensions
    E: vocab_size / number of embeddings
    """
    correct_Bs = logits_BsV.gather(-1, targets_Bs.unsqueeze(-1)).squeeze(-1)
    max_logit_Bs1 = reduce(logits_BsV, "... V -> ... 1", "max")
    exp_BsV = torch.exp(logits_BsV - max_logit_Bs1)
    sum_logit_Bs = reduce(exp_BsV, "... V -> ...", "sum")
    out_Bs = - correct_Bs + max_logit_Bs1.squeeze(-1) + torch.log(sum_logit_Bs)
    out_ = reduce(out_Bs, "... -> ", "mean")
    return out_


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr, weight_decay, betas, eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state: dict[str, Any] = self.state[p]
                    t = state.get("t", 1)
                    m = state["m"] if "m" in state else torch.zeros_like(p)
                    v = state["v"] if "v" in state else torch.zeros_like(p)
                    lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                    p.data -= lr * weight_decay * p.data
                    m = betas[0] * m + (1 - betas[0]) * p.grad
                    v = betas[1] * v + (1 - betas[1]) * (p.grad ** 2)
                    p.data -= lr_t * m / (torch.sqrt(v) + eps)

                    state["t"] = t + 1
                    state["m"] = m
                    state["v"] = v
        return loss


def get_lr_cosine_schedule(t: float, a_max: float, a_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return t / t_w * a_max
    elif t <= t_c:
        return a_min + 0.5 * (1 + math.cos((t-t_w) / (t_c-t_w) * math.pi)) * (a_max - a_min)
    else: # t > t_c
        return a_min


def gradient_norm(parameters: list[nn.Parameter]):
    trainable = [p for p in parameters if p.grad is not None]
    square_sum = 0
    for p in trainable:
        square_sum += reduce(p.grad ** 2, "... -> ", "sum")
    norm = torch.sqrt(square_sum)
    return norm


def gradient_clipping(parameters: list[nn.Parameter], m: float, eps: float = 1e-6):
    trainable = [p for p in parameters if p.grad is not None]
    norm = gradient_norm(trainable)
    scale = (m / (norm + eps)).clamp(max=1.0) # branchless clipping
    for p in trainable:
        p.grad.mul_(scale)
    return norm


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    if len(x) < context_length + 1:
        raise ValueError(f"Sequence too short ({len(x)=}) for {context_length=}")
    idxs = np.random.randint(0, len(x)-context_length, (batch_size,1))
    idxs = idxs + np.arange(context_length)
    tensor_x = torch.from_numpy(x[idxs]).pin_memory().to(device, non_blocking=True).to(torch.long)
    tensor_y = torch.from_numpy(x[idxs+1]).pin_memory().to(device, non_blocking=True).to(torch.long)
    return tensor_x, tensor_y


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    if isinstance(out, (str, os.PathLike)):
        ctx_mgr = open(out, "wb")
    else:
        ctx_mgr = nullcontext(out)

    with ctx_mgr as out_f:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
        }
        torch.save(checkpoint, out_f)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
):
    if isinstance(src, (str, os.PathLike)):
        ctx_mgr = open(src, "rb")
    else:
        ctx_mgr = nullcontext(src)

    with ctx_mgr as src_f:
        checkpoint = torch.load(src_f)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']
