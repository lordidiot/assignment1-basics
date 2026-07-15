from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

import torch
import numpy as np
from tqdm import tqdm
import wandb

from cs336_basics.transformer import AdamW, TransformerLM, cross_entropy_loss, get_batch, get_lr_cosine_schedule, gradient_clipping, gradient_norm


LOG_INTERVAL = 25


class TokenLoader:
    def __init__(
        self,
        tokens_path: Path,
        dtype: np.dtype,
        shape: tuple[int],
        batch_size: int,
        context_length: int,
        device: str,
    ):
        self.tokens = np.memmap(tokens_path, dtype=dtype, shape=shape)
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self) -> tuple[torch.LongTensor, torch.LongTensor]:
        # returns (x, y)
        return get_batch(self.tokens, self.batch_size, self.context_length, self.device)


def main(
    corpus_path: Path,
    lr: float,
    batch_size: int,
    steps: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    theta: int,
    random_seed: int,
):
    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    with open(corpus_path / "meta.json", "r") as meta_f:
        corpus_meta = json.load(meta_f)

    # Experiment configuration
    model_dtype = torch.float32
    config = {
        "dataset": corpus_path.name,
        "lr": lr,
        "min_lr": lr * 0.1,
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": int(steps * 0.08),
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "vocab_size": corpus_meta["vocab_size"],
        "context_length": context_length,
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "theta": theta,
        "random_seed": random_seed,
        "model_dtype": str(model_dtype),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["num_layers"],
        config["d_model"],
        config["num_heads"],
        config["d_ff"],
        config["theta"],
        device=config["device"],
        dtype=model_dtype
    )
    assert corpus_meta["dtype"] == "uint16"
    token_loader = TokenLoader(
        corpus_path / "tokens.bin", np.uint16, tuple(corpus_meta["shape"]),
        config["batch_size"], config["context_length"], config["device"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    x, y = token_loader.get_batch()
    with wandb.init(
        entity="lordidiot-",
        project="cs336",
        config=config,
    ) as run:
        for step in tqdm(range(config["steps"])):
            # cosine schedule
            lr_t = get_lr_cosine_schedule(step, config["lr"], config["min_lr"], config["warmup_steps"], config["steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr_t

            optimizer.zero_grad()
            logits = model(x.clone())
            loss = cross_entropy_loss(logits, y.clone())
            loss.backward()
            grad_norm = gradient_norm(model.parameters())
            gradient_clipping(model.parameters(), config["gradient_clip"])
            clip_grad_norm = gradient_norm(model.parameters())
            optimizer.step()

            if step % LOG_INTERVAL == 0:
                run.log({
                    "train_loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "clip_grad_norm": clip_grad_norm.item(),
                    "lr": lr_t,
                    "gpu/mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu/mem_reserved_gb":  torch.cuda.memory_reserved() / 1e9,
                    "gpu/mem_peak_gb":      torch.cuda.max_memory_allocated() / 1e9,
                }, step=step)
                torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--theta", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=45067)
    args = parser.parse_args()

    sys.exit(main(
        args.corpus_path,
        args.lr,
        args.batch_size,
        args.steps,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.theta,
        args.random_seed,
    ))