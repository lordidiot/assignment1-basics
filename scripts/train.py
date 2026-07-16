from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys
import time

import torch
import numpy as np
from tqdm import tqdm
import wandb

from cs336_basics.transformer import AdamW, TransformerLM, cross_entropy_loss, get_batch, get_lr_cosine_schedule, gradient_clipping, gradient_norm, save_checkpoint


LOG_INTERVAL = 25
VAL_INTERVAL = 1000
CS336_DIR = Path(__file__).resolve().parent.parent

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

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (x, y)
        return get_batch(self.tokens, self.batch_size, self.context_length, self.device)


class Validation:
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

    def get_loss(self, model: torch.nn.Module) -> float:
        with torch.inference_mode():
            losses = []
            lens = []

            N = len(self.tokens)
            chunk_size = self.batch_size * self.context_length
            for batch_start in tqdm(range(0, N, chunk_size)):
                chunk = self.tokens[batch_start:batch_start+chunk_size+1]
                xs = []
                ys = []
                leftover = None
                for seq_start in range(0, len(chunk), self.context_length):
                    seq = chunk[seq_start:seq_start+self.context_length+1]
                    if len(seq) < self.context_length + 1:
                        # last sequence, leftover
                        leftover = seq
                    else:
                        xs.append(seq[:-1])
                        ys.append(seq[1:])

                xs = torch.tensor(np.stack(xs, 0), dtype=torch.long, device=self.device)
                ys = torch.tensor(np.stack(ys, 0), dtype=torch.long, device=self.device)
                logits = model(xs)
                losses.append(cross_entropy_loss(logits, ys).cpu().numpy())
                lens.append(xs.shape[0] * xs.shape[1])

                if leftover is not None and len(leftover) > 1:
                    x = torch.tensor(leftover[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
                    y = torch.tensor(leftover[1:], dtype=torch.long, device=self.device).unsqueeze(0)
                    logits = model(x)
                    losses.append(cross_entropy_loss(logits, y).cpu().numpy())
                    lens.append(x.shape[0] * x.shape[1])

            val_loss = np.sum(np.stack(losses) * (np.array(lens) / np.sum(lens))).item()
            return val_loss


def main(
    corpus_path: Path,
    val_path: Path,
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

    with open(val_path / "meta.json", "r") as val_f:
        val_meta = json.load(val_f)

    # Experiment configuration
    model_dtype = torch.float32
    config = {
        "dataset": corpus_path.name,
        "validation": val_path.name,
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
    validation = Validation(
        val_path / "tokens.bin", np.uint16, tuple(val_meta["shape"]),
        config["batch_size"], config["context_length"], config["device"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    with wandb.init(
        entity="lordidiot-",
        project="cs336",
        config=config,
    ) as run:
        run_dir = CS336_DIR / "out" / run.name
        run_dir.mkdir()

        for step in tqdm(range(config["steps"])):
            # cosine schedule
            lr_t = get_lr_cosine_schedule(step, config["lr"], config["min_lr"], config["warmup_steps"], config["steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr_t

            t = time.perf_counter()
            x, y = token_loader.get_batch()
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            grad_norm = gradient_norm(model.parameters())
            gradient_clipping(model.parameters(), config["gradient_clip"])
            clip_grad_norm = gradient_norm(model.parameters())
            optimizer.step()
            train_iter_s = time.perf_counter() - t


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

            if step % VAL_INTERVAL == 0:
                t = time.perf_counter()
                val_loss = validation.get_loss(model)
                val_iter_s = time.perf_counter() - t
                run.log({
                    "val_loss": val_loss,
                }, step=step)

            if step == 0:
                run.summary["timing/train_iter_s"] = train_iter_s
                run.summary["timing/val_iter_s"] = val_iter_s
                val_train_ratio = val_iter_s / (train_iter_s * VAL_INTERVAL + val_iter_s)
                print(f"{train_iter_s=}, {val_iter_s=}, {val_train_ratio=}")

        # Log one last time
        run.log({
            "val_loss": validation.get_loss(model),
        }, step=step)

        with open(run_dir / "final.pt", "wb") as f:
            save_checkpoint(model, optimizer, step, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("val_path", type=Path)
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
        args.val_path,
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