from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys
from threading import Thread
import time

import torch
import numpy as np
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import tqdm
from queue import Queue
import wandb

from cs336_basics.transformer import AdamW, TransformerLM, cross_entropy_loss, get_batch, get_lr_cosine_schedule, gradient_clipping, gradient_norm, save_checkpoint


WARMUP = 50
TIMING_WINDOW = 25
LOG_INTERVAL = 25
VAL_INTERVAL_TOKENS = 26214400 # 1 validation every VAL_INTERVAL_TOKENS (old: 800 steps)
VAL_BATCH = 128
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
        self._batch_queue = Queue(3)
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        while True:
            x, y = get_batch(self.tokens, self.batch_size, self.context_length, self.device)
            self._batch_queue.put((x, y))

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._batch_queue.get()


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
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            losses = []
            lens = []

            N = len(self.tokens)
            chunk_size = self.batch_size * self.context_length
            for batch_start in range(0, N, chunk_size):
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
                losses.append(cross_entropy_loss(logits, ys))
                lens.append(xs.shape[0] * xs.shape[1])

                if leftover is not None and len(leftover) > 1:
                    x = torch.tensor(leftover[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
                    y = torch.tensor(leftover[1:], dtype=torch.long, device=self.device).unsqueeze(0)
                    logits = model(x)
                    losses.append(cross_entropy_loss(logits, y))
                    lens.append(x.shape[0] * x.shape[1])

            losses = torch.stack(losses).cpu().numpy()
            val_loss = np.sum(losses * (np.array(lens) / np.sum(lens))).item()
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
    val_interval_steps = int(VAL_INTERVAL_TOKENS / config["context_length"] / config["batch_size"])

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
        VAL_BATCH, config["context_length"], config["device"]
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
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)

        val_iter_s = None
        for step in tqdm(range(config["steps"])):
            # Measurement code
            if step == WARMUP:
                torch.cuda.synchronize()
                timing_start = time.perf_counter()
            if step == WARMUP + TIMING_WINDOW:
                torch.cuda.synchronize()
                train_iter_s = (time.perf_counter() - timing_start) / TIMING_WINDOW
                run.summary["timing/train_iter_s"] = train_iter_s
                run.summary["timing/val_iter_s"] = val_iter_s
                val_train_ratio = val_iter_s / (train_iter_s * val_interval_steps + val_iter_s)
                print(f"{train_iter_s=}, {val_iter_s=}, {val_train_ratio=}")

            # Training step
            lr_t = get_lr_cosine_schedule(step, config["lr"], config["min_lr"], config["warmup_steps"], config["steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr_t
            x, y = token_loader.get_batch()
            optimizer.zero_grad()
            with torch.autocast(config["device"], dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
            loss.backward()
            grad_norm = gradient_clipping(model.parameters(), config["gradient_clip"])
            optimizer.step()

            # Logging stuff
            if step % LOG_INTERVAL == 0:
                run.log({
                    "train_loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": lr_t,
                    "gpu/mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu/mem_reserved_gb":  torch.cuda.memory_reserved() / 1e9,
                    "gpu/mem_peak_gb":      torch.cuda.max_memory_allocated() / 1e9,
                }, step=step)
                torch.cuda.reset_peak_memory_stats()
            if step % val_interval_steps == 0:
                if val_iter_s is None:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                val_loss = validation.get_loss(model)
                if val_iter_s is None:
                    torch.cuda.synchronize()
                    val_iter_s = time.perf_counter() - t
                run.log({
                    "val_loss": val_loss,
                }, step=step)

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