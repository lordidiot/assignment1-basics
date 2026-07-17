from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Generator

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM, load_checkpoint, softmax

PROMPT = "> "
EOS = "<|endoftext|>"
EOS_TOKEN = 0
GPT4_SPECIAL_TOKENS = [EOS]

# https://en.wikipedia.org/wiki/ANSI_escape_code#Control_Sequence_Introducer_commands
def _csi(final: str):
    return lambda n=1: f"\033[{n}{final}"
ansi_prev_line = _csi("F")
ansi_forward = _csi("C")


def generate(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    temp: float,
    top_p: float | None = None,
    max_tokens: int = 256,
) -> Generator[int]:
    while len(prefix) < max_tokens:
        logits = model(prefix)[-1]
        probs = softmax(logits / temp, 0)
        if top_p is not None:
            s = probs.sort(descending=True)
            remove = (torch.cumsum(s.values, 0) > top_p).to(int).sum().item() - 1
            if remove > 0:
                probs[s.indices[-remove:]] = 0
        token_id = torch.multinomial(probs, 1) # multinomial will reweight
        prefix = torch.cat((prefix, token_id))
        token_id = token_id.item()
        yield token_id
        if token_id == EOS_TOKEN:
            break

def main(
    checkpoint_path: Path,
    config_path: Path,
    tokenizer_path: Path,
    temp: float,
    top_p: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(config_path, "r") as f:
        config = json.load(f)
    if config["model_dtype"] == "torch.float32":
        model_dtype = torch.float32
    else:
        raise ValueError(f"Unknown model_dtype: {config["model_dtype"]}")

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
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer.from_files(
        str(tokenizer_path / "vocab.pkl"),
        str(tokenizer_path / "merges.pkl"),
        GPT4_SPECIAL_TOKENS,
    )

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        while True:
            prefix = input(PROMPT)
            if not prefix or prefix.lower() == "q":
                break
            sys.stdout.write(ansi_prev_line() + ansi_forward(len(prefix) + len(PROMPT)))
            prefix_tokens = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
            for token in generate(model, prefix_tokens, temp, top_p=top_p):
                sys.stdout.write(tokenizer.decode([token]))
            print("\n\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.98)
    args = parser.parse_args()
    main(
        args.checkpoint_path,
        args.config_path,
        args.tokenizer_path,
        args.temp,
        args.top_p,
    )