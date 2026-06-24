"""
Samples documents from a file (using GPT4 special tokens)
and tokenises them using a trained tokeniser.
"""

import argparse
from pathlib import Path
import random
import sys

from cs336_basics.tokenizer import Tokenizer


def main(tokenizer_path: Path, data_path: Path, random_seed: int) -> int:
    random.seed(random_seed)

    EOS = "<|endoftext|>"
    GPT4_SPECIAL_TOKENS = [EOS]
    tokenizer = Tokenizer.from_files(
        str(tokenizer_path / "vocab.pkl"),
        str(tokenizer_path / "merges.pkl"),
        GPT4_SPECIAL_TOKENS
    )

    # TODO: Might die on huge files, but our validation sets are small
    with open(data_path, "r") as f:
        data = f.read()
    documents = data.split(EOS)
    sampled_documents = random.sample(documents, len(documents))

    bytes_total = sum(len(document.encode()) for document in sampled_documents)
    tokens_total = sum(
        len(tokenizer.encode(document))
        for document in sampled_documents
    )
    print(f"{bytes_total=}, {tokens_total=}")
    print(f"Compression ratio: {bytes_total / tokens_total}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--random-seed", type=int, default=45067)
    args = parser.parse_args()
    sys.exit(main(args.tokenizer, args.data, args.random_seed))
