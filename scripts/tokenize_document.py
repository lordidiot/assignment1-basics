"""
Tokenizes documents from a file using GPT4 special tokens and specified tokenizer.
Processes documents in chunks to reduce peak memory usage.
"""

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
import sys
import tempfile
import time
from itertools import islice
from typing import Iterator

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


EOS = "<|endoftext|>"
GPT4_SPECIAL_TOKENS = [EOS]

_tokenizer = None


def init_worker(tokenizer_path: Path) -> None:
    global _tokenizer
    _tokenizer = Tokenizer.from_files(
        str(tokenizer_path / "vocab.pkl"),
        str(tokenizer_path / "merges.pkl"),
        GPT4_SPECIAL_TOKENS,
    )


def tokenize_document(document: str) -> np.ndarray:
    token_ids = _tokenizer.encode(document)
    return np.array(token_ids, dtype=np.uint16)


def iter_documents(data_path: Path, read_size: int = 1024 * 1024) -> Iterator[str]:
    """
    Streaming equivalent of:

        open(data_path).read().split(EOS)

    but without loading the whole file into memory.
    """
    buffer = ""

    with open(data_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                break

            buffer += chunk
            parts = buffer.split(EOS)

            # Everything except the final part is a complete document.
            yield from parts[:-1]

            # The final part might be incomplete, so keep it.
            buffer = parts[-1]

    # Matches str.split(EOS): always yields the final remainder,
    # including "" if the file ended with EOS.
    yield buffer


def batched(iterator: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def combine_chunk_pickles(chunk_paths: list[Path], out_path: Path) -> None:
    """
    Combines chunk pickles into one big list, preserving original order.

    Warning: this final step still requires enough memory to hold the full list.
    """
    combined = []

    for chunk_path in tqdm(chunk_paths, desc="Combining chunks"):
        with open(chunk_path, "rb") as f:
            combined.extend(pickle.load(f))

    with open(out_path, "wb") as f:
        pickle.dump(combined, f)


def main(
    tokenizer_path: Path,
    data_path: Path,
    out_path: Path,
    docs_per_chunk: int = 100_000,
    pool_chunksize: int = 100,
) -> int:
    start_time = time.perf_counter()

    chunk_paths: list[Path] = []

    with tempfile.TemporaryDirectory(
        prefix="tokenized_chunks_",
        dir=out_path.parent,
    ) as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        with Pool(
            processes=cpu_count(),
            initializer=init_worker,
            initargs=(tokenizer_path,),
        ) as pool:
            documents = iter_documents(data_path)

            for chunk_idx, document_chunk in enumerate(
                batched(documents, docs_per_chunk)
            ):
                print(
                    f"Tokenizing chunk {chunk_idx} "
                    f"with {len(document_chunk)} documents"
                )

                token_ids_list = list(
                    tqdm(
                        pool.imap(
                            tokenize_document,
                            document_chunk,
                            chunksize=pool_chunksize,
                        ),
                        total=len(document_chunk),
                        desc=f"Chunk {chunk_idx}",
                    )
                )

                chunk_path = tmpdir / f"chunk_{chunk_idx:06d}.pkl"

                with open(chunk_path, "wb") as f:
                    pickle.dump(token_ids_list, f)

                chunk_paths.append(chunk_path)

                # Explicitly drop references before moving to next chunk.
                del document_chunk
                del token_ids_list

        print(f"Combining {len(chunk_paths)} chunk files")
        combine_chunk_pickles(chunk_paths, out_path)

    print(f"Took {time.perf_counter() - start_time:.2f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--docs-per-chunk", type=int, default=100_000)
    parser.add_argument("--pool-chunksize", type=int, default=100)

    args = parser.parse_args()

    sys.exit(
        main(
            args.tokenizer,
            args.data,
            args.out,
            docs_per_chunk=args.docs_per_chunk,
            pool_chunksize=args.pool_chunksize,
        )
    )