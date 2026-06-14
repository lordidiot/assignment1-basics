import argparse
from dataclasses import dataclass
import os
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
import pickle
import sys
import time
from typing import BinaryIO, Optional

import regex as re
import multiprocessing

from tqdm import tqdm

PRETOKENIZER_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenizer_split(data: bytes) -> list[bytes]:
    return list(map(lambda s: s.encode("utf-8"), PRETOKENIZER_RE.findall(data.decode("utf-8"))))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path (str): Path to a text file with BPE tokenizer training data.
        vocab_size (int): Maximum final vocabulary size. Includes the initial byte
            vocabulary, tokens produced from merges, and any special tokens.
        special_tokens (list[str]): Strings to add to the vocabulary. During training,
            treat them as hard boundaries that prevent merges across their spans,
            but exclude them when computing merge statistics.

    Returns:
        vocab (dict[int, bytes]): Mapping from token ID to token bytes.
        merges (list[tuple[bytes, bytes]]): BPE merges in order of creation.
            Each item is (<token1>, <token2>) indicating a merge.
    """
    num_processes = os.cpu_count()
    special_tokens_re = re.compile(b"|".join(re.escape(special.encode("utf-8")) for special in special_tokens))
    with open(input_path, "rb") as input_file:
        chunk_boundaries = find_chunk_boundaries(input_file, num_processes, special_tokens_re)
        boundaries = list(zip(chunk_boundaries, chunk_boundaries[1:]))

    pretokenize_and_get_counts_partial = partial(pretokenize_and_get_counts, input_path, special_tokens_re)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        counters = pool.map(pretokenize_and_get_counts_partial, boundaries)
    pretoken_counter = sum(counters, start=Counter())
    del pretoken_counter[b""] # Delete empty pretokens

    return train_merges(pretoken_counter, vocab_size, special_tokens)


@dataclass(eq=False)
class Node:
    token: bytes
    count: int
    prev: Optional["Node"] = None
    next: Optional["Node"] = None


Pair = tuple[bytes, bytes]


def train_merges(
    pretoken_counter: Counter[bytes],
    vocab_size: int,
    special_tokens: list[str],
):
    pair_counter: Counter[Pair] = Counter()
    pair_to_positions: defaultdict[Pair, list[Node]] = defaultdict(list)

    for pretoken in pretoken_counter:
        # print(f"\n================== {pretoken=}")
        count = pretoken_counter[pretoken]
        prev_node = Node(pretoken[0:1], count)
        for i in range(1, len(pretoken)):
            pair = (prev_node.token, pretoken[i:i+1])

            node = Node(pair[1], count, prev=prev_node)
            prev_node.next = node

            # print(f"{pair=} {count=}")
            pair_counter[pair] += count
            pair_to_positions[pair].append(prev_node)

            prev_node = node
    
    vocab = list(i.encode("utf-8") for i in special_tokens)
    vocab += (i.to_bytes() for i in range(0x100))
    merges = []

    for _ in tqdm(range(vocab_size - len(vocab))):
        best: tuple[int, Pair] = (0, (b"", b""))
        for pair in pair_counter:
            current = (pair_counter[pair], pair)
            if current > best:
                best = current
        
        # print(f"{best=}")
        merge_pair_count, merge_pair = best
        if not merge_pair_count:
            break

        merged = merge_pair[0] + merge_pair[1]
        merges.append(merge_pair)
        vocab.append(merged)

        removed: set[Node] = set() # To handle cases like "aaa"
        for node in list(pair_to_positions[merge_pair]):
            if node in removed:
                continue # skip

            merge_node = Node(merged, node.count)
            count = node.count

            # before -X-> node(pair0) -> pair1 -Y-> after
            # unlink X
            if (before := node.prev) is not None:
                p = (before.token, node.token)
                p_prime = (before.token, merge_node.token)
                pair_counter[p] -= count
                pair_to_positions[p].remove(before)
                removed.add(before)
                before.next = merge_node
                merge_node.prev = before
                pair_counter[p_prime] += count
                pair_to_positions[p_prime].append(before)
            
            # unlink Y
            if (after := node.next.next) is not None:
                p = (node.next.token, after.token)
                p_prime = (merge_node.token, after.token)
                pair_counter[p] -= count
                pair_to_positions[p].remove(node.next)
                removed.add(node.next)
                merge_node.next = after
                after.prev = merge_node
                pair_counter[p_prime] += count
                pair_to_positions[p_prime].append(merge_node)

        del pair_counter[merge_pair]
        del pair_to_positions[merge_pair]

    return {k:v for k,v in enumerate(vocab)}, merges



def pretokenize_and_get_counts(
    input_path: str,
    special_tokens_re: re.Pattern[bytes],
    boundary: tuple[int, int],
) -> Counter[bytes]:
    pretoken_counter = Counter()
    start, end = boundary
    bytes_left = end - start

    with open(input_path, "rb") as input_file:
        mini_chunk_size = 4096
        buffer = b""
        input_file.seek(start, os.SEEK_SET)

        while bytes_left:
            to_read = min(mini_chunk_size, bytes_left)
            bytes_left -= to_read
            buffer += input_file.read(to_read)

            documents = special_tokens_re.split(buffer)
            for document in documents[:-1]:
                pretoken_counter.update(pretokenizer_split(document))
            buffer = documents[-1]

        if buffer:
            pretoken_counter.update(pretokenizer_split(buffer))
    
    return pretoken_counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens_re: re.Pattern[bytes],
):
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries)-1):
        initial_position = max(chunk_boundaries[bi-1], chunk_boundaries[bi])
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"": # EOF
                chunk_boundaries[bi] = file_size
                break
            
            found_at = special_tokens_re.search(mini_chunk)
            if found_at is not None:
                chunk_boundaries[bi] = initial_position + found_at.start()
                break
            initial_position += mini_chunk_size
    
    return sorted(set(chunk_boundaries))


def simple_test():
    import tempfile
    with tempfile.NamedTemporaryFile("wb") as f:
        dataset = b"<|endoftext|>".join([b"low"] * 5 + [b"lower"] * 2 + [b"widest"] * 3 + [b"newest"] * 6)
        f.write(dataset)
        f.seek(0, os.SEEK_SET)
        vocab, merges = train_bpe(f.name, 1+256+6, ["<|endoftext|>"])

    print("\033[2J\033[H", end="")
    print(f"{vocab=}\n\n{merges=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("vocab_size", type=int)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    GPT4_SPECIAL_TOKENS = [
        "<|endoftext|>"
    ]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, GPT4_SPECIAL_TOKENS)
    end_time = time.perf_counter()

    with open(output_dir / "logs.txt", "w") as f:
        def print_and_write(text: str):
            print(text)
            f.write(text)

        duration = end_time - start_time
        print_and_write(f"Training on {args.input_path}")
        print_and_write(f"vocab_size = {args.vocab_size}")
        print_and_write(f"{duration=}")
        print_and_write(f"{len(vocab)=}")

    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
