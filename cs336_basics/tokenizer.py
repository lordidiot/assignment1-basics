import argparse
from dataclasses import dataclass
import heapq
import os
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
import pickle
import time
from typing import BinaryIO, Iterable, Iterator, Optional, Self

import regex as re
import multiprocessing

from tqdm import tqdm

PRETOKENIZER_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenizer_split(data: bytes) -> list[bytes]:
    return list(map(lambda s: s.encode("utf-8"), PRETOKENIZER_RE.findall(data.decode("utf-8"))))

def pretokenizer_split_str(text: str) -> list[bytes]:
    return list(map(lambda s: s.encode("utf-8"), PRETOKENIZER_RE.findall(text)))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: Path | None = None
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
    pretoken_counter_pkl = output_dir / "pretoken_counter.pkl" if output_dir else None
    if (
        pretoken_counter_pkl is not None
        and pretoken_counter_pkl.exists()
    ):
        # load
        with open(pretoken_counter_pkl, "rb") as f:
            pretoken_counter = pickle.load(f)
    else:
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

        # save
        if pretoken_counter_pkl is not None:
            with open(pretoken_counter_pkl, "wb") as f:
                pickle.dump(pretoken_counter, f)

    return train_merges(pretoken_counter, vocab_size, special_tokens)


@dataclass(eq=False, slots=True)
class Node:
    token_id: int
    count: int
    prev: Optional["Node"] = None
    next: Optional["Node"] = None


Pair = tuple[int, int]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self._token_id_to_bytes = vocab
        self._bytes_to_token_id = {b:i for i,b in vocab.items()}

        self._merge_to_priority_and_id: dict[Pair, tuple[int, int]] = {}
        for priority, merge in enumerate(merges):
            pair = (self._bytes_to_token_id[merge[0]], self._bytes_to_token_id[merge[1]])
            new_token_id = self._bytes_to_token_id[merge[0] + merge[1]]
            self._merge_to_priority_and_id[pair] = (priority, new_token_id)

        if special_tokens:
            special_tokens = sorted(special_tokens, key=lambda t: len(t), reverse=True)
            self._special_tokens_re = re.compile(
                "(" + "|".join(re.escape(special) for special in special_tokens) + ")"
            )
        else:
            # Never matches, used so re.split() doesn't split anything
            self._special_tokens_re = re.compile(r"\A(?!)")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> Self:
        with (
            open(vocab_filepath, "rb") as vocab_file,
            open(merges_filepath, "rb") as merges_file
        ):
            vocab = pickle.load(vocab_file)
            merges = pickle.load(merges_file)
            return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for i, pretoken_group in enumerate(self._special_tokens_re.splititer(text)):
                if pretoken_group == "":
                    continue

                if i % 2: # special token
                    yield self._bytes_to_token_id[pretoken_group.encode()]
                else:
                    for pretoken in pretokenizer_split_str(pretoken_group):
                        for token_id in self._encode_pretoken(pretoken):
                            yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        > In the case that the input token IDs do not produce a valid Unicode
        > string, you should replace the malformed bytes with the official Unicode
        > replacement character U+FFFD.3 The errors argument of bytes.decode controls
        > how Unicode decoding errors are handled, and using errors='replace' will
        > automatically replace malformed data with the replacement marker.
        """
        return b"".join(self._token_id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        # tuple ordered to prioritise (merge ordering) followed by (position)
        # (merge_priority, position, node, pair)
        pq: list[tuple[int, int, Node, Pair]] = []
        head = Node(-1, 0) # we abuse count for tombstoning (0 = dead, 1 = live)
        prev = head
        for i in range(len(pretoken)):
            node = Node(self._bytes_to_token_id[pretoken[i:i+1]], 1, prev=prev)
            prev.next = node
            if prev != head:
                pair = (prev.token_id, node.token_id)
                if pair in self._merge_to_priority_and_id:
                    pq.append((self._merge_to_priority_and_id[pair][0], i, prev, pair))
            prev = node
        tail = Node(-1, 0)
        prev.next = tail

        heapq.heapify(pq)
        while pq:
            _merge_priority, position, a, (a_id, b_id) = heapq.heappop(pq)
            b = a.next
            if (
                not a.count
                or not b.count
                or a.token_id != a_id # I _think_ this should never happen
                or b.token_id != b_id
            ):
                continue

            merged_token_id = self._merge_to_priority_and_id[(a.token_id, b.token_id)][1]
            merged_node = Node(merged_token_id, 1, prev=a.prev, next=b.next)
            # unlink
            a.prev.next = merged_node
            b.next.prev = merged_node
            # overload count as tombstone (0=dead)
            a.count = 0
            b.count = 0
            a.next = None
            a.prev = None
            b.next = None
            b.prev = None

            # Invariant: head & tail will always be token_id=-1,
            # and I assume that this will never appear in merges
            if (pair := (merged_node.prev.token_id, merged_node.token_id)) in self._merge_to_priority_and_id:
                heapq.heappush(pq, (self._merge_to_priority_and_id[pair][0], position-1, merged_node.prev, pair))

            if (pair := (merged_node.token_id, merged_node.next.token_id)) in self._merge_to_priority_and_id:
                heapq.heappush(pq, (self._merge_to_priority_and_id[pair][0], position, merged_node, pair))

        token_ids = []
        node = head.next
        while node != tail:
            token_ids.append(node.token_id)
            node = node.next
        return token_ids


def train_merges(
    pretoken_counter: Counter[bytes],
    vocab_size: int,
    special_tokens: list[str],
):
    pair_counter: Counter[Pair] = Counter()
    pair_to_positions: defaultdict[Pair, list[Node]] = defaultdict(list)

    token_id_to_bytes = list(i.encode("utf-8") for i in special_tokens)
    token_id_to_bytes += (i.to_bytes() for i in range(0x100))
    bytes_to_token_id = {b:i for i,b in enumerate(token_id_to_bytes)}
    merges: list[tuple[bytes, bytes]] = []

    for pretoken in pretoken_counter:
        count = pretoken_counter[pretoken]
        prev_node = Node(bytes_to_token_id[pretoken[0:1]], count)
        for i in range(1, len(pretoken)):
            pair = (prev_node.token_id, bytes_to_token_id[pretoken[i:i+1]])

            node = Node(pair[1], count, prev=prev_node)
            prev_node.next = node

            pair_counter[pair] += count
            pair_to_positions[pair].append(prev_node)

            prev_node = node
    
    for _ in tqdm(range(vocab_size - len(bytes_to_token_id))):
        # Invariant: needs to be tuple[int, tuple[bytes, bytes]]
        # so that we handle the tiebreaking logic for equal counts
        best: tuple[int, tuple[bytes, bytes]] = (0, (b"", b""))
        for pair in pair_counter:
            current = (pair_counter[pair], (token_id_to_bytes[pair[0]], token_id_to_bytes[pair[1]]))
            if current > best:
                best = current
        
        merge_pair_count, merge_pair_bytes = best
        merge_pair: tuple[int, int] = tuple(bytes_to_token_id[i] for i in merge_pair_bytes)
        if not merge_pair_count:
            break

        merged_bytes = merge_pair_bytes[0] + merge_pair_bytes[1]
        merges.append(merge_pair_bytes)
        if merged_bytes not in bytes_to_token_id:
            token_id_to_bytes.append(merged_bytes)
            bytes_to_token_id[merged_bytes] = len(bytes_to_token_id)
        merged_token_id = bytes_to_token_id[merged_bytes]

        # Need to copy because we mutate pair_to_positions
        for node in list(pair_to_positions[merge_pair]):
            if (
                node.token_id != merge_pair[0]
                or node.next is None
                or node.next.token_id != merge_pair[1]
            ):
                continue # dead node

            merge_node = Node(merged_token_id, node.count)
            count = node.count

            # before -X-> node(pair0) -> pair1 -Y-> after
            # unlink X
            if (before := node.prev) is not None:
                p = (before.token_id, node.token_id)
                p_prime = (before.token_id, merge_node.token_id)
                pair_counter[p] -= count
                before.next = merge_node
                merge_node.prev = before
                pair_counter[p_prime] += count
                pair_to_positions[p_prime].append(before)
            
            # unlink Y
            if (after := node.next.next) is not None:
                p = (node.next.token_id, after.token_id)
                p_prime = (merge_node.token_id, after.token_id)
                pair_counter[p] -= count
                merge_node.next = after
                after.prev = merge_node
                pair_counter[p_prime] += count
                pair_to_positions[p_prime].append(merge_node)
            
            node.next.next = None
            node.next.prev = None
            node.next = None
            node.prev = None

        del pair_counter[merge_pair]
        del pair_to_positions[merge_pair]

    return {i:b for b,i in bytes_to_token_id.items()}, merges


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
    vocab, merges = train_bpe(args.input_path, args.vocab_size, GPT4_SPECIAL_TOKENS, output_dir)
    end_time = time.perf_counter()

    with open(output_dir / "logs.txt", "w") as f:
        def print_and_write(text: str):
            print(text)
            f.write(text + '\n')

        duration = end_time - start_time
        print_and_write(f"Training on {args.input_path}")
        print_and_write(f"vocab_size = {args.vocab_size}")
        print_and_write(f"{duration=}")
        print_and_write(f"{len(vocab)=}")

    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
