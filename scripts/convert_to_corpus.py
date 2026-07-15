"""
Convert from list of numpy arrays to a single contiguous numpy array
"""

import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from tqdm import tqdm

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <input_pkl> <output_dir> <eos_token_id>")
    sys.exit(1)

input_pkl = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
eos_token_id = int(sys.argv[3])

with open(input_pkl, "rb") as input_f:
    documents = pickle.load(input_f)
    dtype = documents[0].dtype
    assert dtype == np.dtype(np.uint16)
    output_dir.mkdir(exist_ok = True)

    token_count = 0
    with open(output_dir / "tokens.bin", "wb") as tokens_f:
        for document in tqdm(documents):
            document.tofile(tokens_f)
            tokens_f.write(eos_token_id.to_bytes(2, 'little')) # uint16
            token_count += len(document) + 1

    with open(output_dir / "meta.json", "w") as meta_f:
        json.dump({
            "dtype": str(dtype),
            "shape": [token_count],
            "eos_token_id": eos_token_id,
        }, meta_f)