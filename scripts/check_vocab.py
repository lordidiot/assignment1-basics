import argparse
from pathlib import Path
import pickle
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir_1", type=Path)
    parser.add_argument("output_dir_2", type=Path)
    args = parser.parse_args()

    output_dir_1: Path = args.output_dir_1
    output_dir_2: Path = args.output_dir_2

    if not output_dir_1.exists():
        print(f"{output_dir_1} does not exist")
        return 1

    if not output_dir_2.exists():
        print(f"{output_dir_2} does not exist")
        return 1

    with (
        open(output_dir_1 / "vocab.pkl", "rb") as f1,
        open(output_dir_2 / "vocab.pkl", "rb") as f2
    ):
        vocab1 = pickle.load(f1)
        vocab2 = pickle.load(f2)
        print(f"{(vocab1 == vocab2)=}")
        print(f"{len(vocab1)=}")
        print(f"{len(vocab2)=}")
        for i in range(min(len(vocab1), len(vocab2))):
            if vocab1[i] != vocab2[i]:
                print(f"{i:03}: {vocab1[i]=}, {vocab2[i]=}")

    return 0


if __name__ == '__main__':
    sys.exit(main())