"""Step 2: convert a cached ``.trees`` file to a GRG on disk.

Kept separate from step 1 and step 3 so that neither the coalescent simulation
nor the GRG construction is repeated when only the timing code changes.
"""

import argparse
import time

import utils


def main():
    parser = argparse.ArgumentParser(description="Convert a .trees file to a GRG")
    parser.add_argument("--trees", type=str, required=True, help="Input .trees path")
    parser.add_argument("--output", type=str, required=True, help="Output .grg path")
    args = parser.parse_args()

    start = time.perf_counter()
    utils.trees_to_grg(args.trees, args.output)
    print(f"Built {args.output} in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
