"""Download gemotions data to a local directory.

    python scripts/fetch_gemotions.py --dest data/gemotions --cache-layers 40
"""

import argparse

from cogmirror.fetch import fetch_gemotions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/gemotions")
    ap.add_argument("--cache-layers", type=int, nargs="*", default=[])
    ap.add_argument("--no-stories", action="store_true")
    args = ap.parse_args()
    print(fetch_gemotions(args.dest, args.cache_layers, stories=not args.no_stories))


if __name__ == "__main__":
    main()
